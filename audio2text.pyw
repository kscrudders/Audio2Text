import os
import sys
import wave
import time
import threading
import pyaudio
import audioop
import numpy as np
import webrtcvad
import contextlib
import queue
import collections
import tkinter as tk
import tkinter.filedialog as filedialog
import customtkinter as ctk
from pathlib import Path
from typing import Optional, List, Union, Tuple
from dataclasses import dataclass
import re
import math
from pydub import AudioSegment

# --- AI IMPORTS ---
NEMO_ERROR = None
try:
    import torch
    import gc
    from nemo.collections.speechlm2.models import SALM
    from nemo.collections.asr.models import ASRModel
    NEMO_AVAILABLE = True
except Exception as e:
    NEMO_AVAILABLE = False
    NEMO_ERROR = str(e)

# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"  # handling for large audio files

# --- CONSTANTS ---
SAMPLE_RATE = 44100
VAD_RATE = 16000
CHUNK_SIZE = 1024
FRAME_MS = 30

TARGET_DB = -30.0
MAX_GAIN = 30.0
ATTACK = 0.1
RELEASE = 0.01

OVERLAP_MS = 2000            # ~2.0s audio overlap on each side of 60s chunks

# Generation / chunking
MAX_TOKENS = 512  # 256 is often too small for >~60s of speech;
CHUNK_MS_CANARY = 60 * 1000      # 60s for Canary
CHUNK_MS_PARAKEET = 4 * 60 * 1000 # 5 mins for Parakeet


def app_dir() -> Path:
    """Directory containing the running app (exe) or script."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


# ============================================================================
# Overlap + alignment-aware stitching helpers (token-level)
# ============================================================================

_PUNCT_STRIP_RE = re.compile(r"(^\W+)|(\W+$)", re.UNICODE)


def _tok(text: str) -> List[str]:
    return [t for t in text.strip().split() if t]


def _norm(t: str) -> str:
    t = t.lower()
    t = _PUNCT_STRIP_RE.sub("", t)
    return t


def _detok(tokens: List[str]) -> str:
    if not tokens:
        return ""
    text = " ".join(tokens)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


@dataclass
class Match:
    i0: int
    j0: int
    length: int


def longest_common_substring(a: List[str], b: List[str]) -> Match:
    """
    Longest contiguous match between token lists a and b.
    Complexity: O(len(a)*len(b)), but we keep windows small (<= ~120 tokens).
    """
    if not a or not b:
        return Match(0, 0, 0)

    prev = [0] * (len(b) + 1)
    best_len = 0
    best_end_i = 0
    best_end_j = 0

    for i in range(1, len(a) + 1):
        curr = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > best_len:
                    best_len = curr[j]
                    best_end_i = i
                    best_end_j = j
        prev = curr

    return Match(best_end_i - best_len, best_end_j - best_len, best_len)


_FILLERS = {"um", "uh", "yeah", "okay", "ok", "like", "hmm"}


def _repetition_penalty(norm_tokens: List[str]) -> float:
    penalty = 0.0
    run = 1
    for i in range(1, len(norm_tokens)):
        if norm_tokens[i] and norm_tokens[i] == norm_tokens[i - 1]:
            run += 1
        else:
            if run > 1:
                penalty += (run - 1) ** 2
            run = 1
    if run > 1:
        penalty += (run - 1) ** 2
    return penalty


def _quality_score(tokens: List[str]) -> float:
    nt = [_norm(t) for t in tokens]
    rep = _repetition_penalty(nt)
    filler = sum(1 for x in nt if x in _FILLERS)
    return -(2.0 * rep + 0.3 * filler)


def _trim_join_runs(prefix: List[str], suffix: List[str], max_trim: int = 32) -> List[str]:
    """
    If boundary produces 'yeah | yeah', trim duplicates from suffix.
    Conservative: only removes exact normalized duplicates at the join.
    """
    out = list(suffix)
    trims = 0
    while prefix and out and trims < max_trim:
        if _norm(prefix[-1]) and _norm(prefix[-1]) == _norm(out[0]):
            out.pop(0)
            trims += 1
        else:
            break
    return out


def stitch_tokens(
    prev_tokens: List[str],
    curr_tokens: List[str],
    prev_score: Optional[float] = None,
    curr_score: Optional[float] = None,
    window: int = 120,
    min_match: int = 5,
) -> List[str]:
    """
    Alignment-aware stitch using longest common substring over normalized tokens.
    If scores available, keeps overlap from higher-score chunk; else uses heuristic.
    """
    if not prev_tokens:
        return list(curr_tokens)
    if not curr_tokens:
        return list(prev_tokens)

    tail = prev_tokens[-window:]
    head = curr_tokens[:window]

    norm_tail = [_norm(t) for t in tail]
    norm_head = [_norm(t) for t in head]

    m = longest_common_substring(norm_tail, norm_head)

    # No robust overlap match -> trim boundary duplicates and concat
    if m.length < min_match:
        suffix = _trim_join_runs(prev_tokens, curr_tokens)
        return prev_tokens + suffix

    # Map match indices back to full prev/curr
    tail_start_global = len(prev_tokens) - len(tail)
    prev_match_start = tail_start_global + m.i0
    prev_match_end = prev_match_start + m.length
    curr_match_start = m.j0
    curr_match_end = curr_match_start + m.length

    prev_overlap = prev_tokens[prev_match_start:prev_match_end]
    curr_overlap = curr_tokens[curr_match_start:curr_match_end]

    if prev_score is not None and curr_score is not None:
        keep_current = (curr_score > prev_score)
    else:
        keep_current = (_quality_score(curr_overlap) > _quality_score(prev_overlap))

    if keep_current:
        prefix = prev_tokens[:prev_match_start]
        suffix = curr_tokens[curr_match_start:]
        suffix = _trim_join_runs(prefix, suffix)
        return prefix + suffix

    prefix = prev_tokens
    suffix = curr_tokens[curr_match_end:]
    suffix = _trim_join_runs(prefix, suffix)
    return prefix + suffix


class OverlapStitcher:
    def __init__(self, window: int = 120, min_match: int = 5):
        self.window = window
        self.min_match = min_match
        self.tokens: List[str] = []
        self.last_score: Optional[float] = None

    def add(self, chunk_text: str, chunk_score: Optional[float] = None) -> str:
        ct = _tok(chunk_text)
        self.tokens = stitch_tokens(
            self.tokens,
            ct,
            prev_score=self.last_score,
            curr_score=chunk_score,
            window=self.window,
            min_match=self.min_match,
        )
        self.last_score = chunk_score
        return _detok(self.tokens)

    def text(self) -> str:
        return _detok(self.tokens)


# ============================================================================
# Model Selection Screen
# ============================================================================

class ModelSelectionScreen(ctk.CTkToplevel):
    def __init__(self, parent, vram_gb: float, has_gpu: bool):
        super().__init__(parent)
        
        self.selected_model = None
        self.vram_gb = vram_gb
        self.has_gpu = has_gpu
        
        self.title("Select Model")
        self.geometry("500x400")
        self.resizable(False, False)
        
        # Make modal
        self.transient(parent)
        self.grab_set()
        
        # Center the window
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (250)
        y = (self.winfo_screenheight() // 2) - (200)
        self.geometry(f"500x400+{x}+{y}")
        
        # Main container
        main_frame = ctk.CTkFrame(self, fg_color="transparent")
        main_frame.pack(fill="both", expand=True, padx=40, pady=40)
        
        # Title
        title = ctk.CTkLabel(
            main_frame,
            text="Choose AI Model",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=(0, 10))
        
        # VRAM info
        if has_gpu:
            vram_label = ctk.CTkLabel(
                main_frame,
                text=f"Detected VRAM: {vram_gb:.1f} GB",
                font=("Arial", 12),
                text_color="gray"
            )
            vram_label.pack(pady=(0, 30))
        else:
            vram_label = ctk.CTkLabel(
                main_frame,
                text="No GPU detected - using CPU",
                font=("Arial", 12),
                text_color="orange"
            )
            vram_label.pack(pady=(0, 30))
        
        # Model options
        if vram_gb >= 15.0:
            # Show both options
            self._create_model_button(
                main_frame,
                "⚡ Faster Inference",
                "nvidia/parakeet-tdt-0.6b-v3",
                "Smaller model, faster processing\n*WIP random crashing bug*",
                "parakeet"
            )
            
            self._create_model_button(
                main_frame,
                "🎯 Higher Accuracy",
                "nvidia/canary-qwen-2.5b",
                "Larger model, more accurate results\nRequires 12+ GB VRAM",
                "canary"
            )
        else:
            # Only show smaller model
            self._create_model_button(
                main_frame,
                "⚡ Faster Inference",
                "nvidia/parakeet-tdt-0.6b-v3",
                "Optimized for your system\nBest option for available VRAM",
                "parakeet",
                auto_select=True
            )
            
            # Show disabled larger model option
            disabled_frame = ctk.CTkFrame(main_frame, fg_color="#2B2B2B", corner_radius=10)
            disabled_frame.pack(fill="x", pady=10)
            
            disabled_label = ctk.CTkLabel(
                disabled_frame,
                text="🎯 Higher Accuracy\n(Requires 15+ GB VRAM)",
                font=("Arial", 14),
                text_color="gray"
            )
            disabled_label.pack(pady=20)
    
    def _create_model_button(self, parent, title: str, model_path: str, 
                             description: str, model_key: str, auto_select: bool = False):
        """Create a clickable model selection card"""
        frame = ctk.CTkFrame(parent, fg_color="#2B2B2B", corner_radius=10)
        frame.pack(fill="x", pady=10)
        
        button = ctk.CTkButton(
            frame,
            text=f"{title}\n\n{description}",
            font=("Arial", 14),
            height=80,
            fg_color="#3B8ED0",
            hover_color="#36719F",
            command=lambda: self._select_model(model_key, model_path)
        )
        button.pack(fill="both", expand=True, padx=3, pady=3)
        
        if auto_select:
            # Auto-select after a brief moment for better UX
            self.after(100, lambda: self._select_model(model_key, model_path))
    
    def _select_model(self, model_key: str, model_path: str):
        """Handle model selection"""
        self.selected_model = {
            'key': model_key,
            'path': model_path,
            'is_salm': (model_key == 'canary')
        }
        self.grab_release()
        self.destroy()
    
    def get_selection(self) -> Optional[dict]:
        """Wait for user selection and return the chosen model"""
        self.wait_window()
        return self.selected_model


# ============================================================================
# App
# ============================================================================

class VoiceRecorder(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- WAVEFORM STATE ---
        self.audio_queue = queue.Queue()
        self.waveform_data = collections.deque(maxlen=200)  # Store last N amplitude points
        self.waveform_running = False

        # --- CONFIG: FFmpeg Setup for Robustness ---
        self.ffmpeg_path = app_dir() / "ffmpeg.exe"
        if self.ffmpeg_path.exists():
            AudioSegment.converter = str(self.ffmpeg_path)
            ffprobe_path = app_dir() / "ffprobe.exe"
            if ffprobe_path.exists():
                AudioSegment.ffprobe = str(ffprobe_path)

        # --- 1. WINDOW GEOMETRY ---
        self.geometry("900x600")
        self.title("Audio2Text")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.app_running = True
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(5, weight=1)

        # --- 2. STATUS HEADER ---
        self.status_label = ctk.CTkLabel(
            self,
            text="Initializing...",
            font=("Arial", 14),
            text_color="gray"
        )
        self.status_label.grid(row=0, column=0, pady=(20, 10), sticky="ew")

        # --- 3. CONTROLS CONTAINER ---
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, pady=10)

        self.btn_group_1 = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.btn_group_1.pack(side="left", padx=30)

        self.record_button = ctk.CTkButton(
            self.btn_group_1,
            text="🎤",
            font=("Arial", 30),
            width=80,
            height=80,
            corner_radius=40,
            fg_color="#3B8ED0",
            hover_color="#36719F",
            command=self.click_handler,
            state="disabled"  # Disabled until model loads
        )
        self.record_button.pack()

        self.record_label = ctk.CTkLabel(self.btn_group_1, text="Record", font=("Arial", 12))
        self.record_label.pack(pady=(5, 0))

        self.btn_group_2 = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.btn_group_2.pack(side="left", padx=30)

        self.file_button = ctk.CTkButton(
            self.btn_group_2,
            text="📂",
            font=("Arial", 30),
            width=80,
            height=80,
            corner_radius=40,
            fg_color="#3B8ED0",
            hover_color="#36719F",
            command=self.select_file_handler,
            state="disabled"  # Disabled until model loads
        )
        self.file_button.pack()

        self.load_label = ctk.CTkLabel(self.btn_group_2, text="Load Audio", font=("Arial", 12))
        self.load_label.pack(pady=(5, 0))

        # --- 4. MAIN TIMER ---
        self.timer_label = ctk.CTkLabel(
            self,
            text="00:00:00",
            font=("Roboto Medium", 40),
            text_color="#3B8ED0"
        )
        self.timer_label.grid(row=2, column=0, pady=(10, 5), sticky="ew")

        # --- 4.5 WAVEFORM CANVAS ---
        self.waveform_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.waveform_frame.grid(row=3, column=0, pady=(0, 10), sticky="ew", padx=40)
        self.waveform_frame.grid_remove() # Hidden by default

        self.waveform_canvas = tk.Canvas(
            self.waveform_frame,
            bg="#1E1E1E",
            height=60,
            highlightthickness=0,
            borderwidth=0
        )
        self.waveform_canvas.pack(fill="both", expand=True)

        # --- 5. PROGRESS BAR ---
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.progress_frame.grid(row=4, column=0, pady=(0, 10), sticky="ew", padx=60)
        self.progress_frame.grid_remove()

        self.progress_bar = ctk.CTkProgressBar(self.progress_frame, mode="indeterminate", width=500)
        self.progress_bar.pack()

        self.processing_timer_label = ctk.CTkLabel(self.progress_frame, text="Processing: 0.0s", font=("Arial", 12))
        self.processing_timer_label.pack(pady=5)

        # --- 6. TEXT AREA ---
        self.text_area = ctk.CTkTextbox(
            self,
            font=("Roboto", 14),
            wrap="word",
            corner_radius=10
        )
        self.text_area.grid(row=5, column=0, padx=40, pady=(0, 20), sticky="nsew")

        # --- 7. FOOTER ---
        self.copy_button = ctk.CTkButton(
            self,
            text="Copy to Clipboard",
            command=self.copy_to_clipboard,
            fg_color="transparent",
            border_width=1,
            text_color=("gray10", "#DCE4EE")
        )
        self.copy_button.grid(row=6, column=0, pady=(0, 20))

        # --- INTERNAL STATE ---
        self.recording = False
        self.transcribing = False
        self.start_time = 0
        self.transcribe_start_time = 0

        self.model = None
        self.model_loading = False
        self.is_salm = False  # Track if using SALM (Canary) or ASR (Parakeet)
        self.device = None
        self.use_amp = False

        self.center_window()
        self.cleanup_old_recordings(keep_last=5)

        if NEMO_AVAILABLE:
            self.update_text_area("Checking system capabilities...")
            # Show model selection screen after a brief delay
            self.after(500, self.show_model_selection)
        else:
            self.update_text_area(f"⚠️ NeMo Error:\n{NEMO_ERROR}")

    def show_model_selection(self):
        """Show model selection screen based on VRAM availability"""
        vram_gb = 0.0
        has_gpu = False
        
        try:
            if torch.cuda.is_available():
                has_gpu = True
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024 ** 3)
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                has_gpu = True
                vram_gb = 8.0  # Assume reasonable amount for MPS
        except Exception:
            pass
        
        # Show selection screen
        selection_screen = ModelSelectionScreen(self, vram_gb, has_gpu)
        selection = selection_screen.get_selection()
        
        if selection:
            # Load the selected model
            self.update_text_area(f"Loading {selection['path']}...\n(This runs in background)")
            threading.Thread(
                target=self.load_ai_model,
                args=(selection['path'], selection['is_salm']),
                daemon=True
            ).start()
        else:
            # User closed the window without selecting
            self.update_text_area("No model selected. Please restart to choose a model.")

    def on_closing(self) -> None:
        """Clean shutdown of threads and window."""
        self.app_running = False
        self.recording = False
        self.transcribing = False
        if NEMO_AVAILABLE and torch.cuda.is_available():
            self.model = None
            gc.collect()
            torch.cuda.empty_cache()
        self.destroy()

    def center_window(self) -> None:
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")

    def cleanup_old_recordings(self, keep_last: int = 100) -> None:
        """Deletes old recordings to save space."""
        try:
            recordings_dir = app_dir() / "recordings"
            if not recordings_dir.exists():
                return
            files = sorted(recordings_dir.glob("recording_*.wav"), key=lambda x: x.stat().st_mtime)
            if len(files) > keep_last:
                for old_file in files[:-keep_last]:
                    try:
                        old_file.unlink()
                    except OSError:
                        pass
        except Exception:
            pass

    # --- UI HELPERS ---

    def update_text_area(self, text: str, clear: bool = False) -> None:
        def _update():
            if not self.winfo_exists():
                return
            if clear:
                self.text_area.delete("0.0", "end")
            if text:
                self.text_area.insert("end", text + "\n")
                self.text_area.see("end")

        self.after(0, _update)

    def set_status(self, text: str, color: Optional[str] = None) -> None:
        def _update():
            if not self.winfo_exists():
                return
            self.status_label.configure(text=text)
            if color:
                self.status_label.configure(text_color=color)

        self.after(0, _update)

    def show_progress(self, show: bool = True) -> None:
        def _update():
            if not self.winfo_exists():
                return
            if show:
                self.transcribing = True
                self.transcribe_start_time = time.time()
                self.progress_frame.grid()
                self.progress_bar.start()
                self.update_processing_timer()
            else:
                self.transcribing = False
                self.progress_frame.grid_remove()
                self.progress_bar.stop()

        self.after(0, _update)

    def update_processing_timer(self) -> None:
        if self.transcribing and self.app_running:
            elapsed = time.time() - self.transcribe_start_time
            self.processing_timer_label.configure(text=f"Processing: {elapsed:.1f}s")
            self.after(100, self.update_processing_timer)

    def reset_timer(self) -> None:
        self.timer_label.configure(text="00:00:00")

    def copy_to_clipboard(self) -> None:
        text = self.text_area.get("0.0", "end").strip()
        if not text:
            return
        self.clipboard_clear()
        self.clipboard_append(text)
        self.copy_button.configure(text="Copied!", fg_color="#2CC985", text_color="white")
        self.after(2000, lambda: self.copy_button.configure(
            text="Copy to Clipboard",
            fg_color="transparent",
            text_color=("gray10", "#DCE4EE")
        ))

    # --- WAVEFORM VISUALIZATION ---

    def update_waveform(self) -> None:
        if not self.waveform_running:
            return

        # 1. Drain the queue to get latest amplitudes
        while not self.audio_queue.empty():
            try:
                amp = self.audio_queue.get_nowait()
                self.waveform_data.append(amp)
            except queue.Empty:
                break

        # 2. Draw
        self.waveform_canvas.delete("all")
        width = self.waveform_canvas.winfo_width()
        height = self.waveform_canvas.winfo_height()
        center_y = height / 2

        if not self.waveform_data or width <= 1:
            self.after(30, self.update_waveform)
            return

        # Normalize data to fit height
        # Simple visualization: Line connecting points
        points = []
        n_points = len(self.waveform_data)
        
        # We want to stretch the available data points across the width
        # or scroll them. Let's scroll: newest on right.
        
        step_x = width / (self.waveform_data.maxlen - 1)
        
        for i, amp in enumerate(self.waveform_data):
            # Amp is 0.0 to 1.0 (approx)
            # Visualize as mirrored wave or just top
            # Let's do mirrored for "Audacity style"
            
            # Apply 5.0x GAIN for visualization and clamp to 1.0
            # Scale to 90% of half-height
            scaled_amp = min(1.0, amp * 6.0) * (height / 2) * 0.95
            
            x = int(i * step_x)
            y_top = int(center_y - scaled_amp)
            y_bottom = int(center_y + scaled_amp)
            
            # Draw a vertical line for this time slice
            # self.waveform_canvas.create_line(x, y_top, x, y_bottom, fill="#3B8ED0", width=2)
            
            # Better: create a polygon or line graph?
            # Vertical bars look like Audacity zoomed out
            if scaled_amp > 1:
                 self.waveform_canvas.create_line(x, y_top, x, y_bottom, fill="#3B8ED0", width=max(1, step_x * 0.8))
            else:
                 self.waveform_canvas.create_line(x, center_y, x, center_y+1, fill="#3B8ED0", width=max(1, step_x * 0.8))

        self.after(10, self.update_waveform)

    # --- MODEL LOADING ---

    def load_ai_model(self, model_path: str, is_salm: bool) -> None:
        try:
            self.model_loading = True
            self.is_salm = is_salm

            vram_gb = 0.0
            d_name = "CPU"

            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.use_amp = True
                
                # Check VRAM
                props = torch.cuda.get_device_properties(0)
                vram_gb = props.total_memory / (1024 ** 3)
                d_name = f"{props.name} ({vram_gb:.1f} GB)"
                
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
                self.use_amp = False
                d_name = "Apple MPS"
            else:
                self.device = torch.device("cpu")
                self.use_amp = False
                d_name = "CPU"

            model_class = SALM if is_salm else ASRModel
            load_msg = f"Loading {model_path}..."

            self.set_status(load_msg, "orange")
            
            # Check local cache override
            base = app_dir()
            local_path = base / "canary_model"
            if local_path.exists() and (local_path / "model_config.yaml").exists():
                pass

            self.model = model_class.from_pretrained(model_path)
            self.model = self.model.to(self.device)
            self.model.eval()

            self.set_status(f"Ready ({d_name})", "#2CC985")
            self.update_text_area(f"Loaded: {model_path}\nReady to record.", clear=True)
            
            # Enable buttons
            self.after(0, lambda: self.record_button.configure(state="normal"))
            self.after(0, lambda: self.file_button.configure(state="normal"))

        except Exception as e:
            self.update_text_area(f"Model Load Failed: {e}")
            self.set_status("Model load failed", "red")
        finally:
            self.model_loading = False

    # --- TRANSCRIPTION HELPERS ---

    def transcribe_chunk_with_score(self, wav_path: str) -> Tuple[str, Optional[float]]:
        """
        Transcribe a WAV chunk. 
        If is_salm=True (Canary), attempts to compute avg log-prob via generate().
        If is_salm=False (Parakeet), uses transcribe() API and returns no score.
        """
        
        # --- PATH A: Parakeet (ASR Model / TDT) ---
        if not self.is_salm:
            try:
                # Ensure the file exists
                import os
                if not os.path.exists(wav_path):
                    return f"[ERROR: File not found: {wav_path}]", None
                
                with torch.no_grad():
                    # FIX: Force num_workers=0 to prevent Windows multiprocessing crash
                    # inside the GUI thread.
                    output = self.model.transcribe(
                        paths2audio_files=[wav_path],
                        batch_size=1,
                        num_workers=0  # <--- CRITICAL FIX
                    )
                    
                    if output and len(output) > 0:
                        hypothesis = output[0]
                        if hasattr(hypothesis, 'text'):
                            return hypothesis.text, None
                        else:
                            return str(hypothesis), None
                        
                    return "", None
                    
            except TypeError as e:
                try:
                    with torch.no_grad():
                        output = self.model.transcribe(audio=[wav_path])
                        if output and len(output) > 0:
                            hypothesis = output[0]
                            if hasattr(hypothesis, 'text'):
                                return hypothesis.text, None
                            return str(hypothesis), None
                        return "", None
                except Exception as inner_e:
                    # RAISE the error instead of returning it as text
                    raise RuntimeError(f"Parakeet processing error: {str(inner_e)}")
                    
            except Exception as e:
                # RAISE the error instead of returning it as text
                raise RuntimeError(f"Parakeet transcription failed: {str(e)}")

        # --- PATH B: Canary (Speech LLM) ---
        prompts = [[{
            "role": "user",
            "content": f"Transcribe: {self.model.audio_locator_tag}",
            "audio": [wav_path],
        }]]

        amp_ctx = torch.amp.autocast(device_type=self.device.type) if self.use_amp else contextlib.nullcontext()

        with torch.no_grad():
            with amp_ctx:
                try:
                    out = self.model.generate(
                        prompts=prompts,
                        max_new_tokens=MAX_TOKENS,
                        return_dict_in_generate=True,
                        output_scores=True,
                        do_sample=False,
                    )
                    seq = out.sequences[0]
                    scores = out.scores
                    gen_len = len(scores)

                    avg_lp = None
                    if gen_len > 0:
                        gen_ids = seq[-gen_len:]
                        lps = []
                        for step_logits, tok_id in zip(scores, gen_ids):
                            lp = torch.log_softmax(step_logits[0], dim=-1)[tok_id].item()
                            lps.append(lp)
                        avg_lp = float(sum(lps) / len(lps))

                    text = self.model.tokenizer.ids_to_text(seq.cpu())
                    return text, avg_lp

                except (TypeError, AttributeError) as e:
                    try:
                        # Fallback for older Canary versions
                        answer_ids = self.model.generate(prompts=prompts, max_new_tokens=MAX_TOKENS)
                        text = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
                        return text, None
                    except Exception as inner_e:
                        return f"[CANARY ERROR: {str(inner_e)}]", None

    # --- TRANSCRIPTION ---
    def run_transcription(self, filepath: Union[str, Path]) -> None:
        """
        Overlap + alignment-aware stitching to reduce boundary duplication/restarts.
        Uses short chunks to avoid degeneration when MAX_TOKENS is limited.
        """
        if self.model_loading:
            self.update_text_area("Model still loading, please wait...")
            return

        if not NEMO_AVAILABLE or self.model is None:
            self.update_text_area("Model not ready or NeMo library missing.")
            return

        self.show_progress(True)

        def _transcribe_thread():

            temp_files: List[Path] = []
            stitcher = OverlapStitcher(window=120, min_match=5)

            try:
                self.set_status("Loading Audio...", "orange")

                audio = AudioSegment.from_file(str(filepath))
                audio = audio.set_channels(1).set_frame_rate(16000)

                duration_ms = len(audio)

                # --- SAFETY CHECK (Issue 1) ---
                if not self.is_salm and duration_ms < 5000:
                    self.update_text_area("⚠️ Audio too short for Parakeet.\nPlease record at least 5 seconds.", clear=True)
                    self.set_status("Audio too short", "orange")
                    return
                # ------------------------

                # Dynamic Chunk Length Selection
                if self.is_salm:
                    CHUNK_LENGTH_MS = CHUNK_MS_CANARY
                else:
                    CHUNK_LENGTH_MS = CHUNK_MS_PARAKEET

                total_chunks = max(1, math.ceil(duration_ms / CHUNK_LENGTH_MS))

                recordings_dir = app_dir() / "recordings"
                recordings_dir.mkdir(exist_ok=True)

                self.update_text_area(f"Audio Length: {duration_ms/1000/60:.2f} mins", clear=True)
                self.update_text_area(
                    f"Chunking: {CHUNK_LENGTH_MS/1000:.0f}s core, {OVERLAP_MS/1000:.1f}s overlap\n"
                )

                for i in range(total_chunks):
                    core_start = i * CHUNK_LENGTH_MS
                    core_end = min((i + 1) * CHUNK_LENGTH_MS, duration_ms)

                    # Initial slice calculation
                    slice_start = max(0, core_start - OVERLAP_MS)
                    slice_end = min(duration_ms, core_end + OVERLAP_MS)

                    # --- DYNAMIC OVERLAP ADJUSTMENT (Issue 2) ---
                    # Ensure final segments for Parakeet are at least 5000ms
                    if not self.is_salm:
                        current_len = slice_end - slice_start
                        min_required = 5000  # 5 seconds minimum
                        # Update status with debug info
                            
                        if current_len < min_required:
                            # Calculate how much padding we are missing
                            padding_needed = min_required - current_len
                            # Extend backwards, ensuring we don't go below index 0
                            slice_start = max(0, slice_start - (padding_needed+2000))
                            
                            # Update status with debug info
                            # self.set_status(f"Adjusting: Len {current_len}ms + {padding_needed}ms pad", "orange")
                    # --------------------------------------------

                    chunk_audio = audio[slice_start:slice_end]
                    if len(chunk_audio) < 100:
                        continue

                    chunk_filename = f"temp_chunk_{time.time_ns()}_{i}.wav"
                    chunk_path = recordings_dir / chunk_filename
                    chunk_audio.export(str(chunk_path), format="wav")
                    temp_files.append(chunk_path)

                    self.set_status(f"Transcribing {i+1}/{total_chunks}...", "#3B8ED0")

                    text_part, score = self.transcribe_chunk_with_score(str(chunk_path))
                    stitcher.add(text_part, chunk_score=score)

                    self.update_text_area(f"--- Chunk {i+1}/{total_chunks} ---\n{text_part}\n")

                    # VRAM cleanup
                    if self.use_amp and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                final_text = stitcher.text()
                self.update_text_area(final_text, clear=True)

                # Save to .txt next to the original file
                try:
                    txt_path = Path(filepath).with_suffix(".txt")
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(final_text)
                except Exception:
                    pass

                elapsed = time.time() - self.transcribe_start_time
                self.set_status(f"Done ({elapsed:.1f}s)", "#2CC985")

            except Exception as e:
                # Free up VRAM so the user can successfully retry
                if self.use_amp and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # Format a user-friendly recovery message
                safe_path = Path(filepath).resolve()
                error_msg = (
                    f"⚠️ Transcription Failed!\n\n"
                    f"Error Details: {str(e)}\n\n"
                    f"Your audio was safely saved before the crash.\n"
                    f"File Location: {safe_path}\n\n"
                    f"To try again:\n"
                    f"1. Click the 📂 (Load Audio) button\n"
                    f"2. Select the file mentioned above."
                )
                
                self.update_text_area(error_msg, clear=True)
                self.set_status("Failed (Audio Saved)", "red")

            finally:
                self.show_progress(False)
                for p in temp_files:
                    try:
                        if p.exists():
                            p.unlink()
                    except Exception:
                        pass
                if self.use_amp and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        threading.Thread(target=_transcribe_thread, daemon=True).start()

    # --- AUDIO LOGIC ---

    def select_file_handler(self) -> None:
        filename = filedialog.askopenfilename(
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg")]
        )
        if filename:
            self.update_text_area(f"File: {os.path.basename(filename)}", clear=True)
            self.run_transcription(filename)

    def click_handler(self) -> None:
        if self.recording:
            self.recording = False
            self.record_button.configure(fg_color="#3B8ED0")
            self.record_label.configure(text="Record")
            self.set_status("Stopping...", "orange")
            
            # Stop waveform
            self.waveform_running = False
            self.waveform_frame.grid_remove()
            self.waveform_data.clear()

            self.after(500, self.reset_timer)
        else:
            self.recording = True
            
            # Start waveform
            self.waveform_running = True
            self.waveform_frame.grid()
            self.waveform_data.clear()
            self.update_waveform()

            self.record_button.configure(fg_color="#E74C3C")
            self.record_label.configure(text="Stop")
            self.set_status("Recording...", "#E74C3C")
            threading.Thread(target=self.record, daemon=True).start()

    def normalize_audio(self, raw_bytes: bytes) -> bytes:
        """
        Normalizes audio using WebRTC VAD to detect speech segments and apply gain.
        Runs on the background thread.
        """
        if not raw_bytes:
            return raw_bytes

        try:
            vad_bytes, _ = audioop.ratecv(raw_bytes, 2, 1, SAMPLE_RATE, VAD_RATE, None)
            vad = webrtcvad.Vad(2)

            audio_in = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
            output_audio = np.zeros_like(audio_in)

            samples_vad = int(VAD_RATE * FRAME_MS / 1000)
            bytes_vad_frame = samples_vad * 2
            samples_in = int(SAMPLE_RATE * FRAME_MS / 1000)

            ptr_vad = 0
            ptr_in = 0
            current_gain = 1.0

            while ptr_vad + bytes_vad_frame <= len(vad_bytes):
                if ptr_in + samples_in > len(audio_in):
                    break

                frame_vad = vad_bytes[ptr_vad: ptr_vad + bytes_vad_frame]
                is_speech = vad.is_speech(frame_vad, VAD_RATE)

                idx_start = ptr_in
                idx_end = ptr_in + samples_in
                chunk = audio_in[idx_start:idx_end]

                target_gain = 1.0
                if is_speech:
                    rms = np.sqrt(np.mean(chunk ** 2))
                    if rms > 0:
                        rms_db = 20 * np.log10((rms / 32768.0) + 1e-9)
                        if rms_db < TARGET_DB:
                            gain_needed = 10 ** ((TARGET_DB - rms_db) / 20)
                            target_gain = min(gain_needed, MAX_GAIN)
                            target_gain = max(target_gain, 1.0)

                if target_gain > current_gain:
                    current_gain = (1 - ATTACK) * current_gain + ATTACK * target_gain
                else:
                    current_gain = (1 - RELEASE) * current_gain + RELEASE * target_gain

                output_audio[idx_start:idx_end] = chunk * current_gain

                ptr_vad += bytes_vad_frame
                ptr_in += samples_in

            if ptr_in < len(audio_in):
                output_audio[ptr_in:] = audio_in[ptr_in:] * current_gain

            output_audio = np.clip(output_audio, -32767, 32767)
            return output_audio.astype(np.int16).tobytes()

        except Exception:
            return raw_bytes

    def record(self) -> None:
        p = None
        stream = None
        # Track the button lock state locally so we only update UI when it changes
        button_locked = False 

        try:
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )

            frames: List[bytes] = []
            start = time.time()

            while self.recording and self.app_running:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                frames.append(data)

                # --- Waveform Update ---
                try:
                    audio_chunk = np.frombuffer(data, dtype=np.int16)
                    peak = np.max(np.abs(audio_chunk))
                    norm_peak = peak / 32768.0
                    self.audio_queue.put(norm_peak)
                except Exception:
                    pass
                # -----------------------

                passed = time.time() - start

                # ============================================================
                #  UI BEHAVIOR: Lock 'Stop' button during unsafe windows
                # ============================================================
                if not self.is_salm:  # Only for Parakeet
                    # Lock if < 5s OR within the 5s window after a 5 min (300s) chunk
                    should_lock = (passed % (CHUNK_MS_PARAKEET/1000)) < 5

                    if should_lock != button_locked:
                        button_locked = should_lock
                        if should_lock:
                            # Disable button
                            #self.after(0, lambda: self.record_button.configure(
                            #    state="disabled",
                            #    fg_color="gray",
                            #    text="More audio needed..."
                            #))
                            self.after(0, lambda: self.record_button.configure(
                                state="disabled",
                                fg_color="gray"
                            ))
                        else:
                            # Re-enable button
                            self.after(0, lambda: self.record_button.configure(
                                state="normal",
                                fg_color="#E74C3C"
                            ))
                # ============================================================

                self.after(0, lambda t=passed: self.timer_label.configure(
                    text=time.strftime("%H:%M:%S", time.gmtime(t))
                ))

            if stream is not None and stream.is_active():
                stream.stop_stream()
                stream.close()
            stream = None

            if p is not None:
                p.terminate()

            if not frames:
                self.set_status("No audio recorded", "orange")
                # Ensure button is reset to normal state
                self.after(0, lambda: self.record_button.configure(
                    fg_color="#3B8ED0", 
                    state="normal", 
                    text="Record"
                ))
                return

            full_data = b"".join(frames)
            del frames

            normalized = self.normalize_audio(full_data)

            recordings_dir = app_dir() / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            filename = recordings_dir / f"recording_{int(time.time())}.wav"

            with wave.open(str(filename), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(normalized)

            self.set_status("File saved.", "white")
            self.cleanup_old_recordings()
            self.run_transcription(filename)

        except Exception as e:
            self.set_status(f"Recording Error: {str(e)}", "red")
            self.recording = False
            self.waveform_running = False
            self.waveform_frame.grid_remove()
            
            # Reset button appearance AND state
            self.after(0, lambda: self.record_button.configure(
                fg_color="#3B8ED0", 
                state="normal"
            ))
            self.after(0, lambda: self.record_label.configure(text="Record"))

        finally:
            # Final safety check to ensure button is clickable if thread dies
            if not self.recording:
                 self.after(0, lambda: self.record_button.configure(state="normal"))

            try:
                if stream is not None:
                    stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                if p is not None:
                    p.terminate()
            except Exception:
                pass


if __name__ == "__main__":
    app = VoiceRecorder()
    app.mainloop()