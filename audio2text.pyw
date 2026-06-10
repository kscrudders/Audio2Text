import os
import sys
from pathlib import Path
# --- VENV BOOTSTRAP ---
def _ensure_venv():
    # 1. Define the path to your venv's pythonw.exe
    # This assumes your venv is a folder named '.venv' in the same directory as the script
    script_dir = Path(__file__).resolve().parent
    venv_python = script_dir / "Scripts" / "pythonw.exe"
    # 2. Check if we are already running in that venv
    # If the venv exists and we aren't using its interpreter, re-launch ourselves
    if venv_python.exists() and sys.executable.lower() != str(venv_python).lower():
        import os
        # os.execl replaces the current process with the venv process
        os.execl(str(venv_python), str(venv_python), *sys.argv)
# Run the bootstrap before anything else (like AI imports) starts
_ensure_venv()
# ----------------------
# ----------------------------------------------------------------------------
# FAST STARTUP
# ----------------------------------------------------------------------------
# The heavy AI libraries (torch + NeMo) can take 10-30s to import on a cold
# start. If they are imported up front, the window cannot appear until that wait
# is over -- so it looks like nothing is happening. Instead we:
#   1) paint a tiny splash NOW using only the standard library (appears instantly),
#   2) do the quick imports and build the main window,
#   3) import torch/NeMo in a BACKGROUND thread while the window is already up,
#      showing a live "Loading AI engine..." indicator.
# ----------------------------------------------------------------------------
import time
import threading
import tkinter as tk
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # set before torch import
def _build_splash() -> tk.Tk:
    """A borderless 'Starting up...' card painted with stdlib tkinter only."""
    s = tk.Tk()
    s.overrideredirect(True)
    w, h = 380, 170
    x = (s.winfo_screenwidth() - w) // 2
    y = (s.winfo_screenheight() - h) // 2
    s.geometry(f"{w}x{h}+{x}+{y}")
    s.configure(bg="#1E1E1E")
    try:
        s.attributes("-topmost", True)
    except Exception:
        pass
    frame = tk.Frame(s, bg="#1E1E1E", highlightbackground="#3B8ED0", highlightthickness=2)
    frame.pack(fill="both", expand=True)
    tk.Label(frame, text="Audio2Text", bg="#1E1E1E", fg="#3B8ED0",
             font=("Arial", 24, "bold")).pack(pady=(36, 6))
    tk.Label(frame, text="Starting up\u2026", bg="#1E1E1E", fg="#DCE4EE",
             font=("Arial", 13)).pack()
    tk.Label(frame, text="First launch can take a little while.", bg="#1E1E1E",
             fg="gray", font=("Arial", 9)).pack(pady=(10, 0))
    s.update_idletasks()
    s.lift()
    s.update()  # force an immediate paint before the slow imports below
    return s
_SPLASH = _build_splash()
# --- Quick imports (a second or two): GUI toolkit, audio I/O, math ---
import wave
import tempfile
import audioop  # used by the CLASSIC record path's VAD-gated normalizer
import contextlib
import queue
import collections
import re
import math
import numpy as np
import pyaudio
import webrtcvad
import tkinter.filedialog as filedialog
import customtkinter as ctk
from typing import Optional, List, Union, Tuple   # NOTE: capital-U "Union"
from dataclasses import dataclass
from pydub import AudioSegment
# --- Heavy AI backend: imported lazily in a background thread (see _load_backend) ---
torch = None
gc = None
SALM = None
ASRModel = None
NEMO_AVAILABLE = False
NEMO_ERROR = None
BACKEND_READY = threading.Event()
def _load_backend() -> None:
    """Import torch + NeMo off the main thread so the GUI stays responsive."""
    global torch, gc, SALM, ASRModel, NEMO_AVAILABLE, NEMO_ERROR
    # Some libraries try to install signal handlers at import time, which raises
    # in a non-main thread. We are a GUI app (no console), so swallow those.
    import signal
    _orig_signal = signal.signal
    def _safe_signal(sig, handler):
        try:
            return _orig_signal(sig, handler)
        except (ValueError, OSError):
            return None
    signal.signal = _safe_signal
    try:
        import torch as _torch
        import gc as _gc
        from nemo.collections.speechlm2.models import SALM as _SALM
        from nemo.collections.asr.models import ASRModel as _ASRModel
        torch = _torch
        gc = _gc
        SALM = _SALM
        ASRModel = _ASRModel
        NEMO_AVAILABLE = True
    except Exception as e:
        NEMO_AVAILABLE = False
        NEMO_ERROR = str(e)
    finally:
        signal.signal = _orig_signal
        BACKEND_READY.set()
# --- CONFIGURATION ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
# ============================================================================
# CONSTANTS -- CLASSIC RECORD PATH (record at 44.1k, normalize, transcribe at end)
# ============================================================================
SAMPLE_RATE = 44100
VAD_RATE = 16000
CHUNK_SIZE = 1024
FRAME_MS = 30
TARGET_DB = -30.0
MAX_GAIN = 30.0
ATTACK = 0.1
RELEASE = 0.01
# ============================================================================
# CONSTANTS -- BATCH / FILE PATH (chunked transcription of finished audio)
# ============================================================================
OVERLAP_MS = 2000             # ~2.0s audio overlap on each side of 60s chunks
MAX_TOKENS = 512              # 256 is often too small for >~60s of speech
CHUNK_MS_CANARY = 60 * 1000   # 60s for Canary
CHUNK_MS_PARAKEET = 60 * 1000 # 60s for Parakeet
# ============================================================================
# CONSTANTS -- LIVE TRANSCRIPTION
# ----------------------------------------------------------------------------
# Canary/Parakeet are NOT streaming models: each call transcribes a finished
# audio segment. "Live" here = capture continuously, use VAD to detect pauses,
# and run the model on (a) the in-progress buffer every PARTIAL_INTERVAL_MS as a
# tentative line, and (b) the whole phrase whenever you pause (a committed line).
# These knobs trade latency vs. accuracy/fragmentation; tune freely.
# ============================================================================
LIVE_RATE = 16000               # capture rate for live (native rate for both models)
LIVE_VAD_FRAME_MS = 30          # WebRTC VAD frame size: must be 10, 20, or 30
LIVE_VAD_AGGR = 2               # 0..3, higher = treats more borderline audio as silence
SILENCE_COMMIT_MS = 800         # trailing silence that finalizes (commits) a phrase
MIN_VOICED_COMMIT_MS = 150      # require at least this much voiced audio to bother committing
MAX_SEGMENT_MS = 18000          # hard cap: force a commit even if the speaker never pauses
PARTIAL_INTERVAL_MS = 1500      # how often the tentative (in-progress) text refreshes
MIN_PARTIAL_MS = 1200           # don't run a partial on a clip shorter than this
PREROLL_MS = 300                # leading silence retained before speech starts
LIVE_OVERLAP_TAIL_MS = 400      # audio carried into the next segment (helps stitch forced cuts)
LIVE_NORMALIZE = True           # light peak-normalization per segment before transcription
LIVE_TARGET_PEAK = 0.92         # peak-normalize target as a fraction of full scale
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
REPEAT_COLLAPSE_MIN = 3   # runs of N+ identical words collapse to a single copy
def collapse_repeat_runs(tokens: List[str], min_run: int = REPEAT_COLLAPSE_MIN) -> List[str]:
    """
    Collapse runs of >= min_run identical words (case/punctuation-insensitive)
    down to a single copy. Targets the ASR failure mode where silence causes
    the decoder to loop one word for the rest of a chunk
    ('okay okay okay okay ...' -> 'okay').
    Runs shorter than min_run are left untouched (intentional repeats like
    'very very good' survive).
    """
    if not tokens:
        return tokens
    out: List[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        key = _norm(tokens[i])
        j = i + 1
        while j < n and key and _norm(tokens[j]) == key:
            j += 1
        run_len = j - i
        if run_len >= min_run:
            keep = tokens[i]
            # If the run's last token carries trailing punctuation (e.g. 'the the the.')
            # and the kept copy doesn't, move that punctuation onto the kept copy.
            trail = re.search(r"\W+$", tokens[j - 1])
            if trail and not re.search(r"\W+$", keep):
                keep = keep + trail.group(0)
            out.append(keep)
        else:
            out.extend(tokens[i:j])
        i = j
    return out
def clean_repeats_text(text: str, min_run: int = REPEAT_COLLAPSE_MIN) -> str:
    """String-level wrapper around collapse_repeat_runs."""
    return _detok(collapse_repeat_runs(_tok(text), min_run))
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
    Never mutates its inputs; always returns a new list.
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
                "\u26a1 Faster Inference",
                "nvidia/parakeet-tdt-0.6b-v3",
                "Smaller model, faster processing\nGreat low-latency option for live",
                "parakeet"
            )
            self._create_model_button(
                main_frame,
                "\U0001f3af Higher Accuracy",
                "nvidia/canary-qwen-2.5b",
                "Larger model, more accurate results\nRequires 12+ GB VRAM",
                "canary"
            )
        else:
            # Only show smaller model
            self._create_model_button(
                main_frame,
                "\u26a1 Faster Inference",
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
                text="\U0001f3af Higher Accuracy\n(Requires 15+ GB VRAM)",
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
        # --- LIVE TRANSCRIPTION STATE ---
        self.live_running = False
        self._stop_evt = threading.Event()          # set to ask capture loop to stop
        self._capture_done_evt = threading.Event()  # set by capture loop once fully flushed
        self.job_queue: "queue.Queue[dict]" = queue.Queue()
        self._live_threads: List[threading.Thread] = []
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
        # --- 3. CONTROLS CONTAINER (three buttons: Record / Live / Load) ---
        self.controls_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.controls_frame.grid(row=1, column=0, pady=10)
        # 3a. CLASSIC RECORD (transcribe at the end)
        self.btn_group_1 = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.btn_group_1.pack(side="left", padx=20)
        self.record_button = ctk.CTkButton(
            self.btn_group_1,
            text="\U0001f3a4",
            font=("Arial", 30),
            width=80,
            height=80,
            corner_radius=40,
            fg_color="#3B8ED0",
            hover_color="#36719F",
            command=self.record_click_handler,
            state="disabled"  # Disabled until model loads
        )
        self.record_button.pack()
        self.record_label = ctk.CTkLabel(self.btn_group_1, text="Record", font=("Arial", 12))
        self.record_label.pack(pady=(5, 0))
        # 3b. LIVE RECORD (pseudo-streaming, chunked)
        self.btn_group_live = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.btn_group_live.pack(side="left", padx=20)
        self.live_button = ctk.CTkButton(
            self.btn_group_live,
            text="\U0001f4e1",
            font=("Arial", 30),
            width=80,
            height=80,
            corner_radius=40,
            fg_color="#3B8ED0",
            hover_color="#36719F",
            command=self.live_click_handler,
            state="disabled"  # Disabled until model loads
        )
        self.live_button.pack()
        self.live_label = ctk.CTkLabel(self.btn_group_live, text="Live Record", font=("Arial", 12))
        self.live_label.pack(pady=(5, 0))
        # 3c. LOAD AUDIO (single or multi-file batch)
        self.btn_group_2 = ctk.CTkFrame(self.controls_frame, fg_color="transparent")
        self.btn_group_2.pack(side="left", padx=20)
        self.file_button = ctk.CTkButton(
            self.btn_group_2,
            text="\U0001f4c2",
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
        self.waveform_frame.grid_remove()  # Hidden by default
        self.waveform_canvas = tk.Canvas(
            self.waveform_frame,
            bg="#1E1E1E",
            height=60,
            highlightthickness=0,
            borderwidth=0
        )
        self.waveform_canvas.pack(fill="both", expand=True)
        # --- 5. PROGRESS BAR (used by the BATCH / file path) ---
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
        self.recording = False       # CLASSIC record path flag
        self.transcribing = False    # batch / end-of-recording transcription flag
        self.start_time = 0
        self.transcribe_start_time = 0
        self.model = None
        self.model_loading = False
        self.is_salm = False  # Track if using SALM (Canary) or ASR (Parakeet)
        self.device = None
        self.use_amp = False
        self.center_window()
        self.cleanup_old_recordings(keep_last=5)
        # The AI engine (torch + NeMo) is still importing in the background.
        # Show a clearly-alive "loading" state and proceed once it's ready.
        self._loading_start = time.time()
        self.set_status("Loading AI engine\u2026", "orange")
        self.update_text_area(
            "Starting up \u2014 loading the AI engine.\n"
            "The first launch can take a little while.",
            clear=True
        )
        self.progress_frame.grid()
        self.progress_bar.start()
        self.processing_timer_label.configure(text="Loading AI engine\u2026 0.0s")
        threading.Thread(target=_load_backend, daemon=True).start()
        self.after(150, self._await_backend)
    def _await_backend(self) -> None:
        """Poll until torch/NeMo finish importing, then start model selection."""
        if not self.winfo_exists():
            return
        elapsed = time.time() - self._loading_start
        try:
            self.processing_timer_label.configure(text=f"Loading AI engine\u2026 {elapsed:.1f}s")
        except Exception:
            pass
        if BACKEND_READY.is_set():
            try:
                self.progress_bar.stop()
                self.progress_frame.grid_remove()
                self.processing_timer_label.configure(text="Processing: 0.0s")  # reset for batch reuse
            except Exception:
                pass
            if NEMO_AVAILABLE:
                self.update_text_area("Checking system capabilities...", clear=True)
                self.show_model_selection()
            else:
                self.set_status("AI engine failed to load", "red")
                self.update_text_area(f"\u26a0\ufe0f NeMo Error:\n{NEMO_ERROR}", clear=True)
            return
        self.after(150, self._await_backend)
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
        self.live_running = False
        try:
            self._stop_evt.set()
        except Exception:
            pass
        if NEMO_AVAILABLE and torch is not None and torch.cuda.is_available():
            self.model = None
            torch.cuda.empty_cache()
            if gc is not None:
                gc.collect()
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
    def _render_text(self, text: str) -> None:
        """Replace the entire textbox contents (used by the live renderer)."""
        def _u():
            if not self.winfo_exists():
                return
            self.text_area.delete("0.0", "end")
            self.text_area.insert("end", text)
            self.text_area.see("end")
        self.after(0, _u)
    def set_status(self, text: str, color: Optional[str] = None) -> None:
        def _update():
            if not self.winfo_exists():
                return
            self.status_label.configure(text=text)
            if color:
                self.status_label.configure(text_color=color)
        self.after(0, _update)
    def _set_mode_buttons(self, record: Optional[str] = None,
                          live: Optional[str] = None,
                          file: Optional[str] = None) -> None:
        """
        Enable/disable the three mode buttons from any thread.
        Pass 'normal' or 'disabled' for each; None leaves it unchanged.
        """
        def _u():
            if not self.winfo_exists():
                return
            if record is not None:
                self.record_button.configure(state=record)
            if live is not None:
                self.live_button.configure(state=live)
            if file is not None:
                self.file_button.configure(state=file)
        self.after(0, _u)
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
        step_x = width / (self.waveform_data.maxlen - 1)
        for i, amp in enumerate(self.waveform_data):
            # Apply gain for visualization and clamp to 1.0; scale to ~95% of half-height
            scaled_amp = min(1.0, amp * 6.0) * (height / 2) * 0.95
            x = int(i * step_x)
            y_top = int(center_y - scaled_amp)
            y_bottom = int(center_y + scaled_amp)
            if scaled_amp > 1:
                self.waveform_canvas.create_line(x, y_top, x, y_bottom, fill="#3B8ED0", width=max(1, step_x * 0.8))
            else:
                self.waveform_canvas.create_line(x, center_y, x, center_y + 1, fill="#3B8ED0", width=max(1, step_x * 0.8))
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
            if not is_salm:
                try:
                    from omegaconf import open_dict
                    decoding_cfg = self.model.cfg.decoding
                    with open_dict(decoding_cfg):
                        decoding_cfg.strategy = "greedy_batch"
                        decoding_cfg.greedy.use_cuda_graph_decoder = False
                    self.model.change_decoding_strategy(decoding_cfg)
                except Exception as e:
                    # Fallback: direct disable (path varies by NeMo version)
                    try:
                        self.model.decoding.decoding.decoding_computer.disable_cuda_graphs()
                    except Exception:
                        self.update_text_area(f"Note: could not disable CUDA graphs: {e}")
            self.set_status(f"Ready ({d_name})", "#2CC985")
            self.update_text_area(
                f"Loaded: {model_path}\n"
                f"\U0001f3a4 Record = transcribe when you stop \u00b7 "
                f"\U0001f4e1 Live Record = transcribe as you speak \u00b7 "
                f"\U0001f4c2 Load Audio = file(s)",
                clear=True
            )
            # Enable all three mode buttons
            self._set_mode_buttons(record="normal", live="normal", file="normal")
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
                if not os.path.exists(wav_path):
                    return f"[ERROR: File not found: {wav_path}]", None
                with torch.no_grad():
                    # Force num_workers=0 to prevent Windows multiprocessing crash in the GUI thread.
                    output = self.model.transcribe(
                        paths2audio_files=[wav_path],
                        batch_size=1,
                        num_workers=0
                    )
                    if output and len(output) > 0:
                        hypothesis = output[0]
                        if hasattr(hypothesis, 'text'):
                            return hypothesis.text, None
                        else:
                            return str(hypothesis), None
                    return "", None
            except TypeError:
                try:
                    with torch.no_grad():
                        output = self.model.transcribe(audio=[wav_path], num_workers=0)
                        if output and len(output) > 0:
                            hypothesis = output[0]
                            if hasattr(hypothesis, 'text'):
                                return hypothesis.text, None
                            return str(hypothesis), None
                        return "", None
                except Exception as inner_e:
                    raise RuntimeError(f"Parakeet processing error: {str(inner_e)}")
            except Exception as e:
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
                except (TypeError, AttributeError):
                    try:
                        # Fallback for older Canary versions
                        answer_ids = self.model.generate(prompts=prompts, max_new_tokens=MAX_TOKENS)
                        text = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
                        return text, None
                    except Exception as inner_e:
                        return f"[CANARY ERROR: {str(inner_e)}]", None
    # ========================================================================
    # LIVE (pseudo-streaming) TRANSCRIPTION
    # ========================================================================
    def _peak_normalize(self, pcm: bytes) -> bytes:
        """Light, boost-only peak normalization to keep quiet mics audible without clipping."""
        try:
            arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
            if arr.size == 0:
                return pcm
            peak = float(np.max(np.abs(arr)))
            if peak < 1.0:
                return pcm
            gain = (LIVE_TARGET_PEAK * 32767.0) / peak
            gain = max(1.0, min(gain, MAX_GAIN))  # only boost, never attenuate
            if gain == 1.0:
                return pcm
            out = np.clip(arr * gain, -32767, 32767).astype(np.int16)
            return out.tobytes()
        except Exception:
            return pcm
    def _transcribe_pcm(self, pcm: bytes) -> Tuple[str, Optional[float]]:
        """Write a 16k mono PCM buffer to a temp wav and run the loaded model on it."""
        if LIVE_NORMALIZE:
            pcm = self._peak_normalize(pcm)
        tf = tempfile.NamedTemporaryFile(prefix="live_", suffix=".wav", delete=False)
        tmp_path = tf.name
        tf.close()
        try:
            with wave.open(tmp_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(LIVE_RATE)
                wf.writeframes(pcm)
            try:
                text, score = self.transcribe_chunk_with_score(tmp_path)
            except Exception:
                text, score = "", None
            return (text or ""), score
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
    def start_live(self) -> None:
        if self.live_running:
            return
        if self.recording or self.transcribing:
            self.update_text_area("Another recording/transcription is in progress.")
            return
        if self.model_loading or self.model is None:
            self.update_text_area("Model still loading, please wait...")
            return
        # Reset live state (fresh events/queue so the app can be restarted repeatedly)
        self.live_running = True
        self._stop_evt = threading.Event()
        self._capture_done_evt = threading.Event()
        self.job_queue = queue.Queue()
        self.transcribe_start_time = time.time()
        # UI: live button becomes Stop; other modes are locked out
        self.update_text_area("", clear=True)
        self.live_button.configure(fg_color="#E74C3C", text="\u23f9")
        self.live_label.configure(text="Stop")
        self._set_mode_buttons(record="disabled", file="disabled")
        self.set_status("Listening\u2026 (live)", "#E74C3C")
        # Waveform
        self.waveform_running = True
        self.waveform_frame.grid()
        self.waveform_data.clear()
        self.update_waveform()
        # Threads
        t_cap = threading.Thread(target=self._live_capture_loop, daemon=True)
        t_wrk = threading.Thread(target=self._live_worker_loop, daemon=True)
        self._live_threads = [t_cap, t_wrk]
        t_cap.start()
        t_wrk.start()
    def stop_live(self) -> None:
        if not self.live_running:
            return
        self.live_running = False
        self._stop_evt.set()
        # The worker keeps running until the capture loop flushes the final segment
        # and the queue drains; it will call _finish_live() to restore the UI.
        self.set_status("Finishing\u2026", "orange")
        self.live_button.configure(state="disabled", fg_color="gray", text="\u2026")
        self.live_label.configure(text="Stop")
        self.waveform_running = False
        self.waveform_frame.grid_remove()
        self.waveform_data.clear()
    def _live_capture_loop(self) -> None:
        """
        Continuously reads 30ms frames at 16k mono, runs VAD, and decides when to emit
        'partial' (in-progress) and 'final' (committed) transcription jobs.
        Owns: seg buffer, VAD/silence counters, segment id. Passes immutable snapshots
        to the worker via the queue, so no locks are needed.
        """
        p = None
        stream = None
        seg = bytearray()
        had_speech = False
        try:
            vad = webrtcvad.Vad(LIVE_VAD_AGGR)
            samples_per_frame = int(LIVE_RATE * LIVE_VAD_FRAME_MS / 1000)   # 480 @ 16k/30ms
            preroll_bytes = int(LIVE_RATE * PREROLL_MS / 1000) * 2
            overlap_bytes = int(LIVE_RATE * LIVE_OVERLAP_TAIL_MS / 1000) * 2
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=LIVE_RATE,
                input=True,
                frames_per_buffer=samples_per_frame
            )
            silence_ms = 0
            voiced_ms = 0
            seg_id = 1
            last_partial_t = time.time()
            start_t = time.time()
            frame_count = 0
            def seg_ms_now() -> float:
                return (len(seg) // 2) / (LIVE_RATE / 1000.0)
            while not self._stop_evt.is_set():
                data = stream.read(samples_per_frame, exception_on_overflow=False)
                frame_count += 1
                # Waveform peak
                try:
                    arr = np.frombuffer(data, dtype=np.int16)
                    self.audio_queue.put(float(np.max(np.abs(arr))) / 32768.0)
                except Exception:
                    pass
                # VAD
                try:
                    speech = vad.is_speech(data, LIVE_RATE)
                except Exception:
                    speech = True  # if VAD errors, assume speech (don't drop audio)
                seg.extend(data)
                if speech:
                    had_speech = True
                    silence_ms = 0
                    voiced_ms += LIVE_VAD_FRAME_MS
                else:
                    if had_speech:
                        silence_ms += LIVE_VAD_FRAME_MS
                    else:
                        # Keep only a short pre-roll of leading silence so a quiet
                        # speaker never bloats the buffer toward MAX_SEGMENT_MS.
                        if len(seg) > preroll_bytes:
                            del seg[:len(seg) - preroll_bytes]
                # Throttled timer update (~every 240ms)
                if frame_count % 8 == 0:
                    elapsed = time.time() - start_t
                    self.after(0, lambda t=elapsed: self.timer_label.configure(
                        text=time.strftime("%H:%M:%S", time.gmtime(t))
                    ))
                # ----- Commit decision -----
                seg_len_ms = seg_ms_now()
                force = seg_len_ms >= MAX_SEGMENT_MS
                natural_pause = had_speech and silence_ms >= SILENCE_COMMIT_MS
                if natural_pause or (force and had_speech):
                    if voiced_ms >= MIN_VOICED_COMMIT_MS:
                        self.job_queue.put({"type": "final", "seg_id": seg_id, "pcm": bytes(seg)})
                    # Carry a short tail into the next segment to help stitch forced (mid-word) cuts
                    tail = bytes(seg[-overlap_bytes:]) if overlap_bytes > 0 else b""
                    seg = bytearray(tail)
                    seg_id += 1
                    had_speech = False
                    silence_ms = 0
                    voiced_ms = 0
                    last_partial_t = time.time()
                    continue
                if force and not had_speech:
                    # Pure silence somehow hit the cap; drop it.
                    seg = bytearray()
                    continue
                # ----- Partial (tentative) decision -----
                now = time.time()
                if (had_speech
                        and seg_len_ms >= MIN_PARTIAL_MS
                        and (now - last_partial_t) * 1000.0 >= PARTIAL_INTERVAL_MS):
                    self.job_queue.put({"type": "partial", "seg_id": seg_id, "pcm": bytes(seg)})
                    last_partial_t = now
        except Exception as e:
            self.set_status(f"Mic error: {e}", "red")
        finally:
            try:
                if stream is not None:
                    if stream.is_active():
                        stream.stop_stream()
                    stream.close()
            except Exception:
                pass
            try:
                if p is not None:
                    p.terminate()
            except Exception:
                pass
            # Flush whatever phrase is in progress as a final job.
            try:
                if had_speech and len(seg) > 0:
                    # seg_id is fine here; worker tracks committed_seg monotonically
                    self.job_queue.put({"type": "final", "seg_id": 10 ** 9, "pcm": bytes(seg)})
            except Exception:
                pass
            self._capture_done_evt.set()
    def _live_worker_loop(self) -> None:
        """
        Drains the job queue, transcribes, and renders. Coalesces partials (only the
        newest matters), always processes finals in order. Owns: committed token list.
        """
        committed: List[str] = []
        committed_seg = 0
        while True:
            jobs: List[dict] = []
            try:
                jobs.append(self.job_queue.get(timeout=0.15))
                while True:
                    jobs.append(self.job_queue.get_nowait())
            except queue.Empty:
                pass
            finals = [j for j in jobs if j["type"] == "final"]
            partials = [j for j in jobs if j["type"] == "partial"]
            did_work = False
            # Process all finals in order -> these mutate the committed transcript.
            for j in finals:
                text, score = self._transcribe_pcm(j["pcm"])
                toks = _tok(text)
                committed = stitch_tokens(committed, toks, curr_score=score, window=120, min_match=5)
                committed_seg = max(committed_seg, j["seg_id"])
                did_work = True
                # Display-only cleanup; 'committed' stays raw so segment stitching
                # can still align against the true token sequence.
                self._render_text(clean_repeats_text(_detok(committed)))
            # Process only the newest partial, and only if its segment isn't committed yet.
            if partials:
                pj = partials[-1]
                if pj["seg_id"] > committed_seg:
                    text, score = self._transcribe_pcm(pj["pcm"])
                    toks = _tok(text)
                    # Tentative view = committed + this phrase, without mutating committed.
                    merged = stitch_tokens(committed, toks, curr_score=score, window=120, min_match=5)
                    did_work = True
                    self._render_text(clean_repeats_text(_detok(merged)))
            if not did_work:
                if self._capture_done_evt.is_set() and self.job_queue.empty():
                    break
        # Final cleanup: collapse hallucinated word loops (silence artifacts)
        final_text = clean_repeats_text(_detok(committed))
        saved_path = self._save_live_transcript(final_text)
        self._finish_live(final_text, saved_path)
    def _save_live_transcript(self, text: str) -> Optional[Path]:
        if not text.strip():
            return None
        try:
            recordings_dir = app_dir() / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            txt_path = recordings_dir / f"transcript_{int(time.time())}.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
            return txt_path
        except Exception:
            return None
    def _finish_live(self, final_text: str, saved_path: Optional[Path]) -> None:
        def _u():
            if not self.winfo_exists():
                return
            self.live_button.configure(state="normal", fg_color="#3B8ED0", text="\U0001f4e1")
            self.live_label.configure(text="Live Record")
            self.record_button.configure(state="normal")
            self.file_button.configure(state="normal")
            # Make sure the final committed text is what's shown.
            self.text_area.delete("0.0", "end")
            self.text_area.insert("end", final_text)
            self.text_area.see("end")
            if saved_path is not None:
                self.set_status(f"Done \u00b7 saved {saved_path.name}", "#2CC985")
            else:
                self.set_status("Done", "#2CC985")
            self.reset_timer()
        self.after(0, _u)
    # ========================================================================
    # BATCH TRANSCRIPTION (file path + end of classic recording)
    # ========================================================================
    def run_transcription(self, filepaths: Union[str, Path, List[Union[str, Path]]]) -> None:
        """
        Transcribe one file or a batch of files sequentially.
        Accepts a single path (str/Path) -- used by the classic recording flow --
        or a list of paths from the multi-select file dialog. Each file is
        chunked, transcribed, stitched, and saved to a .txt next to the source.
        A failure on one file does not abort the rest of the batch.
        """
        if self.model_loading:
            self.update_text_area("Model still loading, please wait...")
            return
        if not NEMO_AVAILABLE or self.model is None:
            self.update_text_area("Model not ready or NeMo library missing.")
            return
        if self.transcribing:
            self.update_text_area("A transcription is already running. Please wait for it to finish.")
            return
        # Normalize input to a list of Paths
        if isinstance(filepaths, (str, Path)):
            filepaths = [filepaths]
        file_list: List[Path] = [Path(f) for f in filepaths]
        if not file_list:
            return
        self.show_progress(True)
        # Lock all mode buttons for the duration of the batch
        self._set_mode_buttons(record="disabled", live="disabled", file="disabled")
        def _batch_thread():
            n_files = len(file_list)
            # Each entry: (display_name, text_or_error, success_bool)
            results: List[Tuple[str, str, bool]] = []
            n_failed = 0
            batch_start = time.time()
            try:
                for file_idx, filepath in enumerate(file_list):
                    if not self.app_running:
                        return
                    prefix = f"[File {file_idx + 1}/{n_files}] " if n_files > 1 else ""
                    try:
                        final_text = self._transcribe_one_file(filepath, prefix, n_files)
                        results.append((filepath.name, final_text, True))
                        # Save transcript next to the original file immediately,
                        # so completed work survives a crash later in the batch.
                        try:
                            txt_path = filepath.with_suffix(".txt")
                            with open(txt_path, "w", encoding="utf-8") as f:
                                f.write(final_text)
                        except Exception:
                            pass
                    except Exception as e:
                        n_failed += 1
                        # Free VRAM so the NEXT file in the batch can still run
                        try:
                            if self.use_amp and torch.cuda.is_available():
                                torch.cuda.empty_cache()
                            gc.collect()
                        except Exception:
                            pass
                        err_msg = (
                            f"\u26a0\ufe0f FAILED: {e}\n"
                            f"Source file is untouched at: {filepath.resolve()}\n"
                            f"You can retry it alone via the \U0001f4c2 button."
                        )
                        results.append((filepath.name, err_msg, False))
                    # Refresh the text area with everything finished so far
                    self._render_batch_results(results, n_files)
                elapsed = time.time() - batch_start
                if n_files > 1:
                    if n_failed == 0:
                        self.set_status(f"Batch done: {n_files} files ({elapsed:.1f}s)", "#2CC985")
                    else:
                        self.set_status(
                            f"Batch done: {n_files - n_failed} ok, {n_failed} failed ({elapsed:.1f}s)",
                            "orange"
                        )
                else:
                    if n_failed == 0:
                        self.set_status(f"Done ({elapsed:.1f}s)", "#2CC985")
                    else:
                        self.set_status("Failed (Audio Saved)", "red")
            finally:
                self.show_progress(False)
                self._set_mode_buttons(record="normal", live="normal", file="normal")
                try:
                    if self.use_amp and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
        threading.Thread(target=_batch_thread, daemon=True).start()
    def _transcribe_one_file(self, filepath: Path, prefix: str, n_files: int) -> str:
        """
        Chunk + transcribe + stitch a single audio file. Returns the final
        stitched text. Raises on failure (the batch loop catches it).
        Temp chunk files are always cleaned up.
        """
        temp_files: List[Path] = []
        stitcher = OverlapStitcher(window=120, min_match=5)
        try:
            self.set_status(f"{prefix}Loading Audio...", "orange")
            audio = AudioSegment.from_file(str(filepath))
            audio = audio.set_channels(1).set_frame_rate(16000)
            duration_ms = len(audio)
            # Dynamic chunk length selection
            if self.is_salm:
                CHUNK_LENGTH_MS = CHUNK_MS_CANARY
            else:
                CHUNK_LENGTH_MS = CHUNK_MS_PARAKEET
            total_chunks = max(1, math.ceil(duration_ms / CHUNK_LENGTH_MS))
            recordings_dir = app_dir() / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            # Live progress header (don't clear in batch mode -- keep earlier results visible)
            self.update_text_area(
                f"\n--- {prefix}{filepath.name} ---\n"
                f"Audio Length: {duration_ms / 1000 / 60:.2f} mins\n"
                f"Chunking: {CHUNK_LENGTH_MS / 1000:.0f}s core, "
                f"{OVERLAP_MS / 1000:.1f}s overlap\n",
                clear=(n_files == 1)
            )
            for i in range(total_chunks):
                if not self.app_running:
                    raise RuntimeError("Application closing")
                core_start = i * CHUNK_LENGTH_MS
                core_end = min((i + 1) * CHUNK_LENGTH_MS, duration_ms)
                slice_start = max(0, core_start - OVERLAP_MS)
                slice_end = min(duration_ms, core_end + OVERLAP_MS)
                # --- DYNAMIC OVERLAP ADJUSTMENT ---
                # Ensure final segments for Parakeet are at least 5000ms
                if not self.is_salm:
                    current_len = slice_end - slice_start
                    min_required = 5000  # 5 seconds minimum
                    if current_len < min_required:
                        padding_needed = min_required - current_len
                        slice_start = max(0, slice_start - (padding_needed + 2000))
                # ----------------------------------
                chunk_audio = audio[slice_start:slice_end]
                if len(chunk_audio) < 100:
                    continue
                chunk_filename = f"temp_chunk_{time.time_ns()}_{i}.wav"
                chunk_path = recordings_dir / chunk_filename
                chunk_audio.export(str(chunk_path), format="wav")
                temp_files.append(chunk_path)
                self.set_status(f"{prefix}Transcribing {i + 1}/{total_chunks}...", "#3B8ED0")
                text_part, score = self.transcribe_chunk_with_score(str(chunk_path))
                stitcher.add(text_part, chunk_score=score)
                self.update_text_area(f"--- {prefix}Chunk {i + 1}/{total_chunks} ---\n{text_part}\n")
            # Final cleanup: collapse hallucinated word loops (silence artifacts)
            return clean_repeats_text(stitcher.text())
        finally:
            for pth in temp_files:
                try:
                    if pth.exists():
                        pth.unlink()
                except Exception:
                    pass
    def _render_batch_results(self, results: List[Tuple[str, str, bool]], n_files: int) -> None:
        """
        Rewrite the text area with the final transcripts of all completed files.
        Single-file runs keep the original behavior (plain text, no header).
        """
        if n_files == 1 and results:
            # Original single-file behavior: just the transcript (or error)
            self.update_text_area(results[0][1], clear=True)
            return
        blocks = []
        for idx, (name, text, ok) in enumerate(results):
            marker = "\u2705" if ok else "\u274c"
            blocks.append(
                f"{'=' * 60}\n{marker} [{idx + 1}/{n_files}] {name}\n{'=' * 60}\n{text}"
            )
        remaining = n_files - len(results)
        if remaining > 0:
            blocks.append(f"... {remaining} file(s) remaining ...")
        self.update_text_area("\n\n".join(blocks), clear=True)
    # ========================================================================
    # CLASSIC RECORD PATH (record -> normalize -> save WAV -> transcribe at end)
    # ========================================================================
    def record_click_handler(self) -> None:
        """Toggle the classic (transcribe-at-the-end) recording."""
        if self.model_loading or self.model is None:
            return
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
            if self.live_running or self.transcribing:
                self.update_text_area("Another recording/transcription is in progress.")
                return
            self.recording = True
            # Lock out the other modes while recording
            self._set_mode_buttons(live="disabled", file="disabled")
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
        Runs on the background thread. (Classic record path only.)
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
                self.after(0, lambda t=passed: self.timer_label.configure(
                    text=time.strftime("%H:%M:%S", time.gmtime(t))
                ))
            if stream is not None and stream.is_active():
                stream.stop_stream()
                stream.close()
            stream = None
            if p is not None:
                p.terminate()
                p = None
            if not frames:
                self.set_status("No audio recorded", "orange")
                self.after(0, lambda: self.record_button.configure(
                    fg_color="#3B8ED0",
                    state="normal"
                ))
                self._set_mode_buttons(live="normal", file="normal")
                return
            full_data = b"".join(frames)
            del frames
            normalized = self.normalize_audio(full_data)
            recordings_dir = app_dir() / "recordings"
            recordings_dir.mkdir(exist_ok=True)
            filename = recordings_dir / f"recording_{int(time.time())}.wav"
            with wave.open(str(filename), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(SAMPLE_RATE)
                wf.writeframes(normalized)
            self.set_status("File saved.", "white")
            self.cleanup_old_recordings()
            # run_transcription manages button locking from here
            self.run_transcription(filename)
        except Exception as e:
            self.set_status(f"Recording Error: {str(e)}", "red")
            self.recording = False
            self.waveform_running = False
            self.waveform_frame.grid_remove()
            # Reset button appearance AND state; unlock the other modes
            self.after(0, lambda: self.record_button.configure(
                fg_color="#3B8ED0",
                state="normal"
            ))
            self.after(0, lambda: self.record_label.configure(text="Record"))
            self._set_mode_buttons(live="normal", file="normal")
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
    # --- HANDLERS ---
    def live_click_handler(self) -> None:
        """Toggle live transcription on/off."""
        if self.model_loading or self.model is None:
            return
        if self.live_running:
            self.stop_live()
        else:
            self.start_live()
    def select_file_handler(self) -> None:
        """Select one OR multiple audio files; multiple files are queued and
        processed sequentially."""
        if self.live_running or self.recording or self.transcribing:
            return
        filenames = filedialog.askopenfilenames(
            title="Select one or more audio files",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac *.ogg")]
        )
        if not filenames:
            return
        files = list(filenames)
        if len(files) == 1:
            self.update_text_area(f"File: {os.path.basename(files[0])}", clear=True)
        else:
            listing = "\n".join(
                f"  {i + 1}. {os.path.basename(f)}" for i, f in enumerate(files)
            )
            self.update_text_area(
                f"Batch queued ({len(files)} files):\n{listing}\n", clear=True
            )
        self.run_transcription(files)
if __name__ == "__main__":
    # The main window is built without the heavy AI libraries, so it appears
    # quickly; tear down the splash just before it shows.
    try:
        _SPLASH.destroy()
    except Exception:
        pass
    app = VoiceRecorder()
    app.mainloop()
