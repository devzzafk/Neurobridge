"""
main_app.py
===========
Silent Speech: An End-to-End TinyML Assistive Interface
========================================================
Crown-jewel dashboard application.

Features:
  • Modern dark-theme CustomTkinter UI (falls back to styled Tkinter if CTk absent)
  • Live 2-channel EMG oscilloscope (rolling waveform + RMS energy)
  • Real-time Random Forest inference with confidence threshold meter
  • Action Feed that flashes Electric Cyan on gesture detection
  • Macro keyboard mapping (gesture → keypress simulation via pynput)
  • System Status panel: connection, model version, inference latency
  • Built-in dataset generation and model training (no external setup needed)
  • Thread-safe background acquisition + inference pipeline

Run:
    python main_app.py
"""

# ─── Stdlib ───────────────────────────────────────────────────────────────────
import threading
import queue
import time
import os
import sys
import pickle
import collections
from typing import Optional

# ─── Numeric / ML ─────────────────────────────────────────────────────────────
import numpy as np

# ─── GUI ──────────────────────────────────────────────────────────────────────
import tkinter as tk
from tkinter import ttk, font as tkfont, messagebox, simpledialog

try:
    import customtkinter as ctk
    _CTK = True
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
except ImportError:
    _CTK = False

# ─── Keyboard simulation (optional) ───────────────────────────────────────────
try:
    from pynput.keyboard import Controller as KbController, Key
    _PYNPUT = True
except ImportError:
    _PYNPUT = False

# ─── Internal modules ─────────────────────────────────────────────────────────
from data_simulator   import EMGStreamSimulator, generate_dataset, DATASET_FILE
from signal_processing import EMGFilterPipeline, extract_features, RollingRMS, FEATURE_NAMES
from model_trainer     import train, load_model, MODEL_FILE


# ══════════════════════════════════════════════════════════════════════════════
# PALETTE & CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

PAL = {
    "bg"          : "#0d0f14",   # Deep Charcoal
    "surface"     : "#141820",   # Panel surface
    "surface2"    : "#1c2030",   # Elevated surface
    "border"      : "#2a3048",   # Subtle border
    "cyan"        : "#00e5ff",   # Electric Cyan accent
    "cyan_dim"    : "#007a8a",   # Dimmed cyan
    "green"       : "#00ff9d",   # Status good
    "yellow"      : "#ffd740",   # Warning
    "red"         : "#ff4f5e",   # Error / danger
    "text"        : "#e8ecf4",   # Crisp White
    "text_dim"    : "#6b7a99",   # Muted text
    "ch1"         : "#00e5ff",   # Channel 1 waveform
    "ch2"         : "#7c4dff",   # Channel 2 waveform
    "bar_bg"      : "#1c2030",
    "bar_fg"      : "#00e5ff",
    "bar_warn"    : "#ffd740",
    "bar_ok"      : "#00ff9d",
}

FS               = 1000        # Hz
CHUNK_SIZE       = 64          # samples per acquisition tick
DISPLAY_SECONDS  = 3           # seconds of waveform shown in oscilloscope
CONFIDENCE_THRESH = 0.85       # 85% — minimum confidence to trigger action
INFERENCE_WINDOW_MS = 200.0    # ms
INFERENCE_STEP_MS   = 50.0     # ms
FLASH_DURATION_MS   = 600      # ms action feed flash duration
MODEL_VERSION       = "v1.0"

CLASS_NAMES   = ["REST", "CLENCH", "FLICK"]
DEFAULT_MACROS = {
    1: "enter",     # CLENCH → Enter key
    2: "space",     # FLICK  → Spacebar
}
ACTION_LABELS = {
    0: "[  REST  ]",
    1: "[ CLENCH ]  ↵ ENTER",
    2: "[  FLICK ]  ░ SPACE",
}


# ══════════════════════════════════════════════════════════════════════════════
# KEYBOARD MACRO EXECUTOR
# ══════════════════════════════════════════════════════════════════════════════

class MacroExecutor:
    """Fires simulated key presses for mapped gestures."""

    KEY_MAP = {
        "enter"    : Key.enter     if _PYNPUT else None,
        "space"    : Key.space     if _PYNPUT else None,
        "tab"      : Key.tab       if _PYNPUT else None,
        "esc"      : Key.esc       if _PYNPUT else None,
        "up"       : Key.up        if _PYNPUT else None,
        "down"     : Key.down      if _PYNPUT else None,
        "left"     : Key.left      if _PYNPUT else None,
        "right"    : Key.right     if _PYNPUT else None,
        "backspace": Key.backspace  if _PYNPUT else None,
    }

    def __init__(self, mappings: dict):
        """
        mappings : {gesture_class_int: key_name_str}
                   e.g. {1: "enter", 2: "space"}
        """
        self.mappings = dict(mappings)
        self._kb = KbController() if _PYNPUT else None

    def fire(self, gesture_class: int):
        key_name = self.mappings.get(gesture_class)
        if not key_name or not _PYNPUT or self._kb is None:
            return
        key = self.KEY_MAP.get(key_name.lower())
        if key:
            try:
                self._kb.press(key)
                self._kb.release(key)
            except Exception:
                pass
        else:
            # Single character
            try:
                self._kb.press(key_name[0])
                self._kb.release(key_name[0])
            except Exception:
                pass

    def update_mapping(self, gesture_class: int, key_name: str):
        self.mappings[gesture_class] = key_name


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME INFERENCE PIPELINE  (runs in background thread)
# ══════════════════════════════════════════════════════════════════════════════

class InferencePipeline:
    """
    Acquisition → Filtering → Windowing → ML Inference

    Runs in its own daemon thread. Pushes results to a thread-safe queue
    consumed by the UI update loop.
    """

    def __init__(self, model, result_queue: queue.Queue,
                 fs: int = FS, chunk_size: int = CHUNK_SIZE):
        self.model        = model
        self.result_queue = result_queue
        self.fs           = fs
        self.chunk_size   = chunk_size
        self._stop        = threading.Event()

        # Signal pipeline
        self._simulator   = EMGStreamSimulator(fs=fs, chunk_size=chunk_size)
        self._filter      = EMGFilterPipeline(fs=fs)

        # Rolling buffers for windowed inference
        win_samples    = int(INFERENCE_WINDOW_MS * fs / 1000)   # 200 samples
        step_samples   = int(INFERENCE_STEP_MS   * fs / 1000)   # 50 samples
        self._win_len  = win_samples
        self._step_len = step_samples
        self._buf_len  = win_samples + chunk_size                # generous margin
        self._ch1_buf  = collections.deque(maxlen=self._buf_len)
        self._ch2_buf  = collections.deque(maxlen=self._buf_len)
        self._sample_count = 0

        # Rolling RMS for display
        self._rms_ch1 = RollingRMS(window_samples=win_samples)
        self._rms_ch2 = RollingRMS(window_samples=win_samples)

        # Latency tracking
        self._latency_ms = 0.0

        self._thread = threading.Thread(target=self._run, daemon=True, name="InferencePipeline")

    def start(self):
        self._stop.clear()
        self._thread.start()

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            t_tick = time.perf_counter()

            # 1. Acquire
            chunk, _true_state = self._simulator.read()   # (chunk_size, 2)

            # 2. Filter
            filtered = self._filter.process(chunk)        # (chunk_size, 2)

            # 3. Rolling RMS values for UI oscilloscope
            rms_ch1 = self._rms_ch1.update(filtered[:, 0])
            rms_ch2 = self._rms_ch2.update(filtered[:, 1])

            # 4. Accumulate into inference buffer
            self._ch1_buf.extend(filtered[:, 0].tolist())
            self._ch2_buf.extend(filtered[:, 1].tolist())
            self._sample_count += self.chunk_size

            # 5. Inference every step_samples
            pred_class  = 0
            confidence  = 0.0
            proba       = np.zeros(3)

            if self._sample_count >= self._win_len:
                t_inf = time.perf_counter()
                ch1_arr = np.array(list(self._ch1_buf)[-self._win_len:], dtype=np.float32)
                ch2_arr = np.array(list(self._ch2_buf)[-self._win_len:], dtype=np.float32)
                window  = np.stack([ch1_arr, ch2_arr], axis=1)
                feats   = extract_features(window).reshape(1, -1)

                proba_arr  = self.model.predict_proba(feats)[0]
                pred_class = int(np.argmax(proba_arr))
                confidence = float(proba_arr[pred_class])
                proba      = proba_arr
                self._latency_ms = (time.perf_counter() - t_inf) * 1000.0

            # 6. Push result to UI queue
            result = {
                "raw_ch1"    : filtered[:, 0].copy(),
                "raw_ch2"    : filtered[:, 1].copy(),
                "rms_ch1"    : rms_ch1,
                "rms_ch2"    : rms_ch2,
                "pred_class" : pred_class,
                "confidence" : confidence,
                "proba"      : proba,
                "latency_ms" : self._latency_ms,
                "timestamp"  : time.time(),
            }

            try:
                self.result_queue.put_nowait(result)
            except queue.Full:
                # Drop oldest to stay real-time
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result)
                except queue.Empty:
                    pass

            # 7. Real-time pacing: sleep remainder of chunk period
            tick_dur = self.chunk_size / self.fs
            elapsed  = time.perf_counter() - t_tick
            sleep_t  = tick_dur - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)


# ══════════════════════════════════════════════════════════════════════════════
# CANVAS OSCILLOSCOPE WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class OscilloscopeCanvas(tk.Canvas):
    """
    Rolling waveform display for one EMG channel.
    Draws the signal trace and a filled RMS energy bar below.
    """

    def __init__(self, parent, color: str, label: str,
                 display_seconds: float = DISPLAY_SECONDS, fs: int = FS,
                 **kwargs):
        kwargs.setdefault("bg", PAL["surface"])
        kwargs.setdefault("highlightthickness", 1)
        kwargs.setdefault("highlightbackground", PAL["border"])
        super().__init__(parent, **kwargs)

        self._color   = color
        self._label   = label
        self._fs      = fs
        self._n_pts   = int(display_seconds * fs)
        self._buffer  = collections.deque(
            [0.0] * self._n_pts, maxlen=self._n_pts
        )
        self._rms_val = 0.0
        self._scale   = 1.5       # amplitude scale factor

        self.bind("<Configure>", self._on_resize)
        self._w = 1
        self._h = 1

    def _on_resize(self, event):
        self._w = event.width
        self._h = event.height
        self._redraw()

    def push(self, samples: np.ndarray, rms_val: float):
        self._buffer.extend(samples.tolist())
        self._rms_val = rms_val
        self._redraw()

    def _redraw(self):
        w, h = self._w, self._h
        if w < 2 or h < 2:
            return

        self.delete("all")

        # Background
        self.create_rectangle(0, 0, w, h, fill=PAL["surface"], outline="")

        # Grid lines
        n_grid = 5
        for i in range(1, n_grid):
            y = int(h * i / n_grid)
            self.create_line(0, y, w, y, fill=PAL["border"], dash=(3, 6))
        mid_y = h // 2
        self.create_line(0, mid_y, w, mid_y, fill=PAL["border"], dash=(4, 4))

        # Waveform trace
        buf  = list(self._buffer)
        n    = len(buf)
        if n < 2:
            return

        waveform_h = int(h * 0.72)    # top 72% for waveform
        rms_zone_y = waveform_h + 8   # RMS bar starts below

        mid = waveform_h // 2
        scale = mid / (self._scale + 1e-9)

        pts = []
        for i, v in enumerate(buf):
            x = int(i * w / n)
            y = int(mid - v * scale)
            y = max(2, min(waveform_h - 2, y))
            pts.extend([x, y])

        if len(pts) >= 4:
            # Glow effect — draw slightly thicker in a dimmed colour first
            glow_pts = pts
            self.create_line(*glow_pts, fill=self._color + "40",
                             width=4, smooth=True)
            self.create_line(*pts, fill=self._color, width=1.5, smooth=True)

        # Channel label
        self.create_text(10, 8, anchor="nw", text=self._label,
                         fill=self._color, font=("Courier New", 9, "bold"))

        # RMS Energy bar
        bar_h     = 12
        bar_pad_x = 8
        bar_w_max = w - bar_pad_x * 2
        rms_norm  = min(1.0, self._rms_val / self._scale)
        bar_w_fill = int(bar_w_max * rms_norm)

        # Background track
        self.create_rectangle(
            bar_pad_x, rms_zone_y,
            bar_pad_x + bar_w_max, rms_zone_y + bar_h,
            fill=PAL["bar_bg"], outline=PAL["border"]
        )
        # Fill
        if bar_w_fill > 0:
            bar_col = (PAL["bar_ok"]   if rms_norm < 0.5 else
                       PAL["bar_warn"] if rms_norm < 0.8 else
                       PAL["red"])
            self.create_rectangle(
                bar_pad_x, rms_zone_y,
                bar_pad_x + bar_w_fill, rms_zone_y + bar_h,
                fill=bar_col, outline=""
            )

        rms_label = f"RMS: {self._rms_val:.4f}"
        self.create_text(bar_pad_x + 4, rms_zone_y + bar_h // 2,
                         anchor="w", text=rms_label,
                         fill=PAL["bg"], font=("Courier New", 8, "bold"))


# ══════════════════════════════════════════════════════════════════════════════
# CONFIDENCE METER WIDGET
# ══════════════════════════════════════════════════════════════════════════════

class ConfidenceMeter(tk.Canvas):
    """Horizontal progress bar showing live prediction confidence."""

    def __init__(self, parent, **kwargs):
        kwargs.setdefault("height", 28)
        kwargs.setdefault("bg", PAL["surface"])
        kwargs.setdefault("highlightthickness", 0)
        super().__init__(parent, **kwargs)
        self._confidence = 0.0
        self._pred_class = 0
        self.bind("<Configure>", lambda e: self._redraw())

    def update_confidence(self, confidence: float, pred_class: int):
        self._confidence = confidence
        self._pred_class = pred_class
        self._redraw()

    def _redraw(self):
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 2:
            return
        self.delete("all")
        self.create_rectangle(0, 0, w, h, fill=PAL["surface"], outline="")

        # Track
        pad = 6
        bar_h = h - pad * 2
        self.create_rectangle(pad, pad, w - pad, pad + bar_h,
                              fill=PAL["bar_bg"], outline=PAL["border"])

        # Threshold marker at 85%
        thresh_x = int(pad + (w - pad * 2) * CONFIDENCE_THRESH)
        self.create_line(thresh_x, pad - 2, thresh_x, h - pad + 2,
                         fill=PAL["yellow"], width=2, dash=(4, 3))
        self.create_text(thresh_x + 3, pad, anchor="nw",
                         text="85%", fill=PAL["yellow"],
                         font=("Courier New", 7))

        # Fill
        fill_w = int((w - pad * 2) * self._confidence)
        if fill_w > 0:
            above_thresh = self._confidence >= CONFIDENCE_THRESH
            col = PAL["bar_ok"] if above_thresh else PAL["bar_fg"]
            self.create_rectangle(pad, pad, pad + fill_w, pad + bar_h,
                                  fill=col, outline="")

        # Label
        cls = CLASS_NAMES[self._pred_class] if self._pred_class < 3 else "?"
        label = f"CONFIDENCE  {self._confidence * 100:.1f}%  →  {cls}"
        self.create_text(w // 2, h // 2, anchor="center",
                         text=label, fill=PAL["text"],
                         font=("Courier New", 9, "bold"))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class SilentSpeechApp:

    # ── INIT ──────────────────────────────────────────────────────────────────

    def __init__(self):
        self._model    = None
        self._pipeline = None
        self._q        = queue.Queue(maxsize=60)
        self._running  = False
        self._macros   = dict(DEFAULT_MACROS)
        self._executor = MacroExecutor(self._macros)
        self._last_gesture_class = 0
        self._last_gesture_time  = 0.0
        self._gesture_cooldown   = 0.8   # seconds between triggers

        # Tkinter root
        self.root = tk.Tk()
        self.root.title("Silent Speech  ─  TinyML Assistive Interface")
        self.root.configure(bg=PAL["bg"])
        self.root.minsize(1080, 680)
        self.root.geometry("1200x740")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_fonts()
        self._build_ui()
        self._status_msg("Initialising… Loading model.")
        self.root.after(200, self._auto_init)

    # ── FONT SETUP ────────────────────────────────────────────────────────────

    def _build_fonts(self):
        self.FONT_MONO_SM  = ("Courier New", 9)
        self.FONT_MONO_MD  = ("Courier New", 11)
        self.FONT_MONO_LG  = ("Courier New", 14, "bold")
        self.FONT_MONO_XL  = ("Courier New", 22, "bold")
        self.FONT_SANS_SM  = ("Helvetica", 9)
        self.FONT_SANS_MD  = ("Helvetica", 11)
        self.FONT_SANS_LG  = ("Helvetica", 13, "bold")
        self.FONT_TITLE    = ("Helvetica", 15, "bold")

    # ── UI CONSTRUCTION ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = self.root

        # ── Top menu bar ──────────────────────────────────────────────────────
        menubar = tk.Menu(root, bg=PAL["surface2"], fg=PAL["text"],
                          activebackground=PAL["cyan"], activeforeground=PAL["bg"],
                          borderwidth=0)
        root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0,
                            bg=PAL["surface2"], fg=PAL["text"],
                            activebackground=PAL["cyan_dim"],
                            activeforeground=PAL["text"])
        file_menu.add_command(label="Generate Dataset (60s)",
                              command=self._menu_generate_dataset)
        file_menu.add_command(label="Train Model",
                              command=self._menu_train_model)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        menubar.add_cascade(label="File", menu=file_menu)

        settings_menu = tk.Menu(menubar, tearoff=0,
                                bg=PAL["surface2"], fg=PAL["text"],
                                activebackground=PAL["cyan_dim"],
                                activeforeground=PAL["text"])
        settings_menu.add_command(label="Configure Macro Mappings",
                                  command=self._open_macro_settings)
        settings_menu.add_command(label="Toggle Keyboard Output",
                                  command=self._toggle_keyboard)
        menubar.add_cascade(label="Settings", menu=settings_menu)

        help_menu = tk.Menu(menubar, tearoff=0,
                            bg=PAL["surface2"], fg=PAL["text"],
                            activebackground=PAL["cyan_dim"],
                            activeforeground=PAL["text"])
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)

        # ── Outer frame ───────────────────────────────────────────────────────
        outer = tk.Frame(root, bg=PAL["bg"])
        outer.pack(fill="both", expand=True, padx=12, pady=(6, 10))

        # ══════════════════════════════════════════════════
        # TOP PANEL — System Status
        # ══════════════════════════════════════════════════
        top = self._make_panel(outer, "◈  SYSTEM STATUS", side="top",
                               height=90, fill="x", expand=False)

        status_row = tk.Frame(top, bg=PAL["surface"])
        status_row.pack(fill="x", padx=14, pady=8)

        # Connection status LED + label
        self._conn_led = tk.Canvas(status_row, width=14, height=14,
                                   bg=PAL["surface"], highlightthickness=0)
        self._conn_led.pack(side="left", padx=(0, 5))
        self._conn_oval = self._conn_led.create_oval(1, 1, 13, 13, fill=PAL["red"])

        self._conn_label = tk.Label(status_row,
                                    text="DISCONNECTED",
                                    font=self.FONT_MONO_MD,
                                    fg=PAL["red"], bg=PAL["surface"])
        self._conn_label.pack(side="left", padx=(0, 30))

        # Model version
        tk.Label(status_row, text="MODEL:", font=self.FONT_MONO_SM,
                 fg=PAL["text_dim"], bg=PAL["surface"]).pack(side="left")
        self._model_ver_label = tk.Label(status_row, text="─ not loaded ─",
                                          font=self.FONT_MONO_MD,
                                          fg=PAL["yellow"], bg=PAL["surface"])
        self._model_ver_label.pack(side="left", padx=(4, 30))

        # Latency
        tk.Label(status_row, text="INFERENCE:", font=self.FONT_MONO_SM,
                 fg=PAL["text_dim"], bg=PAL["surface"]).pack(side="left")
        self._latency_label = tk.Label(status_row, text="─ ms",
                                        font=self.FONT_MONO_MD,
                                        fg=PAL["cyan"], bg=PAL["surface"])
        self._latency_label.pack(side="left", padx=(4, 30))

        # FPS / throughput
        tk.Label(status_row, text="THROUGHPUT:", font=self.FONT_MONO_SM,
                 fg=PAL["text_dim"], bg=PAL["surface"]).pack(side="left")
        self._fps_label = tk.Label(status_row, text="─ Hz",
                                    font=self.FONT_MONO_MD,
                                    fg=PAL["cyan"], bg=PAL["surface"])
        self._fps_label.pack(side="left", padx=(4, 0))

        # Start / Stop button (right-aligned)
        self._start_btn = tk.Button(
            status_row, text="▶  START",
            font=self.FONT_SANS_LG,
            bg=PAL["green"], fg=PAL["bg"],
            activebackground=PAL["cyan"], activeforeground=PAL["bg"],
            relief="flat", padx=16, pady=4,
            cursor="hand2", command=self._toggle_pipeline
        )
        self._start_btn.pack(side="right", padx=6)

        # Status message bar
        self._status_var = tk.StringVar(value="Ready.")
        self._status_bar = tk.Label(
            top, textvariable=self._status_var,
            font=self.FONT_MONO_SM,
            fg=PAL["text_dim"], bg=PAL["surface"],
            anchor="w"
        )
        self._status_bar.pack(fill="x", padx=14, pady=(0, 4))

        # ══════════════════════════════════════════════════
        # CENTER PANEL — Oscilloscope + Confidence
        # ══════════════════════════════════════════════════
        center = self._make_panel(outer, "◈  BIOMEDICAL SIGNAL OSCILLOSCOPE",
                                  side="top", fill="both", expand=True)

        osc_frame = tk.Frame(center, bg=PAL["surface"])
        osc_frame.pack(fill="both", expand=True, padx=10, pady=(4, 6))
        osc_frame.columnconfigure(0, weight=1)
        osc_frame.columnconfigure(1, weight=1)
        osc_frame.rowconfigure(0, weight=1)

        # Channel 1
        ch1_frame = tk.Frame(osc_frame, bg=PAL["surface2"],
                             highlightbackground=PAL["border"],
                             highlightthickness=1)
        ch1_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self._osc_ch1 = OscilloscopeCanvas(
            ch1_frame, color=PAL["ch1"],
            label="CH1 — ELECTRODE A  (Flexor Digitorum)",
            bg=PAL["surface2"]
        )
        self._osc_ch1.pack(fill="both", expand=True, padx=2, pady=2)

        # Channel 2
        ch2_frame = tk.Frame(osc_frame, bg=PAL["surface2"],
                             highlightbackground=PAL["border"],
                             highlightthickness=1)
        ch2_frame.grid(row=0, column=1, sticky="nsew")
        self._osc_ch2 = OscilloscopeCanvas(
            ch2_frame, color=PAL["ch2"],
            label="CH2 — ELECTRODE B  (Masseter / Jaw)",
            bg=PAL["surface2"]
        )
        self._osc_ch2.pack(fill="both", expand=True, padx=2, pady=2)

        # Confidence meter
        conf_frame = tk.Frame(center, bg=PAL["surface"])
        conf_frame.pack(fill="x", padx=10, pady=(0, 6))
        tk.Label(conf_frame, text="CONFIDENCE THRESHOLD METER",
                 font=self.FONT_MONO_SM, fg=PAL["text_dim"],
                 bg=PAL["surface"]).pack(anchor="w", pady=(0, 2))
        self._conf_meter = ConfidenceMeter(conf_frame, bg=PAL["surface"])
        self._conf_meter.pack(fill="x")

        # ══════════════════════════════════════════════════
        # BOTTOM PANEL — Action Feed
        # ══════════════════════════════════════════════════
        bottom = self._make_panel(outer, "◈  ACTION FEED  — GESTURE OUTPUT",
                                  side="top", fill="x", expand=False,
                                  height=190)

        feed_inner = tk.Frame(bottom, bg=PAL["surface"])
        feed_inner.pack(fill="both", expand=True, padx=10, pady=6)
        feed_inner.columnconfigure(0, weight=1)
        feed_inner.columnconfigure(1, weight=0)

        # Main action label
        self._action_frame = tk.Frame(feed_inner, bg=PAL["surface2"],
                                       highlightbackground=PAL["border"],
                                       highlightthickness=1)
        self._action_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._action_var = tk.StringVar(value="[  STANDBY  ]")
        self._action_label = tk.Label(
            self._action_frame,
            textvariable=self._action_var,
            font=self.FONT_MONO_XL,
            fg=PAL["text_dim"], bg=PAL["surface2"],
            anchor="center"
        )
        self._action_label.pack(fill="both", expand=True, padx=10, pady=10)

        # Probability bars (right column)
        prob_frame = tk.Frame(feed_inner, bg=PAL["surface"],
                              width=200)
        prob_frame.grid(row=0, column=1, sticky="nsew")
        prob_frame.grid_propagate(False)
        tk.Label(prob_frame, text="CLASS PROBABILITIES",
                 font=self.FONT_MONO_SM, fg=PAL["text_dim"],
                 bg=PAL["surface"]).pack(anchor="w", pady=(0, 4))

        self._prob_bars  = []
        self._prob_vars  = []
        self._prob_labels = []
        for i, cname in enumerate(CLASS_NAMES):
            row_f = tk.Frame(prob_frame, bg=PAL["surface"])
            row_f.pack(fill="x", pady=2)
            col = [PAL["text_dim"], PAL["ch1"], PAL["ch2"]][i]
            tk.Label(row_f, text=f"{cname:8s}", font=self.FONT_MONO_SM,
                     fg=col, bg=PAL["surface"], width=8,
                     anchor="w").pack(side="left")
            var = tk.DoubleVar(value=0.0)
            self._prob_vars.append(var)
            bar = ttk.Progressbar(row_f, variable=var, maximum=100,
                                   length=100, mode="determinate")
            bar.pack(side="left", padx=4)
            self._prob_bars.append(bar)
            lbl = tk.Label(row_f, text="0.0%", font=self.FONT_MONO_SM,
                           fg=col, bg=PAL["surface"], width=6)
            lbl.pack(side="left")
            self._prob_labels.append(lbl)

        # Event log
        log_frame = tk.Frame(feed_inner, bg=PAL["surface"])
        log_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        feed_inner.rowconfigure(1, weight=0)
        tk.Label(log_frame, text="EVENT LOG", font=self.FONT_MONO_SM,
                 fg=PAL["text_dim"], bg=PAL["surface"]).pack(anchor="w")
        self._log_text = tk.Text(
            log_frame, height=3,
            font=self.FONT_MONO_SM,
            bg=PAL["surface2"], fg=PAL["text"],
            insertbackground=PAL["cyan"],
            selectbackground=PAL["cyan_dim"],
            relief="flat", state="disabled"
        )
        self._log_text.pack(fill="x")
        self._log_text.tag_config("gesture", foreground=PAL["cyan"])
        self._log_text.tag_config("warn",    foreground=PAL["yellow"])
        self._log_text.tag_config("info",    foreground=PAL["text_dim"])

        # Style progress bars
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TProgressbar",
                        troughcolor=PAL["bar_bg"],
                        background=PAL["cyan"],
                        bordercolor=PAL["border"],
                        lightcolor=PAL["cyan"],
                        darkcolor=PAL["cyan"])

        self._keyboard_enabled = True
        self._fps_counter = collections.deque(maxlen=30)

    def _make_panel(self, parent, title: str, side: str = "top",
                    fill: str = "x", expand: bool = False,
                    height: int = 0) -> tk.Frame:
        """Create a titled dark panel."""
        wrapper = tk.Frame(parent, bg=PAL["bg"])
        if height:
            wrapper.configure(height=height)
        wrapper.pack(side=side, fill=fill, expand=expand, pady=(0, 8))

        header = tk.Frame(wrapper, bg=PAL["bg"])
        header.pack(fill="x")
        tk.Label(header, text=title,
                 font=self.FONT_SANS_LG,
                 fg=PAL["cyan"], bg=PAL["bg"]).pack(side="left", pady=(4, 2))
        tk.Frame(header, bg=PAL["border"], height=1).pack(
            side="left", fill="x", expand=True, padx=8, pady=8)

        inner = tk.Frame(wrapper,
                         bg=PAL["surface"],
                         highlightbackground=PAL["border"],
                         highlightthickness=1)
        inner.pack(fill=fill, expand=expand)
        return inner

    # ── AUTO-INIT ─────────────────────────────────────────────────────────────

    def _auto_init(self):
        """Try to load model; if absent, guide user to generate data + train."""
        if os.path.exists(MODEL_FILE):
            self._load_model_file()
        else:
            self._status_msg("No model found. Generating dataset and training…")
            self._log("No emg_model.pkl found — starting auto-setup.", "warn")
            threading.Thread(target=self._auto_setup, daemon=True).start()

    def _auto_setup(self):
        """Background: generate dataset → train → load model."""
        try:
            self._log("Generating 60-second EMG dataset…", "info")
            generate_dataset(duration_secs=60, verbose=False)
            self._log("Dataset generated. Training Random Forest…", "info")
            train(verbose=False)
            self.root.after(0, self._load_model_file)
        except Exception as e:
            self.root.after(0, lambda: self._status_msg(f"Setup error: {e}"))

    def _load_model_file(self):
        try:
            model, meta = load_model(MODEL_FILE)
            self._model = model
            acc = meta.get("metadata", {}).get("test_accuracy", 0)
            ver = meta.get("metadata", {}).get("trained_at", MODEL_VERSION)[:10]
            self._model_ver_label.config(
                text=f"{MODEL_VERSION}  ({acc*100:.1f}% acc)  [{ver}]",
                fg=PAL["green"]
            )
            self._status_msg(f"Model loaded. Test accuracy: {acc*100:.1f}%")
            self._log(f"Model loaded — accuracy {acc*100:.1f}%", "info")
            self._set_connected(True)
        except Exception as e:
            self._status_msg(f"Model load error: {e}")
            self._log(f"Model load error: {e}", "warn")

    # ── PIPELINE CONTROL ──────────────────────────────────────────────────────

    def _toggle_pipeline(self):
        if self._running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self):
        if self._model is None:
            messagebox.showwarning("No Model",
                "Model not loaded. Please train a model first (File → Train Model).")
            return
        self._pipeline = InferencePipeline(self._model, self._q)
        self._pipeline.start()
        self._running = True
        self._start_btn.config(text="■  STOP", bg=PAL["red"])
        self._status_msg("Pipeline running — streaming 2-ch EMG @ 1000 Hz")
        self._log("Pipeline started.", "info")
        self._set_connected(True)
        self.root.after(50, self._ui_update_loop)

    def _stop_pipeline(self):
        self._running = False
        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None
        self._start_btn.config(text="▶  START", bg=PAL["green"])
        self._status_msg("Pipeline stopped.")
        self._log("Pipeline stopped.", "info")
        self._set_connected(False)

    # ── UI UPDATE LOOP ────────────────────────────────────────────────────────

    def _ui_update_loop(self):
        if not self._running:
            return

        # Drain up to 4 results from queue per tick
        processed = 0
        latest_result = None
        while processed < 4:
            try:
                result = self._q.get_nowait()
                latest_result = result
                processed += 1
            except queue.Empty:
                break

        if latest_result is not None:
            self._apply_result(latest_result)

        self.root.after(33, self._ui_update_loop)   # ~30 FPS UI refresh

    def _apply_result(self, r: dict):
        # Oscilloscope
        self._osc_ch1.push(r["raw_ch1"], r["rms_ch1"])
        self._osc_ch2.push(r["raw_ch2"], r["rms_ch2"])

        # Confidence meter
        self._conf_meter.update_confidence(r["confidence"], r["pred_class"])

        # Probability bars
        proba = r["proba"]
        for i in range(min(3, len(proba))):
            self._prob_vars[i].set(proba[i] * 100)
            self._prob_labels[i].config(text=f"{proba[i]*100:.1f}%")

        # Latency
        self._latency_label.config(text=f"{r['latency_ms']:.1f} ms")

        # FPS tracker
        self._fps_counter.append(r["timestamp"])
        if len(self._fps_counter) >= 2:
            span = self._fps_counter[-1] - self._fps_counter[0]
            if span > 0:
                fps = (len(self._fps_counter) - 1) / span
                self._fps_label.config(text=f"{fps:.0f} Hz")

        # Gesture detection
        pred  = r["pred_class"]
        conf  = r["confidence"]
        now   = time.time()

        if (pred != 0 and
                conf >= CONFIDENCE_THRESH and
                (now - self._last_gesture_time) > self._gesture_cooldown):
            self._last_gesture_time  = now
            self._last_gesture_class = pred
            self._trigger_gesture(pred, conf)
        else:
            action_text = ACTION_LABELS.get(pred, f"[CLASS {pred}]")
            if conf < CONFIDENCE_THRESH:
                action_text = "[  REST  ]"
            self._action_var.set(action_text)
            self._action_label.config(fg=PAL["text_dim"],
                                       bg=PAL["surface2"])
            self._action_frame.config(bg=PAL["surface2"])

    def _trigger_gesture(self, gesture_class: int, confidence: float):
        """Flash UI and fire keyboard macro."""
        action_text = ACTION_LABELS.get(gesture_class, f"[CLASS {gesture_class}]")
        self._action_var.set(action_text)
        self._action_label.config(fg=PAL["bg"], bg=PAL["cyan"])
        self._action_frame.config(bg=PAL["cyan"])
        cls_name = CLASS_NAMES[gesture_class]
        self._log(
            f"GESTURE: {cls_name}  ({confidence*100:.1f}% confidence)  "
            f"→ macro key: [{self._macros.get(gesture_class, 'none')}]",
            "gesture"
        )
        if self._keyboard_enabled:
            self._executor.fire(gesture_class)

        # Schedule flash-off
        self.root.after(FLASH_DURATION_MS, self._flash_off)

    def _flash_off(self):
        self._action_label.config(fg=PAL["text_dim"], bg=PAL["surface2"])
        self._action_frame.config(bg=PAL["surface2"])
        self._action_var.set("[  STANDBY  ]")

    # ── STATUS & LOGGING ──────────────────────────────────────────────────────

    def _status_msg(self, msg: str):
        self._status_var.set(f"  {msg}")

    def _set_connected(self, connected: bool):
        if connected:
            self._conn_led.itemconfig(self._conn_oval, fill=PAL["green"])
            self._conn_label.config(text="CONNECTED", fg=PAL["green"])
        else:
            self._conn_led.itemconfig(self._conn_oval, fill=PAL["red"])
            self._conn_label.config(text="DISCONNECTED", fg=PAL["red"])

    def _log(self, message: str, tag: str = "info"):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}]  {message}\n"
        self._log_text.config(state="normal")
        self._log_text.insert("end", line, tag)
        self._log_text.see("end")
        # Keep last 200 lines
        lines = int(self._log_text.index("end-1c").split(".")[0])
        if lines > 200:
            self._log_text.delete("1.0", "50.0")
        self._log_text.config(state="disabled")

    # ── MENU CALLBACKS ────────────────────────────────────────────────────────

    def _menu_generate_dataset(self):
        if self._running:
            messagebox.showwarning("Running",
                "Stop the pipeline before generating a new dataset.")
            return
        self._status_msg("Generating 60-second dataset…")
        self._log("Starting dataset generation (60s)…", "info")
        threading.Thread(target=self._bg_generate, daemon=True).start()

    def _bg_generate(self):
        try:
            generate_dataset(duration_secs=60, verbose=False)
            self.root.after(0, lambda: self._status_msg(
                f"Dataset saved → {DATASET_FILE}"))
            self.root.after(0, lambda: self._log(
                f"Dataset saved: {DATASET_FILE}", "info"))
        except Exception as e:
            self.root.after(0, lambda: self._status_msg(f"Generate error: {e}"))

    def _menu_train_model(self):
        if self._running:
            messagebox.showwarning("Running",
                "Stop the pipeline before training.")
            return
        if not os.path.exists(DATASET_FILE):
            messagebox.showwarning("No Dataset",
                f"Dataset file '{DATASET_FILE}' not found.\n"
                "Use File → Generate Dataset first.")
            return
        self._status_msg("Training model…")
        self._log("Starting model training…", "info")
        threading.Thread(target=self._bg_train, daemon=True).start()

    def _bg_train(self):
        try:
            train(verbose=False)
            self.root.after(0, self._load_model_file)
            self.root.after(0, lambda: self._log("Training complete.", "info"))
        except Exception as e:
            self.root.after(0, lambda: self._status_msg(f"Train error: {e}"))

    def _toggle_keyboard(self):
        self._keyboard_enabled = not self._keyboard_enabled
        state = "ENABLED" if self._keyboard_enabled else "DISABLED"
        self._status_msg(f"Keyboard macro output: {state}")
        self._log(f"Keyboard output {state}", "warn" if not self._keyboard_enabled else "info")

    # ── MACRO SETTINGS DIALOG ─────────────────────────────────────────────────

    def _open_macro_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Macro Key Mapping")
        win.configure(bg=PAL["bg"])
        win.geometry("420x320")
        win.resizable(False, False)
        win.grab_set()

        tk.Label(win, text="◈  GESTURE → KEYBOARD MACRO MAPPING",
                 font=self.FONT_MONO_MD, fg=PAL["cyan"],
                 bg=PAL["bg"]).pack(pady=(16, 4))

        available_keys = ["enter", "space", "tab", "esc",
                          "up", "down", "left", "right", "backspace",
                          "a", "b", "c", "d", "e", "f", "g", "h",
                          "i", "j", "k", "l", "m", "n"]

        entries = {}
        for cls_id in [1, 2]:
            frame = tk.Frame(win, bg=PAL["surface2"],
                             highlightbackground=PAL["border"],
                             highlightthickness=1)
            frame.pack(fill="x", padx=20, pady=6)
            label_text = f"  Gesture {cls_id} ({CLASS_NAMES[cls_id]:8s})  →  "
            tk.Label(frame, text=label_text, font=self.FONT_MONO_MD,
                     fg=PAL["text"], bg=PAL["surface2"]).pack(side="left",
                     padx=8, pady=8)
            var = tk.StringVar(value=self._macros.get(cls_id, ""))
            combo = ttk.Combobox(frame, textvariable=var,
                                  values=available_keys, width=14,
                                  font=self.FONT_MONO_SM)
            combo.pack(side="left", padx=8)
            entries[cls_id] = var

        tk.Label(win,
                 text="Type any key name or choose from list.\n"
                      "Supported: enter, space, tab, esc, up/down/left/right,\n"
                      "backspace, or any single letter (a–z).",
                 font=self.FONT_SANS_SM, fg=PAL["text_dim"],
                 bg=PAL["bg"], justify="left").pack(padx=20, pady=6)

        def save_and_close():
            for cls_id, var in entries.items():
                key_name = var.get().strip().lower()
                if key_name:
                    self._macros[cls_id] = key_name
                    self._executor.update_mapping(cls_id, key_name)
            self._log(f"Macros updated: {self._macros}", "info")
            win.destroy()

        btn_frame = tk.Frame(win, bg=PAL["bg"])
        btn_frame.pack(pady=12)
        tk.Button(btn_frame, text="  Save  ",
                  font=self.FONT_SANS_MD,
                  bg=PAL["cyan"], fg=PAL["bg"],
                  relief="flat", padx=14, pady=4,
                  cursor="hand2", command=save_and_close).pack(side="left", padx=6)
        tk.Button(btn_frame, text="  Cancel  ",
                  font=self.FONT_SANS_MD,
                  bg=PAL["surface2"], fg=PAL["text"],
                  relief="flat", padx=14, pady=4,
                  cursor="hand2", command=win.destroy).pack(side="left", padx=6)

    # ── ABOUT ─────────────────────────────────────────────────────────────────

    def _show_about(self):
        msg = (
            "Silent Speech\n"
            "An End-to-End TinyML Assistive Interface\n\n"
            f"Pipeline:  2-ch EMG @ 1000 Hz\n"
            f"Filter:    Butterworth BP (20-450 Hz) + 50 Hz Notch\n"
            f"Features:  MAV · RMS · WL  |  Window: 200ms / Step: 50ms\n"
            f"Classifier: Random Forest  (n=150 trees)\n"
            f"Threshold:  {CONFIDENCE_THRESH*100:.0f}% confidence\n\n"
            f"pynput keyboard control: {'available' if _PYNPUT else 'NOT INSTALLED'}\n"
            f"CustomTkinter:           {'available' if _CTK    else 'NOT INSTALLED'}"
        )
        messagebox.showinfo("About Silent Speech", msg)

    # ── CLOSE ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        if self._running:
            self._stop_pipeline()
        self.root.destroy()

    # ── MAIN LOOP ─────────────────────────────────────────────────────────────

    def run(self):
        self.root.mainloop()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 64)
    print("  Silent Speech — TinyML Assistive Interface")
    print("=" * 64)
    if not _PYNPUT:
        print("  [WARN] pynput not installed — keyboard macros disabled.")
        print("         Install with: pip install pynput")
    if not _CTK:
        print("  [INFO] customtkinter not installed — using styled Tkinter.")
        print("         Install with: pip install customtkinter")
    print("  Starting GUI…")
    print()
    app = SilentSpeechApp()
    app.run()


if __name__ == "__main__":
    main()
