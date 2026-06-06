"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SILENT SPEECH: An End-to-End TinyML Assistive Interface            ║
║          Principal Product Engineer & Neurotechnology Solution Architect    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Dependencies:
    pip install numpy scipy scikit-learn matplotlib

Run:
    python app.py
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import queue
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy import signal as scipy_signal
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as mpatches

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & THEME
# ─────────────────────────────────────────────────────────────────────────────
FS               = 1000          # Sampling rate Hz
WINDOW_SAMPLES   = 200           # 200ms rolling window for features
INFERENCE_MS     = 50            # Inference every 50ms
CONFIDENCE_THRESH= 0.85          # 85% confidence gate
MODEL_PATH       = "emg_model.pkl"
BUFFER_SIZE      = 1000          # Samples kept in display ring buffer

# Palette
BG_DEEP          = "#1E1E24"
BG_MID           = "#2A2A33"
BG_PANEL         = "#16161C"
BG_CARD          = "#222229"
CYAN             = "#00F5D4"
CYAN_DIM         = "#007A6A"
WHITE            = "#FFFFFF"
GREY_LIGHT       = "#B0B0C0"
GREY_MID         = "#6B6B80"
RED_ALERT        = "#FF4560"
AMBER            = "#FFB020"
GREEN_OK         = "#00E396"

LABEL_MAP = {0: "[  REST  ]", 1: "[ SELECT ]", 2: "[NEXT PAGE]"}
COLOR_MAP  = {0: GREY_MID,    1: CYAN,        2: RED_ALERT}
ICON_MAP   = {0: "●",         1: "▶",         2: "⚡"}

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC sEMG DATA SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────
class SyntheticEMGSimulator:
    """
    Generates physiologically plausible 2-channel surface EMG signals.

    State 0 – Rest:       Low-amplitude Gaussian baseline noise (~5 µV RMS)
    State 1 – Jaw Clench: Heavy burst, dominant 20-80 Hz, high amplitude (~120 µV RMS)
    State 2 – Finger Flick: Sharp transient, dominant 100-300 Hz, fast decay (~60 µV RMS)
    """

    def __init__(self, fs=FS):
        self.fs = fs
        self.rng = np.random.default_rng(42)

    def _noise(self, n, scale=1.0):
        return self.rng.normal(0, scale, n)

    def generate_rest(self, n_samples):
        ch1 = self._noise(n_samples, scale=5.0)
        ch2 = self._noise(n_samples, scale=4.5)
        return ch1, ch2

    def generate_jaw_clench(self, n_samples):
        t = np.linspace(0, n_samples / self.fs, n_samples)
        # Low-frequency muscle burst (20-80 Hz) with AM envelope
        envelope = 120.0 * (0.5 + 0.5 * np.sin(np.pi * t / (n_samples / self.fs)))
        carrier1 = (np.sin(2 * np.pi * 35 * t) +
                    0.6 * np.sin(2 * np.pi * 55 * t) +
                    0.4 * np.sin(2 * np.pi * 70 * t))
        carrier2 = (np.sin(2 * np.pi * 30 * t) +
                    0.7 * np.sin(2 * np.pi * 50 * t) +
                    0.3 * np.sin(2 * np.pi * 65 * t))
        ch1 = envelope * carrier1 + self._noise(n_samples, scale=8.0)
        ch2 = envelope * carrier2 * 0.85 + self._noise(n_samples, scale=7.0)
        return ch1, ch2

    def generate_finger_flick(self, n_samples):
        t = np.linspace(0, n_samples / self.fs, n_samples)
        # High-frequency sharp transient (100-300 Hz) with exponential decay
        decay = 60.0 * np.exp(-t / (0.05 + 0.02 * self.rng.random()))
        carrier1 = (np.sin(2 * np.pi * 150 * t) +
                    0.5 * np.sin(2 * np.pi * 230 * t) +
                    0.3 * np.sin(2 * np.pi * 280 * t))
        carrier2 = (np.sin(2 * np.pi * 140 * t) +
                    0.55 * np.sin(2 * np.pi * 210 * t) +
                    0.25 * np.sin(2 * np.pi * 270 * t))
        ch1 = decay * carrier1 + self._noise(n_samples, scale=5.0)
        ch2 = decay * carrier2 * 0.9 + self._noise(n_samples, scale=4.5)
        return ch1, ch2

    def generate_dataset(self, duration_seconds=60, fs=None):
        """
        Generate a labelled 60-second sEMG dataset.
        Returns X (n_samples, 2), y (n_samples,)
        """
        if fs is None:
            fs = self.fs
        total = duration_seconds * fs
        ch1_all, ch2_all, labels = [], [], []

        generators = [self.generate_rest, self.generate_jaw_clench, self.generate_finger_flick]
        n_classes = len(generators)
        chunk = total // (n_classes * 10)  # 10 chunks per class

        for _ in range(10):
            for label, gen in enumerate(generators):
                c1, c2 = gen(chunk)
                ch1_all.append(c1)
                ch2_all.append(c2)
                labels.extend([label] * chunk)

        ch1 = np.concatenate(ch1_all)
        ch2 = np.concatenate(ch2_all)
        X = np.column_stack([ch1, ch2])
        y = np.array(labels)
        return X, y


# ─────────────────────────────────────────────────────────────────────────────
# 2. DSP PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
class DSPPipeline:
    """
    Real-time DSP pipeline:
      1. 4th-order Butterworth Bandpass (20 Hz – 450 Hz)
      2. 2nd-order IIR Notch at 50 Hz (Q=30)
      3. Feature extraction: MAV, RMS, ZCR, WL, VAR per channel
    """

    def __init__(self, fs=FS):
        self.fs = fs
        self._build_filters()
        # Persistent filter states for each channel (zi_bp, zi_notch)
        self.zi_bp   = [scipy_signal.sosfilt_zi(self.sos_bp)   for _ in range(2)]
        self.zi_notch= [scipy_signal.sosfilt_zi(self.sos_notch) for _ in range(2)]

    def _build_filters(self):
        # Bandpass 20-450 Hz, 4th order Butterworth
        nyq = self.fs / 2.0
        low  = 20.0  / nyq
        high = 450.0 / nyq
        self.sos_bp = scipy_signal.butter(4, [low, high], btype='bandpass', output='sos')
        # Notch at 50 Hz
        b_notch, a_notch = scipy_signal.iirnotch(50.0, Q=30, fs=self.fs)
        self.sos_notch = scipy_signal.tf2sos(b_notch, a_notch)

    def _scale_zi(self, sos_zi, x_first_sample):
        """Scale initial conditions to avoid transient startup artifact."""
        return sos_zi * x_first_sample if len(x_first_sample) > 0 else sos_zi

    def process_block(self, raw_ch1, raw_ch2):
        """
        Process a block of raw sEMG samples.
        Returns (filtered_ch1, filtered_ch2)
        """
        filtered = []
        for ch_idx, raw in enumerate([raw_ch1, raw_ch2]):
            if len(raw) == 0:
                filtered.append(raw)
                continue
            # Bandpass
            bp_out, self.zi_bp[ch_idx] = scipy_signal.sosfilt(
                self.sos_bp, raw, zi=self.zi_bp[ch_idx])
            # Notch
            nt_out, self.zi_notch[ch_idx] = scipy_signal.sosfilt(
                self.sos_notch, bp_out, zi=self.zi_notch[ch_idx])
            filtered.append(nt_out)
        return filtered[0], filtered[1]

    @staticmethod
    def extract_features(window_ch1, window_ch2):
        """
        Extract time-domain biomedical features from a window:
          MAV  – Mean Absolute Value
          RMS  – Root Mean Square
          ZCR  – Zero Crossing Rate
          WL   – Waveform Length
          VAR  – Variance
        Returns a 1D feature vector of length 10 (5 features × 2 channels).
        """
        features = []
        for w in [window_ch1, window_ch2]:
            if len(w) < 2:
                features.extend([0.0] * 5)
                continue
            mav = np.mean(np.abs(w))
            rms = np.sqrt(np.mean(w ** 2))
            zcr = np.sum(np.diff(np.sign(w)) != 0) / len(w)
            wl  = np.sum(np.abs(np.diff(w)))
            var = np.var(w)
            features.extend([mav, rms, zcr, wl, var])
        return np.array(features, dtype=np.float32)

    @staticmethod
    def batch_extract_features(X, window=WINDOW_SAMPLES):
        """
        Sliding-window feature extraction over a full signal matrix (n_samples, 2).
        Returns feature matrix and labels aligned to window end.
        """
        n = len(X)
        rows = []
        for i in range(window, n, window // 2):  # 50% overlap
            w1 = X[i - window:i, 0]
            w2 = X[i - window:i, 1]
            rows.append(DSPPipeline.extract_features(w1, w2))
        return np.array(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ML MODEL ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class MLModelEngine:
    """
    Manages the Random Forest classifier lifecycle:
      - Autonomous self-training from synthetic data if no model found
      - Prediction with calibrated probability output
      - Save / Load via pickle
    """

    def __init__(self, model_path=MODEL_PATH):
        self.model_path = model_path
        self.pipeline   = None
        self.n_features = 10
        self.classes    = [0, 1, 2]
        self.train_acc  = 0.0
        self.n_estimators = 0

    def _build_pipeline(self):
        clf = RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', clf)
        ])
        return pipe

    def auto_train(self, status_callback=None):
        """
        Full autonomous training pipeline:
          1. Generate synthetic dataset
          2. Filter & extract features
          3. Train RF classifier
          4. Evaluate & save model
        """
        def log(msg):
            if status_callback:
                status_callback(msg)
            print(f"[AutoTrain] {msg}")

        log("Generating 60-second synthetic sEMG dataset…")
        sim = SyntheticEMGSimulator(fs=FS)
        X_raw, y_raw = sim.generate_dataset(duration_seconds=60)

        log("Applying DSP pipeline to dataset…")
        dsp = DSPPipeline(fs=FS)
        # Filter full signal in chunks to preserve filter state
        chunk = FS  # 1-second chunks
        filtered1, filtered2 = [], []
        for start in range(0, len(X_raw), chunk):
            end = start + chunk
            f1, f2 = dsp.process_block(X_raw[start:end, 0], X_raw[start:end, 1])
            filtered1.append(f1)
            filtered2.append(f2)
        f1_full = np.concatenate(filtered1)
        f2_full = np.concatenate(filtered2)
        X_filt  = np.column_stack([f1_full, f2_full])

        log("Extracting biomedical features (MAV, RMS, ZCR, WL, VAR)…")
        X_feat = DSPPipeline.batch_extract_features(X_filt, window=WINDOW_SAMPLES)
        # Align labels with windows
        window_labels = []
        n = len(X_raw)
        for i in range(WINDOW_SAMPLES, n, WINDOW_SAMPLES // 2):
            window_labels.append(y_raw[i - 1])
        y_feat = np.array(window_labels[:len(X_feat)])

        log(f"Feature matrix shape: {X_feat.shape}, Classes: {np.unique(y_feat)}")

        X_tr, X_te, y_tr, y_te = train_test_split(
            X_feat, y_feat, test_size=0.2, random_state=42, stratify=y_feat)

        log("Training Random Forest classifier (150 estimators)…")
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_tr, y_tr)

        y_pred = self.pipeline.predict(X_te)
        self.train_acc = accuracy_score(y_te, y_pred)
        self.n_estimators = 150
        log(f"Evaluation Accuracy: {self.train_acc * 100:.2f}%")

        report = classification_report(y_te, y_pred,
                                       target_names=["Rest", "Jaw Clench", "Finger Flick"])
        log("Classification Report:\n" + report)

        self.save()
        log(f"Model saved → {self.model_path}")
        return self.train_acc

    def save(self):
        payload = {
            'pipeline':    self.pipeline,
            'train_acc':   self.train_acc,
            'n_estimators': self.n_estimators
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.model_path, 'rb') as f:
            payload = pickle.load(f)
        self.pipeline     = payload['pipeline']
        self.train_acc    = payload['train_acc']
        self.n_estimators = payload.get('n_estimators', 150)
        return True

    def predict_proba(self, feature_vector):
        """Returns (predicted_class, probability_array)."""
        x = feature_vector.reshape(1, -1)
        proba = self.pipeline.predict_proba(x)[0]
        pred  = int(np.argmax(proba))
        return pred, proba

    def is_ready(self):
        return self.pipeline is not None


# ─────────────────────────────────────────────────────────────────────────────
# 4. LIVE DATA STREAMING THREAD
# ─────────────────────────────────────────────────────────────────────────────
class LiveStreamThread(threading.Thread):
    """
    Background thread that continuously generates 2-channel mock sEMG data
    and pushes blocks into a shared queue for the UI and inference engine.

    Cycles through states:
      State 0 (Rest)         – 2.0 seconds
      State 1 (Jaw Clench)   – 1.0 second
      State 0 (Rest)         – 1.5 seconds
      State 2 (Finger Flick) – 0.5 seconds
    """

    CYCLE = [
        (0, 2.0),
        (1, 1.0),
        (0, 1.5),
        (2, 0.5),
    ]
    BLOCK_SAMPLES = FS // 20  # 50ms block at 1000 Hz = 50 samples

    def __init__(self, data_queue, stop_event):
        super().__init__(daemon=True)
        self.data_queue  = data_queue
        self.stop_event  = stop_event
        self.sim         = SyntheticEMGSimulator(fs=FS)
        self.rng         = np.random.default_rng()
        self._cycle_idx  = 0
        self._state_remaining = self.CYCLE[0][1]
        self._current_state   = self.CYCLE[0][0]

    def _next_block(self):
        n = self.BLOCK_SAMPLES
        t_block = n / FS

        if self._state_remaining <= 0:
            self._cycle_idx = (self._cycle_idx + 1) % len(self.CYCLE)
            state, dur = self.CYCLE[self._cycle_idx]
            # Add small jitter to transitions
            self._current_state   = state
            self._state_remaining = dur + self.rng.uniform(-0.1, 0.1)

        self._state_remaining -= t_block

        if self._current_state == 0:
            ch1, ch2 = self.sim.generate_rest(n)
        elif self._current_state == 1:
            ch1, ch2 = self.sim.generate_jaw_clench(n)
        else:
            ch1, ch2 = self.sim.generate_finger_flick(n)

        return ch1, ch2, self._current_state

    def run(self):
        interval = self.BLOCK_SAMPLES / FS  # target period in seconds
        while not self.stop_event.is_set():
            t0 = time.perf_counter()
            ch1, ch2, state = self._next_block()
            try:
                self.data_queue.put_nowait((ch1, ch2, state))
            except queue.Full:
                # Drop oldest if queue is saturated
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    pass
                self.data_queue.put_nowait((ch1, ch2, state))

            elapsed = time.perf_counter() - t0
            sleep_t = max(0.0, interval - elapsed)
            time.sleep(sleep_t)


# ─────────────────────────────────────────────────────────────────────────────
# 5. INFERENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class InferenceEngine:
    """
    Maintains a rolling feature window and runs the ML model every INFERENCE_MS.
    Emits events via callback when confidence > CONFIDENCE_THRESH.
    """

    def __init__(self, model: MLModelEngine, dsp: DSPPipeline,
                 on_prediction=None):
        self.model          = model
        self.dsp            = dsp
        self.on_prediction  = on_prediction
        self._buf_ch1       = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._buf_ch2       = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
        self._last_infer    = 0.0
        self._latency_ms    = 0.0
        self._lock          = threading.Lock()

    def ingest(self, raw_ch1, raw_ch2):
        """Push a new raw block through DSP and append to rolling buffer."""
        f1, f2 = self.dsp.process_block(raw_ch1, raw_ch2)
        with self._lock:
            n = len(f1)
            self._buf_ch1 = np.roll(self._buf_ch1, -n)
            self._buf_ch2 = np.roll(self._buf_ch2, -n)
            self._buf_ch1[-n:] = f1
            self._buf_ch2[-n:] = f2

        now = time.perf_counter()
        if (now - self._last_infer) * 1000 >= INFERENCE_MS:
            self._last_infer = now
            self._run_inference()

    def _run_inference(self):
        if not self.model.is_ready():
            return
        t0 = time.perf_counter()
        with self._lock:
            w1 = self._buf_ch1.copy()
            w2 = self._buf_ch2.copy()
        feat = DSPPipeline.extract_features(w1, w2)
        pred, proba = self.model.predict_proba(feat)
        self._latency_ms = (time.perf_counter() - t0) * 1000

        if self.on_prediction:
            confidence = float(proba[pred])
            self.on_prediction(pred, proba, confidence, self._latency_ms)

    @property
    def latency_ms(self):
        return self._latency_ms


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN APPLICATION WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class SilentSpeechApp:
    """
    Silent Speech – End-to-End TinyML Assistive Interface
    Full Tkinter dark-theme dashboard with three panels:
      ┌─────────────────────────────────────┐
      │   TOP:    SYSTEM STATUS             │
      ├─────────────────────────────────────┤
      │   CENTER: BIOMEDICAL SIGNAL FEED    │
      ├─────────────────────────────────────┤
      │   BOTTOM: INTENT ACCELERATOR FEED   │
      └─────────────────────────────────────┘
    """

    UPDATE_MS         = 50      # UI refresh rate
    SIGNAL_HISTORY    = 500     # Samples shown in signal chart
    FLASH_DURATION_MS = 800     # Notification flash duration
    LATENCY_HISTORY   = 40      # Number of latency readings to average

    def __init__(self, root):
        self.root = root
        self._configure_root()

        # ── State ──────────────────────────────────────────────────────────
        self.is_booting         = True
        self.boot_message       = tk.StringVar(value="Initialising…")
        self.model_ready        = False
        self.stream_active      = False
        self.last_pred_class    = 0
        self.last_confidence    = 0.0
        self.last_latency_ms    = 0.0
        self.latency_history    = []
        self.inference_count    = 0
        self.prediction_log     = []        # [(ts, label, conf), …]
        self._flash_active      = False
        self._flash_job         = None
        self._gesture_counts    = {0: 0, 1: 0, 2: 0}

        # ── Ring buffers for waveform display ─────────────────────────────
        self.disp_ch1 = np.zeros(self.SIGNAL_HISTORY)
        self.disp_ch2 = np.zeros(self.SIGNAL_HISTORY)
        self._sig_lock = threading.Lock()

        # ── DSP / Model / Inference ────────────────────────────────────────
        self.dsp    = DSPPipeline(fs=FS)
        self.model  = MLModelEngine(MODEL_PATH)
        self.engine = InferenceEngine(
            self.model, self.dsp, on_prediction=self._on_prediction)

        # ── Data queue & streaming thread ─────────────────────────────────
        self.data_queue  = queue.Queue(maxsize=60)
        self.stop_event  = threading.Event()
        self.stream_thread = None

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_ui()

        # ── Boot sequence (non-blocking) ──────────────────────────────────
        threading.Thread(target=self._boot_sequence, daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════
    # ROOT CONFIGURATION
    # ══════════════════════════════════════════════════════════════════════
    def _configure_root(self):
        self.root.title("SILENT SPEECH  ·  TinyML Assistive Interface  ·  v1.0")
        self.root.configure(bg=BG_DEEP)
        self.root.geometry("1180x820")
        self.root.minsize(960, 700)
        self.root.resizable(True, True)
        # Custom fonts
        self._font_title   = tkfont.Font(family="Helvetica", size=11, weight="bold")
        self._font_label   = tkfont.Font(family="Helvetica", size=9)
        self._font_value   = tkfont.Font(family="Courier",   size=10, weight="bold")
        self._font_big     = tkfont.Font(family="Helvetica", size=26, weight="bold")
        self._font_intent  = tkfont.Font(family="Courier",   size=28, weight="bold")
        self._font_micro   = tkfont.Font(family="Helvetica", size=8)
        self._font_mono    = tkfont.Font(family="Courier",   size=9)

    # ══════════════════════════════════════════════════════════════════════
    # UI CONSTRUCTION
    # ══════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        """Assemble the three-panel dashboard layout."""

        # ── Outer container ───────────────────────────────────────────────
        outer = tk.Frame(self.root, bg=BG_DEEP)
        outer.pack(fill="both", expand=True, padx=8, pady=8)

        # ── Header bar ────────────────────────────────────────────────────
        self._build_header(outer)

        # ── Top Panel: SYSTEM STATUS ───────────────────────────────────────
        self._build_top_panel(outer)

        # ── Center Panel: BIOMEDICAL SIGNAL FEED ──────────────────────────
        self._build_center_panel(outer)

        # ── Bottom Panel: INTENT ACCELERATOR FEED ─────────────────────────
        self._build_bottom_panel(outer)

        # ── Boot overlay ──────────────────────────────────────────────────
        self._build_boot_overlay()

    # ── Header ────────────────────────────────────────────────────────────
    def _build_header(self, parent):
        hdr = tk.Frame(parent, bg=BG_DEEP, height=40)
        hdr.pack(fill="x", pady=(0, 6))

        tk.Label(hdr, text="⬡  SILENT SPEECH", font=self._font_title,
                 bg=BG_DEEP, fg=CYAN).pack(side="left", padx=4)
        tk.Label(hdr,
                 text="End-to-End TinyML Assistive Interface  ·  Neural Intent Recognition Engine",
                 font=self._font_micro, bg=BG_DEEP, fg=GREY_MID).pack(side="left", padx=6)

        # Clock
        self._clock_var = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._clock_var,
                 font=self._font_mono, bg=BG_DEEP, fg=GREY_LIGHT).pack(side="right", padx=8)

        # Thin separator
        sep = tk.Frame(parent, bg=CYAN_DIM, height=1)
        sep.pack(fill="x", pady=(0, 6))

    # ── Top Panel ─────────────────────────────────────────────────────────
    def _build_top_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, bd=0, relief="flat")
        panel.pack(fill="x", pady=(0, 6))

        # Section label
        self._section_label(panel, "▸  SYSTEM STATUS")

        # 4-column card grid
        cards_frame = tk.Frame(panel, bg=BG_PANEL)
        cards_frame.pack(fill="x", padx=8, pady=(0, 8))
        for i in range(4):
            cards_frame.columnconfigure(i, weight=1)

        # Card 1: Connection status
        c1 = self._card(cards_frame, 0, 0)
        tk.Label(c1, text="CONNECTION", font=self._font_micro,
                 bg=BG_CARD, fg=GREY_MID).pack(anchor="w")
        conn_row = tk.Frame(c1, bg=BG_CARD)
        conn_row.pack(anchor="w")
        self._conn_dot = tk.Label(conn_row, text="●", font=self._font_big,
                                  bg=BG_CARD, fg=GREY_MID)
        self._conn_dot.pack(side="left")
        self._conn_label = tk.Label(conn_row, text="INITIALISING",
                                    font=self._font_value, bg=BG_CARD, fg=GREY_LIGHT)
        self._conn_label.pack(side="left", padx=4)

        # Card 2: Model stats
        c2 = self._card(cards_frame, 0, 1)
        tk.Label(c2, text="ML MODEL", font=self._font_micro,
                 bg=BG_CARD, fg=GREY_MID).pack(anchor="w")
        self._model_name = tk.Label(c2, text="Random Forest", font=self._font_value,
                                    bg=BG_CARD, fg=WHITE)
        self._model_name.pack(anchor="w")
        self._model_acc = tk.Label(c2, text="Accuracy: –",
                                   font=self._font_label, bg=BG_CARD, fg=GREY_LIGHT)
        self._model_acc.pack(anchor="w")
        self._model_trees = tk.Label(c2, text="Trees: 150  |  Features: 10",
                                     font=self._font_micro, bg=BG_CARD, fg=GREY_MID)
        self._model_trees.pack(anchor="w")

        # Card 3: Inference latency
        c3 = self._card(cards_frame, 0, 2)
        tk.Label(c3, text="INFERENCE LATENCY", font=self._font_micro,
                 bg=BG_CARD, fg=GREY_MID).pack(anchor="w")
        self._latency_var = tk.StringVar(value="– ms")
        tk.Label(c3, textvariable=self._latency_var,
                 font=self._font_big, bg=BG_CARD, fg=CYAN).pack(anchor="w")
        self._latency_avg = tk.Label(c3, text="Avg: – ms",
                                     font=self._font_micro, bg=BG_CARD, fg=GREY_MID)
        self._latency_avg.pack(anchor="w")

        # Card 4: Session stats
        c4 = self._card(cards_frame, 0, 3)
        tk.Label(c4, text="SESSION STATS", font=self._font_micro,
                 bg=BG_CARD, fg=GREY_MID).pack(anchor="w")
        self._infer_count_var = tk.StringVar(value="Inferences:  0")
        self._rest_count_var  = tk.StringVar(value="REST:  0")
        self._sel_count_var   = tk.StringVar(value="SELECT:  0")
        self._next_count_var  = tk.StringVar(value="NEXT PAGE:  0")
        for v, c in [(self._infer_count_var, WHITE),
                     (self._rest_count_var,  GREY_MID),
                     (self._sel_count_var,   CYAN),
                     (self._next_count_var,  RED_ALERT)]:
            tk.Label(c4, textvariable=v,
                     font=self._font_mono, bg=BG_CARD, fg=c).pack(anchor="w")

    # ── Center Panel ─────────────────────────────────────────────────────
    def _build_center_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, bd=0)
        panel.pack(fill="both", expand=True, pady=(0, 6))

        self._section_label(panel, "▸  BIOMEDICAL SIGNAL FEED  ·  2-Channel sEMG  ·  1000 Hz")

        # Matplotlib figure for waveforms
        self._fig = Figure(figsize=(10, 3.2), dpi=100, facecolor=BG_PANEL)
        self._fig.subplots_adjust(
            left=0.05, right=0.98, top=0.88, bottom=0.14, hspace=0.55)

        # Channel 1 axes
        self._ax1 = self._fig.add_subplot(2, 1, 1)
        self._ax2 = self._fig.add_subplot(2, 1, 2)

        for ax, ch_label, color in [
            (self._ax1, "CH 1  —  Primary Electrode", CYAN),
            (self._ax2, "CH 2  —  Reference Electrode", AMBER)
        ]:
            ax.set_facecolor(BG_PANEL)
            ax.tick_params(colors=GREY_MID, labelsize=7)
            ax.spines['bottom'].set_color(GREY_MID)
            ax.spines['left'].set_color(GREY_MID)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel(ch_label, color=color, fontsize=8, labelpad=4)
            ax.set_xlim(0, self.SIGNAL_HISTORY)
            ax.set_ylim(-200, 200)
            ax.axhline(0, color=GREY_MID, linewidth=0.4, linestyle='--', alpha=0.4)
            ax.grid(axis='y', color=GREY_MID, alpha=0.15, linewidth=0.5)

        self._ax2.set_xlabel("Samples  (rolling 500ms window)", color=GREY_MID, fontsize=7)

        x = np.arange(self.SIGNAL_HISTORY)
        self._line1, = self._ax1.plot(x, np.zeros(self.SIGNAL_HISTORY),
                                       color=CYAN,  linewidth=0.9, alpha=0.92)
        self._line2, = self._ax2.plot(x, np.zeros(self.SIGNAL_HISTORY),
                                       color=AMBER, linewidth=0.9, alpha=0.92)

        # State annotations
        self._state_text1 = self._ax1.text(
            0.99, 0.88, "STATE: –", transform=self._ax1.transAxes,
            color=WHITE, fontsize=8, ha='right', va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_MID, alpha=0.7))
        self._state_text2 = self._ax2.text(
            0.99, 0.88, "MAV: –  |  RMS: –", transform=self._ax2.transAxes,
            color=WHITE, fontsize=8, ha='right', va='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=BG_MID, alpha=0.7))

        canvas = FigureCanvasTkAgg(self._fig, master=panel)
        canvas.get_tk_widget().configure(bg=BG_PANEL, highlightthickness=0)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self._canvas = canvas

    # ── Bottom Panel ─────────────────────────────────────────────────────
    def _build_bottom_panel(self, parent):
        panel = tk.Frame(parent, bg=BG_PANEL, bd=0)
        panel.pack(fill="x", pady=(0, 0))

        self._section_label(panel, "▸  INTENT ACCELERATOR FEED  ·  Confidence Gate: 85%")

        content = tk.Frame(panel, bg=BG_PANEL)
        content.pack(fill="x", padx=8, pady=(0, 8))
        content.columnconfigure(0, weight=3)
        content.columnconfigure(1, weight=1)
        content.columnconfigure(2, weight=1)

        # ── Intent notification bar ──────────────────────────────────────
        self._intent_frame = tk.Frame(
            content, bg=BG_MID, relief="flat", bd=0)
        self._intent_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        icon_col = tk.Frame(self._intent_frame, bg=BG_MID, width=60)
        icon_col.pack(side="left", fill="y")
        icon_col.pack_propagate(False)
        self._intent_icon = tk.Label(
            icon_col, text="●", font=self._font_big,
            bg=BG_MID, fg=GREY_MID)
        self._intent_icon.pack(expand=True)

        text_col = tk.Frame(self._intent_frame, bg=BG_MID)
        text_col.pack(side="left", fill="both", expand=True, padx=(0, 12), pady=8)

        self._intent_action = tk.Label(
            text_col, text="[  REST  ]",
            font=self._font_intent, bg=BG_MID, fg=GREY_MID, anchor="w")
        self._intent_action.pack(fill="x")

        sub_row = tk.Frame(text_col, bg=BG_MID)
        sub_row.pack(fill="x")
        self._intent_conf = tk.Label(
            sub_row, text="Confidence:  –",
            font=self._font_mono, bg=BG_MID, fg=GREY_MID)
        self._intent_conf.pack(side="left")
        self._intent_desc = tk.Label(
            sub_row, text="Waiting for gesture recognition…",
            font=self._font_micro, bg=BG_MID, fg=GREY_MID)
        self._intent_desc.pack(side="left", padx=16)

        # Confidence bar canvas (right side)
        conf_frame = tk.Frame(content, bg=BG_CARD)
        conf_frame.grid(row=0, column=1, sticky="nsew", padx=(0, 6))
        tk.Label(conf_frame, text="CONFIDENCE", font=self._font_micro,
                 bg=BG_CARD, fg=GREY_MID).pack(pady=(6, 0))
        self._conf_canvas = tk.Canvas(
            conf_frame, width=120, height=70,
            bg=BG_CARD, highlightthickness=0)
        self._conf_canvas.pack(expand=True, fill="both", padx=8, pady=4)

        # Probability bars (all 3 classes)
        prob_frame = tk.Frame(content, bg=BG_CARD)
        prob_frame.grid(row=0, column=2, sticky="nsew")
        tk.Label(prob_frame, text="CLASS PROBABILITIES",
                 font=self._font_micro, bg=BG_CARD, fg=GREY_MID).pack(pady=(6, 2))

        self._prob_bars = {}
        bar_colors = {0: GREY_LIGHT, 1: CYAN, 2: RED_ALERT}
        bar_labels  = {0: "REST", 1: "SELECT", 2: "NEXT"}
        for cls in [0, 1, 2]:
            row_f = tk.Frame(prob_frame, bg=BG_CARD)
            row_f.pack(fill="x", padx=8, pady=1)
            tk.Label(row_f, text=f"{bar_labels[cls]:6s}",
                     font=self._font_micro, bg=BG_CARD,
                     fg=bar_colors[cls], width=6, anchor="w").pack(side="left")
            bar_bg = tk.Frame(row_f, bg=GREY_MID, height=12)
            bar_bg.pack(side="left", fill="x", expand=True)
            bar_fill = tk.Frame(bar_bg, bg=bar_colors[cls], height=12, width=0)
            bar_fill.place(x=0, y=0, relheight=1.0)
            pct_lbl = tk.Label(row_f, text="0%",
                               font=self._font_micro, bg=BG_CARD,
                               fg=bar_colors[cls], width=4, anchor="e")
            pct_lbl.pack(side="left", padx=(4, 0))
            self._prob_bars[cls] = (bar_bg, bar_fill, pct_lbl)

        # Recent gesture log
        log_frame = tk.Frame(panel, bg=BG_PANEL)
        log_frame.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(log_frame, text="GESTURE LOG  ›", font=self._font_micro,
                 bg=BG_PANEL, fg=GREY_MID).pack(side="left")
        self._log_text = tk.Label(
            log_frame, text="",
            font=self._font_mono, bg=BG_PANEL, fg=GREY_MID)
        self._log_text.pack(side="left", padx=8)

    # ── Boot Overlay ──────────────────────────────────────────────────────
    def _build_boot_overlay(self):
        self._overlay = tk.Frame(self.root, bg=BG_DEEP)
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=1)

        inner = tk.Frame(self._overlay, bg=BG_DEEP)
        inner.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(inner, text="⬡", font=tkfont.Font(family="Helvetica", size=64),
                 bg=BG_DEEP, fg=CYAN).pack()
        tk.Label(inner, text="SILENT SPEECH",
                 font=tkfont.Font(family="Helvetica", size=20, weight="bold"),
                 bg=BG_DEEP, fg=WHITE).pack()
        tk.Label(inner, text="TinyML Assistive Interface",
                 font=tkfont.Font(family="Helvetica", size=10),
                 bg=BG_DEEP, fg=GREY_MID).pack(pady=(2, 16))

        self._boot_bar_bg = tk.Frame(inner, bg=BG_MID, height=4, width=380)
        self._boot_bar_bg.pack()
        self._boot_bar_fill = tk.Frame(self._boot_bar_bg, bg=CYAN, height=4, width=0)
        self._boot_bar_fill.place(x=0, y=0, relheight=1)

        tk.Label(inner, textvariable=self.boot_message,
                 font=self._font_mono, bg=BG_DEEP, fg=CYAN).pack(pady=(8, 0))

    # ── Helpers ────────────────────────────────────────────────────────────
    def _section_label(self, parent, text):
        row = tk.Frame(parent, bg=BG_PANEL)
        row.pack(fill="x", padx=8, pady=(6, 4))
        tk.Frame(row, bg=CYAN, width=3, height=14).pack(side="left", padx=(0, 6))
        tk.Label(row, text=text, font=self._font_title,
                 bg=BG_PANEL, fg=GREY_LIGHT).pack(side="left")

    def _card(self, parent, row, col, padx=4, pady=4):
        frame = tk.Frame(parent, bg=BG_CARD, bd=0)
        frame.grid(row=row, column=col, sticky="nsew",
                   padx=padx, pady=pady, ipadx=10, ipady=8)
        return frame

    # ══════════════════════════════════════════════════════════════════════
    # BOOT SEQUENCE
    # ══════════════════════════════════════════════════════════════════════
    def _boot_sequence(self):
        def step(msg, pct):
            self.boot_message.set(msg)
            self.root.after(0, lambda p=pct: self._boot_bar_fill.configure(
                width=int(380 * p / 100)))

        step("Checking for pre-trained model…", 5)
        time.sleep(0.3)

        if os.path.exists(MODEL_PATH):
            step("Loading model from disk…", 20)
            try:
                self.model.load()
                step(f"Model loaded  ·  Accuracy: {self.model.train_acc * 100:.1f}%", 60)
                time.sleep(0.4)
            except Exception as e:
                step(f"Load failed ({e}), retraining…", 20)
                self._run_auto_train(step)
        else:
            step("No model found  ·  Starting autonomous training…", 10)
            time.sleep(0.3)
            self._run_auto_train(step)

        step("Initialising DSP pipeline…", 75)
        time.sleep(0.25)

        step("Starting live data stream…", 88)
        self.stream_thread = LiveStreamThread(self.data_queue, self.stop_event)
        self.stream_thread.start()
        self.stream_active = True
        time.sleep(0.2)

        step("System ONLINE", 100)
        time.sleep(0.4)

        self.model_ready = True
        self.is_booting  = False
        self.root.after(0, self._on_boot_complete)

    def _run_auto_train(self, step_fn):
        phases = [
            (10,  "Generating synthetic sEMG dataset…"),
            (30,  "Applying DSP bandpass + notch filters…"),
            (50,  "Extracting biomedical features…"),
            (65,  "Training Random Forest (150 trees)…"),
            (72,  "Evaluating classifier…"),
        ]
        for pct, msg in phases:
            step_fn(msg, pct)
            time.sleep(0.05)

        def status_cb(msg):
            pass  # suppress console spam during animated boot

        self.model.auto_train(status_callback=status_cb)
        step_fn(f"Training complete  ·  Accuracy: {self.model.train_acc * 100:.1f}%", 73)
        time.sleep(0.3)

    def _on_boot_complete(self):
        """Called in the main thread after boot succeeds."""
        # Update status cards
        self._conn_dot.configure(fg=GREEN_OK)
        self._conn_label.configure(text="ONLINE", fg=GREEN_OK)
        acc_str = f"{self.model.train_acc * 100:.1f}%"
        self._model_acc.configure(text=f"Accuracy: {acc_str}", fg=GREEN_OK)

        # Dismiss overlay with fade (simulate with repeated alpha reductions)
        self._dismiss_overlay()

        # Start UI update loop
        self._update_clock()
        self._update_ui()

    def _dismiss_overlay(self, step=0):
        if step > 10:
            self._overlay.destroy()
            return
        # Shrink overlay from bottom
        rel = 1.0 - (step / 10.0)
        self._overlay.place(relx=0, rely=0, relwidth=1, relheight=rel)
        self.root.after(40, lambda: self._dismiss_overlay(step + 1))

    # ══════════════════════════════════════════════════════════════════════
    # INFERENCE CALLBACK  (called from inference thread)
    # ══════════════════════════════════════════════════════════════════════
    def _on_prediction(self, pred_class, proba, confidence, latency_ms):
        self.last_pred_class = pred_class
        self.last_confidence = confidence
        self.last_latency_ms = latency_ms
        self.inference_count += 1
        self._gesture_counts[pred_class] += 1

        self.latency_history.append(latency_ms)
        if len(self.latency_history) > self.LATENCY_HISTORY:
            self.latency_history.pop(0)

        if confidence >= CONFIDENCE_THRESH:
            ts = time.strftime("%H:%M:%S")
            self.prediction_log.append((ts, pred_class, confidence, proba.copy()))
            if len(self.prediction_log) > 50:
                self.prediction_log.pop(0)

    # ══════════════════════════════════════════════════════════════════════
    # DATA PROCESSING LOOP  (runs in UI update, drains queue)
    # ══════════════════════════════════════════════════════════════════════
    def _drain_queue(self):
        blocks_processed = 0
        latest_ch1 = latest_ch2 = None
        while not self.data_queue.empty() and blocks_processed < 8:
            try:
                ch1, ch2, state = self.data_queue.get_nowait()
                self.engine.ingest(ch1, ch2)
                with self._sig_lock:
                    n = len(ch1)
                    self.disp_ch1 = np.roll(self.disp_ch1, -n)
                    self.disp_ch2 = np.roll(self.disp_ch2, -n)
                    self.disp_ch1[-n:] = ch1
                    self.disp_ch2[-n:] = ch2
                latest_ch1 = self.disp_ch1.copy()
                latest_ch2 = self.disp_ch2.copy()
                self._last_stream_state = state
                blocks_processed += 1
            except queue.Empty:
                break
        return latest_ch1, latest_ch2

    # ══════════════════════════════════════════════════════════════════════
    # UI UPDATE LOOP  (runs every UPDATE_MS in Tkinter main thread)
    # ══════════════════════════════════════════════════════════════════════
    def _update_ui(self):
        if self.is_booting:
            return

        # 1. Drain data queue and ingest into inference engine
        ch1_data, ch2_data = self._drain_queue()

        # 2. Update waveform plot
        if ch1_data is not None:
            self._update_waveforms(ch1_data, ch2_data)

        # 3. Update system status cards
        self._update_status_cards()

        # 4. Update intent panel
        self._update_intent_panel()

        # 5. Schedule next tick
        self.root.after(self.UPDATE_MS, self._update_ui)

    def _update_clock(self):
        self._clock_var.set(time.strftime("  %H : %M : %S  ·  %d %b %Y  "))
        self.root.after(1000, self._update_clock)

    def _update_waveforms(self, ch1, ch2):
        """Redraw waveform lines and annotations."""
        x = np.arange(len(ch1))
        self._line1.set_data(x, ch1)
        self._line2.set_data(x, ch2)

        # Auto-scale Y with padding
        for ax, data in [(self._ax1, ch1), (self._ax2, ch2)]:
            peak = max(np.abs(data).max(), 10)
            ax.set_ylim(-peak * 1.25, peak * 1.25)

        # State / feature annotations
        state = getattr(self, '_last_stream_state', 0)
        state_names = {0: "STATE: REST", 1: "STATE: JAW CLENCH", 2: "STATE: FINGER FLICK"}
        state_colors = {0: GREY_LIGHT, 1: CYAN, 2: RED_ALERT}
        self._state_text1.set_text(state_names[state])
        self._state_text1.set_color(state_colors[state])

        mav_val = np.mean(np.abs(ch1[-WINDOW_SAMPLES:]))
        rms_val = np.sqrt(np.mean(ch1[-WINDOW_SAMPLES:] ** 2))
        self._state_text2.set_text(f"MAV: {mav_val:6.1f} µV  |  RMS: {rms_val:6.1f} µV")

        self._canvas.draw_idle()

    def _update_status_cards(self):
        """Refresh latency and session counters."""
        if self.last_latency_ms > 0:
            self._latency_var.set(f"{self.last_latency_ms:.1f} ms")
            if self.latency_history:
                avg = np.mean(self.latency_history)
                self._latency_avg.configure(text=f"Avg: {avg:.1f} ms")

        self._infer_count_var.set(f"Inferences:  {self.inference_count:,}")
        self._rest_count_var.set( f"REST:        {self._gesture_counts[0]:,}")
        self._sel_count_var.set(  f"SELECT:      {self._gesture_counts[1]:,}")
        self._next_count_var.set( f"NEXT PAGE:   {self._gesture_counts[2]:,}")

    def _update_intent_panel(self):
        """Update intent bar, confidence bars, and gesture log."""
        cls   = self.last_pred_class
        conf  = self.last_confidence
        label = LABEL_MAP[cls]

        # Colour and label the intent bar
        active = conf >= CONFIDENCE_THRESH
        bar_bg = CYAN if (active and cls == 1) else \
                 RED_ALERT if (active and cls == 2) else \
                 BG_MID

        self._intent_frame.configure(bg=bar_bg)
        self._intent_icon.configure(
            bg=bar_bg,
            text=ICON_MAP[cls],
            fg=WHITE if active else GREY_MID)
        self._intent_action.configure(
            bg=bar_bg,
            text=label,
            fg=WHITE if active else GREY_MID)
        self._intent_conf.configure(
            bg=bar_bg,
            text=f"Confidence:  {conf * 100:.1f}%",
            fg=WHITE if active else GREY_MID)

        desc_map = {
            0: "Idle state — no active gesture detected",
            1: "Jaw Clench detected  →  Simulating SELECT macro",
            2: "Finger Flick detected  →  Simulating NEXT PAGE macro"
        }
        self._intent_desc.configure(
            bg=bar_bg,
            text=desc_map[cls] if active else "Waiting for gesture recognition…",
            fg=WHITE if active else GREY_MID)

        # Confidence meter (semicircular arc drawn on canvas)
        self._draw_conf_meter(conf, active)

        # Probability bars
        # We need last proba array — pull from prediction log if available
        proba = np.array([0.0, 0.0, 0.0])
        proba[cls] = conf
        if self.prediction_log:
            _, _, _, last_proba = self.prediction_log[-1]
            proba = last_proba

        bar_colors_map = {0: GREY_LIGHT, 1: CYAN, 2: RED_ALERT}
        for c in [0, 1, 2]:
            bar_bg_w, bar_fill_w, pct_lbl = self._prob_bars[c]
            bar_bg_w.update_idletasks()
            total_w = bar_bg_w.winfo_width()
            fill_w  = max(0, int(total_w * proba[c]))
            bar_fill_w.configure(width=fill_w, bg=bar_colors_map[c])
            pct_lbl.configure(text=f"{proba[c] * 100:.0f}%")

        # Gesture log (last 5 confident events)
        log_parts = []
        for ts, gc, gconf, _ in self.prediction_log[-5:]:
            log_parts.append(f"[{ts}] {LABEL_MAP[gc]} {gconf*100:.0f}%")
        self._log_text.configure(text="    ".join(log_parts))

    def _draw_conf_meter(self, conf, active):
        """Draw a simple horizontal confidence gauge on the canvas."""
        cv = self._conf_canvas
        cv.delete("all")
        cv.update_idletasks()
        W = cv.winfo_width()  or 120
        H = cv.winfo_height() or 70

        # Background bar
        bx0, bx1 = 10, W - 10
        by       = H // 2
        cv.create_rectangle(bx0, by - 6, bx1, by + 6,
                             fill=BG_MID, outline="", width=0)
        # Fill
        fill_w = int((bx1 - bx0) * conf)
        fill_color = CYAN if active else CYAN_DIM
        if fill_w > 0:
            cv.create_rectangle(bx0, by - 6, bx0 + fill_w, by + 6,
                                 fill=fill_color, outline="", width=0)
        # Threshold marker
        thresh_x = bx0 + int((bx1 - bx0) * CONFIDENCE_THRESH)
        cv.create_line(thresh_x, by - 12, thresh_x, by + 12,
                       fill=AMBER, width=2)
        cv.create_text(thresh_x, by - 16, text="85%",
                       fill=AMBER, font=("Helvetica", 7))
        # Percentage text
        cv.create_text(W // 2, by + 22,
                       text=f"{conf * 100:.1f} %",
                       fill=WHITE if active else GREY_LIGHT,
                       font=("Courier", 11, "bold"))

    # ══════════════════════════════════════════════════════════════════════
    # SHUTDOWN
    # ══════════════════════════════════════════════════════════════════════
    def shutdown(self):
        self.stop_event.set()
        if self.stream_thread and self.stream_thread.is_alive():
            self.stream_thread.join(timeout=1.0)
        self.root.destroy()


# ─────────────────────────────────────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 72)
    print("  SILENT SPEECH  ·  TinyML Assistive Interface")
    print("  Principal Product Engineer & Neurotechnology Solution Architect")
    print("=" * 72)

    root = tk.Tk()
    app  = SilentSpeechApp(root)

    def on_close():
        print("[Shutdown] Stopping threads and closing application…")
        app.shutdown()

    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        root.mainloop()
    except KeyboardInterrupt:
        on_close()


if __name__ == "__main__":
    main()
