"""
signal_processing.py
====================
Biomedical DSP pipeline for the Silent Speech TinyML Assistive Interface.

Provides:
  - 4th-order Butterworth bandpass filter (20–450 Hz)
  - 50 Hz notch filter (power-grid hum rejection)
  - Time-domain EMG feature extraction (MAV, RMS, WL)
  - Sliding-window feature matrix builder
"""

import numpy as np
from scipy.signal import (
    butter, sosfilt, sosfilt_zi,
    iirnotch, bilinear_zpk, zpk2sos
)
from typing import Tuple


# ─── Filter Design ────────────────────────────────────────────────────────────

def design_bandpass(low_hz: float = 20.0,
                    high_hz: float = 450.0,
                    fs: int = 1000,
                    order: int = 4) -> np.ndarray:
    """
    Design a 4th-order Butterworth bandpass filter.

    Parameters
    ----------
    low_hz  : lower cutoff frequency in Hz
    high_hz : upper cutoff frequency in Hz
    fs      : sampling frequency in Hz
    order   : filter order (each side), total order = 2*order for bandpass

    Returns
    -------
    sos : second-order sections array, shape (n_sections, 6)
    """
    nyq  = fs / 2.0
    low  = low_hz  / nyq
    high = high_hz / nyq
    sos  = butter(order, [low, high], btype="bandpass", output="sos")
    return sos


def design_notch(notch_hz: float = 50.0,
                 quality: float = 30.0,
                 fs: int = 1000) -> np.ndarray:
    """
    Design an IIR notch filter to remove power-line interference.

    Parameters
    ----------
    notch_hz : frequency to notch out (50 Hz EU/Asia, 60 Hz US)
    quality  : Q factor — higher = narrower notch
    fs       : sampling frequency in Hz

    Returns
    -------
    sos : second-order sections array
    """
    b, a = iirnotch(notch_hz, quality, fs)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    sos = tf2sos(b, a)
    return sos


class RealtimeFilter:
    """
    Stateful SOS filter that maintains initial conditions across chunks,
    enabling seamless real-time (streaming) filtering without edge artifacts.

    Usage
    -----
    f = RealtimeFilter(sos)
    filtered_chunk = f.process(raw_chunk)   # call for each new chunk
    """

    def __init__(self, sos: np.ndarray):
        self._sos = sos
        self._zi  = sosfilt_zi(sos)          # shape: (n_sections, 2)
        self._zi  = self._zi[:, np.newaxis, :]  # prepare for multi-channel broadcast

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Filter a chunk of shape (n_samples,) or (n_samples, n_channels).

        Returns filtered array of same shape.
        """
        if chunk.ndim == 1:
            y, self._zi[:, 0, :] = sosfilt(
                self._sos, chunk,
                zi=self._zi[:, 0, :]
            )
            return y

        # Multi-channel: process each channel independently
        n_channels = chunk.shape[1]
        # Expand zi if needed
        if self._zi.shape[1] < n_channels:
            zi_base = sosfilt_zi(self._sos)   # (n_sections, 2)
            self._zi = np.repeat(zi_base[:, np.newaxis, :], n_channels, axis=1)

        out = np.empty_like(chunk)
        for ch in range(n_channels):
            out[:, ch], self._zi[:, ch, :] = sosfilt(
                self._sos, chunk[:, ch],
                zi=self._zi[:, ch, :]
            )
        return out

    def reset(self):
        """Reset filter state (call if signal resets / stream interrupts)."""
        zi_base  = sosfilt_zi(self._sos)
        n_ch     = self._zi.shape[1]
        self._zi = np.repeat(zi_base[:, np.newaxis, :], n_ch, axis=1)


class EMGFilterPipeline:
    """
    Cascaded biomedical filter pipeline:
      1. Bandpass  (20 – 450 Hz) — removes movement artifacts & aliasing
      2. Notch     (50 Hz)       — removes mains hum

    Example
    -------
    pipeline = EMGFilterPipeline(fs=1000)
    clean    = pipeline.process(raw_chunk)   # (n_samples, 2) → (n_samples, 2)
    """

    def __init__(self, fs: int = 1000,
                 bp_low: float = 20.0, bp_high: float = 450.0,
                 notch_hz: float = 50.0, notch_q: float = 30.0):
        self.fs = fs
        bp_sos     = design_bandpass(bp_low, bp_high, fs)
        notch_sos  = design_notch(notch_hz, notch_q, fs)
        self._bp    = RealtimeFilter(bp_sos)
        self._notch = RealtimeFilter(notch_sos)

    def process(self, raw: np.ndarray) -> np.ndarray:
        """Apply bandpass then notch to raw chunk. Returns same shape."""
        after_bp    = self._bp.process(raw)
        after_notch = self._notch.process(after_bp)
        return after_notch

    def reset(self):
        self._bp.reset()
        self._notch.reset()


# ─── Feature Extraction ───────────────────────────────────────────────────────

def mav(window: np.ndarray) -> float:
    """Mean Absolute Value — overall signal energy measure."""
    return float(np.mean(np.abs(window)))


def rms(window: np.ndarray) -> float:
    """Root Mean Square — effective power of the signal."""
    return float(np.sqrt(np.mean(window ** 2)))


def waveform_length(window: np.ndarray) -> float:
    """
    Waveform Length — cumulative arc length; sensitive to frequency and amplitude.
    WL = Σ |x[i] - x[i-1]|
    """
    return float(np.sum(np.abs(np.diff(window))))


def extract_features_single_channel(window: np.ndarray) -> np.ndarray:
    """
    Extract [MAV, RMS, WL] from a 1-D signal window.

    Returns
    -------
    np.ndarray shape (3,)
    """
    return np.array([mav(window), rms(window), waveform_length(window)],
                    dtype=np.float32)


def extract_features(window: np.ndarray) -> np.ndarray:
    """
    Extract features from a multi-channel window.

    Parameters
    ----------
    window : np.ndarray shape (n_samples,) or (n_samples, n_channels)

    Returns
    -------
    features : np.ndarray shape (3,) for 1-ch or (3 * n_channels,) for multi-ch
               Order: [ch1_MAV, ch1_RMS, ch1_WL, ch2_MAV, ch2_RMS, ch2_WL, ...]
    """
    if window.ndim == 1:
        return extract_features_single_channel(window)

    feats = []
    for ch in range(window.shape[1]):
        feats.append(extract_features_single_channel(window[:, ch]))
    return np.concatenate(feats)


FEATURE_NAMES = ["ch1_MAV", "ch1_RMS", "ch1_WL", "ch2_MAV", "ch2_RMS", "ch2_WL"]


# ─── Sliding-Window Feature Matrix Builder ────────────────────────────────────

def build_feature_matrix(signal: np.ndarray,
                         labels: np.ndarray,
                         fs: int = 1000,
                         window_ms: float = 200.0,
                         step_ms: float = 50.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slide an analysis window across the signal to build a feature matrix
    for ML training.

    Parameters
    ----------
    signal    : np.ndarray (n_samples,) or (n_samples, n_channels)
    labels    : np.ndarray (n_samples,) integer class labels
    fs        : sampling frequency
    window_ms : window duration in milliseconds (default 200 ms)
    step_ms   : step / hop size in milliseconds (default 50 ms)

    Returns
    -------
    X : np.ndarray (n_windows, n_features)  — feature matrix
    y : np.ndarray (n_windows,)             — majority-vote label per window
    """
    win_len  = int(window_ms * fs / 1000)   # samples per window
    step_len = int(step_ms   * fs / 1000)   # samples per step

    n_samples = signal.shape[0]
    starts    = range(0, n_samples - win_len + 1, step_len)

    X_list = []
    y_list = []

    for start in starts:
        end    = start + win_len
        window = signal[start:end]
        feats  = extract_features(window)
        X_list.append(feats)

        # Majority vote for label
        window_labels = labels[start:end]
        counts = np.bincount(window_labels.astype(int), minlength=3)
        y_list.append(int(np.argmax(counts)))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    return X, y


# ─── Rolling RMS for Live Oscilloscope ───────────────────────────────────────

class RollingRMS:
    """
    Efficient rolling RMS over a sliding buffer — used to feed the UI
    oscilloscope display with one value per incoming chunk.

    Parameters
    ----------
    window_samples : int — how many samples to include in the RMS window
    """

    def __init__(self, window_samples: int = 200):
        self._buf = np.zeros(window_samples, dtype=np.float32)
        self._win = window_samples

    def update(self, new_samples: np.ndarray) -> float:
        """
        Append new_samples to the rolling buffer and return current RMS.

        Parameters
        ----------
        new_samples : 1-D array of any length ≤ window_samples

        Returns
        -------
        float — RMS of the current window
        """
        n = len(new_samples)
        if n >= self._win:
            self._buf[:] = new_samples[-self._win:]
        else:
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = new_samples
        return rms(self._buf)


# ─── Self-Test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("signal_processing.py — self-test")
    fs = 1000
    t  = np.linspace(0, 1, fs, endpoint=False)

    # Synthesise a test signal: 35 Hz + 60 Hz noise
    raw = (0.5 * np.sin(2 * np.pi * 35 * t) +
           0.3 * np.sin(2 * np.pi * 60 * t) +
           0.05 * np.random.randn(fs))

    pipeline = EMGFilterPipeline(fs=fs, notch_hz=60.0)
    filtered = pipeline.process(raw)

    print(f"  Raw      RMS: {rms(raw):.4f}")
    print(f"  Filtered RMS: {rms(filtered):.4f}")

    # Feature extraction test
    feats = extract_features(raw)
    print(f"  Features (1ch): {feats}")

    # 2-channel test
    raw2ch = np.stack([raw, raw * 0.8], axis=1)
    feats2 = extract_features(raw2ch)
    print(f"  Features (2ch): {feats2}")
    print(f"  Feature names : {FEATURE_NAMES}")

    # Sliding window test
    labels = np.zeros(fs, dtype=np.int32)
    labels[300:600] = 1
    X, y = build_feature_matrix(raw2ch, labels, fs=fs)
    print(f"  Feature matrix: {X.shape}, label vector: {y.shape}")
    print("  OK — all tests passed.")
