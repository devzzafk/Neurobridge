"""
data_simulator.py
=================
Handles raw 2-channel mathematical EMG signal synthesis and labeled CSV training
data generation for the Silent Speech TinyML Assistive Interface.

States:
  0 = REST      — low baseline static noise
  1 = CLENCH    — dense low-frequency burst (jaw clench)
  2 = FLICK     — sudden high-frequency spike (finger flick)
"""

import numpy as np
import pandas as pd
import time
import os

# ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 1000          # Hz
BUFFER_SIZE   = 256           # samples returned per "tick" for live streaming
DATASET_FILE  = "emg_dataset.csv"
RECORD_SECS   = 60            # seconds of labeled training data to generate
NUM_CHANNELS  = 2


# ─── Core Signal Synthesis ────────────────────────────────────────────────────

def _baseline_noise(n_samples: int, amplitude: float = 0.05) -> np.ndarray:
    """White Gaussian noise representing low-level muscle baseline."""
    return np.random.normal(0.0, amplitude, n_samples)


def _clench_burst(n_samples: int, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Dense low-frequency burst simulating a jaw clench.
    Dominant energy 20–80 Hz with amplitude envelope.
    """
    t = np.arange(n_samples) / fs
    # Primary contraction harmonics
    sig  = 0.60 * np.sin(2 * np.pi * 35 * t)
    sig += 0.30 * np.sin(2 * np.pi * 55 * t + np.pi / 4)
    sig += 0.15 * np.sin(2 * np.pi * 75 * t)
    # Amplitude envelope — rise and hold
    envelope = np.clip(np.linspace(0, 1, n_samples) * 3.0, 0, 1)
    sig *= envelope
    # Add physiological noise
    sig += _baseline_noise(n_samples, amplitude=0.08)
    return sig


def _flick_spike(n_samples: int, fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Sudden high-frequency spike simulating a finger flick.
    Short transient burst 150–400 Hz with rapid decay.
    """
    t = np.arange(n_samples) / fs
    sig  = 0.80 * np.sin(2 * np.pi * 200 * t)
    sig += 0.40 * np.sin(2 * np.pi * 300 * t + np.pi / 6)
    sig += 0.20 * np.sin(2 * np.pi * 380 * t)
    # Sharp exponential decay envelope
    decay = np.exp(-t * (fs / (n_samples * 0.3)))
    sig *= decay
    sig += _baseline_noise(n_samples, amplitude=0.04)
    return sig


def synthesize_chunk(state: int, n_samples: int = BUFFER_SIZE,
                     fs: int = SAMPLE_RATE) -> np.ndarray:
    """
    Returns shape (n_samples, 2) float32 array of 2-channel EMG data
    for the requested gesture state.

    Parameters
    ----------
    state     : int  — 0=REST, 1=CLENCH, 2=FLICK
    n_samples : int  — number of time-domain samples
    fs        : int  — sampling frequency in Hz

    Returns
    -------
    np.ndarray shape (n_samples, 2)
    """
    if state == 0:
        ch1 = _baseline_noise(n_samples, amplitude=0.03)
        ch2 = _baseline_noise(n_samples, amplitude=0.04)
    elif state == 1:
        ch1 = _clench_burst(n_samples, fs)
        # Channel 2 has slightly attenuated clench (different electrode site)
        ch2 = _clench_burst(n_samples, fs) * 0.75 + _baseline_noise(n_samples, 0.05)
    elif state == 2:
        ch1 = _flick_spike(n_samples, fs)
        # Channel 2 picks up flick with slight delay phase shift
        ch2 = _flick_spike(n_samples, fs) * 0.85
        shift = int(fs * 0.002)   # 2 ms propagation delay
        if shift < n_samples:
            ch2 = np.roll(ch2, shift)
            ch2[:shift] = _baseline_noise(shift, 0.03)
    else:
        raise ValueError(f"Unknown state {state}. Must be 0, 1, or 2.")

    return np.stack([ch1, ch2], axis=1).astype(np.float32)


# ─── Live Stream Generator ────────────────────────────────────────────────────

class EMGStreamSimulator:
    """
    Infinite generator that yields (chunk, state) tuples simulating a live
    2-channel EMG hardware stream at SAMPLE_RATE Hz.

    The state machine cycles through realistic gesture sequences with
    randomised inter-gesture intervals.
    """

    STATE_NAMES = {0: "REST", 1: "CLENCH", 2: "FLICK"}

    def __init__(self, fs: int = SAMPLE_RATE, chunk_size: int = BUFFER_SIZE):
        self.fs         = fs
        self.chunk_size = chunk_size
        self._state     = 0
        self._counter   = 0
        self._duration  = self._random_duration(0)  # samples remaining in state
        self._rng       = np.random.default_rng()

    def _random_duration(self, state: int) -> int:
        """Return number of samples to remain in a given state."""
        if state == 0:
            # REST lasts 0.8 – 2.5 seconds
            secs = self._rng.uniform(0.8, 2.5)
        elif state == 1:
            # CLENCH lasts 0.3 – 0.8 seconds
            secs = self._rng.uniform(0.3, 0.8)
        else:
            # FLICK is brief: 0.1 – 0.25 seconds
            secs = self._rng.uniform(0.1, 0.25)
        return int(secs * self.fs)

    def _next_state(self) -> int:
        """Transition from REST → gesture, or gesture → REST."""
        if self._state == 0:
            return self._rng.choice([1, 2])
        return 0

    def read(self) -> tuple[np.ndarray, int]:
        """
        Returns the next (chunk, state) pair.
        Shape of chunk: (chunk_size, 2)
        """
        self._counter += self.chunk_size
        if self._counter >= self._duration:
            self._state    = self._next_state()
            self._duration = self._random_duration(self._state)
            self._counter  = 0

        chunk = synthesize_chunk(self._state, self.chunk_size, self.fs)
        return chunk, self._state

    def stream(self):
        """Infinite iterator yielding (chunk, state)."""
        while True:
            yield self.read()


# ─── Dataset Generator ────────────────────────────────────────────────────────

def generate_dataset(duration_secs: int = RECORD_SECS,
                     fs: int = SAMPLE_RATE,
                     output_path: str = DATASET_FILE,
                     verbose: bool = True) -> str:
    """
    Generates a labeled EMG dataset CSV of `duration_secs` seconds.

    Each row = one sample: [ch1, ch2, label]
    Labels: 0=REST, 1=CLENCH, 2=FLICK

    Returns the path to the saved CSV file.
    """
    total_samples = duration_secs * fs
    chunk         = BUFFER_SIZE

    records = []
    simulator = EMGStreamSimulator(fs=fs, chunk_size=chunk)

    if verbose:
        print(f"[DataSimulator] Generating {duration_secs}s dataset "
              f"({total_samples:,} samples) → '{output_path}'")

    collected = 0
    t0 = time.time()

    # State distribution targets (balanced)
    state_counts = {0: 0, 1: 0, 2: 0}

    # Fixed script for reproducible balanced dataset
    # Pattern: REST(2s), CLENCH(0.5s), REST(1.5s), FLICK(0.2s), REST(1s), ...
    script = []
    elapsed = 0.0
    while elapsed < duration_secs:
        script.append((0, min(2.0,  duration_secs - elapsed))); elapsed += 2.0
        if elapsed >= duration_secs: break
        script.append((1, min(0.5,  duration_secs - elapsed))); elapsed += 0.5
        if elapsed >= duration_secs: break
        script.append((0, min(1.5,  duration_secs - elapsed))); elapsed += 1.5
        if elapsed >= duration_secs: break
        script.append((2, min(0.22, duration_secs - elapsed))); elapsed += 0.22
        if elapsed >= duration_secs: break
        script.append((0, min(1.0,  duration_secs - elapsed))); elapsed += 1.0

    for state, secs in script:
        n = int(secs * fs)
        data = synthesize_chunk(state, n, fs)
        labels = np.full(n, state, dtype=np.int8)
        for i in range(n):
            records.append((data[i, 0], data[i, 1], labels[i]))
        state_counts[state] += n
        collected += n

    df = pd.DataFrame(records, columns=["ch1", "ch2", "label"])
    df.to_csv(output_path, index=False)

    elapsed_real = time.time() - t0
    if verbose:
        print(f"[DataSimulator] Done in {elapsed_real:.2f}s")
        print(f"  Total samples : {len(df):,}")
        for s, name in EMGStreamSimulator.STATE_NAMES.items():
            pct = state_counts[s] / len(df) * 100
            print(f"  {name:8s} (class {s}): {state_counts[s]:,} samples ({pct:.1f}%)")
        print(f"  Saved to      : {os.path.abspath(output_path)}")

    return output_path


# ─── CLI Entrypoint ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    generate_dataset(duration_secs=RECORD_SECS, verbose=True)
