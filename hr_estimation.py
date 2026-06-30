"""
HR Estimation from BVP/PPG Signals using Sliding Windows
=========================================================
Pipeline:
  1. Load all .npy chunks per subject
  2. Concatenate chunks per subject → full signal
  3. Apply sliding window (overlapping & non-overlapping)
  4. Detect peaks per window → compute HR
  5. Mean HR per subject
"""

import os
import glob
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import defaultdict


# =============================================================================
# CONFIG
# =============================================================================
FPS = 30                         # Sampling rate (frames per second)
WINDOW_SIZES_SEC = [3, 4, 5]    # Window sizes to test in seconds
OVERLAP_RATIO = 0.5              # 50% overlap for overlapping windows
GT_DIR = "GT_ref/"               # Directory with .npy files

# Peak detection parameters (tuned for BVP signal)
PEAK_PROMINENCE = 0.3
PEAK_DISTANCE_SEC = 0.3          # Min distance between peaks in seconds


# =============================================================================
# STEP 1: LOAD & GROUP BY SUBJECT
# =============================================================================
def load_gt_npy_files(gt_dir):
    """
    Load all .npy files and group by subject ID.
    
    Assumes filename format: {subject_id}_label{n}.npy
    e.g. 101_label0.npy, 101_label1.npy, ...
    
    Returns:
        dict: {subject_id: [array_chunk0, array_chunk1, ...]}
    """
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*_label*.npy')))
    
    if not gt_files:
        print(f"No label files found in {gt_dir}")
        return {}

    subject_chunks = defaultdict(list)
    
    for gt_file in gt_files:
        filename = os.path.basename(gt_file).split('.')[0]  # e.g. '101_label4'
        subject_id = filename.split('_')[0]                 # e.g. '101'
        chunk = np.load(gt_file)
        subject_chunks[subject_id].append(chunk)
    
    print(f"Loaded {len(gt_files)} files across {len(subject_chunks)} subjects")
    for subj, chunks in subject_chunks.items():
        print(f"  Subject {subj}: {len(chunks)} chunks × {chunks[0].shape[0]} samples "
              f"= {len(chunks) * chunks[0].shape[0]} total samples "
              f"≈ {len(chunks) * chunks[0].shape[0] / FPS:.1f}s")
    
    return dict(subject_chunks)


# =============================================================================
# STEP 2: CONCATENATE CHUNKS PER SUBJECT
# =============================================================================
def concatenate_subject_signal(subject_chunks):
    """
    Concatenate all chunks for each subject into one continuous signal.
    
    Returns:
        dict: {subject_id: np.array of shape (total_samples,)}
    """
    subject_signals = {}
    for subject_id, chunks in subject_chunks.items():
        signal = np.concatenate(chunks)
        subject_signals[subject_id] = signal
    return subject_signals


# =============================================================================
# STEP 3A: NON-OVERLAPPING SLIDING WINDOW
# =============================================================================
def sliding_window_non_overlapping(signal, window_size_sec, fps=FPS):
    """
    Split signal into non-overlapping windows of fixed size.
    Step = window_size (no overlap).
    
    Args:
        signal: 1D numpy array (full subject signal)
        window_size_sec: window size in seconds
        fps: sampling rate
    
    Returns:
        list of np.arrays, each of length window_size_samples
    """
    window_size = int(window_size_sec * fps)  # samples
    n_samples = len(signal)
    windows = []
    
    start = 0
    while start + window_size <= n_samples:
        window = signal[start : start + window_size]
        windows.append(window)
        start += window_size  # step = full window (no overlap)
    
    print(f"  [Non-overlapping] window={window_size_sec}s ({window_size} samples): "
          f"{len(windows)} complete windows "
          f"(discarded last {n_samples - start} samples)")
    
    return windows


# =============================================================================
# STEP 3B: OVERLAPPING SLIDING WINDOW
# =============================================================================
def sliding_window_overlapping(signal, window_size_sec, overlap_ratio=OVERLAP_RATIO, fps=FPS):
    """
    Split signal into overlapping windows.
    Step = window_size * (1 - overlap_ratio).
    
    Args:
        signal: 1D numpy array (full subject signal)
        window_size_sec: window size in seconds
        overlap_ratio: fraction of overlap (0.5 = 50%)
        fps: sampling rate
    
    Returns:
        list of np.arrays, each of length window_size_samples
    """
    window_size = int(window_size_sec * fps)               # samples
    step_size   = int(window_size * (1 - overlap_ratio))   # e.g. 50% overlap → step = half window
    n_samples   = len(signal)
    windows     = []
    
    start = 0
    while start + window_size <= n_samples:
        window = signal[start : start + window_size]
        windows.append(window)
        start += step_size
    
    print(f"  [Overlapping {int(overlap_ratio*100)}%] window={window_size_sec}s ({window_size} samples), "
          f"step={step_size} samples: {len(windows)} windows")
    
    return windows


# =============================================================================
# STEP 4: PEAK DETECTION → HR PER WINDOW
# =============================================================================
def detect_hr_from_window(window, fps=FPS,
                           prominence=PEAK_PROMINENCE,
                           min_distance_sec=PEAK_DISTANCE_SEC):
    """
    Detect peaks in a BVP window and compute HR in bpm.
    
    Args:
        window: 1D numpy array (single window of BVP signal)
        fps: sampling rate
        prominence: minimum peak prominence
        min_distance_sec: minimum distance between peaks in seconds
    
    Returns:
        float or None: HR in bpm, or None if insufficient peaks
    """
    min_distance_samples = int(fps * min_distance_sec)
    
    peaks, _ = find_peaks(window,
                           prominence=prominence,
                           distance=min_distance_samples)
    
    if len(peaks) < 2:
        return None  # Not enough peaks to compute HR
    
    # Inter-peak intervals → HR
    peak_intervals_sec = np.diff(peaks) / fps  # seconds between peaks
    hr_per_interval    = 60.0 / peak_intervals_sec  # bpm
    
    return hr_per_interval.mean()


# =============================================================================
# STEP 5: FULL HR ESTIMATION PIPELINE
# =============================================================================
def estimate_hr(signal, window_size_sec, mode='non_overlapping',
                overlap_ratio=OVERLAP_RATIO, fps=FPS):
    """
    Full pipeline: signal → windows → HR per window → mean HR.
    
    Args:
        signal: 1D numpy array (full subject signal)
        window_size_sec: window size in seconds
        mode: 'non_overlapping' or 'overlapping'
        overlap_ratio: only used if mode='overlapping'
        fps: sampling rate
    
    Returns:
        dict with:
            hr_per_window: list of HR values (bpm) per window
            mean_hr: mean HR across all valid windows
            std_hr: std of HR across all valid windows
            n_windows_total: total windows created
            n_windows_valid: windows with enough peaks
    """
    # Get windows
    if mode == 'non_overlapping':
        windows = sliding_window_non_overlapping(signal, window_size_sec, fps)
    elif mode == 'overlapping':
        windows = sliding_window_overlapping(signal, window_size_sec, overlap_ratio, fps)
    else:
        raise ValueError(f"mode must be 'non_overlapping' or 'overlapping', got '{mode}'")
    
    # HR per window
    hr_per_window = []
    for w in windows:
        hr = detect_hr_from_window(w, fps)
        if hr is not None:
            hr_per_window.append(hr)
    
    n_valid = len(hr_per_window)
    
    if n_valid == 0:
        print(f"  ⚠️  No valid windows found for window_size={window_size_sec}s")
        return {
            "hr_per_window": [],
            "mean_hr": None,
            "std_hr": None,
            "n_windows_total": len(windows),
            "n_windows_valid": 0
        }
    
    hr_array = np.array(hr_per_window)
    
    return {
        "hr_per_window": hr_per_window,
        "mean_hr": float(np.mean(hr_array)),
        "std_hr":  float(np.std(hr_array)),
        "n_windows_total": len(windows),
        "n_windows_valid": n_valid
    }


# =============================================================================
# STEP 6: RUN ON ALL SUBJECTS × ALL WINDOW SIZES
# =============================================================================
def run_full_analysis(subject_signals, window_sizes_sec=WINDOW_SIZES_SEC):
    """
    Run HR estimation for all subjects, all window sizes, both modes.
    
    Returns:
        dict: results[subject_id][window_size][mode] = estimate_hr output
    """
    results = {}
    
    for subject_id, signal in subject_signals.items():
        print(f"\n{'='*60}")
        print(f"Subject: {subject_id} | Signal length: {len(signal)} samples "
              f"≈ {len(signal)/FPS:.1f}s")
        print(f"{'='*60}")
        
        results[subject_id] = {}
        
        for ws in window_sizes_sec:
            results[subject_id][ws] = {}
            print(f"\n  Window size: {ws}s")
            
            for mode in ['non_overlapping', 'overlapping']:
                res = estimate_hr(signal, ws, mode=mode)
                results[subject_id][ws][mode] = res
                
                if res['mean_hr'] is not None:
                    print(f"    [{mode:>17}] "
                          f"Mean HR = {res['mean_hr']:5.1f} bpm | "
                          f"Std = {res['std_hr']:4.1f} | "
                          f"Valid windows = {res['n_windows_valid']}/{res['n_windows_total']}")
    
    return results


# =============================================================================
# STEP 7: SUMMARY TABLE
# =============================================================================
def print_summary_table(results):
    """Print a clean summary table of all results."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Subject':<10} {'Window':>8} {'Mode':<20} {'Mean HR':>10} {'Std':>8} {'Valid/Total':>12}")
    print(f"{'-'*80}")
    
    for subject_id in sorted(results.keys()):
        for ws in sorted(results[subject_id].keys()):
            for mode in ['non_overlapping', 'overlapping']:
                res = results[subject_id][ws][mode]
                mean = f"{res['mean_hr']:.1f}" if res['mean_hr'] else "N/A"
                std  = f"{res['std_hr']:.1f}"  if res['std_hr']  else "N/A"
                valid = f"{res['n_windows_valid']}/{res['n_windows_total']}"
                print(f"{subject_id:<10} {str(ws)+'s':>8} {mode:<20} {mean:>10} {std:>8} {valid:>12}")
        print(f"{'-'*80}")


# =============================================================================
# STEP 8: VISUALIZATION
# =============================================================================
def plot_subject_signal_with_windows(signal, subject_id, window_size_sec=3, fps=FPS):
    """
    Plot the full subject signal with window boundaries overlaid.
    """
    window_size = int(window_size_sec * fps)
    time_axis   = np.arange(len(signal)) / fps
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Subject {subject_id} — BVP Signal with {window_size_sec}s Windows", fontsize=13)
    
    for ax, mode, color in zip(axes,
                                ['non_overlapping', 'overlapping'],
                                ['steelblue', 'darkorange']):
        ax.plot(time_axis, signal, color='gray', linewidth=0.8, alpha=0.7, label='BVP signal')
        
        # Get windows
        if mode == 'non_overlapping':
            windows_list = sliding_window_non_overlapping(signal, window_size_sec, fps)
            step = window_size
        else:
            step = int(window_size * (1 - OVERLAP_RATIO))
            windows_list = sliding_window_overlapping(signal, window_size_sec, fps=fps)
        
        # Draw window boundaries
        for i in range(len(windows_list)):
            start_t = (i * step) / fps
            end_t   = start_t + window_size_sec
            ax.axvspan(start_t, end_t, alpha=0.08, color=color)
            ax.axvline(start_t, color=color, linewidth=0.5, alpha=0.5)
        
        # Detect and plot peaks for first window as example
        w0 = windows_list[0]
        peaks, _ = find_peaks(w0, prominence=PEAK_PROMINENCE,
                              distance=int(fps * PEAK_DISTANCE_SEC))
        peak_times = peaks / fps
        ax.plot(peak_times, w0[peaks], 'rv', markersize=8, label='Peaks (window 1)')
        
        ax.set_ylabel("BVP Amplitude")
        ax.set_title(f"Mode: {mode.replace('_', ' ').title()}")
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel("Time (seconds)")
    plt.tight_layout()
    plt.savefig(f"subject_{subject_id}_windows_{window_size_sec}s.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Plot saved: subject_{subject_id}_windows_{window_size_sec}s.png")


# =============================================================================
# MAIN
# =============================================================================
# if __name__ == "__main__":
    
#     # 1. Load files grouped by subject
#     subject_chunks = load_gt_npy_files(GT_DIR)
    
#     # 2. Concatenate chunks per subject
#     subject_signals = concatenate_subject_signal(subject_chunks)
    
#     # 3. Run full analysis
#     results = run_full_analysis(subject_signals, window_sizes_sec=WINDOW_SIZES_SEC)
    
#     # 4. Print summary
#     print_summary_table(results)
    
#     # 5. Plot one subject as example
#     first_subject = sorted(subject_signals.keys())[0]
#     plot_subject_signal_with_windows(subject_signals[first_subject],
#                                       subject_id=first_subject,
#                                       window_size_sec=3)