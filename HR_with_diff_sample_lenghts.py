"""
Part C: Signal Length Truncation Experiment
=============================================
Tests how HR estimate stability depends on TOTAL available signal length,
independent of window size. Simulates "partial recording" scenarios by
truncating each subject's concatenated signal to the first N samples.

Reuses: sliding_window_non_overlapping, sliding_window_overlapping,
        detect_hr_from_window, estimate_hr  (from hr_estimation.py)

Usage:
    subject_signals = concatenate_subject_signal(subject_chunks)
    length_results = run_signal_length_analysis(subject_signals,
                                                  signal_lengths=[160, 320, 480, 1920],
                                                  window_sizes_sec=[3, 4, 5])
    df_length = export_length_results_to_csv(length_results, "hr_length_results.csv")
    plot_length_dashboard(df_length)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hr_estimation import estimate_hr, FPS, OVERLAP_RATIO


# =============================================================================
# STEP 1: TRUNCATE SIGNAL TO N SAMPLES
# =============================================================================
def truncate_signal(signal, n_samples):
    """
    Take the first n_samples from a concatenated subject signal.
    Simulates having only a partial recording available.

    Returns:
        np.array of length min(n_samples, len(signal))
        None if signal is shorter than n_samples (insufficient data)
    """
    if len(signal) < n_samples:
        return None  # subject doesn't have enough data for this length
    return signal[:n_samples]


# =============================================================================
# STEP 2: RUN ANALYSIS ACROSS SIGNAL LENGTHS
# =============================================================================
def run_signal_length_analysis(subject_signals,
                                signal_lengths=(160, 320, 480, 1920),
                                window_sizes_sec=(3, 4, 5)):
    """
    For each subject, truncate to each signal_length, then run sliding-window
    HR estimation at each window_size, both modes.

    Returns:
        dict: results[subject_id][signal_length][window_size][mode] = estimate_hr() output
    """
    results = {}

    for subject_id, full_signal in subject_signals.items():
        results[subject_id] = {}
        print(f"\nSubject {subject_id} (full length: {len(full_signal)} samples)")

        for sig_len in signal_lengths:
            truncated = truncate_signal(full_signal, sig_len)

            if truncated is None:
                print(f"  ⚠️  Skipping length={sig_len} (subject only has {len(full_signal)} samples)")
                results[subject_id][sig_len] = None
                continue

            results[subject_id][sig_len] = {}
            print(f"  Length={sig_len} samples ({sig_len/FPS:.1f}s)")

            for ws in window_sizes_sec:
                results[subject_id][sig_len][ws] = {}

                for mode in ['non_overlapping', 'overlapping']:
                    res = estimate_hr(truncated, ws, mode=mode)
                    results[subject_id][sig_len][ws][mode] = res

                    mean_str = f"{res['mean_hr']:.1f}" if res['mean_hr'] is not None else "N/A"
                    print(f"    [{ws}s, {mode:>15}] Mean HR = {mean_str:>6} bpm | "
                          f"windows = {res['n_windows_valid']}/{res['n_windows_total']}")

    return results


# =============================================================================
# STEP 3: EXPORT TO CSV
# =============================================================================
def export_length_results_to_csv(results, output_path="hr_length_results.csv"):
    """
    Flatten nested results into a tidy DataFrame.

    Columns: subject, signal_length, signal_length_sec, window_size_sec,
             mode, mean_hr, std_hr, n_windows_valid, n_windows_total
    """
    rows = []

    for subject_id, length_dict in results.items():
        for sig_len, window_dict in length_dict.items():
            if window_dict is None:
                continue  # insufficient data for this length
            for ws, mode_dict in window_dict.items():
                for mode, res in mode_dict.items():
                    rows.append({
                        "subject": subject_id,
                        "signal_length": sig_len,
                        "signal_length_sec": round(sig_len / FPS, 1),
                        "window_size_sec": ws,
                        "mode": mode,
                        "mean_hr": res["mean_hr"],
                        "std_hr": res["std_hr"],
                        "n_windows_valid": res["n_windows_valid"],
                        "n_windows_total": res["n_windows_total"],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results exported to: {output_path}")
    print(f"   Shape: {df.shape}")

    return df


# =============================================================================
# STEP 4: VISUALIZATION
# =============================================================================
def plot_length_dashboard(df, save_prefix="hr_length_dashboard"):
    """
    3-panel dashboard showing how signal length affects HR estimation:
        1. Mean HR vs signal length, per window size (averaged across subjects)
        2. Std HR vs signal length, per window size (stability/convergence)
        3. Number of valid windows vs signal length
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Effect of Signal Length on HR Estimation Stability", fontsize=14, fontweight='bold')

    df_nonoverlap = df[df["mode"] == "non_overlapping"]
    window_sizes = sorted(df["window_size_sec"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(window_sizes)))

    # -------------------------------------------------------------------
    # Panel 1: Mean HR vs signal length (averaged across subjects)
    # -------------------------------------------------------------------
    ax1 = axes[0]
    for ws, color in zip(window_sizes, colors):
        subset = df_nonoverlap[df_nonoverlap["window_size_sec"] == ws]
        grouped = subset.groupby("signal_length_sec")["mean_hr"].agg(['mean', 'std']).reset_index()
        ax1.errorbar(grouped["signal_length_sec"], grouped["mean"], yerr=grouped["std"],
                     marker='o', label=f"{ws}s window", color=color, capsize=4)

    ax1.set_xlabel("Available Signal Length (seconds)")
    ax1.set_ylabel("Mean HR across subjects (bpm)")
    ax1.set_title("Does Mean HR Estimate Converge\nas More Signal Becomes Available?")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------
    # Panel 2: Std HR vs signal length (within-subject window variability)
    # -------------------------------------------------------------------
    ax2 = axes[1]
    for ws, color in zip(window_sizes, colors):
        subset = df_nonoverlap[df_nonoverlap["window_size_sec"] == ws]
        grouped = subset.groupby("signal_length_sec")["std_hr"].mean().reset_index()
        ax2.plot(grouped["signal_length_sec"], grouped["std_hr"],
                 marker='o', label=f"{ws}s window", color=color)

    ax2.set_xlabel("Available Signal Length (seconds)")
    ax2.set_ylabel("Mean Std HR across subjects (bpm)")
    ax2.set_title("Does Estimate Stability (Std)\nImprove with More Signal?")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------
    # Panel 3: Number of valid windows vs signal length
    # -------------------------------------------------------------------
    ax3 = axes[2]
    df_overlap = df[df["mode"] == "overlapping"]
    for ws, color in zip(window_sizes, colors):
        subset_no = df_nonoverlap[df_nonoverlap["window_size_sec"] == ws]
        subset_ov = df_overlap[df_overlap["window_size_sec"] == ws]

        grouped_no = subset_no.groupby("signal_length_sec")["n_windows_valid"].mean().reset_index()
        grouped_ov = subset_ov.groupby("signal_length_sec")["n_windows_valid"].mean().reset_index()

        ax3.plot(grouped_no["signal_length_sec"], grouped_no["n_windows_valid"],
                 marker='o', linestyle='-', label=f"{ws}s non-overlap", color=color)
        ax3.plot(grouped_ov["signal_length_sec"], grouped_ov["n_windows_valid"],
                 marker='s', linestyle='--', label=f"{ws}s overlap", color=color, alpha=0.6)

    ax3.set_xlabel("Available Signal Length (seconds)")
    ax3.set_ylabel("Mean Number of Valid Windows")
    ax3.set_title("Window Count Growth\nwith Signal Length")
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Dashboard saved: {save_prefix}.png")

    return fig


# =============================================================================
# MAIN (example usage)
# =============================================================================
# if __name__ == "__main__":
    # Assumes subject_signals already loaded from hr_estimation.py pipeline:
    # subject_chunks  = load_gt_npy_files(GT_DIR)
    # subject_signals = concatenate_subject_signal(subject_chunks)

    # SIGNAL_LENGTHS = [160, 320, 480, 1920]   # samples (5.3s, 10.7s, 16s, 64s)
    # WINDOW_SIZES   = [3, 4, 5]               # seconds

    # length_results = run_signal_length_analysis(subject_signals,
    #                                               signal_lengths=SIGNAL_LENGTHS,
    #                                               window_sizes_sec=WINDOW_SIZES)

    # df_length = export_length_results_to_csv(length_results, "hr_length_results.csv")
    # plot_length_dashboard(df_length)