"""
Part D: Chunk-Level HR Estimation
===================================
Splits the full concatenated signal into equal-sized chunks,
applies sliding-window HR estimation within each chunk,
then takes the mean HR across all chunks.

This preserves the original chunk structure and gives each
chunk equal weight in the final HR estimate.

Pipeline:
    Full signal (1920 samples)
        ↓
    Split into N chunks of size chunk_size
        ↓
    Each chunk → sliding window → mean HR per chunk
        ↓
    mean(HR_chunk1, HR_chunk2, ..., HR_chunkN) → final subject HR

Reuses: estimate_hr() from hr_estimation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os
import glob
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from collections import defaultdict

from hr_estimation import estimate_hr, FPS, OVERLAP_RATIO

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
# STEP 1: SPLIT SIGNAL INTO EQUAL CHUNKS
# =============================================================================
def split_into_chunks(signal, chunk_size):
    """
    Split a continuous signal into equal-sized non-overlapping chunks.
    Discards any remaining samples if signal length is not perfectly divisible.

    Args:
        signal:     1D numpy array (full concatenated subject signal)
        chunk_size: number of samples per chunk (e.g. 160, 320, 480)

    Returns:
        list of np.arrays, each of length chunk_size
        (remainder samples discarded with a warning)
    """
    n_samples   = len(signal)
    n_chunks    = n_samples // chunk_size
    remainder   = n_samples % chunk_size

    if remainder != 0:
        print(f"  ⚠️  Signal length {n_samples} not divisible by chunk_size {chunk_size}. "
              f"Discarding last {remainder} samples — using {n_chunks} complete chunks.")

    chunks = [signal[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]
    return chunks


# =============================================================================
# STEP 2: HR PER CHUNK → MEAN ACROSS CHUNKS
# =============================================================================
def estimate_hr_chunk_mean(signal, chunk_size, window_size_sec,
                            mode='non_overlapping'):
    """
    For a given chunk_size:
        1. Split signal into equal chunks
        2. Run sliding-window HR estimation on each chunk
        3. Return mean HR across all valid chunks

    Args:
        signal:          1D numpy array (full subject signal)
        chunk_size:      samples per chunk (160, 320, 480, ...)
        window_size_sec: sliding window size in seconds
        mode:            'non_overlapping' or 'overlapping'

    Returns:
        dict with:
            chunk_hrs:       list of per-chunk mean HR (one per chunk)
            mean_hr:         mean of chunk_hrs
            std_hr:          std of chunk_hrs
            n_chunks_total:  total chunks created
            n_chunks_valid:  chunks that produced a valid HR (≥2 peaks)
            chunk_size:      samples per chunk (for reference)
            chunk_size_sec:  seconds per chunk
    """
    chunks = split_into_chunks(signal, chunk_size)
    n_chunks_total = len(chunks)

    chunk_hrs = []
    for i, chunk in enumerate(chunks):
        res = estimate_hr(chunk, window_size_sec, mode=mode)
        if res['mean_hr'] is not None:
            chunk_hrs.append(res['mean_hr'])
        else:
            print(f"    ⚠️  Chunk {i+1}: no valid HR (insufficient peaks) — skipped")

    n_valid = len(chunk_hrs)

    if n_valid == 0:
        return {
            "chunk_hrs":      [],
            "mean_hr":        None,
            "std_hr":         None,
            "n_chunks_total": n_chunks_total,
            "n_chunks_valid": 0,
            "chunk_size":     chunk_size,
            "chunk_size_sec": round(chunk_size / FPS, 1),
        }

    return {
        "chunk_hrs":      chunk_hrs,
        "mean_hr":        float(np.mean(chunk_hrs)),
        "std_hr":         float(np.std(chunk_hrs)),
        "n_chunks_total": n_chunks_total,
        "n_chunks_valid": n_valid,
        "chunk_size":     chunk_size,
        "chunk_size_sec": round(chunk_size / FPS, 1),
    }


# =============================================================================
# STEP 3: FULL ANALYSIS ACROSS ALL SUBJECTS × CHUNK SIZES × WINDOW SIZES
# =============================================================================
def run_chunk_analysis(subject_signals,
                        chunk_sizes=(160, 320, 480),
                        window_sizes_sec=(3, 4, 5)):
    """
    Run chunk-level HR estimation for all subjects,
    all chunk sizes, all window sizes, both modes.

    Returns:
        dict: results[subject_id][chunk_size][window_size][mode]
    """
    results = {}

    for subject_id, signal in subject_signals.items():
        results[subject_id] = {}
        print(f"\n{'='*65}")
        print(f"Subject: {subject_id} | Full signal: {len(signal)} samples "
              f"({len(signal)/FPS:.1f}s)")
        print(f"{'='*65}")

        for chunk_size in chunk_sizes:
            n_chunks = len(signal) // chunk_size
            print(f"\n  Chunk size: {chunk_size} samples "
                  f"({chunk_size/FPS:.1f}s) → {n_chunks} chunks")

            results[subject_id][chunk_size] = {}

            for ws in window_sizes_sec:
                results[subject_id][chunk_size][ws] = {}

                for mode in ['non_overlapping', 'overlapping']:
                    res = estimate_hr_chunk_mean(signal, chunk_size, ws, mode)
                    results[subject_id][chunk_size][ws][mode] = res

                    mean_str = f"{res['mean_hr']:.1f}" if res['mean_hr'] is not None else "N/A"
                    std_str  = f"{res['std_hr']:.1f}"  if res['std_hr']  is not None else "N/A"

                    print(f"    [chunk={chunk_size:>3}, win={ws}s, {mode:>15}] "
                        f"Mean HR = {mean_str:>6} bpm | "
                        f"Std = {std_str:>5} | "
                        f"Valid chunks = {res['n_chunks_valid']}/{res['n_chunks_total']}")

    return results


# =============================================================================
# STEP 4: EXPORT TO CSV
# =============================================================================
def export_chunk_results_to_csv(results, output_path="hr_chunk_results.csv"):
    """
    Flatten nested results into a tidy DataFrame.

    Columns:
        subject, chunk_size, chunk_size_sec, window_size_sec, mode,
        mean_hr, std_hr, n_chunks_valid, n_chunks_total
    """
    rows = []

    for subject_id, chunk_dict in results.items():
        for chunk_size, window_dict in chunk_dict.items():
            for ws, mode_dict in window_dict.items():
                for mode, res in mode_dict.items():
                    rows.append({
                        "subject":        subject_id,
                        "chunk_size":     chunk_size,
                        "chunk_size_sec": res["chunk_size_sec"],
                        "window_size_sec": ws,
                        "mode":           mode,
                        "mean_hr":        res["mean_hr"],
                        "std_hr":         res["std_hr"],
                        "n_chunks_valid": res["n_chunks_valid"],
                        "n_chunks_total": res["n_chunks_total"],
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\n✅ Results exported to: {output_path}")
    print(f"   Shape: {df.shape} | Subjects: {df['subject'].nunique()}")

    return df


# =============================================================================
# STEP 5: VISUALIZATION DASHBOARD
# =============================================================================
def plot_chunk_dashboard(df, save_prefix="hr_chunk_dashboard"):
    """
    3-panel dashboard:
        1. Mean HR vs chunk size (averaged across subjects) — does estimate converge?
        2. Std HR vs chunk size — does within-subject variance shrink?
        3. Per-subject HR heatmap across chunk sizes (non-overlapping, 3s window)
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Chunk-Level HR Estimation — Effect of Chunk Size",
                  fontsize=14, fontweight='bold')

    df_no = df[df["mode"] == "non_overlapping"]
    window_sizes = sorted(df["window_size_sec"].unique())
    chunk_sizes  = sorted(df["chunk_size"].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(window_sizes)))

    # -------------------------------------------------------------------
    # Panel 1: Mean HR vs chunk size (population average)
    # -------------------------------------------------------------------
    ax1 = axes[0]
    for ws, color in zip(window_sizes, colors):
        subset = df_no[df_no["window_size_sec"] == ws]
        grouped = subset.groupby("chunk_size_sec")["mean_hr"].agg(
            ['mean', 'std']).reset_index()
        ax1.errorbar(grouped["chunk_size_sec"], grouped["mean"],
                     yerr=grouped["std"], marker='o',
                     label=f"{ws}s window", color=color, capsize=4)

    ax1.set_xlabel("Chunk Size (seconds)")
    ax1.set_ylabel("Mean HR across subjects (bpm)")
    ax1.set_title("Mean HR vs Chunk Size\n(non-overlapping, all subjects)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # -------------------------------------------------------------------
    # Panel 2: Std HR vs chunk size (stability)
    # -------------------------------------------------------------------
    ax2 = axes[1]
    for ws, color in zip(window_sizes, colors):
        subset = df_no[df_no["window_size_sec"] == ws]
        grouped = subset.groupby("chunk_size_sec")["std_hr"].mean().reset_index()
        ax2.plot(grouped["chunk_size_sec"], grouped["std_hr"],
                 marker='o', label=f"{ws}s window", color=color)

    ax2.set_xlabel("Chunk Size (seconds)")
    ax2.set_ylabel("Mean Std HR across subjects (bpm)")
    ax2.set_title("HR Estimate Stability vs Chunk Size\n(lower = more stable)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # -------------------------------------------------------------------
    # Panel 3: Per-subject heatmap (non-overlap, 3s window)
    # -------------------------------------------------------------------
    ax3 = axes[2]
    df_heat = df_no[df_no["window_size_sec"] == 3].copy()
    pivot = df_heat.pivot(index="subject", columns="chunk_size", values="mean_hr")
    pivot = pivot.sort_index()

    im = ax3.imshow(pivot.values, aspect='auto', cmap='RdYlGn')
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels([f"{c}\n({c/FPS:.0f}s)" for c in pivot.columns])
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index, fontsize=7)
    ax3.set_xlabel("Chunk Size (samples / seconds)")
    ax3.set_ylabel("Subject")
    ax3.set_title("Mean HR Heatmap per Subject\n(non-overlapping, 3s window)")
    plt.colorbar(im, ax=ax3, label="Mean HR (bpm)")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Dashboard saved: {save_prefix}.png")

    return fig


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    # Assumes subject_signals already loaded:
    GT_DIR = "/home/muhammadu/pre_proc_datasets/PURE_intra/PURE_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

    subject_chunks  = load_gt_npy_files(GT_DIR)
    subject_signals = concatenate_subject_signal(subject_chunks)

    CHUNK_SIZES  = [160, 320, 480]   # samples → 5.3s, 10.7s, 16.0s
    WINDOW_SIZES = [3, 4, 5]         # seconds

    # Run analysis
    chunk_results = run_chunk_analysis(subject_signals,
                                        chunk_sizes=CHUNK_SIZES,
                                        window_sizes_sec=WINDOW_SIZES)

    # Export + visualize
    df_chunks = export_chunk_results_to_csv(chunk_results, "hr_chunk_results.csv")
    plot_chunk_dashboard(df_chunks)