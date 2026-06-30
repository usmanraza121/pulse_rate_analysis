"""
Part B: Export results to CSV + Visualization Dashboard
==========================================================
Run this AFTER you have the `results` dict from run_full_analysis().

Usage:
    subject_signals = concatenate_subject_signal(subject_chunks)
    results = run_full_analysis(subject_signals, window_sizes_sec=WINDOW_SIZES_SEC)
    print_summary_table(results)

    # NEW PART:
    df = export_results_to_csv(results, "hr_results.csv")
    plot_dashboard(df)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# A. EXPORT RESULTS TO CSV
# =============================================================================
def export_results_to_csv(results, output_path="hr_results.csv"):
    """
    Flatten the nested `results` dict into a tidy DataFrame and save as CSV.

    Columns:
        subject, window_size_sec, mode, mean_hr, std_hr,
        n_windows_valid, n_windows_total, signal_group
    """
    rows = []

    for subject_id, window_dict in results.items():
        for window_size, mode_dict in window_dict.items():
            for mode, res in mode_dict.items():
                rows.append({
                    "subject": subject_id,
                    "window_size_sec": window_size,
                    "mode": mode,
                    "mean_hr": res["mean_hr"],
                    "std_hr": res["std_hr"],
                    "n_windows_valid": res["n_windows_valid"],
                    "n_windows_total": res["n_windows_total"],
                    # Group subjects by their series prefix (101->1, 1001->1, 701->7 etc.)
                    "signal_group": str(subject_id)[0] if len(str(subject_id)) == 3 
                                    else str(subject_id)[:2]
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Results exported to: {output_path}")
    print(f"   Shape: {df.shape}")
    print(f"   Subjects: {df['subject'].nunique()}")
    
    return df


# =============================================================================
# B. VISUALIZATION DASHBOARD
# =============================================================================
def plot_dashboard(df, save_prefix="hr_dashboard"):
    """
    Create a multi-panel dashboard:
        1. Bar chart - Mean HR per subject (colored by group)
        2. Box plot - HR distribution per window size
        3. Scatter - Std vs Mean HR (identify noisy subjects)
        4. Line plot - Mean HR comparison: overlap vs non-overlap
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("HR Estimation Dashboard — Sliding Window Analysis", fontsize=15, fontweight='bold')

    # Use 5s non-overlapping as the "reference" estimate for plots 1 & 3
    df_ref = df[(df["window_size_sec"] == 5) & (df["mode"] == "non_overlapping")].copy()
    df_ref = df_ref.sort_values("subject")

    # -------------------------------------------------------------------
    # Panel 1: Bar chart - Mean HR per subject (colored by group)
    # -------------------------------------------------------------------
    ax1 = axes[0, 0]
    groups = df_ref["signal_group"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(groups)))
    color_map = dict(zip(sorted(groups), colors))
    bar_colors = df_ref["signal_group"].map(color_map)

    ax1.bar(range(len(df_ref)), df_ref["mean_hr"], color=bar_colors)
    ax1.set_xticks(range(len(df_ref)))
    ax1.set_xticklabels(df_ref["subject"], rotation=90, fontsize=7)
    ax1.set_ylabel("Mean HR (bpm)")
    ax1.set_title("Mean HR per Subject (5s window, non-overlapping)\nColored by signal group")
    ax1.axhline(60, color='green', linestyle='--', alpha=0.4, linewidth=0.8, label='60 bpm')
    ax1.axhline(100, color='red', linestyle='--', alpha=0.4, linewidth=0.8, label='100 bpm')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # -------------------------------------------------------------------
    # Panel 2: Box plot - HR distribution per window size
    # -------------------------------------------------------------------
    ax2 = axes[0, 1]
    window_sizes = sorted(df["window_size_sec"].unique())
    box_data = [df[(df["window_size_sec"] == ws) & 
                    (df["mode"] == "non_overlapping")]["mean_hr"].dropna()
                for ws in window_sizes]

    bp = ax2.boxplot(box_data, labels=[f"{ws}s" for ws in window_sizes],
                      patch_artist=True, showmeans=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightsteelblue')

    ax2.set_ylabel("Mean HR (bpm)")
    ax2.set_xlabel("Window Size")
    ax2.set_title("HR Distribution Across Window Sizes\n(non-overlapping mode, all subjects)")
    ax2.grid(True, alpha=0.3, axis='y')

    # -------------------------------------------------------------------
    # Panel 3: Scatter - Std vs Mean HR (identify noisy subjects)
    # -------------------------------------------------------------------
    ax3 = axes[1, 0]
    scatter = ax3.scatter(df_ref["mean_hr"], df_ref["std_hr"],
                          c=bar_colors, s=60, edgecolors='black', linewidth=0.5, alpha=0.8)

    # Annotate top 5 noisiest subjects
    noisy = df_ref.nlargest(5, "std_hr")
    for _, row in noisy.iterrows():
        ax3.annotate(str(row["subject"]),
                     (row["mean_hr"], row["std_hr"]),
                     fontsize=8, xytext=(5, 5), textcoords='offset points')

    ax3.set_xlabel("Mean HR (bpm)")
    ax3.set_ylabel("Std HR (bpm) — noise indicator")
    ax3.set_title("Signal Quality Map\n(High Std = noisy / motion artifact)")
    ax3.axhline(10, color='orange', linestyle='--', alpha=0.5, label='Std=10 threshold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # -------------------------------------------------------------------
    # Panel 4: Line plot - Overlap vs Non-overlap comparison
    # -------------------------------------------------------------------
    ax4 = axes[1, 1]
    df_5s = df[df["window_size_sec"] == 5].copy()
    pivot = df_5s.pivot(index="subject", columns="mode", values="mean_hr").sort_index()

    x = np.arange(len(pivot))
    ax4.plot(x, pivot["non_overlapping"], 'o-', label='Non-overlapping', alpha=0.7, markersize=4)
    ax4.plot(x, pivot["overlapping"], 's-', label='Overlapping (50%)', alpha=0.7, markersize=4)

    ax4.set_xticks(x[::3])  # show every 3rd label to avoid crowding
    ax4.set_xticklabels(pivot.index[::3], rotation=90, fontsize=7)
    ax4.set_ylabel("Mean HR (bpm)")
    ax4.set_title("Overlap vs Non-Overlap Comparison (5s window)")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{save_prefix}.png", dpi=150, bbox_inches='tight')
    plt.show()
    print(f"✅ Dashboard saved: {save_prefix}.png")

    return fig


# =============================================================================
# BONUS: Quick noisy-subject report
# =============================================================================
def flag_noisy_subjects(df, std_threshold=10, window_size=5, mode="non_overlapping"):
    """
    Print a quick report of subjects whose HR estimate has high variability,
    which may indicate motion artifacts or poor signal quality.
    """
    subset = df[(df["window_size_sec"] == window_size) & (df["mode"] == mode)]
    noisy = subset[subset["std_hr"] > std_threshold].sort_values("std_hr", ascending=False)

    print(f"\n{'='*60}")
    print(f"⚠️  NOISY SUBJECTS (Std > {std_threshold} bpm, {window_size}s {mode})")
    print(f"{'='*60}")
    if noisy.empty:
        print("None found — all signals look clean!")
    else:
        for _, row in noisy.iterrows():
            print(f"  Subject {row['subject']:>6}: Mean={row['mean_hr']:.1f} bpm, "
                  f"Std={row['std_hr']:.1f} bpm")

    return noisy


# =============================================================================
# MAIN (example usage — append after your existing pipeline)
# =============================================================================
# if __name__ == "__main__":
#     # These come from your existing code:
#     # subject_chunks  = load_gt_npy_files(GT_DIR)
#     # subject_signals = concatenate_subject_signal(subject_chunks)
#     # results         = run_full_analysis(subject_signals, window_sizes_sec=WINDOW_SIZES_SEC)
#     # print_summary_table(results)

#     # NEW: export + visualize
#     # df = export_results_to_csv(results, "hr_results.csv")
#     # plot_dashboard(df)
#     # flag_noisy_subjects(df, std_threshold=10)