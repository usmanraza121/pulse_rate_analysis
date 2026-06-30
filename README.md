# PPG/BVP Heart Rate Estimation via Sliding Window Peak Detection

This repository implements a sliding-window peak-detection pipeline to estimate heart rate (HR) from ground-truth BVP/PPG signals, comparing overlapping vs. non-overlapping windowing strategies across multiple window sizes.

## Overview

Each subject's BVP signal is provided as a sequence of `.npy` chunks (160 samples each, sampled at 30 fps ≈ 5.3s per chunk). The pipeline:

1. Loads and groups all chunks by subject
2. Concatenates chunks into one continuous signal per subject
3. Applies sliding-window peak detection (`scipy.signal.find_peaks`) to estimate HR per window
4. Aggregates per-window HR into a mean HR estimate per subject
5. Compares results across window sizes (3s, 4s, 5s) and windowing modes (overlapping vs non-overlapping)
6. Exports results to CSV and generates a visualization dashboard

## Repository Structure

```
├── hr_estimation.py          # Core pipeline: loading, windowing, peak detection, HR estimation
├── hr_analysis_export.py     # CSV export + visualization dashboard
├── GT_ref/                   # Input directory containing *_label*.npy files (not included)
├── hr_results.csv            # Generated output: tidy results table
└── hr_dashboard.png          # Generated output: 4-panel visualization
```

## Data Format

Input files follow the naming convention:

```
{subject_id}_label{chunk_index}.npy
```

e.g. `101_label0.npy`, `101_label1.npy`, `1001_label0.npy`, ...

Each `.npy` file is a 1D float64 array of shape `(160,)`, representing 160 samples of BVP signal at 30 fps (~5.3 seconds per chunk). Files are grouped by subject ID and concatenated into one continuous signal before windowing.

> Note: only `*_label*.npy` files are loaded — corresponding `*_input*.npy` files (raw video frames or features) are ignored by this pipeline.

## Pipeline Steps

### 1. Load & Group (`load_gt_npy_files`)
Scans a directory for `*_label*.npy` files and groups chunks by subject ID extracted from the filename prefix.

### 2. Concatenate (`concatenate_subject_signal`)
Joins all chunks per subject into a single continuous array, giving each subject a longer signal (typically ~64-85 seconds) suitable for meaningful windowed analysis.

### 3. Sliding Windows
Two windowing strategies are implemented:
- **Non-overlapping** (`sliding_window_non_overlapping`): step size = window size
- **Overlapping** (`sliding_window_overlapping`): step size = window size × (1 − overlap_ratio), default 50% overlap

Window sizes tested: **3s, 4s, 5s** (configurable via `WINDOW_SIZES_SEC`).

### 4. Peak Detection → HR (`detect_hr_from_window`)
Uses `scipy.signal.find_peaks` with prominence and minimum-distance constraints to detect systolic peaks per window, then converts inter-peak intervals to instantaneous HR (bpm).

### 5. Aggregation (`estimate_hr`)
Combines windowing + peak detection, returning per-window HR list, mean HR, std HR, and window counts (valid vs total) for a given subject/window size/mode.

### 6. Batch Run (`run_full_analysis`)
Runs the full pipeline across all subjects × all window sizes × both modes, printing a live progress log.

## Outputs

### CSV (`export_results_to_csv`)
A tidy long-format table with one row per `(subject, window_size, mode)` combination:

| subject | window_size_sec | mode | mean_hr | std_hr | n_windows_valid | n_windows_total | signal_group |
|---|---|---|---|---|---|---|---|

### Dashboard (`plot_dashboard`)
A 4-panel figure:
1. Mean HR per subject (bar chart, colored by subject group)
2. HR distribution across window sizes (box plot)
3. Std vs Mean HR scatter — flags noisy/motion-affected subjects
4. Overlapping vs non-overlapping HR comparison (line plot)

### Noisy Subject Report (`flag_noisy_subjects`)
Lists subjects whose HR estimate exceeds a configurable standard-deviation threshold, useful for identifying poor signal quality or motion artifacts.

## Usage

```python
from hr_estimation import (
    load_gt_npy_files,
    concatenate_subject_signal,
    run_full_analysis,
    print_summary_table,
)
from hr_analysis_export import (
    export_results_to_csv,
    plot_dashboard,
    flag_noisy_subjects,
)

# 1. Load and process
subject_chunks  = load_gt_npy_files("GT_ref/")
subject_signals = concatenate_subject_signal(subject_chunks)

# 2. Run sliding-window HR estimation
results = run_full_analysis(subject_signals, window_sizes_sec=[3, 4, 5])
print_summary_table(results)

# 3. Export + visualize
df = export_results_to_csv(results, "hr_results.csv")
plot_dashboard(df)
flag_noisy_subjects(df, std_threshold=10)
```

## Configuration

Key parameters in `hr_estimation.py`:

```python
FPS = 30                         # Sampling rate
WINDOW_SIZES_SEC = [3, 4, 5]     # Window sizes to test
OVERLAP_RATIO = 0.5              # 50% overlap for overlapping mode
PEAK_PROMINENCE = 0.3            # Minimum peak prominence
PEAK_DISTANCE_SEC = 0.3          # Minimum distance between peaks (limits max ~200 bpm)
```

## Key Findings (from initial 59-subject run)

- Mean HR is highly consistent across window sizes (typically within ~1 bpm)
- Standard deviation decreases as window size increases (larger windows → more stable estimates)
- Overlapping vs non-overlapping windowing yields nearly identical mean HR, but overlapping produces ~2× more windows, which may benefit downstream variance analysis
- A subset of subjects (e.g. 500-series) show high HR variability (std > 20 bpm), likely indicating motion artifacts or signal quality issues — flagged for further review

## Requirements

```
numpy
scipy
pandas
matplotlib
```

## Future Work

- Extend pipeline to 320-sample (longer) signal chunks for comparison against the 160-sample baseline
- Compare estimated HR against independent ground-truth references (e.g. ECG-derived HR) once available
- Investigate root cause of high-variance subjects (motion artifact filtering, signal quality indices)