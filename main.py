from HR_with_diff_sample_lenghts import export_length_results_to_csv, plot_length_dashboard, run_signal_length_analysis

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

GT_DIR = "/home/muhammadu/pre_proc_datasets/PURE_intra/PURE_SizeW128_SizeH128_ClipLength160_DataTypeStandardized_DataAugNone_LabelTypeStandardized_Crop_faceTrue_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

# 1. Load and process
subject_chunks  = load_gt_npy_files(GT_DIR)
subject_signals = concatenate_subject_signal(subject_chunks)

# 2. Run sliding-window HR estimation
results = run_full_analysis(subject_signals, window_sizes_sec=[3, 4, 5])
print_summary_table(results)

# 3. Export + visualize
df = export_results_to_csv(results, "hr_results.csv")
plot_dashboard(df)
flag_noisy_subjects(df, std_threshold=10)


# ===============HR WITH DIFFERENT SIGNAL LENGTHS========================

# Assumes subject_signals already loaded from hr_estimation.py pipeline:
subject_chunks  = load_gt_npy_files(GT_DIR)
subject_signals = concatenate_subject_signal(subject_chunks)

SIGNAL_LENGTHS = [160, 320, 480, 1920]   # samples (5.3s, 10.7s, 16s, 64s)
WINDOW_SIZES   = [3, 4, 5]               # seconds

length_results = run_signal_length_analysis(subject_signals,
                                              signal_lengths=SIGNAL_LENGTHS,
                                              window_sizes_sec=WINDOW_SIZES)

df_length = export_length_results_to_csv(length_results, "hr_length_results.csv")
plot_length_dashboard(df_length)
