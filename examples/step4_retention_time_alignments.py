import chromalyzer

config = {
    "parallel_processing": True,
    "number_of_splits": 100,

    "mz_list_path": "data/all_mz_values.csv",
    "labels_path": "data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    "features_path": "/usr/scratch/chromalyzer/features/",
    "output_dir_TII_aligned": "/usr/scratch/chromalyzer/TII_aligned/",
    "peaks_dir_path": "/usr/scratch/chromalyzer/peaks/",
    "lambda1": [0.4],
    "lambda2": [2.0],
    "rt1_threshold": [35.04], # 10 * 3.504 = 35.04
    "rt2_threshold": [round(i * 0.008,3) for i in range(1,150)], # 5*0.008 = 0.04 - 150*0.008 = 1.2
}

chromalyzer.retention_times_alignment.retention_times_alignment(config)