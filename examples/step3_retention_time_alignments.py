import chromalyzer

config = {
    "parallel_processing": True,
    "number_of_splits": 20,

    "mz_list_path": "data/all_mz_values.csv",
    "labels_path": "data/labels.csv",
    "m_z_column_name": "M/Z",
    "area_column_name": "Area",
    "first_time_column_name": "1st Time (s)",
    "second_time_column_name": "2nd Time (s)",
    "csv_file_name_column": "csv_file_name",
    "label_column_name": "label",

    "features_path": "/usr/scratch/chromalyzer/test/features/",
    "peaks_dir_path": "/usr/scratch/chromalyzer/test/peaks/",
    "lambda1": [0.4],
    "lambda2": [2.0],
    "rt1_threshold": [10],
    "rt2_threshold": range(5,125,5),
    "rt1_time_step": 3.504,
    "rt2_time_step": 0.008
}

chromalyzer.retention_times_alignment.retention_times_alignment(config)