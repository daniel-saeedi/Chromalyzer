import chromalyzer

config = {
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
    "parameters_selection_path":"/usr/scratch/chromalyzer/test/parameters_selection/",

    # Logistic Regression with L2 regularization
    "C": [1e-4,1e-3,1e-2,1e-1,1e0,1e+1,1e+2,1e+3,1e+4],
    "seed": 42,
    "lambda1": [0.4],
    "lambda2": [2.0],
    "rt1_threshold": [10],
    "rt2_threshold": range(5,125,5),
}

chromalyzer.parameters_selection.parameters_selection(config)