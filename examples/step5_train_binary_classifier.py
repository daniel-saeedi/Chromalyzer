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

    "features_path": "/usr/scratch/chromalyzer/features/",
    "peaks_dir_path": "/usr/scratch/chromalyzer/peaks/",
    "results_dir": "/usr/scratch/chromalyzer/lr_l2_results/",

    # Logistic Regression with L2 regularization
    "C": 0.01,
    "seed": 42,
    "number_of_bootstraps": 1000,
    "lambda1": 0.75,
    "lambda2": 5.0,
    "rt1_threshold": 10,
    "rt2_threshold": 100,
}

chromalyzer.binary_classifier.binary_classifier(config)