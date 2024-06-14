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

    "output_dir_heatmap": "/usr/scratch/chromalyzer/heatmaps/",
    
    "peaks_dir_path": "/usr/scratch/chromalyzer/peaks/",
    "lambda1": 0.75,
    "lambda2": 5.0,
    "peak_max_neighbor_distance": 20,
    "area_min_threshold": 15000,
    "strict_noise_filtering": True,
    "column_noise_removal":{
        "enable": True,
        "max_distance_removal_noisy_columns": 50,
        "noisy_columns": [7700,8700],
        "non_zero_ratio_column_threshold": 0.1
    },

    "enable_noisy_regions": True,
    "noisy_regions": [
        {
            "first_time_start": 0,
            "second_time_start": 0,
            "first_time_end": -1,
            "second_time_end": 1,
            "non_zero_ratio_region_threshold": 0.001
        }
    ],
    
    "convolution_filter": {
        "enable": True,
        "lambda3": 10,
        "rt1_window_size": 2000,
        "rt2_window_size": 0.1,
        "rt1_stride": 2000,
        "rt2_stride": 0.1,
        "non_zero_ratio_mean_subtracted":  0.99999,
        "non_zero_ratio_lambda3_filter": 0.05
    },
    "delta_rt1" : 10,
    "delta_rt2" : 0.12,
    "max_retention_time1_allowed": 11400
}
chromalyzer.find_peaks.extract_peaks(config)

