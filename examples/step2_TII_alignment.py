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

    "output_dir_heatmap": "/usr/scratch/chromalyzer/heatmaps/",

    "output_dir_aligned": "/usr/scratch/chromalyzer/TII_aligned/",

}

chromalyzer.TII_alignment.align(config)