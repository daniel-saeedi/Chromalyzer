import chromalyzer

config = {
    "parallel_processing": True,
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
    "eval_path":"/usr/scratch/chromalyzer/eval/NB/",

    # "model": "svm",
    # "svm": {
    #     "C": [1e-3,1e-2,1e-1,1e0,1e+1,1e+2,1e+3],
    #     "kernel": ["linear","poly","rbf","sigmoid"],
    #     "lambda1": [0.4],
    #     "lambda2": [2.0],
    #     "rt1_threshold": [35.04], # 10 * 3.504 = 35.04
    #     "rt2_threshold": [round(i * 0.008,3) for i in range(1,150)], # 5*0.008 = 0.04 - 150*0.008 = 1.2
    # },

    # "model": "xgboost",
    # "xgboost": {
    #     "n_estimators": [20, 50, 100, 200, 500],
    #     "lambda1": [0.75],
    #     "lambda2": [5.0],
    #     "rt1_threshold": [10],
    #     "rt2_threshold": range(20,120,20)
    # },

    # "model": "rf",
    # "rf": {
    #     "n_estimators": [20, 50, 100, 200, 500],
    #     "lambda1": [0.4],
    #     "lambda2": [2.0],
    #     "rt1_threshold": [35.04], # 10 * 3.504 = 35.04
    #     "rt2_threshold": [round(i * 0.008,3) for i in range(1,150)], # 5*0.008 = 0.04 - 150*0.008 = 1.2
    # }

    # "model": "lr_l2",
    # "lr_l2": {
    #     "C": [1e-4,1e-3,1e-2,1e-1,1e0,1e+1,1e+2,1e+3,1e+4],
    #     "lambda1": [0.4],
    #     "lambda2": [2.0],
        # "rt1_threshold": [35.04], # 10 * 3.504 = 35.04
        # "rt2_threshold": [round(i * 0.008,3) for i in range(1,150)], # 5*0.008 = 0.04 - 150*0.008 = 1.2
    # }


    "model": 'NaiveBayes',
    "NaiveBayes": {
        "alpha": [0.01, 0.1, 0.5, 1, 5, 10],
        "lambda1": [0.4],
        "lambda2": [2.0],
        "rt1_threshold": [35.04], # 10 * 3.504 = 35.04
        "rt2_threshold": [round(i * 0.008,3) for i in range(1,150)], # 5*0.008 = 0.04 - 150*0.008 = 1.2
    }
}

chromalyzer.evaluation.eval(config)