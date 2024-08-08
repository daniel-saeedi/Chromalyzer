import itertools
import os
from loguru import logger
import pandas as pd
import numpy as np
from .utils.heatmap_utils import create_folder_if_not_exists
from .utils.rt_alignment_utils import find_cluster_features, which_cluster
import concurrent.futures

class RetentionTimesAlignment:
    def __init__(self, samples_name_list : pd.DataFrame,
                  mz_list : list, peaks_dir_path : str, features_out_path: str, 
                  rt1_threshold : int, rt2_threshold : int, rt1_time_step : int = 3.504, rt2_time_step : int = 0.008) -> None:
        self.samples_name_list = samples_name_list
        self.mz_list = mz_list
        self.peaks_dir_path = peaks_dir_path
        self.features_out_path = features_out_path
        self.rt1_threshold = rt1_threshold
        self.rt2_threshold = rt2_threshold
        self.rt1_time_step = rt1_time_step
        self.rt2_time_step = rt2_time_step

        create_folder_if_not_exists(self.features_out_path)
    
    """
    This function aligns retention times by considering peaks within a certain threshold as the same peak. 
    For example, if the retention time 1 of a peak is 10.0 and the threshold is 0.1, then all peaks with retention time 1 
    between 9.9 and 10.1 will be considered the same peak.

    Output:
    - samples_features_combined: A matrix where each row corresponds to a sample and each column corresponds to a feature. 
    If the value at (i, j) in the matrix is 1, then the j-th peak is present in the i-th sample.
    - features_df_combined: A DataFrame where each row corresponds to a feature and each column corresponds to a property of the feature.
    The columns are: 'm/z', 'RT1_start', 'RT2_start', 'RT1_end', 'RT2_end', 'RT1_center', 'RT2_center'.
    """
    def align_retention_times(self):
        all_features = None
        all_features_info = pd.DataFrame([])

        for m_z in map(int, self.mz_list):
            features, features_info = self.generate_features(m_z, self.samples_name_list)

            if features_info:
                all_features = features if all_features is None else np.concatenate([all_features, features], axis=1)
                features_info = pd.DataFrame(features_info, columns=['m/z', 'RT1_start', 'RT2_start', 'RT1_end', 'RT2_end', 'RT1_center', 'RT2_center'])
                all_features_info = pd.concat([all_features_info, features_info])

        all_features_info.reset_index(drop=True, inplace=True)
        return all_features, all_features_info

    # Returns cluster maps and clusters information(e.g. their starting and end point) for a specific M/Z
    def generate_features(self, m_z, samples_name):
        # Load peaks data from the specified file
        peaks_file_path = os.path.join(self.peaks_dir_path, f'{m_z}.csv')
        peaks = pd.read_csv(peaks_file_path)

        # Check if peaks data is not empty
        if not peaks.empty:
            # Define the range of retention times
            rt1_min, rt1_max, rt1_time_step = peaks['RT1_center'].min(), peaks['RT1_center'].max(), self.rt1_time_step
            rt2_min, rt2_max, rt2_time_step = peaks['RT2_center'].min(), peaks['RT2_center'].max(), self.rt2_time_step

            # Create arrays of retention times
            rt1_val = np.round(np.arange(rt1_min, rt1_max + rt1_time_step, rt1_time_step), 3)
            rt2_val = np.round(np.sort(np.arange(rt2_min, rt2_max + rt2_time_step, rt2_time_step))[::-1], 3)

            # Initialize an empty heatmap DataFrame
            empty_heatmap = pd.DataFrame(0, index=rt2_val, columns=rt1_val)

            # Fill the heatmap with peak data
            for _, row in peaks.iterrows():
                empty_heatmap.at[row['RT2_center'], row['RT1_center']] = 1

            # Find features using clustering
            cluster_centers, cluster_rectangles = find_cluster_features(empty_heatmap, self.rt1_threshold,self.rt2_threshold)

            # Initialize the cluster map
            cluster_map = np.zeros((len(samples_name), len(cluster_rectangles)))

            # Populate the cluster map for each sample
            for sample_id, sample in enumerate(samples_name):
                sample_peaks_df = peaks[peaks['csv_file_name'] == sample]
                for _, peak in sample_peaks_df.iterrows():
                    clusters = which_cluster(rt2_val, rt1_val, peak, cluster_rectangles)
                    for cluster in clusters:
                        cluster_map[sample_id, cluster] = 1

            # Collect clusters information
            clusters_info = [
                (
                    m_z,
                    rt1_val[cluster_rectangle[1]], rt2_val[cluster_rectangle[0]],  # RT1_start, RT2_start
                    rt1_val[cluster_rectangle[3]], rt2_val[cluster_rectangle[2]],  # RT1_end, RT2_end
                    rt1_val[cluster_centers[idx][1]], rt2_val[cluster_centers[idx][0]]  # RT1_center, RT2_center
                )
                for idx, cluster_rectangle in enumerate(cluster_rectangles)
            ]

            return cluster_map, clusters_info

        else:
            return [], []

def retention_times_alignment_process(params,peaks_dir_path,samples_name_list,mz_list,features_out_path):
    for param in params:
        lam1 = param[0]
        lam2 = param[1]
        rt1_threshold,rt2_threshold = int(param[2]),int(param[3])

        peaks_path = os.path.join(peaks_dir_path, f'peaks_lambda1_{lam1}/', f'lam2_{lam2}/')
        rta = RetentionTimesAlignment(samples_name_list, mz_list, peaks_path, features_out_path, rt1_threshold, rt2_threshold)
        features, feature_info_df = rta.align_retention_times()

        np.save(os.path.join(features_out_path, f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_threshold}_rt2th_{rt2_threshold}.npy'), features)
        feature_info_df.to_csv(os.path.join(features_out_path, f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_threshold}_rt2th_{rt2_threshold}.csv'))

        logger.info(f'Features for lambda1={lam1}, lambda2={lam2}, rt1_threshold={rt1_threshold}, rt2_threshold={rt2_threshold} are saved.')

def retention_times_alignment(config):
    log_path = os.path.join(config['peaks_dir_path'], 'find_peaks.log')
    logger.add(log_path, rotation="10 MB")

    # Load the m/z list
    m_zs = pd.read_csv(config['mz_list_path'])
    mz_list = m_zs[config['m_z_column_name']].tolist()

    # Samples name
    samples = pd.read_csv(config['labels_path'])
    samples_name = samples[config['csv_file_name_column']].tolist()

    lam1 = config['lambda1']
    lam2 = config['lambda2']

    rt1_threshold = config['rt1_threshold']
    rt2_threshold = config['rt2_threshold']

    params_combination = list(itertools.product(lam1,lam2,rt1_threshold,rt2_threshold))

    splits = np.array_split(params_combination, config['number_of_splits'])

    create_folder_if_not_exists(os.path.join(config['features_path'],'peaks_per_parameter/'))

    if config['parallel_processing']:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            executor.map(retention_times_alignment_process, splits, itertools.repeat(config['peaks_dir_path']), 
                         itertools.repeat(samples_name), itertools.repeat(mz_list), 
                         itertools.repeat(config['features_path']))
    else:
        retention_times_alignment_process(params_combination, config['peaks_dir_path'], samples_name, mz_list, config['features_path'])
    
    logger.info('Retention times alignment process is completed.')