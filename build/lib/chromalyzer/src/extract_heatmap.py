from loguru import logger
import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from .utils import heatmap, filter_by_m_z

from .utils import create_folder_if_not_exists

# Extract heatmaps for a given m/z list
def extract_heatmap(args):
    mz_list_path = args.mz_list_path
    labels_path = args.labels_path
    raw_csv_path = args.raw_csv_path
    threshold = args.m_z_threshold
    output_dir_heatmap = args.output_dir_heatmap
    area_column_name = args.area_column_name
    m_z_column_name = args.m_z_column_name
    first_time_column_name = args.first_time_column_name
    second_time_column_name = args.second_time_column_name
    csv_file_name_column = args.csv_file_name_column


    logger.info(f"Extracting heatmaps for m/z list: {mz_list_path}")

    # Load the m/z list
    mz_list = pd.read_csv(mz_list_path)[m_z_column_name]

    # Load the labels
    labels = pd.read_csv(labels_path)

    for csv_file_name in tqdm(labels[csv_file_name_column].tolist()):
        # Load the raw data
        logger.info(f"Reading csv file: {csv_file_name}....")
        raw_csv_path = os.path.join(raw_csv_path, csv_file_name)
        raw_sample = pd.read_csv(raw_csv_path)
        logger.info(f"Finished reading csv file: {csv_file_name}")

        # Create a folder for the csv file
        create_folder_if_not_exists(os.path.join(output_dir_heatmap,f'{csv_file_name}/'))

        # Extract heatmaps for each m/z of the sample
        for idx, row in mz_list.iterrows():
            m_z = int(row[m_z_column_name])
            sample_df = filter_by_m_z(raw_sample, m_z, threshold=threshold, m_z_column_name=m_z_column_name)
            hm = heatmap(sample_df, first_time_column_name=first_time_column_name, second_time_column_name=second_time_column_name, area_column_name=area_column_name)

            # Save the heatmap
            file_path = os.path.join(output_dir_heatmap,f'{csv_file_name}/', str(int(m_z)) + '.npy')
            np.save(file_path, hm.to_numpy())

            # Saving first time array
            first_time = hm.columns.to_numpy()
            file_path = os.path.join(output_dir_heatmap,f'{csv_file_name}/', str(int(m_z)) + '_first_time.npy')
            np.save(file_path, first_time)

            # Saving the second time array
            second_time = hm.index.to_numpy()
            file_path = os.path.join(output_dir_heatmap,f'{csv_file_name}/', str(int(m_z)) + '_second_time.npy')
            np.save(file_path, second_time)

            logger.info(f"Saved heatmap for m/z: {m_z} to {file_path}")


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mz_list_path",
        type=str,
        required=True,
        help="Path to the a csv file containing the m/z list to extract heatmaps for.",
    )

    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to the a csv file containing the labels and sample names.",
    )

    parser.add_argument(
        "--raw_csv_path",
        type=str,
        required=True,
        help="Path to the raw csv file containing the GCxGC data.",
    )
    
    parser.add_argument(
        "--threshold",
        type=str,
        required=True,
        help="Threshold for the heatmap extraction. For example, with a threshold of 0.5 and m/z of 100, the heatmap will be extracted for the range of 99.5 to 100.5.",
    )

    parser.add_argument(
        "--output_dir_heatmap",
        type=str,
        required=True,
        help="Path to the output directory to save the extracted heatmaps.",
    )

    args = parser.parse_args()

    extract_heatmap(args)

