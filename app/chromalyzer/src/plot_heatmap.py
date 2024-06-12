from loguru import logger
import json
import pandas as pd
import os

from .utils import load_heatmap_data, plt_heatmap

def plot_heatmap(config, m_z, sample_name, all_samples):
    output_dir_heatmap = config['output_dir_heatmap']
    csv_file_name_column = config['csv_file_name_column']
    labels_path = config['labels_path']

    if all_samples:
        labels = pd.read_csv(labels_path)
        for sample in labels[csv_file_name_column].tolist():
            heatmap_path = os.path.join(output_dir_heatmap, sample)
            ht_df = load_heatmap_data(heatmap_path, m_z, sample)
            plt_heatmap(output_dir_heatmap, ht_df, full_spectrum=True, title=f'Heatmap for {sample} with m/z {m_z}', save=True)
            logger.info(f'Heatmap for {sample} with m/z {m_z} plotted successfully!')

    else:
        heatmap_path = os.path.join(output_dir_heatmap, sample_name)
        ht_df = load_heatmap_data(heatmap_path, m_z, sample_name)
        plt_heatmap(output_dir_heatmap, ht_df, full_spectrum=True, title=f'Heatmap for {sample_name} with m/z {m_z}', save=True)
        logger.info(f'Heatmap for {sample_name} with m/z {m_z} plotted successfully!')

def add_args(parser):
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the config file.",
    )

    parser.add_argument(
        "--m_z",
        type=float,
        required=True,
        help="Which m/z value to plot heatmap for.",
    )

    parser.add_argument(
        "--sample_name",
        type=str,
        required=False,
        default='',
        help="File Name of the sample to plot heatmap for. This name should be present in the labels.csv",
    )

    parser.add_argument(
        "--all",
        type=bool,
        default=False,
        required=False,
        help="If you want to plot heatmap for all the samples in the labels.csv file, set this flag to True.",
    )

def main(args):
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    m_z = args.m_z
    sample_name = args.sample_name
    all_samples = args.all

    # print(config['extract_heatmaps']['raw_csv_path'])
    plot_heatmap(config,m_z,sample_name,all_samples)
