from loguru import logger
import json
import pandas as pd
import os

from .utils import create_folder_if_not_exists, load_heatmap_data, plt_heatmap
import argparse

def plot_heatmap(config, m_z, sample_name, all_samples):
    output_dir_heatmap = config['output_dir_heatmap']
    csv_file_name_column = config['csv_file_name_column']
    labels_path = config['labels_path']
    plot_dir = config['plot']['plot_dir']

    create_folder_if_not_exists(plot_dir)

    if all_samples:
        labels = pd.read_csv(labels_path)
        for sample in labels[csv_file_name_column].tolist():
            ht_df = load_heatmap_data(output_dir_heatmap, int(m_z), sample)
            save_path = os.path.join(plot_dir, f'm_z_{m_z}_sample_{sample}_heatmap.pdf')
            plt_heatmap(save_path, ht_df, full_spectrum=True, title=f'Heatmap for {sample} with m/z {m_z}', save=True)
            logger.info(f'Heatmap for {sample} with m/z {m_z} plotted successfully!')

    else:
        ht_df = load_heatmap_data(output_dir_heatmap, int(m_z), sample_name)
        save_path = os.path.join(plot_dir, f'm_z_{m_z}_sample_{sample_name}_heatmap.pdf')
        plt_heatmap(save_path, ht_df, full_spectrum=True, title=f'Heatmap for {sample_name} with m/z {m_z}', save=True)
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

    return parser


def main(args):
    with open(args.config_path, "r") as f:
        config = json.load(f)
    
    m_z = args.m_z
    sample_name = args.sample_name
    all_samples = args.all

    plot_heatmap(config,m_z,sample_name,all_samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__,
    )
    parsed_args = add_args(parser).parse_args()
    main(parsed_args)