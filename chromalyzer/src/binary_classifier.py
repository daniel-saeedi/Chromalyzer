import json
from loguru import logger
import argparse
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import scipy.stats as stats
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
from .utils.heatmap_utils import create_folder_if_not_exists
from .utils.plot_utils import plot_2d_features, plot_3d_peaks, plot_3d_peaks_interactable, plot_3d_signatures, plot_3d_signatures_interactable, plot_distribution_of_peaks, plot_pca, plot_top_coefficients, plot_top_features
import joblib

def binary_class_signatures(coefficients_pvalues, features_info, results_path,samples,X_train, csv_file_name_column):
    signatures_class_0 = []
    signatures_class_1 = []

    for index,coefficient,p_value in coefficients_pvalues:
        index = int(index)
        m_z = features_info.iloc[index]['m/z']
        first_time_start = features_info.iloc[index]['RT1_start']
        second_time_start = features_info.iloc[index]['RT2_start']
        first_time_end = features_info.iloc[index]['RT1_end']
        second_time_end = features_info.iloc[index]['RT2_end']

        # What samples contain this signature?
        samples_with_this_sign = ''
        for sample in samples.iloc[np.where(X_train[:,index] == 1)[0].reshape(-1)][csv_file_name_column].to_numpy().tolist():
            samples_with_this_sign += sample +', '

        values, counts = np.unique(samples.iloc[np.where(X_train[:,index] == 1)[0].reshape(-1)].to_numpy().tolist(), return_counts=True)

        if counts.size == 0: continue
        
        cluster_label = int(values[np.argmax(counts)])
        if coefficient < 0:
            # Abiotic
            signatures_class_0.append([p_value,coefficient,m_z,f'[{first_time_start},{first_time_end}]',f'[{second_time_end},{second_time_start}]',samples_with_this_sign,cluster_label])
        else:
            # Biotic
            signatures_class_1.append([p_value,coefficient,m_z,f'[{first_time_start},{first_time_end}]',f'[{second_time_end},{second_time_start}]',samples_with_this_sign,cluster_label])


    create_folder_if_not_exists(os.path.join(results_path))
    signatures_class0 = pd.DataFrame(signatures_class_0,columns=['p_value','coefficient','m/z','RT1','RT2','samples','class'])
    signatures_class1 = pd.DataFrame(signatures_class_1,columns=['p_value','coefficient','m/z','RT1','RT2','samples','class'])

    signatures_class0.to_csv(os.path.join(results_path ,f'lr_l2_class0_signatures.csv'))
    signatures_class1.to_csv(os.path.join(results_path, f'lr_l2_class1_signatures.csv'))

    return signatures_class0, signatures_class1
    

def calculate_pvalues(num_bootstraps, C, seed, X_train, samples, label_column_name):
    # Bootstrap sampling
    coefficients = []
    for i in range(num_bootstraps):
        model = LogisticRegression(penalty='l2', solver='liblinear', C=C, random_state=seed)
        X_boot, y_boot = resample(X_train, samples[label_column_name].to_numpy(), random_state=i)  # Resample with replacement
        
        # Check if both classes are present in the resampled data
        if len(np.unique(y_boot)) < 2:
            continue  # Skip this iteration if only one class is present
        
        model.fit(X_boot, y_boot)  # Fit the model on the bootstrapped sample
        coefficients.append(model.coef_[0])  # Store the coefficients

    coefficients = np.array(coefficients)
    if len(coefficients) == 0:
        raise ValueError("All resampled datasets contained only one class. Unable to calculate p-values.")

    mean_coefs = np.mean(coefficients, axis=0)  # Average coefficient from bootstraps
    standard_errors = np.std(coefficients, axis=0)  # Standard errors of coefficients

    # Calculate Z-scores and p-values
    z_scores = mean_coefs / (standard_errors + 1e-20)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    return p_values

def is_point_inside_rect(rectangle, point):
    x1 = rectangle[0]
    y1 = rectangle[1]
    x2 = rectangle[2]
    y2 = rectangle[3]
    # Check if the point is within the rectangle bounds
    if x1 <= point[0] <= x2 and y1 <= point[1] <= y2:
        return True

    return False

def return_peaks_corresponding_to_cluster(peaks_path,m_z,rt1,rt2,feature_id,coefficient,p_value,samples, label):
    peaks = pd.read_csv(os.path.join(peaks_path, f'{m_z}.csv'))

    peaks_inside_cluster = []
    for idx, peak in peaks.iterrows():
        if is_point_inside_rect([rt1[0],rt2[0],rt1[1],rt2[1]],[peak['RT1_center'],peak['RT2_center']]):
            peaks_inside_cluster.append((feature_id,peak['RT1_center'],peak['RT2_center'],coefficient,p_value,m_z, peak['csv_file_name'], label))
    
    return peaks_inside_cluster

def get_peaks_feature_df(signaturs_combined,peaks_path):
    peaks_features_id = []
    for idx, row in signaturs_combined.iterrows():
        m_z = int(row['m/z'])
        rt1 = eval(row['RT1'])
        rt2 = eval(row['RT2'])
        peaks_features_id += (return_peaks_corresponding_to_cluster(peaks_path, m_z,rt1,rt2,idx,row['coefficient'],row['p_value'],row['samples'],row['class']))
    
    peaks_features_df = pd.DataFrame(np.array(peaks_features_id),columns=['feature_id','RT1','RT2','coefficient','p_value','m/z','sample','class'])

    peaks_features_df['feature_id'] = peaks_features_df['feature_id'].astype(int)
    peaks_features_df['coefficient'] = peaks_features_df['coefficient'].astype(float)
    peaks_features_df['RT1'] = peaks_features_df['RT1'].astype(float)
    peaks_features_df['RT2'] = peaks_features_df['RT2'].astype(float)
    peaks_features_df['m/z'] = peaks_features_df['m/z'].astype(float)
    peaks_features_df['p_value'] = peaks_features_df['p_value'].astype(float)
    num_clusters = peaks_features_df['feature_id'].max()

    return peaks_features_df, num_clusters

def binary_classifier(args):
    log_path = os.path.join(args['results_dir'], 'results.log')
    logger.add(log_path, rotation="10 MB")

    seed = args['seed']
    np.random.seed(seed)
    lam1 = args['lambda1']
    lam2 = args['lambda2']
    rt1_th = args['rt1_threshold']
    rt2_th = args['rt2_threshold']
    C = args['C']

    # Load the data
    features = np.load(os.path.join(args['features_path'], f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.npy'))
    features_info = pd.read_csv(os.path.join(args['features_path'], f'features_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.csv'))

    # Load the labels
    samples = pd.read_csv(args['labels_path'])

    # Unlabel the samples
    unlabeled_samples_index = samples[samples[args['label_column_name']] == -1].index

    # Remove the unlabeled samples indices from the features
    X_train = np.delete(features, unlabeled_samples_index, axis=0)
    y_train = samples.drop(unlabeled_samples_index)['label'].to_numpy()
    train_samples = samples.drop(unlabeled_samples_index)
    X_test = features[unlabeled_samples_index]
    test_samples = samples.iloc[unlabeled_samples_index]

    # Train the model
    lr = LogisticRegression(penalty='l2', solver='liblinear', C=C,random_state=seed)
    lr.fit(X_train, y_train)
    coefficients = lr.coef_[0]

    # Save the trained model
    create_folder_if_not_exists(args['results_dir'])
    joblib.dump(lr, os.path.join(args['results_dir'], f'lr_l2_model_lam1_{lam1}_lam2_{lam2}_rt1th_{rt1_th}_rt2th_{rt2_th}.joblib'))

    # Predicting the labels of the unlabeled samples
    for i in range(len(test_samples)):
        if test_samples.iloc[i]['label'] == -1:
            pred = lr.predict(X_test[i].reshape(1, -1))
            sample_name = test_samples.iloc[i][args['csv_file_name_column']]
            logger.info(f'Predicted label for {sample_name}: {pred}')

    # Get the p-values
    p_values = calculate_pvalues(args['number_of_bootstraps'], C, seed, X_train, train_samples, args['label_column_name'])
    logger.info('P-values calculated.')

    # Combine the coefficients and p-values
    coefficients_pvalues = sorted(enumerate(coefficients), key=lambda x: abs(x[1]), reverse=True)
    coefficients_pvalues = np.array([(index, value, p_values[index]) for index, value in coefficients_pvalues])

    # Save signatures per class
    signatures_class0, signatures_class1 = binary_class_signatures(coefficients_pvalues, features_info, args['results_dir'],train_samples,X_train, args['csv_file_name_column'])
    logger.info('Signatures saved.')

    # Plotting top 10 highest to lowest coefficients
    top_featurs_path = os.path.join(args['results_dir'] ,'top_features/')
    create_folder_if_not_exists(top_featurs_path)
    plot_top_coefficients(coefficients_pvalues, top_featurs_path)
    plot_top_features(X_train, coefficients_pvalues, train_samples, top_featurs_path, args['label_column_name'], args['csv_file_name_column'])
    signaturs_combined = pd.concat([signatures_class0,signatures_class1]).sort_values(by='coefficient',key=abs,ascending=False).reset_index(drop=True)
    signaturs_combined.index = signaturs_combined.index + 1
    signaturs_combined.head(10).to_csv(os.path.join(top_featurs_path,'lr_l2_signatures_combined.csv'))
    logger.info('Top 10 signatures plotted in top_coefficients folder.')

    # Plotting PCA
    plot_pca(features, samples[args['label_column_name']].tolist(), samples[args['csv_file_name_column']].tolist() ,coefficients_pvalues, args['results_dir'])
    logger.info('PCA plot saved.')

    # Plotting 2D plot of peaks and signatures.
    peaks_features_df, num_clusters = get_peaks_feature_df(signaturs_combined,os.path.join(args['peaks_dir_path'], f'peaks_lambda1_{lam1}/',f'lam2_{lam2}/'))
    plot_2d_features(signaturs_combined, peaks_features_df, num_clusters, args['results_dir'])

    logger.info('2D plot of peaks and signatures saved.')
    # Plotting 3D plot of peaks (png)
    plot_3d_peaks(peaks_features_df, samples,args['results_dir'])

    # Plotting 3D plot of signatures (png)
    plot_3d_signatures(signaturs_combined, args['results_dir'])
    logger.info('3D plot of signatures (png) saved.')

    # Interactable 3D plot for peaks
    plot_3d_peaks_interactable(peaks_features_df, args['results_dir'])
    logger.info('3D interactive plot of peaks saved.')

    # Interactable 3D plot for signatures
    plot_3d_signatures_interactable(signaturs_combined, args['results_dir'])
    logger.info('3D interactive plot of signatures saved.')

    # Distribution of peaks across m/z values
    plot_distribution_of_peaks(peaks_features_df, args['results_dir'])
    logger.info('Distribution of peaks across m/z values saved.')