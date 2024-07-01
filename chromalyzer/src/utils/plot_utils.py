import random
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import seaborn as sns
import matplotlib.colors as mcolors
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib as mpl
import matplotlib.cm as cm
from svgpathtools import svg2paths
from svgpath2mpl import parse_path


def plot_top_coefficients(coefficients_pvalues, results_dir, top_n=10):
    """
    Plot the top n coefficients with the highest absolute values.
    """
    
    # Get the top 30 feature importances
    top_features = coefficients_pvalues[:top_n]

    # Extract the importance values and feature names
    coefficientections = [coefficientection for _, coefficientection , _ in top_features]
    feature_ids = [i for i in range(0,top_n)]

    # Plot the feature importances
    plt.figure(figsize=(5, 3))
    for feature_id in feature_ids:
        label = 'Class 0' if coefficientections[feature_id] < 0 else 'Class 1'
        plt.bar(feature_id,abs(coefficientections[feature_id]),label=label,color='#ff3333' if label == 'Class 0' else '#3c5488')
    plt.xlabel('Feature')
    plt.ylabel('Coefficient')
    plt.xticks(feature_ids,range(1,top_n+1), rotation=90)

    # Adjust x-axis limits
    plt.xlim(-0.5, len(feature_ids) - 0.5)

    # Add border lines to left and bottom
    plt.gca().spines['left'].set_color('grey')
    plt.gca().spines['left'].set_linewidth(1)
    plt.gca().spines['bottom'].set_color('grey')
    plt.gca().spines['bottom'].set_linewidth(1)

    # Remove the background
    plt.gca().set_facecolor('white')
    colors = {'Class 0':'#ff3333', 'Class 1':'#3c5488'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.show()
    plt.savefig(os.path.join(results_dir,f'top_{top_n}_coefficient.pdf'),format='pdf', bbox_inches='tight', dpi=300)

def plot_top_features(X_train, coefficients_pvalues, train_samples, results_dir, label_column_name, csv_file_name_column, top_n=10):
    X_selected = X_train[:, coefficients_pvalues[:top_n,0].astype(int)]

    X_sorted = X_selected[train_samples.sort_values(label_column_name).index].copy()

    class_0_count = np.sum(train_samples[label_column_name] == 0)
    X_sorted[class_0_count:,:] = np.where(X_sorted[class_0_count:,:] == 1, 2, X_sorted[class_0_count:,:])


    plt.figure(figsize=(16,12))
    sns.set(font_scale=1.4)

    # Define colors for each value
    colors = ["white", "#ff3333", "#3c5488"]  # Colors for 0, 1, and 2 respectively
    cmap = mcolors.ListedColormap(colors)

    boundaries = [-0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(boundaries, cmap.N, clip=True)


    sns.heatmap(X_sorted, cmap=cmap, yticklabels=train_samples.sort_values(label_column_name)[csv_file_name_column].to_numpy(),xticklabels=range(1,top_n+1) ,cbar=False, linecolor='gray', linewidth=0.5,square=True)
    # Save the plot as a PDF file
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,f'top{top_n}_features.pdf'), format="pdf", bbox_inches='tight',dpi=400)

def add_arrow(x_start,y_start,x_end,y_end, text,line_color='#a1caf7'):
    plt.annotate(text,               # text to display
                 xy=(x_end, y_end),                     # point to annotate
                 xytext=(x_start, y_start),       # position of text
                 textcoords='offset points',    # how to interpret xytext
                 ha='right',                    # horizontal alignment
                 va='center',                   # vertical alignment
                 arrowprops=dict(arrowstyle='-', lw=1, color=line_color), # arrow style and color,
                 fontsize=15,
                 color=line_color)  # background box for text
    
def plot_pca(features, labels, samples_name ,coefficients_pvalues, results_dir, top_n = 20):
    X_selected = features[:, coefficients_pvalues[:top_n,0].astype(int)]

    pca = PCA(n_components=2)
    pca.fit(X_selected)
    X_embedded = pca.transform(X_selected)
    plt.figure(figsize=(10, 5))

    ax = plt.gca()  # Get current axis
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Set the color of the x-axis and y-axis
    plt.gca().spines['bottom'].set_edgecolor('black')  # Color for the x-axis
    plt.gca().spines['left'].set_edgecolor('black')   # Color for the y-axis

    results = pd.DataFrame(np.concatenate([X_embedded,np.array(labels).reshape(len(labels),-1)],axis = 1),columns=['PC1','PC2','label'])

    for label in results['label'].unique():
        if label == -1:
            color = 'black'
            marker = 'O'
            l = 'Unkown'
        else:
            color = '#3c5488' if label == 1 else '#e64b35' # Set color based on label
            marker = 'D' if label == 1 else 'x'  # Set marker based on label
            l = 'Biotic' if label == 1 else 'Abiotic'
        plt.scatter(results[results['label'] == label]['PC1'], results[results['label'] == label]['PC2'], label=l, s=100, c=color,marker=marker)

    for idx, sample_name in enumerate(samples_name):
        # Add text labels with arrows
        x_end, y_end = X_embedded[idx]
        random_directions = [random.choice([-1, 1]), random.choice([-1, 1])]
        add_arrow(x_end + random_directions[0] * np.random.randint(15, 80), x_end + random_directions[1] * np.random.randint(15, 80), x_end, y_end, idx)


    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.gcf().set_facecolor('white')  # gcf() - get current figure
    plt.gca().set_facecolor('white')  # gca() - get current axis
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir,'PCA_top20.pdf'),format='pdf',bbox_inches='tight',dpi=400)
    plt.close()

def plot_2d_features(all_signatures, peaks_features_df, num_clusters, result_dir):
    plt.figure(figsize=(15,5))

    # Assign colors based on the 'coefficient' column
    colors = ['#f8951e' if x < 0 else '#35b549' for x in peaks_features_df['coefficient']]

    # Plot the scatter plot with conditional colors
    plt.scatter(peaks_features_df['RT1'], peaks_features_df['RT2'], c=colors, s=10)

    plt.xlabel('1st Time', fontsize=12)  # Increase font size for x-axis label
    plt.ylabel('2nd Time', fontsize=12)  # Increase font size for y-axis label

    plt.xticks(fontsize=12)  # Increase font size for x-axis ticks
    plt.yticks(fontsize=12)  # Increase font size for y-axis ticks

    maxc_maxr = []

    i = 1
    colors_per_cluster = ['green']
    for cluster_id in range(1,num_clusters+1):
        rt1 =  eval(all_signatures.iloc[cluster_id-1]['RT1'])
        rt2 =  eval(all_signatures.iloc[cluster_id-1]['RT2'])

        minc = rt1[0] - 20
        maxc = rt1[1] + 20
        minr = rt2[0] - 0.01
        maxr = rt2[1] + 0.01

        #  {'Abiotic':'#f8951e', 'Biotic':'#35b549'}
        color = '#f8951e' if all_signatures.iloc[cluster_id-1]['coefficient'] < 0 else '#35b549'

        plt.plot([minc, maxc], [minr, minr], color)  # Top line
        plt.plot([minc, maxc], [maxr, maxr], color)  # Bottom line
        plt.plot([minc, minc], [minr, maxr], color)  # Left line
        plt.plot([maxc, maxc], [minr, maxr], color)  # Right line

        maxc_maxr.append((maxc,maxr))

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')

    # Background color
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    plt.savefig(os.path.join(result_dir,'2d_plots_peaks_signatures.pdf'),format='pdf',bbox_inches='tight', dpi=300)
    plt.close()

def plot_3d_peaks_interactable(peaks_features_df, result_dir):
    peaks_features_df_sorted = peaks_features_df.sort_values(by=['m/z','RT1', 'RT2'], ascending=[True,True, True])
    # Create the initial scatter plot
    fig = px.scatter_3d(peaks_features_df_sorted, x='RT1', y='RT2', z='m/z', color='class', hover_data=['sample','p_value'])
    # Update marker size scaling factor if necessary
    fig.update_traces(marker=dict(sizemode='diameter', sizeref=2, sizemin=1,opacity = 0.5))
    # Update layout
    fig.update_layout(scene=dict(
        xaxis_title='1st Time',
        yaxis_title='2nd Time',
        zaxis_title='m/z',
        aspectratio=dict(x=4, y=1, z=1),
    ))
    fig.write_html(os.path.join(result_dir,'3d_plots_peaks_interactable.html'))

def plot_3d_signatures_interactable(all_signatures, result_dir): 
    rt1_center = []
    rt2_center = []
    for idx, signaure in all_signatures.iterrows():
        rt1 = eval(signaure['RT1'])
        rt2 = eval(signaure['RT2'])

        rt1_center.append((rt1[0] + rt1[1]) / 2)
        rt2_center.append((rt2[0] + rt2[1]) / 2)

    all_signatures['RT1_center'] = rt1_center
    all_signatures['RT2_center'] = rt2_center
    all_signatures['coefficient_abs'] = all_signatures['coefficient'].abs()

    all_signatures['class'] = all_signatures['class'].astype(str)

    all_signatures_sorted = all_signatures.sort_values(by=['m/z','RT1_center', 'RT2_center'], ascending=[True,True, True])

    # absolute value of the coefficient
    all_signatures_sorted['coefficient_abs'] = all_signatures_sorted['coefficient'].abs()
    # Create the initial scatter plot
    fig = px.scatter_3d(all_signatures_sorted, x='RT1_center', y='RT2_center', z='m/z', color='class', size='coefficient_abs', hover_data=['samples','RT1','RT2','p_value'])

    # Update marker size scaling factor if necessary
    fig.update_traces(marker=dict(sizemode='diameter', sizeref=2.*max(all_signatures_sorted['coefficient_abs'])/(4.**4), sizemin=1,opacity = 0.5))

    # Update layout
    fig.update_layout(scene=dict(
        xaxis_title='1st Time',
        yaxis_title='2nd Time',
        zaxis_title='m/z',
        aspectratio=dict(x=4, y=1, z=1),
    ))

    fig.write_html(os.path.join(result_dir,'3d_plots_signatures_interactable.html'))

def plot_3d_signatures(all_signatures, result_dir, view = 'small'):
    all_signatures = all_signatures.copy()
    rt1_center = []
    rt2_center = []
    for idx, signaure in all_signatures.iterrows():
        rt1 = eval(signaure['RT1'])
        rt2 = eval(signaure['RT2'])

        rt1_center.append((rt1[0] + rt1[1]) / 2)
        rt2_center.append((rt2[0] + rt2[1]) / 2)

    all_signatures['RT1_center'] = rt1_center
    all_signatures['RT2_center'] = rt2_center
    all_signatures['coefficient_abs'] = all_signatures['coefficient'].abs()

    all_signatures['class'] = all_signatures['class'].astype(str)

    all_signatures = all_signatures.sort_values(by=['m/z','RT1_center', 'RT2_center'], ascending=[True,True, True])

    if view == 'small':
        legend_sizes = [3000,
                        1500,
                        750,
                        355,
                        150]
    else:
        legend_sizes = [5000,
                        2500,
                        1800,
                        800,
                        400]

    all_signatures['point_size'] = 0
    for idx, row in all_signatures.iterrows():

        if row['coefficient_abs'] >= 0.02:
            all_signatures.at[idx, 'point_size'] = legend_sizes[0]
        elif row['coefficient_abs'] >= 0.015:
            all_signatures.at[idx, 'point_size'] = legend_sizes[1]
        elif row['coefficient_abs'] >= 0.01:
            all_signatures.at[idx, 'point_size'] = legend_sizes[2]
        elif row['coefficient_abs'] >= 0.005:
            all_signatures.at[idx, 'point_size'] = legend_sizes[3]
        else:
            all_signatures.at[idx, 'point_size'] = legend_sizes[4]

    all_signatures = all_signatures.sort_values(by='point_size', ascending=False)

    # Create a dictionary to map class strings to specific colors
    color_map = {'0': '#e64b35', '1': '#3c5488'}

    if view == 'small':
        fig = plt.figure(figsize=(12, 12))
    else:
        fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(111, projection='3d')

    # Convert class to string if not already and adjust RT1_center for minutes and rounding
    all_signatures['class'] = all_signatures['class'].astype(str)
    all_signatures['RT1_center'] = (all_signatures['RT1_center'] / 60).astype(int)

    # Scatter plot
    sc = ax.scatter(
        all_signatures['RT2_center'],
        all_signatures['RT1_center'],
        all_signatures['m/z'],
        c=[color_map[x] for x in all_signatures['class']],
        s=all_signatures['point_size'],
        alpha=1,
        edgecolors='w',
    )

    # Adding labels and title
    ax.set_xlabel('2nd Time (s)', labelpad=20)
    ax.set_ylabel('1st Time (min)', labelpad=20)
    ax.set_zlabel('m/z', labelpad=20)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=20)

    # Set background color to white and edge color to black
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    if view == 'small':
        ax.set_box_aspect([2.5, 4, 2])
    else:
        ax.set_box_aspect([3, 4, 2])

    # Adjusting the view and plot appearance
    # Adjusting the view and plot appearance
    if view == 'small':
        ax.view_init(elev=20, azim=-120)
    else:
        ax.view_init(elev=10, azim=-179)
    plt.tight_layout()

    # Create legend handles
    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=np.sqrt(s), label=label)
                    for s, label in zip(legend_sizes, ['> 0.02', '0.015 - 0.02','0.01 - 0.015','0.05 - 0.01','< 0.005'])]

    # Add the legend to the plot
    legend1 = ax.legend(handles=legend_handles, title="Coefficent", fontsize='16', title_fontsize='16',handlelength=3,labelspacing=2, loc='upper right')
    ax.add_artist(legend1)

    ax.legend(handles=[Patch(facecolor=color_map['0'], label='Abiotic'), Patch(facecolor=color_map['1'], label='Biotic')], fontsize='16', title_fontsize='16',handlelength=3,labelspacing=1, loc='upper left' )

    ax.set_xlim(all_signatures['RT2_center'].min(), 3.5)
    ax.set_ylim(all_signatures['RT1_center'].min()-10, 190)
    ax.set_zlim(all_signatures['m/z'].min(),680)

    # Background color
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    # Change the color of the gridlines
    ax.xaxis._axinfo['grid'].update(color = 'gray', linestyle = '-' , linewidth=0.5)
    ax.yaxis._axinfo['grid'].update(color = 'gray', linestyle = '-', linewidth=0.5)
    ax.zaxis._axinfo['grid'].update(color = 'gray', linestyle = '-', linewidth=0.5)

    plt.savefig(os.path.join(result_dir, 'Signatures_3d.pdf'), format='pdf')
    plt.close()

import matplotlib.markers as mmarkers

def create_half_filled_marker(rotation=45):
    # Define the marker path
    marker = mmarkers.MarkerStyle(marker='o')
    path = marker.get_path().transformed(marker.get_transform())

    # Create a wedge to represent the half-filled portion
    wedge = mmarkers.MarkerStyle(marker='|')
    wedge_path = wedge.get_path().transformed(Affine2D().scale(1.5, 1.5).rotate_deg(rotation))

    # Combine the paths
    combined_path = path.intersect_path(wedge_path)
    
    return combined_path

def get_marker(svg_path):
    cwd = os.getcwd()
    custom_path, attributes = svg2paths(svg_path)
    marker = parse_path(attributes[0]['d'])
    marker.vertices -= marker.vertices.mean(axis=0)
    marker = marker.transformed(mpl.transforms.Affine2D().rotate_deg(180))
    marker = marker.transformed(mpl.transforms.Affine2D().scale(-1,1))
    return marker
def plot_3d_peaks(peaks_features_df, samples,result_dir, label = 'biotic', view = 'small'):
    peaks_features_df = peaks_features_df.copy()
    if label == 'biotic':
        selected_samples = samples[samples['label']==1]['csv_file_name'].unique()
    else:
        selected_samples = samples[samples['label']==0]['csv_file_name'].unique()
    peaks_features_df = peaks_features_df[peaks_features_df['sample'].isin(selected_samples)]

    peaks_features_df['point_size'] = 700
    
    import matplotlib.path as mpath

    verts = [
        (1,-1), #lower right
        (-1,-1), #lower left 
        (-1,1), #upper left 
        (1,1), #upper right
        (1,-1), #lower right
    ]

    codes = [
        Path.MOVETO, #begin the figure in the lower right
        Path.CURVE3, #start a 3 point curve with the control point in lower left
        Path.LINETO, #end curve in the upper left
        Path.CURVE3, #start a new 3 point curve with the upper right as a control point
        Path.LINETO, #end curve in lower right
    ]

    oval = Path(verts,codes)

    star_marker = get_marker('markers/star_custom.svg')
    pentagon_marker = get_marker('markers/pentagon_curved.svg')

    
    markers_biotic = ['P',MarkerStyle("o", fillstyle="left"),'X','H','h','x',star_marker,'d',oval,pentagon_marker]
    colors_biotic = ["#d9ed92", "#b5e48c", "#99d98c", "#76c893", "#52b69a", "#27375f", "#34a0a4", "#1a759f", "#1e6091", "#3c5488"]
    random.seed(43)
    # Shuffle only positions of markers and colors
    random.shuffle(markers_biotic)
    random.shuffle(colors_biotic)

    markers_abiotic = ['v','^','<','>','1','s','p','D']
    colors_abiotic = ["#ff6655", "#e64b35", "#d6452e", "#cc4130", "#b33a28", "#993322", "#803020", "#ff5e4d", "#e6594c", "#d1564b"]
    random.shuffle(markers_abiotic)
    random.shuffle(colors_abiotic)

    if label == 'biotic':
        markers = markers_biotic
        colors = colors_biotic
    else:
        markers = markers_abiotic
        colors = colors_abiotic

    if view == 'small':
        fig = plt.figure(figsize=(12, 12))
    else:
        fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')

    # Convert class to string if not already and adjust RT1_center for minutes and rounding
    peaks_features_df['class'] = peaks_features_df['class'].astype(str)
    peaks_features_df['RT1'] = (peaks_features_df['RT1'] / 60).astype(int)

    # Scatter plot for each sample with unique marker
    for i, sample in enumerate(selected_samples):
        sample_df = peaks_features_df[(peaks_features_df['sample'] == sample)]
        label = samples[samples['csv_file_name'] == sample]['label'].values[0]
        ax.scatter(
            sample_df['RT2'],
            sample_df['RT1'],
            sample_df['m/z'],
            c=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            s=sample_df['point_size'],
            alpha=1,
            edgecolors='w',
            label=sample
        )

    # Adding labels and title
    ax.set_xlabel('2nd Time (s)', labelpad=20)
    ax.set_ylabel('1st Time (min)', labelpad=20)
    ax.set_zlabel('m/z', labelpad=20)

    # Set tick font size
    ax.tick_params(axis='both', which='major', labelsize=20)

    if view == 'small':
        ax.set_box_aspect([2.5, 4, 2])
    else:
        ax.set_box_aspect([3, 4, 2])

    # Adjusting the view and plot appearance
    if view == 'small':
        ax.view_init(elev=20, azim=-120)
    else:
        ax.view_init(elev=20, azim=-179)
    plt.tight_layout()

    ax.set_xlim(peaks_features_df['RT2'].min(), 3.5)
    ax.set_ylim(peaks_features_df['RT1'].min()-3, 190)
    ax.set_zlim(peaks_features_df['m/z'].min(), 680)

    # Background color
    plt.gcf().set_facecolor('white')
    plt.gca().set_facecolor('white')

    # Change the color of the gridlines
    ax.xaxis._axinfo['grid'].update(color = 'gray', linestyle = '-' , linewidth=0.5)
    ax.yaxis._axinfo['grid'].update(color = 'gray', linestyle = '-', linewidth=0.5)
    ax.zaxis._axinfo['grid'].update(color = 'gray', linestyle = '-', linewidth=0.5)

    # Make the edges of the plot black
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # Add legend at the bottom of the plot
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 0), ncol=2)

    # Adjust layout to make room for the legend
    # Adjust layout to make room for the legend
    # plt.subplots_adjust(bottom=0.25,left=0.25,right=0.75,top=0.75)

    plt.savefig(os.path.join(result_dir, f'Peaks_3d_{label}.pdf'), format='pdf')
    plt.close()


def plot_distribution_of_peaks(peaks_features_df, result_dir):
    # Define the color palette
    palette = ['#3c5488', '#e64b35']

    # Define the number of bins and calculate the bin edges
    bin_count = 12
    bin_edges = np.linspace(30, 700, bin_count+1)

    # Create a figure and axis with specified size
    fig, ax = plt.subplots(figsize=(3, 2))

    # Remove the top and right borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Create the histogram plot
    sns.histplot(data=peaks_features_df, x="m/z", hue="class", palette=palette, alpha=0.85, bins=bin_edges)

    # Set the x-axis label
    # ax.set_xlabel('m/z', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Set the y-axis label
    # ax.set_ylabel('# Peaks', fontsize=12)

    # Customizing x-axis tick labels to show bin ranges
    tick_labels = [f'{int(bin_edges[i])}-{int(bin_edges[i+1])}' for i in range(len(bin_edges)-1)]
    ax.set_xticks([(bin_edges[i] + bin_edges[i+1])/2 for i in range(len(bin_edges)-1)])
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=12)

    # Limit the x-axis and y-axis range
    ax.set_xlim(30, 700)
    ax.set_ylim(0, None)  # 'None' lets the upper limit be determined automatically

    # Set the background color
    fig.set_facecolor('white')
    ax.set_facecolor('white')

    # remove grid lines
    ax.grid(False)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_edgecolor('black')
    ax.spines['bottom'].set_edgecolor('black')

    # remove legend
    ax.get_legend().remove()

    # Save the pdf file
    plt.savefig(os.path.join(result_dir,'distribution_of_peaks.pdf'), format='pdf', bbox_inches='tight',dpi=400)

    # Show the plot
    plt.show()
