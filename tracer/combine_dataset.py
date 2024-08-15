import pandas as pd
path_to_peaks = '/usr/scratch/chromalyzer/test/peaks/peaks_lambda1_0.4/lam2_2.0/'
m_z_range = range(30, 701, 1)

labels = pd.read_csv('/usr/scratch/danial_stuff/Chromalyzer/data/labels.csv')
# Load the dataset
combined = []
for m_z in m_z_range:
    print(f"Processing m/z {m_z}")
    peaks = pd.read_csv(f"{path_to_peaks}{m_z}.csv")
    # add m_z column
    peaks['m_z'] = m_z
    merged_df = pd.merge(peaks, labels[['csv_file_name', 'sample_name', 'label']], on='csv_file_name', how='left')
    combined.append(merged_df)

combined_df = pd.concat(combined, axis=0)
combined_df.to_csv('/usr/scratch/chromalyzer/test/peaks/combined.csv', index=False)
