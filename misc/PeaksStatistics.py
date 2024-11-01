import pandas as pd

path_to_peaks = '/usr/scratch/chromalyzer/peaks/combined.csv'

peaks = pd.read_csv(path_to_peaks)

print(peaks['sample_name'].value_counts())