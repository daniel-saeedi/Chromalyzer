import pandas as pd
from concurrent.futures import ProcessPoolExecutor

raw_path = '/usr/scratch/NASA/raw/'
samples = pd.read_csv('data/labels.csv')

min_index = []
max_index = []

min_spectrum = []
max_spectrum = []

min_TOF = []
max_TOF = []

min_mz = []
max_mz = []

min_area = []
max_area = []

min_resolutions = []
max_resolutions = []

def process_sample(row):
    raw_sample = pd.read_csv(f'{raw_path}{row["csv_file_name"]}')
    return {
        'min_spectrum': raw_sample['Spectrum'].min(),
        'max_spectrum': raw_sample['Spectrum'].max(),
        'min_TOF': raw_sample['TOF'].min(),
        'max_TOF': raw_sample['TOF'].max(),
        'min_mz': raw_sample['M/Z'].min(),
        'max_mz': raw_sample['M/Z'].max(),
        'min_area': raw_sample['Area'].min(),
        'max_area': raw_sample['Area'].max(),
        'min_resolutions': raw_sample['Resolution'].min(),
        'max_resolutions': raw_sample['Resolution'].max(),
        'min_index': raw_sample.index.min(),
        'max_index': raw_sample.index.max()
    }

    print(row['csv_file_name'], 'processed')

# Use ProcessPoolExecutor for parallel processing
with ProcessPoolExecutor() as executor:
    results = list(executor.map(process_sample, [row for _, row in samples.iterrows()]))

# Collect the results
for result in results:
    min_spectrum.append(result['min_spectrum'])
    max_spectrum.append(result['max_spectrum'])
    min_TOF.append(result['min_TOF'])
    max_TOF.append(result['max_TOF'])
    min_mz.append(result['min_mz'])
    max_mz.append(result['max_mz'])
    min_area.append(result['min_area'])
    max_area.append(result['max_area'])
    min_resolutions.append(result['min_resolutions'])
    max_resolutions.append(result['max_resolutions'])
    min_index.append(result['min_index'])
    max_index.append(result['max_index'])

print(f'Min Spectrum: {min_spectrum}')
print(f'Max Spectrum: {max_spectrum}')
print(f'Min TOF: {min_TOF}')
print(f'Max TOF: {max_TOF}')
print(f'Min M/Z: {min_mz}')
print(f'Max M/Z: {max_mz}')
print(f'Min Area: {min_area}')
print(f'Max Area: {max_area}')
print(f'Min Resolution: {min_resolutions}')
print(f'Max Resolution: {max_resolutions}')
print(f'Min Index: {min_index}')
print(f'Max Index: {max_index}')