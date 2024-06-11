import pandas as pd
import os

# Create a heatmap from a dataframe
def heatmap(df : pd.DataFrame, first_time_column_name : str = '1st Time (s)', second_time_column_name : 
            str = '2nd Time (s)', area_column_name : str = 'Area') -> pd.DataFrame:
    # Create heatmap data and then transpose it
    heatmap_data = df.pivot_table(values=area_column_name, index=first_time_column_name, columns=second_time_column_name, aggfunc='sum').T
    heatmap_data.fillna(0, inplace=True)
    # Reverse the order of rows to rotate 90 degrees counter-clockwise
    heatmap_data = heatmap_data.iloc[::-1]
    return heatmap_data

# Filter the dataframe by M/Z
def filter_by_m_z(df : pd.DataFrame,M_Z : float, threshold : float = 0.3, m_z_column_name : str = 'M/Z') -> pd.DataFrame:
    new_df = df[(df[m_z_column_name] > M_Z - threshold) & (df[m_z_column_name] < M_Z + threshold)].copy()
    return new_df

# Create a folder if it does not exist
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder created here: {folder_path}")
        os.makedirs(folder_path)