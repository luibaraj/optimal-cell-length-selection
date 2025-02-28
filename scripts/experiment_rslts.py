import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# Combine CSV Files (if needed)
# -------------------------------
# Set the directory containing the CSV files; change as needed
csv_dir = '/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/experiment_rslts'
os.chdir(csv_dir)

# Combine all predicted CSV files (if not already combined)
predicted_files = sorted(glob.glob("predicted_L_rslts*.csv"))
predicted_df = pd.concat([pd.read_csv(f) for f in predicted_files], ignore_index=True)
# predicted_df.to_csv("predicted_L_rslts_combined.csv", index=False)
# print("Combined predicted DataFrame:")
# print(predicted_df.head())

# Combine all nearest CSV files (if not already combined)
nearest_files = sorted(glob.glob("nearest_L_rslts*.csv"))
nearest_df = pd.concat([pd.read_csv(f) for f in nearest_files], ignore_index=True)
# nearest_df.to_csv("nearest_L_rslexts_combined.csv", index=False)
# print("Combined nearest DataFrame:")
# print(nearest_df.head())

# -------------------------------
# Merge DataFrames and Compute Runtime Difference
# -------------------------------
# Merge on 'key' and 'L'
merged_df = pd.merge(predicted_df, nearest_df, on=['key', 'L'], suffixes=('_pred', '_near'))
# Compute runtime difference: (predicted runtime - nearest runtime)
merged_df['runtime_diff'] = merged_df['runtime_pred'] - merged_df['runtime_near']
print(merged_df)

# print("Merged DataFrame with runtime differences:")
# print(merged_df[['key', 'L', 'runtime_pred', 'runtime_near', 'runtime_diff']].head())

# -------------------------------
# Extract Formulation Parameters from Key
# -------------------------------
# The key is in the format: "chiN_fA_tau"
params = merged_df['key'].str.split('_', expand=True)
merged_df['chiN'] = params[0].astype(float)
merged_df['fA'] = params[1].astype(float)
merged_df['tau'] = params[2].astype(float)

# -------------------------------
# Visualization Function for Runtime Difference
# -------------------------------
def visualize_runtime_diff(df, title):
    # Creating 3D Scatter Plot using formulation parameters with runtime difference as color
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter plot: x=chiN, y=fA, z=tau, color by runtime difference
    sc = ax.scatter(df['chiN'], df['fA'], df['tau'],
                    c=df['runtime_diff'], cmap='viridis', marker='o', alpha=0.8)
    
    # Labels and Title
    ax.set_xlabel('chiN')
    ax.set_ylabel('fA')
    ax.set_zlabel('tau')
    ax.set_title(title)
    
    # Adding color bar for runtime difference
    cbar = plt.colorbar(sc, ax=ax, shrink=0.6)
    cbar.set_label('Runtime Difference (sec)')
    
    plt.show()

# # Visualize the runtime differences
# visualize_runtime_diff(merged_df, "Runtime Difference (Predicted - Nearest)")
