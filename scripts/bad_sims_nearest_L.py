import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# Compute Nearest Neighbor Keys for Bad Points
# -------------------------------
def compute_nearest_keys(good_points, df_good_scaled, bad_points):
    """
    For each bad point (scaled), compute its Euclidean distance to all good points.
    Return a list of the nearest neighbor key from df_good_scaled.
    """
    nearest_keys = []
    for bad_pt in bad_points:
        distances = np.sqrt(((good_points - bad_pt) ** 2).sum(axis=1))
        min_idx = distances.argmin()
        nearest_keys.append(df_good_scaled.iloc[min_idx]["key"])
    return nearest_keys

# -------------------------------
# Load Good Formulation Data for a Specified Phase and Create Formulation Keys
# -------------------------------
def load_phase_good_data(data_file, phase, fmt="{:.6f}"):
    """
    Load good formulations with L values from the given file,
    filter by the specified phase, and create a formulation key based on chiN, fA, and tau.
    """
    df_phase = pd.read_csv(
        data_file,
        delimiter="\t",
        skiprows=1,
        names=["chiN", "fA", "tau", "phase", "H", "H1", "L", "Runtime", "Iterations", "STATUS"]
    )
    df_phase = df_phase[df_phase["phase"] == phase]
    df_phase["key"] = (
        df_phase["chiN"].apply(lambda x: fmt.format(x)) + "_" +
        df_phase["fA"].apply(lambda x: fmt.format(x)) + "_" +
        df_phase["tau"].apply(lambda x: fmt.format(x))
    )
    return df_phase

# -------------------------------
# Map L Values to Bad Points Based on Nearest Neighbor Keys for a Specified Phase
# -------------------------------
def map_phase_L_values(df_phase_good, df_nearest, bad_points, scaler, fmt="{:.6f}"):
    """
    For each bad point, use its nearest neighbor key (from df_nearest) to look up the corresponding
    L value in the good formulation data filtered by phase. Inverse transform the scaled bad point values
    to recover the original values.
    """
    # Create a DataFrame for bad points with scaled chiN, fA, tau values
    df_bad_points = pd.DataFrame(bad_points, columns=['chiN', 'fA', 'tau'])
    df_bad_points["nn_key"] = df_nearest["nearest_key"].values

    # Inverse transform the scaled bad points to get original values
    bad_points_orig = scaler.inverse_transform(df_bad_points[['chiN', 'fA', 'tau']])
    
    # Create a key for each bad point using the original unscaled values
    df_bad_points["key"] = [
        f"{row[0]:.6f}_{row[1]:.6f}_{row[2]:.6f}" for row in bad_points_orig
    ]
    
    # Build a mapping from the good formulation data: key -> L value
    key_to_L = dict(zip(df_phase_good["key"], df_phase_good["L"]))
    # Map the nearest neighbor key from the good data to the bad points to retrieve L values
    df_bad_points["L"] = df_bad_points["nn_key"].map(key_to_L)
    return df_bad_points

# -------------------------------
# Main Execution Function
# -------------------------------
def main():
    # User-defined file paths and parameters
    data_file    = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/bad_sims_nearest_L.csv"
    cluster_df = pd.read_csv("/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/clustered_density_data.csv")
    exp_path = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/exp_nearest_L.csv"
    fmt = "{:.6f}"  # Formatting string for keys
    
    # Specify the target phase (change this value to work with a different phase)
    target_phase = "GYR"

    # -------------------------------
    # Load and Process Good and Bad Formulation Data
    # -------------------------------
    # Load CSV files based on cluster data
    df_good = cluster_df[cluster_df['dot_cluster'].isin([2, 4])]
    df_bad = cluster_df[cluster_df['dot_cluster'].isin([0, 1, 3])]

    # Fit a MinMaxScaler on the combined data (using columns chiN, fA, tau)
    all_points = pd.concat([df_good[['chiN', 'fA', 'tau']], df_bad[['chiN', 'fA', 'tau']]])
    scaler = MinMaxScaler().fit(all_points)
    
    # Create unique keys for good formulations based on original values
    df_good["key"] = (
        df_good["chiN"].apply(lambda x: fmt.format(x)) + "_" +
        df_good["fA"].apply(lambda x: fmt.format(x)) + "_" +
        df_good["tau"].apply(lambda x: fmt.format(x))
    )
    
    # Scale both good and bad datasets
    df_good_scaled = df_good.copy()
    df_bad_scaled = df_bad.copy()
    df_good_scaled[['chiN', 'fA', 'tau']] = scaler.transform(df_good[['chiN', 'fA', 'tau']])
    df_bad_scaled[['chiN', 'fA', 'tau']] = scaler.transform(df_bad[['chiN', 'fA', 'tau']])
    
    # Compute nearest neighbor keys for each bad point using the good points
    good_points = df_good_scaled[['chiN', 'fA', 'tau']].to_numpy()
    bad_points  = df_bad_scaled[['chiN', 'fA', 'tau']].to_numpy()
    nearest_keys = compute_nearest_keys(good_points, df_good_scaled, bad_points)
    
    # Build DataFrame of nearest neighbor keys
    df_nearest = pd.DataFrame({'nearest_key': nearest_keys})
    
    # -------------------------------
    # Load Phase-Specific Good Data and Map L Values to Bad Formulations
    # -------------------------------
    df_phase_good = load_phase_good_data(data_file, phase=target_phase, fmt=fmt)
    df_bad_points = map_phase_L_values(df_phase_good, df_nearest, bad_points, scaler, fmt=fmt)
    
    # Save the resulting keys and corresponding L values to CSV
    nearest_L = df_bad_points[["key", "L"]].copy()
    nearest_L.to_csv(output_file, index=False)
    print(f"Nearest L values saved to {output_file}")

    sampled_df = nearest_L.sample(n=100, random_state=42)
    sampled_df.to_csv(exp_path, index=False)
    print(f"Nearest L values for experimentation saved to {exp_path}")

if __name__ == "__main__":
    main()
