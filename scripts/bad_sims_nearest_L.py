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
# Load HEX Good Data and Create Formulation Keys
# -------------------------------
def load_hex_good_data(hex_file, phase="HEX", fmt="{:.6f}"):
    """
    Load HEX good formulations with L values from the given file,
    filter by phase, and create a formulation key based on chiN, fA, and tau.
    """
    df_hex = pd.read_csv(
        hex_file,
        delimiter="\t",
        skiprows=1,
        names=["chiN", "fA", "tau", "phase", "H", "H1", "L", "Runtime", "Iterations", "STATUS"]
    )
    df_hex = df_hex[df_hex["phase"] == phase]
    df_hex["key"] = (
        df_hex["chiN"].apply(lambda x: fmt.format(x)) + "_" +
        df_hex["fA"].apply(lambda x: fmt.format(x)) + "_" +
        df_hex["tau"].apply(lambda x: fmt.format(x))
    )
    return df_hex

# -------------------------------
# Map HEX L Values to Bad Points Based on Nearest Neighbor Keys
# -------------------------------
def map_hex_L_values(df_hex_good, df_nearest, bad_points, scaler, fmt="{:.6f}"):
    """
    For each bad point, use its nearest neighbor key (from df_nearest) to look up the corresponding
    L value in the HEX good data. Inverse transform the scaled bad point values to recover the original values.
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
    
    # Build a mapping from the HEX good data: key -> L value
    key_to_L = dict(zip(df_hex_good["key"], df_hex_good["L"]))
    # Map the nearest neighbor key from good data to the bad points to retrieve L values
    df_bad_points["L"] = df_bad_points["nn_key"].map(key_to_L)
    return df_bad_points

# -------------------------------
# Main Execution Function
# -------------------------------
def main():
    # User-defined file paths and parameters at the top of main
    good_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/good_hex_formuations.csv"
    bad_file  = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/bad_hex_formuations.csv"
    L_file    = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/bad_sims_nearest_L.csv"
    fmt = "{:.6f}"  # Formatting string for keys

    # -------------------------------
    # Load and Process Good and Bad Formulation Data
    # -------------------------------
    # Load CSV files
    df_good = pd.read_csv(good_file)
    df_bad = pd.read_csv(bad_file)
    
    # Filter out unwanted bad formulations (exclude cluster 4)
    df_bad = df_bad[~df_bad['dot_cluster'].isin([4])]
    
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
    
    # Convert scaled data to numpy arrays for computation
    good_points = df_good_scaled[['chiN', 'fA', 'tau']].to_numpy()
    bad_points  = df_bad_scaled[['chiN', 'fA', 'tau']].to_numpy()
    
    # Compute nearest neighbor keys for each bad point using the good points
    nearest_keys = compute_nearest_keys(good_points, df_good_scaled, bad_points)
    
    # Build DataFrame of nearest neighbor keys
    df_nearest = pd.DataFrame({'nearest_key': nearest_keys})
    
    # -------------------------------
    # Load HEX Data and Map L Values to Bad Formulations
    # -------------------------------
    df_hex_good = load_hex_good_data(L_file, phase="HEX", fmt=fmt)
    df_bad_points = map_hex_L_values(df_hex_good, df_nearest, bad_points, scaler, fmt=fmt)
    
    # Save the resulting keys and corresponding L values to CSV
    nearest_L = df_bad_points[["key", "L"]].copy()
    nearest_L.to_csv(output_file, index=False)
    print(f"Nearest L values saved to {output_file}")

if __name__ == "__main__":
    main()
