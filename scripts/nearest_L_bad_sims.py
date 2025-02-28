import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# STEP 1: Load Good and Bad Formulations Data
# -------------------------------
def load_formulation_data(good_file, bad_file):
    """
    Load the CSV files for good and bad formulations.
    """
    df_good = pd.read_csv(good_file)
    df_bad = pd.read_csv(bad_file)
    return df_good, df_bad

# -------------------------------
# STEP 2: Filter Out Unwanted Bad Formulations
# -------------------------------
def filter_bad_formulations(df_bad, excluded_clusters):
    """
    Remove rows from the bad formulations DataFrame whose 'dot_cluster' value 
    is in the list of excluded clusters.
    """
    return df_bad[~df_bad['dot_cluster'].isin(excluded_clusters)]

# -------------------------------
# STEP 3: Combine Datasets and Fit a Scaler
# -------------------------------
def fit_scaler(df_good, df_bad):
    """
    Combine the good and bad formulation data (using only the chiN, fA, and tau columns)
    and fit a MinMaxScaler on the combined dataset.
    """
    all_points = pd.concat([df_good[['chiN', 'fA', 'tau']], df_bad[['chiN', 'fA', 'tau']]])
    scaler = MinMaxScaler().fit(all_points)
    return scaler

# -------------------------------
# STEP 4: Create Formulation Keys for Good Formulations
# -------------------------------
def create_good_keys(df_good, fmt="{:.6f}"):
    """
    Create a 'key' column for the good formulations in the format:
    chiN_fA_tau (with each value formatted to six decimals).
    """
    df_good["key"] = (
        df_good["chiN"].apply(lambda x: fmt.format(x)) + "_" +
        df_good["fA"].apply(lambda x: fmt.format(x)) + "_" +
        df_good["tau"].apply(lambda x: fmt.format(x))
    )
    return df_good

# -------------------------------
# STEP 5: Scale the Good and Bad Datasets
# -------------------------------
def scale_datasets(df_good, df_bad, scaler):
    """
    Create copies of the good and bad DataFrames, then scale their
    'chiN', 'fA', and 'tau' columns using the provided scaler.
    """
    df_good_scaled = df_good.copy()
    df_bad_scaled = df_bad.copy()
    
    df_good_scaled[['chiN', 'fA', 'tau']] = scaler.transform(df_good[['chiN', 'fA', 'tau']])
    df_bad_scaled[['chiN', 'fA', 'tau']] = scaler.transform(df_bad[['chiN', 'fA', 'tau']])
    
    return df_good_scaled, df_bad_scaled

# -------------------------------
# STEP 6: Convert Scaled Coordinates to NumPy Arrays
# -------------------------------
def get_scaled_points(df_good_scaled, df_bad_scaled):
    """
    Convert the scaled 'chiN', 'fA', and 'tau' columns to numpy arrays.
    """
    good_points = df_good_scaled[['chiN', 'fA', 'tau']].to_numpy()
    bad_points = df_bad_scaled[['chiN', 'fA', 'tau']].to_numpy()
    return good_points, bad_points

# -------------------------------
# STEP 7: Compute Nearest Neighbor Keys for Bad Points
# -------------------------------
def compute_nearest_keys(good_points, df_good_scaled, bad_points):
    """
    For each bad point (scaled), compute its Euclidean distance to all good points.
    Then, return a list of the nearest neighbor key from df_good_scaled.
    (Note: df_good_scaled['key'] still holds the original unscaled key.)
    """
    nearest_keys = []
    for bad_pt in bad_points:
        distances = np.sqrt(((good_points - bad_pt) ** 2).sum(axis=1))
        min_idx = distances.argmin()
        nearest_keys.append(df_good_scaled.iloc[min_idx]["key"])
    return nearest_keys

# -------------------------------
# STEP 8: Build DataFrame of Nearest Neighbor Keys
# -------------------------------
def build_nearest_df(nearest_keys):
    """
    Build a DataFrame that contains the nearest neighbor keys.
    """
    return pd.DataFrame({'nearest_key': nearest_keys})

# -------------------------------
# STEP 9: Load HEX Good Data and Create Formulation Keys
# -------------------------------
def load_hex_good_data(hex_file, phase="HEX", fmt="{:.6f}"):
    """
    Load HEX good formulations with L values from the given data file, filter by phase,
    and create a formulation key based on chiN, fA, and tau.
    """
    df_hex = pd.read_csv(
        hex_file,
        delimiter="\t", skiprows=1,
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
# STEP 10: Map HEX L Values to Bad Points Based on Nearest Neighbor Keys
# -------------------------------
def map_hex_L_values(df_hex_good, df_nearest, bad_points, scaler, fmt="{:.6f}"):
    """
    For each bad point, use its nearest neighbor key (from df_nearest) to look up the corresponding
    L value in the HEX good data. Before constructing the key for the bad point, inverse transform
    its scaled chiN, fA, and tau values to recover the original values.
    """
    # Create a DataFrame for bad points with scaled chiN, fA, tau values
    df_bad_points = pd.DataFrame(bad_points, columns=['chiN', 'fA', 'tau'])
    df_bad_points["nn_key"] = df_nearest["nearest_key"].values

    # Inverse transform the scaled bad points to get original values
    bad_points_orig = scaler.inverse_transform(df_bad_points[['chiN', 'fA', 'tau']])
    
    # Create a key for each bad point using the original unscaled values
    df_bad_points["key"] = [f"{row[0]:.6f}_{row[1]:.6f}_{row[2]:.6f}" for row in bad_points_orig]
    
    # Build a mapping from the HEX good data: key -> L value
    key_to_L = dict(zip(df_hex_good["key"], df_hex_good["L"]))
    # Map the nearest neighbor key (nn_key) from good data to the bad points to retrieve L values
    df_bad_points["L"] = df_bad_points["nn_key"].map(key_to_L)
    return df_bad_points

# -------------------------------
# STEP 11: Save the Nearest Neighbor L Values to CSV
# -------------------------------
def save_nearest_L(df_bad_points, output_file):
    """
    Save the DataFrame containing formulation keys and corresponding L values to CSV.
    """
    nearest_L = df_bad_points[["key", "L"]].copy()
    nearest_L.to_csv(output_file, index=False)
    print(f"Nearest L values saved to {output_file}")

# -------------------------------
# STEP 12: Main Execution Function
# -------------------------------
def main():
    # IMPORTANT PRE-REQ:
    # Split the clustered_density_data output into separate files for good and bad densities.
    # 'good_file' and 'bad_file' refer to these split files.
    # 'hex_file' should be a CSV containing the original formulations (chiN, fA, tau) and their converged L values.

    # create csv files for good and bad formulations (preferably not in the same scrpt as this one, e.g., when actually clustering)
    
    clustered_data = pd.read_csv("/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/clustered_density_data.csv")
    cluster1, cluster2 = clustered_data[clustered_data["dot_cluster"] == 1], clustered_data[clustered_data["dot_cluster"] == 2]
    good_formuations = pd.concat([cluster1, cluster2])
    good_formuations.to_csv("/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/good_hex_formuations.csv", index=False)

    cluster0, cluster3, cluster4 = clustered_data[clustered_data["dot_cluster"] == 0], clustered_data[clustered_data["dot_cluster"] == 3]\
                                    ,clustered_data[clustered_data["dot_cluster"] == 4]
    bad_hex_formuations = pd.concat([cluster0, cluster3, cluster4])
    bad_hex_formuations.to_csv("/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/bad_hex_formuations.csv", index=False)


    good_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/good_hex_formuations.csv"
    bad_file  = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/bad_hex_formuations.csv"
    L_file  = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/nearest_L_bad.csv"
    
    # Load good and bad formulation data from CSV files
    df_good, df_bad = load_formulation_data(good_file, bad_file)
    
    # Exclude unwanted clusters from the bad formulations (e.g., cluster 4)
    df_bad = filter_bad_formulations(df_bad, excluded_clusters=[4])
    
    # Fit a MinMaxScaler on the combined good and bad formulations (original, unscaled values)
    scaler = fit_scaler(df_good, df_bad)
    
    # Create a unique key for each good formulation based on its original values
    df_good = create_good_keys(df_good)
    
    # Scale both good and bad datasets using the fitted scaler
    df_good_scaled, df_bad_scaled = scale_datasets(df_good, df_bad, scaler)
    
    # Convert the scaled good and bad formulations to numpy arrays for further computation
    good_points, bad_points = get_scaled_points(df_good_scaled, df_bad_scaled)
    
    # Compute the nearest neighbor keys for each bad point by comparing to good points
    nearest_keys = compute_nearest_keys(good_points, df_good_scaled, bad_points)
    
    # Build a DataFrame of nearest neighbor keys
    df_nearest = build_nearest_df(nearest_keys)
    
    # Load HEX good data (with converged L values) and create keys from the original values
    df_hex_good = load_hex_good_data(L_file, phase="HEX")
    
    # Map the corresponding L values from HEX good data to the bad points using nearest neighbor keys.
    # Before mapping, the scaled bad points are inverse transformed to obtain original values.
    df_bad_points = map_hex_L_values(df_hex_good, df_nearest, bad_points, scaler)
    
    # Save the resulting nearest neighbor L values to a CSV file.
    save_nearest_L(df_bad_points, output_file)

if __name__ == "__main__":
    main()
