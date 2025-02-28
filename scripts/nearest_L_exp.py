import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# STEP 1: Load Good Formulations Data
# -------------------------------
def load_formulation_data(good_file):
    """
    Load the CSV file for good formulations.
    """
    df_good = pd.read_csv(good_file)
    return df_good

# -------------------------------
# STEP 2: Create Formulation Keys for Good Formulations
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
# STEP 3: Fit a Scaler Using the Good Formulations
# -------------------------------
def fit_scaler(df_good):
    """
    Fit a MinMaxScaler on the good formulation data (using only the chiN, fA, and tau columns).
    """
    scaler = MinMaxScaler().fit(df_good[['chiN', 'fA', 'tau']])
    return scaler

# -------------------------------
# STEP 4: Scale the Good Dataset
# -------------------------------
def scale_dataset(df_good, scaler):
    """
    Create a copy of the good DataFrame, then scale its 'chiN', 'fA', and 'tau' columns using the provided scaler.
    """
    df_good_scaled = df_good.copy()
    df_good_scaled[['chiN', 'fA', 'tau']] = scaler.transform(df_good[['chiN', 'fA', 'tau']])
    return df_good_scaled

# -------------------------------
# STEP 5: Compute Nearest Neighbor Keys for Each Good Point
# -------------------------------
def compute_nearest_keys_for_good(df_good_scaled):
    """
    For each good point (scaled), compute its Euclidean distance to all other good points.
    The self-distance is set to infinity to avoid selecting itself.
    Then, return a list of the nearest neighbor key for each good point.
    """
    good_points = df_good_scaled[['chiN', 'fA', 'tau']].to_numpy()
    nearest_keys = []
    
    for i, pt in enumerate(good_points):
        distances = np.sqrt(((good_points - pt) ** 2).sum(axis=1))
        distances[i] = np.inf  # ignore self by setting its distance to infinity
        min_idx = distances.argmin()
        nearest_keys.append(df_good_scaled.iloc[min_idx]["key"])
        
    return nearest_keys

# -------------------------------
# STEP 6: Load Phase Good Data and Create Formulation Keys
# -------------------------------
def load_phase_good_data(phase_file, phase_value, fmt="{:.6f}"):
    """
    Load good formulations (with converged L values) from the given file, filter by phase,
    and create a formulation key based on chiN, fA, and tau.
    
    Parameters:
      phase_file : str
          The path to the file containing the phase data.
      phase_value : str
          The phase value to filter on.
      fmt : str
          The format for the keys.
    """
    df_phase = pd.read_csv(
        phase_file,
        delimiter="\t", skiprows=1,
        names=["chiN", "fA", "tau", "phase", "H", "H1", "L", "Runtime", "Iterations", "STATUS"]
    )
    df_phase = df_phase[df_phase["phase"] == phase_value]
    df_phase["key"] = (
        df_phase["chiN"].apply(lambda x: fmt.format(x)) + "_" +
        df_phase["fA"].apply(lambda x: fmt.format(x)) + "_" +
        df_phase["tau"].apply(lambda x: fmt.format(x))
    )
    return df_phase

# -------------------------------
# STEP 7: Build DataFrame of Nearest Neighbor Keys and Map L Values
# -------------------------------
def build_nearest_df_with_L(df_good_scaled, nearest_keys, key_to_L):
    """
    Build a DataFrame that contains each good formulation's key, its nearest neighbor key,
    and the nearest neighbor's L value (looked up from phase good data).
    """
    df = df_good_scaled.copy()
    df["nearest_key"] = nearest_keys
    df["L"] = df["nearest_key"].map(key_to_L)
    return df[["key", "nearest_key", "L"]]

# -------------------------------
# STEP 8: Main Execution Function
# -------------------------------
def main():
    # File paths and phase selection (phase is now modular)
    good_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/good_hex_formuations.csv"
    phase_file = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/nearest_L_exp.csv"
    phase_value = "HEX"  # Change this value to the desired phase
    
    # STEP 1: Load good formulations.
    df_good = load_formulation_data(good_file)
    print(f"Loaded {len(df_good)} good formulations.")
    
    # STEP 2: Create unique formulation keys for good formulations.
    df_good = create_good_keys(df_good)
    
    # STEP 3: Fit the scaler on the good formulations.
    scaler = fit_scaler(df_good)
    
    # STEP 4: Scale the good dataset.
    df_good_scaled = scale_dataset(df_good, scaler)
    
    # STEP 5: Compute nearest neighbor keys for the good formulations.
    nearest_keys = compute_nearest_keys_for_good(df_good_scaled)
    
    # STEP 6: Load phase good data (with converged L values) and create keys.
    df_phase_good = load_phase_good_data(phase_file, phase_value)
    # Build a mapping: formulation key -> L value
    key_to_L = dict(zip(df_phase_good["key"], df_phase_good["L"]))
    
    # STEP 7: Build a DataFrame with each formulation's key, its nearest neighbor key,
    # and the nearest neighbor's L value (mapped from phase good data).
    df_nearest = build_nearest_df_with_L(df_good_scaled, nearest_keys, key_to_L)
    df_nearest = df_nearest[['key', 'L']]

    # Save the resulting DataFrame to CSV.
    df_nearest.to_csv(output_file, index=False)
    print(f"Nearest neighbor keys and phase L values saved to {output_file}")

if __name__ == "__main__":
    main()
