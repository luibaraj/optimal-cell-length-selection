import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def main():
    # User-defined variables
    good_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/good_hex_formuations.csv"
    phase_file = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/data_with_L/data_v2/data_all_DLHGBAS.txt"
    output_file = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/exp_nearest_L.csv"
    phase_value = "HEX"  # Change this value to the desired phase
    fmt = "{:.6f}"      # Format for keys

    # -------------------------------
    # STEP 1: Load Good Formulations Data
    # -------------------------------
    df_good = pd.read_csv(good_file)
    print(f"Loaded {len(df_good)} good formulations.")

    # -------------------------------
    # STEP 2: Create Formulation Keys for Good Formulations
    # -------------------------------
    df_good["key"] = (
        df_good["chiN"].apply(lambda x: fmt.format(x)) + "_" +
        df_good["fA"].apply(lambda x: fmt.format(x)) + "_" +
        df_good["tau"].apply(lambda x: fmt.format(x))
    )

    # -------------------------------
    # STEP 3: Fit a Scaler and Scale the Good Dataset
    # -------------------------------
    scaler = MinMaxScaler().fit(df_good[['chiN', 'fA', 'tau']])
    df_good_scaled = df_good.copy()
    df_good_scaled[['chiN', 'fA', 'tau']] = scaler.transform(df_good[['chiN', 'fA', 'tau']])

    # -------------------------------
    # STEP 4: Compute Nearest Neighbor Keys for Each Good Point
    # -------------------------------
    good_points = df_good_scaled[['chiN', 'fA', 'tau']].to_numpy()
    nearest_keys = []
    for i, pt in enumerate(good_points):
        distances = np.sqrt(((good_points - pt) ** 2).sum(axis=1))
        distances[i] = np.inf  # Ignore self-distance.
        min_idx = distances.argmin()
        nearest_keys.append(df_good_scaled.iloc[min_idx]["key"])

    # -------------------------------
    # STEP 5: Load Phase Good Data and Create Formulation Keys
    # -------------------------------
    df_phase = pd.read_csv(
        phase_file,
        delimiter="\t", 
        skiprows=1,
        names=["chiN", "fA", "tau", "phase", "H", "H1", "L", "Runtime", "Iterations", "STATUS"]
    )
    df_phase = df_phase[df_phase["phase"] == phase_value]
    df_phase["key"] = (
        df_phase["chiN"].apply(lambda x: fmt.format(x)) + "_" +
        df_phase["fA"].apply(lambda x: fmt.format(x)) + "_" +
        df_phase["tau"].apply(lambda x: fmt.format(x))
    )

    # Build mapping from key to L value.
    key_to_L = dict(zip(df_phase["key"], df_phase["L"]))

    # -------------------------------
    # STEP 6: Build DataFrame of Keys and Map L Values
    # -------------------------------
    # Here we map the nearest neighbor key's L value back to the good formulation.
    df_good_scaled["nearest_key"] = nearest_keys
    df_good_scaled["L"] = df_good_scaled["nearest_key"].map(key_to_L)
    df_nearest = df_good_scaled[["key", "L"]]

    # Save the resulting DataFrame to CSV.
    df_nearest.to_csv(output_file, index=False)
    print(f"Nearest neighbor keys and phase L values saved to {output_file}")

if __name__ == "__main__":
    main()
