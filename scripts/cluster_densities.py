import os
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# -------------------------------
# STEP 1: Collect Density Data from Directory Structure
# -------------------------------
def collect_density_data(base_directory, phase):
    """
    Traverse the directory structure under base_directory to collect density profiles.
    Returns a DataFrame with chiN, fA, tau, and density_profile.
    """
    data_list = []
    for batch in os.listdir(base_directory):
        batch_path = os.path.join(base_directory, batch)
        if os.path.isdir(batch_path) and "batch" in batch.lower():
            test_path = os.path.join(batch_path, "Tests")
            # Process each chiN folder
            for chiN in os.listdir(test_path):
                chiN_path = os.path.join(test_path, chiN)
                if not os.path.isdir(chiN_path):
                    continue
                try:
                    chiN_val = float(chiN.replace("chiN", ""))
                except ValueError:
                    continue
                # Process each fA folder inside chiN
                for fA in os.listdir(chiN_path):
                    fA_path = os.path.join(chiN_path, fA)
                    if not os.path.isdir(fA_path):
                        continue
                    try:
                        fA_val = float(fA.replace("fA", ""))
                    except ValueError:
                        continue
                    # Process each tau folder inside fA
                    for tau in os.listdir(fA_path):
                        if not tau.startswith("tau"):
                            continue
                        tau_path = os.path.join(fA_path, tau)
                        try:
                            tau_val = float(tau.replace("tau", ""))
                        except ValueError:
                            continue
                        
                        # Build the path for the specified phase and density file
                        phase_path = os.path.join(tau_path, phase)
                        density_file = os.path.join(phase_path, "density.dat")
                        if os.path.exists(density_file):
                            with open(density_file, "r") as f:
                                # Read non-header, non-empty lines
                                lines = [line for line in f if not line.lstrip().startswith("#") and line.strip()]
                            # Extract the third column values as floats (if available)
                            profile = [float(line.split()[2]) for line in lines if len(line.split()) >= 3]
                            density_vector = np.array(profile)
                            data_list.append({
                                "chiN": chiN_val,
                                "fA": fA_val,
                                "tau": tau_val,
                                "density_profile": density_vector
                            })
    df = pd.DataFrame(data_list)
    return df

# -------------------------------
# STEP 2: Prepare Density Profile and Compute FFT
# -------------------------------
def prepare_density_profile(profile_list):
    """
    Reshape a flat density profile into a 32x32 image.
    """
    profile_array = np.array(profile_list)
    image = profile_array.reshape((32, 32))
    return image

def compute_normalized_fft(image):
    """
    Compute the normalized 2D FFT of an image.
    """
    fft_image = np.fft.fft2(image)
    norm = np.linalg.norm(fft_image.flatten())
    return fft_image / norm if norm != 0 else fft_image

# -------------------------------
# STEP 3: Compute Dot Product Vector
# -------------------------------
def compute_dot_product_vector(sample_fft, rep_fft_vectors):
    """
    Compute absolute dot product vector between a sample FFT (flattened)
    and each reference FFT vector.
    """
    sample_vec = sample_fft.flatten()
    dp_vector = np.array([np.dot(sample_vec, rep_vec) for rep_vec in rep_fft_vectors.values()])
    return abs(dp_vector)

# -------------------------------
# STEP 4: Process Density Data and Compute FFTs
# -------------------------------
def process_density_data(df, rep_formulations_str):
    """
    Process the density DataFrame:
    - Convert density profiles to images.
    - Compute normalized FFT for each image.
    - Build reference FFT vectors from rep_formulations_str.
    - Compute dot product vectors for each sample.
    """
    df2 = df.copy()
    
    # Create density image and normalized FFT columns
    df2["density_image"] = df2["density_profile"].apply(prepare_density_profile)
    df2["normalized_fft"] = df2["density_image"].apply(compute_normalized_fft)
    
    # Create formatted string representations for matching
    chiN_str = df2["chiN"].apply(lambda x: f"{x:.6f}")
    fA_str   = df2["fA"].apply(lambda x: f"{x:.6f}")
    tau_str  = df2["tau"].apply(lambda x: f"{x:.6f}")
    
    # Build reference FFT vectors from rep_formulations_str dictionary
    rep_fft_vectors = {}
    for rep_name, rep_vals in rep_formulations_str.items():
        rep_chiN = rep_vals["chiN"]
        rep_fA   = rep_vals["fA"]
        rep_tau  = rep_vals["tau"]
        
        rep_rows = df2[
            (chiN_str == rep_chiN) &
            (fA_str   == rep_fA) &
            (tau_str  == rep_tau)
        ]
        # Use the first matching row's FFT vector as reference (if available)
        if not rep_rows.empty:
            rep_fft = rep_rows["normalized_fft"].iloc[0]
            rep_fft_vectors[rep_name] = rep_fft.flatten()
    
    # Compute dot product vectors for each sample and add as a column
    df2["dot_product_vector"] = df2["normalized_fft"].apply(
        lambda fft: compute_dot_product_vector(fft, rep_fft_vectors)
    )
    
    return df2

# -------------------------------
# STEP 5: Perform KMeans Clustering on Dot Product Features
# -------------------------------
def perform_clustering(df, desired_clusters):
    """
    Perform KMeans clustering using the dot product vectors as features.
    """
    dot_product_features = np.vstack(df["dot_product_vector"].values)
    kmeans = KMeans(n_clusters=desired_clusters, random_state=42)
    df["dot_cluster"] = kmeans.fit_predict(dot_product_features)
    return df

# -------------------------------
# STEP 6: Plot Clustering Results
# -------------------------------
def plot_clusters(df):
    """
    Plot scatter plots of chiN vs. fA, chiN vs. tau, and fA vs. tau colored by cluster.
    """
    num_clusters = df["dot_cluster"].nunique()
    cmap = plt.get_cmap("viridis", num_clusters)
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    # Scatter Plot: chiN vs. fA
    sc1 = axs[0].scatter(df["chiN"], df["fA"], c=df["dot_cluster"], cmap=cmap, alpha=0.7)
    axs[0].set_xlabel("chiN")
    axs[0].set_ylabel("fA")
    axs[0].set_title("chiN vs. fA")
    fig.colorbar(sc1, ax=axs[0], ticks=range(num_clusters), label="Cluster")
    
    # Scatter Plot: chiN vs. tau
    sc2 = axs[1].scatter(df["chiN"], df["tau"], c=df["dot_cluster"], cmap=cmap, alpha=0.7)
    axs[1].set_xlabel("chiN")
    axs[1].set_ylabel("tau")
    axs[1].set_title("chiN vs. tau")
    fig.colorbar(sc2, ax=axs[1], ticks=range(num_clusters), label="Cluster")
    
    # Scatter Plot: fA vs. tau
    sc3 = axs[2].scatter(df["fA"], df["tau"], c=df["dot_cluster"], cmap=cmap, alpha=0.7)
    axs[2].set_xlabel("fA")
    axs[2].set_ylabel("tau")
    axs[2].set_title("fA vs. tau")
    fig.colorbar(sc3, ax=axs[2], ticks=range(num_clusters), label="Cluster")
    
    plt.tight_layout()
    plt.show()

# -------------------------------
# STEP 7: Copy Screenshot Files into Cluster Directories
# -------------------------------
def copy_screenshots(df, source_dir, destination_dir):
    """
    For each row in the DataFrame, build the expected screenshot filename,
    and copy it from the source directory to a cluster-specific subdirectory
    under destination_dir.
    """
    for index, row in df.iterrows():
        chiN = row['chiN']
        fA = row['fA']
        tau = row['tau']
        cluster = row['dot_cluster']
        
        file_name = f"chiN{chiN:.6f}_fA{fA:.6f}_tau{tau:.6f}.png"
        source_file = os.path.join(source_dir, file_name)
        
        cluster_dir = os.path.join(destination_dir, f"cluster{cluster}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        destination_file = os.path.join(cluster_dir, file_name)
        if os.path.exists(source_file):
            shutil.copy(source_file, destination_file)
            print(f"Copied {source_file} to {destination_file}")
        else:
            print(f"Warning: File {source_file} does not exist!")

# -------------------------------
# STEP 8: Main Execution Function
# -------------------------------
def main():
    # HEXPhase - Representative formulations for each potential class of density.
    rep_formulations = {
        "correct": {"chiN": "34.924158", "fA": "0.461538", "tau": "0.568966"},
        "hollow": {"chiN": "34.983082", "fA": "0.604545", "tau": "0.563380"},
        "bleeding": {"chiN": "34.887595", "fA": "0.705607", "tau": "0.544304"},
        "double_period": {"chiN": "34.623025", "fA": "0.691176", "tau": "0.607595"},
        "disordered": {"chiN": "15.036952", "fA": "0.138060", "tau": "0.695652"}
    }

    # Base directory containing batch clusters.
    base_directory = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/densities/DLH_32npw"

    # Phase directory name
    phase = "HEXPhase"

    # Number of clusters for KMeans clustering.
    desired_clusters = 5

    # Directory containing density images
    source_dir = "/Users/luisbarajas/Desktop/Projects/Research_Projects/Poly/data/densities/screenshots_HEX"

    # Directory for clustering density images together
    destination_dir = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/clusters"

    # Collect density data from the specified directory structure
    df = collect_density_data(base_directory, phase)
    
    # Process the density data: compute images, FFTs, and dot product vectors
    df_processed = process_density_data(df, rep_formulations)
    
    # Perform KMeans clustering using the computed dot product features
    df_clustered = perform_clustering(df_processed, desired_clusters)
    
    # Plot the clustering results
    plot_clusters(df_clustered)
    
    # Copy screenshot files into cluster directories based on clustering results
    copy_screenshots(df_clustered, source_dir, destination_dir)

    # Save df with dot_cluster labels
    target_save_directory = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results"
    os.makedirs(target_save_directory, exist_ok=True)
    output_csv = os.path.join(target_save_directory, "clustered_density_data.csv")
    df_clustered.to_csv(output_csv, index=False)
    print(f"Saved clustered dataframe to {output_csv}")
if __name__ == "__main__":
    main()
