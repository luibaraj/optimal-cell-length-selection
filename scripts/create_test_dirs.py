import os
import pandas as pd
import shutil

# -------------------------------
# Function to create directory structure for each formulation in the target directory
# -------------------------------
def create_directories(df, base_directory, n_batches, phase):
    """
    For each formulation in df (with 'chiN', 'fA', and 'tau' columns),
    create directories in the target structure as follows:
    
      base_directory/
        └── batchXXX/         (batches split uniformly based on the DataFrame row index)
            └── Tests/
                └── chiNXXXXXX/   (e.g., chiN34.924158)
                    └── fAXXXXXX/ (e.g., fA0.461538)
                        └── tauXXXXXX/  (e.g., tau0.568966)
                            └── <phase>/
    """
    total = len(df)
    for index, row in df.iterrows():
        # Calculate batch index uniformly from the DataFrame row index
        batch_index = int((index * n_batches) / total) + 1
        batch_folder = f"batch{batch_index:03d}"
        
        # Extract and format the values to six decimals
        chiN_val = float(row['chiN'])
        fA_val = float(row['fA'])
        tau_val = float(row['tau'])
        
        chiN_folder = f"chiN{chiN_val:.6f}"
        fA_folder = f"fA{fA_val:.6f}"
        tau_folder = f"tau{tau_val:.6f}"
        
        # Construct the full target directory path
        dir_path = os.path.join(base_directory, batch_folder, "Tests", chiN_folder, fA_folder, tau_folder, phase)
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created target directory: {dir_path}")

# -------------------------------
# Function to search through source batches and copy phase contents
# -------------------------------
def copy_phase_contents(df, source_base, target_base, n_batches, phase):
    """
    For each formulation in df:
      - Determine the target directory based on uniform batch splitting.
      - Search all batch folders in the source directory to find the corresponding formulation folder.
      - If found, copy all contents from the source phase directory to the target phase directory.
    """
    total = len(df)
    for index, row in df.iterrows():
        # Determine target batch folder (uniformly split by index)
        batch_index = int((index * n_batches) / total) + 1
        target_batch_folder = f"batch{batch_index:03d}"
        
        # Extract and format the formulation values
        chiN_val = float(row['chiN'])
        fA_val = float(row['fA'])
        tau_val = float(row['tau'])
        
        chiN_folder = f"chiN{chiN_val:.6f}"
        fA_folder = f"fA{fA_val:.6f}"
        tau_folder = f"tau{tau_val:.6f}"
        
        # Construct the target phase directory path
        target_phase_dir = os.path.join(target_base, target_batch_folder, "Tests", chiN_folder, fA_folder, tau_folder, phase)
        
        # Search through all batch folders in the source directory to find the matching formulation
        source_phase_dir = None
        for batch_folder in os.listdir(source_base):
            # Only consider directories that contain "batch" (case-insensitive)
            if not os.path.isdir(os.path.join(source_base, batch_folder)):
                continue
            if "batch" not in batch_folder.lower():
                continue
            candidate = os.path.join(source_base, batch_folder, "Tests", chiN_folder, fA_folder, tau_folder, phase)
            if os.path.isdir(candidate):
                source_phase_dir = candidate
                break  # Stop searching if a matching folder is found
        
        # If the source phase directory was found, copy its contents
        if source_phase_dir:
            try:
                os.makedirs(target_phase_dir, exist_ok=True)
                # Copy each file and subdirectory from the source to the target
                for item in os.listdir(source_phase_dir):
                    s = os.path.join(source_phase_dir, item)
                    d = os.path.join(target_phase_dir, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d, dirs_exist_ok=True)
                    else:
                        shutil.copy2(s, d)
                print(f"Copied contents from {source_phase_dir} to {target_phase_dir}")
            except Exception as e:
                print(f"Error copying from {source_phase_dir} to {target_phase_dir}: {e}")
        else:
            print(f"Source formulation not found for: chiN={chiN_val:.6f}, fA={fA_val:.6f}, tau={tau_val:.6f}")

# -------------------------------
# Main module
# -------------------------------
def main():
    # User-defined variables
    TARGET_BASE_DIR = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/resimulate_batches"  # Target base directory path
    SOURCE_BASE_DIR = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/data/GYR_32npw"  # Source base directory path
    CSV_PATH = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/results/exp_predicted_L.csv"  # CSV with 'key' column (formulations)
    PHASE = "GYRPhase"  # Phase directory name (e.g., HEXPhase)
    N_BATCHES = 50       # Number of batches to split formulations into for the target structu
    
    # Load the CSV file containing the formulations
    df = pd.read_csv(CSV_PATH)
    # Split the 'key' column into separate columns and convert to floats
    df[['chiN', 'fA', 'tau']] = df['key'].str.split('_', expand=True)
    df['chiN'] = df['chiN'].astype(float)
    df['fA'] = df['fA'].astype(float)
    df['tau'] = df['tau'].astype(float)
    
    # (Optional) Limit the DataFrame for testing; remove for full processing
    # df = df[:5]
    
    # Create the target directory structure
    create_directories(df, TARGET_BASE_DIR, N_BATCHES, PHASE)
    
    # Copy the contents from the source directory to the target directory
    copy_phase_contents(df, SOURCE_BASE_DIR, TARGET_BASE_DIR, N_BATCHES, PHASE)

if __name__ == "__main__":
    main()
