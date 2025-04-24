import os
import glob
import re
import pandas as pd

# 1. Iterate through file paths of the form "*/Test/chiN*/fA*/tau*/HEXPhase"
# Use glob with recursive search.
pattern = "/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/resimulate_batch_rslts/resimulate_batches/batch0*/Tests/chiN*/fA*/tau*/HEXPhase"
hexphase_dirs = glob.glob(pattern)

# 2. Function to extract chiN, fA, and tau values from the directory path as strings.
def extract_parameters_from_path(path):
    # Split the path into its components.
    parts = path.split(os.sep)
    chiN_val = fA_val = tau_val = None
    for part in parts:
        if part.startswith("chiN"):
            chiN_val = part.replace("chiN", "")
        elif part.startswith("fA"):
            fA_val = part.replace("fA", "")
        elif part.startswith("tau"):
            tau_val = part.replace("tau", "")
    return chiN_val, fA_val, tau_val

# 3. Function to extract the iteration count from a file using the regex pattern.
# The regex pattern: '^SCFT Converged in\s+(\d+)\s+time steps'
def extract_iters_from_file(file_path, regex_pattern):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            match = regex_pattern.match(line)
            if match:
                return int(match.group(1))
    # Return None if no match was found.
    return None

# Compile the regex pattern once.
iters_regex = re.compile(r'^SCFT Converged in\s+(\d+)\s+time steps')

# List to collect results for each HEXPhase folder.
results = []

# 4. Loop over each HEXPhase directory
print(len(hexphase_dirs))
for hex_dir in hexphase_dirs:
    if os.path.isdir(hex_dir):
        # 2. Extract chiN, fA, tau values (stored as strings)
        chiN_val, fA_val, tau_val = extract_parameters_from_path(hex_dir)
        if not all([chiN_val, fA_val, tau_val]):
            continue  # Skip if any parameter is missing
        
        # 3. Define the paths to the two output files.
        predicted_file = os.path.join(hex_dir, "HEX_predicted_L.out")
        nearest_file   = os.path.join(hex_dir, "HEX_nearest_L.out")
        
        # Check that both files exist.
        if os.path.exists(predicted_file) and os.path.exists(nearest_file):
            # Extract iteration counts from each file.
            iters_predicted = extract_iters_from_file(predicted_file, iters_regex)
            iters_nearest   = extract_iters_from_file(nearest_file, iters_regex)
            
            if iters_predicted is not None and iters_nearest is not None:
                # 4. Take the difference in iterations from both.
                iters_diff = iters_predicted - iters_nearest
                
                # 5. Store the results in the list.
                results.append({
                    "chiN": chiN_val,
                    "fA": fA_val,
                    "tau": tau_val,
                    "iters_diff": iters_diff
                })

# Create a DataFrame with columns: chiN, fA, tau, and iters_diff.
df = pd.DataFrame(results)
output_csv = '/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/experiment_rslts.csv'
df.to_csv(output_csv, index=False)

