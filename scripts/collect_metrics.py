import os
import glob
import pandas as pd
import re

import os
import glob
import re
import pandas as pd

# -------------------------------
# Function to extract lines containing total runtime and iterations completed from slurm outputs
# -------------------------------

def extract_runtime_and_iters(directory, file_pattern='slurm*'):
    """
    Iterate through files in the given directory matching file_pattern and extract paired values:
      - 'iters': iteration count from a line like "SCFT Converged in 84200 time steps (total)"
      - 'Total_Runtime': runtime from a later line like "TOTAL Runtime: 143.05 sec"
    
    It pairs each iteration count with the next encountered runtime value.
    
    Returns a DataFrame with columns:
      - 'iters': extracted iteration count as float
      - 'Total_Runtime': extracted runtime as float
      - 'group': even-indexed rows are labeled "nearest_L", odd-indexed rows "predicted_L"
    
    Parameters:
    - directory (str): Path to the directory containing the files.
    - file_pattern (str): Glob pattern to match files (default: 'slurm*').
    """
    full_pattern = os.path.join(directory, file_pattern)
    
    # Compile regex patterns:
    iters_pattern = re.compile(r'^SCFT Converged in\s+(\d+)\s+time steps')
    runtime_pattern = re.compile(r'^TOTAL Runtime:\s*([\d\.]+)')
    
    matched_data = []
    
    # Iterate over each file matching the pattern
    for file_path in glob.glob(full_pattern):
        pending_iter = None  # temporary storage for iteration count
        with open(file_path, 'r') as file:
            for line in file:
                # Check for an iteration line
                iters_match = iters_pattern.match(line)
                if iters_match:
                    pending_iter = float(iters_match.group(1))
                    continue  # Continue to next line
                
                # Check for a runtime line
                runtime_match = runtime_pattern.match(line)
                if runtime_match and pending_iter is not None:
                    runtime_val = float(runtime_match.group(1))
                    matched_data.append({
                        'iters': pending_iter,
                        'Total_Runtime': runtime_val
                    })
                    # Clear pending_iter after pairing
                    pending_iter = None
    
    # Create a DataFrame from the collected data
    df = pd.DataFrame(matched_data)
    
    # Add the "group" column: even indices get "nearest_L", odd indices get "predicted_L"
    df['group'] = ['nearest_L' if i % 2 == 0 else 'predicted_L' for i in range(len(df))]
    
    return df



def main():
    # base dir containing the slurm files
    directory = '/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/experiment_rslts'

    # collecting experiment results for each simulation (iter, Total_Runtime, group - nearest_L, predicted_L)
    df = extract_runtime_and_iters(directory)
    print(df[:20])

    df.to_csv("/Users/luisbarajas/Documents/GitHub/ML-enabled-SCFT/scripts/results/experiment_rslts/metrics.csv")

if __name__ == "__main__":
    main()
