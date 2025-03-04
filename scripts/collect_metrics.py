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
      - From a line like "* L value for 20.035500_0.357143_0.818182: 1.6799592857", extract the key "20.035500_0.357143_0.818182" 
        and split it into three float columns: chiN, fA, and tau.
      - From a subsequent line "SCFT Converged in 84200 time steps (total)", extract the iteration count (iters).
      - From a later line "TOTAL Runtime: 143.05 sec", extract the runtime (Total_Runtime).
    
    The function assumes these lines occur in the file in the above relative order.
    
    Returns a DataFrame with columns:
      - 'chiN', 'fA', 'tau': floats extracted from the L value key.
      - 'iters': extracted iteration count as float.
      - 'Total_Runtime': extracted runtime as float.
      - 'group': even-indexed rows are labeled "nearest_L", odd-indexed rows "predicted_L".
    
    Parameters:
    - directory (str): Path to the directory containing the files.
    - file_pattern (str): Glob pattern to match files (default: 'slurm*').
    """
    full_pattern = os.path.join(directory, file_pattern)
    
    # Compile regex patterns:
    # Pattern to capture the L value line:
    lvalue_pattern = re.compile(r'^\S+\s+L value for\s+([\d\.]+)_([\d\.]+)_([\d\.]+):')
    
    # Pattern to capture the iteration count from "SCFT Converged in ... time steps"
    iters_pattern = re.compile(r'^SCFT Converged in\s+(\d+)\s+time steps')
    
    # Pattern to capture the runtime value from "TOTAL Runtime: ..."
    runtime_pattern = re.compile(r'^TOTAL Runtime:\s*([\d\.]+)')
    
    matched_data = []
    
    # Iterate over each file matching the pattern
    for file_path in glob.glob(full_pattern):
        # Initialize state variables
        state = "await_lvalue"  # states: await_lvalue, await_converged, await_runtime
        current_row = {}
        
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                if state == "await_lvalue":
                    # Look for the L value line.
                    l_match = lvalue_pattern.match(line)
                    if l_match:
                        # Extract and convert key parts into floats.
                        try:
                            current_row['chiN'] = float(l_match.group(1))
                            current_row['fA']   = float(l_match.group(2))
                            current_row['tau']  = float(l_match.group(3))
                        except ValueError:
                            # If conversion fails, skip this line.
                            continue
                        state = "await_converged"
                        continue  # Move to next line
                
                elif state == "await_converged":
                    # Look for the converged line.
                    iters_match = iters_pattern.match(line)
                    if iters_match:
                        current_row['iters'] = float(iters_match.group(1))
                        state = "await_runtime"
                        continue  # Move to next line
                
                elif state == "await_runtime":
                    # Look for the runtime line.
                    runtime_match = runtime_pattern.match(line)
                    if runtime_match:
                        current_row['Total_Runtime'] = float(runtime_match.group(1))
                        # Add the current row to the matched data.
                        matched_data.append(current_row)
                        # Reset state for the next group in the same file.
                        current_row = {}
                        state = "await_lvalue"
                        continue
    
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
