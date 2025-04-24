#!/bin/bash
# Define the phase folder name (adjust this as needed)
phase="HEXPhase"

# Define the full paths to your CSV files (adjust these as needed)
nearest_csv="/home/luisbarajas/experiments/nearest_vs_predicted/exp_nearest_L.csv"
predicted_csv="/home/luisbarajas/experiments/nearest_vs_predicted/exp_predicted_L.csv"

# Base directory (assumed to be the current directory: Tests)
base_dir=$(pwd)

# Loop over chiN directories in the Tests folder
for chiN_dir in "$base_dir"/chiN*; do
  # Ensure it's a directory
  [ -d "$chiN_dir" ] || continue
  
  # Extract the numeric part of the chiN directory name
  chiN_val=${chiN_dir##*/}         # e.g., "chiN25.045267"
  chiN_num=${chiN_val#chiN}         # removes "chiN" -> "25.045267"
  
  # Loop over fA directories inside each chiN directory
  for fA_dir in "$chiN_dir"/fA*; do
    [ -d "$fA_dir" ] || continue
    
    # Extract the numeric part of the fA directory name
    fA_val=${fA_dir##*/}            # e.g., "fA0.524390"
    fA_num=${fA_val#fA}             # removes "fA" -> "0.524390"
    
    # Loop over tau directories inside each fA directory
    for tau_dir in "$fA_dir"/tau*; do
      [ -d "$tau_dir" ] || continue
      
      # Extract the numeric part of the tau directory name
      tau_val=${tau_dir##*/}         # e.g., "tau0.918919"
      tau_num=${tau_val#tau}          # removes "tau" -> "0.918919"
      
      # Define the full path to the phase subfolder within the tau directory
      phase_dir="$tau_dir/$phase"
      if [ ! -d "$phase_dir" ]; then
        echo "Phase folder $phase not found in $tau_dir"
        continue
      fi
      
      # Construct the key by combining the numeric values with underscores
      key="${chiN_num}_${fA_num}_${tau_num}"
      
      # Retrieve the nearest L value from the nearest CSV
      nearest_L=$(grep "^${key}," "$nearest_csv" | awk -F, '{print $2}')
      if [ -z "$nearest_L" ]; then
        echo "Nearest L value not found for key $key"
        continue
      fi
      
      # Retrieve the predicted L value from the predicted CSV
      predicted_L=$(grep "^${key}," "$predicted_csv" | awk -F, '{print $2}')
      if [ -z "$predicted_L" ]; then
        echo "Predicted L value not found for key $key"
        continue
      fi
      
      # Change into the phase directory where HEX.in is located
      cd "$phase_dir" || continue
      
      # === First Simulation (using nearest L value) ===
      # Print the nearest L value before simulation
      echo "* L value for $key: $nearest_L"
      
      # Replace the cellscaling value in HEX.in with the nearest L value
      sed -i.bak "s/^\s*cellscaling = .*/      cellscaling = $nearest_L/" HEX.in
      
      # Run the simulation with HEX.in updated with the nearest L value
      /home/luisbarajas/PolyFTS/bin/Release/PolyFTS.x HEX.in > HEX_nearest_L.out
      
      # === Second Simulation (using predicted L value) ===
      # Print the predicted L value before simulation
      echo "* L value for $key: $predicted_L"
      
      # Update HEX.in to use the predicted L value
      sed -i.bak "s/^\s*cellscaling = .*/      cellscaling = $predicted_L/" HEX.in
      
      # Run the simulation with HEX.in updated with the predicted L value
      /home/luisbarajas/PolyFTS/bin/Release/PolyFTS.x HEX.in > HEX_predicted_L.out
      
      # Return to the base directory for the next iteration
      cd "$base_dir" || exit
    done
  done
done
