#!/bin/bash

# -------------------------------
# User-defined variables (adjust as needed)
# -------------------------------
BASE_DIR="/path/to/base_directory"         # Directory containing batch folders
PHASE="HEXPhase"                           # Phase folder name (e.g., HEXPhase)
PREDICTED_CSV="/home/luisbarajas/experiments/nearest_vs_predicted/predicted_L.csv"     # CSV with predicted L values (format: key,L_pred)
NEAREST_CSV="/home/luisbarajas/experiments/nearest_vs_predicted/nearest_L.csv"         # CSV with nearest L values (format: key,L)
NEAREST_RSLTS="/home/luisbarajas/experiments/nearest_vs_predicted/nearest_L_rslts.csv"          # Output CSV for first simulation L values
PREDICTED_RSLTS="/home/luisbarajas/experiments/nearest_vs_predicted/predicted_L_rslts.csv"      # Output CSV for simulation L values after updating
POLYFTS="/home/luisbarajas/PolyFTS/bin/Release/PolyFTS.x"  # Path to PolyFTS.x executable

# -------------------------------
# Initialize output CSVs
# -------------------------------
echo "key,L" > "$NEAREST_RSLTS"
echo "key,L" > "$PREDICTED_RSLTS"

# -------------------------------
# Main Loop: Iterate through chiN directories (found under batch*/Tests/)
# -------------------------------
for chiN_dir in "$BASE_DIR"/batch*/Tests/chiN*; do
    if [ -d "$chiN_dir" ]; then
        # Extract chiN value (remove "chiN" prefix)
        chiN=$(basename "$chiN_dir" | sed 's/^chiN//')
        for fA_dir in "$chiN_dir"/fA*; do
            if [ -d "$fA_dir" ]; then
                # Extract fA value (remove "fA" prefix)
                fA=$(basename "$fA_dir" | sed 's/^fA//')
                for tau_dir in "$fA_dir"/tau*; do
                    if [ -d "$tau_dir" ]; then
                        # Extract tau value (remove "tau" prefix)
                        tau=$(basename "$tau_dir" | sed 's/^tau//')
                        # Construct formulation key: "chiN_fA_tau" with six-decimal formatting
                        formulation_key=$(printf "%.6f_%.6f_%.6f" "$chiN" "$fA" "$tau")
                        
                        phase_dir="$tau_dir/$PHASE"
                        if [ -d "$phase_dir" ]; then
                            HEX_IN="$phase_dir/HEX.in"
                            # Check if HEX.in exists before processing
                            if [ -f "$HEX_IN" ]; then
                                
                                # --- First Simulation: Update HEX.in with Nearest Neighbor L ---
                                # Look up nearest L value from NEAREST_CSV using the formulation key
                                nearest_L=$(grep "^$formulation_key," "$NEAREST_CSV" | cut -d',' -f2)
                                if [ -n "$nearest_L" ]; then
                                    # Update the "cellscaling =" line in HEX.in with the nearest L value; create a backup
                                    if sed -i.bak "s/^cellscaling = .*/cellscaling = $nearest_L/" "$HEX_IN"; then
                                        echo "Updated HEX.in in $phase_dir with nearest L: $nearest_L for $formulation_key"
                                    else
                                        echo "Error updating HEX.in in $phase_dir for $formulation_key" >&2
                                    fi
                                else
                                    echo "No nearest L found for $formulation_key in $NEAREST_CSV" >&2
                                fi
                                
                                # Run the first simulation with the modified HEX.in
                                (cd "$phase_dir" && $POLYFTS HEX.in)
                                
                                # Parse the operators.dat file to extract the leftmost value from its last line
                                if [ -f "$phase_dir/operators.dat" ]; then
                                    L_result=$(tail -n 1 "$phase_dir/operators.dat" | awk '{print $1}')
                                    echo "$formulation_key,$L_result" >> "$NEAREST_RSLTS"
                                else
                                    echo "operators.dat not found in $phase_dir" >&2
                                fi
                                
                                # --- Second Simulation: Update HEX.in with Predicted L ---
                                # Look up the predicted L value for this formulation key from PREDICTED_CSV
                                predicted_L=$(grep "^$formulation_key," "$PREDICTED_CSV" | cut -d',' -f2)
                                if [ -n "$predicted_L" ]; then
                                    # Update the "cellscaling =" line in HEX.in with the predicted L value
                                    if sed -i "s/^cellscaling = .*/cellscaling = $predicted_L/" "$HEX_IN"; then
                                        echo "Updated HEX.in in $phase_dir with predicted L: $predicted_L for $formulation_key"
                                    else
                                        echo "Error updating HEX.in in $phase_dir for $formulation_key" >&2
                                    fi
                                    
                                    # Run the second simulation with the updated HEX.in
                                    (cd "$phase_dir" && $POLYFTS HEX.in)
                                    
                                    # Parse the new operators.dat file for the updated L result
                                    if [ -f "$phase_dir/operators.dat" ]; then
                                        new_L_result=$(tail -n 1 "$phase_dir/operators.dat" | awk '{print $1}')
                                        echo "$formulation_key,$new_L_result" >> "$PREDICTED_RSLTS"
                                    else
                                        echo "operators.dat not found in $phase_dir after update" >&2
                                    fi
                                else
                                    echo "No predicted L found for $formulation_key in $PREDICTED_CSV" >&2
                                fi
                            else
                                echo "HEX.in not found in $phase_dir for $formulation_key" >&2
                            fi
                        else
                            echo "Phase directory $phase_dir does not exist for $formulation_key" >&2
                        fi
                    fi
                done
            fi
        done
    fi
done
