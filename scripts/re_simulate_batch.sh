#!/bin/bash

# -------------------------------
# User-defined variables (adjust as needed)
# -------------------------------
BASE_DIR="path_containing_batches"
PHASE="HEXPhase"
NEAREST_L_CSV="path_to_nearest_L.csv"
CELLSCALING_PATTERN="cellscaling ="

# -------------------------------
# Load nearest_L CSV into an associative array (skip header)
# -------------------------------
declare -A nearestL
while IFS=, read -r key L_value; do
    # Skip header row
    if [[ "$key" == "key" ]]; then continue; fi
    nearestL["$key"]="$L_value"
done < "$NEAREST_L_CSV"

# -------------------------------
# Helper Function: Update HEX.in file
# -------------------------------
update_hex_file() {
    local hex_in_file="$1"
    local L_value="$2"
    local formulation_key="$3"

    # Try to update the file using sed, creating a backup (.bak)
    if sed -i.bak "s/^${CELLSCALING_PATTERN}.*/${CELLSCALING_PATTERN} ${L_value}/" "$hex_in_file"; then
        echo "Updated $hex_in_file with L value: $L_value for formulation: $formulation_key"
    else
        echo "Error: Failed to update $hex_in_file for formulation: $formulation_key" >&2
    fi
}

# -------------------------------
# Main Loop: Iterate through directories and update HEX.in files
# -------------------------------
for batch_dir in "$BASE_DIR"/*; do
    # Check for batch directories (case-insensitive match)
    if [[ -d "$batch_dir" && "$batch_dir" =~ [Bb]atch ]]; then
        TESTS_DIR="$batch_dir/Tests"
        if [[ -d "$TESTS_DIR" ]]; then
            # Iterate through chiN folders
            for chiN_dir in "$TESTS_DIR"/chiN*; do
                if [[ -d "$chiN_dir" ]]; then
                    chiN_folder=$(basename "$chiN_dir")
                    chiN_value=$(echo "$chiN_folder" | sed 's/^chiN//')
                    chiN_formatted=$(printf "%.6f" "$chiN_value")
                    
                    # Iterate through fA folders
                    for fA_dir in "$chiN_dir"/fA*; do
                        if [[ -d "$fA_dir" ]]; then
                            fA_folder=$(basename "$fA_dir")
                            fA_value=$(echo "$fA_folder" | sed 's/^fA//')
                            fA_formatted=$(printf "%.6f" "$fA_value")
                            
                            # Iterate through tau folders
                            for tau_dir in "$fA_dir"/tau*; do
                                if [[ -d "$tau_dir" ]]; then
                                    tau_folder=$(basename "$tau_dir")
                                    tau_value=$(echo "$tau_folder" | sed 's/^tau//')
                                    tau_formatted=$(printf "%.6f" "$tau_value")
                                    
                                    # Construct key in the format: chiN_fA_tau
                                    formulation_key="${chiN_formatted}_${fA_formatted}_${tau_formatted}"
                                    
                                    # Check if key exists in nearestL array
                                    if [[ -n "${nearestL[$formulation_key]}" ]]; then
                                        L_value="${nearestL[$formulation_key]}"
                                        phase_dir="$tau_dir/$PHASE"
                                        
                                        # Check if the phase directory exists
                                        if [[ -d "$phase_dir" ]]; then
                                            hex_in_file="$phase_dir/HEX.in"
                                            
                                            # Check if HEX.in exists; if so, try to update it
                                            if [[ -f "$hex_in_file" ]]; then
                                                update_hex_file "$hex_in_file" "$L_value" "$formulation_key"
                                                
                                                # Run PolyFTS.x with HEX.in after updating L value
                                                if (cd "$phase_dir" && /home/luisbarajas/PolyFTS/bin/Release/PolyFTS.x HEX.in); then
                                                    echo "Successfully executed PolyFTS.x for formulation: $formulation_key"
                                                else
                                                    echo "Error running PolyFTS.x in $phase_dir for formulation: $formulation_key" >&2
                                                fi
                                            else
                                                echo "Warning: HEX.in not found in $phase_dir for formulation: $formulation_key" >&2
                                            fi
                                        else
                                            echo "Warning: Phase directory $phase_dir does not exist for formulation: $formulation_key" >&2
                                        fi
                                    else
                                        echo "Warning: No L value found for formulation key: $formulation_key"
                                    fi
                                fi
                            done
                        fi
                    done
                fi
            done
        fi
    fi
done
