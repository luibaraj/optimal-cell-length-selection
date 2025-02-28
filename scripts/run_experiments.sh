#!/bin/bash
# Set your base directory and the location of your simulation script template
BASE_DIR="/home/luisbarajas/experiments/nearest_vs_predicted/sample_dir"
TEMPLATE="/home/luisbarajas/experiments/nearest_vs_predicted/experiment.sh"  # This is the original simulation script

# Iterate through each batch*/Tests directory under BASE_DIR
for tests_dir in "$BASE_DIR"/batch*/Tests; do
    # Determine the batch directory and extract its name (e.g. batch1)
    batch_dir=$(dirname "$tests_dir")
    batch_name=$(basename "$batch_dir")  # e.g. batch1
    # Extract the numeric part of the batch name (remove "batch")
    batch_num=${batch_name#batch}
    
    # Destination for the new experiment script
    dest_script="$tests_dir/experiment${batch_num}.sh"
    
    # Copy the template into the destination
    cp "$TEMPLATE" "$dest_script"
    
    # --- Modify BASE_DIR ---
    # Replace the BASE_DIR variable so that it reflects the actual batch directory.
    # The template line: BASE_DIR="/path/to/base_directory"
    # will be replaced with, for example: BASE_DIR="/path/to/base_directory/batch1"
    sed -i "s|^BASE_DIR=\"[^\"]*\"|BASE_DIR=\"$batch_dir\"|" "$dest_script"
    
    # --- Modify CSV paths to include the batch number ---
    # For NEAREST_RSLTS, replace the .csv suffix with <batch_num>.csv
    sed -i "s|\(NEAREST_RSLTS=\"/home/luisbarajas/experiments/nearest_vs_predicted/nearest_L_rslts\)\.csv\"|\1${batch_num}.csv\"|" "$dest_script"
    # For PREDICTED_RSLTS, do the same
    sed -i "s|\(PREDICTED_RSLTS=\"/home/luisbarajas/experiments/nearest_vs_predicted/predicted_L_rslts\)\.csv\"|\1${batch_num}.csv\"|" "$dest_script"
    
    # Make the new experiment script executable
    chmod +x "$dest_script"
    
    # Submit the experiment script using sbatch
    sbatch "$dest_script"
done
