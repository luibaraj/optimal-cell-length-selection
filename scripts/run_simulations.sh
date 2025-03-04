#!/bin/bash
# This script must be run from the base directory.

# Get the base directory (assumed to be the current working directory)
base_dir=$(pwd)

# Define the source location of simulate.sh.
# Assuming simulate.sh is located in the same directory as this script.
simulate_script_source="$(dirname "$0")/simulate.sh"

# Loop over each batchXXX directory within the base directory
for batch_dir in "$base_dir"/batch*; do
  # Check that it is a directory
  [ -d "$batch_dir" ] || continue
  
  # Define the Tests directory path within this batch directory
  tests_dir="$batch_dir/Tests"
  
  # If the Tests directory does not exist, print a message and move to the next batch directory
  if [ ! -d "$tests_dir" ]; then
    echo "Tests directory not found in $batch_dir"
    continue
  fi
  
  echo "Processing Tests directory: $tests_dir"
  
  # Copy simulate.sh to the Tests directory
  cp "$simulate_script_source" "$tests_dir" || { echo "Failed to copy simulate.sh to $tests_dir"; continue; }
  
  # Change into the Tests directory
  cd "$tests_dir" || continue
  
  # Submit the simulation job using sbatch
  sbatch simulate.sh
  
  # Return to the base directory before processing the next batch directory
  cd "$base_dir" || exit
done
