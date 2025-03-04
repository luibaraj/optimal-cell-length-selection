#!/bin/bash

# Step 1: Set the base directory as the current directory
BASE_DIR=$(pwd)

# Step 2: Loop over each batch directory
for batch in "$BASE_DIR"/batch*; do
    if [ -d "$batch" ]; then
        echo "Processing batch directory: $batch"

        # Step 3: Define the Tests directory path
        TESTS_DIR="$batch/Tests"
        if [ -d "$TESTS_DIR" ]; then
            echo "Found Tests directory: $TESTS_DIR"

            # Step 4: Loop over slurm files in the Tests directory
            for file in "$TESTS_DIR"/slurm*; do
                # Check if the file exists
                if [ -f "$file" ]; then
                    cp "$file" "$BASE_DIR"
                    echo "Copied: $file to $BASE_DIR"
                fi
            done
        else
            echo "Tests directory not found in $batch"
        fi
    fi
done
