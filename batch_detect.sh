#!/bin/bash

# Directory containing images
INPUT_DIR="assets/Moh. Hatta"

# Model parameters
MODEL="convnext_tiny"
WEIGHTS="weights/convnext_tiny_noLM_final.pth"

# Loop through every .jpg file in the directory
for image_path in "$INPUT_DIR"/*.jpg; do
    if [ -f "$image_path" ]; then
        # Extract only the filename (for display/log)
        filename=$(basename "$image_path")
        
        echo "Processing file: $filename"
        echo "Full path: $image_path"

        # Run detection with the full path
        python detect.py \
            -n "$MODEL" \
            -w "$WEIGHTS" \
            --image-path "$image_path" \
            --save-image \
            --write-annotation
    fi
done
