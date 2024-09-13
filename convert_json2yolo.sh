#!/bin/bash

# Check if labelme2yolo is installed
if ! python3 -m pip show labelme2yolo &> /dev/null; then
    echo "labelme2yolo is not installed. Installing now..."
    python3 -m pip install labelme2yolo
else
    echo "----------------------------"
    echo "| labelme2yolo is available."
    echo "----------------------------"
fi

labelme2yolo --json_dir ./data/ --val_size 0.15 --test_size 0.15 --output_format polygon