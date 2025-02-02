#!/bin/bash

# Define source directories (both images & labels are stored together)
TOOTH_SRC_DIR="./dataset/tooth"
CEJ_BONE_SRC_DIR="./dataset/cej_bone"

# Define destination directories
TOOTH_IMG_DIR="./dataset/tooth/images"
CEJ_BONE_IMG_DIR="./dataset/cej_bone/images"

TOOTH_LBL_DIR="./dataset/tooth/labels"
CEJ_BONE_LBL_DIR="./dataset/cej_bone/labels"

# Define train, val, test subdirectories
for split in train val test; do
    mkdir -p "$TOOTH_IMG_DIR/$split" "$CEJ_BONE_IMG_DIR/$split"
    mkdir -p "$TOOTH_LBL_DIR/$split" "$CEJ_BONE_LBL_DIR/$split"
done

# File extensions to check for images
EXTENSIONS=("jpg" "jpeg" "png")

# Get list of image filenames (without extensions)
shopt -s nullglob  # Avoid errors if no files found
FILENAMES=()

for ext in "${EXTENSIONS[@]}"; do
    for file in "$TOOTH_SRC_DIR"/*."$ext"; do
        [ -e "$file" ] || continue  # Skip if no files found
        FILENAMES+=("$(basename "$file")")
    done
done

# Shuffle filenames randomly
shuffled=($(shuf -e "${FILENAMES[@]}"))

# Compute split sizes
TOTAL=${#shuffled[@]}
TRAIN_SIZE=$((TOTAL * 70 / 100))
VAL_SIZE=$((TOTAL * 15 / 100))
TEST_SIZE=$((TOTAL - TRAIN_SIZE - VAL_SIZE))

# Split dataset
TRAIN_FILES=("${shuffled[@]:0:$TRAIN_SIZE}")
VAL_FILES=("${shuffled[@]:$TRAIN_SIZE:$VAL_SIZE}")
TEST_FILES=("${shuffled[@]:$((TRAIN_SIZE + VAL_SIZE))}")

# Function to move images and labels
move_files() {
    local img_dest_tooth="$1"
    local img_dest_cej="$2"
    local lbl_dest_tooth="$3"
    local lbl_dest_cej="$4"
    shift 4

    for filename in "$@"; do
        if [ -f "$TOOTH_SRC_DIR/$filename" ]; then
            mv "$TOOTH_SRC_DIR/$filename" "$img_dest_tooth/"
        fi
        if [ -f "$CEJ_BONE_SRC_DIR/$filename" ]; then
            mv "$CEJ_BONE_SRC_DIR/$filename" "$img_dest_cej/"
        fi

        # Move JSON files if they exist
        json_file="${filename%.*}.json"
        if [ -f "$TOOTH_SRC_DIR/$json_file" ]; then
            mv "$TOOTH_SRC_DIR/$json_file" "$lbl_dest_tooth/"
        fi
        if [ -f "$CEJ_BONE_SRC_DIR/$json_file" ]; then
            mv "$CEJ_BONE_SRC_DIR/$json_file" "$lbl_dest_cej/"
        fi
    done
}

# Move files to their respective directories
move_files "$TOOTH_IMG_DIR/train" "$CEJ_BONE_IMG_DIR/train" "$TOOTH_LBL_DIR/train" "$CEJ_BONE_LBL_DIR/train" "${TRAIN_FILES[@]}"
move_files "$TOOTH_IMG_DIR/val" "$CEJ_BONE_IMG_DIR/val" "$TOOTH_LBL_DIR/val" "$CEJ_BONE_LBL_DIR/val" "${VAL_FILES[@]}"
move_files "$TOOTH_IMG_DIR/test" "$CEJ_BONE_IMG_DIR/test" "$TOOTH_LBL_DIR/test" "$CEJ_BONE_LBL_DIR/test" "${TEST_FILES[@]}"

echo "Dataset successfully split into Train (70%), Validation (15%), and Test (15%)!"

