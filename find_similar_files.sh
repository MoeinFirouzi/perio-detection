#!/bin/bash

DIR1="./data"
DIR2="./data_stage_2"
DEST1="./dataset/tooth"
DEST2="./dataset/cej_bone"

UNIQUE1="unique_tooth.txt"
UNIQUE2="unique_cej_bone.txt"

# Ensure destination directories exist
mkdir -p "$DEST1"
mkdir -p "$DEST2"

# File extensions to check for images
EXTENSIONS=("jpg" "jpeg" "png")

# Temporary files for tracking unique filenames
> "$UNIQUE1"  # Clear or create file for unique files in DIR1
> "$UNIQUE2"  # Clear or create file for unique files in DIR2

# Loop through images in DIR1 and check for matches in DIR2
for ext in "${EXTENSIONS[@]}"; do
    for file1 in "$DIR1"/*."$ext"; do
        [ -e "$file1" ] || continue  # Skip if no files found
        filename=$(basename -- "$file1")

        # Check if the same image exists in DIR2
        file2="$DIR2/$filename"
        if [ -f "$file2" ]; then
            # Copy matching images to respective destinations
            cp "$file1" "$DEST1/"
            cp "$file2" "$DEST2/"

            # Copy corresponding JSON files if they exist
            json1="${file1%.*}.json"
            json2="${file2%.*}.json"

            if [ -f "$json1" ]; then
                cp "$json1" "$DEST1/"
            fi
            if [ -f "$json2" ]; then
                cp "$json2" "$DEST2/"
            fi

            echo "Copied: $filename and corresponding JSON files."
        else
            echo "$filename" >> "$UNIQUE1"  # Save unique file from DIR1
        fi
    done
done

# Now check for unique files in DIR2
for ext in "${EXTENSIONS[@]}"; do
    for file2 in "$DIR2"/*."$ext"; do
        [ -e "$file2" ] || continue  # Skip if no files found
        filename=$(basename -- "$file2")

        file1="$DIR1/$filename"
        if [ ! -f "$file1" ]; then
            echo "$filename" >> "$UNIQUE2"  # Save unique file from DIR2
        fi
    done
done

echo "File copying completed!"
echo "Unique files list saved to $UNIQUE1 and $UNIQUE2"

