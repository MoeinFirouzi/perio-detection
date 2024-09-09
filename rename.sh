#!/bin/bash

# Initialize a counter
counter=1

# Loop through all jpg and JPG files in the current directory
for file in *.jpg *.JPG; do
  # Check if there are any jpg or JPG files
  if [[ -e "$file" ]]; then
    # Construct the new file name with leading zeros and lowercase extension (e.g., 001.jpg)
    new_name=$(printf "%03d.jpg" "$counter")
    
    # Rename the file
    mv "$file" "$new_name"
    
    # Increment the counter
    counter=$((counter + 1))
  fi
done

