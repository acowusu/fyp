#!/bin/bash

# Check if a directory was provided as an argument
if [ -z "$1" ]; then
  echo "Error: Please specify a target directory."
  echo "Usage: $0 <target_directory>"
  exit 1
fi

target_dir="$1"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Check for existence of urls.txt
if [ ! -f "urls.txt" ]; then
  echo "Error: urls.txt not found."
  exit 1
fi

# Iterate through each URL in the file
while read -r url; do
  filename=$(basename "$url")

  # Download to the target directory
  echo "Downloading $filename to $target_dir..."
  curl -s -o "$target_dir/$filename" "$url"

  # Check if download was successful
  if [ $? -eq 0 ]; then
    echo "Extracting $filename in $target_dir..."
    unzip -q -d "$target_dir" "$target_dir/$filename"  # Extract to the target directory
    rm "$target_dir/$filename"                        # Optionally remove the zip file
  else
    echo "Download failed for $filename"
  fi
done < "urls.txt"

echo "All downloads and extractions completed."