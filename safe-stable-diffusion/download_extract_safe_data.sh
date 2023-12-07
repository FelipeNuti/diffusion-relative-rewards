#!/bin/bash

# URL of the tar file
tar_url="https://thor.robots.ox.ac.uk/reward-diffusion/safe_stable_diffusion_data.tar"

# Directory to extract the tar file
extraction_path="."

# Create the target directory if it doesn't exist
mkdir -p "$extraction_path"

# Define the path to the tar file
tar_file="$extraction_path/safe_stable_diffusion_data.tar"

# Check if the tar file already exists
if [ -e "$tar_file" ]; then
    echo "Tar file already exists. Skipping download."
else
    # Download the tar file
    echo "Downloading $tar_url..."
    curl -L -o "$tar_file" "$tar_url"
fi

# Extract the tar file
echo "Extracting to $extraction_path..."
tar -xf "$tar_file" -C "$extraction_path"

# List files inside the extracted directory
echo "Listing files inside $extraction_path:"
ls -l "$extraction_path"

echo "Extraction complete."
