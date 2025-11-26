#!/bin/bash
# Download TALENT benchmark datasets from Google Drive
# Google Drive folder: https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z

set -e

echo "=========================================="
echo "TALENT Datasets Download Script"
echo "=========================================="

# Create dataset directory
DATASET_DIR="${TALENT_DATA_PATH:-/workspace/datasets/TALENT}"
mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

echo "Downloading to: $DATASET_DIR"

# Check if gdown is installed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Download benchmark_dataset.zip from Google Drive
# The folder ID is: 1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z
echo ""
echo "Downloading TALENT benchmark datasets..."
echo "(This may take a while depending on your internet speed)"
echo ""

# Try to download the benchmark_dataset.zip
# Note: If this fails due to large file size, try the alternative methods below
gdown --fuzzy "https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z" -O benchmark_dataset.zip --folder || {
    echo ""
    echo "WARNING: Automatic download failed. This can happen with large Google Drive folders."
    echo ""
    echo "Please manually download the datasets from:"
    echo "  https://drive.google.com/drive/folders/1j1zt3zQIo8dO6vkO-K-WE6pSrl71bf0z"
    echo ""
    echo "Then extract them to: $DATASET_DIR"
    echo ""
    echo "Alternative: Download individual datasets you need:"
    echo "  - basic_benchmark (300 datasets)"
    echo "  - large_benchmark (22 datasets)"
    echo ""
    exit 1
}

# Extract if zip files were downloaded
if ls *.zip 1> /dev/null 2>&1; then
    echo "Extracting downloaded files..."
    for zipfile in *.zip; do
        echo "Extracting $zipfile..."
        unzip -o "$zipfile" -d .
    done
    echo "Cleaning up zip files..."
    rm -f *.zip
fi

# List downloaded datasets
echo ""
echo "Downloaded datasets:"
ls -la "$DATASET_DIR"

# Count datasets
if [ -d "$DATASET_DIR/basic_benchmark" ]; then
    BASIC_COUNT=$(ls -d "$DATASET_DIR/basic_benchmark"/*/ 2>/dev/null | wc -l)
    echo "Basic benchmark: $BASIC_COUNT datasets"
fi

if [ -d "$DATASET_DIR/large_benchmark" ]; then
    LARGE_COUNT=$(ls -d "$DATASET_DIR/large_benchmark"/*/ 2>/dev/null | wc -l)
    echo "Large benchmark: $LARGE_COUNT datasets"
fi

echo ""
echo "=========================================="
echo "TALENT datasets downloaded successfully!"
echo "=========================================="
echo ""
echo "Dataset path: $DATASET_DIR"
echo "Set environment variable:"
echo "  export TALENT_DATA_PATH=$DATASET_DIR"
echo ""
