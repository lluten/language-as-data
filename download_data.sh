#!/bin/bash
# download_data.sh - Downloads and extracts multiple Leipzig Wortschatz corpora,
# keeping only the sentences.txt files and cleaning up afterward.

set -e

# Define data directory
DATA_DIR="data"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Define corpora: (name, url)
declare -A CORPORA=(
    ["slk_newscrawl_2016_1M"]="https://downloads.wortschatz-leipzig.de/corpora/slk_newscrawl_2016_1M.tar.gz"
    ["tur_news_2024_1M"]="https://downloads.wortschatz-leipzig.de/corpora/tur_news_2024_1M.tar.gz"
)

# Loop through and process each corpus
for CORPUS in "${!CORPORA[@]}"; do
    URL="${CORPORA[$CORPUS]}"
    ARCHIVE_NAME="${CORPUS}.tar.gz"
    FOLDER_NAME="${CORPUS}"

    echo "----------------------------------------"
    echo "Processing $CORPUS..."
    echo "----------------------------------------"

    # Download archive if not already present
    if [ ! -f "$ARCHIVE_NAME" ]; then
        echo "Downloading $CORPUS..."
        if command -v curl &> /dev/null; then
            curl -LO "$URL"
        elif command -v wget &> /dev/null; then
            wget "$URL"
        else
            echo "Error: neither curl nor wget is installed. Please install one of them and re-run the script."
            exit 1
        fi
    else
        echo "Archive $ARCHIVE_NAME already exists, skipping download."
    fi

    # Extract archive
    echo "Extracting $ARCHIVE_NAME..."
    tar -xvzf "$ARCHIVE_NAME"

    # Delete archive
    rm -f "$ARCHIVE_NAME"

    # Keep only the sentences file
    echo "Cleaning up not needed files..."
    find "$FOLDER_NAME" -type f ! -name "*-sentences.txt" -delete
done

echo "----------------------------------------"
echo "All corpora downloaded and prepared successfully!"
echo "Files available at:"
echo "  data/slk_newscrawl_2016_1M/slk_newscrawl_2016_1M-sentences.txt"
echo "  data/tur_news_2024_1M/tur_news_2024_1M-sentences.txt"
echo "----------------------------------------"
