# Define URLs for the files to be downloaded
trainZipUrl="http://images.cocodataset.org/zips/train2017.zip"
valZipUrl="http://images.cocodataset.org/zips/val2017.zip"
annotationsZipUrl="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Define the output folder paths
dataFolderPath="./data"
trainZipPath="$dataFolderPath/train2017.zip"
valZipPath="$dataFolderPath/val2017.zip"
annotationsZipPath="$dataFolderPath/annotations_trainval2017.zip"

# Create the data folder if it doesn't exist
if [ ! -d "$dataFolderPath" ]; then
    mkdir -p "$dataFolderPath"
fi

# Function to download a file with progress
download_file() {
    url=$1
    destination=$2

    echo "Downloading $url..."

    # Use curl with progress bar to download the file
    curl -L -o "$destination" "$url"
}

# Download the ZIP files
download_file "$trainZipUrl" "$trainZipPath"
download_file "$valZipUrl" "$valZipPath"
download_file "$annotationsZipUrl" "$annotationsZipPath"

# Extract the ZIP files
echo "Extracting train2017.zip..."
unzip "$trainZipPath" -d "$dataFolderPath"

echo "Extracting val2017.zip..."
unzip "$valZipPath" -d "$dataFolderPath"

echo "Extracting annotations_trainval2017.zip..."
unzip "$annotationsZipPath" -d "$dataFolderPath"

# Clean up ZIP files
echo "Cleaning up ZIP files..."
rm "$trainZipPath"
rm "$valZipPath"
rm "$annotationsZipPath"

echo "Download and extraction completed!"
