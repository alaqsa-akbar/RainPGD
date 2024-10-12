# Define URLs for the files to be downloaded
$trainZipUrl = "http://images.cocodataset.org/zips/train2017.zip"
$valZipUrl = "http://images.cocodataset.org/zips/val2017.zip"
$annotationsZipUrl = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

# Define the output folder paths
$dataFolderPath = "./data"
$trainZipPath = "$dataFolderPath/train2017.zip"
$valZipPath = "$dataFolderPath/val2017.zip"
$annotationsZipPath = "$dataFolderPath/annotations_trainval2017.zip"

# Create the data folder if it doesn't exist
if (-Not (Test-Path $dataFolderPath)) {
    New-Item -ItemType Directory -Path $dataFolderPath
}

# Function to download a file with progress
function Get-File {
    param (
        [string]$url,
        [string]$destination
    )

    # Get the file size from the server
    $webRequest = Invoke-WebRequest -Uri $url -Method Head
    $totalSize = [int64]$webRequest.Headers["Content-Length"]

    # Start downloading the file with a progress bar
    $response = Invoke-WebRequest -Uri $url -OutFile $destination -UseBasicParsing -PassThru

    $downloadedBytes = 0
    $bufferSize = 4KB

    Write-Host "Downloading $url..."

    [System.IO.FileStream]$fileStream = [System.IO.File]::OpenRead($destination)

    while ($downloadedBytes -lt $totalSize) {
        $downloadedBytes += $bufferSize
        $progressPercent = ($downloadedBytes / $totalSize) * 100

        Write-Progress -Activity "Downloading $url" `
            -Status "$([math]::Round($progressPercent, 2))% complete" `
            -PercentComplete $progressPercent

        Start-Sleep -Milliseconds 100 # Simulate download progress
    }

    $fileStream.Close()
}

# Download the ZIP files with progress bar
Get-File $trainZipUrl $trainZipPath
Get-File $valZipUrl $valZipPath
Get-File $annotationsZipUrl $annotationsZipPath

# Extract the ZIP files
Write-Host "Extracting train2017.zip..."
Expand-Archive -Path $trainZipPath -DestinationPath $dataFolderPath

Write-Host "Extracting val2017.zip..."
Expand-Archive -Path $valZipPath -DestinationPath $dataFolderPath

Write-Host "Extracting annotations_trainval2017.zip..."
Expand-Archive -Path $annotationsZipPath -DestinationPath $dataFolderPath

# Clean up ZIP files
Write-Host "Cleaning up ZIP files..."
Remove-Item $trainZipPath
Remove-Item $valZipPath
Remove-Item $annotationsZipPath

Write-Host "Download and extraction completed!"
