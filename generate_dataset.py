from attack.utils import add_rain
import os
from PIL import Image
import numpy as np
from tqdm import tqdm

# Define the input and output directories
input_dir = './SegmentAndComplete/data/train2017'
output_base_dir = './data'

# Create output directories if they don't exist
rain_types = ['weak', 'heavy', 'torrential']
for rain_type in rain_types:
    output_dir = os.path.join(output_base_dir, rain_type)
    os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        
        # Load the image
        image = Image.open(image_path)

        # Convert the image to an np array (shape: height, width, channels)
        image_np = np.array(image)

        # Rearrange the dimensions to (3, width, height) if needed
        # image_np = np.transpose(image_np, (2, 0, 1))  # (channels, height, width)

        # Apply rain for each rain type and save the result
        for rain_type in rain_types:
            # Apply rain effect
            processed_image_np = add_rain(image_np, rain_type)

            # Rearrange the dimensions back to (height, width, channels) for saving
            # processed_image_np = np.transpose(processed_image_np, (1, 2, 0))  # (height, width, channels)

            # Convert the np array back to an image
            processed_image = Image.fromarray(processed_image_np)

            # Save the processed image to the corresponding directory
            output_path = os.path.join(output_base_dir, rain_type, filename)
            processed_image.save(output_path)

        # print(f"Processed {filename}")
