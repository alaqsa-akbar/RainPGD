from RainPGD.utils import add_rain
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="Generating Adversarial Dataset")
parser.add_argument("-f", "--folder", type=str, default='./data', help='folder containing dataset images')
parser.add_argument("-s", "--split", type=str, default='train', help='dataset split')
parser.add_argument("-o", "--output", type=str, default='./RainyCOCO', help='output directory')
args = parser.parse_args()

# Define the input and output directories
input_dir = f'{args.folder}/{args.split}2017'
output_base_dir = args.output 
name = args.split

# Create output directories if they don't exist
rain_types = ['weak', 'heavy', 'torrential']
for rain_type in rain_types:
    output_dir = os.path.join(output_base_dir, name, rain_type)
    os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in tqdm(os.listdir(input_dir)):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(input_dir, filename)
        
        # Load the image
        image = Image.open(image_path)

        # Convert the image to an np array (shape: height, width, channels)
        image_np = np.array(image)

        # Apply rain for each rain type and save the result
        for rain_type in rain_types:
            # Apply rain effect
            processed_image_np = add_rain(image_np, rain_type)
            
            # Convert the np array back to an image
            processed_image = Image.fromarray(processed_image_np)

            # Save the processed image to the corresponding directory
            output_path = os.path.join(output_base_dir, name, rain_type, filename)
            processed_image.save(output_path)
