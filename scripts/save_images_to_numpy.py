import os
import numpy as np
from PIL import Image


def load_images_from_directory(directory):
    # List all files in the directory
    files = os.listdir(directory)

    # Filter out PNG files
    png_files = [f for f in files if f.endswith('.png') and "labels" in f]

    # Sort files based on the numeric value in the filename
    png_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x.split("_")[-1].replace("-labels", "")))))

    # Load images and convert to NumPy arrays
    images = []
    for file in png_files:
        print(file)
        img_path = os.path.join(directory, file)
        img = Image.open(img_path)  # Open the image
        img_array = np.flipud(np.array(img))  # Convert to NumPy array
        # img_array = np.array(img)
        images.append(img_array)

    # Stack images into a single NumPy array (assuming all images have the same shape)
    images_array = np.array(images)

    return images_array


# Usage
directory = ('./data/patient21/patient21_labels/')
images_np = load_images_from_directory(directory)
np.save('./data/processedpatient21/labels.npy', images_np)
# Now images_np is a NumPy array with the images sorted by the number in their filenames
print(images_np.shape)  # Print the shape of the resulting NumPy array
