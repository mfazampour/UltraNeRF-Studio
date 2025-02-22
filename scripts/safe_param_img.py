import numpy as np
from PIL import Image
import os

# Load the numpy volume (3D array)
volume = np.load('./logs/spine_phantom_rec03/output_maps_spine_phantom_segmentation_010000_0/params/reconstruction.npy')

# Create a directory to store the images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Iterate over the slices (assuming the volume is of shape [Z, Y, X])
for i, slice_ in enumerate(volume):
    print(slice_.shape)
    # Convert the slice to an image
    slice_[slice_>=0.5] = 1.
    slice_[slice_ != 1.] = 0.
    img = Image.fromarray((slice_* 255.).astype(np.uint8).transpose(1, 0))

    # If needed, you can apply normalization or colormap here

    # Save the image
    img.save(os.path.join(output_dir, f'{i:03d}.png'))

print(f"Images saved to {output_dir}")
