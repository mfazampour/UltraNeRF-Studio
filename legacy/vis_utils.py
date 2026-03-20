import io

import matplotlib.pyplot as plt
import numpy as np


def show_colorbar(image, cmap="rainbow"):
    figure = plt.figure(figsize=(5, 5))
    plt.imshow(image.numpy(), cmap=cmap)
    plt.colorbar()
    buf = io.BytesIO()
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format="png")
    plt.close(figure)
    return buf


def define_image_grid_3D_np(x_size, y_size):
    y = np.array(range(x_size))
    x = np.array(range(y_size))
    xv, yv = np.meshgrid(x, y, indexing="ij")
    image_grid_xy = np.vstack((xv.ravel(), yv.ravel()))
    z = np.zeros(image_grid_xy.shape[1])
    image_grid = np.vstack((image_grid_xy, z))
    return image_grid
