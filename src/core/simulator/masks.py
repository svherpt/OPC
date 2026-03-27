import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os


def get_random_dataset_mask(dir_path="example_masks", **kwargs):
    """Reads a random mask from the specified directory and returns it as a numpy array."""
    all_files = [f for f in os.listdir(os.path.join('./data/', dir_path)) if f.endswith('.png')]
    random_file = np.random.choice(all_files)

    return read_mask_from_img(os.path.join(dir_path, random_file), **kwargs)


def get_dataset_masks(dir_path="example_masks", num_masks=5, **kwargs):
    """Reads a specified number of random masks from the given directory and returns them as a list of numpy arrays."""
    all_files = [f for f in os.listdir(os.path.join('./data/', dir_path)) if f.endswith('.png')]
    #Sample without replacement
    selected_files = np.random.choice(all_files, size=num_masks, replace=False)

    return [read_mask_from_img(os.path.join(dir_path, file_path), **kwargs) for file_path in selected_files]


def read_mask_from_img(file_path, **kwargs):
    """Reads a mask from an image file, converts it to binary, and resizes it to the desired grid size."""
    mask_size = kwargs.get("mask_grid_size", 512)
    mask = plt.imread(os.path.join('./data/', file_path))

    mask = (mask > 0.5).astype(np.float64)

    #Since we only have grayscale masks, take only one channel if needed
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]


    #Resize to desired size
    if mask.shape[0] != mask_size or mask.shape[1] != mask_size:
        zoom_factors = (mask_size / mask.shape[0], mask_size / mask.shape[1])
        mask = zoom(mask, zoom_factors, order=0)

    return mask


def visualise_mask(mask):
    """Visualizes the given mask using a grayscale colormap."""
    plt.imshow(mask, cmap='gray')
    plt.title("Generated Mask")
    plt.show()


if __name__ == "__main__":
    test_mask = read_mask_from_img("example_masks/1.glp.png", mask_grid_size=256)
    visualise_mask(test_mask)