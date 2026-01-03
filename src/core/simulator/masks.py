import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os


def get_random_dataset_mask(dir_path = "./data/ganopc-data/artitgt", **kwargs):
    all_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
    random_file = np.random.choice(all_files)

    return read_mask_from_img(dir_path + "/" + random_file, **kwargs)


def get_dataset_masks(dir_path="./data/ganopc-data/artitgt", num_masks=5, **kwargs):
    all_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]

    #Sample without replacement
    selected_files = np.random.choice(all_files, size=num_masks, replace=False)

    return [read_mask_from_img(dir_path + "/" + file_path, **kwargs) for file_path in selected_files]


def read_mask_from_img(file_path, **kwargs):
    mask_size = kwargs.get("mask_grid_size", 512)
    mask = plt.imread(file_path)

    mask = (mask > 0.5).astype(np.float64)

    #Resize to desired size
    if mask.shape[0] != mask_size or mask.shape[1] != mask_size:
        zoom_factors = (mask_size / mask.shape[0], mask_size / mask.shape[1])
        mask = zoom(mask, zoom_factors, order=0)

    return mask


def visualise_mask(mask):
    plt.imshow(mask)
    plt.title("Generated Mask")
    plt.show()

if __name__ == "__main__":
    test_mask = read_mask_from_img("./data/ganopc-data/artitgt/1.glp.png", mask_grid_size=256)
    visualise_mask(test_mask)
    test_mask = read_mask_from_img("./augmented_small/train/inputs/000000.png", mask_grid_size=256)
    visualise_mask(test_mask)