import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os

def get_layered_lines_mask(mask_grid_size, mask_num_layers, mask_num_blocks, mask_border_size, **kwargs):
    layer_growth_factor = 1.5

    mask = np.zeros((mask_grid_size, mask_grid_size))

    layer_height = (mask_grid_size - 2 * mask_border_size) // mask_num_layers
    curr_num_blocks = mask_num_blocks

    # Generate a row with an increasing number of blocks
    for layer in range(mask_num_layers):
        y_start = int(mask_border_size * 1.5) + layer * layer_height
        y_end = y_start + layer_height - mask_border_size

        block_width = (mask_grid_size - 2 * mask_border_size) // ( curr_num_blocks)
        for block in range(curr_num_blocks):
            x_start = int(mask_border_size * 1.5) + block * block_width
            x_end = x_start + block_width - mask_border_size

            mask[y_start:y_end, x_start:x_end] = 1
        
        curr_num_blocks = int(curr_num_blocks * layer_growth_factor)

    return mask

def get_random_dataset_mask(dir_path="./data/ganopc-data/artitgt", **kwargs):
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
    mask_img = plt.imread(file_path)
    
    if mask_img.ndim == 3:
        mask = mask_img[:,:,0]
    else:
        mask = mask_img

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
    initial_mask = get_layered_lines_mask(512, 5, 5, 20)
    visualise_mask(initial_mask)