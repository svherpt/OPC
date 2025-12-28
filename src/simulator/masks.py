import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
import os

def get_layered_lines_mask(mask_grid_size, mask_num_layers, mask_num_blocks, mask_border_size, **kwargs):
    """
    Docstring for get_layered_lines_mask

    :param mask_grid_size: int, size of the square mask
    :param mask_num_layers: int, number of rows in the mask
    :param mask_num_blocks: int, number of blocks in the first layer
    :param mask_border_size: int, size of the border around the mask, also used as spacing between blocks
    :return: np.ndarray, generated mask
    """
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

def get_random_dataset_mask(**kwargs):
    file_path = "./data/ganopc-data/artitgt/"
    all_files = [f for f in os.listdir(file_path) if f.endswith('.glp.png')]
    random_file = np.random.choice(all_files)

    mask_id = random_file.split('.')[0]

    return read_mask_from_img(mask_id, **kwargs)

def get_dataset_masks(num_masks, **kwargs):
    file_path = "./data/ganopc-data/artitgt/"
    all_files = [f for f in os.listdir(file_path) if f.endswith('.glp.png')]

    #Sample without replacement
    selected_files = np.random.choice(all_files, size=num_masks, replace=False)
    selected_ids = [f.split('.')[0] for f in selected_files]

    return [read_mask_from_img(mask_id, **kwargs) for mask_id in selected_ids]

def read_mask_from_img(mask_id, **kwargs):
    mask_size = kwargs.get("mask_grid_size", 512)
    
    file_path = f"./data/ganopc-data/artitgt/{mask_id}.glp.png"
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