import numpy as np
from src.core.ml.models import LithographyUNet
import src.core.simulator.lithography_simulator as simulator
import src.visualizers.simulator.simulation_visualizer as simulation_visualizer
import src.core.simulator.masks as masks
import json
import torch
from PIL import Image
from src.core.ml.litho_mask_optimizer import SourceMaskOptimizer
from src.core.ml.inferer import Inferer
import src.visualizers.ml.optimizer_visualizer as optimization_visualizer
import src.core.simulator.light_sources as light_sources
import time
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import src.visualizers.ml.trainer_visualizer as trainer_visualizer


with open("sim_config.json", "r") as f:
    sim_config = json.load(f)


def main():
    #show_n_masks('./augmented_medium/train/inputs', 10)
    
    #test_defocus_results(sim_config, initial_mask)
    dir_path = 'augmented_small'
    dir_path = 'augmented_medium'
    dir_path = 'augmented_massive'
    
    # simulate()
    
    # random_masks = masks.get_dataset_masks('./data/ganopc-data/artitgt', 1, **sim_config)
    # test_ML_model(sim_config, random_masks)
    
    # show_augmentation()

    #train_model('./augmented_massive', target_type='resists')

    # test_ML_model()
    optimize_model_multihead()

    # visualize_dataloader_sample('./data/augmented_massive', split="train", batch_size=1, index_in_batch=0)

def simulate():
    litho_sim = simulator.LithographySimulator(sim_config)
    source_illumination = light_sources.create_quadrant_source(sim_config)  # shape: (Npupil, Npupil)

    random_mask = masks.read_mask_from_img('ganopc-data/artitgt/343.glp.png', **sim_config)

    # t0 = time.time()
    simResults = litho_sim.simulate(random_mask, source_illumination)
    # t1 = time.time()

    # print(f"Simulation time: {t1 - t0:.2f} seconds")
    simulation_visualizer.visualize_simulation_results(simResults, mask=random_mask, illumination=source_illumination, config=sim_config)  
    

def test_ML_model():
    model = LithographyUNet(base_ch=64)
    model.load_state_dict(torch.load('./models/best_model.pth', map_location='cpu'))
    
    num_samples = 4
    maskList = [masks.get_random_dataset_mask('ganopc-data/artitgt', **sim_config) for _ in range(num_samples)]
    illumList = [light_sources.read_random_illumination_quarter('augmented_massive/test/illuminations', **sim_config) for _ in range(num_samples)]

    results = zip(maskList, illumList)

    fig, axes = plt.subplots(num_samples, 6, figsize=(20, 3.5*num_samples))

    for idx, (mask, illum) in enumerate(results):
        pred_intensity, pred_resist = model.predict(mask, illum)
        #make illum to full quadrant
        full_illumination = light_sources.quadrant_to_full(illum)   

        litho_sim = simulator.LithographySimulator(sim_config)
        results = litho_sim.simulate(mask, full_illumination)
        target_intensity = results['wafer_intensity']
        target_resist = results['resist_profile']


        axes[idx, 0].imshow(mask, cmap='gray')
        axes[idx, 0].set_title("Input Mask")
        axes[idx, 0].axis('off')

        #bi-linear interpolation to same size of mask
        upsampled_illum = light_sources.upsample_illumination(full_illumination, target_size=mask.shape[0])
        axes[idx, 1].imshow(upsampled_illum, cmap='hot')
        axes[idx, 1].set_title("Input Illumination")
        axes[idx, 1].axis('off')

        axes[idx, 2].imshow(target_intensity, cmap='inferno')
        axes[idx, 2].set_title("Target Intensity")
        axes[idx, 2].axis('off')

        axes[idx, 3].imshow(pred_intensity, cmap='inferno')
        axes[idx, 3].set_title("Predicted Intensity")
        axes[idx, 3].axis('off')

        axes[idx, 4].imshow(target_resist, cmap='gray')
        axes[idx, 4].set_title("Target Resist")
        axes[idx, 4].axis('off')

        axes[idx, 5].imshow(pred_resist, cmap='gray')
        axes[idx, 5].set_title("Predicted Resist")
        axes[idx, 5].axis('off')

    plt.tight_layout()
    plt.show()


    



def optimize_model_multihead():
    opt = SourceMaskOptimizer(modelClass=LithographyUNet, 
                   modelPath='./checkpoints/best_model.pth', 
                   device='cuda')

    # target_resist = masks.read_mask_from_img('ganopc-data/artitgt/10605.glp.png', mask_grid_size=256)
    # target_resist = masks.read_mask_from_img('ganopc-data/artitgt/572.glp.png', mask_grid_size=256)
    target_resist = masks.read_mask_from_img('ganopc-data/artitgt/343.glp.png', mask_grid_size=256)



    # This now works from zeros with source-mask optimization!
    optimized_mask, optimized_illum, history = opt.optimize(
        target_resist=target_resist,
        illumination_shape=(32, 32),  # Shape of illumination quadrant
        num_iterations=4500,
        lr_mask=0.2,                  # Learning rate for mask
        lr_illum=0.1,                 # Learning rate for illumination
        initial_blur_mask=10.0,       # Start with heavy blur (coarse)
        final_blur_mask=0.5,          # End with light blur (fine details)
        blur_illum=1.0,               # Constant blur for illumination smoothness
        binarize_final=True,
        binary_iterations=500
    )
    litho_sim = simulator.LithographySimulator(sim_config)

    optimization_visualizer.show_optimization_results(
        target_resist, 
        optimized_mask, 
        optimized_illum,
        opt.model, 
        litho_sim,
        sim_config,
        history
    )

def visualize_dataloader_sample(
    data_dir,
    split="train",
    batch_size=1,
    index_in_batch=0
):
    dataset = LithographyDataset(data_dir, split=split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    #print one batch


    mask_tensor, illum_tensor, intensity_tensor, resist_tensor = next(iter(loader))
    mask = mask_tensor[index_in_batch, 0].cpu().numpy()
    illum = illum_tensor[index_in_batch, 0].cpu().numpy()
    intensity = intensity_tensor[index_in_batch, 0].cpu().numpy()
    resist = resist_tensor[index_in_batch, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title("Mask")
    axes[0].axis("off")

    axes[1].imshow(illum, cmap="inferno")
    axes[1].set_title("Illumination")
    axes[1].axis("off")

    axes[2].imshow(intensity, cmap="inferno")
    axes[2].set_title("Aerial Intensity")
    axes[2].axis("off")

    axes[3].imshow(resist, cmap="gray")
    axes[3].set_title("Resist")
    axes[3].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()