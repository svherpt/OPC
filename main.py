import numpy as np
import src.core.simulator.lithography_simulator as simulator
import src.visualizers.simulator.simulation_visualizer as simulation_visualizer
import src.core.simulator.masks as masks
import json
from src.core.ml.models import MultiTargetUNet
import torch
from PIL import Image
from src.core.ml.litho_mask_optimizer import MaskOptimizer
from src.core.ml.inferer import Inferer
import src.visualizers.ml.optimizer_visualizer as optimization_visualizer
import src.core.simulator.light_sources as light_sources
import time

with open("sim_config.json", "r") as f:
    sim_config = json.load(f)


def main():
    #show_n_masks('./augmented_medium/train/inputs', 10)
    
    #test_defocus_results(sim_config, initial_mask)
    dir_path = 'augmented_small'
    dir_path = 'augmented_medium'
    dir_path = 'augmented_massive'
    
    litho_sim = simulator.LithographySimulator(sim_config)
    source_illumination = light_sources.get_source_grid(sim_config)  # shape: (Npupil, Npupil)

    random_mask = masks.read_mask_from_img('ganopc-data/artitgt/1.glp.png', **sim_config)

    t0 = time.time()
    simResults = litho_sim.simulate(random_mask, source_illumination)
    t1 = time.time()

    print(f"Simulation time: {t1 - t0:.2f} seconds")
    simulation_visualizer.visualize_simulation_results(simResults, mask=random_mask, illumination=source_illumination, config=sim_config)  

    # random_masks = masks.get_dataset_masks('./data/ganopc-data/artitgt', 1, **sim_config)
    # test_ML_model(sim_config, random_masks)
    
    # show_augmentation()

    #train_model('./augmented_massive', target_type='resists')
    #optimize_model_multihead('ganopc-data/artitgt')

    # test_ML_model()
    # optimize_model_multihead()



def test_ML_model():
    inferer = Inferer(modelClass=MultiTargetUNet, model_name='litho_surrogate_multi_good.pth', device='cuda', base_ch=64)
    
    img = masks.read_mask_from_img('ganopc-data/artitgt/10605.glp.png')
    intensity, resist = inferer.predict(img)


    #Print min and max
    print(f"Predicted resist min: {resist.min()}, max: {resist.max()}")

    #show results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Input Mask')
    axs[1].imshow(intensity, cmap='gray')
    axs[1].set_title('Predicted Intensity')
    axs[2].imshow(resist, cmap='gray')
    axs[2].set_title('Predicted Resist')
    plt.show()

    



def optimize_model_multihead():
    opt = MaskOptimizer(modelClass=MultiTargetUNet, 
                   modelPath='./models/litho_surrogate_multi_good.pth', 
                   device='cuda')

    # target_resist = masks.read_mask_from_img('ganopc-data/artitgt/10605.glp.png', mask_grid_size=256)
    # target_resist = masks.read_mask_from_img('ganopc-data/artitgt/572.glp.png', mask_grid_size=256)
    target_resist = masks.read_mask_from_img('ganopc-data/artitgt/71.glp.png', mask_grid_size=256)

    # This now works from zeros!
    optimized_mask, history = opt.optimize(
    target_resist=target_resist,
    num_iterations=4500,
    lr=0.2,               # Learning rate
    initial_blur=10.0,     # Start with heavy blur (coarse)
    final_blur=0.5,       # End with light blur (fine details)
    binarize_final=True,
    binary_iterations=500
)
    
    litho_sim = simulator.LithographySimulator(sim_config)

    optimization_visualizer.show_optimization_results(target_resist, optimized_mask, opt.model, litho_sim, history)


def train_model(input_dir, target_type='resists'):
    trainer = LithoSurrogateTrainerMulti(
    data_dir=input_dir,
    batch_size=16,
    learning_rate=1e-4,
    device='cuda'
    )

    trainer.train(num_epochs=20, save_path='litho_surrogate_multi.pth')
    trainer.visualize_predictions(num_samples=10)


    # trainer = LithoSurrogateTrainer(
    #     data_dir=input_dir,
    #     target_type=target_type,
    #     batch_size=16,
    #     learning_rate=1e-4,
    #     device='cuda',
    #     img_size=256
    # )
    
    # trainer.train(num_epochs=20, save_path='litho_surrogate.pth')
    
    # trainer.visualize_predictions(num_samples=10)


if __name__ == "__main__":
    main()