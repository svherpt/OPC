import numpy as np
import src.core.simulator.lithography_simulator as simulator
import src.visualisers.simulator.simulation_visualiser as simulation_visualiser
import src.core.simulator.masks as masks
import json
from src.core.ml.trainer import LithoSurrogateNet, LithoSurrogateTrainer
from src.core.ml.litho_nn_multi_head import LithoSurrogateTrainerMulti
from src.core.ml.litho_mask_optimizer import LithoMaskOptimizer
import torch
from PIL import Image

with open("sim_config.json", "r") as f:
    sim_config = json.load(f)


def main():
    #show_n_masks('./augmented_medium/train/inputs', 10)
    
    #test_defocus_results(sim_config, initial_mask)
    dir_path = 'augmented_small'
    dir_path = 'augmented_medium'
    dir_path = 'augmented_massive'
    

    # random_masks = masks.get_dataset_masks('./data/ganopc-data/artitgt', 1, **sim_config)
    # test_ML_model(sim_config, random_masks)
    
    # show_augmentation()

    #train_model('./augmented_massive', target_type='resists')
    #optimize_model_multihead('./data/ganopc-data/artitgt')


def optimize_model_multihead(data_dir, model_path='litho_surrogate_multi.pth', num_iterations=1000, learning_rate=0.1):
    litho_sim = simulator.LithographySimulator(sim_config)

    # Load target resist profile
    target_path = "./data/ganopc-data/artitgt/10605.glp.png"
    target_resist = masks.read_mask_from_img(target_path, **sim_config)

    # Load the trained multi-head model
    model = LithoSurrogateTrainerMulti('augmented_small').model.to('cuda')
    model.load_state_dict(torch.load('models/' + model_path))
    model.eval()  # freeze model

    # Initialize the mask optimizer
    optimizer = LithoMaskOptimizer(model, device='cuda', physical_simulator=litho_sim, img_size=target_resist.shape[0])

    # Optimize mask
    optimized_mask, history = optimizer.optimize_mask(
        target_resist=target_resist,
        initial_mask=None,
        num_iterations=num_iterations,
        learning_rate=learning_rate,
        binarization_weight=0.1,
        tv_weight=0.001,
        binarize_final=True
    )

    # Visualize results
    optimizer.visualize_optimization(
        target_resist=target_resist,
        optimized_mask=optimized_mask,
        history=history,
        save_path='optimization_result.png',
        show=True
    )

    # Save optimized mask
    optimizer.save_mask(optimized_mask, 'optimized_mask.png')
    print("Optimization finished and mask saved.")


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