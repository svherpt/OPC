import numpy as np
import src.core.simulator.lithography_simulator as simulator
import src.visualisers.simulator.simulation_visualiser as simulation_visualiser
import src.core.simulator.masks as masks
import json
import src.core.ml.data_augmenter as data_augmenter
from src.core.ml.trainer import LithoSurrogateNet, LithoSurrogateTrainer
from src.core.ml.litho_nn_multi_head import LithoSurrogateTrainerMulti
from src.core.ml.litho_mask_optimizer import LithoMaskOptimizer
import torch
from PIL import Image

with open("sim_config.json", "r") as f:
    sim_config = json.load(f)


def main():

    
    #show_n_masks('./augmented_medium/train/inputs', 10)
    #generate_n_augmentations(num_masks=1000, n_augmentations=10)
    
    #test_defocus_results(sim_config, initial_mask)
    dir_path = './augmented_small'
    dir_path = './augmented_medium'
    dir_path = './augmented_massive'
    #generate_n_augmentations(1000, 10, output_dir=dir_path)

    # sim_config["mask_grid_size"] = 512
    #generate_n_augmentations(1000, 10, output_dir='./augmented_medium')
    # for i in range(5):
    #     generate_n_augmentations(1000, 10, output_dir='./augmented_massive')

    # random_masks = masks.get_dataset_masks('./data/ganopc-data/artitgt', 1, **sim_config)
    # test_ML_model(sim_config, random_masks)
    
    # show_augmentation()

    #train_model('./augmented_massive', target_type='resists')
    optimize_model_multihead('./data/ganopc-data/artitgt')


def show_augmentation():
    test_mask =  masks.read_mask_from_img('./data/ganopc-data/artitgt/1.glp.png', **sim_config)
    augmenter = data_augmenter.MaskAugmenter(seed=42)
    
    # Visualize new augmentations
    augmenter.visualize_augmentations(test_mask)


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


def generate_n_augmentations(num_masks, n_augmentations=5, output_dir='./augmented_medium'):
    augmenter = data_augmenter.MaskAugmenter()

    random_masks = masks.get_dataset_masks('./data/ganopc-data/artitgt', num_masks, **sim_config)
    augmenter.save_dataset(random_masks, output_dir=output_dir, sim_config=sim_config, augmentations_per_mask=n_augmentations, train_split=0.8)


def show_n_masks(dir_path, num_masks=5):
    random_masks = masks.get_dataset_masks(dir_path, num_masks, **sim_config)

    for i, mask in enumerate(random_masks):
        litho_sim = simulator.LithographySimulator(sim_config)
        sim_results = litho_sim.simulate(mask)
        simulation_visualiser.visualize_simulation_results(sim_results, mask=mask, config=sim_config)


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


def test_ML_model(sim_config, mask_list):
    optimizer = MaskOptimizer(sim_config)

    for mask in mask_list:
        mask *= 255
        masks.visualise_mask(mask)

        print(np.max(mask))

        optimized_mask, history = optimizer.optimize_mask(
        target_resist=mask,
        num_iterations=1000,
        )

        optimizer.visualize_optimization(mask, optimized_mask, history)


if __name__ == "__main__":
    main()