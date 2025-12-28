import numpy as np
import src.simulator.lithography_simulator as simulator
import src.simulator.visualiser as visualiser
import src.simulator.masks as masks
import json
import src.ml.data_augmenter as data_augmenter

with open("sim_config.json", "r") as f:
    sim_config = json.load(f)

def main():

    
    show_n_masks(num_masks=10)

    
    #test_defocus_results(sim_config, initial_mask)

def generate_n_augmentations(num_masks, n_augmentations=5):
    augmenter = data_augmenter.MaskAugmenter()

    random_masks = masks.get_dataset_masks(num_masks, **sim_config)
    augmenter.save_dataset(random_masks, output_dir='./augmented_dataset', sim_config=sim_config, augmentations_per_mask=n_augmentations, train_split=0.8)


def show_n_masks(num_masks=5):
    random_masks = masks.get_dataset_masks(num_masks, **sim_config)

    for i, mask in enumerate(random_masks):
        litho_sim = simulator.LithographySimulator(sim_config)
        sim_results = litho_sim.simulate(mask)
        visualiser.visualize_simulation_results(sim_results, mask=mask, config=sim_config)

def test_sigma_results(sim_config, initial_mask):
    sigma_values = np.linspace(0.5, 3.0, 6)
    param_name = "resist_blur_sigma"
    compare_results(sim_config, initial_mask, param_name, sigma_values, 'resist_profile')

def compare_results(initial_config, initial_mask, param_name, param_values, visualise_parameter):
    return_objs = []
    for i in range(6):
        sim_config_copy = initial_config.copy()
        sim_config_copy[param_name] = param_values[i]

        litho_sim = simulator.LithographySimulator(sim_config_copy)
        sim_results = litho_sim.simulate(initial_mask)
        return_objs.append(sim_results)

    visualiser.visualize_comparison_multi(return_objs, masks=[initial_mask]*6, config=initial_config, parameter=visualise_parameter)

if __name__ == "__main__":
    main()