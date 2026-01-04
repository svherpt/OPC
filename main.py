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
    #optimize_model_multihead('ganopc-data/artitgt')

    # test_ML_model()
    optimize_model_multihead()



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
    opt = MaskOptimizer(modelClass=MultiTargetUNet, modelPath='litho_surrogate_multi_good.pth', device='cuda')
    
    target_resist = masks.read_mask_from_img('ganopc-data/artitgt/10605.glp.png')
    mask = torch.rand_like(torch.tensor(target_resist)).numpy()

    target_zeros = np.zeros_like(target_resist)
    random_mask = np.random.randint(0, 2, size=target_resist.shape).astype(np.float32)

    #Print the min and max of both target and initial mask
    print(f"Target resist min: {target_resist.min()}, max: {target_resist.max()}")
    print(f"Initial mask min: {mask.min()}, max: {mask.max()}")


    optimized_mask, history = opt.optimize(
        target_resist=target_resist,
        initial_mask=target_resist,
        num_iterations=10000,
        lr=0.1,
        binarization_weight=0,
        tv_weight=0,
        binarize_final=True
    )


    optimization_visualizer.plot_optimization_result(
        target_resist=target_resist,
        optimized_mask=optimized_mask,
        history=history,
        model=opt.model,
        device=opt.device,
        save_dir='./visualizations',
        show=True
    )
    
    opt.save_mask(optimized_mask, './results/optimized_mask.png')


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