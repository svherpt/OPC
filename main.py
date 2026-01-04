import numpy as np
import src.core.simulator.lithography_simulator as simulator
import src.visualizers.simulator.simulation_visualizer as simulation_visualizer
import src.core.simulator.masks as masks
import json
from src.core.ml.models import MultiTargetUNet
import torch
from PIL import Image
from src.core.ml.litho_mask_optimizer import MaskOptimizer
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

    optimize_model_multihead('augmented_small')


def optimize_model_multihead(data_dir, model_path='litho_surrogate_multi.pth', num_iterations=1000, learning_rate=0.1):
    model = MultiTargetUNet()
    model.load_state_dict(torch.load('./models/litho_surrogate_multi_good.pth'))
    model.eval()
    
    opt = MaskOptimizer(model=model, device='cuda')
    target_resist = masks.read_mask_from_img('ganopc-data/artitgt/10605.glp.png')
    
    optimized_mask, history = opt.optimize(
        target_resist=target_resist,
        initial_mask=None,
        num_iterations=1000,
        lr=0.1,
        binarization_weight=0.1,
        tv_weight=0.001,
        binarize_final=True
    )
    
    optimization_visualizer.plot_optimization_result(
        target_resist=target_resist,
        optimized_mask=optimized_mask,
        history=history,
        model=model,
        device=opt.device,
        save_dir='./visualizations',
        show=True
    )
    
    opt.save_mask(optimized_mask, './results/optimized_mask.png')
    print("Optimization complete!")


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