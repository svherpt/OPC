from models import MultiTargetUNet
from optimizer import MaskOptimizer
from optimization_visualizer import plot_optimization_result, plot_optimization_losses
import torch


def optimize_mask():
    model = MultiTargetUNet(base_ch=64)
    model.load_state_dict(torch.load('./checkpoints/multi_target.pth'))
    model.eval()
    
    opt = MaskOptimizer(model=model, device='cuda')
    
    target_resist = opt.load_target('ganopc-data/artitgt/10605.glp.png')
    
    optimized_mask, history = opt.optimize(
        target_resist=target_resist,
        initial_mask=None,
        num_iterations=1000,
        lr=0.1,
        binarization_weight=0.1,
        tv_weight=0.001,
        binarize_final=True
    )
    
    plot_optimization_result(
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


def optimize_with_initial_mask():
    model = MultiTargetUNet(base_ch=64)
    model.load_state_dict(torch.load('./checkpoints/multi_target.pth'))
    
    opt = MaskOptimizer(model=model, device='cuda')
    
    target_resist = opt.load_target('path/to/target.png')
    initial_mask = opt.load_target('path/to/initial_mask.png')
    
    optimized_mask, history = opt.optimize(
        target_resist=target_resist,
        initial_mask=initial_mask,
        num_iterations=500,
        lr=0.05,
        binarization_weight=0.1,
        tv_weight=0.001,
        binarize_final=True
    )
    
    plot_optimization_result(
        target_resist=target_resist,
        optimized_mask=optimized_mask,
        history=history,
        model=model,
        device=opt.device,
        save_dir='./visualizations',
        show=False
    )
    
    plot_optimization_losses(history, save_dir='./visualizations', show=True)
    
    opt.save_mask(optimized_mask, './results/refined_mask.png')


if __name__ == '__main__':
    optimize_mask()
    # optimize_with_initial_mask()