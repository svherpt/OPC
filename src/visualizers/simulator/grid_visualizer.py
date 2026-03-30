import matplotlib.pyplot as plt
from src.core.simulator.lithography_simulator import LithographySimulator
import src.core.simulator.illuminator as illuminator
import src.core.misc as misc
import src.core.simulator.masks as masks_module
from src.core.data.mask_augmenter import MaskAugmenter
from src.core.data.illumination_augmenter import IlluminationAugmenter

class GridVisualizer:
    """Visualizes an NxN grid of simulation results from different masks and illumination fields."""
    
    def __init__(self, masks, illuminations, config, grid_size=4):
        assert len(masks) == grid_size, f"Must provide exactly {grid_size} masks"
        assert len(illuminations) == grid_size, f"Must provide exactly {grid_size} illumination fields"
        
        self.masks = masks
        self.illuminations = illuminations
        self.config = config
        self.grid_size = grid_size
        self.simulator = LithographySimulator(config)
        
        # Run all simulations
        self.results = {}
        self.run_simulations()
        
    def run_simulations(self):
        """Run all NxN simulations and cache results."""
        total_sims = self.grid_size * self.grid_size
        print(f"Running {total_sims} simulations...")
        for i, mask in enumerate(self.masks):
            for j, illum in enumerate(self.illuminations):
                sim_num = i * self.grid_size + j + 1
                print(f"  Simulation {sim_num}/{total_sims} (mask {i}, illumination {j})")
                self.results[(i, j)] = self.simulator.simulate(mask, illum)
    
    def visualize(self):
        """Create interactive grid visualization with masks on left and illuminations on top."""
        n = self.grid_size
        fig = plt.figure(figsize=(4*n, 4*n))
        
        # Create gridspec: (n+1)x(n+1) layout (1 for labels + n for data)
        gs = fig.add_gridspec(n+1, n+1, hspace=0.3, wspace=0.3)
        
        # Top row: show the N illuminations
        for j in range(n):
            ax = fig.add_subplot(gs[0, j+1])
            illum_full = illuminator.quadrant_to_full(self.illuminations[j])
            ax.imshow(illum_full, cmap='hot')
            ax.set_title(f'Illum {j}', fontsize=12, weight='bold')
            ax.axis('off')
        
        # Left column: show the N masks
        for i in range(n):
            ax = fig.add_subplot(gs[i+1, 0])
            ax.imshow(self.masks[i], cmap='gray')
            ax.set_title(f'Mask {i}', fontsize=12, weight='bold')
            ax.axis('off')
        
        # Main grid: resist profiles from simulations
        axes_grid = {}
        for i in range(n):
            for j in range(n):
                ax = fig.add_subplot(gs[i+1, j+1])
                axes_grid[(i, j)] = ax
        
        # Plot all resist profiles
        self.plot_grid(axes_grid)
        
        plt.suptitle(f'{n}x{n} Simulation Grid: Masks vs Illuminations', fontsize=14, weight='bold')
        plt.show()
    
    def plot_grid(self, axes_grid):
        """Plot all resist profiles in the grid."""
        for (i, j), ax in axes_grid.items():
            result = self.results[(i, j)]
            resist_profile = result['resist_profile']
            
            ax.imshow(resist_profile, cmap='gray')
            ax.set_title(f'M{i}:I{j}', fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])


def visualize_grid(masks, illuminations, config, grid_size=4):
    """Convenience function to create and display a grid visualization."""
    visualizer = GridVisualizer(masks, illuminations, config, grid_size)
    visualizer.visualize()


if __name__ == "__main__":
    grid_size = 8
    
    config = misc.get_simulation_config()
    
    # Generate N random augmented masks
    mask_augmenter = MaskAugmenter()
    base_mask = masks_module.get_random_dataset_mask(**config)
    augmented_masks, _ = mask_augmenter.batch_augment([base_mask], augmentations_per_mask=grid_size)
    
    # Generate N random augmented illuminations
    illum_augmenter = IlluminationAugmenter()
    augmented_illums = [illum_augmenter.augment_illumination(**config) for _ in range(grid_size)]
    
    # Visualize as NxN grid
    visualize_grid(augmented_masks, augmented_illums, config, grid_size)
