import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Colormap configuration
CMAP_CONFIG = {
    "mask": "gray",
    "mask_ft": "viridis",
    "filtered_ft": "viridis",
    "wafer_field": "plasma",
    "wafer_intensity": "hot",
    "resist_profile": "gray"
}

def _plot_image(ax, data, cmap, title, xlabel, ylabel, cbar_label, extent=None, norm=None, set_black_bg=False):
    im = ax.imshow(data, cmap=cmap, origin='lower', extent=extent, norm=norm)
    if set_black_bg:
        ax.set_facecolor('black')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=cbar_label)
    return im


def visualize_simulation_results(return_obj, mask=None, config=None, cmap_config=None):
    # Use custom colormap config or default
    cmaps = cmap_config if cmap_config is not None else CMAP_CONFIG
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    plot_idx = 1
    
    # Calculate frequency grid for proper axes
    freq_extent, spatial_extent = _get_extents(config)
    
    # Plot original mask if provided
    if mask is not None:
        ax = plt.subplot(2, 3, plot_idx)
        _plot_image(ax, mask, cmaps.get("mask", "gray"), 
                   'Original Mask', 'X position (nm)', 'Y position (nm)', 
                   'Transmission', extent=spatial_extent)
        plot_idx += 1
    
    # Plot Fourier transform of mask (magnitude)
    if "mask_ft" in return_obj:
        ax = plt.subplot(2, 3, plot_idx)
        mask_ft_mag = np.abs(return_obj["mask_ft"])
        norm = LogNorm(vmin=mask_ft_mag[mask_ft_mag>0].min(), vmax=mask_ft_mag.max())
        _plot_image(ax, mask_ft_mag, cmaps.get("mask_ft", "viridis"),
                   'Mask Fourier Transform (Magnitude)', 'Frequency X (1/nm)', 
                   'Frequency Y (1/nm)', 'Magnitude (log)', 
                   extent=freq_extent, norm=norm)
        plot_idx += 1
    
    # Plot filtered Fourier transform (magnitude) - with black for cutoff regions
    if "filtered_ft" in return_obj:
        ax = plt.subplot(2, 3, plot_idx)
        filtered_ft_mag = np.abs(return_obj["filtered_ft"])
        filtered_ft_mag_masked = np.copy(filtered_ft_mag)
        filtered_ft_mag_masked[filtered_ft_mag_masked == 0] = np.nan
        norm = LogNorm(vmin=filtered_ft_mag[filtered_ft_mag>0].min(), vmax=filtered_ft_mag.max())
        _plot_image(ax, filtered_ft_mag_masked, cmaps.get("filtered_ft", "viridis"),
                   'Total Filtered FT (After Pupil)', 'Frequency X (1/nm)', 
                   'Frequency Y (1/nm)', 'Magnitude (log)', 
                   extent=freq_extent, norm=norm, set_black_bg=True)
        plot_idx += 1
    
    # Plot wafer field (magnitude)
    if "wafer_field" in return_obj:
        ax = plt.subplot(2, 3, plot_idx)
        wafer_field_mag = np.abs(return_obj["wafer_field"])
        _plot_image(ax, wafer_field_mag, cmaps.get("wafer_field", "plasma"),
                   'Wafer Field (Magnitude)', 'X position (nm)', 'Y position (nm)',
                   'Field magnitude', extent=spatial_extent)
        plot_idx += 1
    
    # Plot wafer intensity
    if "wafer_intensity" in return_obj:
        ax = plt.subplot(2, 3, plot_idx)
        _plot_image(ax, return_obj["wafer_intensity"], cmaps.get("wafer_intensity", "hot"),
                   'Wafer Intensity', 'X position (nm)', 'Y position (nm)',
                   'Intensity', extent=spatial_extent)
        plot_idx += 1
    
    # Plot resist profile
    if "resist_profile" in return_obj:
        ax = plt.subplot(2, 3, plot_idx)
        _plot_image(ax, return_obj["resist_profile"], cmaps.get("resist_profile", "gray"),
                   'Resist Profile (Final Pattern)', 'X position (nm)', 'Y position (nm)',
                   'Exposed (1) / Unexposed (0)', extent=spatial_extent)
        plot_idx += 1
    
    plt.tight_layout()
    plt.show()


def visualize_comparison_multi(return_objs, masks=None, config=None, cmap_config=None, 
                               parameter='resist_profile', titles=None):
    cmaps = cmap_config if cmap_config is not None else CMAP_CONFIG
    n_results = len(return_objs)
    
    # Determine grid layout
    if n_results <= 3:
        nrows, ncols = 1, n_results
        figsize = (6 * n_results, 5)
    elif n_results <= 6:
        nrows, ncols = 2, 3
        figsize = (16, 10)
    elif n_results <= 9:
        nrows, ncols = 3, 3
        figsize = (16, 14)
    else:
        nrows = int(np.ceil(n_results / 4))
        ncols = 4
        figsize = (18, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    if n_results == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_results > 1 else axes
    
    # Get extents
    freq_extent, spatial_extent = _get_extents(config)
    
    # Generate default titles if not provided
    if titles is None:
        titles = [f"Result {i+1}" for i in range(n_results)]
    
    # Plot each result
    for idx, (return_obj, title) in enumerate(zip(return_objs, titles)):
        if idx >= len(axes):
            break
            
        ax = axes[idx] if n_results > 1 else axes
        
        # Get data based on parameter
        if parameter == 'mask':
            if masks is None or idx >= len(masks):
                continue
            data = masks[idx]
            cmap = cmaps.get("mask", "gray")
            xlabel, ylabel = 'X position (nm)', 'Y position (nm)'
            cbar_label = 'Transmission'
            extent = spatial_extent
            norm = None
            set_black_bg = False
            
        elif parameter == 'mask_ft':
            if "mask_ft" not in return_obj:
                continue
            data = np.abs(return_obj["mask_ft"])
            cmap = cmaps.get("mask_ft", "viridis")
            xlabel, ylabel = 'Frequency X (1/nm)', 'Frequency Y (1/nm)'
            cbar_label = 'Magnitude (log)'
            extent = freq_extent
            norm = LogNorm(vmin=data[data>0].min(), vmax=data.max())
            set_black_bg = False
            
        elif parameter == 'filtered_ft':
            if "filtered_ft" not in return_obj:
                continue
            data = np.abs(return_obj["filtered_ft"])
            data_masked = np.copy(data)
            data_masked[data_masked == 0] = np.nan
            data = data_masked
            cmap = cmaps.get("filtered_ft", "viridis")
            xlabel, ylabel = 'Frequency X (1/nm)', 'Frequency Y (1/nm)'
            cbar_label = 'Magnitude (log)'
            extent = freq_extent
            norm = LogNorm(vmin=data[~np.isnan(data)].min(), vmax=np.nanmax(data))
            set_black_bg = True
            
        elif parameter == 'wafer_field':
            if "wafer_field" not in return_obj:
                continue
            data = np.abs(return_obj["wafer_field"])
            cmap = cmaps.get("wafer_field", "plasma")
            xlabel, ylabel = 'X position (nm)', 'Y position (nm)'
            cbar_label = 'Field magnitude'
            extent = spatial_extent
            norm = None
            set_black_bg = False
            
        elif parameter == 'wafer_intensity':
            if "wafer_intensity" not in return_obj:
                continue
            data = return_obj["wafer_intensity"]
            cmap = cmaps.get("wafer_intensity", "hot")
            xlabel, ylabel = 'X position (nm)', 'Y position (nm)'
            cbar_label = 'Intensity'
            extent = spatial_extent
            norm = None
            set_black_bg = False
            
        elif parameter == 'resist_profile':
            if "resist_profile" not in return_obj:
                continue
            data = return_obj["resist_profile"]
            cmap = cmaps.get("resist_profile", "gray")
            xlabel, ylabel = 'X position (nm)', 'Y position (nm)'
            cbar_label = 'Exposed (1) / Unexposed (0)'
            extent = spatial_extent
            norm = None
            set_black_bg = False
        else:
            raise ValueError(f"Unknown parameter: {parameter}")
        
        _plot_image(ax, data, cmap, title, xlabel, ylabel, cbar_label, 
                   extent=extent, norm=norm, set_black_bg=set_black_bg)
    
    # Hide unused subplots
    for idx in range(n_results, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()


def _get_extents(config):
    if config is not None:
        mask_grid_size = config.get("mask_grid_size", 512)
        mask_width_nm = config.get("mask_width_nm", 1000)
        grid_spacing = mask_width_nm / mask_grid_size
        freq = np.fft.fftshift(np.fft.fftfreq(mask_grid_size, d=grid_spacing))
        freq_extent = [freq[0], freq[-1], freq[0], freq[-1]]
        spatial_extent = [-mask_width_nm/2, mask_width_nm/2, -mask_width_nm/2, mask_width_nm/2]
    else:
        freq_extent = None
        spatial_extent = None
    return freq_extent, spatial_extent
