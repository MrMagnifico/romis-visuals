import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from os import path


# General constants
OUTPUT_DIR = path.join("images", "misc-visualisations")

# Parameters for MAPE images and colorbar
MAPE_NORMALISATION  = Normalize(0, 0.35)
MAPE_COLOR_MAP      = "viridis"

def omis_alpha_colorbar():
    # Create figure and axes
    fig = plt.figure()
    ax = fig.add_axes([0.8, 0.05, 0.1, 0.9])

    # Define color bar params
    range       = mpl.colors.Normalize(-1, 1)
    color_map   = mpl.colors.LinearSegmentedColormap.from_list("", [(0, 0.5, 1.0), (0, 0, 0), (1, 0.5, 0)])
    all_ticks   = [-1.0, 0, 1.0]
    cb = mpl.colorbar.ColorbarBase(ax,
                                   orientation='vertical',
                                   drawedges=True,
                                   ticks=all_ticks,
                                   cmap=color_map,
                                   norm=range)
    cb.ax.tick_params(labelsize=16)
    cb.outline.set_color('black')
    cb.outline.set_linewidth(1)
    
    # Save plot and clear canvas
    file_path = path.join(OUTPUT_DIR, "alpha-colorbar.png")
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=420, transparent=True)
    plt.cla()
    plt.clf()

def mape_images_colorbar():
    # Create figure and axes
    fig = plt.figure()
    ax  = fig.add_axes([0.8, 0.05, 0.1, 0.9])

    # Define color bar params
    cb = mpl.colorbar.ColorbarBase(ax, orientation='vertical', drawedges=True, cmap=MAPE_COLOR_MAP, norm=MAPE_NORMALISATION)
    cb.ax.tick_params(labelsize=16)
    cb.outline.set_color('black')
    cb.outline.set_linewidth(1)

    # Save plot and clear canvas
    file_path = path.join(OUTPUT_DIR, "mape-colorbar.png")
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0, dpi=420, transparent=True)
    plt.cla()
    plt.clf()


if __name__ == "__main__":
    # Style settings for all plots
    SMALL_SIZE  = 16
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 28
    plt.style.use("seaborn-v0_8")
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    omis_alpha_colorbar()
    mape_images_colorbar()