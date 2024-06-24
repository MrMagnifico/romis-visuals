import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib import colormaps
from matplotlib import ticker
from matplotlib.colors import Normalize
from os import path
from PIL import Image
from tqdm import tqdm, trange

from misc_visuals import MAPE_NORMALISATION, MAPE_COLOR_MAP
from utils import create_if_not_exists


# General constants
OUTPUT_DIR          = path.join("images", "error-metrics")
REFERENCES_DIR      = path.join("images", "references")
PICKLE_FILES_DIR    = path.join("cache")
ALL_SCENES          = ["cornell-nightclub", "modern-hall", "the-breakfast-room", "the-modern-living-room"]

# Neighbour count/reservoir size related variables
CANDIDATE_COUNTS                = [16, 32, 64, 256]
RESERVOIR_SIZES                 = [2, 4, 6, 8, 10, 12, 14, 16]
NEIGHBOUR_COUNTS                = [0, 2, 4, 6, 8]
RENDERS_PATH_NEIGHBOUR_COUNT    = "C:\\Users\\willy\\Documents\\University Work\\MSc Thesis\\romis\\renders\\Final Results\\neighbour-count-reservoir-size"

# Iterations count related variables
MAX_ITERATIONS          = 16
TECHNIQUE_VARIANT_PAIRS = [("restir", "biased"), ("restir", "unbiased"),
                           ("restir+", "biased"), ("restir+", "unbiased"),
                           ("rmis", "equal"), ("rmis", "balance"),
                           ("romis", "direct"), ("romis", "u1"), ("romis", "u2"), ("romis", "u4")]
RENDERS_PATH_ITERATIONS = "C:\\Users\\willy\\Documents\\University Work\\MSc Thesis\\romis\\renders\\Final Results\\convergence"
LINE_STYLES             = ['dashed', 'solid',
                           'dashed', 'solid',
                           'dashed', 'solid',
                           'dashed', 'solid', 'solid', 'solid']

# MAPE images related variables
COMPARISON_ITERATION    = 5

def mape(sample: Image, ground_truth: Image):
    # Verify that both images are of the same dimensions
    if sample.size != ground_truth.size:
        raise RuntimeError(f"Sample and ground truth images for MAPE calculation are not the same size\n\
                             Sample: {sample.size} - Ground truth: {ground_truth.size}")

    # Compute average MAPE
    sample_arr          = np.array(sample)
    ground_truth_arr    = np.array(ground_truth)
    sample_norms        = np.linalg.norm(sample_arr,        ord=2, axis=-1) # Per-pixel L2 norms
    ground_truth_norms  = np.linalg.norm(ground_truth_arr,  ord=2, axis=-1) # Per-pixel L2 norms
    per_pixel_mapes     = abs((ground_truth_norms - sample_norms) / ground_truth_norms)
    return np.mean(per_pixel_mapes)                                         # Normalise error and return final MAPE value

def mape_image(sample: Image, ground_truth: Image):
    # Verify that both images are of the same dimensions
    if sample.size != ground_truth.size:
        raise RuntimeError(f"Sample and ground truth images for MAPE calculation are not the same size\n\
                             Sample: {sample.size} - Ground truth: {ground_truth.size}")

    # Compute MAPE
    sample_arr          = np.array(sample)
    ground_truth_arr    = np.array(ground_truth)
    sample_norms        = np.linalg.norm(sample_arr,        ord=2, axis=-1) # Per-pixel L2 norms
    ground_truth_norms  = np.linalg.norm(ground_truth_arr,  ord=2, axis=-1) # Per-pixel L2 norms
    per_pixel_mapes     = abs((ground_truth_norms - sample_norms) / ground_truth_norms)
    return per_pixel_mapes

def neighbour_reservoir_mapes(scene_name: str):
    # Fetch reference image
    reference_path  = path.join(REFERENCES_DIR, f"reference-{scene_name}.png")
    ground_truth    = Image.open(reference_path)
    
    # Compute MAPE for all renders
    print("Computing neighbour count/reservoir size MAPEs")
    mapes: dict[tuple[int, int, int], float] = dict()
    for candidate_count in tqdm(CANDIDATE_COUNTS, desc="Candidate counts"):
        for reservoir_size in tqdm(RESERVOIR_SIZES, desc="Reservoir sizes"):
            for neighbour_count in tqdm(NEIGHBOUR_COUNTS, desc="Neighbour counts"):
                render_path = path.join(RENDERS_PATH_NEIGHBOUR_COUNT, scene_name, f"{candidate_count}-candidates-{reservoir_size}-reservoir-{neighbour_count}-neighbour.png")
                render      = Image.open(render_path)
                mapes[(candidate_count, reservoir_size, neighbour_count)] = mape(render, ground_truth)

    # Save MAPEs
    neighbour_reservoir_pickle_path = path.join(PICKLE_FILES_DIR, f"mapes-neighbour-reservoir-{scene_name}.pickle")
    with open(neighbour_reservoir_pickle_path, "wb") as file:
        pickle.dump(mapes, file)

def plot_neighbour_reservoir_mapes(mapes: dict[tuple[int, int, int], float], scene_name: str):
    # Get global minimum and maximum MAPE
    min_mape = min(mapes.values())
    max_mape = max(mapes.values())

    # Generate plot per candidate count
    for candidate_count in CANDIDATE_COUNTS:
        file_path = path.join(OUTPUT_DIR, scene_name, f"neighbour-reservoir-{scene_name}-{candidate_count}.png")

        # Collect (X, Y, Z) values
        x = [] # Reservoir sizes
        y = [] # Neighbour counts
        z = [] # MAPE scores
        for reservoir_size in RESERVOIR_SIZES:
            for neighbour_count in NEIGHBOUR_COUNTS:
                x.append(reservoir_size)
                y.append(neighbour_count)
                z.append(mapes[(candidate_count, reservoir_size, neighbour_count)])

        # Plot triangulated graph
        clr_map = plt.colormaps["viridis"]
        ax      = plt.axes(projection="3d")
        ax.plot_trisurf(x, y, z, cmap='viridis', shade=True, linewidth=0.2, antialiased=True)
        ax.set_xlabel("Reservoir size")
        ax.set_ylabel("Neighbour count")
        ax.set_zlabel(" MAPE")                                              # The space causes the label to be vertical. Yes.
        ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}')) # Integer formatting
        ax.set_zlim3d(0, max_mape)
        ax.view_init(elev=29, azim=52)                                      # Set angle of view
        plt.colorbar(cm.ScalarMappable(norm=Normalize(vmin=min_mape, vmax=max_mape), cmap=clr_map), ax=ax, location="right")
        plt.savefig(file_path, bbox_inches='tight', pad_inches=0.2, dpi=240)
        plt.clf()

def iterations_mapes(scene_name: str):
    # Fetch reference image
    reference_path  = path.join(REFERENCES_DIR, f"reference-{scene_name}.png")
    ground_truth    = Image.open(reference_path)
    
    print("Iterations MAPEs")
    mapes: dict[tuple[str, str], list[float]] = dict()
    for technique, variant in tqdm(TECHNIQUE_VARIANT_PAIRS, desc="Technique-variant pairs..."):
        # Compute MAPE for each iteration
        mapes[(technique, variant)] = []
        for iteration in trange(1, MAX_ITERATIONS + 1, desc="Iterations..."):
            render_path = path.join(RENDERS_PATH_ITERATIONS, scene_name, technique, variant, f"{iteration} iters.png")
            render      = Image.open(render_path)
            mapes[(technique, variant)].append(mape(render, ground_truth))

    # Save MAPEs
    iterations_pickle_path = path.join(PICKLE_FILES_DIR, f"mapes-iterations-{scene_name}.pickle")
    with open(iterations_pickle_path, "wb") as file:
        pickle.dump(mapes, file)

def plot_iterations_mapes(mapes: dict[tuple[str, str], list[float]], scene_name: str):
    # Plot parameters
    COLOR_MAP   = colormaps['tab20'].colors
    LOG_BASE    = 10
    LINE_WIDTH  = 1

    file_path   = path.join(OUTPUT_DIR, scene_name, f"{scene_name}.png")
    x_values    = np.arange(1, MAX_ITERATIONS + 1)
    for pair_idx, (technique, variant) in enumerate(mapes.keys()):
        technique_label = technique.upper()
        variant_label   = variant.capitalize()
        y_values        = mapes[(technique, variant)]
        plt.plot(x_values, y_values, label=f"{technique_label} - {variant_label}",
                 color=COLOR_MAP[pair_idx],
                 linewidth=LINE_WIDTH, linestyle=LINE_STYLES[pair_idx])
    plt.yscale('log', base=LOG_BASE)
    plt.xlabel("Iterations")
    plt.ylabel("MAPE")
    plt.grid(True, linestyle='--', which="both")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left") # Place outside of axes at the top right https://stackoverflow.com/a/43439132
    plt.savefig(file_path, bbox_inches='tight', dpi=240)
    plt.clf()

def compute_mape_heatmaps():
    print("MAPE Heatmaps")
    mapes: dict[tuple[str, str, str], np.ndarray] = dict()
    for scene in tqdm(ALL_SCENES, "Scenes"):
        # Fetch reference image
        reference_path  = path.join(REFERENCES_DIR, f"reference-{scene}.png")
        ground_truth    = Image.open(reference_path)

        # Compute MAPE for each (technique, variant) pair
        for technique, variant in tqdm(TECHNIQUE_VARIANT_PAIRS, desc="Technique-variant pairs..."):
            render_path = path.join(RENDERS_PATH_ITERATIONS, scene, technique, variant, f"{COMPARISON_ITERATION} iters.png")
            render      = Image.open(render_path)
            mapes[(scene, technique, variant)] = mape_image(render, ground_truth)

    # Save MAPEs
    heatmaps_pickle_path = path.join(PICKLE_FILES_DIR, f"mape-heatmaps.pickle")
    with open(heatmaps_pickle_path, "wb") as file:
        pickle.dump(mapes, file)

def plot_mape_heatmaps(mape_images: dict[tuple[str, str, str], np.ndarray]):
    for scene in tqdm(ALL_SCENES, desc="Plotting MAPE heatmaps per scene"):
        scene_path = path.join(OUTPUT_DIR, scene)
        for technique, variant in TECHNIQUE_VARIANT_PAIRS:
            image       = mape_images[(scene, technique, variant)]
            file_path   = path.join(scene_path, f"mape-heatmap-{technique}-{variant}.png")
            plt.imshow(image, cmap=MAPE_COLOR_MAP, norm=MAPE_NORMALISATION)
            plt.gca().set_axis_off()
            plt.savefig(file_path, bbox_inches='tight', transparent=True, pad_inches=0, dpi=258)
            plt.clf()


if __name__ == "__main__":
    # MAPE calculations
    for scene in ALL_SCENES:
        iterations_mapes(scene)
        neighbour_reservoir_mapes(scene)
    compute_mape_heatmaps()

    # Create folder for each scene
    for scene in ALL_SCENES:
        create_if_not_exists(path.join(OUTPUT_DIR, scene))

    # Plot neighbour-reservoir figures
    for scene in ALL_SCENES:
        neighbour_reservoir_pickle_path = path.join(PICKLE_FILES_DIR, f"mapes-neighbour-reservoir-{scene}.pickle")
        with open(neighbour_reservoir_pickle_path, "rb") as file:
            mapes = pickle.load(file)
            plot_neighbour_reservoir_mapes(mapes, scene)

    # Plot MAPE heatmaps
    heatmaps_pickle_path = path.join(PICKLE_FILES_DIR, f"mape-heatmaps.pickle")
    with open(heatmaps_pickle_path, "rb") as file:
        mape_heatmaps = pickle.load(file)
        plot_mape_heatmaps(mape_heatmaps)

    # Change label sizes for better convergence plots visibility
    SMALL_SIZE  = 12
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 24
    plt.style.use("seaborn-v0_8")
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
   
    # Plot iterations figures
    for scene in ALL_SCENES:
        iterations_pickle_path = path.join(PICKLE_FILES_DIR, f"mapes-iterations-{scene}.pickle")
        with open(iterations_pickle_path, "rb") as file:
            mapes = pickle.load(file)
            plot_iterations_mapes(mapes, scene)
