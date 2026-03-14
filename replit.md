# Bonsai Data Representation

## Overview
A scientific tool for reconstructing maximum-likelihood cell-trees from single-cell data and visualizing them with the **Bonsai-scout** interactive web app (Python Shiny).

Originally designed for single-cell RNA-sequencing (scRNA-seq) analysis, but works on any high-dimensional dataset.

## Architecture

- **`bonsai/`** - Core Bonsai tree-reconstruction algorithm (Python)
- **`bonsai_scout/`** - Interactive visualization Shiny web app
  - `app.py` - Main Shiny application entry point
  - `run_bonsai_scout_app.py` - CLI wrapper to run the app with data path arguments
- **`backbone_based_bonsai/`** - Backbone-based variant of the Bonsai algorithm
- **`downstream_analyses/`** - Post-processing analysis scripts
- **`optional_preprocessing/`** - Data preprocessing utilities
- **`examples/`** - Example datasets and run configurations
- **`start_app.py`** - Entry point that starts the Shiny app on port 5000

## Running the App

The workflow runs `python3 start_app.py` which:
1. Sets environment variables pointing to preprocessed example data
2. Launches the Shiny app on `0.0.0.0:5000`

## Example Data

Example data for the simulated binary tree dataset is located at:
- `examples/example_data/simulated_binary_6_gens_samplingNoise/`

Preprocessed visualization data (generated, not committed):
- `examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/bonsai_vis_data.hdf`
- `examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/bonsai_vis_settings.json`

## Workflow to Generate Data from Scratch

1. Run Bonsai algorithm:
   ```
   python3 bonsai/bonsai_main.py --config_filepath examples/1_simple_example/example_configs.yaml --step all
   ```

2. Run visualization preprocessing:
   ```
   python3 bonsai_scout/bonsai_scout_preprocess.py \
     --results_folder examples/1_simple_example/results/simulated_binary_6_gens_samplingNoise/ \
     --annotation_path examples/example_data/simulated_binary_6_gens_samplingNoise/annotation/ \
     --take_all_genes False
   ```

3. Start the app: `python3 start_app.py`

## Dependencies

All packages are installed via pip:
- `shiny==0.9.0`, `shinyswatch==0.6.1`, `faicons==0.2.2` - Web framework
- `numpy`, `scipy`, `pandas`, `scikit-learn` - Scientific computing
- `matplotlib` - Plotting
- `h5py`, `tables` - HDF5 data storage
- `ruamel.yaml`, `natsort`, `psutil` - Utilities

## Port

The Shiny app runs on port **5000** (host: `0.0.0.0`).
