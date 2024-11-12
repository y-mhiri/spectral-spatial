# Spectral-Spatial 

This repository contains code for hyperspectral image denoising using TV based regularizers. Optimisation is performed using Chambolle-Pock algorithm [1]

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/spectral-spatial-analysis.git
    cd spectral-spatial-analysis
    ```

2. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Algorithms

- `chambolle_pock.py`: Implementation of the Chambolle-Pock algorithm.
- `grad_alignement.py`: Implementation of denoising using a gradient alignment regularization.
- `nabla.py`: Contains functions for computing gradients.
- `tvprior.py`: Implementation of denoising using a Total Variation Prior.
- `tv_plus_grad_alignement.py`: Combines Total Variation and gradient alignment.

### Datasets

- `datasets.py`: Functions for loading and processing datasets.
- `show_dataset.ipynb`: Jupyter notebook for visualizing datasets.

### Metrics

- `metrics.py`: Functions for computing various metrics.

### Scripts

- `ACP.ipynb`: Jupyter notebook for ACP analysis.
- `benchmark.py`: Script for benchmarking algorithms.
- `grad_alignement.ipynb`: Jupyter notebook for denoising using a gradient alignment regularization.
- `HyDe.ipynb`: Jupyter notebook for HyDe analysis.
- `tv_plus_grad_alignement.ipynb`: Jupyter notebook for TV + Gradient Alignment.
- `tvprior.ipynb`: Jupyter notebook for denoising using a TV Prior.

### Results
- Contains results from various experiments and analyses.

