import argparse
import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
import seaborn as sns
# import tikzplotlib
import rich
import sys
from tqdm import tqdm

# file_dir = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(file_dir, '../..'))
# from src.utils import (
#         tikzplotlib_fix_ncols
# )

sns.set_style('darkgrid')

# Activate LaTeX text rendering
# if available on your system
# plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def generate_figure(data,
                    folder,
                    save=True, **kwargs):
    



    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(data[:].transpose(), label='Cost function')

#     # Fill between the standard deviation
#     ax.fill_between(n_samples_list,
#                         mse_covariance_mean - mse_covariance_std,
#                         mse_covariance_mean + mse_covariance_std,
#                         color='b', alpha=0.2,
#                         label='Standard deviation')

#     # Plot the lower bound
#     ax_cov.plot(n_samples_list, crb, label='Lower bound',
#                 marker='', c='k', linestyle='-')

    ax.set_xlabel('iterations')
    ax.set_ylabel('Cost function')
    ax.set_title(
            'Cost function w.r.t the number of iterations')
    # log-log plot
    ax.set_yscale('log')

    if save:
        plt.savefig(os.path.join(folder, 'cost_function.png'))
        # tikzplotlib_fix_ncols(fig)
        # tikzplotlib.save(os.path.join(folder, 'cost_function.tex'))
        # print('Saved plot in {}'.format(folder))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', type=str,
                        default='data/',
                        help='Path to the data folder where '
                        'results.pkl is located.')  
    parser.add_argument('--save', action='store_true', default=True,
                        help='Save the plot as pdf and LaTeX code')
    args = parser.parse_args()

    rich.print(
            '[bold green]Plotting evolution of the loss wrt the number of iterations.')
    rich.print('[bold green]Folder: {}'.format(args.storage_path))

    folder = args.storage_path

    root = zarr.open(f'{folder}/results.zarr', mode='r')

    loss_ar = root['loss']
    # Plotting
    generate_figure(loss_ar,
                    args.storage_path,
                    args.save)

    plt.show()