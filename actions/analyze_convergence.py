import argparse
import os
import zarr
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

sns.set_style('darkgrid')
plt.rc('font', family='serif')



def generate_loss_plot(loss_data, filename, max_iter=None, labels=None):


    plt.figure(figsize=(15, 12))
    markers = iter(['o', 'v', 's', 'D', 'P','X'])
    # Tracé pour chaque image
    for i, loss in enumerate(loss_data):
        label = labels[i] if labels is not None else f'graph_{i}'
        if max_iter is None:
            plt.plot(loss, label=label, marker=next(markers, 'o'))
        else:
            plt.plot(loss[0:max_iter], label=label,marker=next(markers))
    
    plt.xlabel('Number of iterations')
    plt.ylabel('Objective function (in log scale)')
    plt.title('')
    plt.yscale('log')
    # plt.ylim(100,400)
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde
    if max_iter:
        filename, _ = filename.split('.png')
        plt.savefig(os.path.join(f'{filename}_{max_iter}.png') , bbox_inches='tight', dpi=300)
    else:
        plt.savefig(filename , bbox_inches='tight', dpi=300)
    plt.close()



def fetch_group_results(group_path):

    try:
        zarr_path = os.path.join(group_path, 'results.zarr')
        group_num = group_path.split('group_')[-1]
        root = zarr.open(zarr_path, mode='r')
        
        losses = root['loss']
        metadata = root.attrs
        return losses, metadata

        
    except Exception as e:
        print(f"[red]Erreur d'analyse : {str(e)}[/red]")
        return False

    

def analyze_results(storage_path, output_folder, min_iter, max_iter):

    groups = glob.glob(os.path.join(storage_path,'group_*'))

    losses_dicts = []
    max_iter_cp_val = []
    lmbda_val = []

    for group_path in groups:    

        if os.path.isfile(os.path.join(group_path, 'info.yaml')):
            
            losses, metadata = fetch_group_results(group_path)
            if metadata['max_iter_cp'] not in max_iter_cp_val:
                max_iter_cp_val.append(metadata['max_iter_cp'])

            if metadata['lmbda'] not in lmbda_val:
                lmbda_val.append(metadata['lmbda'])

            losses_dicts.append({'max_iter_cp': metadata['max_iter_cp'], 
                                 'lmbda' : metadata['lmbda'], 
                                 'losses': losses[0,min_iter:max_iter] if max_iter else losses[0,min_iter:]})
        else:
            print(f'No finished run found at {group_path}.')            
    
    for val in max_iter_cp_val:
        filtered_dicts = [d for d in losses_dicts if d['max_iter_cp'] == val]
        array = np.array([np.reshape(d['losses'],(d['losses'].shape[-1], -1)) for d in filtered_dicts])
        labels = [f'$\lambda = {d["lmbda"]}$' for d in filtered_dicts]
        generate_loss_plot(array, os.path.join(output_folder, f'max_iter_cp_{val}.png'), labels=labels)

    for val in lmbda_val:
        filtered_dicts = [d for d in losses_dicts if d['lmbda'] == val]
        array = np.array([np.reshape(d['losses'],(d['losses'].shape[-1], -1)) for d in filtered_dicts])
        labels = [f'{d["max_iter_cp"]} sub-iteration' if d["max_iter_cp"]==1 else f'{d["max_iter_cp"]} sub-iterations' for d in filtered_dicts]
        generate_loss_plot(array, os.path.join(output_folder, f'lmbda_{val}.png'), labels=labels)
    

            

    return True

def create_tabular(df, output_folder, filename):

    tex_path = os.path.join(output_folder, filename)
    with open(tex_path, 'w') as f:

        columns = list(df.columns)

        f.write("\\begin{tabular}{|c|" + "|".join(["c"]*len(columns)) + "|}\n")
        f.write("\\hline\n")
        f.write(" & ".join(columns) + " \\\\\n")
        f.write("\\hline\n")
        
        for index, row in df.iterrows():
            f.write(" & ".join(row.astype(str)) + " \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
    
    print(f"[green]Tableau LaTeX généré : {tex_path}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', required=True, help='Storage path (automatically filled y quanat).')
    parser.add_argument('--max_iter', type=int, default=None)
    parser.add_argument('--min_iter', type=int, default=0)
    parser.add_argument('--out', type=str, help='Output folder name (stored in run folder).', default="metrics")
    args = parser.parse_args()

    output_folder = os.path.join(args.storage_path, args.out)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Analyse des résultats : {args.storage_path}")
    
    analyze_results(args.storage_path, output_folder, args.min_iter, args.max_iter)