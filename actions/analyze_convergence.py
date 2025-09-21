import argparse
import os
import zarr
import glob

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from rich import print

sns.set_style('darkgrid')
plt.rc('font', family='serif')



def generate_loss_plot(loss_data, folder):
    """Génère la courbe de convergence"""
    if loss_data is None or len(loss_data) == 0:
        return

    plt.figure(figsize=(8, 5))
    
    # Tracé pour chaque image
    for i, loss in enumerate(loss_data):
        plt.plot(loss, label=f'Image {i}')
    
    plt.xlabel('Itérations')
    plt.ylabel('Fonction de coût (échelle log)')
    plt.title('Évolution de la fonction de coût')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    # Sauvegarde
    plt.savefig(os.path.join(folder, 'cost_function.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("[green]Graphique de convergence généré")



def fetch_group_results(group_path):
    # zarr_path, output_folder):
    """Analyse principale des résultats"""
    try:
        zarr_path = os.path.join(group_path, 'results.zarr')
        root = zarr.open(zarr_path, mode='r')
        
        losses = root['loss']
        recons
        metadata = root.attrs

        nimages = len(metadata['image_idx'])
        
        p = str(int(metadata['p'])) if metadata['p'] != float('inf') else 'inf'
        q = str(int(metadata['q'])) if metadata['q'] != float('inf') else 'inf'
        r = str(int(metadata['r'])) if metadata['r'] != float('inf') else 'inf'
        plot_name = f"{metadata['noise_level']}_{metadata["algorithm"]}_l{p}{q}{r}"
            
        return losses, plot_name

        
    except Exception as e:
        print(f"[red]Erreur d'analyse : {str(e)}[/red]")
        return False

def get_convergence(losses):
    


def analyze_results(storage_path, output_folder):

    groups = glob.glob(os.path.join(storage_path,'group_*'))

    first_group = 0
    while not os.path.isfile(os.path.join(groups[first_group], 'info.yaml')):
        first_group += 1

    losses, plot_name = fetch_group_results(groups[first_group+1])
    df = get_convergence(losses)

    plot_loss(losses, niter, plot_name)
    for group_path in groups:

        if os.path.isfile(os.path.join(group_path, 'info.yaml')):
            losses, plot_name = fetch_group_results(group_path)

            niter = get_convergence(losses)
            plot_loss(losses, niter, plot_name)
            


    create_tabular(df, output_folder, f'convergence.tex')

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
    parser.add_argument('--storage_path', required=True, help='Chemin vers les résultats')
    parser.add_argument('--out', type=str, help='Output folder name (stored in run folder).', default="metrics")
    args = parser.parse_args()
    # Détermination des chemins
    output_folder = os.path.join(args.storage_path, args.out)

    os.makedirs(output_folder, exist_ok=True)

    print(f"[bold]Analyse des résultats : {args.storage_path}")
    
    if analyze_results(args.storage_path, output_folder):
        print("[bold green]Analyse terminée avec succès!")
    else:
        print("[bold red]Échec de l'analyse")