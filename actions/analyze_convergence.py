import argparse
import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
from loaders import *

sns.set_style('darkgrid')
plt.rc('font', family='serif')

def generate_latex_table(metrics, folder):
    """Génère un tableau LaTeX des métriques brutes"""
    if not metrics:
        return

    # Trouver le nombre d'images
    n_images = max(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in metrics.values())

    # Créer le tableau LaTeX
    tex_path = os.path.join(folder, 'metrics_table.tex')
    with open(tex_path, 'w') as f:
        f.write("\\begin{tabular}{|c|" + "|".join(["c"]*len(metrics)) + "|}\n")
        f.write("\\hline\n")
        f.write("Image & " + " & ".join(metrics.keys()) + " \\\\\n")
        f.write("\\hline\n")
        
        for img_idx in range(n_images):
            row = [str(img_idx)]
            for metric_name, values in metrics.items():
                try:
                    val = values[img_idx] if isinstance(values, (list, np.ndarray)) and len(values) > img_idx else "-"
                    if isinstance(val, (float, np.floating)):
                        if metric_name == 'PSNR':
                            row.append(f"{val:.2f} dB")
                        elif metric_name == 'SAM':
                            row.append(f"{val:.4f} rad")
                        else:
                            row.append(f"{val:.4f}")
                    else:
                        row.append(str(val))
                except (IndexError, TypeError):
                    row.append("-")
            
            f.write(" & ".join(row) + " \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
    
    print(f"[green]Tableau LaTeX généré : {tex_path}")

def generate_metric_plots(metrics, folder):
    """Génère un graphique séparé pour chaque métrique"""
    if not metrics:
        return

    for metric_name, values in metrics.items():
        if not isinstance(values, (list, np.ndarray)) or len(values) == 0:
            continue

        plt.figure(figsize=(8, 5))
        
        # Gestion des valeurs spéciales
        clean_values = []
        for v in values:
            if isinstance(v, (float, int, np.number)):
                if np.isnan(v):
                    clean_values.append(0)  # Remplacer NaN si nécessaire
                else:
                    clean_values.append(v)
            else:
                clean_values.append(0)  # Valeur par défaut

        # Tracé du graphique
        x = range(len(clean_values))
        plt.plot(x, clean_values, 'o-', markersize=8, linewidth=2)
        
        # Configuration spécifique par métrique
        if metric_name == 'PSNR':
            plt.ylabel('dB')
            plt.ylim(0, 100)  # Plage typique pour PSNR
        elif metric_name == 'SAM':
            plt.ylabel('Radians')
            plt.ylim(0, 3.14)  # Plage 0-π
        else:
            plt.ylabel('Valeur')

        plt.xlabel('Index Image')
        plt.title(f'Évolution de {metric_name}')
        plt.grid(True)
        
        # Sauvegarde
        filename = f"metric_{metric_name.lower().replace(' ', '_')}.png"
        plt.savefig(os.path.join(folder, filename), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[green]Graphique généré : {filename}")

def generate_loss_plot(loss_data, metadata, output_dir):
    """Génère la courbe de convergence"""
    if loss_data is None or len(loss_data) == 0:
        return

    # Tracé pour chaque image
    for i, loss in enumerate(loss_data):
        plt.figure(figsize=(8, 5))
        plt.plot(loss, label=f'Image {i}')
    
        plt.xlabel('itérations')
        plt.ylabel('cost function (in log scale)')
        plt.yscale('log')
        plt.legend()
        plt.grid(True)
        group_num, algorithm = metadata['group_num'],metadata['algorithm']

        filepath = os.path.join(output_dir, f'{group_num:03d}_{i:03d}_{algorithm}_loss.png') 
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()

def analyze_results(metadata, output_dir):

    for group_metadata in metadata:
        group_path = group_metadata["group_path"]
        group_name = os.path.basename(group_path)
        group_num = int(group_name.split('_')[1])
        
        loss = load_zarr_arrays(group_path, ["loss"])["loss"]

        algorithm = group_metadata["parameters"].get('algorithm','Unkown')
        filepath = f'{group_num:03d}_{algorithm}'
        filepath = os.path.join(output_dir,filepath)
        generate_loss_plot(loss, {'algorithm':algorithm,'group_num':group_num}, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', required=True, help='Chemin vers les résultats')
    parser.add_argument('--algorithm', type=str,
                       help='Filter by algorithm name')
    parser.add_argument('--groups', type=int, nargs='+', help='Numéro de groupe', default=None)
    args = parser.parse_args()
    
    
    successful = load_experiment_metadata(args.storage_path)
    
    # Filter by algorithm if specified
    if args.algorithm:
        successful = filter_by_algorithm(successful, args.algorithm)
        print(f"Filtered to {args.algorithm} algorithm")
     # Filter by groups if specified
    if args.groups:
        filtered = []
        for metadata in successful:
            group_name = os.path.basename(metadata['group_path'])
            if group_name.startswith('group_'):
                try:
                    group_num = int(group_name.split('_')[1])
                    if group_num in args.groups:
                        filtered.append(metadata)
                except ValueError:
                    continue
        successful = filtered
        print(f"Processing groups: {args.groups}")
    

    # Create output directory
    output_dir = os.path.join(args.storage_path, 'losses')
    os.makedirs(output_dir, exist_ok=True)
    
    analyze_results(successful, output_dir)