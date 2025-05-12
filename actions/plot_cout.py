import argparse
import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print

sns.set_style('darkgrid')
plt.rc('font', family='serif')

def generate_latex_table(metrics, folder):
    """Génère un tableau LaTeX des métriques brutes avec gestion robuste des valeurs"""
    if not metrics:
        return

    # Trouver le nombre d'images
    n_images = 1  # Par défaut, supposons 1 image
    for metric_name, metric_values in metrics.items():
        if isinstance(metric_values, (list, np.ndarray)):
            if len(metric_values) > 0:
                if isinstance(metric_values[0], (list, np.ndarray)):
                    n_images = max(n_images, len(metric_values[0]))
                else:
                    n_images = max(n_images, len(metric_values))

    # Créer le tableau LaTeX
    tex_path = os.path.join(folder, 'metrics_table.tex')
    with open(tex_path, 'w') as f:
        # En-tête du tableau
        f.write("\\begin{tabular}{|c|" + "|".join(["c"]*len(metrics)) + "|}\n")
        f.write("\\hline\n")
        f.write("Image & " + " & ".join(metrics.keys()) + " \\\\\n")
        f.write("\\hline\n")
        
        # Données pour chaque image
        for img_idx in range(n_images):
            row = [str(img_idx)]
            for metric_name, metric_values in metrics.items():
                val = None
                
                # Extraction de la valeur
                if isinstance(metric_values, (list, np.ndarray)):
                    if len(metric_values) > 0:
                        if isinstance(metric_values[0], (list, np.ndarray)):
                            if len(metric_values[0]) > img_idx:
                                val = metric_values[0][img_idx]
                        elif len(metric_values) > img_idx:
                            val = metric_values[img_idx]
                        elif len(metric_values) == 1:  # Cas spécial pour liste de longueur 1
                            val = metric_values[0]
                
                # Conversion et formatage
                if val is not None:
                    try:
                        if metric_name == 'PSNR':
                            row.append(f"{float(val):.2f} dB")
                        elif metric_name == 'SAM':
                            if isinstance(val, (float, int, np.number)):
                                if np.isnan(val):
                                    row.append("nan")
                                else:
                                    row.append(f"{float(val):.4f}")  # Valeur en radians sans conversion
                            else:
                                row.append("-")
                        elif metric_name in ['CC', 'SSIM', 'RNMSE']:
                            row.append(f"{float(val):.4f}")
                        else:
                            row.append(f"{float(val):.4f}")
                    except (TypeError, ValueError) as e:
                        print(f"[yellow]Erreur conversion {metric_name}: {val} ({type(val)}) - {e}[/yellow]")
                        row.append("-")
                else:
                    row.append("-")
            
            f.write(" & ".join(row) + " \\\\\n")
            f.write("\\hline\n")
        
        f.write("\\end{tabular}\n")
    
    print(f"[bold green]Tableau LaTeX généré : {tex_path}")

def generate_figures(root, folder, save=True):
    """Génère les figures de visualisation"""
    # Figure 1: Courbe de convergence
    if 'loss' in root:
        fig1, ax1 = plt.subplots(1, 1, figsize=(6, 4))
        loss_ar = root['loss'][:]
        ax1.plot(loss_ar.T, label='Fonction de coût')
        ax1.set_xlabel('Itérations')
        ax1.set_ylabel('Fonction de coût')
        ax1.set_title('Évolution de la fonction de coût')
        ax1.set_yscale('log')
        
        if save:
            plt.savefig(os.path.join(folder, 'cost_function.png'), bbox_inches='tight', dpi=300)
            plt.close(fig1)
        else:
            plt.show()
    
    # Récupération des métriques
    metrics = {}
    for k, v in root.attrs.items():
        if k not in ['lambda', 'lambda_m', 'p', 'q', 'r', 'time', 
                    'noise_level', 'sigma', 'scale', 'seed',
                    'data_idx', 'crop', 'crop_size', 'device']:
            if isinstance(v, (list, np.ndarray)):
                metrics[k] = v
                print(f"[cyan]Métrique {k}: {v} (type: {type(v)})[/cyan]")
                if isinstance(v, (list, np.ndarray)):
                    print(f"  Premier élément: {v[0]} (type: {type(v[0])})")
    
    if metrics:
        # Génère le tableau LaTeX
        generate_latex_table(metrics, folder)
        
        # Figure 2: Comparaison des métriques
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        for metric_name, values in metrics.items():
            if len(values) > 0:
                if isinstance(values[0], (list, np.ndarray)):
                    for i, sub_values in enumerate(values):
                        ax2.plot(range(len(sub_values)), sub_values, marker='o', label=f'{metric_name}_{i}')
                else:
                    ax2.plot(range(len(values)), values, marker='o', label=metric_name)
        
        ax2.set_xlabel('Index image')
        ax2.set_ylabel('Valeur métrique')
        ax2.set_title('Comparaison des métriques')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True)
        
        if save:
            plt.savefig(os.path.join(folder, 'metrics_comparison.png'), bbox_inches='tight', dpi=300)
            plt.close(fig2)
        else:
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', type=str, required=True,
                       help='Chemin vers les résultats')
    parser.add_argument('--group', type=int, 
                       help='Numéro de groupe si applicable')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Sauvegarder les résultats')
    args = parser.parse_args()

    print("[bold green]Début de l'analyse des résultats...[/bold green]")
    folder = args.storage_path
    if args.group:
        folder = os.path.join(folder, f'group_{args.group}')

    fig_folder = os.path.join(folder, 'figures')
    os.makedirs(fig_folder, exist_ok=True)

    try:
        root = zarr.open(f"{folder}/results.zarr", mode='r')
        generate_figures(root, fig_folder, args.save)
        print("[bold green]Analyse terminée avec succès![/bold green]")
    except Exception as e:
        print(f"[bold red]Erreur lors de l'analyse: {str(e)}[/bold red]")