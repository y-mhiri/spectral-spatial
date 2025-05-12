import argparse
import os
import numpy as np
import zarr
import matplotlib.pyplot as plt
import seaborn as sns
from rich import print
from glob import glob

sns.set_style('darkgrid')
plt.rc('font', family='serif')

def generate_latex_table(metrics, folder, group_name=None):
    """Génère un tableau LaTeX des métriques brutes avec gestion des groupes"""
    if not metrics:
        return

    # Trouver le nombre d'images
    n_images = 1
    for metric_name, metric_values in metrics.items():
        if isinstance(metric_values, (list, np.ndarray)):
            if len(metric_values) > 0:
                if isinstance(metric_values[0], (list, np.ndarray)):
                    n_images = max(n_images, len(metric_values[0]))
                else:
                    n_images = max(n_images, len(metric_values))

    # Créer le tableau LaTeX
    tex_name = 'metrics_table.tex' if group_name is None else f'metrics_table_{group_name}.tex'
    tex_path = os.path.join(folder, tex_name)
    
    with open(tex_path, 'w') as f:
        # Ajouter un titre avec le nom du groupe si spécifié
        if group_name:
            f.write(f"% Résultats pour le groupe {group_name}\n")
        
        f.write("\\begin{tabular}{|c|" + "|".join(["c"]*len(metrics)) + "|}\n")
        f.write("\\hline\n")
        f.write("Image & " + " & ".join(metrics.keys()) + " \\\\\n")
        f.write("\\hline\n")
        
        # [Le reste de la fonction reste identique...]
        # ... (conserver le même code pour remplir le tableau)

def generate_figures(root, folder, save=True, group_name=None):
    """Génère les figures de visualisation avec support des groupes"""
    # Modifier les noms des fichiers pour inclure le groupe
    fig1_path = os.path.join(folder, 'cost_function.png' if group_name is None else f'cost_function_{group_name}.png')
    fig2_path = os.path.join(folder, 'metrics_comparison.png' if group_name is None else f'metrics_comparison_{group_name}.png')

    # [Le reste de la fonction reste identique, mais utiliser les nouveaux chemins]
    # ... (conserver le même code de génération des figures)

def process_group(group_path, group_name=None):
    """Traite un seul groupe de résultats"""
    print(f"\n[bold cyan]Traitement du groupe {group_name if group_name else 'principal'}...[/bold cyan]")
    
    fig_folder = os.path.join(group_path, 'figures')
    os.makedirs(fig_folder, exist_ok=True)

    try:
        root = zarr.open(f"{group_path}/results.zarr", mode='r')
        generate_figures(root, fig_folder, args.save, group_name)
        print(f"[bold green]Analyse du groupe {group_name if group_name else 'principal'} terminée![/bold green]")
    except Exception as e:
        print(f"[bold red]Erreur lors de l'analyse du groupe {group_name if group_name else 'principal'}: {str(e)}[/bold red]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', type=str, required=True,
                       help='Chemin vers les résultats (doit contenir les sous-dossiers group_*)')
    parser.add_argument('--group', type=int, 
                       help='Numéro de groupe spécifique à analyser')
    parser.add_argument('--all_groups', action='store_true',
                       help='Analyser tous les groupes automatiquement')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Sauvegarder les résultats')
    args = parser.parse_args()

    print("[bold green]Début de l'analyse des résultats...[/bold green]")
    
    if args.all_groups:
        # Traiter tous les groupes automatiquement
        group_dirs = glob(os.path.join(args.storage_path, 'group_*'))
        for group_dir in group_dirs:
            group_name = os.path.basename(group_dir)
            process_group(group_dir, group_name)
        
        # Traiter également le dossier principal s'il existe
        if os.path.exists(os.path.join(args.storage_path, 'results.zarr')):
            process_group(args.storage_path)
    elif args.group is not None:
        # Traiter un groupe spécifique
        group_path = os.path.join(args.storage_path, f'group_{args.group}')
        process_group(group_path, f'group_{args.group}')
    else:
        # Traiter seulement le dossier principal
        process_group(args.storage_path)

    print("\n[bold green]Toutes les analyses sont terminées![/bold green]")