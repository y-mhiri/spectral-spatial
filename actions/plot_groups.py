import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import zarr
from rich import print
from glob import glob

def generate_latex_table(metrics, folder):
    """Génère un tableau LaTeX des métriques brutes."""
    if not metrics:
        return

    # Nombre d'images
    n_images = max(len(v) if isinstance(v, (list, np.ndarray)) else 1 for v in metrics.values())

    # Créer le tableau LaTeX
    tex_path = os.path.join(folder, 'metrics_table.tex')
    with open(tex_path, 'w') as f:
        f.write("\\begin{tabular}{|c|" + "|".join(["c"] * len(metrics)) + "|}\n")
        f.write("\\hline\n")
        f.write("Image & " + " & ".join(metrics.keys()) + " \\\\\n")
        f.write("\\hline\n")

        for img_idx in range(n_images):
            row = [str(img_idx)]
            for metric_name, values in metrics.items():
                try:
                    val = values[img_idx] if isinstance(values, (list, np.ndarray)) and len(values) > img_idx else "-"
                    row.append(f"{val:.4f}" if isinstance(val, (float, np.floating)) else str(val))
                except (IndexError, TypeError):
                    row.append("-")
            f.write(" & ".join(row) + " \\\\\n")
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
    
    print(f"[green]Tableau LaTeX généré : {tex_path}")

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


def compare_groups(storage_path, param_name):
    """Compare les métriques entre groupes pour un paramètre donné."""
    group_dirs = sorted(glob(os.path.join(storage_path, 'group_*')))
    if not group_dirs:
        print("[red]Aucun groupe trouvé pour comparaison[/red]")
        return

    all_metrics = {}
    param_values = []
    group_names = []

    for group_dir in group_dirs:
        try:
            root = zarr.open(os.path.join(group_dir, 'results.zarr'), mode='r')
            params = dict(root.attrs)

            param_value = params.get(param_name)
            group_name = os.path.basename(group_dir)
            group_names.append(group_name)
            if param_value is None:
                print(f"[yellow]Paramètre {param_name} non trouvé dans {group_dir}[/yellow]")
                continue

            param_values.append(param_value)

            for metric_name in params:
                if metric_name not in ['lambda', 'lambda_m', 'p', 'q', 'r', 'time', 
                                       'noise_level', 'sigma', 'scale', 'seed', 
                                       'crop', 'crop_size', 'device', 'data_idx']:
                    metric_value = params[metric_name]
                    if isinstance(metric_value, (list, np.ndarray)):
                        mean_value = np.mean(metric_value)
                        all_metrics.setdefault(metric_name, []).append(mean_value)

        except Exception as e:
            print(f"[red]Erreur lecture {group_dir}: {str(e)}[/red]")

    if not param_values:
        return

    # Création du tableau LaTeX pour les moyennes
    latex_table_path = os.path.join(storage_path, 'metrics_table.tex')
    with open(latex_table_path, 'w') as f:
        f.write("\\begin{tabular}{|c|" + "|".join(["c"] * (len(group_names) + 1)) + "|}\n")
        f.write("\\hline\n")
        f.write("Métrique & " + " & ".join(group_names) + " & Moyenne \\\\\n")
        f.write("\\hline\n")

        for metric_name, values in all_metrics.items():
            mean_of_means = np.mean(values)  # Calcul de la moyenne des moyennes
            f.write(f"{metric_name} & " + " & ".join(f"{val:.4f}" for val in values) + f" & {mean_of_means:.4f} \\\\\n")
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
    
    print(f"[green]Tableau LaTeX généré : {latex_table_path}")

    # Création des graphiques
    output_dir = os.path.join(storage_path, 'comparison')
    os.makedirs(output_dir, exist_ok=True)

    for metric_name, values in all_metrics.items():
        if len(values) != len(param_values):
            continue

        plt.figure(figsize=(10, 6))
        sorted_indices = np.argsort(param_values)
        x = np.array(param_values)[sorted_indices]
        y = np.array(values)[sorted_indices]

        plt.plot(x, y, 'o-', markersize=8, linewidth=2)
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs {param_name}')
        plt.xticks(ticks=range(len(group_names)), labels=group_names, rotation=45)
        plt.grid(True)

        if param_name == 'noise_level':
            plt.xscale('log')

        filename = f"compare_{param_name}_{metric_name.lower()}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[green]Graphique de comparaison généré : {filename}")

def analyze_results(zarr_path, output_folder):
    """Analyse principale des résultats."""
    try:
        root = zarr.open(zarr_path, mode='r')
        os.makedirs(output_folder, exist_ok=True)
        
        metrics = {}
        for k, v in root.attrs.items():
            if k not in ['lambda', 'lambda_m', 'p', 'q', 'r', 'time']:
                if isinstance(v, (list, np.ndarray)):
                    metrics[k] = v
        
        if 'loss' in root:
            generate_loss_plot(root['loss'][:], output_folder)
        
        if metrics:
            generate_latex_table(metrics, output_folder)
        
        return True
    
    except Exception as e:
        print(f"[red]Erreur d'analyse : {str(e)}[/red]")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', required=True, help='Chemin vers les résultats')
    parser.add_argument('--group', type=int, help='Numéro de groupe spécifique')
    parser.add_argument('--all_groups', action='store_true', help='Analyser tous les groupes')
    parser.add_argument('--compare', type=str, help='Paramètre à comparer (noise_level, scale, etc.)')
    args = parser.parse_args()

    if args.all_groups:
        # Analyse de tous les groupes
        group_dirs = glob(os.path.join(args.storage_path, 'group_*'))
        for group_dir in group_dirs:
            group_num = os.path.basename(group_dir).split('_')[1]
            print(f"\n[bold]Analyse du groupe {group_num}...")
            
            results_path = os.path.join(group_dir, "results.zarr")
            output_folder = os.path.join(group_dir, "analysis")
            
            if analyze_results(results_path, output_folder):
                print(f"[green]Groupe {group_num} analysé avec succès!")
            else:
                print(f"[red]Échec analyse groupe {group_num}")

        # Comparaison après l'analyse de tous les groupes
        if args.compare:
            print(f"\n[bold]Comparaison des groupes pour {args.compare}...")
            compare_groups(args.storage_path, args.compare)

    elif args.compare:
        # Mode comparaison entre groupes
        print(f"[bold]Comparaison des groupes pour {args.compare}...")
        compare_groups(args.storage_path, args.compare)
    
    else:
        # Analyse d'un groupe spécifique ou du dossier principal
        results_path = os.path.join(args.storage_path, f"group_{args.group}" if args.group else "", "results.zarr")
        output_folder = os.path.join(args.storage_path, f"group_{args.group}" if args.group else "", "analysis")
        
        print(f"[bold]Analyse des résultats : {results_path}")
        if analyze_results(results_path, output_folder):
            print("[bold green]Analyse terminée avec succès!")
        else:
            print("[bold red]Échec de l'analyse")