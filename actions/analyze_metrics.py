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

def analyze_results(zarr_path, output_folder):
    """Analyse principale des résultats"""
    try:
        root = zarr.open(zarr_path, mode='r')
        
        # Création du dossier de sortie
        os.makedirs(output_folder, exist_ok=True)
        
        # Extraction des métriques
        metrics = {}
        for k, v in root.attrs.items():
            if k not in ['lambda', 'lambda_m', 'p', 'q', 'r', 'time']:
                if isinstance(v, (list, np.ndarray)):
                    metrics[k] = v
        
        # Génération des graphiques
        if 'loss' in root:
            generate_loss_plot(root['loss'][:], output_folder)
        
        if metrics:
            generate_latex_table(metrics, output_folder)
            generate_metric_plots(metrics, output_folder)
        
        return True
    
    except Exception as e:
        print(f"[red]Erreur d'analyse : {str(e)}[/red]")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--storage_path', required=True, help='Chemin vers les résultats')
    parser.add_argument('--group', type=int, help='Numéro de groupe')
    args = parser.parse_args()
    print(args.group)
    # Détermination des chemins
    results_path = os.path.join(args.storage_path, f"group_{args.group}" if args.group is not None else "", "results.zarr")
    output_folder = os.path.join(args.storage_path, f"group_{args.group}" if args.group is not None else "", "analysis")

    print(f"[bold]Analyse des résultats : {results_path}")
    
    if analyze_results(results_path, output_folder):
        print("[bold green]Analyse terminée avec succès!")
    else:
        print("[bold red]Échec de l'analyse")