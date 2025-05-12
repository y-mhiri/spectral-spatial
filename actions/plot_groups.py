import argparse
import os
import numpy as np
import zarr
import glob
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

def organize_data(file_paths, group_param):
    metrics_data = {}
    for file_path in file_paths:
        root = zarr.open(file_path, mode='r')
        data = {}
        for key in root.attrs:
            data[key] = root.attrs[key]

        group_value = data[group_param]
        if group_value not in metrics_data:
            metrics_data[group_value] = []
        metrics_data[group_value].append({
            'CC': data['CC'][0],
            'PSNR': data['PSNR'][0],
            'RNMSE': data['RNMSE'][0],
            'SAM': data['SAM'][0],
            'SSIM': data['SSIM'][0][0],
            'time': data['time']
        })
    return metrics_data

def plot_metrics(metrics_data, group_param, output_folder):
    group_values = sorted(metrics_data.keys())
    metrics_names = ['CC', 'PSNR', 'RNMSE', 'SAM', 'SSIM', 'time']

    for metric_name in metrics_names:
        plt.figure()
        metric_values = [sum([data[metric_name] for data in metrics_data[group_value]]) / len(metrics_data[group_value]) for group_value in group_values]
        plt.plot(group_values, metric_values, 'o-', label=metric_name)
        plt.xlabel(group_param)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs {group_param}')
        plt.legend()
        plt.grid(True)

        # Sauvegarde du graphique
        filename = f"{metric_name.lower()}_vs_{group_param.lower()}.png"
        plt.savefig(os.path.join(output_folder, filename), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"Graphique généré : {filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyse des métriques en fonction des paramètres de groupe.')
    parser.add_argument('--run_path', required=True, help='Chemin vers le dossier des runs')
    parser.add_argument('--run_num', type=int, required=True, help='Numéro du run à analyser')
    parser.add_argument('--group_param', required=True, help='Paramètre de groupe à analyser (e.g., noise_level, lambda, etc.)')
    parser.add_argument('--output_folder', required=True, help='Dossier de sortie pour les graphiques')

    args = parser.parse_args()

    # Chemin vers les fichiers .zarr du run spécifié
    run_path = os.path.join(args.run_path, f"run_{args.run_num}")
    group_paths = glob.glob(os.path.join(run_path, 'group_*'))

    for group_path in group_paths:
        file_paths = glob.glob(os.path.join(group_path, '*.zarr'))
        group_name = os.path.basename(group_path)
        group_output_folder = os.path.join(args.output_folder, group_name)

        # Organiser les données par groupe
        metrics_data = organize_data(file_paths, args.group_param)

        # Tracer les métriques
        plot_metrics(metrics_data, args.group_param, group_output_folder)

if __name__ == "__main__":
    main()
