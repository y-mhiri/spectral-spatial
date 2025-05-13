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
                    clean_values.append(0)
                else:
                    clean_values.append(v)
            else:
                clean_values.append(0)

        # Tracé du graphique
        x = range(len(clean_values))
        plt.plot(x, clean_values, 'o-', markersize=8, linewidth=2)
        
        # Configuration spécifique par métrique
        if metric_name == 'PSNR':
            plt.ylabel('dB')
            plt.ylim(0, 100)
        elif metric_name == 'SAM':
            plt.ylabel('Radians')
            plt.ylim(0, 3.14)
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
    
    for i, loss in enumerate(loss_data):
        plt.plot(loss, label=f'Image {i}')
    
    plt.xlabel('Itérations')
    plt.ylabel('Fonction de coût (échelle log)')
    plt.title('Évolution de la fonction de coût')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(os.path.join(folder, 'cost_function.png'), bbox_inches='tight', dpi=300)
    plt.close()
    print("[green]Graphique de convergence généré")

def compare_groups(storage_path, param_name):
    """Compare les métriques entre groupes pour un paramètre donné"""
    group_dirs = sorted(glob(os.path.join(storage_path, 'group_*')))
    if not group_dirs:
        print("[red]Aucun groupe trouvé pour comparaison[/red]")
        return

    # Récupération des données
    all_metrics = {}
    param_values = []
    
    for group_dir in group_dirs:
        try:
            root = zarr.open(os.path.join(group_dir, 'results.zarr'), mode='r')
            params = dict(root.attrs)
            
            # Valeur du paramètre pour ce groupe
            param_value = params.get(param_name)
            if param_value is None:
                print(f"[yellow]Paramètre {param_name} non trouvé dans {group_dir}[/yellow]")
                continue
            
            param_values.append(param_value)
            
            # Récupération des métriques moyennes
            for metric_name in params:
                if metric_name not in ['lambda', 'lambda_m', 'p', 'q', 'r', 'time', 
                                      'noise_level', 'sigma', 'scale', 'seed',
                                      'data_idx', 'crop', 'crop_size', 'device']:
                    metric_value = params[metric_name]
                    if isinstance(metric_value, (list, np.ndarray)):
                        all_metrics.setdefault(metric_name, []).append(np.mean(metric_value))
        except Exception as e:
            print(f"[red]Erreur lecture {group_dir}: {str(e)}[/red]")

    if not param_values:
        return

    # Création des graphiques
    output_dir = os.path.join(storage_path, 'comparison')
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_name, values in all_metrics.items():
        if len(values) != len(param_values):
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Tri des valeurs selon le paramètre
        sorted_indices = np.argsort(param_values)
        x = np.array(param_values)[sorted_indices]
        y = np.array(values)[sorted_indices]
        
        plt.plot(x, y, 'o-', markersize=8, linewidth=2)
        plt.xlabel(param_name)
        plt.ylabel(metric_name)
        plt.title(f'{metric_name} vs {param_name}')
        plt.grid(True)
        
        if param_name == 'noise_level':
            plt.xscale('log')
        
        filename = f"compare_{param_name}_{metric_name.lower()}.png"
        plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
        plt.close()
        print(f"[green]Graphique de comparaison généré : {filename}")

def analyze_results(zarr_path, output_folder):
    """Analyse principale des résultats"""
    try:
        root = zarr.open(zarr_path, mode='r')
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
    parser.add_argument('--group', type=int, help='Numéro de groupe spécifique')
    parser.add_argument('--all_groups', action='store_true', help='Analyser tous les groupes')
    parser.add_argument('--compare', type=str, help='Paramètre à comparer (noise_level, scale, etc.)')
    args = parser.parse_args()

    if args.compare:
        # Mode comparaison entre groupes
        print(f"[bold]Comparaison des groupes pour {args.compare}...")
        compare_groups(args.storage_path, args.compare)
    elif args.all_groups:
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
    else:
        # Analyse d'un groupe spécifique ou du dossier principal
        results_path = os.path.join(args.storage_path, f"group_{args.group}" if args.group else "", "results.zarr")
        output_folder = os.path.join(args.storage_path, f"group_{args.group}" if args.group else "", "analysis")
        
        print(f"[bold]Analyse des résultats : {results_path}")
        if analyze_results(results_path, output_folder):
            print("[bold green]Analyse terminée avec succès!")
        else:
            print("[bold red]Échec de l'analyse")