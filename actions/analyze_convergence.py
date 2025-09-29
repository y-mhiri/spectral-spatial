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
            plt.plot(loss, label=label, marker=next(markers))
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


        niter = metadata['max_iter_cp']
        nimages = len(metadata['image_idx'])
        
        p = str(int(metadata['p'])) if metadata['p'] != float('inf') else 'inf'
        q = str(int(metadata['q'])) if metadata['q'] != float('inf') else 'inf'
        r = str(int(metadata['r'])) if metadata['r'] != float('inf') else 'inf'
        plot_name = f"{group_num}_{metadata['algorithm']}_l{p}{q}{r}.png"
            
        return losses, plot_name, niter 

        
    except Exception as e:
        print(f"[red]Erreur d'analyse : {str(e)}[/red]")
        return False

    

def analyze_results(storage_path, output_folder, one_plot, max_iter):

    groups = glob.glob(os.path.join(storage_path,'group_*'))

    loss_list = []
    niter_list = []


            
    losses, plot_name, niter = fetch_group_results(groups[0])
    n_images_prev = losses.shape[0]
    for group_path in groups:

        
        if os.path.isfile(os.path.join(group_path, 'info.yaml')):
            
            losses, plot_name, niter = fetch_group_results(group_path)
            generate_loss_plot(losses,  os.path.join(output_folder, plot_name))
            if max_iter:
                generate_loss_plot(losses,  os.path.join(output_folder, plot_name), max_iter)
            n_images_curr = losses.shape[0]
            if one_plot : 
                assert n_images_curr == n_images_prev
                n_images_prev = n_images_curr

            niter_list.append(niter)
            loss_list.append(losses)

    n_images = n_images_curr
    if one_plot:
        loss_array = np.array(loss_list).reshape(n_images, len(loss_list), -1)        

        for im in range(n_images):
            labels = [f'{i} sub-iteration' if i==1 else f'{i} sub-iterations' for i in niter_list]
            generate_loss_plot(loss_array[im], os.path.join(output_folder, f'loss_per_group_image_{im}.png'), labels=labels)
            
            if max_iter:   
                generate_loss_plot(loss_array[im], os.path.join(output_folder, f'loss_per_group_image_{im}.png'),max_iter=max_iter, labels=labels)

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
    parser.add_argument('--one_plot', action='store_true', help='Creates only one plot when True')
    parser.add_argument('--max_iter', type=int, default=None, help='If specified, defines the number of iterations to plot')
    parser.add_argument('--out', type=str, help='Output folder name (stored in run folder).', default="metrics")
    args = parser.parse_args()

    output_folder = os.path.join(args.storage_path, args.out)

    os.makedirs(output_folder, exist_ok=True)

    print(f"Analyse des résultats : {args.storage_path}")
    
    analyze_results(args.storage_path, output_folder, args.one_plot, args.max_iter)