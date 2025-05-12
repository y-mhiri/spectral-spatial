import sys
import os 

# Configuration des chemins
path = '/home/ndiayem/Documents/spectral-spatial/src'
sys.path.append(os.path.join(path, 'datasets'))
sys.path.append(os.path.join(path, 'algorithms'))
sys.path.append(os.path.join(path, 'metrics'))

import argparse
import torch
import zarr
import time
import matplotlib.pyplot as plt
from pansharpening import PANDataset
from alg1 import PANTVGradProj
from torchvision import transforms
from math import sqrt
import yaml
from datetime import datetime
from metrics import compute_metrics


def run_single_experiment(args, group_path=None):
    """Exécute une seule expérience avec les paramètres donnés"""
    # Utiliser le chemin du groupe si spécifié, sinon le chemin de base
    out_path = group_path if group_path else args.storage_path
    
    # Configuration des paramètres
    device = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    torch.manual_seed(args.seed)

    # Chargement des données
    crop_transform = transforms.Compose([transforms.CenterCrop(args.crop_size)])
    dataset = PANDataset(
        root_dir=args.data_path,
        split='train',
        transform=crop_transform if args.crop_center else None,
        normalize=True,
        scale=args.scale,
        sigma=args.sigma,
        sigma1=args.noise_level,
        device=device,
        size=args.crop_size if args.crop_center else None,
        seed=args.seed
    )

    subset = torch.utils.data.Subset(dataset, args.image_idx)
    A, A_adj, R, R_adj = dataset.get_operators()

    # Initialisation du fichier de résultats
    root = zarr.open(f'{out_path}/results.zarr', mode='w')

    # Simulation des données bruitées
    Hsi_noisy = dataset.simulate_low_res_hsi(subset[0].unsqueeze(0))
    Pan_noisy = dataset.get_panchromatic(subset[0].unsqueeze(0))

    root.create_dataset('Hsi_noise', data=Hsi_noisy.cpu().numpy())
    root.create_dataset('Pan_noise', data=Pan_noisy.cpu().numpy())

    # Sauvegarde des paramètres
    root.attrs.update({
        'lambda': args.lmbda,
        'lambda_m': args.lmbda_m,
        'p': args.p,
        'q': args.q,
        'r': args.r,
        'noise_level': args.noise_level,
        'sigma': args.sigma,
        'scale': args.scale,
        'data_idx': args.image_idx,
        'crop': args.crop_center,
        'crop_size': args.crop_size,
        'device': device,
        'seed': args.seed
    })

    # Initialisation des résultats
    metrics = {}
    reconstructed_ar = torch.zeros([len(subset), dataset.nband, args.crop_size, args.crop_size], 
                                 device=device, dtype=dtype)
    loss_ar = torch.zeros([len(subset), args.max_iter], device=device, dtype=dtype)

    # Boucle principale
    for j, data in enumerate(subset):
        print(f"Traitement de l'image {j+1}/{len(subset)}")

        X = data.unsqueeze(0).to(device=device, dtype=dtype)
        Y_H = dataset.simulate_low_res_hsi(data.unsqueeze(0)).to(device=device, dtype=dtype)
        Y_M = dataset.get_panchromatic(data.unsqueeze(0)).to(device=device, dtype=dtype)

        optim = PANTVGradProj(
            A=A,
            Aadj=A_adj,
            spectral_op=R,
            spectral_op_t=R_adj,
            max_iter=args.max_iter,
            lmbda=args.lmbda,
            lmbda_m=args.lmbda_m,
            tau=args.tau,
            tol=args.tol,
            scale=dataset.scale,
            max_iter_gp=args.max_iter_gp,
            p=args.p,
            q=args.q,
            r=args.r,
            verbose=True
        )

        start_time = time.time()
        reconstructed, loss = optim(Y_H, Y_M)
        compute_time = time.time() - start_time

        # Sauvegarde des résultats
        reconstructed_ar[j] = reconstructed
        loss_ar[j] = loss
        root.attrs['time'] = compute_time

        # Calcul des métriques
        sample_metrics = compute_metrics(gt=X, est=reconstructed, numpy=True)
        for metric in sample_metrics:
            if metric in metrics:
                metrics[metric].append(sample_metrics[metric])
            else:
                metrics[metric] = [sample_metrics[metric]]

        torch.cuda.empty_cache()

    # Finalisation de la sauvegarde
    for metric in metrics:
        root.attrs[metric] = metrics[metric]

    root.create_dataset('reconstructed', data=reconstructed_ar.cpu().numpy())
    root.create_dataset('loss', data=loss_ar.cpu().numpy())

    # Sauvegarde des informations supplémentaires
    info = {
        'datasets': {
            args.data_path.split('/')[-1].split('.')[0]: args.data_path
        },
        'experiment': {
            'date': datetime.now().isoformat(),
            'parameters': {
                'lambda': args.lmbda,
                'lambda_m': args.lmbda_m,
                'max_iter': args.max_iter,
                'max_iter_gp': args.max_iter_gp,
                'tau': args.tau,
                'p': args.p,
                'q': args.q,
                'r': args.r,
                'noise_level': args.noise_level,
                'sigma': args.sigma,
                'scale': args.scale
            }
        }
    }

    with open(os.path.join(out_path, 'info.yaml'), 'w') as f:
        yaml.safe_dump(info, f)

    print(f"Expérience terminée. Résultats dans {out_path}")

def main():
    parser = argparse.ArgumentParser()

    # Paramètres de base
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--cuda_avail", type=bool, default=True)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--storage_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="/home/ndiayem/Documents/spectral-spatial/data/harvard.zarr")
    parser.add_argument("--seed", type=int, default=42)

    # Paramètres d'exécution
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--max_iter_gp", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-12)

    # Paramètres des données
    parser.add_argument("--image_idx", nargs="+", type=int, default=[6, 17])
    parser.add_argument("--crop_center", type=bool, default=True)
    parser.add_argument("--crop_size", type=int, default=256)

    # Paramètres algorithmiques (peuvent être des listes pour les groupes)
    parser.add_argument("--lmbda", type=float, nargs="+", required=True)
    parser.add_argument("--lmbda_m", type=float, nargs="+", required=True)
    parser.add_argument("--tau", type=float, nargs="+", required=True)
    parser.add_argument("--p", type=float, nargs="+", required=True)
    parser.add_argument("--q", type=float, nargs="+", required=True)
    parser.add_argument("--r", type=float, nargs="+", required=True)
    parser.add_argument("--noise_level", type=float, nargs="+", required=True)
    parser.add_argument("--sigma", type=float, nargs="+", required=True)
    parser.add_argument("--scale", type=int, nargs="+", required=True)

    # Paramètre pour nommer les groupes
    parser.add_argument("--group_names", type=str, nargs="+", help="Noms des groupes de paramètres")

    args = parser.parse_args()

    # Validation des paramètres
    param_lists = {
        'lmbda': args.lmbda,
        'lmbda_m': args.lmbda_m,
        'tau': args.tau,
        'p': args.p,
        'q': args.q,
        'r': args.r,
        'noise_level': args.noise_level,
        'sigma': args.sigma,
        'scale': args.scale
    }

    # Vérifier que toutes les listes ont la même longueur
    lengths = [len(v) for v in param_lists.values()]
    if len(set(lengths)) > 1:
        raise ValueError("Toutes les listes de paramètres doivent avoir la même longueur")

    num_groups = lengths[0]
    
    # Utiliser les noms de groupes fournis ou générer des noms par défaut
    if args.group_names:
        if len(args.group_names) != num_groups:
            raise ValueError("Le nombre de noms de groupes ne correspond pas au nombre de groupes de paramètres")
        group_names = args.group_names
    else:
        group_names = [f"group_{i}" for i in range(num_groups)]

    print(f"Configuration de {num_groups} groupes de paramètres:")
    for i, name in enumerate(group_names):
        print(f"\nGroupe {name}:")
        for param, values in param_lists.items():
            print(f"  {param}: {values[i]}")

    # Exécution pour chaque groupe
    for group_idx in range(num_groups):
        group_name = group_names[group_idx]
        group_path = os.path.join(args.storage_path, group_name)
        os.makedirs(group_path, exist_ok=True)

        print(f"\n=== Exécution du groupe {group_name} ===")

        # Création d'un namespace avec les paramètres du groupe
        group_args = argparse.Namespace(
            device=args.device,
            cuda_avail=args.cuda_avail,
            dtype=args.dtype,
            storage_path=args.storage_path,  # Non utilisé car on spécifie group_path
            data_path=args.data_path,
            seed=args.seed,
            max_iter=args.max_iter,
            max_iter_gp=args.max_iter_gp,
            tol=args.tol,
            image_idx=args.image_idx,
            crop_center=args.crop_center,
            crop_size=args.crop_size,
            lmbda=args.lmbda[group_idx],
            lmbda_m=args.lmbda_m[group_idx],
            tau=args.tau[group_idx],
            p=args.p[group_idx],
            q=args.q[group_idx],
            r=args.r[group_idx],
            noise_level=args.noise_level[group_idx],
            sigma=args.sigma[group_idx],
            scale=args.scale[group_idx]
        )

        # Exécution de l'expérience
        run_single_experiment(group_args, group_path)

    print("\nToutes les exécutions sont terminées!")

if __name__ == "__main__":
    main()