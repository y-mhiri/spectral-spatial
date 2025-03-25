import torch
import torch.nn as nn
from tqdm.auto import tqdm
from nabla import nabla , nabla_adjoint
from pansharpening import PANDataset

class ProximalGradient(nn.Module):
    """
    Algorithme de gradient proximal pour résoudre le problème de pansharpening hyperspectral.

    Attribut :
    maxiter (int): nombre maximale d'itération
    lambda (float): parametre de regularisation
    lambda_m (float): regularisation sur l'attache aux données
    tau (float): Pas de descend 
    tol (float): critére de convergence
    A (function) : 
    Aadj (function) : 

    Methode :
    Grad_f : calcul le gradient de l'attaches aux données
    proj  : fait une projection de l'image si sa norme est supérieur à 1 
    grad_proj : fait une descend de gradient projecté 
    proxg : calcul l'operateur proximal de g 
    forward : la fonction d'optimisation 
    """

    def __init__(self, A, Aadj,R ,max_iter=100, lmbda=1.0, lmbda_m=1.0, tau=0.1, tol=1e-7, verbose=True):
        super(ProximalGradient, self).__init__()
        self.max_iter = max_iter
        self.lmbda = lmbda  # Paramètre de régularisation pour la variation totale
        self.lmbda_m = lmbda_m  # Paramètre de régularisation pour l'attache aux données multispectrales
        self.tau = tau  # Pas de gradient
        self.tol = tol  # Tolérance pour la convergence
        self.verbose = verbose  # Affichage des informations
        self.A = A  # l'operateur de sous échantillonnage plus flou gaussien
        self.Aadj = Aadj # l'opérateur adjoint de A  
        self.R = R  # reponse spectrale de l'image hyperspectral 
    

    def convergence_criteria(self,U0, U1):
        """
        U1 (torch.tensor):l'image estimée à l'itération i [h,w,c]
        U0 (torch.tensor) : l'image estimée à l'itération i-1
        """
        a = (torch.linalg.norm(U1-U0)/torch.linalg.norm(U0)) < self.tol 
        return a 
    
    def grad_f(self, U, Y_H, Y_M):
        """
        Calcule le gradient de la fonction f(U).

            U (torch.tensor) : l'image hyperspectral estimée de taille l'image original [h,w,c]
            Y_H (torch.tensor) : l'image hyperspectral de base resolution spatiale [h//scale,w//scale,c] avec scale le facteur de sous échantillonnage
            Y_M  (torch.tensor): l'image panchromatique obtenu en faisant une moyenne selon les bandes de l'image original [h,w,1]
            
        """
        # Terme 1 : Gradient de 1/2 ||Y_H -  U B||_F^2
        #grad1 = (U @ B - Y_H) @ B.T # Si B est une matrice
        # Si B est une fonction 
        grad1 =  self.Aadj((self.A(U ,8) - Y_H),8)

        # Terme 2 : Gradient de (lambda_m / 2) ||Y_M - R H U||_F^2
        grad2 = self.lmbda_m * (self.R.T @ (self.R @ U - Y_M))

        return grad1 + grad2
    
    def proj(self, z):
        """
        
        projection sur la boule unité pour la norme l221.

         z (torch.tensor) :  [2,n,m,c]
        """

        return z/ torch.maximum(torch.norm(z, dim=-1, keepdim=True), torch.ones_like(z))
    

    def grad_proj(self,x):
        """
         x (torch.tensor) : [h,w,c]
        """
        n, m,c = x.shape
        z0 = torch.ones((2, n, m,c))
        for i in range(self.max_iter):
            grad_z = -2 * nabla(nabla_adjoint(z0) + x / self.lmbda)  
            z = self.proj(z0 - self.tau * grad_z)
            if self.convergence_criteria(z, z0, self.tol):  
                break
            z0 = z
        return z
    


    
    def proxg(self,x):
        """
        x (torch.tensor) : [h,w,c]
        """
        z = torch.clone(x)
        z = self.grad_proj(z, self.tau,self.max_iter,self.lmbda, self.tol)
        y = x + self.lmbda * nabla_adjoint(z) 
        return y
    

    def forward(self,U, Y_H, Y_M):
        """
        Résout le problème d'optimisation.
        """
        # Initialisation
        h, w, _ = Y_M.shape
        _, _, c = Y_M.shape
        U = torch.zeros((h, w, c))

        # Boucle d'optimisation
        for it in tqdm(range(self.max_iter)):
            # Gradient de f(U)
            grad_U = self.grad_f(U, Y_H, Y_M)

            # Mise à jour de U
            U = self.proxg(U - self.lmbda * grad_U)

        return U
