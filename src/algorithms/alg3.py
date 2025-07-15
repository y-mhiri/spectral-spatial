import torch
from nabla import nabla
from pan_gradient_prox import PANProximalGradient
from tv_grad import TVGradAlignment

class PANTVGradAlignment(PANProximalGradient):
    """
    Calcul de l'opérateur proximale de la TV vectorielle en utilisant chamboll pock.
    Attributs:
        solver (class): résoud l'algo de chamboll pock avec les paramétre nécessaire.

    Methods:
    proxg(input_image)
       cette fonction donne l'opérateur proximale de l'image d'entrée en faisant un algo de chamboll pock

    """
    def __init__(self, params, *args, **kwargs):
        super().__init__(*args, **kwargs)

        params['p'] = self.p
        params['q'] = self.q
        params['r'] = self.r

        self.optim = TVGradAlignment(**params)
        # Assurez-vous que W est initialisé
        self.W = self.optim.W


    def proxg(self,x):
        """
        Donne l'opérateur proximale de la tv vectorielle avec chamboll pock ....
        
        """
        params = {}

        params['compute_L'] = {'nband': x.shape[1]}
        params['K'] = {}
        params['K_adjoint'] = {}
        params['prox_sigma_g_conj'] = {}
        params['prox_tau_f'] = {'y': x, 'sigma2': 1}
        params['loss_fn'] = {}


        return self.optim(x,init=None, verbose=False, params=params, return_loss=False)
    
    def ctv_norm(self, U,eps=1e-8):
            """
            Calcule la norme CTV l^p,q,r avec support pour p,q,r = infini.
            
            Args:
                U (torch.Tensor): Tenseur de gradients [b,c,h,w,2]
                p, q, r (float or torch.inf): Exposants de la norme
                eps (float): Petite valeur pour stabilité numérique
                
            Returns:
                torch.Tensor: Norme CTV [b,1,1,1]
            """
            # Norme p sur les canaux (axis=1)
            if torch.isinf(torch.tensor(self.p)):
                norm_p = torch.amax(torch.abs(U), dim=1, keepdim=True)  # l^infini
            else:
                norm_p = torch.sum(torch.abs(U)**self.p, dim=1, keepdim=True)**(1/(self.p + eps))

            # Norme q sur les dérivées (axis=-1)
            if torch.isinf(torch.tensor(self.q)):
                norm_q = torch.amax(torch.abs(norm_p), dim=-1, keepdim=True)  # l^infini
            else:
                norm_q = torch.sum(norm_p**self.q, dim=-1, keepdim=True)**(1/(self.q + eps))

            # Norme r sur les pixels (axis=(2,3))
            if torch.isinf(torch.tensor(self.r)):
                norm_r = torch.amax(torch.abs(norm_q), dim=(2,3), keepdim=True)  # l^infini
            else:
                norm_r = torch.sum(norm_q**self.r, dim=(2,3), keepdim=True)**(1/(self.r + eps))

            return norm_r
    


    def compute_cost(self, U, Y_H, Y_M):
        """
        Calcule le coût total de la fonction objective.
        
        Args:
            U (torch.Tensor): Image estimée [b,c,h,w]
            Y_H (torch.Tensor): Données hyperspectrales [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Données panchromatiques [b,1,h,w]
            
        Returns:
            float: Coût total
        """
        # Terme d'attache aux données hyperspectrales
        data_term_h = 0.5 * torch.norm(self.A(U)-Y_H)**2
        
        # Terme d'attache aux données panchromatiques
        
        data_term_m = 0.5 * self.lmbda_m * torch.norm(self.spectral_op(U)-Y_M)**2
        
        # Terme de régularisation TV
        grad_U = torch.matmul(self.W,nabla(U).unsqueeze(-1)).squeeze(-1)
        #tv_per_pixel = torch.sqrt(torch.sum(grad_U**2, dim=(1,4)))
        #torch.sum(tv_per_pixel)
        tv_term = self.lmbda * self.ctv_norm(grad_U,eps=1e-8)
        
        return data_term_h + data_term_m + tv_term ,data_term_h,data_term_m,tv_term
    

    def forward(self, Y_H, Y_M):
        """
        Résout le problème d'optimisation complet.
        
        Args:
            Y_H (torch.Tensor): Données hyperspectrales [b,c,h//scale,w//scale]
            Y_M (torch.Tensor): Données panchromatiques [b,1,h,w]
            
        Returns:
            tuple: (Image estimée [b,c,h,w], historique des coûts)
        """
        U = self.Aadj(Y_H).clone()
        cost_history = []
        
        if self.verbose:
            print("\nDébut de l'optimisation:")
            print(f"{'It':<5} | {'Coût total':<12} | {'Data H':<12} | {'Data M':<12} | {'TV':<12} | {'ΔU':<12}")
            print("-" * 80)
        
        cost_history = torch.zeros(self.max_iter)
        for it in range(self.max_iter):
            U_prev = U.clone()
            
            # Étape de gradient
            grad = self.grad_f(U, Y_H, Y_M)
            U = self.proxg(U - self.lmbda * grad)
            
            # Calcul des métriques
            total_cost,data_term_h,data_term_m,tv_term = self.compute_cost(U, Y_H, Y_M)
            delta_U = torch.norm(U - U_prev).item() / (torch.norm(U_prev).item() + 1e-8)
            cost_history[it] = total_cost.item()
            
            # Affichage conditionnel
            if self.verbose and (it % 10 == 0 or it == self.max_iter - 1 or delta_U < self.tol):
                
                print(f"{it:<5} | {total_cost.item():<12.3e} | {data_term_h.item():<12.3e} | "
                      f"{data_term_m.item():<12.3e} | {tv_term.item():<12.3e} | {delta_U:<12.3e}")
                
                if delta_U < self.tol:
                    print(f"\nConvergence atteinte à l'itération {it} (ΔU = {delta_U:.3e} < {self.tol})")
                    break
        
        return U, cost_history