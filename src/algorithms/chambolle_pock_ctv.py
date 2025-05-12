from pan_gradient_prox import PANProximalGradient
from nabla import nabla , nabla_adjoint
import torch




class CHAM_CTV(PANProximalGradient):
    """
    Calcul de l'opérateur proximal de la TV vectorielle par descente de gradient projeté.
    
    Attributs:
        max_iter_gp (int): Nombre maximal d'itérations pour la descente de gradient
        tau (float): Pas de descente
    """

    def __init__(self, A, Aadj,spectral_op,spectral_op_t, max_iter, lmbda, lmbda_m,sigma ,tau,tol, scale,p ,q ,r,verbose):
        """
        Initialise les paramètres de l'algorithme.
        
        Args:
            max_iter_gp (int): Nombre max d'itérations pour la sous-optimisation
            tau (float): Pas de descente pour le gradient projeté
        """
        super().__init__(A, Aadj,spectral_op,spectral_op_t,max_iter, lmbda, lmbda_m, tol, scale,p,q,r,verbose)
        self.p = p 
        self.q = q
        self.r = r
        self.sigma = sigma
        self.tau = tau


        


    def sylvester_solver(self,A, B,C):
        """
        Résout l'équation AX + XB = C par décomposition spectrale
        Version optimisée et robuste pour PyTorch
        
        Paramètres:
            A: torch.Tensor (..., n, n)
            B: torch.Tensor (..., m, m)
            C: torch.Tensor (..., n, m)
        
        Retourne:
            X: Solution (..., n, m)
        """
        def adjoint(V):
            return V.transpose(-1,-2).conj()
        
        # 1. Décomposition spectrale
        R, U = torch.linalg.eig(A)  # A = U diag(R) U^-1
        S, V = torch.linalg.eig(B)  # B = V diag(S) V^-1
        
        # 2. Transformation dans l'espace propre
        F = adjoint(U) @ (C + 0j) @ V  # Equivalent à U^{-1} C V
        
        # 3. Solution diagonale
        W = R[..., :, None] + S[..., None, :]  # AX + XB => R_i + S_j
        Y = F / W
        
        # 4. Retour à l'espace original
        X = U @ Y @ adjoint(V)
        
        # Gestion des cas réels/complexes
        if all(torch.isreal(x.flatten()[0]) for x in [A, B, C]):
            return X.real
        return X
    

    def create_spatial_identity(self,batch=1, channels=31, height=256, width=256):
        """
        Crée un tenseur [batch, channels, height, width] avec des identités spatiales.
        Chaque canal [i,:,:] contient une diagonale de 1 de taille h x w.
        """
        # Initialisation à zéro
        identity = torch.zeros(batch, channels, height, width)
        
        # Remplir la diagonale pour chaque canal
        rows = torch.arange(height)
        cols = torch.arange(width)
        valid_mask = (rows < width) & (cols < height)  # Gère h ≠ w
        identity[:, :, rows[valid_mask], cols[valid_mask]] = 1
        
        return identity
    

    def prox_f_sylvester(self,U,Y_H,Y_M):
        
        b,c,h,w  = U.shape
        R = (1/c)*torch.ones(1,c, device=self.device)
        
        A = self.create_spatial_identity(b,c,h,w) +  self.lmbda*self.lmbda_m * R.T @ R
        B = self.Aadj(self.A(self.lmbda*U))
        C = U + self.Aadj(self.lmbda*Y_H) + self.lmbda_m * R.T @ Y_M
        
        # Solution de AU + UB = C
        return self.sylvester_solver(A, B, C)
        
    
    def project_dual_ball_r(self,U,eps=1e-8):
        """
        Projection sur la boule duale l^{p*,q*,r*} <= 1.
        Gère explicitement p*, q*, r* = infinity.
        """

        def get_dual_exponent(val):
            if val == 1:
                return torch.inf
            elif torch.isinf(torch.tensor(val)):
                return 1.0
            else:
                return 1 / (1 - 1/val)

        p_star = get_dual_exponent(self.p)
        r_star = get_dual_exponent(self.r)
        # Étape 1: Norme p* sur les canaux (axis=1)
        if torch.isinf(torch.tensor(p_star, device=U.device)):
            norm_p_star= torch.amax(torch.abs(U), dim=1, keepdim=True) # l^infini
        else:
            norm_p_star = torch.sum(torch.abs(U)**p_star, dim=1, keepdim=True)**(1/(p_star + eps))


        # Étape 3: Norme r* sur les pixels (axis=(2,3))
        if torch.isinf(torch.tensor(r_star, device=U.device)):
            norm_r_star= torch.amax(torch.abs(norm_p_star), dim=(2,3), keepdim=True)  # l^infini
        else:
            norm_r_star = torch.sum(norm_p_star**r_star, dim=(2,3), keepdim=True)**(1/(r_star + eps))

        # Scaling pour respecter ||U||_{p*,q*,r*} <= 1
        scaling = torch.maximum(torch.tensor(1.0, device=U.device), norm_r_star)
        return U / (scaling + eps)


    def proxg(self, x):
        """Opérateur proximal pour la norme CTV l^p,q,r avec gestion de p,q,r=infini."""
        # Calcul des exposants duaux (gère p,q,r=1 et p,q,r=infini)
        
    

        # Projection duale + formule de Moreau
        
        proj = self.project_dual_ball_r(x,eps=1e-8)
        return proj
    

    def forward(self,y_h,y_m ,init=None, verbose=True, return_loss=True):
        """
            Solve the optimization problem using the Chambolle-Pock algorithm

            Parameters:
            - y: input tensor of shape (batch, channels, height, width)
            - init: initial estimate. If None, set to K^*y
            - verbose: print the progress of the algorithm
            - params: dictionary of additional parameters

            Returns:
            - u: estimate of the solution
            - loss: loss function at each iteration

        """


        
        sigma = self.sigma
        tau = self.tau 


        if init is not None:
            u = torch.clone(init)
        else:
            u = self.Aadj(y_h)

        q = nabla(u)
        v = torch.clone(u) 

  
        loss = torch.zeros(self.max_iter)
        rel = torch.zeros(self.max_iter)

        if verbose:
            print(f'Chambolle Pock algorithm starting...')
        # for it in tqdm(range(self.max_iter)):
        for it in range(self.max_iter):
            
            u_old = torch.clone(u)

            w = self.proxg(q + self.lmbda * sigma * nabla(v))
            u = self.prox_f_sylvester(u - self.lmbda * tau * nabla_adjoint(w),y_h,y_m)

            if self.accelerate:
                self.theta = 1/sqrt(1 + 2 * self.gamma * tau)
                tau = self.gamma * self.theta
                sigma = sigma / self.theta

            v = u + self.theta * (u - u_old)

      
            loss[it] = self.loss_fn(u, y, self.lmbda, **params['loss_fn'])
            rel[it] = torch.norm(u - u_old)/torch.norm(u_old)


            if verbose:
                print('Iteration: ', it, 'relative variation: ', torch.norm(u - u_old)/torch.norm(u_old))

                print('Cost function: ', loss[it])

            
            if rel[it] < self.tol:
                print('Iteration: ', it, 'relative variation: ', torch.norm(u - u_old)/torch.norm(u_old))
                print(f'Converged after {it+1} iterations.')
                break

        if return_loss:
            return u, loss, rel
        else:
            return u
    

