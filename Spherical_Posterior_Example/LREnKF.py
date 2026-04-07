"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time


def LREnKF(Y, X0, A, h, grad_h, t, Noise, r_dims, alpha=None, SIGMA=1e-6):
    """
    Low-Rank Ensemble Kalman Filter (LREnKF) according to Algorithm 1 in
    [Le Provost, M., Baptista, R., Marzouk, Y. and Eldredge, J.D., 2022.
     A low-rank ensemble Kalman filter for elliptic observations.
     Proceedings of the Royal Society A.]

    where:

    Y      : True observations with shape (NUM_SIM x T x dy),
    X0     : Initial particles with shape (NUM_SIM x L x N),
    A      : Deterministic dynamic model (without noise),
    h      : Deterministic observation model (without noise); takes state matrix
             of shape (L x N) and returns observation matrix of shape (dy x N),
    grad_h : Jacobian of the observation model h; callable taking a single state
             vector of shape (L,) and returning the Jacobian of shape (dy x L),
    t      : Time vector (e.g., t = 0.0, dt, 2*dt, ..., tf),
    Noise  : A list [sigma, gamma] defining the noise levels in the dynamics (sigma)
             and observations (gamma), with covariances sigma^2 * I and gamma^2 * I,
    r_dims : [rX, rY] Fixed rank of the state-space projection rX and 
             observation-space projection rY
    alpha  : Optional float in (0, 1). If provided, ranks rX and rY are chosen
             adaptively at each analysis step to capture at least a fraction alpha of
             the cumulative energy of C_X and C_Y, respectively. Overrides rX and rY,
    SIGMA  : A small positive constant (nugget) added to the linear system for
             numerical stability.

    The analysis step implements the factorized Kalman gain:
        T(y, x) = x + Sigma_X^{1/2} V_rX (A_Xtilde A_Ztilde^T) b_tilde,
    where b_tilde solves (A_Ztilde A_Ztilde^T + A_Etilde A_Etilde^T) b_tilde
                       = U_rY^T Sigma_E^{-1/2} (y* 1^T - EZ - EE).

    The state Gramian C_X (L x L) and observation Gramian C_Y (dy x dy) are:
        C_X = 1/(N-1) sum_j (Sigma_E^{-1/2} grad_h(x^j) Sigma_X^{1/2})^T
                            (Sigma_E^{-1/2} grad_h(x^j) Sigma_X^{1/2}),
        C_Y = 1/(N-1) sum_j (Sigma_E^{-1/2} grad_h(x^j) Sigma_X^{1/2})
                            (Sigma_E^{-1/2} grad_h(x^j) Sigma_X^{1/2})^T.
    With Sigma_X = sigma^2 I and Sigma_E = gamma^2 I, the whitened Jacobian
    simplifies to G_j = (sigma / gamma) * grad_h(x^j).
    """

    # Determine the dimensions from X0:
    # X0 has shape (NUM_SIM x L x N) where NUM_SIM is the number of simulations,
    # L is the state dimension, and N is the number of particles.
    NUM_SIM = X0.shape[0]
    L = X0.shape[1]
    N = X0.shape[2]

    # Extract the dimensions from Y:
    # Y has shape (NUM_SIM x T x dy) where T is the number of time steps
    # and dy is the observation dimension.
    T = Y.shape[1]
    dy = Y.shape[2]

    # Unpack the noise levels:
    # sigma corresponds to noise in the hidden state (Sigma_X = sigma^2 * I),
    # gamma corresponds to noise in the observation (Sigma_E = gamma^2 * I).
    sigma = Noise[0]
    gamma = Noise[1]

    # Precompute the whitening ratio used in the Gramian: (sigma / gamma).
    ratio = sigma / gamma
    
    # rX : Fixed rank of the state-space projection (overridden if alpha is set),
    rX = r_dims[0]
    # rY : Fixed rank of the observation-space projection (overridden if alpha is set),
    rY = r_dims[1]

    start_time = time.time()

    # Initialize the output array for LREnKF estimations.
    # Shape: (NUM_SIM x T x N x L).
    X_LREnKF = np.zeros((NUM_SIM, T, N, L))

    # Loop over each simulation.
    for k in range(NUM_SIM):
        # Retrieve the observations for the current simulation.
        y = Y[k,]

        # Set the initial condition for the current simulation (transposed to match dimensions).
        X_LREnKF[k, 0] = X0[k,].T

        # Time stepping for the filter (excluding the final time step).
        for i in range(T - 1):

            # --- Forecast step (identical to standard EnKF) ---
            # Generate process noise for all particles (shape: N x L) and propagate.
            x_noise = sigma * np.random.multivariate_normal(np.zeros(L), np.eye(L), N)
            x_hat = A(X_LREnKF[k, i,].T, t[i]).T + x_noise   # shape: N x L
            # print(x_hat.shape)
            
            # --- Analysis step (LREnKF, Algorithm 1) ---

            # Step 1: Evaluate the observation operator for each forecast particle
            # (lines 1-3 in Algorithm 1).
            # h takes (L x N), returns (dy x N); transpose to get (N x dy).
            Z = h(x_hat.T).T                                   # shape: N x dy

            # Sample observation noise for each particle (shape: N x dy).
            E_noise = gamma * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), N)

            # Step 2: Compute Monte-Carlo approximations of C_X (L x L) and C_Y (dy x dy)
            # using the whitened Jacobians G_j = (sigma/gamma) * grad_h(x^j)
            # (lines 4-5 in Algorithm 1).
            C_X = np.zeros((L, L))
            C_Y = np.zeros((dy, dy))
            for j in range(N):
                Jj = grad_h(x_hat[j,])    # Jacobian at particle j, shape: dy x L
                Gj = ratio * Jj            # Whitened Jacobian, shape: dy x L
                C_X += Gj.T @ Gj           # Contribution to C_X
                C_Y += Gj @ Gj.T           # Contribution to C_Y
            C_X /= (N - 1)
            C_Y /= (N - 1)

            # Step 3: Low-rank eigendecomposition of C_X and C_Y
            # (lines 6-7 in Algorithm 1).
            # np.linalg.eigh returns eigenvalues in ascending order; reverse for descending.
            eigvals_X, eigvecs_X = np.linalg.eigh(C_X)
            eigvals_X = eigvals_X[::-1]
            eigvecs_X = eigvecs_X[:, ::-1]   # Columns are eigenvectors, now sorted descending

            eigvals_Y, eigvecs_Y = np.linalg.eigh(C_Y)
            eigvals_Y = eigvals_Y[::-1]
            eigvecs_Y = eigvecs_Y[:, ::-1]

            # Determine ranks adaptively from the cumulative energy threshold alpha,
            # or use the fixed ranks rX, rY provided by the caller.
            if alpha is not None:
                # Set rX_curr = smallest r such that sum(lambda^2_{X,1:r}) >= alpha * sum(lambda^2_X).
                total_X = np.sum(np.maximum(eigvals_X, 0.0))
                cumsum_X = np.cumsum(np.maximum(eigvals_X, 0.0))
                rX_curr = int(np.searchsorted(cumsum_X, alpha * total_X) + 1)
                rX_curr = min(rX_curr, L)

                total_Y = np.sum(np.maximum(eigvals_Y, 0.0))
                cumsum_Y = np.cumsum(np.maximum(eigvals_Y, 0.0))
                rY_curr = int(np.searchsorted(cumsum_Y, alpha * total_Y) + 1)
                rY_curr = min(rY_curr, dy)
            else:
                rX_curr = min(rX, L)
                rY_curr = min(rY, dy)

            # Leading eigenvectors define the informative subspaces.
            V_rX = eigvecs_X[:, :rX_curr]   # State-space basis,       shape: L  x rX_curr
            U_rY = eigvecs_Y[:, :rY_curr]   # Observation-space basis, shape: dy x rY_curr

            # Step 4: Compute empirical means of the forecast ensemble and observations.
            mu_X = x_hat.mean(axis=0)        # shape: L
            mu_Z = Z.mean(axis=0)            # shape: dy
            mu_E = E_noise.mean(axis=0)      # shape: dy

            # Step 5: Whiten and project particles, observations, and observation noise
            # into the informative subspaces (lines 8-11 in Algorithm 1).
            # x_tilde^j = V_rX^T * Sigma_X^{-1/2} * (x^j - mu_X) = V_rX^T * (x^j-mu_X)/sigma
            # z_tilde^j = U_rY^T * Sigma_E^{-1/2} * (z^j - mu_Z) = U_rY^T * (z^j-mu_Z)/gamma
            # e_tilde^j = U_rY^T * Sigma_E^{-1/2} * (e^j - mu_E) = U_rY^T * (e^j-mu_E)/gamma
            X_tilde = ((x_hat - mu_X) / sigma) @ V_rX    # shape: N x rX_curr
            Z_tilde = ((Z     - mu_Z) / gamma) @ U_rY    # shape: N x rY_curr
            E_tilde = ((E_noise - mu_E) / gamma) @ U_rY  # shape: N x rY_curr

            # Step 6: Form the perturbation matrices in the projected spaces
            # (lines 12-15 in Algorithm 1).
            # A_Xtilde[:,j] = (x_tilde^j - mu_Xtilde) / sqrt(N-1),  shape: rX x N
            # A_Ztilde[:,j] = (z_tilde^j - mu_Ztilde) / sqrt(N-1),  shape: rY x N
            # A_Etilde[:,j] = (e_tilde^j - mu_Etilde) / sqrt(N-1),  shape: rY x N
            mu_Xtilde = X_tilde.mean(axis=0, keepdims=True)
            mu_Ztilde = Z_tilde.mean(axis=0, keepdims=True)
            mu_Etilde = E_tilde.mean(axis=0, keepdims=True)

            A_Xtilde = (X_tilde - mu_Xtilde).T / np.sqrt(N - 1)   # shape: rX x N
            A_Ztilde = (Z_tilde - mu_Ztilde).T / np.sqrt(N - 1)   # shape: rY x N
            A_Etilde = (E_tilde - mu_Etilde).T / np.sqrt(N - 1)   # shape: rY x N

            # Step 7: Solve the linear system for b_tilde (rY x N) via representers
            # (line 16 in Algorithm 1):
            # (A_Ztilde A_Ztilde^T + A_Etilde A_Etilde^T) b_tilde
            #     = U_rY^T Sigma_E^{-1/2} (y*_{i+1} 1^T - EZ - EE)
            # where EZ (dy x N) and EE (dy x N) are the ensemble matrices of z^j and e^j.
            lhs = A_Ztilde @ A_Ztilde.T + A_Etilde @ A_Etilde.T   # shape: rY x rY
            lhs += np.eye(rY_curr) * SIGMA                          # nugget for stability

            EZ = Z.T                   # Full observation ensemble matrix,      shape: dy x N
            EE = E_noise.T             # Full observation noise ensemble matrix, shape: dy x N

            rhs_full = y[i + 1, :] * np.ones(N) - EZ - EE   # shape: dy x N
            rhs = (U_rY.T / gamma) @ rhs_full                       # shape: rY x N

            b_tilde = np.linalg.solve(lhs, rhs)                     # shape: rY x N

            # Step 8: Lift the analysis correction back to the original state space
            # (lines 17-18 in Algorithm 1):
            # x^j_a = x^j + Sigma_X^{1/2} V_rX (A_Xtilde A_Ztilde^T) b_tilde[:,j]
            #       = x^j + sigma * V_rX @ (A_Xtilde @ A_Ztilde^T) @ b_tilde[:,j]
            gain = sigma * V_rX @ (A_Xtilde @ A_Ztilde.T)   # shape: L x rY
            update = (gain @ b_tilde).T                       # shape: N x L
            # print("update : ",gain.shape, b_tilde.shape)
            X_LREnKF[k, i + 1, :, :] = x_hat + update

    print("--- LREnKF running time : %s seconds ---" % (time.time() - start_time))
    # Rearrange the dimensions of the output to (NUM_SIM x T x L x N) for convenient plotting.
    return X_LREnKF.transpose(0, 1, 3, 2)