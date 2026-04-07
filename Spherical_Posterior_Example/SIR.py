"""
@author: Mohammad Al-Jarrah
"""

import numpy as np
import time

def SIR(Y,X0,A,h,t,Noise):
    """
    Sequential importance resampling (SIR) filter function according to Algorithm 2 in 
    [Al-Jarrah, M., Jin, N., Hosseini, B. and Taghvaei, A., 2024, July. 
     Nonlinear Filtering with Brenier Optimal Transport Maps. 
     In International Conference on Machine Learning (pp. 813-839). PMLR.]
    
    where:
        
    Y     : True observations with shape (NUM_SIM x T x dy),
    X0    : Initial particles with shape (NUM_SIM x L x N),
    A     : Deterministic dynamic model (without noise),
    h     : Deterministic observation model (without noise),
    t     : Time vector (e.g., t = 0.0, dt, 2*dt, ..., tf),
    Noise : A list [sigma, gamma] defining the noise levels in the dynamics (sigma) and observations (gamma).
    """

    # Determine the dimensions from X0:
    # X0 has shape (NUM_SIM x L x N) where NUM_SIM is the number of simulations,
    # L is the state dimension, and N is the number of particles.
    NUM_SIM = X0.shape[0]
    L = X0.shape[1]
    N = X0.shape[2]
    
    # Extract the dimensions from Y:
    # Y has shape (NUM_SIM x T x dy) where T is the number of time steps,
    # and dy is the observation dimension.
    T = Y.shape[1]
    dy = Y.shape[2]

    # Unpack the noise levels:
    # sigma corresponds to noise in the hidden state,
    # gamma corresponds to noise in the observation.
    sigma = Noise[0]
    gamma = Noise[1]
    
    # Calculate the time step based on the time vector.
    tau = t[1] - t[0]
    
    start_time = time.time()
    # Initialize the output array for the EnKF estimations.
    # Its shape is (NUM_SIM x T x N x L) to store results for each simulation, time step, particle, and state dimension.
    x_SIR = np.zeros((NUM_SIM, T, N, L))

    # Create a random generator instance for the resampling step.
    rng = np.random.default_rng()
    # Loop over each simulation.
    for k in range(NUM_SIM):
        # Retrieve the observations for the current simulation.
        y = Y[k,]
        
        # Set the initial condition for the current simulation (transposed to match dimensions).
        x_SIR[k, 0] = X0[k,].T 
        
        for i in range(T-1):
            # Generate process noise for all particles at the current time step (shape: N x L).
            x_noise = sigma * np.random.multivariate_normal(np.zeros(L),  np.eye(L), N)

            # Propagate particles using the model A with added noise.
            x_SIR[k,i+1,] = A(x_SIR[k,i,].T,t[i]).T + x_noise
            
            # Calculate the weight for each particle based on its observation likelihood P(Y|X^i).
            W = np.sum((y[i+1,].T - h(x_SIR[k,i+1,].T).T)*(y[i+1,].T - h(x_SIR[k,i+1,].T).T),axis=1)/(2*gamma*gamma)
        
            # Adjust weights by subtracting the minimum value (for numerical stability).
            W = W - np.min(W)
            W = np.exp(-W).T
            # Normalize the weights so that they sum to 1.
            W = W/np.sum(W)
            
            # Resample the particles based on the normalized weights.
            index = rng.choice(np.arange(N), N, p = W)
            # Reassign the resampled particles to the current time step.
            x_SIR[k,i+1,] = x_SIR[k,i+1,index,:]
            
    print("--- SIR time : %s seconds ---" % (time.time() - start_time))
    # Rearrange the dimensions of the output to (NUM_SIM x T x L x N) for convenient plotting.
    return x_SIR.transpose(0, 1, 3, 2)