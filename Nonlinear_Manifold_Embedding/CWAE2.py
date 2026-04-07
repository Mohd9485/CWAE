"""
@author: jarrah
"""

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def CWAE2(Y,X0,A,h,t,Noise,parameters):
    """
    CWAE variant 2 (CWAE2) filter.

    CWAE2 uses the latent encoding ẑ_y as input to the x-encoder, and conditions
    the decoder on the decoded ambient observation ŷ = GY(ẑ_y) rather than on ẑ_y:
        z_y  = ΦY(y),   z_xy = ΦX(ẑ_y, x),   x̂ = GX(ŷ, z_xy),   ŷ = GY(z_y)
    At inference, the decoder receives raw y_true directly — encoder_y is bypassed:
        x̂ = GX(y_true, z_xy),   z_xy ~ N(0, I)
    This differs from CWAE1 (where ΦX sees raw y) and CWAE3 (where GX conditions
    on z_y instead of the decoded ŷ).

    Parameters
    ----------
    Y     : True observations with shape (NUM_SIM x T x dy x 1).
    X0    : Initial particles with shape (NUM_SIM x L x N).
    A     : Deterministic dynamic model (without noise).
    h     : Deterministic observation model (without noise).
    t     : Time vector (e.g., t = 0.0, dt, 2*dt, ..., tf).
    Noise : List [sigma, gamma] — process noise std and observation noise std.
    parameters : Dictionary of hyperparameters for the neural network components.
    """

    # X0 has shape (NUM_SIM x L x N).
    NUM_SIM = X0.shape[0]
    L = X0.shape[1]
    N = X0.shape[2]

    # Y has shape (NUM_SIM x T x dy).
    T = Y.shape[1]
    dy = Y.shape[2]

    INPUT_DIM = [L, dy]

    # sigma: process noise std; gamma: observation noise std.
    sigma = Noise[0]
    gamma = Noise[1]

    # Time step size derived from the time vector.
    tau = t[1] - t[0]

    # Unpack neural network hyperparameters.
    normalization = parameters['normalization']
    latent_dims = parameters['latent_dims']
    NUM_NEURON = parameters['NUM_NEURON']
    BATCH_SIZE = parameters['BATCH_SIZE']
    LearningRate = parameters['LearningRate']
    ITERATION = parameters['ITERATION']
    Final_Number_ITERATION = parameters['Final_Number_ITERATION']
    num_resblocks = parameters['num_resblocks']
    lamb = parameters['lamb']
    n_critic = parameters['n_critic']

    # Use CPU; preferred for low-dimensional problems and Apple M-chip machines.
    device = torch.device('cpu')

    class ResidualBlock(nn.Module):
        def __init__(self, hidden_dim, activation):
            super(ResidualBlock, self).__init__()
            # Each block consists of two linear layers with a skip connection.
            self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
            self.activation = activation

        def forward(self, x):
            identity = x  # save input for skip connection
            out = self.linear1(x)
            out = self.activation(out)
            out = self.linear2(out)
            # Add skip connection and apply activation.
            out = self.activation(out + identity)
            return out

    # Encoder that maps observation y to a latent code z_y.
    class Encoder_Y(nn.Module):
        def __init__(self, input_dim, latent_dims, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the x dimension and
                                   input_dim[1] is the y dimension.
                latent_dims (tuple): A tuple where latent_dims[0] is the latent dimension for x
                                     and latent_dims[1] is the latent dimension for y.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(Encoder_Y, self).__init__()
            self.activation = nn.ReLU()

            # Input layer: transforms the input y to the hidden dimension.
            self.layer_input = nn.Linear(input_dim[1], hidden_dim, bias=False)

            # Stack of residual blocks.
            self.resblocks = nn.ModuleList([
                ResidualBlock(hidden_dim, self.activation) for _ in range(num_resblocks)
            ])

            # Output layer: maps from hidden dimension to the y latent dimension.
            self.layer_out = nn.Linear(hidden_dim, latent_dims[1], bias=False)

        def forward(self, y):
            inp = self.layer_input(y)
            out = inp
            for block in self.resblocks:
                out = block(out)
            out = self.activation(out)
            out = self.layer_out(out)
            return out

    # Encoder that maps (x, z_y) to a latent code z_xy.
    # Unlike CWAE1, this encoder chains off the encoded observation z_y rather than raw y.
    class Encoder_XY(nn.Module):
        def __init__(self, input_dim, latent_dims, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the x dimension and
                                   input_dim[1] is the y dimension.
                latent_dims (tuple): A tuple where latent_dims[0] is the latent dimension for x
                                     and latent_dims[1] is the latent dimension for y.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(Encoder_XY, self).__init__()
            self.activation = nn.ReLU()

            # Input layer: transforms concatenated (x, z_y) to the hidden dimension.
            self.layer_input = nn.Linear(input_dim[0] + latent_dims[1], hidden_dim, bias=False)

            # Stack of residual blocks.
            self.resblocks = nn.ModuleList([
                ResidualBlock(hidden_dim, self.activation) for _ in range(num_resblocks)
            ])

            # Output layer: maps from hidden dimension to the x latent dimension.
            self.layer_out = nn.Linear(hidden_dim, latent_dims[0], bias=False)

        def forward(self, x, zy):
            # Concatenate x and z_y along the feature dimension.
            inp = torch.concat((x, zy), dim=1)
            inp = self.layer_input(inp)
            out = inp
            for block in self.resblocks:
                out = block(out)
            out = self.activation(out)
            out = self.layer_out(out)
            return out

    # Decoder that reconstructs x from the latent code z_xy conditioned on raw y.
    # Unlike CWAE1, this decoder takes raw y directly instead of z_y.
    class Decoder_XY(nn.Module):
        def __init__(self, input_dim, latent_dims, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the x dimension and
                                   input_dim[1] is the y dimension.
                latent_dims (tuple): A tuple where latent_dims[0] is the latent dimension for x
                                     and latent_dims[1] is the latent dimension for y.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(Decoder_XY, self).__init__()
            self.activation = nn.ReLU()

            # Input layer: transforms concatenated (z_xy, y_raw) to the hidden dimension.
            self.layer_input = nn.Linear(latent_dims[0] + input_dim[1], hidden_dim, bias=False)

            # Stack of residual blocks.
            self.resblocks = nn.ModuleList([
                ResidualBlock(hidden_dim, self.activation) for _ in range(num_resblocks)
            ])

            # Output layer: maps from hidden dimension to the x state dimension.
            self.layer_out = nn.Linear(hidden_dim, input_dim[0], bias=False)

        def forward(self, z_xy, y):
            # Concatenate z_xy and y_raw (or decoded y) along the feature dimension.
            inp = torch.concat((z_xy, y), dim=1)
            inp = self.layer_input(inp)
            out = inp
            for block in self.resblocks:
                out = block(out)
            out = self.activation(out)
            out = self.layer_out(out)
            return out

    # Decoder that reconstructs y from the observation latent code z_y.
    class Decoder_Y(nn.Module):
        def __init__(self, input_dim, latent_dims, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the x dimension and
                                   input_dim[1] is the y dimension.
                latent_dims (tuple): A tuple where latent_dims[0] is the latent dimension for x
                                     and latent_dims[1] is the latent dimension for y.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(Decoder_Y, self).__init__()
            self.activation = nn.ReLU()

            # Input layer: transforms z_y to the hidden dimension.
            self.layer_input = nn.Linear(latent_dims[1], hidden_dim, bias=False)

            # Stack of residual blocks.
            self.resblocks = nn.ModuleList([
                ResidualBlock(hidden_dim, self.activation) for _ in range(num_resblocks)
            ])

            # Output layer: maps from hidden dimension to the y observation dimension.
            self.layer_out = nn.Linear(hidden_dim, input_dim[1], bias=False)

        def forward(self, z_y):
            inp = self.layer_input(z_y)
            out = inp
            for block in self.resblocks:
                out = block(out)
            out = self.activation(out)
            out = self.layer_out(out)
            return out

    # Discriminator that operates in the joint latent space to enforce the WAE prior.
    class LatentDiscriminator(nn.Module):
        def __init__(self, input_dim, latent_dims, hidden_dim, num_resblocks=2):
            """
            Parameters:
                input_dim (tuple): A tuple where input_dim[0] is the x dimension and
                                   input_dim[1] is the y dimension.
                latent_dims (tuple): A tuple where latent_dims[0] is the latent dimension for x
                                     and latent_dims[1] is the latent dimension for y.
                hidden_dim (int): The number of neurons in the hidden layers.
                num_resblocks (int): Number of residual blocks to use.
            """
            super(LatentDiscriminator, self).__init__()
            self.activation = nn.ReLU()

            # Input layer: transforms the joint latent code to the hidden dimension.
            self.layer_input = nn.Linear(latent_dims[0] + latent_dims[1], hidden_dim, bias=False)

            # Stack of residual blocks.
            self.resblocks = nn.ModuleList([
                ResidualBlock(hidden_dim, self.activation) for _ in range(num_resblocks)
            ])

            # Output layer: produces a scalar logit for real/fake classification.
            self.layer_out = nn.Linear(hidden_dim, 1, bias=False)

        def forward(self, z):
            inp = self.layer_input(z)
            out = inp
            for block in self.resblocks:
                out = block(out)
            out = self.activation(out)
            out = self.layer_out(out)
            return out

    # Binary cross-entropy loss in logit form (numerically stable).
    _bce_logits = nn.BCEWithLogitsLoss()

    def discriminator_loss(D, z_real, z_fake):
        # z_fake must be detached from the encoders before calling.
        real_logits = D(z_real)
        fake_logits = D(z_fake)
        real_targets = torch.ones_like(real_logits)
        fake_targets = torch.zeros_like(fake_logits)
        return _bce_logits(real_logits, real_targets) + _bce_logits(fake_logits, fake_targets)

    def generator_adv_loss(D, z_fake):
        # Encoder adversarial loss: push D(z_fake) toward 1 (fool the discriminator).
        fake_logits = D(z_fake)
        real_targets = torch.ones_like(fake_logits)
        return _bce_logits(fake_logits, real_targets)

    def WAE_loss(z, z_hat, x, x_hat, y, y_hat, lamb, D):
        # Total WAE loss: reconstruction term + lambda-weighted adversarial regularization.
        recon = ((x - x_hat) * (x - x_hat)).sum(axis=1).mean() + ((y - y_hat) * (y - y_hat)).sum(axis=1).mean()
        adv = generator_adv_loss(D, z_hat)
        return recon + lamb * adv

    def init_weights(m):
        # Xavier uniform initialization for linear layers; bias set to a small positive constant.
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.1)

    def train(encoder_y, encoder_xy, decoder_y, decoder_xy, discriminator, X_Train, Y_Train, iterations, learning_rate, ts, Ts, batch_size, k, K, latent_dims, lamb, n_critic):
        # Set all networks to training mode.
        encoder_y.train()
        encoder_xy.train()
        decoder_xy.train()
        decoder_y.train()
        discriminator.train()

        # Separate Adam optimizers for each network component.
        optimizer_encoder_y = torch.optim.Adam(encoder_y.parameters(), lr=learning_rate[0])
        optimizer_encoder_xy = torch.optim.Adam(encoder_xy.parameters(), lr=learning_rate[1])
        optimizer_decoder_y = torch.optim.Adam(decoder_y.parameters(), lr=learning_rate[2])
        optimizer_decoder_xy = torch.optim.Adam(decoder_xy.parameters(), lr=learning_rate[3])
        optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=learning_rate[4])

        # Exponential learning rate decay applied after each iteration.
        scheduler_encoder_y = ExponentialLR(optimizer_encoder_y, gamma=0.999)
        scheduler_encoder_xy = ExponentialLR(optimizer_encoder_xy, gamma=0.999)
        scheduler_decoder_y = ExponentialLR(optimizer_decoder_y, gamma=0.999)
        scheduler_decoder_xy = ExponentialLR(optimizer_decoder_xy, gamma=0.999)
        scheduler_discriminator = ExponentialLR(optimizer_discriminator, gamma=0.999)

        for i in range(iterations):
            # Sample a random mini-batch without replacement.
            idx = torch.randperm(X_Train.shape[0])[:batch_size]
            X_train = X_Train[idx].clone().detach()
            Y_train = Y_Train[idx].clone().detach()
            # Sample from the prior distribution in joint latent space.
            Z_train = torch.randn((X_train.shape[0], latent_dims[0] + latent_dims[1]), device=device)

            # ---- Discriminator update ----
            # Freeze encoders/decoders; only update the discriminator.
            encoder_y.requires_grad_(False)
            encoder_xy.requires_grad_(False)
            decoder_y.requires_grad_(False)
            decoder_xy.requires_grad_(False)
            discriminator.requires_grad_(True)

            for j in range(n_critic):
                with torch.no_grad():
                    z_y_hat = encoder_y.forward(Y_train)
                    z_xy_hat = encoder_xy.forward(X_train, z_y_hat)
                    z_fake = torch.cat((z_xy_hat, z_y_hat), dim=1)

                loss_D = discriminator_loss(discriminator, z_real=Z_train, z_fake=z_fake)
                optimizer_discriminator.zero_grad()
                loss_D.backward()
                optimizer_discriminator.step()

            # ---- Encoder update ----
            # Freeze decoders and discriminator; only update encoders.
            encoder_y.requires_grad_(True)
            encoder_xy.requires_grad_(True)
            decoder_y.requires_grad_(False)
            decoder_xy.requires_grad_(False)
            discriminator.requires_grad_(False)

            z_y_hat = encoder_y.forward(Y_train)
            z_xy_hat = encoder_xy.forward(X_train, z_y_hat)
            Y_hat = decoder_y.forward(z_y_hat)
            # Decoder_XY conditions on the reconstructed y (Y_hat) rather than z_y.
            X_hat = decoder_xy.forward(z_xy_hat, Y_hat)
            z_hat = torch.cat((z_xy_hat, z_y_hat), dim=1)

            loss_E = WAE_loss(Z_train, z_hat, X_train, X_hat, Y_train, Y_hat, lamb, discriminator)

            optimizer_encoder_y.zero_grad()
            optimizer_encoder_xy.zero_grad()
            loss_E.backward()
            optimizer_encoder_y.step()
            optimizer_encoder_xy.step()

            # ---- Decoder update ----
            # Freeze encoders and discriminator; only update decoders.
            encoder_y.requires_grad_(False)
            encoder_xy.requires_grad_(False)
            decoder_y.requires_grad_(True)
            decoder_xy.requires_grad_(True)
            discriminator.requires_grad_(False)

            with torch.no_grad():
                z_y_hat = encoder_y.forward(Y_train)
                z_xy_hat = encoder_xy.forward(X_train, z_y_hat)
            Y_hat = decoder_y.forward(z_y_hat)
            X_hat = decoder_xy.forward(z_xy_hat, Y_hat)

            # Pure reconstruction loss for decoder training (no adversarial term).
            loss_Dec = ((X_train - X_hat) * (X_train - X_hat)).sum(axis=1).mean() + ((Y_train - Y_hat) * (Y_train - Y_hat)).sum(axis=1).mean()

            optimizer_decoder_y.zero_grad()
            optimizer_decoder_xy.zero_grad()
            loss_Dec.backward()
            optimizer_decoder_y.step()
            optimizer_decoder_xy.step()

            # Advance all LR schedulers.
            scheduler_encoder_y.step()
            scheduler_encoder_xy.step()
            scheduler_decoder_y.step()
            scheduler_decoder_xy.step()
            scheduler_discriminator.step()

            # Log total WAE loss at every 1000 iterations and at the final iteration.
            if (i + 1) == iterations or (i + 1) % 1000 == 0:
                encoder_y.eval()
                encoder_xy.eval()
                decoder_y.eval()
                decoder_xy.eval()
                discriminator.eval()
                with torch.no_grad():
                    z_y_hat = encoder_y.forward(Y_Train)
                    z_xy_hat = encoder_xy.forward(X_Train, z_y_hat)
                    Y_hat = decoder_y.forward(z_y_hat)
                    X_hat = decoder_xy.forward(z_xy_hat, Y_hat)
                    z_hat = torch.cat((z_xy_hat, z_y_hat), dim=1)
                    Z_full = torch.randn((X_Train.shape[0], latent_dims[0] + latent_dims[1]), device=device)
                    loss = WAE_loss(Z_full, z_hat, X_Train, X_hat, Y_Train, Y_hat, lamb, discriminator)
                    print("Simu#%d/%d, Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" %
                          (k + 1, K, ts, Ts - 1, i + 1, iterations, loss.item()))

    start_time = time.time()

    # Output array for filtered particles; shape: (NUM_SIM x T x N x L).
    X_CWAE = torch.zeros((NUM_SIM, T, N, L), device=device, dtype=torch.float32)

    for k in range(NUM_SIM):
        # Observations for this simulation run.
        y = Y[k,]

        # Set initial particles from X0 (transposed to match (N x L) layout).
        X_CWAE[k, 0,] = torch.from_numpy(X0[k,].T).to(torch.float32).to(device)

        ITERS = ITERATION
        LR = LearningRate

        # Instantiate all network components for this simulation.
        encoder_y = Encoder_Y(INPUT_DIM, latent_dims, NUM_NEURON[0], num_resblocks[0])
        encoder_xy = Encoder_XY(INPUT_DIM, latent_dims, NUM_NEURON[1], num_resblocks[1])
        decoder_y = Decoder_Y(INPUT_DIM, latent_dims, NUM_NEURON[2], num_resblocks[2])
        decoder_xy = Decoder_XY(INPUT_DIM, latent_dims, NUM_NEURON[3], num_resblocks[3])
        discriminator = LatentDiscriminator(INPUT_DIM, latent_dims, NUM_NEURON[4], num_resblocks[4])

        encoder_y.to(device)
        encoder_xy.to(device)
        decoder_y.to(device)
        decoder_xy.to(device)
        discriminator.to(device)

        # Apply Xavier weight initialization to all networks.
        encoder_y.apply(init_weights)
        encoder_xy.apply(init_weights)
        decoder_y.apply(init_weights)
        decoder_xy.apply(init_weights)
        discriminator.apply(init_weights)

        for i in range(T - 1):
            # Propagate particles through the dynamics model with process noise.
            x_noise = torch.distributions.MultivariateNormal(torch.zeros(L), covariance_matrix=torch.eye(L))
            X1 = A(X_CWAE[k, i,].T, t[i]).T + sigma * x_noise.sample((N,)).to(device)

            # Compute predicted observations for each particle with observation noise.
            y_noise = torch.distributions.MultivariateNormal(torch.zeros(dy), covariance_matrix=torch.eye(dy))
            Y1 = h(X1.T).T + gamma * y_noise.sample((N,)).to(device)

            # Normalize observations before training if requested (only Y is scaled; X is left in its original scale).
            if normalization == 'Standard':
                scaler_Y = StandardScaler()
                Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()), device=device, dtype=torch.float32)
            elif normalization == 'MinMax':
                scaler_Y = MinMaxScaler()
                Y1 = torch.tensor(scaler_Y.fit_transform(Y1.cpu()), device=device, dtype=torch.float32)

            # Train the CWAE networks on the current forecast ensemble.
            train(encoder_y, encoder_xy, decoder_y, decoder_xy, discriminator, X1, Y1, ITERS, LR, i + 1, T, BATCH_SIZE, k, NUM_SIM, latent_dims, lamb, n_critic)

            # Halve the training iterations after warm-up to speed up later time steps.
            if ITERS > Final_Number_ITERATION and i % 1 == 0 and i >= 5:
                ITERS = int(ITERS / 2)

            # Prepare the true observation at the next time step for conditioning.
            Y1_true = y[i + 1, :]
            Y1_true = np.asarray(Y1_true).reshape(1, dy)
            if normalization == 'Standard' or normalization == 'MinMax':
                Y1_true = scaler_Y.transform(Y1_true)
            Y1_true = torch.from_numpy(Y1_true).to(torch.float32).to(device)

            # Draw new particles by decoding random latent codes conditioned on the true observation.
            # Decoder_XY takes raw y directly — no encoder_y pass needed at inference.
            with torch.no_grad():
                Z_xy_hat = torch.randn((N, latent_dims[0]), device=device)
                X_mapped = decoder_xy.forward(Z_xy_hat, Y1_true.repeat(N, 1))

            # Store the updated particle positions for this time step.
            X_CWAE[k, i + 1,] = X_mapped.detach()

    print("--- CWAE time : %s seconds ---" % (time.time() - start_time))

    # Transpose output to (NUM_SIM x T x L x N) for consistent downstream use.
    return X_CWAE.cpu().numpy().transpose(0, 1, 3, 2)
