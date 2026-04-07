"""
@author: Mohammad Al-Jarrah
"""

# ── Raw SMAC-tuned hyperparameter values for the sphere experiment ─────────── #
# Keys follow the SMAC config space in smac_tune_CWAE_sphere.py:
#   nns1–5  : hidden-layer width multipliers (actual width = nns * 16)
#   nbs1–5  : number of residual blocks per network (5 networks: ΦY, ΦX, GY, GX, D)
#   lr1–5   : per-network learning rates
#   batch_size : mini-batch size multiplier (actual size = batch_size * 16)
#   lamb    : WAE regularisation weight balancing reconstruction vs. latent penalty
#   n_critic : discriminator update steps per generator step

CWAE1_VALUES = {
        'batch_size': 4,
        'lamb': 1.2925032791965,
        'lr1': 0.0008170723266,
        'lr2': 0.000406541925,
        'lr3': 6.47460293e-05,
        'lr4': 0.003050878356,
        'lr5': 0.0001154771163,
        'n_critic': 12,
        'nbs1': 2,
        'nbs2': 3,
        'nbs3': 3,
        'nbs4': 2,
        'nbs5': 2,
        'nns1': 3,
        'nns2': 4,
        'nns3': 2,
        'nns4': 8,
        'nns5': 6,
    }


CWAE2_VALUES = {
        'batch_size': 4,
        'lamb': 1.2925032791965,
        'lr1': 0.0008170723266,
        'lr2': 0.000406541925,
        'lr3': 6.47460293e-05,
        'lr4': 0.003050878356,
        'lr5': 0.0001154771163,
        'n_critic': 12,
        'nbs1': 2,
        'nbs2': 3,
        'nbs3': 3,
        'nbs4': 2,
        'nbs5': 2,
        'nns1': 3,
        'nns2': 4,
        'nns3': 2,
        'nns4': 8,
        'nns5': 6,

    }

CWAE3_VALUES = {
        'batch_size': 12,
        'lamb': 0.1808507626172,
        'lr1': 7.92130658e-05,
        'lr2': 0.008027111071,
        'lr3': 0.0001711245632,
        'lr4': 0.0087275933214,
        'lr5': 0.0002955757829,
        'n_critic': 4,
        'nbs1': 3,
        'nbs2': 1,
        'nbs3': 2,
        'nbs4': 3,
        'nbs5': 2,
        'nns1': 5,
        'nns2': 7,
        'nns3': 3,
        'nns4': 6,
        'nns5': 5,
    }


# ── Parameter builders ─────────────────────────────────────────────────────── #
# Each function converts the raw SMAC values into the dictionary format expected
# by the CWAE filter functions. Integer multipliers are scaled to their true sizes
# here so the filter code never needs to know the encoding convention.

def get_cwae1_parameters(N, latent_dims):
    values = CWAE1_VALUES

    # Scale raw integer multipliers to actual hidden-layer widths and batch size.
    nns_ = [values[f"nns{i}"] for i in range(1, 6)]
    nns_ = [x * 16 for x in nns_]
    nbs_ = [values[f"nbs{i}"] for i in range(1, 6)]
    lr_  = [values[f"lr{i}"]  for i in range(1, 6)]

    parameters_CWAE = {
        "normalization": "Standard",
        "latent_dims": latent_dims,
        "NUM_NEURON": nns_,
        "BATCH_SIZE": int(values["batch_size"])*16,
        "LearningRate": lr_,
        "ITERATION": int(3e3),             
        "Final_Number_ITERATION": int(1e3), 
        "num_resblocks": nbs_,
        "lamb": float(values["lamb"]),
        "n_critic": int(values["n_critic"]),
    }
    return parameters_CWAE

def get_cwae2_parameters(N, latent_dims):
    values = CWAE2_VALUES

    nns_ = [values[f"nns{i}"] for i in range(1, 6)]
    nns_ = [x * 16 for x in nns_]
    nbs_ = [values[f"nbs{i}"] for i in range(1, 6)]
    lr_  = [values[f"lr{i}"]  for i in range(1, 6)]

    parameters_CWAE = {
        "normalization": "Standard",
        "latent_dims": latent_dims,
        "NUM_NEURON": nns_,
        "BATCH_SIZE": int(values["batch_size"])*16,
        "LearningRate": lr_,
        "ITERATION": int(3e3),
        "Final_Number_ITERATION": int(1e3),
        "num_resblocks": nbs_,
        "lamb": float(values["lamb"]),
        "n_critic": int(values["n_critic"]),
    }
    return parameters_CWAE

def get_cwae3_parameters(N, latent_dims):
    values = CWAE3_VALUES

    nns_ = [values[f"nns{i}"] for i in range(1, 6)]
    nns_ = [x * 16 for x in nns_]
    nbs_ = [values[f"nbs{i}"] for i in range(1, 6)]
    lr_  = [values[f"lr{i}"]  for i in range(1, 6)]

    parameters_CWAE = {
        "normalization": "Standard",
        "latent_dims": latent_dims,
        "NUM_NEURON": nns_,
        "BATCH_SIZE": int(values["batch_size"])*16,
        "LearningRate": lr_,
        "ITERATION": int(3e3),
        "Final_Number_ITERATION": int(1e3),
        "num_resblocks": nbs_,
        "lamb": float(values["lamb"]),
        "n_critic": int(values["n_critic"]),
    }
    return parameters_CWAE













