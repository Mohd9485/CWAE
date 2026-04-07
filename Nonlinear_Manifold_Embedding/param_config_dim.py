"""
@author: Mohammad Al-Jarrah
"""
# ── Raw SMAC-tuned hyperparameter values for the sphere experiment ─────────── #
# Keys follow the SMAC config space in smac_tune_CWAE_sphere.py:
#   nns1–5  : hidden-layer width multipliers 
#   nbs1–5  : number of residual blocks per network (5 networks: ΦY, ΦX, GY, GX, D)
#   lr1–5   : per-network learning rates
#   batch_size : mini-batch size multiplier 
#   lamb    : WAE regularisation weight balancing reconstruction vs. latent penalty
#   n_critic : discriminator update steps per generator step


CWAE1_VALUES = {
    "small": {
        'batch_size': 209,
        'iter': 2000,
        'lamb': 0.1444535097495,
        'lr1': 0.0011437607684,
        'lr2': 0.0092644255424,
        'lr3': 0.0028842507935,
        'lr4': 0.0010976034315,
        'lr5': 0.0006801562836,
        'n_critic': 5,
        'nbs1': 1,
        'nbs2': 3,
        'nbs3': 1,
        'nbs4': 1,
        'nbs5': 1,
        'nns1': 60,
        'nns2': 96,
        'nns3': 93,
        'nns4': 66,
        'nns5': 93,
    },
    "large": {
        'batch_size': 4*16,
        'lamb': 9.1545712684591,
        'lr1': 0.0077393936221,
        'lr2': 6.97053236e-05,
        'lr3': 0.0006812830228,
        'lr4': 0.0003481239632,
        'lr5': 1.00910786e-05,
        'n_critic': 9,
        'nbs1': 3,
        'nbs2': 2,
        'nbs3': 3,
        'nbs4': 3,
        'nbs5': 3,
        'nns1': 5*16,
        'nns2': 3*16,
        'nns3': 4*16,
        'nns4': 6*16,
        'nns5': 4*16,
    
    }
}

CWAE2_VALUES = {
    "small": {
        "batch_size": 40,
        "iter": 2000,
        "lamb": 1.8532680339669,
        "lr1": 0.0006881528207,
        "lr2": 0.0020234241865,
        "lr3": 0.0040499874107,
        "lr4": 0.0056614739432,
        "lr5": 0.0001858075174,
        "n_critic": 15,
        "nbs1": 2,
        "nbs2": 3,
        "nbs3": 2,
        "nbs4": 3,
        "nbs5": 2,
        "nns1": 115,
        "nns2": 32,
        "nns3": 96,
        "nns4": 116,
        "nns5": 80,
    },
    "large": {
        'batch_size': 8*16,
        'lamb': 1.8823341007444,
        'lr1': 1.57793857e-05,
        'lr2': 7.9911348e-05,
        'lr3': 0.0095013353517,
        'lr4': 0.0010693083867,
        'lr5': 0.000226071317,
        'n_critic': 5,
        'nbs1': 1,
        'nbs2': 3,
        'nbs3': 1,
        'nbs4': 2,
        'nbs5': 1,
        'nns1': 2*16,
        'nns2': 2*16,
        'nns3': 8*16,
        'nns4': 7*16,
        'nns5': 6*16, 
    }
}
CWAE3_VALUES = {
    "small": {
        'batch_size': 117,
        "iter": 2000,
        'lamb': 9.4578999218123,
        'lr1': 0.0002256105044,
        'lr2': 0.0026916800397,
        'lr3': 0.0003223297291,
        'lr4': 0.0005232531081,
        'lr5': 1.01026019e-05,
        'n_critic': 15,
        'nbs1': 1,
        'nbs2': 1,
        'nbs3': 2,
        'nbs4': 3,
        'nbs5': 3,
        'nns1': 62,
        'nns2': 41,
        'nns3': 118,
        'nns4': 37,
        'nns5': 57,
    },
    "large": {
        'batch_size': 7*16,
        'lamb': 0.3572038452769,
        'lr1': 0.0007398958862,
        'lr2': 2.32998323e-05,
        'lr3': 0.0011373828117,
        'lr4': 0.0099720768306,
        'lr5': 0.0036741953117,
        'n_critic': 13,
        'nbs1': 3,
        'nbs2': 1,
        'nbs3': 1,
        'nbs4': 1,
        'nbs5': 2,
        'nns1': 8*16,
        'nns2': 7*16,
        'nns3': 4*16,
        'nns4': 8*16,
        'nns5': 1*16,
    }
}

# ── Parameter builders ─────────────────────────────────────────────────────── #
# Each function converts the raw SMAC values into the dictionary format expected
# by the CWAE filter functions. Integer multipliers are scaled to their true sizes
# here so the filter code never needs to know the encoding convention.

def get_cwae1_parameters(n, latent_dims):
    # define your rule for small vs large n
    if n <= 5:
        values = CWAE1_VALUES["small"]
    else:
        values = CWAE1_VALUES["large"]

    nns_ = [values[f"nns{i}"] for i in range(1, 6)]
    nbs_ = [values[f"nbs{i}"] for i in range(1, 6)]
    lr_  = [values[f"lr{i}"]  for i in range(1, 6)]

    parameters_CWAE = {
        "normalization": "Standard",
        "latent_dims": latent_dims,
        "NUM_NEURON": nns_,
        "BATCH_SIZE": int(values["batch_size"]),
        "LearningRate": lr_,
        'ITERATION': int(2e3),
        "Final_Number_ITERATION": int(1e3),
        "num_resblocks": nbs_,
        "lamb": float(values["lamb"]),
        "n_critic": int(values["n_critic"]),
    }
    return parameters_CWAE

def get_cwae2_parameters(n, latent_dims):
    # define your rule for small vs large n
    if n <= 5:
        values = CWAE2_VALUES["small"]
    else:
        values = CWAE2_VALUES["large"]

    nns_ = [values[f"nns{i}"] for i in range(1, 6)]
    nbs_ = [values[f"nbs{i}"] for i in range(1, 6)]
    lr_  = [values[f"lr{i}"]  for i in range(1, 6)]

    parameters_CWAE = {
        "normalization": "Standard",
        "latent_dims": latent_dims,
        "NUM_NEURON": nns_,
        "BATCH_SIZE": int(values["batch_size"]),
        "LearningRate": lr_,
        'ITERATION': int(2e3),
        "Final_Number_ITERATION": int(1e3),
        "num_resblocks": nbs_,
        "lamb": float(values["lamb"]),
        "n_critic": int(values["n_critic"]),
    }
    return parameters_CWAE

def get_cwae3_parameters(n, latent_dims):
    # define your rule for small vs large n
    if n <= 5:
        values = CWAE3_VALUES["small"]
    else:
        values = CWAE3_VALUES["large"]

    nns_ = [values[f"nns{i}"] for i in range(1, 6)]
    nbs_ = [values[f"nbs{i}"] for i in range(1, 6)]
    lr_  = [values[f"lr{i}"]  for i in range(1, 6)]

    parameters_CWAE = {
        "normalization": "Standard",
        "latent_dims": latent_dims,
        "NUM_NEURON": nns_,
        "BATCH_SIZE": int(values["batch_size"]),
        "LearningRate": lr_,
        'ITERATION': int(2e3),
        "Final_Number_ITERATION": int(1e3),
        "num_resblocks": nbs_,
        "lamb": float(values["lamb"]),
        "n_critic": int(values["n_critic"]),
    }
    return parameters_CWAE













