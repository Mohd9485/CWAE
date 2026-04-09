#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 00:36:54 2026

@author: jarrah
"""
CWAE1_VALUES = {
    "small": {
        'batch_size': 1,
        'lamb': 0.00336634027762,
        'lr1': 0.000470631048994,
        'lr2': 0.0008359479402223,
        'lr3': 0.00602885134977,
        'lr4': 0.001296207611921,
        'lr5': 0.0001352184803291,
    },
    "large": {
        'batch_size': 1,
        'lamb': 0.00336634027762,
        'lr1': 0.000470631048994,
        'lr2': 0.0008359479402223,
        'lr3': 0.00602885134977,
        'lr4': 0.001296207611921,
        'lr5': 0.0001352184803291,
    }
}

CWAE2_VALUES = {
    "small": {
        'batch_size': 4,
        'lamb': 0.436634027762,
        'lr1': 0.000470631048994,
        'lr2': 0.0008359479402223,
        'lr3': 0.00602885134977,
        'lr4': 0.001296207611921,
        'lr5': 0.0001352184803291,
    },
    "large": {
        'batch_size': 8,
        'lamb': 2.06634027762,
        'lr1': 0.0000500631048994,
        'lr2': 0.00005059479402223,
        'lr3': 0.000352885134977,
        'lr4': 0.0003096207611921,
        'lr5': 0.00010184803291,
    }
}
CWAE3_VALUES = {
    "small": {
        'batch_size': 1,
        'lamb': 0.001897196623392,
        'lr1': 0.0001136384020772,
        'lr2': 0.003013391313466,
        'lr3': 0.006469222643601,
        'lr4': 0.00848586399119,
        'lr5': 3.992259860268e-04,
    },
    "large": {
        'batch_size': 2,
        'lamb': 0.001897196623392,
        'lr1': 0.0001136384020772,
        'lr2': 0.003013391313466,
        'lr3': 0.006469222643601,
        'lr4': 0.00848586399119,
        'lr5': 3.992259860268e-04,
    }
}



OTF_VALUES = {
    "small":{
        'lr1':0.000359192322780721,
        'lr2':0.003910496591912005,
        'nns':2,
        'nbs1':1,
        'nbs2':1,
        'bs':6,
        'n_critic':5,
        'kernel_size':3,
        'sch_gamma':0.9981473099849163,
    },
   "large":{
        'lr1':0.000359192322780721,
        'lr2':0.003910496591912005,
        'nns':2,
        'nbs1':1,
        'nbs2':1,
        'bs':6,
        'n_critic':5,
        'kernel_size':3,
        'sch_gamma':0.9981473099849163,
    }
    
}

SAMTF_VALUES = {
    "small":{
      'bs': 3,
      'eps_G': 0.4994764995163,
      'eps_I': 0.2576148600441,
      'lr1': 0.000169810252,
      'lr2': 0.0001251818457,
      'n_critic': 6,
      'nbs1': 1,
      'nbs2': 2,
      'nns': 8,
    },
    "large":{
      'bs': 14,
      'eps_G': 0.1760115504407,
      'eps_I': 0.4448314050306,
      'lr1': 0.0010914585094,
      'lr2': 5.96520669e-05,
      'n_critic': 7,
      'nbs1': 3,
      'nbs2': 1,
  'nns': 5,
    }
}

def get_cwae1_parameters(N, device, latent_dims):
    # define your rule for small vs large N
    if N < 1000:
        values = CWAE1_VALUES["small"]
    else:
        values = CWAE1_VALUES["large"]

    lr_  = {f"lr{i}": values[f"lr{i}"] for i in range(1, 6)}

    parameters_CWAE = {
        "latent_dims": latent_dims,
        "BATCH_SIZE": int(values["batch_size"])*8,
        "epochs": int(50),
        "Final_Number_ITERATION": int(1e3),
        "lamb": float(values["lamb"]),
        "val_split": float(0.2),
        "device": device,
    }
    return parameters_CWAE | lr_

def get_cwae2_parameters(N, device, latent_dims):
    # define your rule for small vs large N
    if N < 1000:
        values = CWAE2_VALUES["small"]
    else:
        values = CWAE2_VALUES["large"]

    lr_  = {f"lr{i}": values[f"lr{i}"] for i in range(1, 6)}

    parameters_CWAE = {
        "latent_dims": latent_dims,
        "BATCH_SIZE": int(values["batch_size"])*8,
        "epochs": int(130),
        'fix_iter': 100000,
        "Final_Number_ITERATION": int(1e5),
        "lamb": float(values["lamb"]),
        "val_split": float(0.2),
        "device": device,
    }
    return parameters_CWAE | lr_

def get_cwae3_parameters(N, device, latent_dims):
    # define your rule for small vs large N
    if N < 1000:
        values = CWAE3_VALUES["small"]
    else:
        values = CWAE3_VALUES["large"]

    lr_  = {f"lr{i}": values[f"lr{i}"] for i in range(1, 6)}

    parameters_CWAE = {
        "latent_dims": latent_dims,
        "BATCH_SIZE": int(values["batch_size"])*8,
        "epochs": int(50),
        "Final_Number_ITERATION": int(1e3),
        "lamb": float(values["lamb"]),
        "val_split": float(0.2),
        "device": device,
    }
    return parameters_CWAE | lr_

def get_otf_parameters(N, latent_dims):
    # define your rule for small vs large N
    if N < 1000:
        values = OTF_VALUES["small"]
    else:
        values = OTF_VALUES["large"]

    parameters_OTF = {
        # 'INPUT_DIM': [L, dy],
        'BATCH_SIZE': int(values["bs"])*16,
        'LearningRate': [float(values["lr1"]) , float(values["lr2"])],  # Learning rates for the mapping networks.
        'ITERATION': int(3e3),
        'Final_Number_ITERATION': int(64 / 16),
        'K_in': int(values["n_critic"]),
        'fix_iter': 5,
        'num_resblocks': [int(values["nbs1"]),int(values["nbs2"])],  # Number of residual blocks for the two networks (f,T).
        'base_channels': values["nns"],
        'scheduler_gamma': values["sch_gamma"],
        'kernel_size': values["kernel_size"],
    }
    return parameters_OTF


def get_samtf_parameters(N, latent_dims):
    # define your rule for small vs large N
    if N < 1000:
        values = SAMTF_VALUES["small"]
    else:
        values = SAMTF_VALUES["large"]

    parameters_SAMTF = {
        'normalization': "Standard",   # Options: 'None', 'Standard', 'MinMax'
        # 'INPUT_DIM': [L, dy],
        'NUM_NEURON': int(values["nns"])*16,
        'BATCH_SIZE': int(values["bs"])*16,
        'LearningRate': [float(values["lr1"]) , float(values["lr2"])],  # Learning rates for the mapping networks.
        'ITERATION': int(3e3),
        'Final_Number_ITERATION': int(64 / 16),
        'K_in': int(values["n_critic"]),
        'fix_iter': 5,
        'num_resblocks': [int(values["nbs1"]),int(values["nbs2"])],  # Number of residual blocks for the two networks (f,T).
        'epsilon_I': float(values["eps_I"]),      # data-free LIS tolerance
        'epsilon_G': float(values["eps_G"]),      # data-dependent LIS tolerance epsilon_G = epsilon_I / 2 a reliable default
        'restart_freq': 5, # Restart frequency to rebuild transport map from scratch
        'r_x': latent_dims[0], # State-space LIS dimension  r_x 
        'r_y': latent_dims[1] # Observation-space LIS dimension  r_y
    }
    return parameters_SAMTF













