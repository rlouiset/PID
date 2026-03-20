import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils.helper_modules import Sequential2 # noqa
from unimodals.common_models import MLP, Linear, MLP3
from synthetic.get_data import get_dataloader
import torch
import torch.nn as nn

from math import *
import matplotlib.pyplot as plt

from synthetic.updated_redundancy_aware_supervised_learning import train, test
from utils_ours import return_redundancy_test_performances, compute_PID_categorical
from fusions.common_fusions import Concat
from fusions.transformers_fusion import FusionTransformerWrapper

import argparse

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", nargs='+', default=30, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--n-latent", default=512, type=int)
parser.add_argument("--rank", default=32, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--weight", default=1, type=float)
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()

softmax = torch.nn.Softmax(dim=-1)

def extract_representations(model, dataloader, device):
    model.eval()

    reps0, reps1, targets = [], [], []

    with torch.no_grad():
        for j in dataloader:
            inputs = [j[0].to(device).float(),
                      j[1].to(device).float()]
            y = j[2]

            # Forward pass (fills model.reps)
            _ = model(inputs)

            reps0.append(model.reps[0].cpu())
            reps1.append(model.reps[1].cpu())
            targets.append(y.cpu())

    return {
        "modality0": torch.cat(reps0, dim=0),
        "modality1": torch.cat(reps1, dim=0),
        "targets": torch.cat(targets, dim=0),
    }



def RUS_adjustment(rus):
    """
    Adjusts the input tensors (r, u1, u2, s) while preserving certain sums
    and the original device of the tensors. The adjustment aims to make the
    means of these components non-negative based on a specific priority:

    1. If the mean of 'r' (R_mean) or 's' (S_mean) is negative, an adjustment
       factor is calculated to make both R_mean and S_mean non-negative.
       This adjustment might consequently alter the means of 'u1' (U1_mean)
       and 'u2' (U2_mean), potentially making them negative.

    2. If R_mean and S_mean are already non-negative, but U1_mean or U2_mean
       is negative, the adjustment factor is calculated to make both U1_mean
       and U2_mean non-negative. This adjustment might, in turn, make
       R_mean or S_mean negative if they were small positive values.

    The adjustment maintains the following sum properties for the means:
    - (R_mean + U1_mean + U2_mean + S_mean) remains unchanged.
    - (R_mean + U1_mean) remains unchanged.
    - (R_mean + U2_mean) remains unchanged.

    Args:
        rus (tuple or list): A collection of four PyTorch tensors (r, u1, u2, s).

    Returns:
        tuple: A tuple of four adjusted PyTorch tensors (r_adjusted, u1_adjusted,
               u2_adjusted, s_adjusted), on the same device as the input tensors.
    """
    r_orig, u_1_orig, u_2_orig, s_orig = rus

    R_mean = r_orig.detach().mean()
    U1_mean = u_1_orig.detach().mean()
    U2_mean = u_2_orig.detach().mean()
    S_mean = s_orig.detach().mean()

    adj_factor = torch.tensor(0.0, dtype=R_mean.dtype, device=R_mean.device)

    # Priority 1: Address negative mean of r or s
    if R_mean < 0 or S_mean < 0:
        adj_factor = -torch.min(R_mean, S_mean)

    # Priority 2: If means of r and s are non-negative, address negative mean of u1 or u2
    elif U1_mean < 0 or U2_mean < 0:
        adj_factor = torch.min(U1_mean, U2_mean)

    r_adjusted = r_orig + adj_factor
    u_1_adjusted = u_1_orig - adj_factor
    u_2_adjusted = u_2_orig - adj_factor
    s_adjusted = s_orig + adj_factor

    return r_adjusted, u_1_adjusted, u_2_adjusted, s_adjusted


def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):
    """
    probs:   (N, C) probabilities
    targets: (N,)
    """
    probs = torch.clamp(probs, min=eps, max=1.0)
    log_probs = torch.log(probs)

    ce = -log_probs[torch.arange(targets.shape[0]), targets.long()].mean()
    acc = (probs.argmax(dim=1) == targets).float().mean()

    return acc.item(), ce.item()

def extract_split(model, loader, device):
    d = extract_representations(model, loader, device)

    X = {
        "modality0": d["modality0"].float(),
        "modality1": d["modality1"].float()
    }
    y = d["targets"].float()

    return X, y


def compute_redundancy_metrics(y_pred_dict):
    results = {}
    for key in ["modality0", "modality1", "average"]:
        acc, ce = traditional_cross_entropy_from_probs(
            softmax(y_pred_dict[key]),
            y_pred_dict["targets"]
        )
        results[key] = {"accuracy": acc, "cross_entropy": ce}
    return results


def print_model_metrics(dict_of_metrics):
    print(f"{'Model':<12s} | {'Acc':>8s} | {'CE':>8s}")
    print("-" * 36)

    print(f"{'Joint':<12s} | {dict_of_metrics['joint_acc']:8.4f} | {dict_of_metrics['joint_ce']:8.4f}")
    print(f"{'Modality 0':<12s} | {dict_of_metrics['modalities_acc'][0]:8.4f} | {dict_of_metrics['modalities_ce'][0]:8.4f}")
    print(f"{'Modality 1':<12s} | {dict_of_metrics['modalities_acc'][1]:8.4f} | {dict_of_metrics['modalities_ce'][1]:8.4f}")


def print_redundancy_metrics(results):
    mapping = {
        "modality0": "Red Mod 0",
        "modality1": "Red Mod 1",
        "average": "Red Joint"
    }

    for key, name in mapping.items():
        acc = results[key]["accuracy"]
        ce = results[key]["cross_entropy"]
        print(f"{name:<22s} | {acc:8.4f} | {ce:8.4f}")


# Wrap main code to protect multiprocessing
if __name__ == "__main__":
    # Load data
    traindata, validdata, _, testdata = get_dataloader(
        path=args.data_path,
        keys=args.keys,
        modalities=args.modalities,
        batch_size=args.bs,
        num_workers=args.num_workers
    )

    # Specify model
    if len(args.input_dim) == 1:
        input_dims = args.input_dim * len(args.modalities)
    else:
        input_dims = args.input_dim

    encoders = [Linear(input_dim, args.hidden_dim).to(device) for input_dim in input_dims]

    heads = [MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device) for input_dim in input_dims]

    fusion = nn.Sequential(
        Concat(),
        MLP3(len(args.modalities)*args.hidden_dim, args.n_latent, args.n_latent)
    ).to(device)

    head = MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device)

    # Training
    model = train(
        encoders,
        fusion,
        head,
        heads,
        traindata,
        validdata,
        args.epochs,
        objective=torch.nn.CrossEntropyLoss(),
        optimtype=torch.optim.AdamW,
        lr=args.lr,
        save=args.saved_model,
        weight_decay=args.weight_decay
    )

    # Testing
    dict_of_metrics = test(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss())

    # Extract representations
    X_train_dict, y_train = extract_split(model, traindata, device)
    X_val_dict, y_val = extract_split(model, validdata, device)
    X_test_dict, y_test = extract_split(model, testdata, device)

    weights_test = testdata.dataset.data["weights"]

    # Redundancy evaluation
    y_pred_dict = return_redundancy_test_performances(
        X_train_dict,
        X_val_dict,
        X_test_dict,
        y_train,
        y_val,
        y_test,
        "redundancy",
        distribution_target="categorical",
        num_classes=args.num_classes
    )

    results = compute_redundancy_metrics(y_pred_dict)

    for k, v in results.items():
        print(f"redundancy representations - {k:10s} | acc = {v['accuracy']:.4f}, CE = {v['cross_entropy']:.4f}")

    print('---')

    print_model_metrics(dict_of_metrics)

    print('---')

    print_redundancy_metrics(results)

    compute_PID_categorical(
        dict_of_metrics["joint_ce"],
        dict_of_metrics["modalities_ce"][0],
        dict_of_metrics["modalities_ce"][1],
        max(results["modality0"]["cross_entropy"], results["modality1"]["cross_entropy"]),
        num_classes=args.num_classes
    )

    N = len(y_pred_dict["targets"])

    # PointWise PID
    list_of_pointwise_pid = []
    for j_i, m0_i, m1_i, r_i, y_i in zip(dict_of_metrics["pred_joint"], dict_of_metrics["pred_modalities"][0],
                                dict_of_metrics["pred_modalities"][1], y_pred_dict["average"], y_pred_dict["targets"]):

        y_i = y_i.long()

        total_contribution = log(args.num_classes) + torch.log(torch.clamp(softmax(j_i), min=1e-12, max=1.0))[y_i]

        r_contribution = log(args.num_classes) + torch.log(torch.clamp(softmax(r_i), min=1e-12, max=1.0))[y_i]

        m0_contribution = log(args.num_classes) + torch.log(torch.clamp(softmax(m0_i), min=1e-12, max=1.0))[y_i] - r_contribution
        m1_contribution = log(args.num_classes) + torch.log(torch.clamp(softmax(m1_i), min=1e-12, max=1.0))[y_i] - r_contribution

        s_contribution = total_contribution - m0_contribution - m1_contribution - r_contribution

        list_of_pointwise_pid.append([m0_contribution, m1_contribution, r_contribution, s_contribution])

    list_of_pointwise_pid = np.array(list_of_pointwise_pid)

    print(list_of_pointwise_pid[:5])

    # r, u0, u1, s = RUS_adjustment([list_of_pointwise_pid[:, 2], list_of_pointwise_pid[:, 0], list_of_pointwise_pid[:, 1], list_of_pointwise_pid[:, 0]])
    # list_of_pointwise_pid = np.array([u0, u1, r, s]).T

    print(np.mean(list_of_pointwise_pid[:, 0]))
    print(np.mean(list_of_pointwise_pid[:, 1]))
    print(np.mean(list_of_pointwise_pid[:, 2]))
    print(np.mean(list_of_pointwise_pid[:, 3]))

    # pid
    pid = np.maximum(list_of_pointwise_pid, 0)
    row_sums = pid.sum(axis=1, keepdims=True)
    # Replace zero rows with uniform distribution
    zero_rows = row_sums.squeeze() == 0
    pid[zero_rows] = 1.0 / pid.shape[1]
    # Now renormalize safely
    pid_norm = pid / pid.sum(axis=1, keepdims=True)

    # L2 normalize both vectors
    pid_l2 = pid_norm / np.linalg.norm(pid_norm, axis=1, keepdims=True)
    weights_l2 = weights_test / np.linalg.norm(weights_test, axis=1, keepdims=True)

    sim_pointwise = np.sum(pid_l2 * weights_l2, axis=1)

    print("Mean true per-sample cosine similarity:", sim_pointwise.mean())
