import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import log
import argparse
import multiprocessing
import sys, os

from unimodals.common_models import MLP, Linear, MLP3
from synthetic.get_data import get_dataloader
from synthetic.updated_redundancy_aware_supervised_learning import train, test
from utils_ours import (
    return_redundancy_test_performances,
    compute_PID_categorical_with_source_decomposition
)
from fusions.common_fusions import Concat

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
        print(f"{name:<12s} | {acc:8.4f} | {ce:8.4f}")


# ---------- BASIC UTILS ----------

def ce_per_sample(targets, probs, eps=1e-12):
    """
    Per-sample cross-entropy:
        CE(x) = -log p(y|x)
    """
    probs = torch.clamp(probs, eps, 1.0)
    return -torch.log(probs[torch.arange(len(targets)), targets])


def compute_probs_and_ce(logits_list, targets):
    """
    From logits → probabilities + CE for each modality
    """
    probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
    ce_list = [ce_per_sample(targets, probs) for probs in probs_list]
    return probs_list, ce_list


def compute_pointwise_information(ce_list, log_py):
    """
    PMI proxy:
        i = log p(y|x) - log p(y)
          = -CE - log p(y)
    """
    return [-ce - log_py for ce in ce_list]


# ---------- CCS REDUNDANCY ----------

def compute_ccs_and_selection(probs_list, ce_list, same_sign, log_py):
    """
    Compute:
        - CCS redundancy
        - associated predictions

    Rule:
        if sign agreement:
            redundancy = worst CE
            prediction = worst modality
        else:
            redundancy = baseline (-log p(y))
            prediction = uniform
    """
    ce_stack = torch.stack(ce_list, dim=1)        # (N, 2)
    probs_stack = torch.stack(probs_list, dim=1)  # (N, 2, C)

    # Worst modality (higher CE)
    worst_idx = torch.argmax(ce_stack, dim=1)     # (N,)
    worst_ce = torch.max(ce_stack, dim=1).values  # (N,)

    # Baseline (independence)
    baseline = -log_py * torch.ones_like(worst_ce)

    # CCS redundancy
    ccs = torch.where(same_sign, worst_ce, baseline)

    # Select corresponding prediction
    selected = probs_stack[torch.arange(probs_stack.size(0)), worst_idx]

    # Uniform fallback
    num_classes = probs_list[0].size(1)
    uniform = torch.full_like(selected, 1.0 / num_classes)

    selected = torch.where(same_sign.unsqueeze(1), selected, uniform)

    return ccs, selected


# ---------- PID ----------
def compute_pointwise_pid(dict_of_metrics, num_classes):
    """
    Compute per-sample PID:
        total = redundancy + unique_0 + unique_1 + synergy
    """
    logK = log(num_classes)
    pid_list = []

    for j, m0, m1, r, y in zip(
        dict_of_metrics["pred_joint"],
        dict_of_metrics["pred_modalities"][0],
        dict_of_metrics["pred_modalities"][1],
        dict_of_metrics["redundancy_preds"],
        dict_of_metrics["true_labels"]
    ):
        y = y.long()

        def logp(p):
            return torch.log(torch.clamp(p, 1e-12, 1.0))

        pj = logp(F.softmax(j, dim=0))[y]
        pm0 = logp(F.softmax(m0, dim=0))[y]
        pm1 = logp(F.softmax(m1, dim=0))[y]
        pr = logp(r)[y]

        total = logK + pj

        # redundancy = max between CCS and source redundancy
        r_val = logK + pr

        # unique information
        u0 = logK + pm0 - r_val
        u1 = logK + pm1 - r_val

        # synergy
        s = total - u0 - u1 - r_val

        pid_list.append([u0, u1, r_val, s])

    return np.array(pid_list)

def compute_pointwise_pid_with_source(dict_of_metrics, num_classes):
    """
    Compute per-sample PID:
        total = redundancy + unique_0 + unique_1 + synergy
    """
    logK = log(num_classes)
    pid_list = []

    for j, m0, m1, r, r_src, y in zip(
        dict_of_metrics["pred_joint"],
        dict_of_metrics["pred_modalities"][0],
        dict_of_metrics["pred_modalities"][1],
        dict_of_metrics["redundancy_preds"],
        dict_of_metrics["source_redundancy_preds"],
        dict_of_metrics["true_labels"]
    ):
        y = y.long()

        def logp(p):
            return torch.log(torch.clamp(p, 1e-12, 1.0))

        pj = logp(F.softmax(j, dim=0))[y]
        pm0 = logp(F.softmax(m0, dim=0))[y]
        pm1 = logp(F.softmax(m1, dim=0))[y]
        pr = logp(r)[y]
        pr_src = logp(r_src)[y]

        total = logK + pj

        # redundancy = max between CCS and source redundancy
        r_val = max(logK + pr, logK + pr_src)

        # unique information
        u0 = logK + pm0 - r_val
        u1 = logK + pm1 - r_val

        # synergy
        s = total - u0 - u1 - r_val

        pid_list.append([u0, u1, r_val, s])

    return np.array(pid_list)


def normalize_pid(pid):
    """
    Ensure valid PID:
        - non-negative
        - sums to 1
    """
    pid = np.maximum(pid, 0)

    row_sums = pid.sum(axis=1, keepdims=True)
    zero_rows = row_sums.squeeze() == 0

    pid[zero_rows] = 1.0 / pid.shape[1]
    pid /= pid.sum(axis=1, keepdims=True)

    return pid


def cosine_similarity(a, b):
    """
    Row-wise cosine similarity
    """
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)

# =========================
# MAIN
# =========================

if __name__ == "__main__":

    # ========= 1. LOAD DATA =========
    traindata, validdata, _, testdata = get_dataloader(
        path=args.data_path,
        keys=args.keys,
        modalities=args.modalities,
        batch_size=args.bs,
        num_workers=args.num_workers
    )

    # ========= 2. BUILD MODEL =========
    input_dims = args.input_dim * len(args.modalities) if len(args.input_dim) == 1 else args.input_dim

    encoders = [Linear(d, args.hidden_dim).to(device) for d in input_dims]
    heads = [MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device) for _ in input_dims]

    fusion = nn.Sequential(
        Concat(),
        MLP3(len(args.modalities) * args.hidden_dim, args.n_latent, args.n_latent)
    ).to(device)

    head = MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device)

    # ========= 3. TRAIN =========
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

    # ========= 4. TEST =========
    dict_of_metrics = test(
        model,
        testdata,
        no_robust=True,
        criterion=torch.nn.CrossEntropyLoss()
    )

    print_model_metrics(dict_of_metrics)

    # ========= 5. EXTRACT REPRESENTATIONS =========
    X_train_dict, y_train = extract_split(model, traindata, device)
    X_val_dict, y_val = extract_split(model, validdata, device)
    X_test_dict, y_test = extract_split(model, testdata, device)

    weights_test = testdata.dataset.data["weights"]

    # ========= 6. CCS REDUNDANCY =========
    targets = dict_of_metrics["true_labels"].long()
    logits_list = dict_of_metrics["pred_modalities"]

    log_py = np.log(1.0 / args.num_classes)

    probs_list, ce_list = compute_probs_and_ce(logits_list, targets)
    i_list = compute_pointwise_information(ce_list, log_py)

    same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])

    ccs, selected_preds = compute_ccs_and_selection(
        probs_list, ce_list, same_sign, log_py
    )

    dict_of_metrics["redundancy_ce"] = ccs.mean().item()
    dict_of_metrics["redundancy_preds"] = selected_preds

    # ========= 7. SOURCE REDUNDANCY =========
    y_pred_dict = return_redundancy_test_performances(
        X_train_dict, X_val_dict, X_test_dict,
        y_train, y_val, y_test,
        "redundancy",
        distribution_target="categorical",
        num_classes=args.num_classes
    )

    results = compute_redundancy_metrics(y_pred_dict)
    print_model_metrics(dict_of_metrics)
    print_redundancy_metrics(results)

    dict_of_metrics["source_redundancy_ce"] = results["average"]["cross_entropy"]
    dict_of_metrics["source_redundancy_preds"] = y_pred_dict["average"]

    # ========= 8 GLOBAL PID =========
    compute_PID_categorical_with_source_decomposition(
        dict_of_metrics["joint_ce"],
        dict_of_metrics["modalities_ce"][0],
        dict_of_metrics["modalities_ce"][1],
        dict_of_metrics["redundancy_ce"],
        dict_of_metrics["source_redundancy_ce"],
        num_classes=args.num_classes
    )

    # ========= 9. POINTWISE PID WITH SOURCE =========
    pid_source = compute_pointwise_pid_with_source(dict_of_metrics, args.num_classes)
    pid_norm = normalize_pid(pid_source)
    sim = cosine_similarity(pid_norm, weights_test)
    print("Mean true per-sample cosine similarity with source:", sim.mean())

    # ========= 9. POINTWISE PID WITHOUT SOURCE =========
    pid = compute_pointwise_pid(dict_of_metrics, args.num_classes)
    pid_norm = normalize_pid(pid)
    sim = cosine_similarity(pid_norm, weights_test)
    print("Mean true per-sample cosine similarity without source:", sim.mean())