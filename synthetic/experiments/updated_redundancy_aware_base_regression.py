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
            # j = [mod0, mod1, labels]
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


def EncourageAlignment(ce_weight=2, align_weight=args.weight, criterion=torch.nn.CrossEntropyLoss()):
    def _actualfunc(pred, truth, args):
        ce_loss = criterion(pred, truth)
        outs = args['reps']
        outs[0] = outs[0].view(-1,).cpu().detach().numpy()
        outs[1] = outs[1].view(-1,).cpu().detach().numpy()
        align_loss = np.dot(outs[0], outs[1])/(np.linalg.norm(outs[0])*np.linalg.norm(outs[1]))
        return align_loss * align_weight + ce_loss * ce_weight
    return _actualfunc

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
    """encoders = [nn.Sequential(Linear(input_dim, args.hidden_dim).to(device),
                                nn.ReLU(),
                                Linear(args.hidden_dim, args.hidden_dim).to(device)) for input_dim in input_dims]"""

    heads = [MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device) for input_dim in input_dims]

    fusion = nn.Sequential(
        Concat(),
        MLP3(len(args.modalities)*args.hidden_dim, args.n_latent, args.n_latent)
    ).to(device)
    """fusion = FusionTransformerWrapper(args.hidden_dim,
        n_heads=8,
        n_layers=2,
        fusion="concat",
        pool="cls",
        dropout=0.0)"""
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
        objective=torch.nn.MSELoss(),
        optimtype=torch.optim.Adam,
        lr=0.001, # args.lr,
        save=args.saved_model,
        weight_decay=args.weight_decay,
        task="regression"
    )

    # Testing
    print("Testing:")
    dict_of_metrics = test(model, testdata, no_robust=True, criterion=torch.nn.MSELoss(), task="regression")

    X_train_dict = extract_representations(model, traindata, device)
    y_train = X_train_dict["targets"].float()

    X_val_dict = extract_representations(model, validdata, device)
    y_val = X_val_dict["targets"].float()

    X_test_dict = extract_representations(model, testdata, device)
    y_test = X_test_dict["targets"].float()

    X_train_dict = {
        "modality0": X_train_dict["modality0"].float(),
        "modality1": X_train_dict["modality1"].float()
    }

    X_val_dict = {
        "modality0": X_val_dict["modality0"].float(),
        "modality1": X_val_dict["modality1"].float()
    }

    X_test_dict = {
        "modality0": X_test_dict["modality0"].float(),
        "modality1": X_test_dict["modality1"].float()
    }

    y_pred_dict = return_redundancy_test_performances(X_train_dict, X_val_dict, X_test_dict, y_train, y_val, y_test, "redundancy", distribution_target="gaussian", num_classes=1)

    print(dict_of_metrics["joint_ce"])
    print(dict_of_metrics["modalities_ce"])

    results = {}
    for key in ["modality0", "modality1", "average"]:
        print(y_pred_dict[key].shape)
        acc, ce = traditional_cross_entropy_from_probs(softmax(y_pred_dict[key]), y_pred_dict["targets"])
        results[key] = {"accuracy": acc, "cross_entropy": ce}

    for k, v in results.items():
        print(f"{k:10s} | acc = {v['accuracy']:.4f}, CE = {v['cross_entropy']:.4f}")

    print('---')

    compute_PID_categorical(dict_of_metrics["joint_ce"], dict_of_metrics["modalities_ce"][0],
                            dict_of_metrics["modalities_ce"][1], results["average"]["cross_entropy"], num_classes=args.num_classes)

