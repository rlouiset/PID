"""
LSMI PID on latent representations of a trained MLP multimodal model.

Pipeline:
  1. Train MLP encoders + fusion (same as ccs_source_aware_base)
  2. Extract hidden_dim-d representations for each modality
  3. Train LSMI discriminators (p(y|z1), p(y|z2), p(y|z1,z2)) on those reps
  4. Train MargKernel entropy estimators (H(Z1), H(Z2)) on those reps
  5. Compute LSMI PID — same metric computation as lsmi_base
"""
from __future__ import print_function
import argparse
import math
import multiprocessing
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unimodals.common_models import MLP, Linear, MLP3
from synthetic.get_data import get_dataloader
from synthetic.updated_redundancy_aware_supervised_learning import train, test
from fusions.common_fusions import Concat
from utils_lsmi import MargKernel, cls_network, feature_dataset, setup_seed

multiprocessing.set_start_method('fork', force=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ─── Args ────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
# Data / model (mirrors ccs_source_aware_base)
parser.add_argument("--data-path",    default="SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str)
parser.add_argument("--keys",         nargs='+', default=['a','b','c','d','e','label'], type=str)
parser.add_argument("--modalities",   nargs='+', default=[0, 1], type=int)
parser.add_argument("--bs",           default=32,    type=int)
parser.add_argument("--num-workers",  default=4,     type=int)
parser.add_argument("--input-dim",    nargs='+',     default=30,   type=int)
parser.add_argument("--hidden-dim",   default=512,   type=int)
parser.add_argument("--n-latent",     default=512,   type=int)
parser.add_argument("--num-classes",  default=2,     type=int)
parser.add_argument("--epochs",       default=30,    type=int)
parser.add_argument("--lr",           default=1e-4,  type=float)
parser.add_argument("--weight-decay", default=0.01,  type=float)
parser.add_argument("--saved-model",  default=None,  type=str)
# LSMI
parser.add_argument("--embed-size",      default=32,  type=int,
                    help="Hidden dim of LSMI discriminators")
parser.add_argument("--lsmi-bs",         default=512, type=int)
parser.add_argument("--epochs-disc",     default=30,  type=int)
parser.add_argument("--epochs-entropy",  default=30,  type=int)
parser.add_argument("--seed",            default=42,  type=int)
args = parser.parse_args()

setup_seed(args.seed)


# ─── Utilities ───────────────────────────────────────────────────────────────

def RUS_adjustment(rus):
    r_orig, u_1_orig, u_2_orig, s_orig = rus
    R_mean  = r_orig.detach().mean()
    U1_mean = u_1_orig.detach().mean()
    U2_mean = u_2_orig.detach().mean()
    S_mean  = s_orig.detach().mean()
    adj = torch.tensor(0.0, dtype=R_mean.dtype, device=R_mean.device)
    if R_mean < 0 or S_mean < 0:
        adj = -torch.min(R_mean, S_mean)
    elif U1_mean < 0 or U2_mean < 0:
        adj = torch.min(U1_mean, U2_mean)
    return r_orig + adj, u_1_orig - adj, u_2_orig - adj, s_orig + adj


def normalize_pid(pid):
    pid = np.maximum(pid, 0)
    pid /= pid.sum(axis=1, keepdims=True) + 1e-12
    return pid


def cosine_similarity(a, b):
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.sum(a * b, axis=1)


def print_model_metrics(d):
    print(f"{'Model':<12s} | {'Acc':>8s} | {'CE':>8s}")
    print("-" * 36)
    print(f"{'Joint':<12s} | {d['joint_acc']:8.4f} | {d['joint_ce']:8.4f}")
    print(f"{'Modality 0':<12s} | {d['modalities_acc'][0]:8.4f} | {d['modalities_ce'][0]:8.4f}")
    print(f"{'Modality 1':<12s} | {d['modalities_acc'][1]:8.4f} | {d['modalities_ce'][1]:8.4f}")


# ─── Representation extraction ────────────────────────────────────────────────

def extract_representations(model, loader):
    """Extract hidden_dim-d encoder outputs (model.reps) for both modalities."""
    model.eval()
    reps0, reps1, targets = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = [batch[0].to(device).float(), batch[1].to(device).float()]
            _ = model(inputs)
            reps0.append(model.reps[0].cpu())
            reps1.append(model.reps[1].cpu())
            targets.append(batch[2].cpu())
    return torch.cat(reps0).float(), torch.cat(reps1).float(), torch.cat(targets)


# ─── LSMI estimators ─────────────────────────────────────────────────────────

def obtain_discriminator(train_loader, repr_dim, embed_size, n_classes, n_epochs):
    """
    Train three classifiers on latent representations:
      d1(z1)     → p(y|z1)    → I(Z1; Y)
      d2(z2)     → p(y|z2)    → I(Z2; Y)
      dj(z1, z2) → p(y|z1,z2) → I(Z1,Z2; Y)
    """
    d1 = cls_network(repr_dim,          embed_size, n_classes).to(device)
    d2 = cls_network(repr_dim,          embed_size, n_classes).to(device)
    dj = cls_network(repr_dim * 2,      embed_size, n_classes).to(device)
    models    = [d1, d2, dj]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(n_epochs):
        losses = 0.0; num_samples = 0
        for batch in train_loader:
            z1, z2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            loss = (criterion(d1(z1), y) + criterion(d2(z2), y)
                    + criterion(dj(torch.cat([z1, z2], dim=1)), y))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses += loss.item() * len(y); num_samples += len(y)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  [disc]    Epoch [{epoch+1}/{n_epochs}]  Loss: {losses/num_samples:.4f}')
    return models


def obtain_entropy_estimator(train_loader, repr_dim, n_epochs):
    """Train MargKernel models for H(Z1) and H(Z2)."""
    mk1 = MargKernel(repr_dim).to(device)
    mk2 = MargKernel(repr_dim).to(device)
    models    = [mk1, mk2]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    for epoch in range(n_epochs):
        for m in models: m.train()
        losses = 0.0
        for batch in train_loader:
            z1, z2 = batch[0].to(device), batch[1].to(device)
            loss = mk1(z1) + mk2(z2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses += loss.item() * len(z1)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  [entropy] Epoch [{epoch+1}/{n_epochs}]  Loss: {losses/len(train_loader.dataset):.4f}')
    return models


def get_mutual_info(loader, model, modality, n_classes):
    """I(Zi; Y) ≈ E[ log p(y|zi) ] + log K  (uniform prior, K = n_classes)."""
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            z1, z2, y = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            if modality == 'modality_1':
                x = z1
            elif modality == 'modality_2':
                x = z2
            else:
                x = torch.cat([z1, z2], dim=1)
            rows = torch.arange(x.size(0), device=device)
            out  = model(x)
            info.append(math.log(n_classes) + F.log_softmax(out, dim=1)[rows, y])
    return torch.cat(info).detach()


def get_entropy(loader, model, modality):
    """H(Zi) via MargKernel."""
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if modality == 'modality_1' else batch[1]).to(device)
            info.append(model(x))
    return torch.cat(info).detach()


def LSMI_estimation(loader, discriminator, entropy_estimator, n_classes):
    I1Y  = get_mutual_info(loader, discriminator[0], 'modality_1',  n_classes)
    I2Y  = get_mutual_info(loader, discriminator[1], 'modality_2',  n_classes)
    I12Y = get_mutual_info(loader, discriminator[2], 'modality_12', n_classes)
    H1   = get_entropy(loader, entropy_estimator[0], 'modality_1')
    H2   = get_entropy(loader, entropy_estimator[1], 'modality_2')

    r_plus  = torch.minimum(H1, H2)
    r_minus = torch.minimum(H1 - I1Y, H2 - I2Y)
    r  = r_plus - r_minus
    u1 = I1Y  - r
    u2 = I2Y  - r
    s  = I12Y - r - u1 - u2

    r_adj, u1_adj, u2_adj, s_adj = RUS_adjustment([r, u1, u2, s])
    print(f"R={r_adj.mean():.4f}  U1={u1_adj.mean():.4f}  U2={u2_adj.mean():.4f}  S={s_adj.mean():.4f}")

    return r.cpu().numpy(), u1.cpu().numpy(), u2.cpu().numpy(), s.cpu().numpy()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    # ========= 1. LOAD DATA =========
    traindata, validdata, _, testdata = get_dataloader(
        path=args.data_path,
        keys=args.keys,
        modalities=args.modalities,
        batch_size=args.bs,
        num_workers=args.num_workers,
    )

    # ========= 2. BUILD MODEL =========
    input_dims = (args.input_dim * len(args.modalities)
                  if len(args.input_dim) == 1 else args.input_dim)

    encoders = [Linear(d, args.hidden_dim).to(device) for d in input_dims]
    heads    = [MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device)
                for _ in input_dims]
    fusion   = nn.Sequential(
        Concat(),
        MLP3(len(args.modalities) * args.hidden_dim, args.n_latent, args.n_latent),
    ).to(device)
    head     = MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device)

    # ========= 3. TRAIN =========
    model = train(
        encoders, fusion, head, heads,
        traindata, validdata,
        args.epochs,
        objective=torch.nn.CrossEntropyLoss(),
        optimtype=torch.optim.AdamW,
        lr=args.lr,
        save=args.saved_model,
        weight_decay=args.weight_decay,
    )

    # ========= 4. TEST =========
    dict_of_metrics = test(model, testdata, no_robust=True,
                           criterion=torch.nn.CrossEntropyLoss())
    print_model_metrics(dict_of_metrics)

    # ========= 5. EXTRACT REPRESENTATIONS =========
    print("\nExtracting representations...")
    train_z1, train_z2, y_train = extract_representations(model, traindata)
    test_z1,  test_z2,  y_test  = extract_representations(model, testdata)
    print(f"  Repr shapes: {train_z1.shape}, {train_z2.shape}")

    repr_dim = train_z1.shape[1]  # = hidden_dim

    # ========= 6. BUILD LSMI LOADERS =========
    lsmi_train = DataLoader(
        feature_dataset(train_z1, train_z2, y_train),
        batch_size=args.lsmi_bs, shuffle=True,  num_workers=0)
    lsmi_test  = DataLoader(
        feature_dataset(test_z1,  test_z2,  y_test),
        batch_size=args.lsmi_bs, shuffle=False, num_workers=0)

    # ========= 7. TRAIN LSMI ESTIMATORS =========
    print("\nTraining discriminators  (p(y|z1), p(y|z2), p(y|z1,z2))...")
    discriminator = obtain_discriminator(
        lsmi_train, repr_dim, args.embed_size, args.num_classes, args.epochs_disc)

    print("\nTraining entropy estimators  (H(Z1), H(Z2))...")
    entropy_estimator = obtain_entropy_estimator(
        lsmi_train, repr_dim, args.epochs_entropy)

    # ========= 8. LSMI (PID) =========
    print("\nTrain PID:")
    LSMI_estimation(lsmi_train, discriminator, entropy_estimator, args.num_classes)

    print("\nTest PID:")
    r, u1, u2, s = LSMI_estimation(lsmi_test, discriminator, entropy_estimator, args.num_classes)

    # ========= 9. RAW / ADJUSTED / NORMALISED =========
    pid = np.stack([u1, u2, r, s], axis=1)  # [U1, U2, R, S]

    print("\nbefore adjustment")
    print("u1:", np.mean(pid[:, 0]))
    print("u2:", np.mean(pid[:, 1]))
    print("r: ", np.mean(pid[:, 2]))
    print("s: ", np.mean(pid[:, 3]))

    r_adj, u1_adj, u2_adj, s_adj = RUS_adjustment(
        [torch.tensor(r), torch.tensor(u1), torch.tensor(u2), torch.tensor(s)])
    r_adj, u1_adj, u2_adj, s_adj = (r_adj.numpy(), u1_adj.numpy(),
                                     u2_adj.numpy(), s_adj.numpy())

    print("\nafter adjustment")
    print("r: ", np.mean(r_adj))
    print("u1:", np.mean(u1_adj))
    print("u2:", np.mean(u2_adj))
    print("s: ", np.mean(s_adj))

    pid_norm = normalize_pid(pid)

    weights_test = testdata.dataset.data["weights"]
    sim = cosine_similarity(pid_norm, weights_test)
    print("\nMean true per-sample cosine similarity:", sim.mean())
