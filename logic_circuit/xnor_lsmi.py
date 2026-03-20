import torch
from torch.utils.data import Dataset

import argparse

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

from utils_ours import return_redundancy_test_performances, compute_PID_categorical

from utils_lsmi import MargKernel, cls_network
from utils_lsmi import get_loader, setup_seed


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


def obtain_feature_input(batch, device):
    modal_1 = batch[0].to(device)
    modal_2 = batch[1].to(device)
    labels = batch[2].to(device)
    return modal_1, modal_2, labels


def get_entropy(dataloader, model, modality='modality_1', cfg=None):
    model.eval()
    info = []
    with torch.no_grad():
        losses = 0.0
        for batch in dataloader:
            modal_1, modal_2, _ = obtain_feature_input(batch, device=cfg.device)
            if modality == "modality_1":
                input_data = modal_1
            elif modality == "modality_2":
                input_data = modal_2
            batch_size = input_data.shape[0]
            loss = model(input_data)
            info.append(loss)
            losses = losses + torch.mean(loss).item() * batch_size
    info = torch.cat(info, dim=0).detach()
    return info


def get_mutual_info(dataloader, model, modality='modality_1', cfg=None):
    model.eval()
    info = []
    with torch.no_grad():
        infos = 0.0
        for batch in dataloader:
            modal_1, modal_2, labels = obtain_feature_input(batch, device=cfg.device)
            if modality == "modality_1":
                input_data = modal_1
            elif modality == "modality_2":
                input_data = modal_2
            elif modality == "modality_12":
                input_data = torch.cat([modal_1, modal_2], dim=1)
            batch_size = input_data.shape[0]
            rows = torch.arange(batch_size)
            out = model(input_data)
            info_cur = np.log(cfg.n_classes) + torch.nn.Softmax(dim=1)(out)[rows, labels].log()
            info.append(info_cur)
            infos = infos + torch.mean(info_cur).item() * batch_size
    info = torch.cat(info, dim=0).detach()
    return info


def LSMI_estimation(dataloader, discriminator, entropy_estimator, cfg=None):
    I_X1Y = get_mutual_info(dataloader, discriminator[0], modality='modality_1', cfg=cfg)
    I_X2Y = get_mutual_info(dataloader, discriminator[1], modality='modality_2', cfg=cfg)
    I_X1X2Y = get_mutual_info(dataloader, discriminator[2], modality='modality_12', cfg=cfg)
    H_X1 = get_entropy(dataloader, entropy_estimator[0], modality='modality_1', cfg=cfg)
    H_X2 = get_entropy(dataloader, entropy_estimator[1], modality='modality_2', cfg=cfg)

    r_plus = torch.minimum(H_X1, H_X2)
    r_minus = torch.minimum(H_X1 - I_X1Y, H_X2 - I_X2Y)

    r = r_plus - r_minus
    u_1 = I_X1Y - r
    u_2 = I_X2Y - r
    s = I_X1X2Y - r - u_1 - u_2
    r_adjusted, u_1_adjusted, u_2_adjusted, s_adjusted = RUS_adjustment([r, u_1, u_2, s])

    R = torch.mean(r_adjusted)
    U_1 = torch.mean(u_1_adjusted)
    U_2 = torch.mean(u_2_adjusted)
    S = torch.mean(s_adjusted)

    print(f"R: {R.item():.4f}, U1: {U_1.item():.4f}, U2: {U_2.item():.4f}, S: {S.item():.4f}")

    return r, u_1, u_2, s


def obtain_discriminator(cfg, train_loader):
    model_1 = cls_network(input_dim=cfg.input_size_1, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(
        cfg.device)
    model_2 = cls_network(input_dim=cfg.input_size_2, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(
        cfg.device)
    model_j = cls_network(input_dim=cfg.input_size_1 + cfg.input_size_2, hidden_dim=cfg.embed_size,
                          output_dim=cfg.n_classes).to(cfg.device)
    models = [model_1, model_2, model_j]
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = cfg.num_epochs_discriminator
    for epoch in range(num_epochs):
        losses = 0.0
        num_samples = 0
        for batch in train_loader:
            modal_1, modal_2, labels = obtain_feature_input(batch, device=cfg.device)
            batch_size = modal_1.shape[0]
            out_1 = models[0](modal_1)
            out_2 = models[1](modal_2)
            out_j = models[2](torch.cat([modal_1, modal_2], dim=1))
            optimizer.zero_grad()
            loss_1 = criterion(out_1, labels)
            loss_2 = criterion(out_2, labels)
            loss_j = criterion(out_j, labels)
            loss = loss_1 + loss_2 + loss_j
            loss.backward()
            optimizer.step()
            losses += loss.item() * batch_size
            num_samples += batch_size
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses / num_samples:.4f}')

    return models


def obtain_entropy_estimator(cfg, train_loader):
    model_1 = MargKernel(dim=cfg.input_size_1).to(cfg.device)
    model_2 = MargKernel(dim=cfg.input_size_2).to(cfg.device)
    models = [model_1, model_2]
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    num_epochs = cfg.num_epochs_entropy_estimator
    for epoch in range(num_epochs):
        for model in models:
            model.train()
        losses = 0.0
        for batch in train_loader:
            modal_1, modal_2, _ = obtain_feature_input(batch, device=cfg.device)
            batch_size = modal_1.shape[0]
            loss_1 = model_1(modal_1)
            loss_2 = model_2(modal_2)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            losses += loss.item() * batch_size
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses / len(train_loader.dataset):.4f}')

    return models


def estimation_main(cfg, feature_dir=None):
    train_loader, val_loader = get_loader(cfg, feature_dir)
    discriminator = obtain_discriminator(cfg, train_loader=train_loader)
    entropy_estimator = obtain_entropy_estimator(cfg, train_loader=train_loader)
    LSMI_estimation(train_loader, discriminator, entropy_estimator, cfg)
    LSMI_estimation(val_loader, discriminator, entropy_estimator, cfg)


softmax = torch.nn.Softmax(dim=-1)

def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):

    probs = torch.clamp(probs, min=eps, max=1.0)

    log_probs = torch.log(probs)

    ce = -log_probs[torch.arange(targets.shape[0]), targets.long()].mean()

    acc = (probs.argmax(dim=1) == targets).float().mean()

    return acc.item(), ce.item()

def test_unit(model, device, loader, unimodal=None):

    model.eval()

    total_acc = 0
    total_ce = 0
    total_n = 0

    with torch.no_grad():

        for x1, x2, y in loader:

            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            if unimodal is None:
                logits, _, _ = model(x1, x2)
            else:
                logits = model(x1, x2, unimodal)

            probs = torch.exp(logits)

            acc, ce = traditional_cross_entropy_from_probs(probs, y)

            batch_size = y.size(0)

            total_acc += acc * batch_size
            total_ce += ce * batch_size
            total_n += batch_size

    avg_acc = total_acc / total_n
    avg_ce = total_ce / total_n

    print(f"[{unimodal if unimodal else 'joint'} testset] CE: {avg_ce:.4f}, Accuracy: {avg_acc:.4f}")

    return avg_acc, avg_ce

class XNORDataset(Dataset):
    def __init__(self, n_samples=10000):
        self.x1 = torch.randint(0, 2, (n_samples, 1)).float()
        self.x2 = torch.randint(0, 2, (n_samples, 1)).float()

        self.y = (self.x1 == self.x2).long().squeeze()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx]

class LogicNet(nn.Module):

    def __init__(self, repr_dim=8):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, repr_dim)
        )

        self.enc2 = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, repr_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2 * repr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2)
        )

        self.head1 = nn.Linear(repr_dim, 2)
        self.head2 = nn.Linear(repr_dim, 2)

    def forward(self, x1, x2, unimodal=None):

        z1 = self.enc1(x1)
        z2 = self.enc2(x2)

        if unimodal == "modality0":
            return F.log_softmax(self.head1(z1), dim=-1)

        if unimodal == "modality1":
            return F.log_softmax(self.head2(z2), dim=-1)

        z = torch.cat([z1, z2], dim=-1)

        joint_logits = self.classifier(z)

        return (
            F.log_softmax(joint_logits, dim=-1),
            F.log_softmax(self.head1(z1), dim=-1),
            F.log_softmax(self.head2(z2), dim=-1),
        )

    def get_representations(self, x1, x2):
        return self.enc1(x1), self.enc2(x2)

def train(model, loader, optimizer, device):

    model.train()

    for x1, x2, y in loader:

        x1 = x1.to(device)
        x2 = x2.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        joint, m1, m2 = model(x1, x2)

        loss = F.nll_loss(joint, y)
        loss += F.nll_loss(m1, y)
        loss += F.nll_loss(m2, y)

        loss.backward()
        optimizer.step()

def extract_representations(model, loader, device):

    model.eval()

    z1_list = []
    z2_list = []
    y_list = []

    with torch.no_grad():

        for x1, x2, y in loader:

            x1 = x1.to(device)
            x2 = x2.to(device)

            z1, z2 = model.get_representations(x1, x2)

            z1_list.append(z1.cpu())
            z2_list.append(z2.cpu())
            y_list.append(y)

    return (
        torch.cat(z1_list),
        torch.cat(z2_list),
        torch.cat(y_list),
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = XNORDataset(20000)
test_set = XNORDataset(5000)

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512)

model = LogicNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(50):
    train(model, train_loader, optimizer, device)

train_z1, train_z2, y_train = extract_representations(model, train_loader, device)
test_z1, test_z2, y_test = extract_representations(model, test_loader, device)

# ----- evaluate joint and unimodal performances -----

joint_acc, joint_ce = test_unit(model, device, test_loader)
m1_acc, m1_ce = test_unit(model, device, test_loader, "modality0")
m2_acc, m2_ce = test_unit(model, device, test_loader, "modality1")

print("Joint ce: " + str(joint_ce) + " - " +
      "Mod1 ce: " + str(m1_ce) + " - " +
      "Mod2 ce: " + str(m2_ce))

print("Joint acc: " + str(joint_acc) + " - " +
      "Mod1 acc: " + str(m1_acc) + " - " +
      "Mod2 acc: " + str(m2_acc))
# -------------------------
# Argparse
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--out-pt", type=str, default="lsmi_data.pt")
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--embed-size", type=int, default=32)
parser.add_argument("--epochs-disc", type=int, default=50)
parser.add_argument("--epochs-entropy", type=int, default=50)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# -------------------------
# Save .pt for LSMI
# -------------------------
lsmi_data = {
    "train_modal_1_features": train_z1.float(),
    "train_modal_2_features": train_z2.float(),
    "train_targets": y_train,
    "val_modal_1_features": test_z1.float(),
    "val_modal_2_features": test_z2.float(),
    "val_targets": y_test,
}

torch.save(lsmi_data, "lsmi_data.pt")
print(f"[✓] Saved LSMI data to {args.out_pt}")

# -------------------------
# Fake cfg replacement
# -------------------------
class CFG:
    pass

cfg = CFG()
cfg.device = torch.device(args.device)
cfg.batch_size = args.batch_size
cfg.num_workers = args.num_workers
cfg.embed_size = args.embed_size
cfg.n_classes = 2
cfg.num_epochs_discriminator = args.epochs_disc
cfg.num_epochs_entropy_estimator = args.epochs_entropy

cfg.input_size_1 = train_z1.shape[1]
cfg.input_size_2 = train_z2.shape[1]

setup_seed(args.seed)

# -------------------------
# Load data
# -------------------------
train_loader, val_loader = get_loader(cfg, args.out_pt)

# -------------------------
# Train estimators
# -------------------------
discriminator = obtain_discriminator(cfg, train_loader)
entropy_estimator = obtain_entropy_estimator(cfg, train_loader)

# -------------------------
# LSMI (PID)
# -------------------------
print("\nTrain PID:")
LSMI_estimation(train_loader, discriminator, entropy_estimator, cfg)

print("\nValidation PID:")
r, u1, u2, s = LSMI_estimation(val_loader, discriminator, entropy_estimator, cfg)

print((100*r[:10]).int().numpy())
print((100*u1[:10]).int().numpy())
print((100*u2[:10]).int().numpy())
print((100*s[:10]).int().numpy())
print(train_set.x1[:10, 0])
print(train_set.x2[:10, 0])
