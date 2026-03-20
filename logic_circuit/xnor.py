import torch
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

from utils_ours_source import return_redundancy_test_performances, compute_PID_categorical

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

X_train_dict = {
    "modality0": train_z1.float(),
    "modality1": train_z2.float()
}

X_test_dict = {
    "modality0": test_z1.float(),
    "modality1": test_z2.float()
}

y_pred_dict = return_redundancy_test_performances(
    X_train_dict,
    X_test_dict,
    X_test_dict,
    y_train,
    y_test,
    y_test,
    "redundancy",
    distribution_target="categorical",
    lambda_reg=10, num_classes=2, h_dim=1024
)

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


# ----- evaluate redundancy predictor -----

results = {}

for key in ["modality0", "modality1", "average"]:

    acc_, ce_ = traditional_cross_entropy_from_probs(
        softmax(y_pred_dict[key]),
        y_pred_dict["targets"]
    )

    results[key] = {
        "accuracy": acc_,
        "cross_entropy": ce_
    }

for k, v in results.items():
    print(
        "redundancy representations - "
        + f"{k:10s} | acc = {v['accuracy']:.4f}, CE = {v['cross_entropy']:.4f}"
    )

print(torch.sum(y_test==0))
print(torch.sum(y_test==1))

# ----- compute PID -----

compute_PID_categorical(
    joint_ce,                      # H(Y|Z1,Z2)
    m1_ce,                         # H(Y|Z1)
    m2_ce,                         # H(Y|Z2)
    results["average"]["cross_entropy"],   # H(Y|ZR)
    num_classes=2
)
