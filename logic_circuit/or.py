import torch
from torch.utils.data import Dataset

import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from torch.utils.data import DataLoader
import torch.optim as optim

from utils_ours import return_redundancy_test_performances

softmax = torch.nn.Softmax(dim=-1)

def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):

    probs = torch.clamp(probs, min=eps, max=1.0)

    log_probs = torch.log(probs)

    ce = -log_probs[torch.arange(targets.shape[0]), targets.long()].mean()

    acc = (probs.argmax(dim=1) == targets).float().mean()

    return acc.item(), ce.item()

def compute_log_py(targets, num_classes):
    """
    Compute log p(y) for each sample
    """
    counts = torch.bincount(targets, minlength=num_classes).float()
    probs = counts / counts.sum()
    probs = torch.clamp(probs, 1e-12, 1.0)

    log_py_all = torch.log(probs)  # shape (C,)

    # map to each sample
    return log_py_all[targets]     # shape (N,)

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


def test_unit(model, device, loader, unimodal=None):

    model.eval()

    total_acc = 0
    total_ce = 0
    total_n = 0

    probs_list = []

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

            probs_list.extend(list(probs.detach().cpu().numpy()))

            acc, ce = traditional_cross_entropy_from_probs(probs, y)

            batch_size = y.size(0)

            total_acc += acc * batch_size
            total_ce += ce * batch_size
            total_n += batch_size

    avg_acc = total_acc / total_n
    avg_ce = total_ce / total_n

    print(f"[{unimodal if unimodal else 'joint'} testset] CE: {avg_ce:.4f}, Accuracy: {avg_acc:.4f}")

    return avg_acc, avg_ce, torch.tensor(probs_list)

class ORDataset(Dataset):
    def __init__(self, n_samples=10000):
        self.x1 = torch.randint(0, 2, (n_samples, 1)).float()
        self.x2 = torch.randint(0, 2, (n_samples, 1)).float()

        self.y = ((self.x1 + self.x2) > 0).long().squeeze()

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

def compute_entropy_from_labels(y, num_classes):
    counts = torch.bincount(y, minlength=num_classes).float()
    probs = counts / counts.sum()
    probs = torch.clamp(probs, 1e-12, 1.0)
    return -torch.sum(probs * torch.log(probs)).item()

def ce_per_sample(targets, probs, eps=1e-12):
    """
    Per-sample cross-entropy:
        CE(x) = -log p(y|x)
    """
    probs = torch.clamp(probs, eps, 1.0)
    return -torch.log(probs[torch.arange(len(targets)), targets])

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

def compute_ce_from_probs(probs_list, targets):
    """
    From logits → probabilities + CE for each modality
    """
    ce_list = [ce_per_sample(targets, probs) for probs in probs_list]
    return ce_list

# ---------- PID ----------
def compute_pointwise_pid_from_probs(dict_of_metrics, num_classes):
    """ Compute per-sample PID: total = redundancy + unique_0 + unique_1 + synergy """
    targets = dict_of_metrics["true_labels"].long()
    log_py = compute_log_py(targets, num_classes)  # (N,)

    pid_list = []

    for i, (j, m0, m1, npr, y, log_py_i) in enumerate(zip(
        dict_of_metrics["probs_joint"],
        dict_of_metrics["probs_modalities"][0],
        dict_of_metrics["probs_modalities"][1],
        dict_of_metrics["redundancy_pointwise_ce"],
        targets,
        log_py
    )):
        y = y.long()

        # log p(y|x)
        pj = logp(j)[y]
        pm0 = logp(m0)[y]
        pm1 = logp(m1)[y]

        # CE = -log p(y|x)
        joint_ce = -pj
        modality0_ce = -pm0
        modality1_ce = -pm1
        redundancy_ce = npr

        # per-sample entropy H(Y=y) = -log p(y)
        h_y = -log_py_i

        # ===== CLIPPING =====
        modality0_ce = min(modality0_ce, h_y)
        modality1_ce = min(modality1_ce, h_y)
        redundancy_ce = min(redundancy_ce, h_y)

        redundancy_ce = max(redundancy_ce, joint_ce, modality0_ce, modality1_ce)

        modality0_ce = max(modality0_ce, joint_ce)
        modality1_ce = max(modality1_ce, joint_ce)

        """print("m0-: ", modality0_ce)
        print("m1-: ", modality1_ce)"""

        modality0_ce = min(modality0_ce, redundancy_ce)
        modality1_ce = min(modality1_ce, redundancy_ce)

        # ===== INFORMATION =====
        total = h_y - joint_ce

        r_val = h_y - redundancy_ce

        u0 = h_y - modality0_ce - r_val # max(0, h_y - modality0_ce - r_val)
        u1 = h_y - modality1_ce - r_val # max(0, h_y - modality1_ce - r_val)

        s = total - u0 - u1 - r_val

        """print("m0+: ", modality0_ce)
        print("m1+: ", modality1_ce)
        print("r: ", redundancy_ce)
        print("hy: ", h_y)
        print("u0: ", u0)
        print("u1: ", u1)
        print('')

        if i > 10:
            print(debug)"""

        """if s < 0:
            r_val -= s
            u0 = h_y - modality0_ce - r_val # max(0, h_y - modality0_ce - r_val)
            u1 = h_y - modality1_ce - r_val # max(0, h_y - modality1_ce - r_val)
            s = 0"""

        pid_list.append([u0, u1, r_val, s])

    return np.array(pid_list)

def compute_pointwise_pid_with_source_from_probs(dict_of_metrics, num_classes):
    """ Compute per-sample PID: total = redundancy (x% Source) + unique_0 + unique_1 + synergy """
    targets = dict_of_metrics["true_labels"].long()
    log_py = compute_log_py(targets, num_classes)

    pid_list = []

    for i, (j, m0, m1, npr, r_src, y, log_py_i) in enumerate(zip(
        dict_of_metrics["probs_joint"],
        dict_of_metrics["probs_modalities"][0],
        dict_of_metrics["probs_modalities"][1],
        dict_of_metrics["redundancy_pointwise_ce"],
        dict_of_metrics["source_redundancy_preds"],
        targets,
        log_py
    )):
        y = y.long()

        pj = logp(j)[y]
        pm0 = logp(m0)[y]
        pm1 = logp(m1)[y]
        pr_src = logp(F.softmax(r_src, dim=0))[y]

        joint_ce = -pj
        modality0_ce = -pm0
        modality1_ce = -pm1
        redundancy_ce = npr
        source_redundancy_ce = -pr_src

        h_y = -log_py_i

        # ===== CLIPPING =====
        modality0_ce = min(modality0_ce, h_y)
        modality1_ce = min(modality1_ce, h_y)
        redundancy_ce = min(redundancy_ce, h_y)
        source_redundancy_ce = min(source_redundancy_ce, h_y)

        redundancy_ce = max(redundancy_ce, joint_ce, modality0_ce, modality1_ce)
        source_redundancy_ce = max(source_redundancy_ce, joint_ce, modality0_ce, modality1_ce)

        modality0_ce = max(modality0_ce, joint_ce)
        modality1_ce = max(modality1_ce, joint_ce)

        modality0_ce = min(modality0_ce, redundancy_ce)
        modality1_ce = min(modality1_ce, redundancy_ce)

        # ===== INFORMATION =====
        total = h_y - joint_ce

        # your design choice: strongest redundancy
        r_val = max(h_y - redundancy_ce, h_y - source_redundancy_ce)

        u0 = max(0, h_y - modality0_ce - r_val)
        u1 = max(0, h_y - modality1_ce - r_val)

        s = total - u0 - u1 - r_val

        """if s < 0:
            r_val -= s
            u0 = h_y - modality0_ce - r_val # max(0, h_y - modality0_ce - r_val)
            u1 = h_y - modality1_ce - r_val # max(0, h_y - modality1_ce - r_val)
            s = 0"""

        pid_list.append([u0, u1, r_val, s])

    return np.array(pid_list)

def compute_PID_categorical_with_source_decomposition(
    joint_ce,
    modality0_ce,
    modality1_ce,
    redundancy_ce,
    source_redundancy_ce,
    num_classes,
    targets
):
    import torch

    # ===== 1. TRUE GLOBAL ENTROPY =====
    H_Y = compute_entropy_from_targets(targets, num_classes)

    print("H(Y)", H_Y)
    print("joint_ce", joint_ce)
    print("redundancy_ce", redundancy_ce)
    print("source redundancy_ce", source_redundancy_ce)
    print("modality0_ce", modality0_ce)
    print("modality1_ce", modality1_ce)
    print('')

    # ===== 2. CLIP using H(Y) =====
    modality0_ce = min(modality0_ce, H_Y)
    modality1_ce = min(modality1_ce, H_Y)
    redundancy_ce = min(redundancy_ce, H_Y)
    source_redundancy_ce = min(source_redundancy_ce, H_Y)

    # ===== 3. YOUR STRUCTURAL CONSTRAINTS =====
    redundancy_ce = max(redundancy_ce, joint_ce, modality0_ce, modality1_ce)
    source_redundancy_ce = max(source_redundancy_ce, joint_ce, modality0_ce, modality1_ce)

    # keep only shared redundancy
    redundancy_ce = min(redundancy_ce, source_redundancy_ce)

    modality0_ce = max(modality0_ce, joint_ce)
    modality1_ce = max(modality1_ce, joint_ce)

    modality0_ce = min(modality0_ce, redundancy_ce)
    modality1_ce = min(modality1_ce, redundancy_ce)

    # ===== 4. INFORMATION TERMS (FIXED) =====
    I = H_Y - joint_ce

    I_R = H_Y - redundancy_ce
    I_R_source = H_Y - source_redundancy_ce

    I_U0 = (H_Y - modality0_ce) - I_R
    I_U1 = (H_Y - modality1_ce) - I_R

    I_S = I - I_U0 - I_U1 - I_R

    # ===== 5. NON-NEGATIVITY =====
    if I_S < 0:
        I_R -= I_S
        I_R_source -= I_S
        I_U0 = (H_Y - modality0_ce) - I_R
        I_U1 = (H_Y - modality1_ce) - I_R
        I_S = 0

    ratio_source = I_R_source / (I_R + 1e-10)

    print("R=" + str(I_R)[:5] + "(" + str(100*ratio_source)[:5] + "% Source)")
    print("U0=", str(I_U0)[:5])
    print("U1=", str(I_U1)[:5])
    print("S=", str(I_S)[:5])
    print("I=", str(I)[:5])


def compute_pointwise_information(ce_list, log_py):
    """
    PMI proxy:
        i = log p(y|x) - log p(y)
          = -CE - log p(y)
    ce: (N,)
    log_py: (N,)
    """
    return [-ce - log_py for ce in ce_list]

# ---------- CCS REDUNDANCY ----------
def compute_ccs_and_selection(ce_list, same_sign, log_py):
    """
    Compute:
        - CCS redundancy

    Rule:
        if sign agreement:
            redundancy = worst CE
            prediction = worst modality
        else:
            redundancy = baseline (-log p(y))
            prediction = uniform
    """
    ce_stack = torch.stack(ce_list, dim=1)        # (N, 2)

    # Worst modality (higher CE)
    worst_ce = torch.max(ce_stack, dim=1).values  # (N,)

    # Baseline (independence)
    baseline = -log_py  # (N,)

    # CCS redundancy
    ccs = torch.where(same_sign, worst_ce, baseline)

    return ccs

def logp(p):
    return torch.log(torch.clamp(p, 1e-12, 1.0))

def compute_entropy_from_targets(targets, num_classes):
    import torch

    targets = torch.tensor(targets).long()
    counts = torch.bincount(targets, minlength=num_classes).float()
    probs = counts / counts.sum()
    probs = torch.clamp(probs, 1e-12, 1.0)

    return -torch.sum(probs * torch.log(probs)).item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = ORDataset(20000)
test_set = ORDataset(5000)

num_classes = 2

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
test_loader = DataLoader(test_set, batch_size=512)

model = LogicNet().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    train(model, train_loader, optimizer, device)

train_z1, train_z2, y_train = extract_representations(model, train_loader, device)
test_z1, test_z2, y_test = extract_representations(model, test_loader, device)

# ----- evaluate joint and unimodal performances -----

joint_acc, joint_ce, joint_probs = test_unit(model, device, test_loader)
m1_acc, m1_ce, m1_probs = test_unit(model, device, test_loader, "modality0")
m2_acc, m2_ce, m2_probs = test_unit(model, device, test_loader, "modality1")

print("Joint ce: " + str(joint_ce) + " - " +
      "Mod1 ce: " + str(m1_ce) + " - " +
      "Mod2 ce: " + str(m2_ce))

print("Joint acc: " + str(joint_acc) + " - " +
      "Mod1 acc: " + str(m1_acc) + " - " +
      "Mod2 acc: " + str(m2_acc))

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
    num_classes=num_classes, h_dim=1024
)


log_py = compute_log_py(y_test, num_classes)
logits_list = [m1_probs, m2_probs]

ce_list = compute_ce_from_probs(logits_list, y_test)
i_list = compute_pointwise_information(ce_list, log_py)

same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])

ccs = compute_ccs_and_selection(
    ce_list, same_sign, log_py
)

redundancy_ce = ccs.mean().item()
redundancy_pointwise_ce = ccs.numpy()

dict_of_metrics = {"joint_ce": joint_ce,
                   "joint_acc":joint_acc,
                   "probs_joint":joint_probs,
                   "modalities_ce": [m1_ce, m2_ce],
                   "modalities_acc": [m1_acc, m2_acc],
                   "probs_modalities": [m1_probs, m2_probs],
                   "redundancy_ce": redundancy_ce,
                   "redundancy_pointwise_ce": redundancy_pointwise_ce,
                   "true_labels": y_test,}


def compute_redundancy_metrics(y_pred_dict):
    results = {}
    for key in ["modality0", "modality1", "average"]:
        acc, ce = traditional_cross_entropy_from_probs(
            softmax(y_pred_dict[key]),
            y_pred_dict["targets"]
        )
        results[key] = {"accuracy": acc, "cross_entropy": ce}
    return results

results = compute_redundancy_metrics(y_pred_dict)
print_model_metrics(dict_of_metrics)
print_redundancy_metrics(results)

dict_of_metrics["source_redundancy_pointwise_ce"] = results["average"]["cross_entropy"]
dict_of_metrics["source_redundancy_preds"] = y_pred_dict["average"]

# ========= 8 GLOBAL PID =========
compute_PID_categorical_with_source_decomposition(
    dict_of_metrics["joint_ce"],
    dict_of_metrics["modalities_ce"][0],
    dict_of_metrics["modalities_ce"][1],
    dict_of_metrics["redundancy_ce"],
    dict_of_metrics["source_redundancy_pointwise_ce"],
    num_classes=num_classes,
    targets=dict_of_metrics["true_labels"]
)

# ========= 9. POINTWISE PID WITH SOURCE =========
pid_source = compute_pointwise_pid_with_source_from_probs(dict_of_metrics, num_classes)
print(np.mean(pid_source, axis=0))

# ========= 9. POINTWISE PID WITHOUT SOURCE =========
pid = compute_pointwise_pid_from_probs(dict_of_metrics, num_classes)
print(np.mean(pid, axis=0))

pid = - (torch.tensor(pid) + log_py[:, None])
print(np.mean(pid.numpy(), axis=0))

compute_PID_categorical_with_source_decomposition(
    np.mean(pid.numpy(), axis=0)[-1],
    np.mean(pid.numpy(), axis=0)[0],
    np.mean(pid.numpy(), axis=0)[1],
    np.mean(pid.numpy(), axis=0)[2],
    dict_of_metrics["source_redundancy_pointwise_ce"],
    num_classes=num_classes,
    targets=dict_of_metrics["true_labels"]
)

"""print(test_set.x1[0:15, 0])
print(test_set.x2[0:15, 0])"""
"""print(pid[0:15])
print(log_py[0:15])"""
"""print(m1_probs[0:15])
print(m2_probs[0:15])"""

