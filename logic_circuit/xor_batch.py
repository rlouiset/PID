import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
from tqdm import tqdm

# ─── Dataset ───

class XORDataset(Dataset):

    def __init__(self, n_samples=10000, noise_std=0.1):

        self.x1 = torch.randint(0, 2, (n_samples, 1)).float()
        self.x2 = torch.randint(0, 2, (n_samples, 1)).float()

        # XOR logic
        self.y = ((self.x1 + self.x2) % 2).long().squeeze()
        self.noise_std = noise_std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]

        noise1 = torch.randn_like(x1) * self.noise_std
        noise2 = torch.randn_like(x2) * self.noise_std

        return x1 + noise1, x2 + noise2, self.y[idx]


# ─── MultimodalDataset (needed by critic_ce_alignment) ───

class MultimodalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return tuple([self.data[i][idx] for i in range(len(self.data))] + [self.labels[idx]])


# ─── MLP helper ───

def mlp(dim, hidden_dim, output_dim, layers, activation):
    act = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation]
    seq = [nn.Linear(dim, hidden_dim), act()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), act()]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)


# ─── Discriminator ───

class Discrim(nn.Module):
    def __init__(self, x_dim, hidden_dim, num_labels, layers, activation):
        super().__init__()
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)

    def forward(self, *x):
        x = torch.cat(x, dim=-1)
        return self.mlp(x)


# ─── CE Alignment ───

class CEAlignment(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation):
        super().__init__()
        self.num_labels = num_labels
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation)
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)

    def forward(self, x1, x2, x1_probs, x2_probs):
        B = x1.size(0)
        q_x1 = self.mlp1(x1).view(B, self.num_labels, -1)
        q_x2 = self.mlp2(x2).view(B, self.num_labels, -1)

        q_x1 = (q_x1 - q_x1.mean(dim=-1, keepdim=True)) / (q_x1.var(dim=-1, keepdim=True) + 1e-8).sqrt()
        q_x2 = (q_x2 - q_x2.mean(dim=-1, keepdim=True)) / (q_x2.var(dim=-1, keepdim=True) + 1e-8).sqrt()

        align_logits = torch.einsum('b y d, b z d -> b y z', q_x1, q_x2) / math.sqrt(q_x1.size(-1))
        align = torch.exp(align_logits)

        normalized = []
        for i in range(self.num_labels):
            current = align[..., i]
            for _ in range(50):
                current = current / (current.sum(dim=-1, keepdim=True) + 1e-8) * x2_probs
                current = current / (current.sum(dim=1, keepdim=True) + 1e-8) * x1_probs
            normalized.append(current)
        normalized = torch.stack(normalized, dim=-1)
        return normalized


# ─── CE Alignment Information ───

class CEAlignmentInformation(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, discrim_1, discrim_2, discrim_12, p_y):
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        self.discrim_1 = discrim_1
        self.discrim_2 = discrim_2
        self.discrim_12 = discrim_12
        for d in [self.discrim_1, self.discrim_2, self.discrim_12]:
            if isinstance(d, nn.Module):
                d.eval()
        self.register_buffer('p_y', p_y)

    def align_parameters(self):
        return list(self.align.parameters())

    def forward(self, x1, x2, y):
        with torch.no_grad():
            p_y_x1 = torch.softmax(self.discrim_1(x1), dim=-1)
            p_y_x2 = torch.softmax(self.discrim_2(x2), dim=-1)

        align = self.align(x1.flatten(1), x2.flatten(1), p_y_x1, p_y_x2)

        y_oh = nn.functional.one_hot(y.squeeze(-1).long(), num_classes=self.num_labels).float()
        self.p_y[self.p_y == 0] += 1e-8
        self.p_y[self.p_y == 1] -= 1e-8

        q_x2_x1y = align / (align.sum(dim=1, keepdim=True) + 1e-8)
        log_term = torch.log(q_x2_x1y + 1e-8) - torch.log(
            torch.einsum('aby, ay -> ab', q_x2_x1y, p_y_x1) + 1e-8
        )[:, :, None]

        loss = torch.mean(torch.sum(torch.sum(p_y_x1[:, None, :] * q_x2_x1y * log_term, dim=-1), dim=-1))

        with torch.no_grad():
            p_y_x1x2 = torch.softmax(self.discrim_12(x1, x2), dim=-1)

        p1 = p_y_x1.detach().clone().clamp(min=1e-8)
        p2 = p_y_x2.detach().clone().clamp(min=1e-8)
        p12 = p_y_x1x2.detach().clone().clamp(min=1e-8)

        mi_y_x1 = torch.mean(torch.sum(p_y_x1 * (torch.log(p1) - torch.log(self.p_y)[None]), dim=-1))
        mi_y_x2 = torch.mean(torch.sum(p_y_x2 * (torch.log(p2) - torch.log(self.p_y)[None]), dim=-1))
        mi_y_x1x2 = torch.mean(torch.sum(p_y_x1x2 * (torch.log(p12) - torch.log(self.p_y)[None]), dim=-1))

        mi_q_y_x1x2 = p_y_x1[:, None, :] * q_x2_x1y * (
            log_term + torch.log(p_y_x1 + 1e-8)[:, None, :] - torch.log(self.p_y + 1e-8)[None, None, :]
        )
        mi_q_y_x1x2 = torch.mean(torch.sum(torch.sum(mi_q_y_x1x2, dim=-1), dim=-1))

        redundancy = mi_y_x1 + mi_y_x2 - mi_q_y_x1x2
        unique1 = mi_q_y_x1x2 - mi_y_x2
        unique2 = mi_q_y_x1x2 - mi_y_x1
        synergy = mi_y_x1x2 - mi_q_y_x1x2

        return loss, torch.stack([redundancy, unique1, unique2, synergy], dim=0), align


# ─── Joint discriminator wrapper ───
# The original code passes [x1, x2] as a list to discrim_12,
# but we also need it to work when called as discrim_12(x1, x2).

class JointDiscrim(nn.Module):
    def __init__(self, x_dim, hidden_dim, num_labels, layers, activation):
        super().__init__()
        self.net = mlp(x_dim, hidden_dim, num_labels, layers, activation)

    def forward(self, *args):
        # Accept either (x1, x2) or ([x1, x2],)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            x = torch.cat(args[0], dim=-1)
        else:
            x = torch.cat(args, dim=-1)
        return self.net(x)


# ─── Training loops ───

def train_discrim_simple(model, dataloader, epochs=40, lr=1e-3, mode='joint'):
    """mode: 'x1', 'x2', or 'joint'"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in dataloader:
            optimizer.zero_grad()
            if mode == 'x1':
                logits = model(x1)
            elif mode == 'x2':
                logits = model(x2)
            else:
                logits = model(x1, x2)
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"  Discrim [{mode}] epoch {epoch+1}: loss={total_loss / len(dataloader):.4f}")
    model.eval()
    return model


def train_ce_alignment(model, dataloader, epochs=10, lr=1e-3):
    opt = optim.Adam(model.align_parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in dataloader:
            opt.zero_grad()
            loss, _, _ = model(x1, x2, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (epoch + 1) % 5 == 0:
            print(f"  CE-Align epoch {epoch+1}: loss={total_loss / len(dataloader):.4f}")
    model.eval()


def eval_ce_alignment(model, dataloader):
    model.eval()
    results = []
    with torch.no_grad():
        for x1, x2, y in dataloader:
            _, result, _ = model(x1, x2, y)
            results.append(result)
    return torch.stack(results, dim=0)


# ─── Main ───

if __name__ == "__main__":
    hidden_dim = 16
    num_classes = 2
    batch_size = 512
    n_train = 20000
    n_test = 5000
    discrim_epochs = 40
    ce_epochs = 10

    train_set = XORDataset(n_train, noise_std=0.1)
    test_set = XORDataset(n_test, noise_std=0.1)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Step 1: Train discriminators p(y|x1), p(y|x2), p(y|x1,x2)
    print("Training discriminators...")
    discrim_1 = Discrim(x_dim=1, hidden_dim=hidden_dim, num_labels=num_classes, layers=2, activation='relu')
    discrim_2 = Discrim(x_dim=1, hidden_dim=hidden_dim, num_labels=num_classes, layers=2, activation='relu')
    discrim_12 = JointDiscrim(x_dim=2, hidden_dim=hidden_dim, num_labels=num_classes, layers=2, activation='relu')

    train_discrim_simple(discrim_1, train_loader, epochs=discrim_epochs, mode='x1')
    train_discrim_simple(discrim_2, train_loader, epochs=discrim_epochs, mode='x2')
    train_discrim_simple(discrim_12, train_loader, epochs=discrim_epochs, mode='joint')

    # Step 2: Estimate p(y)
    p_y = torch.zeros(num_classes)
    for _, _, y in train_loader:
        for c in range(num_classes):
            p_y[c] += (y == c).sum()
    p_y /= p_y.sum()
    print(f"p(y) = {p_y}")

    # Step 3: Build CE-PID model and train alignment
    print("Training CE alignment...")
    model = CEAlignmentInformation(
        x1_dim=1, x2_dim=1,
        hidden_dim=hidden_dim, embed_dim=10, num_labels=num_classes,
        layers=2, activation='relu',
        discrim_1=discrim_1, discrim_2=discrim_2, discrim_12=discrim_12,
        p_y=p_y
    )

    train_ce_alignment(model, train_loader, epochs=ce_epochs, lr=1e-3)

    # Step 4: Evaluate
    print("Evaluating...")
    results = eval_ce_alignment(model, test_loader)

    res = results.cpu().numpy()
    values = np.mean(res, axis=0)
    values = values / np.log(2)  # convert nats → bits
    values = np.maximum(values, 0)

    print(f"\n=== CE-PID for OR gate ===")
    print(f"Redundancy: {values[0]:.4f} bits")
    print(f"Unique X1:  {values[1]:.4f} bits")
    print(f"Unique X2:  {values[2]:.4f} bits")
    print(f"Synergy:    {values[3]:.4f} bits")