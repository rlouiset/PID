"""
Batch (CE-Alignment) PID on *latent representations* of AV-MNIST.

Global-only estimator (no pointwise output): trains discriminators p(y|z1),
p(y|z2), p(y|z1,z2) plus a Sinkhorn-normalized alignment Q(z2|z1,y) on learned
CNN embeddings, then reads off R/U1/U2/S from the resulting mutual-information
terms. Adapted from logic_circuit/or_batch.py.

Shares the CNN-training / representation-extraction code with
avmnist_repr_lsmi.py (same args, same architecture, same LR schedule) so the
three AV-MNIST estimators -- CCS (avmnist.py), LSMI (avmnist_repr_lsmi.py),
and Batch (here) -- run on directly comparable representations for global PID.
"""
from __future__ import print_function
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from model import CNN
from dataset import AV_dataset
from utils import pad_or_crop, softmax, traditional_cross_entropy_from_probs
from utils_lsmi import feature_dataset, setup_seed

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose


# ─── Args ────────────────────────────────────────────────────────────────────

def config():
    parser = argparse.ArgumentParser(description='Batch (CE-Alignment) PID on AV-MNIST latent representations')
    parser.add_argument('--batch-size',           type=int,   default=1024)
    parser.add_argument('--test-batch-size',      type=int,   default=1024)
    parser.add_argument('--epoch',                type=int,   default=30,
                        help='Epochs to train the CNN encoder')
    parser.add_argument('--lr',                   type=float, default=0.001)
    parser.add_argument('--log-interval',         type=int,   default=30)
    parser.add_argument('--seed',                 type=int,   default=1)
    # Batch / CE-alignment
    parser.add_argument('--ce-hidden-dim',        type=int,   default=128,
                        help='Hidden dim of the CE-alignment discriminators (mirrors --lsmi-embed-size)')
    parser.add_argument('--ce-embed-dim',         type=int,   default=10,
                        help='Per-class embedding dim used by the alignment module')
    parser.add_argument('--ce-bs',                type=int,   default=1024,
                        help='Batch size for discriminator / alignment training (mirrors --lsmi-bs)')
    parser.add_argument('--epochs-discriminator', type=int,   default=30)
    parser.add_argument('--epochs-ce',            type=int,   default=10,
                        help='Epochs to train the alignment module')
    return parser

args = config().parse_args()
setup_seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data loading (identical to avmnist_repr_lsmi.py) ─────────────────────────

def load_fsdd():
    from torchfsdd import TorchFSDDGenerator, TrimSilence
    audio_transforms = Compose([
        TrimSilence(threshold=1e-6),
        MelSpectrogram(sample_rate=8000, n_mels=64, n_fft=512, hop_length=128),
        AmplitudeToDB(),
        pad_or_crop,
    ])
    fsdd = TorchFSDDGenerator(
        version='local',
        path='/lustre/fswork/projects/rech/haj/uik24xv/datasets/torch-fsdd/lib/test/data/v1.0.10', # '/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10',
        transforms=audio_transforms, load_all=True,
    )
    return fsdd.train_test_split(test_size=0.2)


def prepare_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    v_train = datasets.MNIST('/lustre/fswork/projects/rech/haj/uik24xv/datasets/MNIST/', train=True,  download=False, transform=transform)
    v_test  = datasets.MNIST('/lustre/fswork/projects/rech/haj/uik24xv/datasets/MNIST/', train=False,                transform=transform)
    a_train, a_test = load_fsdd()

    loader_kwargs = dict(num_workers=0, pin_memory=True, drop_last=False)
    train_loader = DataLoader(AV_dataset(v_train, a_train),
                              batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(AV_dataset(v_test,  a_test),
                              batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


# ─── CNN training (identical to avmnist_repr_lsmi.py) ─────────────────────────

def train_epoch(model, loader, optimizer, epoch):
    model.train()
    last_loss = 0.0
    for batch_idx, (imgs, audios, labels) in enumerate(loader):
        imgs, audios, labels = imgs.to(device), audios.to(device), labels.to(device)
        optimizer.zero_grad()
        out, out_img, out_aud = model(imgs, audios, unimodal="train")
        loss = F.nll_loss(out, labels) + F.nll_loss(out_img, labels) + F.nll_loss(out_aud, labels)
        loss.backward()
        optimizer.step()
        last_loss = loss.item()
        if batch_idx % args.log_interval == 0:
            print(f'  Epoch {epoch} [{batch_idx * len(imgs)}/{len(loader.dataset)}]'
                  f'  loss={loss.item():.4f}')
    return last_loss


def evaluate(model, loader):
    joint_acc,  joint_ce,  _ = test_unit(model, loader)
    visual_acc, visual_ce, _ = test_unit(model, loader, 'visual')
    audio_acc,  audio_ce,  _ = test_unit(model, loader, 'audio')
    print(f"  Joint  acc={joint_acc:.4f}  CE={joint_ce:.4f}")
    print(f"  Visual acc={visual_acc:.4f}  CE={visual_ce:.4f}")
    print(f"  Audio  acc={audio_acc:.4f}  CE={audio_ce:.4f}")


def test_unit(model, loader, unimodal=None):
    model.eval()
    total_acc = total_ce = total_n = 0
    probs_list = []
    with torch.no_grad():
        for imgs, audios, labels in loader:
            imgs, audios, labels = imgs.to(device), audios.to(device), labels.to(device)
            logits = model(imgs, audios) if unimodal is None else model(imgs, audios, unimodal)
            probs  = torch.exp(logits)
            probs_list.append(probs.cpu())
            acc, ce = traditional_cross_entropy_from_probs(probs, labels)
            total_acc += acc * len(labels); total_ce += ce * len(labels); total_n += len(labels)
    return total_acc / total_n, total_ce / total_n, torch.cat(probs_list)


def extract_representations(model, loader):
    """Extract 128-d embeddings from the CNN (before classification heads)."""
    model.eval()
    img_list, aud_list, label_list = [], [], []
    with torch.no_grad():
        for imgs, audios, labels in loader:
            img_repr, aud_repr = model.get_representations(imgs.to(device), audios.to(device))
            img_list.append(img_repr.cpu())
            aud_list.append(aud_repr.cpu())
            label_list.append(labels)
    return torch.cat(img_list).float(), torch.cat(aud_list).float(), torch.cat(label_list)


# ─── CE-Alignment batch estimator (adapted from logic_circuit/or_batch.py) ────

def mlp(dim, hidden_dim, output_dim, layers, activation):
    act = {'relu': nn.ReLU, 'tanh': nn.Tanh}[activation]
    seq = [nn.Linear(dim, hidden_dim), act()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), act()]
    seq += [nn.Linear(hidden_dim, output_dim)]
    return nn.Sequential(*seq)


class Discrim(nn.Module):
    def __init__(self, x_dim, hidden_dim, num_labels, layers, activation):
        super().__init__()
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)

    def forward(self, *x):
        x = torch.cat(x, dim=-1)
        return self.mlp(x)


class JointDiscrim(nn.Module):
    """Same as Discrim, but also accepts a single [x1, x2] list/tuple argument."""
    def __init__(self, x_dim, hidden_dim, num_labels, layers, activation):
        super().__init__()
        self.net = mlp(x_dim, hidden_dim, num_labels, layers, activation)

    def forward(self, *args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            x = torch.cat(args[0], dim=-1)
        else:
            x = torch.cat(args, dim=-1)
        return self.net(x)


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


def train_discrim_simple(model, dataloader, epochs, lr, mode):
    """mode: 'x1', 'x2', or 'joint'"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
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


def train_ce_alignment(model, dataloader, epochs, lr):
    opt = optim.Adam(model.align_parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
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
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            _, result, _ = model(x1, x2, y)
            results.append(result)
    return torch.stack(results, dim=0)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    n_classes = 10

    # ========= 1. DATA =========
    print("Loading AV-MNIST...")
    AV_train, AV_test = prepare_dataset()

    # ========= 2. TRAIN CNN =========
    model     = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print("\nTraining CNN encoder...")
    for epoch in range(1, args.epoch + 1):
        print(f"\n=== Epoch {epoch} ===")
        train_epoch(model, AV_train, optimizer, epoch)
        scheduler.step()

    print("\nTest performance:")
    evaluate(model, AV_test)

    # ========= 3. EXTRACT REPRESENTATIONS =========
    # z1 = 128-d image embedding,  z2 = 128-d audio embedding
    print("\nExtracting representations...")
    train_z1, train_z2, y_train = extract_representations(model, AV_train)
    test_z1,  test_z2,  y_test  = extract_representations(model, AV_test)
    print(f"  Image repr: {train_z1.shape}   Audio repr: {train_z2.shape}")

    feat_dim = train_z1.shape[1]  # 128

    # ========= 4. BUILD CE-ALIGNMENT FEATURE LOADERS =========
    ce_train = DataLoader(
        feature_dataset(train_z1, train_z2, y_train),
        batch_size=args.ce_bs, shuffle=True,  num_workers=0)
    ce_test  = DataLoader(
        feature_dataset(test_z1,  test_z2,  y_test),
        batch_size=args.ce_bs, shuffle=False, num_workers=0)

    # ========= 5. TRAIN DISCRIMINATORS p(y|z1), p(y|z2), p(y|z1,z2) =========
    print("\nTraining discriminators...")
    discrim_1  = Discrim(feat_dim, args.ce_hidden_dim, n_classes, layers=2, activation='relu').to(device)
    discrim_2  = Discrim(feat_dim, args.ce_hidden_dim, n_classes, layers=2, activation='relu').to(device)
    discrim_12 = JointDiscrim(2 * feat_dim, args.ce_hidden_dim, n_classes, layers=2, activation='relu').to(device)

    train_discrim_simple(discrim_1,  ce_train, epochs=args.epochs_discriminator, lr=1e-3, mode='x1')
    train_discrim_simple(discrim_2,  ce_train, epochs=args.epochs_discriminator, lr=1e-3, mode='x2')
    train_discrim_simple(discrim_12, ce_train, epochs=args.epochs_discriminator, lr=1e-3, mode='joint')

    # ========= 6. ESTIMATE p(y) =========
    p_y = torch.bincount(y_train, minlength=n_classes).float()
    p_y /= p_y.sum()
    p_y = p_y.to(device)
    print(f"p(y) = {p_y}")

    # ========= 7. TRAIN CE-ALIGNMENT =========
    print("\nTraining CE alignment...")
    ce_model = CEAlignmentInformation(
        x1_dim=feat_dim, x2_dim=feat_dim,
        hidden_dim=args.ce_hidden_dim, embed_dim=args.ce_embed_dim, num_labels=n_classes,
        layers=2, activation='relu',
        discrim_1=discrim_1, discrim_2=discrim_2, discrim_12=discrim_12,
        p_y=p_y,
    ).to(device)

    train_ce_alignment(ce_model, ce_train, epochs=args.epochs_ce, lr=1e-3)

    # ========= 8. GLOBAL PID =========
    print("\nEvaluating...")
    results = eval_ce_alignment(ce_model, ce_test)

    res = results.cpu().numpy()
    values = np.mean(res, axis=0)
    values = values / np.log(2)  # nats → bits
    values = np.maximum(values, 0)

    print(f"\n=== Batch (CE-Alignment) PID for AV-MNIST ===")
    print(f"Redundancy:   {values[0]:.4f} bits")
    print(f"Unique image: {values[1]:.4f} bits")
    print(f"Unique audio: {values[2]:.4f} bits")
    print(f"Synergy:      {values[3]:.4f} bits")