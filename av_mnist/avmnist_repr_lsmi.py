"""
LSMI-based PID on *latent representations* of AV-MNIST.

Pipeline (Algorithm 1 from arxiv 2506.17248):
  1. Train CNN → extract 128-d embeddings  z1=image, z2=audio
  2. Train three discriminators on those embeddings:
       - d1(z1)      →  p(y|z1)    →  I(Z1; Y)
       - d2(z2)      →  p(y|z2)    →  I(Z2; Y)
       - dj(z1, z2)  →  p(y|z1,z2) →  I(Z1,Z2; Y)
     All via:  I(Zi; Y) ≈ E[ log p(y|zi) ] + log K  (uniform prior, K classes)
  3. Train two MargKernel models:
       - mk1  →  H(Z1)  (differential entropy of image rep distribution)
       - mk2  →  H(Z2)  (differential entropy of audio rep distribution)
  4. LSMI PID:
       R = min(H1,H2) – min(H1–I1Y, H2–I2Y)
       U1 = I1Y – R,  U2 = I2Y – R,  S = I12Y – R – U1 – U2

Note: running LSMI on latent 128-d representations (not raw 28×28 images /
spectrograms) means the discriminators are light MLPs and the entropy estimator
models the learned embedding distribution — which is more meaningful for PID.
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
from utils_lsmi import MargKernel, cls_network, feature_dataset, setup_seed

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose


# ─── Args ────────────────────────────────────────────────────────────────────

def config():
    parser = argparse.ArgumentParser(description='LSMI PID on AV-MNIST latent representations')
    parser.add_argument('--batch-size',           type=int,   default=1024)
    parser.add_argument('--test-batch-size',      type=int,   default=1024)
    parser.add_argument('--epoch',                type=int,   default=30,
                        help='Epochs to train the CNN encoder')
    parser.add_argument('--lr',                   type=float, default=0.001)
    parser.add_argument('--log-interval',         type=int,   default=30)
    parser.add_argument('--seed',                 type=int,   default=1)
    # LSMI
    parser.add_argument('--lsmi-embed-size',      type=int,   default=128,
                        help='Hidden dim of LSMI discriminators')
    parser.add_argument('--lsmi-bs',              type=int,   default=1024,
                        help='Batch size for LSMI estimator training / inference')
    parser.add_argument('--epochs-discriminator', type=int,   default=30)
    parser.add_argument('--epochs-entropy',       type=int,   default=30)
    return parser

args = config().parse_args()
setup_seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Data loading ─────────────────────────────────────────────────────────────

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
        path='/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10',
        transforms=audio_transforms, load_all=True,
    )
    return fsdd.train_test_split(test_size=0.2)


def prepare_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    v_train = datasets.MNIST('data', train=True,  download=True, transform=transform)
    v_test  = datasets.MNIST('data', train=False,                transform=transform)
    a_train, a_test = load_fsdd()

    loader_kwargs = dict(num_workers=0, pin_memory=True, drop_last=False)
    train_loader = DataLoader(AV_dataset(v_train, a_train),
                              batch_size=args.batch_size, shuffle=True,  **loader_kwargs)
    test_loader  = DataLoader(AV_dataset(v_test,  a_test),
                              batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)
    return train_loader, test_loader


# ─── CNN training ─────────────────────────────────────────────────────────────

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


# ─── Representation extraction ────────────────────────────────────────────────

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


# ─── LSMI estimators ─────────────────────────────────────────────────────────

def obtain_discriminator(train_loader, input_size_1, input_size_2, embed_size, n_classes, n_epochs):
    """
    Train three classifiers on latent representations:
      d1(z1)     → p(y|z1)    → used to estimate I(Z1; Y)
      d2(z2)     → p(y|z2)    → used to estimate I(Z2; Y)
      dj(z1, z2) → p(y|z1,z2) → used to estimate I(Z1,Z2; Y)
    Estimator: I(Zi; Y) ≈ E[log p(y|zi)] + log K  (uniform class prior)
    """
    d1 = cls_network(input_size_1,                embed_size, n_classes).to(device)
    d2 = cls_network(input_size_2,                embed_size, n_classes).to(device)
    dj = cls_network(input_size_1 + input_size_2, embed_size, n_classes).to(device)
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
            print(f"  [disc]    epoch {epoch+1}/{n_epochs}  loss={losses/num_samples:.4f}")
    return models


def obtain_entropy_estimator(train_loader, input_size_1, input_size_2, n_epochs):
    """
    Train two MargKernel models:
      mk1 → H(Z1)  (differential entropy of image representation distribution)
      mk2 → H(Z2)  (differential entropy of audio representation distribution)
    """
    mk1 = MargKernel(input_size_1).to(device)
    mk2 = MargKernel(input_size_2).to(device)
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
            print(f"  [entropy] epoch {epoch+1}/{n_epochs}  loss={losses/len(train_loader.dataset):.4f}")
    return models


def get_mutual_info(loader, model, modality, n_classes):
    """I(Zi; Y) ≈ E[ log p(y|zi) ] + log K  (K = n_classes, uniform prior)."""
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
    """H(Zi) estimated as (negative) average log-density under MargKernel."""
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            x = (batch[0] if modality == 'modality_1' else batch[1]).to(device)
            info.append(model(x))
    return torch.cat(info).detach()

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

def LSMI_estimation(loader, discriminators, entropy_estimators, n_classes, split_name=""):
    """
    Raw LSMI PID (no RUS adjustment on individual samples).
    RUS / correction is applied at distribution level in the caller.
    """
    I1Y  = get_mutual_info(loader, discriminators[0], 'modality_1',  n_classes)
    I2Y  = get_mutual_info(loader, discriminators[1], 'modality_2',  n_classes)
    I12Y = get_mutual_info(loader, discriminators[2], 'modality_12', n_classes)
    H1   = get_entropy(loader, entropy_estimators[0], 'modality_1')
    H2   = get_entropy(loader, entropy_estimators[1], 'modality_2')

    r_plus  = torch.minimum(H1, H2)
    r_minus = torch.minimum(H1 - I1Y, H2 - I2Y)
    r  = r_plus - r_minus
    u1 = I1Y  - r
    u2 = I2Y  - r
    s  = I12Y - r - u1 - u2
    r_adjusted, u_1_adjusted, u_2_adjusted, s_adjusted = RUS_adjustment([r, u1, u2, s])

    R = torch.mean(r_adjusted)
    U_1 = torch.mean(u_1_adjusted)
    U_2 = torch.mean(u_2_adjusted)
    S = torch.mean(s_adjusted)

    print(f"R: {R.item():.4f}, U1: {U_1.item():.4f}, U2: {U_2.item():.4f}, S: {S.item():.4f}")

    return r, u1, u2, s


def normalize_pid(pid):
    pid = np.maximum(pid, 0)
    pid /= pid.sum(axis=1, keepdims=True) + 1e-12
    return pid


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

    # ========= 4. BUILD LSMI FEATURE LOADERS =========
    lsmi_train = DataLoader(
        feature_dataset(train_z1, train_z2, y_train),
        batch_size=args.lsmi_bs, shuffle=True,  num_workers=0)
    lsmi_test  = DataLoader(
        feature_dataset(test_z1,  test_z2,  y_test),
        batch_size=args.lsmi_bs, shuffle=False, num_workers=0)

    feat_dim = train_z1.shape[1]  # 128

    # ========= 5. TRAIN LSMI ESTIMATORS =========
    print("\nTraining LSMI discriminators  (p(y|z1), p(y|z2), p(y|z1,z2))...")
    discriminators = obtain_discriminator(
        lsmi_train, feat_dim, feat_dim,
        embed_size=args.lsmi_embed_size,
        n_classes=n_classes,
        n_epochs=args.epochs_discriminator,
    )

    print("\nTraining LSMI entropy estimators  (H(Z1), H(Z2))...")
    entropy_estimators = obtain_entropy_estimator(
        lsmi_train, feat_dim, feat_dim,
        n_epochs=args.epochs_entropy,
    )

    # ========= 6. LSMI PID (raw) =========
    print("\nPID (RUS adjustment):")
    LSMI_estimation(lsmi_train, discriminators, entropy_estimators, n_classes, "train")
    r, u1, u2, s = LSMI_estimation(lsmi_test, discriminators, entropy_estimators, n_classes, "test")

    # ========= 7. POINTWISE DISTRIBUTION =========
    # Stack as [U_image, U_audio, R, S]
    pid = np.stack([u1.cpu().numpy(), u2.cpu().numpy(),
                    r.cpu().numpy(),  s.cpu().numpy()], axis=1)

    print("\nMean pointwise PID [U_image, U_audio, R, S] (test):")
    print(np.mean(pid, axis=0))
    pid_norm = normalize_pid(pid)
    print("Normalised mean:", np.mean(pid_norm, axis=0))

