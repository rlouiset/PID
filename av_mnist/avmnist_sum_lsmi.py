"""LSMI PID on representations learned by the AV-MNIST multimodal model.

Same CNN_sum model, AV_dataset_sum, and training protocol as avmnist_sum_update.py.
LSMI discriminators and entropy estimators are trained on the representations
extracted from the trained multimodal encoder, not on the raw inputs.
"""
from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import CNN_sum
from dataset import AV_dataset_sum
from utils import TARGET_FRAMES, softmax, traditional_cross_entropy_from_probs

from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose
from torchfsdd import TorchFSDDGenerator, TrimSilence
from math import *

from utils_lsmi import MargKernel, cls_network, get_loader, setup_seed

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# ─── Config ───────────────────────────────────────────────────────────────────

def config():
    parser = argparse.ArgumentParser(description='AV-MNIST LSMI PID on learned representations')
    parser.add_argument('--batch-size',      type=int,   default=1024)
    parser.add_argument('--test-batch-size', type=int,   default=1024)
    parser.add_argument('--epoch',           type=int,   default=50)
    parser.add_argument('--lr',              type=float, default=0.001)
    parser.add_argument('--gamma',           type=float, default=0.996)
    parser.add_argument('--seed',            type=int,   default=1)
    parser.add_argument('--log-interval',    type=int,   default=30)
    parser.add_argument('--save-model',      action='store_true', default=False)
    parser.add_argument('--depth',           type=int,   default=6)
    parser.add_argument('--fuse_depth',      type=int,   default=2)
    print(parser.parse_args(), '\n')
    return parser


# ─── Audio / image transforms ─────────────────────────────────────────────────

def pad_or_crop(spec):
    n_mels, T = spec.shape
    if T > TARGET_FRAMES:
        start = np.random.randint(0, T - TARGET_FRAMES + 1)
        return spec[:, start:start + TARGET_FRAMES]
    if T < TARGET_FRAMES:
        pad = TARGET_FRAMES - T
        return F.pad(spec, (pad // 2, pad - pad // 2))
    return spec


def normalize_spec(spec):
    return (spec - spec.mean()) / (spec.std() + 1e-6)


def add_noise(spec, noise_level=0.05):
    return spec + torch.randn_like(spec) * noise_level


def freq_mask(spec, max_width=8):
    spec = spec.clone()
    F_, _ = spec.shape
    w = np.random.randint(0, max_width)
    s = np.random.randint(0, max(1, F_ - w))
    spec[s:s + w, :] = 0
    return spec


def time_mask(spec, max_width=10):
    _, T = spec.shape
    w = np.random.randint(0, max_width)
    s = np.random.randint(0, max(1, T - w))
    spec[:, s:s + w] = 0
    return spec


def augment(spec):
    if np.random.rand() < 0.8:
        spec = add_noise(spec)
    if np.random.rand() < 0.5:
        spec = freq_mask(spec)
    if np.random.rand() < 0.5:
        spec = time_mask(spec)
    return spec


def get_audio_transforms(train=True):
    base = [
        TrimSilence(threshold=1e-6),
        MelSpectrogram(sample_rate=8000, n_mels=64, n_fft=512, hop_length=128),
        AmplitudeToDB(),
        pad_or_crop,
    ]
    if train:
        base.append(augment)
    base.append(normalize_spec)
    return Compose(base)


def get_image_transforms(train=True):
    base = []
    if train:
        base.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)))
    base += [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    return transforms.Compose(base)


def load_fsdd():
    train_tf = get_audio_transforms(train=True)
    test_tf  = get_audio_transforms(train=False)
    fsdd_train = TorchFSDDGenerator(
        version='local',
        path='/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10',
        transforms=train_tf, load_all=True,
    )
    fsdd_test = TorchFSDDGenerator(
        version='local',
        path='/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10',
        transforms=test_tf, load_all=True,
    )
    train_set, _ = fsdd_train.train_test_split(test_size=0.15)
    _,  test_set = fsdd_test.train_test_split(test_size=0.15)
    return train_set, test_set


def prepare_dataset(args, cutoff_sum):
    kw_tr = {'batch_size': args.batch_size,      'shuffle': True,
             'num_workers': 0, 'pin_memory': True, 'drop_last': False}
    kw_te = {'batch_size': args.test_batch_size, 'shuffle': False,
             'num_workers': 0, 'pin_memory': True, 'drop_last': False}

    v_train = datasets.MNIST('data', train=True,  download=True,
                             transform=get_image_transforms(train=True))
    v_test  = datasets.MNIST('data', train=False,
                             transform=get_image_transforms(train=False))
    a_train, a_test = load_fsdd()

    AV_train = DataLoader(AV_dataset_sum(v_train, a_train, cutoff_sum,
                                         samples_per_combination=30), **kw_tr)
    AV_test  = DataLoader(AV_dataset_sum(v_test,  a_test,  cutoff_sum,
                                         samples_per_combination=10),  **kw_te)
    return AV_train, AV_test


# ─── Model eval helpers ───────────────────────────────────────────────────────

def test_unit(model, loader, unimodal=None):
    model.eval()
    total_acc = total_ce = total_n = 0
    total_img_acc = total_aud_acc = 0
    probs_list = []

    with torch.no_grad():
        for imgs, audios, labels, labels_img, labels_aud in loader:
            imgs, audios, labels = imgs.to(device), audios.to(device), labels.to(device)
            labels_img, labels_aud = labels_img.to(device), labels_aud.to(device)

            if unimodal is None:
                output, output_img, output_aud, out_dig_img, out_dig_aud = \
                    model.forward(imgs, audios, unimodal="train")
                logits = output
            else:
                logits = model(imgs, audios, unimodal)
                out_dig_img = out_dig_aud = None

            probs = torch.exp(logits)
            probs_list.append(probs.cpu())

            acc, ce = traditional_cross_entropy_from_probs(probs, labels)
            bs = labels.size(0)
            total_acc += acc * bs; total_ce += ce * bs; total_n += bs

            if unimodal is None:
                total_img_acc += (out_dig_img.argmax(1) == labels_img).float().mean().item() * bs
                total_aud_acc += (out_dig_aud.argmax(1) == labels_aud).float().mean().item() * bs

    res = {"acc": total_acc / total_n, "ce": total_ce / total_n,
           "probs": torch.cat(probs_list)}
    if unimodal is None:
        res["img_digit_acc"] = total_img_acc / total_n
        res["aud_digit_acc"] = total_aud_acc / total_n
    return res


def test(model, loader):
    joint = test_unit(model, loader)
    vis   = test_unit(model, loader, 'visual')
    aud   = test_unit(model, loader, 'audio')
    return {
        "joint_acc": joint["acc"], "joint_ce": joint["ce"],
        "vis_acc":   vis["acc"],   "vis_ce":   vis["ce"],
        "aud_acc":   aud["acc"],   "aud_ce":   aud["ce"],
        "img_digit_acc": joint["img_digit_acc"],
        "aud_digit_acc": joint["aud_digit_acc"],
    }


def extract_representations(model, loader):
    model.eval()
    vis_list, aud_list, y_list = [], [], []
    img_lbl_list, aud_lbl_list = [], []

    with torch.no_grad():
        for imgs, audios, labels, img_labels, aud_labels in loader:
            img_repr, aud_repr = model.get_representations(imgs.to(device), audios.to(device))
            vis_list.append(img_repr.cpu())
            aud_list.append(aud_repr.cpu())
            y_list.append(labels)
            img_lbl_list.append(img_labels)
            aud_lbl_list.append(aud_labels)

    return (torch.cat(vis_list), torch.cat(aud_list), torch.cat(y_list),
            torch.cat(img_lbl_list), torch.cat(aud_lbl_list))


# ─── LSMI helpers ─────────────────────────────────────────────────────────────

def _input(batch):
    return batch[0].to(device), batch[1].to(device), batch[2].to(device)


def RUS_adjustment(rus):
    r, u1, u2, s = rus
    R_m, U1_m, U2_m, S_m = (x.detach().mean() for x in (r, u1, u2, s))
    adj = torch.tensor(0.0, dtype=R_m.dtype, device=R_m.device)
    if R_m < 0 or S_m < 0:
        adj = -torch.min(R_m, S_m)
    elif U1_m < 0 or U2_m < 0:
        adj = torch.min(U1_m, U2_m)
    return r + adj, u1 - adj, u2 - adj, s + adj


def get_mutual_info(loader, model, modality, n_classes):
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            m1, m2, labels = _input(batch)
            x = m1 if modality == 'modality_1' else \
                m2 if modality == 'modality_2' else torch.cat([m1, m2], dim=1)
            out = model(x)
            rows = torch.arange(x.size(0), device=x.device)
            info.append(np.log(n_classes) + torch.nn.Softmax(dim=1)(out)[rows, labels].log())
    return torch.cat(info).detach()


def get_entropy(loader, model, modality):
    model.eval()
    info = []
    with torch.no_grad():
        for batch in loader:
            m1, m2, _ = _input(batch)
            x = m1 if modality == 'modality_1' else m2
            info.append(model(x))
    return torch.cat(info).detach()


def obtain_discriminator(train_loader, input_size_1, input_size_2, embed_size,
                          n_classes, num_epochs):
    model_1 = cls_network(input_size_1, embed_size, n_classes).to(device)
    model_2 = cls_network(input_size_2, embed_size, n_classes).to(device)
    model_j = cls_network(input_size_1 + input_size_2, embed_size, n_classes).to(device)
    models    = [model_1, model_2, model_j]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0.; n = 0
        for batch in train_loader:
            m1, m2, labels = _input(batch)
            out_1 = models[0](m1)
            out_2 = models[1](m2)
            out_j = models[2](torch.cat([m1, m2], dim=1))
            optimizer.zero_grad()
            loss = criterion(out_1, labels) + criterion(out_2, labels) + criterion(out_j, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * m1.size(0); n += m1.size(0)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  Discriminator  epoch [{epoch+1}/{num_epochs}]  loss={total_loss/n:.4f}')
    return models


def obtain_entropy_estimator(train_loader, input_size_1, input_size_2, embed_size, num_epochs):
    model_1 = MargKernel(dim=input_size_1).to(device)
    model_2 = MargKernel(dim=input_size_2).to(device)
    models    = [model_1, model_2]
    optimizer = torch.optim.Adam([p for m in models for p in m.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(num_epochs):
        for m in models:
            m.train()
        total_loss = 0.; n = 0
        for batch in train_loader:
            m1, m2, _ = _input(batch)
            loss = model_1(m1) + model_2(m2)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * m1.size(0); n += m1.size(0)
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'  Entropy est.   epoch [{epoch+1}/{num_epochs}]  loss={total_loss/n:.4f}')
    return models


def LSMI_estimation(loader, discriminator, entropy_estimator, n_classes):
    I_X1Y  = get_mutual_info(loader, discriminator[0], 'modality_1', n_classes)
    I_X2Y  = get_mutual_info(loader, discriminator[1], 'modality_2', n_classes)
    I_X12Y = get_mutual_info(loader, discriminator[2], 'modality_12', n_classes)
    H_X1   = get_entropy(loader, entropy_estimator[0], 'modality_1')
    H_X2   = get_entropy(loader, entropy_estimator[1], 'modality_2')

    r_plus  = torch.minimum(H_X1, H_X2)
    r_minus = torch.minimum(H_X1 - I_X1Y, H_X2 - I_X2Y)
    r  = r_plus - r_minus
    u1 = I_X1Y  - r
    u2 = I_X2Y  - r
    s  = I_X12Y - r - u1 - u2

    r_adj, u1_adj, u2_adj, s_adj = RUS_adjustment([r, u1, u2, s])
    print(f"  R={r_adj.mean():.4f}  U1={u1_adj.mean():.4f}  "
          f"U2={u2_adj.mean():.4f}  S={s_adj.mean():.4f}")
    return r, u1, u2, s


def normalize_pid(pid):
    pid_ = np.maximum(pid, 0)
    pid_ /= pid_.sum(axis=1, keepdims=True) + 1e-12
    return pid_


# ─── Main ─────────────────────────────────────────────────────────────────────

def mnist(args):
    cutoff_sum = 8

    # ── 1. Data ───────────────────────────────────────────────────────────────
    AV_train, AV_test = prepare_dataset(args, cutoff_sum=cutoff_sum)

    # ── 2. Model ──────────────────────────────────────────────────────────────
    model = CNN_sum(num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint = torch.load(f"cnn_sum{cutoff_sum}_model.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    # ── 3. Test ───────────────────────────────────────────────────────────────
    test_metrics = test(model, AV_test)
    print(f"Joint CE={test_metrics['joint_ce']:.4f}  Acc={test_metrics['joint_acc']:.4f}")
    print(f"Visual CE={test_metrics['vis_ce']:.4f}  Acc={test_metrics['vis_acc']:.4f}")
    print(f"Audio  CE={test_metrics['aud_ce']:.4f}  Acc={test_metrics['aud_acc']:.4f}")

    # ── 4. Extract representations from the multimodal model ──────────────────
    print("\nExtracting representations…")
    train_vis, train_aud, y_train, train_img_lbl, train_aud_lbl = \
        extract_representations(model, AV_train)
    test_vis,  test_aud,  y_test,  test_img_lbl,  test_aud_lbl  = \
        extract_representations(model, AV_test)
    print(f"  vis={train_vis.shape}  aud={train_aud.shape}")

    # ── 5. Save feature tensors for LSMI loader ───────────────────────────────
    torch.save({
        "train_modal_1_features": train_vis.float(),
        "train_modal_2_features": train_aud.float(),
        "train_targets":          y_train,
        "val_modal_1_features":   test_vis.float(),
        "val_modal_2_features":   test_aud.float(),
        "val_targets":            y_test,
    }, "lsmi.pt")
    print("[✓] Saved LSMI features to lsmi.pt")

    # ── 6. LSMI config ────────────────────────────────────────────────────────
    class CFG:
        pass
    cfg            = CFG()
    cfg.device     = device
    cfg.batch_size = 512
    cfg.num_workers = 0

    input_size_1 = train_vis.shape[1]
    input_size_2 = train_aud.shape[1]
    embed_size   = 128
    n_classes    = 2
    n_ep_disc    = 30
    n_ep_ent     = 30

    setup_seed(args.seed)

    # ── 7. Load feature loaders ───────────────────────────────────────────────
    train_loader, val_loader = get_loader(cfg, "lsmi.pt")

    # ── 8. Train LSMI estimators on learned representations ───────────────────
    print("\nTraining discriminators…")
    discriminator = obtain_discriminator(
        train_loader, input_size_1, input_size_2, embed_size, n_classes, n_ep_disc)

    print("\nTraining entropy estimators…")
    entropy_estimator = obtain_entropy_estimator(
        train_loader, input_size_1, input_size_2, embed_size, n_ep_ent)

    # ── 9. LSMI PID ───────────────────────────────────────────────────────────
    print("\n=== LSMI TRAIN PID ===")
    LSMI_estimation(train_loader, discriminator, entropy_estimator, n_classes)

    print("\n=== LSMI TEST PID ===")
    r, u1, u2, s = LSMI_estimation(val_loader, discriminator, entropy_estimator, n_classes)

    # ── 10. Stack pointwise PID ───────────────────────────────────────────────
    pid_lsmi = np.stack([
        u1.cpu().numpy(), u2.cpu().numpy(),
        r.cpu().numpy(),  s.cpu().numpy(),
    ], axis=1)   # columns: [U_vis, U_aud, R, S]

    # ── 11. RUS adjustment ────────────────────────────────────────────────────
    r, u1, u2, s = RUS_adjustment([r, u1, u2, s])
    print("\n=== AFTER RUS ADJUSTMENT ===")
    print(f"R={r.mean():.4f}  U_vis={u1.mean():.4f}  "
          f"U_aud={u2.mean():.4f}  S={s.mean():.4f}")

    # ── 12. Subgroup analysis ──────────────────────────────────────────────────
    synergy, redundancy, u0_list, u1_list = [], [], [], []
    for img_label, aud_label, pid_val in zip(test_img_lbl, test_aud_lbl, pid_lsmi):
        t = torch.tensor(pid_val)[None, :]
        if img_label + aud_label > cutoff_sum:
            if   img_label > cutoff_sum and aud_label > cutoff_sum: redundancy.append(t)
            elif img_label <= cutoff_sum and aud_label > cutoff_sum: u1_list.append(t)
            elif img_label > cutoff_sum and aud_label <= cutoff_sum: u0_list.append(t)
            else: synergy.append(t)
        else:
            synergy.append(t)

    print("\n=== SUBGROUP PID (mean [U_vis, U_aud, R, S]) ===")
    print("Redundancy :", torch.mean(torch.cat(redundancy), dim=0))
    print("Unique_vis :", torch.mean(torch.cat(u0_list),    dim=0))
    print("Unique_aud :", torch.mean(torch.cat(u1_list),    dim=0))
    print("Synergy    :", torch.mean(torch.cat(synergy),    dim=0))

    # ── 13. Normalised cosine similarity ──────────────────────────────────────
    list_pid = [torch.cat(redundancy), torch.cat(u0_list),
                torch.cat(u1_list),    torch.cat(synergy)]
    list_lbl = [
        torch.cat([torch.tensor([0, 0, 1, 0])[None, :]] * len(list_pid[0])),
        torch.cat([torch.tensor([1, 0, 0, 0])[None, :]] * len(list_pid[1])),
        torch.cat([torch.tensor([0, 1, 0, 0])[None, :]] * len(list_pid[2])),
        torch.cat([torch.tensor([0, 0, 0, 1])[None, :]] * len(list_pid[3])),
    ]

    pid       = torch.cat(list_pid).float().numpy()
    pid_labels = torch.cat(list_lbl).float().numpy()

    pid_norm  = normalize_pid(pid)
    pid_l2    = pid_norm  / (np.linalg.norm(pid_norm,   axis=1, keepdims=True) + 1e-12)
    labels_l2 = pid_labels / (np.linalg.norm(pid_labels, axis=1, keepdims=True) + 1e-12)
    print("\nBefore correction  cosine sim:", np.sum(pid_l2 * labels_l2, axis=1).mean())

    for i, p in enumerate(pid):
        if p[0] < 0 and p[1] >= 0:
            pid[i] = [0, p[1], p[2], p[3] + p[0]]
        if p[1] < 0 and p[0] >= 0:
            pid[i] = [p[0], 0, p[2], p[3] + p[1]]

    pid_norm  = normalize_pid(pid)
    pid_l2    = pid_norm  / (np.linalg.norm(pid_norm,   axis=1, keepdims=True) + 1e-12)
    print("After  correction  cosine sim:", np.sum(pid_l2 * labels_l2, axis=1).mean())


if __name__ == '__main__':
    args = config().parse_args()
    mnist(args)
