""" Adapted from https://github.com/pytorch/examples/blob/main/mnist/main.py """
from __future__ import print_function
import argparse
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
from utils import pad_or_crop, TARGET_FRAMES, softmax, traditional_cross_entropy_from_probs
from utils_ours import return_redundancy_test_performances

import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose




def config():
    parser = argparse.ArgumentParser(description='Pytorch audiovisual MNIST digit classification')
    parser.add_argument('--model', type=str, default='CNN', help='FCN or CNN')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N', help='input batch size for testing')
    parser.add_argument('--epoch', type=int, default=30, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.996, metavar='M', help='Learning rate step gamma=')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
    parser.add_argument('--log-interval', type=int, default=30, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    parser.add_argument("--depth", type=int, default=6, help='number of layers ')
    parser.add_argument("--fuse_depth", type=int, default=2, help='fuse at which layer')
    print(parser.parse_args(), '\n')
    return parser

def load_fsdd():
    from torchfsdd import TorchFSDDGenerator, TrimSilence
    from torchaudio.transforms import MFCC
    from torchvision.transforms import Compose, Resize
    # Set number of features and classes
    n_mfcc = 28
    n_digits = 10

    # Specify transformations to be applied to the raw audio
    transforms = Compose([
        TrimSilence(threshold=1e-6),

        MelSpectrogram(
            sample_rate=8000,
            n_mels=64,
            n_fft=512,
            hop_length=128
        ),

        AmplitudeToDB(),

        pad_or_crop
    ])

    # Initialize a generator for a local version of FSDD
    fsdd = TorchFSDDGenerator(version='local', path='/lustre/fswork/projects/rech/haj/uik24xv/datasets/torch-fsdd/lib/test/data/v1.0.10', transforms=transforms,
                              load_all=True) #  '/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10'

    # Create two Torch datasets for a train-test split from the generator
    train_set, test_set = fsdd.train_test_split(test_size=0.2)
    return train_set, test_set


def prepare_dataset(args):
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': True}
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'drop_last': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    v_train = datasets.MNIST('/lustre/fswork/projects/rech/haj/uik24xv/datasets/MNIST/', train=True, download=False, transform=transform)
    v_test = datasets.MNIST('/lustre/fswork/projects/rech/haj/uik24xv/datasets/MNIST/', train=False, transform=transform)
    a_train, a_test = load_fsdd()

    # Create a multimodal dataset instance and its DataLoader
    AV_trainset = AV_dataset(v_train, a_train)
    AV_testset = AV_dataset(v_test, a_test)
    AV_train = DataLoader(AV_trainset, **train_kwargs)
    AV_test = DataLoader(AV_testset, **test_kwargs)

    # Iterate over the DataLoader to get batches of paired data
    # for batch in AV_train:
    #     imgs, audios, labels = batch
    #     # Do whatever you need with the paired data
    #     print("MNIST images:", imgs.shape)
    #     print("Spoken digit audios:", audios.shape)
    #     print("Labels:", labels)
    #     display(imgs)

    return AV_train, AV_test


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (imgs, audios, labels) in enumerate(train_loader):
        imgs, audios, labels = imgs.to(device), audios.to(device), labels.to(device)
        optimizer.zero_grad()
        output, output_img, output_aud = model.forward(imgs, audios, unimodal="train")
        loss = F.nll_loss(output, labels)
        loss += F.nll_loss(output_img, labels)
        loss += F.nll_loss(output_aud, labels)
        if batch_idx == 0:
            Ls = loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(imgs), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
        loss.backward()
        optimizer.step()
    return Ls


def test_unit(model, device, loader, unimodal=None):

    model.eval()

    total_acc = 0
    total_ce = 0
    total_n = 0

    probs_list = []

    with torch.no_grad():
        for imgs, audios, labels in loader:

            imgs = imgs.to(device)
            audios = audios.to(device)
            labels = labels.to(device)

            if unimodal is None:
                logits = model(imgs, audios)
            else:
                logits = model(imgs, audios, unimodal)

            probs = torch.exp(logits)

            probs_list.append(probs.cpu())

            acc, ce = traditional_cross_entropy_from_probs(probs, labels)

            batch_size = labels.size(0)
            total_acc += acc * batch_size
            total_ce += ce * batch_size
            total_n += batch_size

    return (
        total_acc / total_n,
        total_ce / total_n,
        torch.cat(probs_list)
    )

def test(model, device, loader):

    joint_acc, joint_ce, joint_probs = test_unit(model, device, loader)
    vis_acc, vis_ce, vis_probs = test_unit(model, device, loader, 'visual')
    aud_acc, aud_ce, aud_probs = test_unit(model, device, loader, 'audio')

    return {
        "joint_acc": joint_acc,
        "joint_ce": joint_ce,
        "joint_probs": joint_probs,

        "vis_acc": vis_acc,
        "vis_ce": vis_ce,
        "vis_probs": vis_probs,

        "aud_acc": aud_acc,
        "aud_ce": aud_ce,
        "aud_probs": aud_probs,
    }

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
        """modality0_ce = min(modality0_ce, h_y)
        modality1_ce = min(modality1_ce, h_y)"""

        # redundancy_ce = max(redundancy_ce, joint_ce)
        joint_ce = min(redundancy_ce, joint_ce)

        redundancy_ce = min(redundancy_ce, h_y)

        modality0_ce = max(modality0_ce, joint_ce)
        modality1_ce = max(modality1_ce, joint_ce)

        """modality0_ce = min(modality0_ce, redundancy_ce)
        modality1_ce = min(modality1_ce, redundancy_ce)"""

        # ===== INFORMATION =====
        total = h_y - joint_ce

        r_val = h_y - redundancy_ce

        u0 = h_y - modality0_ce - r_val
        u1 = h_y - modality1_ce - r_val

        s = total - u0 - u1 - r_val

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
        """modality0_ce = min(modality0_ce, h_y)
        modality1_ce = min(modality1_ce, h_y)"""

        # redundancy_ce = max(redundancy_ce, joint_ce)
        joint_ce = min(redundancy_ce, joint_ce)
        # source_redundancy_ce = max(source_redundancy_ce, joint_ce)
        source_redundancy_ce = min(redundancy_ce, source_redundancy_ce)

        redundancy_ce = min(redundancy_ce, h_y)
        source_redundancy_ce = min(source_redundancy_ce, h_y)

        modality0_ce = max(modality0_ce, joint_ce)
        modality1_ce = max(modality1_ce, joint_ce)

        """modality0_ce = min(modality0_ce, redundancy_ce)
        modality1_ce = min(modality1_ce, redundancy_ce)"""

        # ===== INFORMATION =====
        total = h_y - joint_ce

        # your design choice: strongest redundancy
        r_val = max(h_y - redundancy_ce, h_y - source_redundancy_ce)

        u0 = h_y - modality0_ce - r_val
        u1 = h_y - modality1_ce - r_val

        s = total - u0 - u1 - r_val

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
    print('')

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

    print("R=" + str(I_R) + " (" + str(100*ratio_source) + "% Source)")
    print("U0=", str(I_U0))
    print("U1=", str(I_U1))
    print("S=", str(I_S))
    print("I=", str(I))

def ce_per_sample(targets, probs, eps=1e-12):
    """
    Per-sample cross-entropy:
        CE(x) = -log p(y|x)
    """
    probs = torch.clamp(probs, eps, 1.0)
    return -torch.log(probs[torch.arange(len(targets)), targets])

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


def logp(p):
    return torch.log(torch.clamp(p, 1e-12, 1.0))

def compute_entropy_from_targets(targets, num_classes):
    import torch

    targets = torch.tensor(targets).long()
    counts = torch.bincount(targets, minlength=num_classes).float()
    probs = counts / counts.sum()
    probs = torch.clamp(probs, 1e-12, 1.0)

    return -torch.sum(probs * torch.log(probs)).item()


def extract_representations(model, loader, device):

    model.eval()

    visual_list = []
    audio_list = []
    label_list = []

    with torch.no_grad():

        for imgs, audios, labels in loader:

            imgs = imgs.to(device)
            audios = audios.to(device)

            img_repr, aud_repr = model.get_representations(imgs, audios)

            visual_list.append(img_repr.cpu())
            audio_list.append(aud_repr.cpu())
            label_list.append(labels)

    visual_repr = torch.cat(visual_list)
    audio_repr = torch.cat(audio_list)
    labels = torch.cat(label_list)

    return visual_repr, audio_repr, labels

def mnist(args):

    # =======================
    # 1. DATA
    # =======================
    AV_train, AV_test = prepare_dataset(args)

    device = "cuda"

    model = CNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    print(model)

    # =======================
    # 2. TRAINING
    # =======================
    for epoch in range(1, args.epoch + 1):

        print(f"\n===== Epoch {epoch} =====")

        train(args, model, device, AV_train, optimizer, epoch)
        scheduler.step()

        test_metrics = test(model, device, AV_test)

        print(
            f"Joint CE: {test_metrics['joint_ce']:.4f} | "
            f"Visual CE: {test_metrics['vis_ce']:.4f} | "
            f"Audio CE: {test_metrics['aud_ce']:.4f}"
        )

        print(
            f"Joint Acc: {test_metrics['joint_acc']:.4f} | "
            f"Visual Acc: {test_metrics['vis_acc']:.4f} | "
            f"Audio Acc: {test_metrics['aud_acc']:.4f}"
        )

    # =======================
    # 3. EXTRACT REPRESENTATIONS
    # =======================
    train_vis, train_aud, y_train = extract_representations(model, AV_train, device)
    test_vis, test_aud, y_test = extract_representations(model, AV_test, device)

    # =======================
    # 4. GET PROBABILITIES
    # =======================
    test_metrics = test(model, device, AV_test)

    joint_probs = test_metrics["joint_probs"]
    vis_probs = test_metrics["vis_probs"]
    aud_probs = test_metrics["aud_probs"]

    joint_ce = test_metrics["joint_ce"]
    vis_ce = test_metrics["vis_ce"]
    aud_ce = test_metrics["aud_ce"]

    joint_acc = test_metrics["joint_acc"]
    vis_acc = test_metrics["vis_acc"]
    aud_acc = test_metrics["aud_acc"]

    # =======================
    # 4b. MUTUAL INFORMATION I(Y; (X1,X2))
    # =======================
    H_Y = compute_entropy_from_targets(y_test, num_classes=10)
    mi_joint = H_Y - joint_ce
    print(f"\nI(Y; (X1,X2)) = H(Y) - H(Y|X1,X2) = {H_Y:.4f} - {joint_ce:.4f} = {mi_joint:.4f} nats")

    # =======================
    # 5. CCS REDUNDANCY
    # =======================
    log_py = compute_log_py(y_test, num_classes=10)

    ce_list = compute_ce_from_probs(
        [vis_probs, aud_probs],
        y_test
    )

    i_list = compute_pointwise_information(ce_list, log_py)

    same_sign = torch.sign(i_list[0]) == torch.sign(i_list[1])

    ccs = compute_ccs_and_selection(
        ce_list, same_sign, log_py
    )
    print(ccs[:5])

    redundancy_pointwise_ce = ccs.numpy()
    redundancy_ce = ccs.mean().item()

    print(f"\nCCS redundancy CE: {redundancy_ce:.4f}")

    # =======================
    # 6. SOURCE REDUNDANCY (LEARNED)
    # =======================
    X_train_dict = {
        "modality0": train_vis.float(),
        "modality1": train_aud.float()
    }

    X_test_dict = {
        "modality0": test_vis.float(),
        "modality1": test_aud.float()
    }

    print(y_test[:20])

    y_pred_dict = return_redundancy_test_performances(
        X_train_dict,
        X_train_dict,
        X_test_dict,
        y_train,
        y_train,
        y_test,
        "redundancy",
        distribution_target="categorical",
        lambda_reg=1,
        num_classes=10
    )

    results = {}
    for key in ["modality0", "modality1", "average"]:
        acc_, ce_ = traditional_cross_entropy_from_probs(
            softmax(y_pred_dict[key]),
            y_pred_dict["targets"]
        )
        results[key] = {"accuracy": acc_, "cross_entropy": ce_}

    print("\n=== Learned redundancy performances ===")
    for k, v in results.items():
        print(f"{k:10s} | acc = {v['accuracy']:.4f}, CE = {v['cross_entropy']:.4f}")

    # =======================
    # 7. BUILD METRICS DICT
    # =======================
    dict_of_metrics = {
        "joint_ce": joint_ce,
        "joint_acc": joint_acc,
        "probs_joint": joint_probs,

        "modalities_ce": [vis_ce, aud_ce],
        "modalities_acc": [vis_acc, aud_acc],
        "probs_modalities": [vis_probs, aud_probs],

        "redundancy_ce": redundancy_ce,
        "redundancy_pointwise_ce": redundancy_pointwise_ce,

        "true_labels": y_test,

        "source_redundancy_pointwise_ce": results["average"]["cross_entropy"],
        "source_redundancy_preds": y_pred_dict["average"]
    }

    # =======================
    # 8. GLOBAL PID
    # =======================
    print("\n=== GLOBAL PID (WITH SOURCE) ===")

    compute_PID_categorical_with_source_decomposition(
        joint_ce,
        vis_ce,
        aud_ce,
        redundancy_ce,
        results["average"]["cross_entropy"],
        num_classes=10,
        targets=y_test
    )

    # =======================
    # 9. POINTWISE PID
    # =======================
    print("\n=== POINTWISE PID ===")

    pid = compute_pointwise_pid_from_probs(dict_of_metrics, num_classes=10)
    pid_source = compute_pointwise_pid_with_source_from_probs(dict_of_metrics, num_classes=10)

    print("PID mean [U0, U1, R, S]:", np.mean(pid, axis=0))
    print("PID + source mean [U0, U1, R, S]:", np.mean(pid_source, axis=0))

if __name__ == '__main__':
    args = config().parse_args()
    torch.manual_seed(args.seed)

    mnist(args)
