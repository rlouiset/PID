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
from model import CNN_sum
from dataset import AV_dataset_sum
from utils_ours import return_redundancy_test_performances, compute_PID_categorical

import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose

from math import *

TARGET_FRAMES = 64

softmax = torch.nn.Softmax(dim=-1)

import argparse
import pickle
import torch
import numpy as np

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

def pad_or_crop(spec):
    # spec shape: (n_mels, time)
    n_mels, T = spec.shape

    if T > TARGET_FRAMES:
        spec = spec[:, :TARGET_FRAMES]  # crop

    elif T < TARGET_FRAMES:
        pad = TARGET_FRAMES - T
        spec = F.pad(spec, (0, pad))  # pad time dimension

    return spec


def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):
    probs = torch.clamp(probs, min=eps, max=1.0)
    log_probs = torch.log(probs)

    ce = -log_probs[torch.arange(targets.shape[0]), targets.long()].mean()
    acc = (probs.argmax(dim=1) == targets).float().mean()

    return acc.item(), ce.item()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def config():
    parser = argparse.ArgumentParser(description='Pytorch audiovisual MNIST digit classification')
    parser.add_argument('--model', type=str, default='CNN', help='FCN or CNN')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N', help='input batch size for testing')
    parser.add_argument('--epoch', type=int, default=2, metavar='N', help='number of epochs to train')
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


def vis(args, Ls, acc, V_acc, A_acc):
    L = args.depth + 1
    filename = "{}_L{}_Lf{}_lr{}_seed{}".format(args.model, L, args.fuse_depth, args.lr, args.seed)

    import pandas as pd
    df = pd.DataFrame({'Ls': Ls,
                       'Eg': acc,
                       'Eg_A': V_acc,
                       'Eg_B': A_acc})
    df.to_csv("{}.csv".format(filename))

    import matplotlib
    import matplotlib.pyplot as plt
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.figure(figsize=(4, 3))
    plt.plot(Ls / Ls[0], c='k', lw=1.5, label="Loss")
    plt.plot(A_acc, c='fuchsia', lw=1.5, label="Audio acc")
    plt.plot(V_acc, c='b', lw=1.5, label="Visual acc")
    plt.plot(acc, 'k--', lw=1.5, label="AV acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss & Accuracy")
    plt.xlim((0, args.epoch - 1))
    plt.legend()
    plt.tight_layout(pad=0.5)
    plt.savefig("{}.svg".format(filename))
    # plt.show()


def display(X):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    img = X[0, 0, :, :].cpu().detach().numpy()
    plt.imshow(img, cmap='gray')
    plt.show()


def load_fsdd():
    from torchfsdd import TorchFSDDGenerator, TrimSilence
    from torchaudio.transforms import MFCC
    from torchvision.transforms import Compose, Resize

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
    fsdd = TorchFSDDGenerator(version='local', path='/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10',
                              transforms=transforms,
                              load_all=True)  # '/Users/robinlouiset/Documents/torch-fsdd/lib/test/data/v1.0.10'

    # Create two Torch datasets for a train-test split from the generator
    train_set, test_set = fsdd.train_test_split(test_size=0.1)
    return train_set, test_set


def prepare_dataset(args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': False,
                   'drop_last': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    v_train = datasets.MNIST('data', train=True, download=True, transform=transform)
    v_test = datasets.MNIST('data', train=False, transform=transform)
    a_train, a_test = load_fsdd()

    # Create a multimodal dataset instance and its DataLoader
    AV_trainset = AV_dataset_sum(v_train, a_train)
    AV_testset = AV_dataset_sum(v_test, a_test)
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
    for batch_idx, (imgs, audios, labels, labels_img, labels_aud) in enumerate(train_loader):
        imgs, audios, labels = imgs.to(device), audios.to(device), labels.to(device)
        labels_img, labels_aud = labels_img.to(device), labels_aud.to(device)
        optimizer.zero_grad()
        output, output_img, output_aud, output_digit_img, output_digit_aud = model.forward(imgs, audios,
                                                                                           unimodal="train")
        # loss = F.cross_entropy(output, labels)
        # loss += F.cross_entropy(output_img, labels)
        # loss += F.cross_entropy(output_aud, labels)
        loss = F.nll_loss(output_digit_img, labels_img)
        loss += F.nll_loss(output_digit_aud, labels_aud)

        print("img digit acc:", (output_digit_img.argmax(1) == labels_img).float().mean())
        print("aud digit acc:", (output_digit_aud.argmax(1) == labels_aud).float().mean())
        print("fusion acc:", (output.argmax(1) == labels).float().mean())
        print('---')

        if epoch > 25:
            loss += F.nll_loss(output, labels)
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


def test_unit(model, device, test_loader, unimodal=None):
    model.eval()

    total_acc = 0
    total_ce = 0
    total_n = 0

    with torch.no_grad():
        for imgs, audios, labels, _, _ in test_loader:
            imgs = imgs.to(device)
            audios = audios.to(device)
            labels = labels.to(device)

            logits = model(imgs, audios, unimodal)

            probs = torch.exp(logits)

            acc, ce = traditional_cross_entropy_from_probs(probs, labels)

            batch_size = labels.size(0)

            total_acc += acc * batch_size
            total_ce += ce * batch_size
            total_n += batch_size

    avg_acc = total_acc / total_n
    avg_ce = total_ce / total_n

    print(
        "[{} testset] CE: {:.4f}, Accuracy: {:.4f}".format(
            unimodal, avg_ce, avg_acc
        )
    )

    return avg_acc, avg_ce


def test(model, device, test_loader):
    acc, ce = test_unit(model, device, test_loader)
    visual_acc, visual_ce = test_unit(model, device, test_loader, 'visual')
    audio_acc, audio_ce = test_unit(model, device, test_loader, 'audio')
    return acc, ce, visual_acc, visual_ce, audio_acc, audio_ce


def extract_representations(model, loader, device):
    model.eval()

    visual_list = []
    audio_list = []
    label_list = []

    visual_label_list = []
    audio_label_list = []

    with torch.no_grad():
        for imgs, audios, labels, img_labels, audio_labels in loader:
            imgs = imgs.to(device)
            audios = audios.to(device)

            img_repr, aud_repr = model.get_representations(imgs, audios)

            visual_list.append(img_repr.cpu())
            audio_list.append(aud_repr.cpu())
            label_list.append(labels)

            visual_label_list.append(img_labels)
            audio_label_list.append(audio_labels)

    visual_repr = torch.cat(visual_list)
    audio_repr = torch.cat(audio_list)
    labels = torch.cat(label_list)

    img_labels = torch.cat(visual_label_list)
    audio_labels = torch.cat(audio_label_list)

    return visual_repr, audio_repr, labels, img_labels, audio_labels


def mnist(args):
    AV_train, AV_test = prepare_dataset(args)

    Ls = np.zeros(args.epoch)
    acc, V_acc, A_acc = np.copy(Ls), np.copy(Ls), np.copy(Ls)
    ce, V_ce, A_ce = np.copy(Ls), np.copy(Ls), np.copy(Ls)

    model = CNN_sum(num_classes=2).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epoch + 1):
        acc[epoch - 1], ce[epoch - 1], V_acc[epoch - 1], V_ce[epoch - 1], A_acc[epoch - 1], A_ce[epoch - 1] = test(
            model, device, AV_test)
        Ls[epoch - 1] = train(args, model, device, AV_train, optimizer, epoch)

    vis(args, Ls, acc, V_acc, A_acc)

    train_vis, train_aud, y_train, img_labels, audio_labels = extract_representations(model, AV_train, device)
    test_vis, test_aud, y_test, img_labels, audio_labels = extract_representations(model, AV_test, device)

    # -------------------------
    # Save .pt for LSMI
    # -------------------------
    lsmi_data = {
        "train_modal_1_features": train_vis.float(),
        "train_modal_2_features": train_aud.float(),
        "train_targets": y_train,
        "val_modal_1_features": test_vis.float(),
        "val_modal_2_features": test_aud.float(),
        "val_targets": y_test,
    }

    torch.save(lsmi_data, args.out_pt)
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
    cfg.n_classes = args.num_classes
    cfg.num_epochs_discriminator = args.epochs_disc
    cfg.num_epochs_entropy_estimator = args.epochs_entropy

    cfg.input_size_1 = train_vis.shape[1]
    cfg.input_size_2 = train_aud.shape[1]

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

    # stack PID atoms in the same order as weights
    list_of_pointwise_pid = np.stack([u1, u2, r, s], axis=1)  # shape (N, 4)

    # RUS adjustement
    r, u1, u2, s = RUS_adjustment([torch.tensor(r), torch.tensor(u1), torch.tensor(u2), torch.tensor(s)])
    r, u1, u2, s = r.numpy(), u1.numpy(), u2.numpy(), s.numpy()

    print("after adjustement")
    print("r: ", np.mean(r))
    print("u1: ", np.mean(u1))
    print("u2: ", np.mean(u2))
    print("s: ", np.mean(s))

    # Subgroups
    synergy_combinations = []
    redundancy_combinations = []
    unicity_0_combinations = []
    unicity_1_combinations = []
    for img_label, aud_label, pointwise_pids in zip(img_labels, audio_labels, list_of_pointwise_pid):
        if img_label + aud_label > 6:
            if img_label > 6 and aud_label > 6:
                redundancy_combinations.append(torch.tensor(pointwise_pids)[None, :])
            elif img_label < 7 and aud_label > 6:
                unicity_1_combinations.append(torch.tensor(pointwise_pids)[None, :])
            elif img_label > 6 and aud_label < 7:
                unicity_0_combinations.append(torch.tensor(pointwise_pids)[None, :])
            else:
                synergy_combinations.append(torch.tensor(pointwise_pids)[None, :])
        else:
            synergy_combinations.append(torch.tensor(pointwise_pids)[None, :])

    print(torch.cat(synergy_combinations).shape)

    print("Synergy Combinations:", torch.mean(torch.cat(synergy_combinations), dim=0))
    print("Redundancy Combinations:", torch.mean(torch.cat(redundancy_combinations), dim=0))
    print("Unicity 0 Combinations:", torch.mean(torch.cat(unicity_0_combinations), dim=0))
    print("Unicity 1 Combinations:", torch.mean(torch.cat(unicity_1_combinations), dim=0))

    # POINTWISE COSINE SIMILARITY
    list_of_pointwise_pids = [torch.cat(redundancy_combinations), torch.cat(unicity_0_combinations),
                              torch.cat(unicity_1_combinations), torch.cat(synergy_combinations)]
    list_pointwise_labels = [torch.cat([torch.tensor([0, 0, 1, 0])[None, :]]*len(list_of_pointwise_pids)),
                             torch.cat([torch.tensor([1, 0, 0, 0])[None, :]] * len(list_of_pointwise_pids)),
                             torch.cat([torch.tensor([0, 1, 0, 0])[None, :]] * len(list_of_pointwise_pids)),
                             torch.cat([torch.tensor([0, 0, 0, 1])[None, :]] * len(list_of_pointwise_pids))]

    pid_names = ["R", "U0", "U1", "S"]
    for pid, pid_labels, pid_name in zip(list_of_pointwise_pids, list_pointwise_labels, pid_names):
        pid = np.maximum(pid, 0)
        row_sums = pid.sum(axis=1, keepdims=True)

        # Replace zero rows with uniform distribution
        zero_rows = row_sums.squeeze() == 0
        pid[zero_rows] = 1.0 / pid.shape[1]
        # Now renormalize safely
        pid_norm = pid / pid.sum(axis=1, keepdims=True)

        # L2 normalize both vectors
        pid_l2 = pid_norm / np.linalg.norm(pid_norm, axis=1, keepdims=True)
        pid_labels_l2 = pid_labels / np.linalg.norm(pid_labels, axis=1, keepdims=True)

        sim_pointwise = np.sum(pid_l2 * pid_labels_l2, axis=1)

        print("Mean true per-sample cosine similarity:", sim_pointwise.mean())


if __name__ == '__main__':
    args = config().parse_args()
    torch.manual_seed(args.seed)

    mnist(args)
