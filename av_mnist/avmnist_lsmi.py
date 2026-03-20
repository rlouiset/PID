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
    fsdd = TorchFSDDGenerator(version='local', path='/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10', transforms=transforms,
                              load_all=True) #  '/Users/robinlouiset/Documents/torch-fsdd/lib/test/data/v1.0.10'

    # Create two Torch datasets for a train-test split from the generator
    train_set, test_set = fsdd.train_test_split(test_size=0.2)
    return train_set, test_set


def prepare_dataset(args):
    train_kwargs = {'batch_size': args.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': args.test_batch_size, 'shuffle': False}
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
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

    print(model)

    # =======================
    # 2. TRAINING
    # =======================
    for epoch in range(1, args.epoch + 1):

        print(f"\n===== Epoch {epoch} =====")

        train(args, model, device, AV_train, optimizer, epoch)

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

    torch.save(lsmi_data, "lsmi.pt")
    print(f"[✓] Saved LSMI data to lsmi.pt")

    # -------------------------
    # Fake cfg replacement
    # -------------------------
    class CFG:
        pass

    cfg = CFG()
    cfg.device = "cuda"
    cfg.batch_size = 512
    cfg.num_workers = 0
    cfg.embed_size = 128
    cfg.n_classes = 2
    cfg.num_epochs_discriminator = 30
    cfg.num_epochs_entropy_estimator = 30

    cfg.input_size_1 = train_vis.shape[1]
    cfg.input_size_2 = train_aud.shape[1]

    setup_seed(args.seed)

    # -------------------------
    # Load data
    # -------------------------
    train_loader, val_loader = get_loader(cfg, "lsmi.pt")

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
    list_of_pointwise_pid = np.stack([u1.detach().cpu().numpy(),
                                      u2.detach().cpu().numpy(),
                                      r.detach().cpu().numpy(),
                                      s.detach().cpu().numpy()], axis=1)  # shape (N, 4)

    # RUS adjustement
    r, u1, u2, s = RUS_adjustment([torch.tensor(r), torch.tensor(u1), torch.tensor(u2), torch.tensor(s)])
    r, u1, u2, s = (r.detach().cpu().numpy(), u1.detach().cpu().numpy(),
                    u2.detach().cpu().numpy(), s.detach().cpu().numpy())

    print("after adjustement")
    print("r: ", np.mean(r))
    print("u1: ", np.mean(u1))
    print("u2: ", np.mean(u2))
    print("s: ", np.mean(s))

if __name__ == '__main__':
    args = config().parse_args()
    torch.manual_seed(args.seed)

    mnist(args)
