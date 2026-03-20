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
