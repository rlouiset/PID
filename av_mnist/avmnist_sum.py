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
from dataset import AV_dataset_sum
from utils_ours import return_redundancy_test_performances, compute_PID_categorical

import torch
import torch.nn.functional as F
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
from torchvision.transforms import Compose

TARGET_FRAMES = 64

softmax = torch.nn.Softmax(dim=-1)

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
    parser.add_argument('--epoch', type=int, default=30, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate')
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
    if args.model == 'FCN':
        L = args.depth
    elif args.model == 'CNN':
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
    fsdd = TorchFSDDGenerator(version='local', path='/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10', transforms=transforms,
                              load_all=True) #  '/Users/robinlouiset/Documents/torch-fsdd/lib/test/data/v1.0.10'

    # Create two Torch datasets for a train-test split from the generator
    train_set, test_set = fsdd.train_test_split(test_size=0.1)
    return train_set, test_set


def prepare_dataset(args):
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    cuda_kwargs = {'num_workers': 0,
                   'pin_memory': True,
                   'shuffle': True}
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
        output, output_img, output_aud, output_digit_img, output_digit_aud = model.forward(imgs, audios, unimodal="train")
        loss = 4 * F.cross_entropy(output[labels==0], labels[labels==0]) + F.cross_entropy(output[labels==1], labels[labels==1])
        loss += F.cross_entropy(output_digit_img, labels_img)
        loss += F.cross_entropy(output_digit_aud, labels_aud)
        loss += 4 * F.cross_entropy(output_img[labels == 0], labels[labels == 0]) + F.cross_entropy(output_img[labels == 1], labels[labels == 1])
        loss += 4 * F.cross_entropy(output_aud[labels==0], labels[labels==0]) + F.cross_entropy(output_aud[labels==1], labels[labels==1])
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
        for imgs, audios, labels, _ ,_ in test_loader:

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

    with torch.no_grad():

        for imgs, audios, labels, _, _ in loader:

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
    AV_train, AV_test = prepare_dataset(args)

    Ls = np.zeros(args.epoch)
    acc, V_acc, A_acc = np.copy(Ls), np.copy(Ls), np.copy(Ls)
    ce, V_ce, A_ce = np.copy(Ls), np.copy(Ls), np.copy(Ls)

    model = CNN(num_classes=2).to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epoch + 1):
        acc[epoch - 1],  ce[epoch - 1],  V_acc[epoch - 1], V_ce[epoch - 1],  A_acc[epoch - 1], A_ce[epoch - 1] = test(model, device, AV_test)
        Ls[epoch - 1] = train(args, model, device, AV_train, optimizer, epoch)
        scheduler.step()

    vis(args, Ls, acc, V_acc, A_acc)

    train_vis, train_aud, y_train = extract_representations(model, AV_train, device)
    test_vis, test_aud, y_test = extract_representations(model, AV_test, device)

    X_train_dict = {
        "modality0": train_vis.float(),
        "modality1": train_aud.float()
    }

    X_test_dict = {
        "modality0": test_vis.float(),
        "modality1": test_aud.float()
    }

    y_pred_dict = return_redundancy_test_performances(X_train_dict, X_train_dict, X_test_dict, y_train, y_train, y_test,
                                                      "redundancy", distribution_target="categorical",
                                                      num_classes=2)

    results = {}
    for key in ["modality0", "modality1", "average"]:
        acc_, ce_ = traditional_cross_entropy_from_probs(softmax(y_pred_dict[key]), y_pred_dict["targets"])
        results[key] = {"accuracy": acc_, "cross_entropy": ce_}

    for k, v in results.items():
        print("redundancy representations - " + f"{k:10s} | acc = {v['accuracy']:.4f}, CE = {v['cross_entropy']:.4f}")

    compute_PID_categorical(ce[-1], V_ce[-1], A_ce[-1], results["average"]["cross_entropy"],
                            num_classes=2)

if __name__ == '__main__':
    args = config().parse_args()
    torch.manual_seed(args.seed)

    mnist(args)
