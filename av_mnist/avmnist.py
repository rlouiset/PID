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
from model import CNN, FCN
from dataset import AV_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def config():
    parser = argparse.ArgumentParser(description='Pytorch audiovisual MNIST digit classification')
    parser.add_argument('--model', type=str, default='CNN', help='FCN or CNN')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N', help='input batch size for training')
    parser.add_argument('--test-batch-size', type=int, default=1024, metavar='N', help='input batch size for testing')
    parser.add_argument('--epoch', type=int, default=200, metavar='N', help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.04, metavar='LR', help='learning rate')
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
    # Set number of features and classes
    n_mfcc = 28
    n_digits = 10

    # Specify transformations to be applied to the raw audio
    transforms = Compose([
        # Trim silence from the start and end of the audio
        TrimSilence(threshold=1e-6),
        # Generate n_mfcc+1 MFCCs (and remove the first one since it is a constant offset)
        lambda audio: MFCC(sample_rate=8e3, n_mfcc=n_mfcc + 1)(audio)[1:, :],
        # Standardize MFCCs for each frame
        lambda mfcc: (mfcc - mfcc.mean(axis=0)) / mfcc.std(axis=0),
        # Transpose from DxT to TxD
        lambda mfcc: mfcc.transpose(1, 0),
        # Resize into 28x28
        lambda mfcc: Resize(size=(28, 28))(mfcc.unsqueeze(0)).squeeze(),
    ])

    # Initialize a generator for a local version of FSDD
    fsdd = TorchFSDDGenerator(version='local', path='/home/rlouiset/PID/torch-fsdd/lib/test/data/v1.0.10', transforms=transforms,
                              load_all=True)

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
        output = model(imgs, audios)
        loss = F.cross_entropy(output, labels)
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
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for imgs, audios, labels in test_loader:
            imgs, audios, labels = imgs.to(device), audios.to(device), labels.to(device)
            output = model(imgs, audios, unimodal)
            test_loss += F.cross_entropy(output, labels, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    print('[{} testset] Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        unimodal, test_loss, correct, len(test_loader.dataset), 100. * test_acc))
    return test_acc


def test(model, device, test_loader):
    acc = test_unit(model, device, test_loader)
    visual_acc = test_unit(model, device, test_loader, 'visual')
    audio_acc = test_unit(model, device, test_loader, 'audio')
    return acc, visual_acc, audio_acc


def mnist(args):
    AV_train, AV_test = prepare_dataset(args)

    Ls = np.zeros(args.epoch)
    acc, V_acc, A_acc = np.copy(Ls), np.copy(Ls), np.copy(Ls)

    if args.model == 'FCN':
        model = FCN(args.depth, args.fuse_depth).to(device)
    elif args.model == 'CNN':
        model = CNN(args.depth, args.fuse_depth).to(device)
    else:
        raise NotImplementedError
    print(model)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epoch + 1):
        acc[epoch - 1], V_acc[epoch - 1], A_acc[epoch - 1] = test(model, device, AV_test)
        Ls[epoch - 1] = train(args, model, device, AV_train, optimizer, epoch)
        scheduler.step()

    vis(args, Ls, acc, V_acc, A_acc)


if __name__ == '__main__':
    args = config().parse_args()
    torch.manual_seed(args.seed)

    mnist(args)
