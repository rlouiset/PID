import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import torch
import sys
import os
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.getcwd()))
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from utils.helper_modules import Sequential2 # noqa
from unimodals.common_models import MLP, Linear, MLP3
from synthetic.get_data import get_dataloader
import torch
import torch.nn as nn

from synthetic.updated_redundancy_aware_supervised_learning import train, test
from synthetic.supervised_learning import MMDL

from fusions.common_fusions import Concat


# Datasets

class MultimodalDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        self.num_modalities = len(self.data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return tuple([self.data[i][idx] for i in range(self.num_modalities)] + [self.labels[idx]])


# Models

batch_size = 32

def sinkhorn_probs(matrix, x1_probs, x2_probs, n_iters=50):
    for _ in range(n_iters):
        matrix = matrix / (matrix.sum(dim=0, keepdim=True) + 1e-8) * x2_probs[None]
        matrix = matrix / (matrix.sum(dim=1, keepdim=True) + 1e-8) * x1_probs[:, None]
    return matrix


def mlp(dim, hidden_dim, output_dim, layers, activation):
    activation = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
    }[activation]

    seq = [nn.Linear(dim, hidden_dim), activation()]
    for _ in range(layers):
        seq += [nn.Linear(hidden_dim, hidden_dim), activation()]
    seq += [nn.Linear(hidden_dim, output_dim)]

    return nn.Sequential(*seq)


def simple_discrim(xs, y, num_labels):
    shape = [x.size(1) for x in xs] + [num_labels]
    p = torch.ones(*shape) * 1e-8
    for i in range(len(y)):
        p[tuple([torch.argmax(x[i]).item() for x in xs] + [y[i].item()])] += 1
    p /= torch.sum(p)

    def f(*x):
        x = [torch.argmax(xx, dim=1) for xx in x]
        return torch.log(p[tuple(x)])

    return f


class Discrim(nn.Module):
    def __init__(self, x_dim, hidden_dim, num_labels, layers, activation):
        super().__init__()
        self.mlp = mlp(x_dim, hidden_dim, num_labels, layers, activation)

    def forward(self, *x):
        x = torch.cat(x, dim=-1)
        return self.mlp(x)

import math

class CEAlignment(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation):
        super().__init__()
        self.num_labels = num_labels
        self.mlp1 = mlp(x1_dim, hidden_dim, embed_dim * num_labels, layers, activation)  # Outputs [B, Y*embed]
        self.mlp2 = mlp(x2_dim, hidden_dim, embed_dim * num_labels, layers, activation)

    def forward(self, x1, x2, x1_probs, x2_probs):  # x1_probs = p_y_x1 [B, Y]
        # Embed x1/x2 into per-class representations: [B, Y, embed]
        batch_size = x1.size(0)  # Get batch size explicitly
        q_x1 = self.mlp1(x1).view(batch_size, self.num_labels, -1)  # [B, Y, embed]
        q_x2 = self.mlp2(x2).view(batch_size, self.num_labels, -1)  # [B, Y, embed]

        # Rest of your code stays the same...
        # L2 normalize per class embedding (stabilizes dot products)
        q_x1 = (q_x1 - q_x1.mean(dim=-1, keepdim=True)) / (q_x1.var(dim=-1, keepdim=True) + 1e-8).sqrt()
        q_x2 = (q_x2 - q_x2.mean(dim=-1, keepdim=True)) / (q_x2.var(dim=-1, keepdim=True) + 1e-8).sqrt()

        # Fix: Correct einsum for dot-product similarity per batch, per y1-y2 pair
        # q_x1 [B, Y, D], q_x2 [B, Y, D] -> align [B, Y, Y] (similarity(y1 from x1, y2 from x2))
        align_logits = torch.einsum('b y d, b z d -> b y z', q_x1, q_x2) / math.sqrt(q_x1.size(-1))
        align = torch.exp(align_logits)

        # Sinkhorn normalization: Normalize align to match marginals p(y|x1), p(y|x2) per batch
        # (This approximates optimal transport coupling Q(y2 | x1, y1) ~ p(y2|x2))
        normalized = []
        for i in range(self.num_labels):  # Per output class? But original loops over last dim; adjust if needed
            current = align[..., i]  # [B, Y] if align [B,Y,Y] and i over last Y? Wait, original assumes [B,Y,Y]
            # Original loop is over align.size(-1)=Y, so current [B, Y_in] where Y_in=Y
            for _ in range(50):  # Reduced iters for efficiency (original has 500, but converges fast)
                current = current / (current.sum(dim=-1, keepdim=True) + 1e-8) * x2_probs  # Normalize rows to p(y2|x2)
                current = current / (current.sum(dim=1, keepdim=True) + 1e-8) * x1_probs  # Normalize cols to p(y1|x1)
            normalized.append(current)
        normalized = torch.stack(normalized, dim=-1)  # [B, Y, Y] if stacked over Y

        if torch.isnan(normalized).any():
            raise ValueError('NaN in normalized align; check embeddings or probs.')

        return normalized  # [B, Y, Y]


class CEAlignmentInformation(nn.Module):
    def __init__(self, x1_dim, x2_dim, hidden_dim, embed_dim, num_labels,
                 layers, activation, discrim_1, discrim_2, discrim_12, p_y):
        super().__init__()
        self.num_labels = num_labels
        self.align = CEAlignment(x1_dim, x2_dim, hidden_dim, embed_dim, num_labels, layers, activation)
        self.discrim_1 = discrim_1
        if isinstance(self.discrim_1, nn.Module):
            self.discrim_1.eval()
        self.discrim_2 = discrim_2
        if isinstance(self.discrim_2, nn.Module):
            self.discrim_2.eval()
        self.discrim_12 = discrim_12
        if isinstance(self.discrim_12, nn.Module):
            self.discrim_12.eval()
        self.register_buffer('p_y', p_y)
        # self.critic_1y = SeparableCritic(x1_dim, y_dim, hidden_dim, embed_dim, layers, activation)
        # self.critic_2y = SeparableCritic(x2_dim, y_dim, hidden_dim, embed_dim, layers, activation)
        # self.critic_12y = SeparableCritic(x1_dim + x2_dim, y_dim, hidden_dim, embed_dim, layers, activation)

    def align_parameters(self):
        return list(self.align.parameters())

    def forward(self, x1, x2, y):
        # print('forward', x1.shape, x2.shape, y.shape)
        with torch.no_grad():
            a = self.discrim_1(x1)
            # print('a', a.shape)
            b = self.discrim_2(x2)
            # print('b', b.shape)
            p_y_x1 = nn.Softmax(dim=-1)(a)
            p_y_x2 = nn.Softmax(dim=-1)(b)
        align = self.align(torch.flatten(x1, 1, -1), torch.flatten(x2, 1, -1), p_y_x1, p_y_x2)
        # print(p_y_x2)
        # print(self.p_y)
        # print(y.squeeze(-1))
        y = nn.functional.one_hot(y.squeeze(-1).long(), num_classes=self.num_labels)
        self.p_y[self.p_y == 0] += 1e-8
        self.p_y[self.p_y == 1] -= 1e-8

        # sample method: P(X1)
        # coeff: P(Y | X1) Q(X2 | X1, Y)
        # log term: log Q(X2 | X1, Y) - logsum_Y' Q(X2 | X1, Y') Q(Y' | X1)

        q_x2_x1y = align / (torch.sum(align, dim=1, keepdim=True) + 1e-8)
        # print(torch.cat([1 - y, y], dim=-1).shape)
        log_term = torch.log(q_x2_x1y + 1e-8) - torch.log(torch.einsum('aby, ay -> ab', q_x2_x1y, p_y_x1) + 1e-8)[:, :, None]
        # print(q_x2_x1y)
        # print(log_term)
        # That's all we need for optimization purposes
        loss = torch.mean(torch.sum(torch.sum(p_y_x1[:, None, :] * q_x2_x1y * log_term, dim=-1), dim=-1))
        # Now, we calculate the MI terms
        p_y_x1_sampled = torch.sum(p_y_x1 * y, dim=-1)
        p_y_x2_sampled = torch.sum(p_y_x2 * y, dim=-1)
        # print(p_y_x2_sampled)
        with torch.no_grad():
            p_y_x1x2 = nn.Softmax(dim=-1)(self.discrim_12([x1, x2]))
        p_y_x1x2_sampled = torch.sum(p_y_x1x2 * y, dim=-1)
        p_y_sampled = torch.sum(self.p_y[None] * y, dim=-1)

        p1 = p_y_x1.detach().clone()
        p1[p1 == 0] += 1e-8
        log_p_y_x1 = torch.log(p1)
        # log_p_y_x1[log_p_y_x1 == float("-Inf")] += 1e-8
        p2 = p_y_x2.detach().clone()
        p2[p2 == 0] += 1e-8
        log_p_y_x2 = torch.log(p2)
        # log_p_y_x2[log_p_y_x2 == float("-Inf")] += 1e-8
        p12 = p_y_x1x2.detach().clone()
        p12[p12 == 0] += 1e-8
        log_p_y_x1x2 = torch.log(p12)
        # log_p_y_x1x2[log_p_y_x1x2 == float("-Inf")] += 1e-8

        # mi_y_x1 = torch.mean(torch.log(p_y_x1_sampled) - torch.log(p_y_sampled))
        mi_y_x1 = torch.mean(torch.sum(p_y_x1 * (log_p_y_x1 - torch.log(self.p_y)[None]), dim=-1))
        # mi_y_x2 = torch.mean(torch.log(p_y_x2_sampled) - torch.log(p_y_sampled))
        mi_y_x2 = torch.mean(torch.sum(p_y_x2 * (log_p_y_x2 - torch.log(self.p_y)[None]), dim=-1))
        # mi_y_x1x2 = torch.mean(torch.log(p_y_x1x2_sampled) - torch.log(p_y_sampled))
        mi_y_x1x2 = torch.mean(torch.sum(p_y_x1x2 * (log_p_y_x1x2 - torch.log(self.p_y)[None, None]), dim=-1))
        mi_q_y_x1x2 = p_y_x1[:, None, :] * q_x2_x1y * (log_term + torch.log(p_y_x1 + 1e-8)[:, None, :] - torch.log(self.p_y + 1e-8)[None, None, :])
        '''
        if not self.training:
            print(p_y_x1)
            print(q_x2_x1y)
            print(log_term)
            print(torch.log(p_y_x1))
            print(torch.log(self.p_y))
            print(log_term + torch.log(p_y_x1)[:, None, :] - torch.log(self.p_y)[None, None, :])
        '''
        mi_q_y_x1x2 = torch.sum(torch.sum(mi_q_y_x1x2, dim=-1), dim=-1) # anchored by x1 -- take mean to get MI
        mi_q_y_x1x2 = torch.mean(mi_q_y_x1x2)

        '''
        if not self.training:
            print(torch.stack([mi_y_x1, mi_y_x2, mi_y_x1x2, mi_q_y_x1x2]))
        '''
        # print('   m', torch.stack([mi_y_x1, mi_y_x2, mi_y_x1x2, mi_q_y_x1x2]))

        redundancy = mi_y_x1 + mi_y_x2 - mi_q_y_x1x2
        unique1 = mi_q_y_x1x2 - mi_y_x2
        unique2 = mi_q_y_x1x2 - mi_y_x1
        synergy = mi_y_x1x2 - mi_q_y_x1x2

        # print('   r', torch.stack([redundancy, unique1, unique2, synergy]))

        return loss, torch.stack([redundancy, unique1, unique2, synergy], dim=0), align

# Training Loops
from tqdm import tqdm


def train_discrim(model, train_loader, optimizer, data_type, num_epoch=40):
    for _iter in range(num_epoch):
        print(_iter)
        for i_batch, data_batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()

            inputs = []
            for j in range(len(data_type)):
                xs = [data_batch[data_type[j][i] - 1] for i in range(len(data_type[j]))]
                x_batch = torch.cat(xs, dim=1)
                if j != len(data_type) - 1:
                    x_batch = x_batch.float()
                inputs.append(x_batch)
            y = inputs[-1]
            inputs = inputs[:-1]

            logits = model(*inputs)
            loss = nn.CrossEntropyLoss()(logits, y.squeeze(-1))
            loss.backward()

            optimizer.step()

            if (_iter + 1) % 20 == 0 and i_batch % 1024 == 0:
                print('iter: ', _iter, ' i_batch: ', i_batch, ' loss: ', loss.item())


def eval_discrim(model, test_loader, data_type):
    losses = []
    for i_batch, data_batch in enumerate(test_loader):
        inputs = []
        for j in range(len(data_type)):
            xs = [data_batch[data_type[j][i] - 1] for i in range(len(data_type[j]))]
            x_batch = torch.cat(xs, dim=1)
            if j != len(data_type) - 1:
                x_batch = x_batch.float()
            inputs.append(x_batch)
        y = inputs[-1]
        inputs = inputs[:-1]

        logits = model(*inputs)
        loss = nn.CrossEntropyLoss()(logits, y.squeeze(-1))
        losses.append(loss.item())

        if i_batch % 1024 == 0:
            print('i_batch: ', i_batch, ' loss: ', loss.item())
    print('Eval loss:', sum(losses) / len(losses))


def train_ce_alignment(model, train_loader, opt_align, data_type, num_epoch=10):
    for _iter in range(num_epoch):
        print(_iter)
        for i_batch, data_batch in enumerate(tqdm(train_loader)):
            opt_align.zero_grad()

            x1s = [data_batch[data_type[0][i] - 1] for i in range(len(data_type[0]))]
            x2s = [data_batch[data_type[1][i] - 1] for i in range(len(data_type[1]))]
            ys = [data_batch[data_type[2][i] - 1] for i in range(len(data_type[2]))]

            x1_batch = torch.cat(x1s, dim=1).float()
            x2_batch = torch.cat(x2s, dim=1).float()

            y_batch = ys[0]

            loss, _, _ = model(x1_batch, x2_batch, y_batch)
            loss.backward()

            opt_align.step()

            # if (_iter + 1) % 1 == 0 and i_batch % 1 == 0:
            #     print('iter: ', _iter, ' i_batch: ', i_batch, ' align_loss: ', loss.item())


def eval_ce_alignment(model, test_loader, data_type):
    results = []
    aligns = []

    for i_batch, data_batch in enumerate(test_loader):
        x1s = [data_batch[data_type[0][i] - 1] for i in range(len(data_type[0]))]
        x2s = [data_batch[data_type[1][i] - 1] for i in range(len(data_type[1]))]
        ys = [data_batch[data_type[2][i] - 1] for i in range(len(data_type[2]))]

        x1_batch = torch.cat(x1s, dim=1).float()
        x2_batch = torch.cat(x2s, dim=1).float()
        y_batch = ys[0]

        with torch.no_grad():
            _, result, align = model(x1_batch, x2_batch, y_batch)
        results.append(result)
        aligns.append(align)

    results = torch.stack(results, dim=0)

    return results, aligns


def critic_ce_alignment(x1, x2, labels, num_labels, train_ds, test_ds, discrim_1=None, discrim_2=None, discrim_12=None,
                        learned_discrim=True, shuffle=True, discrim_epochs=40, ce_epochs=10):
    if discrim_1 is not None:
        model_discrim_1, model_discrim_2, model_discrim_12 = discrim_1, discrim_2, discrim_12
    elif learned_discrim:
        model_discrim_1 = Discrim(x_dim=x1.size(1), hidden_dim=32, num_labels=num_labels, layers=3,
                                  activation='relu')
        model_discrim_2 = Discrim(x_dim=x2.size(1), hidden_dim=32, num_labels=num_labels, layers=3,
                                  activation='relu')
        model_discrim_12 = Discrim(x_dim=x1.size(1) + x2.size(1), hidden_dim=32, num_labels=num_labels, layers=3,
                                   activation='relu')

        for model, data_type in [
            (model_discrim_1, ([1], [0])),
            (model_discrim_2, ([2], [0])),
            (model_discrim_12, ([1], [2], [0])),
        ]:
            optimizer = optim.Adam(model.parameters(), lr=1e-3)
            train_loader1 = DataLoader(train_ds, shuffle=shuffle, drop_last=True,
                                       batch_size=batch_size,
                                       num_workers=1)
            train_discrim(model, train_loader1, optimizer, data_type=data_type, num_epoch=discrim_epochs)
            model.eval()
            test_loader1 = DataLoader(test_ds, shuffle=False, drop_last=False,
                                      batch_size=batch_size, num_workers=1)
            eval_discrim(model, test_loader1, data_type=data_type)
    else:
        model_discrim_1 = simple_discrim(x1, labels, num_labels)
        model_discrim_2 = simple_discrim(x2, labels, num_labels)
        model_discrim_12 = simple_discrim([x1, x2], labels, num_labels)

    p_y = torch.sum(nn.functional.one_hot(labels.squeeze(-1)), dim=0) / len(labels)

    # print(p_y)

    def product(x):
        return x[0] * product(x[1:]) if x else 1

    model = CEAlignmentInformation(x1_dim=product(x1.shape[1:]), x2_dim=product(x2.shape[1:]),
                                   hidden_dim=32, embed_dim=10, num_labels=num_labels, layers=3, activation='relu',
                                   discrim_1=model_discrim_1, discrim_2=model_discrim_2, discrim_12=model_discrim_12,
                                   p_y=p_y)
    opt_align = optim.Adam(model.align_parameters(), lr=1e-3)

    train_loader1 = DataLoader(train_ds, shuffle=shuffle, drop_last=True,
                               batch_size=batch_size,
                               num_workers=1)
    test_loader1 = DataLoader(test_ds, shuffle=False, drop_last=True,
                              batch_size=batch_size,
                              num_workers=1)

    # Train and estimate mutual information
    model.train()
    train_ce_alignment(model, train_loader1, opt_align, data_type=([1], [2], [0]), num_epoch=ce_epochs)

    model.eval()
    results, aligns = eval_ce_alignment(model, test_loader1, data_type=([1], [2], [0]))
    return results, aligns, (model, model_discrim_1, model_discrim_2, model_discrim_12, p_y)

class CEPID(nn.Module):
    def __init__(
        self,
        x1_dim,
        x2_dim,
        num_labels,
        discrim_1,
        discrim_2,
        discrim_12,
        p_y,
        hidden_dim=64,
        embed_dim=16,
        layers=2,
        activation='relu',
    ):
        super().__init__()

        self.num_labels = num_labels
        self.align = CEAlignment(
            x1_dim, x2_dim,
            hidden_dim, embed_dim,
            num_labels, layers, activation
        )

        self.discrim_1 = discrim_1.eval()
        self.discrim_2 = discrim_2.eval()
        self.discrim_12 = discrim_12.eval()

        self.register_buffer("p_y", p_y)

    def forward(self, x1, x2, y):
        with torch.no_grad():
            p_y_x1 = torch.softmax(self.discrim_1(x1), dim=-1)
            p_y_x2 = torch.softmax(self.discrim_2(x2), dim=-1)
            p_y_x12 = torch.softmax(self.discrim_12(torch.cat([x1, x2], dim=1)), dim=-1)

        align = self.align(x1, x2, p_y_x1, p_y_x2)

        y_oh = torch.nn.functional.one_hot(y, self.num_labels).float()

        # q(x2 | x1, y)
        q = align / (align.sum(dim=1, keepdim=True) + 1e-8)

        log_term = (
            torch.log(q + 1e-8)
            - torch.log(
                torch.einsum("aby,ay->ab", q, p_y_x1)[:, :, None] + 1e-8
            )
        )

        mi_q = torch.mean(
            torch.sum(
                p_y_x1[:, None, :] * q * (
                    log_term
                    + torch.log(p_y_x1 + 1e-8)[:, None, :]
                    - torch.log(self.p_y + 1e-8)[None, None, :]
                ),
                dim=(-1, -2)
            )
        )

        mi_x1 = torch.mean(
            torch.sum(p_y_x1 * (torch.log(p_y_x1 + 1e-8) - torch.log(self.p_y)), dim=-1)
        )
        mi_x2 = torch.mean(
            torch.sum(p_y_x2 * (torch.log(p_y_x2 + 1e-8) - torch.log(self.p_y)), dim=-1)
        )
        mi_x12 = torch.mean(
            torch.sum(p_y_x12 * (torch.log(p_y_x12 + 1e-8) - torch.log(self.p_y)), dim=-1)
        )

        R = mi_x1 + mi_x2 - mi_q
        U1 = mi_q - mi_x2
        U2 = mi_q - mi_x1
        S = mi_x12 - mi_q

        return R, U1, U2, S

def CE_PID_estimation(dataloader, discriminator, cfg):
    device = cfg.device

    # Estimate p(y)
    ys = []
    for _, _, y in dataloader:
        ys.append(y)
    y_all = torch.cat(ys).to(device)
    p_y = torch.bincount(y_all, minlength=cfg.n_classes).float()
    p_y /= p_y.sum()

    model = CEPID(
        x1_dim=cfg.input_size_1,
        x2_dim=cfg.input_size_2,
        num_labels=cfg.n_classes,
        discrim_1=discriminator[0],
        discrim_2=discriminator[1],
        discrim_12=discriminator[2],
        p_y=p_y,
    ).to(device)

    Rs, U1s, U2s, Ss = [], [], [], []

    with torch.no_grad():
        for x1, x2, y in dataloader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            R, U1, U2, S = model(x1, x2, y)
            Rs.append(R)
            U1s.append(U1)
            U2s.append(U2)
            Ss.append(S)

    R = torch.mean(torch.stack(Rs))
    U1 = torch.mean(torch.stack(U1s))
    U2 = torch.mean(torch.stack(U2s))
    S = torch.mean(torch.stack(Ss))

    print(f"[CE-PID] R={R:.4f}, U1={U1:.4f}, U2={U2:.4f}, S={S:.4f}")
    return R, U1, U2, S


def get_mm_dataset(modalities, data):
    L = []
    for mod in modalities:
        L.append(data[f'x{mod}'])
    labels = replace_neg(data['labels'])
    return MultimodalDataset(L, labels)

import multiprocessing
multiprocessing.set_start_method('fork', force=True)

import pickle

device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--data-path", default="SIMPLE_DATA_DIM=3_STD=0.5.pickle", type=str, help="input path of synthetic dataset")
parser.add_argument("--keys", nargs='+', default=['a','b','c','d','e','label'], type=str, help="keys to access data of each modality and label, assuming dataset is structured as a dict")
parser.add_argument("--modalities", nargs='+', default=[0,1], type=int, help="specify the index of modalities in keys")
parser.add_argument("--bs", default=32, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--input-dim", nargs='+', default=30, type=int)
parser.add_argument("--hidden-dim", default=512, type=int)
parser.add_argument("--n-latent", default=512, type=int)
parser.add_argument("--rank", default=32, type=int)
parser.add_argument("--num-classes", default=2, type=int)
parser.add_argument("--epochs", default=30, type=int)
parser.add_argument("--lr", default=1e-4, type=float)
parser.add_argument("--weight-decay", default=0.01, type=float)
parser.add_argument("--weight", default=1, type=float)
parser.add_argument("--saved-model", default=None, type=str)
args = parser.parse_args()


class ConcatEarly(nn.Module):
    """Concatenation of input data on dimension 2."""

    def __init__(self):
        """Initialize ConcatEarly Module."""
        super(ConcatEarly, self).__init__()

    def forward(self, modalities):
        """
        Forward Pass of ConcatEarly.

        :param modalities: An iterable of modalities to combine
        """
        return torch.cat(modalities, dim=1)

# Wrap main code to protect multiprocessing
if __name__ == "__main__":
    # Load data
    traindata, validdata, _, testdata = get_dataloader(
        path=args.data_path,
        keys=args.keys,
        modalities=args.modalities,
        batch_size=args.bs,
        num_workers=args.num_workers
    )

    # -------------------------
    # Load pickle
    # -------------------------
    with open(args.data_path, "rb") as f:
        data = pickle.load(f)

    print(data.keys())

    X1_train = torch.from_numpy(data["train"]["0"]).float()
    X2_train = torch.from_numpy(data["train"]["1"]).float()
    y_train = torch.from_numpy(data["train"]["label"]).long().squeeze()

    X1_val = torch.from_numpy(data["test"]["0"]).float()
    X2_val = torch.from_numpy(data["test"]["1"]).float()
    y_val = torch.from_numpy(data["test"]["label"]).long().squeeze()


    # Specify model
    if len(args.input_dim) == 1:
        input_dims = args.input_dim * len(args.modalities)
    else:
        input_dims = args.input_dim
    encoders = [Linear(input_dim, args.hidden_dim).to(device) for input_dim in input_dims]
    #encoders = [nn.Sequential(Linear(input_dim, args.hidden_dim).to(device),
    #                          nn.ReLU(),
    #                          Linear(args.hidden_dim, args.hidden_dim).to(device)) for input_dim in input_dims]

    heads = [MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device) for input_dim in input_dims]

    fusion = nn.Sequential(
        Concat(),
        MLP3(len(args.modalities) * args.hidden_dim, args.n_latent, args.n_latent)
    ).to(device)
    # fusion = ConcatEarly().cuda()
    head = MLP(args.n_latent, args.hidden_dim, args.num_classes).to(device)

    # Training
    model = train(
        encoders,
        fusion,
        head,
        heads,
        traindata,
        validdata,
        args.epochs,
        objective=torch.nn.CrossEntropyLoss(),
        optimtype=torch.optim.AdamW,
        lr=args.lr,
        save=args.saved_model,
        weight_decay=args.weight_decay
    )

    # Testing
    print("Testing (Ours):")
    # model = torch.load(args.saved_model, weights_only=True).to(device)
    test(model, testdata, no_robust=True, criterion=torch.nn.CrossEntropyLoss())

    model12 = MMDL(model.encoders, model.fuse, model.head).to(device)
    model1 = nn.Sequential(model.encoders[0], model.heads[0])
    model2 = nn.Sequential(model.encoders[1], model.heads[1])

    import numpy as np

    train_ds = MultimodalDataset(
        data=[X1_train, X2_train],
        labels=y_train
    )

    test_ds = MultimodalDataset(
        data=[X1_val, X2_val],
        labels=y_val
    )

    replace_neg = np.vectorize(lambda x: 0 if x <= 0 else 1)
    pred = replace_neg(y_train.cpu().numpy())
    # print(pred, len(pred))

    results = critic_ce_alignment(
        X1_train,
        X2_train,
        y_train,
        num_labels=args.num_classes,
        train_ds=train_ds,
        test_ds=test_ds,
        discrim_1=model1,
        discrim_2=model2,
        discrim_12=model12,
        learned_discrim=True,
        shuffle=True,
        discrim_epochs=40,
        ce_epochs=10
    )

    res = results[0].cpu().numpy()
    values = np.mean(res, axis=0)
    values = values / np.log(2)
    values = np.maximum(values, 0)
    print(', '.join([str(v) for v in values]))
    print("Redundancy:", values[0])
    print("Unique1:", values[1])
    print("Unique1:", values[2])
    print("Synergy:", values[3])
