import os
import sys
sys.path.append(os.getcwd())
import math
import numpy as np
import torch
from torch import nn
import pickle
from itertools import chain, combinations
from collections import namedtuple
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num-data", default=10000, type=int)
parser.add_argument("--modality-number", default=2, type=int)
parser.add_argument("--feature-dim", default=100, type=int)
parser.add_argument("--feature-sep", default=0.5, type=float)
parser.add_argument("--label-dim", nargs='+', default=[50], type=int)
parser.add_argument("--transform-dim", default=100, type=int)
parser.add_argument('--setting', default='redundancy', type=str)
parser.add_argument('--mix-ratio', nargs='+', default=None, type=float)
parser.add_argument('--out-path', default='MultiBench/synthetic', type=str)
args = parser.parse_args()

def save_data(data, filename):
    with open(os.path.join(args.out_path, filename), 'wb') as f:
        pickle.dump(data, f)


num_data = args.num_data
n_modality = args.modality_number
mix_ratio = args.mix_ratio if args.mix_ratio else np.random.rand(4,)
dim_info = {'redundancy':[0, 0, 1], 'uniqueness0':[1, 0, 0], 'uniqueness1':[0, 1, 0], 'synergy':[1, 1, 0]}

if args.setting not in dim_info.keys():
    assert len(mix_ratio) == 4, "mix_ratio has the wrong shape"
    assert np.sum(mix_ratio) == 1, "mix_ratio does not sum to 1"
    mix_ratio = mix_ratio
else:
    if args.setting == "redundancy":
        mix_ratio = [0, 0, 1, 0]
    if args.setting == "uniqueness0":
        mix_ratio = [1, 0, 0, 0]
    if args.setting == "uniqueness1":
        mix_ratio = [0, 1, 0, 0]
    if args.setting == "synergy":
        mix_ratio = [0, 0, 0, 1]
mix_ratio = np.array(mix_ratio)

# first create one dataset per redundancy, uniquenes and synergy
intersections = chain.from_iterable(combinations(np.arange(n_modality), r) for r in range(1, n_modality+1))
intersections = [''.join([str(i) for i in x]) for x in intersections]
feature_dim = [args.feature_dim] * len(intersections)
feature_sep = args.feature_sep
DimInfo = namedtuple('DimInfo', ['dim', 'sep'])

dict_of_feature_dim_info = {}
dict_of_label_dim_info = {}
for setting in dim_info:
    label_dim = np.array(dim_info[setting]) * np.array(feature_dim)
    feature_dim_info = dict()
    label_dim_info = dict()
    for (i, x) in enumerate(intersections):
        feature_dim_info[x] = DimInfo(feature_dim[i], feature_sep)
        label_dim_info[x] = label_dim[i]
    dict_of_feature_dim_info[setting] = feature_dim_info
    dict_of_label_dim_info[setting] = label_dim_info

dict_of_total_data = {}
dict_of_total_labels = {}
for setting in dim_info:
    total_data = [[] for _ in range(n_modality)]
    total_reg = []

    dataset = []
    for _ in range(num_data):

        if setting != "synergy":
            transforms = dict()
            for x in intersections:
                transforms[x] = np.random.uniform(0.0, 1.0,
                                                  (dict_of_feature_dim_info[setting][x].dim, args.transform_dim))
            label_transform = nn.Sequential(nn.Dropout(0.1))


            raw_features = dict()
            for k, d in dict_of_feature_dim_info[setting].items():
                raw_features[k] = np.random.multivariate_normal(np.zeros((d.dim,)), np.eye(d.dim)*d.sep, (1,))[0]
            modality_data = []
            for i in range(n_modality):
                modality_data.append([])
                for k, v in raw_features.items():
                    if str(i) in str(k):
                        modality_data[-1].append(v @ transforms[k])
            modality_data = [np.concatenate(data) for data in modality_data]

            label_components = []
            for k, d in dict_of_label_dim_info[setting].items():
                label_components.append(raw_features[k][:d])
            label_vector = np.concatenate(label_components)
            label_vector = label_transform(torch.Tensor(label_vector)).detach().numpy()
            label_reg = np.mean(label_vector)
            dataset.append((label_reg, modality_data))

        else:
            transforms_0 = dict()
            for x in intersections:
                transforms_0[x] = np.random.uniform(0.0, 1.0,  (dict_of_feature_dim_info[setting][x].dim, args.transform_dim))
            label_transform_0 = nn.Sequential(nn.Dropout(0.0))

            transforms_1 = dict()
            for x in intersections:
                transforms_1[x] = np.random.uniform(0.0, 1.0,
                                                    (dict_of_feature_dim_info[setting][x].dim, args.transform_dim))
            label_transform_1 = nn.Sequential(nn.Dropout(0.0))

            raw_features = dict()
            for k, d in dict_of_feature_dim_info[setting].items():
                raw_features[k] = np.random.multivariate_normal(np.zeros((d.dim,)), np.eye(d.dim)*d.sep, (1,))[0]
            modality_data = []
            for i in range(n_modality):
                modality_data.append([])
                for k, v in raw_features.items():
                    if str(i) in str(k):
                        if i == 0:
                            modality_data[-1].append(v @ transforms_0[k])
                        if i == 1:
                            modality_data[-1].append(v @ transforms_1[k])
            modality_data = [np.concatenate(data) for data in modality_data]

            label_vector_0 = np.copy(np.array(raw_features['0'])[:args.label_dim[0]])
            label_vector_0 = label_transform_0(torch.Tensor(label_vector_0)).detach().numpy()
            label_reg_0 = np.mean(label_vector_0)

            label_vector_1 = np.copy(np.array(raw_features['1'])[:args.label_dim[0]])
            label_vector_1 = label_transform_1(torch.Tensor(label_vector_1)).detach().numpy()
            label_reg_1 = np.mean(label_vector_1)

            label_reg = label_reg_0 * (1 - label_reg_1) + (1 - label_reg_0) * label_reg_1  # in [0,1]

            dataset.append((label_reg, modality_data))

    dataset = sorted(dataset, key=lambda x: x[0])

    total_reg = np.array([dataset_i[0] for dataset_i in dataset])
    for i in range(n_modality):
        for x in dataset:
            total_data[i].append(x[1][i])
    for i in range(n_modality):
        total_data[i] = np.vstack(total_data[i])
        assert(len(total_data[i]) == len(total_reg))

    dict_of_total_data[setting] = total_data
    total_reg = total_reg - np.min(total_reg)
    total_reg = total_reg / np.max(total_reg)
    dict_of_total_labels[setting] = total_reg

# mix data if needed, get total_wieghts
concentration = 10.0  # ↑ less variance, ↓ more randomness
dirichlet_alpha = concentration * mix_ratio
total_weights = np.random.dirichlet(dirichlet_alpha+1e-5, size=num_data) # shape: (num_data, 4)

# get total data, labels and weight
total_data = [[] for _ in range(n_modality)]
total_labels = []
for i in range(num_data):
    for m in range(n_modality):
        x = (
            total_weights[i][0]  * dict_of_total_data["uniqueness0"][m][i]
          + total_weights[i][1]  * dict_of_total_data["uniqueness1"][m][i]
          + total_weights[i][2] * dict_of_total_data["redundancy"][m][i]
          + total_weights[i][3] * dict_of_total_data["synergy"][m][i]
        )
        total_data[m].append(x)
    total_labels.append(0.25*dict_of_total_labels["uniqueness0"][i]
                        + 0.25*dict_of_total_labels["uniqueness1"][i]
                        + 0.25*dict_of_total_labels["redundancy"][i]
                        + 0.25*dict_of_total_labels["synergy"][i]
                        )

# ---------- FINALIZE DATA ----------
total_data = [np.vstack(x) for x in total_data]
total_labels = np.array(total_labels)
total_weights = np.array(total_weights)

# ---------- TRAIN / VAL / TEST SPLIT ----------
X = np.array(total_data).transpose((1, 0, 2))  # (N, M, D)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X,
    total_labels,
    total_weights,
    test_size=0.3
)

X_valid, X_test, y_valid, y_test, w_valid, w_test = train_test_split(
    X_test,
    y_test,
    w_test,
    test_size=0.5
)

# ---------- STORE ----------
data = dict()
data['train'] = dict()
data['valid'] = dict()
data['test'] = dict()

for i in range(n_modality):
    data['train'][str(i)] = X_train[:, i, :]
    data['valid'][str(i)] = X_valid[:, i, :]
    data['test'][str(i)]  = X_test[:, i, :]

data['train']['label'] = y_train
data['valid']['label'] = y_valid
data['test']['label']  = y_test

data['train']['weights'] = w_train
data['valid']['weights'] = w_valid
data['test']['weights']  = w_test

print(args.setting)
print("Train weights mean:", data['train']['weights'].mean(axis=0))
print("Valid weights mean:", data['valid']['weights'].mean(axis=0))
print("Test weights mean :", data['test']['weights'].mean(axis=0))
print('---')

# ---------- SAVE ----------
save_data(data, "reg_DATA_{}.pickle".format(args.setting))
