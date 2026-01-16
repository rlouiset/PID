import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pickle
import torch
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from synthetic.supervised_learning import test
from synthetic.ensemble import test as test_ensemble
from synthetic.get_data import get_dataloader
from sklearn.preprocessing import normalize
from sklearn.decomposition import *
from sklearn.cluster import *
from synthetic.rus import *
import pandas as pd


def clustering(X, pca=False, n_clusters=20, n_components=5):
  X = np.nan_to_num(X)
  if len(X.shape) > 2:
    X = X.reshape(X.shape[0],-1)
  if pca:
    # print(np.any(np.isnan(X)), np.all(np.isfinite(X)))
    X = normalize(X)
    X = PCA(n_components=n_components).fit_transform(X)
  kmeans = KMeans(n_clusters=n_clusters).fit(X)
  return kmeans.labels_, X

for setting in ['redundancy', 'uniqueness0', 'uniqueness1', 'synergy']:
  data_dir = 'synthetic/experiments/DATA_{}.pickle'.format(setting)
  dataset = pd.read_pickle(data_dir)
  n_components = 2
  data_cluster = dict()
  for split in ['valid', 'test']:
    data_cluster[split] = dict()
    data = dataset[split]
    kmeans_0, data_0 = clustering(data['0'], pca=True, n_components=n_components, n_clusters=20)
    data_cluster[split]['0'] = kmeans_0.reshape(-1,1)
    kmeans_1, data_1 = clustering(data['1'], pca=True, n_components=n_components, n_clusters=20)
    data_cluster[split]['1'] = kmeans_1.reshape(-1,1)
    data_cluster[split]['label'] = data['label']
  with open('synthetic/experiments/DATA_{}_cluster.pickle'.format(setting), 'wb') as f:
      pickle.dump(data_cluster, f)

for i in range(1, 7):
  data_dir = 'synthetic/experiments/DATA_mix{}.pickle'.format(i)
  dataset = pd.read_pickle(data_dir)
  n_components = 2
  data_cluster = dict()
  for split in ['valid', 'test']:
    data_cluster[split] = dict()
    data = dataset[split]
    kmeans_0, data_0 = clustering(data['0'], pca=True, n_components=n_components, n_clusters=20)
    data_cluster[split]['0'] = kmeans_0.reshape(-1,1)
    kmeans_1, data_1 = clustering(data['1'], pca=True, n_components=n_components, n_clusters=20)
    data_cluster[split]['1'] = kmeans_1.reshape(-1,1)
    data_cluster[split]['label'] = data['label']
  with open('synthetic/experiments/DATA_mix{}_cluster.pickle'.format(i), 'wb') as f:
      pickle.dump(data_cluster, f)

for i in range(1, 6):
    data_dir = 'synthetic/experiments/DATA_synthetic{}.pickle'.format(i)
    dataset = pd.read_pickle(data_dir)
    n_components = 2
    data_cluster = dict()
    for split in ['valid', 'test']:
        data_cluster[split] = dict()
        data = dataset[split]
        kmeans_0, data_0 = clustering(data['0'], pca=True, n_components=n_components, n_clusters=20)
        data_cluster[split]['0'] = kmeans_0.reshape(-1,1)
        kmeans_1, data_1 = clustering(data['1'], pca=True, n_components=n_components, n_clusters=20)
        data_cluster[split]['1'] = kmeans_1.reshape(-1,1)
        data_cluster[split]['label'] = data['label']
    with open('synthetic/experiments/DATA_synthetic{}_cluster.pickle'.format(i), 'wb') as f:
        pickle.dump(data_cluster, f)

if os.path.isfile('synthetic/experiments/datasets.pickle'):
    with open('synthetic/experiments/datasets.pickle', 'rb') as f:
        results = pickle.load(f)
else:
    results = dict()
SETTINGS = ['redundancy', 'uniqueness0', 'uniqueness1', 'synergy'] + ['mix{}'.format(i) for i in range(1,7)] + ['synthetic{}'.format(i) for i in range(1,6)]
for setting in ['mix6']:
    with open('synthetic/experiments/DATA_{}_cluster.pickle'.format(setting), 'rb') as f:
        dataset = pickle.load(f)
    print(setting)
    data = (dataset['test']['0'], dataset['test']['1'], dataset['test']['label'])
    print("ok")
    P, maps = convert_data_to_distribution(*data)
    print("ok2")
    result = get_measure(P)
    print("ok3")
    results[setting] = result
    print()

with open('synthetic/experiments/datasets.pickle', 'wb') as f:
    pickle.dump(results, f)
