import random
import torch
from torch.utils.data import Dataset, DataLoader


class AV_dataset(Dataset):
    def __init__(self, mnist_dataset, audio_dataset):
        self.mnist_dataset = mnist_dataset
        self.audio_dataset = audio_dataset

        # Group MNIST samples by label
        self.visual_dict = {}
        for idx in range(len(self.mnist_dataset)):
            image, label = self.mnist_dataset[idx]
            if label not in self.visual_dict:
                self.visual_dict[label] = []
            self.visual_dict[label].append((image, label))

        # Group spoken digit samples by label
        self.audio_dict = {}
        for idx in range(len(self.audio_dataset)):
            audio, label = self.audio_dataset[idx]
            audio = torch.unsqueeze(audio, 0)
            if label not in self.audio_dict:
                self.audio_dict[label] = []
            self.audio_dict[label].append((audio, label))

        # Ensure that each label has at least one sample in both datasets
        common_labels = set(self.visual_dict.keys()) & set(self.audio_dict.keys())
        assert len(common_labels) > 0, "No common labels found between MNIST and spoken digit datasets"

        # Pair samples with the same label
        self.paired_samples = []
        for label in common_labels:
            mnist_samples = self.visual_dict[label]
            spoken_digit_samples = self.audio_dict[label]
            for mnist_sample in mnist_samples:
                # Randomly select a spoken digit sample with the same label
                spoken_digit_sample = random.choice(spoken_digit_samples)
                self.paired_samples.append((mnist_sample[0], spoken_digit_sample[0], label))


    def __len__(self):
        return len(self.paired_samples)


    def __getitem__(self, idx):
        return self.paired_samples[idx]

def build_label_index(dataset):
    label_to_indices = {i: [] for i in range(10)}

    for idx in range(len(dataset)):
        _, label = dataset[idx]
        label_to_indices[int(label)].append(idx)

    return label_to_indices

class AV_dataset_sum(Dataset):

    def __init__(self, mnist_dataset, audio_dataset, cutoff_sum, samples_per_combination=100):
        self.mnist_dataset = mnist_dataset
        self.audio_dataset = audio_dataset
        self.cutoff_sum = cutoff_sum

        # index by label
        self.mnist_index = build_label_index(mnist_dataset)
        self.audio_index = build_label_index(audio_dataset)

        # build balanced pairs
        self.pairs = []

        for d_img in range(10):
            for d_aud in range(10):

                img_indices = self.mnist_index[d_img]
                aud_indices = self.audio_index[d_aud]

                if len(img_indices) == 0 or len(aud_indices) == 0:
                    continue

                for _ in range(samples_per_combination):
                    i = random.choice(img_indices)
                    j = random.choice(aud_indices)

                    self.pairs.append((i, j, d_img, d_aud))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)  # 🔑 use pairs length

    def __getitem__(self, idx):

        i, j, img_label, audio_label = self.pairs[idx]

        # fetch data
        img, _ = self.mnist_dataset[i]
        audio, _ = self.audio_dataset[j]

        # ensure shape (1, F, T)
        audio = torch.unsqueeze(audio, 0)

        # compute target
        digit_sum = img_label + audio_label
        target = 1 if digit_sum > self.cutoff_sum else 0

        return (
            img,
            audio,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(img_label, dtype=torch.long),
            torch.tensor(audio_label, dtype=torch.long),
        )

class AV_dataset_sum_dependent(Dataset):

    def __init__(self, mnist_dataset, audio_dataset, cutoff_sum, samples_per_combination=100):
        self.mnist_dataset = mnist_dataset
        self.audio_dataset = audio_dataset
        self.cutoff_sum = cutoff_sum

        self.mnist_index = build_label_index(mnist_dataset)
        self.audio_index = build_label_index(audio_dataset)

        self.pairs = []

        for d_img in range(10):
            for d_aud in range(10):

                # Enforce dependence: both above or both below cutoff
                img_above = d_img > cutoff_sum
                aud_above = d_aud > cutoff_sum
                if img_above != aud_above:
                    continue  # skip unique combinations

                img_indices = self.mnist_index[d_img]
                aud_indices = self.audio_index[d_aud]

                if len(img_indices) == 0 or len(aud_indices) == 0:
                    continue

                for _ in range(samples_per_combination):
                    i = random.choice(img_indices)
                    j = random.choice(aud_indices)
                    self.pairs.append((i, j, d_img, d_aud))

        random.shuffle(self.pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j, img_label, audio_label = self.pairs[idx]

        img, _ = self.mnist_dataset[i]
        audio, _ = self.audio_dataset[j]
        audio = torch.unsqueeze(audio, 0)

        digit_sum = img_label + audio_label
        target = 1 if digit_sum > self.cutoff_sum else 0

        return (
            img,
            audio,
            torch.tensor(target, dtype=torch.long),
            torch.tensor(img_label, dtype=torch.long),
            torch.tensor(audio_label, dtype=torch.long),
        )