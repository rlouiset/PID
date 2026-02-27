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