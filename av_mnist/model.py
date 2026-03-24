import torch
import torch.nn as nn
import torch.nn.functional as F

from unimodals.common_models import MLP3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display(X):
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    img = [x.cpu().detach().numpy() for x in X]
    f = plt.figure()
    for i in range(len(X)):
        f.add_subplot(1, len(X), i + 1)
        plt.imshow(img[i], cmap='gray')
    plt.show()


def bernoulli(prob, shape):
    a = torch.bernoulli(prob * torch.ones(shape[0])).to(device)
    return a[:, None, None, None].expand(shape)


def corrupt(xA, xB, feed_A=0.8, feed_AB=0.2):
    A, AB = bernoulli(feed_A, xA.shape), bernoulli(feed_AB, xA.shape)
    n = torch.normal(0, 0.15, size=xA.shape).to(device)
    A_only = A * (1 - AB)
    B_only = 1 - A - AB
    xA = xA * (AB + A_only) + (0.1 * xA + n) * (1 - AB - A_only)
    xB = xB * (AB + B_only) + (0.1 * xB + n) * (1 - AB - B_only)
    # display([xA[0,0,:,:], xB[0,0,:,:]])
    return xA, xB

class CNN(nn.Module):

    def __init__(self, num_classes=10, emb_dim=128):
        super(CNN, self).__init__()

        # -------- Image branch (MNIST) --------
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.image_fc = nn.Linear(128 * 4 * 4, emb_dim)

        # -------- Audio branch (spectrogram) --------
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.audio_fc = nn.Linear(128 * 4 * 4, emb_dim)

        # -------- Fusion classifier --------
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.audio_classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.visual_classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, xA, xB, unimodal=None):

        # Image embedding
        img = self.image_encoder(xA)
        img = torch.flatten(img, 1)
        img = self.image_fc(img)

        # Audio embedding
        aud = self.audio_encoder(xB)
        aud = torch.flatten(aud, 1)
        aud = self.audio_fc(aud)

        if unimodal=="train":
            x = torch.cat((img, aud), dim=1)
            x = self.classifier(x)

            img = self.visual_classifier(img.detach())
            aud = self.audio_classifier(aud.detach())

            return F.log_softmax(x, dim=1), F.log_softmax(img, dim=1), F.log_softmax(aud, dim=1)

        if unimodal is None:
            # Fusion
            x = torch.cat((img, aud), dim=1)
            x = self.classifier(x)
        elif unimodal == "visual":
            x = self.visual_classifier(img)
        else:
            x = self.audio_classifier(aud)

        return F.log_softmax(x, dim=1)

    def get_representations(self, xA, xB):
        # Image embedding
        img = self.image_encoder(xA)
        img = torch.flatten(img, 1)
        img = self.image_fc(img)

        # Audio embedding
        aud = self.audio_encoder(xB)
        aud = torch.flatten(aud, 1)
        aud = self.audio_fc(aud)

        return img, aud

class CNN_sum(nn.Module):

    def __init__(self, num_classes=2, emb_dim=128):
        super(CNN_sum, self).__init__()

        # -------- Image branch (MNIST) --------
        self.image_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )

        self.image_fc = nn.Linear(128 * 4 * 4, emb_dim)

        # -------- Audio branch (spectrogram) --------
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((4,4))
        )

        self.dropout = nn.Dropout(p=0.1)

        self.audio_fc = nn.Linear(128 * 4 * 4, emb_dim)

        # -------- Fusion classifier --------
        self.classifier = MLP3(emb_dim * 2, 512, 2)

        self.audio_classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.visual_classifier = nn.Sequential(
            nn.Linear(emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.audio_digit_classifier = nn.Sequential(
            nn.Linear(emb_dim, 10)
        )

        self.visual_digit_classifier = nn.Sequential(
            nn.Linear(emb_dim, 10)
        )

    def forward(self, xA, xB, unimodal=None):

        # Image embedding
        img = self.image_encoder(xA)
        img = torch.flatten(img, 1)
        img = self.image_fc(self.dropout(img))

        # Audio embedding
        aud = self.audio_encoder(xB)
        aud = torch.flatten(aud, 1)
        aud = self.audio_fc(self.dropout(aud))

        if unimodal=="train":
            x = torch.cat((img, aud), dim=1)
            x = self.classifier(x)

            img_digit = self.visual_digit_classifier(img)
            aud_digit = self.audio_digit_classifier(aud)

            img_label = self.visual_classifier(img.detach())
            aud_label = self.audio_classifier(aud.detach())

            return F.log_softmax(x, dim=1), F.log_softmax(img_label, dim=1), F.log_softmax(aud_label, dim=1), F.log_softmax(img_digit, dim=1), F.log_softmax(aud_digit, dim=1)

        if unimodal is None:
            # Fusion
            x = torch.cat((img, aud), dim=1)
            x = self.classifier(x)
        elif unimodal == "visual":
            x = self.visual_classifier(img)
        else:
            x = self.audio_classifier(aud)

        return F.log_softmax(x, dim=1)

    def get_representations(self, xA, xB):
        # Image embedding
        img = self.image_encoder(xA)
        img = torch.flatten(img, 1)
        img = self.image_fc(img)

        # Audio embedding
        aud = self.audio_encoder(xB)
        aud = torch.flatten(aud, 1)
        aud = self.audio_fc(aud)

        return img, aud