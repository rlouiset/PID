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