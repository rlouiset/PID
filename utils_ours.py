# =========================
# Imports
# =========================
import gc
import os
from typing import Literal, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from math import *

# =========================
# Dataset & DataModule
# =========================
class MultimodalRepresentationsDataset(Dataset):
    """
    Dataset wrapping precomputed multimodal representations.
    """

    def __init__(self, X_dict: Dict[str, torch.Tensor], y: torch.Tensor, ids: torch.Tensor):
        self.X_dict = X_dict
        self.y = y
        self.ids = ids

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            self.X_dict["modality0"][idx],
            self.X_dict["modality1"][idx],
            self.y[idx],
            self.ids[idx]
        )


class MultimodalRepresentationsDataModule(pl.LightningDataModule):
    """
    Lightning DataModule for multimodal representations.
    """

    def __init__(self, train_ds, val_ds, test_ds, batch_size=1024):
        super().__init__()
        self.batch_size = batch_size
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False)


# =========================
# Models
# =========================
class MLP(nn.Module):
    """Two-layer perceptron used as prediction head."""

    def __init__(self, indim, hiddim, outdim, dropout=False, dropoutp=0.1):
        super().__init__()
        self.fc1 = nn.Linear(indim, hiddim)
        self.fc2 = nn.Linear(hiddim, outdim)
        self.dropout = nn.Dropout(dropoutp) if dropout else nn.Identity()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class RedundancyRepresentationModel(nn.Module):
    """
    Learns redundancy representations across two modalities and
    predicts targets from each redundancy embedding.
    """

    def __init__(self, indim_0, indim_1, num_classes=1, hdim=1024):
        super().__init__()

        latdim = min(indim_0, indim_1)

        self.projector_0 = nn.Sequential(
            nn.Linear(indim_0, 2*hdim),
            nn.ReLU(),
            nn.Linear(2*hdim, latdim),
        )

        self.projector_1 = nn.Sequential(
            nn.Linear(indim_1, 2*hdim),
            nn.ReLU(),
            nn.Linear(2*hdim, latdim),
        )

        self.head = MLP(latdim, hdim, num_classes)

    def forward(self, modality0, modality1):
        z0 = self.projector_0(modality0)
        z1 = self.projector_1(modality1)

        z0 = z0 / z0.norm(p=2, dim=1, keepdim=True)
        z1 = z1 / z1.norm(p=2, dim=1, keepdim=True)

        y_pred_0 = self.head(z0.detach())
        y_pred_1 = self.head(z1.detach())

        y_pred = self.head(((z1.detach() + z0.detach()) / 2))

        return z0, y_pred_0, z1, y_pred_1, y_pred


# =========================
# Losses
# =========================
def supclip_continuous(c, s, y, ids, sigma_y=0.01, tau=0.5):
    """Supervised CLIP-style loss for continuous targets."""
    is_same_id = (ids[:, None] == ids[None, :]).float()

    label_sim = torch.cdist(y[:, None], y[:, None]).pow(2)
    label_sim = torch.exp(-label_sim / (2 * sigma_y))

    dist = torch.cdist((c+s)/2, (c+s)/2).pow(2)

    num = (dist * label_sim).mean()

    den = (torch.exp(-dist / (2 * tau))).mean(1).log().mean()
    return num + den

def supclip_categorical(c, s, y, ids, tau=0.1):
    """Supervised CLIP-style loss for categorical targets."""
    is_same_id = (ids[:, None] == ids[None, :]).float()

    is_same_label = (y[:, None] == y[None, :]).float()

    distance_matrix = torch.cdist(c, s, p=2.0).pow(2)
    num = (distance_matrix * is_same_label).sum() / is_same_label.sum()

    den = (distance_matrix.mul(-1/(2*tau)).exp()).mean(1).log().mean()
    return num + den

def continuous_supervised_kernel_alignment(ZA, ZB, y, tau=0.1, sigma_y=0.1):
    """
    ZA, ZB: (N, D) unit-norm representations
    y: (N,) continuous labels
    tau: kernel bandwidth for ZA/ZB
    sigma_y: kernel bandwidth in label space
    """
    # similarity in representation space
    sim = torch.cdist(ZA, ZB).pow(2)        # (N,N)
    K = torch.exp(-sim / (2 * tau))         # Gaussian kernel

    # similarity in label space
    label_dist = torch.cdist(y[:, None], y[:, None]).pow(2)
    S = torch.exp(-label_dist / (2 * sigma_y))

    # positive / negative weighting
    pos_weight = S
    neg_weight = 1.0 - S

    # compute weighted losses
    pos_loss = ((K - 1.0) ** 2 * pos_weight).sum() / pos_weight.sum()
    neg_loss = (K ** 2 * neg_weight).sum() / neg_weight.sum()

    return pos_loss + neg_loss

def supervised_kernel_alignment(ZA, ZB, y, tau=0.1):
    sim = torch.cdist(ZA, ZB).pow(2)
    K = torch.exp(-sim / (2 * tau))

    pos = (y[:, None] == y[None, :]).float()
    neg = 1.0 - pos

    pos_loss = ((K - 1.0) ** 2 * pos).sum() / pos.sum()
    neg_loss = ((K) ** 2 * neg).sum() / neg.sum()

    return pos_loss + neg_loss

# =========================
# Lightning Module
# =========================
class RedundancyRepresentationLightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        distribution_target: str = "gaussian",
        lambda_reg=10,
        lr: float = 1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = model

        self.distribution_target = distribution_target
        self.lr = lr

        self.lambda_reg = lambda_reg

        self.test_preds = []
        self.test_targets = []

    def forward(self, modality0, modality1):
        return self.model(modality0, modality1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # -------------------------
    # Shared step
    # -------------------------
    def _shared_step(self, batch, stage: str):
        modality0, modality1, y, ids = batch
        z0, y_pred_0, z1, y_pred_1, y_pred = self.forward(modality0, modality1)

        if self.distribution_target == "gaussian":
            pred_loss = F.mse_loss(y_pred.squeeze(), y)
            pred_align_loss = F.mse_loss(y_pred_0.squeeze(), y_pred_1.squeeze())
            sup_clip_loss = 100 * supclip_continuous(z0, z1, y, ids)
            align_loss = 1000 * (z0 - z1).norm(p=2, dim=1).pow(2).mean()
        elif self.distribution_target == "categorical":
            pred_loss = F.cross_entropy(y_pred_0, y) + F.cross_entropy(y_pred_1, y) + F.cross_entropy(y_pred, y)
            sup_clip_loss = supervised_kernel_alignment(z0, z1, y)
            align_loss = self.lambda_reg*(z0 - z1).norm(p=2, dim=1).pow(2).mean()
        else:
            raise NotImplementedError

        loss = pred_loss + sup_clip_loss + align_loss
        self.log(f"{stage}/pred_loss", pred_loss, prog_bar=True)
        self.log(f"{stage}/align_loss", align_loss, prog_bar=True)
        self.log(f"{stage}/sup_clip_loss", sup_clip_loss, prog_bar=True)
        # self.log(f"{stage}/pred_align_loss", pred_align_loss, prog_bar=True)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        return loss, y_pred_0, y_pred_1, y_pred, y

    # -------------------------
    # Train / Val / Test
    # -------------------------
    def training_step(self, batch, batch_idx):
        loss, _, _, _, _ = self._shared_step(batch, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "Val")

    def test_step(self, batch, batch_idx):
        _, y_pred_0, y_pred_1, _, y = self._shared_step(batch, "Test")

        ce0 = F.cross_entropy(y_pred_0, y, reduction="none")
        ce1 = F.cross_entropy(y_pred_1, y, reduction="none")

        # mask: True where model 0 is worse (higher CE)
        mask = ce0 > ce1  # shape: (batch,)

        worst_logits = torch.where(mask[:, None], y_pred_0, y_pred_1).detach()

        self.test_preds.append({
            "modality0": y_pred_0.detach().cpu(),
            "modality1": y_pred_1.detach().cpu(),
            "average": worst_logits.detach().cpu()
        })
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        """
        Return predictions as a dictionary.
        """
        self.y_pred_dict = {
            "modality0": torch.cat([p["modality0"] for p in self.test_preds]),
            "modality1": torch.cat([p["modality1"] for p in self.test_preds]),
            "average": torch.cat([p["average"] for p in self.test_preds]),
            "targets": torch.cat(self.test_targets),
        }

# =========================
# Trainer helper
# =========================
def create_redundancy_trainer(
    max_epochs=100,
    config_name="model",
    accelerator: Literal["cpu", "gpu", "auto"] = "auto",
    checkpoint_dir="checkpoints",
    monitor_metric="Val/loss",
):

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename=f"{config_name}-best",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
    )

    return pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        callbacks=[checkpoint_cb],
    )


# =========================
# Training / Evaluation API
# =========================
def return_redundancy_test_performances(
    X_train_dict, X_val_dict, X_test_dict,
    y_train, y_val, y_test,
    config_name,
    ids_train=None, ids_val=None, ids_test=None,
    distribution_target="gaussian", lambda_reg=10, num_classes=1, h_dim=1024, lr=1e-4
):

    if ids_train is None:
        ids_train = torch.arange(len(y_train), dtype=torch.long)
    if ids_val is None:
        ids_val = torch.arange(len(y_val), dtype=torch.long)
    if ids_test is None:
        ids_test = torch.arange(len(y_test), dtype=torch.long)

    train_ds = MultimodalRepresentationsDataset(X_train_dict, y_train, ids_train)
    val_ds = MultimodalRepresentationsDataset(X_val_dict, y_val, ids_val)
    test_ds = MultimodalRepresentationsDataset(X_test_dict, y_test, ids_test)

    datamodule = MultimodalRepresentationsDataModule(train_ds, val_ds, test_ds)

    indim_1, indim_2 = X_train_dict["modality0"].shape[1], X_train_dict["modality1"].shape[1]
    model = RedundancyRepresentationModel(indim_1, indim_2, num_classes=num_classes, hdim=h_dim)

    pl_model = RedundancyRepresentationLightningModel(
        model,
        distribution_target=distribution_target,
        lambda_reg=lambda_reg,
        lr=lr
    )

    checkpoint_dir = f"checkpoints/{config_name}/redundancy"
    os.makedirs(checkpoint_dir, exist_ok=True)

    trainer = create_redundancy_trainer(
        config_name=config_name,
        checkpoint_dir=checkpoint_dir,
    )

    trainer.fit(pl_model, datamodule=datamodule)

    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"Best redundancy model saved at: {best_model_path}")

    best_model = RedundancyRepresentationLightningModel.load_from_checkpoint(best_model_path, weights_only=False)

    trainer.test(best_model, datamodule=datamodule)

    y_pred_dict = best_model.y_pred_dict

    # After trainer.test(...)
    del trainer, pl_model, model, best_model, datamodule
    gc.collect()
    torch.cuda.empty_cache()

    return y_pred_dict