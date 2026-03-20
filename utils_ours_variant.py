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
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=True)


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

    def __init__(self, indim_0, indim_1, heads, num_classes=1, hdim=1024):
        super().__init__()

        latdim = min(indim_0, indim_1)

        self.head_0 = MLP(indim_0, hdim, num_classes)
        self.head_1 = MLP(indim_1, hdim, num_classes)

        """# Freeze head_0 and head_1
        for p in self.head_0.parameters():
            p.requires_grad = False
        self.head_0.eval()

        for p in self.head_1.parameters():
            p.requires_grad = False
        self.head_1.eval()"""

        self.projector_0_R = nn.Sequential(
            nn.Linear(indim_0, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, latdim),
        )

        self.projector_1_R = nn.Sequential(
            nn.Linear(indim_1, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, latdim),
        )

        self.projector_R = nn.Sequential(
            nn.Linear(2 * indim_0, 2 * hdim),
            nn.ReLU(),
            nn.Linear(2 * hdim, latdim),
        )

        self.head_0_R = MLP(indim_0, hdim, num_classes)
        self.head_1_R = MLP(indim_1, hdim, num_classes)
        self.head_R = MLP(2 * indim_0, hdim, num_classes)

    def forward(self, modality0, modality1):
        y_pred_0 = self.head_0(modality0)
        y_pred_1 = self.head_1(modality1)

        z0_R = self.projector_0_R(modality0)
        z1_R = self.projector_1_R(modality1)

        y_pred_0_R = self.head_0_R(modality0)
        y_pred_1_R = self.head_1_R(modality1)

        y_pred_R = self.head_R(torch.cat((modality0, modality1), dim=1))

        return y_pred_0, y_pred_1, y_pred_0_R, y_pred_1_R

# =========================
# Lightning Module
# =========================
class RedundancyRepresentationLightningModel(pl.LightningModule):
    def __init__(
        self,
        model,
        distribution_target: str = "gaussian",
        lr: float = 1e-4,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = model

        self.distribution_target = distribution_target
        self.lr = lr

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
        y_pred_0, y_pred_1, y_pred_0_R, y_pred_1_R = self.forward(modality0, modality1)

        if self.distribution_target == "gaussian":
            pred_loss = F.mse_loss(y_pred.squeeze(), y)
            pred_kl_loss = 0
            mse_loss = 0
            pred_align_loss = F.mse_loss(y_pred_0.squeeze(), y_pred_1.squeeze())
            sup_clip_loss = 100 * supclip_continuous(z0, z1, y, ids)
            align_loss = 1000 * (z0 - z1).norm(p=2, dim=1).pow(2).mean()
        elif self.distribution_target == "categorical":

            pred_loss0 = F.cross_entropy(y_pred_0, y, reduction="mean")
            pred_loss1 = F.cross_entropy(y_pred_1, y, reduction="mean")

            loss = pred_loss0 + pred_loss1

            ce0 = F.cross_entropy(y_pred_0, y, reduction="none")
            ce1 = F.cross_entropy(y_pred_1, y, reduction="none")

            # mask: True where model 0 is worse (higher CE)
            mask = ce0 > ce1  # shape: (batch,)

            worst_logits = torch.where(mask[:, None], y_pred_0, y_pred_1).detach()
            worst_probs = torch.softmax(worst_logits, dim=1)

            log_probs_0_R = F.log_softmax(y_pred_0_R, dim=1)
            log_probs_1_R = F.log_softmax(y_pred_1_R, dim=1)

            loss += -(worst_probs * log_probs_0_R).sum(dim=1).mean()
            loss += -(worst_probs * log_probs_1_R).sum(dim=1).mean()

            """pred_loss0 = F.cross_entropy(y_pred_0, y, reduction="mean")
            pred_loss1 = F.cross_entropy(y_pred_1, y, reduction="mean")

            ce0 = F.cross_entropy(y_pred_0, y, reduction="none")
            ce1 = F.cross_entropy(y_pred_1, y, reduction="none")

            log_py = torch.log(torch.tensor(0.25))

            i0 = -ce0 - log_py
            i1 = -ce1 - log_py

            same_sign = torch.sign(i0) == torch.sign(i1)

            ce_stack = torch.stack([ce0, ce1], dim=1)
            argmax_ce = torch.argmax(ce_stack, dim=1)

            pred_stack = torch.stack([y_pred_0, y_pred_1], dim=1)
            ccs_logits = pred_stack[torch.arange(pred_stack.size(0)), argmax_ce]

            mask = same_sign.float()

            pred_loss = pred_loss0 + pred_loss1

            loss0 = F.mse_loss(y_pred_0_R, ccs_logits.detach(), reduction="none").mean(dim=1)
            loss1 = F.mse_loss(y_pred_1_R, ccs_logits.detach(), reduction="none").mean(dim=1)

            loss = pred_loss + (loss0 * mask).mean() + (loss1 * mask).mean()"""

        else:
            raise NotImplementedError

        # self.log(f"{stage}/pred_align_loss", pred_align_loss, prog_bar=True)
        self.log(f"{stage}/loss", loss, prog_bar=True)
        return loss, y_pred_0, y_pred_1, y_pred_0_R, y_pred_1_R, y

    # -------------------------
    # Train / Val / Test
    # -------------------------
    def training_step(self, batch, batch_idx):
        loss, _, _, _, _, _ = self._shared_step(batch, "Train")
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "Val")

    def test_step(self, batch, batch_idx):
        _, y_pred_0, y_pred_1, y_pred_0_R, y_pred_1_R, y = self._shared_step(batch, "Test")

        self.test_preds.append({
            "modality0": y_pred_0_R.detach().cpu(),
            "modality1": y_pred_1_R.detach().cpu()
        })
        self.test_targets.append(y.detach().cpu())

    def on_test_epoch_end(self):
        """
        Return predictions as a dictionary.
        """
        y_pred_0_R = torch.cat([p["modality0"] for p in self.test_preds])
        y_pred_1_R = torch.cat([p["modality1"] for p in self.test_preds])

        y = torch.cat(self.test_targets)

        ce0 = F.cross_entropy(y_pred_0_R, y.long(), reduction="mean")
        ce1 = F.cross_entropy(y_pred_1_R, y.long(), reduction="mean")

        if ce0 > ce1:
           worst_logits = y_pred_0_R
        else:
            worst_logits = y_pred_1_R

        self.y_pred_dict = {
            "modality0": y_pred_0_R,
            "modality1": y_pred_1_R,
            "average": worst_logits,
            "targets": y,
        }

# =========================
# Trainer helper
# =========================
def create_redundancy_trainer(
    max_epochs=50,
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
    heads, config_name,
    ids_train=None, ids_val=None, ids_test=None,
    distribution_target="gaussian", num_classes=1, h_dim=1024
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
    model = RedundancyRepresentationModel(indim_1, indim_2, heads, num_classes=num_classes, hdim=h_dim)

    pl_model = RedundancyRepresentationLightningModel(
        model,
        distribution_target=distribution_target
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

    best_path = trainer.checkpoint_callback.best_model_path
    ckpt = torch.load(best_path, weights_only=False, map_location="cpu")
    epoch = ckpt["epoch"]
    print("Best epoch:", epoch)

    # After trainer.test(...)
    del trainer, pl_model, model, best_model, datamodule
    gc.collect()
    torch.cuda.empty_cache()

    return y_pred_dict

def compute_PID_categorical(joint_ce, modality0_ce, modality1_ce, redundancy_ce, num_classes):
    print("joint_ce", joint_ce)
    print("redundancy_ce", redundancy_ce)
    print("modality0_ce", modality0_ce)
    print("modality1_ce", modality1_ce)

    redundancy_ce = min(max(redundancy_ce, joint_ce), log(num_classes))

    modality0_ce = max(modality0_ce, joint_ce)
    modality1_ce = max(modality1_ce, joint_ce)

    modality0_ce = min(min(modality0_ce, redundancy_ce), log(num_classes))
    modality1_ce = min(min(modality1_ce, redundancy_ce), log(num_classes))

    # joint_ce = min(modality0_ce, modality1_ce, joint_ce)

    I = log(num_classes) - joint_ce

    I_R = log(num_classes) - redundancy_ce
    I_U0 = (log(num_classes) - modality0_ce) - I_R
    I_U1 = (log(num_classes) - modality1_ce) - I_R

    I_S = max(0, I - I_U0 - I_U1 - I_R)

    print("R=", I_R)
    print("U0=", I_U0)
    print("U1=", I_U1)
    print("S=", I_S)
    print("I=", I)