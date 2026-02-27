"""Implements supervised learning training procedures."""
import torch
from torch import nn
import time
import numpy as np
from eval_scripts.performance import AUPRC, f1_score, accuracy, eval_affect
from eval_scripts.complexity import all_in_one_train, all_in_one_test
from eval_scripts.robustness import relative_robustness, effective_robustness, single_plot
from test import *
from tqdm import tqdm
import pickle
import copy
from itertools import chain
from math import log
from sklearn.metrics import r2_score

# import pdb

softmax = nn.Softmax(dim=-1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def traditional_cross_entropy_from_probs(probs, targets, eps=1e-12):
    """
    probs:   Tensor (N, C) — probabilities (must sum to 1)
    targets: Tensor (N,)   — class indices
    """
    probs = torch.clamp(probs, min=eps, max=1.0)
    log_probs = torch.log(probs)
    loss = -log_probs[torch.arange(targets.shape[0]), targets]
    return loss.mean()

class RedundancyAwareMMDL(nn.Module):
    """Implements Redundancy Aware MMDL classifier."""

    def __init__(self, encoders, heads, fusion, head, has_padding=False):
        """Instantiate MMDL Module

        Args:
            encoders (List): List of nn.Module encoders, one per modality.
            fusion (nn.Module): Fusion module
            head (nn.Module): Classifier module
            has_padding (bool, optional): Whether input has padding or not. Defaults to False.
        """
        super(RedundancyAwareMMDL, self).__init__()

        assert len(encoders) == 2, ("The current implementation of UnicityAwareMMDL only takes 2 modalities as inputs")

        self.encoders = nn.ModuleList(encoders)
        self.heads = nn.ModuleList(heads)

        self.fuse = fusion
        self.head = head
        self.has_padding = has_padding
        self.fuseout = None
        self.reps = []

    def forward(self, inputs):
        """Apply MMDL to Layer Input.

        Args:
            inputs (torch.Tensor): Layer Input

        Returns:
            torch.Tensor: Layer Output
        """
        outs = []
        if self.has_padding:
            for i in range(len(inputs[0])):
                outs.append(self.encoders[i](
                    [inputs[0][i], inputs[1][i]]))
        else:
            for i in range(len(inputs)):
                outs.append(self.encoders[i](inputs[i]))
        self.reps = outs
        if self.has_padding:
            if isinstance(outs[0], torch.Tensor):
                out = self.fuse(outs)
            else:
                out = self.fuse([i[0] for i in outs])
        else:
            out = self.fuse(outs)
        self.fuseout = out
        if type(out) is tuple:
            out = out[0]
        if self.has_padding and not isinstance(outs[0], torch.Tensor):
            out_pred_joint = self.head([out, inputs[1][0]])
        else:
            out_pred_joint = self.head(out)

        outs_modalities = []
        for i, outs_i in enumerate(outs):
            outs_modalities.append(self.heads[i](outs_i))

        return out_pred_joint, outs, outs_modalities


def deal_with_objective(objective, pred, truth, args):
    """Alter inputs depending on objective function, to deal with different objective arguments."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if type(objective) == nn.CrossEntropyLoss:
        if len(truth.size()) == len(pred.size()):
            truth1 = truth.squeeze(len(pred.size()) - 1)
        else:
            truth1 = truth
        return objective(pred, truth1.long().to(device))
    elif type(objective) == nn.MSELoss or type(objective) == nn.modules.loss.BCEWithLogitsLoss or type(
            objective) == nn.L1Loss:
        return objective(pred[:, 0], truth.float().to(device))
    else:
        return objective(pred, truth.to(device), args)


def train(
        encoders, fusion, head, heads, train_dataloader, valid_dataloader,
        total_epochs, additional_optimizing_modules=[],
        is_packed=False,
        early_stop=False, task="classification", optimtype=torch.optim.RMSprop, lr=0.001, weight_decay=0.0,
        objective=nn.CrossEntropyLoss(), auprc=False, save='best.pt', validtime=False, objective_args_dict=None,
        input_to_float=True, clip_val=8,
        track_complexity=False, pretrain=None):
    """
    Handle running a simple supervised training loop.

    :param fusion: fusion module, takes in outputs of encoders in a list and outputs fused representation
    :param total_epochs: maximum number of epochs to train
    :param additional_optimizing_modules: list of modules, include all modules that you want to be optimized by the optimizer other than those in encoders, fusion, head (for example, decoders in MVAE)
    :param is_packed: whether the input modalities are packed in one list or not (default is False, which means we expect input of [tensor(20xmodal1_size),(20xmodal2_size),(20xlabel_size)] for batch size 20 and 2 input modalities)
    :param early_stop: whether to stop early if valid performance does not improve over 7 epochs
    :param task: type of task, currently support "classification","regression","multilabel"
    :param optimtype: type of optimizer to use
    :param lr: learning rate
    :param weight_decay: weight decay of optimizer
    :param objective: objective function, which is either one of CrossEntropyLoss, MSELoss or BCEWithLogitsLoss or a custom objective function that takes in three arguments: prediction, ground truth, and an argument dictionary.
    :param auprc: whether to compute auprc score or not
    :param save: the name of the saved file for the model with current best validation performance
    :param validtime: whether to show valid time in seconds or not
    :param objective_args_dict: the argument dictionary to be passed into objective function. If not None, at every batch the dict's "reps", "fused", "inputs", "training" fields will be updated to the batch's encoder outputs, fusion module output, input tensors, and boolean of whether this is training or validation, respectively.
    :param input_to_float: whether to convert input to float type or not
    :param clip_val: grad clipping limit
    :param track_complexity: whether to track training complexity or not
    """
    if not pretrain:
        model = (RedundancyAwareMMDL(encoders, heads, fusion, head,
                                  has_padding=is_packed)
                 .to(device))
    else:
        model = torch.load(pretrain).to(device)

    best_model = copy.deepcopy(model)

    def _trainprocess(best_model):
        additional_params = []
        for m in additional_optimizing_modules:
            additional_params.extend(
                [p for p in m.parameters() if p.requires_grad])
        op = optimtype([p for p in model.parameters() if p.requires_grad] +
                       additional_params, lr=lr, weight_decay=weight_decay)
        bestvalloss = 10000
        bestloss = 10000
        bestacc = 0
        bestf1 = 0
        patience = 0

        def _processinput(inp):
            if input_to_float:
                return inp.float()
            else:
                return inp

        for epoch in range(total_epochs):
            totalloss = 0.0
            totals = 0
            model.train()
            for j in train_dataloader:
                op.zero_grad()
                out_pred_joint, outs, outs_modalities = model([_processinput(i).to(device)
                                                                                      for i in j[:-1]])
                if not (objective_args_dict is None):
                    objective_args_dict['reps'] = model.reps
                    objective_args_dict['fused'] = model.fuseout
                    objective_args_dict['inputs'] = j[:-1]
                    objective_args_dict['training'] = True
                    objective_args_dict['model'] = model

                loss = deal_with_objective(
                    objective, out_pred_joint, j[-1], objective_args_dict)

                for out in outs_modalities:
                    loss += deal_with_objective(
                        objective, out, j[-1], objective_args_dict)

                totalloss += loss * len(j[-1])
                totals += len(j[-1])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                op.step()
            print("Epoch " + str(epoch) + " train loss: " + str(totalloss / totals))
            validstarttime = time.time()
            if validtime:
                print("train total: " + str(totals))
            model.eval()
            with torch.no_grad():
                totalloss = 0.0
                pred = []
                true = []
                pts = []
                for j in valid_dataloader:
                    model.eval()

                    # UNPACK MODEL OUTPUT
                    out_pred_joint, outs, outs_modalities = model(
                        [_processinput(i).to(device) for i in j[:-1]]
                    )

                    if not (objective_args_dict is None):
                        objective_args_dict['reps'] = model.reps
                        objective_args_dict['fused'] = model.fuseout
                        objective_args_dict['inputs'] = j[:-1]
                        objective_args_dict['training'] = False

                    # LOSS ON JOINT PREDICTION AND SINGLE MODALITIES
                    loss = deal_with_objective(
                        objective, out_pred_joint, j[-1], objective_args_dict
                    )

                    for out in outs_modalities:
                        loss += deal_with_objective(
                            objective, out, j[-1], objective_args_dict)

                    totalloss += loss * len(j[-1])

                    # PREDICTIONS ONLY FROM JOINT PREDICTION
                    if task == "classification":
                        pred.append(torch.argmax(out_pred_joint, dim=1))
                    elif task == "multilabel":
                        pred.append(torch.sigmoid(out_pred_joint).round())

                    true.append(j[-1])

                    if auprc:
                        # pdb.set_trace()
                        sm = softmax(out)
                        pts += [(sm[i][1].item(), j[-1][i].item())
                                for i in range(j[-1].size(0))]
            if pred:
                pred = torch.cat(pred, 0)
            true = torch.cat(true, 0)
            totals = true.shape[0]
            valloss = totalloss / totals
            if task == "classification":
                acc = accuracy(true, pred)
                print("Epoch " + str(epoch) + " valid loss: " + str(valloss) +
                      " acc: " + str(acc))
                """if acc > bestacc:
                    patience = 0
                    bestacc = acc
                    print("Saving Best")
                    if save:
                        best_model = copy.deepcopy(model)
                        torch.save(model, save)
                else:
                    patience += 1"""
                if loss < bestloss:
                    patience = 0
                    bestloss = loss
                    print("Saving Best")
                    if save:
                        best_model = copy.deepcopy(model)
                        torch.save(model, save)
                else:
                    patience += 1
            elif task == "multilabel":
                f1_micro = f1_score(true, pred, average="micro")
                f1_macro = f1_score(true, pred, average="macro")
                print("Epoch " + str(epoch) + " valid loss: " + str(valloss) +
                      " f1_micro: " + str(f1_micro) + " f1_macro: " + str(f1_macro))
                if f1_macro > bestf1:
                    patience = 0
                    bestf1 = f1_macro
                    print("Saving Best")
                    if save:
                        torch.save(model, save)
                else:
                    patience += 1
            elif task == "regression":
                print("Epoch " + str(epoch) + " valid loss: " + str(valloss.item()))
                if valloss < bestvalloss:
                    patience = 0
                    bestvalloss = valloss
                    print("Saving Best")
                    if save:
                        torch.save(model, save)
                else:
                    patience += 1
            if early_stop and patience > 7:
                break
            if auprc:
                print("AUPRC: " + str(AUPRC(pts)))
            validendtime = time.time()
            if validtime:
                print("valid time:  " + str(validendtime - validstarttime))
                print("Valid total: " + str(totals))
        if not save:
            print("Testing: ")
            if task == "classification":
                print("acc:", bestacc)
            elif task == "multilabel":
                print("f1 macro:", bestf1)
            elif task == "regression":
                print("loss:", bestvalloss)
        return best_model

    if track_complexity:
        t, mem, num_params = all_in_one_train(_trainprocess, [model] + additional_optimizing_modules)
        return t, mem, num_params
    else:
        best_model = _trainprocess(best_model)

    return best_model

def single_test(
    model,
    test_dataloader,
    criterion=nn.CrossEntropyLoss(),
    task="classification",
    auprc=False,
    input_to_float=True,
    cluster=None,
    sub_acc=None,
    save_preds=None
):
    """Run single test for model and compute accuracies and cross-entropies."""

    def _processinput(inp):
        return inp.float() if input_to_float else inp

    with torch.no_grad():
        totalloss_joint = 0.0
        totalloss_modalities = [0.0 for _ in range(2)]

        pred_joint = []
        true = []

        pred_modalities = [[] for _ in range(2)]

        totals = 0

        for _, j in enumerate(test_dataloader):
            model.eval()

            inputs = [_processinput(i).to(device) for i in j[:-1]]
            targets = j[-1].to(device)
            batch_size = targets.size(0)
            totals += batch_size

            out_pred_joint, outs, outs_modalities = model(inputs)

            # ---------- TARGET FORMAT ----------
            if isinstance(criterion, nn.CrossEntropyLoss):
                if len(targets.size()) == len(out_pred_joint.size()):
                    targets_ce = targets.squeeze(len(out_pred_joint.size()) - 1).long()
                else:
                    targets_ce = targets.long()
            else:
                targets_ce = targets

            # ---------- JOINT LOSS ----------
            if isinstance(criterion, nn.CrossEntropyLoss):
                loss_joint = traditional_cross_entropy_from_probs(softmax(out_pred_joint), targets_ce) # loss_joint = criterion(out_pred_joint, targets_ce)
            else:
                loss_joint = criterion(out_pred_joint[:, 0], targets_ce)
            totalloss_joint += loss_joint.item() * batch_size

            # ---------- MODALITY LOSSES ----------
            for idx, out_mod in enumerate(outs_modalities):
                if isinstance(criterion, nn.CrossEntropyLoss):
                    loss_mod = traditional_cross_entropy_from_probs(softmax(out_mod), targets_ce)  # criterion(out_mod, targets_ce)
                else:
                    loss_mod = criterion(out_mod[:, 0], targets_ce)
                totalloss_modalities[idx] += loss_mod.item() * batch_size

            # ---------- PREDICTIONS ----------
            if task == "classification":
                pred_joint.append(out_pred_joint)
                for idx, out_mod in enumerate(outs_modalities):
                    pred_modalities[idx].append(out_mod)
            elif task == "regression":
                pred_joint.append(out_pred_joint[:, 0])
                for idx, out_mod in enumerate(outs_modalities):
                    pred_modalities[idx].append(out_mod)

            true.append(targets)

        # ---------- CONCATENATE ----------
        pred_joint = torch.cat(pred_joint, 0)
        true = torch.cat(true, 0)
        pred_modalities = [torch.cat(p, 0) for p in pred_modalities]

        # ---------- AVERAGE LOSSES ----------
        ce_joint = totalloss_joint / totals
        ce_modalities = [l / totals for l in totalloss_modalities]

        # ---------- METRICS ----------
        if task == "classification":
            acc_joint = accuracy(true, torch.argmax(pred_joint, dim=1))
            acc_modalities = [accuracy(true, torch.argmax(p, dim=1)) for p in pred_modalities]
        elif task == "regression":
            acc_joint = r2_score(true, pred_joint)
            acc_modalities = [r2_score(true, p) for p in pred_modalities]

        """print(f"Joint accuracy/R2: {acc_joint:.4f}, Loss: {ce_joint:.4f}")
        for idx, (acc, ce) in enumerate(zip(acc_modalities, ce_modalities)):
            print(f"Modality head {idx} accuracy/R2: {acc:.4f}, CE: {ce:.4f}")"""

        if save_preds:
            preds_np = pred_joint.cpu().numpy().reshape(-1,)
            with open(save_preds, "wb") as f:
                pickle.dump(preds_np, f)

        # Return all accuracies as dict
        return {
            "joint_acc": acc_joint,
            "modalities_acc": acc_modalities,
            "joint_ce": ce_joint,
            "modalities_ce": ce_modalities,
            "pred_joint": pred_joint,
            "true_labels": true,
            "pred_modalities": pred_modalities

        }

def test(
    model,
    test_dataloaders_all,
    dataset="default",
    method_name="My method",
    criterion=nn.CrossEntropyLoss(),
    task="classification",
    auprc=False,
    input_to_float=True,
    no_robust=False,
    cluster=None,
    sub_acc=None,
    save_preds=None
):
    """
    Handle getting test results for a supervised model.
    Compatible with UnicityAwareMMDL and per-head accuracies.
    """

    # ---------- NO ROBUSTNESS ----------
    if no_robust:
        # If test_dataloaders_all is a dict, pick first DataLoader automatically
        if isinstance(test_dataloaders_all, dict):
            first_key = list(test_dataloaders_all.keys())[0]
            first_loader = test_dataloaders_all[first_key][0]
        else:
            first_loader = test_dataloaders_all  # assume it's already a DataLoader
        return single_test(
            model,
            first_loader,
            criterion,
            task,
            auprc,
            input_to_float,
            cluster=cluster,
            sub_acc=sub_acc,
            save_preds=save_preds,
        )

    # ---------- CLEAN TEST ----------
    def _testprocess():
        single_test(
            model,
            test_dataloaders_all[list(test_dataloaders_all.keys())[0]][0],
            criterion,
            task,
            auprc,
            input_to_float,
        )

    dict_of_metrics = all_in_one_test(_testprocess, [model])

    # ---------- ROBUSTNESS TEST ----------
    for noisy_modality, test_dataloaders in test_dataloaders_all.items():
        print(f"Testing on noisy data ({noisy_modality})...")

        # For robustness, store per-head curves
        robustness_curve = {
            "joint": [],
            "modalities": [[] for _ in range(2)],
            "redundancies": [[] for _ in range(2)]
        }

        for test_dataloader in tqdm(test_dataloaders):
            results = single_test(
                model,
                test_dataloader,
                criterion,
                task,
                auprc,
                input_to_float,
                cluster=cluster,
                sub_acc=sub_acc,
                save_preds=save_preds,
            )

            # Append to robustness lists
            robustness_curve["joint"].append(results["joint"])
            for idx, acc in enumerate(results["modalities"]):
                robustness_curve["modalities"][idx].append(acc)
            for idx, acc in enumerate(results["redundancies"]):
                robustness_curve["redundancies"][idx].append(acc)

        robustness_key = f"{dataset} {noisy_modality}"

        # ---------- PRINT & PLOT ----------
        print(f"Joint relative robustness ({noisy_modality}): {relative_robustness(robustness_curve['joint'], robustness_key)}")
        print(f"Joint effective robustness ({noisy_modality}): {effective_robustness(robustness_curve['joint'], robustness_key)}")
        single_plot(
            robustness_curve["joint"],
            robustness_key,
            xlabel="Noise level",
            ylabel="accuracy",
            fig_name=f"{method_name}-{robustness_key}-{noisy_modality}-joint",
            method=method_name
        )

        # Modalities
        for idx, acc_list in enumerate(robustness_curve["modalities"]):
            print(f"Modality head {idx} relative robustness: {relative_robustness(acc_list, f'{robustness_key}_mod{idx}')}")
            print(f"Modality head {idx} effective robustness: {effective_robustness(acc_list, f'{robustness_key}_mod{idx}')}")
            single_plot(
                acc_list,
                f"{robustness_key}_mod{idx}",
                xlabel="Noise level",
                ylabel="accuracy",
                fig_name=f"{method_name}-{robustness_key}-{noisy_modality}-mod{idx}",
                method=method_name
            )

        # Redundancies
        for idx, acc_list in enumerate(robustness_curve["redundancies"]):
            print(f"Unicity head {idx} relative robustness: {relative_robustness(acc_list, f'{robustness_key}_uni{idx}')}")
            print(f"Unicity head {idx} effective robustness: {effective_robustness(acc_list, f'{robustness_key}_uni{idx}')}")
            single_plot(
                acc_list,
                f"{robustness_key}_uni{idx}",
                xlabel="Noise level",
                ylabel="accuracy",
                fig_name=f"{method_name}-{robustness_key}-{noisy_modality}-uni{idx}",
                method=method_name
            )

        print("Plots saved for joint, modalities, and unicity heads.")

    return dict_of_metrics
