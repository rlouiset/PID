from __future__ import print_function, division
import argparse
import pickle
import torch
import numpy as np

import torch
import torch.nn as nn
import normflows as nf

import numpy as np
import scipy.linalg as la


def pinv(a):
    """
    la.pinv sometimes raises an la.LinAlgError: SVD did not converge. This
    appears to be some kind of BLAS/LAPACK bug:
    https://github.com/numpy/numpy/issues/12941.

    The solution appears to be to use the gesvd driver for the SVD, which is
    slower, but does not appear to cause this problem. In our case, this
    means we need a manual implementation of pinv that uses this driver.

    This function is a simplified version of the scipy implementation:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/linalg/_basic.py#L1319-L1464
    """

    # NOTE: We could use a try-except block to try using the default la.pinv
    # first, and only use this function if that fails, but since we do not use
    # pinv in any time-critical loops, it might be better to simply have a
    # slower but more accurate result.
    u, s, vh = la.svd(a, lapack_driver='gesvd')
    t = u.dtype.char.lower()
    maxS = np.max(s)

    atol = 0.
    rtol = max(a.shape) * np.finfo(t).eps

    val = atol + maxS * rtol
    rank = np.sum(s > val)

    u = u[:, :rank]
    u /= s[:rank]
    B = (u @ vh[:rank]).conj().T

    return B


def solve(a, b):
    return la.solve(a, b, assume_a='pos')


def whiten(cov, dm, dx, dy, ret_channel_params=False):
    """
    Whiten the X- and Y-channel noise covariances, and return a new joint
    covariance matrix between M, X and Y.

    Also standardizes the covariance of M to an identity matrix.

    Assumes invertibility of matrices where required:
        sig_m, sig_x__m, sig_y__m
    """

    # Variable name convention
    # - No separator implies joint auto-covariance between variables
    # - One underscore refers to cross-covariance between variables
    # - Two underscores refers to conditioning
    #
    # Examples
    # - sig_mxy: joint (auto-)covariance between M, X and Y
    # - sig_xy_m: cross-covariance between the stacked vector (X, Y) and M
    # - sig_x_m__y: conditional cross-covariance matrix between X and M given Y

    # First standardize M
    sig_mxy = cov.copy()
    sig_m = cov[:dm, :dm]
    sig_mxy[:, :dm] = solve(la.sqrtm(sig_m).real, sig_mxy[:, :dm].T).T
    sig_mxy[:dm, :] = solve(la.sqrtm(sig_m).real, sig_mxy[:dm, :])
    sig_m = sig_mxy[:dm, :dm]  # Redefine sig_m

    # Extract necessary parameters
    sig_x = sig_mxy[dm:dm+dx, dm:dm+dx]
    sig_y = sig_mxy[dm+dx:, dm+dx:]
    sig_x_m = sig_mxy[dm:dm+dx, :dm]  # Also equal to hx pre-whitening
    sig_y_m = sig_mxy[dm+dx:, :dm]    # Also equal to hy pre-whitening

    # Compute channel noise covariance matrices (TODO: Skip the solve here because sig_m = I?)
    sig_x__m = sig_x - sig_x_m @ solve(sig_m, sig_x_m.T)
    sig_y__m = sig_y - sig_y_m @ solve(sig_m, sig_y_m.T)

    # Whiten the X-channel
    sig_mxy[:, dm:dm+dx] = solve(la.sqrtm(sig_x__m).real, sig_mxy[:, dm:dm+dx].T).T
    sig_mxy[dm:dm+dx, :] = solve(la.sqrtm(sig_x__m).real, sig_mxy[dm:dm+dx, :])

    # Whiten the Y-channel
    sig_mxy[:, dm+dx:] = solve(la.sqrtm(sig_y__m).real, sig_mxy[:, dm+dx:].T).T
    sig_mxy[dm+dx:, :] = solve(la.sqrtm(sig_y__m).real, sig_mxy[dm+dx:, :])

    # Extract the final joint covariance of (X, Y) given M
    sig_xy = sig_mxy[dm:, dm:]
    sig_xy_m = sig_mxy[dm:, :dm]
    sig_xy__m = sig_xy - sig_xy_m @ solve(sig_m, sig_xy_m.T) # TODO: Skip solve?

    if ret_channel_params:
        return sig_mxy, sig_x_m, sig_y_m, sig_xy_m, sig_xy__m
    return sig_mxy


def create_flows(n_flows, latent_size, q0, flow_type='RealNVP'):
    flows = norm_flows(n_flows, latent_size, flow_type)
    return nf.NormalizingFlow(
        q0=q0,
        flows=flows,
    )


class CartesianProductFlow(nn.Module):
    def __init__(self, dm, dx, dy, n_flows, encoder=None, gamma=1.0):
        super(CartesianProductFlow, self).__init__()
        self.dm = dm
        self.dx = dx
        self.dy = dy

        self.model_m = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dm),
            flows=norm_flows(n_flows, dm),
        )
        self.model_x = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dx),
            flows=norm_flows(n_flows, dx),
        )
        self.model_y = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dy),
            flows=norm_flows(n_flows, dy),
        )

        self.q_mx = nf.distributions.base.GaussianPCA(dm + dx, dm)
        self.q_my = nf.distributions.base.GaussianPCA(dm + dy, dm)

        self.encoder = encoder
        self.gamma = gamma

    def forward(self, m, x, y):
        if self.encoder is not None:
            x, y = self.encoder(x, y)

        z_x = self.model_x.inverse(x)
        z_y = self.model_y.inverse(y)
        z_m = self.model_m.inverse(m)

        return z_m, z_x, z_y

    def learning_loss(self, m, x, y):
        lmi_loss = 0.0
        if self.encoder is not None:
            lmi_loss += self.encoder.learning_loss(x, y, m)
            x, y = self.encoder(x, y)

        z_m, log_det_m = self.model_m.inverse_and_log_det(m)
        z_x, log_det_x = self.model_x.inverse_and_log_det(x)
        z_y, log_det_y = self.model_y.inverse_and_log_det(y)

        z_mx = torch.cat([z_m, z_x], dim=-1)
        z_my = torch.cat([z_m, z_y], dim=-1)
        log_prob = self.q_mx.log_prob(z_mx) + self.q_my.log_prob(z_my)
        log_det = log_det_m * 2 + log_det_x + log_det_y

        loss = -torch.mean(log_prob + log_det)
        if self.encoder is not None:
            loss += lmi_loss * self.gamma

        return loss

    def estimate_latent_mean(self, m, x, y):
        z_m, z_x, z_y = self.forward(m, x, y)
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        return torch.mean(z_combined, dim=0)

    def stack_mxy(self, m, x, y):
        z_m, z_x, z_y = self.forward(m, x, y)
        z_mxy = torch.cat([z_m, z_x, z_y], dim=-1)
        return z_mxy

    def estimate_latent_cov(self, m, x, y):
        z_m, z_x, z_y = self.forward(m, x, y)
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        return torch.cov(z_combined.T)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class BroadcastChannelFlow(nn.Module):
    def __init__(self, dm, dx, dy, n_flows, encoder=None, gamma=1.0):
        super(BroadcastChannelFlow, self).__init__()
        self.dm = dm
        self.dx = dx
        self.dy = dy

        context_encoder_x = nf.nets.MLP([dm, dx * 2], init_zeros=True)
        context_encoder_y = nf.nets.MLP([dm, dy * 2], init_zeros=True)

        self.nf_m = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dm),
            flows=norm_flows(n_flows, dm),
        )
        self.nf_x = nf.NormalizingFlow(
            q0=nf.distributions.base.ConditionalDiagGaussian(dx, context_encoder_x),
            flows=norm_flows(n_flows, dx),
        )
        self.nf_y = nf.NormalizingFlow(
            q0=nf.distributions.base.ConditionalDiagGaussian(dy, context_encoder_y),
            flows=norm_flows(n_flows, dy),
        )

        self.encoder = encoder
        self.gamma = gamma

    def forward(self, m, x, y):
        if self.encoder is not None:
            x, y = self.encoder(x, y)
        z_m = self.nf_m.inverse(m)
        z_x = self.nf_x.inverse(x)
        z_y = self.nf_y.inverse(y)
        return z_m, z_x, z_y

    def learning_loss(self, m, x, y):
        lmi_loss = 0.0
        if self.encoder is not None:
            lmi_loss += self.encoder.learning_loss(x, y, m)
            x, y = self.encoder(x, y)

        z_m, log_det_m = self.nf_m.inverse_and_log_det(m)
        z_x, log_det_x = self.nf_x.inverse_and_log_det(x)
        z_y, log_det_y = self.nf_y.inverse_and_log_det(y)
        log_det = log_det_m + log_det_x + log_det_y
        log_prob = (self.nf_m.q0.log_prob(z_m) + self.nf_x.q0.log_prob(z_x, context=z_m)
                    + self.nf_y.q0.log_prob(z_y, context=z_m))
        loss = -(log_prob + log_det).mean()

        if self.encoder is not None:
            loss += lmi_loss * self.gamma

        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


def norm_flows(n_flows, latent_size, flow_type='RealNVP'):
    if flow_type == 'Planar':
        flows = [nf.flows.Planar((latent_size,)) for k in range(n_flows)]
    elif flow_type == 'Radial':
        flows = [nf.flows.Radial((latent_size,)) for k in range(n_flows)]
    elif flow_type == 'Spline':
        flows = []
        for i in range(n_flows):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, 2, 128)]
            flows += [nf.flows.LULinearPermute(latent_size)]
    elif flow_type == 'RealNVP':
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
        flows = []
        for i in range(n_flows):
            s = nf.nets.MLP([latent_size, latent_size // 2, 64, latent_size], init_zeros=True)
            t = nf.nets.MLP([latent_size, latent_size // 2, 64, latent_size], init_zeros=True)
            if i % 2 == 0:
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
            flows += [nf.flows.ActNorm(latent_size)]
    else:
        raise NotImplementedError

    return flows


def glows(n_flows, n_bottleneck, channels, hidden_channels, input_shape, num_classes):
    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(n_flows):
        flows_ = []
        for j in range(n_bottleneck):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (n_flows + 1 - i), hidden_channels,
                                          split_mode='channel', scale=True)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (n_flows - i), input_shape[1] // 2 ** (n_flows - i),
                            input_shape[2] // 2 ** (n_flows - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (n_flows + 1), input_shape[1] // 2 ** n_flows,
                            input_shape[2] // 2 ** n_flows)
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

    return q0, flows, merges


def conditional_flows(n_flows, latent_size, hidden_units, hidden_layers, context_size, flow_type='AutoregressiveNeuralSpline'):
    flows = []
    if flow_type == 'MaskedAffineAutoregressive':
        for i in range(n_flows):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units,
                                                          context_features=context_size,
                                                          num_blocks=hidden_layers)]
            flows += [nf.flows.LULinearPermute(latent_size)]
    elif flow_type == 'AutoregressiveNeuralSpline':
        for i in range(n_flows):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,
                                                                     num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]
    elif flow_type == 'CoupledNeuralSpline':
        for i in range(n_flows):
            flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,
                                                              num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

    return flows


import numpy as np
import math
import torch
from torch.special import log_ndtr

import warnings
import numpy as np
import scipy.linalg as la
import numpy.linalg as npla

import sys
import warnings

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.linalg as npla


def objective(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    obj = 0.5 / np.log(2) * npla.slogdet(
        np.eye(dm) + hy.T @ hy + B.T @ la.solve(S, B)
    )[1]
    return obj


def gradient(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    S_inv_B = la.solve(S, B)
    g_sig = S_inv_B @ la.solve(np.eye(dm) + hy.T @ hy + B.T @ S_inv_B,
                               S_inv_B.T @ sig - hy.T)
    return g_sig


def project(sig_temp):
    """
    Returns a projection of sig_temp which satisfies:
        I - sig_temp @ sig_temp.T is positive semi-definite.
    Also returns a flag to indicate whether the projection changed the matrix.
    """
    dx, dy = sig_temp.shape

    # Have we projected at least once?
    proj_once = False

    # Full conditional cov matrix of X, Y given M
    covxy__m = np.block([[np.eye(dx), sig_temp],
                         [sig_temp.T, np.eye(dy)]])

    # Project covxy__m back onto the PSD cone
    lamda, V = la.eigh(covxy__m)
    lamda = lamda.real  # Real symmetric matrix should have real eigenvalues
    if (lamda > 0).all():
        return sig_temp, False

    V = V.real
    lamda[lamda < 0] = 0
    lamda[lamda > 1] = 1
    covxy__m = V @ np.diag(lamda) @ V.T

    covx__m = covxy__m[:dx, :dx]
    covy__m = covxy__m[dx:, dx:]

    # Regularization to use
    reg = 1e-12

    while reg < 1.1:
        # Pull sig out of covxy and re-standardize
        # Setting several elements of lamda to zero will likely cause covx__m
        # and/or covy__m to be rank-deficient. So we regularize, and keep
        # increasing the amount of regularization until the projection works.
        sig_temp_proj = la.solve(
            reg * np.eye(dx) + la.sqrtm(covx__m).real,
            la.solve(
                reg * np.eye(dy) + la.sqrtm(covy__m).real, covxy__m[:dx, dx:].T
            ).T
        )

        new_covxy__m = np.block([[np.eye(dx), sig_temp_proj],
                                 [sig_temp_proj.T, np.eye(dy)]])
        new_lamda = la.eigvalsh(new_covxy__m)

        if (new_lamda > 0).all():
            break
        else:
            reg *= 10

    if reg >= 1.1:
        warnings.warn('Projection failed: could not find a feasible point')

    return sig_temp_proj, True


def pinv(a):
    """
    la.pinv sometimes raises an la.LinAlgError: SVD did not converge. This
    appears to be some kind of BLAS/LAPACK bug:
    https://github.com/numpy/numpy/issues/12941.

    The solution appears to be to use the gesvd driver for the SVD, which is
    slower, but does not appear to cause this problem. In our case, this
    means we need a manual implementation of pinv that uses this driver.

    This function is a simplified version of the scipy implementation:
    https://github.com/scipy/scipy/blob/v1.11.1/scipy/linalg/_basic.py#L1319-L1464
    """

    # NOTE: We could use a try-except block to try using the default la.pinv
    # first, and only use this function if that fails, but since we do not use
    # pinv in any time-critical loops, it might be better to simply have a
    # slower but more accurate result.
    u, s, vh = la.svd(a, lapack_driver='gesvd')
    t = u.dtype.char.lower()
    maxS = np.max(s)

    atol = 0.
    rtol = max(a.shape) * np.finfo(t).eps

    val = atol + maxS * rtol
    rank = np.sum(s > val)

    u = u[:, :rank]
    u /= s[:rank]
    B = (u @ vh[:rank]).conj().T

    return B


def exact_tilde_union_info_minimizer(hx, hy, plot=False, ret_obj=False, reg=1e-7, verbose=False):
    dx, dm = hx.shape
    dy, dm_ = hy.shape
    if dm != dm_:
        raise ValueError('Incompatible shapes for Hx and Hy')

    # Initialize sig
    # XXX: Choice of which to pinv is arbitrary - can we average instead?
    # (based on the two equations: we can either write it out in terms of
    # hx - sig @ hy or hy - sig.T @ hx)
    sig_temp = hx @ pinv(hy)
    sig_temp_proj = project(sig_temp)[0]
    sig = sig_temp_proj.copy()

    # Gradient descent
    eta_sig = 1e-3 * np.ones((dx, dy))
    beta = 0.9       # Factor to increase or decrease LR for Rprop
    alpha = 0.999    # Slow decay of overall learning rate

    #reg = 1e-7       # Regularization in the objective for matrix inverse
    noise_std = 0    # Standard deviation of noise to add to the gradient
    #stop_threshold = 1e-6  # Absolute difference in objective for stopping
    stop_threshold = reg
    max_iterations = 20000
    patience = 20    # Num iters with small gradient before stopping (min=1)
    extra_iters = 0  # Num of extra iters after stop criterion is attained

    minima = None
    g_sig_prev = None
    running_obj = []
    if plot:
        running_sig_pre_proj = [sig_temp,]
        running_sig_post_proj = [sig_temp_proj,]
        running_grad = []
        running_eta = []
    i = 1
    extra = 0

    obj_hist = np.array([])
    while True:
        # Evaluate the objective
        obj = objective(sig, hx, hy, dm, dx, dy, reg)

        if minima is None or obj < min(running_obj):
            minima = (sig.copy(), obj)

        if len(running_obj) >= patience:
            if extra == 0:
                if (np.abs(np.array(running_obj[-patience:]) - obj) < stop_threshold).all() or i >= max_iterations:
                    if i >= max_iterations:
                        warnings.warn('Exceeded maximum number of iterations. May not have converged.')
                    if extra_iters == 0: break
                    extra += 1
            elif extra > extra_iters:
                break
            else:
                extra += 1

        if np.isnan(obj):
            running_obj.append(np.inf)
        else:
            running_obj.append(obj)
        i += 1

        g_sig = gradient(sig, hx, hy, dm, dx, dy, reg)
        g_sig = np.sign(g_sig).astype(int)

        # Backtracking with Rprop would have to work by ensuring that
        # gradients are moving in the right direction along all dimensions.
        #
        # This won't work well in conjunction with *projected* gradient
        # descent, because the projection step may be parallel and opposite
        # to the gradient step in one dimension. If so, that dimension can
        # never be made to move in the right direction, meaning
        # backtracking will fail before convergence.

        # Vanilla gradient descent
        sig_plus = sig - alpha**i * eta_sig * g_sig

        # Project sig back onto the PSD cone
        sig_proj, _ = project(sig_plus)

        # Learning rate update
        if g_sig_prev is not None:
            sign_changed = - g_sig * g_sig_prev  # -1 if sign did not change, +1 if sign changed
            eta_sig *= beta**sign_changed

        g_sig_prev = g_sig

        if plot:
            running_eta.append(eta_sig)
            running_grad.append(g_sig)
            running_sig_pre_proj.append(sig_plus)
            running_sig_post_proj.append(sig_proj)

        sig[:, :] = sig_proj

        if verbose:
            obj_hist = np.append(obj_hist, obj)

    if plot:
        running_sig_pre_proj = np.array(running_sig_pre_proj).squeeze()
        running_sig_post_proj = np.array(running_sig_post_proj).squeeze()
        running_grad = np.array(running_grad).squeeze()

        nrows = 2
        ncols = 2
        plt.figure(figsize=(10, 7))
        plt.subplot(nrows, ncols, 1)
        plt.semilogy(running_obj)
        plt.title('Convergence of objective')
        plt.ylabel('Objective')
        plt.xlabel('Iteration')

        if dx == 1 or dy == 1:
            plt.subplot(nrows, ncols, 2)
            plt.plot(running_sig_pre_proj)
            plt.plot(running_sig_post_proj)
            plt.plot(running_grad)
            plt.title('Convergence of minimizer')
            plt.ylabel('$\Sigma_i$')
            plt.xlabel('Iteration')
        elif dx == 2 and dy == 2:
            plt.subplot(nrows, ncols, 2)
            pre_proj = running_sig_pre_proj.reshape((running_sig_pre_proj.shape[0], -1))
            post_proj = running_sig_post_proj.reshape((running_sig_post_proj.shape[0], -1))
            plt.plot(post_proj)
            plt.title('Convergence of minimizer')
            plt.ylabel('$\Sigma_i$')
            plt.xlabel('Iteration')

        if dx == 1 and dy == 1:
            plt.subplot(nrows, ncols, 3)
            x_ = np.linspace(-1, 1, 100)
            #x_ = np.linspace(0.69, 0.71, 100)
            x = 0.5 * (x_[1:] + x_[:-1])
            t = np.arange(len(running_obj) + 1)
            objs = []
            for xi in x:
                sig = np.array([[xi]])
                if np.any(np.eye(dx) - sig @ sig.T <= 0):
                    objs.append(np.nan)
                    continue
                obj = objective(sig, hx, hy, dm, dx, dy, reg)
                objs.append(obj)
            objs = np.repeat(np.array(objs).reshape((1, -1)), len(running_obj), axis=0)
            plt.pcolormesh(t, x_, objs.T, cmap='jet')
            plt.colorbar()
            plt.plot(running_sig_post_proj, 'w-')
            #plt.ylim((0.69, 0.71))

        if (dx == 2 and dy == 1) or (dx == 1 and dy == 2):
            plt.subplot(nrows, ncols, 3)
            x, y = np.mgrid[-1:1:100j, -1:1:100j]
            sigs = np.moveaxis(np.array((x, y)), 0, 2)
            objs = []
            for sig in sigs.reshape((-1, 2)):
                sig = sig.reshape((dx, dy))
                if np.any(npla.eigvals(np.eye(dx) - sig @ sig.T) < 0):
                    objs.append(np.nan)
                    continue
                obj = objective(sig, hx, hy, dm, dx, dy, reg)
                objs.append(obj)
            objs = np.array(objs).reshape(sigs.shape[:2])
            plt.pcolormesh(x, y, objs[:-1, :-1], cmap='jet')
            plt.colorbar()
            #for pre, post in zip(running_sig_pre_proj, running_sig_post_proj):
            #    plt.plot([pre[0], post[0]], [pre[1], post[1]], 'k-')
            #for pre, post in zip(running_sig_pre_proj[1:], running_sig_post_proj[:-1]):
            #    plt.plot([pre[0], post[0]], [pre[1], post[1]], 'w-')
            #plt.plot(running_sig_pre_proj[:, 0], running_sig_pre_proj[:, 1], 'k-')
            plt.plot(running_sig_post_proj[:, 0], running_sig_post_proj[:, 1], 'w-')
            plt.plot(running_sig_post_proj[0, 0], running_sig_post_proj[0, 1], 'ko')

        if dx == 2 and dy == 2:
            plt.subplot(nrows, ncols, 3)
            x = np.linspace(-1, 1, 100)
            for i in range(2):
                for j in range(2):
                    post_proj = running_sig_post_proj[-1]
                    sigs = post_proj * np.ones((100, 2, 2))
                    sigs[:, i, j] = x
                    objs = []
                    for sig in sigs:
                        if np.any(npla.eigvals(np.eye(dx) - sig @ sig.T) < 0):
                            objs.append(np.nan)
                            continue
                        obj = objective(sig, hx, hy, dm, dx, dy, reg)
                        objs.append(obj)
                    plt.plot(x, objs, label=('$\Sigma_{%d%d}$' % (i, j)))
            plt.title('Objective around optima')
            plt.xlabel('$\Sigma_{ij}$')
            plt.ylabel('Objective')
            plt.legend()

        #plt.subplot(nrows, ncols, 4)
        #plt.semilogy(running_eta)

        plt.show()

    sig, obj = minima

    if ret_obj:
        return sig, obj, i, obj_hist
    return sig


def bias(d, n):
    return sum(np.log(1 - k / n) for k in range(1, d+1)) / np.log(2) / 2


def compute_bias(du, dv, n):
    """
    Compute the bias in the mutual information estimate based on the work of
    Cai et al. (J. Mult. Anal., 2015).

    This value needs to be subtracted from the mutual information estimate to
    recover the unbiased mutual information.
    """
    # Bias of differential entropy estimate
    return bias(du, n) + bias(dv, n) - bias(du + dv, n)


def debias(imxy, bias_):
    """Remove bias while ensuring non-negativity."""

    return np.maximum(imxy - bias_, 0)


def exact_gauss_tilde_pid(cov, dm, dx, dy, verbose=False, ret_t_sigt=False,
                          plot=False, unbiased=False, sample_size=None):

    # XXX: Debiasing has not been thoroughly tested.
    # Right now, we assume that the proportion of bias in the union information
    # is the same as the proportion of bias in I(M ; (X, Y)).

    # Regularization
    reg = 1e-7

    if unbiased == True and sample_size is None:
        raise ValueError('Must supply sample_size when requesting unbiased estimates')

    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy + reg * np.eye(*sigxy.shape), hxy))[1] / np.log(2)

    if unbiased:
        imx = debias(imx, compute_bias(dm, dx, sample_size))
        imy = debias(imy, compute_bias(dm, dy, sample_size))
        imxy_debiased = debias(imxy, compute_bias(dm, dx + dy, sample_size))

        # But ensure that the debiased imxy does not go below the debiased imx
        # or the debiased imy, as this will make PID values negative
        imxy_debiased = max(imxy_debiased, imx, imy)
    else:
        imxy_debiased = imxy

    debias_factor = imxy_debiased / imxy

    #sig = exact_tilde_union_info_minimizer(hx, hy, plot=plot)
    sig, obj, _, obj_hist = exact_tilde_union_info_minimizer(hx, hy, plot=plot, ret_obj=True, reg=reg)
    covxy__m = np.block([[np.eye(dx), sig], [sig.T, np.eye(dy)]])
    #covxy = covxy__m + np.vstack((hx, hy)) @ np.vstack((hx, hy)).T

    #union_info = 0.5 / np.log(2) * npla.slogdet(
    #    np.eye(dm) + hxy.T @ la.solve(covxy__m + 1e-7 * np.eye(*covxy__m.shape), hxy))[1]
    #union_info = obj
    union_info = objective(sig, hx, hy, dm, dx, dy, reg=reg)

    union_info *= debias_factor

    # Union info is lower bounded by max{I(M; X), I(M; Y)} and upper bounded by
    # min{I(M; X) + I(M; Y), I(M; (X, Y))}: imposing this ensures positivity of
    # the PID terms
    union_info = max(union_info, imx, imy)
    union_info = min(union_info, imx + imy, imxy_debiased)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy_debiased - union_info

    #uix = (union_info - imy) * debias_factor
    #uiy = (union_info - imx) * debias_factor
    #ri = (imx + imy - union_info) * debias_factor
    #si = (imxy - union_info) * debias_factor

    # Return union_info and None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    if verbose:
        return ret, obj_hist

    return ret

# Suppress the specific warning: linAlgWarning: Ill-conditioned matrix since we can just resolve it or approximate it
# warnings.filterwarnings("ignore", category=la.LinAlgWarning)

def objective(sig, hx, hy, dm, dx, dy, reg):  # the objective function
    dx, dy = sig.shape

    H = np.concatenate((hx, hy), axis=0)
    sig_all = np.block([[np.eye(dx), sig],
                        [sig.T, np.eye(dy)]])

    obj = 0.5 / np.log(2) * (npla.slogdet(H @ H.T + sig_all)[1] - npla.slogdet(sig_all)[1])
    return obj


def gradient(sig, hx, hy, dm, dx, dy, reg):
    H = np.concatenate((hx, hy), axis=0)
    sig_all = np.block([[np.eye(dx), sig],
                        [sig.T, np.eye(dy)]])

    G = H @ H.T + sig_all
    G_11 = G[0:dx, 0:dx]
    G_12 = G[0:dx, dx:dx + dy]
    G_22 = G[dx:dx + dy, dx:dx + dy]
    G11_inv_G12 = la.solve(G_11, G_12)

    # Matrix computation intermediate steps to compute the gradient
    A = G_22 - G_12.T @ G11_inv_G12 + reg * np.eye(dy)
    A1 = np.eye(dy) - sig.T @ sig + reg * np.eye(dy)

    sol_AB = npla.solve(A, G11_inv_G12.T)
    sol_AB1 = npla.solve(A1, sig.T)
    g_sig = -sol_AB + sol_AB1
    g_sig = g_sig.T
    return g_sig


def thin_project(sig_temp):  # project the matrix onto the PSD cone, but only need to work with the upper triangular part

    U, S, VT = npla.svd(sig_temp, full_matrices=False)
    S_clamped = np.clip(S, 0, 0.99999999999999999)  # clamp to [1e-10,0.99999999999999999] to avoid numerical issues
    S_clamped_matrix = np.diag(S_clamped)
    sig_proj = U @ S_clamped_matrix @ VT

    return sig_proj, True


def exact_thin_pid_minimizer(hx, hy, plot=False, ret_obj=False, reg=1e-7, max_iters=20000, verbose=False):
    dx, dm = hx.shape
    dy, dm_ = hy.shape
    if dm != dm_:
        raise ValueError('Incompatible shapes for Hx and Hy')

    swap = False
    if dx < dy:  # Swap if necessary, since we assume dx >= dy
        dx, dy = dy, dx
        hy, hx = hx, hy
        swap = True

    # Gradient descent
    eta_sig = 1e-3 * np.ones((dx, dy))
    beta = 0.9  # Factor to increase or decrease LR for Rprop
    alpha = 0.999  # Slow decay of overall learning rate

    stop_threshold = reg
    max_iterations = max_iters  # Maximum number of iterations
    patience = 20  # Num iters with small gradient before stopping (min=1)
    extra_iters = 0  # Num of extra iters after stop criterion is attained

    minima = None
    g_sig_prev = None
    running_obj = []
    i = 1
    extra = 0

    sig_temp = hx @ pinv(hy)

    try:
        sig_temp_proj, _ = thin_project(sig_temp)
    except npla.LinAlgError:
        warnings.warn('Thin projection failed, falling back to tilde projection.')
        sig_temp_proj, _ = project(sig_temp)
    sig = sig_temp_proj.copy()

    obj_hist = np.array([])
    while True:
        # Evaluate the objective
        obj = objective(sig, hx, hy, dm, dx, dy, reg)

        if minima is None or obj < min(running_obj):
            minima = (sig.copy(), obj)

        if len(running_obj) >= patience:
            if extra == 0:
                if (np.abs(np.array(running_obj[-patience:]) - obj) < stop_threshold).all() or i >= max_iterations:
                    if i >= max_iterations:

                        warnings.warn('Exceeded maximum number of iterations. May not have converged.')
                    if extra_iters == 0: break
                    extra += 1
            elif extra > extra_iters:
                break
            else:
                extra += 1

        if np.isnan(obj):
            running_obj.append(np.inf)
        else:
            running_obj.append(obj)
        i += 1

        g_sig = gradient(sig, hx, hy, dm, dx, dy, reg)
        g_sig = np.sign(g_sig).astype(int)

        # gradient descent
        sig_plus = sig - alpha ** i * eta_sig * g_sig
        # project sig back onto the PSD cone, but only need to work with the upper triangular part
        try:
            sig_proj, _ = thin_project(sig_plus)
        except npla.LinAlgError:
            warnings.warn('Thin projection failed, falling back to tilde projection.')
            sig_proj, _ = project(sig_plus)

        # Learning rate update
        if g_sig_prev is not None:
            sign_changed = - g_sig * g_sig_prev  # -1 if sign did not change, +1 if sign changed
            eta_sig *= beta ** sign_changed

        g_sig_prev = g_sig
        sig[:, :] = sig_proj

        if verbose:
            obj_hist = np.append(obj_hist, obj)

    sig, obj = minima

    if swap:
        sig = sig.T

    if ret_obj:
        return sig, obj, i, obj_hist
    return sig


def bias(d, n):
    return sum(np.log(1 - k / n) for k in range(1, d+1)) / np.log(2) / 2


def compute_bias(du, dv, n):
    """
    Compute the bias in the mutual information estimate based on the work of
    Cai et al. (J. Mult. Anal., 2015).

    This value needs to be subtracted from the mutual information estimate to
    recover the unbiased mutual information.
    """
    # Bias of differential entropy estimate
    return bias(du, n) + bias(dv, n) - bias(du + dv, n)


def debias(imxy, bias_):
    """Remove bias while ensuring non-negativity."""

    return np.maximum(imxy - bias_, 0)


def exact_gauss_thin_pid(cov, dm, dx, dy, verbose=False, ret_t_sigt=False,
                          plot=False, unbiased=False, sample_size=None):

    # XXX: Debiasing has not been thoroughly tested.
    # Right now, we assume that the proportion of bias in the union information
    # is the same as the proportion of bias in I(M ; (X, Y)).

    # Regularization
    reg = 1e-7

    if unbiased == True and sample_size is None:
        raise ValueError('Must supply sample_size when requesting unbiased estimates')

    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy + reg * np.eye(*sigxy.shape), hxy))[1] / np.log(2)

    if unbiased:
        imx = debias(imx, compute_bias(dm, dx, sample_size))
        imy = debias(imy, compute_bias(dm, dy, sample_size))
        imxy_debiased = debias(imxy, compute_bias(dm, dx + dy, sample_size))

        # But ensure that the debiased imxy does not go below the debiased imx
        # or the debiased imy, as this will make PID values negative
        imxy_debiased = max(imxy_debiased, imx, imy)
    else:
        imxy_debiased = imxy

    debias_factor = imxy_debiased / imxy

    sig, obj, _, obj_hist = exact_thin_pid_minimizer(hx, hy, plot=plot, ret_obj=True, reg=reg, verbose=verbose)

    union_info = objective(sig, hx, hy, dm, dx, dy, reg=reg)
    union_info *= debias_factor

    # Union info is lower bounded by max{I(M; X), I(M; Y)} and upper bounded by
    # min{I(M; X) + I(M; Y), I(M; (X, Y))}: imposing this ensures positivity of
    # the PID terms
    union_info = max(union_info, imx, imy)
    union_info = min(union_info, imx + imy, imxy_debiased)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy_debiased - union_info

    # Return union_info and None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    if verbose:
        return ret, obj_hist

    return ret

from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def fit(model, dataloader, n_epochs, lr, device='cpu', verbose=False):
    nfm = fit_flows(model, dataloader, n_epochs, lr, device, verbose=verbose)
    ret = fit_pid(nfm, dataloader, device=device, verbose=verbose)
    return ret


def fit_flows(model, dataloader, n_epochs, lr, device='cpu', verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)
    data_iter = iter(dataloader)

    loss_hist = np.array([])
    model.train()
    for i in tqdm(range(n_epochs)):
        try:
            x, y, m = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y, m = next(data_iter)
        optimizer.zero_grad()
        loss = model.learning_loss(m.to(device), x.to(device), y.to(device))

        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
            scheduler.step()

        loss_hist = np.append(loss_hist, loss.detach().to('cpu').numpy())

    if verbose:
        print(f"Final Loss: {loss_hist[-1]:.4f}")
        plt.figure(figsize=(5, 5))
        plt.plot(loss_hist, label='loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()

    return model


def fit_pid(model, dataloader, device='cpu', verbose=False, ret_t_sigt=False):
    z_mxy = []
    model.eval()
    for x_batch, y_batch, m_batch in dataloader:
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
        with torch.no_grad():
            z_m, z_x, z_y = model(m_batch, x_batch, y_batch)
            z_mxy.append(torch.cat([z_m, z_x, z_y], dim=-1))

    z_mxy = torch.cat(z_mxy, dim=0)
    cov = torch.cov(z_mxy.T).cpu().numpy()
    trained_cov = covariance_to_correlation(cov)
    ret = exact_gauss_thin_pid(trained_cov, model.dm, model.dx, model.dy, verbose=False, ret_t_sigt=ret_t_sigt)

    return ret


def covariance_to_correlation(covariance_matrix):
    covariance_matrix = np.array(covariance_matrix)

    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square")

    if not np.allclose(covariance_matrix, covariance_matrix.T):
        raise ValueError("Covariance matrix must be symmetric")

    std_devs = np.sqrt(np.diag(covariance_matrix))
    outer_std = np.outer(std_devs, std_devs)
    correlation_matrix = covariance_matrix / outer_std
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix


def standardize_data(data):
    # standardize data along columns
    return (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)


def train_flow(m_data, x_data, y_data, n_flows, n_epochs=100, batch_size=64, lr=2e-4, encoder=None, verbose=False,
               device='cuda'):
    # Initialize model
    dim_x, dim_y, dim_m = x_data.shape[1], y_data.shape[1], m_data.shape[1]

    encoder = encoder.to(device) if encoder is not None else None
    flow = CartesianProductFlow(dim_m, dim_x, dim_y, n_flows).to(device)

    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Create and standardize dataset
    x_data = torch.tensor(x_data, dtype=torch.float32)
    x_data_standardized = standardize_data(x_data)

    y_data = torch.tensor(y_data, dtype=torch.float32)
    y_data_standardized = standardize_data(y_data)

    m_data = torch.tensor(m_data, dtype=torch.float32)
    m_data_standardized = standardize_data(m_data)

    dataset = TensorDataset(x_data_standardized, y_data_standardized, m_data_standardized)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []

    for epoch in tqdm(range(n_epochs)):
        epoch_losses = []

        for x_batch, y_batch, m_batch in dataloader:
            x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)

            optimizer.zero_grad()

            # Forward pass
            m_features, x_features, y_features = encoder(m_batch, x_batch, y_batch) if encoder is not None else (
            m_batch, x_batch, y_batch)
            # z_m, z_x, z_y, log_det = flow(m_features, x_features, y_features)

            # Compute loss
            loss = flow.learning_loss(m_features, x_features, y_features)

            # Backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)

    # Plot loss curve
    if verbose:
        final_loss = losses[-1]
        print(f"Final Loss: {final_loss:.4f}")

    # save the encoder and flow
    torch.save(encoder, 'encoder.pt')
    torch.save(flow, 'flow.pt')

    z_combined = []
    for x_batch, y_batch, m_batch in dataloader:
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
        with torch.no_grad():
            m_batch, x_batch, y_batch = encoder(m_batch, x_batch, y_batch) if encoder is not None else (
            m_batch, x_batch, y_batch)
            z_m, z_x, z_y = flow(m_batch, x_batch, y_batch)
            z_combined.append(torch.cat([z_m, z_x, z_y], dim=-1))
    z_combined = torch.cat(z_combined, dim=0)
    cov = torch.cov(z_combined.T).cpu().numpy()

    return flow, cov, losses


def trained_covariance(m_data, x_data, y_data, flow_path, encoder_path=None, device='cuda'):
    encoder = torch.load(encoder_path) if encoder_path is not None else None
    flow = torch.load(flow_path)

    x_data = torch.tensor(x_data, dtype=torch.float)
    x_data_standardized = standardize_data(x_data)

    y_data = torch.tensor(y_data, dtype=torch.float)
    y_data_standardized = standardize_data(y_data)

    m_data = torch.tensor(m_data, dtype=torch.float)
    m_data_standardized = standardize_data(m_data)

    dataset = TensorDataset(x_data_standardized, y_data_standardized, m_data_standardized)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    z_combined = []
    for x_batch, y_batch, m_batch in dataloader:
        x_batch, y_batch, m_batch = x_batch.to(device), y_batch.to(device), m_batch.to(device)
        with torch.no_grad():
            m_batch, x_batch, y_batch = encoder(m_batch, x_batch, y_batch) if encoder is not None else (
            m_batch, x_batch, y_batch)
            z_m, z_x, z_y, log_det = flow(m_batch, x_batch, y_batch)
            z_combined.append(torch.cat([z_m, z_x, z_y], dim=-1))
    z_combined = torch.cat(z_combined, dim=0)
    cov = torch.cov(z_combined.T).cpu().numpy()

    return cov


"""def flow_pid(m, x, y, n_flows=3, n_epochs=250, batch_size=64, lr=2e-4, encoder=None, verbose=False, ret_t_sigt=False,
             device='cpu'):
    trained_flow, trained_cov, training_losses = train_flow(m, x, y, n_flows=n_flows,
                                                            n_epochs=n_epochs, batch_size=batch_size, lr=lr,
                                                            encoder=encoder, verbose=verbose, device=device)

    trained_cov = covariance_to_correlation(trained_cov)
    ret = exact_gauss_thin_pid(trained_cov, m.shape[1], x.shape[1], y.shape[1], verbose=False, ret_t_sigt=ret_t_sigt)
    return ret"""

# -------------------------
# Argparse
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--pickle-path", type=str, required=True)
parser.add_argument("--out-pt", type=str, default=None)
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--embed-size", type=int, default=32)
parser.add_argument("--num-classes", type=int, required=True)
parser.add_argument("--epochs-disc", type=int, default=30)
parser.add_argument("--epochs-entropy", type=int, default=30)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# -------------------------
# Load pickle
# -------------------------
with open(args.pickle_path, "rb") as f:
    data = pickle.load(f)

print(data.keys())

X1_train = torch.from_numpy(data["train"]["0"]).float()
X2_train = torch.from_numpy(data["train"]["1"]).float()
y_train = torch.from_numpy(data["train"]["label"]).long().squeeze()

X1_test = torch.from_numpy(data["test"]["0"]).float()
X2_test = torch.from_numpy(data["test"]["1"]).float()
y_test   = torch.from_numpy(data["test"]["label"]).long().squeeze()

X1_val = torch.from_numpy(data["valid"]["0"]).float()
X2_val = torch.from_numpy(data["valid"]["1"]).float()
y_val   = torch.from_numpy(data["valid"]["label"]).long().squeeze()

# -------------------------
# Fake cfg replacement
# -------------------------
class CFG:
    pass

cfg = CFG()
cfg.device = torch.device(args.device)
cfg.batch_size = args.batch_size
cfg.num_workers = args.num_workers
cfg.embed_size = args.embed_size
cfg.n_classes = args.num_classes
cfg.num_epochs_discriminator = args.epochs_disc
cfg.num_epochs_entropy_estimator = args.epochs_entropy

cfg.input_size_1 = X1_train.shape[1]
cfg.input_size_2 = X2_train.shape[1]

from utils_lsmi import setup_seed
setup_seed(args.seed)

def standardize_data(data):
    data = np.nan_to_num((data - data.mean(axis=0)) / data.std(axis=0))
    data = np.clip(data, a_min=-10, a_max=10)
    return data

# TRAIN
y_train_standardized = standardize_data(y_train.float())
X1_train_standardized = standardize_data(X1_train)
X2_train_standardized = standardize_data(X2_train)

train_dataset = TensorDataset(
    torch.from_numpy(X1_train_standardized).float(),
    torch.from_numpy(X2_train_standardized).float(),
    torch.from_numpy(y_train_standardized[:, None]).float(),
)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)

# VAL
y_val_standardized = standardize_data(y_val.float())
X1_val_standardized = standardize_data(X1_val)
X2_val_standardized = standardize_data(X2_val)

val_dataset = TensorDataset(
    torch.from_numpy(X1_val_standardized).float(),
    torch.from_numpy(X2_val_standardized).float(),
    torch.from_numpy(y_val_standardized[:, None]).float(),
)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, drop_last=True)

# TEST
y_test_standardized = standardize_data(y_test.float())
X1_test_standardized = standardize_data(X1_test)
X2_test_standardized = standardize_data(X2_test)

test_dataset = TensorDataset(
    torch.from_numpy(X1_test_standardized).float(),
    torch.from_numpy(X2_test_standardized).float(),
    torch.from_numpy(y_test_standardized[:, None]).float(),
)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, drop_last=False)

import copy

patience = 5                     # number of iterations to wait
best_val_loss = float("inf")
patience_counter = 0
best_model_state = None

val_loss_hist = []

train_iter = iter(train_loader)
###### flow pid ######
enable_cuda=True
device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
print(f"Using device: {device}")

model = CartesianProductFlow(
    dm=1,
    dx=200,
    dy=200,
    n_flows=6,
).to(device)

# Train model
max_iter = 5000

loss_hist = np.array([])

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-4)

for i in tqdm(range(max_iter)):
    try:
        x, y, m = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        x, y, m = next(train_iter)

    model.train()
    optimizer.zero_grad()

    loss = model.learning_loss(m.to(device), x.to(device), y.to(device))

    if not (torch.isnan(loss) or torch.isinf(loss)):
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss_hist = np.append(loss_hist, loss.detach().cpu().numpy())

    if i % 10 == 0:
        # -----------------
        # VALIDATION
        # -----------------
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for x_val, y_val, m_val in val_loader:
                batch_loss = model.learning_loss(
                    m_val.to(device),
                    x_val.to(device),
                    y_val.to(device),
                )
                val_loss += batch_loss.item()

        val_loss /= len(val_loader)
        val_loss_hist.append(val_loss)

        # -----------------
        # EARLY STOPPING
        # -----------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping at iteration {i}")
            break

print(f"Final loss: {loss_hist[-1]}")

ret = fit_pid(model, test_loader, device=device, verbose=True)
norm = ret[7] + ret[5] + ret[6] + ret[8]
print(f"Flow PID, R: {ret[7]}, U1: {ret[5]}, U2: {ret[6]}, S: {ret[8]}")