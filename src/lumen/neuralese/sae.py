"""
A sparse autoencoder (SAE) for dictionary learning on residual-stream
activations, following the methodology of Anthropic's "Towards
Monosemanticity" (Bricken et al. 2023) and Cunningham et al. 2023, scaled
down to a toy model: an overcomplete, L1-regularized single-hidden-layer
autoencoder is trained to reconstruct activations, on the hypothesis that
its hidden units ("features") pick out more interpretable, closer-to-
monosemantic directions than the raw activation basis.

    z = ReLU(W_enc @ x + b_enc)              # sparse code, dim > d_model
    x_hat = W_dec @ z + b_dec                # reconstruction
    loss = ||x - x_hat||^2 + l1_coef * |z|_1

This module trains the SAE, reports reconstruction / sparsity metrics, and
checks how strongly each learned feature correlates with the known EVAL vs
DEPLOY mode label -- the one piece of "ground truth latent structure" this
toy model has, which real SAE work usually doesn't get to check against.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List

from .tensor import Tensor, Adam


@dataclass
class SAEConfig:
    d_in: int = 32
    d_hidden: int = 128       # overcomplete: d_hidden > d_in
    l1_coef: float = 1e-2
    lr: float = 2e-3
    seed: int = 0


@dataclass
class SAETrainResult:
    sae: "SparseAutoencoder"
    loss_history: List[float]
    recon_mse_history: List[float]
    l0_history: List[float]           # mean number of active (nonzero) features per example
    dead_feature_frac: float           # fraction of features that never fire on the eval set


class SparseAutoencoder:
    def __init__(self, config: SAEConfig):
        self.cfg = config
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / np.sqrt(config.d_in)
        self.W_enc = Tensor(rng.uniform(-scale, scale, (config.d_in, config.d_hidden)), requires_grad=True)
        self.b_enc = Tensor(np.zeros(config.d_hidden), requires_grad=True)
        # Decoder initialized as (approx) transpose of encoder, a standard SAE init trick
        self.W_dec = Tensor(self.W_enc.data.T.copy(), requires_grad=True)
        self.b_dec = Tensor(np.zeros(config.d_in), requires_grad=True)

    def params(self):
        return [self.W_enc, self.b_enc, self.W_dec, self.b_dec]

    def encode(self, x: Tensor) -> Tensor:
        return (x @ self.W_enc + self.b_enc).relu()

    def decode(self, z: Tensor) -> Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def _normalize_decoder(self):
        """Keep decoder columns (feature directions) at unit norm, a standard
        SAE trick that prevents the L1 penalty from being trivially dodged
        by shrinking the decoder and inflating the encoder."""
        norms = np.linalg.norm(self.W_dec.data, axis=1, keepdims=True) + 1e-8
        self.W_dec.data /= norms


def _l1_tensor(z: Tensor) -> Tensor:
    """Mean L1 norm of z over the batch. relu(z) then relu(-z) trick avoids
    needing an abs() op: |z| = relu(z) + relu(-z)."""
    abs_z = z.relu() + (Tensor(0.0) - z).relu()
    return abs_z.sum(axis=-1).mean()


def train_sae(activations: np.ndarray, config: SAEConfig, n_epochs: int = 60,
              batch_size: int = 64, verbose: bool = False) -> SAETrainResult:
    """activations: (N, d_in) numpy array of residual-stream vectors to fit."""
    sae = SparseAutoencoder(config)
    opt = Adam(sae.params(), lr=config.lr)
    n = activations.shape[0]
    rng = np.random.default_rng(config.seed + 1)

    loss_hist, mse_hist, l0_hist = [], [], []
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = Tensor(activations[idx])
            x_hat, z = sae.forward(xb)
            diff = x_hat - xb
            mse = (diff * diff).sum(axis=-1).mean()
            l1 = _l1_tensor(z)
            loss = mse + config.l1_coef * l1
            opt.zero_grad()
            loss.backward()
            opt.step()
            sae._normalize_decoder()

        x_hat, z = sae.forward(Tensor(activations))
        diff = x_hat - Tensor(activations)
        mse_val = float((diff.data ** 2).sum(axis=-1).mean())
        l0_val = float((z.data > 1e-6).sum(axis=-1).mean())
        loss_hist.append(mse_val + config.l1_coef * float(np.abs(z.data).sum(axis=-1).mean()))
        mse_hist.append(mse_val)
        l0_hist.append(l0_val)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1:3d}  recon_mse={mse_val:.4f}  mean_L0={l0_val:.1f}/{config.d_hidden}")

    final_z = sae.encode(Tensor(activations)).data
    dead_frac = float((final_z.max(axis=0) < 1e-6).mean())
    return SAETrainResult(sae, loss_hist, mse_hist, l0_hist, dead_frac)


def feature_mode_correlation(sae: SparseAutoencoder, activations: np.ndarray, mode_labels: np.ndarray) -> np.ndarray:
    """Point-biserial correlation of each learned feature's activation with
    the binary EVAL(0)/DEPLOY(1) label. Returns (d_hidden,) array. This is
    only checkable here because the toy model gives us ground-truth latent
    structure to correlate against -- real SAE feature interpretation
    usually has to rely on reading top-activating examples instead."""
    z = sae.encode(Tensor(activations)).data  # (N, d_hidden)
    y = mode_labels.astype(np.float64)
    y_c = y - y.mean()
    z_c = z - z.mean(axis=0, keepdims=True)
    num = z_c.T @ y_c
    denom = (np.linalg.norm(z_c, axis=0) * np.linalg.norm(y_c)) + 1e-8
    return num / denom
