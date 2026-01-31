#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key
from opt_einsum import contract

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..._utils import _get_size, _identity
from ...activations._stan import Stan
from ..core._base import _AbstractBaseModel
from ..layers._linear import Linear as RealLinear


def _inv_softplus(y: Array) -> Array:
    """Inverse of softplus for y > 0 in a numerically stable way."""
    return jnp.where(y > 20.0, y, jnp.log(jnp.expm1(y)))


class _InputComplexify(StrictModule):
    phi: Array  # (in_features,)

    def __init__(self, in_features: int, key: Array):
        self.phi = jr.uniform(key, (in_features,), minval=-jnp.pi, maxval=jnp.pi)

    def __call__(self, x: Array) -> Array:
        x = jnp.asarray(x)
        return x * jnp.exp(1j * self.phi)


class _ComplexLinearSoftplus(StrictModule):
    rho: Array
    theta: Array
    bias_rho: Array
    bias_phase: Array

    def __init__(
        self,
        in_features: int,
        out_features: int,
        key: Array,
        init_mag: float | None = None,
    ):
        k1, k2, k3, k4 = jr.split(key, 4)
        self.theta = jr.uniform(
            k1, (out_features, in_features), minval=-jnp.pi, maxval=jnp.pi
        )
        self.bias_phase = jr.uniform(k2, (out_features,), minval=-jnp.pi, maxval=jnp.pi)

        r0 = (1.0 / jnp.sqrt(in_features)) if init_mag is None else float(init_mag)
        jitter_W = 0.05 * jr.normal(k3, (out_features, in_features))
        jitter_b = 0.05 * jr.normal(k4, (out_features,))
        self.rho = _inv_softplus(jnp.clip(r0 + jitter_W, 1e-5, None))
        self.bias_rho = _inv_softplus(jnp.clip(0.01 + jitter_b, 1e-5, None))

    def __call__(self, x: Array) -> Array:
        rW = jax.nn.softplus(self.rho)
        phW = jnp.exp(1j * self.theta)
        W = rW * phW

        rb = jax.nn.softplus(self.bias_rho)
        phb = jnp.exp(1j * self.bias_phase)
        b = rb * phb

        if x.ndim == 1:
            return W @ x + b
        else:
            return x @ W.T + b


class _ModReLU(StrictModule):
    b: Array
    eps: float = 1e-6

    def __init__(self, features: int, b_init: float = 0.0):
        self.b = jnp.full((features,), b_init)

    def __call__(self, z: Array) -> Array:
        r = jnp.abs(z)
        u = z / (r + self.eps)
        m = jnp.maximum(r + self.b, 0.0)
        return m * u


class _ActionNet(StrictModule):
    l1: RealLinear
    l2: RealLinear
    l3: RealLinear

    def __init__(
        self,
        in_complex_features: int,
        K: int,
        width: int,
        key: Array,
        *,
        rwf: bool | tuple[float, float] = True,
    ):
        k1, k2, k3 = jr.split(key, 3)
        self.l1 = RealLinear(
            in_size=2 * in_complex_features,
            out_size=width,
            activation=Stan(shape=width),
            rwf=rwf,
            key=k1,
        )
        self.l2 = RealLinear(
            in_size=width,
            out_size=width,
            activation=Stan(shape=width),
            rwf=rwf,
            key=k2,
        )
        self.l3 = RealLinear(in_size=width, out_size=K, activation=None, rwf=rwf, key=k3)

    def __call__(self, z: Array) -> Array:
        if z.ndim > 1:
            v = jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=-1)
        else:
            v = jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=0)
        h = self.l1(v)
        h = self.l2(h)
        return self.l3(h)


class _SumOverPathsDense(StrictModule):
    """Vectorized K-path complex dense block.

    Stores parameters for all K paths in stacked arrays and computes all
    path outputs in a single batched operation.
    """

    # Parameters stacked along K dimension
    rho: Array  # (K, out, in)
    theta: Array  # (K, out, in)
    bias_rho: Array  # (K, out)
    bias_phase: Array  # (K, out)

    action: _ActionNet
    gating_logits: Array  # (K,)
    K: int
    phase_scale: float

    def __init__(
        self,
        in_features: int,
        out_features: int,
        K: int,
        key: Array,
        width_action: int = 64,
        phase_scale: float = 1.0,
        learn_gates: bool = True,
        *,
        rwf: bool | tuple[float, float] = True,
    ):
        k_theta, k_bphase, k_rho, k_brho, k_action = jr.split(key, 5)
        # Initialize theta and bias_phase uniformly
        self.theta = jr.uniform(
            k_theta, (K, out_features, in_features), minval=-jnp.pi, maxval=jnp.pi
        )
        self.bias_phase = jr.uniform(
            k_bphase, (K, out_features), minval=-jnp.pi, maxval=jnp.pi
        )

        # Initialize magnitudes near r0 with small jitter, then invert softplus
        r0 = 1.0 / jnp.sqrt(in_features)
        jitter_W = 0.05 * jr.normal(k_rho, (K, out_features, in_features))
        jitter_b = 0.05 * jr.normal(k_brho, (K, out_features))
        self.rho = _inv_softplus(jnp.clip(r0 + jitter_W, 1e-5, None))
        self.bias_rho = _inv_softplus(jnp.clip(0.01 + jitter_b, 1e-5, None))

        # Action network and gating
        self.action = _ActionNet(in_features, K, width_action, k_action, rwf=rwf)
        logits = jnp.zeros((K,))
        self.gating_logits = logits if learn_gates else jax.lax.stop_gradient(logits)
        self.K = K
        self.phase_scale = phase_scale

    def __call__(self, x: Array) -> Array:
        # Compute complex weights for all paths
        rW = jax.nn.softplus(self.rho)  # (K, out, in)
        phW = jnp.exp(1j * self.theta)  # (K, out, in)
        W = rW * phW  # (K, out, in) complex

        rb = jax.nn.softplus(self.bias_rho)  # (K, out)
        phb = jnp.exp(1j * self.bias_phase)  # (K, out)
        b = rb * phb  # (K, out)

        g = jax.nn.softmax(self.gating_logits)  # (K,)
        alpha = self.phase_scale * self.action(x)  # (K,) or (B,K)

        if x.ndim == 1:
            # Path outputs: (K,out)
            y_paths = contract("koi,i->ko", W, x) + b
            amp = g * jnp.exp(1j * alpha)  # (K,)
            y = contract("ko,k->o", y_paths, amp)
            return y
        else:
            # Batched: x (B,in), alpha (B,K)
            y_paths = contract("koi,bi->bko", W, x) + b  # (B,K,out)
            amp = jnp.exp(1j * alpha) * g  # (B,K)
            y = contract("bko,bk->bo", y_paths, amp)
            return y


class _RealConcatDense(StrictModule):
    linear: RealLinear

    def __init__(
        self,
        in_complex_features: int,
        out_features: int,
        key: Array,
        *,
        rwf: bool | tuple[float, float] = True,
    ):
        self.linear = RealLinear(
            in_size=2 * in_complex_features,
            out_size=out_features,
            activation=None,
            rwf=rwf,
            key=key,
        )

    def __call__(self, z: Array) -> Array:
        if z.ndim > 1:
            v = jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=-1)
        else:
            v = jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=0)
        return self.linear(v)


class FeynmaNN(_AbstractBaseModel):
    r"""Feynman path-integral style network with complex hidden blocks.

    This model builds a complex-valued hidden state and updates it with a
    *sum-over-paths* block. For a hidden vector $z$ and $K$ paths, one block
    computes

    $$
    \text{Block}(z)
      = \sum_{k=1}^{K} g_k\;e^{i\,\alpha_k(z)}\,(W_k z + b_k),
    $$

    where $g=\text{softmax}(\text{logits})$ are learned gates and
    $\alpha(z)$ is produced by a small real action network (scaled by
    `phase_scale`).

    The nonlinearity is ModReLU:

    $$
    \text{ModReLU}(z)=\mathop{\text{max}}(|z|+b,0)\,\frac{z}{|z|+\varepsilon}.
    $$

    Stacking `depth` blocks yields a complex latent representation which is
    mapped to the requested output either by a real readout
    (concatenating $\Re z$ and $\Im z$) or by a complex linear projection.
    """

    # Standard (complex) path-integral components
    complexify: _InputComplexify | None
    blocks: tuple[_SumOverPathsDense, ...] | None
    activs: tuple[_ModReLU, ...] | None
    readout: _RealConcatDense | None
    # Optional complex projection to requested feature size
    latent_project: _ComplexLinearSoftplus | None

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]
    final_activation: Callable

    def __init__(
        self,
        *,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        width_size: int,
        depth: int,
        num_paths: int = 4,
        width_action: int = 32,
        phase_scale: float = 1.0,
        final_activation: Callable | None = None,
        modrelu_bias_init: float = 0.0,
        learn_gates: bool = True,
        key: Key[Array, ""] = DOC_KEY0,
        rwf: bool | tuple[float, float] = False,
        keep_output_complex: bool = False,
    ):
        in_features = _get_size(in_size)
        out_features = _get_size(out_size)

        # Activation outside readout to match MLP contract
        final_act = _identity if final_activation is None else final_activation

        # Pre-initialize all fields
        self.complexify = None
        self.blocks = None
        self.activs = None
        self.readout = None
        self.latent_project = None

        keys = jr.split(key, depth + 2)
        self.complexify = _InputComplexify(in_features, keys[0])

        blocks = []
        activs = []
        fin = in_features
        for d in range(depth):
            blocks.append(
                _SumOverPathsDense(
                    fin,
                    width_size,
                    num_paths,
                    keys[1 + d],
                    width_action=width_action,
                    phase_scale=phase_scale,
                    learn_gates=learn_gates,
                    rwf=rwf,
                )
            )
            activs.append(_ModReLU(width_size, b_init=modrelu_bias_init))
            fin = width_size

        self.blocks = tuple(blocks)
        self.activs = tuple(activs)
        if keep_output_complex:
            self.latent_project = _ComplexLinearSoftplus(fin, out_features, keys[-1])
        else:
            self.readout = _RealConcatDense(fin, out_features, keys[-1], rwf=rwf)

        self.in_size = in_size
        self.out_size = out_size
        self.final_activation = final_act

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        # Handle scalar input
        if self.in_size == "scalar":
            if x.shape != ():
                raise ValueError(f"`x` must have scalar shape, got {x.shape}")
            x = jnp.broadcast_to(x, (1,))

        assert (
            self.complexify is not None
            and self.blocks is not None
            and self.activs is not None
        )
        z = self.complexify(x)
        for block, act in zip(self.blocks, self.activs):
            z = act(block(z))
        if self.latent_project is not None:
            y = self.latent_project(z)
        else:
            assert self.readout is not None
            y = self.readout(z)

        # Scalarize if needed (handle batched and unbatched)
        if self.out_size == "scalar":
            if y.ndim == 1 and y.shape == (1,):
                y = jnp.reshape(y, ())
            elif y.ndim >= 2 and y.shape[-1] == 1:
                y = jnp.squeeze(y, axis=-1)

        y = self.final_activation(y)
        return y
