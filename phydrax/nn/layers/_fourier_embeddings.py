#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.lax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._doc import DOC_KEY0
from .._base import _AbstractBaseModel
from .._keys import EvalKey
from .._utils import _canonical_size, _get_size, _get_value_shape, _tuple, SizeLike


def _canonical_passthrough(
    passthrough: Sequence[int],
    in_dim: int,
    /,
) -> tuple[int, ...]:
    indices = tuple(int(index) for index in passthrough)
    if len(set(indices)) != len(indices):
        raise ValueError(f"`passthrough` indices must be unique, got {indices}.")
    if any(index < 0 or index >= in_dim for index in indices):
        raise ValueError(
            f"`passthrough` indices must lie in [0, {in_dim}), got {indices}."
        )
    return indices


def _as_wavevectors(
    wavevectors: ArrayLike,
    in_dim: int,
    /,
    *,
    name: str,
) -> Array:
    matrix = jnp.asarray(wavevectors, dtype=float)
    if matrix.ndim == 0:
        if in_dim != 1:
            raise ValueError(
                f"`{name}` must have trailing dimension {in_dim}, got scalar input."
            )
        matrix = matrix.reshape((1, 1))
    elif matrix.ndim == 1:
        if in_dim == 1:
            matrix = matrix.reshape((-1, 1))
        elif matrix.shape[0] == in_dim:
            matrix = matrix.reshape((1, in_dim))
        else:
            raise ValueError(
                f"`{name}` must have trailing dimension {in_dim}, got {matrix.shape}."
            )
    elif matrix.ndim != 2 or matrix.shape[1] != in_dim:
        raise ValueError(
            f"`{name}` must have shape (num_wavevectors, {in_dim}), got {matrix.shape}."
        )
    if matrix.shape[0] == 0:
        raise ValueError(f"`{name}` must contain at least one wavevector.")
    return matrix


def _as_phases(
    phases: ArrayLike | None,
    num_wavevectors: int,
    /,
) -> Array:
    if phases is None:
        return jnp.zeros((num_wavevectors,), dtype=float)
    values = jnp.asarray(phases, dtype=float)
    if values.ndim == 0:
        return jnp.broadcast_to(values, (num_wavevectors,))
    if values.shape != (num_wavevectors,):
        raise ValueError(
            f"`phases` must be scalar or have shape ({num_wavevectors},), "
            f"got {values.shape}."
        )
    return values


def _periodic_feature_size(
    out_size: int,
    passthrough: tuple[int, ...],
    include_constant: bool,
    /,
) -> int:
    out_size = int(out_size)
    if out_size <= 0:
        raise ValueError(f"`out_size` must be positive, got {out_size}.")
    extra_features = len(passthrough) + int(include_constant)
    periodic_size = out_size - extra_features
    if periodic_size <= 0 or periodic_size % 2 != 0:
        raise ValueError(
            "`out_size - len(passthrough) - int(include_constant)` must be "
            f"a positive even number, got {periodic_size}."
        )
    return periodic_size


def _random_wavevectors(
    *,
    in_dim: int,
    feature_size: int,
    mu: ArrayLike | Sequence[ArrayLike],
    sigma: ArrayLike | Sequence[ArrayLike],
    key: Key[Array, ""],
) -> Array:
    mu_in = _tuple(mu)
    if mu_in is None or len(mu_in) == 0:
        raise ValueError("`mu` must contain at least one value.")
    sigma_in = _tuple(sigma)
    if sigma_in is None or len(sigma_in) == 0:
        raise ValueError("`sigma` must contain at least one value.")
    num_mu = len(mu_in)
    num_sigma = len(sigma_in)
    means = mu_in * num_sigma
    standard_deviations = sigma_in * num_mu
    num_blocks = num_mu * num_sigma
    if feature_size % (2 * num_blocks) != 0:
        divisor = 2 * num_blocks
        raise ValueError(
            "`feature_size` must be divisible by "
            f"`2 * (len(mu) * len(sigma)) = {divisor}`, got {feature_size}."
        )
    rows_per_block = feature_size // (2 * num_blocks)
    keys = jr.split(key, num_blocks)
    blocks = tuple(
        jr.normal(subkey, (rows_per_block, in_dim)) * sigma_ + mu_
        for subkey, mu_, sigma_ in zip(
            keys,
            means,
            standard_deviations,
            strict=True,
        )
    )
    return jnp.concatenate(blocks, axis=0)


class _AbstractFourierFeatureEmbeddings(_AbstractBaseModel):
    in_size: int | tuple[int, ...] | Literal["scalar"] = eqx.field(kw_only=True)
    out_size: int = eqx.field(kw_only=True)

    embedding_matrix: Array = eqx.field(kw_only=True)
    phases: Array = eqx.field(kw_only=True)
    passthrough: tuple[int, ...] = eqx.field(kw_only=True)
    include_constant: bool = eqx.field(kw_only=True)
    trainable: bool = eqx.field(kw_only=True)

    def _initialize(
        self,
        *,
        in_size: SizeLike,
        embedding_matrix: ArrayLike,
        phases: ArrayLike | None,
        passthrough: Sequence[int],
        include_constant: bool,
        trainable: bool,
    ) -> None:
        in_size_c = _canonical_size(in_size)
        in_dim = _get_size(in_size_c)
        matrix = _as_wavevectors(
            embedding_matrix,
            in_dim,
            name="embedding_matrix",
        )
        passthrough_indices = _canonical_passthrough(passthrough, in_dim)

        self.in_size = in_size_c
        self.out_size = (
            2 * int(matrix.shape[0]) + len(passthrough_indices) + int(include_constant)
        )
        self.embedding_matrix = matrix
        self.phases = _as_phases(phases, int(matrix.shape[0]))
        self.passthrough = passthrough_indices
        self.include_constant = bool(include_constant)
        self.trainable = bool(trainable)

    def __call__(
        self,
        x: Array,
        /,
        *,
        key: EvalKey = DOC_KEY0,
    ) -> Array:
        del key
        x_arr = jnp.asarray(x)
        in_shape = _get_value_shape(self.in_size)
        if self.in_size == "scalar":
            if x_arr.shape == ():
                x_vec = x_arr.reshape((1,))
            elif x_arr.shape == (1,):
                x_vec = x_arr
            else:
                raise ValueError(
                    "`x` must have scalar shape () or (1,) for "
                    f"in_size='scalar', got {x_arr.shape}."
                )
        else:
            if x_arr.shape != in_shape:
                raise ValueError(f"`x` must have shape {in_shape}, got {x_arr.shape}.")
            x_vec = x_arr.reshape((_get_size(self.in_size),))

        embedding_matrix = (
            self.embedding_matrix
            if self.trainable
            else jax.lax.stop_gradient(self.embedding_matrix)
        )
        phases = jax.lax.stop_gradient(self.phases)
        projected = embedding_matrix @ x_vec + phases
        periodic = jnp.concatenate((jnp.cos(projected), jnp.sin(projected)))

        features = [periodic]
        if self.passthrough:
            features.append(x_vec[jnp.asarray(self.passthrough)])
        if self.include_constant:
            features.append(jnp.ones((1,), dtype=periodic.dtype))
        return periodic if len(features) == 1 else jnp.concatenate(features)


class ExplicitFourierFeatureEmbeddings(_AbstractFourierFeatureEmbeddings):
    r"""Fourier features defined by explicit, fixed wavevectors.

    For a matrix of row wavevectors $B$ and phases $p$, this returns
    $[\cos(Bx+p),\sin(Bx+p)]$, followed by selected raw coordinates and an
    optional constant. Wavevectors use angular-frequency units.
    """

    def __init__(
        self,
        *,
        in_size: SizeLike,
        wavevectors: ArrayLike,
        phases: ArrayLike | None = None,
        passthrough: Sequence[int] = (),
        include_constant: bool = False,
    ):
        r"""**Arguments:**

        - `in_size`: Input value size. The input is flattened to a vector.
        - `wavevectors`: Scalar, vector, or matrix of angular wavevectors.
        - `phases`: Optional scalar or one phase per wavevector.
        - `passthrough`: Flattened input coordinates appended without encoding.
        - `include_constant`: Append a constant-one feature.
        """
        self._initialize(
            in_size=in_size,
            embedding_matrix=wavevectors,
            phases=phases,
            passthrough=passthrough,
            include_constant=include_constant,
            trainable=False,
        )

    @classmethod
    def from_periodic_modes(
        cls,
        *,
        in_size: SizeLike,
        coordinate: int,
        period: float,
        modes: Sequence[int],
        phases: ArrayLike | None = None,
        passthrough: Sequence[int] = (),
        include_constant: bool = False,
    ) -> ExplicitFourierFeatureEmbeddings:
        """Construct positive integer harmonics for one periodic coordinate."""
        in_dim = _get_size(_canonical_size(in_size))
        coordinate = int(coordinate)
        if coordinate < 0 or coordinate >= in_dim:
            raise ValueError(f"`coordinate` must lie in [0, {in_dim}), got {coordinate}.")
        period = float(period)
        if period <= 0.0:
            raise ValueError(f"`period` must be positive, got {period}.")
        mode_values = tuple(int(mode) for mode in modes)
        if not mode_values or any(mode <= 0 for mode in mode_values):
            raise ValueError(
                f"`modes` must contain positive integers, got {mode_values}."
            )
        if len(set(mode_values)) != len(mode_values):
            raise ValueError(f"`modes` must be unique, got {mode_values}.")

        wavevectors = jnp.zeros((len(mode_values), in_dim), dtype=float)
        angular_frequencies = (
            2.0 * jnp.pi * jnp.asarray(mode_values, dtype=float) / period
        )
        wavevectors = wavevectors.at[:, coordinate].set(angular_frequencies)
        return cls(
            in_size=in_size,
            wavevectors=wavevectors,
            phases=phases,
            passthrough=passthrough,
            include_constant=include_constant,
        )


class MultiscaleFourierFeatureEmbeddings(_AbstractFourierFeatureEmbeddings):
    r"""Deterministic multiscale Fourier features.

    Each positive scale multiplies every base wavevector. When
    `base_wavevectors=None`, the coordinate-axis unit vectors are used.
    """

    scales: Array
    base_wavevectors: Array

    def __init__(
        self,
        *,
        in_size: SizeLike,
        scales: ArrayLike | Sequence[float] = (1.0, 2.0, 4.0, 8.0),
        base_wavevectors: ArrayLike | None = None,
        phases: ArrayLike | None = None,
        passthrough: Sequence[int] = (),
        include_constant: bool = False,
    ):
        r"""**Arguments:**

        - `in_size`: Input value size. The input is flattened to a vector.
        - `scales`: Positive angular-frequency multipliers.
        - `base_wavevectors`: Wavevectors multiplied by every scale. Defaults to
          the coordinate-axis unit vectors.
        - `phases`: Optional scalar or one phase per resulting wavevector.
        - `passthrough`: Flattened input coordinates appended without encoding.
        - `include_constant`: Append a constant-one feature.
        """
        in_dim = _get_size(_canonical_size(in_size))
        scale_values = jnp.asarray(scales, dtype=float)
        if scale_values.ndim == 0:
            scale_values = scale_values.reshape((1,))
        if scale_values.ndim != 1 or scale_values.shape[0] == 0:
            raise ValueError(
                f"`scales` must be a nonempty one-dimensional array, got {scale_values.shape}."
            )
        if bool(jnp.any(~jnp.isfinite(scale_values))) or bool(
            jnp.any(scale_values <= 0.0)
        ):
            raise ValueError("`scales` must contain finite positive values.")

        base = (
            jnp.eye(in_dim, dtype=float)
            if base_wavevectors is None
            else _as_wavevectors(
                base_wavevectors,
                in_dim,
                name="base_wavevectors",
            )
        )
        if bool(jnp.any(jnp.linalg.norm(base, axis=1) == 0.0)):
            raise ValueError("`base_wavevectors` must not contain a zero wavevector.")
        wavevectors = (scale_values[:, None, None] * base[None, :, :]).reshape(
            (-1, in_dim)
        )

        self.scales = scale_values
        self.base_wavevectors = base
        self._initialize(
            in_size=in_size,
            embedding_matrix=wavevectors,
            phases=phases,
            passthrough=passthrough,
            include_constant=include_constant,
            trainable=False,
        )


class HybridFourierFeatureEmbeddings(_AbstractFourierFeatureEmbeddings):
    r"""Fixed deterministic wavevectors with a Gaussian random spectral tail."""

    deterministic_wavevector_count: int
    random_feature_size: int

    def __init__(
        self,
        *,
        in_size: SizeLike,
        deterministic_wavevectors: ArrayLike,
        random_out_size: int = 32,
        deterministic_phases: ArrayLike | None = None,
        random_mu: ArrayLike | Sequence[ArrayLike] = 0.0,
        random_sigma: ArrayLike | Sequence[ArrayLike] = 1.0,
        passthrough: Sequence[int] = (),
        include_constant: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""**Arguments:**

        - `in_size`: Input value size. The input is flattened to a vector.
        - `deterministic_wavevectors`: Fixed angular wavevectors guaranteed to
          appear in the embedding.
        - `random_out_size`: Number of cosine/sine features in the random tail.
        - `deterministic_phases`: Optional phases for deterministic wavevectors.
        - `random_mu`: Gaussian means for random spectral blocks.
        - `random_sigma`: Gaussian standard deviations for random spectral blocks.
        - `passthrough`: Flattened input coordinates appended without encoding.
        - `include_constant`: Append a constant-one feature.
        - `key`: PRNG key for the random spectral tail.
        """
        in_dim = _get_size(_canonical_size(in_size))
        deterministic = _as_wavevectors(
            deterministic_wavevectors,
            in_dim,
            name="deterministic_wavevectors",
        )
        random_out_size = int(random_out_size)
        if random_out_size <= 0 or random_out_size % 2 != 0:
            raise ValueError(
                f"`random_out_size` must be a positive even number, got {random_out_size}."
            )
        random = _random_wavevectors(
            in_dim=in_dim,
            feature_size=random_out_size,
            mu=random_mu,
            sigma=random_sigma,
            key=key,
        )
        deterministic_phase_values = _as_phases(
            deterministic_phases,
            int(deterministic.shape[0]),
        )
        phases = jnp.concatenate(
            (
                deterministic_phase_values,
                jnp.zeros((random.shape[0],), dtype=float),
            )
        )

        self.deterministic_wavevector_count = int(deterministic.shape[0])
        self.random_feature_size = random_out_size
        self._initialize(
            in_size=in_size,
            embedding_matrix=jnp.concatenate((deterministic, random), axis=0),
            phases=phases,
            passthrough=passthrough,
            include_constant=include_constant,
            trainable=False,
        )


class RandomFourierFeatureEmbeddings(_AbstractFourierFeatureEmbeddings):
    r"""Gaussian random Fourier feature embedding.

    Samples one or more Gaussian wavevector blocks and returns their cosine and
    sine features. Multiple `mu` and `sigma` values form their Cartesian product.
    """

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: int = 32,
        mu: ArrayLike | Sequence[ArrayLike] = 0.0,
        sigma: ArrayLike | Sequence[ArrayLike] = 1.0,
        phases: ArrayLike | None = None,
        passthrough: Sequence[int] = (),
        include_constant: bool = False,
        trainable: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""**Arguments:**

        - `in_size`: Input value size. The input is flattened to a vector.
        - `out_size`: Total output size, including passthrough and constant features.
        - `mu`: Gaussian means (scalar or sequence for multiblock sampling).
        - `sigma`: Gaussian standard deviations (scalar or sequence for multiblock
          sampling).
        - `phases`: Optional scalar or one phase per sampled wavevector.
        - `passthrough`: Flattened input coordinates appended without encoding.
        - `include_constant`: Append a constant-one feature.
        - `trainable`: Learn sampled wavevectors instead of stopping their gradients.
        - `key`: PRNG key.
        """
        in_size_c = _canonical_size(in_size)
        in_dim = _get_size(in_size_c)
        passthrough_indices = _canonical_passthrough(passthrough, in_dim)
        periodic_size = _periodic_feature_size(
            out_size,
            passthrough_indices,
            include_constant,
        )
        embedding_matrix = _random_wavevectors(
            in_dim=in_dim,
            feature_size=periodic_size,
            mu=mu,
            sigma=sigma,
            key=key,
        )
        self._initialize(
            in_size=in_size_c,
            embedding_matrix=embedding_matrix,
            phases=phases,
            passthrough=passthrough_indices,
            include_constant=include_constant,
            trainable=trainable,
        )


class TrainableFourierFeatureEmbeddings(_AbstractFourierFeatureEmbeddings):
    r"""Fourier embedding with unrestricted trainable wavevectors.

    The wavevectors may be initialized explicitly or sampled from Gaussian
    blocks. Their gradients are not stopped during evaluation.
    """

    def __init__(
        self,
        *,
        in_size: SizeLike,
        out_size: int = 32,
        initial_wavevectors: ArrayLike | None = None,
        mu: ArrayLike | Sequence[ArrayLike] = 0.0,
        sigma: ArrayLike | Sequence[ArrayLike] = 1.0,
        phases: ArrayLike | None = None,
        passthrough: Sequence[int] = (),
        include_constant: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""**Arguments:**

        - `in_size`: Input value size. The input is flattened to a vector.
        - `out_size`: Total output size, including passthrough and constant features.
        - `initial_wavevectors`: Optional explicit initialization. Its row count must
          match the periodic part of `out_size`.
        - `mu`: Gaussian means used when `initial_wavevectors` is omitted.
        - `sigma`: Gaussian standard deviations used when `initial_wavevectors` is
          omitted.
        - `phases`: Optional fixed scalar or one phase per wavevector.
        - `passthrough`: Flattened input coordinates appended without encoding.
        - `include_constant`: Append a constant-one feature.
        - `key`: PRNG key used for Gaussian initialization.
        """
        in_size_c = _canonical_size(in_size)
        in_dim = _get_size(in_size_c)
        passthrough_indices = _canonical_passthrough(passthrough, in_dim)
        periodic_size = _periodic_feature_size(
            out_size,
            passthrough_indices,
            include_constant,
        )
        if initial_wavevectors is None:
            embedding_matrix = _random_wavevectors(
                in_dim=in_dim,
                feature_size=periodic_size,
                mu=mu,
                sigma=sigma,
                key=key,
            )
        else:
            embedding_matrix = _as_wavevectors(
                initial_wavevectors,
                in_dim,
                name="initial_wavevectors",
            )
            expected_rows = periodic_size // 2
            if embedding_matrix.shape[0] != expected_rows:
                raise ValueError(
                    "`initial_wavevectors` row count must equal "
                    f"(out_size - extra features) / 2 = {expected_rows}, "
                    f"got {embedding_matrix.shape[0]}."
                )

        self._initialize(
            in_size=in_size_c,
            embedding_matrix=embedding_matrix,
            phases=phases,
            passthrough=passthrough_indices,
            include_constant=include_constant,
            trainable=True,
        )


__all__ = [
    "ExplicitFourierFeatureEmbeddings",
    "HybridFourierFeatureEmbeddings",
    "MultiscaleFourierFeatureEmbeddings",
    "RandomFourierFeatureEmbeddings",
    "TrainableFourierFeatureEmbeddings",
]
