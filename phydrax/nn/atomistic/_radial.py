#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact MACE-family radial realization: Bessel basis, cutoff, Agnesi, radial MLP.

The admitted family is closed: a Bessel basis evaluated at the (optionally
Agnesi-transformed) distance, multiplied by the polynomial cutoff envelope of the
untransformed distance, followed by a bias-free radial MLP with normalized-SiLU
hidden activations and an optional ``tanh(x**2)`` output postprocessing. Other
bases, transforms, biases or activations are not approximated by this owner.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from ..._fingerprint import canonical_fingerprint
from ..._model import register_artifact_value
from ..._polynomial._orthogonal import standard_normal_hermite_rule_data
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, ParameterOwner
from ..._validation import finite_real_scalar, positive_finite_float, positive_integer
from ...typing import as_host_array, Dim, Float, HostFloat64, parse, PRNGKey, Scalar
from ._fixed_binding import require_identical


RadialPostprocess: TypeAlias = Literal["none", "tanh-square"]

# Gauss-Hermite nodes for the native normalized-SiLU second moment. SiLU squared
# is smooth with polynomial growth, so this rule resolves the expectation to
# float64 rounding; the constant is a declared native value, distinct from any
# provider's sampled normalization constant.
_SILU_MOMENT_NODES = 160


class RadialBasisDim(Dim):
    """Bessel radial basis functions."""


class RadialSpeciesDim(Dim):
    """Model species in declared order."""


def _float_dtype(dtype: DTypeLike, /) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("Radial realizations require float32 or float64 dtype.")
    return resolved


def normalized_silu_scale() -> float:
    """Return ``1 / sqrt(E[silu(z)**2])`` for standard-normal ``z`` by quadrature."""
    rule = standard_normal_hermite_rule_data(_SILU_MOMENT_NODES, dtype=jnp.float64)
    nodes = np.asarray(rule.nodes, dtype=np.float64)
    weights = np.asarray(rule.weights, dtype=np.float64)
    silu = nodes / (1.0 + np.exp(-nodes))
    return float(1.0 / np.sqrt(np.sum(weights * silu * silu)))


@final
class BesselRadialBasis(StrictModule, NonTrainableState):
    """Fixed-frequency Bessel basis ``prefactor * sin(w_n r) / r`` on ``r > 0``."""

    __strict_contract__ = True

    frequencies: Float[RadialBasisDim]
    prefactor: Float[Scalar]
    cutoff: float = eqx.field(static=True)
    basis_count: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        frequencies: ArrayLike,
        prefactor: float,
        cutoff: float,
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        dtype_ = _float_dtype(dtype)
        values = as_host_array(
            frequencies, HostFloat64[RadialBasisDim], "frequencies"
        ).astype(dtype_)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Bessel frequencies must be finite and positive.")
        factor = float(
            np.asarray(positive_finite_float(prefactor, "prefactor"), dtype=dtype_)
        )
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("The Bessel prefactor must be finite and positive in dtype.")
        cutoff_ = positive_finite_float(cutoff, "cutoff")
        self.frequencies = jnp.asarray(values)
        self.prefactor = jnp.asarray(np.asarray(factor, dtype=dtype_))
        self.cutoff = cutoff_
        self.basis_count = values.shape[0]
        # The identity records the realized (dtype-rounded) prefactor, exactly as
        # it records the realized frequencies, so re-admitting the stored arrays
        # reproduces it.
        self.basis_id = canonical_fingerprint(
            {
                "kind": "bessel-radial-basis",
                "cutoff": cutoff_,
                "frequencies": [float(value) for value in values],
                "prefactor": factor,
                "dtype": dtype_.name,
            }
        )

    @classmethod
    def native(
        cls,
        cutoff: float,
        basis_count: int,
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ) -> BesselRadialBasis:
        """Construct ``w_n = n pi / cutoff`` and ``prefactor = sqrt(2 / cutoff)``."""
        cutoff_ = positive_finite_float(cutoff, "cutoff")
        count = positive_integer(basis_count, "basis_count")
        frequencies = (math.pi / cutoff_) * np.arange(1, count + 1, dtype=np.float64)
        return cls(frequencies, math.sqrt(2.0 / cutoff_), cutoff_, dtype=dtype)

    def __call__(self, distance: Array, /) -> Array:
        radius = distance[..., None]
        frequencies = self.frequencies.reshape(
            (1,) * distance.ndim + self.frequencies.shape
        )
        return self.prefactor * jnp.sin(frequencies * radius) / radius


@final
class PolynomialCutoff(StrictModule, NonTrainableState):
    """Polynomial envelope with value, slope and curvature zero at the cutoff.

    ``f(u) = 1 - (p+1)(p+2)/2 u^p + p(p+2) u^(p+1) - p(p+1)/2 u^(p+2)`` for
    ``u = r / cutoff < 1`` and exactly zero beyond. The join is C2; the third
    derivative jumps at the cutoff.
    """

    cutoff: float = eqx.field(static=True)
    power: int = eqx.field(static=True)

    def __init__(self, cutoff: float, power: int, /) -> None:
        self.cutoff = positive_finite_float(cutoff, "cutoff")
        self.power = positive_integer(power, "power")

    @property
    def regularity_order(self) -> int:
        return 2

    def __call__(self, distance: Array, /) -> Array:
        p = float(self.power)
        ratio = distance / self.cutoff
        ratio_p = ratio**self.power
        envelope = (
            1.0
            - 0.5 * (p + 1.0) * (p + 2.0) * ratio_p
            + p * (p + 2.0) * ratio_p * ratio
            - 0.5 * p * (p + 1.0) * ratio_p * ratio * ratio
        )
        return jnp.where(distance < self.cutoff, envelope, jnp.zeros_like(envelope))


@final
class AgnesiTransform(StrictModule, NonTrainableState):
    """Species-pair Agnesi distance transform with explicit per-species radii.

    ``r0 = (R[sender] + R[receiver]) / 2`` and
    ``t = 1 / (1 + a (r/r0)^q / (1 + (r/r0)^(q-p)))``. Radii are ordered by the
    model's declared species, and the formula is used as written; no pair
    symmetry is assumed by downstream table preparation.
    """

    __strict_contract__ = True

    covalent_radii: Float[RadialSpeciesDim]
    atomic_numbers: tuple[int, ...] = eqx.field(static=True)
    a: float = eqx.field(static=True)
    q: float = eqx.field(static=True)
    p: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        covalent_radii: ArrayLike,
        /,
        *,
        atomic_numbers: Sequence[int],
        a: float,
        q: float,
        p: float,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        dtype_ = _float_dtype(dtype)
        radii = as_host_array(
            covalent_radii, HostFloat64[RadialSpeciesDim], "covalent_radii"
        )
        numbers = tuple(
            positive_integer(value, "atomic_numbers") for value in atomic_numbers
        )
        if len(numbers) != radii.shape[0]:
            raise ValueError("atomic_numbers must match covalent_radii.")
        if len(set(numbers)) != len(numbers):
            raise ValueError("atomic_numbers must be distinct.")
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("Agnesi covalent radii must be finite and positive.")
        a_ = finite_real_scalar(a, "a")
        q_ = finite_real_scalar(q, "q")
        p_ = finite_real_scalar(p, "p")
        self.covalent_radii = jnp.asarray(radii.astype(dtype_))
        self.atomic_numbers = numbers
        self.a = a_
        self.q = q_
        self.p = p_
        self.transform_id = canonical_fingerprint(
            {
                "kind": "agnesi-distance-transform",
                "atomic_numbers": list(numbers),
                "covalent_radii": [float(value) for value in radii.astype(dtype_)],
                "a": a_,
                "q": q_,
                "p": p_,
                "dtype": dtype_.name,
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.atomic_numbers)

    def __call__(
        self, distance: Array, sender_species: Array, receiver_species: Array, /
    ) -> Array:
        r0 = 0.5 * (
            self.covalent_radii[sender_species] + self.covalent_radii[receiver_species]
        )
        ratio = distance / r0
        return 1.0 / (1.0 + self.a * ratio**self.q / (1.0 + ratio ** (self.q - self.p)))


@final
class RadialEmbedding(StrictModule):
    """Edge radial features ``bessel(transform(r)) * cutoff(r)``."""

    basis: BesselRadialBasis
    cutoff: PolynomialCutoff
    transform: AgnesiTransform | None
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: BesselRadialBasis,
        cutoff: PolynomialCutoff,
        /,
        *,
        transform: AgnesiTransform | None = None,
    ) -> None:
        if not isinstance(basis, BesselRadialBasis):
            raise TypeError("basis must be a BesselRadialBasis.")
        if not isinstance(cutoff, PolynomialCutoff):
            raise TypeError("cutoff must be a PolynomialCutoff.")
        if transform is not None and not isinstance(transform, AgnesiTransform):
            raise TypeError("transform must be an AgnesiTransform or None.")
        if basis.cutoff != cutoff.cutoff:
            raise ValueError("The Bessel basis and cutoff must share one cutoff radius.")
        if (
            transform is not None
            and transform.covalent_radii.dtype != basis.frequencies.dtype
        ):
            raise TypeError("The Agnesi transform and Bessel basis must share one dtype.")
        self.basis = basis
        self.cutoff = cutoff
        self.transform = transform
        self.embedding_id = canonical_fingerprint(
            {
                "kind": "radial-embedding",
                "basis": basis.basis_id,
                "cutoff": cutoff.cutoff,
                "cutoff_power": cutoff.power,
                "transform": None if transform is None else transform.transform_id,
            }
        )

    @property
    def radius(self) -> float:
        return self.cutoff.cutoff

    @property
    def basis_count(self) -> int:
        return self.basis.basis_count

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.basis.frequencies.dtype)

    @property
    def pair_dependent(self) -> bool:
        """Whether features depend on the ordered sender/receiver species."""
        return self.transform is not None

    def validate(self) -> None:
        """Re-admit every fixed field from the arrays that execute.

        This is a host boundary over concrete arrays. Bessel frequencies and
        prefactor, the polynomial cutoff and the Agnesi radii and constants are
        rebuilt through their constructors from the stored values (float32
        realizations stay authoritative), and every recomputed identity and
        static field must equal the recorded one; recorded labels are never
        trusted.
        """
        basis = self.basis
        dtype = _float_dtype(basis.frequencies.dtype)
        prefactor = np.asarray(jax.device_get(basis.prefactor))
        if prefactor.shape != () or prefactor.dtype != dtype:
            raise ValueError("The Bessel prefactor must be a scalar of the basis dtype.")
        transform = self.transform
        rebuilt = RadialEmbedding(
            BesselRadialBasis(
                np.asarray(jax.device_get(basis.frequencies), dtype=np.float64),
                float(prefactor),
                basis.cutoff,
                dtype=dtype,
            ),
            PolynomialCutoff(self.cutoff.cutoff, self.cutoff.power),
            transform=None
            if transform is None
            else AgnesiTransform(
                np.asarray(jax.device_get(transform.covalent_radii), dtype=np.float64),
                atomic_numbers=transform.atomic_numbers,
                a=transform.a,
                q=transform.q,
                p=transform.p,
                dtype=dtype,
            ),
        )
        require_identical(self, rebuilt, "Radial embedding", ValueError)

    def unmasked(
        self, distance: Array, sender_species: Array, receiver_species: Array, /
    ) -> Array:
        """Evaluate on the admitted domain ``r > 0`` without padding semantics."""
        transformed = (
            distance
            if self.transform is None
            else self.transform(distance, sender_species, receiver_species)
        )
        return self.basis(transformed) * self.cutoff(distance)[..., None]

    def __call__(
        self,
        distance: ArrayLike,
        sender_species: ArrayLike,
        receiver_species: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
    ) -> Array:
        radius = jnp.asarray(distance, dtype=self.dtype)
        sender = jnp.asarray(sender_species, dtype=jnp.int32)
        receiver = jnp.asarray(receiver_species, dtype=jnp.int32)
        if sender.shape != radius.shape or receiver.shape != radius.shape:
            raise ValueError("Species indices must match the distance shape.")
        if valid is None:
            return self.unmasked(radius, sender, receiver)
        mask = jnp.asarray(valid, dtype=jnp.bool_)
        if mask.shape != radius.shape:
            raise ValueError("valid must match the distance shape.")
        # Padded lanes evaluate at an admitted interior radius so neither the
        # primal nor its derivatives see the singular r = 0 point.
        sentinel = jnp.asarray(0.5 * self.radius, dtype=self.dtype)
        safe = jnp.where(mask, radius, sentinel)
        zero = jnp.zeros_like(sender)
        features = self.unmasked(
            safe, jnp.where(mask, sender, zero), jnp.where(mask, receiver, zero)
        )
        return jnp.where(mask[..., None], features, jnp.zeros_like(features))


@final
class RadialMLP(StrictModule, ParameterOwner):
    """Bias-free radial MLP in the original unit-variance layer parameterization.

    Layer ``l`` applies ``x @ (W_l / sqrt(h_in))``; hidden layers apply
    ``activation_scale * silu``; the final layer is linear, followed by the
    declared postprocessing. ``weights`` are PARAMETER leaves in source layout.
    """

    weights: tuple[Array, ...]
    widths: tuple[int, ...] = eqx.field(static=True)
    activation_scale: float = eqx.field(static=True)
    postprocess: RadialPostprocess = eqx.field(static=True)

    def __init__(
        self,
        weights: Sequence[ArrayLike],
        /,
        *,
        activation_scale: float,
        postprocess: RadialPostprocess = "none",
    ) -> None:
        postprocess_ = parse(postprocess, RadialPostprocess, "postprocess")
        scale = positive_finite_float(activation_scale, "activation_scale")
        arrays = tuple(jnp.asarray(weight) for weight in weights)
        if not arrays:
            raise ValueError("A radial MLP requires at least one layer.")
        dtype = arrays[0].dtype
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("Radial MLP weights must be float32 or float64.")
        if any(weight.ndim != 2 or weight.dtype != dtype for weight in arrays):
            raise ValueError("Radial MLP weights must be rank-two with one dtype.")
        widths = [arrays[0].shape[0]]
        for index, weight in enumerate(arrays):
            if weight.shape[0] != widths[-1] or min(weight.shape) <= 0:
                raise ValueError(f"Radial MLP layer {index} has inconsistent widths.")
            widths.append(weight.shape[1])
        self.weights = arrays
        self.widths = tuple(widths)
        self.activation_scale = scale
        self.postprocess = postprocess_

    @classmethod
    def initialize(
        cls,
        widths: Sequence[int],
        /,
        *,
        key: PRNGKey,
        dtype: DTypeLike = jnp.float64,
        activation_scale: float | None = None,
        postprocess: RadialPostprocess = "none",
    ) -> RadialMLP:
        """Draw unit-normal weights, the original radial-network initialization."""
        sizes = tuple(positive_integer(value, "widths") for value in widths)
        if len(sizes) < 2:
            raise ValueError("A radial MLP requires input and output widths.")
        dtype_ = _float_dtype(dtype)
        keys = jr.split(key, len(sizes) - 1)
        weights = tuple(
            jr.normal(layer_key, (fan_in, fan_out), dtype=dtype_)
            for layer_key, fan_in, fan_out in zip(keys, sizes[:-1], sizes[1:])
        )
        scale = normalized_silu_scale() if activation_scale is None else activation_scale
        return cls(weights, activation_scale=scale, postprocess=postprocess)

    @property
    def input_width(self) -> int:
        return self.widths[0]

    @property
    def output_width(self) -> int:
        return self.widths[-1]

    @property
    def embedding_width(self) -> int:
        """Width entering the final linear layer."""
        return self.widths[-2]

    @property
    def final_projection(self) -> Array:
        """Effective final linear map ``W_last / sqrt(h_in)``."""
        weight = self.weights[-1]
        return weight / jnp.sqrt(jnp.asarray(weight.shape[0], dtype=weight.dtype))

    def _layer(self, value: Array, weight: Array, /) -> Array:
        return value @ (
            weight / jnp.sqrt(jnp.asarray(weight.shape[0], dtype=weight.dtype))
        )

    def penultimate(self, features: Array, /) -> Array:
        """Return the hidden state entering the final linear layer."""
        value = features
        for weight in self.weights[:-1]:
            value = self.activation_scale * jax.nn.silu(self._layer(value, weight))
        return value

    def project(self, embedding: Array, /) -> Array:
        """Apply the final linear layer and postprocessing to a penultimate state."""
        output = embedding @ self.final_projection
        match self.postprocess:
            case "none":
                return output
            case "tanh-square":
                return jnp.tanh(output * output)
            case unreachable:
                assert_never(unreachable)

    def __call__(self, features: Array, /) -> Array:
        if features.shape[-1:] != (self.input_width,):
            raise ValueError(f"Radial MLP features must end in width {self.input_width}.")
        return self.project(self.penultimate(features))


for _artifact in (
    AgnesiTransform,
    BesselRadialBasis,
    PolynomialCutoff,
    RadialEmbedding,
    RadialMLP,
):
    register_artifact_value(f"phydrax.nn.atomistic:{_artifact.__name__}", _artifact)
del _artifact


__all__ = [
    "AgnesiTransform",
    "BesselRadialBasis",
    "normalized_silu_scale",
    "PolynomialCutoff",
    "RadialEmbedding",
    "RadialMLP",
    "RadialPostprocess",
]
