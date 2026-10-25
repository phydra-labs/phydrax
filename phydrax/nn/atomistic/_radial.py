#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact MACE-family radial realizations: bases, cutoff, transforms, radial MLPs.

The admitted family is closed and source-exact. A Bessel, Gaussian or Chebyshev
basis is evaluated at the (optionally Agnesi- or Soft-transformed) distance and
multiplied by the polynomial cutoff envelope of the untransformed distance. Edge
weights come from either the bias-free unit-variance radial MLP with
normalized-SiLU hidden activations (and an optional ``tanh(x**2)`` output), or the
biased ``Linear -> LayerNorm -> SiLU`` radial MLP of the nonlinear interaction
family. Other bases, transforms, biases or activations are not approximated by
this owner.
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
from ..._model._fixed_binding import require_identical
from ..._polynomial._orthogonal import standard_normal_hermite_host_arrays
from ..._strict import StrictModule
from ..._trainable import NonTrainableState, ParameterOwner
from ..._validation import finite_real_scalar, positive_finite_float, positive_integer
from ...typing import as_host_array, Dim, Float, HostFloat64, parse, PRNGKey, Scalar


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
    nodes, weights = standard_normal_hermite_host_arrays(
        _SILU_MOMENT_NODES, dtype=np.float64
    )
    silu = nodes / (1.0 + np.exp(-nodes))
    return float(1.0 / np.sqrt(np.sum(weights * silu * silu)))


def normalized_sigmoid_scale() -> float:
    """Return ``1 / sqrt(E[sigmoid(z)**2])`` for standard-normal ``z`` by quadrature."""
    nodes, weights = standard_normal_hermite_host_arrays(
        _SILU_MOMENT_NODES, dtype=np.float64
    )
    sigmoid = 1.0 / (1.0 + np.exp(-nodes))
    return float(1.0 / np.sqrt(np.sum(weights * sigmoid * sigmoid)))


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
class GaussianRadialBasis(StrictModule, NonTrainableState):
    """Gaussian basis ``exp(c (r - mu_n)**2)`` with source-declared centers.

    The source family places ``count >= 2`` centers ``mu_n`` on
    ``linspace(0, cutoff, count)`` and uses ``c = -1 / (2 (cutoff / (count - 1))**2)``
    computed in float64 and applied in the compute dtype. Centers are the stored
    (possibly float32-rounded) realization; the coefficient is re-derived from the
    declaration and never trusted from a label.
    """

    __strict_contract__ = True

    centers: Float[RadialBasisDim]
    coefficient: float = eqx.field(static=True)
    cutoff: float = eqx.field(static=True)
    basis_count: int = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self,
        centers: ArrayLike,
        cutoff: float,
        /,
        *,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        dtype_ = _float_dtype(dtype)
        values = as_host_array(centers, HostFloat64[RadialBasisDim], "centers").astype(
            dtype_
        )
        cutoff_ = positive_finite_float(cutoff, "cutoff")
        count = values.shape[0]
        if count < 2:
            raise ValueError("A Gaussian radial basis declares at least two centers.")
        if not np.all(np.isfinite(values)):
            raise ValueError("Gaussian centers must be finite.")
        declared = np.linspace(0.0, cutoff_, count, dtype=np.float64).astype(dtype_)
        if not np.allclose(
            values, declared, rtol=0.0, atol=8.0 * float(np.finfo(dtype_).eps) * cutoff_
        ):
            raise ValueError("Gaussian centers must be the declared uniform grid.")
        coefficient = -0.5 / (cutoff_ / (count - 1)) ** 2
        self.centers = jnp.asarray(values)
        self.coefficient = coefficient
        self.cutoff = cutoff_
        self.basis_count = count
        self.basis_id = canonical_fingerprint(
            {
                "kind": "gaussian-radial-basis",
                "cutoff": cutoff_,
                "centers": [float(value) for value in values],
                "coefficient": coefficient,
                "dtype": dtype_.name,
            }
        )

    @classmethod
    def native(
        cls, cutoff: float, basis_count: int, /, *, dtype: DTypeLike = jnp.float64
    ) -> GaussianRadialBasis:
        cutoff_ = positive_finite_float(cutoff, "cutoff")
        count = positive_integer(basis_count, "basis_count")
        return cls(
            np.linspace(0.0, cutoff_, count, dtype=np.float64), cutoff_, dtype=dtype
        )

    def rebuilt(self) -> GaussianRadialBasis:
        """Re-admit this basis from the stored centers."""
        return GaussianRadialBasis(
            np.asarray(jax.device_get(self.centers), dtype=np.float64),
            self.cutoff,
            dtype=self.centers.dtype,
        )

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.centers.dtype)

    def __call__(self, distance: Array, /) -> Array:
        shifted = distance[..., None] - self.centers.reshape(
            (1,) * distance.ndim + self.centers.shape
        )
        coefficient = jnp.asarray(self.coefficient, dtype=shifted.dtype)
        return jnp.exp(coefficient * shifted * shifted)


@final
class ChebyshevRadialBasis(StrictModule, NonTrainableState):
    """First-kind Chebyshev polynomials ``T_n(r)``, ``n = 1..count``, of the distance.

    The source evaluates the polynomials at the raw (or transformed) distance in
    the model length unit, not at a cutoff-normalized argument; values grow as
    ``|r|**n`` outside ``[-1, 1]``. Native evaluation uses the exact three-term
    recurrence, a polynomial identical to the source's trigonometric branch on
    ``|r| < 1``.
    """

    cutoff: float = eqx.field(static=True)
    basis_count: int = eqx.field(static=True)
    precision: str = eqx.field(static=True)
    basis_id: str = eqx.field(static=True)

    def __init__(
        self, cutoff: float, basis_count: int, /, *, dtype: DTypeLike = jnp.float64
    ) -> None:
        dtype_ = _float_dtype(dtype)
        cutoff_ = positive_finite_float(cutoff, "cutoff")
        count = positive_integer(basis_count, "basis_count")
        self.cutoff = cutoff_
        self.basis_count = count
        self.precision = dtype_.name
        self.basis_id = canonical_fingerprint(
            {
                "kind": "chebyshev-radial-basis",
                "cutoff": cutoff_,
                "count": count,
                "dtype": dtype_.name,
            }
        )

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.precision)

    def rebuilt(self) -> ChebyshevRadialBasis:
        return ChebyshevRadialBasis(self.cutoff, self.basis_count, dtype=self.dtype)

    def __call__(self, distance: Array, /) -> Array:
        argument = distance.astype(self.dtype)
        previous = jnp.ones_like(argument)
        current = argument
        values = [current]
        for _ in range(1, self.basis_count):
            previous, current = current, 2.0 * argument * current - previous
            values.append(current)
        return jnp.stack(values, axis=-1)


RadialBasis: TypeAlias = BesselRadialBasis | GaussianRadialBasis | ChebyshevRadialBasis


def _rebuilt_basis(basis: RadialBasis, /) -> RadialBasis:
    """Re-admit one basis from its stored arrays (float32 realizations stay)."""
    match basis:
        case BesselRadialBasis():
            prefactor = np.asarray(jax.device_get(basis.prefactor))
            dtype = _float_dtype(basis.frequencies.dtype)
            if prefactor.shape != () or prefactor.dtype != dtype:
                raise ValueError(
                    "The Bessel prefactor must be a scalar of the basis dtype."
                )
            return BesselRadialBasis(
                np.asarray(jax.device_get(basis.frequencies), dtype=np.float64),
                float(prefactor),
                basis.cutoff,
                dtype=dtype,
            )
        case GaussianRadialBasis() | ChebyshevRadialBasis():
            return basis.rebuilt()
        case unreachable:
            assert_never(unreachable)


def _basis_dtype(basis: RadialBasis, /) -> np.dtype:
    match basis:
        case BesselRadialBasis():
            return np.dtype(basis.frequencies.dtype)
        case GaussianRadialBasis() | ChebyshevRadialBasis():
            return basis.dtype
        case unreachable:
            assert_never(unreachable)


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
        cutoff = jnp.asarray(self.cutoff, dtype=distance.dtype)
        ratio = distance / cutoff
        ratio_p = ratio**self.power
        envelope = (
            jnp.ones((), dtype=distance.dtype)
            - jnp.asarray(0.5 * (p + 1.0) * (p + 2.0), dtype=distance.dtype) * ratio_p
            + jnp.asarray(p * (p + 2.0), dtype=distance.dtype) * ratio_p * ratio
            - jnp.asarray(0.5 * p * (p + 1.0), dtype=distance.dtype)
            * ratio_p
            * ratio
            * ratio
        )
        return jnp.where(distance < cutoff, envelope, jnp.zeros_like(envelope))


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
        r0 = jnp.asarray(0.5, dtype=self.covalent_radii.dtype) * (
            self.covalent_radii[sender_species] + self.covalent_radii[receiver_species]
        )
        ratio = distance / r0
        one = jnp.ones((), dtype=ratio.dtype)
        a = jnp.asarray(self.a, dtype=ratio.dtype)
        q = jnp.asarray(self.q, dtype=ratio.dtype)
        exponent = jnp.asarray(self.q - self.p, dtype=ratio.dtype)
        return one / (one + a * ratio**q / (one + ratio**exponent))


@final
class SoftTransform(StrictModule, NonTrainableState):
    """Species-pair tanh-blended distance transform of the source family.

    ``r0 = R[sender] + R[receiver]`` (a sum, unlike Agnesi's mean), ``p0 = 3 r0 / 4``,
    ``p1 = 4 r0 / 3``, ``m = (p0 + p1) / 2``, ``s = (1 + tanh(alpha (r - m) /
    (p1 - p0))) / 2`` and ``t = p0 + (r - p0) s``. The map is smooth; it tends to
    ``p0`` below and to ``r`` above the blend. Radii are in the model length unit
    and ordered by the declared species.
    """

    __strict_contract__ = True

    covalent_radii: Float[RadialSpeciesDim]
    atomic_numbers: tuple[int, ...] = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    transform_id: str = eqx.field(static=True)

    def __init__(
        self,
        covalent_radii: ArrayLike,
        /,
        *,
        atomic_numbers: Sequence[int],
        alpha: float,
        dtype: DTypeLike = jnp.float64,
    ) -> None:
        dtype_ = _float_dtype(dtype)
        radii = as_host_array(
            covalent_radii, HostFloat64[RadialSpeciesDim], "covalent_radii"
        )
        numbers = tuple(
            positive_integer(value, "atomic_numbers") for value in atomic_numbers
        )
        if len(numbers) != radii.shape[0] or len(set(numbers)) != len(numbers):
            raise ValueError("atomic_numbers must be distinct and match covalent_radii.")
        if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
            raise ValueError("Soft-transform covalent radii must be finite and positive.")
        alpha_ = finite_real_scalar(alpha, "alpha")
        self.covalent_radii = jnp.asarray(radii.astype(dtype_))
        self.atomic_numbers = numbers
        self.alpha = alpha_
        self.transform_id = canonical_fingerprint(
            {
                "kind": "soft-distance-transform",
                "atomic_numbers": list(numbers),
                "covalent_radii": [float(value) for value in radii.astype(dtype_)],
                "alpha": alpha_,
                "dtype": dtype_.name,
            }
        )

    @property
    def species_count(self) -> int:
        return len(self.atomic_numbers)

    def __call__(
        self, distance: Array, sender_species: Array, receiver_species: Array, /
    ) -> Array:
        dtype = self.covalent_radii.dtype
        r0 = self.covalent_radii[sender_species] + self.covalent_radii[receiver_species]
        lower = jnp.asarray(0.75, dtype=dtype) * r0
        upper = jnp.asarray(4.0 / 3.0, dtype=dtype) * r0
        middle = jnp.asarray(0.5, dtype=dtype) * (lower + upper)
        steepness = jnp.asarray(self.alpha, dtype=dtype) / (upper - lower)
        blend = jnp.asarray(0.5, dtype=dtype) * (
            jnp.ones((), dtype=dtype) + jnp.tanh(steepness * (distance - middle))
        )
        return lower + (distance - lower) * blend


DistanceTransform: TypeAlias = AgnesiTransform | SoftTransform


def _rebuilt_transform(
    transform: DistanceTransform, dtype: np.dtype, /
) -> DistanceTransform:
    radii = np.asarray(jax.device_get(transform.covalent_radii), dtype=np.float64)
    match transform:
        case AgnesiTransform():
            return AgnesiTransform(
                radii,
                atomic_numbers=transform.atomic_numbers,
                a=transform.a,
                q=transform.q,
                p=transform.p,
                dtype=dtype,
            )
        case SoftTransform():
            return SoftTransform(
                radii,
                atomic_numbers=transform.atomic_numbers,
                alpha=transform.alpha,
                dtype=dtype,
            )
        case unreachable:
            assert_never(unreachable)


@final
class RadialEmbedding(StrictModule):
    """Edge radial features ``basis(transform(r)) * cutoff(r)``."""

    basis: RadialBasis
    cutoff: PolynomialCutoff
    transform: DistanceTransform | None
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self,
        basis: RadialBasis,
        cutoff: PolynomialCutoff,
        /,
        *,
        transform: DistanceTransform | None = None,
    ) -> None:
        if not isinstance(
            basis, (BesselRadialBasis, GaussianRadialBasis, ChebyshevRadialBasis)
        ):
            raise TypeError("basis must be a Bessel, Gaussian or Chebyshev basis.")
        if not isinstance(cutoff, PolynomialCutoff):
            raise TypeError("cutoff must be a PolynomialCutoff.")
        if transform is not None and not isinstance(
            transform, (AgnesiTransform, SoftTransform)
        ):
            raise TypeError("transform must be an Agnesi or Soft transform or None.")
        if basis.cutoff != cutoff.cutoff:
            raise ValueError("The radial basis and cutoff must share one cutoff radius.")
        if transform is not None and transform.covalent_radii.dtype != _basis_dtype(
            basis
        ):
            raise TypeError("The distance transform and basis must share one dtype.")
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
        return _basis_dtype(self.basis)

    @property
    def pair_dependent(self) -> bool:
        """Whether features depend on the ordered sender/receiver species."""
        return self.transform is not None

    def validate(self) -> None:
        """Re-admit every fixed field from the arrays that execute.

        This is a host boundary over concrete arrays. Basis arrays (Bessel
        frequencies and prefactor, Gaussian centers), the polynomial cutoff and
        the transform radii and constants are rebuilt through their constructors
        from the stored values (float32 realizations stay authoritative), and
        every recomputed identity and static field must equal the recorded one;
        recorded labels are never trusted.
        """
        dtype = _float_dtype(self.dtype)
        transform = self.transform
        rebuilt = RadialEmbedding(
            _rebuilt_basis(self.basis),
            PolynomialCutoff(self.cutoff.cutoff, self.cutoff.power),
            transform=None if transform is None else _rebuilt_transform(transform, dtype),
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
            hidden = jax.nn.silu(self._layer(value, weight))
            value = jnp.asarray(self.activation_scale, dtype=hidden.dtype) * hidden
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


@final
class LayerNormRadialMLP(StrictModule, ParameterOwner):
    """Biased ``Linear -> LayerNorm -> SiLU`` edge network of the nonlinear family.

    Hidden layer ``l`` applies ``x @ W_l + b_l``, then a biased-variance layer
    normalization ``(y - mean) / sqrt(var + epsilon) * gamma_l + beta_l`` and the
    plain (unnormalized) SiLU; the final layer is affine, followed by the declared
    postprocessing. ``weights[l]`` has shape ``(fan_in, fan_out)`` (the transpose
    of the source ``Linear.weight``); every array is a PARAMETER leaf.
    """

    weights: tuple[Array, ...]
    biases: tuple[Array, ...]
    norm_scales: tuple[Array, ...]
    norm_shifts: tuple[Array, ...]
    widths: tuple[int, ...] = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    postprocess: RadialPostprocess = eqx.field(static=True)

    def __init__(
        self,
        weights: Sequence[ArrayLike],
        biases: Sequence[ArrayLike],
        norm_scales: Sequence[ArrayLike],
        norm_shifts: Sequence[ArrayLike],
        /,
        *,
        epsilon: float,
        postprocess: RadialPostprocess = "none",
    ) -> None:
        postprocess_ = parse(postprocess, RadialPostprocess, "postprocess")
        epsilon_ = positive_finite_float(epsilon, "epsilon")
        arrays = tuple(jnp.asarray(weight) for weight in weights)
        offsets = tuple(jnp.asarray(bias) for bias in biases)
        scales = tuple(jnp.asarray(value) for value in norm_scales)
        shifts = tuple(jnp.asarray(value) for value in norm_shifts)
        if not arrays:
            raise ValueError("A radial MLP requires at least one layer.")
        dtype = arrays[0].dtype
        if dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
            raise TypeError("Radial MLP parameters must be float32 or float64.")
        widths = [arrays[0].shape[0] if arrays[0].ndim == 2 else 0]
        for index, weight in enumerate(arrays):
            if weight.ndim != 2 or weight.dtype != dtype or weight.shape[0] != widths[-1]:
                raise ValueError(f"Radial MLP layer {index} has inconsistent widths.")
            widths.append(weight.shape[1])
        if min(widths) <= 0:
            raise ValueError("Radial MLP widths must be positive.")
        hidden = tuple(widths[1:-1])
        if (
            tuple(bias.shape for bias in offsets) != tuple((w,) for w in widths[1:])
            or tuple(scale.shape for scale in scales) != tuple((w,) for w in hidden)
            or tuple(shift.shape for shift in shifts) != tuple((w,) for w in hidden)
            or any(value.dtype != dtype for value in (*offsets, *scales, *shifts))
        ):
            raise ValueError(
                "Radial MLP biases and layer normalizations must match their widths."
            )
        self.weights = arrays
        self.biases = offsets
        self.norm_scales = scales
        self.norm_shifts = shifts
        self.widths = tuple(widths)
        self.epsilon = epsilon_
        self.postprocess = postprocess_

    @classmethod
    def initialize(
        cls,
        widths: Sequence[int],
        /,
        *,
        key: PRNGKey,
        dtype: DTypeLike = jnp.float64,
        epsilon: float = 1.0e-5,
        postprocess: RadialPostprocess = "none",
    ) -> LayerNormRadialMLP:
        """Draw ``U(-1/sqrt(fan_in), 1/sqrt(fan_in))`` affine maps and unit norms."""
        sizes = tuple(positive_integer(value, "widths") for value in widths)
        if len(sizes) < 2:
            raise ValueError("A radial MLP requires input and output widths.")
        dtype_ = _float_dtype(dtype)
        keys = jr.split(key, 2 * (len(sizes) - 1))
        weights = []
        biases = []
        for index, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            bound = 1.0 / math.sqrt(fan_in)
            weights.append(
                jr.uniform(keys[2 * index], (fan_in, fan_out), dtype_, -bound, bound)
            )
            biases.append(
                jr.uniform(keys[2 * index + 1], (fan_out,), dtype_, -bound, bound)
            )
        hidden = sizes[1:-1]
        return cls(
            weights,
            biases,
            tuple(jnp.ones((width,), dtype=dtype_) for width in hidden),
            tuple(jnp.zeros((width,), dtype=dtype_) for width in hidden),
            epsilon=epsilon,
            postprocess=postprocess,
        )

    @property
    def input_width(self) -> int:
        return self.widths[0]

    @property
    def output_width(self) -> int:
        return self.widths[-1]

    def __call__(self, features: Array, /) -> Array:
        if features.shape[-1:] != (self.input_width,):
            raise ValueError(f"Radial MLP features must end in width {self.input_width}.")
        value = features
        last = len(self.weights) - 1
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            value = value @ weight + bias
            if index == last:
                break
            mean = jnp.mean(value, axis=-1, keepdims=True)
            centered = value - mean
            variance = jnp.mean(centered * centered, axis=-1, keepdims=True)
            epsilon = jnp.asarray(self.epsilon, dtype=value.dtype)
            value = jax.nn.silu(
                centered * jax.lax.rsqrt(variance + epsilon) * self.norm_scales[index]
                + self.norm_shifts[index]
            )
        match self.postprocess:
            case "none":
                return value
            case "tanh-square":
                return jnp.tanh(value * value)
            case unreachable:
                assert_never(unreachable)


for _artifact in (
    AgnesiTransform,
    BesselRadialBasis,
    ChebyshevRadialBasis,
    GaussianRadialBasis,
    LayerNormRadialMLP,
    PolynomialCutoff,
    RadialEmbedding,
    RadialMLP,
    SoftTransform,
):
    register_artifact_value(f"phydrax.nn.atomistic:{_artifact.__name__}", _artifact)
del _artifact


__all__ = [
    "AgnesiTransform",
    "BesselRadialBasis",
    "ChebyshevRadialBasis",
    "DistanceTransform",
    "GaussianRadialBasis",
    "LayerNormRadialMLP",
    "normalized_sigmoid_scale",
    "normalized_silu_scale",
    "PolynomialCutoff",
    "RadialBasis",
    "RadialEmbedding",
    "RadialMLP",
    "RadialPostprocess",
    "SoftTransform",
]
