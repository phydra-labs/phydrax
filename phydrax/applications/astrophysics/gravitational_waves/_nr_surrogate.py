#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from ...._differentiation import DerivativeContract, DerivativeSurface
from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._interpolation import linear_interpolate
from ...._physical import RelativityScaleContract
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....discretization.spectral import SphericalSpectralDiscretization
from ....interchange._report import AdapterLoss, AdapterReport, AdapterStatus
from .._photometry import ObservationDataProvenance
from ._status import GravitationalWaveStatus


_ALIGNED_PARAMETER_KEYS = ("mass_ratio", "primary_spin", "secondary_spin")
_ALIGNED_FIT_COORDINATES = (
    "log_mass_ratio",
    "effective_spin",
    "antisymmetric_spin",
)
_ALIGNED_PARAMETERIZATION_ID = (
    "mass-ratio-primary-spin-secondary-spin-to-logq-effective-antisymmetric"
)
_SPIN_WEIGHT_ID = "spin:-2"
_STRAIN_CONVENTION = "h=plus-i*cross"
_MODE_AMPLITUDE_NORMALIZATION = "r*h/M"
_NORMALIZED_SURROGATE_FORMAT = "phydrax-aligned-nr-polynomial-eim"
_GEOMETRIC_TIME_UNIT = "total-mass-geometric"


def _positive_capacity(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _array_metadata(value: ArrayLike, name: str, /) -> tuple[tuple[int, ...], int]:
    if not isinstance(value, (jax.Array, np.ndarray)):
        raise TypeError(f"{name} must be a concrete NumPy or JAX array.")
    shape = tuple(value.shape)
    entries = math.prod(shape)
    return shape, entries * np.dtype(value.dtype).itemsize


def _real_scalar(value: ArrayLike, dtype: jnp.dtype, name: str, /) -> Array:
    scalar = jnp.asarray(value)
    if (
        scalar.shape != ()
        or jnp.issubdtype(scalar.dtype, jnp.complexfloating)
        or jnp.issubdtype(scalar.dtype, jnp.bool_)
    ):
        raise TypeError(f"{name} must be one real numeric scalar.")
    return scalar.astype(dtype)


class NRSurrogateResourcePolicy(StrictModule, NonTrainableState):
    """Finite normalized-array, compiled-mode, and output capacities."""

    maximum_normalized_bytes: int = eqx.field(static=True)
    maximum_time_samples: int = eqx.field(static=True)
    maximum_modes: int = eqx.field(static=True)
    maximum_nodes_per_field: int = eqx.field(static=True)
    maximum_terms_per_node: int = eqx.field(static=True)
    maximum_polynomial_order: int = eqx.field(static=True)
    maximum_reconstruction_entries: int = eqx.field(static=True)
    maximum_angular_bandlimit: int = eqx.field(static=True)
    maximum_modal_entries: int = eqx.field(static=True)
    maximum_output_samples: int = eqx.field(static=True)
    maximum_normalization_losses: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_normalized_bytes: int = 512 * 1024**2,
        maximum_time_samples: int = 1_000_000,
        maximum_modes: int = 64,
        maximum_nodes_per_field: int = 4_096,
        maximum_terms_per_node: int = 1_024,
        maximum_polynomial_order: int = 32,
        maximum_reconstruction_entries: int = 64_000_000,
        maximum_angular_bandlimit: int = 32,
        maximum_modal_entries: int = 32_000_000,
        maximum_output_samples: int = 1_000_000,
        maximum_normalization_losses: int = 64,
    ):
        names = (
            "maximum_normalized_bytes",
            "maximum_time_samples",
            "maximum_modes",
            "maximum_nodes_per_field",
            "maximum_terms_per_node",
            "maximum_polynomial_order",
            "maximum_reconstruction_entries",
            "maximum_angular_bandlimit",
            "maximum_modal_entries",
            "maximum_output_samples",
            "maximum_normalization_losses",
        )
        values = tuple(
            _positive_capacity(value, name)
            for value, name in zip(
                (
                    maximum_normalized_bytes,
                    maximum_time_samples,
                    maximum_modes,
                    maximum_nodes_per_field,
                    maximum_terms_per_node,
                    maximum_polynomial_order,
                    maximum_reconstruction_entries,
                    maximum_angular_bandlimit,
                    maximum_modal_entries,
                    maximum_output_samples,
                    maximum_normalization_losses,
                ),
                names,
                strict=True,
            )
        )
        (
            self.maximum_normalized_bytes,
            self.maximum_time_samples,
            self.maximum_modes,
            self.maximum_nodes_per_field,
            self.maximum_terms_per_node,
            self.maximum_polynomial_order,
            self.maximum_reconstruction_entries,
            self.maximum_angular_bandlimit,
            self.maximum_modal_entries,
            self.maximum_output_samples,
            self.maximum_normalization_losses,
        ) = values
        self.policy_id = canonical_fingerprint(
            {
                "kind": "nr-surrogate-resource-policy",
                **dict(zip(names, values, strict=True)),
            }
        )


def _aligned_spin_surrogate_coordinates_raw(
    mass_ratio: ArrayLike,
    primary_spin: ArrayLike,
    secondary_spin: ArrayLike,
    /,
) -> Array:
    ratio = jnp.asarray(mass_ratio)
    spin_primary = jnp.asarray(primary_spin)
    spin_secondary = jnp.asarray(secondary_spin)
    ratio, spin_primary, spin_secondary = jnp.broadcast_arrays(
        ratio, spin_primary, spin_secondary
    )
    eta = ratio / (1.0 + ratio) ** 2
    chi_eff = (ratio * spin_primary + spin_secondary) / (1.0 + ratio)
    chi_hat = (chi_eff - (38.0 / 113.0) * eta * (spin_primary + spin_secondary)) / (
        1.0 - (76.0 / 113.0) * eta
    )
    chi_a = 0.5 * (spin_primary - spin_secondary)
    return jnp.stack((jnp.log(ratio), chi_hat, chi_a), axis=-1)


def aligned_spin_surrogate_coordinates(
    mass_ratio: ArrayLike,
    primary_spin: ArrayLike,
    secondary_spin: ArrayLike,
    /,
) -> Array:
    """Map aligned-binary parameters to smooth reduced fit coordinates.

    The mass ratio convention is ``q = primary_mass / secondary_mass >= 1``.
    The returned coordinates are ``(log(q), chi_hat, chi_a)`` using the
    leading post-Newtonian effective-spin combination.
    """

    raw = tuple(
        jnp.asarray(value) for value in (mass_ratio, primary_spin, secondary_spin)
    )
    if any(
        jnp.iscomplexobj(value) or jnp.issubdtype(value.dtype, jnp.bool_) for value in raw
    ):
        raise TypeError("Aligned surrogate coordinates require real numeric arrays.")
    dtype = jnp.result_type(*raw, 1.0)
    ratio, spin_primary, spin_secondary = jnp.broadcast_arrays(
        *(value.astype(dtype) for value in raw)
    )
    valid = (
        jnp.isfinite(ratio)
        & jnp.isfinite(spin_primary)
        & jnp.isfinite(spin_secondary)
        & (ratio >= 1.0)
        & (jnp.abs(spin_primary) <= 1.0)
        & (jnp.abs(spin_secondary) <= 1.0)
    )
    safe_ratio = jnp.where(valid, ratio, 1.0)
    safe_primary = jnp.where(valid, spin_primary, 0.0)
    safe_secondary = jnp.where(valid, spin_secondary, 0.0)
    coordinates = _aligned_spin_surrogate_coordinates_raw(
        safe_ratio, safe_primary, safe_secondary
    )
    return eqx.error_if(
        coordinates,
        jnp.any(~valid),
        "Aligned surrogate coordinates require finite q >= 1 and spins in [-1, 1].",
    )


class PolynomialEmpiricalField(StrictModule, NonTrainableState):
    """Static padded polynomial nodes followed by empirical reconstruction."""

    reconstruction_matrix: Array
    coefficients: Array
    orders: Array
    active_terms: Array
    resource_policy: NRSurrogateResourcePolicy
    parameter_count: int = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    term_count: int = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    array_bytes: int = eqx.field(static=True)
    resource_policy_id: str = eqx.field(static=True)
    content_sha256: str = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        reconstruction_matrix: ArrayLike,
        coefficients: ArrayLike,
        orders: ArrayLike,
        active_terms: ArrayLike,
        /,
        *,
        field_id: str,
        resource_policy: NRSurrogateResourcePolicy | None = None,
    ):
        policy = (
            NRSurrogateResourcePolicy() if resource_policy is None else resource_policy
        )
        if not isinstance(policy, NRSurrogateResourcePolicy):
            raise TypeError("resource_policy must be NRSurrogateResourcePolicy or None.")
        reconstruction_shape, _ = _array_metadata(
            reconstruction_matrix, "reconstruction_matrix"
        )
        coefficient_shape, _ = _array_metadata(coefficients, "coefficients")
        order_shape, _ = _array_metadata(orders, "orders")
        active_shape, _ = _array_metadata(active_terms, "active_terms")
        if len(coefficient_shape) != 2 or len(order_shape) != 3:
            raise ValueError(
                "Polynomial coefficients and orders must have ranks two and three."
            )
        node_count, term_count = coefficient_shape
        if (
            not node_count
            or not term_count
            or order_shape[:2] != coefficient_shape
            or order_shape[2] == 0
            or active_shape != coefficient_shape
            or len(reconstruction_shape) != 2
            or reconstruction_shape[1] != node_count
            or reconstruction_shape[0] < 2
        ):
            raise ValueError("Polynomial empirical field arrays are shape-incompatible.")
        field_bytes = (
            8 * math.prod(reconstruction_shape)
            + 8 * math.prod(coefficient_shape)
            + 4 * math.prod(order_shape)
            + math.prod(active_shape)
        )
        if (
            reconstruction_shape[0] > policy.maximum_time_samples
            or node_count > policy.maximum_nodes_per_field
            or term_count > policy.maximum_terms_per_node
            or math.prod(reconstruction_shape) > policy.maximum_reconstruction_entries
            or field_bytes > policy.maximum_normalized_bytes
        ):
            raise ValueError("Polynomial empirical field exceeds its resource policy.")
        reconstruction_host = np.asarray(reconstruction_matrix)
        coefficient_host = np.asarray(coefficients)
        order_host = np.asarray(orders)
        active_host = np.asarray(active_terms)
        identifier = str(field_id).strip()
        if not identifier:
            raise ValueError("field_id must be non-empty.")
        if np.iscomplexobj(reconstruction_host) or np.iscomplexobj(coefficient_host):
            raise ValueError("Polynomial empirical fields require real arrays.")
        if order_host.dtype.kind not in ("i", "u") or active_host.dtype.kind != "b":
            raise TypeError(
                "Polynomial orders must be integers and active_terms boolean."
            )
        if coefficient_host.ndim != 2 or order_host.ndim != 3:
            raise ValueError(
                "Polynomial coefficients and orders must have ranks two and three."
            )
        node_count, term_count = coefficient_host.shape
        if (
            node_count == 0
            or term_count == 0
            or order_host.shape[:2] != coefficient_host.shape
            or order_host.shape[2] == 0
            or active_host.shape != coefficient_host.shape
            or reconstruction_host.ndim != 2
            or reconstruction_host.shape[1] != node_count
            or reconstruction_host.shape[0] < 2
        ):
            raise ValueError("Polynomial empirical field arrays are shape-incompatible.")
        if (
            np.any(~np.isfinite(reconstruction_host))
            or np.any(~np.isfinite(coefficient_host))
            or np.any(order_host < 0)
            or np.any(~np.any(active_host, axis=1))
            or np.any(coefficient_host[~active_host] != 0.0)
            or np.any(order_host[~active_host] != 0)
        ):
            raise ValueError("Polynomial empirical field data are invalid or unmasked.")
        if np.any(order_host > policy.maximum_polynomial_order):
            raise ValueError("Polynomial orders exceed the resource policy.")
        reconstruction_real = np.asarray(reconstruction_host, dtype=np.float64)
        coefficient_real = np.asarray(coefficient_host, dtype=np.float64)
        order_integer = np.asarray(order_host, dtype=np.int32)
        active_boolean = np.asarray(active_host, dtype=np.bool_)
        self.reconstruction_matrix = jnp.asarray(reconstruction_real)
        self.coefficients = jnp.asarray(coefficient_real)
        self.orders = jnp.asarray(order_integer)
        self.active_terms = jnp.asarray(active_boolean)
        self.resource_policy = policy
        self.parameter_count = order_integer.shape[2]
        self.node_count = int(node_count)
        self.term_count = int(term_count)
        self.sample_count = reconstruction_real.shape[0]
        self.array_bytes = field_bytes
        self.resource_policy_id = policy.policy_id
        content_sha256 = array_tree_fingerprint(
            {
                "reconstruction": reconstruction_real,
                "coefficients": coefficient_real,
                "orders": order_integer,
                "active_terms": active_boolean,
            }
        )["sha256"]
        self.content_sha256 = content_sha256
        self.field_id = canonical_fingerprint(
            {
                "kind": "polynomial-empirical-field",
                "label": identifier,
                "content": content_sha256,
                "resource_policy": policy.policy_id,
            }
        )

    def node_values(self, fit_coordinates: ArrayLike, /) -> Array:
        coordinates = jnp.asarray(fit_coordinates)
        if coordinates.shape != (self.parameter_count,):
            raise ValueError(
                f"Fit coordinates must have shape {(self.parameter_count,)}."
            )
        factors = jnp.where(
            self.orders == 0,
            jnp.ones((), dtype=coordinates.dtype),
            coordinates[None, None, :] ** self.orders,
        )
        monomials = jnp.prod(factors, axis=-1)
        return jnp.sum(
            jnp.where(
                self.active_terms,
                self.coefficients * monomials,
                jnp.zeros((), dtype=monomials.dtype),
            ),
            axis=-1,
        )

    def evaluate(self, fit_coordinates: ArrayLike, /) -> Array:
        return contract(
            "tn,n->t", self.reconstruction_matrix, self.node_values(fit_coordinates)
        )


def _normalized_surrogate_content_sha256(
    geometric_time: np.ndarray,
    modes: tuple[tuple[int, int], ...],
    real_fields: tuple[PolynomialEmpiricalField, ...],
    imaginary_fields: tuple[PolynomialEmpiricalField, ...],
    fit_coordinate_lower: np.ndarray,
    fit_coordinate_upper: np.ndarray,
    /,
    *,
    maximum_mass_ratio: float,
    maximum_spin_magnitude: float,
    frame_id: str,
    time_origin_id: str,
    mode_normalization: str,
) -> str:
    return canonical_fingerprint(
        {
            "kind": "aligned-nr-polynomial-surrogate-content",
            "time": array_tree_fingerprint(geometric_time),
            "modes": [list(mode) for mode in modes],
            "real_fields": [field.content_sha256 for field in real_fields],
            "imaginary_fields": [field.content_sha256 for field in imaginary_fields],
            "fit_coordinate_lower": fit_coordinate_lower.tolist(),
            "fit_coordinate_upper": fit_coordinate_upper.tolist(),
            "maximum_mass_ratio": maximum_mass_ratio,
            "maximum_spin_magnitude": maximum_spin_magnitude,
            "frame": frame_id,
            "time_origin": time_origin_id,
            "mode_normalization": mode_normalization,
            "strain_convention": _STRAIN_CONVENTION,
            "amplitude_normalization": _MODE_AMPLITUDE_NORMALIZATION,
        }
    )


def aligned_nr_surrogate_semantic_bindings(
    geometric_time: ArrayLike,
    modes: Sequence[tuple[int, int]],
    fit_coordinate_lower: ArrayLike,
    fit_coordinate_upper: ArrayLike,
    /,
    *,
    maximum_mass_ratio: float,
    maximum_spin_magnitude: float,
    frame_id: str,
    time_origin_id: str,
    mode_normalization: str,
    differentiation_id: str,
    resource_policy: NRSurrogateResourcePolicy | None = None,
) -> dict[str, str]:
    """Build the artifact's exact internal semantic identities."""

    policy = NRSurrogateResourcePolicy() if resource_policy is None else resource_policy
    if not isinstance(policy, NRSurrogateResourcePolicy):
        raise TypeError("resource_policy must be NRSurrogateResourcePolicy or None.")
    time_shape, _ = _array_metadata(geometric_time, "geometric_time")
    lower_shape, _ = _array_metadata(fit_coordinate_lower, "fit_coordinate_lower")
    upper_shape, _ = _array_metadata(fit_coordinate_upper, "fit_coordinate_upper")
    raw_modes = tuple(modes)
    if (
        len(time_shape) != 1
        or time_shape[0] < 2
        or time_shape[0] > policy.maximum_time_samples
        or lower_shape != (3,)
        or upper_shape != (3,)
        or not raw_modes
        or len(raw_modes) > policy.maximum_modes
        or any(
            not isinstance(mode, tuple)
            or len(mode) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in mode
            )
            for mode in raw_modes
        )
    ):
        raise ValueError("Surrogate semantic inputs are malformed or over capacity.")
    mode_values = tuple((int(ell), int(order)) for ell, order in raw_modes)
    if len(set(mode_values)) != len(mode_values) or any(
        ell < 2 or order < 0 or order > ell for ell, order in mode_values
    ):
        raise ValueError("Surrogate semantic mode set is invalid.")
    time_host = np.asarray(geometric_time, dtype=np.float64)
    lower_host = np.asarray(fit_coordinate_lower, dtype=np.float64)
    upper_host = np.asarray(fit_coordinate_upper, dtype=np.float64)
    ratio_limit = float(maximum_mass_ratio)
    spin_limit = float(maximum_spin_magnitude)
    frame = str(frame_id).strip()
    time_origin = str(time_origin_id).strip()
    normalization = str(mode_normalization).strip()
    differentiation = str(differentiation_id).strip()
    if (
        np.any(~np.isfinite(time_host))
        or np.any(np.diff(time_host) <= 0.0)
        or np.any(~np.isfinite(lower_host))
        or np.any(~np.isfinite(upper_host))
        or np.any(lower_host >= upper_host)
        or not np.isfinite(ratio_limit)
        or ratio_limit < 1.0
        or not np.isfinite(spin_limit)
        or spin_limit <= 0.0
        or spin_limit > 1.0
        or not frame
        or not time_origin
        or not normalization
        or not differentiation
    ):
        raise ValueError("Surrogate semantic support or identity is invalid.")
    mode_set_id = canonical_fingerprint(
        {
            "kind": "spin-weighted-mode-set",
            "spin_weight": -2,
            "modes": [list(mode) for mode in mode_values],
        }
    )
    physical_support_id = canonical_fingerprint(
        {
            "kind": "aligned-nr-physical-support",
            "parameterization": _ALIGNED_PARAMETERIZATION_ID,
            "mass_ratio": [1.0, ratio_limit],
            "primary_spin": [-spin_limit, spin_limit],
            "secondary_spin": [-spin_limit, spin_limit],
        }
    )
    fit_support_id = canonical_fingerprint(
        {
            "kind": "aligned-nr-fit-support",
            "coordinate_names": list(_ALIGNED_FIT_COORDINATES),
            "lower": lower_host.tolist(),
            "upper": upper_host.tolist(),
        }
    )
    time_support_id = canonical_fingerprint(
        {
            "kind": "aligned-nr-time-support",
            "time_origin": time_origin,
            "geometric_time": array_tree_fingerprint(time_host),
        }
    )
    return {
        "mode_basis_id": normalization,
        "quantity_id": _MODE_AMPLITUDE_NORMALIZATION,
        "time_reference_id": time_origin,
        "unit_id": _GEOMETRIC_TIME_UNIT,
        "frame_id": frame,
        "strain_convention": _STRAIN_CONVENTION,
        "spin_weight_id": _SPIN_WEIGHT_ID,
        "mode_set_id": mode_set_id,
        "parameterization_id": _ALIGNED_PARAMETERIZATION_ID,
        "physical_support_id": physical_support_id,
        "fit_support_id": fit_support_id,
        "time_support_id": time_support_id,
        "differentiation_id": differentiation,
    }


class AlignedNRSurrogateArtifact(StrictModule, NonTrainableState):
    """Caller-supplied nonprecessing numerical-relativity mode surrogate.

    Only nonnegative-m modes are stored. Evaluation reconstructs negative-m
    modes through ``h[l,-m] = (-1)^l conjugate(h[l,m])``. Direct arrays are
    content-bound but never authenticated as a Phydrax or external release;
    every result therefore remains explicitly unqualified.
    """

    geometric_time: Array
    fit_coordinate_lower: Array
    fit_coordinate_upper: Array
    real_fields: tuple[PolynomialEmpiricalField, ...]
    imaginary_fields: tuple[PolynomialEmpiricalField, ...]
    provenance: ObservationDataProvenance
    resource_policy: NRSurrogateResourcePolicy
    normalization_report: AdapterReport
    differentiation: DerivativeContract
    modes: tuple[tuple[int, int], ...] = eqx.field(static=True)
    fit_coordinate_names: tuple[str, ...] = eqx.field(static=True)
    mode_set_id: str = eqx.field(static=True)
    parameterization_id: str = eqx.field(static=True)
    physical_support_id: str = eqx.field(static=True)
    fit_support_id: str = eqx.field(static=True)
    time_support_id: str = eqx.field(static=True)
    maximum_mass_ratio: float = eqx.field(static=True)
    maximum_spin_magnitude: float = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    time_origin_id: str = eqx.field(static=True)
    mode_normalization: str = eqx.field(static=True)
    strain_convention: str = eqx.field(static=True)
    amplitude_normalization: str = eqx.field(static=True)
    normalized_content_sha256: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    source_authenticated: bool = eqx.field(static=True)
    trust_id: str = eqx.field(static=True)
    array_bytes: int = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometric_time: ArrayLike,
        modes: Sequence[tuple[int, int]],
        real_fields: Sequence[PolynomialEmpiricalField],
        imaginary_fields: Sequence[PolynomialEmpiricalField],
        fit_coordinate_lower: ArrayLike,
        fit_coordinate_upper: ArrayLike,
        provenance: ObservationDataProvenance,
        /,
        *,
        maximum_mass_ratio: float,
        maximum_spin_magnitude: float,
        frame_id: str,
        time_origin_id: str,
        mode_normalization: str,
        artifact_id: str,
        resource_policy: NRSurrogateResourcePolicy | None = None,
    ):
        policy = (
            NRSurrogateResourcePolicy() if resource_policy is None else resource_policy
        )
        if not isinstance(policy, NRSurrogateResourcePolicy):
            raise TypeError("resource_policy must be NRSurrogateResourcePolicy or None.")
        time_shape, _ = _array_metadata(geometric_time, "geometric_time")
        lower_shape, _ = _array_metadata(fit_coordinate_lower, "fit_coordinate_lower")
        upper_shape, _ = _array_metadata(fit_coordinate_upper, "fit_coordinate_upper")
        if (
            len(time_shape) != 1
            or time_shape[0] < 2
            or time_shape[0] > policy.maximum_time_samples
            or lower_shape != (3,)
            or upper_shape != (3,)
        ):
            raise ValueError("Surrogate grids or bounds exceed their resource policy.")
        time_host = np.asarray(geometric_time, dtype=np.float64)
        lower_host = np.asarray(fit_coordinate_lower, dtype=np.float64)
        upper_host = np.asarray(fit_coordinate_upper, dtype=np.float64)
        raw_modes = tuple(modes)
        if any(
            not isinstance(mode, tuple)
            or len(mode) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in mode
            )
            for mode in raw_modes
        ):
            raise TypeError("Surrogate modes must be integer (ell, m) tuples.")
        mode_values = tuple((int(ell), int(order)) for ell, order in raw_modes)
        if (
            not mode_values
            or len(mode_values) > policy.maximum_modes
            or len(set(mode_values)) != len(mode_values)
            or any(ell < 2 or order < 0 or order > ell for ell, order in mode_values)
        ):
            raise ValueError(
                "Surrogate modes are empty, invalid, duplicated, or over capacity."
            )
        real_values = tuple(real_fields)
        imaginary_values = tuple(imaginary_fields)
        if len(real_values) != len(mode_values) or len(imaginary_values) != len(
            mode_values
        ):
            raise ValueError("Every surrogate mode requires real and imaginary fields.")
        if any(
            not isinstance(field, PolynomialEmpiricalField)
            for field in real_values + imaginary_values
        ):
            raise TypeError("Surrogate fields must be PolynomialEmpiricalField values.")
        fields = real_values + imaginary_values
        if any(
            field.sample_count != time_host.size
            or field.parameter_count != 3
            or field.resource_policy_id != policy.policy_id
            for field in fields
        ):
            raise ValueError(
                "Surrogate fields must share the time grid, coordinates, and resource policy."
            )
        aggregate_bytes = (
            8 * math.prod(time_shape)
            + 8 * math.prod(lower_shape)
            + 8 * math.prod(upper_shape)
            + sum(field.array_bytes for field in fields)
        )
        if aggregate_bytes > policy.maximum_normalized_bytes:
            raise ValueError(
                "Surrogate normalized arrays exceed their aggregate byte cap."
            )
        frame = str(frame_id).strip()
        time_origin = str(time_origin_id).strip()
        mode_normalization_ = str(mode_normalization).strip()
        label = str(artifact_id).strip()
        ratio_limit = float(maximum_mass_ratio)
        spin_limit = float(maximum_spin_magnitude)
        if not isinstance(provenance, ObservationDataProvenance):
            raise TypeError("provenance must be ObservationDataProvenance.")
        if (
            np.any(~np.isfinite(time_host))
            or np.any(np.diff(time_host) <= 0.0)
            or np.any(~np.isfinite(lower_host))
            or np.any(~np.isfinite(upper_host))
            or np.any(lower_host >= upper_host)
            or not np.isfinite(ratio_limit)
            or ratio_limit < 1.0
            or not np.isfinite(spin_limit)
            or spin_limit <= 0.0
            or spin_limit > 1.0
            or not frame
            or not time_origin
            or not mode_normalization_
            or not label
        ):
            raise ValueError("Surrogate support, grids, or identity are invalid.")
        semantic_bindings = aligned_nr_surrogate_semantic_bindings(
            time_host,
            mode_values,
            lower_host,
            upper_host,
            maximum_mass_ratio=ratio_limit,
            maximum_spin_magnitude=spin_limit,
            frame_id=frame,
            time_origin_id=time_origin,
            mode_normalization=mode_normalization_,
            differentiation_id=provenance.differentiation.contract_id,
            resource_policy=policy,
        )
        normalized_content = _normalized_surrogate_content_sha256(
            time_host,
            mode_values,
            real_values,
            imaginary_values,
            lower_host,
            upper_host,
            maximum_mass_ratio=ratio_limit,
            maximum_spin_magnitude=spin_limit,
            frame_id=frame,
            time_origin_id=time_origin,
            mode_normalization=mode_normalization_,
        )
        source_differentiation = provenance.differentiation
        # Stored values are not differentiable after normalization, and the
        # piecewise reconstruction claims no global higher-order regularity.
        differentiation = DerivativeContract(
            (
                entry
                for entry in source_differentiation.surfaces
                if entry.surface is not DerivativeSurface.STORED_VALUES
            ),
            route=source_differentiation.route,
            conditions=source_differentiation.conditions,
            nondifferentiable_outputs=source_differentiation.nondifferentiable_outputs,
        )
        differentiation_losses = (
            ()
            if differentiation.contract_id == source_differentiation.contract_id
            else (
                AdapterLoss(
                    "/differentiation",
                    "import",
                    "transformed",
                    "Polynomial reconstruction and bounded piecewise interpolation "
                    "do not preserve stored-value or global higher-order derivatives.",
                    changes_interpretation=False,
                    affected_capability_ids=(
                        source_differentiation.contract_id,
                        differentiation.contract_id,
                    ),
                ),
            )
        )
        if len(differentiation_losses) > policy.maximum_normalization_losses:
            raise ValueError("Surrogate normalization losses exceed the resource policy.")
        normalization_report = AdapterReport(
            (
                AdapterStatus.DECLARED_LOSS
                if differentiation_losses
                else AdapterStatus.LOSSLESS
            ),
            "phydrax-arrays",
            _NORMALIZED_SURROGATE_FORMAT,
            source_id=provenance.provenance_id,
            target_id=normalized_content,
            preserved_fields=(
                "fit-coordinate-support",
                "geometric-time",
                "mode-frame",
                "mode-normalization",
                "strain-convention",
                "polynomial-node-models",
                "reconstruction-matrices",
            ),
            losses=differentiation_losses,
            stage="nr-surrogate-caller-normalization",
        )
        if not normalization_report.valid:
            raise ValueError("Surrogate normalization report is not valid.")
        source_binding_id = canonical_fingerprint(
            {
                "kind": "caller-asserted-nr-surrogate-source-binding",
                "provenance": provenance.provenance_id,
                "normalization_report": normalization_report.report_id,
                "normalized_content_sha256": normalized_content,
                "differentiation": differentiation.contract_id,
            }
        )
        trust_id = canonical_fingerprint(
            {
                "kind": "nr-surrogate-caller-asserted-trust",
                "source_authenticated": False,
            }
        )

        self.geometric_time = jnp.asarray(time_host)
        self.fit_coordinate_lower = jnp.asarray(lower_host)
        self.fit_coordinate_upper = jnp.asarray(upper_host)
        self.real_fields = real_values
        self.imaginary_fields = imaginary_values
        self.provenance = provenance
        self.resource_policy = policy
        self.normalization_report = normalization_report
        self.differentiation = differentiation
        self.modes = mode_values
        self.fit_coordinate_names = _ALIGNED_FIT_COORDINATES
        self.mode_set_id = semantic_bindings["mode_set_id"]
        self.parameterization_id = _ALIGNED_PARAMETERIZATION_ID
        self.physical_support_id = semantic_bindings["physical_support_id"]
        self.fit_support_id = semantic_bindings["fit_support_id"]
        self.time_support_id = semantic_bindings["time_support_id"]
        self.maximum_mass_ratio = ratio_limit
        self.maximum_spin_magnitude = spin_limit
        self.frame_id = frame
        self.time_origin_id = time_origin
        self.mode_normalization = mode_normalization_
        self.strain_convention = _STRAIN_CONVENTION
        self.amplitude_normalization = _MODE_AMPLITUDE_NORMALIZATION
        self.normalized_content_sha256 = normalized_content
        self.source_binding_id = source_binding_id
        self.source_authenticated = False
        self.trust_id = trust_id
        self.array_bytes = aggregate_bytes
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "aligned-nr-polynomial-surrogate-artifact",
                "label": label,
                "normalized_content_sha256": normalized_content,
                "source_binding": source_binding_id,
                "resource_policy": policy.policy_id,
                "differentiation": differentiation.contract_id,
                "provenance": provenance.provenance_id,
                "trust": trust_id,
            }
        )


class NRSurrogateModeResult(StrictModule):
    geometric_time: Array
    coefficients: Array
    fit_coordinates: Array
    symmetry_defect: Array
    valid: Array
    qualified: Array
    status: Array
    intrinsic_derivative_valid: Array
    artifact_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    strain_convention: str = eqx.field(static=True)
    time_origin_id: str = eqx.field(static=True)
    mode_normalization: str = eqx.field(static=True)
    amplitude_normalization: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    differentiation_id: str = eqx.field(static=True)
    trust_id: str = eqx.field(static=True)


class NRSurrogatePolarizations(StrictModule):
    time: Array
    values: Array
    support: Array
    valid: Array
    qualified: Array
    status: Array
    intrinsic_derivative_valid: Array
    extrinsic_derivative_valid: Array
    mass_scaling_derivative_valid: Array
    time_derivative_valid: Array
    time_unit: str = eqx.field(static=True)
    mass_frame_id: str = eqx.field(static=True)
    waveform_id: str = eqx.field(static=True)
    frame_id: str = eqx.field(static=True)
    strain_convention: str = eqx.field(static=True)
    time_origin_id: str = eqx.field(static=True)
    mode_normalization: str = eqx.field(static=True)
    amplitude_normalization: str = eqx.field(static=True)
    source_binding_id: str = eqx.field(static=True)
    differentiation_id: str = eqx.field(static=True)
    trust_id: str = eqx.field(static=True)

    @property
    def plus(self) -> Array:
        return self.values[0]

    @property
    def cross(self) -> Array:
        return self.values[1]


class AlignedNRSurrogatePlan(StrictModule, NonTrainableState):
    """JAX-native evaluation of a normalized aligned-spin NR mode surrogate."""

    artifact: AlignedNRSurrogateArtifact
    angular: SphericalSpectralDiscretization
    scale: RelativityScaleContract
    symmetry_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        artifact: AlignedNRSurrogateArtifact,
        angular: SphericalSpectralDiscretization,
        scale: RelativityScaleContract,
        /,
        *,
        symmetry_tolerance: float = 1.0e-10,
    ):
        if not isinstance(artifact, AlignedNRSurrogateArtifact):
            raise TypeError("artifact must be AlignedNRSurrogateArtifact.")
        if not isinstance(angular, SphericalSpectralDiscretization):
            raise TypeError("angular must be SphericalSpectralDiscretization.")
        if angular.layout.spin != -2 or angular.layout.reality:
            raise ValueError(
                "NR waveform synthesis requires a complex spin-minus-two basis."
            )
        if angular.layout.bandlimit <= max(ell for ell, _ in artifact.modes):
            raise ValueError("Angular bandlimit must contain every surrogate mode.")
        if artifact.mode_normalization != angular.layout.normalization:
            raise ValueError(
                "Artifact mode normalization does not match the angular basis."
            )
        modal_entries = (
            math.prod(angular.layout.coefficient_shape) * artifact.geometric_time.size
        )
        if (
            angular.layout.bandlimit > artifact.resource_policy.maximum_angular_bandlimit
            or modal_entries > artifact.resource_policy.maximum_modal_entries
        ):
            raise ValueError(
                "Angular mode storage exceeds the surrogate resource policy."
            )
        if (
            not artifact.normalization_report.valid
            or artifact.normalization_report.target_id
            != artifact.normalized_content_sha256
        ):
            raise ValueError("Surrogate normalization binding is invalid.")
        if not isinstance(scale, RelativityScaleContract):
            raise TypeError("scale must be RelativityScaleContract.")
        if scale.scale_id != RelativityScaleContract.si().scale_id:
            raise ValueError(
                "Physical NR surrogate evaluation currently requires the exact SI relativity scale."
            )
        tolerance = float(symmetry_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("symmetry_tolerance must be finite and nonnegative.")
        self.artifact = artifact
        self.angular = angular
        self.scale = scale
        self.symmetry_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "aligned-nr-surrogate-plan",
                "artifact": artifact.artifact_id,
                "angular": angular.prepared_id,
                "scale": scale.scale_id,
                "symmetry_tolerance": tolerance,
                "source_binding": artifact.source_binding_id,
                "resource_policy": artifact.resource_policy.policy_id,
                "physical_mass_frame": "detector-frame-redshifted-mass",
            }
        )

    def evaluate_modes(
        self,
        parameters: PyTree[ArrayLike],
        /,
        *,
        reference_phase: ArrayLike = 0.0,
    ) -> NRSurrogateModeResult:
        """Reconstruct native-grid modes with ``h_lm -> exp(-i m phase) h_lm``."""
        if not isinstance(parameters, Mapping) or set(parameters) != set(
            _ALIGNED_PARAMETER_KEYS
        ):
            raise ValueError(
                f"Aligned NR surrogate parameters must be exactly {list(_ALIGNED_PARAMETER_KEYS)}."
            )
        dtype = self.artifact.geometric_time.dtype
        ratio = _real_scalar(parameters["mass_ratio"], dtype, "mass_ratio")
        primary_spin = _real_scalar(parameters["primary_spin"], dtype, "primary_spin")
        secondary_spin = _real_scalar(
            parameters["secondary_spin"], dtype, "secondary_spin"
        )
        phase = _real_scalar(reference_phase, dtype, "reference_phase")
        raw = jnp.stack((ratio, primary_spin, secondary_spin, phase))
        finite_inputs = jnp.all(jnp.isfinite(raw))
        physical_support = (
            (ratio >= 1.0)
            & (ratio <= self.artifact.maximum_mass_ratio)
            & (jnp.abs(primary_spin) <= self.artifact.maximum_spin_magnitude)
            & (jnp.abs(secondary_spin) <= self.artifact.maximum_spin_magnitude)
        )
        physical_interior = (
            (ratio > 1.0)
            & (ratio < self.artifact.maximum_mass_ratio)
            & (jnp.abs(primary_spin) < self.artifact.maximum_spin_magnitude)
            & (jnp.abs(secondary_spin) < self.artifact.maximum_spin_magnitude)
        )
        physical_evaluation_support = finite_inputs & physical_support
        safe_ratio = jnp.where(physical_evaluation_support, ratio, 1.0)
        safe_primary = jnp.where(physical_evaluation_support, primary_spin, 0.0)
        safe_secondary = jnp.where(physical_evaluation_support, secondary_spin, 0.0)
        period = jnp.asarray(2.0 * jnp.pi, dtype=phase.dtype)
        safe_phase = jnp.where(jnp.isfinite(phase), jnp.remainder(phase, period), 0.0)
        fit_coordinates = _aligned_spin_surrogate_coordinates_raw(
            safe_ratio, safe_primary, safe_secondary
        )
        fit_support = jnp.all(
            (fit_coordinates >= self.artifact.fit_coordinate_lower)
            & (fit_coordinates <= self.artifact.fit_coordinate_upper)
        )
        fit_interior = jnp.all(
            (fit_coordinates > self.artifact.fit_coordinate_lower)
            & (fit_coordinates < self.artifact.fit_coordinate_upper)
        )
        evaluation_coordinates = jnp.clip(
            fit_coordinates,
            self.artifact.fit_coordinate_lower,
            self.artifact.fit_coordinate_upper,
        )
        coefficient_shape = self.angular.layout.coefficient_shape + (
            self.artifact.geometric_time.size,
        )
        modal = jnp.zeros(coefficient_shape, dtype=jnp.complex128)
        symmetry_defect = jnp.asarray(0.0, dtype=self.artifact.geometric_time.dtype)
        center = self.angular.layout.bandlimit - 1
        for (ell, order), real_field, imaginary_field in zip(
            self.artifact.modes,
            self.artifact.real_fields,
            self.artifact.imaginary_fields,
            strict=True,
        ):
            mode = (
                real_field.evaluate(evaluation_coordinates)
                + 1.0j * imaginary_field.evaluate(evaluation_coordinates)
            ) * jnp.exp(-1.0j * order * safe_phase)
            modal = modal.at[ell, center + order].set(mode)
            if order > 0:
                modal = modal.at[ell, center - order].set(
                    (-1.0 if ell % 2 else 1.0) * jnp.conj(mode)
                )
            else:
                required = (-1.0 if ell % 2 else 1.0) * jnp.conj(mode)
                symmetry_defect = jnp.maximum(
                    symmetry_defect, jnp.max(jnp.abs(mode - required))
                )
        finite_modes = jnp.all(jnp.isfinite(modal))
        valid = (
            finite_inputs
            & physical_support
            & fit_support
            & finite_modes
            & (symmetry_defect <= self.symmetry_tolerance)
        )
        qualified = valid & jnp.asarray(self.artifact.source_authenticated)
        intrinsic_derivative_valid = (
            valid
            & physical_interior
            & fit_interior
            & (
                DerivativeSurface.PHYSICAL_PARAMETER
                in self.artifact.differentiation.supported_surfaces
            )
        )
        status = jnp.where(
            valid,
            int(GravitationalWaveStatus.SUCCESS),
            jnp.where(
                ~finite_inputs,
                int(GravitationalWaveStatus.NONFINITE_WAVEFORM),
                jnp.where(
                    ~physical_support | ~fit_support,
                    int(GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT),
                    jnp.where(
                        ~finite_modes,
                        int(GravitationalWaveStatus.NONFINITE_WAVEFORM),
                        int(GravitationalWaveStatus.APPROXIMATION_FAILURE),
                    ),
                ),
            ),
        ).astype(jnp.int32)
        return NRSurrogateModeResult(
            self.artifact.geometric_time,
            jnp.where(valid, modal, jnp.zeros_like(modal)),
            fit_coordinates,
            symmetry_defect,
            valid,
            qualified,
            status,
            intrinsic_derivative_valid,
            self.artifact.artifact_id,
            self.artifact.frame_id,
            self.artifact.strain_convention,
            self.artifact.time_origin_id,
            self.artifact.mode_normalization,
            self.artifact.amplitude_normalization,
            self.artifact.source_binding_id,
            self.artifact.differentiation.contract_id,
            self.artifact.trust_id,
        )

    def evaluate_geometric(
        self,
        geometric_time: ArrayLike,
        parameters: PyTree[ArrayLike],
        /,
        *,
        inclination: ArrayLike,
        azimuth: ArrayLike = 0.0,
        reference_phase: ArrayLike = 0.0,
        frame_angle: ArrayLike = 0.0,
    ) -> NRSurrogatePolarizations:
        """Synthesize dimensionless ``r h / M`` on requested ``t / M`` samples."""
        query_raw = jnp.asarray(geometric_time)
        if jnp.iscomplexobj(query_raw) or jnp.issubdtype(query_raw.dtype, jnp.bool_):
            raise TypeError("Surrogate evaluation time must be a real numeric vector.")
        query = query_raw.astype(self.artifact.geometric_time.dtype)
        output_modal_entries = (
            math.prod(self.angular.layout.coefficient_shape) * query.size
        )
        if (
            query.ndim != 1
            or query.size == 0
            or query.size > self.artifact.resource_policy.maximum_output_samples
            or output_modal_entries > self.artifact.resource_policy.maximum_modal_entries
        ):
            raise ValueError(
                "Surrogate evaluation time must be a non-empty vector within the output-sample capacity."
            )
        dtype = self.artifact.geometric_time.dtype
        inclination_ = _real_scalar(inclination, dtype, "inclination")
        azimuth_ = _real_scalar(azimuth, dtype, "azimuth")
        frame_angle_ = _real_scalar(frame_angle, dtype, "frame_angle")
        angles = jnp.stack((inclination_, azimuth_, frame_angle_))
        finite_angles = jnp.all(jnp.isfinite(angles))
        angular_support = (inclination_ >= 0.0) & (inclination_ <= jnp.pi)
        angle_period = jnp.asarray(2.0 * jnp.pi, dtype=angles.dtype)
        safe_inclination = jnp.where(
            finite_angles & angular_support,
            inclination_,
            jnp.asarray(0.5 * jnp.pi, dtype=inclination_.dtype),
        )
        safe_azimuth = jnp.where(
            jnp.isfinite(azimuth_),
            jnp.remainder(azimuth_, angle_period),
            0.0,
        )
        safe_frame_angle = jnp.where(
            jnp.isfinite(frame_angle_),
            jnp.remainder(frame_angle_, angle_period),
            0.0,
        )
        modes = self.evaluate_modes(parameters, reference_phase=reference_phase)
        interpolated = linear_interpolate(
            modes.geometric_time,
            modes.coefficients,
            query,
            axis=-1,
            bounds="fill",
            fill_value=0.0j,
        )
        modal_at_query = jnp.moveaxis(interpolated.values, 0, -1)
        strain = self.angular.evaluate_angles(
            modal_at_query,
            safe_inclination,
            safe_azimuth,
            frame_angle=safe_frame_angle,
        )
        finite_samples = jnp.isfinite(query) & jnp.isfinite(strain)
        support = interpolated.support & angular_support
        valid = modes.valid & finite_angles & support & finite_samples
        knot_index = jnp.clip(
            jnp.searchsorted(modes.geometric_time, query, side="left"),
            0,
            modes.geometric_time.size - 1,
        )
        at_knot = query == modes.geometric_time[knot_index]
        intrinsic_derivative_valid = valid & modes.intrinsic_derivative_valid
        extrinsic_derivative_valid = (
            valid
            & (
                DerivativeSurface.MODEL_PARAMETER
                in self.artifact.differentiation.supported_surfaces
            )
            & (inclination_ > 0.0)
            & (inclination_ < jnp.pi)
        )
        mass_scaling_derivative_valid = jnp.zeros_like(valid)
        qualified = valid & modes.qualified
        time_derivative_valid = (
            valid
            & (
                DerivativeSurface.INPUT
                in self.artifact.differentiation.supported_surfaces
            )
            & ~at_knot
        )
        values = jnp.stack((jnp.real(strain), -jnp.imag(strain)))
        values = jnp.where(valid[None, :], values, jnp.zeros_like(values))
        status = jnp.where(
            valid,
            int(GravitationalWaveStatus.SUCCESS),
            jnp.where(
                ~finite_angles
                | ~finite_samples
                | (modes.status == int(GravitationalWaveStatus.NONFINITE_WAVEFORM)),
                int(GravitationalWaveStatus.NONFINITE_WAVEFORM),
                jnp.where(
                    modes.status != int(GravitationalWaveStatus.SUCCESS),
                    modes.status,
                    int(GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT),
                ),
            ),
        ).astype(jnp.int32)
        return NRSurrogatePolarizations(
            query,
            values,
            support,
            valid,
            qualified,
            status,
            intrinsic_derivative_valid,
            extrinsic_derivative_valid,
            mass_scaling_derivative_valid,
            time_derivative_valid,
            "total-mass-geometric-time",
            "source-frame-geometric-mass",
            self.plan_id,
            self.artifact.frame_id,
            self.artifact.strain_convention,
            self.artifact.time_origin_id,
            self.artifact.mode_normalization,
            self.artifact.amplitude_normalization,
            self.artifact.source_binding_id,
            self.artifact.differentiation.contract_id,
            self.artifact.trust_id,
        )

    def evaluate_physical(
        self,
        time_seconds: ArrayLike,
        parameters: PyTree[ArrayLike],
        /,
        *,
        detector_frame_total_mass_kg: ArrayLike,
        luminosity_distance_m: ArrayLike,
        inclination: ArrayLike,
        azimuth: ArrayLike = 0.0,
        reference_phase: ArrayLike = 0.0,
        frame_angle: ArrayLike = 0.0,
    ) -> NRSurrogatePolarizations:
        """Synthesize observer strain using detector-frame mass in SI units."""

        time_raw = jnp.asarray(time_seconds)
        if jnp.iscomplexobj(time_raw) or jnp.issubdtype(time_raw.dtype, jnp.bool_):
            raise TypeError("Physical surrogate time must be a real numeric vector.")
        time = time_raw.astype(self.artifact.geometric_time.dtype)
        mass = _real_scalar(
            detector_frame_total_mass_kg,
            self.artifact.geometric_time.dtype,
            "detector_frame_total_mass_kg",
        )
        distance = _real_scalar(
            luminosity_distance_m,
            self.artifact.geometric_time.dtype,
            "luminosity_distance_m",
        )
        finite_scaling = jnp.isfinite(mass) & jnp.isfinite(distance)
        physical_scaling = finite_scaling & (mass > 0.0) & (distance > 0.0)
        safe_mass = jnp.where(physical_scaling, mass, 1.0)
        safe_distance = jnp.where(physical_scaling, distance, 1.0)
        mass_time = self.scale.mass_to_geometric_time(safe_mass)
        mass_length = self.scale.mass_to_geometric_length(safe_mass)
        finite_conversion = jnp.isfinite(mass_time) & jnp.isfinite(mass_length)
        safe_mass_time = jnp.where(finite_conversion & (mass_time > 0.0), mass_time, 1.0)
        geometric = self.evaluate_geometric(
            time / safe_mass_time,
            parameters,
            inclination=inclination,
            azimuth=azimuth,
            reference_phase=reference_phase,
            frame_angle=frame_angle,
        )
        candidate_values = geometric.values * mass_length / safe_distance
        finite_values = jnp.all(jnp.isfinite(candidate_values), axis=0)
        valid = (
            geometric.valid
            & physical_scaling
            & finite_conversion
            & (mass_time > 0.0)
            & finite_values
        )
        qualified = valid & geometric.qualified
        intrinsic_derivative_valid = geometric.intrinsic_derivative_valid & valid
        time_derivative_valid = geometric.time_derivative_valid & valid
        extrinsic_derivative_valid = geometric.extrinsic_derivative_valid & valid
        mass_scaling_derivative_valid = time_derivative_valid & valid
        values = jnp.where(
            valid[None, :],
            candidate_values,
            jnp.zeros_like(candidate_values),
        )
        status = jnp.where(
            valid,
            int(GravitationalWaveStatus.SUCCESS),
            jnp.where(
                ~finite_scaling
                | ~finite_conversion
                | ~finite_values
                | (geometric.status == int(GravitationalWaveStatus.NONFINITE_WAVEFORM)),
                int(GravitationalWaveStatus.NONFINITE_WAVEFORM),
                jnp.where(
                    ~physical_scaling | (mass_time <= 0.0),
                    int(GravitationalWaveStatus.OUTSIDE_WAVEFORM_SUPPORT),
                    geometric.status,
                ),
            ),
        ).astype(jnp.int32)
        return NRSurrogatePolarizations(
            time,
            values,
            geometric.support,
            valid,
            qualified,
            status,
            intrinsic_derivative_valid,
            extrinsic_derivative_valid,
            mass_scaling_derivative_valid,
            time_derivative_valid,
            "seconds",
            "detector-frame-redshifted-mass",
            self.plan_id,
            self.artifact.frame_id,
            self.artifact.strain_convention,
            self.artifact.time_origin_id,
            self.artifact.mode_normalization,
            "dimensionless-strain",
            self.artifact.source_binding_id,
            self.artifact.differentiation.contract_id,
            self.artifact.trust_id,
        )


__all__ = [
    "AlignedNRSurrogateArtifact",
    "AlignedNRSurrogatePlan",
    "NRSurrogateModeResult",
    "NRSurrogatePolarizations",
    "NRSurrogateResourcePolicy",
    "PolynomialEmpiricalField",
    "aligned_spin_surrogate_coordinates",
]
