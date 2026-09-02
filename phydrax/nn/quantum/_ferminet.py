#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.special as jsp_special
from jaxtyping import Array, Key

from phydrax.ein import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._precision import real_precision_dtype_name
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic._types import AtomicStructure
from ...operators.quantum._amplitude import LogAmplitude
from ...operators.quantum._electronic_advanced import ElectronicVMCResourcePlan
from ..parameters import PositiveTransform


def _signed_log_components(value: Array, /) -> tuple[Array, Array, Array]:
    """Decode signs and log magnitudes without flushing subnormal values."""
    dtype = value.dtype
    if dtype == jnp.dtype(jnp.float16):
        unsigned_dtype = jnp.uint16
        total_bits, fraction_bits, exponent_bits = 16, 10, 5
        exponent_bias, subnormal_offset = 15, -24
    elif dtype == jnp.dtype(jnp.bfloat16):
        unsigned_dtype = jnp.uint16
        total_bits, fraction_bits, exponent_bits = 16, 7, 8
        exponent_bias, subnormal_offset = 127, -133
    elif dtype == jnp.dtype(jnp.float32):
        unsigned_dtype = jnp.uint32
        total_bits, fraction_bits, exponent_bits = 32, 23, 8
        exponent_bias, subnormal_offset = 127, -149
    elif dtype == jnp.dtype(jnp.float64):
        unsigned_dtype = jnp.uint64
        total_bits, fraction_bits, exponent_bits = 64, 52, 11
        exponent_bias, subnormal_offset = 1023, -1074
    else:
        raise TypeError(
            "Stable signed products require float16, bfloat16, float32, or float64."
        )

    bits = jax.lax.bitcast_convert_type(value, unsigned_dtype)
    sign_shift = jnp.asarray(total_bits - 1, dtype=unsigned_dtype)
    fraction_shift = jnp.asarray(fraction_bits, dtype=unsigned_dtype)
    magnitude_mask = jnp.asarray((1 << (total_bits - 1)) - 1, dtype=unsigned_dtype)
    fraction_mask = jnp.asarray((1 << fraction_bits) - 1, dtype=unsigned_dtype)
    exponent_mask = jnp.asarray((1 << exponent_bits) - 1, dtype=unsigned_dtype)
    negative = (bits >> sign_shift) != 0
    magnitude = bits & magnitude_mask
    exponent = (magnitude >> fraction_shift) & exponent_mask
    fraction = magnitude & fraction_mask
    nonzero = magnitude != 0

    safe_fraction = jnp.where(fraction != 0, fraction, 1)
    log_two = jnp.log(jnp.asarray(2.0, dtype=dtype))
    subnormal_log = (
        jnp.log(safe_fraction.astype(dtype))
        + jnp.asarray(subnormal_offset, dtype=dtype) * log_two
    )
    normal_log = (
        jnp.log1p(fraction.astype(dtype) / jnp.asarray(1 << fraction_bits, dtype=dtype))
        + (exponent.astype(dtype) - jnp.asarray(exponent_bias, dtype=dtype)) * log_two
    )
    finite_log = jnp.where(exponent == 0, subnormal_log, normal_log)
    special_log = jnp.log(jnp.abs(value))
    log_magnitude = jnp.where(exponent == exponent_mask, special_log, finite_log)
    return negative, nonzero, log_magnitude


def _stable_signed_product_primal(value: Array, log_scale: Array, /) -> Array:
    negative, nonzero, log_magnitude = _signed_log_components(value)
    combined_log_magnitude = jnp.where(nonzero, log_magnitude + log_scale, 0.0)
    scaled_magnitude = jnp.exp(combined_log_magnitude)
    signed_value = jnp.where(negative, -scaled_magnitude, scaled_magnitude)
    return jnp.where(nonzero, signed_value, jnp.zeros_like(value))


def _apply_linear_stable_signed_product(value: Array, log_scale: Array, /) -> Array:
    def inverse_scale(argument):
        return _stable_signed_product_primal(argument, -log_scale)

    def solve(_inverse_scale, right_hand_side):
        return _stable_signed_product_primal(right_hand_side, log_scale)

    return jax.lax.custom_linear_solve(
        inverse_scale,
        value,
        solve,
        symmetric=True,
    )


@jax.custom_jvp
def _stable_signed_product(value: Array, log_scale: Array, /) -> Array:
    """Evaluate ``value * exp(log_scale)`` without a tiny-times-huge product."""
    return _stable_signed_product_primal(value, log_scale)


@_stable_signed_product.defjvp
def _stable_signed_product_jvp(primals, tangents):
    value, log_scale = primals
    value_tangent, log_scale_tangent = tangents
    primal = _stable_signed_product(value, log_scale)
    value_contribution = _apply_linear_stable_signed_product(value_tangent, log_scale)
    scale_contribution = _apply_linear_stable_signed_bilinear_product(
        value, log_scale_tangent, log_scale
    )
    return primal, value_contribution + scale_contribution


@jax.custom_jvp
def _stable_log_abs(value: Array, /) -> Array:
    """Evaluate ``log(abs(value))`` without flushing subnormal values."""
    _, nonzero, log_magnitude = _signed_log_components(value)
    return jnp.where(nonzero, log_magnitude, -jnp.inf)


@_stable_log_abs.defjvp
def _stable_log_abs_jvp(primals, tangents):
    (value,) = primals
    (value_tangent,) = tangents
    primal = _stable_log_abs(value)
    negative, nonzero, _ = _signed_log_components(value)
    reciprocal_log_scale = jnp.where(nonzero, -primal, 0.0)
    tangent = _apply_linear_stable_signed_product(value_tangent, reciprocal_log_scale)
    signed_tangent = jnp.where(negative, -tangent, tangent)
    return primal, jnp.where(nonzero, signed_tangent, 0.0)


def _stable_signed_bilinear_product_primal(
    left: Array, right: Array, log_scale: Array, /
) -> Array:
    left_negative, left_nonzero, left_log_magnitude = _signed_log_components(left)
    right_negative, right_nonzero, right_log_magnitude = _signed_log_components(right)
    nonzero = left_nonzero & right_nonzero
    combined_log_magnitude = jnp.where(
        nonzero,
        left_log_magnitude + right_log_magnitude + log_scale,
        0.0,
    )
    scaled_magnitude = jnp.exp(combined_log_magnitude)
    signed_value = jnp.where(
        left_negative ^ right_negative, -scaled_magnitude, scaled_magnitude
    )
    return jnp.where(nonzero, signed_value, jnp.zeros_like(left))


@jax.custom_jvp
def _zero_multiplier_linear_product(
    multiplier: Array,
    value: Array,
    log_scale: Array,
    /,
) -> Array:
    """Represent an exact-zero product while retaining its cross derivative."""
    return jnp.zeros_like(value)


@_zero_multiplier_linear_product.defjvp
def _zero_multiplier_linear_product_jvp(primals, tangents):
    _, value, log_scale = primals
    multiplier_tangent, _, _ = tangents
    primal = jnp.zeros_like(value)
    tangent = _apply_linear_stable_signed_bilinear_product(
        value,
        multiplier_tangent,
        log_scale,
    )
    return primal, tangent


def _apply_linear_stable_signed_bilinear_product(
    multiplier: Array, value: Array, log_scale: Array, /
) -> Array:
    negative, nonzero, _ = _signed_log_components(multiplier)
    combined_log_scale = log_scale + _stable_log_abs(multiplier)
    nonzero_value = _apply_linear_stable_signed_product(value, combined_log_scale)
    signed_nonzero_value = jnp.where(negative, -nonzero_value, nonzero_value)
    zero_multiplier = jnp.where(nonzero, jnp.zeros_like(multiplier), multiplier)
    zero_linear_value = jnp.where(nonzero, jnp.zeros_like(value), value)
    zero_value = _zero_multiplier_linear_product(
        zero_multiplier,
        zero_linear_value,
        log_scale,
    )
    return jnp.where(nonzero, signed_nonzero_value, zero_value)


@jax.custom_jvp
def _zero_multiplier_trilinear_product(
    multiplier: Array,
    left: Array,
    value: Array,
    log_scale: Array,
    /,
) -> Array:
    """Represent one exact-zero factor while retaining higher cross derivatives."""
    return jnp.zeros_like(value)


@_zero_multiplier_trilinear_product.defjvp
def _zero_multiplier_trilinear_product_jvp(primals, tangents):
    _, left, value, log_scale = primals
    multiplier_tangent, _, _, _ = tangents
    primal = jnp.zeros_like(value)
    tangent = _apply_linear_stable_signed_trilinear_product(
        left,
        value,
        multiplier_tangent,
        log_scale,
    )
    return primal, tangent


def _apply_linear_stable_signed_trilinear_product(
    left: Array, right: Array, value: Array, log_scale: Array, /
) -> Array:
    right_negative, right_nonzero, _ = _signed_log_components(right)
    right_log_scale = log_scale + _stable_log_abs(right)
    nonzero_right_value = _apply_linear_stable_signed_bilinear_product(
        left, value, right_log_scale
    )
    signed_nonzero_right_value = jnp.where(
        right_negative, -nonzero_right_value, nonzero_right_value
    )
    zero_right = jnp.where(right_nonzero, jnp.zeros_like(right), right)
    zero_left = jnp.where(right_nonzero, jnp.zeros_like(left), left)
    zero_linear_value = jnp.where(right_nonzero, jnp.zeros_like(value), value)
    zero_right_value = _zero_multiplier_trilinear_product(
        zero_right,
        zero_left,
        zero_linear_value,
        log_scale,
    )
    return jnp.where(right_nonzero, signed_nonzero_right_value, zero_right_value)


@jax.custom_jvp
def _stable_signed_bilinear_product(
    left: Array, right: Array, log_scale: Array, /
) -> Array:
    """Evaluate ``left * right * exp(log_scale)`` in a signed log domain."""
    return _stable_signed_bilinear_product_primal(left, right, log_scale)


@_stable_signed_bilinear_product.defjvp
def _stable_signed_bilinear_product_jvp(primals, tangents):
    left, right, log_scale = primals
    left_tangent, right_tangent, log_scale_tangent = tangents
    primal = _stable_signed_bilinear_product(left, right, log_scale)
    left_contribution = _apply_linear_stable_signed_bilinear_product(
        right, left_tangent, log_scale
    )
    right_contribution = _apply_linear_stable_signed_bilinear_product(
        left, right_tangent, log_scale
    )
    scale_contribution = _apply_linear_stable_signed_trilinear_product(
        left, right, log_scale_tangent, log_scale
    )
    tangent = left_contribution + right_contribution + scale_contribution
    return primal, tangent


def _polynomial_determinant(matrix: Array, /) -> Array:
    """Determinant from Newton identities, including derivatives at singularity."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square.")
    size = int(matrix.shape[0])
    power = jnp.eye(size, dtype=matrix.dtype)
    traces = []
    for _ in range(size):
        power = contract("ij,jk->ik", power, matrix)
        traces.append(jnp.trace(power))
    elementary = [jnp.ones((), dtype=matrix.dtype)]
    for degree in range(1, size + 1):
        value = (
            sum(
                (
                    (-1.0) ** (power_index - 1)
                    * elementary[degree - power_index]
                    * traces[power_index - 1]
                    for power_index in range(1, degree + 1)
                ),
                jnp.zeros((), dtype=matrix.dtype),
            )
            / degree
        )
        elementary.append(value)
    return elementary[-1]


def _scaled_determinant_factors(
    raw_orbitals: Array, log_envelope: Array, /
) -> tuple[Array, Array, Array]:
    if (
        raw_orbitals.shape != log_envelope.shape
        or raw_orbitals.ndim != 3
        or raw_orbitals.shape[-2] != raw_orbitals.shape[-1]
    ):
        raise ValueError(
            "raw_orbitals and log_envelope must be matching determinant batches "
            "of square matrices."
        )
    _, raw_nonzero, raw_log_magnitude = _signed_log_components(raw_orbitals)
    combined_log_magnitude = jnp.where(
        raw_nonzero,
        raw_log_magnitude + log_envelope,
        -jnp.inf,
    )
    row_has_entry = jnp.any(raw_nonzero, axis=-1)
    row_shift = jax.lax.stop_gradient(
        jnp.where(
            row_has_entry,
            jnp.max(combined_log_magnitude, axis=-1),
            0.0,
        )
    )
    row_scaled_log_magnitude = combined_log_magnitude - row_shift[:, :, None]
    column_has_entry = jnp.any(raw_nonzero, axis=-2)
    column_shift = jax.lax.stop_gradient(
        jnp.where(
            column_has_entry,
            jnp.max(row_scaled_log_magnitude, axis=-2),
            0.0,
        )
    )
    scaled_log_envelope = log_envelope - row_shift[:, :, None] - column_shift[:, None, :]
    scaled_orbitals = _stable_signed_product(raw_orbitals, scaled_log_envelope)
    return scaled_orbitals, row_shift, column_shift


def _scaled_determinant_components(
    raw_orbitals: Array, log_envelope: Array, /
) -> tuple[Array, Array]:
    scaled_orbitals, row_shift, column_shift = _scaled_determinant_factors(
        raw_orbitals, log_envelope
    )
    scaled_determinant = jax.vmap(_polynomial_determinant)(scaled_orbitals)
    determinant_log_scale = jnp.sum(row_shift, axis=-1) + jnp.sum(column_shift, axis=-1)
    return scaled_determinant, determinant_log_scale


def _scaled_log_determinants(
    raw_orbitals: Array, log_envelope: Array, /
) -> tuple[Array, Array]:
    """Evaluate stable determinant signs and log magnitudes."""
    scaled_determinant, determinant_log_scale = _scaled_determinant_components(
        raw_orbitals, log_envelope
    )
    negative, nonzero, _ = _signed_log_components(scaled_determinant)
    determinant_log_abs = jnp.where(
        nonzero,
        _stable_log_abs(scaled_determinant) + determinant_log_scale,
        -jnp.inf,
    )
    sign = jnp.where(nonzero, jnp.where(negative, -1.0, 1.0), 0.0)
    return sign, determinant_log_abs


def _stable_determinant_mixture(
    scaled_determinant: Array,
    determinant_log_scale: Array,
    coefficients: Array,
    /,
) -> tuple[Array, Array, Array]:
    if (
        scaled_determinant.ndim != 1
        or determinant_log_scale.shape != scaled_determinant.shape
        or coefficients.shape != scaled_determinant.shape
    ):
        raise ValueError(
            "Determinants, log scales, and coefficients must be matching vectors."
        )
    determinant_defined = jnp.isfinite(scaled_determinant) & jnp.isfinite(
        determinant_log_scale
    )
    coefficient_defined = jnp.isfinite(coefficients)
    _, determinant_bit_nonzero, determinant_log_magnitude = _signed_log_components(
        scaled_determinant
    )
    _, coefficient_bit_nonzero, coefficient_log_magnitude = _signed_log_components(
        coefficients
    )
    any_defined = jnp.any(determinant_defined)
    determinant_nonzero = determinant_defined & determinant_bit_nonzero
    active_product = determinant_nonzero & coefficient_defined & coefficient_bit_nonzero
    any_active_product = jnp.any(active_product)
    any_nonzero_determinant = jnp.any(determinant_nonzero)
    determinant_physical_log = jnp.where(
        determinant_nonzero,
        determinant_log_magnitude + determinant_log_scale,
        -jnp.inf,
    )
    active_product_log = jnp.where(
        active_product,
        coefficient_log_magnitude + determinant_physical_log,
        -jnp.inf,
    )
    product_shift = jnp.max(active_product_log)
    zero_coefficient_shift = jnp.max(determinant_physical_log)
    singular_shift = jnp.max(
        jnp.where(determinant_defined, determinant_log_scale, -jnp.inf)
    )
    safe_singular_shift = jnp.where(any_defined, singular_shift, 0.0)
    fallback_shift = jnp.where(
        any_nonzero_determinant, zero_coefficient_shift, safe_singular_shift
    )
    determinant_shift = jax.lax.stop_gradient(
        jnp.where(any_active_product, product_shift, fallback_shift)
    )
    safe_determinant = jnp.where(determinant_defined, scaled_determinant, 0.0)
    safe_coefficient = jnp.where(coefficient_defined, coefficients, 0.0)
    relative_log_scale = jnp.where(
        determinant_defined,
        determinant_log_scale - determinant_shift,
        0.0,
    )
    scaled_terms = _stable_signed_bilinear_product(
        safe_coefficient, safe_determinant, relative_log_scale
    )
    scaled_sum = jnp.sum(scaled_terms)
    scaled_sum_negative, scaled_sum_nonzero, _ = _signed_log_components(scaled_sum)
    nonzero = (
        any_defined
        & jnp.all(coefficient_defined)
        & scaled_sum_nonzero
        & jnp.isfinite(scaled_sum)
    )
    log_abs = jnp.where(
        nonzero,
        determinant_shift + _stable_log_abs(scaled_sum),
        -jnp.inf,
    )
    phase = jnp.where(scaled_sum_negative, -1.0 + 0.0j, 1.0 + 0.0j)
    valid = jnp.all(determinant_defined) & jnp.all(coefficient_defined)
    return log_abs, phase, valid


class _FermiNetConfiguration(StrictModule, NonTrainableState):
    spin_labels: Array
    electron_count: int = eqx.field(static=True)
    spin_up_count: int = eqx.field(static=True)
    spin_down_count: int = eqx.field(static=True)
    hidden_features: int = eqx.field(static=True)
    pair_features: int = eqx.field(static=True)
    layer_count: int = eqx.field(static=True)
    determinant_count: int = eqx.field(static=True)
    compute_dtype: str = eqx.field(static=True)
    minimum_envelope_decay: float = eqx.field(static=True)


class _FermiNetLayer(StrictModule):
    pair_weight: Array
    pair_bias: Array
    one_weight: Array
    one_bias: Array


class FermiNet(StrictModule):
    """Rotation-invariant generalized-determinant molecular FermiNet.

    The static leading electron block is spin-up and the trailing block is
    spin-down. Shared one- and two-electron streams are permutation equivariant
    within each spin block. Full generalized determinants therefore change sign
    under same-spin exchanges without imposing a spatial exchange rule on
    opposite-spin coordinates.
    """

    layers: tuple[_FermiNetLayer, ...]
    orbital_weight: Array
    orbital_bias: Array
    raw_envelope_decay: Array
    envelope_logits: Array
    determinant_coefficients: Array
    nuclei: AtomicStructure
    configuration: _FermiNetConfiguration
    resource_plan: ElectronicVMCResourcePlan
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        nuclei: AtomicStructure,
        electron_count: int,
        spin_up_count: int,
        /,
        *,
        hidden_features: int = 64,
        pair_features: int = 32,
        layer_count: int = 4,
        determinant_count: int = 16,
        compute_dtype: Any = "float64",
        resource_plan: ElectronicVMCResourcePlan | None = None,
        minimum_envelope_decay: float = 1e-6,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        if not isinstance(nuclei, AtomicStructure):
            raise TypeError("nuclei must be an AtomicStructure.")
        if nuclei.has_periodic_metadata:
            raise ValueError("FermiNet supports finite nonperiodic molecules only.")
        electrons = int(electron_count)
        spin_up = int(spin_up_count)
        hidden = int(hidden_features)
        pair_hidden = int(pair_features)
        layers = int(layer_count)
        determinants = int(determinant_count)
        minimum_decay = float(minimum_envelope_decay)
        resource = (
            ElectronicVMCResourcePlan(
                electrons,
                determinant_count=determinants,
            )
            if resource_plan is None
            else resource_plan
        )
        if not isinstance(resource, ElectronicVMCResourcePlan):
            raise TypeError("resource_plan must be ElectronicVMCResourcePlan or None.")
        if (
            resource.electron_count != electrons
            or resource.determinant_count != determinants
        ):
            raise ValueError(
                "FermiNet electron/determinant counts must match resource_plan."
            )
        if spin_up < 0 or spin_up > electrons:
            raise ValueError("spin_up_count must lie between zero and electron_count.")
        if hidden <= 0 or pair_hidden <= 0 or layers <= 0 or determinants <= 0:
            raise ValueError(
                "hidden_features, pair_features, layer_count, and "
                "determinant_count must be positive."
            )
        if not math.isfinite(minimum_decay) or minimum_decay <= 0.0:
            raise ValueError("minimum_envelope_decay must be finite and positive.")
        dtype = real_precision_dtype_name(compute_dtype)
        minimum_decay = max(minimum_decay, float(jnp.finfo(jnp.dtype(dtype)).tiny))
        atom_capacity = int(nuclei.atomic_numbers.shape[0])
        one_input = 2 * atom_capacity + 2
        pair_input = 2
        keys = jr.split(key, 2 * layers + 6)
        modules: list[_FermiNetLayer] = []
        current_one = one_input
        current_pair = pair_input
        for index in range(layers):
            pair_weight = jr.normal(
                keys[2 * index], (current_pair, pair_hidden), dtype=dtype
            ) / jnp.sqrt(jnp.asarray(current_pair, dtype=dtype))
            pair_bias = jnp.zeros((pair_hidden,), dtype=dtype)
            aggregated = 3 * current_one + 2 * pair_hidden
            one_weight = jr.normal(
                keys[2 * index + 1], (aggregated, hidden), dtype=dtype
            ) / jnp.sqrt(jnp.asarray(aggregated, dtype=dtype))
            one_bias = jnp.zeros((hidden,), dtype=dtype)
            modules.append(
                _FermiNetLayer(
                    pair_weight=pair_weight,
                    pair_bias=pair_bias,
                    one_weight=one_weight,
                    one_bias=one_bias,
                )
            )
            current_one = hidden
            current_pair = pair_hidden
        orbital_weight = jr.normal(
            keys[-6], (determinants, hidden, electrons), dtype=dtype
        ) / jnp.sqrt(jnp.asarray(hidden, dtype=dtype))
        orbital_bias = 0.05 * jr.normal(keys[-5], (determinants, electrons), dtype=dtype)
        inverse_softplus_one = jnp.log(jnp.expm1(jnp.asarray(1.0, dtype=dtype)))
        raw_envelope_decay = inverse_softplus_one + 0.05 * jr.normal(
            keys[-4], (determinants, electrons, atom_capacity), dtype=dtype
        )
        envelope_logits = 0.05 * jr.normal(
            keys[-3], (determinants, electrons, atom_capacity), dtype=dtype
        )
        determinant_coefficients = jnp.ones((determinants,), dtype=dtype) / jnp.sqrt(
            jnp.asarray(determinants, dtype=dtype)
        )
        spin_labels = jnp.concatenate(
            (
                jnp.zeros((spin_up,), dtype=jnp.int32),
                jnp.ones((electrons - spin_up,), dtype=jnp.int32),
            )
        )
        self.layers = tuple(modules)
        self.orbital_weight = orbital_weight
        self.orbital_bias = orbital_bias
        self.raw_envelope_decay = raw_envelope_decay
        self.envelope_logits = envelope_logits
        self.determinant_coefficients = determinant_coefficients
        self.nuclei = nuclei
        self.resource_plan = resource
        self.configuration = _FermiNetConfiguration(
            spin_labels=spin_labels,
            electron_count=electrons,
            spin_up_count=spin_up,
            spin_down_count=electrons - spin_up,
            hidden_features=hidden,
            pair_features=pair_hidden,
            layer_count=layers,
            determinant_count=determinants,
            compute_dtype=dtype,
            minimum_envelope_decay=minimum_decay,
        )
        self.network_id = canonical_fingerprint(
            {
                "kind": "ferminet",
                "structure": nuclei.structure_id,
                "scale": nuclei.scale.scale_id,
                "electron_count": electrons,
                "spin_up_count": spin_up,
                "hidden_features": hidden,
                "pair_features": pair_hidden,
                "layer_count": layers,
                "determinant_count": determinants,
                "compute_dtype": dtype,
                "minimum_envelope_decay": minimum_decay,
                "determinant_mode": "full-generalized",
            }
        )

    @property
    def configuration_shape(self) -> tuple[int, int]:
        return (self.configuration.electron_count, 3)

    @property
    def envelope_decay(self) -> Array:
        """Strictly positive physical decay parameters for every envelope."""
        return PositiveTransform(self.configuration.minimum_envelope_decay)(
            self.raw_envelope_decay
        )

    def _distances(self, electrons: Array, /) -> tuple[Array, Array]:
        dtype = jnp.dtype(self.configuration.compute_dtype)
        coordinate = jnp.asarray(electrons, dtype=dtype)
        active = self.nuclei.active_mask
        nuclei = jnp.where(active[:, None], self.nuclei.positions.astype(dtype), 0.0)
        length_factor = jnp.asarray(self.nuclei.scale.length_to_reference, dtype=dtype)
        electron_nuclear_squared = jnp.sum(
            (coordinate[:, None, :] - nuclei[None, :, :]) ** 2, axis=-1
        )
        electron_nuclear = (
            jnp.sqrt(jnp.where(active[None, :], electron_nuclear_squared, 1.0))
            * length_factor
        )
        electron_nuclear = jnp.where(active[None, :], electron_nuclear, 0.0)
        identity = jnp.eye(self.configuration.electron_count, dtype=bool)
        electron_pair_squared = jnp.sum(
            (coordinate[:, None, :] - coordinate[None, :, :]) ** 2,
            axis=-1,
        )
        electron_pair = (
            jnp.sqrt(jnp.where(identity, 1.0, electron_pair_squared)) * length_factor
        )
        electron_pair = jnp.where(identity, 0.0, electron_pair)
        return electron_nuclear, electron_pair

    def _aggregate(self, one: Array, pair: Array, /) -> Array:
        labels = self.configuration.spin_labels
        electrons = self.configuration.electron_count
        identity = jnp.eye(electrons, dtype=bool)
        up = labels == 0
        down = labels == 1
        up_count = jnp.maximum(jnp.sum(up), 1)
        down_count = jnp.maximum(jnp.sum(down), 1)
        global_up = jnp.sum(jnp.where(up[:, None], one, 0.0), axis=0) / up_count
        global_down = jnp.sum(jnp.where(down[:, None], one, 0.0), axis=0) / down_count
        pair_up_mask = (~identity) & up[None, :]
        pair_down_mask = (~identity) & down[None, :]
        pair_up_count = jnp.maximum(jnp.sum(pair_up_mask, axis=-1), 1)
        pair_down_count = jnp.maximum(jnp.sum(pair_down_mask, axis=-1), 1)
        pair_up = (
            jnp.sum(jnp.where(pair_up_mask[..., None], pair, 0.0), axis=1)
            / pair_up_count[:, None]
        )
        pair_down = (
            jnp.sum(jnp.where(pair_down_mask[..., None], pair, 0.0), axis=1)
            / pair_down_count[:, None]
        )
        return jnp.concatenate(
            (
                one,
                jnp.broadcast_to(global_up, one.shape),
                jnp.broadcast_to(global_down, one.shape),
                pair_up,
                pair_down,
            ),
            axis=-1,
        )

    def _single(self, electrons: Array, /) -> LogAmplitude:
        if electrons.shape != self.configuration_shape:
            raise ValueError(
                "FermiNet configurations must have shape "
                f"{self.configuration_shape}; got {electrons.shape}."
            )
        electron_nuclear, electron_pair = self._distances(electrons)
        active = self.nuclei.active_mask
        spin_one_hot = jax.nn.one_hot(
            self.configuration.spin_labels,
            2,
            dtype=jnp.dtype(self.configuration.compute_dtype),
        )
        one = jnp.concatenate(
            (
                electron_nuclear,
                jnp.exp(-electron_nuclear) * active[None, :],
                spin_one_hot,
            ),
            axis=-1,
        )
        pair = jnp.stack((electron_pair, jnp.exp(-electron_pair)), axis=-1)
        for layer in self.layers:
            pair = jnp.tanh(pair @ layer.pair_weight + layer.pair_bias)
            one = jnp.tanh(self._aggregate(one, pair) @ layer.one_weight + layer.one_bias)

        raw_orbitals = contract("if,dfj->dij", one, self.orbital_weight)
        raw_orbitals = raw_orbitals + self.orbital_bias[:, None, :]
        decay = self.envelope_decay
        masked_logits = jnp.where(active[None, None, :], self.envelope_logits, -jnp.inf)
        log_mixing = jax.nn.log_softmax(masked_logits, axis=-1)
        log_envelope = jsp_special.logsumexp(
            log_mixing[:, :, None, :]
            - decay[:, :, None, :] * electron_nuclear[None, None, :, :],
            axis=-1,
        )
        log_envelope = jnp.swapaxes(log_envelope, 1, 2)

        scaled_determinant, determinant_log_scale = _scaled_determinant_components(
            raw_orbitals, log_envelope
        )
        log_abs, phase, determinant_valid = _stable_determinant_mixture(
            scaled_determinant,
            determinant_log_scale,
            self.determinant_coefficients,
        )
        input_valid = jnp.all(jnp.isfinite(electrons))
        return LogAmplitude(log_abs, phase, valid=input_valid & determinant_valid)

    def __call__(self, electrons: Array, /) -> LogAmplitude:
        """Evaluate one configuration or an arbitrary leading batch of walkers."""
        coordinate = jnp.asarray(electrons)
        if (
            coordinate.ndim < 2
            or tuple(coordinate.shape[-2:]) != self.configuration_shape
        ):
            raise ValueError(
                "FermiNet inputs must end in shape "
                f"{self.configuration_shape}; got {coordinate.shape}."
            )
        if coordinate.ndim == 2:
            return self._single(coordinate)
        batch_shape = tuple(int(size) for size in coordinate.shape[:-2])
        flat_count = math.prod(batch_shape)
        values = jax.vmap(self._single)(
            coordinate.reshape((flat_count,) + self.configuration_shape)
        )
        return LogAmplitude(
            values.log_abs.reshape(batch_shape),
            values.phase.reshape(batch_shape),
            valid=values.valid.reshape(batch_shape),
        )


__all__ = ["FermiNet"]
