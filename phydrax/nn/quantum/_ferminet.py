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
from opt_einsum import contract

from ..._doc import DOC_KEY0
from ..._fingerprint import canonical_fingerprint
from ..._precision import real_precision_dtype_name
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...atomistic import AtomicStructure
from ...operators.quantum import LogAmplitude
from ..parameters import PositiveTransform


@jax.custom_jvp
def _stable_signed_product(value: Array, log_scale: Array, /) -> Array:
    """Evaluate ``value * exp(log_scale)`` without a tiny-times-huge product."""
    nonzero = value != 0.0
    safe_magnitude = jnp.where(nonzero, jnp.abs(value), 1.0)
    combined_log_magnitude = jnp.where(
        nonzero, jnp.log(safe_magnitude) + log_scale, 0.0
    )
    return jnp.where(
        nonzero, jnp.sign(value) * jnp.exp(combined_log_magnitude), 0.0
    )


@_stable_signed_product.defjvp
def _stable_signed_product_jvp(primals, tangents):
    value, log_scale = primals
    value_tangent, log_scale_tangent = tangents
    primal = _stable_signed_product(value, log_scale)
    tangent = _stable_signed_product(value_tangent, log_scale)
    tangent = tangent + primal * log_scale_tangent
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
        value = sum(
            (
                (-1.0) ** (power_index - 1)
                * elementary[degree - power_index]
                * traces[power_index - 1]
                for power_index in range(1, degree + 1)
            ),
            jnp.zeros((), dtype=matrix.dtype),
        ) / degree
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
    raw_nonzero = raw_orbitals != 0.0
    safe_raw_magnitude = jnp.where(
        raw_nonzero, jnp.abs(raw_orbitals), jnp.ones((), raw_orbitals.dtype)
    )
    combined_log_magnitude = jnp.where(
        raw_nonzero,
        jnp.log(safe_raw_magnitude) + log_envelope,
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
    scaled_log_envelope = (
        log_envelope - row_shift[:, :, None] - column_shift[:, None, :]
    )
    scaled_orbitals = _stable_signed_product(
        raw_orbitals, scaled_log_envelope
    )
    return scaled_orbitals, row_shift, column_shift


def _scaled_determinant_components(
    raw_orbitals: Array, log_envelope: Array, /
) -> tuple[Array, Array]:
    scaled_orbitals, row_shift, column_shift = _scaled_determinant_factors(
        raw_orbitals, log_envelope
    )
    scaled_determinant = jax.vmap(_polynomial_determinant)(scaled_orbitals)
    determinant_log_scale = jnp.sum(row_shift, axis=-1) + jnp.sum(
        column_shift, axis=-1
    )
    return scaled_determinant, determinant_log_scale


def _scaled_log_determinants(
    raw_orbitals: Array, log_envelope: Array, /
) -> tuple[Array, Array]:
    """Evaluate stable determinant signs and log magnitudes."""
    scaled_determinant, determinant_log_scale = _scaled_determinant_components(
        raw_orbitals, log_envelope
    )
    nonzero = scaled_determinant != 0.0
    safe_magnitude = jnp.where(nonzero, jnp.abs(scaled_determinant), 1.0)
    determinant_log_abs = jnp.where(
        nonzero,
        jnp.log(safe_magnitude) + determinant_log_scale,
        -jnp.inf,
    )
    return jnp.sign(scaled_determinant), determinant_log_abs


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
        if electrons <= 0:
            raise ValueError("electron_count must be positive.")
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
        minimum_decay = max(
            minimum_decay, float(jnp.finfo(jnp.dtype(dtype)).tiny)
        )
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
        orbital_bias = 0.05 * jr.normal(
            keys[-5], (determinants, electrons), dtype=dtype
        )
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
        nuclei = jnp.where(
            active[:, None], self.nuclei.positions.astype(dtype), 0.0
        )
        length_factor = jnp.asarray(
            self.nuclei.scale.length_to_reference, dtype=dtype
        )
        electron_nuclear_squared = jnp.sum(
            (coordinate[:, None, :] - nuclei[None, :, :]) ** 2, axis=-1
        )
        electron_nuclear = (
            jnp.sqrt(
                jnp.where(active[None, :], electron_nuclear_squared, 1.0)
            )
            * length_factor
        )
        electron_nuclear = jnp.where(active[None, :], electron_nuclear, 0.0)
        identity = jnp.eye(self.configuration.electron_count, dtype=bool)
        electron_pair_squared = jnp.sum(
            (coordinate[:, None, :] - coordinate[None, :, :]) ** 2,
            axis=-1,
        )
        electron_pair = (
            jnp.sqrt(jnp.where(identity, 1.0, electron_pair_squared))
            * length_factor
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
        pair_up = jnp.sum(
            jnp.where(pair_up_mask[..., None], pair, 0.0), axis=1
        ) / pair_up_count[:, None]
        pair_down = jnp.sum(
            jnp.where(pair_down_mask[..., None], pair, 0.0), axis=1
        ) / pair_down_count[:, None]
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

        scaled_determinant, determinant_log_scale = (
            _scaled_determinant_components(raw_orbitals, log_envelope)
        )
        determinant_defined = jnp.isfinite(scaled_determinant) & jnp.isfinite(
            determinant_log_scale
        )
        any_determinant = jnp.any(determinant_defined)
        determinant_shift = jax.lax.stop_gradient(
            jnp.max(
                jnp.where(
                    determinant_defined, determinant_log_scale, -jnp.inf
                )
            )
        )
        safe_determinant_shift = jnp.where(any_determinant, determinant_shift, 0.0)
        safe_scaled_determinant = jnp.where(
            determinant_defined, scaled_determinant, 0.0
        )
        relative_log_scale = jnp.where(
            determinant_defined,
            determinant_log_scale - safe_determinant_shift,
            0.0,
        )
        scaled_determinant = _stable_signed_product(
            safe_scaled_determinant, relative_log_scale
        )

        coefficient_scale = jax.lax.stop_gradient(
            jnp.max(jnp.abs(self.determinant_coefficients))
        )
        coefficient_scale_valid = jnp.isfinite(coefficient_scale) & (
            coefficient_scale > 0.0
        )
        safe_coefficient_scale = jnp.where(
            coefficient_scale_valid, coefficient_scale, 1.0
        )
        scaled_sum = jnp.sum(
            (self.determinant_coefficients / safe_coefficient_scale)
            * scaled_determinant
        )
        nonzero = (
            any_determinant
            & coefficient_scale_valid
            & (scaled_sum != 0.0)
            & jnp.isfinite(scaled_sum)
        )
        log_abs = jnp.where(
            nonzero,
            jnp.log(safe_coefficient_scale)
            + safe_determinant_shift
            + jnp.log(jnp.abs(scaled_sum)),
            -jnp.inf,
        )
        phase = jnp.where(scaled_sum < 0.0, -1.0 + 0.0j, 1.0 + 0.0j)
        input_valid = jnp.all(jnp.isfinite(electrons))
        determinant_valid = jnp.all(determinant_defined) & jnp.all(
            jnp.isfinite(self.determinant_coefficients)
        )
        return LogAmplitude(log_abs, phase, valid=input_valid & determinant_valid)

    def __call__(self, electrons: Array, /) -> LogAmplitude:
        """Evaluate one configuration or an arbitrary leading batch of walkers."""
        coordinate = jnp.asarray(electrons)
        if coordinate.ndim < 2 or tuple(coordinate.shape[-2:]) != self.configuration_shape:
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
