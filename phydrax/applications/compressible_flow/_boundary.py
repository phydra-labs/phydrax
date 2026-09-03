#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
    HomogeneousMixtureEulerSystem,
)
from ...equations._hyperbolic_systems import AbstractNormalCharacteristicSystem


class CharacteristicReflectionLedger(StrictModule):
    incoming_characteristics: Array
    outgoing_characteristics: Array
    reflected_characteristics: Array
    incoming_energy: Array
    reflected_energy: Array
    reflection_coefficient: Array
    admissible: Array
    boundary_id: str = eqx.field(static=True)


class CharacteristicBoundaryResult(StrictModule):
    boundary_state: Array
    eigenvalues: Array
    ledger: CharacteristicReflectionLedger


class CharacteristicNonreflectingBoundaryPlan(StrictModule, NonTrainableState):
    """Freeze incoming characteristics to far field and pass outgoing waves."""

    relaxation: float = eqx.field(static=True)
    sonic_tolerance: float = eqx.field(static=True)
    boundary_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        relaxation: float = 1.0,
        sonic_tolerance: float = 1.0e-10,
    ):
        relaxation_ = float(relaxation)
        tolerance = float(sonic_tolerance)
        if (
            not np.isfinite(relaxation_)
            or not 0.0 <= relaxation_ <= 1.0
            or not np.isfinite(tolerance)
            or tolerance < 0.0
        ):
            raise ValueError("Characteristic boundary parameters are invalid.")
        self.relaxation = relaxation_
        self.sonic_tolerance = tolerance
        self.boundary_id = canonical_fingerprint(
            {
                "kind": "compressible-characteristic-nonreflecting-boundary",
                "relaxation": relaxation_,
                "sonic_tolerance": tolerance,
            }
        )

    def apply(
        self,
        system: AbstractNormalCharacteristicSystem,
        interior_state: ArrayLike,
        far_field_state: ArrayLike,
        outward_normal: ArrayLike,
        args=None,
        /,
    ) -> CharacteristicBoundaryResult:
        if not isinstance(system, AbstractNormalCharacteristicSystem):
            raise TypeError("Characteristic boundary requires normal characteristics.")
        interior = jnp.asarray(interior_state)
        far_field = jnp.asarray(far_field_state, dtype=interior.dtype)
        normal = jnp.asarray(outward_normal, dtype=interior.dtype)
        if interior.shape != far_field.shape or interior.ndim < 1:
            raise ValueError("Interior and far-field states must have equal shape.")
        if normal.shape != interior.shape[:-1] + (system.dimension,):
            raise ValueError("Boundary normal shape is incompatible with the state.")
        normal_norm = jnp.sqrt(
            oe.contract("...d,...d->...", normal, normal, backend="jax")
        )
        normal = (
            eqx.error_if(
                normal,
                jnp.any(~jnp.isfinite(normal_norm) | (normal_norm <= 0.0)),
                "Boundary normals must be finite and nonzero.",
            )
            / normal_norm[..., None]
        )
        left, right, eigenvalues = system.normal_eigensystem(
            interior, far_field, normal, args
        )
        expected_matrix = interior.shape[:-1] + (
            interior.shape[-1],
            interior.shape[-1],
        )
        if left.shape != expected_matrix or right.shape != expected_matrix:
            raise ValueError("Normal characteristic matrices have incompatible shapes.")
        delta = far_field - interior
        characteristics = oe.contract("...ij,...j->...i", left, delta, backend="jax")
        incoming_mask = eigenvalues < -self.sonic_tolerance
        outgoing_mask = eigenvalues > self.sonic_tolerance
        incoming = jnp.where(incoming_mask, characteristics, 0.0)
        correction_characteristics = self.relaxation * incoming
        correction = oe.contract(
            "...ij,...j->...i", right, correction_characteristics, backend="jax"
        )
        boundary = interior + correction
        reconstructed = oe.contract(
            "...ij,...j->...i", left, boundary - interior, backend="jax"
        )
        reflected = jnp.where(outgoing_mask, reconstructed, 0.0)
        incoming_energy = 0.5 * jnp.sum(incoming * incoming, axis=-1)
        reflected_energy = 0.5 * jnp.sum(reflected * reflected, axis=-1)
        coefficient = jnp.where(
            incoming_energy > 0.0,
            jnp.sqrt(reflected_energy / incoming_energy),
            jnp.zeros_like(incoming_energy),
        )
        admissible = system.admissible(boundary)
        boundary = eqx.error_if(
            boundary,
            jnp.any(~admissible | ~jnp.isfinite(boundary)),
            "Characteristic boundary produced an inadmissible state.",
        )
        ledger = CharacteristicReflectionLedger(
            incoming,
            jnp.where(outgoing_mask, characteristics, 0.0),
            reflected,
            incoming_energy,
            reflected_energy,
            coefficient,
            admissible,
            self.boundary_id,
        )
        return CharacteristicBoundaryResult(boundary, eigenvalues, ledger)

    __call__ = apply


class CompressibleSpongeLedger(StrictModule):
    profile: Array
    conservative_rate: Array
    mass_rate: Array
    species_mass_rate: Array
    momentum_rate: Array
    total_energy_rate: Array
    entropy_rate: Array
    fluctuation_energy_before: Array
    fluctuation_energy_after: Array
    reflection_coefficient: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class CompressibleSpongeResult(StrictModule):
    source: Array
    predicted_state: Array
    ledger: CompressibleSpongeLedger


class CompressibleSpongePlan(StrictModule, NonTrainableState):
    """Conserved-variable relaxation with conservative and entropy ledgers."""

    system: (
        HomogeneousMixtureEulerSystem | HomogeneousMixtureCompressibleNavierStokesSystem
    )
    target_state: Array
    strength: float = eqx.field(static=True)
    start_coordinate: float = eqx.field(static=True)
    end_coordinate: float = eqx.field(static=True)
    profile_power: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: HomogeneousMixtureEulerSystem
        | HomogeneousMixtureCompressibleNavierStokesSystem,
        target_state: ArrayLike,
        /,
        *,
        strength: float,
        start_coordinate: float,
        end_coordinate: float,
        profile_power: float = 2.0,
    ):
        target = jnp.asarray(target_state)
        strength_ = float(strength)
        start = float(start_coordinate)
        end = float(end_coordinate)
        power = float(profile_power)
        if (
            not isinstance(
                system,
                (
                    HomogeneousMixtureEulerSystem,
                    HomogeneousMixtureCompressibleNavierStokesSystem,
                ),
            )
            or target.ndim != 1
            or target.shape[0] != system.component_count
            or not np.all(np.isfinite(np.asarray(target)))
            or not np.isfinite(strength_)
            or strength_ <= 0.0
            or not np.isfinite(start)
            or not np.isfinite(end)
            or end <= start
            or not np.isfinite(power)
            or power <= 0.0
        ):
            raise ValueError("Compressible sponge parameters are invalid.")
        if not bool(system.admissible(target)):
            raise ValueError("Compressible sponge target must be canonically admissible.")
        self.system = system
        self.target_state = target
        self.strength = strength_
        self.start_coordinate = start
        self.end_coordinate = end
        self.profile_power = power
        self.plan_id = canonical_fingerprint(
            {
                "kind": "compressible-conservative-sponge",
                "system": system.system_id,
                "target": array_tree_fingerprint(target),
                "strength": strength_,
                "start_coordinate": start,
                "end_coordinate": end,
                "profile_power": power,
            }
        )

    @property
    def dimension(self) -> int:
        return self.system.dimension

    def profile(self, coordinate: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinate)
        fraction = jnp.clip(
            (value - self.start_coordinate)
            / (self.end_coordinate - self.start_coordinate),
            0.0,
            1.0,
        )
        return self.strength * fraction**self.profile_power

    def apply(
        self,
        conserved: ArrayLike,
        coordinate: ArrayLike,
        /,
        *,
        step_size: ArrayLike = 0.0,
        weights: ArrayLike | None = None,
    ) -> CompressibleSpongeResult:
        system = self.system
        state = jnp.asarray(conserved)
        if state.ndim < 1 or state.shape[-1] != system.component_count:
            raise ValueError("Sponge state has the wrong component count.")
        spatial_shape = state.shape[:-1]
        coordinate_ = jnp.asarray(coordinate, dtype=state.dtype)
        if coordinate_.shape != spatial_shape:
            raise ValueError("Sponge coordinate must match the state grid.")
        profile = self.profile(coordinate_)
        target = jnp.broadcast_to(self.target_state.astype(state.dtype), state.shape)
        state = eqx.error_if(
            state,
            jnp.any(~system.admissible(state))
            | jnp.any(~system.admissible(target))
            | jnp.any(~jnp.isfinite(state)),
            "Sponge state and target must be finite and admissible.",
        )
        source = -profile[..., None] * (state - target)
        step = jnp.asarray(step_size, dtype=state.dtype)
        if step.shape != ():
            raise ValueError("Sponge step_size must be scalar.")
        step = eqx.error_if(
            step,
            ~jnp.isfinite(step) | (step < 0.0),
            "Sponge step_size must be finite and nonnegative.",
        )
        predicted = state + step * source
        weights_ = (
            jnp.ones(spatial_shape, dtype=state.dtype)
            if weights is None
            else jnp.asarray(weights, dtype=state.dtype)
        )
        if weights_.shape != spatial_shape:
            raise ValueError("Sponge weights must match the state grid.")
        weights_ = eqx.error_if(
            weights_,
            jnp.any(~jnp.isfinite(weights_) | (weights_ < 0.0)),
            "Sponge weights must be finite and nonnegative.",
        )
        axes = tuple(range(len(spatial_shape)))
        integrated = source * weights_[..., None]
        for axis in sorted(axes, reverse=True):
            integrated = jnp.sum(integrated, axis=axis)
        entropy_variables = system.entropy_variables(state)
        entropy_density = oe.contract(
            "...i,...i->...", entropy_variables, source, backend="jax"
        )
        entropy_integral = entropy_density * weights_
        before_density = 0.5 * jnp.sum((state - target) ** 2, axis=-1)
        after_density = 0.5 * jnp.sum((predicted - target) ** 2, axis=-1)
        before = before_density * weights_
        after = after_density * weights_
        for axis in sorted(axes, reverse=True):
            entropy_integral = jnp.sum(entropy_integral, axis=axis)
            before = jnp.sum(before, axis=axis)
            after = jnp.sum(after, axis=axis)
        reflection = jnp.where(
            before > 0.0,
            jnp.sqrt(jnp.maximum(after, 0.0) / before),
            jnp.zeros_like(before),
        )
        finite = (
            jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(predicted))
            & jnp.all(system.admissible(predicted))
        )
        ledger = CompressibleSpongeLedger(
            profile,
            integrated,
            jnp.sum(integrated[: system.species_count]),
            integrated[: system.species_count],
            integrated[system.species_count : -1],
            integrated[-1],
            entropy_integral,
            before,
            after,
            reflection,
            finite,
            self.plan_id,
        )
        return CompressibleSpongeResult(source, predicted, ledger)

    __call__ = apply


__all__ = [
    "CharacteristicBoundaryResult",
    "CharacteristicNonreflectingBoundaryPlan",
    "CharacteristicReflectionLedger",
    "CompressibleSpongeLedger",
    "CompressibleSpongePlan",
    "CompressibleSpongeResult",
]
