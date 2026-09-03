#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._gas_dynamics import (
    HomogeneousMixtureCompressibleNavierStokesSystem,
    HomogeneousMixtureEulerSystem,
)


CompressibleWaveKind: TypeAlias = Literal[
    "isentropic", "acoustic", "entropy", "vorticity"
]


class CompressibleReferenceWaveEvidence(StrictModule):
    primitive: Array
    conserved: Array
    pressure_relation_residual: Array
    transverse_velocity_residual: Array
    characteristic_identity_residual: Array
    entropy_supported: Array
    admissible: Array
    finite: Array
    wave_id: str = eqx.field(static=True)


class CompressibleReferenceWavePlan(StrictModule, NonTrainableState):
    """Small canonical characteristic wave about one homogeneous gas state."""

    base_primitive: Array
    wave_vector: tuple[float, ...] = eqx.field(static=True)
    kind: CompressibleWaveKind = eqx.field(static=True)
    amplitude: float = eqx.field(static=True)
    propagation_sign: int = eqx.field(static=True)
    wave_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: CompressibleWaveKind,
        base_primitive: ArrayLike,
        wave_vector: Sequence[float],
        /,
        *,
        amplitude: float = 1.0e-3,
        propagation_sign: int = 1,
    ):
        base = jnp.asarray(base_primitive)
        wave = tuple(float(value) for value in wave_vector)
        amplitude_ = float(amplitude)
        sign = int(propagation_sign)
        if (
            kind not in ("isentropic", "acoustic", "entropy", "vorticity")
            or base.ndim != 1
            or len(wave) not in (1, 2, 3)
            or any(not np.isfinite(value) for value in (*np.asarray(base), *wave))
            or np.linalg.norm(np.asarray(wave)) <= 0.0
            or not np.isfinite(amplitude_)
            or amplitude_ <= 0.0
            or sign not in (-1, 1)
        ):
            raise ValueError("Compressible reference-wave parameters are invalid.")
        self.kind = kind
        self.base_primitive = base
        self.wave_vector = wave
        self.amplitude = amplitude_
        self.propagation_sign = sign
        self.wave_id = canonical_fingerprint(
            {
                "kind": "canonical-compressible-reference-wave",
                "wave_kind": kind,
                "base_primitive": tuple(float(value) for value in np.asarray(base)),
                "wave_vector": wave,
                "amplitude": amplitude_,
                "propagation_sign": sign,
            }
        )

    @property
    def dimension(self) -> int:
        return len(self.wave_vector)

    def _mode(
        self, system: HomogeneousMixtureEulerSystem, /
    ) -> tuple[Array, Array, Array, Array, Array]:
        if not isinstance(system, HomogeneousMixtureEulerSystem):
            raise TypeError("Reference waves require HomogeneousMixtureEulerSystem.")
        if system.dimension != self.dimension or self.base_primitive.shape != (
            system.component_count,
        ):
            raise ValueError("Reference-wave base state does not match the gas system.")
        base = system.primitive_to_conserved(self.base_primitive)
        if not bool(system.admissible(base) & system.entropy_evidence(base)):
            raise ValueError(
                "Reference-wave base state lacks admissibility and entropy evidence."
            )
        wave = jnp.asarray(self.wave_vector, dtype=base.dtype)
        wave_norm = jnp.sqrt(contract("d,d->", wave, wave, backend="jax"))
        direction = wave / wave_norm
        left, right, speeds = system.normal_eigensystem(base, base, direction)
        if self.kind in ("isentropic", "acoustic"):
            mode = 0 if self.propagation_sign < 0 else system.component_count - 1
        elif self.kind == "entropy":
            mode = 1
        else:
            if system.dimension == 1:
                raise ValueError(
                    "Vorticity reference waves require at least two dimensions."
                )
            mode = 1 + system.species_count
        identity = contract("ij,jk->ik", left, right, backend="jax") - jnp.eye(
            system.component_count, dtype=base.dtype
        )
        return base, right[:, mode], speeds[mode], direction, identity

    def conserved(
        self,
        system: HomogeneousMixtureEulerSystem,
        coordinates: ArrayLike,
        time: ArrayLike,
        /,
    ) -> Array:
        base, mode, speed, _, _ = self._mode(system)
        points = jnp.asarray(coordinates)
        if points.shape[-1:] != (self.dimension,):
            raise ValueError("Reference-wave coordinates have the wrong dimension.")
        time_value = jnp.asarray(time, dtype=points.dtype)
        wave = jnp.asarray(self.wave_vector, dtype=points.dtype)
        phase = contract("...d,d->...", points, wave, backend="jax") - (
            speed * jnp.sqrt(contract("d,d->", wave, wave, backend="jax")) * time_value
        )
        if self.kind in ("isentropic", "acoustic"):
            scale = self.amplitude / system.density(base)
        else:
            scale = self.amplitude
        return base + scale * jnp.cos(phase)[..., None] * mode

    def primitive(
        self,
        system: HomogeneousMixtureEulerSystem,
        coordinates: ArrayLike,
        time: ArrayLike,
        /,
    ) -> Array:
        return system.conserved_to_primitive(self.conserved(system, coordinates, time))

    def evaluate(
        self,
        system: HomogeneousMixtureEulerSystem,
        coordinates: ArrayLike,
        time: ArrayLike,
        /,
    ) -> CompressibleReferenceWaveEvidence:
        base, _, _, direction, identity = self._mode(system)
        conserved = self.conserved(system, coordinates, time)
        primitive = system.conserved_to_primitive(conserved)
        recovered = system.recover_thermodynamics(conserved)
        base_recovered = system.recover_thermodynamics(base)
        density_delta = system.density(conserved) - system.density(base)
        pressure_delta = recovered.state.pressure - base_recovered.state.pressure
        if self.kind in ("isentropic", "acoustic"):
            pressure_residual = pressure_delta - (
                base_recovered.state.frozen_sound_speed_squared * density_delta
            )
        else:
            pressure_residual = pressure_delta
        velocity = primitive[..., system.species_count : -1]
        base_velocity = self.base_primitive[system.species_count : -1]
        fluctuation = velocity - base_velocity
        longitudinal = (
            contract("...d,d->...", fluctuation, direction, backend="jax")[..., None]
            * direction
        )
        transverse = fluctuation - longitudinal
        transverse_residual = longitudinal if self.kind == "vorticity" else transverse
        return CompressibleReferenceWaveEvidence(
            primitive,
            conserved,
            pressure_residual,
            transverse_residual,
            jnp.max(jnp.abs(identity)),
            system.entropy_evidence(conserved),
            system.admissible(conserved),
            jnp.all(jnp.isfinite(conserved)),
            self.wave_id,
        )


class ManufacturedViscousNSEvidence(StrictModule):
    state: Array
    forcing: Array
    temporal_rate: Array
    inviscid_divergence: Array
    viscous_divergence: Array
    identity_residual: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class ManufacturedViscousNSPlan(StrictModule, NonTrainableState):
    """Automatic strong-form forcing for a smooth canonical mixture NS state."""

    exact_state: Callable = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    exact_state_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dimension: int,
        exact_state: Callable[[Array, Array, Any], Array],
        exact_state_id: str,
        /,
    ):
        dimension_ = int(dimension)
        identifier = str(exact_state_id)
        if dimension_ not in (1, 2, 3) or not callable(exact_state) or not identifier:
            raise ValueError("Manufactured viscous-NS plan inputs are invalid.")
        self.dimension = dimension_
        self.exact_state = exact_state
        self.exact_state_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "manufactured-viscous-canonical-mixture-ns",
                "dimension": dimension_,
                "exact_state": identifier,
            }
        )

    def evaluate(
        self,
        system: HomogeneousMixtureCompressibleNavierStokesSystem,
        time: ArrayLike,
        coordinates: ArrayLike,
        args: Any = None,
        /,
    ) -> ManufacturedViscousNSEvidence:
        if (
            not isinstance(system, HomogeneousMixtureCompressibleNavierStokesSystem)
            or system.dimension != self.dimension
        ):
            raise TypeError(
                "Manufactured viscous NS requires a matching canonical mixture system."
            )
        time_value = jnp.asarray(time)
        points = jnp.asarray(coordinates)
        if time_value.shape != () or points.shape[-1:] != (self.dimension,):
            raise ValueError("Manufactured NS time/coordinate shapes are invalid.")
        flat = points.reshape((-1, self.dimension))

        def point_terms(point):
            def state_at_time(local_time):
                return jnp.asarray(self.exact_state(local_time, point, args))

            def state_at_point(local_point):
                return jnp.asarray(self.exact_state(time_value, local_point, args))

            state = state_at_point(point)
            temporal = jax.jacfwd(state_at_time)(time_value)

            def inviscid_tensor(local_point):
                local_state = state_at_point(local_point)
                return jnp.stack(
                    tuple(
                        system.physical_flux(local_state, axis, args)
                        for axis in range(self.dimension)
                    ),
                    axis=-1,
                )

            inviscid_gradient = jax.jacfwd(inviscid_tensor)(point)
            inviscid_divergence = jnp.trace(inviscid_gradient, axis1=-2, axis2=-1)

            def viscous_tensor(local_point):
                local_state = state_at_point(local_point)
                conserved_gradient = jax.jacfwd(state_at_point)(local_point)
                return system.viscous_flux(local_state, conserved_gradient, args)

            viscous_gradient = jax.jacfwd(viscous_tensor)(point)
            viscous_divergence = jnp.trace(viscous_gradient, axis1=-2, axis2=-1)
            forcing = temporal + inviscid_divergence - viscous_divergence
            identity = temporal + inviscid_divergence - viscous_divergence - forcing
            return (
                state,
                forcing,
                temporal,
                inviscid_divergence,
                viscous_divergence,
                identity,
            )

        values = jax.vmap(point_terms)(flat)
        output_shape = points.shape[:-1] + (system.component_count,)
        reshaped = tuple(value.reshape(output_shape) for value in values)
        return ManufacturedViscousNSEvidence(
            *reshaped,
            jnp.all(jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in reshaped))),
            self.plan_id,
        )

    __call__ = evaluate


__all__ = [
    "CompressibleReferenceWaveEvidence",
    "CompressibleReferenceWavePlan",
    "ManufacturedViscousNSEvidence",
    "ManufacturedViscousNSPlan",
]
