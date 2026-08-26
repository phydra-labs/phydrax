#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..discretization.spectral import HermitianSpectralCoordinates
from ..dynamics import (
    AbstractDifferentiableEvolution,
    ContinuousSystem,
    DiscreteSystem,
    EvolutionStep,
    EvolutionTangentStep,
    StateLayout,
)


class _HermitianContinuousVectorField(StrictModule):
    system: ContinuousSystem
    coordinates: HermitianSpectralCoordinates

    def __call__(self, time: Array, values: Array, args: Any) -> Array:
        state = self.coordinates.from_real_coordinates(values)
        rate = self.system.evaluate(time, state, args)
        return self.coordinates.to_real_coordinates(rate)


class _HermitianDiscreteTransition(StrictModule):
    system: DiscreteSystem
    coordinates: HermitianSpectralCoordinates

    def __call__(self, coordinate: Array, values: Array, args: Any) -> Array:
        state = self.coordinates.from_real_coordinates(values)
        following = self.system.evaluate(coordinate, state, args)
        return self.coordinates.to_real_coordinates(following)


class HermitianCoordinateEvolution(AbstractDifferentiableEvolution):
    """Expose a Hermitian full-complex evolution in independent real coordinates."""

    evolution: AbstractDifferentiableEvolution
    coordinates: HermitianSpectralCoordinates
    system: ContinuousSystem | DiscreteSystem
    evolution_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    discretization_id: str = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)
    tangent_method_id: str = eqx.field(static=True)

    def __init__(
        self,
        evolution: AbstractDifferentiableEvolution,
        coordinates: HermitianSpectralCoordinates,
        /,
    ):
        if not isinstance(evolution, AbstractDifferentiableEvolution):
            raise TypeError("evolution must be an AbstractDifferentiableEvolution.")
        if not isinstance(coordinates, HermitianSpectralCoordinates):
            raise TypeError("coordinates must be HermitianSpectralCoordinates.")
        if evolution.state_layout.shape != coordinates.state_shape:
            raise ValueError("Evolution and Hermitian spectral state shapes must match.")
        if evolution.system.input_layout is not None:
            raise ValueError(
                "HermitianCoordinateEvolution initially requires autonomous dynamics."
            )
        layout = StateLayout(
            (coordinates.coordinate_size,),
            axes=("spectral_coordinate",),
            layout_id=f"state-layout:{coordinates.coordinate_id}",
        )
        if isinstance(evolution.system, ContinuousSystem):
            system: ContinuousSystem | DiscreteSystem = ContinuousSystem(
                _HermitianContinuousVectorField(evolution.system, coordinates),
                state_layout=layout,
                system_id=f"{evolution.system.system_id}:hermitian-coordinates",
            )
        else:
            system = DiscreteSystem(
                _HermitianDiscreteTransition(evolution.system, coordinates),
                state_layout=layout,
                system_id=f"{evolution.system.system_id}:hermitian-coordinates",
            )
        identifier = canonical_fingerprint(
            {
                "kind": "hermitian-coordinate-evolution-v1",
                "evolution": evolution.evolution_id,
                "coordinates": coordinates.coordinate_id,
            }
        )
        self.evolution = evolution
        self.coordinates = coordinates
        self.system = system
        self.evolution_id = identifier
        self.method_id = evolution.method_id
        self.backend_id = evolution.backend_id
        self.discretization_id = evolution.discretization_id
        self.approximation_id = evolution.approximation_id
        self.tangent_method_id = (
            f"{evolution.tangent_method_id}:hermitian-coordinate-conjugation"
        )

    def advance(
        self,
        state: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionStep:
        coordinates = self.coordinates.validate_coordinates(state)
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")
        result = self.evolution.advance(
            self.coordinates.from_real_coordinates(coordinates),
            source,
            target,
            args,
        )
        final_coordinates = self.coordinates.to_real_coordinates(result.final_state)
        return EvolutionStep(
            source_coordinate=source,
            target_coordinate=target,
            final_state=final_coordinates,
            valid=result.valid,
            status=result.status,
            backend_status=result.backend_status,
            system_id=self.system.system_id,
            evolution_id=self.evolution_id,
            method_id=self.method_id,
            backend_id=self.backend_id,
            discretization_id=self.discretization_id,
            approximation_id=self.approximation_id,
        )

    def tangent_action(
        self,
        state: ArrayLike,
        tangent: ArrayLike,
        source_coordinate: ArrayLike,
        target_coordinate: ArrayLike,
        args: Any = None,
        /,
    ) -> EvolutionTangentStep:
        coordinates = self.coordinates.validate_coordinates(state)
        vector = self.coordinates.validate_coordinates(tangent)
        source = jnp.asarray(source_coordinate)
        target = jnp.asarray(target_coordinate)
        if source.shape != () or target.shape != ():
            raise ValueError("Evolution segment coordinates must be scalar.")
        result = self.evolution.tangent_action(
            self.coordinates.from_real_coordinates(coordinates),
            self.coordinates.from_real_coordinates(vector),
            source,
            target,
            args,
        )
        primal = EvolutionStep(
            source_coordinate=source,
            target_coordinate=target,
            final_state=self.coordinates.to_real_coordinates(result.primal.final_state),
            valid=result.primal.valid,
            status=result.primal.status,
            backend_status=result.primal.backend_status,
            system_id=self.system.system_id,
            evolution_id=self.evolution_id,
            method_id=self.method_id,
            backend_id=self.backend_id,
            discretization_id=self.discretization_id,
            approximation_id=self.approximation_id,
        )
        propagated = self.coordinates.to_real_coordinates(result.tangent)
        return EvolutionTangentStep(
            primal=primal,
            tangent=propagated,
            valid=result.valid,
            status=result.status,
            tangent_method_id=self.tangent_method_id,
        )


__all__ = ["HermitianCoordinateEvolution"]
