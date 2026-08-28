#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._numerics._compensated import compensated_sum
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import ParticleDiscretization


ParticlePopulationRole: TypeAlias = Literal[
    "dynamic-fluid",
    "static-boundary",
    "prescribed-boundary",
    "dynamic-rigid",
    "material-phase",
]


class ParticlePopulation(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    population_id: str = eqx.field(static=True)
    role: ParticlePopulationRole = eqx.field(static=True)
    particles: ParticleDiscretization
    state_shape: tuple[int, ...] | None = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        particles: ParticleDiscretization,
        /,
        *,
        role: ParticlePopulationRole,
        state_shape: Sequence[int] | None = None,
        population_id: str | None = None,
    ):
        name_ = str(name)
        if not name_:
            raise ValueError("Particle population name must be non-empty.")
        if not isinstance(particles, ParticleDiscretization):
            raise TypeError("particles must be a ParticleDiscretization.")
        allowed = (
            "dynamic-fluid",
            "static-boundary",
            "prescribed-boundary",
            "dynamic-rigid",
            "material-phase",
        )
        if role not in allowed:
            raise ValueError("Unknown particle population role.")
        shape = None if state_shape is None else tuple(int(size) for size in state_shape)
        if shape is not None and (not shape or any(size <= 0 for size in shape)):
            raise ValueError("state_shape must contain positive dimensions or be None.")
        dynamic = role in ("dynamic-fluid", "dynamic-rigid", "material-phase")
        if dynamic != (shape is not None):
            raise ValueError(
                "Dynamic populations require state_shape; static roles forbid it."
            )
        generated = canonical_fingerprint(
            {
                "kind": "particle-population",
                "name": name_,
                "particles": particles.prepared_id,
                "role": role,
                "state_shape": None if shape is None else list(shape),
            }
        )
        identifier = generated if population_id is None else str(population_id)
        if not identifier:
            raise ValueError("population_id must be non-empty.")
        self.name = name_
        self.population_id = identifier
        self.role = role
        self.particles = particles
        self.state_shape = shape

    @property
    def dynamic(self) -> bool:
        return self.state_shape is not None


class ParticleInteractionKey(StrictModule, NonTrainableState):
    target_population_id: str = eqx.field(static=True)
    source_population_id: str = eqx.field(static=True)
    interaction_kind: str = eqx.field(static=True)
    reciprocal: bool = eqx.field(static=True)
    key_id: str = eqx.field(static=True)

    def __init__(
        self,
        target_population: ParticlePopulation,
        source_population: ParticlePopulation,
        interaction_kind: str,
        /,
        *,
        reciprocal: bool,
    ):
        if not isinstance(target_population, ParticlePopulation) or not isinstance(
            source_population, ParticlePopulation
        ):
            raise TypeError("Interaction endpoints must be ParticlePopulation values.")
        kind = str(interaction_kind)
        if not kind:
            raise ValueError("interaction_kind must be non-empty.")
        self.target_population_id = target_population.population_id
        self.source_population_id = source_population.population_id
        self.interaction_kind = kind
        self.reciprocal = bool(reciprocal)
        self.key_id = canonical_fingerprint(
            {
                "kind": "particle-interaction-key",
                "target": target_population.population_id,
                "source": source_population.population_id,
                "interaction": kind,
                "reciprocal": bool(reciprocal),
            }
        )


class ParticleAssemblyStateLayout(StrictModule, NonTrainableState):
    population_names: tuple[str, ...] = eqx.field(static=True)
    population_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    offsets: tuple[int, ...] = eqx.field(static=True)
    total_size: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)

    def __init__(self, populations: Sequence[ParticlePopulation], /):
        dynamic = tuple(population for population in populations if population.dynamic)
        names = tuple(population.name for population in dynamic)
        shapes = tuple(population.state_shape for population in dynamic)
        resolved_shapes = tuple(shape for shape in shapes if shape is not None)
        sizes = tuple(int(jnp.prod(jnp.asarray(shape))) for shape in resolved_shapes)
        offsets = []
        current = 0
        for size in sizes:
            offsets.append(current)
            current += size
        self.population_names = names
        self.population_shapes = resolved_shapes
        self.offsets = tuple(offsets)
        self.total_size = current
        self.layout_id = canonical_fingerprint(
            {
                "kind": "particle-assembly-state-layout",
                "names": list(names),
                "shapes": [list(shape) for shape in resolved_shapes],
                "offsets": offsets,
            }
        )

    def pack(self, states: Sequence[ArrayLike], /) -> Array:
        values = tuple(jnp.asarray(value) for value in states)
        if len(values) != len(self.population_shapes):
            raise ValueError("Assembly state count does not match dynamic populations.")
        for value, shape in zip(values, self.population_shapes, strict=True):
            if value.shape != shape:
                raise ValueError("Assembly state shape does not match its population.")
        if not values:
            return jnp.zeros((0,), dtype=float)
        return jnp.concatenate(tuple(value.reshape((-1,)) for value in values))

    def unpack(self, state: ArrayLike, /) -> tuple[Array, ...]:
        value = jnp.asarray(state)
        if value.shape != (self.total_size,):
            raise ValueError("Packed assembly state has the wrong size.")
        outputs = []
        for offset, shape in zip(self.offsets, self.population_shapes, strict=True):
            size = int(jnp.prod(jnp.asarray(shape)))
            outputs.append(value[offset : offset + size].reshape(shape))
        return tuple(outputs)


class ParticleAssemblyPlan(StrictModule, NonTrainableState):
    populations: tuple[ParticlePopulation, ...]
    state_layout: ParticleAssemblyStateLayout
    plan_id: str = eqx.field(static=True)

    def __init__(self, populations: Sequence[ParticlePopulation], /):
        values = tuple(populations)
        if not values or any(
            not isinstance(population, ParticlePopulation) for population in values
        ):
            raise ValueError("ParticleAssemblyPlan requires particle populations.")
        names = tuple(population.name for population in values)
        identifiers = tuple(population.population_id for population in values)
        if len(set(names)) != len(names) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Particle population names and IDs must be unique.")
        self.populations = values
        self.state_layout = ParticleAssemblyStateLayout(values)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-assembly-plan",
                "populations": list(identifiers),
                "state_layout": self.state_layout.layout_id,
            }
        )

    def population(self, name: str, /) -> ParticlePopulation:
        matches = tuple(
            population for population in self.populations if population.name == name
        )
        if len(matches) != 1:
            raise KeyError(f"Unknown particle population {name!r}.")
        return matches[0]


class ParticleInteractionLedger(StrictModule):
    target_force: Array
    source_reaction: Array
    action_reaction_defect: Array
    target_power: Array
    source_power: Array
    pair_count: Array

    @classmethod
    def from_forces(
        cls,
        target_force: ArrayLike,
        source_reaction: ArrayLike,
        target_velocity: ArrayLike,
        source_velocity: ArrayLike,
        pair_count: ArrayLike,
        /,
    ) -> "ParticleInteractionLedger":
        target = jnp.asarray(target_force)
        source = jnp.asarray(source_reaction)
        target_v = jnp.asarray(target_velocity)
        source_v = jnp.asarray(source_velocity)
        if target.shape != target_v.shape or source.shape != source_v.shape:
            raise ValueError("Interaction force and velocity shapes must match.")
        total_target = compensated_sum(target, axis=0)
        total_source = compensated_sum(source, axis=0)
        return cls(
            total_target,
            total_source,
            total_target + total_source,
            compensated_sum(jnp.sum(target * target_v, axis=-1)),
            compensated_sum(jnp.sum(source * source_v, axis=-1)),
            jnp.asarray(pair_count, dtype=jnp.int32),
        )


__all__ = [
    "ParticleAssemblyPlan",
    "ParticleAssemblyStateLayout",
    "ParticleInteractionKey",
    "ParticleInteractionLedger",
    "ParticlePopulation",
    "ParticlePopulationRole",
]
