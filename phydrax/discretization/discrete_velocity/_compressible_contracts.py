#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


CompressibleKineticModelKind = Literal[
    "guided-d3q39",
    "entropic-d3q343",
    "filtered-d3q33",
    "adaptive-gauge",
]
KineticPopulationRole = Literal[
    "particle",
    "internal-energy",
    "thermal",
]


class KineticPopulationFieldSpec(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    role: KineticPopulationRole = eqx.field(static=True)
    population_count: int = eqx.field(static=True)
    positive: bool = eqx.field(static=True)
    checkpoint_required: bool = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: KineticPopulationRole,
        population_count: int,
        /,
        *,
        positive: bool = True,
        checkpoint_required: bool = True,
    ):
        if not name:
            raise ValueError("Population field name must be non-empty.")
        if role not in ("particle", "internal-energy", "thermal"):
            raise ValueError(f"Unknown kinetic population role {role!r}.")
        count = int(population_count)
        if count < 1:
            raise ValueError("population_count must be positive.")
        self.name = str(name)
        self.role = role
        self.population_count = count
        self.positive = bool(positive)
        self.checkpoint_required = bool(checkpoint_required)


class KineticPopulationLayout(StrictModule, NonTrainableState):
    fields: tuple[KineticPopulationFieldSpec, ...]
    layout_id: str = eqx.field(static=True)

    def __init__(self, fields: tuple[KineticPopulationFieldSpec, ...], /):
        items = tuple(fields)
        if not items or any(
            not isinstance(item, KineticPopulationFieldSpec) for item in items
        ):
            raise TypeError("fields must contain KineticPopulationFieldSpec objects.")
        names = tuple(item.name for item in items)
        roles = tuple(item.role for item in items)
        if len(set(names)) != len(names) or len(set(roles)) != len(roles):
            raise ValueError("Population names and scientific roles must be unique.")
        self.fields = items
        self.layout_id = canonical_fingerprint(
            {
                "kind": "kinetic-population-layout",
                "fields": [
                    {
                        "name": item.name,
                        "role": item.role,
                        "population_count": item.population_count,
                        "positive": item.positive,
                        "checkpoint_required": item.checkpoint_required,
                    }
                    for item in items
                ],
            }
        )

    def index(self, role: KineticPopulationRole, /) -> int:
        matches = tuple(
            index for index, field in enumerate(self.fields) if field.role == role
        )
        if len(matches) != 1:
            raise ValueError(f"Population role {role!r} is absent from this layout.")
        return matches[0]


class CompressibleKineticPopulationState(StrictModule):
    populations: tuple[Array, ...]
    equilibrium_dual: Array
    stabilizer: Array
    frame_velocity: Array
    frame_temperature_scale: Array
    layout: KineticPopulationLayout = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self,
        populations: tuple[ArrayLike, ...],
        equilibrium_dual: ArrayLike,
        stabilizer: ArrayLike,
        frame_velocity: ArrayLike,
        frame_temperature_scale: ArrayLike,
        layout: KineticPopulationLayout,
        model_id: str,
        rule_id: str,
    ):
        if not isinstance(layout, KineticPopulationLayout):
            raise TypeError("layout must be a KineticPopulationLayout.")
        model_identifier = str(model_id).strip()
        rule_identifier = str(rule_id).strip()
        if not model_identifier or not rule_identifier:
            raise ValueError("model_id and rule_id must be nonempty.")
        values = tuple(jnp.asarray(value) for value in populations)
        if len(values) != len(layout.fields):
            raise ValueError("Population tuple does not match the declared layout.")
        spatial_shape = values[0].shape[:-1]
        for value, field in zip(values, layout.fields, strict=True):
            if value.shape != spatial_shape + (field.population_count,):
                raise ValueError(
                    f"Population field {field.name!r} must have shape "
                    f"{spatial_shape + (field.population_count,)}; got {value.shape}."
                )
        dual = jnp.asarray(equilibrium_dual)
        stabilization = jnp.broadcast_to(jnp.asarray(stabilizer), spatial_shape)
        frame = jnp.asarray(frame_velocity)
        scale = jnp.broadcast_to(jnp.asarray(frame_temperature_scale), spatial_shape)
        if dual.shape[:-1] != spatial_shape:
            raise ValueError("equilibrium_dual must share the population spatial shape.")
        if frame.shape[:-1] != spatial_shape:
            raise ValueError("frame_velocity must share the population spatial shape.")
        if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in values):
            raise TypeError("Kinetic populations must be real-valued.")
        self.populations = values
        self.equilibrium_dual = dual
        self.stabilizer = stabilization
        self.frame_velocity = frame
        self.frame_temperature_scale = scale
        self.layout = layout
        self.model_id = model_identifier
        self.rule_id = rule_identifier

    @property
    def spatial_shape(self) -> tuple[int, ...]:
        return tuple(self.populations[0].shape[:-1])

    def population(self, role: KineticPopulationRole, /) -> Array:
        return self.populations[self.layout.index(role)]


class CompressibleKineticMacroscopicState(StrictModule):
    density: Array
    momentum: Array
    velocity: Array
    translational_energy: Array
    internal_energy: Array
    total_energy: Array
    temperature: Array
    pressure: Array
    stress: Array
    heat_flux: Array
    finite: Array
    admissible: Array


class CompressibleKineticConservationEvidence(StrictModule):
    mass_defect: Array
    momentum_defect: Array
    energy_defect: Array
    entropy_change: Array
    minimum_population: Array
    finite: Array
    successful: Array


class CompressibleKineticStepResult(StrictModule):
    candidate: CompressibleKineticPopulationState
    accepted: CompressibleKineticPopulationState
    macroscopic: CompressibleKineticMacroscopicState
    conservation: CompressibleKineticConservationEvidence
    status: Array
    successful: Array
    model_id: str = eqx.field(static=True)


class CompressibleKineticSupportTuple(StrictModule, NonTrainableState):
    model_kind: CompressibleKineticModelKind = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    collision_id: str = eqx.field(static=True)
    transport_id: str = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        model_kind: CompressibleKineticModelKind,
        rule_id: str,
        collision_id: str,
        transport_id: str,
        precision_id: str,
        execution_id: str,
    ):
        if model_kind not in (
            "guided-d3q39",
            "entropic-d3q343",
            "filtered-d3q33",
            "adaptive-gauge",
        ):
            raise ValueError(f"Unknown compressible kinetic model {model_kind!r}.")
        identifiers = (rule_id, collision_id, transport_id, precision_id, execution_id)
        if any(not value for value in identifiers):
            raise ValueError("Support-tuple identifiers must be non-empty.")
        self.model_kind = model_kind
        self.rule_id = rule_id
        self.collision_id = collision_id
        self.transport_id = transport_id
        self.precision_id = precision_id
        self.execution_id = execution_id
        self.support_id = canonical_fingerprint(
            {
                "kind": "compressible-kinetic-support",
                "model": model_kind,
                "rule": rule_id,
                "collision": collision_id,
                "transport": transport_id,
                "precision": precision_id,
                "execution": execution_id,
            }
        )


__all__ = [
    "CompressibleKineticConservationEvidence",
    "CompressibleKineticMacroscopicState",
    "CompressibleKineticModelKind",
    "CompressibleKineticPopulationState",
    "CompressibleKineticStepResult",
    "CompressibleKineticSupportTuple",
    "KineticPopulationFieldSpec",
    "KineticPopulationLayout",
    "KineticPopulationRole",
]
