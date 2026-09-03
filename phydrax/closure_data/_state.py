#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


StateRepresentation = Literal["nondimensional", "dimensional"]


class FlowStateSchema(StrictModule, NonTrainableState):
    """Ordered flow-state components with physical units and reference scales."""

    component_names: tuple[str, ...] = eqx.field(static=True)
    component_units: tuple[str, ...] = eqx.field(static=True)
    reference_scales: tuple[float, ...] = eqx.field(static=True)
    density_name: str | None = eqx.field(static=True)
    velocity_names: tuple[str, ...] = eqx.field(static=True)
    species_names: tuple[str, ...] = eqx.field(static=True)
    total_energy_name: str | None = eqx.field(static=True)
    enthalpy_name: str | None = eqx.field(static=True)
    component_axis: int = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)

    def __init__(
        self,
        component_names: tuple[str, ...],
        component_units: tuple[str, ...],
        reference_scales: tuple[float, ...],
        /,
        *,
        density_name: str | None = None,
        velocity_names: tuple[str, ...] = (),
        species_names: tuple[str, ...] = (),
        total_energy_name: str | None = None,
        enthalpy_name: str | None = None,
        component_axis: int = -1,
    ):
        names = tuple(str(value).strip() for value in component_names)
        units = tuple(str(value).strip() for value in component_units)
        scales = tuple(float(value) for value in reference_scales)
        density = None if density_name is None else str(density_name).strip()
        velocities = tuple(str(value).strip() for value in velocity_names)
        species = tuple(str(value).strip() for value in species_names)
        energy = None if total_energy_name is None else str(total_energy_name).strip()
        enthalpy = None if enthalpy_name is None else str(enthalpy_name).strip()
        role_names = (
            *((density,) if density is not None else ()),
            *velocities,
            *species,
            *((energy,) if energy is not None else ()),
            *((enthalpy,) if enthalpy is not None else ()),
        )
        if (
            not names
            or len(set(names)) != len(names)
            or len(units) != len(names)
            or len(scales) != len(names)
            or any(not value for value in (*names, *units, *role_names))
            or any(not np.isfinite(value) or value <= 0.0 for value in scales)
            or any(value not in names for value in role_names)
            or len(set(velocities)) != len(velocities)
            or len(set(species)) != len(species)
            or set(velocities) & set(species)
            or int(component_axis) != -1
        ):
            raise ValueError("Flow-state schema metadata is invalid.")
        self.component_names = names
        self.component_units = units
        self.reference_scales = scales
        self.density_name = density
        self.velocity_names = velocities
        self.species_names = species
        self.total_energy_name = energy
        self.enthalpy_name = enthalpy
        self.component_axis = -1
        self.schema_id = canonical_fingerprint(
            {
                "kind": "flow-state-schema",
                "component_names": list(names),
                "component_units": list(units),
                "reference_scales": list(scales),
                "density_name": density,
                "velocity_names": list(velocities),
                "species_names": list(species),
                "total_energy_name": energy,
                "enthalpy_name": enthalpy,
                "component_axis": -1,
            }
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.component_names

    @property
    def units(self) -> tuple[str, ...]:
        return self.component_units

    @property
    def component_count(self) -> int:
        return len(self.component_names)

    def index(self, name: str, /) -> int:
        value = str(name).strip()
        if value not in self.component_names:
            raise KeyError(f"Unknown flow-state component {value!r}.")
        return self.component_names.index(value)

    def unit(self, name: str, /) -> str:
        return self.component_units[self.index(name)]

    def validate(self, values: ArrayLike, /, *, owner: str = "Flow state") -> Array:
        array = jnp.asarray(values)
        if array.ndim < 1 or array.shape[-1] != self.component_count:
            raise ValueError(
                f"{owner} must end with {self.component_count} components; "
                f"got shape {array.shape}."
            )
        if not jnp.issubdtype(array.dtype, jnp.inexact):
            raise TypeError(f"{owner} must use an inexact dtype.")
        return array

    def dimensionalize(self, values: ArrayLike, /) -> Array:
        array = self.validate(values)
        return array * jnp.asarray(self.reference_scales, dtype=array.dtype)

    def nondimensionalize(self, values: ArrayLike, /) -> Array:
        array = self.validate(values)
        return array / jnp.asarray(self.reference_scales, dtype=array.dtype)


class ClosureSnapshot(StrictModule, NonTrainableState):
    """One immutable, identified flow snapshot; mesh ownership remains external."""

    values: Array
    schema: FlowStateSchema
    time: float = eqx.field(static=True)
    case_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    time_block_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    representation: StateRepresentation = eqx.field(static=True)
    parent_ids: tuple[str, ...] = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        schema: FlowStateSchema,
        /,
        *,
        time: float,
        case_id: str,
        trajectory_id: str,
        realization_id: str,
        time_block_id: str,
        mesh_id: str,
        representation: StateRepresentation = "nondimensional",
        parent_ids: tuple[str, ...] = (),
    ):
        if not isinstance(schema, FlowStateSchema):
            raise TypeError("schema must be a FlowStateSchema.")
        array = schema.validate(values, owner="Closure snapshot")
        time_value = float(time)
        identifiers = tuple(
            str(value).strip()
            for value in (
                case_id,
                trajectory_id,
                realization_id,
                time_block_id,
                mesh_id,
            )
        )
        parents = tuple(str(value).strip() for value in parent_ids)
        representation_ = str(representation).strip()
        if (
            not np.isfinite(time_value)
            or any(not value for value in (*identifiers, *parents))
            or representation_ not in ("nondimensional", "dimensional")
        ):
            raise ValueError("Closure snapshot metadata is invalid.")
        content = array_tree_fingerprint(array)
        self.values = array
        self.schema = schema
        self.time = time_value
        (
            self.case_id,
            self.trajectory_id,
            self.realization_id,
            self.time_block_id,
            self.mesh_id,
        ) = identifiers
        self.representation = representation_
        self.parent_ids = parents
        self.snapshot_id = canonical_fingerprint(
            {
                "kind": "closure-snapshot",
                "schema": schema.schema_id,
                "time": time_value,
                "identifiers": list(identifiers),
                "representation": representation_,
                "parents": list(parents),
                "content": content,
            }
        )

    def dimensionalize(self) -> ClosureSnapshot:
        if self.representation != "nondimensional":
            raise ValueError("Only nondimensional snapshots can be dimensionalized.")
        return ClosureSnapshot(
            self.schema.dimensionalize(self.values),
            self.schema,
            time=self.time,
            case_id=self.case_id,
            trajectory_id=self.trajectory_id,
            realization_id=self.realization_id,
            time_block_id=self.time_block_id,
            mesh_id=self.mesh_id,
            representation="dimensional",
            parent_ids=(*self.parent_ids, self.snapshot_id),
        )

    def nondimensionalize(self) -> ClosureSnapshot:
        if self.representation != "dimensional":
            raise ValueError("Only dimensional snapshots can be nondimensionalized.")
        return ClosureSnapshot(
            self.schema.nondimensionalize(self.values),
            self.schema,
            time=self.time,
            case_id=self.case_id,
            trajectory_id=self.trajectory_id,
            realization_id=self.realization_id,
            time_block_id=self.time_block_id,
            mesh_id=self.mesh_id,
            representation="nondimensional",
            parent_ids=(*self.parent_ids, self.snapshot_id),
        )


class ClosureSeries(StrictModule, NonTrainableState):
    """A strictly time-ordered trajectory with one schema and mesh identity."""

    snapshots: tuple[ClosureSnapshot, ...]
    schema: FlowStateSchema
    case_id: str = eqx.field(static=True)
    trajectory_id: str = eqx.field(static=True)
    realization_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)
    series_id: str = eqx.field(static=True)

    def __init__(self, snapshots: tuple[ClosureSnapshot, ...], /):
        values = tuple(snapshots)
        if not values or any(not isinstance(item, ClosureSnapshot) for item in values):
            raise ValueError("ClosureSeries requires at least one closure snapshot.")
        first = values[0]
        schema_id = first.schema.schema_id
        identity = (
            first.case_id,
            first.trajectory_id,
            first.realization_id,
            first.mesh_id,
            first.representation,
        )
        if any(
            item.schema.schema_id != schema_id
            or (
                item.case_id,
                item.trajectory_id,
                item.realization_id,
                item.mesh_id,
                item.representation,
            )
            != identity
            for item in values
        ):
            raise ValueError("ClosureSeries snapshots must share schema and identity.")
        times = tuple(item.time for item in values)
        if any(right <= left for left, right in zip(times[:-1], times[1:], strict=True)):
            raise ValueError("ClosureSeries times must be strictly increasing.")
        self.snapshots = values
        self.schema = first.schema
        self.case_id = first.case_id
        self.trajectory_id = first.trajectory_id
        self.realization_id = first.realization_id
        self.mesh_id = first.mesh_id
        self.series_id = canonical_fingerprint(
            {
                "kind": "closure-series",
                "schema": schema_id,
                "snapshot_ids": [item.snapshot_id for item in values],
                "identity": list(identity),
            }
        )

    @property
    def times(self) -> Array:
        return jnp.asarray(tuple(item.time for item in self.snapshots))

    def stack(self) -> Array:
        shapes = tuple(item.values.shape for item in self.snapshots)
        if any(shape != shapes[0] for shape in shapes):
            raise ValueError("ClosureSeries snapshots must share shape before stacking.")
        return jnp.stack(tuple(item.values for item in self.snapshots), axis=0)

    @property
    def representation(self) -> StateRepresentation:
        return self.snapshots[0].representation

    def dimensionalize(self) -> ClosureSeries:
        if self.representation != "nondimensional":
            raise ValueError("Only nondimensional series can be dimensionalized.")
        return ClosureSeries(tuple(item.dimensionalize() for item in self.snapshots))

    def nondimensionalize(self) -> ClosureSeries:
        if self.representation != "dimensional":
            raise ValueError("Only dimensional series can be nondimensionalized.")
        return ClosureSeries(tuple(item.nondimensionalize() for item in self.snapshots))


__all__ = [
    "ClosureSeries",
    "ClosureSnapshot",
    "FlowStateSchema",
    "StateRepresentation",
]
