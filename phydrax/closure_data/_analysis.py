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
from ._filters import FavreFilter, PreparedFilter
from ._state import ClosureSnapshot


AnalysisNodeKind = Literal[
    "reynolds_sgs_stress",
    "favre_sgs_stress",
    "sgs_energy",
    "reynolds_species_flux",
    "favre_species_flux",
    "reynolds_enthalpy_flux",
    "favre_enthalpy_flux",
    "source_residual",
    "periodic_les_reynolds_stress",
    "periodic_les_stress_divergence",
    "periodic_les_energy_transfer",
    "periodic_les_scalar_flux",
]
ClosureTargetKind = Literal[
    "sgs_stress",
    "sgs_energy",
    "species_flux",
    "enthalpy_flux",
    "source",
    "sgs_stress_divergence",
    "sgs_transfer",
    "scalar_flux",
]


class ClosureField(StrictModule, NonTrainableState):
    """Named physical array with exact units and immutable upstream lineage."""

    values: Array
    name: str = eqx.field(static=True)
    units: str = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        /,
        *,
        name: str,
        units: str,
        schema_id: str,
        lineage_ids: tuple[str, ...],
    ):
        array = jnp.asarray(values)
        name_ = str(name).strip()
        units_ = str(units).strip()
        schema = str(schema_id).strip()
        lineage = tuple(str(value).strip() for value in lineage_ids)
        if (
            not name_
            or not units_
            or not schema
            or not lineage
            or any(not value for value in lineage)
            or not jnp.issubdtype(array.dtype, jnp.inexact)
        ):
            raise ValueError("Closure field metadata is invalid.")
        self.values = array
        self.name = name_
        self.units = units_
        self.schema_id = schema
        self.lineage_ids = lineage
        self.field_id = canonical_fingerprint(
            {
                "kind": "closure-field",
                "name": name_,
                "units": units_,
                "schema": schema,
                "lineage": list(lineage),
                "content": array_tree_fingerprint(array),
            }
        )

    @classmethod
    def from_snapshot(cls, snapshot: ClosureSnapshot, name: str, /) -> ClosureField:
        if not isinstance(snapshot, ClosureSnapshot):
            raise TypeError("snapshot must be a ClosureSnapshot.")
        index = snapshot.schema.index(name)
        return cls(
            snapshot.values[..., index],
            name=name,
            units=snapshot.schema.unit(name),
            schema_id=snapshot.schema.schema_id,
            lineage_ids=(*snapshot.parent_ids, snapshot.snapshot_id),
        )

    @classmethod
    def velocity_from_snapshot(cls, snapshot: ClosureSnapshot, /) -> ClosureField:
        names = snapshot.schema.velocity_names
        if not names:
            raise ValueError("Snapshot schema declares no velocity components.")
        indices = tuple(snapshot.schema.index(name) for name in names)
        units = tuple(snapshot.schema.unit(name) for name in names)
        if any(unit != units[0] for unit in units):
            raise ValueError("Velocity components must use one physical unit.")
        return cls(
            snapshot.values[..., jnp.asarray(indices)],
            name="velocity",
            units=units[0],
            schema_id=snapshot.schema.schema_id,
            lineage_ids=(*snapshot.parent_ids, snapshot.snapshot_id),
        )


class ClosureAnalysisNode(StrictModule, NonTrainableState):
    """Typed deterministic operation node identified entirely by semantic inputs."""

    kind: AnalysisNodeKind = eqx.field(static=True)
    input_ids: tuple[str, ...] = eqx.field(static=True)
    output_name: str = eqx.field(static=True)
    output_units: str = eqx.field(static=True)
    parameters: tuple[tuple[str, str], ...] = eqx.field(static=True)
    node_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: AnalysisNodeKind,
        input_ids: tuple[str, ...],
        /,
        *,
        output_name: str,
        output_units: str,
        parameters: tuple[tuple[str, str], ...] = (),
    ):
        kind_ = str(kind).strip()
        inputs = tuple(str(value).strip() for value in input_ids)
        output = str(output_name).strip()
        units = str(output_units).strip()
        parameters_ = tuple(
            sorted((str(key).strip(), str(value).strip()) for key, value in parameters)
        )
        if (
            kind_
            not in (
                "reynolds_sgs_stress",
                "favre_sgs_stress",
                "sgs_energy",
                "reynolds_species_flux",
                "favre_species_flux",
                "reynolds_enthalpy_flux",
                "favre_enthalpy_flux",
                "source_residual",
                "periodic_les_reynolds_stress",
                "periodic_les_stress_divergence",
                "periodic_les_energy_transfer",
                "periodic_les_scalar_flux",
            )
            or not inputs
            or any(not value for value in inputs)
            or not output
            or not units
            or any(not key or not value for key, value in parameters_)
            or len({key for key, _ in parameters_}) != len(parameters_)
        ):
            raise ValueError("Closure analysis node metadata is invalid.")
        self.kind = kind_
        self.input_ids = inputs
        self.output_name = output
        self.output_units = units
        self.parameters = parameters_
        self.node_id = canonical_fingerprint(
            {
                "kind": "closure-analysis-node",
                "operation": kind_,
                "inputs": list(inputs),
                "output_name": output,
                "output_units": units,
                "parameters": [list(value) for value in parameters_],
            }
        )


class ClosureAnalysisDAG(StrictModule, NonTrainableState):
    """Topologically validated immutable lineage graph for closure quantities."""

    external_input_ids: tuple[str, ...] = eqx.field(static=True)
    nodes: tuple[ClosureAnalysisNode, ...]
    dag_id: str = eqx.field(static=True)

    def __init__(
        self,
        external_input_ids: tuple[str, ...],
        nodes: tuple[ClosureAnalysisNode, ...] = (),
        /,
    ):
        external = tuple(str(value).strip() for value in external_input_ids)
        nodes_ = tuple(nodes)
        if (
            any(not value for value in external)
            or len(set(external)) != len(external)
            or any(not isinstance(node, ClosureAnalysisNode) for node in nodes_)
        ):
            raise ValueError("Closure analysis DAG inputs or nodes are invalid.")
        available = set(external)
        for node in nodes_:
            if any(value not in available for value in node.input_ids):
                raise ValueError(
                    "Closure analysis DAG contains a missing or forward dependency."
                )
            if node.node_id in available:
                raise ValueError(
                    "Closure analysis DAG contains a duplicate node identity."
                )
            available.add(node.node_id)
        self.external_input_ids = external
        self.nodes = nodes_
        self.dag_id = canonical_fingerprint(
            {
                "kind": "closure-analysis-dag",
                "external_inputs": list(external),
                "nodes": [node.node_id for node in nodes_],
            }
        )

    def append(self, node: ClosureAnalysisNode, /) -> ClosureAnalysisDAG:
        if not isinstance(node, ClosureAnalysisNode):
            raise TypeError("node must be a ClosureAnalysisNode.")
        return ClosureAnalysisDAG(self.external_input_ids, (*self.nodes, node))


class ClosureTarget(StrictModule, NonTrainableState):
    """Materialized closure target carrying its exact analysis node and units."""

    values: Array
    node: ClosureAnalysisNode
    target_kind: ClosureTargetKind = eqx.field(static=True)
    schema_id: str = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: ArrayLike,
        node: ClosureAnalysisNode,
        /,
        *,
        target_kind: ClosureTargetKind,
        schema_id: str,
    ):
        if not isinstance(node, ClosureAnalysisNode):
            raise TypeError("node must be a ClosureAnalysisNode.")
        array = jnp.asarray(values)
        kind = str(target_kind).strip()
        schema = str(schema_id).strip()
        if (
            kind
            not in (
                "sgs_stress",
                "sgs_energy",
                "species_flux",
                "enthalpy_flux",
                "source",
                "sgs_stress_divergence",
                "sgs_transfer",
                "scalar_flux",
            )
            or not schema
        ):
            raise ValueError("Closure target metadata is invalid.")
        self.values = array
        self.node = node
        self.target_kind = kind
        self.schema_id = schema
        self.lineage_ids = (*node.input_ids, node.node_id)
        self.target_id = canonical_fingerprint(
            {
                "kind": "closure-target",
                "target_kind": kind,
                "schema": schema,
                "node": node.node_id,
                "content": array_tree_fingerprint(array),
            }
        )

    @property
    def units(self) -> str:
        return self.node.output_units


class ClosureQualityReport(StrictModule, NonTrainableState):
    """Finite-value and magnitude evidence over a set of materialized targets."""

    finite_fraction: Array
    rms: Array
    maximum_absolute: Array
    target_ids: tuple[str, ...] = eqx.field(static=True)
    sample_count: int = eqx.field(static=True)
    nonfinite_count: int = eqx.field(static=True)
    maximum_allowed: float | None = eqx.field(static=True)
    passed: bool = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        targets: tuple[ClosureTarget, ...],
        /,
        *,
        maximum_allowed: float | None = None,
    ):
        values = tuple(targets)
        if not values or any(not isinstance(value, ClosureTarget) for value in values):
            raise ValueError("Quality reports require at least one closure target.")
        limit = None if maximum_allowed is None else float(maximum_allowed)
        if limit is not None and (not np.isfinite(limit) or limit < 0.0):
            raise ValueError("maximum_allowed must be finite and nonnegative.")
        flattened = jnp.concatenate(
            tuple(value.values.reshape((-1,)) for value in values)
        )
        finite = jnp.isfinite(flattened)
        safe = jnp.where(finite, flattened, jnp.zeros_like(flattened))
        count = int(flattened.size)
        nonfinite = int(np.count_nonzero(~np.asarray(finite)))
        maximum = jnp.max(jnp.abs(safe), initial=0.0)
        passed = nonfinite == 0 and (limit is None or float(np.asarray(maximum)) <= limit)
        self.finite_fraction = jnp.sum(finite) / count
        self.rms = jnp.sqrt(jnp.sum(jnp.abs(safe) ** 2) / count)
        self.maximum_absolute = maximum
        self.target_ids = tuple(value.target_id for value in values)
        self.sample_count = count
        self.nonfinite_count = nonfinite
        self.maximum_allowed = limit
        self.passed = passed
        self.report_id = canonical_fingerprint(
            {
                "kind": "closure-quality-report",
                "targets": list(self.target_ids),
                "sample_count": count,
                "nonfinite_count": nonfinite,
                "maximum_allowed": limit,
                "passed": passed,
            }
        )


def sgs_stress_target(
    velocity: ClosureField,
    prepared_filter: PreparedFilter,
    /,
    *,
    density: ClosureField | None = None,
) -> ClosureTarget:
    _require_field(velocity, "velocity")
    if velocity.values.ndim != prepared_filter.spatial_rank + 1:
        raise ValueError("Velocity must have exactly one trailing component axis.")
    outer = velocity.values[..., :, None] * velocity.values[..., None, :]
    if density is None:
        mean = prepared_filter.apply(velocity.values)
        values = prepared_filter.apply(outer) - mean[..., :, None] * mean[..., None, :]
        node_kind: AnalysisNodeKind = "reynolds_sgs_stress"
        inputs = (velocity.field_id,)
        units = f"({velocity.units})^2"
    else:
        _compatible_fields(velocity, density)
        rho = _scalar_values(density)
        favre = FavreFilter(prepared_filter)
        mean_density = favre.mean_density(rho)
        mean = favre.apply(velocity.values, rho)
        values = (
            prepared_filter.apply(rho[..., None, None] * outer)
            - mean_density[..., None, None] * mean[..., :, None] * mean[..., None, :]
        )
        node_kind = "favre_sgs_stress"
        inputs = (velocity.field_id, density.field_id)
        units = f"({density.units})*({velocity.units})^2"
    node = ClosureAnalysisNode(
        node_kind,
        inputs,
        output_name="sgs_stress",
        output_units=units,
        parameters=(("filter_id", prepared_filter.prepared_id),),
    )
    return ClosureTarget(
        values,
        node,
        target_kind="sgs_stress",
        schema_id=velocity.schema_id,
    )


def sgs_energy_target(stress: ClosureTarget, /) -> ClosureTarget:
    if not isinstance(stress, ClosureTarget) or stress.target_kind != "sgs_stress":
        raise TypeError("stress must be an SGS stress ClosureTarget.")
    if stress.values.ndim < 2 or stress.values.shape[-1] != stress.values.shape[-2]:
        raise ValueError("SGS stress must end with a square tensor.")
    values = 0.5 * jnp.trace(stress.values, axis1=-2, axis2=-1)
    node = ClosureAnalysisNode(
        "sgs_energy",
        (stress.node.node_id,),
        output_name="sgs_energy",
        output_units=stress.units,
    )
    return ClosureTarget(
        values,
        node,
        target_kind="sgs_energy",
        schema_id=stress.schema_id,
    )


def species_flux_target(
    velocity: ClosureField,
    species: ClosureField,
    prepared_filter: PreparedFilter,
    /,
    *,
    density: ClosureField | None = None,
) -> ClosureTarget:
    return _scalar_flux_target(
        velocity,
        species,
        prepared_filter,
        density=density,
        target_kind="species_flux",
    )


def enthalpy_flux_target(
    velocity: ClosureField,
    enthalpy: ClosureField,
    prepared_filter: PreparedFilter,
    /,
    *,
    density: ClosureField | None = None,
) -> ClosureTarget:
    return _scalar_flux_target(
        velocity,
        enthalpy,
        prepared_filter,
        density=density,
        target_kind="enthalpy_flux",
    )


def source_target(
    fine_source: ClosureField,
    resolved_source: ClosureField,
    prepared_filter: PreparedFilter,
    /,
) -> ClosureTarget:
    _compatible_fields(fine_source, resolved_source)
    if fine_source.units != resolved_source.units:
        raise ValueError("Fine and resolved source units must match exactly.")
    filtered = prepared_filter.apply(fine_source.values)
    if filtered.shape != resolved_source.values.shape:
        raise ValueError("Filtered and resolved source arrays must share shape.")
    node = ClosureAnalysisNode(
        "source_residual",
        (fine_source.field_id, resolved_source.field_id),
        output_name="source_closure",
        output_units=fine_source.units,
        parameters=(("filter_id", prepared_filter.prepared_id),),
    )
    return ClosureTarget(
        filtered - resolved_source.values,
        node,
        target_kind="source",
        schema_id=fine_source.schema_id,
    )


def _scalar_flux_target(
    velocity: ClosureField,
    scalar: ClosureField,
    prepared_filter: PreparedFilter,
    /,
    *,
    density: ClosureField | None,
    target_kind: Literal["species_flux", "enthalpy_flux"],
) -> ClosureTarget:
    _compatible_fields(velocity, scalar)
    scalar_values = _scalar_values(scalar)
    if velocity.values.ndim != prepared_filter.spatial_rank + 1:
        raise ValueError("Velocity must have exactly one trailing component axis.")
    if density is None:
        mean_velocity = prepared_filter.apply(velocity.values)
        mean_scalar = prepared_filter.apply(scalar_values)
        values = (
            prepared_filter.apply(velocity.values * scalar_values[..., None])
            - mean_velocity * mean_scalar[..., None]
        )
        node_kind: AnalysisNodeKind = (
            "reynolds_species_flux"
            if target_kind == "species_flux"
            else "reynolds_enthalpy_flux"
        )
        inputs = (velocity.field_id, scalar.field_id)
        units = f"({velocity.units})*({scalar.units})"
    else:
        _compatible_fields(velocity, density)
        rho = _scalar_values(density)
        favre = FavreFilter(prepared_filter)
        mean_density = favre.mean_density(rho)
        mean_velocity = favre.apply(velocity.values, rho)
        mean_scalar = favre.apply(scalar_values, rho)
        values = (
            prepared_filter.apply(
                rho[..., None] * velocity.values * scalar_values[..., None]
            )
            - mean_density[..., None] * mean_velocity * mean_scalar[..., None]
        )
        node_kind = (
            "favre_species_flux"
            if target_kind == "species_flux"
            else "favre_enthalpy_flux"
        )
        inputs = (velocity.field_id, scalar.field_id, density.field_id)
        units = f"({density.units})*({velocity.units})*({scalar.units})"
    node = ClosureAnalysisNode(
        node_kind,
        inputs,
        output_name=target_kind,
        output_units=units,
        parameters=(("filter_id", prepared_filter.prepared_id),),
    )
    return ClosureTarget(
        values,
        node,
        target_kind=target_kind,
        schema_id=velocity.schema_id,
    )


def _scalar_values(field: ClosureField) -> Array:
    values = field.values
    if values.ndim >= 1 and values.shape[-1:] == (1,):
        return values[..., 0]
    return values


def _require_field(value: ClosureField, name: str) -> None:
    if not isinstance(value, ClosureField):
        raise TypeError(f"{name} must be a ClosureField.")


def _compatible_fields(left: ClosureField, right: ClosureField) -> None:
    _require_field(left, "left")
    _require_field(right, "right")
    if left.schema_id != right.schema_id:
        raise ValueError("Closure fields must share one schema identity.")
    rank = min(left.values.ndim, right.values.ndim)
    if left.values.shape[:rank] != right.values.shape[:rank]:
        raise ValueError("Closure fields have incompatible spatial shapes.")


__all__ = [
    "AnalysisNodeKind",
    "ClosureAnalysisDAG",
    "ClosureAnalysisNode",
    "ClosureField",
    "ClosureQualityReport",
    "ClosureTarget",
    "ClosureTargetKind",
    "enthalpy_flux_target",
    "sgs_energy_target",
    "sgs_stress_target",
    "source_target",
    "species_flux_target",
]
