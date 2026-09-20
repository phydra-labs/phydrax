#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization._local_variational import PreparedLocalRegion
from ...discretization.fem._mortar import (
    FiniteElementMortarMetricData,
    FiniteElementMortarPlan,
)
from ...discretization.fem._reference_operator import PreparedFiniteElementReference
from ._ir import (
    LocalActionIR,
    operator_program_from_local_ir,
    OperatorProgram,
)


class WorksetSignature(StrictModule, NonTrainableState):
    region_kind: str = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    reference_action_ids: tuple[str, ...] = eqx.field(static=True)
    field_layout_ids: tuple[str, ...] = eqx.field(static=True)
    geometry_action_id: str = eqx.field(static=True)
    coefficient_layout_ids: tuple[str, ...] = eqx.field(static=True)
    precision_id: str = eqx.field(static=True)
    ir_semantics_id: str = eqx.field(static=True)
    local_kernel: str = eqx.field(static=True)
    provider_selection_id: str | None = eqx.field(static=True)
    execution_kind: str | None = eqx.field(static=True)
    operator_realization: str | None = eqx.field(static=True)
    reference_realization_id: str | None = eqx.field(static=True)
    local_widths: tuple[tuple[str, int], ...] = eqx.field(static=True)
    neighbor_local_widths: tuple[tuple[str, int], ...] = eqx.field(static=True)
    material_id: str | None = eqx.field(static=True)
    signature_id: str = eqx.field(static=True)

    def __init__(
        self,
        region_kind: str,
        block_name: str,
        cell_kind: str,
        rule_id: str,
        local_widths: Mapping[str, int] | Sequence[tuple[str, int]],
        /,
        *,
        support_id: str,
        entity_set_id: str,
        reference_action_ids: Sequence[str],
        field_layout_ids: Sequence[str],
        geometry_action_id: str,
        coefficient_layout_ids: Sequence[str] = (),
        precision_id: str,
        ir_semantics_id: str,
        local_kernel: str,
        provider_selection_id: str | None = None,
        execution_kind: str | None = None,
        operator_realization: str | None = None,
        reference_realization_id: str | None = None,
        neighbor_local_widths: Mapping[str, int]
        | Sequence[tuple[str, int]]
        | None = None,
        material_id: str | None = None,
    ):
        region = str(region_kind)
        block = str(block_name)
        cell = str(cell_kind)
        rule = str(rule_id)
        support = str(support_id)
        entity_set = str(entity_set_id)
        references = tuple(sorted(set(str(value) for value in reference_action_ids)))
        fields = tuple(sorted(set(str(value) for value in field_layout_ids)))
        geometry = str(geometry_action_id)
        layouts = tuple(sorted(str(value) for value in coefficient_layout_ids))
        precision = str(precision_id)
        semantics = str(ir_semantics_id)
        kernel = str(local_kernel)
        selection = None if provider_selection_id is None else str(provider_selection_id)
        execution = None if execution_kind is None else str(execution_kind)
        operator_realization_ = (
            None if operator_realization is None else str(operator_realization)
        )
        reference_realization = (
            None if reference_realization_id is None else str(reference_realization_id)
        )
        widths = tuple(
            sorted(
                (str(name), int(width))
                for name, width in (
                    local_widths.items()
                    if isinstance(local_widths, Mapping)
                    else local_widths
                )
            )
        )
        neighbor_widths = (
            widths
            if neighbor_local_widths is None
            else tuple(
                sorted(
                    (str(name), int(width))
                    for name, width in (
                        neighbor_local_widths.items()
                        if isinstance(neighbor_local_widths, Mapping)
                        else neighbor_local_widths
                    )
                )
            )
        )
        material = None if material_id is None else str(material_id)
        selection_values = (
            selection,
            execution,
            operator_realization_,
            reference_realization,
        )
        identities = (
            region,
            block,
            cell,
            rule,
            support,
            entity_set,
            geometry,
            precision,
            semantics,
            kernel,
            *references,
            *fields,
        )
        if (
            any(not value for value in identities)
            or not references
            or not fields
            or any(not value for value in layouts)
            or not widths
            or any(not name or width <= 0 for name, width in widths)
            or set(name for name, _ in neighbor_widths) != set(name for name, _ in widths)
            or any(not name or width <= 0 for name, width in neighbor_widths)
            or (
                any(value is None for value in selection_values)
                and any(value is not None for value in selection_values)
            )
            or any(value == "" for value in selection_values if value is not None)
        ):
            raise ValueError("Workset signature identities and widths must be complete.")
        self.region_kind = region
        self.block_name = block
        self.cell_kind = cell
        self.support_id = support
        self.entity_set_id = entity_set
        self.rule_id = rule
        self.reference_action_ids = references
        self.field_layout_ids = fields
        self.geometry_action_id = geometry
        self.coefficient_layout_ids = layouts
        self.precision_id = precision
        self.ir_semantics_id = semantics
        self.local_kernel = kernel
        self.provider_selection_id = selection
        self.execution_kind = execution
        self.operator_realization = operator_realization_
        self.reference_realization_id = reference_realization
        self.local_widths = widths
        self.neighbor_local_widths = neighbor_widths
        self.material_id = material
        self.signature_id = canonical_fingerprint(
            {
                "kind": "local-workset-signature",
                "region": region,
                "block": block,
                "cell": cell,
                "support": support,
                "entity_set": entity_set,
                "rule": rule,
                "reference_actions": references,
                "field_layouts": fields,
                "geometry_action": geometry,
                "coefficient_layouts": layouts,
                "precision": precision,
                "ir_semantics": semantics,
                "local_kernel": kernel,
                "provider_selection": selection,
                "execution_kind": execution,
                "operator_realization": operator_realization_,
                "reference_realization": reference_realization,
                "local_widths": [list(item) for item in widths],
                "neighbor_local_widths": [list(item) for item in neighbor_widths],
                "material": material,
            }
        )


class CompiledWorkset(StrictModule, NonTrainableState):
    signature: WorksetSignature
    local_region: PreparedLocalRegion | None
    reference: PreparedFiniteElementReference | None
    neighbor_reference: PreparedFiniteElementReference | None
    mortar: FiniteElementMortarPlan | None
    mortar_metric: FiniteElementMortarMetricData | None
    action_indices: Array
    action_index_values: tuple[int, ...] = eqx.field(static=True)
    entity_index_values: tuple[int, ...] = eqx.field(static=True)
    entity_indices: Array
    owner_cells: Array
    neighbor_cells: Array
    owner_local_entities: Array
    neighbor_local_entities: Array
    owner_permutations: Array
    neighbor_permutations: Array
    neighbor_trace_permutations: Array
    gathers: tuple[tuple[str, Array], ...]
    neighbor_gathers: tuple[tuple[str, Array], ...]
    valid: Array
    workset_id: str = eqx.field(static=True)

    def __init__(
        self,
        signature: WorksetSignature,
        action_indices: ArrayLike,
        entity_indices: ArrayLike,
        owner_cells: ArrayLike,
        neighbor_cells: ArrayLike,
        gathers: Mapping[str, ArrayLike] | Sequence[tuple[str, ArrayLike]],
        /,
        *,
        local_region: PreparedLocalRegion | None = None,
        reference: PreparedFiniteElementReference | None = None,
        neighbor_reference: PreparedFiniteElementReference | None = None,
        mortar: FiniteElementMortarPlan | None = None,
        mortar_metric: FiniteElementMortarMetricData | None = None,
        neighbor_gathers: Mapping[str, ArrayLike]
        | Sequence[tuple[str, ArrayLike]]
        | None = None,
        owner_local_entities: ArrayLike | None = None,
        neighbor_local_entities: ArrayLike | None = None,
        owner_permutations: ArrayLike | None = None,
        neighbor_permutations: ArrayLike | None = None,
        neighbor_trace_permutations: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ):
        if not isinstance(signature, WorksetSignature):
            raise TypeError("signature must be WorksetSignature.")
        actions = np.asarray(action_indices, dtype=np.int32)
        entities = np.asarray(entity_indices, dtype=np.int32)
        owners = np.asarray(owner_cells, dtype=np.int32)
        neighbors = np.asarray(neighbor_cells, dtype=np.int32)
        if actions.ndim != 1 or entities.ndim != 1:
            raise ValueError("Workset action/entity indices must be rank-1.")
        if owners.shape != entities.shape or neighbors.shape != entities.shape:
            raise ValueError("Workset owner/neighbor routes must match entities.")
        gather_items = tuple(
            sorted(
                (str(name), np.asarray(route, dtype=np.int32))
                for name, route in (
                    gathers.items() if isinstance(gathers, Mapping) else gathers
                )
            )
        )
        if set(name for name, _ in gather_items) != set(
            name for name, _ in signature.local_widths
        ):
            raise ValueError("Workset gathers must match signature field widths.")
        count = entities.size
        for name, route in gather_items:
            width = dict(signature.local_widths)[name]
            if route.shape != (count, width):
                raise ValueError("Workset gather shape does not match its signature.")
        if neighbor_gathers is None:
            neighbor_items = tuple(
                (
                    name,
                    np.full(
                        (count, dict(signature.neighbor_local_widths)[name]),
                        -1,
                        dtype=np.int32,
                    ),
                )
                for name, _ in gather_items
            )
        else:
            neighbor_items = tuple(
                sorted(
                    (str(name), np.asarray(route, dtype=np.int32))
                    for name, route in (
                        neighbor_gathers.items()
                        if isinstance(neighbor_gathers, Mapping)
                        else neighbor_gathers
                    )
                )
            )
        if tuple(name for name, _ in neighbor_items) != tuple(
            name for name, _ in gather_items
        ) or any(
            route.shape != (count, dict(signature.neighbor_local_widths)[name])
            for name, route in neighbor_items
        ):
            raise ValueError("Neighbor gathers must match neighbor signature layouts.")

        def route(values, default, dtype):
            return (
                np.full((count,), default, dtype=dtype)
                if values is None
                else np.asarray(values, dtype=dtype)
            )

        def permutation(values):
            result = (
                np.ones((count,), dtype=np.int32)
                if values is None
                else np.asarray(values, dtype=np.int32)
            )
            if result.ndim not in (1, 2) or result.shape[0] != count:
                raise ValueError(
                    "Workset facet permutations require one scalar or route per entity."
                )
            return result

        owner_local = route(owner_local_entities, -1, np.int32)
        neighbor_local = route(neighbor_local_entities, -1, np.int32)
        owner_permutation = permutation(owner_permutations)
        neighbor_permutation = permutation(neighbor_permutations)
        trace_permutations = (
            np.empty((count, 0), dtype=np.int32)
            if neighbor_trace_permutations is None
            else np.asarray(neighbor_trace_permutations, dtype=np.int32)
        )
        if trace_permutations.ndim != 2 or trace_permutations.shape[0] != count:
            raise ValueError(
                "Neighbor trace permutations require one point route per entity."
            )
        if owner_local.shape != (count,) or neighbor_local.shape != (count,):
            raise ValueError("Workset local-entity routes are invalid.")
        if reference is not None and (
            not isinstance(reference, PreparedFiniteElementReference)
            or reference.prepared_id not in signature.reference_action_ids
        ):
            raise ValueError("Prepared reference does not match the workset signature.")
        if neighbor_reference is not None and (
            not isinstance(neighbor_reference, PreparedFiniteElementReference)
            or neighbor_reference.prepared_id not in signature.reference_action_ids
        ):
            raise ValueError("Neighbor reference does not match the workset signature.")
        if local_region is not None:
            if not isinstance(local_region, PreparedLocalRegion):
                raise TypeError("local_region must be PreparedLocalRegion or None.")
            if (
                local_region.geometry_actions.action_id != signature.geometry_action_id
                or tuple(
                    sorted(value.action_id for value in local_region.reference_actions)
                )
                != signature.reference_action_ids
                or tuple(local_region.entity_indices) != tuple(entities)
            ):
                raise ValueError("Prepared local region does not match its workset.")
        if mortar is not None and not isinstance(mortar, FiniteElementMortarPlan):
            raise TypeError("mortar must be FiniteElementMortarPlan or None.")
        if mortar_metric is not None and not isinstance(
            mortar_metric, FiniteElementMortarMetricData
        ):
            raise TypeError(
                "mortar_metric must be FiniteElementMortarMetricData or None."
            )
        if (mortar is None) != (mortar_metric is None):
            raise ValueError(
                "Mortar reference and metric data must be supplied together."
            )
        valid_ = (
            np.ones((count,), dtype=np.bool_)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
        )
        if valid_.shape != (count,):
            raise ValueError("Workset validity must have one entry per entity.")
        self.signature = signature
        self.local_region = local_region
        self.reference = reference
        self.neighbor_reference = neighbor_reference
        self.mortar = mortar
        self.mortar_metric = mortar_metric
        self.action_indices = jnp.asarray(actions)
        self.action_index_values = tuple(actions)
        self.entity_index_values = tuple(entities)
        self.entity_indices = jnp.asarray(entities)
        self.owner_cells = jnp.asarray(owners)
        self.neighbor_cells = jnp.asarray(neighbors)
        self.owner_local_entities = jnp.asarray(owner_local)
        self.neighbor_local_entities = jnp.asarray(neighbor_local)
        self.owner_permutations = jnp.asarray(owner_permutation)
        self.neighbor_permutations = jnp.asarray(neighbor_permutation)
        self.neighbor_trace_permutations = jnp.asarray(trace_permutations)
        self.gathers = tuple((name, jnp.asarray(route)) for name, route in gather_items)
        self.neighbor_gathers = tuple(
            (name, jnp.asarray(route)) for name, route in neighbor_items
        )
        self.valid = jnp.asarray(valid_)
        self.workset_id = canonical_fingerprint(
            {
                "kind": "compiled-local-workset",
                "signature": signature.signature_id,
                "local_region": (
                    None if local_region is None else local_region.region_id
                ),
                "prepared_reference": (
                    None if reference is None else reference.prepared_id
                ),
                "neighbor_reference": (
                    None if neighbor_reference is None else neighbor_reference.prepared_id
                ),
                "mortar": None if mortar is None else mortar.plan_id,
                "mortar_metric": (
                    None if mortar_metric is None else mortar_metric.metric_id
                ),
                "actions": array_tree_fingerprint(actions),
                "entities": array_tree_fingerprint(entities),
                "owners": array_tree_fingerprint(owners),
                "neighbors": array_tree_fingerprint(neighbors),
                "owner_local_entities": array_tree_fingerprint(owner_local),
                "neighbor_local_entities": array_tree_fingerprint(neighbor_local),
                "owner_permutations": array_tree_fingerprint(owner_permutation),
                "neighbor_permutations": array_tree_fingerprint(neighbor_permutation),
                "neighbor_trace_permutations": array_tree_fingerprint(trace_permutations),
                "gathers": [
                    [name, array_tree_fingerprint(route)] for name, route in gather_items
                ],
                "neighbor_gathers": [
                    [name, array_tree_fingerprint(route)]
                    for name, route in neighbor_items
                ],
                "valid": array_tree_fingerprint(valid_),
            }
        )

    def gather(self, field_name: str, values: ArrayLike, /) -> Array:
        name = str(field_name)
        routes = dict(self.gathers)
        if name not in routes:
            raise KeyError(f"Workset has no field gather {name!r}.")
        return jnp.asarray(values)[routes[name]]

    def gather_neighbor(self, field_name: str, values: ArrayLike, /) -> Array:
        name = str(field_name)
        routes = dict(self.neighbor_gathers)
        if name not in routes:
            raise KeyError(f"Workset has no neighbor gather {name!r}.")
        safe = jnp.maximum(routes[name], 0)
        gathered = jnp.asarray(values)[safe]
        valid = routes[name] >= 0
        return jnp.where(
            valid.reshape(valid.shape + (1,) * (gathered.ndim - valid.ndim)),
            gathered,
            jnp.zeros_like(gathered),
        )


class WorksetBucket(StrictModule, NonTrainableState):
    worksets: tuple[CompiledWorkset, ...]
    signature_id: str = eqx.field(static=True)
    entity_count: int = eqx.field(static=True)
    entity_capacity: int = eqx.field(static=True)
    resident_bytes: int = eqx.field(static=True)
    bucket_id: str = eqx.field(static=True)

    def __init__(self, worksets: Sequence[CompiledWorkset], /):
        values = tuple(worksets)
        if not values or len({value.signature.signature_id for value in values}) != 1:
            raise ValueError(
                "Workset buckets require one or more equal-signature worksets."
            )
        arrays = []
        for value in values:
            arrays.extend(
                (
                    value.entity_indices,
                    value.owner_cells,
                    value.neighbor_cells,
                    value.owner_local_entities,
                    value.neighbor_local_entities,
                    value.valid,
                )
            )
            arrays.extend(route for _name, route in value.gathers)
            arrays.extend(route for _name, route in value.neighbor_gathers)
        self.worksets = values
        self.signature_id = values[0].signature.signature_id
        self.entity_count = sum(value.entity_indices.size for value in values)
        self.entity_capacity = max(value.entity_indices.size for value in values)
        self.resident_bytes = sum(np.asarray(value).nbytes for value in arrays)
        self.bucket_id = canonical_fingerprint(
            {
                "kind": "compiled-workset-bucket",
                "signature": self.signature_id,
                "worksets": [value.workset_id for value in values],
                "entity_capacity": self.entity_capacity,
                "resident_bytes": self.resident_bytes,
            }
        )


def bucket_worksets(
    worksets: Sequence[CompiledWorkset],
    /,
) -> tuple[WorksetBucket, ...]:
    groups: dict[str, list[CompiledWorkset]] = {}
    for workset in worksets:
        groups.setdefault(workset.signature.signature_id, []).append(workset)
    return tuple(WorksetBucket(groups[identifier]) for identifier in sorted(groups))


class WorksetProgram(StrictModule, NonTrainableState):
    ir: LocalActionIR
    worksets: tuple[CompiledWorkset, ...]
    operator_program: OperatorProgram
    buckets: tuple[WorksetBucket, ...]
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        ir: LocalActionIR,
        worksets: Sequence[CompiledWorkset],
        /,
        *,
        operator_program: OperatorProgram | None = None,
    ):
        worksets_ = tuple(worksets)
        if not isinstance(ir, LocalActionIR) or not worksets_:
            raise ValueError("WorksetProgram requires an IR and worksets.")
        bucket_id = canonical_fingerprint(
            {
                "kind": "finite-element-operator-program-buckets",
                "signatures": tuple(
                    sorted(value.signature.signature_id for value in worksets_)
                ),
            }
        )
        operator_program_ = (
            operator_program_from_local_ir(ir, bucket_id=bucket_id)
            if operator_program is None
            else operator_program
        )
        if not isinstance(operator_program_, OperatorProgram):
            raise TypeError("operator_program must be OperatorProgram or None.")
        self.ir = ir
        self.worksets = worksets_
        self.operator_program = operator_program_
        self.buckets = bucket_worksets(worksets_)
        self.program_id = canonical_fingerprint(
            {
                "kind": "local-workset-program",
                "ir": ir.ir_id,
                "worksets": [workset.workset_id for workset in worksets_],
                "operator_program": operator_program_.program_id,
                "buckets": [value.bucket_id for value in self.buckets],
            }
        )


__all__ = [
    "CompiledWorkset",
    "WorksetBucket",
    "WorksetProgram",
    "WorksetSignature",
    "bucket_worksets",
]
