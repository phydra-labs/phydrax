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
    local_widths: tuple[tuple[str, int], ...] = eqx.field(static=True)
    neighbour_local_widths: tuple[tuple[str, int], ...] = eqx.field(static=True)
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
        neighbour_local_widths: Mapping[str, int]
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
        neighbour_widths = (
            widths
            if neighbour_local_widths is None
            else tuple(
                sorted(
                    (str(name), int(width))
                    for name, width in (
                        neighbour_local_widths.items()
                        if isinstance(neighbour_local_widths, Mapping)
                        else neighbour_local_widths
                    )
                )
            )
        )
        material = None if material_id is None else str(material_id)
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
            or set(name for name, _ in neighbour_widths)
            != set(name for name, _ in widths)
            or any(not name or width <= 0 for name, width in neighbour_widths)
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
        self.local_widths = widths
        self.neighbour_local_widths = neighbour_widths
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
                "local_widths": [list(item) for item in widths],
                "neighbour_local_widths": [list(item) for item in neighbour_widths],
                "material": material,
            }
        )


class CompiledWorkset(StrictModule, NonTrainableState):
    signature: WorksetSignature
    local_region: PreparedLocalRegion | None
    reference: PreparedFiniteElementReference | None
    neighbour_reference: PreparedFiniteElementReference | None
    mortar: FiniteElementMortarPlan | None
    mortar_metric: FiniteElementMortarMetricData | None
    action_indices: Array
    action_index_values: tuple[int, ...] = eqx.field(static=True)
    entity_index_values: tuple[int, ...] = eqx.field(static=True)
    entity_indices: Array
    owner_cells: Array
    neighbour_cells: Array
    owner_local_entities: Array
    neighbour_local_entities: Array
    owner_permutations: Array
    neighbour_permutations: Array
    neighbour_trace_permutations: Array
    gathers: tuple[tuple[str, Array], ...]
    neighbour_gathers: tuple[tuple[str, Array], ...]
    valid: Array
    workset_id: str = eqx.field(static=True)

    def __init__(
        self,
        signature: WorksetSignature,
        action_indices: ArrayLike,
        entity_indices: ArrayLike,
        owner_cells: ArrayLike,
        neighbour_cells: ArrayLike,
        gathers: Mapping[str, ArrayLike] | Sequence[tuple[str, ArrayLike]],
        /,
        *,
        local_region: PreparedLocalRegion | None = None,
        reference: PreparedFiniteElementReference | None = None,
        neighbour_reference: PreparedFiniteElementReference | None = None,
        mortar: FiniteElementMortarPlan | None = None,
        mortar_metric: FiniteElementMortarMetricData | None = None,
        neighbour_gathers: Mapping[str, ArrayLike]
        | Sequence[tuple[str, ArrayLike]]
        | None = None,
        owner_local_entities: ArrayLike | None = None,
        neighbour_local_entities: ArrayLike | None = None,
        owner_permutations: ArrayLike | None = None,
        neighbour_permutations: ArrayLike | None = None,
        neighbour_trace_permutations: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ):
        if not isinstance(signature, WorksetSignature):
            raise TypeError("signature must be WorksetSignature.")
        actions = np.asarray(action_indices, dtype=np.int32)
        entities = np.asarray(entity_indices, dtype=np.int32)
        owners = np.asarray(owner_cells, dtype=np.int32)
        neighbours = np.asarray(neighbour_cells, dtype=np.int32)
        if actions.ndim != 1 or entities.ndim != 1:
            raise ValueError("Workset action/entity indices must be rank-1.")
        if owners.shape != entities.shape or neighbours.shape != entities.shape:
            raise ValueError("Workset owner/neighbour routes must match entities.")
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
        if neighbour_gathers is None:
            neighbour_items = tuple(
                (
                    name,
                    np.full(
                        (count, dict(signature.neighbour_local_widths)[name]),
                        -1,
                        dtype=np.int32,
                    ),
                )
                for name, _ in gather_items
            )
        else:
            neighbour_items = tuple(
                sorted(
                    (str(name), np.asarray(route, dtype=np.int32))
                    for name, route in (
                        neighbour_gathers.items()
                        if isinstance(neighbour_gathers, Mapping)
                        else neighbour_gathers
                    )
                )
            )
        if tuple(name for name, _ in neighbour_items) != tuple(
            name for name, _ in gather_items
        ) or any(
            route.shape != (count, dict(signature.neighbour_local_widths)[name])
            for name, route in neighbour_items
        ):
            raise ValueError("Neighbour gathers must match neighbour signature layouts.")

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
        neighbour_local = route(neighbour_local_entities, -1, np.int32)
        owner_permutation = permutation(owner_permutations)
        neighbour_permutation = permutation(neighbour_permutations)
        trace_permutations = (
            np.empty((count, 0), dtype=np.int32)
            if neighbour_trace_permutations is None
            else np.asarray(neighbour_trace_permutations, dtype=np.int32)
        )
        if trace_permutations.ndim != 2 or trace_permutations.shape[0] != count:
            raise ValueError(
                "Neighbour trace permutations require one point route per entity."
            )
        if owner_local.shape != (count,) or neighbour_local.shape != (count,):
            raise ValueError("Workset local-entity routes are invalid.")
        if reference is not None and (
            not isinstance(reference, PreparedFiniteElementReference)
            or reference.prepared_id not in signature.reference_action_ids
        ):
            raise ValueError("Prepared reference does not match the workset signature.")
        if neighbour_reference is not None and (
            not isinstance(neighbour_reference, PreparedFiniteElementReference)
            or neighbour_reference.prepared_id not in signature.reference_action_ids
        ):
            raise ValueError("Neighbour reference does not match the workset signature.")
        if local_region is not None:
            if not isinstance(local_region, PreparedLocalRegion):
                raise TypeError("local_region must be PreparedLocalRegion or None.")
            if (
                local_region.geometry_actions.action_id != signature.geometry_action_id
                or tuple(
                    sorted(value.action_id for value in local_region.reference_actions)
                )
                != signature.reference_action_ids
                or tuple(int(value) for value in local_region.entity_indices)
                != tuple(int(value) for value in entities)
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
            np.ones((count,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != (count,):
            raise ValueError("Workset validity must have one entry per entity.")
        self.signature = signature
        self.local_region = local_region
        self.reference = reference
        self.neighbour_reference = neighbour_reference
        self.mortar = mortar
        self.mortar_metric = mortar_metric
        self.action_indices = jnp.asarray(actions)
        self.action_index_values = tuple(int(value) for value in actions)
        self.entity_index_values = tuple(int(value) for value in entities)
        self.entity_indices = jnp.asarray(entities)
        self.owner_cells = jnp.asarray(owners)
        self.neighbour_cells = jnp.asarray(neighbours)
        self.owner_local_entities = jnp.asarray(owner_local)
        self.neighbour_local_entities = jnp.asarray(neighbour_local)
        self.owner_permutations = jnp.asarray(owner_permutation)
        self.neighbour_permutations = jnp.asarray(neighbour_permutation)
        self.neighbour_trace_permutations = jnp.asarray(trace_permutations)
        self.gathers = tuple((name, jnp.asarray(route)) for name, route in gather_items)
        self.neighbour_gathers = tuple(
            (name, jnp.asarray(route)) for name, route in neighbour_items
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
                "neighbour_reference": (
                    None
                    if neighbour_reference is None
                    else neighbour_reference.prepared_id
                ),
                "mortar": None if mortar is None else mortar.plan_id,
                "mortar_metric": (
                    None if mortar_metric is None else mortar_metric.metric_id
                ),
                "actions": array_tree_fingerprint(actions),
                "entities": array_tree_fingerprint(entities),
                "owners": array_tree_fingerprint(owners),
                "neighbours": array_tree_fingerprint(neighbours),
                "owner_local_entities": array_tree_fingerprint(owner_local),
                "neighbour_local_entities": array_tree_fingerprint(neighbour_local),
                "owner_permutations": array_tree_fingerprint(owner_permutation),
                "neighbour_permutations": array_tree_fingerprint(neighbour_permutation),
                "neighbour_trace_permutations": array_tree_fingerprint(
                    trace_permutations
                ),
                "gathers": [
                    [name, array_tree_fingerprint(route)] for name, route in gather_items
                ],
                "neighbour_gathers": [
                    [name, array_tree_fingerprint(route)]
                    for name, route in neighbour_items
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

    def gather_neighbour(self, field_name: str, values: ArrayLike, /) -> Array:
        name = str(field_name)
        routes = dict(self.neighbour_gathers)
        if name not in routes:
            raise KeyError(f"Workset has no neighbour gather {name!r}.")
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
                    value.neighbour_cells,
                    value.owner_local_entities,
                    value.neighbour_local_entities,
                    value.valid,
                )
            )
            arrays.extend(route for _name, route in value.gathers)
            arrays.extend(route for _name, route in value.neighbour_gathers)
        self.worksets = values
        self.signature_id = values[0].signature.signature_id
        self.entity_count = sum(int(value.entity_indices.size) for value in values)
        self.entity_capacity = max(int(value.entity_indices.size) for value in values)
        self.resident_bytes = sum(int(np.asarray(value).nbytes) for value in arrays)
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
