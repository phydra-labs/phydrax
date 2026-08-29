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
from ._ir import LocalActionIR


class WorksetSignature(StrictModule, NonTrainableState):
    region_kind: str = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)
    local_widths: tuple[tuple[str, int], ...] = eqx.field(static=True)
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
        material_id: str | None = None,
    ):
        region = str(region_kind)
        block = str(block_name)
        cell = str(cell_kind)
        rule = str(rule_id)
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
        material = None if material_id is None else str(material_id)
        if (
            not region
            or not block
            or not cell
            or not rule
            or not widths
            or any(not name or width <= 0 for name, width in widths)
        ):
            raise ValueError("Workset signature fields must be complete and positive.")
        self.region_kind = region
        self.block_name = block
        self.cell_kind = cell
        self.rule_id = rule
        self.local_widths = widths
        self.material_id = material
        self.signature_id = canonical_fingerprint(
            {
                "kind": "finite-element-workset-signature",
                "region": region,
                "block": block,
                "cell": cell,
                "rule": rule,
                "local_widths": [list(item) for item in widths],
                "material": material,
            }
        )


class CompiledWorkset(StrictModule, NonTrainableState):
    signature: WorksetSignature
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
        neighbour_gathers: Mapping[str, ArrayLike]
        | Sequence[tuple[str, ArrayLike]]
        | None = None,
        owner_local_entities: ArrayLike | None = None,
        neighbour_local_entities: ArrayLike | None = None,
        owner_permutations: ArrayLike | None = None,
        neighbour_permutations: ArrayLike | None = None,
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
                (name, np.full_like(route, -1)) for name, route in gather_items
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
            route.shape != dict(gather_items)[name].shape
            for name, route in neighbour_items
        ):
            raise ValueError("Neighbour gathers must match owner gather layouts.")

        def route(values, default, dtype):
            return (
                np.full((count,), default, dtype=dtype)
                if values is None
                else np.asarray(values, dtype=dtype)
            )

        owner_local = route(owner_local_entities, -1, np.int32)
        neighbour_local = route(neighbour_local_entities, -1, np.int32)
        owner_permutation = route(owner_permutations, 1, np.int8)
        neighbour_permutation = route(neighbour_permutations, 1, np.int8)
        if any(
            value.shape != (count,)
            for value in (
                owner_local,
                neighbour_local,
                owner_permutation,
                neighbour_permutation,
            )
        ):
            raise ValueError("Workset local-entity/permutation routes are invalid.")
        valid_ = (
            np.ones((count,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != (count,):
            raise ValueError("Workset validity must have one entry per entity.")
        self.signature = signature
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
        self.gathers = tuple((name, jnp.asarray(route)) for name, route in gather_items)
        self.neighbour_gathers = tuple(
            (name, jnp.asarray(route)) for name, route in neighbour_items
        )
        self.valid = jnp.asarray(valid_)
        self.workset_id = canonical_fingerprint(
            {
                "kind": "compiled-finite-element-workset",
                "signature": signature.signature_id,
                "actions": array_tree_fingerprint(actions),
                "entities": array_tree_fingerprint(entities),
                "owners": array_tree_fingerprint(owners),
                "neighbours": array_tree_fingerprint(neighbours),
                "owner_local_entities": array_tree_fingerprint(owner_local),
                "neighbour_local_entities": array_tree_fingerprint(neighbour_local),
                "owner_permutations": array_tree_fingerprint(owner_permutation),
                "neighbour_permutations": array_tree_fingerprint(neighbour_permutation),
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


class WorksetProgram(StrictModule, NonTrainableState):
    ir: LocalActionIR
    worksets: tuple[CompiledWorkset, ...]
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        ir: LocalActionIR,
        worksets: Sequence[CompiledWorkset],
        /,
    ):
        worksets_ = tuple(worksets)
        if not isinstance(ir, LocalActionIR) or not worksets_:
            raise ValueError("WorksetProgram requires an IR and worksets.")
        self.ir = ir
        self.worksets = worksets_
        self.program_id = canonical_fingerprint(
            {
                "kind": "finite-element-workset-program",
                "ir": ir.ir_id,
                "worksets": [workset.workset_id for workset in worksets_],
            }
        )


__all__ = ["CompiledWorkset", "WorksetProgram", "WorksetSignature"]
