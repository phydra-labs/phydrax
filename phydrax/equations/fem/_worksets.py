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
    term_indices: Array
    entity_indices: Array
    owner_cells: Array
    neighbour_cells: Array
    gathers: tuple[tuple[str, Array], ...]
    valid: Array
    workset_id: str = eqx.field(static=True)

    def __init__(
        self,
        signature: WorksetSignature,
        term_indices: ArrayLike,
        entity_indices: ArrayLike,
        owner_cells: ArrayLike,
        neighbour_cells: ArrayLike,
        gathers: Mapping[str, ArrayLike] | Sequence[tuple[str, ArrayLike]],
        /,
        *,
        valid: ArrayLike | None = None,
    ):
        if not isinstance(signature, WorksetSignature):
            raise TypeError("signature must be WorksetSignature.")
        terms = np.asarray(term_indices, dtype=np.int32)
        entities = np.asarray(entity_indices, dtype=np.int32)
        owners = np.asarray(owner_cells, dtype=np.int32)
        neighbours = np.asarray(neighbour_cells, dtype=np.int32)
        if terms.ndim != 1 or entities.ndim != 1:
            raise ValueError("Workset term/entity indices must be rank-1.")
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
        valid_ = (
            np.ones((count,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != (count,):
            raise ValueError("Workset validity must have one entry per entity.")
        self.signature = signature
        self.term_indices = jnp.asarray(terms)
        self.entity_indices = jnp.asarray(entities)
        self.owner_cells = jnp.asarray(owners)
        self.neighbour_cells = jnp.asarray(neighbours)
        self.gathers = tuple((name, jnp.asarray(route)) for name, route in gather_items)
        self.valid = jnp.asarray(valid_)
        self.workset_id = canonical_fingerprint(
            {
                "kind": "compiled-finite-element-workset",
                "signature": signature.signature_id,
                "terms": array_tree_fingerprint(terms),
                "entities": array_tree_fingerprint(entities),
                "owners": array_tree_fingerprint(owners),
                "neighbours": array_tree_fingerprint(neighbours),
                "gathers": [
                    [name, array_tree_fingerprint(route)] for name, route in gather_items
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
