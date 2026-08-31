#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._coefficients import PrimeField
from ._complex import CellSubcomplex
from ._filtration import CellFiltration
from ._homology import compute_homology
from ._persistence import compute_persistence, PersistenceResult
from ._resources import TopologyResourcePolicy


class VineyardResult(StrictModule, NonTrainableState):
    """Exact persistence snapshots and stable cell-pair lineage on one layout."""

    snapshots: tuple[PersistenceResult, ...]
    times: Array
    lineage: tuple[Array, ...]
    vineyard_id: str = eqx.field(static=True)

    def __init__(
        self,
        snapshots: Sequence[PersistenceResult],
        times: Array,
        lineage: Sequence[Array],
        /,
    ):
        values = tuple(snapshots)
        lineages = tuple(jnp.asarray(value, dtype=jnp.int32) for value in lineage)
        times_ = jnp.asarray(times)
        if times_.shape != (len(values),) or len(lineages) != max(0, len(values) - 1):
            raise ValueError("Vineyard snapshots, times, and lineage do not align.")
        self.snapshots = values
        self.times = times_
        self.lineage = lineages
        self.vineyard_id = canonical_fingerprint(
            {
                "kind": "vineyard-result",
                "snapshots": [value.result_id for value in values],
                "time_count": int(times_.shape[0]),
            }
        )


def compute_vineyard(
    filtrations: Sequence[CellFiltration],
    times: Array,
    /,
    *,
    coefficients: PrimeField,
    resources: TopologyResourcePolicy | None = None,
) -> VineyardResult:
    """Recompute exact snapshots and retain deterministic creator-cell lineage."""
    values = tuple(filtrations)
    if not values:
        raise ValueError("Vineyards require at least one filtration.")
    layout_ids = {value.complex.layout.layout_id for value in values}
    topology_ids = {value.complex.topology.topology_id for value in values}
    if len(layout_ids) != 1 or len(topology_ids) != 1:
        raise ValueError("Vineyards require one fixed topology and compact layout.")
    snapshots = tuple(
        compute_persistence(
            value,
            coefficients=coefficients,
            resources=resources,
        )
        for value in values
    )
    lineage = []
    for source, target in zip(snapshots[:-1], snapshots[1:], strict=True):
        target_keys = {
            (int(degree), int(entity)): index
            for index, (degree, entity) in enumerate(
                zip(
                    np.asarray(target.pairing.degrees),
                    np.asarray(target.pairing.birth_entity_ids),
                    strict=True,
                )
            )
        }
        lineage.append(
            jnp.asarray(
                [
                    target_keys.get((int(degree), int(entity)), -1)
                    for degree, entity in zip(
                        np.asarray(source.pairing.degrees),
                        np.asarray(source.pairing.birth_entity_ids),
                        strict=True,
                    )
                ],
                dtype=jnp.int32,
            )
        )
    return VineyardResult(snapshots, times, lineage)


class ZigzagCellOperation(StrictModule, NonTrainableState):
    """One closure-checked cell insertion or removal request."""

    action: Literal["insert", "remove"] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    ambient_cell: int = eqx.field(static=True)
    operation_id: str = eqx.field(static=True)

    def __init__(
        self,
        action: Literal["insert", "remove"],
        degree: int,
        ambient_cell: int,
        /,
    ):
        if action not in ("insert", "remove"):
            raise ValueError("Zigzag action must be insert or remove.")
        if int(degree) < 0 or int(ambient_cell) < 0:
            raise ValueError("Zigzag cell coordinates must be non-negative.")
        self.action = action
        self.degree = int(degree)
        self.ambient_cell = int(ambient_cell)
        self.operation_id = canonical_fingerprint(
            {
                "kind": "zigzag-cell-operation",
                "action": action,
                "degree": int(degree),
                "cell": int(ambient_cell),
            }
        )


class ZigzagTopologyResult(StrictModule, NonTrainableState):
    """Exact homology history of one validated insert/remove operation stream."""

    operations: tuple[ZigzagCellOperation, ...]
    betti_history: Array
    state_ids: tuple[str, ...] = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(self, operations, betti_history, state_ids, /):
        self.operations = tuple(operations)
        self.betti_history = jnp.asarray(betti_history, dtype=jnp.int32)
        self.state_ids = tuple(state_ids)
        self.result_id = canonical_fingerprint(
            {
                "kind": "zigzag-topology-result",
                "operations": [value.operation_id for value in self.operations],
                "states": list(self.state_ids),
            }
        )


class MonotoneZigzagIntervals(StrictModule, NonTrainableState):
    """Exact interval decomposition of a closure-valid insertion stream."""

    persistence: PersistenceResult
    operations: tuple[ZigzagCellOperation, ...]
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        persistence: PersistenceResult,
        operations: Sequence[ZigzagCellOperation],
        /,
    ):
        values = tuple(operations)
        self.persistence = persistence
        self.operations = values
        self.result_id = canonical_fingerprint(
            {
                "kind": "monotone-zigzag-intervals",
                "persistence": persistence.result_id,
                "operations": [value.operation_id for value in values],
            }
        )


def compute_monotone_zigzag_intervals(
    ambient: CellSubcomplex,
    initial_masks,
    operations: Sequence[ZigzagCellOperation],
    /,
    *,
    coefficients: PrimeField,
    resources: TopologyResourcePolicy | None = None,
) -> MonotoneZigzagIntervals:
    """Decompose an all-insertion zigzag exactly as a cellular filtration."""
    values = tuple(operations)
    if any(value.action != "insert" for value in values):
        raise ValueError(
            "Monotone interval decomposition requires an all-insertion stream; "
            "mixed insert/remove streams use compute_zigzag_topology."
        )
    masks = [np.asarray(value, dtype=bool).copy() for value in initial_masks]
    filtration_values = [
        np.zeros_like(np.asarray(mask), dtype=float) for mask in initial_masks
    ]
    CellSubcomplex(ambient.topology, masks)
    for step, operation in enumerate(values, start=1):
        if masks[operation.degree][operation.ambient_cell]:
            raise ValueError("Monotone insertion targets an already active cell.")
        masks[operation.degree][operation.ambient_cell] = True
        filtration_values[operation.degree][operation.ambient_cell] = float(step)
        CellSubcomplex(ambient.topology, masks)
    final = CellSubcomplex(ambient.topology, masks)
    filtration = CellFiltration(
        final,
        filtration_values,
        source_id=f"monotone-zigzag:{final.subcomplex_id}",
    )
    persistence = compute_persistence(
        filtration,
        coefficients=coefficients,
        resources=resources,
    )
    return MonotoneZigzagIntervals(persistence, values)


def compute_zigzag_topology(
    ambient: CellSubcomplex,
    initial_masks,
    operations: Sequence[ZigzagCellOperation],
    /,
    *,
    coefficients: PrimeField,
    resources: TopologyResourcePolicy | None = None,
) -> ZigzagTopologyResult:
    """Validate an operation stream and compute exact field homology after each step."""
    masks = [np.asarray(value, dtype=bool).copy() for value in initial_masks]
    state = CellSubcomplex(ambient.topology, masks)
    states = [state]
    for operation in operations:
        degree = operation.degree
        if degree > ambient.max_degree or operation.ambient_cell >= masks[degree].size:
            raise ValueError(
                "Zigzag operation addresses a cell outside the ambient complex."
            )
        if operation.action == "insert":
            if masks[degree][operation.ambient_cell]:
                raise ValueError("Zigzag insertion targets an already active cell.")
            masks[degree][operation.ambient_cell] = True
        else:
            if not masks[degree][operation.ambient_cell]:
                raise ValueError("Zigzag removal targets an inactive cell.")
            masks[degree][operation.ambient_cell] = False
        state = CellSubcomplex(ambient.topology, masks)
        states.append(state)
    results = tuple(
        compute_homology(
            state,
            coefficients=coefficients,
            resources=resources,
        )
        for state in states
    )
    maximum = ambient.max_degree + 1
    history = np.zeros((len(results), maximum), dtype=np.int32)
    for row, result in enumerate(results):
        for value in result.degrees:
            if value.degree >= 0:
                history[row, value.degree] = value.dimension
    return ZigzagTopologyResult(
        tuple(operations),
        history,
        tuple(state.subcomplex_id for state in states),
    )


__all__ = [
    "MonotoneZigzagIntervals",
    "VineyardResult",
    "ZigzagCellOperation",
    "ZigzagTopologyResult",
    "compute_vineyard",
    "compute_monotone_zigzag_intervals",
    "compute_zigzag_topology",
]
