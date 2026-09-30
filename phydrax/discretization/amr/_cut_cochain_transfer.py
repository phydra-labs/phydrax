#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Commuting and metric-adjoint cochain transfer across cut-complex epochs."""

from __future__ import annotations

from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...exterior._complex import ComplexBoundary
from ...linalg import (
    AbstractLinearOperator,
    adjoint,
    ComplexMap,
    ComplexMapEvidence,
    DenseLinearOperator,
    transpose,
)
from ...typing import parse
from .._cochain import CochainDiscretization
from ._cut_cochain import CutCellCochainPlan
from ._cut_transition import MultivaluedCutCellTransition


@final
class CutCellCochainTransferPlan(StrictModule, NonTrainableState):
    """Minimum-change commuting projection on admitted active coordinates.

    Relative pairings are restricted before inversion. Public actions accept
    full storage coordinates and zero-extend the active-coordinate results.
    """

    source_plan: CutCellCochainPlan
    target_plan: CutCellCochainPlan
    source_state: CochainDiscretization
    target_state: CochainDiscretization
    complex_map: ComplexMap
    evidence: ComplexMapEvidence
    transpose_operators: tuple[AbstractLinearOperator, ...]
    adjoint_operators: tuple[AbstractLinearOperator, ...]
    source_indices: tuple[Array, ...]
    target_indices: tuple[Array, ...]
    boundary: ComplexBoundary = eqx.field(static=True)
    topology_changed: bool = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_plan: CutCellCochainPlan,
        target_plan: CutCellCochainPlan,
        source_state: CochainDiscretization,
        target_state: CochainDiscretization,
        /,
        *,
        cell_transition: MultivaluedCutCellTransition | None = None,
        tolerance: float = 1.0e-9,
        boundary: ComplexBoundary = "absolute",
    ) -> None:
        if not isinstance(source_plan, CutCellCochainPlan) or not isinstance(
            target_plan, CutCellCochainPlan
        ):
            raise TypeError("Cut cochain transfer requires source/target plans.")
        if not isinstance(source_state, CochainDiscretization) or not isinstance(
            target_state, CochainDiscretization
        ):
            raise TypeError("Cut cochain transfer requires cochain realizations.")
        boundary = parse(boundary, ComplexBoundary, "boundary")
        for metric in (*source_state.hodges, *target_state.hodges):
            metric.admit()
        source_topology = source_state.topology
        target_topology = target_state.topology
        if (
            source_topology.topology_id != source_plan.complex.mesh.topology.topology_id
            or target_topology.topology_id
            != target_plan.complex.mesh.topology.topology_id
        ):
            raise ValueError("Cut cochain plans and realizations are incompatible.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Cut cochain transfer tolerance must be positive.")
        dimension = source_topology.dimension
        if target_topology.dimension != dimension:
            raise ValueError("Cut cochain transfer dimensions must match.")
        source = source_state.hilbert_complex(boundary=boundary)
        target = target_state.hilbert_complex(boundary=boundary)
        source_indices = tuple(
            np.asarray(source_state.active_indices(k, boundary=boundary), dtype=np.int32)
            for k in range(dimension + 1)
        )
        target_indices = tuple(
            np.asarray(target_state.active_indices(k, boundary=boundary), dtype=np.int32)
            for k in range(dimension + 1)
        )
        source_d = tuple(
            incidence.scipy_boundary()
            .T.tocsr()[source_indices[k + 1]][:, source_indices[k]]
            .toarray()
            for k, incidence in enumerate(source_topology.incidences)
        )
        target_d = tuple(
            incidence.scipy_boundary()
            .T.tocsr()[target_indices[k + 1]][:, target_indices[k]]
            .toarray()
            for k, incidence in enumerate(target_topology.incidences)
        )
        topology_changed = source_topology.topology_id != target_topology.topology_id
        source_counts = tuple(entity.count for entity in source_topology.entity_sets)
        target_counts = tuple(entity.count for entity in target_topology.entity_sets)
        primal: list[np.ndarray] = []
        if not topology_changed:
            for lower, upper in zip(source_indices, target_indices, strict=True):
                primal.append((upper[:, None] == lower[None, :]).astype(np.float64))
        else:
            if (
                not isinstance(cell_transition, MultivaluedCutCellTransition)
                or cell_transition.source.topology_id != source_plan.complex.topology_id
                or cell_transition.target.topology_id != target_plan.complex.topology_id
            ):
                raise ValueError(
                    "Changed cut topology requires its exact cell common refinement."
                )
            remap = cell_transition.remap
            cell_matrix = np.zeros(
                (target_counts[-1], source_counts[-1]), dtype=np.float64
            )
            weights = (
                np.asarray(remap.intersection_measures)
                / np.asarray(remap.source_volumes)[
                    np.asarray(remap.source_indices, dtype=np.int32)
                ]
            )
            np.add.at(
                cell_matrix,
                (
                    np.asarray(remap.target_routes, dtype=np.int32),
                    np.asarray(remap.source_indices, dtype=np.int32),
                ),
                weights,
            )
            primal = [np.empty((0, 0), dtype=np.float64) for _ in source_counts]
            primal[-1] = cell_matrix[np.ix_(target_indices[-1], source_indices[-1])]
            for degree in reversed(range(dimension)):
                desired = primal[degree + 1] @ source_d[degree]
                pseudoinverse = np.linalg.pinv(target_d[degree], rcond=tolerance_)
                source_ids = np.asarray(
                    source_topology.entities(degree).entity_ids, dtype=np.int64
                )[source_indices[degree]]
                target_ids = np.asarray(
                    target_topology.entities(degree).entity_ids, dtype=np.int64
                )[target_indices[degree]]
                geometric = (target_ids[:, None] == source_ids[None, :]).astype(
                    np.float64
                )
                null_projection = (
                    np.eye(target_ids.size, dtype=np.float64)
                    - pseudoinverse @ target_d[degree]
                )
                primal[degree] = pseudoinverse @ desired + null_projection @ geometric
        defects = tuple(
            float(
                np.max(
                    np.abs(target_d[k] @ primal[k] - primal[k + 1] @ source_d[k]),
                    initial=0.0,
                )
            )
            for k in range(dimension)
        )
        if max(defects, default=0.0) > tolerance_:
            raise ValueError(
                "Cut cochain topology change has no admitted commuting projection."
            )
        map_id = canonical_fingerprint(
            {
                "kind": "cut-cell-cochain-complex-map",
                "source": source.complex_id,
                "target": target.complex_id,
                "boundary": boundary,
                "source_mask": source_indices,
                "target_mask": target_indices,
                "matrices": [array_tree_fingerprint(matrix) for matrix in primal],
            }
        )
        maps = tuple(
            DenseLinearOperator(
                matrix,
                source=source.space(k),
                target=target.space(k),
                operator_id=canonical_fingerprint({"map": map_id, "degree": k}),
            )
            for k, matrix in enumerate(primal)
        )
        self.source_plan = source_plan
        self.target_plan = target_plan
        self.source_state = source_state
        self.target_state = target_state
        self.complex_map = ComplexMap(source, target, maps, map_id=map_id)
        self.evidence = ComplexMapEvidence(
            jnp.asarray(defects, dtype=jnp.float64), jnp.asarray(True, dtype=jnp.bool_)
        )
        self.transpose_operators = tuple(transpose(operator) for operator in maps)
        self.adjoint_operators = tuple(adjoint(operator) for operator in maps)
        self.source_indices = tuple(jnp.asarray(indices) for indices in source_indices)
        self.target_indices = tuple(jnp.asarray(indices) for indices in target_indices)
        self.boundary = boundary
        self.topology_changed = topology_changed
        self.plan_id = map_id

    @staticmethod
    def _apply(
        operator: AbstractLinearOperator,
        values: ArrayLike,
        indices: Array,
        output_indices: Array,
        input_size: int,
        output_size: int,
        /,
    ) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != input_size:
            raise ValueError(
                "Cut cochain values must begin with the expected entity count."
            )
        selected = value[indices]
        if selected.ndim == 1:
            image = operator.mv(selected)
        else:
            columns = selected.reshape((indices.size, prod(value.shape[1:]))).T
            image = jax.vmap(operator.mv)(columns).T.reshape(
                (output_indices.size,) + value.shape[1:]
            )
        return (
            jnp.zeros((output_size,) + value.shape[1:], dtype=image.dtype)
            .at[output_indices]
            .set(image)
        )

    def _degree(self, degree: int, /) -> int:
        if degree < 0 or degree >= len(self.complex_map.maps):
            raise ValueError("Cut cochain transfer degree is out of range.")
        return degree

    def apply(self, degree: int, values: ArrayLike, /) -> Array:
        k = self._degree(degree)
        return self._apply(
            self.complex_map.maps[k],
            values,
            self.source_indices[k],
            self.target_indices[k],
            self.source_state.topology.entities(k).count,
            self.target_state.topology.entities(k).count,
        )

    def transpose(self, degree: int, values: ArrayLike, /) -> Array:
        k = self._degree(degree)
        return self._apply(
            self.transpose_operators[k],
            values,
            self.target_indices[k],
            self.source_indices[k],
            self.target_state.topology.entities(k).count,
            self.source_state.topology.entities(k).count,
        )

    def adjoint(self, degree: int, values: ArrayLike, /) -> Array:
        k = self._degree(degree)
        return self._apply(
            self.adjoint_operators[k],
            values,
            self.target_indices[k],
            self.source_indices[k],
            self.target_state.topology.entities(k).count,
            self.source_state.topology.entities(k).count,
        )


__all__ = ["CutCellCochainTransferPlan"]
