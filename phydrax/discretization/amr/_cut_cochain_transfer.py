#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Commuting and metric-adjoint cochain transfer across cut-complex epochs."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cut_cochain import CutCellCochainPlan, CutCellCochainState
from ._cut_transition import MultivaluedCutCellTransition


class CutCellCochainTransferEvidence(StrictModule, NonTrainableState):
    """Degree-wise commuting defects for one admitted topology transfer."""

    commuting_defects: tuple[float, ...] = eqx.field(static=True)
    maximum_commuting_defect: float = eqx.field(static=True)
    topology_changed: bool = eqx.field(static=True)
    valid: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class CutCellCochainTransferPlan(StrictModule, NonTrainableState):
    """Minimum-change commuting projections with exact transpose/metric adjoint."""

    source_plan: CutCellCochainPlan
    target_plan: CutCellCochainPlan
    source_state: CutCellCochainState
    target_state: CutCellCochainState
    primal_matrices: tuple[Array, ...]
    transpose_matrices: tuple[Array, ...]
    adjoint_matrices: tuple[Array, ...]
    evidence: CutCellCochainTransferEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_plan: CutCellCochainPlan,
        target_plan: CutCellCochainPlan,
        source_state: CutCellCochainState,
        target_state: CutCellCochainState,
        /,
        *,
        cell_transition: MultivaluedCutCellTransition | None = None,
        tolerance: float = 1.0e-9,
    ):
        if not isinstance(source_plan, CutCellCochainPlan) or not isinstance(
            target_plan, CutCellCochainPlan
        ):
            raise TypeError("Cut cochain transfer requires source/target plans.")
        if not isinstance(source_state, CutCellCochainState) or not isinstance(
            target_state, CutCellCochainState
        ):
            raise TypeError("Cut cochain transfer requires accepted metric states.")
        if (
            source_state.complex_id != source_plan.complex.topology_id
            or target_state.complex_id != target_plan.complex.topology_id
            or not bool(source_state.metrics.valid)
            or not bool(target_state.metrics.valid)
        ):
            raise ValueError("Cut cochain plans and metric states are incompatible.")
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ <= 0.0:
            raise ValueError("Cut cochain transfer tolerance must be positive.")
        source_topology = source_state.topology.topology
        target_topology = target_state.topology.topology
        dimension = source_topology.dimension
        if target_topology.dimension != dimension:
            raise ValueError("Cut cochain transfer dimensions must match.")
        source_counts = tuple(entity.count for entity in source_topology.entity_sets)
        target_counts = tuple(entity.count for entity in target_topology.entity_sets)
        topology_changed = (
            source_plan.complex.topology_id != target_plan.complex.topology_id
        )

        def derivative(topology, degree: int) -> np.ndarray:
            incidence = topology.incidences[degree]
            relation = incidence.relation
            valid = np.asarray(relation.valid, dtype=bool)
            matrix = np.zeros((relation.target_size, relation.source_size), dtype=float)
            np.add.at(
                matrix,
                (
                    np.asarray(relation.target_indices, dtype=np.int32)[valid],
                    np.asarray(relation.source_indices, dtype=np.int32)[valid],
                ),
                np.asarray(incidence.signs, dtype=float)[valid],
            )
            return matrix

        if not topology_changed and source_counts == target_counts:
            primal = [np.eye(count, dtype=float) for count in source_counts]
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
            cell_matrix = np.zeros((target_counts[-1], source_counts[-1]), dtype=float)
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
            primal = [np.empty((0, 0)) for _ in source_counts]
            primal[-1] = cell_matrix
            for degree in reversed(range(dimension)):
                target_derivative = derivative(target_topology, degree)
                source_derivative = derivative(source_topology, degree)
                desired = primal[degree + 1] @ source_derivative
                pseudoinverse = np.linalg.pinv(target_derivative, rcond=tolerance_)
                source_ids = np.asarray(
                    source_topology.entities(degree).entity_ids, dtype=np.int64
                )
                target_ids = np.asarray(
                    target_topology.entities(degree).entity_ids, dtype=np.int64
                )
                source_by_id = {
                    int(identifier): index for index, identifier in enumerate(source_ids)
                }
                geometric = np.zeros(
                    (target_counts[degree], source_counts[degree]), dtype=float
                )
                for target_index, identifier in enumerate(target_ids):
                    source_index = source_by_id.get(int(identifier))
                    if source_index is not None:
                        geometric[target_index, source_index] = 1.0
                null_projection = np.eye(target_counts[degree]) - (
                    pseudoinverse @ target_derivative
                )
                primal[degree] = pseudoinverse @ desired + null_projection @ geometric
        defects = []
        for degree in range(dimension):
            commutator = derivative(target_topology, degree) @ primal[degree] - primal[
                degree + 1
            ] @ derivative(source_topology, degree)
            defects.append(float(np.max(np.abs(commutator), initial=0.0)))
        maximum_defect = max(defects, default=0.0)
        if maximum_defect > tolerance_:
            raise ValueError(
                "Cut cochain topology change has no admitted commuting projection."
            )
        transpose_matrices = tuple(matrix.T for matrix in primal)
        adjoints = []
        for degree, matrix in enumerate(primal):
            source_metric = np.asarray(
                source_state.metrics.hodge_stars[degree], dtype=float
            )
            target_metric = np.asarray(
                target_state.metrics.hodge_stars[degree], dtype=float
            )
            adjoints.append((matrix.T * target_metric[None, :]) / source_metric[:, None])
        evidence = CutCellCochainTransferEvidence(
            commuting_defects=tuple(defects),
            maximum_commuting_defect=maximum_defect,
            topology_changed=topology_changed,
            valid=True,
            evidence_id=canonical_fingerprint(
                {
                    "kind": "cut-cell-cochain-transfer-evidence",
                    "source": source_plan.complex.topology_id,
                    "target": target_plan.complex.topology_id,
                    "defects": defects,
                    "tolerance": tolerance_,
                }
            ),
        )
        self.source_plan = source_plan
        self.target_plan = target_plan
        self.source_state = source_state
        self.target_state = target_state
        self.primal_matrices = tuple(jnp.asarray(matrix) for matrix in primal)
        self.transpose_matrices = tuple(
            jnp.asarray(matrix) for matrix in transpose_matrices
        )
        self.adjoint_matrices = tuple(jnp.asarray(matrix) for matrix in adjoints)
        self.evidence = evidence
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cut-cell-cochain-transfer-plan",
                "source": source_state.state_id,
                "target": target_state.state_id,
                "primal": [array_tree_fingerprint(matrix) for matrix in primal],
                "evidence": evidence.evidence_id,
            }
        )

    @staticmethod
    def _apply(matrix: Array, values: ArrayLike, size: int, name: str, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != size:
            raise ValueError(f"{name} must begin with the expected entity count.")
        return jnp.tensordot(matrix.astype(value.dtype), value, axes=((1,), (0,)))

    def apply(self, degree: int, values: ArrayLike, /) -> Array:
        degree_ = int(degree)
        if degree_ < 0 or degree_ >= len(self.primal_matrices):
            raise ValueError("Cut cochain transfer degree is out of range.")
        return self._apply(
            self.primal_matrices[degree_],
            values,
            self.primal_matrices[degree_].shape[1],
            "Source cochain",
        )

    def transpose(self, degree: int, values: ArrayLike, /) -> Array:
        degree_ = int(degree)
        if degree_ < 0 or degree_ >= len(self.transpose_matrices):
            raise ValueError("Cut cochain transpose degree is out of range.")
        return self._apply(
            self.transpose_matrices[degree_],
            values,
            self.transpose_matrices[degree_].shape[1],
            "Target cotangent",
        )

    def adjoint(self, degree: int, values: ArrayLike, /) -> Array:
        degree_ = int(degree)
        if degree_ < 0 or degree_ >= len(self.adjoint_matrices):
            raise ValueError("Cut cochain adjoint degree is out of range.")
        return self._apply(
            self.adjoint_matrices[degree_],
            values,
            self.adjoint_matrices[degree_].shape[1],
            "Target metric value",
        )


__all__ = [
    "CutCellCochainTransferEvidence",
    "CutCellCochainTransferPlan",
]
