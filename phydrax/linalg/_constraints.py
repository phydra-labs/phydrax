#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import equinox as eqx
import jax
import numpy as np
from jax import Array, core as jax_core
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._assembly import plan_sparse_assembly, prepare_sparse_assembly, SparseAssemblyPolicy
from ._operators import AbstractLinearOperator, FunctionLinearOperator
from ._properties import LinearCapabilityError, OperatorProperties
from ._spaces import AbstractVectorSpace, DualSpace


if TYPE_CHECKING:
    from ..sparse import SparseCoordinateOperator


class ConstraintMap(StrictModule, NonTrainableState):
    """Affine parameterization ``u = P z + g`` over declared vector spaces."""

    full_space: AbstractVectorSpace
    reduced_space: AbstractVectorSpace
    prolongation: AbstractLinearOperator
    dual_pullback: AbstractLinearOperator
    constraint_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        full_space: AbstractVectorSpace,
        reduced_space: AbstractVectorSpace,
        prolongation: AbstractLinearOperator,
        /,
        *,
        constraint_id: str | None = None,
    ) -> None:
        if not isinstance(full_space, AbstractVectorSpace) or not isinstance(
            reduced_space, AbstractVectorSpace
        ):
            raise TypeError("Constraint spaces must be AbstractVectorSpace values.")
        if not prolongation.source.compatible(
            reduced_space
        ) or not prolongation.target.compatible(full_space):
            raise ValueError("Constraint prolongation must map reduced to full space.")
        if prolongation.batch_shape:
            raise ValueError("Constraint prolongation must be unbatched.")
        full_dual = DualSpace(full_space)
        reduced_dual = DualSpace(reduced_space)
        pullback = FunctionLinearOperator(
            lambda value: prolongation.transpose_mv(value),
            source=full_dual,
            target=reduced_dual,
            transpose_action=lambda value: prolongation.mv(value),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constraint-dual-pullback",
                    "prolongation": prolongation.operator_id,
                }
            ),
        )
        if constraint_id is not None and not isinstance(constraint_id, str):
            raise TypeError("constraint_id must be a string or None.")
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "constraint-map",
                    "full_space": full_space.space_id,
                    "reduced_space": reduced_space.space_id,
                    "prolongation": prolongation.operator_id,
                }
            )
            if constraint_id is None
            else constraint_id
        )
        if not resolved_id:
            raise ValueError("constraint_id must be non-empty.")
        self.full_space = full_space
        self.reduced_space = reduced_space
        self.prolongation = prolongation
        self.dual_pullback = pullback
        self.constraint_id = resolved_id

    def expand(self, reduced: object, lift: object, /) -> PyTree[Array]:
        reduced_ = self.reduced_space.validate(reduced)
        lift_ = self.full_space.validate(lift)
        return self.full_space.validate(
            jax.tree.map(
                lambda correction, offset: correction + offset,
                self.prolongation.mv(reduced_),
                lift_,
            )
        )

    def homogeneous_correction(self, reduced: object, /) -> PyTree[Array]:
        return self.full_space.validate(
            self.prolongation.mv(self.reduced_space.validate(reduced))
        )

    def pullback_dual(self, residual: PyTree[Any], /) -> PyTree[Array]:
        return self.dual_pullback.mv(residual)

    def reduce_vector(self, residual: PyTree[Any], /) -> PyTree[Array]:
        return self.prolongation.adjoint_mv(self.full_space.validate(residual))

    def coordinate_selection(self) -> np.ndarray | None:
        """Full coordinates kept by a pure coordinate-selection prolongation.

        Returns ``selected`` with ``selected[r]`` the flattened full coordinate of
        reduced coordinate ``r`` when ``P`` stores exactly one unit coefficient
        per reduced coordinate on distinct full coordinates (strong Dirichlet
        elimination), and ``None`` for every other prolongation, including
        traced structure.
        """
        from ..sparse import RowRelation, SparseCoordinateOperator

        prolongation = self.prolongation
        if (
            not isinstance(prolongation, SparseCoordinateOperator)
            or prolongation.block_shape is not None
        ):
            return None
        relation = prolongation.relation
        if isinstance(relation, RowRelation):
            relation = relation.as_edge_relation()
        leaves = (
            relation.source_indices,
            relation.target_indices,
            relation.valid,
            prolongation.coefficients,
        )
        if any(isinstance(leaf, jax_core.Tracer) for leaf in leaves):
            return None
        valid = np.asarray(relation.valid, dtype=np.bool_)
        sources = np.asarray(relation.source_indices)[valid]
        targets = np.asarray(relation.target_indices)[valid]
        coefficients = np.asarray(prolongation.coefficients)[valid]
        size = self.reduced_space.size
        if sources.size != size or np.any(coefficients != 1):
            return None
        order = np.argsort(sources, kind="stable")
        if not np.array_equal(sources[order], np.arange(size)):
            return None
        selected = targets[order].astype(np.int64)
        if np.unique(selected).size != size:
            return None
        return selected

    def reduce_sparse_operator(
        self,
        operator: AbstractLinearOperator,
        /,
        *,
        policy: SparseAssemblyPolicy | None = None,
    ) -> SparseCoordinateOperator:
        """Exactly eliminate constrained coordinates from structurally sparse storage.

        For a coordinate-selection constraint (see `coordinate_selection`) this
        returns the free-by-free block ``P^T A P`` of ``operator`` as sparse
        coordinates on the reduced space: the kept entries of ``A``'s exact
        sparse assembly, renumbered, without densifying anything. Columns of
        constrained coordinates belong to the right-hand side through the lift
        ``g`` (``-P^T (A g - f)``) and are not stored here. Claimed
        self-adjointness, (semi)definiteness and diagonality survive the
        congruence with ``"transformed"`` evidence when ``A`` certifies them.

        Refuses constraints that are not coordinate selections, operators whose
        sparse lowering would materialize, and operators that do not act on the
        full space.
        """
        from ..sparse import EdgeRelation, SparseCoordinateOperator

        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not operator.source.compatible(self.full_space):
            raise ValueError("The operator must act on the constraint's full space.")
        if operator.target.compatible(self.full_space):
            target: AbstractVectorSpace = self.reduced_space
        elif operator.target.compatible(DualSpace(self.full_space)):
            target = DualSpace(self.reduced_space)
        else:
            raise ValueError(
                "The operator must map the full space to itself or to its dual."
            )
        selected = self.coordinate_selection()
        if selected is None:
            raise LinearCapabilityError(
                "Sparse elimination requires a concrete coordinate-selection "
                "constraint; general affine constraints need a Galerkin product."
            )
        plan = plan_sparse_assembly(operator, policy)
        if plan.uses_materialization:
            raise LinearCapabilityError(
                "Sparse elimination requires a structurally sparse lowering; "
                "the operator would be densely materialized."
            )
        assembled = prepare_sparse_assembly(plan, operator).operator
        if not isinstance(assembled, SparseCoordinateOperator):
            raise TypeError("Sparse assembly did not produce coordinate storage.")
        values = assembled.coefficients
        reduced_index = np.full((self.full_space.size,), -1, dtype=np.int64)
        reduced_index[selected] = np.arange(selected.size, dtype=np.int64)
        rows = reduced_index[np.asarray(plan.row_indices, dtype=np.int64)]
        columns = reduced_index[np.asarray(plan.column_indices, dtype=np.int64)]
        kept = np.flatnonzero((rows >= 0) & (columns >= 0))
        size = self.reduced_space.size
        relation = EdgeRelation(
            columns[kept].astype(np.int32),
            rows[kept].astype(np.int32),
            source_size=size,
            target_size=size,
        )
        properties = operator.properties
        claims = {
            name: True
            for name in (
                "diagonal",
                "self_adjoint",
                "positive_semidefinite",
                "positive_definite",
            )
            if getattr(properties, name)
        }
        return SparseCoordinateOperator(
            relation,
            values[kept],
            source=self.reduced_space,
            target=target,
            properties=OperatorProperties(
                **claims,
                evidence={
                    name: "transformed" for name in claims if properties.certifies(name)
                },
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constraint-eliminated-sparse-operator",
                    "constraint": self.constraint_id,
                    "operator": operator.operator_id,
                    "plan": plan.plan_id,
                }
            ),
        )


def compose_constraint_maps(
    outer: ConstraintMap,
    inner: ConstraintMap,
    /,
    *,
    constraint_id: str | None = None,
) -> ConstraintMap:
    """Compose nested affine charts' homogeneous maps.

    If ``outer`` expands ``y`` into ``x`` and ``inner`` expands ``z`` into
    ``y``, the returned map expands ``z`` into ``x``. Their dynamic lifts
    compose as ``outer.expand(inner_lift, outer_lift)``.
    """
    if not isinstance(outer, ConstraintMap) or not isinstance(inner, ConstraintMap):
        raise TypeError("outer and inner must be ConstraintMap values.")
    if not inner.full_space.compatible(outer.reduced_space):
        raise ValueError(
            "The inner full space must match the outer reduced space exactly."
        )
    if constraint_id is not None and not isinstance(constraint_id, str):
        raise TypeError("constraint_id must be a string or None.")
    identifier = (
        canonical_fingerprint(
            {
                "kind": "composed-constraint-map",
                "outer": outer.constraint_id,
                "inner": inner.constraint_id,
            }
        )
        if constraint_id is None
        else constraint_id
    )
    return ConstraintMap(
        outer.full_space,
        inner.reduced_space,
        outer.prolongation @ inner.prolongation,
        constraint_id=identifier,
    )


__all__ = ["ConstraintMap", "compose_constraint_maps"]
