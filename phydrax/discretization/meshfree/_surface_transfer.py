# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Sparse concentration remaps with measured conservation and adjoint evidence."""

from __future__ import annotations

from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...sparse import EdgeRelation, RowRelation, SparseCoordinateOperator
from ...typing import Dim, Float64, parse
from .._gram import diagonal_gram_space
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._topology_epoch import TopologyEpoch, TopologyEpochTransition
from .._transfer import FieldTransfer, TransferProperties
from ._stencils import PreparedLocalStencils


SurfaceTransferMode: TypeAlias = Literal["high-order", "positive"]


class TransferSourceDim(Dim):
    """Old compact coordinates."""


class TransferTargetDim(Dim):
    """New compact coordinates."""


class TransferRouteDim(Dim):
    """Flat sparse transfer routes."""


class TransferSlotDim(Dim):
    """Fixed row-local transfer neighbors."""


@final
class SurfaceTransferEvidence(StrictModule):
    __strict_contract__ = True
    column_residual: Float64[TransferSourceDim]
    row_residual: Float64[TransferTargetDim]
    conservative: bool = eqx.field(static=True)
    constant_preserving: bool = eqx.field(static=True)
    nonnegative: bool = eqx.field(static=True)
    correction_norm: float = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)


@final
class PreparedSurfaceTransfer(StrictModule):
    __strict_contract__ = True
    transfer: FieldTransfer
    evidence: SurfaceTransferEvidence
    source_measures: Float64[TransferSourceDim]
    target_measures: Float64[TransferTargetDim]

    def apply(self, concentrations: ArrayLike, /) -> Array:
        return self.transfer.primal_operator.mv(jnp.asarray(concentrations))

    def apply_content(self, content: ArrayLike, /) -> Array:
        """Convert extensive content to concentration before applying T."""
        return self.target_measures * self.apply(
            jnp.asarray(content) / self.source_measures
        )

    def epoch_transition(
        self, source: TopologyEpoch, target: TopologyEpoch, /
    ) -> TopologyEpochTransition:
        return TopologyEpochTransition(
            source,
            target,
            self.transfer,
            self.source_measures,
            self.target_measures,
            measure_defect_bound=jnp.abs(self.evidence.column_residual),
        )


@final
class SurfaceTransferPlan(StrictModule):
    __strict_contract__ = True
    relation: EdgeRelation | RowRelation
    coefficients: Float64[TransferRouteDim] | Float64[TransferTargetDim, TransferSlotDim]
    source_measures: Float64[TransferSourceDim]
    target_measures: Float64[TransferTargetDim]
    mode: SurfaceTransferMode = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        relation: EdgeRelation | RowRelation,
        coefficients: ArrayLike,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        source_id: str,
        target_id: str,
        mode: SurfaceTransferMode = "high-order",
        tolerance: float = 1e-10,
    ) -> None:
        if not isinstance(relation, (EdgeRelation, RowRelation)):
            raise TypeError("Cross-target transfer needs a native sparse relation.")
        if isinstance(relation, RowRelation):
            if relation.case_shape or len(relation.target_shape) != 1:
                raise ValueError(
                    "Surface transfers require one unbatched compact target vector."
                )
            target_size = relation.targets_per_case
        else:
            target_size = relation.target_size
        old = np.asarray(source_measures, dtype=np.float64)
        new = np.asarray(target_measures, dtype=np.float64)
        if old.shape != (relation.source_size,) or new.shape != (target_size,):
            raise ValueError("Cross-target measures must match compact relation spaces.")
        if not np.all(np.isfinite(old) & (old > 0)) or not np.all(
            np.isfinite(new) & (new > 0)
        ):
            raise ValueError("Conservative transfers require positive active measures.")
        base = np.asarray(coefficients, dtype=np.float64)
        if base.shape != relation.route_shape or not np.all(np.isfinite(base)):
            raise ValueError(
                "Cross-target stencil coefficients must be finite and match routes."
            )
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Transfer tolerance must be positive and finite.")
        if any(
            not isinstance(value, str) or not value.strip() or value != value.strip()
            for value in (source_id, target_id)
        ):
            raise ValueError(
                "Transfer supports require explicit canonical source_id and target_id."
            )
        mode_ = parse(mode, SurfaceTransferMode, "mode")
        self.relation, self.coefficients = relation, jnp.asarray(base)
        self.source_measures, self.target_measures = jnp.asarray(old), jnp.asarray(new)
        self.mode, self.tolerance = mode_, float(tolerance)
        self.source_id, self.target_id = source_id, target_id

    @classmethod
    def from_stencils(
        cls,
        stencils: PreparedLocalStencils,
        source_measures: ArrayLike,
        target_measures: ArrayLike,
        /,
        *,
        functional_index: int = 0,
        mode: SurfaceTransferMode = "high-order",
        tolerance: float = 1e-10,
    ) -> SurfaceTransferPlan:
        if not isinstance(stencils, PreparedLocalStencils):
            raise TypeError("Cross-target preparation requires PreparedLocalStencils.")
        if stencils.report.refused_rows or not 0 <= functional_index < len(
            stencils.weights
        ):
            raise ValueError(
                "Cross-target functional index or refused stencil rows are invalid."
            )
        return cls(
            stencils.neighborhood.relation,
            stencils.weights[functional_index],
            source_measures,
            target_measures,
            source_id=f"{stencils.neighborhood.neighborhood_id}:source",
            target_id=f"{stencils.neighborhood.neighborhood_id}:target",
            mode=mode,
            tolerance=tolerance,
        )

    def prepare(self) -> PreparedSurfaceTransfer:
        """Bounded host sparse preparation, never a global dense matrix.

        High-order: the minimum Euclidean correction on each existing column
        solves w_new.T T = w_old.T exactly. Positive: nonnegative base affinities
        define a column allocation of the old measure. Neither method claims
        constant preservation without checking every actual row.
        """
        edge = (
            self.relation
            if isinstance(self.relation, EdgeRelation)
            else self.relation.as_edge_relation()
        )
        valid = np.asarray(edge.valid)
        columns = np.asarray(edge.source_indices)[valid]
        rows = np.asarray(edge.target_indices)[valid]
        original = np.asarray(self.coefficients).reshape(-1)[valid]
        old, new = np.asarray(self.source_measures), np.asarray(self.target_measures)
        coverage = np.bincount(columns, minlength=old.size)
        if np.any(coverage == 0):
            raise ValueError(
                "Conservation is infeasible: an old point has no cross-target route."
            )
        if self.mode == "high-order":
            moment = np.bincount(
                columns, weights=new[rows] * original, minlength=old.size
            )
            denominator = np.bincount(columns, weights=new[rows] ** 2, minlength=old.size)
            values = original + new[rows] * ((old - moment) / denominator)[columns]
        else:
            if np.any(original < 0):
                raise ValueError(
                    "Positive transfer requires nonnegative base allocation coefficients."
                )
            affinity = original
            normalizer = np.bincount(
                columns, weights=new[rows] * affinity, minlength=old.size
            )
            if np.any(normalizer <= 0):
                raise ValueError(
                    "Nonnegative allocation is infeasible: a column has no positive affinity."
                )
            values = affinity * (old / normalizer)[columns]
        column_residual = (
            np.bincount(columns, weights=new[rows] * values, minlength=old.size) - old
        )
        row_residual = np.bincount(rows, weights=values, minlength=new.size) - 1
        conservative = bool(np.max(np.abs(column_residual) / old) <= self.tolerance)
        constant = bool(np.max(np.abs(row_residual)) <= self.tolerance)
        positive = bool(np.all(values >= 0))
        if not conservative:
            raise ValueError(
                "Sparse conservative correction failed its measured constraint."
            )
        source_metric_id = canonical_fingerprint(
            {"support": self.source_id, "measures": old}
        )
        target_metric_id = canonical_fingerprint(
            {"support": self.target_id, "measures": new}
        )
        source_space, _source_mass = diagonal_gram_space(
            old, dtype=np.float64, space_id=source_metric_id
        )
        target_space, _target_mass = diagonal_gram_space(
            new, dtype=np.float64, space_id=target_metric_id
        )
        relation = EdgeRelation(columns, rows, source_size=old.size, target_size=new.size)
        reverse = EdgeRelation(rows, columns, source_size=new.size, target_size=old.size)
        identifier = canonical_fingerprint(
            {
                "kind": "surface-transfer",
                "source": self.source_id,
                "target": self.target_id,
                "rows": rows,
                "columns": columns,
                "values": values,
                "source_measures": old,
                "target_measures": new,
            }
        )
        primal = SparseCoordinateOperator(
            relation,
            values,
            source=source_space,
            target=target_space,
            operator_id=identifier + ":primal",
        )
        dual = SparseCoordinateOperator(
            reverse,
            values,
            source=target_space,
            target=source_space,
            operator_id=identifier + ":dual",
        )
        hilbert = SparseCoordinateOperator(
            reverse,
            values * new[rows] / old[columns],
            source=target_space,
            target=source_space,
            operator_id=identifier + ":hilbert",
        )
        source = DiscreteFieldSpace(
            "concentration",
            self.source_id,
            TensorDofLayout(("point",), (old.size,)),
            source_space,
            representation="point_value",
        )
        target = DiscreteFieldSpace(
            "concentration",
            self.target_id,
            TensorDofLayout(("point",), (new.size,)),
            target_space,
            representation="point_value",
        )
        transfer = FieldTransfer(
            source,
            target,
            primal,
            dual_pullback_operator=dual,
            hilbert_adjoint_operator=hilbert,
            properties=TransferProperties(
                conservative=True,
                constant_preserving=constant,
                positivity_preserving=positive,
                adjoint_paired=True,
                differentiable_geometry=False,
            ),
        )
        evidence = SurfaceTransferEvidence(
            jnp.asarray(column_residual),
            jnp.asarray(row_residual),
            conservative,
            constant,
            positive,
            float(np.linalg.norm(values - original)),
            self.tolerance,
        )
        return PreparedSurfaceTransfer(
            transfer, evidence, self.source_measures, self.target_measures
        )


__all__ = [
    "SurfaceTransferMode",
    "SurfaceTransferEvidence",
    "PreparedSurfaceTransfer",
    "SurfaceTransferPlan",
]
