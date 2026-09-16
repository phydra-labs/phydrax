#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import AbstractLinearOperator
from ._reduction import TrialTestReduction


class ReducedLiftArtifact(StrictModule, NonTrainableState):
    """Affine boundary lift with trace and time-derivative evidence."""

    lift_coordinates: Array
    trace_values: Array
    trace_operator: AbstractLinearOperator
    term_ids: tuple[str, ...] = eqx.field(static=True)
    state_space_id: str = eqx.field(static=True)
    boundary_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    evidence_ids: tuple[str, ...] = eqx.field(static=True)
    maximum_trace_defect: float = eqx.field(static=True)
    artifact_id: str = eqx.field(static=True)

    def __init__(
        self,
        reduction: TrialTestReduction,
        trace_operator: AbstractLinearOperator,
        lift_terms: Sequence[PyTree[Array]],
        boundary_terms: ArrayLike,
        /,
        *,
        term_ids: Sequence[str],
        evidence_ids: Sequence[str],
        tolerance: float = 1.0e-10,
    ):
        if not isinstance(reduction, TrialTestReduction):
            raise TypeError("reduction must be a TrialTestReduction.")
        if not isinstance(trace_operator, AbstractLinearOperator):
            raise TypeError("trace_operator must be an AbstractLinearOperator.")
        if not trace_operator.source.compatible(reduction.trial.full_space):
            raise ValueError("Trace source must match the trial full space.")
        lifts = tuple(reduction.trial.full_space.validate(value) for value in lift_terms)
        terms = tuple(str(value) for value in term_ids)
        evidence = tuple(str(value) for value in evidence_ids)
        if not lifts or len(lifts) != len(terms):
            raise ValueError("Lift terms and term IDs must be non-empty and aligned.")
        if any(not value for value in (*terms, *evidence)):
            raise ValueError("Lift and evidence IDs must be non-empty.")
        coordinates = jnp.stack(
            tuple(reduction.trial.full_space.flatten(value) for value in lifts)
        )
        boundary = jnp.asarray(boundary_terms)
        if boundary.shape != (len(lifts), trace_operator.target.size):
            raise ValueError(
                "boundary_terms must match lift terms and trace output size."
            )
        actual = jnp.stack(
            tuple(
                trace_operator.target.flatten(trace_operator.mv(value)) for value in lifts
            )
        )
        trial_vectors = reduction.trial.prolongation.mv_block(
            jnp.eye(reduction.trial_rank, dtype=coordinates.dtype)
        )
        homogeneous = trace_operator.mv_block(trial_vectors)
        defect = max(
            float(np.max(np.abs(np.asarray(actual - boundary)))),
            float(np.max(np.abs(np.asarray(homogeneous)))),
        )
        tolerance_ = float(tolerance)
        if not np.isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if defect > tolerance_:
            raise ValueError("Lift or trial basis violates the declared trace contract.")
        self.lift_coordinates = coordinates
        self.trace_values = boundary
        self.trace_operator = trace_operator
        self.term_ids = terms
        self.state_space_id = reduction.trial.full_space.space_id
        self.boundary_space_id = trace_operator.target.space_id
        self.support_id = reduction.support_id
        self.geometry_id = reduction.geometry_id
        self.evidence_ids = evidence
        self.maximum_trace_defect = defect
        self.artifact_id = canonical_fingerprint(
            {
                "kind": "reduced-lift-artifact",
                "reduction": reduction.reduction_id,
                "trace": trace_operator.operator_id,
                "terms": list(terms),
                "support": reduction.support_id,
                "geometry": reduction.geometry_id,
                "evidence": list(evidence),
                "content": array_tree_fingerprint(
                    {"lift": coordinates, "boundary": boundary}
                )["sha256"],
            }
        )

    def evaluate(self, coefficients: ArrayLike, /):
        values = jnp.asarray(coefficients)
        if values.shape[-1:] != (len(self.term_ids),):
            raise ValueError("Lift coefficients must end in the lift-term axis.")
        coordinates = contract("...p,pn->...n", values, self.lift_coordinates)
        if coordinates.ndim == 1:
            return coordinates
        return coordinates

    def boundary_values(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape[-1:] != (len(self.term_ids),):
            raise ValueError("Lift coefficients must end in the lift-term axis.")
        return contract("...p,pb->...b", values, self.trace_values)


__all__ = ["ReducedLiftArtifact"]
