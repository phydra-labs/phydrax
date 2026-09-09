#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, FailurePolicy, RankPolicy
from ...linalg.svd import svd, SVDProblem, SVDSolvePolicy
from ._calibration import BatteryCalibrationPlan, PreparedBatteryCalibration


def _parameter_coordinate_ids(position: PyTree[Any], /) -> tuple[str, ...]:
    identifiers: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(position)[0]:
        name = jax.tree_util.keystr(path) or "<root>"
        shape = np.asarray(leaf).shape
        if not shape:
            identifiers.append(name)
            continue
        for index in np.ndindex(shape):
            suffix = ",".join(str(value) for value in index)
            identifiers.append(f"{name}[{suffix}]")
    return tuple(identifiers)


class _BatteryIdentifiabilityNumerics(StrictModule):
    whitened_jacobian: Array
    fisher_information: Array
    singular_values: Array
    numerical_rank: Array
    condition_number: Array
    weak_directions: Array
    weak_direction_mask: Array
    svd_status: Array
    successful: Array


class BatteryIdentifiabilityPlan(StrictModule):
    """Whitened-Jacobian coordinates and native numerical-rank policy."""

    calibration: PreparedBatteryCalibration
    rank_policy: RankPolicy
    parameter_ids: tuple[str, ...] = eqx.field(static=True)
    parameter_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        calibration: BatteryCalibrationPlan | PreparedBatteryCalibration,
        /,
        *,
        rank_policy: RankPolicy | None = None,
    ):
        if isinstance(calibration, BatteryCalibrationPlan):
            prepared = calibration.prepare()
        elif isinstance(calibration, PreparedBatteryCalibration):
            prepared = calibration
        else:
            raise TypeError(
                "calibration must be a BatteryCalibrationPlan or "
                "PreparedBatteryCalibration."
            )
        policy = RankPolicy() if rank_policy is None else rank_policy
        if not isinstance(policy, RankPolicy):
            raise TypeError("rank_policy must be a phydrax.linalg.RankPolicy or None.")
        parameter_ids = _parameter_coordinate_ids(prepared.plan.parameter_space.initial)
        if not parameter_ids:
            raise ValueError("Identifiability requires at least one search coordinate.")
        self.calibration = prepared
        self.rank_policy = policy
        self.parameter_ids = parameter_ids
        self.parameter_count = len(parameter_ids)
        self.problem_id = prepared.problem_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "battery-identifiability-plan",
                "calibration_preparation_id": prepared.preparation_id,
                "problem_id": prepared.problem_id,
                "parameter_ids": list(parameter_ids),
                "rank_policy": {
                    "relative_cutoff": policy.relative_cutoff,
                    "absolute_cutoff": policy.absolute_cutoff,
                    "require_full_rank": policy.require_full_rank,
                },
            }
        )

    def evaluate_numerics(
        self, position: PyTree[Any] | None = None, /
    ) -> _BatteryIdentifiabilityNumerics:
        """Return JIT-compatible numerical evidence without assigning a host ID."""

        point = (
            self.calibration.plan.parameter_space.initial
            if position is None
            else position
        )
        return _evaluate_battery_identifiability_numerics(self, point)


class BatteryIdentifiabilityReport(StrictModule):
    """Fisher and weak-direction evidence from one whitened residual Jacobian."""

    whitened_jacobian: Array
    fisher_information: Array
    singular_values: Array
    numerical_rank: Array
    condition_number: Array
    weak_directions: Array
    weak_direction_mask: Array
    svd_status: Array
    successful: Array
    parameter_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    calibration_preparation_id: str = eqx.field(static=True)
    position_fingerprint: str = eqx.field(static=True)
    evidence_fingerprint: str = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        whitened_jacobian: Array,
        fisher_information: Array,
        singular_values: Array,
        numerical_rank: Array,
        condition_number: Array,
        weak_directions: Array,
        weak_direction_mask: Array,
        svd_status: Array,
        successful: Array,
        /,
        *,
        parameter_ids: tuple[str, ...],
        plan_id: str,
        problem_id: str,
        calibration_preparation_id: str,
        position_fingerprint: str,
        evidence_fingerprint: str,
        report_id: str,
    ):
        jacobian = jnp.asarray(whitened_jacobian)
        fisher = jnp.asarray(fisher_information)
        values = jnp.asarray(singular_values)
        rank = jnp.asarray(numerical_rank, dtype=jnp.int32)
        condition = jnp.asarray(condition_number)
        weak = jnp.asarray(weak_directions)
        weak_mask = jnp.asarray(weak_direction_mask, dtype=bool)
        status = jnp.asarray(svd_status, dtype=jnp.int32)
        success = jnp.asarray(successful, dtype=bool)
        count = len(parameter_ids)
        if jacobian.ndim != 2 or jacobian.shape[1] != count:
            raise ValueError(
                "Whitened Jacobian must have one column per parameter coordinate."
            )
        if fisher.shape != (count, count):
            raise ValueError(
                "Fisher information must be square in parameter coordinates."
            )
        if values.shape != (count,):
            raise ValueError(
                "Singular values must contain one value per parameter coordinate."
            )
        if weak.shape != (count, count) or weak_mask.shape != (count,):
            raise ValueError("Weak-direction evidence has incompatible parameter shape.")
        if any(value.shape != () for value in (rank, condition, status, success)):
            raise ValueError(
                "Identifiability rank, condition, status, and success must be scalar."
            )
        identifiers = (
            plan_id,
            problem_id,
            calibration_preparation_id,
            position_fingerprint,
            evidence_fingerprint,
            report_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("Identifiability provenance IDs must be non-empty strings.")
        self.whitened_jacobian = jacobian
        self.fisher_information = fisher
        self.singular_values = values
        self.numerical_rank = rank
        self.condition_number = condition
        self.weak_directions = weak
        self.weak_direction_mask = weak_mask
        self.svd_status = status
        self.successful = success
        self.parameter_ids = parameter_ids
        self.plan_id = plan_id
        self.problem_id = problem_id
        self.calibration_preparation_id = calibration_preparation_id
        self.position_fingerprint = position_fingerprint
        self.evidence_fingerprint = evidence_fingerprint
        self.report_id = report_id


def _evaluate_battery_identifiability_numerics(
    plan: BatteryIdentifiabilityPlan,
    position: PyTree[Any],
    /,
) -> _BatteryIdentifiabilityNumerics:
    plan.calibration.plan.parameter_space.constrain(position)
    flat_point, unravel = ravel_pytree(position)
    if int(flat_point.size) != plan.parameter_count:
        raise ValueError("Identifiability position does not match parameter coordinates.")

    def residual_from_flat(flat: Array, /) -> Array:
        return plan.calibration.residual(unravel(flat))

    residual = residual_from_flat(flat_point)
    jacobian = jax.jacfwd(residual_from_flat)(flat_point)
    finite_evaluation = jnp.all(jnp.isfinite(residual)) & jnp.all(jnp.isfinite(jacobian))
    safe_jacobian = jnp.where(finite_evaluation, jacobian, jnp.zeros_like(jacobian))
    fisher = safe_jacobian.T @ safe_jacobian

    row_count = max(int(safe_jacobian.shape[0]), plan.parameter_count)
    if row_count == int(safe_jacobian.shape[0]):
        svd_matrix = safe_jacobian
    else:
        svd_matrix = jnp.pad(
            safe_jacobian,
            ((0, row_count - int(safe_jacobian.shape[0])), (0, 0)),
        )
    operator = DenseLinearOperator(
        svd_matrix,
        operator_id=f"{plan.plan_id}:whitened-jacobian",
    )
    decomposition = svd(
        SVDProblem(operator, problem_id=f"{plan.problem_id}:identifiability"),
        policy=SVDSolvePolicy(
            count=plan.parameter_count,
            rank=plan.rank_policy,
            failure=FailurePolicy("status"),
        ),
    )
    singular_values = decomposition.singular_values
    numerical_rank = decomposition.numerical_rank
    full_rank = numerical_rank == plan.parameter_count
    condition = jnp.where(
        full_rank,
        singular_values[0] / singular_values[-1],
        jnp.asarray(jnp.inf, dtype=singular_values.dtype),
    )
    weak_mask = jnp.arange(plan.parameter_count, dtype=jnp.int32) >= numerical_rank
    right_vectors = jnp.asarray(decomposition.right_vectors)
    weak_directions = jnp.where(weak_mask[:, None], right_vectors.T, 0.0)
    retained_mask = ~weak_mask
    retained_converged = jnp.all(
        (~retained_mask) | jnp.asarray(decomposition.converged, dtype=bool)
    )
    decomposition_valid = (
        jnp.all(jnp.isfinite(singular_values))
        & jnp.all(jnp.isfinite(right_vectors))
        & retained_converged
    )
    rank_requirement = (not plan.rank_policy.require_full_rank) | full_rank
    successful = finite_evaluation & decomposition_valid & rank_requirement
    return _BatteryIdentifiabilityNumerics(
        whitened_jacobian=jacobian,
        fisher_information=fisher,
        singular_values=singular_values,
        numerical_rank=numerical_rank,
        condition_number=condition,
        weak_directions=weak_directions,
        weak_direction_mask=weak_mask,
        svd_status=decomposition.status,
        successful=successful,
    )


def evaluate_battery_identifiability(
    plan: BatteryIdentifiabilityPlan,
    position: PyTree[Any] | None = None,
    /,
) -> BatteryIdentifiabilityReport:
    """Evaluate numerical evidence and bind its concrete content identity."""

    if not isinstance(plan, BatteryIdentifiabilityPlan):
        raise TypeError("plan must be a BatteryIdentifiabilityPlan.")
    point = (
        plan.calibration.plan.parameter_space.initial if position is None else position
    )
    if any(
        isinstance(leaf, jax.core.Tracer) for leaf in jax.tree_util.tree_leaves(point)
    ):
        raise TypeError(
            "Report identity requires a concrete position; use "
            "BatteryIdentifiabilityPlan.evaluate_numerics inside JIT."
        )
    numerics = plan.evaluate_numerics(point)
    position_fingerprint = array_tree_fingerprint(point)["sha256"]
    evidence_fingerprint = array_tree_fingerprint(
        {
            "whitened_jacobian": numerics.whitened_jacobian,
            "fisher_information": numerics.fisher_information,
            "singular_values": numerics.singular_values,
            "numerical_rank": numerics.numerical_rank,
            "condition_number": numerics.condition_number,
            "weak_directions": numerics.weak_directions,
            "weak_direction_mask": numerics.weak_direction_mask,
            "svd_status": numerics.svd_status,
            "successful": numerics.successful,
        }
    )["sha256"]
    report_id = canonical_fingerprint(
        {
            "kind": "battery-identifiability-report",
            "plan_id": plan.plan_id,
            "problem_id": plan.problem_id,
            "calibration_preparation_id": plan.calibration.preparation_id,
            "position_fingerprint": position_fingerprint,
            "evidence_fingerprint": evidence_fingerprint,
        }
    )
    return BatteryIdentifiabilityReport(
        numerics.whitened_jacobian,
        numerics.fisher_information,
        numerics.singular_values,
        numerics.numerical_rank,
        numerics.condition_number,
        numerics.weak_directions,
        numerics.weak_direction_mask,
        numerics.svd_status,
        numerics.successful,
        parameter_ids=plan.parameter_ids,
        plan_id=plan.plan_id,
        problem_id=plan.problem_id,
        calibration_preparation_id=plan.calibration.preparation_id,
        position_fingerprint=position_fingerprint,
        evidence_fingerprint=evidence_fingerprint,
        report_id=report_id,
    )


__all__ = [
    "BatteryIdentifiabilityPlan",
    "BatteryIdentifiabilityReport",
    "evaluate_battery_identifiability",
]
