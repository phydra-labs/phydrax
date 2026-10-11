"""Native whitened-Jacobian rank and physical confounding evidence."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.flatten_util import ravel_pytree
from jax.typing import ArrayLike

from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, FailurePolicy, RankPolicy
from ...linalg.svd import (
    require_exact_svd_rank,
    svd,
    SVDProblem,
    SVDSolvePolicy,
    SVDSolveResult,
)
from ...observation import CoordinateLayout
from ...typing import as_array, Float
from ._calibration import FuelCalibrationParameters, FuelCalibrationPlan
from ._observables import FuelObservationDim
from ._sensitivity import FuelParameterDim


class FuelIdentifiabilityReport(StrictModule):
    """Data-only identifiability of calibration parameters from a whitened Jacobian.

    Rows of ``weak_directions`` are the right singular vectors beyond the
    numerical rank, which the data do not constrain; rows within the rank are
    zero. ``normalized_column_correlation`` exposes confounded parameter pairs.
    """

    __strict_contract__ = True
    whitened_jacobian: Float[FuelObservationDim, FuelParameterDim]
    fisher_information: Float[FuelParameterDim, FuelParameterDim]
    normalized_column_correlation: Float[FuelParameterDim, FuelParameterDim]
    weak_directions: Float[FuelParameterDim, FuelParameterDim]
    decomposition: SVDSolveResult
    parameter_layout: CoordinateLayout
    observation_layout: CoordinateLayout
    condition_number: Array
    finite: Array
    successful: Array


def evaluate_fuel_identifiability(
    whitened_jacobian: ArrayLike,
    parameter_layout: CoordinateLayout,
    observation_layout: CoordinateLayout,
    /,
    *,
    rank_policy: RankPolicy | None = None,
) -> FuelIdentifiabilityReport:
    """Diagnose data-only confounding; priors do not enter the rank.

    Right singular vectors keep the caller's physical or search coordinate
    labels. Zero columns are unobserved parameters, not evidence of
    decorrelation. Decomposition status and triplet residuals are reported
    unchanged.
    """
    matrix = as_array(
        whitened_jacobian,
        Float[FuelObservationDim, FuelParameterDim],
        "whitened_jacobian",
    )
    parameter_count = parameter_layout.size
    if matrix.shape != (observation_layout.size, parameter_count):
        raise ValueError(
            "Jacobian axes must match the observation and parameter layouts."
        )
    finite = jnp.all(jnp.isfinite(matrix))
    matrix = eqx.error_if(
        matrix, ~finite, "Identifiability requires a finite whitened Jacobian."
    )
    # Zero rows give the SVD at least as many rows as parameter columns, so an
    # underdetermined Jacobian still yields a full right singular basis.
    padded = jnp.pad(matrix, ((0, max(0, parameter_count - matrix.shape[0])), (0, 0)))
    policy = RankPolicy() if rank_policy is None else rank_policy
    decomposition = svd(
        SVDProblem(
            DenseLinearOperator(
                padded,
                operator_id=f"{observation_layout.layout_id}:{parameter_layout.layout_id}:whitened-jacobian",
            )
        ),
        policy=SVDSolvePolicy(
            count=parameter_count, rank=policy, failure=FailurePolicy("status")
        ),
    )
    fisher = matrix.T @ matrix
    norms = jnp.sqrt(jnp.diag(fisher))
    denominator = norms[:, None] * norms[None, :]
    correlation = jnp.where(
        denominator > 0, fisher / jnp.where(denominator > 0, denominator, 1.0), jnp.nan
    )
    rank = require_exact_svd_rank(decomposition)
    weak = jnp.arange(parameter_count, dtype=jnp.int32) >= rank
    vectors = jnp.asarray(decomposition.right_vectors)
    condition = jnp.where(
        rank == parameter_count,
        decomposition.singular_values[0] / decomposition.singular_values[-1],
        jnp.inf,
    )
    retained_converged = jnp.all(weak | decomposition.converged)
    return FuelIdentifiabilityReport(
        whitened_jacobian=matrix,
        fisher_information=fisher,
        normalized_column_correlation=correlation,
        weak_directions=jnp.where(weak[:, None], vectors.T, 0.0),
        decomposition=decomposition,
        parameter_layout=parameter_layout,
        observation_layout=observation_layout,
        condition_number=condition,
        finite=finite,
        successful=finite
        & retained_converged
        & jnp.all(jnp.isfinite(decomposition.singular_values))
        & jnp.all(jnp.isfinite(vectors))
        & ((not policy.require_full_rank) | (rank == parameter_count)),
    )


def evaluate_fuel_calibration_identifiability(
    plan: FuelCalibrationPlan,
    position: FuelCalibrationParameters | None = None,
    /,
    *,
    rank_policy: RankPolicy | None = None,
) -> FuelIdentifiabilityReport:
    """Whitened data-only Jacobian of the coupled-rod residual at a search position.

    Forward-mode AD of ``plan.residual`` over every flattened search coordinate;
    labels are the PyTree paths of those coordinates.
    """
    point = plan.parameter_space.initial if position is None else position
    flat, unravel = ravel_pytree(point)
    labels: list[str] = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(point)[0]:
        prefix = jax.tree_util.keystr(path)
        for index in np.ndindex(leaf.shape):
            labels.append(f"{prefix}[{','.join(str(axis) for axis in index)}]")
    parameter_layout = CoordinateLayout(tuple(labels))
    observation_layout = CoordinateLayout(
        tuple(
            f"{rod.case.case_id}:{label}"
            for rod in plan.rods
            for label in rod.observation.layout.labels
        )
    )

    def residual(values: Array) -> Array:
        return plan.residual(unravel(values))

    return evaluate_fuel_identifiability(
        jax.jacfwd(residual)(flat),
        parameter_layout,
        observation_layout,
        rank_policy=rank_policy,
    )
