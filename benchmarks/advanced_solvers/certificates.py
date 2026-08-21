#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import itertools
from collections.abc import Mapping
from typing import Any

import numpy as np

from .problems import (
    BenchmarkProblem,
    ContinuationProblem,
    GeneralEigenProblem,
    NonlinearProblem,
    OptimizationProblem,
    SparseLinearProblem,
)


def independent_certificate(
    problem: BenchmarkProblem,
    solution: Any,
    auxiliary: Mapping[str, Any],
    /,
) -> dict[str, Any]:
    """Evaluate a backend-independent residual or problem certificate in NumPy."""
    if isinstance(problem, SparseLinearProblem):
        return _linear_certificate(problem, solution)
    if isinstance(problem, NonlinearProblem):
        return _nonlinear_certificate(problem, solution)
    if isinstance(problem, GeneralEigenProblem):
        return _eigen_certificate(problem, solution, auxiliary)
    if isinstance(problem, ContinuationProblem):
        return _continuation_certificate(problem, solution, auxiliary)
    if isinstance(problem, OptimizationProblem):
        return _optimization_certificate(problem, solution)
    raise TypeError(f"Unsupported benchmark problem type {type(problem).__name__!r}")


def _linear_certificate(
    problem: SparseLinearProblem,
    solution: Any,
) -> dict[str, Any]:
    matrix = problem.matrix
    value = np.asarray(solution)
    rhs = problem.rhs
    residual = matrix @ value - rhs
    residual_norm = float(np.linalg.norm(residual))
    rhs_norm = float(np.linalg.norm(rhs))
    value_norm = float(np.linalg.norm(value))
    matrix_norm = float(np.linalg.norm(matrix, ord=2))
    relative = residual_norm / _positive_scale(rhs_norm)
    backward = residual_norm / _positive_scale(matrix_norm * value_norm + rhs_norm)
    details: dict[str, Any] = {
        "matrix_norm_2": matrix_norm,
        "rhs_norm_2": rhs_norm,
    }
    if residual.ndim == 2:
        rhs_column_norms = np.linalg.norm(rhs, axis=0)
        relative_columns = np.linalg.norm(residual, axis=0) / np.maximum(
            rhs_column_norms,
            np.finfo(np.float64).tiny,
        )
        details["relative_residual_per_rhs"] = [float(item) for item in relative_columns]
    return _certificate(
        "linear-system",
        residual_norm=residual_norm,
        relative_residual=relative,
        backward_error=backward,
        details=details,
    )


def _nonlinear_certificate(
    problem: NonlinearProblem,
    solution: Any,
) -> dict[str, Any]:
    value = np.asarray(solution, dtype=np.float64)
    if problem.variant == "vi":
        residual = problem.natural_map(value)
        kind = "variational-inequality-natural-map"
        scale = max(1.0, float(np.linalg.norm(value)))
        details = {
            "active_lower_count": int(np.count_nonzero(np.isclose(value, problem.lower))),
            "feasible": bool(
                np.all(value >= problem.lower) and np.all(value <= problem.upper)
            ),
        }
    else:
        residual = problem.residual(value)
        kind = "nonlinear-root"
        jacobian_norm = float(np.linalg.norm(problem.jacobian(value), ord=2))
        scale = max(
            1.0,
            jacobian_norm * float(np.linalg.norm(value))
            + float(np.linalg.norm(problem.target)),
        )
        details = {"jacobian_norm_2": jacobian_norm}
    residual_norm = float(np.linalg.norm(residual))
    reference = max(1.0, float(np.linalg.norm(problem.target)))
    return _certificate(
        kind,
        residual_norm=residual_norm,
        relative_residual=residual_norm / reference,
        backward_error=residual_norm / scale,
        details=details,
    )


def _eigen_certificate(
    problem: GeneralEigenProblem,
    solution: Any,
    auxiliary: Mapping[str, Any],
) -> dict[str, Any]:
    matrix = problem.matrix.astype(np.complex128)
    vectors = np.asarray(solution, dtype=np.complex128)
    matrix_norm = float(np.linalg.norm(matrix, ord=2))
    vector_norm = float(np.linalg.norm(vectors))
    if "schur_form" in auxiliary:
        form = np.asarray(auxiliary["schur_form"], dtype=np.complex128)
        residual = matrix @ vectors - vectors @ form
        eigenvalues = np.linalg.eigvals(form)
        relation_norm = float(np.linalg.norm(form))
        kind = "schur-relation"
        details = {
            "unitarity_error": float(
                np.linalg.norm(vectors.conj().T @ vectors - np.eye(vectors.shape[1]))
            ),
        }
    else:
        eigenvalues = np.asarray(auxiliary["eigenvalues"], dtype=np.complex128)
        residual = matrix @ vectors - vectors * eigenvalues[np.newaxis, :]
        relation_norm = float(np.linalg.norm(eigenvalues))
        kind = "eigenpair-relation"
        details = {}
    reference_spectrum = np.linalg.eigvals(matrix)
    selected_indices = np.argsort(np.abs(reference_spectrum))[-problem.eigenpairs :]
    selected_reference = reference_spectrum[selected_indices]
    returned_count = int(eigenvalues.size)
    count_satisfied = returned_count == problem.eigenpairs
    membership_error = _eigen_membership_error(eigenvalues, selected_reference)
    membership_tolerance = max(
        1e-7,
        100.0
        * np.sqrt(np.finfo(np.float64).eps)
        * max(1.0, float(np.max(np.abs(selected_reference)))),
    )
    membership_satisfied = bool(
        count_satisfied and membership_error <= membership_tolerance
    )
    details.update(
        {
            "requested_eigenpairs": problem.eigenpairs,
            "returned_eigenpairs": returned_count,
            "count_satisfied": count_satisfied,
            "largest_magnitude_membership_error": membership_error,
            "membership_tolerance": membership_tolerance,
            "largest_magnitude_membership_satisfied": membership_satisfied,
        }
    )
    residual_norm = float(np.linalg.norm(residual))
    relative = residual_norm / _positive_scale(matrix_norm * vector_norm)
    backward = residual_norm / _positive_scale(
        matrix_norm * vector_norm + vector_norm * relation_norm
    )
    details["matrix_norm_2"] = matrix_norm
    return _certificate(
        kind,
        residual_norm=residual_norm,
        relative_residual=relative,
        backward_error=backward,
        details=details,
    )


def _continuation_certificate(
    problem: ContinuationProblem,
    solution: Any,
    auxiliary: Mapping[str, Any],
) -> dict[str, Any]:
    states = np.asarray(solution, dtype=np.float64)
    coordinates = np.asarray(auxiliary["coordinates"], dtype=np.float64)
    if states.shape[0] != coordinates.shape[0]:
        raise ValueError(
            "continuation states and coordinates must have equal point counts"
        )
    residuals = np.stack(
        [
            problem.residual(state, coordinate)
            for state, coordinate in zip(states, coordinates, strict=True)
        ]
    )
    residual_norm = float(np.linalg.norm(residuals))
    scale = max(
        1.0,
        float(np.linalg.norm(states * states)) + float(np.linalg.norm(coordinates)),
    )
    flattened_states = np.reshape(states, (states.shape[0], -1))
    point_norms = np.reshape(np.abs(residuals), (states.shape[0], -1)).max(axis=1)
    finite_branch = bool(
        states.shape[0] >= 3
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(coordinates))
    )
    scalar_states = flattened_states[:, 0]
    state_sign_change = bool(np.any(scalar_states[:-1] * scalar_states[1:] <= 0.0))
    coordinate_steps = np.diff(coordinates)
    nonzero_steps = coordinate_steps[coordinate_steps != 0.0]
    tangent_coordinate_sign_change = bool(
        nonzero_steps.size >= 2 and np.any(nonzero_steps[:-1] * nonzero_steps[1:] < 0.0)
    )
    fold_bracket = state_sign_change and tangent_coordinate_sign_change
    branch_successful = bool(np.asarray(auxiliary["branch_successful"]))
    residual_tolerance = float(auxiliary["residual_tolerance"])
    residuals_satisfied = bool(
        point_norms.size > 0 and float(np.max(point_norms)) <= residual_tolerance
    )
    successful_fold_traversal = bool(
        branch_successful and finite_branch and fold_bracket and residuals_satisfied
    )
    return _certificate(
        "continuation-branch-residual",
        residual_norm=residual_norm,
        relative_residual=residual_norm / max(1.0, float(np.linalg.norm(coordinates))),
        backward_error=residual_norm / scale,
        details={
            "branch_points": int(states.shape[0]),
            "branch_successful": branch_successful,
            "finite_branch": finite_branch,
            "maximum_point_residual": float(np.max(point_norms)),
            "residual_tolerance": residual_tolerance,
            "residuals_satisfied": residuals_satisfied,
            "minimum_coordinate": float(np.min(coordinates)),
            "maximum_coordinate": float(np.max(coordinates)),
            "state_sign_change": state_sign_change,
            "tangent_coordinate_sign_change": tangent_coordinate_sign_change,
            "fold_bracket": fold_bracket,
            "successful_fold_traversal": successful_fold_traversal,
        },
    )


def _optimization_certificate(
    problem: OptimizationProblem,
    solution: Any,
) -> dict[str, Any]:
    value = np.asarray(solution, dtype=np.float64)
    if value.shape != problem.initial.shape:
        raise ValueError("optimization result has the wrong parameter shape")
    gradient = problem.gradient(value)
    if problem.variant == "constrained":
        equality = abs(problem.equality(value))
        inequality_violation = max(problem.inequality(value), 0.0)
        equality_gradient = 2.0 * value
        denominator = float(equality_gradient @ equality_gradient)
        multiplier = (
            -float(gradient @ equality_gradient) / denominator
            if denominator > np.finfo(np.float64).tiny
            else 0.0
        )
        dual = gradient + multiplier * equality_gradient
        residual_norm = max(
            equality,
            inequality_violation,
            float(np.linalg.norm(dual, ord=np.inf)),
        )
        details = {
            "objective": problem.objective(value),
            "objective_gap": abs(
                problem.objective(value) - problem.objective(problem.optimum)
            ),
            "distance_to_reference": float(np.linalg.norm(value - problem.optimum)),
            "equality_violation": equality,
            "inequality_violation": inequality_violation,
            "estimated_equality_multiplier": multiplier,
            "dual_stationarity_norm": float(np.linalg.norm(dual, ord=np.inf)),
        }
        scale = 1.0 + float(np.linalg.norm(gradient, ord=np.inf))
        kind = "optimization-kkt"
    elif problem.variant == "proximal":
        stationarity = problem.proximal_stationarity(value)
        residual_norm = float(np.linalg.norm(stationarity, ord=np.inf))
        details = {
            "objective": problem.objective(value),
            "objective_gap": abs(
                problem.objective(value) - problem.objective(problem.optimum)
            ),
            "distance_to_reference": float(np.linalg.norm(value - problem.optimum)),
            "proximal_gradient_mapping_norm": residual_norm,
        }
        scale = 1.0 + float(np.linalg.norm(value, ord=np.inf))
        kind = "optimization-proximal-stationarity"
    else:
        residual_norm = float(np.linalg.norm(gradient, ord=np.inf))
        details = {
            "objective": problem.objective(value),
            "objective_gap": abs(
                problem.objective(value) - problem.objective(problem.optimum)
            ),
            "distance_to_reference": float(np.linalg.norm(value - problem.optimum)),
            "gradient_norm": residual_norm,
        }
        scale = 1.0 + abs(problem.objective(value))
        kind = "optimization-stationarity"
    relative = residual_norm / _positive_scale(scale)
    return _certificate(
        kind,
        residual_norm=residual_norm,
        relative_residual=relative,
        backward_error=relative,
        details=details,
    )


def _eigen_membership_error(
    returned: np.ndarray,
    selected_reference: np.ndarray,
    /,
) -> float:
    returned_values = np.ravel(np.asarray(returned, dtype=np.complex128))
    reference_values = np.ravel(np.asarray(selected_reference, dtype=np.complex128))
    if returned_values.size == 0:
        return float("inf")
    best = float("inf")
    for assignment in itertools.permutations(
        range(reference_values.size),
        returned_values.size,
    ):
        errors = np.abs(returned_values - reference_values[np.asarray(assignment)])
        best = min(best, float(np.max(errors)))
    return best


def _certificate(
    kind: str,
    *,
    residual_norm: float,
    relative_residual: float,
    backward_error: float,
    details: dict[str, Any],
) -> dict[str, Any]:
    values = (residual_norm, relative_residual, backward_error)
    if not all(np.isfinite(value) and value >= 0.0 for value in values):
        raise FloatingPointError("independent certificate produced non-finite evidence")
    return {
        "kind": kind,
        "residual_norm": residual_norm,
        "relative_residual": relative_residual,
        "backward_error": backward_error,
        "independently_computed": True,
        "evaluator": "benchmarks.advanced_solvers.certificates",
        "details": details,
    }


def _positive_scale(value: float) -> float:
    return max(float(value), np.finfo(np.float64).tiny)


__all__ = ["independent_certificate"]
