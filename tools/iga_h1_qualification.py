#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax


jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._io import write_json_atomic


_DEGREE = 2
_SPAN_COUNTS = (2, 4, 8, 16)
_SOLVER_RTOL = 1.0e-10


def _basis_values(grid, points: np.ndarray, degree: int | None = None) -> np.ndarray:
    knots = np.asarray(grid.knots, dtype=float)
    p = grid.degree if degree is None else degree
    x = np.asarray(points, dtype=float).reshape((-1,))
    values = np.zeros((x.size, knots.size - 1), dtype=float)
    for index in range(knots.size - 1):
        values[:, index] = (knots[index] <= x) & (x < knots[index + 1])
    for order in range(1, p + 1):
        next_values = np.zeros((x.size, values.shape[1] - 1), dtype=float)
        for index in range(next_values.shape[1]):
            left_width = knots[index + order] - knots[index]
            right_width = knots[index + order + 1] - knots[index + 1]
            if left_width > 0.0:
                next_values[:, index] += (
                    (x - knots[index]) / left_width * values[:, index]
                )
            if right_width > 0.0:
                next_values[:, index] += (
                    (knots[index + order + 1] - x) / right_width * values[:, index + 1]
                )
        values = next_values
    at_right = x == float(grid.active_interval[1])
    if np.any(at_right):
        values[at_right] = 0.0
        values[at_right, -1] = 1.0
    return values


def _basis_and_derivative(grid, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = _basis_values(grid, points)
    lower = _basis_values(grid, points, grid.degree - 1)
    knots = np.asarray(grid.knots, dtype=float)
    derivative = np.zeros_like(values)
    for index in range(values.shape[1]):
        left_width = knots[index + grid.degree] - knots[index]
        right_width = knots[index + grid.degree + 1] - knots[index + 1]
        if left_width > 0.0:
            derivative[:, index] += grid.degree / left_width * lower[:, index]
        if right_width > 0.0:
            derivative[:, index] -= grid.degree / right_width * lower[:, index + 1]
    return values, derivative


def _quadrature(grid, points_per_axis: int):
    nodes, weights = np.polynomial.legendre.leggauss(points_per_axis)
    points = []
    scaled_weights = []
    breakpoints = np.asarray(grid.breakpoints, dtype=float)
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        midpoint = 0.5 * (lower + upper)
        half_width = 0.5 * (upper - lower)
        points.extend(midpoint + half_width * nodes)
        scaled_weights.extend(half_width * weights)
    xi, eta = np.meshgrid(points, points, indexing="ij")
    wx, wy = np.meshgrid(scaled_weights, scaled_weights, indexing="ij")
    return xi.reshape((-1,)), eta.reshape((-1,)), (wx * wy).reshape((-1,))


def _evaluate(grid, geometry, coefficients, points_per_axis: int):
    xi, eta, parameter_weights = _quadrature(grid, points_per_axis)
    nx, dnx = _basis_and_derivative(grid, xi)
    ny, dny = _basis_and_derivative(grid, eta)
    basis = (nx[:, :, None] * ny[:, None, :]).reshape((xi.size, -1))
    derivative_xi = (dnx[:, :, None] * ny[:, None, :]).reshape((xi.size, -1))
    derivative_eta = (nx[:, :, None] * dny[:, None, :]).reshape((xi.size, -1))
    weights = np.asarray(geometry.weights, dtype=float).reshape((-1,))
    weighted_basis = basis * weights
    denominator = np.sum(weighted_basis, axis=1)
    denominator_xi = np.sum(derivative_xi * weights, axis=1)
    denominator_eta = np.sum(derivative_eta * weights, axis=1)
    rational = weighted_basis / denominator[:, None]
    rational_xi = (
        weights
        * (derivative_xi * denominator[:, None] - basis * denominator_xi[:, None])
        / denominator[:, None] ** 2
    )
    rational_eta = (
        weights
        * (derivative_eta * denominator[:, None] - basis * denominator_eta[:, None])
        / denominator[:, None] ** 2
    )
    control_points = np.asarray(geometry.control_points, dtype=float).reshape((-1, 2))
    physical_points = rational @ control_points
    dx_dxi = rational_xi @ control_points
    dx_deta = rational_eta @ control_points
    determinant = dx_dxi[:, 0] * dx_deta[:, 1] - dx_deta[:, 0] * dx_dxi[:, 1]
    gradient_x = (
        rational_xi * dx_deta[:, 1, None] - rational_eta * dx_dxi[:, 1, None]
    ) / determinant[:, None]
    gradient_y = (
        -rational_xi * dx_deta[:, 0, None] + rational_eta * dx_dxi[:, 0, None]
    ) / determinant[:, None]
    physical_weights = parameter_weights * determinant
    result = {
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "physical_points": physical_points,
        "physical_weights": physical_weights,
        "xi": xi,
        "eta": eta,
    }
    if coefficients is not None:
        coefficients_ = np.asarray(coefficients, dtype=float).reshape((-1,))
        result["field"] = rational @ coefficients_
        result["field_gradient"] = np.stack(
            (gradient_x @ coefficients_, gradient_y @ coefficients_), axis=-1
        )
    return result


def _polynomial_coefficients(grid, values) -> np.ndarray:
    sites = np.asarray(grid.greville_abscissae, dtype=float)
    collocation = _basis_values(grid, sites)
    return np.linalg.solve(collocation, np.asarray(values(sites), dtype=float))


def _geometry(case: str, grid):
    if case == "affine-square":
        sites = np.asarray(grid.greville_abscissae, dtype=float)
        xx, yy = np.meshgrid(sites, sites, indexing="ij")
        return phx.discretization.iga.NURBSGeometryState(
            jnp.asarray(np.stack((xx, yy), axis=-1)),
            jnp.ones(xx.shape),
        )
    radial = _polynomial_coefficients(grid, lambda value: 1.0 + value)
    homogeneous_x = _polynomial_coefficients(grid, lambda value: 1.0 - value**2)
    homogeneous_y = _polynomial_coefficients(grid, lambda value: 2.0 * value)
    homogeneous_weight = _polynomial_coefficients(grid, lambda value: 1.0 + value**2)
    weights = np.broadcast_to(homogeneous_weight[None, :], (radial.size, radial.size))
    homogeneous_points = np.stack(
        (
            radial[:, None] * homogeneous_x[None, :],
            radial[:, None] * homogeneous_y[None, :],
        ),
        axis=-1,
    )
    return phx.discretization.iga.NURBSGeometryState(
        jnp.asarray(homogeneous_points / weights[..., None]),
        jnp.asarray(weights),
    )


def _exact(case: str, points: np.ndarray):
    x = points[:, 0]
    y = points[:, 1]
    if case == "affine-square":
        value = np.sin(np.pi * x) * np.sin(np.pi * y)
        gradient = np.stack(
            (
                np.pi * np.cos(np.pi * x) * np.sin(np.pi * y),
                np.pi * np.sin(np.pi * x) * np.cos(np.pi * y),
            ),
            axis=-1,
        )
        return value, gradient
    radius = np.sqrt(x**2 + y**2)
    sine_angle = 2.0 * x * y / radius**2
    cosine_angle = (x**2 - y**2) / radius**2
    phase = np.pi * (radius - 1.0)
    amplitude = np.sin(phase)
    radial_derivative = np.pi * np.cos(phase) * sine_angle
    angular_derivative = 2.0 * amplitude * cosine_angle
    gradient = np.stack(
        (
            x / radius * radial_derivative - y / radius**2 * angular_derivative,
            y / radius * radial_derivative + x / radius**2 * angular_derivative,
        ),
        axis=-1,
    )
    return amplitude * sine_angle, gradient


def _source(case: str):
    if case == "affine-square":
        return phx.equations.coefficient(
            lambda points, args: (
                2.0
                * jnp.pi**2
                * jnp.sin(jnp.pi * points[..., 0])
                * jnp.sin(jnp.pi * points[..., 1])
            ),
            coefficient_id="iga-affine-square-source",
        )

    def quarter_annulus(points, args):
        x = points[..., 0]
        y = points[..., 1]
        radius = jnp.sqrt(x**2 + y**2)
        sine_angle = 2.0 * x * y / radius**2
        phase = jnp.pi * (radius - 1.0)
        amplitude = jnp.sin(phase)
        return (
            jnp.pi**2 * amplitude
            - jnp.pi * jnp.cos(phase) / radius
            + 4.0 * amplitude / radius**2
        ) * sine_angle

    return phx.equations.coefficient(
        quarter_annulus,
        coefficient_id="iga-quarter-annulus-source",
    )


def _compile(case, grid, geometry, points_per_axis, policy, version):
    plan = phx.discretization.iga.IsogeometricPlan.isoparametric(
        (grid, grid),
        geometry,
        field_name="u",
        axis_names=("xi", "eta"),
        quadrature_policy=phx.discretization.iga.IsogeometricQuadraturePolicy(
            points_per_axis
        ),
        qualification_policy=policy,
    )
    discretization = plan.prepare(numeric_version=version)
    form = phx.equations.FiniteElementForm(
        f"iga-{case}-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction("u", _source(case)),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=discretization.homogeneous_trace_constraint("u"),
        execution_policy=phx.equations.FiniteElementExecutionPolicy(
            realization="matrix_free",
            local_kernel="sum_factorized",
        ),
    )
    return discretization, compiled


def _norm(value) -> jax.Array:
    return jnp.sqrt(jnp.real(jnp.vdot(value, value)))


def _field_errors(case, grid, geometry, coefficients, points_per_axis):
    evaluation = _evaluate(grid, geometry, coefficients, points_per_axis)
    exact_value, exact_gradient = _exact(case, evaluation["physical_points"])
    value_error = evaluation["field"] - exact_value
    gradient_error = evaluation["field_gradient"] - exact_gradient
    weights = evaluation["physical_weights"]
    l2_error = np.sqrt(np.sum(weights * value_error**2))
    h1_error = np.sqrt(np.sum(weights * np.sum(gradient_error**2, axis=-1)))
    return float(l2_error), float(h1_error)


def _private_sparse_stiffness(grid, geometry, points_per_axis):
    evaluation = _evaluate(grid, geometry, None, points_per_axis)
    weights = evaluation["physical_weights"]
    gradient_x = evaluation["gradient_x"]
    gradient_y = evaluation["gradient_y"]
    dense = (gradient_x.T * weights) @ gradient_x + (gradient_y.T * weights) @ gradient_y
    count = grid.coefficient_count
    free = np.asarray(
        [i * count + j for i in range(1, count - 1) for j in range(1, count - 1)],
        dtype=int,
    )
    reduced = dense[np.ix_(free, free)]
    rows, columns = np.nonzero(reduced)
    return rows, columns, reduced[rows, columns], reduced.shape[0]


def _sparse_matvec(rows, columns, data, size, vector):
    result = np.zeros((size,), dtype=float)
    np.add.at(result, rows, data * np.asarray(vector)[columns])
    return result


def _taylor_evidence(compiled, discretization, geometry, state, policy):
    direction = jnp.sin(
        jnp.arange(geometry.control_points.size, dtype=geometry.control_points.dtype)
        + 1.0
    ).reshape(geometry.control_points.shape)
    direction = 1.0e-2 * direction / jnp.maximum(_norm(direction), 1.0)

    def residual(alpha):
        perturbed = phx.discretization.iga.NURBSGeometryState(
            geometry.control_points + alpha * direction,
            geometry.weights,
        )
        runtime = discretization.prepare_runtime(
            perturbed,
            numeric_version="qualification-taylor",
        )
        context = phx.equations.FiniteElementExecutionContext(runtime)
        return compiled.residual(state, context)

    base, tangent = jax.jvp(residual, (jnp.asarray(0.0),), (jnp.asarray(1.0),))
    steps = np.geomspace(1.0e-2, 1.0e-5, policy.taylor_step_count)
    errors = []
    for step in steps:
        remainder = residual(jnp.asarray(step)) - base - step * tangent
        errors.append(float(_norm(remainder)))
    floor = float(128.0 * jnp.finfo(base.dtype).eps * jnp.maximum(_norm(base), 1.0))
    slopes = []
    for index in range(len(steps) - 1):
        if errors[index] > floor and errors[index + 1] > floor:
            slopes.append(
                float(
                    np.log(errors[index] / errors[index + 1])
                    / np.log(steps[index] / steps[index + 1])
                )
            )
    passed = len(slopes) >= policy.taylor_minimum_intervals and all(
        policy.taylor_slope_min <= slope <= policy.taylor_slope_max for slope in slopes
    )
    return {
        "errors": errors,
        "non_roundoff_slopes": slopes,
        "passed": passed,
        "steps": steps.tolist(),
    }


def _level(case, span_count, policy):
    grid = phx.discretization.iga.BSplineGrid.open_uniform(
        _DEGREE,
        span_count,
        interval=(0.0, 1.0),
    )
    geometry = _geometry(case, grid)
    q = _DEGREE + 1
    discretization, compiled = _compile(
        case, grid, geometry, q, policy, f"{case}-n{span_count}-q{q}"
    )
    system, right_hand_side = compiled.linear_system()
    solve_policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    result = phx.linalg.solve(system, right_hand_side, policy=solve_policy)
    full = compiled.expand(result.value)
    _, reference_compiled = _compile(
        case,
        grid,
        geometry,
        q + policy.quadrature_reference_increment,
        policy,
        f"{case}-n{span_count}-q{q + policy.quadrature_reference_increment}",
    )
    reference_system, reference_rhs = reference_compiled.linear_system()
    reference_result = phx.linalg.solve(
        reference_system,
        reference_rhs,
        policy=solve_policy,
    )
    reference_full = reference_compiled.expand(reference_result.value)
    l2_error, h1_error = _field_errors(
        case,
        grid,
        geometry,
        full,
        q + policy.quadrature_reference_increment,
    )
    evaluation = _evaluate(
        grid,
        geometry,
        full - reference_full,
        q + policy.quadrature_reference_increment,
    )
    quadrature_defect = float(
        np.sqrt(
            np.sum(
                evaluation["physical_weights"]
                * (
                    evaluation["field"] ** 2
                    + np.sum(evaluation["field_gradient"] ** 2, axis=-1)
                )
            )
        )
    )
    residual = compiled.residual(result.value)
    normalized_residual = float(
        _norm(residual) / jnp.maximum(_norm(right_hand_side), 1.0)
    )
    operator = compiled.affine_operator()
    size = compiled.state_space.size
    probe = jnp.sin(jnp.arange(size, dtype=full.dtype) + 1.0)
    dual_probe = jnp.cos(jnp.arange(size, dtype=full.dtype) + 0.5)
    image = operator.mv(probe)
    transpose_image = operator.transpose_mv(dual_probe)
    duality_defect = float(
        jnp.abs(jnp.vdot(image, dual_probe) - jnp.vdot(probe, transpose_image))
        / jnp.maximum(_norm(image) * _norm(dual_probe), 1.0)
    )
    rows, columns, data, sparse_size = _private_sparse_stiffness(grid, geometry, q)
    sparse_image = _sparse_matvec(rows, columns, data, sparse_size, probe)
    parity_defect = float(
        np.linalg.norm(np.asarray(image) - sparse_image)
        / max(float(np.linalg.norm(sparse_image)), 1.0)
    )
    evidence = discretization.default_geometry_evidence
    epsilon = float(jnp.finfo(full.dtype).eps)
    weight_tolerance, rank_tolerance, orientation_tolerance = policy.geometry_tolerances(
        full.dtype
    )
    residual_bound = max(
        policy.residual_factor * _SOLVER_RTOL,
        policy.residual_epsilon_factor * epsilon,
    )
    parity_bound = max(
        policy.parity_factor * _SOLVER_RTOL,
        policy.parity_epsilon_factor * epsilon,
    )
    duality_bound = max(
        policy.duality_factor * _SOLVER_RTOL,
        policy.duality_epsilon_factor * epsilon,
    )
    quadrature_bound = policy.quadrature_error_fraction * max(h1_error, residual_bound)
    passed = bool(
        jnp.all(result.successful)
        and jnp.all(reference_result.successful)
        and normalized_residual <= residual_bound
        and parity_defect <= parity_bound
        and duality_defect <= duality_bound
        and quadrature_defect <= quadrature_bound
    )
    return {
        "_compiled": compiled,
        "_discretization": discretization,
        "_geometry": geometry,
        "_grid": grid,
        "_state": result.value,
        "duality_bound": duality_bound,
        "duality_defect": duality_defect,
        "geometry_evidence": {
            "orientation_tolerance": float(orientation_tolerance),
            "rank_tolerance": float(rank_tolerance),
            "weight_tolerance": float(weight_tolerance),
            "coordinate_scale": float(evidence.coordinate_scale),
            "evidence_id": evidence.evidence_id,
            "minimum_orientation_ratio": float(evidence.minimum_orientation_ratio),
            "minimum_rank_ratio": float(evidence.minimum_rank_ratio),
            "minimum_weight_ratio": float(evidence.minimum_weight_ratio),
        },
        "h1_error": h1_error,
        "l2_error": l2_error,
        "normalized_free_residual": normalized_residual,
        "parity_bound": parity_bound,
        "parity_defect": parity_defect,
        "passed": passed,
        "quadrature_bound": quadrature_bound,
        "q_vs_q_plus_2_defect": quadrature_defect,
        "residual_bound": residual_bound,
        "solver_successful": bool(jnp.all(result.successful)),
        "span_count_per_axis": span_count,
    }


def _rate(errors):
    widths = 1.0 / np.asarray(_SPAN_COUNTS, dtype=float)
    return float(np.polyfit(np.log(widths), np.log(np.asarray(errors)), 1)[0])


def _case(case, policy):
    levels = [_level(case, spans, policy) for spans in _SPAN_COUNTS]
    h1_rate = _rate([level["h1_error"] for level in levels])
    l2_rate = _rate([level["l2_error"] for level in levels])
    finest = levels[-1]
    taylor = _taylor_evidence(
        finest["_compiled"],
        finest["_discretization"],
        finest["_geometry"],
        finest["_state"],
        policy,
    )
    evaluation = _evaluate(
        finest["_grid"],
        finest["_geometry"],
        None,
        _DEGREE + 1 + policy.quadrature_reference_increment,
    )
    if case == "affine-square":
        expected_map = np.stack((evaluation["xi"], evaluation["eta"]), axis=-1)
    else:
        radius = 1.0 + evaluation["xi"]
        parameter = evaluation["eta"]
        expected_map = np.stack(
            (
                radius * (1.0 - parameter**2) / (1.0 + parameter**2),
                radius * 2.0 * parameter / (1.0 + parameter**2),
            ),
            axis=-1,
        )
    map_defect = float(np.max(np.abs(evaluation["physical_points"] - expected_map)))
    physical_linear = _evaluate(
        finest["_grid"],
        finest["_geometry"],
        finest["_geometry"].control_points[..., 0],
        _DEGREE + 1 + policy.quadrature_reference_increment,
    )
    reproduction_defect = float(
        np.max(
            np.abs(physical_linear["field"] - physical_linear["physical_points"][:, 0])
        )
    )
    reproduction_bound = max(finest["residual_bound"], finest["duality_bound"])
    for level in levels:
        for private_key in (
            "_compiled",
            "_discretization",
            "_geometry",
            "_grid",
            "_state",
        ):
            del level[private_key]
    h1_bound = _DEGREE - policy.h1_rate_slack
    l2_bound = _DEGREE + 1.0 - policy.l2_rate_slack
    passed = bool(
        len(levels) == policy.refinement_levels
        and all(level["passed"] for level in levels)
        and h1_rate >= h1_bound
        and l2_rate >= l2_bound
        and taylor["passed"]
        and map_defect <= reproduction_bound
        and reproduction_defect <= reproduction_bound
    )
    return {
        "geometry_claim": (
            "affine map; sampled regularity evidence"
            if case == "affine-square"
            else "exact rational quarter-annulus map; sampled regularity evidence"
        ),
        "h1_rate": h1_rate,
        "h1_rate_bound": h1_bound,
        "l2_rate": l2_rate,
        "l2_rate_bound": l2_bound,
        "levels": levels,
        "map_defect": map_defect,
        "name": case,
        "passed": passed,
        "physical_linear_reproduction_bound": reproduction_bound,
        "physical_linear_reproduction_defect": reproduction_defect,
        "taylor": taylor,
    }


def run():
    policy = phx.discretization.iga.IsogeometricH1QualificationPolicy()
    if policy.refinement_levels != len(_SPAN_COUNTS):
        raise ValueError("The frozen IGA qualification requires exactly four levels.")
    cases = [_case(name, policy) for name in ("affine-square", "quarter-annulus")]
    payload = {
        "cases": cases,
        "degree": _DEGREE,
        "kind": "iga-h1-qualification",
        "policy": {
            "duality_epsilon_factor": policy.duality_epsilon_factor,
            "duality_factor": policy.duality_factor,
            "h1_rate_slack": policy.h1_rate_slack,
            "l2_rate_slack": policy.l2_rate_slack,
            "orientation_tolerance": policy.orientation_tolerance,
            "parity_epsilon_factor": policy.parity_epsilon_factor,
            "parity_factor": policy.parity_factor,
            "policy_id": policy.policy_id,
            "quadrature_error_fraction": policy.quadrature_error_fraction,
            "quadrature_reference_increment": policy.quadrature_reference_increment,
            "rank_tolerance": policy.rank_tolerance,
            "refinement_levels": policy.refinement_levels,
            "residual_epsilon_factor": policy.residual_epsilon_factor,
            "residual_factor": policy.residual_factor,
            "taylor_minimum_intervals": policy.taylor_minimum_intervals,
            "taylor_slope_max": policy.taylor_slope_max,
            "taylor_slope_min": policy.taylor_slope_min,
            "taylor_step_count": policy.taylor_step_count,
            "weight_tolerance": policy.weight_tolerance,
        },
        "solver_relative_tolerance": _SOLVER_RTOL,
        "span_counts_per_axis": list(_SPAN_COUNTS),
    }
    payload["passed"] = bool(all(case["passed"] for case in cases))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify the S1 scalar-H1 isogeometric path."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/iga_h1_qualification.json"),
    )
    arguments = parser.parse_args()
    payload = run()
    rendered = json.dumps(payload, allow_nan=False, indent=2, sort_keys=True)
    print(rendered)
    write_json_atomic(arguments.output, payload)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
