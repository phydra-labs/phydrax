#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Independent oracles and observation-derived gates of the native design loop."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, cast

import numpy as np
import pytest

import phydrax as phx
from phydrax.meshing.qualification._design_oracle import (
    design_qualification_cases,
    DESIGN_QUALIFICATION_DEFAULT_CASE,
)
from tools.meshing_qualification._controls import DesignCase, QualificationControls
from tools.meshing_qualification.scenarios import design


# Affine-triangle moments and Duffy quadrature are exact for these polynomial
# integrands; only floating-point summation separates them.
_MOMENT_RTOL = 1e-12
# Central differences of a quartic moment with h=1e-6 on unit-scale vertices.
_CENTRAL_DIFFERENCE_ATOL = 1e-8


def _certified(points: np.ndarray, cells: np.ndarray) -> phx.meshing.CellMeshingResult:
    mesh = phx.discretization.CellMesh.from_triangles(points, cells)
    return phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())


def _p1_space(
    result: phx.meshing.CellMeshingResult,
) -> phx.discretization.FiniteElementDiscretization:
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("triangle", 1)
    )
    return phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()


_FOLDED = (
    np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.5), (0.0, 1.0, 0.25), (1.2, 1.1, -0.3)),
        dtype=np.float64,
    ),
    np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32),
)


def test_design_cell_errors_partition_the_independent_l2_error() -> None:
    result = _certified(*_FOLDED)
    space = _p1_space(result)
    values = space.dof_maps[0].dof_coordinates[:, 1]
    cells = design._design_qualification_cell_errors(result, space, values, 6)
    physical = design._design_qualification_physical(result, space, values, 6)
    assert cells.shape == (2,)
    assert np.all(cells > 0.0)
    assert float(np.sum(cells)) == pytest.approx(
        physical["field_L2_error"] ** 2, rel=_MOMENT_RTOL
    )


def test_design_moment_gradient_is_the_solve_free_qoi_reference() -> None:
    unit = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    total, _ = design._design_qualification_moment_gradient(
        unit, (np.asarray(((0, 1, 2),)),)
    )
    # ∫_T x² over the unit right triangle is 1/12.
    assert total == pytest.approx(1.0 / 12.0, rel=_MOMENT_RTOL)

    points, cells = _FOLDED
    total, gradient = design._design_qualification_moment_gradient(points, (cells,))
    result = _certified(points, cells)
    space = _p1_space(result)
    exact = design._design_qualification_physical(
        result, space, space.dof_maps[0].dof_coordinates[:, 0], 4
    )["exact_integral"]
    assert total == pytest.approx(exact, rel=_MOMENT_RTOL)
    step = 1.0e-6
    for vertex in range(points.shape[0]):
        for axis in range(3):
            upper, lower = points.copy(), points.copy()
            upper[vertex, axis] += step
            lower[vertex, axis] -= step
            central = (
                design._design_qualification_moment_gradient(upper, (cells,))[0]
                - design._design_qualification_moment_gradient(lower, (cells,))[0]
            ) / (2.0 * step)
            assert gradient[vertex, axis] == pytest.approx(
                central, abs=_CENTRAL_DIFFERENCE_ATOL
            )


def test_design_moment_gradient_refuses_degenerate_triangles() -> None:
    collinear = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="nondegenerate"):
        design._design_qualification_moment_gradient(
            collinear, (np.asarray(((0, 1, 2),)),)
        )


def test_design_host_qoi_integrates_p1_fields_exactly() -> None:
    points, cells = _FOLDED
    dofs = (cells,)
    area = sum(
        0.5
        * np.linalg.norm(
            np.cross(points[c[1]] - points[c[0]], points[c[2]] - points[c[0]])
        )
        for c in cells
    )
    ones = np.ones(points.shape[0])
    assert design._design_qualification_host_qoi(
        points, ones, (cells,), dofs
    ) == pytest.approx(area, rel=_MOMENT_RTOL)
    affine = 2.0 * points[:, 0] - points[:, 2]
    expected = sum(
        0.5
        * np.linalg.norm(
            np.cross(points[c[1]] - points[c[0]], points[c[2]] - points[c[0]])
        )
        * np.mean(affine[c])
        for c in cells
    )
    assert design._design_qualification_host_qoi(
        points, affine, (cells,), dofs
    ) == pytest.approx(expected, rel=_MOMENT_RTOL)


def test_design_learned_features_read_the_declared_load_hessian() -> None:
    # The first face is normal to x, where the x² load has no in-face curvature;
    # the second shares its edge and leaves the x = 0.5 plane.
    points = np.asarray(
        ((0.5, 0.0, 0.0), (0.5, 1.0, 0.0), (0.5, 0.0, 1.0), (1.5, 1.0, 0.0)),
        dtype=np.float64,
    )
    cells = np.asarray(((0, 1, 2), (2, 1, 3)), dtype=np.int32)
    features = design._design_qualification_cell_features(_certified(points, cells))
    assert features.shape == (2, len(design._LEARNED_FEATURE_IDS))
    assert features[0] == pytest.approx((0.0, 0.0, 0.5 * np.sqrt(2.0) ** 4))
    # Edge x-increments 0, 1, -1 give s = Σ eᵀ(2e₀e₀ᵀ)e/2 = 2; the area is √2/2
    # and the longest edge is √3.
    area, diameter = np.sqrt(2.0) / 2.0, np.sqrt(3.0)
    assert features[1] == pytest.approx(
        (area * 4.0, area * 2.0 * diameter**2, area * diameter**4)
    )


def test_design_rank_correlation_is_monotone_invariant() -> None:
    observed = np.asarray((0.1, 0.4, 0.2, 0.9))
    assert design._design_qualification_rank_correlation(
        observed**3, observed
    ) == pytest.approx(1.0)
    assert design._design_qualification_rank_correlation(
        -observed, observed
    ) == pytest.approx(-1.0)
    with pytest.raises(ValueError, match="equally long"):
        design._design_qualification_rank_correlation(observed, observed[:2])


def test_design_rank_correlation_averages_tied_ranks() -> None:
    # Ordinal tie-breaking would rank (1, 1, 2, 2) as (0, 1, 2, 3) and report
    # 0.6; Spearman shares tied ranks, giving the Pearson value of
    # (0.5, 0.5, 2.5, 2.5) against (1, 0, 3, 2), which is 2/sqrt(5).
    scores = np.asarray((1.0, 1.0, 2.0, 2.0))
    errors = np.asarray((0.2, 0.1, 0.4, 0.3))
    assert design._design_qualification_rank_correlation(
        scores, errors
    ) == pytest.approx(2.0 / np.sqrt(5.0))
    # A constant marker carries no ordering; ordinal ranks would report 1.
    assert np.isnan(
        design._design_qualification_rank_correlation(
            np.zeros(4), np.asarray((0.1, 0.2, 0.3, 0.4))
        )
    )


def _inscribed_sphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Return a unit-sphere inscribed polyhedron by projected octahedron splits."""
    points = [
        np.asarray(vertex, dtype=np.float64)
        for vertex in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    ]
    faces = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]  # fmt: skip
    for _ in range(level):
        midpoints: dict[tuple[int, int], int] = {}

        def midpoint(first: int, second: int) -> int:
            key = (min(first, second), max(first, second))
            if key not in midpoints:
                middle = points[first] + points[second]
                points.append(middle / np.linalg.norm(middle))
                midpoints[key] = len(points) - 1
            return midpoints[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        faces = refined
    return np.asarray(points), np.asarray(faces, dtype=np.int64)


_UNIT_SPHERE = _inscribed_sphere(3)
_RADIUS = 0.75
# Nearest lattice sign events of the radius-0.75 sphere on the 9-point
# [-1.4, 1.4]³ lattice: the node (0.7, 0, 0) inside, (0.7, 0.35, 0) outside.
_EVENTS = {"lower": 0.7 - _RADIUS, "upper": float(np.hypot(0.7, 0.35)) - _RADIUS}


def _moment(radius: float) -> float:
    points, cells = _UNIT_SPHERE
    return design._design_qualification_moment_gradient(radius * points, (cells,))[0]


def _moment_derivative(radius: float) -> float:
    # Every inscribed vertex moves along its unit direction: dP/dR = P/R.
    points, cells = _UNIT_SPHERE
    _, gradient = design._design_qualification_moment_gradient(
        radius * points, (cells,)
    )
    return float(np.sum(gradient * points))


def test_design_taylor_slope_is_a_log_log_fit_of_real_remainders() -> None:
    gradient = _moment_derivative(_RADIUS)
    base = _moment(_RADIUS)
    offsets = [_EVENTS["lower"] * fraction for fraction in design._TAYLOR_FRACTIONS]

    def remainders(slope: float) -> list[float]:
        return [
            abs(_moment(_RADIUS + offset) - base - offset * slope)
            for offset in offsets
        ]

    fitted, pairwise = design._design_qualification_taylor_slopes(
        design._TAYLOR_FRACTIONS, remainders(gradient)
    )
    assert abs(fitted - 2.0) <= design._TAYLOR_SLOPE_DEVIATION_BOUND
    assert max(abs(slope - 2.0) for slope in pairwise) <= (
        design._TAYLOR_SLOPE_DEVIATION_BOUND
    )
    # A wrong gradient leaves a first-order remainder at these steps.
    wrong, _ = design._design_qualification_taylor_slopes(
        design._TAYLOR_FRACTIONS, remainders(1.1 * gradient)
    )
    assert abs(wrong - 2.0) > design._TAYLOR_SLOPE_DEVIATION_BOUND
    with pytest.raises(ValueError, match="finite positive"):
        design._design_qualification_taylor_slopes(design._TAYLOR_FRACTIONS, [1.0, 0.0, 1.0])


def test_design_case_table_is_canonical_and_fresh() -> None:
    cases = design_qualification_cases()
    assert set(cases) == {"routine-sphere", "larger-sphere", "combined-design-target"}
    assert DESIGN_QUALIFICATION_DEFAULT_CASE == "routine-sphere"
    cases["routine-sphere"]["radius"] = 2.0
    assert design_qualification_cases()["routine-sphere"]["radius"] == 0.75
    for resolution in (2, 4, 5):
        counts = {
            name: design._design_qualification_lattice_count(
                QualificationControls(resolution=resolution),
                design._design_qualification_case(
                    QualificationControls(design_case=cast(DesignCase, name))
                ),
            )
            for name in cases
        }
        assert counts == {
            "routine-sphere": 2 * resolution + 1,
            "larger-sphere": 4 * resolution + 1,
            "combined-design-target": 2 * resolution + 1,
        }
    assert design_qualification_cases()["routine-sphere"]["design_target"] is None
    assert design_qualification_cases()["combined-design-target"]["design_target"]
    assert QualificationControls().design_case == DESIGN_QUALIFICATION_DEFAULT_CASE
    with pytest.raises(ValueError, match="registered design case"):
        design._design_qualification_case(
            QualificationControls(design_case=cast(DesignCase, "unknown"))
        )


def _derivative_record(radius: float, epoch: str) -> dict[str, Any]:
    """Return derivative observations measured on the inscribed sphere moment."""
    points, cells = _UNIT_SPHERE
    gradient = _moment_derivative(radius)
    base = _moment(radius)
    step = design._DERIVATIVE_STEP * design._DERIVATIVE_DIRECTION
    finite_difference = (_moment(radius + step) - _moment(radius - step)) / (2.0 * step)
    _, vertex_gradient = design._design_qualification_moment_gradient(
        radius * points, (cells,)
    )
    analytic = float(
        np.sum(vertex_gradient * ((radius + step) * points - (radius - step) * points))
        / (2.0 * step)
    )
    distance = -_EVENTS["lower"]
    remainders = [
        abs(_moment(radius - distance * f) - base + distance * f * gradient)
        for f in design._TAYLOR_FRACTIONS
    ]
    checks = {
        name: {"observed": 1.0e-9, "bound": bound, "passed": True}
        for name, bound in design._DESIGN_DERIVATIVE_BOUNDS.items()
    }
    return {
        "epoch_id": epoch,
        "parameter_events": dict(_EVENTS),
        "adjoint": {
            "value": base,
            "host_value": base,
            "solve_free_moment": base,
            "adjoint_gradient": gradient,
            "tangent_gradient": gradient,
            "finite_difference_gradient": finite_difference,
            "analytic_reference_gradient": analytic,
            "taylor": {
                "direction": -1.0,
                "event_distance": distance,
                "fractions": list(design._TAYLOR_FRACTIONS),
                "remainders": remainders,
                "roundoff_bound": float(np.finfo(np.float64).eps)
                * cells.shape[0]
                * base,
            },
        },
        "checks": checks,
        "qualifications": {
            stage: {"valid": True} for stage in ("geometry", "pde", "transfer")
        },
        "binding_issues": [],
        "event_nondifferentiability": {
            "lattice_crossing": {
                "before": {
                    "status": 0,
                    "margins": [["lattice-sign", 3.0e-5], ["positive-face-area", 1e-9]],
                },
                "after": {
                    "status": int(design.ImplicitSurfaceStatus.SIGN_PATTERN_CHANGED),
                    "margins": [["lattice-sign", -3.0e-5], ["positive-face-area", 1e-9]],
                },
                "invalidation": ["event-boundary", "stopped-geometry-realization-event"],
            },
            "topology_event": {
                "invalidation": ["epoch-or-parameter-changed", "stopped-topology-event"],
                "transition_is_derivative": False,
                "transition_valid": True,
            },
        },
    }


def _design_step(source: str, designed: str) -> dict[str, Any]:
    """Return the combined design equation solved on the inscribed sphere moment."""
    target = 4.0 * np.pi * _RADIUS**4 / 3.0
    bracket = [
        _RADIUS + design._DESIGN_BRACKET_FRACTION * _EVENTS["lower"],
        _RADIUS + design._DESIGN_BRACKET_FRACTION * _EVENTS["upper"],
    ]
    # The fixed inscribed topology scales homogeneously: Q(R) = Q(1) R⁴.
    root = (target / _moment(1.0)) ** 0.25
    return {
        "status": "SUCCESS",
        "initial_radius": _RADIUS,
        "event_free_bracket": bracket,
        "bracket_endpoints": [
            {"radius": value, "solve_free_residual": _moment(value) - target}
            for value in bracket
        ],
        "root_radius": root,
        "independent_solve_free_QoI": _moment(root),
        "published": {"result_id": designed},
        "initial_epoch": {
            "result_id": source,
            "derivatives": {"parameter_events": dict(_EVENTS)},
        },
    }


def _controls(*, combined: bool = False) -> QualificationControls:
    return QualificationControls(
        resolution=4,
        design_case="combined-design-target" if combined else "routine-sphere",
    )


def _complete_record(*, combined: bool = False) -> dict[str, Any]:
    source, designed = "source-epoch", "designed-epoch"
    epoch_source = designed if combined else source
    learned = {
        "proposal": "learned",
        "status": "completed",
        "candidate_id": "learned-route",
        "condition": 17.0,
        "raw_scores": [0.3, 0.1, 0.7, 0.2],
        "trust_bounds": {
            "training_relative_residual": {"observed": 0.16, "bound": 0.5, "passed": True},
            "held_out_observed_error_rank_correlation": {
                "observed": 0.8,
                "bound": 0.5,
                "passed": True,
            },
        },
        "trusted_proposal": {"safety_audit_passed": True, "compliance_issues": []},
        "adversarial_refusals": [{"expected_refusal": True}] * 2,
        "training": {
            "training_result_id": "uniform-target",
            "application_source_result_id": epoch_source,
            "fit_rank": len(design._LEARNED_FEATURE_IDS),
            "training_samples": 16,
            "application_samples": 4,
            "training_relative_residual": 0.16,
        },
        "learned_acceptance": {
            "receipt_published": True,
            "source_accepted_state_id": "accepted-0",
        },
        "physical_reanalysis_rollback": {
            "committed": False,
            "source_state_identically_retained": True,
            "source_mesh_identically_retained": True,
            "receipt_present": True,
            "receipt_published": False,
            "actual_staged_solution_defect": 0.4,
            "accepted_state_id": "accepted-0",
        },
    }
    selected = {
        "proposal": "analytic",
        "status": "completed",
        "candidate_id": "analytic-route",
        "condition": 27.0,
        "native_event": {"converged": True, "transition_present": True},
        "feasibility_issues": [],
        "state_entries": ["u-entry", "density-entry", "history-entry"],
    }
    uniform = {
        "proposal": "uniform",
        "status": "completed",
        "candidate_id": "uniform-route",
        "condition": 18.0,
    }
    root = (4.0 * np.pi * _RADIUS**4 / 3.0 / _moment(1.0)) ** 0.25
    record: dict[str, Any] = {
        "status": "passed",
        "case": design._design_qualification_case(_controls(combined=combined)),
        "case_observation": {"lattice_points_per_axis": [9, 9, 9]},
        "epochs": [
            {
                "epoch": 0,
                "source": {"result_id": epoch_source},
                "baseline": {"field_L2_error": 0.0215},
                "trials": [uniform, selected, learned],
                "selected_candidate_id": "analytic-route",
                "receipt_published": True,
                "source_accepted_state_id": "accepted-0",
                "independent_reanalysis": {
                    "issues": [],
                    "estimator_id": "independent-host-Duffy-10",
                    "selected_estimator_id": "independent-host-Duffy-8",
                },
                "continued_physical": {"field_L2_error": 0.0144},
            }
        ],
        "fixed_epoch_derivatives": _derivative_record(
            root if combined else _RADIUS, epoch_source
        ),
        "physical_error": 0.0144,
        "end_to_end_seconds": 65.0,
        "peak_process_resident_bytes": 1,
        "memory_budget_status": "within",
        "baseline_operator_condition": 19.0,
        "failed_positive_attempts": [],
    }
    if combined:
        record["design_step"] = _design_step(source, designed)
    return record


def _set(path: tuple[str | int, ...], value: object) -> Callable[[dict[str, Any]], None]:
    def mutate(record: dict[str, Any]) -> None:
        target: Any = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return mutate


def _scale(path: tuple[str | int, ...], factor: float) -> Callable[[dict[str, Any]], None]:
    def mutate(record: dict[str, Any]) -> None:
        target: Any = record
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = target[path[-1]] * factor

    return mutate


def _first_order_remainders(record: dict[str, Any]) -> None:
    # Remainders of an adjoint that is wrong by 10%: first order in the step.
    derivatives = record["fixed_epoch_derivatives"]
    adjoint = derivatives["adjoint"]
    distance = adjoint["taylor"]["event_distance"]
    adjoint["taylor"]["remainders"] = [
        abs(0.1 * adjoint["adjoint_gradient"] * distance * fraction)
        for fraction in design._TAYLOR_FRACTIONS
    ]


_EPOCH = ("epochs", 0)
_LEARNED = (*_EPOCH, "trials", 2)
_SELECTED = (*_EPOCH, "trials", 1)
_DERIVATIVES = ("fixed_epoch_derivatives",)
_ADJOINT = (*_DERIVATIVES, "adjoint")
_CROSSING = (*_DERIVATIVES, "event_nondifferentiability", "lattice_crossing")
_TOPOLOGY = (*_DERIVATIVES, "event_nondifferentiability", "topology_event")
_MUTATIONS = (
    pytest.param(
        "fixed_epoch_derivatives_qualified",
        _scale((*_ADJOINT, "analytic_reference_gradient"), 1.0 + 1.0e-5),
        id="analytic-reference-gradient-off-despite-recorded-pass",
    ),
    pytest.param(
        "fixed_epoch_derivatives_qualified",
        _scale((*_ADJOINT, "finite_difference_gradient"), 1.0 + 1.0e-5),
        id="finite-difference-gradient-off",
    ),
    pytest.param(
        "fixed_epoch_derivatives_qualified",
        _scale((*_ADJOINT, "host_value"), 1.0 + 1.0e-6),
        id="solved-qoi-not-the-solve-free-moment",
    ),
    pytest.param(
        "fixed_epoch_derivatives_qualified",
        _first_order_remainders,
        id="taylor-remainders-first-order",
    ),
    pytest.param(
        "fixed_epoch_derivatives_qualified",
        _set((*_DERIVATIVES, "parameter_events"), {"lower": -1.0e-5, "upper": 1.0}),
        id="finite-difference-step-outside-event-margin",
    ),
    pytest.param(
        "fixed_epoch_derivatives_qualified",
        _set((*_DERIVATIVES, "checks", "pde_finite_difference", "observed"), 3.0e-6),
        id="pde-finite-difference-observation-over-bound",
    ),
    pytest.param(
        "fixed_epoch_derivatives_qualified",
        _set((*_DERIVATIVES, "qualifications", "pde", "valid"), False),
        id="pde-qualification-invalid",
    ),
    pytest.param(
        "event_nondifferentiability_observed",
        _set((*_CROSSING, "after", "status"), 0),
        id="owner-did-not-refuse-crossing",
    ),
    pytest.param(
        "event_nondifferentiability_observed",
        _set((*_CROSSING, "after", "margins"), [["lattice-sign", 1.0e-5]]),
        id="after-probe-did-not-cross-lattice-sign",
    ),
    pytest.param(
        "event_nondifferentiability_observed",
        _set((*_CROSSING, "invalidation"), ["event-boundary"]),
        id="crossing-not-invalidated-as-geometry-event",
    ),
    pytest.param(
        "event_nondifferentiability_observed",
        _set((*_TOPOLOGY, "invalidation"), ["epoch-or-parameter-changed"]),
        id="topology-event-not-invalidated",
    ),
    pytest.param(
        "trusted_learned_proposal_admitted",
        _set((*_LEARNED, "trust_bounds", "training_relative_residual", "observed"), 0.6),
        id="training-residual-over-bound-despite-recorded-pass",
    ),
    pytest.param(
        "trusted_learned_proposal_admitted",
        _set(
            (*_LEARNED, "trust_bounds", "held_out_observed_error_rank_correlation", "observed"),
            0.4,
        ),
        id="held-out-rank-correlation-under-bound",
    ),
    pytest.param(
        "trusted_learned_proposal_admitted",
        _set((*_LEARNED, "raw_scores"), [0.3, float("nan"), 0.7, 0.2]),
        id="nonfinite-learned-score",
    ),
    pytest.param(
        "trusted_learned_proposal_admitted",
        _set((*_LEARNED, "training", "fit_rank"), 2),
        id="rank-deficient-feature-fit",
    ),
    pytest.param(
        "trusted_learned_proposal_admitted",
        _set((*_LEARNED, "training", "training_result_id"), "source-epoch"),
        id="trained-on-application-mesh",
    ),
    pytest.param(
        "trusted_learned_proposal_admitted",
        _set((*_LEARNED, "adversarial_refusals"), [{"expected_refusal": True}]),
        id="adversarial-refusal-missing",
    ),
    pytest.param(
        "native_events_accepted",
        _set((*_SELECTED, "native_event", "converged"), False),
        id="native-event-not-converged",
    ),
    pytest.param(
        "all_state_transfer_published",
        _set((*_SELECTED, "feasibility_issues"), ["state-disposition-incomplete"]),
        id="state-disposition-incomplete",
    ),
    pytest.param(
        "independent_reanalysis_accepted",
        _set((*_EPOCH, "independent_reanalysis", "issues"), ["physical-reanalysis-rejected"]),
        id="reanalysis-rejected",
    ),
    pytest.param(
        "independent_reanalysis_accepted",
        _set(
            (*_EPOCH, "independent_reanalysis", "estimator_id"),
            "independent-host-Duffy-8",
        ),
        id="reanalysis-reuses-selection-estimator",
    ),
    pytest.param(
        "failed_reanalysis_rollback_reused",
        _set((*_LEARNED, "physical_reanalysis_rollback", "committed"), True),
        id="failed-reanalysis-committed",
    ),
    pytest.param(
        "failed_reanalysis_rollback_reused",
        _set((*_LEARNED, "physical_reanalysis_rollback", "actual_staged_solution_defect"), 0.0),
        id="rollback-without-staged-defect",
    ),
    pytest.param(
        "failed_reanalysis_rollback_reused",
        _set((*_EPOCH, "source_accepted_state_id"), "accepted-other"),
        id="rollback-state-not-reused",
    ),
    pytest.param(
        "physical_objective_improved_to_target",
        _set((*_EPOCH, "continued_physical", "field_L2_error"), 0.03),
        id="physical-error-not-improved",
    ),
    pytest.param(
        "resource_budgets_within",
        _set(("peak_process_resident_bytes",), None),
        id="memory-peak-unassessed",
    ),
    pytest.param(
        "resource_budgets_within",
        _set(("peak_process_resident_bytes",), 9 * 1024**3),
        id="memory-peak-over-budget",
    ),
    pytest.param(
        "resource_budgets_within",
        _set((*_SELECTED, "condition"), 2.0e12),
        id="trial-condition-over-budget",
    ),
    pytest.param(
        "case_contract_observed",
        _set(("case_observation", "lattice_points_per_axis"), [17, 17, 17]),
        id="lattice-not-the-case-lattice",
    ),
    pytest.param(
        "case_contract_observed",
        _set(("case", "lattice_factor"), 4),
        id="recorded-case-not-canonical",
    ),
)


@pytest.mark.parametrize(("gate", "mutate"), _MUTATIONS)
def test_design_completion_gates_follow_each_observation(
    gate: str, mutate: Callable[[dict[str, Any]], None]
) -> None:
    args = _controls()
    complete = design._design_qualification_completion_gates(_complete_record(), args)
    assert all(complete.values()), complete
    record = deepcopy(_complete_record())
    mutate(record)
    gates = design._design_qualification_completion_gates(record, args)
    assert {name for name, value in gates.items() if not value} == {gate}


def _straddle_lost(record: dict[str, Any]) -> None:
    endpoint = record["design_step"]["bracket_endpoints"][0]
    endpoint["solve_free_residual"] = abs(endpoint["solve_free_residual"])


@pytest.mark.parametrize(
    "mutate",
    (
        pytest.param(
            _set(("design_step", "status"), "RESIDUAL_STAGNATION"), id="root-not-converged"
        ),
        pytest.param(
            _scale(("design_step", "root_radius"), 1.05), id="root-outside-bracket"
        ),
        pytest.param(
            _scale(("design_step", "independent_solve_free_QoI"), 1.0 + 1.0e-6),
            id="root-misses-authored-sphere-moment",
        ),
        pytest.param(_straddle_lost, id="bracket-does-not-straddle-target"),
        pytest.param(
            _set(("design_step", "event_free_bracket", 1), _RADIUS + _EVENTS["upper"]),
            id="bracket-reaches-lattice-event",
        ),
        pytest.param(
            _set(("design_step", "initial_radius"), 0.8), id="target-not-authored-radius"
        ),
        pytest.param(
            _set(("fixed_epoch_derivatives", "epoch_id"), "source-epoch"),
            id="derivatives-not-requalified-at-design",
        ),
        pytest.param(
            _set(("epochs", 0, "source", "result_id"), "source-epoch"),
            id="epochs-not-on-designed-source",
        ),
    ),
)
def test_design_combined_case_requires_its_observed_design_step(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    args = _controls(combined=True)
    complete = design._design_qualification_completion_gates(
        _complete_record(combined=True), args
    )
    assert all(complete.values()), complete
    record = _complete_record(combined=True)
    mutate(record)
    gates = design._design_qualification_completion_gates(record, args)
    assert {name for name, value in gates.items() if not value} == {
        "case_contract_observed"
    }


def test_design_inscribed_moment_poses_a_bracketed_design_equation() -> None:
    # The represented polyhedron's moment lies below the authored sphere's;
    # the target is reachable only by a bracketed radius increase.
    step = _design_step("source-epoch", "designed-epoch")
    target = 4.0 * np.pi * _RADIUS**4 / 3.0
    assert _moment(_RADIUS) < target
    lower, upper = step["event_free_bracket"]
    assert lower < _RADIUS < step["root_radius"] < upper
    assert _RADIUS + _EVENTS["lower"] < lower and upper < _RADIUS + _EVENTS["upper"]
    assert abs(_moment(step["root_radius"]) - target) <= (
        design._DESIGN_INDEPENDENT_QOI_BOUND
    )
    assert _moment_derivative(step["root_radius"]) > 0.0


def test_design_completion_requires_passed_status_and_every_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(design, "peak_resident_bytes", lambda: 1)
    args = _controls()
    for status, broken, expected in (
        ("passed", False, "complete"),
        ("passed", True, "incomplete"),
        ("failed", False, "incomplete"),
    ):
        record = _complete_record()
        record["status"] = status
        record["active_phase"] = "continued_solve"
        if broken:
            record["epochs"][0]["receipt_published"] = False
        design._design_qualification_completion(args, [], record, perf_counter())
        assert record["mandatory_workflow_completion"] == expected
        assert record["case_contract_completion"] == expected
        assert all(record["completion_gates"].values()) is not broken
        json.dumps(record)


def test_design_cold_continuation_binds_the_exact_uninterrupted_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from types import SimpleNamespace

    import jax.numpy as jnp

    from tools import meshing_design_source as source

    accepted = jnp.asarray((0.25, -1.0, 3.0), dtype=jnp.float64)
    continued = accepted + jnp.asarray((7.9e-8, 0.0, -1.0e-9), dtype=jnp.float64)
    identity = {"accepted_state_id": "a", "field_fingerprint": {"sha256": "f"}}
    monkeypatch.setattr(source, "_identity", lambda _: identity)
    state: Any = SimpleNamespace(accepted_state=SimpleNamespace(fields=(accepted,)))
    record = {
        "status": "published",
        "content_id": "c",
        "identity": identity,
        "uninterrupted_continuation": {
            "fingerprint": phx.array_tree_fingerprint(continued),
            "values": np.asarray(continued).tolist(),
            "change_from_accepted": float(jnp.max(jnp.abs(continued - accepted))),
        },
    }
    reference = json.loads(json.dumps(record))
    # Exact JSON round trip of the warm continued solve, not the accepted seed.
    restored = source._uninterrupted_reference(reference, state, "c")
    assert np.array_equal(np.asarray(restored), np.asarray(continued))
    with pytest.raises(RuntimeError, match="published record"):
        source._uninterrupted_reference(reference, state, "other")
    changed = deepcopy(reference)
    changed["identity"]["field_fingerprint"]["sha256"] = "g"
    with pytest.raises(RuntimeError, match="identity"):
        source._uninterrupted_reference(changed, state, "c")
    tampered = deepcopy(reference)
    tampered["uninterrupted_continuation"]["values"] = np.asarray(accepted).tolist()
    with pytest.raises(RuntimeError, match="fingerprint"):
        source._uninterrupted_reference(tampered, state, "c")

    def corrupt(values: Any) -> dict[str, Any]:
        # A self-consistent forged record: its fingerprint matches its values.
        forged = deepcopy(reference)
        forged["uninterrupted_continuation"]["values"] = values
        forged["uninterrupted_continuation"]["fingerprint"] = json.loads(
            json.dumps(
                phx.array_tree_fingerprint(
                    jnp.asarray(np.asarray(values, dtype=np.float64))
                ),
                sort_keys=True,
            )
        )
        return forged

    nonfinite = np.asarray(continued).tolist()
    nonfinite[0] = float("nan")
    for forged in (corrupt(nonfinite), corrupt(np.asarray(continued)[:2].tolist())):
        with pytest.raises(RuntimeError, match="finite with the accepted-field shape"):
            source._uninterrupted_reference(forged, state, "c")
    malformed = deepcopy(reference)
    malformed["uninterrupted_continuation"]["values"] = ["u", 0.0, 1.0]
    with pytest.raises(RuntimeError, match="numeric array"):
        source._uninterrupted_reference(malformed, state, "c")
    for field in ("fingerprint", "values", "change_from_accepted"):
        missing = deepcopy(reference)
        del missing["uninterrupted_continuation"][field]
        with pytest.raises(RuntimeError, match="must be a mapping with"):
            source._uninterrupted_reference(missing, state, "c")
    misreported = deepcopy(reference)
    misreported["uninterrupted_continuation"]["change_from_accepted"] = 0.0
    with pytest.raises(RuntimeError, match="change_from_accepted"):
        source._uninterrupted_reference(misreported, state, "c")
    for partial in (None, [], "published", 1, {}):
        with pytest.raises(RuntimeError, match="published"):
            source._uninterrupted_reference(partial, state, "c")
    for field in ("status", "content_id", "identity", "uninterrupted_continuation"):
        missing_root = deepcopy(reference)
        del missing_root[field]
        with pytest.raises(RuntimeError, match="published record|identity|must be a mapping"):
            source._uninterrupted_reference(missing_root, state, "c")
    for partial in (None, [], "values", 0, {}):
        missing_continuation = deepcopy(reference)
        missing_continuation["uninterrupted_continuation"] = partial
        with pytest.raises(RuntimeError, match="must be a mapping with"):
            source._uninterrupted_reference(missing_continuation, state, "c")


def test_design_affine_carrier_proof_is_reused_only_in_its_bound_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.discretization import _cell_geometry_transfer as transfer
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        owned_coordinate_preparation,
    )

    mesh = phx.discretization.CellMesh.from_triangles(*_FOLDED)
    geometry = CellGeometrySpec.affine(mesh)
    prove = transfer._affine_source_carrier
    calls = 0

    def counted(*args: Any, **kwargs: Any) -> bool:
        nonlocal calls
        calls += 1
        return prove(*args, **kwargs)

    monkeypatch.setattr(transfer, "_affine_source_carrier", counted)
    for _ in range(2):
        previous_calls = calls
        with owned_coordinate_preparation(100000, 10000000) as budget:
            assert transfer.is_affine_cell_geometry(mesh, geometry, budget=budget)
            charged = budget.work_units
            assert charged > 0
            assert transfer.is_affine_cell_geometry(mesh, geometry, budget=budget)
            assert budget.work_units == charged
            assert calls == previous_calls + 1
            altered = phx.discretization.CellMesh.from_triangles(
                _FOLDED[0] + np.asarray((0.125, 0.0, 0.0)), _FOLDED[1]
            )
            assert not transfer.is_affine_cell_geometry(altered, geometry, budget=budget)
            assert calls == previous_calls + 2


@pytest.mark.parametrize(
    "reference",
    (
        None, [], {}, {"design_case": []}, {"design_case": "unknown"},
        {"design_case": "routine-sphere"},
        {"design_case": "routine-sphere", "source_clock_seconds": 121.0},
        {"design_case": "routine-sphere", "source_clock_seconds": 120.0},
        {"design_case": "routine-sphere", "source_clock_seconds": 120.0,
         "original_campaign": {"status": "passed"}},
    ),
)
def test_design_consumer_refuses_partial_producer_before_archive(
    tmp_path: Path, reference: Any
) -> None:
    from tools import meshing_design_source as source

    record = tmp_path / "produce.json"
    record.write_text(json.dumps(reference))
    with pytest.raises(RuntimeError, match="consume.reference"):
        source.consume(tmp_path / "absent-archive", "absent-content", record)


def test_design_source_cases_keep_registered_science_and_authored_limits() -> None:
    from tools import meshing_design_source as source

    for name in design_qualification_cases():
        controls = source.original_controls(cast(DesignCase, name))
        assert controls.design_case == name
        assert (
            controls.resolution, controls.capacity, controls.repeats,
            controls.target_error, controls.adaptation_rounds,
            controls.maximum_memory_bytes, controls.maximum_condition,
        ) == (4, 20000, 3, 0.05, 1, 8 * 1024**3, 1.0e12)
        assert (controls.maximum_work_units, controls.timeout) == (
            (320_000, 120.0) if name == "routine-sphere" else (480_000, 180.0)
        )


@pytest.mark.parametrize(
    ("design_case", "work", "wall"),
    (("routine-sphere", 320_000, 120.0), ("larger-sphere", 480_000, 180.0),
     ("combined-design-target", 480_000, 180.0)),
)
@pytest.mark.parametrize(
    "expired_phase", (None, "producer", "cold", "killed-producer", "killed-cold")
)
def test_design_whole_launcher_binds_processes_and_never_accepts_reporting_grace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, expired_phase: str | None,
    design_case: DesignCase, work: int, wall: float,
) -> None:
    from types import SimpleNamespace

    from tools import meshing_design_source as source

    published = {
        "status": "published", "content_id": "actual-content",
        "identity": {"accepted": "all-fields"}, "interpreter_process": 101,
    }
    continued = {
        "status": "passed", "identity": published["identity"],
        "interpreter_process": 102,
    }
    calls: list[tuple[list[str], float]] = []
    reference = tmp_path / "produce.json"

    def launch(command: list[str], *, cwd: Path, timeout: float) -> Any:
        del cwd
        calls.append((command, timeout))
        phase = "producer" if len(calls) == 1 else "cold"
        if phase == "cold":
            assert json.loads(reference.read_text()) == published
            assert command[4:7] == [
                str((tmp_path / "archive.npz").resolve()),
                published["content_id"], str(reference.resolve()),
            ]
        if expired_phase == f"killed-{phase}":
            raise source.CampaignProcessTimeout(
                "original whole workflow clock watchdog expired",
                stdout=f"observed-{phase}-stdout",
                stderr=f"observed-{phase}-diagnostics",
                peak_resident_bytes=123456,
            )
        return SimpleNamespace(
            returncode=0, stderr=f"observed-{phase}-diagnostics",
            deadline_expired=expired_phase == phase,
            stdout=json.dumps(published if phase == "producer" else continued),
        )

    monkeypatch.setattr(source, "run_campaign_process", launch)
    if expired_phase is not None:
        with pytest.raises(TimeoutError, match="original whole workflow clock") as failure:
            source.qualify(tmp_path / "archive.npz", reference, design_case=design_case)
        phase = expired_phase.removeprefix("killed-")
        if expired_phase.startswith("killed-"):
            assert isinstance(failure.value, source.CampaignProcessTimeout)
            notes = "\n".join(failure.value.__notes__)
            assert f"observed-{phase}-diagnostics" in notes
            assert f"observed-{phase}-stdout" in notes
            assert failure.value.peak_resident_bytes == 123456
        else:
            assert f"observed-{phase}-diagnostics" in str(failure.value)
        assert len(calls) == (1 if phase == "producer" else 2)
    else:
        result = source.qualify(
            tmp_path / "archive.npz", reference, design_case=design_case
        )
        assert result["producer_process"] == 101
        assert result["cold"] == continued
        assert 0.0 < calls[1][1] <= calls[0][1] <= wall
        assert calls[0][0][5:7] == ["--design-case", design_case]
        for command, _ in calls:
            assert command[-4:] == [
                "--maximum-work-units", str(work), "--timeout", str(wall),
            ]


def test_design_mutable_routed_images_renew_bank_and_incidence_checks() -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as enclosure
    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
        owned_coordinate_preparation,
    )

    element = BarycentricCellGeometryElement(
        coordinate_lagrange_element("triangle", 1),
        np.asarray(((1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.0, 0.0, 1.0))),
    )
    bank = tuple(tuple(Fraction(value) for value in row) for row in ((0, 0), (1, 0), (0, 1)))
    route = np.asarray(((0, 1, 2),), dtype=np.int64)
    vertices = route.copy()
    with owned_coordinate_preparation(100000, 10000000) as budget:
        def images(controls: Any, routes: np.ndarray = route) -> Any:
            return enclosure.coordinate_vertex_images(
                ((element, routes, np.repeat(vertices, routes.shape[0], axis=0)),),
                controls, 3, mismatch="design shared vertex mismatch", budget=budget,
            )

        original = images(bank)
        first_work = budget.work_units
        assert images(bank) == original
        assert budget.work_units > first_work
        changed = (bank[0], (Fraction(2), Fraction(0)), bank[2])
        assert images(changed) != original
        with pytest.raises(ValueError, match="shared vertex mismatch"):
            images(bank, np.asarray(((0, 1, 2), (0, 2, 1))))
        with pytest.raises(ValueError, match="out-of-range row"):
            images(bank, np.asarray(((0, 1, 3),)))


def test_design_retained_tapes_compose_real_qoi_and_transfer_adjoints() -> None:
    import jax.numpy as jnp

    points = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    coefficients = jnp.asarray((1.0, 2.0, 3.0))
    radius = jnp.asarray(1.3)

    def coordinates(value: Any) -> Any:
        return points * value

    def pde(value: Any) -> Any:
        return coefficients * value**2

    geometry, solution = design._design_qualification_linearizations(
        coordinates, pde, radius
    )
    cells = (jnp.asarray(((0, 1, 2),)),)
    value, adjoint, tangent = design._design_qualification_adjoint_values(
        geometry, solution, cells, cells
    )
    # Area scales as r² and the independently supplied field as r².
    assert float(value) == pytest.approx(float(radius**4), rel=1e-12)
    assert float(adjoint) == pytest.approx(float(4.0 * radius**3), rel=1e-12)
    assert float(tangent) == pytest.approx(float(adjoint), rel=1e-12)

    matrix = jnp.asarray(((0.25, 0.75, 0.0), (0.0, 0.5, 0.5)))
    outer = phx.linalg.prepare_linearization(lambda values: matrix @ values, solution.primal)
    direction, step = jnp.asarray(0.7), 1.0e-5
    lower = matrix @ pde(radius - step * direction)
    upper = matrix @ pde(radius + step * direction)
    checked = design._design_qualification_derivative_probe(
        (solution, outer), direction, lower, upper, step
    )
    assert checked["finite_difference_error"] < 1e-8
    assert checked["JVP_VJP_duality_error"] < 1e-12


def test_design_quality_reuses_kernel_across_active_counts_without_changing_bits() -> None:
    import jax

    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.meshing import _quality as quality
    from tools.compile_counter import CompileCounter

    source = coordinate_lagrange_element("triangle", 1)
    rule = quality._mapped_quality_rule("triangle", source)
    coordinates = jax.device_put(np.broadcast_to(_FOLDED[0][:3], (19, 3, 3)).copy())
    jax.clear_caches()
    with CompileCounter() as first:
        baseline = jax.block_until_ready(
            quality._mapped_sample_worksets("triangle", source, coordinates[:17], None, rule)
        )
    with CompileCounter() as reused:
        changed = jax.block_until_ready(
            quality._mapped_sample_worksets("triangle", source, coordinates, None, rule)
        )
    assert any("_mapped_sample_values" in event.kernel_name for event in first.events)
    assert not any("_mapped_sample_values" in event.kernel_name for event in reused.events)
    for left, right in zip(jax.tree.leaves(baseline), jax.tree.leaves(changed), strict=True):
        np.testing.assert_array_equal(
            np.asarray(left).view(np.uint8), np.asarray(right[:17]).view(np.uint8)
        )


def test_design_tape_actions_reuse_executables_across_derivative_consumers() -> None:
    import jax
    import jax.numpy as jnp

    from tools.compile_counter import CompileCounter

    points = jnp.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    coefficients = jnp.asarray((1.0, 2.0, 3.0))
    radius = jnp.asarray(1.3, dtype=jnp.float64)
    direction = jnp.asarray(0.7, dtype=jnp.float64)

    def coordinates(value: Any) -> Any:
        return points * value

    def pde(value: Any) -> Any:
        return coefficients * value**2

    geometry, solution = design._design_qualification_linearizations(
        coordinates, pde, radius
    )
    cells = (jnp.asarray(((0, 1, 2),)),)
    step = 1.0e-5
    with CompileCounter() as first:
        for tape, function in ((geometry, coordinates), (solution, pde)):
            design._design_qualification_derivative_probe(
                (tape,),
                direction,
                function(radius - step * direction),
                function(radius + step * direction),
                step,
            )
    with CompileCounter() as reused:
        value, adjoint, tangent = jax.block_until_ready(
            design._design_qualification_adjoint_values(geometry, solution, cells, cells)
        )
    assert any(
        "_design_qualification_tape_jvp" in event.kernel_name for event in first.events
    )
    assert any(
        "_design_qualification_tape_vjp" in event.kernel_name for event in first.events
    )
    assert not any(
        "_design_qualification_tape_" in event.kernel_name for event in reused.events
    )
    assert float(value) == pytest.approx(float(radius**4), rel=1e-12)
    assert float(adjoint) == pytest.approx(float(4.0 * radius**3), rel=1e-12)
    assert float(tangent) == pytest.approx(float(adjoint), rel=1e-12)


def test_design_reanalysis_reexecutes_assembly_and_solve_without_duplicate_lowering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jax.numpy as jnp

    result = _certified(*_FOLDED)
    scalar, all_fields, problem, fields = design._design_qualification_prepare(result)
    seed = jnp.zeros(scalar.field_spaces[0].layout.size, dtype=jnp.float64)
    values, compiler, _ = design._compiled_solver_stages(problem, seed, 1)
    assert compiler["solver"]["successful"]
    values = design._design_qualification_array(values)
    _, reanalysis, _, records = design._design_qualification_transaction(
        result, result, scalar, all_fields, problem, fields, values
    )
    calls = {"assembly": 0, "solve": 0}
    original_assembly = design._prepare_corpus_epoch_system
    original_solve = design._solve_corpus_epoch_system

    def assembly(*args: Any, **kwargs: Any) -> Any:
        calls["assembly"] += 1
        return original_assembly(*args, **kwargs)

    def solve(epoch: Any, initial: Any, *args: Any, **kwargs: Any) -> Any:
        calls["solve"] += 1
        np.testing.assert_array_equal(np.asarray(initial), np.asarray(seed))
        return original_solve(epoch, initial, *args, **kwargs)

    monkeypatch.setattr(design, "_prepare_corpus_epoch_system", assembly)
    monkeypatch.setattr(design, "_solve_corpus_epoch_system", solve)
    carried = (
        values, jnp.full_like(values, 2.0),
        all_fields.dof_maps[2].dof_coordinates[:, 1] + 0.5,
    )
    for _ in range(2):
        evidence = reanalysis(result.mesh, carried, None, None)
        assert evidence.revision_id == result.result_id
        assert records[-1]["assembly_compilation_count"] == 0
        assert records[-1]["solve_compilation_count"] == 0
    assert calls == {"assembly": 2, "solve": 2}
    assert records[-1]["recomputed_solution_defect"] <= 1e-10
    with pytest.raises(RuntimeError, match="independently recomputed PDE"):
        reanalysis(result.mesh, (values + 0.01, *carried[1:]), None, None)
    assert calls == {"assembly": 3, "solve": 3}


@pytest.mark.parametrize("depth", (1, 2))
def test_design_corner_basis_reuses_code_not_source_proofs_or_numerical_actions(
    monkeypatch: pytest.MonkeyPatch, depth: int,
) -> None:
    import equinox as eqx
    import jax

    from phydrax.discretization._cell_geometry import (
        BarycentricCellGeometryElement,
        coordinate_lagrange_element,
        ScalarCoordinateElement,
    )
    from phydrax.meshing.qualification import _design_oracle as oracle
    from tools.compile_counter import CompileCounter

    root = coordinate_lagrange_element("triangle", 1)
    reference = jax.device_put(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    )
    elements = []
    for index in range(21):
        weight = (index + 1) / 32.0
        element = BarycentricCellGeometryElement(
            root,
            np.asarray(((1.0, -0.0, 0.0), (weight, 1.0 - weight, 0.0), (0.0, 0.0, 1.0))),
        )
        if depth == 2:
            element = BarycentricCellGeometryElement(
                element,
                np.asarray(((0.75, 0.25, 0.0), (0.0, 0.75, 0.25), (0.0, 0.0, 1.0))),
            )
        elements.append(element)

    @eqx.filter_jit
    def prior(element: ScalarCoordinateElement, points: jax.Array) -> jax.Array:
        return element.tabulate(points)[0]

    expected = [np.asarray(prior(element, reference)) for element in elements]
    identities = [element.element_id for element in elements]
    fingerprints = [phx.array_tree_fingerprint(element) for element in elements]
    assert len(set(identities)) == 21
    proof = oracle._require_affine_triangle
    proved: list[ScalarCoordinateElement] = []

    def observed(element: ScalarCoordinateElement) -> ScalarCoordinateElement:
        proved.append(element)
        return proof(element)

    monkeypatch.setattr(oracle, "_require_affine_triangle", observed)
    jax.clear_caches()
    with CompileCounter() as compiler:
        actual = [
            np.asarray(oracle.affine_triangle_corner_basis(element))
            for element in elements
        ]
    assert sum("_corner_basis" in event.kernel_name for event in compiler.events) == 1
    assert len(proved) == len(elements)
    assert all(left is right for left, right in zip(proved, elements, strict=True))
    for left, right in zip(actual, expected, strict=True):
        assert left.dtype == right.dtype == np.dtype(np.float64)
        assert left.shape == right.shape == (3, 3)
        assert left.tobytes() == right.tobytes()
    assert actual[0].tobytes() != actual[-1].tobytes()
    assert [element.element_id for element in elements] == identities
    assert [phx.array_tree_fingerprint(element) for element in elements] == fingerprints


@pytest.mark.parametrize(
    ("work", "wall"),
    ((0, 180.0), (-1, 180.0), (True, 180.0), (1.5, 180.0),
     (480_000, 0.0), (480_000, -1.0), (480_000, True),
     (480_000, float("nan")), (480_000, float("inf"))),
)
def test_design_authored_controls_refuse_invalid_limits(work: Any, wall: Any) -> None:
    from tools import meshing_design_source as source

    with pytest.raises(ValueError, match="maximum_work_units|timeout"):
        source.original_controls("larger-sphere", maximum_work_units=work, timeout=wall)


def _authored_design_reference(controls: QualificationControls) -> dict[str, Any]:
    return {
        "design_case": controls.design_case,
        "source_clock_seconds": controls.timeout,
        "maximum_work_units": controls.maximum_work_units,
        "original_campaign": {
            "status": "passed",
            "controls": {
                name: getattr(controls, name)
                for name in (
                    "design_case", "resolution", "capacity", "adaptation_rounds",
                    "repeats", "maximum_work_units", "timeout",
                )
            },
            "case": design._design_qualification_case(controls),
        },
    }


@pytest.mark.parametrize(
    "path",
    (("maximum_work_units",), ("source_clock_seconds",),
     ("original_campaign", "controls", "maximum_work_units"),
     ("original_campaign", "controls", "timeout")),
)
def test_design_consumer_binds_explicit_producer_limits(
    tmp_path: Path, path: tuple[str, ...],
) -> None:
    from tools import meshing_design_source as source

    controls = source.original_controls(
        "larger-sphere", maximum_work_units=479_999, timeout=179.0
    )
    reference = _authored_design_reference(controls)
    record = tmp_path / "produce.json"
    record.write_text(json.dumps(reference))
    observed, _ = source._design_consumer_reference(
        record, maximum_work_units=479_999, timeout=179.0
    )
    assert (observed.maximum_work_units, observed.timeout) == (479_999, 179.0)
    with pytest.raises(RuntimeError, match="resource controls"):
        source._design_consumer_reference(record)
    owner = reference
    for name in path[:-1]:
        owner = owner[name]
    owner[path[-1]] -= 1
    record.write_text(json.dumps(reference))
    with pytest.raises(RuntimeError, match="consume.reference"):
        source._design_consumer_reference(
            record, maximum_work_units=479_999, timeout=179.0
        )


@pytest.mark.parametrize("changed", ("maximum_work_units", "maximum_wall_seconds"))
def test_design_consumer_refuses_archive_control_mismatch_before_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    from types import SimpleNamespace

    from tools import meshing_design_source as source

    controls = source.original_controls("larger-sphere")
    reference = tmp_path / "produce.json"
    reference.write_text(json.dumps(_authored_design_reference(controls)))
    limits = {"maximum_work_units": 480_000, "maximum_wall_seconds": 180.0}
    limits[changed] -= 1
    monkeypatch.setattr(
        source, "read_meshing_source_execution_controls",
        lambda *args, **kwargs: SimpleNamespace(limits=SimpleNamespace(**limits)),
    )
    with pytest.raises(RuntimeError, match="consume.archive resource controls"):
        source.consume(tmp_path / "archive.npz", "content", reference)


@pytest.mark.parametrize("operation", ("produce", "qualify", "consume"))
@pytest.mark.parametrize("explicit", (False, True))
def test_design_source_cli_forwards_authored_limits(
    monkeypatch: pytest.MonkeyPatch, operation: str, explicit: bool,
) -> None:
    from tools import meshing_design_source as source

    arguments = {
        "produce": ["archive.npz"],
        "qualify": ["archive.npz", "produce.json"],
        "consume": ["archive.npz", "content", "produce.json"],
    }[operation]
    options = ["--maximum-work-units", "479999", "--timeout", "179"] if explicit else []
    calls: list[dict[str, Any]] = []

    def run(*args: Any, **kwargs: Any) -> dict[str, object]:
        del args
        calls.append(kwargs)
        return {"status": "parser-only"}

    monkeypatch.setattr(source, operation, run)
    monkeypatch.setattr(source.sys, "argv", ["design", operation, *arguments, *options])
    source.main()
    assert calls[0]["maximum_work_units"] == (479_999 if explicit else None)
    assert calls[0]["timeout"] == (179.0 if explicit else None)
    if operation != "consume":
        assert calls[0]["design_case"] == "routine-sphere"


@pytest.mark.parametrize("inherited_expiry", (102.0, 101.0))
def test_design_cold_keeps_inherited_deadline_at_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inherited_expiry: float,
) -> None:
    from types import SimpleNamespace

    from tools import meshing_design_source as source
    from tools.meshing_qualification import _runtime as runtime

    controls = source.original_controls("larger-sphere")
    reference = tmp_path / "produce.json"
    reference.write_text(json.dumps(_authored_design_reference(controls)))
    clock = [100.0]
    monkeypatch.setattr(source, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runtime, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(runtime, "_LAUNCH_EXPIRES_AT", inherited_expiry)

    def bootstrap(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        clock[0] += 1.0
        return SimpleNamespace(
            execution_evidence=None,
            limits=SimpleNamespace(maximum_work_units=480_000, maximum_wall_seconds=180.0),
        )

    class RequestObserved(Exception):
        pass

    def request(*args: Any, **kwargs: Any) -> Any:
        del args
        assert kwargs["workflow_remaining_seconds"] == 1.0
        raise RequestObserved

    monkeypatch.setattr(source, "read_meshing_source_execution_controls", bootstrap)
    monkeypatch.setattr(source, "execute_native_request", request)
    expected = RequestObserved if inherited_expiry == 102.0 else TimeoutError
    with pytest.raises(expected):
        source.consume(tmp_path / "archive.npz", "content", reference)


@pytest.mark.meshcore
def test_design_authored_work_reaches_request_identity_and_native_r_minus_one() -> None:
    from phydrax._meshcore import MeshcoreError, NativeExecutionScope
    from tools import meshing_design_source as source

    identities = []
    for work in (17, 16):
        controls = source.original_controls(
            "routine-sphere", maximum_work_units=work, timeout=179.0
        )
        _, _, specification, _ = design._design_qualification_source(
            controls, design._design_qualification_case(controls)
        )
        limits = specification.limits
        assert (limits.maximum_work_units, limits.maximum_wall_seconds) == (work, 179.0)
        identities.append((specification.specification_id, limits.limits_id))
        if work == 17:
            with NativeExecutionScope(limits.native_limits()) as scope:
                scope.charge(work=17)
                assert scope.remaining().remaining_work_units == 0
        else:
            with (
                pytest.raises(MeshcoreError, match="CAPACITY_EXCEEDED") as refusal,
                NativeExecutionScope(limits.native_limits()) as scope,
            ):
                scope.charge(work=17)
            assert refusal.value.work_counters is not None
    assert identities[0][0] != identities[1][0]
    assert identities[0][1] != identities[1][1]
