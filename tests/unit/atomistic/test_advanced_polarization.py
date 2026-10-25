#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax import config as pxconfig
from phydrax._model._ports import ValuePort
from phydrax.atomistic._conditioning import (
    AtomisticConditionLevel,
    AtomisticFieldReference,
    AtomisticObservable,
)
from phydrax.atomistic._multipole_basis import GaussianMultipoleWidths
from phydrax.atomistic._multipole_electrostatics import (
    EwaldMultipoleRoute,
    FiniteMultipoleRoute,
    ParticleMeshMultipoleRoute,
)
from phydrax.atomistic._polarizable_force_field import (
    AngleAnglePotential,
    Buffered147Potential,
    ChargeFluxPotential,
    ChargePenetrationPotential,
    ChargeTransferPotential,
    DampedDispersionPotential,
    evaluate_polarizable_term,
    OutOfPlaneBendPotential,
    PauliRepulsionPotential,
    PolarizableForceFieldPlan,
    StretchBendPotential,
)
from phydrax.atomistic._polarization import (
    evaluate_implicit_polarization_jvp,
    evaluate_prepared_polarization,
    InducedDipolePolarizationPotential,
    PermanentMultipoleSiteData,
    PolarizationCurvatureProbe,
    PolarizationOperatorPlan,
    PolarizationPlan,
    PolarizationPreconditionerKind,
    PolarizationPreconditionerPlan,
    PolarizationScaleData,
    PolarizationSolverKind,
    PolarizationStabilityPolicy,
    prepared_polarization_energy,
)
from phydrax.atomistic._reciprocal_mesh import ReciprocalMeshPlan, ReciprocalModeSetPlan
from phydrax.units import DIMENSIONLESS


@pytest.fixture(autouse=True)
def _double_precision() -> Iterator[None]:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        yield


# --- shared fixtures -----------------------------------------------------------


def _multipoles(charges: Any, polarizabilities: Any, *, damping: Any = 4.0) -> Any:
    count = len(charges)
    return PermanentMultipoleSiteData(
        np.asarray(charges, dtype=np.float64),
        np.zeros((count, 3)),
        np.zeros((count, 3, 3)),
        np.asarray(polarizabilities, dtype=np.float64),
        np.full((count,), damping, dtype=np.float64),
    )


def _scales(
    pairs: list[list[int]], d: list[float], p: list[float], u: list[float]
) -> PolarizationScaleData:
    return PolarizationScaleData(
        np.asarray(pairs, dtype=np.int64),
        np.asarray(d, dtype=np.float64),
        np.asarray(p, dtype=np.float64),
        np.asarray(u, dtype=np.float64),
        site_capacity=2,
    )


def _two_site_case() -> Any:
    positions = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=jnp.float64)
    multipoles = _multipoles([1.0, -1.0], [0.1, 0.2], damping=10.0)
    return positions, multipoles


def _two_site_reference(coupling: float, field: float) -> np.ndarray:
    """Closed-form induced dipoles of the axial two-site problem.

    Along the axis the (damped) dipole tensor is ``lambda * 2/R^3`` and both
    charge fields are ``lambda / R^2`` toward +x, so the 2x2 system is
    ``[[1/a0, -c], [-c, 1/a1]] u = [f, f]``.
    """
    matrix = np.asarray([[10.0, -coupling], [-coupling, 5.0]])
    expected = np.zeros((2, 3))
    expected[:, 0] = np.linalg.solve(matrix, np.asarray([field, field]))
    return expected


def _damped_two_site_reference() -> np.ndarray:
    damping = 1.0 - np.exp(-10.0 * 2.0**3)
    return _two_site_reference(0.25 * damping, 0.25 * damping)


def _plan(**kwargs: Any) -> PolarizationPlan:
    settings: dict[str, Any] = {
        "coulomb_constant": 1.0,
        "maximum_iterations": 16,
        "tolerance": 1.0e-10,
    }
    settings.update(kwargs)
    route = settings.get("route")
    if route is not None and route.periodic and "stability" not in settings:
        # Two-site periodic cases: the bounded 6x6 dense-spectrum reference.
        settings["stability"] = PolarizationStabilityPolicy("dense-spectrum")
    return PolarizationPlan(**settings)


def _central_energy_derivative(
    prepared: Any, positions: Any, direction: Any, step: float, **kwargs: Any
) -> float:
    plus = prepared_polarization_energy(prepared, positions + step * direction, **kwargs)[
        0
    ]
    minus = prepared_polarization_energy(
        prepared, positions - step * direction, **kwargs
    )[0]
    return float((plus - minus) / (2.0 * step))


PERIODIC_POSITIONS = jnp.asarray([[1.0, 1.0, 1.0], [3.0, 1.0, 1.0]], dtype=jnp.float64)
CUBIC = 8.0 * np.eye(3)


def _ewald_route(extents: tuple[int, int, int] = (6, 6, 6)) -> EwaldMultipoleRoute:
    return EwaldMultipoleRoute(
        0.7,
        3.9,
        ReciprocalModeSetPlan(extents),
        real_space="explicit-image-reference",
    )


# --- 1. analytic two-site solve -------------------------------------------------


def test_two_site_pcg_solve_matches_closed_form_and_jit() -> None:
    positions, multipoles = _two_site_case()
    prepared = _plan(tolerance=1.0e-12).prepare(multipoles)
    result = prepared.solve(positions)
    state = result.state
    expected = _damped_two_site_reference()
    assert bool(result.successful & state.successful)
    assert bool(state.converged) and bool(state.stationary)
    assert state.functional == "stationary"
    assert state.derivative_mode == "implicit-stationary"
    np.testing.assert_allclose(
        state.induced_dipoles, expected, rtol=1.0e-10, atol=1.0e-13
    )
    compiled = jax.jit(lambda value: prepared.solve(value).state.induced_dipoles)(
        positions
    )
    np.testing.assert_allclose(compiled, expected, rtol=1.0e-10, atol=1.0e-13)


# --- 2. truncated conjugate gradient semantics ----------------------------------


def test_tcg_exhausting_krylov_space_matches_pcg_but_is_not_converged() -> None:
    positions, multipoles = _two_site_case()
    pcg = _plan(tolerance=1.0e-12).prepare(multipoles).solve(positions)
    tcg = (
        _plan(tolerance=1.0e-12, solver_kind=PolarizationSolverKind.TCG, tcg_order=4)
        .prepare(multipoles)
        .solve(positions)
    )
    assert bool(pcg.successful & tcg.successful)
    assert not bool(tcg.state.converged)
    assert tcg.state.functional == "tcg-order-4"
    np.testing.assert_allclose(
        tcg.state.induced_dipoles, pcg.state.induced_dipoles, rtol=1.0e-8, atol=1.0e-11
    )


def test_tcg_order_one_is_force_valid_algorithmic_functional() -> None:
    positions, multipoles = _two_site_case()
    tcg = _plan(tolerance=1.0, solver_kind="tcg", tcg_order=1).prepare(multipoles)
    pcg = _plan(tolerance=1.0e-12).prepare(multipoles)
    state = tcg.solve(positions).state
    assert not bool(state.converged)
    assert bool(state.successful) and bool(state.force_valid)
    assert state.functional == "tcg-order-1"
    evaluation = evaluate_prepared_polarization(tcg, positions)
    assert bool(evaluation.successful)
    assert evaluation.evidence.mode == "algorithmic-fixed-order"
    assert evaluation.evidence.functional == "tcg-order-1"
    # The Hylleraas functional is minimised by the stationary solution.
    tcg_energy = prepared_polarization_energy(tcg, positions)[0]
    pcg_energy = prepared_polarization_energy(pcg, positions)[0]
    assert float(tcg_energy) >= float(pcg_energy) - 1.0e-14
    # Forces are exact derivatives of the TCG functional itself.
    step = 1.0e-4
    for site, axis in ((1, 0), (0, 0), (1, 1)):
        direction = jnp.zeros_like(positions).at[site, axis].set(1.0)
        finite = _central_energy_derivative(tcg, positions, direction, step)
        np.testing.assert_allclose(
            -float(evaluation.forces[site, axis]), finite, rtol=1.0e-6, atol=1.0e-10
        )


def test_tcg_refuses_predictor_state() -> None:
    positions, multipoles = _two_site_case()
    tcg = _plan(solver_kind="tcg", tcg_order=2).prepare(multipoles)
    history = tcg.initial_predictor_state()
    with pytest.raises(ValueError, match="TCG"):
        tcg.solve(positions, predictor_state=history)


# --- 3. preconditioners ---------------------------------------------------------


def test_preconditioners_apply_and_agree_on_solution() -> None:
    positions, multipoles = _two_site_case()
    operator = _plan().operator.prepare(multipoles)
    residual = jnp.asarray([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    local = (
        PolarizationPreconditionerPlan(PolarizationPreconditionerKind.POLARIZABILITY)
        .prepare(operator)
        .apply(residual)
    )
    identity = (
        PolarizationPreconditionerPlan(PolarizationPreconditionerKind.IDENTITY)
        .prepare(operator)
        .apply(residual)
    )
    assert bool(local.successful & identity.successful)
    np.testing.assert_allclose(local.value, residual * np.asarray([[0.1], [0.2]]))
    np.testing.assert_allclose(identity.value, residual)
    solutions = [
        _plan(tolerance=1.0e-12, preconditioner_kind=kind)
        .prepare(multipoles)
        .solve(positions)
        for kind in ("polarizability", "identity")
    ]
    assert all(bool(item.successful) for item in solutions)
    np.testing.assert_allclose(
        solutions[0].state.induced_dipoles,
        solutions[1].state.induced_dipoles,
        rtol=1.0e-9,
        atol=1.0e-12,
    )


def test_polarizability_preconditioner_uses_unit_row_for_nonpolarizable_site() -> None:
    multipoles = _multipoles([0.0, 0.0], [0.2, 0.0])
    operator = _plan().operator.prepare(multipoles)
    residual = jnp.asarray([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    local = (
        PolarizationPreconditionerPlan("polarizability").prepare(operator).apply(residual)
    )
    np.testing.assert_allclose(local.value, residual * np.asarray([[0.2], [1.0]]))


# --- 4. warm start ---------------------------------------------------------------


def test_pcg_warm_start_reuses_previous_solution() -> None:
    positions, multipoles = _two_site_case()
    prepared = _plan(tolerance=1.0e-10).prepare(multipoles)
    cold = prepared.solve(positions)
    warm = prepared.solve(positions, predictor_state=cold.predictor_state)
    assert cold.predictor_state.history.shape == (2, 2, 3)
    assert int(cold.predictor_state.valid_count) == 1
    np.testing.assert_allclose(
        warm.initial_dipoles, cold.state.induced_dipoles, rtol=0.0, atol=0.0
    )
    assert int(warm.state.iterations) <= int(cold.state.iterations)
    assert bool(warm.successful)


# --- 5. polarization catastrophe -------------------------------------------------


def test_polarization_catastrophe_is_reported_unsuccessful() -> None:
    positions, _ = _two_site_case()
    unstable = _multipoles([1.0, -1.0], [10.0, 10.0], damping=10.0)
    result = (
        _plan(maximum_iterations=8, tolerance=1.0e-10).prepare(unstable).solve(positions)
    )
    assert not bool(result.successful)
    assert not bool(result.state.stability_proven)
    assert result.state.stability_certificate == "scaled-block-gershgorin"
    assert float(result.state.stability_lower_bound) < 0.0


def test_unstable_mode_orthogonal_to_field_is_not_certified() -> None:
    """Review counterexample: r=1, k=1, q=(1,1), alpha=(3/4,3/4), no damping.

    The axial block [[4/3,-2],[-2,4/3]] has eigenvalue -2/3 on the symmetric
    mode, orthogonal to the field (-1,1); CG converges with u . E_p > 0, so
    solution curvature must not certify stability.
    """
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    multipoles = _multipoles([1.0, 1.0], [0.75, 0.75], damping=1.0)
    for stability in (
        PolarizationStabilityPolicy("scaled-block-gershgorin"),
        PolarizationStabilityPolicy("dense-spectrum"),
    ):
        prepared = _plan(damping="none", stability=stability).prepare(multipoles)
        result = prepared.solve(positions)
        assert float(result.stability.solution_curvature) > 0.0
        assert not bool(result.state.stability_proven)
        assert not bool(result.successful)
        assert not bool(result.state.force_valid)
        evaluation = evaluate_prepared_polarization(prepared, positions)
        assert not bool(evaluation.successful)
        assert bool(jnp.all(jnp.isnan(evaluation.forces)))
    dense = _plan(
        damping="none", stability=PolarizationStabilityPolicy("dense-spectrum")
    ).prepare(multipoles)
    np.testing.assert_allclose(
        float(dense.solve(positions).state.stability_lower_bound),
        -2.0 / 3.0 * 0.75,
        rtol=1.0e-10,
    )


def test_assumed_positive_definite_is_reported_as_assumption_not_proof() -> None:
    positions, multipoles = _two_site_case()
    result = (
        _plan(stability=PolarizationStabilityPolicy("assumed-positive-definite"))
        .prepare(multipoles)
        .solve(positions)
    )
    assert bool(result.successful)
    assert result.state.stability_assumed
    assert not bool(result.state.stability_proven)
    assert bool(jnp.isnan(result.state.stability_lower_bound))


def test_gershgorin_and_dense_certificates_prove_the_stable_case() -> None:
    positions, multipoles = _two_site_case()
    damping = 1.0 - np.exp(-10.0 * 2.0**3)
    # Off-diagonal axial block 2 lambda / R^3 scaled by sqrt(alpha0 alpha1).
    gershgorin = 1.0 - np.sqrt(0.02) * 2.0 * damping / 8.0
    for certificate, lower in (
        ("scaled-block-gershgorin", gershgorin),
        ("dense-spectrum", gershgorin),
    ):
        result = (
            _plan(stability=PolarizationStabilityPolicy(certificate))
            .prepare(multipoles)
            .solve(positions)
        )
        assert bool(result.successful)
        assert bool(result.state.stability_proven)
        np.testing.assert_allclose(
            float(result.state.stability_lower_bound), lower, rtol=1.0e-10
        )


def test_periodic_plans_require_an_explicit_stability_policy() -> None:
    with pytest.raises(ValueError, match="explicit stability policy"):
        PolarizationPlan(coulomb_constant=1.0, route=_ewald_route())
    with pytest.raises(ValueError, match="nonperiodic operators only"):
        PolarizationPlan(
            coulomb_constant=1.0,
            route=_ewald_route(),
            stability=PolarizationStabilityPolicy("scaled-block-gershgorin"),
        ).prepare(_two_site_case()[1], cell=phx.discretization.PeriodicCell(CUBIC))
    with pytest.raises(ValueError, match="maximum_dense_bytes"):
        _plan(
            stability=PolarizationStabilityPolicy(
                "dense-spectrum", maximum_dense_bytes=64
            )
        ).prepare(_two_site_case()[1])


def test_noop_listed_pair_does_not_trigger_minimum_image_refusal() -> None:
    cell = phx.discretization.PeriodicCell(10.0 * np.eye(3))
    positions = jnp.asarray([[1.0, 1.0, 1.0], [5.0, 5.0, 1.0]], dtype=jnp.float64)
    multipoles = _multipoles([0.5, -0.5], [0.1, 0.1], damping=1.0)
    plan = PolarizationOperatorPlan(
        coulomb_constant=1.0,
        route=EwaldMultipoleRoute(0.35, 9.0, ReciprocalModeSetPlan((4, 4, 4))),
    )
    trial = jnp.asarray([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]], dtype=jnp.float64)
    unlisted = plan.prepare(multipoles, cell=cell).apply(
        positions, trial, cell_vectors=cell.vectors
    )
    listed = plan.prepare(
        multipoles,
        scaling=_scales([[0, 1]], [1.0], [1.0], [1.0]),
        cell=cell,
    ).apply(positions, trial, cell_vectors=cell.vectors)
    assert bool(unlisted.successful & listed.successful)
    np.testing.assert_allclose(listed.action, unlisted.action, rtol=1.0e-12, atol=1.0e-14)
    # A contributing scale on that non-unique minimum image is refused.
    contributing = plan.prepare(
        multipoles,
        scaling=_scales([[0, 1]], [0.5], [0.5], [0.5]),
        cell=cell,
    ).apply(positions, trial, cell_vectors=cell.vectors)
    assert not bool(contributing.successful)


def test_curvature_probe_ritz_sign_detects_indefinite_operator() -> None:
    positions, stable = _two_site_case()
    unstable = _multipoles([1.0, -1.0], [10.0, 10.0], damping=10.0)
    probe = PolarizationCurvatureProbe(subspace_dimension=4, maximum_steps=20)
    plan = _plan(
        maximum_iterations=8,
        curvature_probe=probe,
        stability=PolarizationStabilityPolicy("assumed-positive-definite"),
    )
    # Under the assumption policy only detection can refuse: a negative Ritz
    # value (an upper bound of lambda_min) proves indefiniteness.
    bad = plan.prepare(unstable).solve(positions)
    good = plan.prepare(stable).solve(positions)
    assert bad.stability.probed and good.stability.probed
    assert float(bad.stability.minimum_ritz_value) < 0.0
    assert bool(bad.stability.indefinite_detected)
    assert not bool(bad.successful)
    assert float(good.stability.minimum_ritz_value) > 0.0
    assert bool(good.successful)
    assert not bool(good.state.stability_proven)


# --- 6. sparse pair scaling ------------------------------------------------------


def test_sparse_scaling_scales_each_field_column() -> None:
    positions, multipoles = _two_site_case()
    scaling = _scales([[0, 1]], [0.0], [0.5], [0.25])
    trial = jnp.asarray([[0.1, 0.03, 0.0], [-0.2, 0.0, 0.05]])
    scaled = _plan().operator.prepare(multipoles, scaling=scaling).apply(positions, trial)
    full = _plan().operator.prepare(multipoles).apply(positions, trial)
    assert bool(scaled.successful & full.successful)
    assert float(jnp.max(jnp.abs(full.p_field))) > 0.0
    assert float(jnp.max(jnp.abs(full.u_field))) > 0.0
    np.testing.assert_allclose(scaled.d_field, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(
        scaled.p_field, 0.5 * full.p_field, rtol=1.0e-12, atol=1.0e-15
    )
    np.testing.assert_allclose(
        scaled.u_field, 0.25 * full.u_field, rtol=1.0e-12, atol=1.0e-15
    )


def test_nonpolarizable_site_has_zero_mutual_field_and_identity_row() -> None:
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    multipoles = _multipoles([0.0, 0.0], [0.2, 0.0])
    operator = _plan().operator.prepare(multipoles)
    trial = jnp.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    result = operator.apply(positions, trial)
    assert bool(result.successful)
    np.testing.assert_allclose(result.u_field, 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.action[0], 0.0, atol=1.0e-15)
    np.testing.assert_allclose(result.action[1], trial[1])


def test_coincident_fully_excluded_pair_is_successful_with_zero_fields() -> None:
    multipoles = _multipoles([1.0, -1.0], [0.2, 0.3])
    excluded = _scales([[0, 1]], [0.0], [0.0], [0.0])
    result = (
        _plan()
        .operator.prepare(multipoles, scaling=excluded)
        .apply(jnp.zeros((2, 3)), jnp.asarray([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]))
    )
    assert bool(result.successful)
    np.testing.assert_allclose(result.d_field, 0.0)
    np.testing.assert_allclose(result.p_field, 0.0)
    np.testing.assert_allclose(result.u_field, 0.0)


def test_scale_data_refuses_duplicate_and_self_pairs() -> None:
    with pytest.raises(ValueError, match="once"):
        _scales([[0, 1], [1, 0]], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="distinct"):
        _scales([[1, 1]], [0.0], [0.0], [0.0])


# --- 7. forces and implicit response ---------------------------------------------


def test_pcg_forces_match_finite_differences_of_energy() -> None:
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.3, -0.1], [0.4, 1.9, 0.5]], dtype=jnp.float64
    )
    multipoles = _multipoles([0.6, -0.4, -0.2], [0.3, 0.2, 0.25], damping=2.0)
    prepared = _plan(maximum_iterations=40, tolerance=1.0e-12).prepare(multipoles)
    evaluation = evaluate_prepared_polarization(prepared, positions)
    assert bool(evaluation.successful)
    assert evaluation.evidence.mode == "implicit-stationary"
    np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=1.0e-10)
    step = 1.0e-4
    for site in range(3):
        for axis in range(3):
            direction = jnp.zeros_like(positions).at[site, axis].set(1.0)
            finite = _central_energy_derivative(prepared, positions, direction, step)
            np.testing.assert_allclose(
                -float(evaluation.forces[site, axis]), finite, rtol=1.0e-6, atol=1.0e-9
            )


def test_implicit_jvp_matches_finite_difference_of_induced_dipoles() -> None:
    positions = jnp.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.3, -0.1], [0.4, 1.9, 0.5]], dtype=jnp.float64
    )
    multipoles = _multipoles([0.6, -0.4, -0.2], [0.3, 0.2, 0.25], damping=2.0)
    prepared = _plan(maximum_iterations=40, tolerance=1.0e-12).prepare(multipoles)
    direction = jnp.asarray(
        np.random.default_rng(3).normal(size=(3, 3)), dtype=jnp.float64
    )
    result = evaluate_implicit_polarization_jvp(prepared, positions, direction)
    assert bool(result.successful)
    assert result.evidence.mode == "implicit-stationary"
    step = 1.0e-5
    plus = prepared.solve(positions + step * direction).state.induced_dipoles
    minus = prepared.solve(positions - step * direction).state.induced_dipoles
    np.testing.assert_allclose(
        result.tangent, (plus - minus) / (2.0 * step), rtol=1.0e-6, atol=1.0e-9
    )
    np.testing.assert_allclose(
        result.primal, prepared.solve(positions).state.induced_dipoles
    )


def test_force_tolerance_gate_refuses_forces_of_loose_solve() -> None:
    positions, multipoles = _two_site_case()
    prepared = _plan(
        maximum_iterations=12, tolerance=1.0e-2, force_tolerance=1.0e-14
    ).prepare(multipoles)
    evaluation = evaluate_prepared_polarization(prepared, positions)
    assert bool(evaluation.state.converged)
    assert float(evaluation.state.residual) > 1.0e-14
    assert not bool(evaluation.state.force_valid)
    assert not bool(evaluation.successful)
    assert bool(jnp.all(jnp.isnan(evaluation.forces)))
    assert bool(jnp.isnan(evaluation.energy))


# --- 8. periodic routes ----------------------------------------------------------


def test_periodic_and_nonperiodic_cell_vector_contracts() -> None:
    positions, multipoles = _two_site_case()
    nonperiodic = _plan().prepare(multipoles)
    with pytest.raises(ValueError, match="cell_vectors require"):
        nonperiodic.solve(positions, cell_vectors=jnp.asarray(CUBIC))
    periodic = _plan(route=_ewald_route()).prepare(
        multipoles, cell=phx.discretization.PeriodicCell(CUBIC)
    )
    with pytest.raises(ValueError, match="requires cell_vectors"):
        periodic.solve(PERIODIC_POSITIONS)
    assert bool(
        periodic.solve(PERIODIC_POSITIONS, cell_vectors=jnp.asarray(CUBIC)).successful
    )


def test_periodic_pair_exclusion_removes_minimum_image_damped_member() -> None:
    _, multipoles = _two_site_case()
    cell = phx.discretization.PeriodicCell(CUBIC)
    vectors = jnp.asarray(CUBIC)
    trial = jnp.asarray([[0.1, 0.02, 0.0], [-0.2, 0.0, 0.04]])
    plan = _plan(route=_ewald_route())
    excluded_scales = _scales([[0, 1]], [0.0], [0.0], [0.0])
    full = plan.operator.prepare(multipoles, cell=cell).apply(
        PERIODIC_POSITIONS, trial, cell_vectors=vectors
    )
    excluded = plan.operator.prepare(
        multipoles, scaling=excluded_scales, cell=cell
    ).apply(PERIODIC_POSITIONS, trial, cell_vectors=vectors)
    finite = _plan().operator.prepare(multipoles).apply(PERIODIC_POSITIONS, trial)
    assert bool(full.successful & excluded.successful & finite.successful)
    for name in ("d_field", "p_field", "u_field"):
        np.testing.assert_allclose(
            getattr(excluded, name) - getattr(full, name),
            -getattr(finite, name),
            rtol=1.0e-10,
            atol=1.0e-13,
        )


def test_skew_triclinic_cell_is_supported_and_singular_cell_fails() -> None:
    _, multipoles = _two_site_case()
    skew = np.asarray([[8.0, 0.0, 0.0], [4.0, 7.0, 0.0], [1.0, 2.0, 8.0]])
    plan = _plan(route=_ewald_route())
    skew_result = plan.prepare(
        multipoles, cell=phx.discretization.PeriodicCell(skew)
    ).solve(PERIODIC_POSITIONS, cell_vectors=jnp.asarray(skew))
    assert bool(skew_result.successful)
    assert bool(jnp.all(jnp.isfinite(skew_result.state.induced_dipoles)))
    cubic = plan.prepare(multipoles, cell=phx.discretization.PeriodicCell(CUBIC))
    singular = cubic.solve(PERIODIC_POSITIONS, cell_vectors=jnp.zeros((3, 3)))
    assert not bool(singular.successful)


def test_periodic_cell_gradient_matches_finite_differences() -> None:
    """``cell_gradient`` is ``dE/dh`` at fixed Cartesian positions."""
    _, multipoles = _two_site_case()
    prepared = _plan(
        route=_ewald_route(), tolerance=1.0e-12, maximum_iterations=30
    ).prepare(multipoles, cell=phx.discretization.PeriodicCell(CUBIC))
    vectors = jnp.asarray(CUBIC)
    evaluation = evaluate_prepared_polarization(
        prepared, PERIODIC_POSITIONS, cell_vectors=vectors
    )
    assert bool(evaluation.successful)
    assert evaluation.cell_gradient is not None
    step = 1.0e-4
    for row, column in ((0, 0), (1, 1), (1, 0), (2, 1)):
        delta = jnp.zeros((3, 3)).at[row, column].set(step)
        plus = prepared_polarization_energy(
            prepared, PERIODIC_POSITIONS, cell_vectors=vectors + delta
        )[0]
        minus = prepared_polarization_energy(
            prepared, PERIODIC_POSITIONS, cell_vectors=vectors - delta
        )[0]
        np.testing.assert_allclose(
            float(evaluation.cell_gradient[row, column]),
            float((plus - minus) / (2.0 * step)),
            rtol=1.0e-5,
            atol=1.0e-9,
        )


def test_particle_mesh_route_agrees_with_ewald_route() -> None:
    _, multipoles = _two_site_case()
    cell = phx.discretization.PeriodicCell(CUBIC)
    system = phx.atomistic.AtomisticSystemPlan(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([1, 1], dtype=np.int64),
        np.asarray([1.0, 1.0], dtype=np.float64),
        phx.atomistic.AtomisticUnitSystem.reduced(),
        atom_type_ids=np.asarray([0, 0], dtype=np.int64),
        charges=np.asarray([1.0, -1.0], dtype=np.float64),
        cell=cell,
    ).prepare()
    ewald = _plan(route=_ewald_route((8, 8, 8)), tolerance=1.0e-12).prepare(
        multipoles, cell=system.cell
    )
    mesh = _plan(
        route=ParticleMeshMultipoleRoute(
            0.7, 3.9, ReciprocalMeshPlan((16, 16, 16), spline_degree=4)
        ),
        tolerance=1.0e-12,
    ).prepare(multipoles, cell=system.cell, particles=system.particle_discretization())
    vectors = jnp.asarray(CUBIC)
    reference = ewald.solve(PERIODIC_POSITIONS, cell_vectors=vectors)
    candidate = mesh.solve(PERIODIC_POSITIONS, cell_vectors=vectors)
    assert bool(reference.successful & candidate.successful)
    np.testing.assert_allclose(
        candidate.state.induced_dipoles,
        reference.state.induced_dipoles,
        rtol=1.0e-4,
        atol=1.0e-6,
    )


# --- 9. Gaussian sites -----------------------------------------------------------


def test_narrow_gaussian_sites_approach_undamped_point_sites() -> None:
    positions, multipoles = _two_site_case()
    point = _plan(damping="none", tolerance=1.0e-12).prepare(multipoles).solve(positions)
    narrow = (
        _plan(site_kind="gaussian", damping="none", tolerance=1.0e-12)
        .prepare(
            multipoles,
            widths=GaussianMultipoleWidths(
                jnp.asarray([0.05]), np.asarray([0, 0], dtype=np.int64)
            ),
        )
        .solve(positions)
    )
    assert bool(point.successful & narrow.successful)
    np.testing.assert_allclose(
        point.state.induced_dipoles, _two_site_reference(0.25, 0.25), rtol=1.0e-10
    )
    np.testing.assert_allclose(
        narrow.state.induced_dipoles,
        point.state.induced_dipoles,
        rtol=1.0e-9,
        atol=1.0e-12,
    )


def test_wide_gaussian_sites_match_autodiff_of_erf_kernel() -> None:
    positions, multipoles = _two_site_case()
    width = 0.8
    scale = np.sqrt(2.0 * (width**2 + width**2))

    def potential(distance: Any) -> Any:
        return jax.scipy.special.erf(distance / scale) / distance

    distance = jnp.asarray(2.0, dtype=jnp.float64)
    field = -float(jax.grad(potential)(distance))
    coupling = float(jax.grad(jax.grad(potential))(distance))
    result = (
        _plan(site_kind="gaussian", damping="none", tolerance=1.0e-12)
        .prepare(
            multipoles,
            widths=GaussianMultipoleWidths(
                jnp.asarray([width]), np.asarray([0, 0], dtype=np.int64)
            ),
        )
        .solve(positions)
    )
    assert bool(result.successful)
    np.testing.assert_allclose(
        result.state.induced_dipoles,
        _two_site_reference(coupling, field),
        rtol=1.0e-9,
        atol=1.0e-12,
    )


def test_gaussian_sites_refuse_exponential_cubic_damping() -> None:
    with pytest.raises(ValueError, match="damping='none'"):
        PolarizationPlan(
            coulomb_constant=1.0, site_kind="gaussian", damping="exponential-cubic"
        )


# --- 10. typed polarizability inputs ---------------------------------------------


def _dimensionless_port() -> ValuePort:
    return ValuePort(
        "test.atom-polarizability",
        event_shape=(),
        component_ids=("polarizability",),
        representation="atom-polarizability",
        dimensions=(DIMENSIONLESS,),
        normalization_id="identity",
    )


def test_potential_refuses_reference_ratio_and_unitless_polarizability() -> None:
    _, multipoles = _two_site_case()
    ratio = AtomisticFieldReference(
        "polarizability-model",
        AtomisticObservable.ATOMIC_POLARIZABILITY,
        _dimensionless_port(),
        level=AtomisticConditionLevel.ATOM,
    )
    with pytest.raises(ValueError, match="ATOMIC_POLARIZABILITY"):
        InducedDipolePolarizationPotential(
            _plan(), multipoles, polarizability_field=ratio
        )
    unitless = AtomisticFieldReference(
        "polarizability-model",
        AtomisticObservable.POLARIZABILITY,
        _dimensionless_port(),
        level=AtomisticConditionLevel.ATOM,
    )
    with pytest.raises(ValueError, match="explicit physical unit"):
        InducedDipolePolarizationPotential(
            _plan(), multipoles, polarizability_field=unitless
        )


# --- 11. bounded resources -------------------------------------------------------


def test_finite_route_workspace_budget_is_enforced_at_preparation() -> None:
    _, multipoles = _two_site_case()
    plan = _plan(route=FiniteMultipoleRoute(maximum_workspace_bytes=1))
    with pytest.raises(ValueError, match="maximum_workspace_bytes"):
        plan.prepare(multipoles)


# --- advanced non-polarization terms ----------------------------------------------


def _advanced_terms() -> Any:
    positions = jnp.asarray(
        [
            [1.0, 0.0, 0.2],
            [0.0, 0.0, 0.0],
            [0.0, 1.1, 0.0],
            [-0.4, 0.2, 1.1],
        ]
    )
    count = positions.shape[0]
    empty_angles = np.empty((0, 3), dtype=np.int32)
    terms = (
        Buffered147Potential(np.full(count, 0.55), np.full(count, 0.2)),
        ChargePenetrationPotential(
            np.asarray([0.5, 0.4, 0.3, 0.2], dtype=np.float64),
            np.asarray([-0.4, -0.3, -0.2, -0.1], dtype=np.float64),
            np.full(count, 1.6),
        ),
        ChargeTransferPotential(np.full(count, 0.3), np.full(count, 1.7)),
        ChargeFluxPotential(
            np.asarray([0.2, -0.1, -0.1, 0.0], dtype=np.float64),
            np.asarray([[0, 1]], dtype=np.int64),
            np.asarray([0.03], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            empty_angles,
            np.empty((0, 2)),
            np.empty((0,)),
        ),
        DampedDispersionPotential(
            np.full(count, 0.12),
            np.full(count, 1.8),
            c8=np.full(count, 0.04),
            c10=np.full(count, 0.01),
        ),
        PauliRepulsionPotential(np.full(count, 1.2), np.full(count, 2.0)),
        StretchBendPotential(
            count,
            np.asarray([[0, 1, 2]], dtype=np.int64),
            np.asarray([[0.2, -0.1]], dtype=np.float64),
            np.asarray([[1.0, 1.1]], dtype=np.float64),
            np.asarray([1.4], dtype=np.float64),
        ),
        AngleAnglePotential(
            count,
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            np.asarray([0.15], dtype=np.float64),
            np.asarray([[1.4, 1.2]], dtype=np.float64),
        ),
        OutOfPlaneBendPotential(
            count,
            np.asarray([[0, 1, 2, 3]], dtype=np.int64),
            np.asarray([0.4], dtype=np.float64),
        ),
    )
    return positions, terms


def test_advanced_terms_are_invariant_and_forces_match_finite_differences() -> None:
    for term_index in range(9):
        positions, terms = _advanced_terms()
        term = terms[term_index]
        evaluation = evaluate_polarizable_term(term, positions)
        translated = term.energy(positions + jnp.asarray([2.0, -1.0, 0.5]))
        rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        rotated = term.energy(positions @ rotation.T)
        assert bool(evaluation.successful)
        np.testing.assert_allclose(
            translated, evaluation.energy, rtol=2.0e-5, atol=2.0e-6
        )
        np.testing.assert_allclose(rotated, evaluation.energy, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=3.0e-5)

        direction = jnp.zeros_like(positions).at[0, 0].set(1.0)
        step = 2.0e-3
        finite_difference = (
            term.energy(positions + step * direction)
            - term.energy(positions - step * direction)
        ) / (2.0 * step)
        analytic = -evaluation.forces[0, 0]
        np.testing.assert_allclose(analytic, finite_difference, rtol=5.0e-3, atol=5.0e-5)


def test_damped_dispersion_is_finite_at_close_contact() -> None:
    term = DampedDispersionPotential(
        np.asarray([0.2, 0.3], dtype=np.float64), np.asarray([2.0, 2.4], dtype=np.float64)
    )
    positions = jnp.asarray([[0.0, 0.0, 0.0], [1.0e-4, 0.0, 0.0]])
    evaluation = evaluate_polarizable_term(term, positions)
    assert bool(evaluation.successful)
    assert bool(jnp.isfinite(evaluation.energy))
    assert bool(jnp.all(jnp.isfinite(evaluation.forces)))


def test_charge_flux_conserves_total_charge_and_force_field_is_coherent() -> None:
    positions, terms = _advanced_terms()
    charge_flux = terms[3]
    charges = charge_flux.charges(positions)
    np.testing.assert_allclose(
        jnp.sum(charges), jnp.sum(charge_flux.reference_charges), atol=1.0e-7
    )
    multipoles = _multipoles([0.2, -0.1, -0.1, 0.0], np.full(4, 0.03))
    polarization = PolarizationPlan(
        coulomb_constant=1.0,
        maximum_iterations=24,
        tolerance=1.0e-6,
        force_tolerance=2.0e-6,
    )
    force_field = PolarizableForceFieldPlan(
        terms, polarization=polarization, force_balance_tolerance=2.0e-4
    ).prepare(multipoles=multipoles)
    evaluation = force_field.evaluate(positions)
    assert evaluation.term_energies.shape == (9,)
    assert bool(evaluation.successful)
    assert bool(evaluation.qualification.polarization_force_valid)
    np.testing.assert_allclose(jnp.sum(evaluation.forces, axis=0), 0.0, atol=2.0e-4)
