import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _flow_plan(path=None):
    return phx.solver.MarkovCubaturePlan(
        phx.discretization.TemporalMesh.uniform(
            0.0,
            1.0,
            1,
            role="driver",
        ),
        phx.integration.GaussianCubatureRule(1, 3),
        method="stratonovich-flow",
        path=path,
        flow_substeps=2,
    )


def _stochastic_problem(*, interpretation="stratonovich", structure="additive"):
    return phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
        wiener_terms=(
            phx.solver.WienerTerm(
                "noise",
                lambda time, state, args: time * jnp.ones(state.shape + (1,)),
                (1,),
                structure=structure,
            ),
        ),
        interpretation=interpretation,
    )


def test_straight_gaussian_paths_are_degree_three_signature_certified():
    rule = phx.integration.GaussianCubatureRule(3, 3)
    path = phx.stochastic.straight_wiener_cubature_path(rule.prepared)
    replay = phx.stochastic.straight_wiener_cubature_path(rule.prepared)

    assert path.noise_dimension == 3
    assert path.path_count == rule.num_points
    assert path.segment_count == 1
    assert path.gaussian_degree == 3
    assert path.signature_degree == 3
    assert path.source_rule_id == rule.rule_id
    assert path.path_id == replay.path_id
    assert jnp.array_equal(path.increments[:, 0], rule.prepared.points)
    assert jnp.array_equal(path.weights, rule.prepared.weights)


def test_expected_brownian_signature_uses_noncommutative_generator():
    from phydrax.stochastic._cubature_path import _expected_wiener_signature_level

    fourth = _expected_wiener_signature_level(2, 4, np.dtype(float))
    sixth = _expected_wiener_signature_level(2, 6, np.dtype(float))

    assert fourth[1, 1, 1, 1] == pytest.approx(1.0 / 8.0)
    assert fourth[1, 1, 2, 2] == pytest.approx(1.0 / 8.0)
    assert fourth[2, 2, 1, 1] == pytest.approx(1.0 / 8.0)
    assert fourth[1, 2, 1, 2] == 0.0
    assert fourth[1, 2, 2, 1] == 0.0
    assert sixth[1, 1, 2, 2, 1, 1] == pytest.approx(1.0 / 48.0)
    assert sixth[1, 2, 1, 2, 1, 2] == 0.0


def test_straight_gaussian_paths_reject_false_higher_signature_certificate():
    rule = phx.integration.GaussianCubatureRule(2, 5)

    with pytest.raises(ValueError, match="fourth signature level"):
        phx.stochastic.straight_wiener_cubature_path(rule.prepared)


def test_positive_path_fitting_recertifies_without_degree_downgrade():
    rule = phx.integration.GaussianCubatureRule(1, 3)
    initial = phx.stochastic.straight_wiener_cubature_path(rule.prepared)
    fitted = phx.stochastic.fit_wiener_cubature_path(
        1,
        3,
        path_count=initial.path_count,
        segment_count=initial.segment_count,
        initial_data=initial,
        optimizer=lambda residual, vector: vector,
    )

    assert fitted.signature_degree == 3
    assert fitted.family == "fitted-positive-signature"
    assert jnp.max(fitted.signature_residuals) < 1.0e-12


def test_path_identity_is_invariant_to_input_path_order():
    rule = phx.integration.GaussianCubatureRule(1, 3)
    canonical = phx.stochastic.straight_wiener_cubature_path(rule.prepared)
    reversed_path = phx.stochastic.WienerCubaturePathData(
        canonical.increments[::-1],
        canonical.segment_widths,
        canonical.weights[::-1],
        gaussian_degree=3,
        signature_degree=3,
        family="straight-gaussian",
        source_rule_id=rule.rule_id,
    )

    assert canonical.path_id == reversed_path.path_id
    assert jnp.array_equal(canonical.increments, reversed_path.increments)
    assert jnp.array_equal(canonical.weights, reversed_path.weights)


def test_path_data_rejects_false_signature_certificates():
    with pytest.raises(ValueError, match="second signature level"):
        phx.stochastic.WienerCubaturePathData(
            jnp.asarray([[[-2.0]], [[2.0]]]),
            jnp.asarray([1.0]),
            jnp.asarray([0.5, 0.5]),
            gaussian_degree=3,
            signature_degree=3,
            family="invalid-variance",
            source_rule_id="invalid-variance",
        )

    with pytest.raises(ValueError, match="time-space signatures"):
        phx.stochastic.WienerCubaturePathData(
            jnp.asarray([[[1.0], [0.0]], [[0.0], [-1.0]]]),
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.5, 0.5]),
            gaussian_degree=3,
            signature_degree=3,
            family="invalid-timing",
            source_rule_id="invalid-timing",
        )

    positive = math.sqrt(2.0)
    negative = -1.0 / math.sqrt(2.0)
    with pytest.raises(ValueError, match="third signature level"):
        phx.stochastic.WienerCubaturePathData(
            jnp.asarray([[[negative]], [[positive]]]),
            jnp.asarray([1.0]),
            jnp.asarray([2.0 / 3.0, 1.0 / 3.0]),
            gaussian_degree=3,
            signature_degree=3,
            family="invalid-skew",
            source_rule_id="invalid-skew",
        )


def test_controlled_flow_consumes_every_path_segment_at_its_physical_time():
    rule = phx.integration.GaussianCubatureRule(1, 3)
    path = phx.stochastic.WienerCubaturePathData(
        jnp.asarray([[[-1.0], [0.0]], [[1.0], [0.0]]]),
        jnp.asarray([0.5, 0.5]),
        jnp.asarray([0.5, 0.5]),
        gaussian_degree=3,
        signature_degree=3,
        family="early-control",
        source_rule_id=rule.rule_id,
    )
    solution = jax.jit(
        lambda: phx.solver.solve_markov_cubature(
            _stochastic_problem(),
            _flow_plan(path),
        )
    )()
    mask = solution.mask[-1]
    values = jnp.sort(solution.points[-1, :, 0][mask])
    weights = jnp.exp(solution.log_weights[-1][mask])

    assert solution.successful
    assert jnp.allclose(values, jnp.asarray([-0.25, 0.25]), atol=1e-12)
    assert jnp.allclose(weights, jnp.asarray([0.5, 0.5]), atol=1e-12)


def test_controlled_flow_gates_interpretation_structure_and_path_degree():
    flow_plan = _flow_plan()
    with pytest.raises(ValueError, match="requires a Stratonovich problem"):
        phx.solver.solve_markov_cubature(
            _stochastic_problem(interpretation="ito"),
            flow_plan,
        )
    with pytest.raises(NotImplementedError, match="additive or commutative"):
        phx.solver.solve_markov_cubature(
            _stochastic_problem(structure="general"),
            flow_plan,
        )

    rule = phx.integration.GaussianCubatureRule(1, 3)
    degree_two = phx.stochastic.WienerCubaturePathData(
        rule.prepared.points[:, None, :],
        jnp.asarray([1.0]),
        rule.prepared.weights,
        gaussian_degree=3,
        signature_degree=2,
        family="degree-two",
        source_rule_id=rule.rule_id,
    )
    with pytest.raises(ValueError, match="signature degree at least three"):
        _flow_plan(degree_two)
