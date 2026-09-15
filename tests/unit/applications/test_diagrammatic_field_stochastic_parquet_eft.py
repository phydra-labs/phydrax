#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import math

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications import diagrammatic_field as df


def _catalogue_diagram(order):
    phi = df.FieldSpec("catalogue-phi", statistics="boson")
    propagator = df.PropagatorSpec(phi, mass=1.0)
    rule = df.VertexRule(
        f"catalogue-phi4-{order}",
        (phi, phi, phi, phi),
        1.0,
        perturbative_order=order,
    )
    route = df.MomentumRoute([0.75], 0.0)
    return df.DiagramGraph(
        (df.VertexInsertion("v", rule),),
        (df.PropagatorLine("p", propagator, "v", "v", df.MomentumRoute([0.0])),),
        (
            df.ExternalLeg("in", phi, "v", route, incoming=True),
            df.ExternalLeg("out", phi, "v", route, incoming=False),
        ),
    )


def _eft_basis():
    phi = df.FieldSpec("eft-phi", statistics="boson")
    return df.EFTOperatorBasis((df.EFTOperator("phi2", (phi, phi), 2.0),))


def test_diagram_monte_carlo_satisfies_birth_death_and_worm_detailed_balance():
    diagrams = (_catalogue_diagram(1), _catalogue_diagram(2))
    prepared = df.DiagramMonteCarloPlan(
        steps=2_000,
        maximum_diagrams=2,
        maximum_neighbors=1,
        maximum_order=2,
    ).prepare(diagrams, jnp.asarray([1.0 + 0.0j, -0.5 + 0.25j]))
    result = prepared.run(jr.key(41))

    np.testing.assert_allclose(prepared.detailed_balance_residual, 0.0, atol=1e-15)
    assert result.evidence.successful
    assert result.state.accepted_moves[0] > 0
    assert result.state.accepted_moves[1] > 0
    assert result.state.accepted_moves[2] > 0
    assert result.state.order_histogram[1] > 0
    assert result.state.order_histogram[2] > 0
    np.testing.assert_allclose(jnp.sum(result.order_probabilities), 1.0)
    assert 0.0 <= result.evidence.average_sign <= 1.0
    assert result.evidence.phase_standard_error >= 0.0


def test_finite_parquet_iteration_reaches_the_analytic_scalar_fixed_point():
    bare = jnp.asarray([[1.0]])
    bubbles = jnp.full((3, 1, 1), 0.1)
    result = (
        df.ParquetIterationPlan(
            maximum_iterations=512,
            tolerance=1e-12,
            damping=0.8,
        )
        .prepare(bare, bubbles)
        .iterate()
    )

    expected = 1.0 / (1.0 - 3.0 * 0.1**2)
    assert result.evidence.converged
    assert result.evidence.successful
    assert result.evidence.fixed_point_residual <= 1e-12
    np.testing.assert_allclose(result.full_vertex, [[expected]], rtol=1e-10)


def test_native_matching_and_rg_running_recover_known_one_operator_flow():
    basis = _eft_basis()
    matched = (
        df.EFTMatchingPlan(basis, jnp.asarray([[1.0]]))
        .prepare()
        .match(
            jnp.asarray([2.0]),
            scale=1.0,
            statistical_covariance=jnp.asarray([[0.04]]),
        )
    )
    running = (
        df.RGFlowPlan(basis, jnp.asarray([[0.2]]))
        .prepare(
            1.0,
            math.e,
            steps=64,
        )
        .run(matched.state)
    )

    assert matched.evidence.successful
    assert running.evidence.successful
    np.testing.assert_allclose(
        running.state.coefficients,
        [2.0 * math.exp(0.2)],
        rtol=1e-9,
    )
    np.testing.assert_allclose(
        running.state.statistical_covariance,
        [[0.04 * math.exp(0.4)]],
        rtol=1e-9,
    )


def test_eft_truncation_and_statistical_uncertainties_remain_independent():
    basis = _eft_basis()
    state = df.WilsonCoefficientState(
        jnp.asarray([-2.0 + 0.0j]),
        jnp.asarray([[0.04]]),
        1.0,
        True,
        basis.basis_id,
    )
    low_q_plan = df.EFTObservablePlan(
        basis,
        jnp.asarray([1.0]),
        df.PowerCountingRule(0.25, 1.0),
        first_omitted_power=2,
        omitted_coefficient_scale=2.0,
    ).prepare()
    high_q_plan = df.EFTObservablePlan(
        basis,
        jnp.asarray([1.0]),
        df.PowerCountingRule(0.5, 1.0),
        first_omitted_power=2,
        omitted_coefficient_scale=2.0,
    ).prepare()
    changed_statistics = df.WilsonCoefficientState(
        state.coefficients,
        jnp.asarray([[0.09]]),
        state.scale,
        True,
        basis.basis_id,
    )

    baseline = low_q_plan.evaluate(state)
    power_changed = high_q_plan.evaluate(state)
    statistics_changed = low_q_plan.evaluate(changed_statistics)

    np.testing.assert_allclose(baseline.value, -2.0)
    np.testing.assert_allclose(baseline.evidence.phase, -1.0 + 0.0j)
    assert baseline.evidence.real_sign == -1
    np.testing.assert_allclose(baseline.statistical_uncertainty, 0.2)
    np.testing.assert_allclose(baseline.truncation_uncertainty, 0.125)
    np.testing.assert_allclose(
        baseline.statistical_uncertainty,
        power_changed.statistical_uncertainty,
    )
    assert power_changed.truncation_uncertainty > baseline.truncation_uncertainty
    np.testing.assert_allclose(
        baseline.truncation_uncertainty,
        statistics_changed.truncation_uncertainty,
    )
    assert statistics_changed.statistical_uncertainty > baseline.statistical_uncertainty
