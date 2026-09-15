import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import functional_rg as frg


def test_regulator_threshold_and_truncation_identities():
    regulator = frg.Regulator.optimized()
    quadrature = frg.ThresholdQuadraturePlan(3.0, quadrature_order=16, momentum_upper=8.0)
    masses = jnp.asarray([0.0, 0.5, 2.0])
    eta = jnp.asarray(0.1)
    threshold = jax.jit(quadrature.evaluate)(regulator, masses, eta)
    expected = (2.0 / 3.0) * (1.0 - eta / 5.0) / (1.0 + masses)
    np.testing.assert_allclose(threshold.value, expected, rtol=1.0e-13, atol=1.0e-13)
    assert bool(jnp.all(threshold.admissible))
    np.testing.assert_allclose(threshold.quadrature_error, 0.0, atol=0.0)
    power_law = quadrature.evaluate(frg.Regulator.power_law(2.0), masses)
    assert bool(jnp.all(power_law.admissible))
    assert bool(jnp.all(power_law.value > 0.0))

    nodes = jnp.linspace(0.0, 2.0, 17)
    prepared = frg.ONLocalPotentialPlan(
        4, 3.0, regulator, quadrature, approximation="lpa-prime"
    ).prepare(nodes)
    state = frg.ONPotentialState(-0.1 * nodes + 0.5 * nodes**2)
    assert prepared.evaluate(state).anomalous_dimension > 0.0
    identity = prepared.truncation_identity(state)
    assert bool(identity.satisfied)
    np.testing.assert_allclose(identity.origin_mass_splitting, 0.0, atol=1.0e-13)
    np.testing.assert_allclose(identity.component_trace_residual, 0.0, atol=1.0e-13)

    matsubara = frg.MatsubaraThresholdPlan(8)
    zero_temperature = matsubara.evaluate(0.0, 0.2, 0.4)
    np.testing.assert_allclose(zero_temperature.boson, 1.2**-1.5)
    np.testing.assert_allclose(zero_temperature.fermion, 1.4**-1.5)
    yukawa = frg.GrossNeveuYukawaFlowPlan(2, 1, 3.0, matsubara, representation="yukawa")
    fermion_state = frg.FermionBosonTruncationState(0.1, 0.5, 0.2, 0.0)
    truncation = yukawa.truncation_identity(fermion_state, 0.2)
    assert bool(truncation.satisfied)
    np.testing.assert_allclose(truncation.inactive_beta_residual, 0.0, atol=0.0)

    gross_neveu = frg.GrossNeveuYukawaFlowPlan(
        2, 1, 3.0, matsubara, representation="gross-neveu"
    )
    gross_state = frg.FermionBosonTruncationState(0.0, 0.0, 0.0, 0.3)
    gross_flow = gross_neveu.evaluate(gross_state, 0.2)
    np.testing.assert_allclose(gross_flow.beta[:3], 0.0, atol=0.0)
    assert gross_flow.beta[3] != 0.0
    assert bool(gross_neveu.truncation_identity(gross_state, 0.2).satisfied)


def test_wilson_fisher_like_fixed_point_and_critical_exponents():
    regulator = frg.Regulator.optimized()
    quadrature = frg.ThresholdQuadraturePlan(3.0, quadrature_order=16, momentum_upper=8.0)
    flow = frg.PolynomialONFlowPlan(1, 3.0, regulator, quadrature, coupling_count=2)
    search = frg.FixedPointSearchPlan(
        2,
        maximum_iterations=40,
        absolute_tolerance=1.0e-9,
        relative_tolerance=1.0e-8,
    )
    fixed_point = jax.jit(search.search)(flow, jnp.asarray([-0.1, 5.0]))
    assert bool(fixed_point.converged)
    assert fixed_point.residual_norm < 1.0e-8
    assert fixed_point.couplings[0] < 0.0
    assert fixed_point.couplings[1] > 1.0
    assert fixed_point.critical_exponents[0] > 0.0
    assert fixed_point.critical_exponents[1] < 0.0


def test_scheme_refinement_and_vertex_grid_evidence():
    regulator = frg.Regulator.exponential()
    coarse_rule = frg.ThresholdQuadraturePlan(
        3.0, quadrature_order=8, momentum_upper=12.0
    )
    refined_rule = frg.ThresholdQuadraturePlan(
        3.0, quadrature_order=32, momentum_upper=12.0
    )
    nodes = jnp.linspace(0.0, 2.0, 17)
    state = frg.ONPotentialState(-0.1 * nodes + 0.5 * nodes**2)
    coarse = frg.ONLocalPotentialPlan(2, 3.0, regulator, coarse_rule).prepare(nodes)
    refined = frg.ONLocalPotentialPlan(2, 3.0, regulator, refined_rule).prepare(nodes)
    evidence = frg.evaluate_scheme_refinement(
        coarse,
        refined,
        state,
        absolute_tolerance=1.0e-3,
        relative_tolerance=1.0e-3,
    )
    assert bool(evidence.accepted)
    assert evidence.refined_indicator < evidence.coarse_indicator
    assert evidence.absolute_difference < 1.0e-3

    momentum = jnp.linspace(0.0, 6.0, 12)
    vertex = frg.MomentumVertexGridFlowPlan(
        momentum,
        frg.Regulator.optimized(),
        frg.ThresholdQuadraturePlan(3.0, quadrature_order=16, momentum_upper=8.0),
        angular_order=8,
    ).prepare()
    vertex_state = frg.MomentumVertexState(1.0 + momentum**2, 0.5 + 0.02 * momentum**2)
    evaluation = jax.jit(vertex.evaluate)(vertex_state, jnp.asarray(0.0))
    assert bool(evaluation.admissible)
    assert evaluation.beta_four_point_vertex.shape == momentum.shape
    assert jnp.ptp(evaluation.beta_four_point_vertex) > 0.0


def test_fixed_resource_guards_fail_before_large_allocations():
    with pytest.raises(ValueError, match="node budget"):
        frg.ThresholdQuadraturePlan(3.0, quadrature_order=64, maximum_nodes=16)
    rule = frg.ThresholdQuadraturePlan(3.0, quadrature_order=8)
    with pytest.raises(ValueError, match="vertex-flow budget"):
        frg.MomentumVertexGridFlowPlan(
            jnp.linspace(0.0, 10.0, 128),
            frg.Regulator.optimized(),
            rule,
            angular_order=16,
            maximum_grid_entries=100,
        )
