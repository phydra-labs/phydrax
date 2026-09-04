#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.equations._ksgs import (
    BuoyancyKSGSInputs,
    BuoyancyKSGSPlan,
    DynamicKSGSInputs,
    DynamicKSGSPlan,
    KSGSCoefficients,
    KSGSInputs,
    KSGSState,
    LowReKSGSCoefficients,
    LowReKSGSInputs,
    LowReKSGSPlan,
    replace_ksgs_kinetic_energy,
    StaticKSGSPlan,
)
from phydrax.equations._les_closures import (
    LESFilterScale,
    LESParameterProvenance,
    ResolvedLESFilter,
)


def _coefficients(*, eddy=1.0, limit=10.0):
    return KSGSCoefficients(eddy, 1.0, 3.0, 2.0, limit)


def _filter(
    name,
    *,
    family="explicit-filter",
    axis_names=("x", "y", "z"),
    topology="tensor-product",
    boundary_class="periodic",
    commutation_status="modeled",
    repeated_filter_semantics="composed",
):
    scale_rule = {
        "explicit-filter": "kernel-equivalent",
        "implicit-grid-volume": "volume-equivalent",
        "sharp-fourier-projection": "cutoff-equivalent",
    }[family]
    return ResolvedLESFilter(
        name,
        family=family,
        axis_names=axis_names,
        topology=topology,
        boundary_class=boundary_class,
        scale_rule=scale_rule,
        commutation_status=commutation_status,
        repeated_filter_semantics=repeated_filter_semantics,
    )


def _provenance():
    return LESParameterProvenance(
        _filter("explicit box filter"),
        "unit-mac-grid",
        "incompressible",
        source_kind="user",
        evidence_ids=(),
    )


def _base_inputs(gradient=None, *, width=1.0, viscosity=0.25, diffusion=0.0):
    if gradient is None:
        gradient = jnp.zeros((3, 3))
    return KSGSInputs(
        jnp.asarray(gradient),
        LESFilterScale(jnp.full((3,), width)),
        jnp.asarray(viscosity),
        jnp.asarray(diffusion),
    )


def _all_state_equal(left, right):
    return all(
        np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def test_static_zero_and_exact_equilibrium_limits():
    plan = StaticKSGSPlan(_coefficients(), _provenance())
    zero = plan.evaluate(plan.initialize_state(0.0), _base_inputs())
    assert zero.eddy_viscosity == 0.0
    assert zero.contributions.raw_production == 0.0
    assert zero.contributions.dissipation == 0.0
    assert zero.contributions.rhs == 0.0
    assert zero.diffusivity == 0.25
    assert bool(zero.evidence.kinetic_energy_nonnegative)
    assert bool(zero.evidence.finite)

    equilibrium_gradient = jnp.diag(jnp.asarray((0.5, -0.5, 0.0)))
    equilibrium = plan.evaluate(
        plan.initialize_state(1.0), _base_inputs(equilibrium_gradient)
    )
    np.testing.assert_allclose(equilibrium.contributions.production, 1.0)
    np.testing.assert_allclose(equilibrium.contributions.dissipation, 1.0)
    np.testing.assert_allclose(equilibrium.contributions.rhs, 0.0, atol=1.0e-7)
    assert not bool(equilibrium.evidence.production_limited)


def test_eddy_and_diffusion_coefficients_scale_without_changing_dissipation():
    gradient = jnp.diag(jnp.asarray((0.1, -0.1, 0.0)))
    inputs = _base_inputs(gradient)
    first_plan = StaticKSGSPlan(_coefficients(eddy=0.5), _provenance())
    first = first_plan.evaluate(first_plan.initialize_state(1.0), inputs)
    second_plan = StaticKSGSPlan(_coefficients(eddy=1.0), _provenance())
    second = second_plan.evaluate(second_plan.initialize_state(1.0), inputs)
    np.testing.assert_allclose(second.eddy_viscosity, 2.0 * first.eddy_viscosity)
    np.testing.assert_allclose(
        second.contributions.raw_production,
        2.0 * first.contributions.raw_production,
    )
    np.testing.assert_allclose(
        second.diffusivity - 0.25, 2.0 * (first.diffusivity - 0.25)
    )
    np.testing.assert_allclose(
        second.contributions.dissipation, first.contributions.dissipation
    )
    pre_operator = second_plan.transport(
        second.state,
        inputs.filter_scale,
        inputs.molecular_kinematic_viscosity,
    )
    np.testing.assert_allclose(pre_operator.eddy_viscosity, second.eddy_viscosity)
    np.testing.assert_allclose(pre_operator.diffusivity, second.diffusivity)


def test_production_dissipation_signs_and_explicit_production_limit():
    plan = StaticKSGSPlan(_coefficients(limit=2.0), _provenance())
    gradient = jnp.diag(jnp.asarray((10.0, -10.0, 0.0)))
    result = plan.evaluate(
        plan.initialize_state(1.0), _base_inputs(gradient, diffusion=-0.5)
    )
    assert result.contributions.raw_production > 0.0
    np.testing.assert_allclose(result.contributions.production, 2.0)
    np.testing.assert_allclose(result.contributions.dissipation, 1.0)
    assert result.contributions.production_limit_reduction > 0.0
    assert bool(result.evidence.production_limited)
    assert bool(result.evidence.production_nonnegative)
    assert bool(result.evidence.dissipation_nonnegative)
    np.testing.assert_allclose(result.contributions.rhs, 0.5)


def test_buoyancy_has_stable_sink_and_unstable_source_signs():
    plan = BuoyancyKSGSPlan(_coefficients(), _provenance())
    state = plan.initialize_state(1.0)
    base = _base_inputs()
    stable = plan.evaluate(state, BuoyancyKSGSInputs(base, 4.0))
    unstable = plan.evaluate(state, BuoyancyKSGSInputs(base, -4.0))
    np.testing.assert_allclose(stable.contributions.buoyancy, -8.0)
    np.testing.assert_allclose(unstable.contributions.buoyancy, 8.0)
    np.testing.assert_allclose(
        stable.contributions.buoyancy, -unstable.contributions.buoyancy
    )


def test_dynamic_update_history_acceptance_and_exact_restart_identity():
    plan = DynamicKSGSPlan(
        _coefficients(eddy=0.25),
        _provenance(),
        _filter("explicit test box filter"),
        2.0,
    )
    state = plan.initialize_state(1.0)
    identity = jnp.eye(3)
    first_inputs = DynamicKSGSInputs(_base_inputs(), 0.5 * identity, identity, 0.2, True)
    first = plan.evaluate(state, first_inputs)
    np.testing.assert_allclose(first.state.dynamic_numerator, 0.3)
    np.testing.assert_allclose(first.state.dynamic_denominator, 0.6)
    np.testing.assert_allclose(first.state.eddy_viscosity_coefficient, 0.5)
    assert first.state.dynamic_updates == 1
    assert bool(first.evidence.dynamic_update_accepted)

    paused = plan.evaluate(
        first.state,
        DynamicKSGSInputs(_base_inputs(), identity, identity, 1.0, False),
    )
    assert _all_state_equal(paused.state, first.state)

    restarted = KSGSState(
        jnp.array(first.state.kinetic_energy),
        jnp.array(first.state.dynamic_numerator),
        jnp.array(first.state.dynamic_denominator),
        jnp.array(first.state.eddy_viscosity_coefficient),
        jnp.array(first.state.dynamic_updates),
    )
    second_inputs = DynamicKSGSInputs(_base_inputs(), identity, identity, 0.2, True)
    uninterrupted = plan.evaluate(first.state, second_inputs)
    resumed = plan.evaluate(restarted, second_inputs)
    assert _all_state_equal(uninterrupted.state, resumed.state)
    np.testing.assert_allclose(
        uninterrupted.contributions.rhs, resumed.contributions.rhs, rtol=0.0, atol=0.0
    )


def test_dynamic_filter_semantics_are_compatible_and_non_aliasing():
    provenance = _provenance()
    coefficients = _coefficients()
    ratio = 2.0
    invalid_filters = (
        (provenance.resolved_filter, "differ"),
        (_filter("other axes", axis_names=("a", "b", "c")), "matching axes"),
        (_filter("other topology", topology="unstructured"), "topology"),
        (_filter("other boundary", boundary_class="wall-bounded"), "boundary"),
        (
            _filter("implicit", family="implicit-grid-volume"),
            "explicit or sharp",
        ),
        (
            _filter("unknown commute", commutation_status="unmodeled"),
            "commutation",
        ),
        (
            _filter("unknown repeat", repeated_filter_semantics="unmodeled"),
            "repeated",
        ),
    )
    for test_filter, message in invalid_filters:
        with pytest.raises(ValueError, match=message):
            DynamicKSGSPlan(coefficients, provenance, test_filter, ratio)

    commuting_provenance = LESParameterProvenance(
        _filter("commuting resolved", commutation_status="commuting"),
        "unit-mac-grid",
        "incompressible",
        source_kind="user",
        evidence_ids=(),
    )
    sharp_test_filter = _filter(
        "sharp test projection",
        family="sharp-fourier-projection",
        commutation_status="commuting",
        repeated_filter_semantics="idempotent",
    )
    plan = DynamicKSGSPlan(coefficients, commuting_provenance, sharp_test_filter, ratio)
    assert plan.test_filter.filter_id == sharp_test_filter.filter_id


def test_low_re_damping_and_viscous_dissipation_are_explicit():
    plan = LowReKSGSPlan(_coefficients(), LowReKSGSCoefficients(2.0, 2.0), _provenance())
    result = plan.evaluate(
        plan.initialize_state(1.0),
        LowReKSGSInputs(
            _base_inputs(viscosity=1.0),
            jnp.asarray(1.0),
            jnp.asarray((3.0, 4.0, 0.0)),
        ),
    )
    np.testing.assert_allclose(result.eddy_viscosity, 1.0 - np.exp(-2.0))
    np.testing.assert_allclose(result.contributions.low_re_dissipation, 50.0)
    np.testing.assert_allclose(result.contributions.rhs, -51.0)
    assert bool(result.evidence.dissipation_nonnegative)


def test_negative_kinetic_energy_is_refused_without_a_floor_eager_and_jit():
    plan = StaticKSGSPlan(_coefficients(), _provenance())
    inputs = _base_inputs()
    with pytest.raises(Exception, match="negative"):
        plan.evaluate(plan.initialize_state(-1.0), inputs)

    compiled = jax.jit(
        lambda value: (
            plan.evaluate(plan.initialize_state(value), inputs).contributions.rhs
        )
    )
    with pytest.raises(Exception, match="negative"):
        compiled(jnp.asarray(-1.0)).block_until_ready()


def test_static_transition_is_jittable_differentiable_and_fixed_shape():
    plan = StaticKSGSPlan(_coefficients(), _provenance())
    inputs = _base_inputs(jnp.diag(jnp.asarray((0.2, -0.2, 0.0))))

    def rhs(kinetic):
        return plan.evaluate(plan.initialize_state(kinetic), inputs).contributions.rhs

    primal, tangent = jax.jvp(jax.jit(rhs), (jnp.asarray(1.2),), (jnp.asarray(0.3),))
    assert jnp.isfinite(primal)
    assert jnp.isfinite(tangent)

    state = plan.initialize_state(jnp.asarray((1.0, 2.0)))
    field_inputs = KSGSInputs(
        jnp.zeros((2, 3, 3)),
        LESFilterScale(jnp.ones((2, 3))),
        jnp.asarray((0.1, 0.2)),
        jnp.asarray((0.0, 0.0)),
    )
    result = jax.jit(plan.evaluate)(state, field_inputs)
    assert result.contributions.rhs.shape == state.kinetic_energy.shape
    assert _all_state_equal(result.state, state)

    replaced = replace_ksgs_kinetic_energy(result.state, jnp.asarray((1.5, 2.5)))
    np.testing.assert_array_equal(replaced.kinetic_energy, jnp.asarray((1.5, 2.5)))
    assert _all_state_equal(
        KSGSState(
            result.state.kinetic_energy,
            replaced.dynamic_numerator,
            replaced.dynamic_denominator,
            replaced.eddy_viscosity_coefficient,
            replaced.dynamic_updates,
        ),
        result.state,
    )


def test_nonfinite_backend_term_is_reported_by_evidence():
    plan = StaticKSGSPlan(_coefficients(), _provenance())
    result = plan.evaluate(plan.initialize_state(1.0), _base_inputs(diffusion=jnp.nan))
    assert not bool(result.evidence.finite)
