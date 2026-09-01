#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.solid_mechanics._mixed_hyperelastic import (
    mixed_hyperelastic_form,
    mixed_pressure_first_piola,
    MixedAugmentedLagrangianPlan,
    MixedHyperelasticLaw,
    MixedHyperelasticModel,
    prepare_mixed_neural_stationarity,
    prepare_mixed_neural_virtual_work,
)
from phydrax.discretization.fem._mixed_constraint import PressureGaugePolicy
from phydrax.nn.parameters import ParameterSubspace
from phydrax.solver import PreparedFieldEquilibrium


def _isochoric_energy(deformation_bar):
    dimension = deformation_bar.shape[0]
    return 1.5 * (jnp.sum(deformation_bar * deformation_bar) - dimension)


def _log_volume(deformation):
    return jnp.log(jnp.linalg.det(deformation))


def _law(*, bulk_modulus=None):
    return MixedHyperelasticLaw(
        _isochoric_energy,
        _log_volume,
        bulk_modulus=bulk_modulus,
        minimum_jacobian=1.0e-8,
    )


def test_exact_and_finite_bulk_laws_have_the_declared_pressure_equations_and_blocks():
    deformation = jnp.asarray(((1.2, 0.1), (0.0, 0.9)))
    pressure = jnp.asarray(2.5)
    exact = _law()
    finite = _law(bulk_modulus=80.0)

    exact_response = exact.evaluate(deformation, pressure)
    finite_response = finite.evaluate(deformation, pressure)
    exact_tangent = exact.block_tangent(deformation, pressure)
    finite_tangent = finite.block_tangent(deformation, pressure)
    constraint_gradient = jax.grad(_log_volume)(deformation)

    np.testing.assert_allclose(
        exact_response.constraint_residual,
        _log_volume(deformation),
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        finite_response.constraint_residual,
        _log_volume(deformation) - pressure / 80.0,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        exact_response.pressure_first_piola,
        pressure * constraint_gradient,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        mixed_pressure_first_piola(deformation, pressure, _log_volume),
        pressure * constraint_gradient,
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        exact_tangent.deformation_pressure,
        exact_tangent.pressure_deformation,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(exact_tangent.pressure_pressure, 0.0, atol=0.0)
    np.testing.assert_allclose(
        finite_tangent.pressure_pressure,
        -1.0 / 80.0,
        rtol=1e-13,
        atol=1e-13,
    )
    assert exact.formulation == "exact"
    assert finite.formulation == "finite-bulk"
    assert bool(exact_response.evidence.valid)
    assert bool(finite_response.evidence.valid)


def test_isochoric_response_is_scale_invariant_and_invalid_j_is_explicit_evidence():
    law = _law()
    deformation = jnp.asarray(((1.1, 0.2), (0.1, 0.95)))

    np.testing.assert_allclose(
        law.isochoric_value(3.0 * deformation),
        law.isochoric_value(deformation),
        rtol=1e-12,
        atol=1e-12,
    )
    invalid = law.evaluate(jnp.asarray(((-1.0, 0.0), (0.0, 1.0))), 0.0)

    assert not bool(invalid.evidence.jacobian_valid)
    assert not bool(invalid.evidence.valid)


def test_mixed_form_declares_exact_and_finite_bulk_block_dependencies_without_penalty_aliasing():
    exact = mixed_hyperelastic_form("u", "p", MixedHyperelasticModel(_law()))
    finite = mixed_hyperelastic_form(
        "u", "p", MixedHyperelasticModel(_law(bulk_modulus=100.0))
    )

    assert exact.actions[0].input_fields == ("u", "p")
    assert exact.actions[1].input_fields == ("u",)
    assert finite.actions[0].input_fields == ("u", "p")
    assert finite.actions[1].input_fields == ("u", "p")


def _neural_root():
    functions = {
        "u": jnp.asarray((0.4, -0.2)),
        "p": jnp.asarray(0.3),
    }
    subspace = ParameterSubspace(functions, {"u": True, "p": True})
    return functions, subspace


def test_mixed_neural_stationarity_uses_field_equilibrium_and_refuses_implicit_gauge():
    functions, subspace = _neural_root()
    law = _law()
    gauge = PressureGaugePolicy("mean-zero")

    def action(fields, realization, args):
        del args
        return 0.5 * jnp.vdot(fields["u"], fields["u"]) + fields["p"] * (
            jnp.sum(fields["u"]) - realization
        )

    prepared = prepare_mixed_neural_stationarity(
        functions,
        action,
        subspace,
        law,
        gauge,
        gauge_enforced=True,
        realization=0.1,
        realization_id="mixed-stationarity-points",
        provenance_id="mixed-stationarity-realization",
    )
    state = prepared.initial_state
    residual = prepared.problem.residual(state)
    tangent = jax.jacfwd(prepared.problem.residual)(state)

    assert isinstance(prepared, PreparedFieldEquilibrium)
    assert residual.shape == state.shape
    assert jnp.count_nonzero(jnp.abs(tangent) > 0.0) > state.size
    with pytest.raises(ValueError, match="explicitly enforced gauge"):
        prepare_mixed_neural_stationarity(
            functions,
            action,
            subspace,
            law,
            gauge,
            gauge_enforced=False,
            realization=0.1,
            realization_id="mixed-stationarity-points",
            provenance_id="mixed-stationarity-realization",
        )
    frozen_pressure = ParameterSubspace(functions, {"u": True, "p": False})
    with pytest.raises(ValueError, match="must select field 'p'"):
        prepare_mixed_neural_stationarity(
            functions,
            action,
            frozen_pressure,
            law,
            gauge,
            gauge_enforced=True,
            realization=0.1,
            realization_id="mixed-stationarity-points",
            provenance_id="mixed-stationarity-realization",
        )


def test_mixed_neural_virtual_work_retains_u_p_pullback_and_finite_bulk_refuses_gauge():
    functions, subspace = _neural_root()
    exact = _law()

    def field_jet(fields, realization, args):
        del realization, args
        return {"u": 2.0 * fields["u"], "p": fields["p"]}

    def virtual_work(fields, jets, realization, args):
        del fields, realization, args
        return {"u": jets["u"] + jets["p"], "p": jnp.sum(jets["u"])}

    prepared = prepare_mixed_neural_virtual_work(
        functions,
        field_jet,
        virtual_work,
        subspace,
        None,
        exact,
        PressureGaugePolicy("pinned"),
        gauge_enforced=True,
        realization_id="mixed-virtual-work-points",
        provenance_id="mixed-virtual-work-realization",
    )

    assert isinstance(prepared, PreparedFieldEquilibrium)
    assert prepared.problem.residual(prepared.initial_state).shape == (
        subspace.total_dimension,
    )
    with pytest.raises(ValueError, match="must not impose"):
        prepare_mixed_neural_virtual_work(
            functions,
            field_jet,
            virtual_work,
            subspace,
            None,
            _law(bulk_modulus=50.0),
            PressureGaugePolicy("mean-zero"),
            gauge_enforced=True,
            realization_id="mixed-virtual-work-points",
            provenance_id="mixed-virtual-work-realization",
        )


def test_augmented_lagrangian_updates_multiplier_and_rolls_back_every_outer_field():
    law = MixedHyperelasticLaw(
        _isochoric_energy,
        lambda deformation: jnp.linalg.det(deformation) - 1.0,
        minimum_jacobian=1.0e-8,
    )
    plan = MixedAugmentedLagrangianPlan(
        law,
        initial_penalty=10.0,
        penalty_growth=4.0,
        constraint_reduction=0.25,
    )
    initial = plan.initialize(jnp.asarray(((1.2, 0.0), (0.0, 1.0))))
    inner = plan.inner_response(initial, jnp.asarray(((1.1, 0.0), (0.0, 1.0))))
    np.testing.assert_allclose(inner.pressure, 1.0, atol=1e-12)
    accepted = plan.advance(
        initial,
        jnp.asarray(((1.04, 0.0), (0.0, 1.0))),
        inner_successful=True,
    )

    np.testing.assert_allclose(accepted.accepted_state.pressure, 0.4, atol=1e-12)
    assert bool(accepted.evidence.accepted)
    assert bool(accepted.evidence.constraint_reduced)
    assert not bool(accepted.evidence.penalty_increased)

    rejected = plan.advance(
        accepted.accepted_state,
        jnp.asarray(((-1.0, 0.0), (0.0, 1.0))),
        inner_successful=True,
    )

    assert not bool(rejected.evidence.accepted)
    assert bool(rejected.evidence.rollback_applied)
    np.testing.assert_array_equal(
        rejected.accepted_state.deformation_gradient,
        accepted.accepted_state.deformation_gradient,
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.pressure,
        accepted.accepted_state.pressure,
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.penalty,
        accepted.accepted_state.penalty,
    )
    np.testing.assert_array_equal(
        rejected.accepted_state.outer_iteration,
        accepted.accepted_state.outer_iteration,
    )
