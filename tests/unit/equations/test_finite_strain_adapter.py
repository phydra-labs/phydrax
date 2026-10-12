#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


pytestmark = pytest.mark.strict_jax


def _parameters() -> phx.applications.solid_mechanics.FiniteStrainJ2Parameters:
    return phx.applications.solid_mechanics.FiniteStrainJ2Parameters(20.0, 80.0, 0.5, 2.0)


@pytest.mark.parametrize(
    "stretch", [1.0, 1.002, 1.15], ids=["rest", "elastic", "plastic"]
)
def test_adapter_matches_direct_j2(stretch: float) -> None:
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    adapter = phx.applications.solid_mechanics.FiniteStrainConstitutiveModel(
        material, 2.0
    )
    state = material.initialize_state((), jnp.float64)
    deformation = jnp.diag(jnp.asarray((stretch, 1.0, 0.999), dtype=jnp.float64))
    direct = material.evaluate_linearized(
        deformation, state, 2.0, _parameters(), 0.0, 0.1
    )
    adapted = adapter.evaluate(deformation, state, _parameters(), 0.0, 0.1)
    assert adapter.state_shape == (7,)
    assert adapted.trial_state.shape == (7,)
    np.testing.assert_allclose(adapted.response, direct.response.first_piola, atol=1e-12)
    np.testing.assert_allclose(
        adapted.trial_state, direct.response.trial_state, atol=1e-12
    )
    np.testing.assert_allclose(adapted.energy, direct.response.reference_energy_density)
    np.testing.assert_allclose(adapted.dissipation, direct.response.dissipation_increment)
    if adapted.consistent_tangent is None:
        raise RuntimeError("The finite-strain adapter must provide its tangent.")
    np.testing.assert_allclose(adapted.consistent_tangent, direct.algorithmic_tangent)
    np.testing.assert_array_equal(
        adapted.diagnostics["branch_code"], direct.response.branch_code
    )
    assert bool(adapted.valid)
    assert bool(adapted.diagnostics["tangent_successful"])
    assert material.slot_semantic_id == "phydrax.equations.implicit-mpm-constitutive-plan"
    assert adapter.slot_semantic_id == "phydrax.equations.constitutive-model"


def test_adapter_batched_jit_and_directional_tangent() -> None:
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    density = jnp.asarray(((1.0, 2.0), (3.0, 4.0)), dtype=jnp.float64)
    adapter = phx.applications.solid_mechanics.FiniteStrainConstitutiveModel(
        material, density
    )
    state = material.initialize_state((2, 2), jnp.float64)
    deformation = jnp.broadcast_to(
        jnp.diag(jnp.asarray((1.15, 0.97, 0.93))), (2, 2, 3, 3)
    )
    response = eqx.filter_jit(adapter.evaluate)(
        deformation, state, _parameters(), 0.0, 0.1
    )
    assert response.response.shape == (2, 2, 3, 3)
    assert response.consistent_tangent is not None
    assert response.consistent_tangent.shape == (2, 2, 3, 3, 3, 3)
    assert bool(jnp.all(response.valid))
    direction = jnp.asarray(((0.3, 0.2, -0.1), (0.2, -0.1, 0.1), (-0.1, 0.1, 0.2)))
    epsilon = 1e-6
    plus = material.evaluate(
        deformation + epsilon * jnp.broadcast_to(direction, deformation.shape),
        state,
        density,
        _parameters(),
        0.0,
        0.1,
    )
    minus = material.evaluate(
        deformation - epsilon * jnp.broadcast_to(direction, deformation.shape),
        state,
        density,
        _parameters(),
        0.0,
        0.1,
    )
    predicted = phx.ein.contract(
        "...ijkl,kl->...ij", response.consistent_tangent, direction
    )
    np.testing.assert_allclose(
        predicted,
        (plus.first_piola - minus.first_piola) / (2 * epsilon),
        rtol=2e-6,
        atol=2e-7,
    )


def test_adapter_material_transaction_commit_and_rollback() -> None:
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    adapter = phx.applications.solid_mechanics.FiniteStrainConstitutiveModel(
        material, 2.0, model_id="j2-fem"
    )
    site = phx.equations.MaterialSiteId("gauss-point")
    plan = phx.equations.MaterialIntegrationPlan(((site, adapter),))
    initial = material.initialize_state((), jnp.float64)
    transaction = plan.initialize((initial,))
    deformation = jnp.diag(jnp.asarray((1.2, 0.98, 0.95), dtype=jnp.float64))
    response = plan.evaluate(site, deformation, transaction, _parameters(), 0.0, 0.1)
    assert bool(response.valid)
    candidate = transaction.with_trials({site: response.trial_state})
    np.testing.assert_array_equal(candidate.state(site).committed, initial)
    np.testing.assert_array_equal(candidate.rollback().state(site).trial, initial)
    committed = candidate.commit()
    np.testing.assert_array_equal(committed.state(site).committed, response.trial_state)
    assert committed.state(site).state_version == 1
    direct = material.evaluate(
        deformation, response.trial_state, 2.0, _parameters(), 0.1, 0.1
    )
    continued = plan.evaluate(site, deformation, committed, _parameters(), 0.1, 0.1)
    np.testing.assert_allclose(continued.response, direct.first_piola, atol=1e-11)


def test_adapter_failed_material_retains_committed_state() -> None:
    material = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    adapter = phx.applications.solid_mechanics.FiniteStrainConstitutiveModel(
        material, 2.0
    )
    initial = material.initialize_state((), jnp.float64)
    response = adapter.evaluate(
        jnp.diag(jnp.asarray((-1.0, 1.0, 1.0))), initial, _parameters(), 0.0, 0.1
    )
    assert not bool(response.valid)
    np.testing.assert_array_equal(response.trial_state, initial)


def test_adapter_exposes_split_tangent_failure() -> None:
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(3)
    adapter = phx.applications.solid_mechanics.FiniteStrainConstitutiveModel(
        material, 1.0
    )
    parameters = phx.applications.solid_mechanics.MPMPhaseFieldParameters(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0),
        1.0,
        0.1,
    )
    response = adapter.evaluate(
        jnp.eye(3, dtype=jnp.float64), jnp.asarray((0.2, 0.0)), parameters, 0.0, 0.1
    )
    assert bool(response.diagnostics["material_successful"])
    assert not bool(response.diagnostics["tangent_successful"])
    assert not bool(response.valid)
    assert bool(jnp.all(jnp.isfinite(response.response)))


@pytest.mark.parametrize("density", [0.0, -1.0, np.inf, np.nan])
def test_adapter_rejects_invalid_reference_density(density: float) -> None:
    with pytest.raises(ValueError, match="density"):
        phx.applications.solid_mechanics.FiniteStrainConstitutiveModel(
            phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan(), density
        )


def test_adapter_rejects_non_three_dimensional_material() -> None:
    with pytest.raises(ValueError, match="3-D"):
        phx.applications.solid_mechanics.FiniteStrainConstitutiveModel(
            phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(2),
            1.0,
        )
