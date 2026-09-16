#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.operators.quantum._electronic_transport import (
    elastic_transmission,
    ElasticDisorderEnsemblePlan,
    electronic_contact_current_kernels,
    ELECTRONIC_TRANSPORT_CONVENTION,
    evaluate_elastic_disorder_ensemble,
    FermionicKeldyshTransportState,
    retarded_embedding,
    retarded_open_system_point,
)


def test_nonorthogonal_retarded_embedding_is_causal_and_energy_shift_invariant():
    energy = 0.4 + 0.2j
    lead_onsite = -0.3
    coupling_h = jnp.asarray([[0.7 + 0.1j]])
    coupling_s = jnp.asarray([[0.15 - 0.02j]])
    surface = jnp.asarray([[1.0 / (energy - lead_onsite)]])
    original = retarded_embedding(energy, surface, coupling_h, coupling_s)

    shift = 2.5
    shifted = retarded_embedding(
        energy + shift,
        jnp.asarray([[1.0 / ((energy + shift) - (lead_onsite + shift))]]),
        coupling_h + shift * coupling_s,
        coupling_s,
    )

    assert bool(original.evidence.causal)
    assert float(original.evidence.broadening_minimum_eigenvalue) >= -1.0e-12
    assert jnp.allclose(shifted.self_energy, original.self_energy)
    assert jnp.allclose(shifted.broadening, original.broadening)


def test_contact_currents_use_positive_into_region_and_signed_electron_charge():
    energy = jnp.asarray(0.0 + 0.0j)
    surface = jnp.asarray([[-1.0j]])
    coupling = jnp.asarray([[1.0]])
    left = retarded_embedding(energy, surface, coupling)
    right = retarded_embedding(energy, surface, coupling)
    point = retarded_open_system_point(
        energy,
        jnp.asarray([[0.0]]),
        jnp.asarray([[1.0]]),
        (left, right),
    )
    currents = electronic_contact_current_kernels(
        point,
        jnp.asarray([1.0, 0.25]),
        jnp.asarray([-1.0, 1.0]),
    )

    assert bool(point.successful)
    assert elastic_transmission(
        point.green, left.broadening, right.broadening
    ) == pytest.approx(1.0)
    assert currents.particle_current_into_region[0] == pytest.approx(0.75)
    assert currents.particle_current_into_region[1] == pytest.approx(-0.75)
    assert currents.particle_continuity_residual == pytest.approx(0.0)
    assert currents.charge_current_into_region[0] < 0.0
    assert currents.charge_current_into_region[1] > 0.0
    assert currents.heat_current_into_region[0] > 0.0
    assert currents.heat_current_into_region[1] > 0.0
    assert ELECTRONIC_TRANSPORT_CONVENTION.bloch_phase_sign == 1
    assert ELECTRONIC_TRANSPORT_CONVENTION.retarded_time_sign == -1


def test_elastic_disorder_ensemble_retains_failed_realizations():
    plan = ElasticDisorderEnsemblePlan(
        ("clean", "positive", "negative"), jnp.asarray([0.5, 0.25, 0.25])
    )
    result = evaluate_elastic_disorder_ensemble(
        plan,
        jnp.asarray([[1.0], [3.0], [-1.0]]),
        jnp.asarray([True, False, True]),
    )

    assert result.mean[0] == pytest.approx(1.0)
    assert result.realization_values[:, 0].tolist() == pytest.approx([1.0, 3.0, -1.0])
    assert result.realization_successful.tolist() == [True, False, True]
    assert not bool(result.successful)
    assert result.effective_sample_size == pytest.approx(8.0 / 3.0)


def test_fermionic_adapter_state_checks_car_causality_adjoint_and_conservation():
    retarded = jnp.asarray([[[[0.0j]], [[0.0j]]], [[[1.0j]], [[0.0j]]]])
    advanced = jnp.swapaxes(jnp.swapaxes(retarded.conj(), -1, -2), 0, 1)
    spectral = retarded - advanced
    state = FermionicKeldyshTransportState(
        jnp.zeros_like(spectral),
        spectral,
        retarded,
        advanced,
        jnp.asarray([0.0, 1.0]),
        mode_order_id="site-spin-order",
        source_id="fermionic-control",
        particle_continuity_residual=0.0,
    )
    leaking = FermionicKeldyshTransportState(
        jnp.zeros_like(spectral),
        spectral,
        retarded,
        advanced,
        jnp.asarray([0.0, 1.0]),
        mode_order_id="site-spin-order",
        source_id="fermionic-control",
        particle_continuity_residual=1.0e-3,
    )

    assert state.statistics == "fermionic"
    assert bool(state.valid)
    assert not bool(leaking.valid)
