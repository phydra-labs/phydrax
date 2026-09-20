#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.chemistry.periodic._boltzmann import (
    ConstantRelaxationTime,
    PeriodicBoltzmannPlan,
)
from phydrax.chemistry.periodic._kubo import (
    conserved_collinear_spin_evidence,
    finite_frequency_kubo_response,
    KuboDiamagneticSumRule,
    KuboLinewidth,
    PeriodicKuboPlan,
    require_conserved_collinear_spin,
)


_E = 1.602176634e-19
_KB = 1.380649e-23


def _two_level_kubo(shift=0.0):
    gap = 1.5 * _E
    temperature = 300.0
    velocity = 2.0e5
    energies = jnp.asarray([[-0.5 * gap + shift, 0.5 * gap + shift]])
    velocities = jnp.zeros((1, 2, 2, 1), dtype="complex128")
    velocities = velocities.at[0, 0, 1, 0].set(velocity)
    velocities = velocities.at[0, 1, 0, 0].set(velocity)
    occupation_difference = np.tanh(gap / (4.0 * _KB * temperature))
    expected = np.pi * _E**2 / 1.0e-28 * occupation_difference / gap * velocity**2
    return PeriodicKuboPlan(
        energies,
        velocities,
        jnp.asarray([1.0]),
        KuboDiamagneticSumRule(
            jnp.asarray([[expected]]), source_id="analytic-two-level-diamagnetic"
        ),
        chemical_potential_joule=shift,
        temperature_kelvin=temperature,
        cell_volume_m3=1.0e-28,
    )


def test_kubo_separates_drude_and_regular_response_with_passivity_and_f_sum():
    plan = _two_level_kubo()
    raw = plan.raw_transitions()
    gap_frequency = 1.5 * _E / 1.0545718176461565e-34
    response = finite_frequency_kubo_response(
        plan,
        jnp.asarray([0.8 * gap_frequency, gap_frequency, 1.2 * gap_frequency]),
        KuboLinewidth(0.02 * _E, mechanism_id="declared-elastic-linewidth"),
        f_sum_tolerance=1.0e-12,
    )

    assert raw.drude_weight[0, 0] == pytest.approx(0.0)
    assert raw.f_sum_relative_residual == pytest.approx(0.0, abs=1.0e-12)
    assert jnp.real(response.regular_conductivity_siemens_per_m[1, 0, 0]) > 0.0
    assert bool(response.evidence.passive)
    assert bool(response.evidence.sum_rule_satisfied)
    assert not response.evidence.includes_drude
    assert not response.evidence.infers_relaxation_time
    with pytest.raises(ValueError, match="strictly increasing positive"):
        finite_frequency_kubo_response(
            plan,
            jnp.asarray([0.0, gap_frequency]),
            KuboLinewidth(0.02 * _E, mechanism_id="declared-elastic-linewidth"),
        )


def test_kubo_is_energy_shift_invariant_and_drude_is_degenerate_gauge_invariant():
    unshifted = _two_level_kubo().raw_transitions()
    shifted = _two_level_kubo(4.0 * _E).raw_transitions()
    assert jnp.allclose(unshifted.transition_factors, shifted.transition_factors)
    assert jnp.allclose(
        unshifted.regular_spectral_weight, shifted.regular_spectral_weight
    )

    energies = jnp.zeros((1, 2))
    velocity = jnp.asarray([[[[1.0], [0.0]], [[0.0], [-1.0]]]])
    rotation = jnp.asarray([[1.0, 1.0], [-1.0, 1.0]]) / jnp.sqrt(2.0)
    rotated_matrix = rotation.T @ velocity[0, :, :, 0] @ rotation
    rotated = rotated_matrix[None, :, :, None]
    rule = KuboDiamagneticSumRule(jnp.zeros((1, 1)), source_id="unused-drude-rule")
    common = dict(
        chemical_potential_joule=0.0,
        temperature_kelvin=300.0,
        cell_volume_m3=1.0,
    )
    original = PeriodicKuboPlan(
        energies, velocity, jnp.ones(1), rule, **common
    ).raw_transitions()
    changed = PeriodicKuboPlan(
        energies, rotated, jnp.ones(1), rule, **common
    ).raw_transitions()
    assert jnp.allclose(original.drude_weight, changed.drude_weight)


def test_collinear_spin_response_requires_commutator_closure():
    energies = jnp.asarray([[0.0, 2.0 * _E]])
    conserved = conserved_collinear_spin_evidence(
        energies,
        jnp.asarray([[[0.5, 0.0], [0.0, -0.5]]]),
        source_id="collinear-spin",
    )
    mixed = conserved_collinear_spin_evidence(
        energies,
        jnp.asarray([[[0.0, 0.5], [0.5, 0.0]]]),
        source_id="nonconserved-spin",
    )

    require_conserved_collinear_spin(conserved)
    assert bool(conserved.conserved)
    assert not bool(mixed.conserved)
    with pytest.raises(ValueError, match=r"\[H,Sz\]=0"):
        require_conserved_collinear_spin(mixed)


def _boltzmann_plan(tau, shift=0.0, *, rank_deficient=False):
    energies = jnp.asarray(
        [
            [-0.05 * _E + shift, 0.08 * _E + shift],
            [0.03 * _E + shift, -0.07 * _E + shift],
        ]
    )
    velocities = jnp.asarray(
        [
            [[1.2e5, 0.4e5], [-0.7e5, 1.1e5]],
            [[0.8e5, -1.3e5], [-1.0e5, -0.6e5]],
        ]
    )
    if rank_deficient:
        velocities = velocities.at[:, :, 1].set(0.0)
    return PeriodicBoltzmannPlan(
        energies,
        velocities,
        jnp.asarray([0.5, 0.5]),
        ConstantRelaxationTime(tau, mechanism_id="supplied-elastic-tau"),
        chemical_potential_joule=shift,
        temperature_kelvin=350.0,
        cell_volume_m3=2.0e-28,
    )


def test_constant_tau_boltzmann_scales_transport_and_obeys_onsager():
    base = _boltzmann_plan(1.0e-14).evaluate()
    doubled = _boltzmann_plan(2.0e-14).evaluate()

    assert jnp.allclose(
        doubled.electrical_conductivity_siemens_per_m,
        2.0 * base.electrical_conductivity_siemens_per_m,
    )
    assert jnp.allclose(
        doubled.electronic_thermal_conductivity_watt_per_m_kelvin,
        2.0 * base.electronic_thermal_conductivity_watt_per_m_kelvin,
    )
    assert jnp.allclose(doubled.seebeck_volt_per_kelvin, base.seebeck_volt_per_kelvin)
    assert jnp.allclose(base.peltier_volt, 350.0 * base.seebeck_volt_per_kelvin)
    assert int(base.evidence.electrical_rank) == 2
    assert bool(base.evidence.passive)
    assert not base.evidence.relaxation_inferred_from_linewidth


def test_boltzmann_is_energy_shift_invariant_and_refuses_missing_velocity_rank():
    base = _boltzmann_plan(1.0e-14).evaluate()
    shifted = _boltzmann_plan(1.0e-14, shift=3.0 * _E).evaluate()

    assert jnp.allclose(
        base.electrical_conductivity_siemens_per_m,
        shifted.electrical_conductivity_siemens_per_m,
    )
    assert jnp.allclose(base.seebeck_volt_per_kelvin, shifted.seebeck_volt_per_kelvin)
    with pytest.raises(ValueError, match="rank deficient"):
        _boltzmann_plan(1.0e-14, rank_deficient=True).evaluate()
    with pytest.raises(ValueError, match="positive"):
        ConstantRelaxationTime(0.0, mechanism_id="invalid")
