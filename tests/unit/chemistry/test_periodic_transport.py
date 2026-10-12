#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax.chemistry.periodic._boltzmann import (
    ConstantRelaxationTime,
    PeriodicBoltzmannPlan,
)
from phydrax.chemistry.periodic._kubo import (
    conserved_collinear_spin_evidence,
    finite_frequency_kubo_response,
    KuboDiamagneticSumRule,
    KuboFrequencyPolicy,
    KuboLinewidth,
    PeriodicKuboPlan,
    require_conserved_collinear_spin,
)


_E = 1.602176634e-19
_KB = 1.380649e-23


def _two_level_kubo(shift: Any = 0.0) -> Any:
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


def test_periodic_transport_scenario_1() -> None:
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
        energies,
        velocity,
        jnp.ones(1),
        rule,
        # ty: ignore[invalid-argument-type]
        **common,
    ).raw_transitions()
    changed = PeriodicKuboPlan(
        energies,
        rotated,
        jnp.ones(1),
        rule,
        # ty: ignore[invalid-argument-type]
        **common,
    ).raw_transitions()
    assert jnp.allclose(original.drude_weight, changed.drude_weight)
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


def _kubo_loop_reference(
    energies: NDArray[np.float64],
    velocities: NDArray[np.complex128],
    weights: NDArray[np.float64],
    frequencies: NDArray[np.float64],
    width: float,
) -> tuple[NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
    """Independent SI frequency/transition loops, including both optical poles."""
    hbar = 1.0545718176461565e-34
    occupations = 1.0 / (1.0 + np.exp(energies / (_KB * 300.0)))
    dimension = velocities.shape[-1]
    conductivity = np.zeros((frequencies.size, dimension, dimension), dtype=np.complex128)
    spectral_weight = np.zeros((dimension, dimension), dtype=np.float64)
    for k in range(energies.shape[0]):
        for n in range(energies.shape[1]):
            for m in range(n + 1, energies.shape[1]):
                delta = energies[k, m] - energies[k, n]
                factor = weights[k] * (occupations[k, n] - occupations[k, m]) / delta
                for a in range(dimension):
                    for b in range(dimension):
                        product = (
                            velocities[k, n, m, a] * velocities[k, n, m, b].conjugate()
                        )
                        spectral_weight[a, b] += (
                            np.pi * _E**2 / 1.0e-28 * factor * product.real
                        )
                        for f, omega in enumerate(frequencies):
                            poles = product / (omega - delta / hbar + 1j * width / hbar)
                            poles += product.conjugate() / (
                                omega + delta / hbar + 1j * width / hbar
                            )
                            conductivity[f, a, b] += 1j * _E**2 / 1.0e-28 * factor * poles
    return conductivity, spectral_weight, occupations


@pytest.mark.parametrize("k_count", [1, 2], ids=["one-k", "two-k"])
@pytest.mark.parametrize("dimension", [1, 2], ids=["one-axis", "two-axes"])
@pytest.mark.parametrize(
    ("policy", "chunk_size"),
    [("chunked", 1), ("chunked", 2), ("chunked", 8), ("parallel", 8)],
    ids=["single-frequency", "tail-chunk", "default-chunk", "parallel"],
)
def test_kubo_frequency_transition_loop_parity(
    k_count: int, dimension: int, policy: KuboFrequencyPolicy, chunk_size: int
) -> None:
    energies = np.asarray([[-0.4 * _E, 0.6 * _E], [-0.3 * _E, 0.9 * _E]])[:k_count]
    velocities = np.zeros((k_count, 2, 2, dimension), dtype=np.complex128)
    for k in range(k_count):
        for a in range(dimension):
            velocities[k, 0, 1, a] = 1.0e5 * (k + 1.0 + 0.4 * a + 0.3j * a)
            velocities[k, 1, 0, a] = velocities[k, 0, 1, a].conjugate()
    weights = np.arange(1, k_count + 1, dtype=np.float64)
    weights /= np.sum(weights)
    frequencies = np.asarray([0.2, 0.95, 1.7]) * _E / 1.0545718176461565e-34
    width = 0.04 * _E
    expected, spectral_weight, occupations = _kubo_loop_reference(
        energies, velocities, weights, frequencies, width
    )
    plan = PeriodicKuboPlan(
        energies,
        velocities,
        weights,
        KuboDiamagneticSumRule(spectral_weight, source_id="independent-transition-loop"),
        chemical_potential_joule=0.0,
        temperature_kelvin=300.0,
        cell_volume_m3=1.0e-28,
    )
    response = finite_frequency_kubo_response(
        plan,
        frequencies,
        KuboLinewidth(width, mechanism_id="loop-reference-linewidth"),
        f_sum_tolerance=1.0e-12,
        frequency_policy=policy,
        frequency_chunk_size=chunk_size,
    )
    np.testing.assert_allclose(
        response.regular_conductivity_siemens_per_m, expected, rtol=2.0e-11, atol=2.0e-13
    )
    np.testing.assert_allclose(
        response.raw.occupations, occupations, rtol=2.0e-11, atol=2.0e-15
    )
    np.testing.assert_allclose(
        response.raw.regular_spectral_weight, spectral_weight, rtol=2.0e-11
    )
    np.testing.assert_array_equal(
        response.raw.drude_weight, np.zeros((dimension, dimension))
    )
    assert float(response.raw.f_sum_relative_residual) < 1.0e-12
    assert bool(response.successful)
    assert bool(response.evidence.finite)
    assert bool(response.evidence.passive)
    assert bool(response.evidence.sum_rule_satisfied)
    assert not response.evidence.includes_drude
    assert not response.evidence.infers_relaxation_time
    assert response.plan_id == plan.plan_id
    assert response.raw.plan_id == plan.plan_id
    assert response.evidence.linewidth_mechanism_id == "loop-reference-linewidth"
    effective_chunk = min(chunk_size, 3) if policy == "chunked" else 3
    assert response.evidence.frequency_policy == policy
    assert response.evidence.frequency_chunk_size == effective_chunk
    assert response.evidence.logical_frequency_workspace_bytes == (
        16 * effective_chunk * k_count * 4 * dimension**2
    )


def test_kubo_frequency_policy_identity_is_separate_from_scientific_plan() -> None:
    plan = _two_level_kubo()
    frequencies = np.asarray([0.2, 0.95, 1.7]) * _E / 1.0545718176461565e-34
    linewidth = KuboLinewidth(0.04 * _E, mechanism_id="identity-reference")
    chunked = finite_frequency_kubo_response(
        plan, frequencies, linewidth, frequency_chunk_size=1
    )
    parallel = finite_frequency_kubo_response(
        plan, frequencies, linewidth, frequency_policy="parallel"
    )
    assert chunked.plan_id == parallel.plan_id == plan.plan_id
    assert chunked.evidence.execution_policy_id != parallel.evidence.execution_policy_id
    np.testing.assert_allclose(
        chunked.regular_conductivity_siemens_per_m,
        parallel.regular_conductivity_siemens_per_m,
        rtol=2.0e-11,
    )


@pytest.mark.parametrize("chunk_size", [0, -1], ids=["zero", "negative"])
def test_kubo_refuses_nonpositive_frequency_chunks(chunk_size: int) -> None:
    with pytest.raises(
        ValueError, match="frequency_chunk_size must be a positive integer"
    ):
        finite_frequency_kubo_response(
            _two_level_kubo(),
            np.asarray([1.0e15]),
            KuboLinewidth(0.04 * _E, mechanism_id="chunk-refusal"),
            frequency_chunk_size=chunk_size,
        )


def _boltzmann_plan(tau: Any, shift: Any = 0.0, *, rank_deficient: Any = False) -> Any:
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


def test_boltzmann_is_energy_shift_invariant_and_refuses_missing_velocity_rank() -> None:
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
