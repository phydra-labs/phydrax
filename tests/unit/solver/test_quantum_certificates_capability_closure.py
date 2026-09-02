# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.operators.quantum import (
    ElectronicIntegralHamiltonian,
    ElectronicVMCResourcePlan,
)
from phydrax.solver._adaptive_tdvp import AdaptiveTDVPPlan, solve_adaptive_tdvp
from phydrax.solver._finite_subspace_tdvp import (
    FiniteVariationalSubspaceTDVPProblem,
    prepare_finite_subspace_tdvp,
    solve_finite_subspace_tdvp,
)
from phydrax.solver._open_certificates import (
    certify_finite_lindblad_steady_state,
    certify_finite_refinement,
    certify_process_identifiability,
)


def test_finite_subspace_cayley_retains_norm_energy_and_reversibility():
    overlap = jnp.eye(2, dtype=jnp.complex64)
    hamiltonian = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    problem = FiniteVariationalSubspaceTDVPProblem(
        overlap,
        hamiltonian,
        jnp.array([1, 0], dtype=jnp.complex64),
        problem_id="rabi",
    )
    plan = prepare_finite_subspace_tdvp(problem, step_size=0.05, num_steps=8)
    result = solve_finite_subspace_tdvp(problem, plan)
    assert bool(result.valid)
    assert jnp.max(jnp.abs(result.norm_drifts)) < 1e-5
    assert jnp.max(jnp.abs(result.energy_drifts)) < 1e-5
    assert result.claim.startswith("cayley")


def test_finite_subspace_cayley_rejects_cross_problem_plan_reuse():
    overlap = jnp.eye(2, dtype=jnp.complex64)
    hamiltonian = jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64)
    source = FiniteVariationalSubspaceTDVPProblem(
        overlap,
        hamiltonian,
        jnp.array([1, 0], dtype=jnp.complex64),
        problem_id="source",
    )
    other = FiniteVariationalSubspaceTDVPProblem(
        overlap,
        hamiltonian,
        jnp.array([0, 1], dtype=jnp.complex64),
        problem_id="other",
    )
    plan = prepare_finite_subspace_tdvp(source, step_size=0.05, num_steps=1)

    with pytest.raises(ValueError, match="different finite-subspace problem"):
        solve_finite_subspace_tdvp(other, plan)


def test_adaptive_tdvp_separates_sampling_uncertainty_from_temporal_defect():
    plan = AdaptiveTDVPPlan(
        (0.0, 0.2),
        initial_step_size=0.05,
        step_size_bounds=(1e-3, 0.1),
        absolute_tolerance=1e-3,
        maximum_attempts=16,
        maximum_accepted_steps=16,
    )

    def vector_field(parameters, time, key):
        del time, key
        return -1j * parameters, jnp.asarray(0.0)

    result = solve_adaptive_tdvp(
        vector_field,
        jnp.array([1.0 + 0.0j]),
        plan,
        key=jr.key(3),
    )
    assert bool(result.valid)
    assert not jnp.any(result.noise_dominated)
    assert jnp.any(result.accepted_attempts)


def test_adaptive_tdvp_uses_real_time_dtype_for_complex_parameters():
    plan = AdaptiveTDVPPlan(
        (0.0, 0.2),
        initial_step_size=0.05,
        step_size_bounds=(0.01, 0.1),
        absolute_tolerance=1e-3,
        maximum_attempts=4,
        maximum_accepted_steps=4,
    )
    initial = jnp.array([1.0 + 0.0j], dtype=jnp.complex64)
    callback_time_dtypes = []

    def time_dependent_vector_field(parameters, time, key):
        del key
        callback_time_dtypes.append(time.dtype)
        stopped = jnp.where(time < 0.1, 0.0, 0.0)
        return stopped * parameters, jnp.asarray(0.0, dtype=time.dtype)

    result = solve_adaptive_tdvp(
        time_dependent_vector_field,
        initial,
        plan,
        key=jr.key(31),
    )

    real_dtype = initial.real.dtype
    assert callback_time_dtypes
    assert all(dtype == real_dtype for dtype in callback_time_dtypes)
    assert result.accepted_times.dtype == real_dtype
    assert result.attempt_times.dtype == real_dtype
    assert result.attempt_step_sizes.dtype == real_dtype
    assert bool(result.valid)
    accepted_times = result.accepted_times[result.accepted_mask]
    assert jnp.allclose(
        accepted_times,
        jnp.asarray([0.0, 0.05, 0.15, 0.2], dtype=real_dtype),
    )


def test_adaptive_tdvp_propagates_velocity_uncertainty_over_the_step():
    plan = AdaptiveTDVPPlan(
        (0.0, 0.1),
        initial_step_size=0.05,
        step_size_bounds=(0.01, 0.05),
        absolute_tolerance=1e-3,
        maximum_attempts=2,
        maximum_accepted_steps=2,
    )

    def vector_field(parameters, time, key):
        del time, key
        return jnp.zeros_like(parameters), jnp.asarray(1e-2)

    result = solve_adaptive_tdvp(
        vector_field,
        jnp.array([1.0]),
        plan,
        key=jr.key(4),
    )

    assert bool(result.valid)
    assert jnp.all(result.accepted_attempts)
    assert jnp.allclose(result.sampling_uncertainties, 5e-4)


def test_adaptive_tdvp_exhausts_constant_sampling_noise_without_step_collapse():
    plan = AdaptiveTDVPPlan(
        (0.0, 0.1),
        initial_step_size=0.05,
        step_size_bounds=(0.01, 0.05),
        absolute_tolerance=1e-3,
        maximum_attempts=4,
        maximum_accepted_steps=4,
    )

    def vector_field(parameters, time, key):
        del time, key
        return jnp.zeros_like(parameters), jnp.asarray(1e-1)

    result = solve_adaptive_tdvp(
        vector_field,
        jnp.array([1.0]),
        plan,
        key=jr.key(5),
    )

    assert bool(result.overflow)
    assert jnp.all(result.noise_dominated)
    assert not jnp.any(result.accepted_attempts)
    assert jnp.allclose(result.attempt_step_sizes, 0.05)
    assert jnp.allclose(result.sampling_uncertainties, 5e-3)


def test_finite_steady_state_refinement_and_quotient_identifiability_claims():
    identity = jnp.eye(2, dtype=jnp.complex64).reshape(-1)
    trace = jnp.array([1, 0, 0, 1], dtype=jnp.complex64)
    liouvillian = 0.5 * identity[:, None] @ trace[None, :] - jnp.eye(
        4, dtype=jnp.complex64
    )
    steady = certify_finite_lindblad_steady_state(liouvillian, 2)
    assert bool(steady.valid)
    assert bool(steady.unique)
    assert jnp.allclose(steady.density, jnp.eye(2) / 2, atol=1e-5)

    refinement = certify_finite_refinement(
        jnp.array([2, 4, 8]),
        jnp.array([1.0, 0.6, 0.5]),
        jnp.array([0.2, 0.05, 0.01]),
        axis="cutoff",
        tolerance=0.11,
    )
    assert bool(refinement.stabilized)
    assert refinement.estimate_kind == "difference"

    design = jnp.array([[1.0, 0.0], [0.0, 0.0]])
    gauge = jnp.array([[0.0], [1.0]])
    identified = certify_process_identifiability(design, gauge, design_id="finite-design")
    assert bool(identified.valid)
    assert bool(identified.identifiable)


def test_detailed_balance_gap_is_scale_invariant_and_rejects_growth():
    identity = jnp.eye(2, dtype=jnp.complex64).reshape(-1)
    trace = jnp.array([1, 0, 0, 1], dtype=jnp.complex64)
    liouvillian = 0.5 * identity[:, None] @ trace[None, :] - jnp.eye(
        4, dtype=jnp.complex64
    )
    metric = jnp.eye(4, dtype=jnp.complex64)
    baseline = certify_finite_lindblad_steady_state(
        liouvillian,
        2,
        detailed_balance_symmetrizer=metric,
    )
    rescaled = certify_finite_lindblad_steady_state(
        liouvillian,
        2,
        detailed_balance_symmetrizer=7.0 * metric,
    )

    assert bool(baseline.valid)
    assert bool(rescaled.valid)
    assert jnp.allclose(baseline.certified_gap, rescaled.certified_gap)
    assert jnp.allclose(baseline.certified_gap, 1.0)

    growing = liouvillian.at[1, 1].add(1.5)
    invalid = certify_finite_lindblad_steady_state(
        growing,
        2,
        detailed_balance_symmetrizer=metric,
    )
    assert not bool(invalid.valid)
    assert bool(invalid.unique)
    assert bool(invalid.physical)
    assert jnp.isnan(invalid.certified_gap)


def test_resource_admission_and_finite_no_pair_metadata():
    resource = ElectronicVMCResourcePlan(
        8,
        determinant_count=2,
        maximum_pair_elements=10_000,
        maximum_determinant_work=10_000,
    )
    assert resource.electron_count == 8
    one = jnp.eye(2, dtype=jnp.complex64)
    two = jnp.zeros((2, 2, 2, 2), dtype=jnp.complex64)
    operator = ElectronicIntegralHamiltonian(
        one,
        two,
        representation="four-component-no-pair",
        projector_id="positive-energy:finite-basis",
    )
    assert bool(operator.valid)
    assert "no-pair" in operator.claim
