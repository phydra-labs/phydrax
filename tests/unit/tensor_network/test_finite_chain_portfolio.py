#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.solver._dmrg import FiniteDMRGPolicy, FiniteDMRGProblem, solve_finite_dmrg
from phydrax.solver._finite_response import (
    FiniteResponsePolicy,
    FiniteResponseProblem,
    solve_finite_excited_state,
    solve_finite_response,
)
from phydrax.solver._matrix_product_tdvp import (
    FiniteTDVPPolicy,
    FiniteTDVPProblem,
    solve_finite_tdvp,
)
from phydrax.tensor_network._core import MatrixProductState
from phydrax.tensor_network._models import (
    build_local_term_mpo,
    build_string_mpo,
    FiniteLocalTerm,
    FixedStructureMPOCoefficients,
)
from phydrax.tensor_network._mpo import (
    product_mpo,
    variational_compress_mps,
    VariationalCompressionPolicy,
)
from phydrax.tensor_network._observables import (
    finite_correlation_matrix,
    finite_entanglement_spectrum,
    finite_reduced_density,
    finite_transfer_spectrum,
)
from phydrax.tensor_network._thermal import (
    finite_temperature_purification,
    FiniteThermalPolicy,
    thermal_mpo_expectation,
)


def _product_state(*vectors):
    return MatrixProductState(
        tuple(
            jnp.asarray(vector, dtype=jnp.complex128)[None, :, None] for vector in vectors
        )
    )


def test_local_and_string_mpo_builders_have_dense_and_hermiticity_evidence():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    result = build_local_term_mpo(
        (2, 2), (FiniteLocalTerm(0, (z,)), FiniteLocalTerm(1, (z,), coefficient=2.0))
    )
    expected = jnp.kron(z, jnp.eye(2)) + 2.0 * jnp.kron(jnp.eye(2), z)
    assert jnp.allclose(result.operator.to_dense(), expected)
    assert result.evidence.hermitian
    assert result.evidence.hermiticity_residual < 1e-12
    string = build_string_mpo((2, 2), 0, (z, z))
    assert jnp.allclose(string.operator.to_dense(), jnp.kron(z, z))


def test_local_mpo_builder_supports_heterogeneous_site_dimensions():
    x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    number = jnp.diag(jnp.arange(3.0)).astype(jnp.complex128)
    result = build_local_term_mpo(
        (2, 3, 2),
        (
            FiniteLocalTerm(0, (x, number), coefficient=0.4),
            FiniteLocalTerm(2, (x,), coefficient=-0.2),
        ),
    )
    expected = 0.4 * jnp.kron(jnp.kron(x, number), jnp.eye(2)) - 0.2 * jnp.kron(
        jnp.eye(6), x
    )

    assert result.operator.output_dimensions == (2, 3, 2)
    assert result.operator.input_dimensions == (2, 3, 2)
    assert jnp.allclose(result.operator.to_dense(), expected)
    assert bool(result.evidence.hermitian)


def test_finite_dmrg_reports_galerkin_global_residual_and_variance():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    hamiltonian = build_local_term_mpo(
        (2, 2), (FiniteLocalTerm(0, (z,)), FiniteLocalTerm(1, (z,)))
    ).operator
    result = solve_finite_dmrg(
        FiniteDMRGProblem(_product_state([1.0, 0.0], [1.0, 0.0]), hamiltonian),
        FiniteDMRGPolicy(maximum_bond_dimension=2, maximum_sweeps=4),
    )
    used = result.diagnostics.active_sweeps
    assert jnp.any(used)
    assert jnp.nanmin(result.diagnostics.projected_residual_history) < 1e-7
    assert jnp.nanmin(result.diagnostics.global_residual_history) < 1e-7
    assert jnp.nanmin(result.diagnostics.energy_variance_history) < 1e-12
    assert jnp.allclose(result.energy, -2.0, atol=1e-7)


def test_excited_state_projector_targeting_reports_reference_overlap():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    hamiltonian = build_local_term_mpo(
        (2, 2), (FiniteLocalTerm(0, (z,)), FiniteLocalTerm(1, (z,)))
    ).operator
    result = solve_finite_excited_state(
        FiniteDMRGProblem(
            _product_state([1.0, 0.0], [1.0, 0.0]),
            hamiltonian,
            problem_id="excited-target",
        ),
        (_product_state([0.0, 1.0], [0.0, 1.0]),),
        jnp.asarray([5.0]),
        FiniteDMRGPolicy(maximum_bond_dimension=2, maximum_sweeps=4),
    )
    assert result.reference_overlaps.shape == (1,)
    assert result.projector_hermiticity_residuals[0] < 1e-10
    assert result.reference_overlaps[0] < 1e-6


def test_finite_tdvp_real_and_imaginary_time_semantics_are_normalized():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    hamiltonian = product_mpo(z[None, ...])
    plus = _product_state(jnp.asarray([1.0, 1.0]) / jnp.sqrt(2.0))
    real = solve_finite_tdvp(
        FiniteTDVPProblem(plus, hamiltonian),
        FiniteTDVPPolicy("real-time", step_size=0.02, steps=2),
    )
    assert real.successful
    assert jnp.allclose(real.diagnostics.norm_history, 1.0, atol=1e-6)
    assert jnp.allclose(real.diagnostics.normalized_energy_history, 0.0, atol=1e-6)
    imaginary = solve_finite_tdvp(
        FiniteTDVPProblem(plus, hamiltonian),
        FiniteTDVPPolicy("imaginary-time", step_size=0.05, steps=2),
    )
    assert imaginary.successful
    assert jnp.allclose(imaginary.diagnostics.norm_history, 1.0, atol=1e-6)
    assert (
        imaginary.diagnostics.normalized_energy_history[-1]
        < imaginary.diagnostics.normalized_energy_history[0]
    )
    assert imaginary.checkpoint.completed_steps == 2


def test_two_site_tdvp_uses_fixed_schedule_and_truncation_capacity():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    hamiltonian = build_local_term_mpo(
        (2, 2), (FiniteLocalTerm(0, (z,)), FiniteLocalTerm(1, (z,)))
    ).operator
    schedule = FixedStructureMPOCoefficients(
        (hamiltonian,),
        jnp.asarray([[1.0], [0.5]], dtype=jnp.float64),
    )
    result = solve_finite_tdvp(
        FiniteTDVPProblem(_product_state([1.0, 0.0], [1.0, 0.0]), schedule),
        FiniteTDVPPolicy(
            "real-time",
            step_size=0.01,
            steps=1,
            algorithm="two-site",
            maximum_bond_dimension=2,
        ),
    )
    assert result.diagnostics.truncation_history.shape == (1, 2)
    assert result.diagnostics.active_steps[0]
    assert schedule.operator_at(0).structure_id == schedule.operator_at(1).structure_id


def test_variational_compression_returns_fixed_objective_and_residual_histories():
    state = _product_state([1.0, 1.0], [1.0, -1.0]).normalized()
    compressed, evidence = variational_compress_mps(
        state,
        VariationalCompressionPolicy(
            maximum_bond_dimension=1,
            maximum_sweeps=2,
            gradient_step=0.01,
        ),
    )
    assert compressed.bond_dimensions == (1,)
    assert evidence.objective_history.shape == (3,)
    assert evidence.gradient_residual_history.shape == (2,)
    assert jnp.all(jnp.isfinite(evidence.objective_history[evidence.active_sweeps.sum()]))


def test_beta_zero_purification_is_maximally_mixed_and_normalized():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    hamiltonian = product_mpo(z[None, ...])
    thermal = finite_temperature_purification(
        hamiltonian,
        0.0,
        FiniteThermalPolicy(maximum_bond_dimension=2, maximum_order=4),
    )
    assert thermal.evidence.successful
    assert jnp.allclose(thermal.evidence.normalized_trace, 1.0)
    assert jnp.allclose(
        thermal_mpo_expectation(thermal.purification, hamiltonian), 0.0, atol=1e-12
    )


def test_response_zero_time_sum_rule_and_finite_fourier_history():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    problem = FiniteResponseProblem(
        _product_state([1.0, 0.0]),
        product_mpo(z[None, ...]),
        product_mpo(x[None, ...]),
        jnp.linspace(-3.0, 3.0, 33),
    )
    result = solve_finite_response(
        problem,
        FiniteResponsePolicy(
            step_size=0.02, steps=3, maximum_bond_dimension=1, tdvp_algorithm="one-site"
        ),
    )
    assert result.evidence.successful
    assert jnp.allclose(result.evidence.sum_rule, 1.0, atol=1e-10)
    assert result.evidence.sum_rule_residual < 1e-10
    assert jnp.all(jnp.isfinite(result.spectrum))


def test_correlations_reduced_density_and_entanglement_are_computed():
    z = jnp.diag(jnp.asarray([1.0, -1.0], dtype=jnp.complex128))
    state = _product_state([1.0, 0.0], [1.0, 0.0])
    correlations = finite_correlation_matrix(state, z)
    transfer = finite_transfer_spectrum(state, 0, maximum_modes=2)
    reduced = finite_reduced_density(state, 0, 1)
    entanglement = finite_entanglement_spectrum(state, 0, maximum_rank=2)
    assert jnp.allclose(correlations.expectation, 1.0)
    assert jnp.allclose(correlations.connected, 0.0)
    assert reduced.valid
    assert transfer.successful
    assert jnp.allclose(transfer.spectral_radius, 1.0)
    assert jnp.allclose(reduced.density, jnp.diag(jnp.asarray([1.0, 0.0])))
    assert jnp.allclose(entanglement.entropy, 0.0)
