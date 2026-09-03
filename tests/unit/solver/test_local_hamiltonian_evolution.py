#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import pytest

import phydrax as phx


s = phx.solver
q = phx.operators.quantum


def _paulis():
    x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.complex128)
    z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.complex128)
    return x, z


def test_single_term_local_evolution_matches_direct_exponential_and_saves_states():
    _, z = _paulis()
    layout = q.HilbertRegisterLayout(("q",), (2,))
    hamiltonian = s.LocalHamiltonian(
        layout,
        (s.LocalHamiltonianTerm(z, ("q",)),),
    )
    schedule = s.FixedGridLocalHamiltonian(
        hamiltonian,
        jnp.asarray([0.0, 0.2]),
        jnp.ones((1, 1)),
    )
    prepared = s.prepare_local_hamiltonian_evolution(
        schedule,
        policy=s.LocalHamiltonianEvolutionPolicy(save_indices=(0, 1)),
    )
    initial = jnp.asarray([1.0, 1.0], dtype=jnp.complex128) / jnp.sqrt(2.0)
    result = jax.jit(s.solve_local_hamiltonian_evolution)(prepared, initial)
    expected = jsp.linalg.expm(-0.2j * z) @ initial

    assert bool(result.successful)
    assert jnp.allclose(result.final_state, expected, atol=1e-12)
    assert result.saved_states.shape == (2, 2)
    assert jnp.allclose(result.saved_states[0], initial)
    assert jnp.allclose(result.saved_states[1], expected)


def test_second_order_product_formula_has_expected_global_convergence():
    x, z = _paulis()
    layout = q.HilbertRegisterLayout(("q",), (2,))
    hamiltonian = s.LocalHamiltonian(
        layout,
        (
            s.LocalHamiltonianTerm(x, ("q",), term_id="x"),
            s.LocalHamiltonianTerm(z, ("q",), term_id="z"),
        ),
    )
    initial = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
    expected = jsp.linalg.expm(-0.7j * (x + z)) @ initial

    def error(interval_count):
        grid = jnp.linspace(0.0, 0.7, interval_count + 1)
        schedule = s.FixedGridLocalHamiltonian(
            hamiltonian,
            grid,
            jnp.ones((interval_count, 2)),
        )
        prepared = s.prepare_local_hamiltonian_evolution(
            schedule,
            policy=s.LocalHamiltonianEvolutionPolicy(order=2),
        )
        result = s.solve_local_hamiltonian_evolution(prepared, initial)
        return jnp.linalg.norm(result.final_state - expected)

    coarse = error(8)
    fine = error(16)
    assert coarse / fine > 3.7
    assert coarse / fine < 4.3


def test_local_hamiltonian_materialization_and_heterogeneous_batch_evolution():
    x, _ = _paulis()
    number = jnp.diag(jnp.arange(3.0)).astype(jnp.complex128)
    layout = q.HilbertRegisterLayout(("q", "r"), (2, 3))
    term = s.LocalHamiltonianTerm.from_product((x, number), ("q", "r"))
    hamiltonian = s.LocalHamiltonian(layout, (term,))
    dense = s.materialize_local_hamiltonian(hamiltonian)
    schedule = s.FixedGridLocalHamiltonian(
        hamiltonian,
        jnp.asarray([0.0, 0.1]),
        jnp.ones((1, 1)),
    )
    prepared = s.prepare_local_hamiltonian_evolution(schedule)
    initial = jnp.eye(6, dtype=jnp.complex128)
    result = s.solve_local_hamiltonian_evolution(prepared, initial)

    assert jnp.allclose(dense, jnp.kron(x, number))
    assert result.final_state.shape == (6, 6)
    assert jnp.allclose(result.final_state, jsp.linalg.expm(-0.1j * dense).T)


def test_local_hamiltonian_refresh_preserves_structure_and_gradients():
    x, _ = _paulis()
    layout = q.HilbertRegisterLayout(("q",), (2,))

    def schedule(amplitude):
        hamiltonian = s.LocalHamiltonian(
            layout,
            (s.LocalHamiltonianTerm(amplitude * x, ("q",), term_id="drive"),),
        )
        return s.FixedGridLocalHamiltonian(
            hamiltonian,
            jnp.asarray([0.0, 0.3]),
            jnp.ones((1, 1)),
        )

    prepared = s.prepare_local_hamiltonian_evolution(schedule(jnp.asarray(1.0)))
    initial = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)
    derivative = jax.grad(
        lambda amplitude: jnp.real(
            s.solve_local_hamiltonian_evolution(
                s.refresh_local_hamiltonian_evolution(
                    prepared,
                    schedule(amplitude),
                ),
                initial,
            ).final_state[1]
        )
    )(jnp.asarray(1.0))

    assert jnp.isfinite(derivative)
    assert (
        int(
            s.refresh_local_hamiltonian_evolution(
                prepared,
                schedule(jnp.asarray(1.1)),
            ).numeric_version
        )
        == 1
    )


def test_reversible_product_formula_matches_algorithmic_reverse_mode():
    x, z = _paulis()
    layout = q.HilbertRegisterLayout(("q",), (2,))
    grid = jnp.linspace(0.0, 0.8, 9)
    initial = jnp.asarray([1.0, 0.0], dtype=jnp.complex128)

    def schedule(generator_scale, coefficient):
        hamiltonian = s.LocalHamiltonian(
            layout,
            (
                s.LocalHamiltonianTerm(
                    generator_scale * x,
                    ("q",),
                    term_id="variable-x",
                ),
                s.LocalHamiltonianTerm(z, ("q",), term_id="fixed-z"),
            ),
        )
        values = jnp.stack(
            (
                jnp.full((8,), coefficient),
                jnp.linspace(0.2, 0.5, 8),
            ),
            axis=1,
        )
        return s.FixedGridLocalHamiltonian(hamiltonian, grid, values)

    nominal = schedule(jnp.asarray(1.0), jnp.asarray(0.7))
    automatic = s.prepare_local_hamiltonian_evolution(
        nominal,
        policy=s.LocalHamiltonianEvolutionPolicy(differentiation="autodiff"),
    )
    reversible = s.prepare_local_hamiltonian_evolution(
        nominal,
        policy=s.LocalHamiltonianEvolutionPolicy(
            differentiation="reversible-product-formula"
        ),
    )

    def loss(prepared, generator_scale, coefficient):
        result = s.solve_local_hamiltonian_evolution(
            s.refresh_local_hamiltonian_evolution(
                prepared,
                schedule(generator_scale, coefficient),
            ),
            initial,
        )
        return jnp.real(result.final_state[1] * jnp.conj(result.final_state[1]))

    automatic_gradient = jax.grad(
        lambda generator_scale, coefficient: loss(
            automatic,
            generator_scale,
            coefficient,
        ),
        argnums=(0, 1),
    )(jnp.asarray(1.0), jnp.asarray(0.7))
    reversible_gradient = jax.grad(
        lambda generator_scale, coefficient: loss(
            reversible,
            generator_scale,
            coefficient,
        ),
        argnums=(0, 1),
    )(jnp.asarray(1.0), jnp.asarray(0.7))

    assert jnp.allclose(reversible_gradient[0], automatic_gradient[0], atol=1e-9)
    assert jnp.allclose(reversible_gradient[1], automatic_gradient[1], atol=1e-9)


def test_local_hamiltonian_lowers_exactly_to_heterogeneous_noncontiguous_mpo():
    x, z = _paulis()
    number = jnp.diag(jnp.arange(3.0)).astype(jnp.complex128)
    layout = q.HilbertRegisterLayout(("a", "r", "b"), (2, 3, 2))
    hamiltonian = s.LocalHamiltonian(
        layout,
        (
            s.LocalHamiltonianTerm.from_product((x, x), ("a", "b")),
            s.LocalHamiltonianTerm.from_product((number,), ("r",)),
            s.LocalHamiltonianTerm.from_product((z,), ("a",)),
        ),
    )
    lowering = s.lower_local_hamiltonian_to_mpo(hamiltonian)
    dense = s.materialize_local_hamiltonian(hamiltonian)
    schedule = s.FixedGridLocalHamiltonian(
        hamiltonian,
        jnp.asarray([0.0, 0.1, 0.2]),
        jnp.asarray([[1.0, 0.5, 0.2], [0.7, 0.4, 0.1]]),
    )
    coefficient_basis = s.fixed_grid_local_hamiltonian_mpo_coefficients(
        schedule,
        lowering,
    )

    assert bool(lowering.evidence.valid)
    assert lowering.evidence.chain_order == ("a", "r", "b")
    assert jnp.allclose(lowering.operator.to_dense(), dense)
    assert coefficient_basis.coefficients.shape == (3, 3)
    assert jnp.allclose(
        coefficient_basis.coefficients[-1],
        schedule.coefficients[-1],
    )

    unfactored = s.LocalHamiltonian(
        q.HilbertRegisterLayout(("a", "b"), (2, 2)),
        (s.LocalHamiltonianTerm(jnp.kron(x, x), ("a", "b")),),
    )
    with pytest.raises(ValueError, match="product_factors"):
        s.lower_local_hamiltonian_to_mpo(unfactored)


def test_local_hamiltonian_rejects_invalid_structure_and_reports_bad_invariants():
    x, _ = _paulis()
    layout = q.HilbertRegisterLayout(("q",), (2,))
    with pytest.raises(ValueError, match="unique"):
        s.LocalHamiltonianTerm(x, ("q", "q"))
    with pytest.raises(ValueError, match="target dimensions"):
        s.LocalHamiltonian(
            layout,
            (s.LocalHamiltonianTerm(jnp.eye(3), ("q",)),),
        )
    invalid_term = s.LocalHamiltonianTerm(
        jnp.asarray([[0.0, 1.0], [0.0, 0.0]], dtype=jnp.complex128),
        ("q",),
    )
    invalid = s.FixedGridLocalHamiltonian(
        s.LocalHamiltonian(layout, (invalid_term,)),
        jnp.asarray([0.0, 0.1]),
        jnp.ones((1, 1)),
    )
    prepared = s.prepare_local_hamiltonian_evolution(invalid)
    result = s.solve_local_hamiltonian_evolution(
        prepared,
        jnp.asarray([1.0, 0.0], dtype=jnp.complex128),
    )
    assert not bool(result.successful)
    assert int(result.status) == int(s.LocalHamiltonianEvolutionStatus.INVALID_INPUT)
