import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.models import PortHamiltonianVectorField


def test_port_hamiltonian_structural_matrices_are_exactly_valid():
    model = PortHamiltonianVectorField(
        state_size=4,
        energy_width=10,
        energy_depth=2,
        initial_damping=0.03,
        key=jr.key(0),
    )
    interconnection = model.interconnection_matrix()
    dissipation = model.dissipation_matrix()

    assert jnp.array_equal(interconnection, -interconnection.T)
    assert jnp.array_equal(jnp.diag(interconnection), jnp.zeros(4))
    assert jnp.allclose(dissipation, dissipation.T)
    assert jnp.min(jnp.linalg.eigvalsh(dissipation)) > 0.0


def test_unforced_port_hamiltonian_energy_rate_equals_negative_dissipation():
    model = PortHamiltonianVectorField(
        state_size=3,
        energy_width=9,
        energy_depth=2,
        initial_damping=0.02,
        key=jr.key(2),
    )
    states = jr.normal(jr.key(3), (7, 3))
    rates = jax.jit(jax.vmap(model.energy_rate))(states)
    dissipations = jax.vmap(model.dissipation_rate)(states)

    assert jnp.all(rates <= 1e-10)
    assert jnp.allclose(rates, -dissipations, atol=1e-10, rtol=1e-10)


def test_conservative_port_hamiltonian_has_zero_instantaneous_energy_rate():
    model = PortHamiltonianVectorField(
        state_size=4,
        energy_width=8,
        energy_depth=2,
        dissipative=False,
        key=jr.key(4),
    )
    states = jr.normal(jr.key(5), (6, 4))
    rates = jax.vmap(model.energy_rate)(states)

    assert jnp.allclose(rates, jnp.zeros_like(rates), atol=1e-10, rtol=0.0)


def test_controlled_port_hamiltonian_obeys_the_power_balance():
    model = PortHamiltonianVectorField(
        state_size=3,
        control_size=2,
        energy_width=8,
        energy_depth=2,
        initial_damping=0.01,
        key=jr.key(6),
    )
    state = jnp.asarray([0.4, -0.2, 0.7])
    control = jnp.asarray([0.3, -0.5])
    gradient = model.energy_gradient(state)
    supplied_power = jnp.vdot(gradient, model.control_matrix @ control).real

    assert jnp.allclose(
        model.energy_rate((state, control)),
        supplied_power - model.dissipation_rate(state),
        atol=1e-10,
        rtol=1e-10,
    )
    assert model((state, control)).shape == (3,)


def test_parameter_updates_cannot_break_hamiltonian_matrix_invariants():
    model = PortHamiltonianVectorField(
        state_size=3,
        energy_width=7,
        energy_depth=2,
        key=jr.key(8),
    )
    updates = jax.tree.map(
        lambda leaf: (
            0.1 * jr.normal(jr.key(9), leaf.shape) if eqx.is_array(leaf) else None
        ),
        model,
    )
    updated = eqx.apply_updates(model, updates)
    interconnection = updated.interconnection_matrix()
    dissipation = updated.dissipation_matrix()

    assert jnp.array_equal(interconnection, -interconnection.T)
    assert jnp.min(jnp.linalg.eigvalsh(dissipation)) > 0.0
