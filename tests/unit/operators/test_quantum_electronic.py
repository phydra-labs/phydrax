import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


class _Hydrogenic(eqx.Module):
    alpha: jax.Array

    def __call__(self, electrons):
        radius = jnp.sqrt(jnp.sum(electrons[0] ** 2))
        return phx.operators.LogAmplitude(-self.alpha * radius)


class _Constant(eqx.Module):
    offset: jax.Array

    def __call__(self, electrons):
        return phx.operators.LogAmplitude(self.offset + 0.0 * jnp.sum(electrons))


def _structure(charges, positions, *, name="molecule", cell=None, periodic_axes=None):
    scale = phx.atomistic.AtomisticScaleContract("bohr", "hartree")
    return phx.atomistic.AtomicStructure(
        jnp.asarray(charges, dtype=jnp.int32),
        jnp.asarray(positions, dtype=jnp.float64),
        jnp.ones((len(charges),), dtype=jnp.float64),
        scale,
        cell=cell,
        periodic_axes=periodic_axes,
        name=name,
    )


def test_h_and_h2_coulomb_values_and_analytic_hydrogen_local_energy():
    hydrogen = _structure([1], [[0.0, 0.0, 0.0]], name="H")
    hamiltonian = phx.operators.ElectronicCoulombHamiltonian(hydrogen, 1)
    coordinate = jnp.asarray([[[2.0, 0.0, 0.0]]], dtype=jnp.float64)

    coulomb = phx.operators.evaluate_local_operator(
        _Constant(jnp.asarray(0.0)), hamiltonian, coordinate
    )
    exact = phx.operators.evaluate_local_operator(
        _Hydrogenic(jnp.asarray(1.0)), hamiltonian, coordinate
    )
    assert coulomb.valid[0]
    assert jnp.allclose(coulomb.value[0], -0.5)
    assert jnp.allclose(exact.value[0], -0.5, rtol=1e-11, atol=1e-11)

    hydrogen_molecule = _structure(
        [1, 1], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], name="H2"
    )
    h2_hamiltonian = phx.operators.ElectronicCoulombHamiltonian(
        hydrogen_molecule, 1
    )
    h2 = phx.operators.evaluate_local_operator(
        _Constant(jnp.asarray(0.0)),
        h2_hamiltonian,
        jnp.zeros((1, 1, 3), dtype=jnp.float64),
    )
    assert h2.valid[0]
    assert jnp.allclose(h2.value[0], -1.5)


def test_helium_coulomb_symmetry_translation_and_rotation_invariance():
    helium = _structure([2], [[0.0, 0.0, 0.0]], name="He")
    hamiltonian = phx.operators.ElectronicCoulombHamiltonian(helium, 2)
    electrons = jnp.asarray(
        [[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=jnp.float64
    )
    model = _Constant(jnp.asarray(0.0))
    baseline = phx.operators.evaluate_local_operator(model, hamiltonian, electrons)
    exchanged = phx.operators.evaluate_local_operator(
        model, hamiltonian, electrons[:, ::-1]
    )
    assert jnp.allclose(baseline.value, -3.5)
    assert jnp.allclose(exchanged.value, baseline.value)

    rotation = jnp.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    translated_helium = _structure([2], [[2.0, -3.0, 0.5]], name="He-shifted")
    transformed = electrons @ rotation.T + jnp.asarray([2.0, -3.0, 0.5])
    transformed_hamiltonian = phx.operators.ElectronicCoulombHamiltonian(
        translated_helium, 2
    )
    transformed_value = phx.operators.evaluate_local_operator(
        model, transformed_hamiltonian, transformed
    )
    assert jnp.allclose(transformed_value.value, baseline.value)


def test_exact_and_chunked_kinetic_trace_match_with_jit_vjp_and_gradient():
    hydrogen = _structure([1], [[0.0, 0.0, 0.0]], name="H")
    exact = phx.operators.ElectronicCoulombHamiltonian(
        hydrogen,
        1,
        kinetic=phx.operators.ElectronicKineticPolicy(trace_method="exact"),
    )
    chunked = phx.operators.ElectronicCoulombHamiltonian(
        hydrogen,
        1,
        kinetic=phx.operators.ElectronicKineticPolicy(
            trace_method="chunked-exact", coordinate_chunk_size=2
        ),
    )
    coordinate = jnp.asarray([[[1.3, -0.2, 0.4]]], dtype=jnp.float64)
    model = _Hydrogenic(jnp.asarray(0.8))
    exact_value = jax.jit(phx.operators.evaluate_local_operator)(
        model, exact, coordinate
    )
    chunked_value = jax.jit(phx.operators.evaluate_local_operator)(
        model, chunked, coordinate
    )
    assert jnp.allclose(exact_value.value, chunked_value.value, rtol=1e-11, atol=1e-11)
    assert jnp.array_equal(exact_value.work_count, jnp.asarray([3]))
    assert exact_value.compute_dtype == "float64"
    assert exact_value.method_id != chunked_value.method_id

    value, pullback = jax.vjp(
        lambda alpha: phx.operators.evaluate_local_operator(
            _Hydrogenic(alpha), exact, coordinate
        ).value,
        model.alpha,
    )
    cotangent = pullback(jnp.ones_like(value))[0]
    direct = jax.grad(
        lambda alpha: jnp.real(
            phx.operators.evaluate_local_operator(
                _Hydrogenic(alpha), exact, coordinate
            ).value[0]
        )
    )(model.alpha)
    assert jnp.isfinite(cotangent)
    assert jnp.allclose(cotangent, direct)


def test_coincident_singularities_are_invalid_and_never_clipped():
    hydrogen = _structure([1], [[0.0, 0.0, 0.0]], name="H")
    hamiltonian = phx.operators.ElectronicCoulombHamiltonian(hydrogen, 1)
    estimate = phx.operators.evaluate_local_operator(
        _Hydrogenic(jnp.asarray(1.0)),
        hamiltonian,
        jnp.zeros((1, 1, 3), dtype=jnp.float64),
    )
    assert not estimate.valid[0]
    assert estimate.status[0] == int(
        phx.operators.LocalOperatorStatus.SINGULAR_CONFIGURATION
    )
    assert jnp.isnan(estimate.value[0])

    coincident_nuclei = _structure(
        [1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], name="singular-H2"
    )
    singular_hamiltonian = phx.operators.ElectronicCoulombHamiltonian(
        coincident_nuclei, 1
    )
    nuclear = phx.operators.evaluate_local_operator(
        _Constant(jnp.asarray(0.0)),
        singular_hamiltonian,
        jnp.asarray([[[1.0, 0.0, 0.0]]]),
    )
    assert nuclear.status[0] == int(
        phx.operators.LocalOperatorStatus.SINGULAR_CONFIGURATION
    )


def test_periodic_electronic_systems_and_stochastic_trace_are_explicitly_rejected():
    periodic = _structure(
        [1],
        [[0.0, 0.0, 0.0]],
        name="periodic-H",
        cell=jnp.eye(3),
        periodic_axes=jnp.asarray([True, False, False]),
    )
    with pytest.raises(ValueError, match="finite nonperiodic"):
        phx.operators.ElectronicCoulombHamiltonian(periodic, 1)
    with pytest.raises(ValueError, match="exact"):
        phx.operators.ElectronicKineticPolicy(trace_method="stochastic")


def test_initial_walkers_and_state_dependent_proposal_are_replayable_and_corrected():
    hydrogen = _structure([1], [[0.0, 0.0, 0.0]], name="H")
    first = phx.operators.electronic_initial_walkers(jr.key(7), hydrogen, 1, 8)
    second = phx.operators.electronic_initial_walkers(jr.key(7), hydrogen, 1, 8)
    assert first.shape == (8, 1, 3)
    assert first.dtype == jnp.float64
    assert jnp.array_equal(first, second)

    proposal = phx.operators.harmonic_mean_electron_proposal(
        hydrogen, 1, step_size=0.3
    )
    current = jnp.asarray([[1.0, 0.0, 0.0]])
    proposed = jnp.asarray([[2.0, 0.0, 0.0]])
    forward = proposal.log_prob(proposed, current)
    reverse = proposal.log_prob(current, proposed)
    assert jnp.isfinite(forward)
    assert jnp.isfinite(reverse)
    assert not jnp.allclose(forward, reverse)
