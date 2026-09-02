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

    hydrogen_molecule = _structure([1, 1], [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], name="H2")
    h2_hamiltonian = phx.operators.ElectronicCoulombHamiltonian(hydrogen_molecule, 1)
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
    electrons = jnp.asarray([[[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=jnp.float64)
    model = _Constant(jnp.asarray(0.0))
    baseline = phx.operators.evaluate_local_operator(model, hamiltonian, electrons)
    exchanged = phx.operators.evaluate_local_operator(
        model, hamiltonian, electrons[:, ::-1]
    )
    assert jnp.allclose(baseline.value, -3.5)
    assert jnp.allclose(exchanged.value, baseline.value)

    rotation = jnp.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
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
    exact_value = jax.jit(phx.operators.evaluate_local_operator)(model, exact, coordinate)
    chunked_value = jax.jit(phx.operators.evaluate_local_operator)(
        model, chunked, coordinate
    )
    assert jnp.allclose(exact_value.value, chunked_value.value, rtol=1e-11, atol=1e-11)
    assert jnp.array_equal(exact_value.work_count, jnp.asarray([3]))
    assert exact_value.compute_dtype == "float64"
    assert exact_value.method_id != chunked_value.method_id

    value, pullback = jax.vjp(
        lambda alpha: (
            phx.operators.evaluate_local_operator(
                _Hydrogenic(alpha), exact, coordinate
            ).value
        ),
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


def test_electronic_scales_require_explicit_bohr_hartree_reference_conversion():
    bad_scale = phx.atomistic.AtomisticScaleContract("angstrom", "electronvolt")
    bad_structure = phx.atomistic.AtomicStructure(
        jnp.asarray([1], dtype=jnp.int32),
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.ones((1,), dtype=jnp.float64),
        bad_scale,
        name="mis-scaled-H",
    )
    with pytest.raises(ValueError, match="Bohr.*Hartree|physical conversion"):
        phx.operators.ElectronicCoulombHamiltonian(bad_structure, 1)

    physical_scale = phx.atomistic.AtomisticScaleContract(
        "angstrom",
        "electronvolt",
        length_to_reference=1.8897261254578281,
        energy_to_reference=0.03674932217565499,
    )
    physical_structure = phx.atomistic.AtomicStructure(
        jnp.asarray([1], dtype=jnp.int32),
        jnp.zeros((1, 3), dtype=jnp.float64),
        jnp.ones((1,), dtype=jnp.float64),
        physical_scale,
        name="scaled-H",
    )
    estimate = phx.operators.evaluate_local_operator(
        _Constant(jnp.asarray(0.0)),
        phx.operators.ElectronicCoulombHamiltonian(physical_structure, 1),
        jnp.asarray([[[1.0, 0.0, 0.0]]], dtype=jnp.float64),
    )
    assert estimate.valid[0]
    assert jnp.allclose(estimate.value[0], -14.3996454784255, rtol=1e-12)


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

    proposal = phx.operators.harmonic_mean_electron_proposal(hydrogen, 1, step_size=0.3)
    current = jnp.asarray([[1.0, 0.0, 0.0]])
    proposed = jnp.asarray([[2.0, 0.0, 0.0]])
    forward = proposal.log_prob(proposed, current)
    reverse = proposal.log_prob(current, proposed)
    assert jnp.isfinite(forward)
    assert jnp.isfinite(reverse)
    assert not jnp.allclose(forward, reverse)


def _with_antisymmetric_hermitian_pair(two_body, bra, ket, value):
    for first, second, bra_phase in (
        (bra[0], bra[1], 1),
        (bra[1], bra[0], -1),
    ):
        for third, fourth, ket_phase in (
            (ket[0], ket[1], 1),
            (ket[1], ket[0], -1),
        ):
            phase = bra_phase * ket_phase
            two_body = two_body.at[first, second, third, fourth].set(phase * value)
            two_body = two_body.at[third, fourth, first, second].set(
                phase * jnp.conj(value)
            )
    return two_body


def _explicit_two_body_matrix_element(two_body, bra, ket):
    bra_state = tuple(bool(value) for value in bra)
    orbital_count = len(bra_state)
    element = jnp.asarray(0.0, dtype=two_body.dtype)
    for first in range(orbital_count):
        for second in range(orbital_count):
            for third in range(orbital_count):
                for fourth in range(orbital_count):
                    state = list(bool(value) for value in ket)
                    phase = 1
                    valid = True
                    for orbital, occupied_after in (
                        (third, False),
                        (fourth, False),
                        (second, True),
                        (first, True),
                    ):
                        if state[orbital] == occupied_after:
                            valid = False
                            break
                        phase *= (-1) ** sum(state[:orbital])
                        state[orbital] = occupied_after
                    if valid and tuple(state) == bra_state:
                        element = (
                            element
                            + 0.25 * phase * two_body[first, second, third, fourth]
                        )
    return element


def test_finite_ewald_excludes_only_same_particle_zero_image_self_terms():
    self_energy, self_evidence = phx.operators.periodic_coulomb_energy(
        jnp.asarray([[0.125, 0.25, 0.375]]),
        jnp.asarray([1.0]),
        jnp.eye(3),
        real_image_radius=0,
        reciprocal_radius=0,
        screening=1.0,
        uniform_background=True,
    )
    assert jnp.isfinite(self_energy)
    assert bool(self_evidence.valid)

    coincident_energy, coincident_evidence = phx.operators.periodic_coulomb_energy(
        jnp.asarray([[0.125, 0.25, 0.375], [0.125, 0.25, 0.375]]),
        jnp.asarray([1.0, -1.0]),
        jnp.eye(3),
        real_image_radius=0,
        reciprocal_radius=0,
        screening=1.0,
    )
    assert not jnp.isfinite(coincident_energy)
    assert not bool(coincident_evidence.valid)


@pytest.mark.parametrize("real_image_radius", [0, 1, 3])
def test_finite_ewald_rejects_periodically_equivalent_particles_at_every_cutoff(
    real_image_radius,
):
    energy, evidence = phx.operators.periodic_coulomb_energy(
        jnp.asarray([[0.125, 0.25, 0.375], [2.125, -2.75, 4.375]]),
        jnp.asarray([1.0, -1.0]),
        jnp.eye(3),
        real_image_radius=real_image_radius,
        reciprocal_radius=0,
        screening=1.0,
    )

    assert not jnp.isfinite(energy)
    assert not bool(evidence.valid)


def test_finite_ewald_canonicalizes_wrapped_fractional_positions():
    positions = jnp.asarray([[0.125, 0.25, 0.375], [0.625, 0.75, 0.875]])
    shifts = jnp.asarray([[2.0, -3.0, 4.0], [-1.0, 5.0, -2.0]])
    charges = jnp.asarray([1.0, -1.0])
    cell = jnp.asarray([[1.5, 0.1, 0.0], [0.0, 1.25, 0.2], [0.1, 0.0, 1.75]])
    baseline_energy, baseline_evidence = phx.operators.periodic_coulomb_energy(
        positions,
        charges,
        cell,
        real_image_radius=1,
        reciprocal_radius=1,
        screening=0.8,
    )
    wrapped_energy, wrapped_evidence = phx.operators.periodic_coulomb_energy(
        positions + shifts,
        charges,
        cell,
        real_image_radius=1,
        reciprocal_radius=1,
        screening=0.8,
    )

    assert jnp.allclose(wrapped_energy, baseline_energy)
    assert eqx.tree_equal(wrapped_evidence, baseline_evidence)


def test_integral_hamiltonian_validity_includes_two_body_bra_ket_hermiticity():
    one_body = jnp.zeros((4, 4), dtype=jnp.complex64)
    two_body = jnp.zeros((4, 4, 4, 4), dtype=jnp.complex64)
    two_body = two_body.at[0, 1, 2, 3].set(1.0)
    two_body = two_body.at[1, 0, 2, 3].set(-1.0)
    two_body = two_body.at[0, 1, 3, 2].set(-1.0)
    two_body = two_body.at[1, 0, 3, 2].set(1.0)

    hamiltonian = phx.operators.ElectronicIntegralHamiltonian(one_body, two_body)

    assert jnp.allclose(hamiltonian.antisymmetry_residual, 0.0)
    assert hamiltonian.hermiticity_residual > 0.0
    assert not bool(hamiltonian.valid)


def test_double_connections_use_exact_spectator_dependent_fermionic_parity():
    one_body = jnp.zeros((6, 6), dtype=jnp.complex64)
    two_body = jnp.zeros((6, 6, 6, 6), dtype=jnp.complex64)
    positive_value = jnp.asarray(2.0 + 0.25j, dtype=jnp.complex64)
    negative_value = jnp.asarray(3.0 - 0.5j, dtype=jnp.complex64)
    two_body = _with_antisymmetric_hermitian_pair(
        two_body, (4, 5), (0, 1), positive_value
    )
    two_body = _with_antisymmetric_hermitian_pair(
        two_body, (4, 5), (0, 2), negative_value
    )
    hamiltonian = phx.operators.ElectronicIntegralHamiltonian(one_body, two_body)
    occupation = jnp.asarray([True, True, True, True, False, False])
    connections = hamiltonian.connected(occupation)

    assert bool(hamiltonian.valid)
    for removed, signed_value in (
        ((0, 1), positive_value),
        ((0, 2), -negative_value),
    ):
        target = occupation.at[jnp.asarray(removed)].set(False)
        target = target.at[jnp.asarray([4, 5])].set(True)
        matching = jnp.all(connections.configurations == target, axis=1)
        assert int(jnp.sum(matching)) == 1
        connected_element = connections.matrix_elements[jnp.argmax(matching)]
        explicit_element = _explicit_two_body_matrix_element(two_body, target, occupation)
        assert jnp.allclose(explicit_element, signed_value)
        assert jnp.allclose(connected_element, explicit_element)
