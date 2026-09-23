import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _dipole_unit():
    return phx.units.derived_unit(
        "e*bohr-excited-test",
        ((phx.units.ELEMENTARY_CHARGE, 1), (phx.units.BOHR, 1)),
    )


def _tda_manifold(amplitudes, energies):
    energies = jnp.asarray(energies)
    return phx.chemistry.ElectronicManifoldResult(
        energies,
        -1.0 + energies,
        phx.chemistry.TDAStateRepresentation(amplitudes),
        jnp.zeros((energies.size, 3)),
        jnp.zeros((energies.size,)),
        jnp.zeros((energies.size,)),
        True,
        "tda-test",
        "singlet",
        tuple((index,) for index in range(energies.size)),
        phx.units.HARTREE,
        _dipole_unit(),
        provider_id="tda-test-provider",
        request_id="tda-test-request",
        state_space_id="tda-test-space",
    )


def test_rpa_representation_and_analytic_tda_couplings_close_exact_small_models():
    rpa = phx.chemistry.RandomPhaseApproximationPlan(
        jnp.asarray([[1.0]]),
        jnp.asarray([[0.2]]),
        jnp.asarray([[1.0, 0.0, 0.0]]),
        -1.0,
        1,
        "rpa-test",
        phx.units.HARTREE,
        _dipole_unit(),
    ).solve()
    representation = phx.chemistry.TDAStateRepresentation(jnp.eye(2))
    response_derivative = jnp.zeros((2, 2, 1)).at[0, 1, 0].set(0.2).at[1, 0, 0].set(0.2)
    couplings, weighted = phx.chemistry.tda_derivative_couplings(
        representation,
        jnp.asarray([1.0, 2.0]),
        response_derivative,
    )
    properties = phx.chemistry.tda_property_derivatives(
        jnp.diag(jnp.asarray([1.0, 2.0])),
        representation,
        jnp.asarray([1.0, 2.0]),
        response_derivative,
        jnp.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        jnp.zeros((2, 3, 1)),
    )

    assert bool(rpa.successful)
    np.testing.assert_allclose(rpa.excitation_energies, np.sqrt(0.96), atol=2.0e-12)
    np.testing.assert_allclose(rpa.representation.symplectic_norms, 1.0, atol=2.0e-12)
    np.testing.assert_allclose(weighted[0, 1, 0], 0.2, atol=1.0e-14)
    np.testing.assert_allclose(couplings[0, 1, 0], 0.2, atol=1.0e-14)
    np.testing.assert_allclose(couplings[1, 0, 0], -0.2, atol=1.0e-14)
    assert bool(properties.successful)
    np.testing.assert_allclose(
        properties.transition_dipole_derivatives[0, :, 0],
        [0.0, -0.2, 0.0],
        atol=1.0e-13,
    )


def test_overlap_tracking_recovers_swapped_roots():
    previous = _tda_manifold(jnp.eye(2), [1.0, 2.0])
    current = _tda_manifold(jnp.asarray([[0.0, 1.0], [1.0, 0.0]]), [1.1, 1.9])
    tracking = phx.chemistry.track_excited_states(previous, current)

    assert bool(tracking.successful)
    np.testing.assert_array_equal(tracking.permutation, [1, 0])
    np.testing.assert_allclose(tracking.assigned_overlaps, 1.0, atol=1.0e-14)
    np.testing.assert_allclose(tracking.unitarity_residual, 0.0, atol=1.0e-14)


def test_meci_branching_plane_and_surface_hopping_preserve_structural_invariants():
    def crossing_surface(positions):
        x = positions[0, 0]
        gradients = jnp.zeros((2, 1, 3)).at[0, 0, 0].set(1.0).at[1, 0, 0].set(-1.0)
        coupling = jnp.asarray([[0.0, 1.0, 0.0]])
        return phx.chemistry.TwoStateSurfaceEvaluation(
            jnp.asarray([x, -x]),
            gradients,
            coupling,
            True,
            ("s0", "s1"),
            "linear-crossing",
        )

    crossing_provider = phx.chemistry.CallableTwoStateSurfaceProvider(
        crossing_surface, "linear-crossing"
    )
    crossing = phx.chemistry.MinimumEnergyCrossingPlan(
        phx.chemistry.CrossingKind.MECI,
        crossing_provider,
        gap_tolerance=1.0e-10,
        gradient_tolerance=1.0e-10,
        geometry_step=0.1,
        gap_step=0.5,
        maximum_iterations=8,
    ).run(jnp.asarray([[0.5, 0.0, 0.0]]))

    derivative_coupling = (
        jnp.zeros((2, 2, 1, 3)).at[0, 1, 0, 0].set(0.1).at[1, 0, 0, 0].set(-0.1)
    )

    def dynamics_surface(positions):
        del positions
        return phx.chemistry.NonadiabaticSurfaceEvaluation(
            jnp.asarray([0.0, 0.2]),
            jnp.zeros((2, 1, 3)),
            derivative_coupling,
            ("s0", "s1"),
            "constant-surfaces",
        )

    dynamics_provider = phx.chemistry.CallableNonadiabaticSurfaceProvider(
        dynamics_surface, "constant-surfaces"
    )
    dynamics = phx.chemistry.FewestSwitchesSurfaceHoppingPlan(
        dynamics_provider,
        jnp.asarray([1.0]),
        0.001,
        electronic_substeps=2,
    )
    initial = dynamics.initialize(
        jnp.zeros((1, 3)),
        jnp.asarray([[0.2, 0.0, 0.0]]),
        jnp.asarray([1.0 + 0.0j, 0.0j]),
        0,
        jax.random.PRNGKey(11),
    )
    step = dynamics.step(initial)

    assert bool(crossing.successful)
    np.testing.assert_allclose(crossing.energy_gap, 0.0, atol=1.0e-12)
    assert int(crossing.branching.rank) == 2
    assert bool(step.successful)
    np.testing.assert_allclose(
        jnp.real(
            jnp.vdot(
                step.state.electronic_coefficients, step.state.electronic_coefficients
            )
        ),
        1.0,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        step.state.total_energy, initial.total_energy, atol=2.0e-12
    )
