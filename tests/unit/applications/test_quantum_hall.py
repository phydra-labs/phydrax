from fractions import Fraction

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.operators.quantum import (
    PeriodicPrincipalLayerLeadPlan,
    prepare_periodic_lead_embedding,
)
from phydrax.sampling import SingleElectronSphereProposal
from phydrax.units import ANGSTROM, ELECTRONVOLT


_ELECTRONVOLT_JOULE = 1.602_176_634e-19


def _energy_scale():
    return qh.QuantumHallEnergyScale(
        ELECTRONVOLT,
        _ELECTRONVOLT_JOULE,
        "electronvolt",
    )


def test_haldane_model_is_hermitian_and_has_two_bands():
    prepared = qh.HaldaneModelPlan(
        1.0,
        0.1,
        0.2,
        np.pi / 2.0,
        1.0,
        ELECTRONVOLT,
        ANGSTROM,
    ).prepare()
    evaluation = prepared.evaluate(np.asarray(((0.0, 0.0), (1.0 / 3.0, 1.0 / 3.0))))

    np.testing.assert_allclose(
        evaluation.hamiltonians,
        np.conj(np.swapaxes(evaluation.hamiltonians, -1, -2)),
        atol=1.0e-12,
    )
    assert evaluation.hamiltonians.shape == (2, 2, 2)
    assert bool(evaluation.successful)


def test_kane_mele_rashba_phase_has_nontrivial_time_reversal_invariant():
    result = qh.evaluate_kane_mele_topology(
        qh.KaneMeleModelPlan(
            1.0,
            0.0,
            0.2,
            0.05,
            1.0,
            ELECTRONVOLT,
            ANGSTROM,
        ),
        mesh_shape=(8, 8),
    )

    assert int(result.z2.invariant) == 1
    assert float(result.z2.endpoint_kramers_residual) < 1.0e-10
    assert bool(result.successful)


def test_projected_sphere_uses_direct_lz_sector_and_matrix_free_operator():
    sphere = qh.HaldaneSpherePlan(
        2,
        qh.MonopoleLandauLevel(3, 0, qh.SPIN_POLARIZED_ELECTRON),
        "fermion",
        _energy_scale(),
        filling=Fraction(1, 3),
        shift=3,
    )
    pseudopotentials = qh.HaldanePseudopotentialPlan(
        sphere,
        {1: 1.0, 3: 0.0},
        "v1-parent",
    )
    prepared = qh.prepare_haldane_sphere_hamiltonian(
        pseudopotentials,
        twice_projection=0,
    )

    assert prepared.many_body.dimension == 2
    assert not prepared.many_body.operator.capabilities.materialize
    np.testing.assert_array_equal(prepared.many_body.twice_projections, 0)
    vector = jnp.asarray((1.0, -1.0j))
    assert prepared.many_body.operator.mv(vector).shape == vector.shape


def test_laughlin_amplitude_is_antisymmetric_for_odd_exponent():
    amplitude = qh.LaughlinSphereAmplitude(3, 3)
    configuration = jnp.asarray(((0.8, -0.2), (1.4, 0.7), (2.0, -1.1)))
    exchanged = configuration[jnp.asarray((1, 0, 2))]
    first = amplitude(configuration)
    second = amplitude(exchanged)
    ratio = jnp.exp(second.log_abs - first.log_abs) * second.phase * jnp.conj(first.phase)

    np.testing.assert_allclose(ratio, -1.0, atol=1.0e-10)
    assert bool(first.valid & first.nonzero)


def test_monopole_attention_is_antisymmetric_and_vmc_prepares():
    sphere = qh.HaldaneSpherePlan(
        2,
        qh.MonopoleLandauLevel(3, 0, qh.SPIN_POLARIZED_ELECTRON),
        "fermion",
        _energy_scale(),
    )
    prepared = qh.prepare_landau_level_mixing_vmc(
        qh.LandauLevelMixingVMCPlan(
            sphere,
            0.5,
            chain_count=4,
            hidden_dimension=4,
            layer_count=1,
            determinant_count=1,
        ),
        jr.key(3),
    )
    configuration = prepared.problem.initial_configurations[0]
    exchanged = configuration[jnp.asarray((1, 0))]
    first = prepared.model(configuration)
    second = prepared.model(exchanged)
    ratio = jnp.exp(second.log_abs - first.log_abs) * second.phase * jnp.conj(first.phase)

    np.testing.assert_allclose(ratio, -1.0, atol=1.0e-8)
    assert prepared.problem.initial_configurations.shape == (4, 2, 2)


def test_sphere_proposal_is_supported_and_hastings_symmetric():
    configuration = jnp.asarray(((0.8, -0.2), (1.4, 0.7)))
    move = SingleElectronSphereProposal(2, 0.3).propose(jr.key(11), configuration)

    assert bool(move.valid)
    np.testing.assert_allclose(move.log_forward, move.log_reverse, atol=1.0e-12)
    assert float(move.payload.angular_displacement) <= 0.3


def test_subband_form_factor_has_exact_zero_momentum_limit():
    positions = np.linspace(-5.0e-9, 5.0e-9, 101)
    density = np.full_like(positions, 1.0 / (positions[-1] - positions[0]))
    plan = qh.SubbandCoulombFormFactorPlan(
        positions,
        density,
        normalization_tolerance=1.0e-12,
    )
    result = qh.evaluate_subband_coulomb_form_factor(
        plan,
        np.asarray((0.0, 1.0e8, 2.0e8)),
    )

    np.testing.assert_allclose(result.form_factors[0], 1.0, atol=1.0e-12)
    assert np.all(np.diff(np.asarray(result.form_factors)) <= 0.0)
    assert bool(result.successful)


def test_periodic_chain_lead_is_causal_and_has_psd_broadening():
    plan = PeriodicPrincipalLayerLeadPlan(
        np.asarray(((0.0,),)),
        np.asarray(((-1.0,),)),
        tolerance=1.0e-10,
        maximum_iterations=80,
    )
    result = prepare_periodic_lead_embedding(plan, 0.0, broadening=1.0e-5)

    assert bool(result.causal)
    assert bool(result.broadening_positive_semidefinite)
    assert bool(result.successful)
