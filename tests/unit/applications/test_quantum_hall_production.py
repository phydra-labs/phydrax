from fractions import Fraction

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax import solver
from phydrax.applications import quantum_hall as qh
from phydrax.operators.periodic import (
    evaluate_periodic_sheet_hall,
    PeriodicSheetHallPlan,
)
from phydrax.operators.quantum import (
    AbelianGroup,
    many_body_twist_chern,
    PeriodicPrincipalLayerLeadPlan,
)
from phydrax.operators.quantum.lattice import (
    FixedAbelianChargeBasis,
    SectorBasisResourcePolicy,
)
from phydrax.solver import UniformVUMPSPolicy
from phydrax.tensor_network import (
    su2_wigner_3j,
    UniformAbelianMatrixProductOperator,
    UniformAbelianMatrixProductState,
    UniformMatrixProductOperator,
    UniformMatrixProductState,
)
from phydrax.units import ELECTRONVOLT


_ELECTRONVOLT_JOULE = 1.602_176_634e-19


def _scale():
    return qh.QuantumHallEnergyScale(
        ELECTRONVOLT,
        _ELECTRONVOLT_JOULE,
        "electronvolt",
    )


def test_landau_manifold_separates_physical_flux_from_orbital_spin():
    lowest = qh.MonopoleLandauLevel(5, 0, qh.SPIN_POLARIZED_ELECTRON)
    second = qh.MonopoleLandauLevel(5, 1, qh.SPIN_POLARIZED_ELECTRON)

    assert lowest.twice_orbital_spin == 5
    assert lowest.orbital_count == 6
    assert second.twice_orbital_spin == 7
    assert second.orbital_count == 8
    assert second.twice_monopole_strength == lowest.twice_monopole_strength


def test_higher_landau_coulomb_channels_are_finite_and_complete():
    manifold = qh.MonopoleLandauLevel(5, 1, qh.SPIN_POLARIZED_ELECTRON)
    sphere = qh.HaldaneSpherePlan(
        2,
        manifold,
        "fermion",
        _scale(),
        filling=Fraction(1, 3),
        shift=1,
    )
    result = qh.coulomb_haldane_pseudopotentials(sphere)

    assert tuple(channel for channel, _ in result.relative_channels) == (1, 3, 5, 7)
    assert np.all(np.isfinite(tuple(value for _, value in result.relative_channels)))


def test_modular_charge_basis_selects_magnetic_momentum_without_enumeration():
    basis = FixedAbelianChargeBasis(
        ("m0", "m1", "m2"),
        (((0, 0), (1, 0)), ((0, 0), (1, 1)), ((0, 0), (1, 2))),
        ("particle", "momentum"),
        (1, 1),
        AbelianGroup((None, 3)),
        resources=SectorBasisResourcePolicy(
            maximum_dimension=8,
            maximum_table_bytes=1_000_000,
        ),
    )

    assert basis.dimension == 1
    np.testing.assert_array_equal(basis.coordinate(0), np.asarray((0, 1, 0)))
    assert int(basis.rank(np.asarray((0, 1, 0)))) == 0


def test_effective_mixing_is_hermitian_and_refuses_singular_denominators():
    active = qh.MonopoleLandauLevel(3, 0, qh.SPIN_POLARIZED_ELECTRON)
    virtual = qh.MonopoleLandauLevel(3, 1, qh.SPIN_POLARIZED_ELECTRON)
    plan = qh.LandauLevelMixingEffectivePlan(
        active,
        (virtual,),
        ("m1", "m3"),
        np.asarray((2.0,)),
        np.asarray(((1.0 + 1.0j, 0.5),)),
        kappa=0.2,
        three_body_ids=("m3",),
        three_body_vertices=np.asarray(((0.25,),)),
    )
    result = qh.evaluate_effective_landau_level_interaction(plan)

    np.testing.assert_allclose(
        result.two_body_correction, np.conj(result.two_body_correction.T)
    )
    assert bool(result.successful)


def test_many_body_twist_bundle_has_zero_chern_for_constant_manifold():
    states = np.zeros((3, 3, 2, 1), dtype=np.complex128)
    states[..., 0, 0] = 1.0
    result = many_body_twist_chern(states, np.ones((3, 3)))

    assert int(result.nearest_integer) == 0
    np.testing.assert_allclose(result.normalization_residual, 0.0)
    assert bool(result.successful)


def test_sheet_hall_response_recovers_one_conductance_quantum():
    charge = 1.602_176_634e-19
    plan = PeriodicSheetHallPlan(
        np.asarray(((-1.0,), (-1.0,))),
        np.asarray(((1.0,), (1.0,))),
        np.asarray((0.5, 0.5)),
        charge_coulomb=charge,
        chemical_potential_joule=0.0,
        temperature_kelvin=0.0,
    )
    result = evaluate_periodic_sheet_hall(plan)

    np.testing.assert_allclose(result.effective_chern, 1.0)
    np.testing.assert_allclose(
        result.sheet_conductance_siemens,
        charge**2 / 6.626_070_15e-34,
    )
    assert bool(result.successful)


def test_wigner_three_j_matches_spin_half_singlet_value():
    np.testing.assert_allclose(
        su2_wigner_3j(1, 1, 0, 1, -1, 0),
        1.0 / np.sqrt(2.0),
    )


def test_explicit_multilevel_and_torus_sectors_lower_matrix_free():
    manifold = qh.MonopoleLandauLevel(1, 0, qh.SPIN_POLARIZED_ELECTRON)
    term = qh.ProjectedOrbitalTerm((0,), (0,), 1.0)
    prepared = qh.prepare_multi_landau_level_sphere(
        qh.MultiLandauLevelSpherePlan(
            1,
            (manifold,),
            qh.HallChargeSector({"particle-number": 1, "twice-projection": -1}),
            (term,),
        )
    )
    np.testing.assert_allclose(
        prepared.operator.mv(jnp.ones((1,), dtype=jnp.complex128)), 1.0
    )

    torus = qh.prepare_torus_hamiltonian(
        qh.TorusProjectedPlan(
            1,
            qh.MagneticTorusGeometry(2),
            0,
            (qh.TorusOrbitalTerm((0,), (0,), 1.0, (0, 0)),),
        )
    )
    assert torus.basis.dimension == 1
    np.testing.assert_allclose(
        torus.operator.mv(jnp.ones((1,), dtype=jnp.complex128)), 1.0
    )


def test_infinite_cylinder_reuses_uniform_vumps_with_charge_identity():
    state = UniformMatrixProductState(
        (jnp.asarray((0.0, 1.0), dtype=jnp.complex128)[None, :, None],)
    )
    operator = UniformMatrixProductOperator(
        (jnp.diag(jnp.asarray((1.0, -1.0), dtype=jnp.complex128))[None, :, :, None],)
    )
    group = AbelianGroup((None,))
    charges = (((0,), (1,)),)
    result = qh.solve_infinite_hall_cylinder(
        qh.InfiniteHallCylinderPlan(
            1,
            1,
            8.0,
            UniformAbelianMatrixProductState(
                state, group, ("particle-number",), charges, (1,)
            ),
            UniformAbelianMatrixProductOperator(
                operator, group, ("particle-number",), charges
            ),
            UniformVUMPSPolicy(maximum_iterations=3, gradient_step=0.05),
        )
    )

    assert bool(result.charge_exact)
    assert bool(result.successful)


def test_localized_transport_preserves_input_and_propagates_solve_evidence():
    rates = np.asarray(((7.0, 1.0), (1.0, 7.0)))
    original = rates.copy()
    plan = qh.LocalizedHallNetworkPlan(
        rates,
        np.asarray(((1.0, 0.0),)),
        np.asarray(((0.0, 1.0),)),
        np.ones(2),
        "two-site-network",
    )
    result = qh.solve_localized_hall_transport(plan)

    np.testing.assert_array_equal(rates, original)
    np.testing.assert_allclose(result.populations, (2.0, 1.0))
    assert bool(result.linear_solve.successful)
    assert bool(result.successful)


def test_open_transport_retains_unsymmetrized_failed_density_evidence():
    liouvillian = np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
        ),
        dtype=np.complex128,
    )
    liouvillian[1, 3] = -1.0j
    result = qh.solve_open_hall_transport(
        qh.OpenHallTransportPlan(
            liouvillian,
            np.eye(2, dtype=np.complex128)[None, ...],
            "nonhermitian-test",
        )
    )

    np.testing.assert_allclose(result.density_matrix[0, 1], 1.0j)
    np.testing.assert_allclose(result.density_matrix[1, 0], 0.0)
    assert bool(result.linear_solve.successful)
    assert not bool(result.successful)


def _three_terminal_problem():
    lead = PeriodicPrincipalLayerLeadPlan(
        np.zeros((1, 1)),
        -np.ones((1, 1)),
        maximum_iterations=4,
    )
    contacts = tuple(
        solver.PeriodicLeadContactPlan(lead, np.ones((1, 1)), f"contact-{index}")
        for index in range(3)
    )
    return solver.MultiTerminalCoherentProblem(
        np.zeros((1, 1)),
        np.ones((1, 1)),
        contacts,
        np.asarray((0.0,)),
        np.asarray((1.0e-21,)),
        np.asarray((-1.0e-21, 1.0e-21, 0.0)),
        np.full((3,), 300.0),
    )


def _coherent_fixture(problem, *, problem_id=None):
    transmission = jnp.asarray((((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)),))
    zeros = jnp.zeros((3,))
    return solver.MultiTerminalCoherentResult(
        transmission,
        zeros,
        zeros,
        zeros,
        zeros,
        jnp.asarray(0.0),
        jnp.asarray(0.0),
        jnp.asarray((True,)),
        jnp.asarray(True),
        problem.problem_id if problem_id is None else problem_id,
        "coherent-fixture",
    )


def test_transport_probes_reject_foreign_and_singular_contact_rosters():
    problem = _three_terminal_problem()
    foreign = _coherent_fixture(problem, problem_id="foreign-problem")
    with pytest.raises(ValueError, match="identities"):
        solver.solve_dephasing_probes(
            problem,
            foreign,
            solver.TransportProbePlan((2,)),
        )

    coherent = _coherent_fixture(problem)
    with pytest.raises(ValueError, match="physical contact"):
        solver.solve_dephasing_probes(
            problem,
            coherent,
            solver.TransportProbePlan((0, 1, 2)),
        )


def test_voltage_probe_reports_native_linear_solve_status():
    problem = _three_terminal_problem()
    result = solver.solve_voltage_probes(
        problem,
        _coherent_fixture(problem),
        solver.TransportProbePlan((2,), residual_tolerance=1.0e-8),
    )

    assert int(result.linear_solve_status) == 0
    assert result.problem_id == problem.problem_id
    assert bool(result.successful)


def test_scba_rejects_nonvector_local_coupling_roster():
    energies = np.linspace(-1.0, 1.0, 8)
    matrices = np.broadcast_to(np.eye(2), (8, 2, 2))
    with pytest.raises(ValueError, match="coupling"):
        solver.KeldyshSCBAProblem(
            energies,
            matrices,
            np.zeros_like(matrices),
            np.zeros_like(matrices),
            np.ones((2, 2)),
            phonon_bins=1,
            phonon_occupation=0.0,
        )


def test_multiterminal_coherent_maps_the_energy_grid():
    problem = _three_terminal_problem()
    result = solver.solve_multiterminal_coherent(problem)

    assert result.transmissions.shape == (1, 3, 3)
    assert result.point_successful.shape == (1,)
    assert result.problem_id == problem.problem_id
