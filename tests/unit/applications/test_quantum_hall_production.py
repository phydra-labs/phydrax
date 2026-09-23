from fractions import Fraction

import jax.numpy as jnp
import numpy as np

from phydrax.applications import quantum_hall as qh
from phydrax.operators.periodic import (
    evaluate_periodic_sheet_hall,
    PeriodicSheetHallPlan,
)
from phydrax.operators.quantum import AbelianGroup, many_body_twist_chern
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
    term = qh.ProjectedOrbitalTerm((0,), (0,), 1.0, "number-zero")
    prepared = qh.prepare_multi_landau_level_sphere(
        qh.MultiLandauLevelSpherePlan(
            1,
            (manifold,),
            qh.HallChargeSector(
                {"particle-number": 1, "twice-projection": -1}
            ),
            (term,),
        )
    )
    np.testing.assert_allclose(prepared.operator.mv(jnp.ones((1,), dtype=jnp.complex128)), 1.0)

    torus = qh.prepare_torus_hamiltonian(
        qh.TorusProjectedPlan(
            1,
            qh.MagneticTorusGeometry(2),
            0,
            (qh.TorusOrbitalTerm((0,), (0,), 1.0, (0, 0), "torus-number"),),
        )
    )
    assert torus.basis.dimension == 1
    np.testing.assert_allclose(torus.operator.mv(jnp.ones((1,), dtype=jnp.complex128)), 1.0)


def test_infinite_cylinder_reuses_uniform_vumps_with_charge_identity():
    state = UniformMatrixProductState(
        (jnp.asarray((0.0, 1.0), dtype=jnp.complex128)[None, :, None],)
    )
    operator = UniformMatrixProductOperator(
        (
            jnp.diag(jnp.asarray((1.0, -1.0), dtype=jnp.complex128))[
                None, :, :, None
            ],
        )
    )
    group = AbelianGroup((None,))
    charges = (((0,), (1,)),)
    result = qh.solve_infinite_hall_cylinder(
        qh.InfiniteHallCylinderPlan(
            1,
            1,
            8.0,
            UniformAbelianMatrixProductState(state, group, charges, (1,)),
            UniformAbelianMatrixProductOperator(operator, group, charges),
            UniformVUMPSPolicy(maximum_iterations=3, gradient_step=0.05),
        )
    )

    assert bool(result.charge_exact)
    assert bool(result.successful)
