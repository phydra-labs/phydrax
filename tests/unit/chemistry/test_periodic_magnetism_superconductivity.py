import jax.numpy as jnp
import numpy as np

from phydrax.chemistry.periodic._magnetism import (
    evaluate_spin_resolved_bands,
    prepare_spin_orbit_operator,
    SpinorBasisConvention,
    SpinOrbitCouplingPlan,
    SpinResolvedBandObservablePlan,
)
from phydrax.chemistry.periodic._orbital_model import (
    PeriodicBlochGauge,
    PeriodicOrbitalBasisPlan,
    PeriodicOrbitalPencilPlan,
)
from phydrax.chemistry.periodic._superconductivity import (
    BdGChernPlan,
    evaluate_bdg_chern,
    PairingChannelPlan,
    solve_superconducting_mean_field,
    SuperconductingMeanFieldPlan,
)
from phydrax.chemistry.periodic._topology import PeriodicChernRefinementEvidence
from phydrax.discretization import PeriodicCell
from phydrax.discretization._reciprocal import (
    ReciprocalConnectivityPlan,
    ReciprocalMeshPlan,
)
from phydrax.nonlinear import NonlinearTermination
from phydrax.operators.periodic._family import (
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
)
from phydrax.operators.quantum._fermionic_fock import FermionModeOrder
from phydrax.sparse import EdgeRelation
from phydrax.units import ANGSTROM, ELECTRONVOLT


def _orbital_basis(labels, *, rank=2):
    cell = PeriodicCell(np.eye(rank))
    return PeriodicOrbitalBasisPlan(
        cell,
        tuple(labels),
        np.zeros((len(labels), rank)),
        ANGSTROM,
        PeriodicBlochGauge("lattice"),
    )


def _matrix_family(matrix, rank, *, hermitian):
    matrix = np.asarray(matrix, dtype=complex)
    count = matrix.shape[0]
    target = np.repeat(np.arange(count), count)
    source = np.tile(np.arange(count), count)
    reverse = np.arange(count * count).reshape((count, count)).T.reshape((-1,))
    relation = EdgeRelation(source, target, source_size=count, target_size=count)
    plan = PeriodicTranslationFamilyPlan(
        relation,
        np.zeros((count * count, rank), dtype=int),
        reverse,
        hermitian=hermitian,
        maximum_dense_entries=1_000_000,
    )
    state = PeriodicTranslationFamilyState(
        plan, matrix[target, source].reshape((-1, 1, 1))
    )
    return prepare_periodic_translation_family(plan, state)


def _onsite_pencil(basis, matrix):
    family = _matrix_family(matrix, basis.cell.rank, hermitian=True)
    return PeriodicOrbitalPencilPlan.orthonormal(
        basis, family.plan, family.state, ELECTRONVOLT
    ).prepare()


def test_supplied_p_orbital_l_dot_s_splitting_and_spin_projection():
    basis = _orbital_basis(("px", "py", "pz"))
    convention = SpinorBasisConvention(basis)
    angular = np.asarray(
        [
            [[0, 0, 0], [0, 0, -1j], [0, 1j, 0]],
            [[0, 0, 1j], [0, 0, 0], [-1j, 0, 0]],
            [[0, -1j, 0], [1j, 0, 0], [0, 0, 0]],
        ],
        dtype=complex,
    )
    prepared = prepare_spin_orbit_operator(
        SpinOrbitCouplingPlan(
            convention,
            angular,
            source_id="supplied-p-shell",
            maximum_spinor_count=8,
        ),
        2.0,
    )
    eigenvalues = jnp.linalg.eigvalsh(prepared.onsite_matrix)
    assert jnp.allclose(eigenvalues[:2], -2.0)
    assert jnp.allclose(eigenvalues[2:], 1.0)
    assert prepared.family.output_size == 6
    assert prepared.hermiticity_residual < 1.0e-12

    coefficients = jnp.eye(6, dtype=complex)[None, :, :]
    observable = evaluate_spin_resolved_bands(
        SpinResolvedBandObservablePlan(convention),
        jnp.arange(6.0)[None, :],
        coefficients,
    )
    assert jnp.allclose(
        observable.spin_expectations[:, :, 2],
        jnp.asarray([[0.5, -0.5] * 3]),
    )
    assert observable.spin_bound_residual < 1.0e-12


def test_finite_channel_gap_closure_and_trivial_class_d_chern_composition():
    basis = _orbital_basis(("up", "down"))
    pencil = _onsite_pencil(basis, np.zeros((2, 2)))
    mesh = ReciprocalMeshPlan.monkhorst_pack(basis.cell, (2, 2))
    minus = np.asarray([3, 2, 1, 0])
    singlet = np.asarray([[0.0, 1.0], [-1.0, 0.0]])
    pairing_family = _matrix_family(singlet, basis.cell.rank, hermitian=False)
    channels = PairingChannelPlan(
        FermionModeOrder(("up", "down")),
        mesh,
        minus,
        (pairing_family,),
        [[2.0]],
        ("onsite-singlet",),
    )
    mean_field = solve_superconducting_mean_field(
        SuperconductingMeanFieldPlan(
            pencil,
            channels,
            ensemble="fixed-chemical-potential",
            temperature_energy=0.1,
            chemical_potential=0.0,
            damping=0.5,
            termination=NonlinearTermination(
                absolute_residual=1.0e-8,
                relative_residual=1.0e-8,
                maximum_steps=128,
            ),
            minimum_gap=1.0e-4,
            maximum_mode_count=4,
        ),
        [0.9 + 0.0j],
    )
    assert bool(mean_field.successful)
    assert mean_field.channel_amplitudes[0].real > 0.0
    assert jnp.abs(mean_field.channel_amplitudes[0].imag) < 1.0e-10
    assert mean_field.spectrum.minimum_direct_gap > 0.5

    connectivity = ReciprocalConnectivityPlan.regular(mesh).prepare()
    nambu_connection = np.broadcast_to(
        np.eye(4, dtype=complex), (connectivity.edge_count, 4, 4)
    ).copy()
    topology = evaluate_bdg_chern(
        BdGChernPlan(
            connectivity,
            nambu_connection,
            basis_id=mean_field.bdg.plan.convention.convention_id,
            source_id="orthonormal-basis-connection",
            minimum_gap=1.0e-4,
            energy_unit=ELECTRONVOLT,
            refinement=PeriodicChernRefinementEvidence((2, 2), (4, 4), 0.0, 0.0, 1.0e-8),
        ),
        mean_field,
    )
    assert int(topology.chern.nearest_integer) == 0
    assert topology.chern.quantization_residual < 1.0e-12
    assert bool(topology.successful)
