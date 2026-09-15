import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import PeriodicCell
from phydrax.discretization._reciprocal import ReciprocalMeshPlan
from phydrax.operators.periodic._family import (
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
)
from phydrax.operators.quantum._fermionic_fock import FermionModeOrder
from phydrax.operators.quantum._magnetism import (
    dmi_spin_model,
    heisenberg_spin_model,
    prepare_quantum_spin_model,
    quantum_spin_sector_operator,
    spin_matrices,
)
from phydrax.operators.quantum._spin_wave import (
    evaluate_linear_spin_wave,
    LinearSpinWavePlan,
    lower_collinear_spin_wave_bonds,
    prepare_linear_spin_wave,
    SpinWaveReferenceState,
)
from phydrax.operators.quantum._superconductivity import (
    evaluate_bdg_spectrum,
    evaluate_pairing_observables,
    FermionicBdGPlan,
    FermionicPairingPlan,
    NambuConvention,
    prepare_fermionic_bdg,
)
from phydrax.operators.quantum.lattice._compile import QuantumLatticeResourcePolicy
from phydrax.operators.quantum.lattice._sector import SectorBasisResourcePolicy
from phydrax.sparse import EdgeRelation


def _constant_family(matrix, *, hermitian):
    value = np.asarray(matrix, dtype=complex)
    count = value.shape[0]
    target = np.repeat(np.arange(count), count)
    source = np.tile(np.arange(count), count)
    reverse = np.arange(count * count).reshape((count, count)).T.reshape((-1,))
    relation = EdgeRelation(source, target, source_size=count, target_size=count)
    plan = PeriodicTranslationFamilyPlan(
        relation,
        np.zeros((count * count, 1), dtype=int),
        reverse,
        hermitian=hermitian,
    )
    state = PeriodicTranslationFamilyState(
        plan, value[target, source].reshape((-1, 1, 1))
    )
    return prepare_periodic_translation_family(plan, state)


def test_spin_matrices_use_s_not_pauli_and_obey_su2_casimir():
    sx, sy, sz, _, _ = spin_matrices(1)
    identity = jnp.eye(2)
    assert jnp.allclose(sx @ sy - sy @ sx, 1.0j * sz)
    assert jnp.allclose(sx @ sx + sy @ sy + sz @ sz, 0.75 * identity)
    assert jnp.allclose(jnp.linalg.eigvalsh(sz), jnp.asarray([-0.5, 0.5]))


def test_heisenberg_dimer_lowers_to_fixed_projection_without_ambient_basis():
    model = heisenberg_spin_model(
        ("left", "right"), (1, 1), [[0, 1]], [1.0], energy_unit="reduced-energy"
    )
    prepared = prepare_quantum_spin_model(
        model,
        QuantumLatticeResourcePolicy(
            maximum_terms=8,
            maximum_factors_per_term=2,
            maximum_branches_per_input=32,
            maximum_sector_dimension=4,
            maximum_workspace_bytes=100_000,
        ),
    )
    operator = quantum_spin_sector_operator(
        model,
        prepared,
        0,
        sector_resources=SectorBasisResourcePolicy(
            maximum_dimension=4, maximum_table_bytes=10_000
        ),
    )
    assert operator.source.size == 2
    # In the ascending basis {|down,up>, |up,down>}, H=-S1·S2.
    columns = jnp.stack(
        tuple(operator.mv(jnp.eye(2, dtype=complex)[:, index]) for index in range(2)),
        axis=1,
    )
    assert jnp.allclose(jnp.linalg.eigvalsh(columns), jnp.asarray([-0.25, 0.75]))


def test_dmi_orientation_and_fixed_sz_conservation_are_compiler_proved():
    resources = QuantumLatticeResourcePolicy(
        maximum_terms=16,
        maximum_factors_per_term=2,
        maximum_branches_per_input=64,
        maximum_sector_dimension=4,
        maximum_workspace_bytes=100_000,
    )
    sector_resources = SectorBasisResourcePolicy(
        maximum_dimension=4, maximum_table_bytes=10_000
    )
    longitudinal = dmi_spin_model(
        ("left", "right"),
        (1, 1),
        [[0, 1]],
        [[0.0, 0.0, 1.0]],
        energy_unit="reduced-energy",
    )
    prepared_longitudinal = prepare_quantum_spin_model(longitudinal, resources)
    operator = quantum_spin_sector_operator(
        longitudinal,
        prepared_longitudinal,
        0,
        sector_resources=sector_resources,
    )
    assert operator.properties.certifies("self_adjoint")

    transverse = dmi_spin_model(
        ("left", "right"),
        (1, 1),
        [[0, 1]],
        [[1.0, 0.0, 0.0]],
        energy_unit="reduced-energy",
    )
    prepared_transverse = prepare_quantum_spin_model(transverse, resources)
    with pytest.raises(ValueError, match="charge map|declared unique"):
        quantum_spin_sector_operator(
            transverse,
            prepared_transverse,
            0,
            sector_resources=sector_resources,
        )


def test_collinear_lswt_returns_positive_krein_modes_and_rejects_instability():
    reference = SpinWaveReferenceState([[0.0, 0.0, 1.0]], [1.0])
    mesh = ReciprocalMeshPlan.monkhorst_pack(PeriodicCell([[1.0]]), (1,))
    plan = LinearSpinWavePlan(
        reference,
        mesh,
        [0],
        [0],
        energy_unit="reduced-energy",
        maximum_site_count=2,
    )
    lowering = lower_collinear_spin_wave_bonds(
        plan,
        [[[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]],
        np.empty((0,), dtype=int),
        np.empty((0,), dtype=int),
        np.empty((0, 1), dtype=int),
        np.empty((0, 3, 3)),
        zeeman_energies=[[0.0, 0.0, 2.0]],
    )
    prepared = prepare_linear_spin_wave(
        plan,
        lowering.normal_family,
        lowering.pairing_family,
        lowering.reference_torques,
    )
    result = evaluate_linear_spin_wave(prepared)
    assert jnp.allclose(result.frequencies, jnp.asarray([[2.0]]))
    assert jnp.allclose(result.krein_norms, jnp.asarray([[1.0]]))
    assert result.paraunitarity_residual < 1.0e-10

    unstable = prepare_linear_spin_wave(
        plan,
        _constant_family([[-1.0]], hermitian=True),
        _constant_family([[0.0]], hermitian=False),
        [[0.0, 0.0, 0.0]],
    )
    with pytest.raises(ValueError, match="Krein|stability|branch"):
        evaluate_linear_spin_wave(unstable)


def test_fermionic_bdg_phs_pair_antisymmetry_and_double_counting_are_explicit():
    order = FermionModeOrder(("up", "down"))
    pairing_plan = FermionicPairingPlan(
        order,
        [0],
        [1.0],
        mesh_id="gamma",
        energy_unit="reduced-energy",
    )
    plan = FermionicBdGPlan(NambuConvention(order), pairing_plan, maximum_mode_count=4)
    pairing = np.asarray([[[0.0, 0.7], [-0.7, 0.0]]])
    prepared = prepare_fermionic_bdg(
        plan,
        [[[0.2, 0.0], [0.0, 0.2]]],
        pairing,
        chemical_potential=0.0,
        double_counting_constant=0.3,
        normal_family_id="normal",
        pairing_family_id="singlet",
    )
    spectrum = evaluate_bdg_spectrum(prepared)
    assert spectrum.particle_hole_residual < 1.0e-10
    assert spectrum.spectral_pairing_residual < 1.0e-10
    assert jnp.allclose(
        spectrum.eigenvalues,
        -spectrum.eigenvalues[:, ::-1],
    )
    observables = evaluate_pairing_observables(prepared, temperature_energy=0.1)
    assert jnp.allclose(
        observables.grand_potential,
        observables.quasiparticle_grand_potential
        + observables.nambu_half_trace_correction
        + observables.double_counting_constant,
    )

    with pytest.raises(ValueError, match="antisymmetry"):
        prepare_fermionic_bdg(
            plan,
            [[[0.2, 0.0], [0.0, 0.2]]],
            [[[0.0, 0.7], [0.7, 0.0]]],
            chemical_potential=0.0,
            normal_family_id="normal",
            pairing_family_id="invalid",
        )
