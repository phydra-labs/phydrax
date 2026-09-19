#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np

from phydrax.algebraic import SparsePolynomialSystem
from phydrax.applications import (
    conformal_bootstrap as cb,
    fuzzy_space,
    numerical_relativity as nr,
    phase_field,
    spin_foam,
    spin_network,
    supersymmetric_lattice,
)
from phydrax.geometry import complex as complex_geometry
from phydrax.interchange.hep import interpret_slha2, parse_slha, serialize_slha
from phydrax.linalg import certify_interval_psd, PfaffianStatus
from phydrax.operators.quantum.lattice import (
    FiniteGroupActionPlan,
    FiniteGroupIrrepPlan,
    FixedBosonNumberBasis,
    MonomialConfigurationGenerator,
    OrbitSectorResourcePolicy,
    prepare_finite_group_irrep,
    prepare_finite_group_irrep_basis,
    SectorBasisResourcePolicy,
    SU2SectorResourcePolicy,
)
from phydrax.optics import wave
from phydrax.particle_physics import integrate_native_rge, NativeSpectrumModelPlan


def _s3_irrep_basis():
    resources = SectorBasisResourcePolicy(
        maximum_dimension=32,
        maximum_table_bytes=1_000_000,
    )
    basis = FixedBosonNumberBasis(
        ("a", "b", "c"),
        (2, 2, 2),
        1,
        resources=resources,
    )
    rotation = MonomialConfigurationGenerator(
        "r",
        basis.site_ids,
        basis.site_dimensions,
        (1, 2, 0),
        order=3,
    )
    reflection = MonomialConfigurationGenerator(
        "s",
        basis.site_ids,
        basis.site_dimensions,
        (1, 0, 2),
        order=2,
    )
    action = FiniteGroupActionPlan(
        basis,
        (rotation, reflection),
        OrbitSectorResourcePolicy(
            maximum_group_order=6,
            maximum_orbit_dimension=8,
            maximum_table_bytes=1_000_000,
        ),
    )
    angle = 2.0 * np.pi / 3.0
    matrices = {
        "r": np.asarray(
            (
                (np.cos(angle), -np.sin(angle)),
                (np.sin(angle), np.cos(angle)),
            )
        ),
        "s": np.diag((1.0, -1.0)),
    }
    return prepare_finite_group_irrep_basis(
        prepare_finite_group_irrep(
            action,
            FiniteGroupIrrepPlan("standard", matrices),
        )
    )


def test_higher_dimensional_irrep_projector_is_complete() -> None:
    basis = _s3_irrep_basis()
    embedding = np.asarray(basis.embedding)
    assert basis.multiplicity == 1
    assert basis.dimension == 2
    assert np.allclose(embedding.conj().T @ embedding, np.eye(2), atol=1e-12)
    assert float(basis.evidence.covariance_residual) < 1e-12


def test_generic_virasoro_level_one_and_matrix_sos_certificate() -> None:
    plan = cb.VirasoroEllipticBlockPlan(
        2.0,
        (0.1, 0.2, 0.3, 0.4),
        0.7,
        maximum_level=2,
    )
    block = cb.prepare_virasoro_elliptic_block(plan)
    expected = ((0.7 + 0.1 - 0.2) * (0.7 + 0.3 - 0.4)) / (2.0 * 0.7)
    assert np.allclose(np.asarray(block.coefficients[1]), expected, atol=1e-13)

    polynomial = cb.PolynomialMatrixBlock(
        cb.DampedRationalPrefactor("1", "1"),
        [[[["1", "1"]], [["0"]]], [[["0"]], [["1", "1"]]]],
    )
    program = cb.ConformalPolynomialMatrixProgram(
        ("0",),
        ("1",),
        (polynomial,),
        frontend_id="matrix-control",
        frontend_precision_bits=128,
    )
    witness = cb.HalfLineSOSWitness(
        (("1", "0"), ("0", "1")),
        (("1", "0"), ("0", "1")),
        matrix_dimension=2,
    )
    certificate = cb.certify_pmp_continuum_positivity(
        program,
        ("1",),
        (witness,),
    )
    assert certificate.continuum_positive


def test_scalable_pfaffian_obeys_determinant_identity() -> None:
    generator = np.random.default_rng(42)
    raw = generator.normal(size=(8, 8)) + 1j * generator.normal(size=(8, 8))
    matrix = raw - raw.T
    evidence = supersymmetric_lattice.scalable_pfaffian(
        matrix,
        supersymmetric_lattice.ScalablePfaffianPlan(maximum_dimension=16),
    )
    assert bool(evidence.accepted)
    assert np.allclose(complex(evidence.pfaffian) ** 2, np.linalg.det(matrix))


def test_scalable_pfaffian_phase_survives_ordinary_value_overflow() -> None:
    first = 1.0e200 * np.exp(0.2j)
    second = 1.0e200 * np.exp(0.3j)
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 1], matrix[1, 0] = first, -first
    matrix[2, 3], matrix[3, 2] = second, -second

    evidence = supersymmetric_lattice.scalable_pfaffian(
        matrix,
        supersymmetric_lattice.ScalablePfaffianPlan(
            maximum_dimension=4,
            pivot_tolerance=0.0,
        ),
    )

    assert bool(evidence.accepted)
    assert not bool(evidence.value_finite)
    assert int(evidence.native_status) == int(PfaffianStatus.NONFINITE_VALUE)
    assert np.allclose(float(evidence.phase), 0.5)
    assert np.allclose(float(evidence.log_magnitude), 2.0 * np.log(1.0e200))


def test_flat_kahler_metric_has_zero_native_ricci() -> None:
    metric = np.tile(np.eye(2, dtype=complex), (2, 1, 1))
    jet = complex_geometry.KahlerMetricJet(
        metric,
        np.zeros((2, 2, 2, 2), dtype=complex),
        np.zeros((2, 2, 2, 2), dtype=complex),
        np.zeros((2, 2, 2, 2, 2), dtype=complex),
        ("first", "second"),
    )
    evidence = complex_geometry.evaluate_calabi_yau_ricci(
        jet,
        maximum_ricci_norm=1e-13,
    )
    assert bool(evidence.accepted)
    assert np.allclose(evidence.ricci_tensors, 0.0)


def test_zero_spherical_ads_data_remains_exact() -> None:
    points = np.linspace(0.0, np.pi / 2.0 - 0.05, 33)
    plan = nr.SphericalConformalAdSPlan(
        points,
        time_step=1e-4,
        steps=2,
    )
    initial = nr.prepare_spherical_ads_initial_data(
        plan,
        np.zeros(points.size),
        np.zeros(points.size),
    )
    run = nr.run_spherical_conformal_ads(plan, initial)
    assert run.evidence.status == "complete"
    assert np.allclose(run.final_state.scalar, 0.0)
    assert np.allclose(run.final_state.metric_a, 1.0)


def test_many_body_fuzzy_sphere_reproduces_two_particle_multiplets() -> None:
    plan = fuzzy_space.FuzzySphereManyBodyPlan(
        2,
        2,
        "boson",
        {0: 1.0, 4: 2.0},
        maximum_basis_dimension=32,
        maximum_nonzero_routes=1024,
    )
    prepared = fuzzy_space.prepare_fuzzy_sphere_many_body(plan)
    eigenvalues = np.linalg.eigvalsh(np.asarray(prepared.dense()))
    assert bool(prepared.evidence.accepted)
    assert np.allclose(eigenvalues, (1.0, 2.0, 2.0, 2.0, 2.0, 2.0))


def test_sm_gauge_running_matches_one_loop_closed_form() -> None:
    plan = NativeSpectrumModelPlan("sm-one-loop", scheme="msbar")
    initial = np.asarray((0.36, 0.65, 1.17, 0.94, 0.024, 0.01, 0.13, -7800.0))
    low, high = 91.1876, 1000.0
    history = integrate_native_rge(plan, initial, low, high)
    final = float(history.parameters[-1, 0])
    coefficient = 41.0 / 6.0
    expected_inverse_square = initial[0] ** -2 - coefficient / (8.0 * np.pi**2) * np.log(
        high / low
    )
    assert np.allclose(final, expected_inverse_square**-0.5, rtol=5e-7)


def test_mapped_infinite_phi4_kink_has_correct_sector() -> None:
    potential = phase_field.PolynomialDefectPotential(
        SparsePolynomialSystem.from_coo(
            ("phi",),
            ("potential",),
            (0, 0, 0),
            ((4,), (2,), (0,)),
            (0.25, -0.5, 0.25),
        ),
        "phi4",
    )
    plan = phase_field.MappedInfiniteDefectPlan(
        potential,
        np.asarray(((1.0,),)),
        np.asarray((-1.0,)),
        np.asarray((1.0,)),
        collocation_count=33,
        residual_tolerance=1e-7,
    )
    result = phase_field.solve_mapped_infinite_defect(plan)
    assert bool(result.evidence.converged)
    assert np.allclose(result.evidence.topological_charge, (2.0,))


def test_zero_coupling_multimode_envelope_is_identity() -> None:
    time = np.linspace(-2.0, 2.0, 16, endpoint=False)
    plan = wave.CoupledEnvelopePlan(
        time,
        (0.0,),
        (0.0,),
        (10.0, 12.0),
        np.zeros((2, 16)),
        np.zeros((2, 16)),
        np.zeros(2),
        np.zeros((2, 2, 2, 2)),
        np.zeros((2, 2, 2)),
        np.zeros((2, 2, 2)),
        np.ones(16),
        np.zeros(2),
        raman_fraction=0.0,
        propagation_distance=0.02,
        step_size=0.01,
    )
    field = np.zeros((1, 1, 2, 16), dtype=complex)
    field[0, 0, 0] = np.exp(-(time**2))
    field[0, 0, 1] = 0.5 * np.exp(-0.5 * time**2)
    result = wave.propagate_coupled_envelope(plan, field)
    assert np.allclose(result.final_field, field, atol=1e-11)


def test_spin_recoupling_and_zero_spin_15j_are_exact() -> None:
    resources = SU2SectorResourcePolicy(
        maximum_product_dimension=10_000,
        maximum_sector_dimension=1_000,
        maximum_matrix_elements=1_000_000,
    )
    edges = tuple(
        spin_network.SpinNetworkEdge(f"e{index}", "left", "right", 1)
        for index in range(4)
    )
    graph = spin_network.SpinNetworkGraphPlan(
        ("left", "right"),
        edges,
        {
            "left": tuple(f"e{index}" for index in range(4)),
            "right": tuple(f"e{index}" for index in range(4)),
        },
        resources,
    )
    first = spin_network.prepare_recoupled_vertex_basis(
        graph,
        spin_network.SpinNetworkCouplingTree(
            "left",
            ("e0", "e1", "e2", "e3"),
            "first",
        ),
    )
    second = spin_network.prepare_recoupled_vertex_basis(
        graph,
        spin_network.SpinNetworkCouplingTree(
            "left",
            ("e1", "e2", "e0", "e3"),
            "second",
        ),
    )
    recoupling = spin_network.spin_network_recoupling_matrix(first, second)
    assert bool(recoupling.accepted)
    assert np.allclose(
        np.asarray(recoupling.matrix) @ np.asarray(recoupling.matrix).conj().T,
        np.eye(2),
    )
    assert np.allclose(spin_foam.su2_15j_symbol((0,) * 10, (0,) * 5, resources), 1.0)


def test_slha2_preserves_qnumbers_header_semantics() -> None:
    data = """BLOCK MASS
  25 1.25000000E+02
BLOCK NMIX Q= 1.0E+03
  1 1 1.0
  1 2 0.0
  2 1 0.0
  2 2 1.0
BLOCK QNUMBERS 9000001
  1 0
  2 1
  3 1
  4 0
"""
    document = parse_slha(data)
    semantics = interpret_slha2(document)
    assert semantics.quantum_numbers[0].pdg_id == 9000001
    restored = parse_slha(serialize_slha(document))
    assert restored.blocks[-1].header_arguments == ("9000001",)


def test_interval_psd_refuses_indefinite_matrix() -> None:
    positive = certify_interval_psd(((2, -0.25), (-0.25, 2)))
    indefinite = certify_interval_psd(((1, 2), (2, 1)))
    assert positive.positive_semidefinite
    assert not indefinite.positive_semidefinite


def test_generic_liouville_upsilon_obeys_reflection() -> None:
    coupling = 0.8
    background_charge = coupling + 1.0 / coupling
    argument = 0.7 + 0.2j
    direct = cb.liouville_upsilon(
        coupling,
        argument,
        precision_digits=30,
    )
    reflected = cb.liouville_upsilon(
        coupling,
        background_charge - argument,
        precision_digits=30,
    )
    assert np.allclose(direct, reflected, rtol=1e-12, atol=1e-12)


def test_bfss_fermion_preparation_certifies_rational_action() -> None:
    bosonic = supersymmetric_lattice.BFSSPlan(
        4,
        1,
        1,
        coupling=1.0,
        time_spacing=1.0,
    )
    configuration = supersymmetric_lattice.BFSSConfiguration(
        np.zeros((4, 1, 1, 1), dtype=complex),
        np.zeros((4, 1, 1, 1), dtype=complex),
        np.ones((4, 1, 1), dtype=complex),
        np.ones((4, 1, 1), dtype=complex),
    )
    plan = supersymmetric_lattice.BFSSFermionPlan(
        bosonic,
        np.ones((1, 1, 1)),
        np.ones((1, 1)),
        regulator_mass=0.1,
        spectral_lower=0.4,
        spectral_upper=0.7,
        rational_poles=6,
        rational_verification_points=257,
        maximum_rational_error=0.01,
    )
    prepared = supersymmetric_lattice.prepare_bfss_fermion_rhmc(
        plan,
        configuration,
    )
    assert bool(prepared.evidence.accepted)
    assert float(prepared.evidence.antisymmetry_residual) < 1e-12


def test_nonzero_sl2c_boost_is_unitary_and_composes() -> None:
    evidence = spin_foam.evaluate_sl2c_boost(
        spin_foam.SL2CPrincipalSeriesPlan(0.2, 1, 5),
        1,
        0.7,
    )
    assert bool(evidence.accepted)
    assert float(evidence.unitarity_residual) < 1e-12
    assert float(evidence.composition_residual) < 1e-12


def test_projective_variety_reuses_canonical_sparse_polynomials() -> None:
    system = SparsePolynomialSystem.from_coo(
        ("z0", "z1", "z2", "z3", "z4"),
        ("quintic",),
        (0, 0, 0, 0, 0),
        tuple(tuple(int(row == column) * 5 for column in range(5)) for row in range(5)),
        np.ones(5),
    )
    plan = complex_geometry.ProjectiveVarietyPlan(
        "hypersurface",
        system,
        (1, 1, 1, 1, 1),
        (5,),
    )
    point = np.asarray(
        ((1.0, np.exp(1j * np.pi / 5.0), 0.0, 0.0, 0.0),),
        dtype=complex,
    )
    evidence = complex_geometry.assess_projective_variety(plan, point)
    assert plan.system.system_id == system.system_id
    assert bool(evidence.accepted)
