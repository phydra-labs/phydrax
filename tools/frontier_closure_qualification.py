#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Cross-frontier closure qualification with raw bounded scientific evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phydrax.algebraic import SparsePolynomialSystem
from phydrax.applications import (
    conformal_bootstrap as cb,
    fuzzy_space,
    numerical_relativity as nr,
    phase_field,
    spin_foam,
    supersymmetric_lattice,
)
from phydrax.geometry import complex as complex_geometry
from phydrax.linalg import certify_interval_psd
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
from phydrax.qualification import frontier_closure_obligations


def run_qualification() -> dict[str, object]:
    sector_resources = SectorBasisResourcePolicy(
        maximum_dimension=32,
        maximum_table_bytes=1_000_000,
    )
    direct = FixedBosonNumberBasis(
        ("a", "b", "c"),
        (2, 2, 2),
        1,
        resources=sector_resources,
    )
    rotation = MonomialConfigurationGenerator(
        "r", direct.site_ids, direct.site_dimensions, (1, 2, 0), order=3
    )
    reflection = MonomialConfigurationGenerator(
        "s", direct.site_ids, direct.site_dimensions, (1, 0, 2), order=2
    )
    action = FiniteGroupActionPlan(
        direct,
        (rotation, reflection),
        OrbitSectorResourcePolicy(
            maximum_group_order=6,
            maximum_orbit_dimension=8,
            maximum_table_bytes=1_000_000,
        ),
    )
    angle = 2.0 * np.pi / 3.0
    irrep = prepare_finite_group_irrep(
        action,
        FiniteGroupIrrepPlan(
            "standard",
            {
                "r": np.asarray(
                    (
                        (np.cos(angle), -np.sin(angle)),
                        (np.sin(angle), np.cos(angle)),
                    )
                ),
                "s": np.diag((1.0, -1.0)),
            },
        ),
    )
    irrep_basis = prepare_finite_group_irrep_basis(irrep)

    virasoro = cb.prepare_virasoro_elliptic_block(
        cb.VirasoroEllipticBlockPlan(
            2.0,
            (0.1, 0.2, 0.3, 0.4),
            0.7,
            maximum_level=2,
        )
    )
    expected_level_one = ((0.7 + 0.1 - 0.2) * (0.7 + 0.3 - 0.4)) / 1.4
    polynomial = cb.PolynomialMatrixBlock(
        cb.DampedRationalPrefactor("1", "1"),
        [[[["1", "1"]], [["0"]]], [[["0"]], [["1", "1"]]]],
    )
    pmp = cb.ConformalPolynomialMatrixProgram(
        ("0",),
        ("1",),
        (polynomial,),
        frontend_id="qualification-control",
        frontend_precision_bits=128,
    )
    continuum = cb.certify_pmp_continuum_positivity(
        pmp,
        ("1",),
        (
            cb.HalfLineSOSWitness(
                (("1", "0"), ("0", "1")),
                (("1", "0"), ("0", "1")),
                matrix_dimension=2,
            ),
        ),
    )

    generator = np.random.default_rng(11)
    raw = generator.normal(size=(8, 8)) + 1j * generator.normal(size=(8, 8))
    pfaffian = supersymmetric_lattice.scalable_pfaffian(
        raw - raw.T,
        supersymmetric_lattice.ScalablePfaffianPlan(maximum_dimension=16),
    )

    metric = np.tile(np.eye(2, dtype="complex128"), (2, 1, 1))
    ricci = complex_geometry.evaluate_calabi_yau_ricci(
        complex_geometry.KahlerMetricJet(
            metric,
            np.zeros((2, 2, 2, 2), complex),
            np.zeros((2, 2, 2, 2), complex),
            np.zeros((2, 2, 2, 2, 2), complex),
            ("first", "second"),
        ),
        maximum_ricci_norm=1e-13,
    )
    quintic_system = SparsePolynomialSystem.from_coo(
        ("z0", "z1", "z2", "z3", "z4"),
        ("quintic",),
        (0, 0, 0, 0, 0),
        tuple(tuple(int(row == column) * 5 for column in range(5)) for row in range(5)),
        np.ones(5),
    )
    quintic = complex_geometry.assess_projective_variety(
        complex_geometry.ProjectiveVarietyPlan(
            "hypersurface",
            quintic_system,
            (1, 1, 1, 1, 1),
            (5,),
        ),
        np.asarray(
            ((1.0, np.exp(1j * np.pi / 5.0), 0.0, 0.0, 0.0),),
            dtype="complex128",
        ),
    )

    radial_points = np.linspace(0.0, np.pi / 2.0 - 0.05, 33)
    ads_plan = nr.SphericalConformalAdSPlan(
        radial_points,
        time_step=1e-4,
        steps=2,
    )
    ads = nr.run_spherical_conformal_ads(
        ads_plan,
        nr.prepare_spherical_ads_initial_data(
            ads_plan,
            np.zeros(radial_points.size),
            np.zeros(radial_points.size),
        ),
    )

    fuzzy = fuzzy_space.prepare_fuzzy_sphere_many_body(
        fuzzy_space.FuzzySphereManyBodyPlan(
            2,
            2,
            "boson",
            {0: 1.0, 4: 2.0},
            maximum_basis_dimension=32,
            maximum_nonzero_routes=1024,
        )
    )
    fuzzy_eigenvalues = np.linalg.eigvalsh(np.asarray(fuzzy.dense()))

    sm_plan = NativeSpectrumModelPlan("sm-one-loop", scheme="msbar")
    sm_initial = np.asarray((0.36, 0.65, 1.17, 0.94, 0.024, 0.01, 0.13, -7800.0))
    sm = integrate_native_rge(sm_plan, sm_initial, 91.1876, 1000.0)
    expected_g1 = (
        sm_initial[0] ** -2 - (41.0 / 6.0) / (8.0 * np.pi**2) * np.log(1000.0 / 91.1876)
    ) ** -0.5

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
    defect = phase_field.solve_mapped_infinite_defect(
        phase_field.MappedInfiniteDefectPlan(
            potential,
            ((1.0,),),
            (-1.0,),
            (1.0,),
            collocation_count=33,
            residual_tolerance=1e-7,
        )
    )

    time = np.linspace(-2.0, 2.0, 16, endpoint=False)
    envelope_plan = wave.CoupledEnvelopePlan(
        time,
        (0.0,),
        (0.0,),
        (10.0,),
        np.zeros((1, 16)),
        np.zeros((1, 16)),
        np.zeros(1),
        np.zeros((1, 1, 1, 1)),
        np.zeros((1, 1, 1)),
        np.zeros((1, 1, 1)),
        np.ones(16),
        np.zeros(1),
        raman_fraction=0.0,
        propagation_distance=0.02,
        step_size=0.01,
    )
    envelope_initial = np.exp(-(time**2))[None, None, None, :]
    envelope = wave.propagate_coupled_envelope(envelope_plan, envelope_initial)

    spin_resources = SU2SectorResourcePolicy(
        maximum_product_dimension=10_000,
        maximum_sector_dimension=1_000,
        maximum_matrix_elements=1_000_000,
    )
    symbol = spin_foam.su2_15j_symbol((0,) * 10, (0,) * 5, spin_resources)
    boost = spin_foam.evaluate_sl2c_boost(
        spin_foam.SL2CPrincipalSeriesPlan(0.2, 1, 3),
        1,
        0.4,
    )

    positive = certify_interval_psd(((2, -0.25), (-0.25, 2)))
    criteria = {
        "closure_obligation_count": len(frontier_closure_obligations()),
        "interval_psd_margin": float(positive.lower_margin),
        "irrep_covariance_residual": float(irrep_basis.evidence.covariance_residual),
        "virasoro_level_one_error": float(
            abs(complex(virasoro.coefficients[1]) - expected_level_one)
        ),
        "continuum_pmp_positive": continuum.continuum_positive,
        "pfaffian_determinant_residual": float(pfaffian.determinant_identity_residual),
        "maximum_ricci_norm": float(np.max(np.asarray(ricci.ricci_norms))),
        "canonical_projective_variety": bool(quintic.accepted),
        "ads_scalar_norm": float(np.linalg.norm(np.asarray(ads.final_state.scalar))),
        "fuzzy_spectrum_error": float(
            np.linalg.norm(fuzzy_eigenvalues - np.asarray((1, 2, 2, 2, 2, 2)))
        ),
        "sm_g1_error": float(abs(float(sm.parameters[-1, 0]) - expected_g1)),
        "defect_residual": float(defect.evidence.residual_norm),
        "envelope_identity_error": float(
            np.max(np.abs(np.asarray(envelope.final_field) - envelope_initial))
        ),
        "zero_spin_15j_error": float(abs(symbol - 1.0)),
        "sl2c_unitarity_residual": float(boost.unitarity_residual),
    }
    successful = bool(
        criteria["closure_obligation_count"] == 9
        and criteria["interval_psd_margin"] > 0.0
        and criteria["irrep_covariance_residual"] < 1e-10
        and criteria["virasoro_level_one_error"] < 1e-12
        and criteria["continuum_pmp_positive"]
        and criteria["pfaffian_determinant_residual"] < 1e-8
        and criteria["maximum_ricci_norm"] < 1e-12
        and criteria["canonical_projective_variety"]
        and criteria["ads_scalar_norm"] < 1e-12
        and criteria["fuzzy_spectrum_error"] < 1e-10
        and criteria["sm_g1_error"] < 1e-6
        and criteria["defect_residual"] < 1e-6
        and criteria["envelope_identity_error"] < 1e-10
        and criteria["zero_spin_15j_error"] < 1e-12
        and criteria["sl2c_unitarity_residual"] < 1e-10
    )
    return {
        "kind": "computational-frontier-closure-qualification",
        "criteria": criteria,
        "successful": successful,
        "claim": "bounded-native-closure-evidence-with-explicit-permanent-scientific-nonclaims",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    encoded = json.dumps(run_qualification(), indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
