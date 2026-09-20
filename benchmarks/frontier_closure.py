#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from benchmarks._runtime import capture_environment, logical_array_bytes, measure_host
from phydrax.algebraic import SparsePolynomialSystem
from phydrax.applications import (
    conformal_bootstrap as cb,
    fuzzy_space,
    numerical_relativity as nr,
    phase_field,
    spin_foam,
    supersymmetric_lattice,
)
from phydrax.operators.quantum.lattice import (
    FiniteGroupActionPlan,
    FiniteGroupIrrepPlan,
    FixedBosonNumberBasis,
    MonomialConfigurationGenerator,
    OrbitSectorResourcePolicy,
    prepare_finite_group_irrep,
    prepare_finite_group_irrep_basis,
    SectorBasisResourcePolicy,
)
from phydrax.optics import wave
from phydrax.particle_physics import integrate_native_rge, NativeSpectrumModelPlan


def _timed_case(name: str, function):
    value, seconds = measure_host(function)
    return {
        "name": name,
        "host_seconds": seconds,
        "logical_bytes": logical_array_bytes(value),
        "successful": True,
    }, value


def benchmark_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []

    resources = SectorBasisResourcePolicy(
        maximum_dimension=32,
        maximum_table_bytes=1_000_000,
    )
    direct = FixedBosonNumberBasis(
        ("a", "b", "c"),
        (2, 2, 2),
        1,
        resources=resources,
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
    irrep_plan = FiniteGroupIrrepPlan(
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
    )
    case, irrep = _timed_case(
        "finite-group-irrep",
        lambda: prepare_finite_group_irrep_basis(
            prepare_finite_group_irrep(action, irrep_plan)
        ),
    )
    case["scientific_residual"] = float(irrep.evidence.covariance_residual)
    cases.append(case)

    case, virasoro = _timed_case(
        "generic-virasoro",
        lambda: cb.prepare_virasoro_elliptic_block(
            cb.VirasoroEllipticBlockPlan(
                2.0,
                (0.1, 0.2, 0.3, 0.4),
                0.7,
                maximum_level=5,
            )
        ),
    )
    case["scientific_residual"] = float(
        np.max(np.asarray(virasoro.gram_condition_numbers))
    )
    cases.append(case)

    generator = np.random.default_rng(21)
    raw = generator.normal(size=(32, 32)) + 1j * generator.normal(size=(32, 32))
    matrix = raw - raw.T
    case, pfaffian = _timed_case(
        "scalable-pfaffian",
        lambda: supersymmetric_lattice.scalable_pfaffian(
            matrix,
            supersymmetric_lattice.ScalablePfaffianPlan(maximum_dimension=64),
        ),
    )
    case["scientific_residual"] = float(pfaffian.determinant_identity_residual)
    cases.append(case)

    case, fuzzy = _timed_case(
        "fuzzy-many-body",
        lambda: fuzzy_space.prepare_fuzzy_sphere_many_body(
            fuzzy_space.FuzzySphereManyBodyPlan(
                2,
                2,
                "boson",
                {0: 1.0, 4: 2.0},
                maximum_basis_dimension=32,
                maximum_nonzero_routes=1024,
            )
        ),
    )
    case["scientific_residual"] = float(fuzzy.evidence.hermiticity_residual)
    cases.append(case)

    sm_plan = NativeSpectrumModelPlan("sm-one-loop", scheme="msbar")
    sm_initial = np.asarray((0.36, 0.65, 1.17, 0.94, 0.024, 0.01, 0.13, -7800.0))
    case, history = _timed_case(
        "adaptive-native-rge",
        lambda: integrate_native_rge(sm_plan, sm_initial, 91.1876, 10_000.0),
    )
    case["accepted_steps"] = history.accepted_step_sizes.size
    cases.append(case)

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
    case, defect = _timed_case(
        "mapped-infinite-defect",
        lambda: phase_field.solve_mapped_infinite_defect(
            phase_field.MappedInfiniteDefectPlan(
                potential,
                ((1.0,),),
                (-1.0,),
                (1.0,),
                collocation_count=33,
                residual_tolerance=1e-7,
            )
        ),
    )
    case["scientific_residual"] = float(defect.evidence.residual_norm)
    cases.append(case)

    radial_points = np.linspace(0.0, np.pi / 2.0 - 0.05, 33)
    ads_plan = nr.SphericalConformalAdSPlan(
        radial_points,
        time_step=1e-4,
        steps=4,
    )
    ads_initial = nr.gaussian_spherical_ads_initial_data(
        ads_plan,
        amplitude=1e-4,
        center=0.5,
        width=0.15,
    )
    case, ads = _timed_case(
        "nonlinear-spherical-ads",
        lambda: nr.run_spherical_conformal_ads(ads_plan, ads_initial),
    )
    case["scientific_residual"] = float(ads.evidence.relative_mass_change)
    cases.append(case)

    time = np.linspace(-2.0, 2.0, 32, endpoint=False)
    envelope_plan = wave.CoupledEnvelopePlan(
        time,
        (0.0,),
        (0.0,),
        (10.0, 12.0),
        np.zeros((2, 32)),
        np.zeros((2, 32)),
        np.zeros(2),
        np.zeros((2, 2, 2, 2)),
        np.zeros((2, 2, 2)),
        np.zeros((2, 2, 2)),
        np.ones(32),
        np.zeros(2),
        raman_fraction=0.0,
        propagation_distance=0.1,
        step_size=0.01,
    )
    envelope_initial = np.zeros((1, 1, 2, 32), dtype="complex128")
    envelope_initial[0, 0, 0] = np.exp(-(time**2))
    envelope_initial[0, 0, 1] = 0.5 * np.exp(-0.5 * time**2)
    case, envelope = _timed_case(
        "coupled-envelope",
        lambda: wave.propagate_coupled_envelope(envelope_plan, envelope_initial),
    )
    case["scientific_residual"] = float(envelope.evidence.relative_energy_change)
    cases.append(case)

    case, boost = _timed_case(
        "sl2c-boost-kernel",
        lambda: spin_foam.evaluate_sl2c_boost(
            spin_foam.SL2CPrincipalSeriesPlan(0.2, 1, 15),
            1,
            0.7,
        ),
    )
    case["scientific_residual"] = float(boost.unitarity_residual)
    cases.append(case)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str)
    arguments = parser.parse_args()
    payload = {
        "environment": capture_environment().to_dict(),
        "cases": benchmark_cases(),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        Path(arguments.output).write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
