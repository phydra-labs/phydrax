#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic analytic qualification of the diagnostic CT forward path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


_M2_PER_KG = phx.units.derived_unit(
    "m2/kg", ((phx.units.METER, 2), (phx.units.KILOGRAM, -1))
)


def _reference():
    return phx.qualification.ReferenceArtifactManifest(
        "synthetic-diagnostic-photon-qualification",
        checksum_algorithm="sha256",
        checksum="2" * 64,
        size_bytes=1,
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"coefficient": 1.0},
        uncertainty={"value": 0.0},
        lineage_ids=("analytic-two-material-two-energy",),
    )


def qualify() -> dict[str, object]:
    tomography = phx.imaging.tomography
    grid = phx.equations.PhotonEnergyGrid(np.asarray((40.0, 80.0)) * 1.602176634e-16)
    provenance = phx.nuclear.NuclearDataProvenance(
        _reference(),
        "https://example.invalid/analytic-diagnostic-photon",
        "synthetic-diagnostic-photon",
        "qualification",
        "two-material-two-energy",
    )
    coefficients = phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        grid,
        ("material-a", "material-b"),
        np.asarray(((1.0, 2.0), (3.0, 4.0))),
        _M2_PER_KG,
        provenance,
        phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
    )
    contract = phx.SpatialCoordinateContract(
        phx.units.METER,
        coordinate_system="cartesian-world",
        reference_frame="qualification-ct",
    )
    rays = phx.measurement.RaySampleSupport(
        np.asarray(((-1.0, 0.5, 0.5),)),
        np.asarray(((1.0, 0.0, 0.0),)),
        ("ray-0",),
        contract,
        far=np.asarray((5.0,)),
    )
    support = tomography.ProjectionSupport(rays, (1,), ("view-0",))
    transform = tomography.VoxelXRayTransformPlan(
        support,
        (2, 1, 1),
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        contract,
    )
    projector = tomography.MaterialBasisProjectionPlan(
        transform, coefficients.material_ids
    )
    protocol = tomography.CTAcquisitionProtocol(
        (
            tomography.CTViewAcquisition(
                "view-0",
                tomography.TubeSpectrum(grid, np.asarray((100.0, 200.0)), "absolute"),
                tomography.FilterStack(grid, np.asarray((1.0, 0.5))),
                tomography.BowtieTransmission(grid, np.asarray((1.0, 0.5))),
                tomography.AECSetting(2.0, "absolute"),
                tomography.DetectorResponse(
                    grid,
                    np.asarray((1.0, 2.0)),
                    dark_signal=5.0,
                    gain=3.0,
                ),
            ),
        )
    )
    detector = tomography.PolychromaticDetectorPlan(support, protocol, coefficients)
    density = jnp.asarray((0.1, 0.2)).reshape((2, 1, 1))
    fractions = jnp.asarray(((1.0, 0.0), (0.0, 1.0))).reshape((2, 1, 1, 2))
    areal_mass = projector.project(density, fractions)
    scatter = jnp.asarray(7.0)
    result = detector.evaluate(
        areal_mass,
        scatter_signal=scatter,
        scatter_label=tomography.ScatterLabel(
            "analytic-qualification-scatter", "absolute"
        ),
    )
    expected_primary = 200.0 * np.exp(-0.7) + 200.0 * np.exp(-1.0)
    expected_signal = 5.0 + 3.0 * (expected_primary + 7.0)
    gradient = jax.grad(lambda mass: jnp.sum(detector.evaluate(mass).expected_signal))(
        areal_mass
    )
    error = float(jnp.max(jnp.abs(result.expected_signal - expected_signal)))
    accepted = bool(
        result.successful
        & (error < 1.0e-4)
        & jnp.all(jnp.isfinite(gradient))
        & jnp.all(gradient < 0.0)
    )
    return {
        "accepted": accepted,
        "case": "analytic-two-material-two-energy",
        "areal_mass_kg_m2": np.asarray(areal_mass).tolist(),
        "expected_signal": expected_signal,
        "observed_signal": float(result.expected_signal[0]),
        "absolute_error": error,
        "gradient": np.asarray(gradient).tolist(),
        "identities": {
            "energy_grid": grid.grid_id,
            "coefficient_table": coefficients.table_id,
            "projection_plan": projector.plan_id,
            "detector_plan": detector.plan_id,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = qualify()
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
