"""Cherenkov light of a 3 MeV electron in a water cube read out at its walls.

A condensed-history electron records its steps; the steps emit dispersive
Frank-Tamm photons; the photons are transported to the cube walls, which are
photodetectors. The stopping power and the dispersion are synthetic round
numbers, not a qualified water model.
"""

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.optics.geometric import NonSequentialSurfaceKind
from phydrax.optics.transport import (
    charged_steps_from_transport,
    CherenkovEmission,
    detect_optical_arrivals,
    emit_optical_photons,
    ExplicitPhotonSource,
    OpticalEmissionStatus,
    OpticalMonteCarloPlan,
    OpticalPhotodetector,
    OpticalPhotonSourcePlan,
    PhotodetectionPlan,
    prepare_optical_monte_carlo,
    simulate_optical_photons,
    SpectralOpticalMedium,
)


relativity = phx.RelativityScaleContract.si()
side = 0.04  # meters

# Charged transport: one water voxel with a flat ~2 MeV/cm stopping power.
manifest = phx.qualification.ReferenceArtifactManifest(
    "synthetic-water-stopping",
    checksum_algorithm="sha256",
    checksum="f" * 64,
    size_bytes=1,
    license_id="synthetic-permissive",
    commercial_use_permitted=True,
    redistribution_permitted=True,
    training_use_permitted=False,
    export_permitted=True,
    export_classification="unrestricted",
    nondimensionalization={"energy_eV": 1.0},
    uncertainty={"relative": 0.0},
    lineage_ids=("synthetic:water",),
)
charged = phx.solver.ChargedParticleTransportPlan(
    phx.discretization.VoxelRadiationGeometryPlan(
        jnp.zeros(3),
        jnp.full(3, side),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        material_count=1,
    ),
    phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((10.0, 2.0e7)),
        jnp.full((1, 2), 2.0e8),
        jnp.zeros((1, 2)),
        jnp.zeros((1, 2)),
        ("water",),
        manifest,
    ),
    maximum_steps=512,
    maximum_step_length=1e-3,
    maximum_fractional_energy_loss=0.05,
    cutoff_energy_ev=1.0e4,
    step_bank_capacity=512,
)
electron = charged.simulate(
    jnp.asarray(((0.5 * side, 0.5 * side, 0.005),)),
    jnp.asarray(((0.0, 0.0, 1.0),)),
    jnp.asarray((3.0e6,)),
    jnp.asarray((int(phx.equations.ChargedRadiationParticleKind.ELECTRON),)),
    jr.key(0),
)

# Water (medium 0) with a linear dispersion 1.345 -> 1.330 over 300-600 nm,
# inside air (medium 1).
grid = np.linspace(250e-9, 650e-9, 41)
water_index = 1.345 - 0.015 * (grid - 300e-9) / 300e-9
medium = SpectralOpticalMedium(
    grid,
    np.stack((water_index, np.ones_like(grid))),
    np.full((2, grid.size), np.inf),
)
sources = OpticalPhotonSourcePlan(
    relativity=relativity,
    photon_capacity=4096,
    cherenkov=CherenkovEmission(
        medium, np.linspace(300e-9, 600e-9, 129), radiating_media=(True, False)
    ),
)
steps = charged_steps_from_transport(electron, (0,), relativity)
emission = emit_optical_photons(sources, steps, jr.key(1))
if int(emission.status) != int(OpticalEmissionStatus.SUCCESS):
    raise RuntimeError(
        f"Emission refused: {OpticalEmissionStatus(int(emission.status))!r}"
    )

# The six cube faces are one photodetector.
low, high = np.zeros(3), np.full(3, side)
vertices, triangles = [], []
for axis in range(3):
    for face, bound in enumerate((low, high)):
        u, v = (value for value in range(3) if value != axis)
        corners = []
        for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
            corner = np.empty(3)
            corner[axis], corner[u], corner[v] = (
                bound[axis],
                (low, high)[a][u],
                (low, high)[b][v],
            )
            corners.append(corner)
        outward = np.zeros(3)
        outward[axis] = 1.0 if face else -1.0
        base = len(vertices)
        vertices.extend(corners)
        for triangle in ((0, 1, 2), (0, 2, 3)):
            p = [corners[index] for index in triangle]
            forward = np.cross(p[1] - p[0], p[2] - p[0]) @ outward > 0.0
            triangles.append(
                [base + i for i in (triangle if forward else triangle[::-1])]
            )
surfaces = phx.optics.geometric.NonSequentialSurfaceTable(
    np.asarray(vertices),
    np.asarray(triangles),
    np.zeros(12, dtype=np.int32),
    np.ones(12, dtype=np.int32),
    np.asarray((1.33, 1.0)),
    surface_ids=np.repeat(np.arange(6), 2),
    surface_kinds=np.full(12, int(NonSequentialSurfaceKind.DETECTOR)),
    detector_indices=np.zeros(12, dtype=np.int32),
)
optical = simulate_optical_photons(
    prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            surfaces,
            medium,
            relativity=relativity,
            maximum_interactions=8,
            detector_arrival_capacity=1,
        )
    ),
    ExplicitPhotonSource(emission.state),
    jr.key(2),
)
detection = detect_optical_arrivals(
    PhotodetectionPlan(
        OpticalPhotodetector(
            np.linspace(300e-9, 600e-9, 5),
            np.asarray(((0.2, 0.25, 0.3, 0.25, 0.2),)),
            np.zeros((1, 3)),
            transit_times=40e-9,
            transit_time_spreads=0.6e-9,
        ),
        phx.applications.detector.SensitiveHitPlan(
            np.asarray((0,)), channel_count=1, conditions_id="water-cube"
        ),
        gate=(0.0, 1e-6),
        hit_capacity=4096,
    ),
    optical.detector_arrivals,
    jr.key(3),
    event_ids=np.asarray((0,)),
    photon_events=np.zeros(sources.photon_capacity, dtype=np.int64),
)
if not (bool(optical.all_successful) and bool(detection.all_successful)):
    raise RuntimeError(
        "Optical transport or photodetection evidence rejected the result."
    )
print(
    {
        "charged_steps": int(electron.step_count[0]),
        "expected_cherenkov_photons": round(float(emission.cherenkov_expected[0]), 1),
        "emitted_photons": int(emission.allocated_count),
        "emitted_energy_eV": round(
            float(emission.cherenkov_photon_energy[0])
            / float(phx.ElectromagneticScaleContract.si().elementary_charge),
            1,
        ),
        "expected_photoelectrons": round(
            float(detection.expected_photoelectrons[0, 0]), 1
        ),
        "photoelectrons": int(detection.photoelectron_counts[0, 0]),
    }
)
