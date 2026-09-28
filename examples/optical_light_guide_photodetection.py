"""Read a ground-surface scintillator light guide out through a photomultiplier."""

import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.applications.detector import (
    DigitizationPlan,
    digitize_sensitive_hits,
    SensitiveHitPlan,
)
from phydrax.optics.geometric import NonSequentialSurfaceKind
from phydrax.optics.transport import (
    detect_optical_arrivals,
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalPhotodetector,
    PhotodetectionPlan,
    prepare_optical_monte_carlo,
    simulate_optical_photons,
    TissueOpticalMedium,
    UnifiedSurfaceModel,
)


# A 2 cm x 2 cm x 20 cm acrylic bar (medium 0, n = 1.49) in air (medium 1),
# lengths in meters. Faces (x-, x+, y-, y+, z-, z+) are surface ids 0..5:
# ground sides, a white painted end at z = 0, and a photomultiplier at z = 0.2.
half, length = 0.01, 0.2
low = np.asarray((-half, -half, 0.0))
high = np.asarray((half, half, length))
vertices, triangles = [], []
for axis in range(3):
    for side, bound in enumerate((low, high)):
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
        outward[axis] = 1.0 if side else -1.0
        base = len(vertices)
        vertices.extend(corners)
        for triangle in ((0, 1, 2), (0, 2, 3)):
            p = [corners[index] for index in triangle]
            forward = np.cross(p[1] - p[0], p[2] - p[0]) @ outward > 0.0
            triangles.append(
                [base + i for i in (triangle if forward else triangle[::-1])]
            )
kinds = np.repeat(
    np.asarray(
        [int(NonSequentialSurfaceKind.DIELECTRIC)] * 4
        + [int(NonSequentialSurfaceKind.MIRROR), int(NonSequentialSurfaceKind.DETECTOR)]
    ),
    2,
)
surfaces = phx.optics.geometric.NonSequentialSurfaceTable(
    np.asarray(vertices),
    np.asarray(triangles),
    np.zeros(12, dtype=np.int32),
    np.ones(12, dtype=np.int32),
    np.asarray((1.49, 1.0)),
    surface_ids=np.repeat(np.arange(6), 2),
    surface_kinds=kinds,
    detector_indices=np.where(kinds == int(NonSequentialSurfaceKind.DETECTOR), 0, -1),
)
surface_model = UnifiedSurfaceModel(
    ["ground"] * 4 + ["ground-front-painted", "polished"],
    sigma_alpha=(0.05,) * 4 + (0.0, 0.0),
    specular_spike=(0.1,) * 4 + (0.0, 0.0),
    specular_lobe=(0.85,) * 4 + (0.0, 0.0),
    reflectivity=(1.0,) * 4 + (0.95, 1.0),
)
# Clear acrylic: 3 m bulk absorption length, no scattering.
medium = TissueOpticalMedium(
    np.asarray((1.0 / 3.0, 0.0)), np.zeros(2), np.zeros(2), np.asarray((1.49, 1.0))
)
prepared = prepare_optical_monte_carlo(
    OpticalMonteCarloPlan(
        surfaces,
        medium,
        relativity=phx.RelativityScaleContract.si(),
        maximum_interactions=256,
        surface_model=surface_model,
        detector_arrival_capacity=1,
        photon_batch_size=1024,
    )
)

# 8 scintillation flashes of 256 isotropic 420 nm photons at the bar's middle.
events, per_event = 8, 256
count = events * per_event
rng = np.random.default_rng(0)
directions = rng.normal(size=(count, 3))
photons = launch_optical_photons(
    np.tile((0.0, 0.0, 0.5 * length), (count, 1)),
    directions,
    0,
    wavelengths=420.0e-9,
)
transport = simulate_optical_photons(prepared, ExplicitPhotonSource(photons), jr.key(0))
if not bool(transport.all_successful):
    raise RuntimeError("Optical transport evidence rejected the result.")

# Bialkali photomultiplier: QE(lambda), 90% collection, 40 ns transit with 0.6 ns
# spread, gamma single-photoelectron gain (mean 1, sd 0.35), 500 Hz dark rate.
photodetector = OpticalPhotodetector(
    np.asarray((300e-9, 350e-9, 400e-9, 450e-9, 500e-9, 600e-9)),
    np.asarray((0.05, 0.22, 0.27, 0.25, 0.18, 0.04)),
    np.asarray(((0.0, 0.0, length),)),
    collection_efficiency=0.9,
    transit_times=40e-9,
    transit_time_spreads=0.6e-9,
    single_photoelectron_charges=1.0,
    single_photoelectron_spreads=0.35,
    dark_count_rates=500.0,
)
hit_plan = SensitiveHitPlan(np.asarray((0,)), channel_count=1, conditions_id="bar")
detection = detect_optical_arrivals(
    PhotodetectionPlan(
        photodetector,
        hit_plan,
        gate=(0.0, 200e-9),
        hit_capacity=per_event,
        dark_count_capacity=4,
    ),
    transport.detector_arrivals,
    jr.key(1),
    event_ids=np.arange(events),
    photon_events=np.repeat(np.arange(events), per_event),
)
if not bool(detection.all_successful):
    raise RuntimeError("Photodetection evidence rejected the result.")
digits = digitize_sensitive_hits(
    DigitizationPlan(
        np.ones(1),
        np.full(1, 0.1),
        np.zeros((1, 1)),
        adc_lsb=0.05,
        threshold=0.5,
        maximum_adc=4095,
        conditions_id="bar",
    ),
    detection.hits,
    jr.key(2),
)
print(
    {
        "light_collection": float(transport.tallies.detector[0]),
        "standard_error": float(transport.standard_errors.detector[0]),
        "expected_photoelectrons": np.asarray(detection.expected_photoelectrons)[:, 0]
        .round(2)
        .tolist(),
        "photoelectrons": np.asarray(detection.photoelectron_counts)[:, 0].tolist(),
        "adc": np.asarray(digits.digits.signals)[:, 0].tolist(),
    }
)
