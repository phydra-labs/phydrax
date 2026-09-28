#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from phydrax._physical import RelativityScaleContract
from phydrax.applications.detector import (
    DigitizationPlan,
    digitize_sensitive_hits,
    SensitiveHitPlan,
)
from phydrax.optics.geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)
from phydrax.optics.transport import (
    detect_optical_arrivals,
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalPhotodetector,
    OpticalSurfaceFinish,
    OpticalTransportResult,
    OpticalVarianceReduction,
    PhotodetectionPlan,
    prepare_optical_monte_carlo,
    simulate_optical_photons,
    TissueOpticalMedium,
    UnifiedSurfaceModel,
)


_SI = RelativityScaleContract.si()
_C = float(_SI.speed_of_light)
_WAVELENGTH = 420e-9
_DETECTOR = int(NonSequentialSurfaceKind.DETECTOR)
_ABSORBER = int(NonSequentialSurfaceKind.ABSORBER)
_DIELECTRIC = int(NonSequentialSurfaceKind.DIELECTRIC)
_MIRROR = int(NonSequentialSurfaceKind.MIRROR)


def _box(
    lower: tuple[float, float, float],
    upper: tuple[float, float, float],
    kinds: tuple[int, ...],
    indices: tuple[float, float],
) -> NonSequentialSurfaceTable:
    """Closed box; faces ``(x-, x+, y-, y+, z-, z+)`` are surface ids 0..5.

    Normals point outward, from the inside medium 0 to the outside medium 1;
    the single detector face is detector 0.
    """

    low = np.asarray(lower)
    high = np.asarray(upper)
    vertices = []
    triangles = []
    for axis in range(3):
        for side, bound in enumerate((low, high)):
            others = [value for value in range(3) if value != axis]
            corners = []
            for a, b in ((0, 0), (1, 0), (1, 1), (0, 1)):
                corner = np.empty(3)
                corner[axis] = bound[axis]
                corner[others[0]] = (low, high)[a][others[0]]
                corner[others[1]] = (low, high)[b][others[1]]
                corners.append(corner)
            base = len(vertices)
            vertices.extend(corners)
            outward = np.zeros(3)
            outward[axis] = 1.0 if side else -1.0
            for triangle in ((0, 1, 2), (0, 2, 3)):
                points = [corners[index] for index in triangle]
                normal = np.cross(points[1] - points[0], points[2] - points[0])
                ordered = triangle if normal @ outward > 0.0 else triangle[::-1]
                triangles.append([base + index for index in ordered])
    face_kinds = np.repeat(np.asarray(kinds), 2)
    return NonSequentialSurfaceTable(
        np.asarray(vertices),
        np.asarray(triangles),
        np.zeros(12, dtype=np.int32),
        np.ones(12, dtype=np.int32),
        np.asarray(indices),
        surface_ids=np.repeat(np.arange(6), 2),
        surface_kinds=face_kinds,
        detector_indices=np.where(face_kinds == _DETECTOR, 0, -1),
    )


def _clear(indices: tuple[float, float]) -> TissueOpticalMedium:
    return TissueOpticalMedium(np.zeros(2), np.zeros(2), np.zeros(2), np.asarray(indices))


def _simulate(
    surfaces: NonSequentialSurfaceTable,
    indices: tuple[float, float],
    model: UnifiedSurfaceModel,
    photons: Any,
    *,
    maximum_interactions: int,
    seed: int = 0,
    **kwargs: Any,
) -> OpticalTransportResult:
    plan = OpticalMonteCarloPlan(
        surfaces,
        _clear(indices),
        relativity=_SI,
        maximum_interactions=maximum_interactions,
        surface_model=model,
        **kwargs,
    )
    return simulate_optical_photons(
        prepare_optical_monte_carlo(plan), ExplicitPhotonSource(photons), jr.key(seed)
    )


def _guide_faces(
    length: float, sides: OpticalSurfaceFinish = "polished", **model: Any
) -> tuple[NonSequentialSurfaceTable, UnifiedSurfaceModel]:
    kinds = (_DIELECTRIC,) * 4 + (_ABSORBER, _DETECTOR)
    surfaces = _box((-1.0, -50.0, 0.0), (1.0, 50.0, length), kinds, (1.5, 1.0))
    finishes: list[OpticalSurfaceFinish] = [sides] * 4 + ["polished", "polished"]
    return surfaces, UnifiedSurfaceModel(finishes, **model)


def _fresnel_power(theta: float, n1: float, n2: float) -> tuple[float, float]:
    """Textbook (Hecht §4.6) s and p reflectances."""

    cosine = np.cos(theta)
    transmitted = np.sqrt(1.0 - (n1 / n2 * np.sin(theta)) ** 2)
    r_s = (n1 * cosine - n2 * transmitted) / (n1 * cosine + n2 * transmitted)
    r_p = (n2 * cosine - n1 * transmitted) / (n2 * cosine + n1 * transmitted)
    return float(r_s**2), float(r_p**2)


def test_light_guide_delivers_trapped_light_and_one_partial_reflection() -> None:
    # TIR-trapped: 30 deg off axis meets the side walls at 60 deg > 41.8 deg.
    surfaces, model = _guide_faces(10.0)
    count = 8
    angle = np.deg2rad(30.0)
    direction = np.asarray([np.sin(angle), 0.0, np.cos(angle)])
    origins = np.tile(np.asarray([0.0, 0.0, 0.1]), (count, 1))
    trapped = launch_optical_photons(
        origins,
        np.tile(direction, (count, 1)),
        0,
        wavelengths=_WAVELENGTH,
        transverse_axes=np.tile(np.asarray([0.0, 1.0, 0.0]), (count, 1)),
        jones_vectors=np.tile(np.asarray([1.0, 1.0j]) / np.sqrt(2.0), (count, 1)),
    )
    result = _simulate(
        surfaces,
        (1.5, 1.0),
        model,
        trapped,
        maximum_interactions=16,
        detector_arrival_capacity=1,
    )
    assert bool(result.all_successful)
    np.testing.assert_allclose(np.asarray(result.per_photon_tallies.detector), 1.0)
    np.testing.assert_allclose(np.asarray(result.per_photon_tallies.escape), 0.0)
    arrivals = result.detector_arrivals
    path = (10.0 - 0.1) / np.cos(angle)
    np.testing.assert_allclose(
        np.asarray(arrivals.times)[:, 0], 1.5 * path / _C, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(arrivals.incidence_cosines)[:, 0], np.cos(angle)
    )

    # One side reflection at 30 deg incidence, then the detector face: the
    # expected-split detected weight is the textbook R_s or R_p.
    surfaces, model = _guide_faces(1.0)
    angle = np.deg2rad(60.0)
    direction = np.asarray([np.sin(angle), 0.0, np.cos(angle)])
    reference = _fresnel_power(np.deg2rad(30.0), 1.5, 1.0)
    for component in (0, 1):
        jones = np.zeros((1, 2), dtype=np.complex128)
        jones[0, component] = 1.0
        photon = launch_optical_photons(
            origins[:1],
            direction[None, :],
            0,
            wavelengths=_WAVELENGTH,
            transverse_axes=np.asarray([[0.0, 1.0, 0.0]]),
            jones_vectors=jones,
        )
        split = _simulate(
            surfaces,
            (1.5, 1.0),
            model,
            photon,
            maximum_interactions=8,
            branch_capacity=4,
            variance_reduction=OpticalVarianceReduction(
                interface_branching="expected-split"
            ),
        )
        assert bool(split.all_successful)
        np.testing.assert_allclose(
            float(split.tallies.detector[0]), reference[component], rtol=1e-12
        )
        np.testing.assert_allclose(
            float(split.tallies.escape), 1.0 - reference[component], rtol=1e-12
        )


def _cavity_launch(count: int, rng: np.random.Generator) -> Any:
    """Uniform points on the five wall faces with inward cosine directions."""

    faces = rng.integers(0, 5, count)
    axis = faces // 2
    side = faces % 2
    positions = rng.uniform(-1.0, 1.0, (count, 3))
    inward = np.zeros((count, 3))
    rows = np.arange(count)
    positions[rows, axis] = np.where(side == 1, 1.0, -1.0) * (1.0 - 1e-7)
    inward[rows, axis] = np.where(side == 1, -1.0, 1.0)
    cosine = np.sqrt(rng.uniform(size=count))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    first = np.roll(inward, 1, axis=1)
    second = np.cross(inward, first)
    directions = cosine[:, None] * inward + np.sqrt(1.0 - cosine**2)[:, None] * (
        np.cos(azimuth)[:, None] * first + np.sin(azimuth)[:, None] * second
    )
    return positions, directions


def _reference_cavity(
    positions: np.ndarray, directions: np.ndarray, reflectivity: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Independent Lambertian random walk in the cube; detector is z = +1."""

    rng = np.random.default_rng(seed)
    count = positions.shape[0]
    position = positions.copy()
    direction = directions.copy()
    path = np.zeros(count)
    alive = np.ones(count, dtype=bool)
    detected = np.zeros(count, dtype=bool)
    while np.any(alive):
        with np.errstate(divide="ignore"):
            distances = np.where(
                direction > 0.0,
                (1.0 - position) / direction,
                np.where(direction < 0.0, (-1.0 - position) / direction, np.inf),
            )
        axis = np.argmin(distances, axis=1)
        step = distances[np.arange(count), axis]
        position = np.where(
            alive[:, None], position + step[:, None] * direction, position
        )
        path = np.where(alive, path + step, path)
        positive = direction[np.arange(count), axis] > 0.0
        on_detector = alive & (axis == 2) & positive
        detected |= on_detector
        survive = alive & ~on_detector & (rng.uniform(size=count) < reflectivity)
        inward = np.zeros((count, 3))
        inward[np.arange(count), axis] = np.where(positive, -1.0, 1.0)
        cosine = np.sqrt(rng.uniform(size=count))
        azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
        first = np.roll(inward, 1, axis=1)
        second = np.cross(inward, first)
        new = cosine[:, None] * inward + np.sqrt(1.0 - cosine**2)[:, None] * (
            np.cos(azimuth)[:, None] * first + np.sin(azimuth)[:, None] * second
        )
        direction = np.where(survive[:, None], new, direction)
        alive = survive
    return detected, path


@pytest.mark.parametrize("reflectivity", [0.0, 0.8], ids=["black", "white"])
def test_lambertian_cavity_matches_reciprocity_and_independent_random_walk(
    reflectivity: float,
) -> None:
    count = 4000
    rng = np.random.default_rng(21)
    positions, directions = _cavity_launch(count, rng)
    kinds = (_MIRROR,) * 5 + (_DETECTOR,)
    surfaces = _box((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0), kinds, (1.0, 1.0))
    model = UnifiedSurfaceModel(
        ["ground-front-painted"] * 5 + ["polished"],
        reflectivity=(reflectivity,) * 5 + (1.0,),
    )
    photons = launch_optical_photons(positions, directions, 0, wavelengths=_WAVELENGTH)
    result = _simulate(
        surfaces,
        (1.0, 1.0),
        model,
        photons,
        maximum_interactions=96,
        detector_arrival_capacity=1,
        photon_batch_size=1000,
    )
    assert bool(result.all_successful)
    assert float(result.maximum_absolute_ledger_residual) < 1e-12
    detected = np.asarray(result.per_photon_tallies.detector)[:, 0]
    np.testing.assert_allclose(
        np.asarray(result.per_photon_tallies.absorption)[:, 0], 1.0 - detected
    )
    fraction = float(np.mean(detected))
    if reflectivity == 0.0:
        # View-factor reciprocity: A_wall F(wall->port) = A_port F(port->wall).
        expected = 0.2
        assert abs(fraction - expected) < 5.0 * np.sqrt(expected * 0.8 / count)
        return
    reference, path = _reference_cavity(positions, directions, reflectivity, seed=5)
    expected = float(np.mean(reference))
    spread = np.sqrt(
        fraction * (1 - fraction) / count + expected * (1 - expected) / count
    )
    assert abs(fraction - expected) < 5.0 * spread
    arrivals = result.detector_arrivals
    times = np.asarray(arrivals.times)[:, 0][np.asarray(arrivals.active)[:, 0]]
    reference_times = path[reference] / _C
    time_spread = np.sqrt(
        np.var(times) / times.size + np.var(reference_times) / reference_times.size
    )
    assert abs(np.mean(times) - np.mean(reference_times)) < 5.0 * time_spread


def test_ground_light_guide_feeds_photodetection_and_digitization() -> None:
    count = 256
    rng = np.random.default_rng(4)
    polar = rng.uniform(0.0, np.deg2rad(70.0), count)
    azimuth = rng.uniform(0.0, 2.0 * np.pi, count)
    directions = np.stack(
        (
            np.sin(polar) * np.cos(azimuth),
            np.sin(polar) * np.sin(azimuth),
            np.cos(polar),
        ),
        axis=-1,
    )
    origins = np.tile(np.asarray([0.0, 0.0, 0.2]), (count, 1))
    photons = launch_optical_photons(
        origins, directions, 0, wavelengths=_WAVELENGTH, first_identity=(0, 40)
    )
    surfaces, model = _guide_faces(
        3.0,
        "ground",
        sigma_alpha=(0.1,) * 4 + (0.0, 0.0),
        specular_spike=(0.2,) * 4 + (0.0, 0.0),
        specular_lobe=(0.6,) * 4 + (0.0, 0.0),
    )
    options = {"maximum_interactions": 48, "detector_arrival_capacity": 1}
    whole = _simulate(
        surfaces, (1.5, 1.0), model, photons, photon_batch_size=256, **options
    )
    permutation = rng.permutation(count)
    shuffled = _simulate(
        surfaces,
        (1.5, 1.0),
        model,
        jax.tree_util.tree_map(lambda value: value[permutation], photons),
        photon_batch_size=37,
        **options,
    )
    assert bool(whole.all_successful)
    for left, right in zip(
        jax.tree_util.tree_leaves(whole.detector_arrivals),
        jax.tree_util.tree_leaves(shuffled.detector_arrivals),
        strict=True,
    ):
        np.testing.assert_allclose(
            np.asarray(left)[permutation], np.asarray(right), rtol=1e-12, atol=1e-18
        )
    arrivals = whole.detector_arrivals
    delivered = np.asarray(arrivals.active)[:, 0]
    assert 0 < np.sum(delivered) < count

    photodetector = OpticalPhotodetector(
        np.asarray([380e-9, 460e-9]),
        np.asarray([0.3, 0.2]),
        np.zeros((1, 3)),
        collection_efficiency=0.9,
        transit_times=30e-9,
        transit_time_spreads=1e-9,
        single_photoelectron_charges=1.0,
        single_photoelectron_spreads=0.4,
        dark_count_rates=1e6,
    )
    hit_plan = SensitiveHitPlan(np.asarray([0]), channel_count=1, conditions_id="guide")
    plan = PhotodetectionPlan(
        photodetector,
        hit_plan,
        gate=(0.0, 200e-9),
        hit_capacity=count,
        dark_count_capacity=8,
    )
    events = 4
    detection = detect_optical_arrivals(
        plan,
        arrivals,
        jr.key(3),
        event_ids=np.arange(events),
        photon_events=np.arange(count) % events,
    )
    assert bool(detection.all_successful)
    probability = 0.25 * 0.9
    weights = np.where(delivered, np.asarray(arrivals.weights)[:, 0], 0.0)
    per_event = np.bincount(np.arange(count) % events, weights=weights, minlength=events)
    np.testing.assert_allclose(
        np.asarray(detection.expected_photoelectrons)[:, 0], probability * per_event
    )
    active = np.asarray(detection.hits.active)
    photon_hits = active & ~np.asarray(detection.dark)
    np.testing.assert_array_equal(
        np.sum(photon_hits, axis=-1), np.asarray(detection.photoelectron_counts)[:, 0]
    )
    digitization = digitize_sensitive_hits(
        DigitizationPlan(
            np.ones(1),
            np.zeros(1),
            np.zeros((1, 1)),
            adc_lsb=1e-3,
            threshold=0.0,
            maximum_adc=1 << 30,
            conditions_id="guide",
        ),
        detection.hits,
        jr.key(0),
    )
    charge = np.sum(np.where(active, np.asarray(detection.hits.energies), 0.0), axis=-1)
    np.testing.assert_allclose(np.asarray(digitization.deposited_signal)[:, 0], charge)
    np.testing.assert_allclose(
        np.asarray(digitization.digits.signals)[:, 0] * 1e-3, charge, atol=5e-4
    )
    np.testing.assert_array_equal(np.asarray(detection.hits.event_ids), np.arange(events))
    assert jnp.issubdtype(detection.hits.energies.dtype, jnp.floating)
