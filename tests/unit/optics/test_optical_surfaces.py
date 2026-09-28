#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy import integrate, stats

from phydrax._physical import RelativityScaleContract
from phydrax.optics.geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)
from phydrax.optics.transport import (
    OpticalMonteCarloPlan,
    OpticalSurfaceHit,
    OpticalSurfaceInteraction,
    TissueOpticalMedium,
    UnifiedSurfaceModel,
)


_MIRROR = int(NonSequentialSurfaceKind.MIRROR)
_DIELECTRIC = int(NonSequentialSurfaceKind.DIELECTRIC)


def _geometry(
    angles: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random planes of incidence: directions, normals, and unit ``s`` axes."""

    count = angles.shape[0]
    normals = rng.normal(size=(count, 3))
    normals /= np.linalg.norm(normals, axis=-1)[:, None]
    tangents = np.cross(normals, rng.normal(size=(count, 3)))
    tangents /= np.linalg.norm(tangents, axis=-1)[:, None]
    directions = np.cos(angles)[:, None] * normals + np.sin(angles)[:, None] * tangents
    axes = np.cross(directions, normals)
    axes /= np.linalg.norm(axes, axis=-1)[:, None]
    return directions, normals, axes


def _interact(
    model: UnifiedSurfaceModel,
    directions: np.ndarray,
    normals: np.ndarray,
    axes: np.ndarray,
    jones: np.ndarray,
    indices: tuple[float, float],
    *,
    kind: int = _DIELECTRIC,
    surface: int = 0,
    seed: int = 0,
) -> OpticalSurfaceInteraction:
    count = directions.shape[0]
    hit = OpticalSurfaceHit(
        jnp.asarray(directions),
        jnp.asarray(normals),
        jnp.full((count,), indices[0]),
        jnp.full((count,), indices[1]),
        jnp.full((count,), kind, dtype=jnp.int32),
        jnp.full((count,), 500e-9),
        jnp.asarray(jones, dtype=jnp.complex128),
        jnp.asarray(axes),
        jnp.full((count,), surface, dtype=jnp.int32),
    )
    keys = jax.vmap(lambda lane: jr.fold_in(jr.key(seed), lane))(jnp.arange(count))
    return model.interact(hit, keys)


def _random_jones(count: int, rng: np.random.Generator) -> np.ndarray:
    jones = rng.normal(size=(count, 2)) + 1j * rng.normal(size=(count, 2))
    return jones / np.linalg.norm(jones, axis=-1)[:, None]


def _field(jones: np.ndarray, direction: np.ndarray, axis: np.ndarray) -> np.ndarray:
    return jones[:, :1] * axis + jones[:, 1:] * np.cross(direction, axis)


@pytest.mark.parametrize(
    ("indices", "maximum_angle"),
    [((1.0, 1.5), 1.45), ((1.5, 1.0), 0.95 * np.arcsin(1.0 / 1.5))],
    ids=["external", "internal-below-critical"],
)
def test_polarized_fresnel_fields_satisfy_maxwell_boundary_conditions(
    indices: tuple[float, float], maximum_angle: float
) -> None:
    # Independent reference: the reflected and transmitted fields rebuilt from
    # powers and Jones vectors must satisfy continuity of tangential E and H
    # (H proportional to n d × E) for arbitrary complex polarization.
    rng = np.random.default_rng(7)
    count = 64
    angles = rng.uniform(0.0, maximum_angle, count)
    directions, normals, axes = _geometry(angles, rng)
    jones = _random_jones(count, rng)
    result = _interact(
        UnifiedSurfaceModel(["polished"]), directions, normals, axes, jones, indices
    )
    n1, n2 = indices
    reflectance = np.asarray(result.reflectance)
    transmittance = np.asarray(result.transmittance)
    np.testing.assert_allclose(reflectance + transmittance, 1.0, atol=1e-13)
    assert np.all(np.asarray(result.reflection_valid))
    assert np.all(np.asarray(result.transmission_valid))
    reflected_directions = np.asarray(result.reflected_directions)
    transmitted_directions = np.asarray(result.transmitted_directions)
    incident_cosine = np.cos(angles)
    transmitted_cosine = np.sum(transmitted_directions * normals, axis=-1)
    np.testing.assert_allclose(
        n1 * np.sqrt(1.0 - incident_cosine**2),
        n2 * np.sqrt(1.0 - transmitted_cosine**2),
        atol=1e-12,
    )
    incident = _field(jones, directions, axes)
    reflected = np.sqrt(reflectance)[:, None] * _field(
        np.asarray(result.reflected_jones),
        reflected_directions,
        np.asarray(result.reflected_axes),
    )
    transmitted_amplitude = np.sqrt(
        transmittance * n1 * incident_cosine / (n2 * transmitted_cosine)
    )
    transmitted = transmitted_amplitude[:, None] * _field(
        np.asarray(result.transmitted_jones),
        transmitted_directions,
        np.asarray(result.transmitted_axes),
    )

    def tangential(vectors: np.ndarray) -> np.ndarray:
        return np.cross(normals, vectors)

    np.testing.assert_allclose(
        tangential(incident + reflected), tangential(transmitted), atol=1e-12
    )
    magnetic_incident = n1 * np.cross(directions, incident)
    magnetic_reflected = n1 * np.cross(reflected_directions, reflected)
    magnetic_transmitted = n2 * np.cross(transmitted_directions, transmitted)
    np.testing.assert_allclose(
        tangential(magnetic_incident + magnetic_reflected),
        tangential(magnetic_transmitted),
        atol=1e-12,
    )


def test_total_internal_reflection_phase_and_ellipticity_match_born_and_wolf() -> None:
    rng = np.random.default_rng(11)
    n1, n2 = 1.51, 1.0
    relative = n2 / n1
    critical = np.arcsin(relative)
    angles = np.concatenate(
        (rng.uniform(critical + 1e-3, 0.5 * np.pi - 1e-3, 31), [np.deg2rad(54.6)])
    )
    directions, normals, axes = _geometry(angles, rng)
    jones = np.tile(np.asarray([1.0, 1.0]) / np.sqrt(2.0), (angles.size, 1))
    result = _interact(
        UnifiedSurfaceModel(["polished"]), directions, normals, axes, jones, (n1, n2)
    )
    np.testing.assert_allclose(np.asarray(result.reflectance), 1.0, atol=1e-14)
    np.testing.assert_allclose(np.asarray(result.transmittance), 0.0, atol=0.0)
    reflected = np.asarray(result.reflected_jones)
    np.testing.assert_allclose(np.abs(reflected), 1.0 / np.sqrt(2.0), atol=1e-13)
    # Born & Wolf §1.5.4: tan(delta / 2) = cos(theta) sqrt(sin² theta - n²) /
    # sin² theta for delta = phi_p - phi_s; the exp(-i omega t) amplitudes
    # retard p relative to s, so delta is negative.
    sine = np.sin(angles)
    expected = -2.0 * np.arctan(np.cos(angles) * np.sqrt(sine**2 - relative**2) / sine**2)
    phase = np.angle(reflected[:, 1] / reflected[:, 0])
    np.testing.assert_allclose(phase, expected, atol=1e-12)
    stokes_circular = 2.0 * np.imag(np.conj(reflected[:, 0]) * reflected[:, 1])
    np.testing.assert_allclose(stokes_circular, np.sin(expected), atol=1e-12)
    # Fresnel rhomb (n = 1.51, 54.6 deg): one reflection retards by 45 deg.
    assert abs(np.rad2deg(phase[-1]) + 45.0) < 0.1


@pytest.mark.parametrize("component", [0, 1], ids=["s", "p"])
def test_fresnel_power_coefficients_are_reciprocal(component: int) -> None:
    rng = np.random.default_rng(3 + component)
    n1, n2 = 1.0, 1.7
    forward = rng.uniform(0.0, 1.5, 40)
    backward = np.arcsin(n1 / n2 * np.sin(forward))
    jones = np.zeros((40, 2), dtype=np.complex128)
    jones[:, component] = 1.0
    model = UnifiedSurfaceModel(["polished"])
    outward = _interact(model, *_geometry(forward, rng), jones, (n1, n2))
    inward = _interact(model, *_geometry(backward, rng), jones, (n2, n1))
    np.testing.assert_allclose(
        np.asarray(outward.transmittance), np.asarray(inward.transmittance), atol=1e-13
    )
    np.testing.assert_allclose(
        np.asarray(outward.reflectance), np.asarray(inward.reflectance), atol=1e-13
    )


def test_ground_micro_facet_tilts_follow_the_truncated_gaussian_law() -> None:
    # Lobe-only mirror at normal incidence: the outgoing angle from -d is
    # 2 alpha, and lobes must stay on the incident side (alpha < pi/4).
    sigma = 0.3
    count = 20000
    normal = np.asarray([0.0, 0.0, 1.0])
    directions = np.tile(normal, (count, 1))
    axes = np.tile(np.asarray([1.0, 0.0, 0.0]), (count, 1))
    jones = np.tile(np.asarray([1.0, 0.0]), (count, 1))
    model = UnifiedSurfaceModel(["ground"], sigma_alpha=sigma, specular_lobe=1.0)
    result = _interact(
        model,
        directions,
        directions,
        axes,
        jones,
        (1.0, 1.0),
        kind=_MIRROR,
    )
    assert np.all(np.asarray(result.reflection_valid))
    outgoing = np.asarray(result.reflected_directions)
    tilt = 0.5 * np.arccos(np.clip(-outgoing[:, 2], -1.0, 1.0))
    upper = 0.25 * np.pi

    def density(alpha: float) -> float:
        return float(np.exp(-(alpha**2) / (2.0 * sigma**2)) * np.sin(alpha))

    total = integrate.quad(density, 0.0, upper)[0]

    def cdf(values: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                integrate.quad(density, 0.0, min(value, upper))[0] / total
                for value in values
            ]
        )

    assert stats.kstest(tilt, cdf).pvalue > 1e-3
    azimuth = np.arctan2(outgoing[:, 1], outgoing[:, 0])
    assert stats.kstest(azimuth, stats.uniform(-np.pi, 2.0 * np.pi).cdf).pvalue > 1e-3


def test_unified_reflection_branches_occur_with_declared_probabilities() -> None:
    probabilities = {"spike": 0.2, "lobe": 0.3, "backscatter": 0.1, "lambertian": 0.4}
    count = 20000
    theta = np.deg2rad(40.0)
    normal = np.asarray([0.0, 0.0, 1.0])
    direction = np.asarray([np.sin(theta), 0.0, np.cos(theta)])
    directions = np.tile(direction, (count, 1))
    normals = np.tile(normal, (count, 1))
    axes = np.tile(np.asarray([0.0, -1.0, 0.0]), (count, 1))
    jones = np.tile(np.asarray([1.0, 0.0]), (count, 1))
    model = UnifiedSurfaceModel(
        ["ground"],
        sigma_alpha=0.002,
        specular_spike=probabilities["spike"],
        specular_lobe=probabilities["lobe"],
        backscatter=probabilities["backscatter"],
    )
    result = _interact(
        model, directions, normals, axes, jones, (1.0, 1.0), kind=_MIRROR, seed=5
    )
    outgoing = np.asarray(result.reflected_directions)
    specular = direction - 2.0 * np.dot(direction, normal) * normal
    from_specular = np.linalg.norm(outgoing - specular, axis=-1)
    from_back = np.linalg.norm(outgoing + direction, axis=-1)
    spike = from_specular < 1e-12
    back = from_back < 1e-12
    lobe = ~spike & (from_specular < 0.02)
    lambertian = ~spike & ~back & ~lobe
    for name, observed in (
        ("spike", spike),
        ("lobe", lobe),
        ("backscatter", back),
        ("lambertian", lambertian),
    ):
        p = probabilities[name]
        assert abs(np.sum(observed) - count * p) < 5.0 * np.sqrt(count * p * (1.0 - p)), (
            name
        )
    # Lambertian directions are cosine-weighted about -n: E[cos] = 2/3.
    cosines = -outgoing[lambertian] @ normal
    assert abs(np.mean(cosines) - 2.0 / 3.0) < 5.0 * np.sqrt(1.0 / 18.0 / cosines.size)
    np.testing.assert_allclose(np.asarray(result.reflectance), 1.0)
    jones_norm = np.sum(np.abs(np.asarray(result.reflected_jones)) ** 2, axis=-1)
    np.testing.assert_allclose(jones_norm, 1.0, atol=1e-12)
    transverse = np.sum(np.asarray(result.reflected_axes) * outgoing, axis=-1)
    np.testing.assert_allclose(transverse, 0.0, atol=1e-12)


def test_nearly_smooth_ground_dielectric_splits_like_the_polished_boundary() -> None:
    count = 20000
    theta = np.deg2rad(70.0)
    direction = np.asarray([np.sin(theta), 0.0, np.cos(theta)])
    normal = np.asarray([0.0, 0.0, 1.0])
    directions = np.tile(direction, (count, 1))
    normals = np.tile(normal, (count, 1))
    axes = np.tile(np.asarray([0.0, -1.0, 0.0]), (count, 1))
    jones = np.tile(np.asarray([1.0, 0.0]), (count, 1))
    polished = _interact(
        UnifiedSurfaceModel(["polished"]),
        directions[:1],
        normals[:1],
        axes[:1],
        jones[:1],
        (1.0, 1.5),
    )
    rough = _interact(
        UnifiedSurfaceModel(["ground"], sigma_alpha=1e-4, specular_lobe=1.0),
        directions,
        normals,
        axes,
        jones,
        (1.0, 1.5),
        seed=9,
    )
    reflectance = float(polished.reflectance[0])
    reflected = np.asarray(rough.reflectance) == 1.0
    np.testing.assert_array_equal(np.asarray(rough.transmittance), (~reflected) * 1.0)
    assert np.all(np.asarray(rough.transmission_valid) == ~reflected)
    tolerance = 5.0 * np.sqrt(count * reflectance * (1.0 - reflectance))
    assert abs(np.sum(reflected) - count * reflectance) < tolerance


def test_painted_surfaces_absorb_by_reflectivity_and_never_transmit() -> None:
    rng = np.random.default_rng(1)
    count = 16
    directions, normals, axes = _geometry(rng.uniform(0.0, 1.2, count), rng)
    jones = _random_jones(count, rng)
    model = UnifiedSurfaceModel(
        ["polished-front-painted", "ground-front-painted"], reflectivity=(0.7, 0.9)
    )
    specular = _interact(model, directions, normals, axes, jones, (1.0, 1.5))
    diffuse = _interact(model, directions, normals, axes, jones, (1.0, 1.5), surface=1)
    np.testing.assert_allclose(np.asarray(specular.reflectance), 0.7)
    np.testing.assert_allclose(np.asarray(diffuse.reflectance), 0.9)
    for result in (specular, diffuse):
        np.testing.assert_allclose(np.asarray(result.transmittance), 0.0)
        assert not np.any(np.asarray(result.transmission_valid))
        leaving = np.sum(np.asarray(result.reflected_directions) * normals, axis=-1)
        assert np.all(leaving < 0.0)
    unknown = _interact(model, directions, normals, axes, jones, (1.0, 1.5), surface=2)
    assert not np.any(np.asarray(unknown.reflection_valid))


@pytest.mark.parametrize(
    ("finishes", "kwargs", "error", "match"),
    [
        (["satin"], {}, ValueError, "finishes"),
        ("polished", {}, TypeError, "sequence"),
        (["polished"], {"sigma_alpha": 0.1}, ValueError, "only to the ground"),
        (["ground"], {"sigma_alpha": -0.1}, ValueError, "non-negative"),
        (["ground"], {"sigma_alpha": np.inf}, ValueError, "finite"),
        (["ground"], {"specular_spike": 0.7, "backscatter": 0.4}, ValueError, "exceed"),
        (["ground"], {"specular_lobe": -0.1}, ValueError, r"\[0, 1\]"),
        (["polished"], {"backscatter": 0.2}, ValueError, "only to the ground"),
        (["polished"], {"reflectivity": 1.2}, ValueError, "reflectivity"),
        (
            ["ground", "polished"],
            {"sigma_alpha": (0.1, 0.1, 0.1)},
            ValueError,
            "per surface",
        ),
        (["ground"], {"facet_attempts": 0}, ValueError, "positive"),
    ],
    ids=[
        "unknown-finish",
        "bare-string",
        "polished-roughness",
        "negative-roughness",
        "nonfinite-roughness",
        "probabilities-exceed-one",
        "negative-probability",
        "polished-branches",
        "reflectivity-above-one",
        "wrong-surface-count",
        "no-attempts",
    ],
)
def test_invalid_surface_declarations_are_refused(
    finishes: Any, kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        UnifiedSurfaceModel(finishes, **kwargs)


def test_transport_plan_refuses_surface_tables_the_model_cannot_describe() -> None:
    vertices = np.asarray(
        [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]]
    )
    triangles = np.asarray([[0, 1, 2], [0, 2, 3]])
    medium = TissueOpticalMedium(
        np.zeros(2), np.zeros(2), np.zeros(2), np.asarray([1.0, 1.5])
    )

    def plan(model: UnifiedSurfaceModel, **table: Any) -> OpticalMonteCarloPlan:
        surfaces = NonSequentialSurfaceTable(
            vertices,
            triangles,
            np.asarray([0, 0]),
            np.asarray([1, 1]),
            np.asarray([1.0, 1.5]),
            surface_ids=np.asarray([0, 0]),
            **table,
        )
        return OpticalMonteCarloPlan(
            surfaces,
            medium,
            relativity=RelativityScaleContract.si(),
            maximum_interactions=4,
            surface_model=model,
        )

    plan(UnifiedSurfaceModel(["ground"], sigma_alpha=0.1))
    with pytest.raises(ValueError, match="one finish per surface id"):
        plan(UnifiedSurfaceModel(["polished", "polished"]))
    detector = {
        "surface_kinds": np.full((2,), int(NonSequentialSurfaceKind.DETECTOR)),
        "detector_indices": np.zeros((2,), dtype=np.int32),
    }
    plan(UnifiedSurfaceModel(["polished"]), **detector)
    with pytest.raises(ValueError, match="take no UNIFIED finish"):
        plan(UnifiedSurfaceModel(["ground-front-painted"]), **detector)
