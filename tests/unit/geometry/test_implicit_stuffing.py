"""Isosurface stuffing of interval programs, checked against analytic solids."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax._meshcore import verify_volume
from phydrax.geometry._interval_enclosure import prepare_interval_function
from phydrax.geometry._interval_native import NativeIntervalProgram


RADIUS = 0.371
MAJOR = 0.742


def _sphere(x: Array) -> Array:
    return jnp.stack((jnp.sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]) - RADIUS,))


def _torus(x: Array) -> Array:
    ring = jnp.sqrt(x[0] * x[0] + x[1] * x[1]) - MAJOR
    return jnp.stack((jnp.sqrt(ring * ring + x[2] * x[2]) - RADIUS,))


def _sphere_distance(points: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.norm(points, axis=1) - RADIUS)


def _torus_distance(points: np.ndarray) -> np.ndarray:
    ring = np.hypot(points[:, 0], points[:, 1]) - MAJOR
    return np.abs(np.hypot(ring, points[:, 2]) - RADIUS)


def _boundary(cells: np.ndarray) -> np.ndarray:
    faces = np.concatenate(
        (
            cells[:, [1, 2, 3]],
            cells[:, [0, 3, 2]],
            cells[:, [0, 1, 3]],
            cells[:, [0, 2, 1]],
        )
    )
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    return faces[counts[inverse.reshape((-1,))] == 1]


def _minimum_dihedral_degrees(points: np.ndarray, cells: np.ndarray) -> float:
    corners = [points[cells[:, k]] for k in range(4)]
    faces = ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))
    normals = []
    for a, b, c in faces:
        normal = np.cross(corners[b] - corners[a], corners[c] - corners[a])
        normals.append(normal / np.linalg.norm(normal, axis=1, keepdims=True))
    smallest = 180.0
    for i in range(4):
        for j in range(i + 1, 4):
            cosine = np.clip(np.sum(normals[i] * normals[j], axis=1), -1.0, 1.0)
            smallest = min(smallest, float(np.min(180.0 - np.degrees(np.arccos(cosine)))))
    return smallest


def _program(field: Callable[[Array], Array], inputs: int) -> NativeIntervalProgram:
    program = prepare_interval_function(
        field, inputs, batch_capacity=1, execution="batched"
    ).native_program
    assert program is not None
    return program


@pytest.mark.parametrize(
    ("field", "distance", "half_extent", "volume"),
    [
        pytest.param(
            _sphere,
            _sphere_distance,
            (0.4, 0.4, 0.4),
            4.0 / 3.0 * np.pi * RADIUS**3,
            id="sphere",
        ),
        pytest.param(
            _torus,
            _torus_distance,
            (1.15, 1.15, 0.4),
            2.0 * np.pi**2 * MAJOR * RADIUS**2,
            id="torus",
        ),
    ],
)
def test_stuffed_solid_is_a_certified_tiling_close_to_the_zero_set(
    field: Callable[[Array], Array],
    distance: Callable[[np.ndarray], np.ndarray],
    half_extent: tuple[float, float, float],
    volume: float,
) -> None:
    program = _program(field, 3)
    upper = np.asarray(half_extent, dtype=np.float64)
    spacing = 0.1
    stuffed = program.stuff_tetrahedra(-upper, upper, spacing)
    points, cells = stuffed.points, stuffed.cells
    boundary = _boundary(cells)

    assert stuffed.degenerate_count == 0
    assert verify_volume(points, cells, boundary).certified
    # Every boundary vertex is a cut point or a warped lattice vertex: on the zero set.
    on_surface = np.unique(boundary.reshape((-1,)))
    assert np.max(distance(points[on_surface])) < 1e-12
    # Flat triangles between surface points deviate by the chord error, at most
    # curvature * edge^2 / 8; warping (alpha < 1/2) keeps edges below 1.5 spacings.
    weights = np.random.default_rng(0).dirichlet((1.0, 1.0, 1.0), size=16)
    samples = np.einsum("kj,ijl->ikl", weights, points[boundary]).reshape((-1, 3))
    assert np.max(distance(samples)) <= (1.5 * spacing) ** 2 / (8.0 * RADIUS)
    corners = [points[cells[:, k]] for k in range(4)]
    signed = np.einsum(
        "ij,ij->i",
        np.cross(corners[1] - corners[0], corners[2] - corners[0]),
        corners[3] - corners[0],
    )
    assert np.all(signed > 0.0)
    assert abs(np.sum(signed) / 6.0 - volume) < 0.02 * volume
    assert _minimum_dihedral_degrees(points, cells) > 10.0


def test_stuffing_refuses_a_program_without_three_inputs() -> None:
    program = _program(lambda x: jnp.stack((x[0] * x[0] + x[1] * x[1] - 1.0,)), 2)
    box = np.ones((3,), dtype=np.float64)
    with pytest.raises(ValueError, match="three inputs"):
        program.stuff_tetrahedra(-box, box, 0.1)
