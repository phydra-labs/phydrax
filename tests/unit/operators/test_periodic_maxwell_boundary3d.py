from itertools import product

import jax
import numpy as np

from phydrax.discretization import PeriodicCell
from phydrax.discretization.bem import (
    OrientedTriangleSurfaceComplex3D,
    RWGSurfaceCurrentSpace3D,
)
from phydrax.operators.integral.layer_potential._maxwell3d import MaxwellEFIEPolicy3D
from phydrax.operators.integral.layer_potential._periodic_maxwell_boundary3d import (
    PeriodicMaxwellBoundaryPolicy3D,
    prepare_periodic_maxwell_boundary_3d,
)


def test_periodic_mfie_has_actual_central_magnetic_action_and_oriented_jump() -> None:
    with jax.enable_x64(True):
        vertices = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        faces = np.asarray([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int32)
        surface = OrientedTriangleSurfaceComplex3D(vertices, faces)
        space = RWGSurfaceCurrentSpace3D(surface)
        k = 0.8
        lattice = 8.0 * np.eye(3, dtype=np.float64)
        bloch = np.asarray([0.1, -0.2, 0.3], dtype=np.float64)
        prepared = prepare_periodic_maxwell_boundary_3d(
            space,
            PeriodicCell(lattice),
            wavenumber=k,
            bloch_wavevector=bloch,
            formulation="mfie",
            policy=PeriodicMaxwellBoundaryPolicy3D(image_cutoff=1),
            free_space_policy=MaxwellEFIEPolicy3D(
                regular_order=3,
                singular_order=3,
                near_order=3,
                absolute_tolerance=1.0,
                relative_tolerance=1.0,
                max_kh=2.0,
            ),
        )
        areas = np.asarray(surface.face_areas)
        normals = np.asarray(surface.face_normals)
        centers = np.asarray(surface.face_centroids)
        edges = np.asarray(surface.face_edges)
        opposite = vertices[np.asarray(surface.opposite_vertices)]
        scales = (
            np.asarray(surface.face_edge_signs)
            * np.asarray(surface.edge_lengths)[edges]
            / (2.0 * areas[:, None])
        )
        quadrature = np.asarray(
            [
                [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
                [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
                [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            ],
            dtype=np.float64,
        )
        gram = np.zeros((space.size, space.size), dtype=np.float64)
        for face, triangle in enumerate(faces):
            for point in quadrature @ vertices[triangle]:
                basis = scales[face, :, None] * (point - opposite[face])
                gram[np.ix_(edges[face], edges[face])] += (
                    areas[face] / 3.0 * (basis @ basis.T)
                )
        expected = 0.5 * gram.astype(np.complex128)
        central = np.zeros_like(expected)
        basis = scales[:, :, None] * (centers[:, None] - opposite)
        for index in product((-1, 0, 1), repeat=3):
            translation = np.asarray(index) @ lattice
            phase = np.exp(1j * (translation @ bloch))
            for target in range(4):
                for source in range(4):
                    if index == (0, 0, 0) and target == source:
                        continue
                    displacement = centers[target] - centers[source] - translation
                    radius = np.linalg.norm(displacement)
                    green_gradient = (
                        np.exp(1j * k * radius)
                        * (1j * k * radius - 1.0)
                        * displacement
                        / (4.0 * np.pi * radius**3)
                    )
                    field = np.cross(
                        normals[target], np.cross(green_gradient, basis[source])
                    )
                    local = (
                        phase * areas[target] * areas[source] * (basis[target] @ field.T)
                    )
                    expected[np.ix_(edges[target], edges[source])] += local
                    if index == (0, 0, 0):
                        central[np.ix_(edges[target], edges[source])] += local
        np.testing.assert_allclose(prepared.operator.matrix, expected, atol=1e-12)
        assert np.linalg.norm(central) > 1e-3
        assert prepared.evidence.image_count == 26
        assert not prepared.evidence.infinite_lattice_certified
