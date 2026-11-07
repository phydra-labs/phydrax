#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._precision import PrecisionRequest, PrecisionResolution
from phydrax.discretization import CellMesh
from phydrax.discretization._cell_complex import PolyhedralConnectivity
from phydrax.discretization.finite_volume import (
    FiniteVolumePrecisionPolicy,
    GeostrophicBalancePlan,
    HybridDiffusionBoundary,
    HybridMimeticDiffusion,
    lower_dgsem_shallow_water,
    lower_global_spectral_shallow_water,
    lower_sbp_shallow_water,
    lower_triangle_unstructured_shallow_water,
    prepare_polyhedral_finite_volume_geometry,
    PreparedShallowWaterBathymetry,
    ShallowWaterCharacteristicOpenBoundary,
    ShallowWaterEquilibriumWENOZPlan,
    ShallowWaterHydrostaticHLLPlan,
    ShallowWaterNormalDischargeBoundary,
    ShallowWaterWetDryPolicy,
    UnstructuredFiniteVolumePlan,
)
from phydrax.equations._finite_volume_advanced import (
    BedloadSedimentPlan,
    HydrostaticLayerCoupling,
    MultilayerShallowWaterSystem,
    ShallowWaterExnerSystem,
)
from phydrax.equations._hyperbolic_systems import ShallowWaterSystem
from phydrax.linalg import DenseLU, LinearSolvePolicy


@pytest.mark.parametrize(
    "vertices",
    (
        (
            (0.22265625, 0.9890625, 0.6000000000000001),
            (0.23323934062070742, 1.0, 0.598034986748605),
            (0.22573931060433014, 1.0, 0.5983017514542878),
            (0.22959894883118698, 1.0, 0.6012504013680428),
        ),
        (
            (0.20947265625, 0.9603515625, 0.55),
            (0.2164175237621592, 1.0, 0.5328684535026391),
            (0.19505740198062824, 1.0, 0.5379454072437395),
            (0.18939760179963475, 0.9763986037901143, 0.5552360742001581),
        ),
    ),
)
def test_translated_small_polyhedral_cell_preserves_affine_mimetic_geometry(
    vertices: tuple,
) -> None:
    # Actual unit-box tetrahedra 1230 and 4873 expose cancellation in area
    # vectors and face centroids, respectively.
    coordinates = np.asarray(vertices)
    mesh = CellMesh.from_polyhedra(
        coordinates,
        (
            (
                (1, 2, 3),
                (0, 3, 2),
                (0, 1, 3),
                (0, 2, 1),
            ),
        ),
    )
    geometry = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
    exact = tuple(tuple(Fraction(float(value)) for value in row) for row in coordinates)
    connectivity = mesh.connectivity
    assert isinstance(connectivity, PolyhedralConnectivity)
    faces = np.asarray(connectivity.face_vertex_values)
    offsets = np.asarray(connectivity.face_vertex_offsets)
    for row, area in enumerate(np.asarray(geometry.area_vectors)):
        face_vertices = faces[offsets[row] : offsets[row + 1]]
        a, b, c = (exact[int(vertex)] for vertex in face_vertices)
        u = tuple(b[i] - a[i] for i in range(3))
        v = tuple(c[i] - a[i] for i in range(3))
        expected = np.asarray(
            (
                float((u[1] * v[2] - u[2] * v[1]) / 2),
                float((u[2] * v[0] - u[0] * v[2]) / 2),
                float((u[0] * v[1] - u[1] * v[0]) / 2),
            )
        )
        expected *= int(np.asarray(geometry.owner_signs)[row])
        np.testing.assert_allclose(area, expected, rtol=0.0, atol=6.0e-20)
    diffusion = HybridMimeticDiffusion(geometry)
    gradient = jnp.asarray((0.3, -0.5, 0.9))
    expected_cells = geometry.cell_centers @ gradient + 2.0
    expected_faces = geometry.face_centers @ gradient + 2.0
    boundary = HybridDiffusionBoundary(
        geometry,
        dirichlet={int(row): float(value) for row, value in enumerate(expected_faces)},
    )
    tensor = jnp.asarray(((2.0, 0.4, -0.2), (0.4, 1.5, 0.3), (-0.2, 0.3, 3.0)))
    result = diffusion.solve(tensor, boundary, policy=LinearSolvePolicy(DenseLU()))
    assert result.successful
    np.testing.assert_allclose(result.value[:1], expected_cells, rtol=0.0, atol=1.0e-11)
    np.testing.assert_allclose(result.value[1:], expected_faces, rtol=0.0, atol=1.0e-11)


def test_fv_capability_closure_scenario_1() -> None:
    coordinates = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        ],
        dtype="float64",
    )
    cells = (
        (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
    )
    mesh = CellMesh.from_polyhedra(coordinates, cells)
    geometry = prepare_polyhedral_finite_volume_geometry(mesh)
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
    np.testing.assert_allclose(geometry.cell_volumes, (1.0,))
    np.testing.assert_allclose(geometry.cell_centers, ((0.5, 0.5, 0.5),))
    np.testing.assert_allclose(geometry.closure_residual, (0.0,), atol=1e-14)
    np.testing.assert_allclose(finite_volume.cell_volumes, (1.0,))
    assert finite_volume.connectivity is mesh.connectivity
    owner_offset = (
        finite_volume.face_centers - finite_volume.cell_centers[finite_volume.owner_cells]
    )
    assert np.any(np.asarray(finite_volume.owner_signs) < 0.0)
    assert np.all(
        np.sum(np.asarray(owner_offset) * np.asarray(finite_volume.area_vectors), axis=-1)
        > 0.0
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(finite_volume.area_vectors), axis=0), 0.0, atol=1e-14
    )
    coordinates = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
            (2, 0, 0),
            (2, 1, 0),
            (2, 0, 1),
            (2, 1, 1),
        ],
        dtype="float64",
    )
    cells = (
        (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
        (
            (1, 2, 9, 8),
            (5, 10, 11, 6),
            (1, 8, 10, 5),
            (8, 9, 11, 10),
            (9, 2, 6, 11),
            (2, 1, 5, 6),
        ),
    )
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(
        CellMesh.from_polyhedra(coordinates, cells)
    ).prepare()

    owner = np.asarray(finite_volume.owner_cells)
    neighbor = np.asarray(finite_volume.neighbor_cells)
    face_centers = np.asarray(finite_volume.face_centers)
    cell_centers = np.asarray(finite_volume.cell_centers)
    area_vectors = np.asarray(finite_volume.area_vectors)
    interior = neighbor >= 0
    assert np.count_nonzero(interior) == 1
    assert np.all(
        np.sum((face_centers - cell_centers[owner]) * area_vectors, axis=-1) > 0.0
    )
    assert np.all(
        np.sum(
            (face_centers[interior] - cell_centers[neighbor[interior]])
            * area_vectors[interior],
            axis=-1,
        )
        < 0.0
    )

    face_flux = area_vectors[:, 0]
    cell_flux = np.zeros((finite_volume.cell_count,))
    np.add.at(cell_flux, owner, face_flux)
    np.add.at(cell_flux, neighbor[interior], -face_flux[interior])
    np.testing.assert_allclose(cell_flux, 0.0, atol=1e-14)
    bed = jnp.asarray((0.0, 0.2, 0.5, 0.9, 1.0, 0.7, 0.3, 0.1))
    surface = jnp.ones_like(bed)
    state = jnp.stack((surface - bed, jnp.zeros_like(bed)), axis=-1)
    plan = ShallowWaterEquilibriumWENOZPlan(characteristic=True)
    left, right, bed_left, bed_right, evidence = jax.jit(plan.reconstruct)(state, bed)
    np.testing.assert_allclose(left[..., 0] + bed_left, 1.0, atol=2e-6)
    np.testing.assert_allclose(right[..., 0] + bed_right, 1.0, atol=2e-6)
    assert np.asarray(evidence.dry_stencil_fallback).any()
    for lower in (
        lower_triangle_unstructured_shallow_water,
        lower_sbp_shallow_water,
        lower_global_spectral_shallow_water,
        lower_dgsem_shallow_water,
    ):
        bed = jnp.asarray((0.0, 0.2, 0.5, 0.1))
        bathymetry = PreparedShallowWaterBathymetry(
            bed,
            bed.shape,
            geometry_id="backend-grid",
            precision_id="float32",
            dtype=jnp.float32,
        )
        derivative = lambda value, axis, args: jnp.roll(value, -1) - jnp.roll(value, 1)
        prepared = lower(
            bathymetry=bathymetry,
            derivative=derivative,
            dimension=1,
            gravity=9.81,
            geometry_id="backend-grid",
        )
        state = jnp.stack((1.0 - bed, jnp.zeros_like(bed)), axis=-1)
        np.testing.assert_allclose(prepared.residual(state), 0.0)


def _surface_volume(coordinates: np.ndarray, cell: tuple) -> float:
    """Origin-based divergence volume of a cell's faces triangulated about
    their vertex centroids (quadrilaterals) or as given (triangles)."""
    volume = 0.0
    for loop in cell:
        polygon = coordinates[np.asarray(loop)]
        if polygon.shape[0] == 3:
            fan = (polygon,)
        else:
            apex = np.mean(polygon, axis=0)
            fan = tuple(
                np.stack((apex, polygon[i], polygon[(i + 1) % polygon.shape[0]]))
                for i in range(polygon.shape[0])
            )
        volume += sum(float(np.dot(t[0], np.cross(t[1], t[2]))) / 6.0 for t in fan)
    return volume


def test_polyhedral_warped_faces_close_and_match_their_triangulated_volume() -> None:
    lift = 0.3
    # Two stacked unit hexahedra; raising one corner of the shared face makes it
    # a bilinear (warped) quadrilateral while every side face stays planar.
    coordinates = np.asarray(
        [
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1 + lift),
            (0, 1, 1),
            (0, 0, 2),
            (1, 0, 2),
            (1, 1, 2),
            (0, 1, 2),
        ],
        dtype="float64",
    )
    cells = (
        (
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ),
        (
            (4, 7, 6, 5),
            (8, 9, 10, 11),
            (4, 5, 9, 8),
            (5, 6, 10, 9),
            (6, 7, 11, 10),
            (7, 4, 8, 11),
        ),
    )
    mesh = CellMesh.from_polyhedra(coordinates, cells)
    geometry = prepare_polyhedral_finite_volume_geometry(mesh)

    # The vertex-centroid fan of a bilinear graph face integrates its height
    # exactly, so the cell volumes equal the exact trilinear volumes.
    np.testing.assert_allclose(
        geometry.cell_volumes, (1.0 + lift / 4.0, 1.0 - lift / 4.0), rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        geometry.cell_volumes,
        [_surface_volume(coordinates, cell) for cell in cells],
        rtol=0,
        atol=1e-15,
    )
    np.testing.assert_allclose(geometry.closure_residual, 0.0, atol=1e-15)
    area_vectors = np.asarray(geometry.face_area_vectors)
    measures = np.asarray(geometry.face_measures)
    weights = np.asarray(geometry.face_quadrature_weights)
    valid = np.asarray(geometry.face_quadrature_valid)
    # Planar quadrilaterals keep two fan slots; the warped one uses four.
    assert weights.shape[1] == 4
    assert sorted(np.count_nonzero(valid, axis=1).tolist()) == [2] * 10 + [4]
    np.testing.assert_allclose(np.sum(weights, axis=1), measures, rtol=1e-15)
    warped = int(np.flatnonzero(np.count_nonzero(valid, axis=1) == 4)[0])
    shared = coordinates[[4, 5, 6, 7]]
    newell = 0.5 * np.sum(np.cross(shared, np.roll(shared, -1, axis=0)), axis=0)
    np.testing.assert_allclose(
        np.abs(area_vectors[warped]), np.abs(newell), rtol=0, atol=1e-15
    )

    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
    np.testing.assert_allclose(np.sum(finite_volume.cell_volumes), 2.0, atol=1e-15)
    owner = np.asarray(finite_volume.owner_cells)
    neighbor = np.asarray(finite_volume.neighbor_cells)
    interior = neighbor >= 0
    outward = np.asarray(finite_volume.area_vectors)
    closure = np.zeros((finite_volume.cell_count, 3))
    np.add.at(closure, owner, outward)
    np.add.at(closure, neighbor[interior], -outward[interior])
    np.testing.assert_allclose(closure, 0.0, atol=1e-15)


def test_polyhedral_twisted_prism_uses_one_shared_warped_side_triangulation() -> None:
    angle = 0.2
    base = np.asarray([(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)])
    center = np.mean(base, axis=0)
    rotation = np.asarray(
        [(np.cos(angle), -np.sin(angle)), (np.sin(angle), np.cos(angle))]
    )
    top = (base - center) @ rotation.T + center
    coordinates = np.concatenate(
        (
            np.column_stack((base, np.zeros(3))),
            np.column_stack((top, np.full(3, 0.5))),
        )
    )
    cells = (((0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)),)
    mesh = CellMesh.from_polyhedra(coordinates, cells)
    geometry = prepare_polyhedral_finite_volume_geometry(mesh)

    valid = np.asarray(geometry.face_quadrature_valid)
    assert sorted(np.count_nonzero(valid, axis=1).tolist()) == [1, 1, 4, 4, 4]
    np.testing.assert_allclose(
        geometry.cell_volumes,
        (_surface_volume(coordinates, cells[0]),),
        rtol=0,
        atol=1e-15,
    )
    # A single cell owns every face, so its outward area vectors close exactly.
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(mesh).prepare()
    np.testing.assert_allclose(
        np.sum(finite_volume.area_vectors, axis=0), 0.0, atol=1e-15
    )
    np.testing.assert_allclose(geometry.closure_residual, 0.0, atol=1e-15)
    np.testing.assert_allclose(
        np.sum(geometry.cell_quadrature_weights), geometry.cell_volumes[0], atol=1e-15
    )


def test_fv_capability_closure_scenario_2() -> None:
    system = ShallowWaterSystem(2)
    plan = ShallowWaterHydrostaticHLLPlan()
    normal = jnp.asarray((3.0, 4.0)) / 5.0
    left = jnp.asarray((1.0, 0.0, 0.0))
    right = jnp.asarray((0.5, 0.0, 0.0))
    # ty: ignore[invalid-argument-type]
    result = plan.normal_face_contribution(system, left, right, 0.0, 0.5, normal)
    np.testing.assert_allclose(result.normal_flux[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(result.left_correction[0], 0.0)
    np.testing.assert_allclose(result.right_correction[0], 0.0)
    policy = ShallowWaterWetDryPolicy()
    discharge = ShallowWaterNormalDischargeBoundary(
        lambda t, x, args: jnp.ones(x.shape[:-1]),
        lambda t, x, args: 2 * jnp.ones(x.shape[:-1]),
        boundary_id="inlet",
    )
    trace = discharge.trace(
        0.0,
        jnp.asarray((1.0, 0.0, 0.0)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0, 0.0)),
        jnp.asarray(0.0),
        policy,
    )
    np.testing.assert_allclose(trace.boundary_mass_flux, 1.0)
    open_boundary = ShallowWaterCharacteristicOpenBoundary(
        lambda t, x, args: jnp.asarray(1.0),
        lambda t, x, args: jnp.asarray((0.0, 0.0)),
        boundary_id="open",
    )
    open_trace = open_boundary.trace(
        0.0,
        jnp.asarray((1.0, 0.1, 0.0)),
        jnp.asarray((0.0, 0.0)),
        jnp.asarray((1.0, 0.0)),
        jnp.asarray(0.0),
        policy,
        9.81,
    )
    assert bool(open_trace.ready)
    reference = GeostrophicBalancePlan(jnp.ones((2,)), jnp.zeros((2, 1)), "f-plane")
    prepared = reference.prepare(
        jnp.zeros((2,)), "grid", lambda state, args: jnp.zeros_like(state)
    )
    np.testing.assert_array_equal(
        prepared.deviation_residual(lambda state, args: state, prepared.reference_state),
        prepared.reference_state,
    )
    coupling = HydrostaticLayerCoupling.from_densities(jnp.asarray((1025.0, 1000.0)))
    system = MultilayerShallowWaterSystem(coupling, 1)
    state = jnp.asarray((1.0, 0.5, 0.0, 0.0))
    assert bool(system.admissible(state))
    sediment = BedloadSedimentPlan(2.65, 1e-3, 0.047, 0.4)
    np.testing.assert_allclose(
        sediment.bedload(jnp.asarray(1.0), jnp.asarray((0.0,))), 0.0
    )
    exner = ShallowWaterExnerSystem(
        ShallowWaterSystem(), sediment, bed_bounds=(-1.0, 1.0)
    )
    assert bool(exner.admissible(jnp.asarray((1.0, 0.0, 0.0))))
    request = PrecisionRequest(
        "finite-volume",
        {
            "storage": "float16",
            "compute": "float32",
            "accumulation": "float32",
            "certification": "float32",
            "output": "float16",
            "checkpoint": "float16",
        },
    )
    resolution = PrecisionResolution(request, "test-provider", dict(request.requested))
    precision = FiniteVolumePrecisionPolicy("float16", resolution=resolution)
    accepted = precision.quantize_and_validate(
        jnp.asarray((1.0, 0.0)), lambda value: value[..., 0] >= 0
    )
    assert accepted.dtype == jnp.float16
    with pytest.raises(ValueError, match="provider precision"):
        FiniteVolumePrecisionPolicy("float16")
