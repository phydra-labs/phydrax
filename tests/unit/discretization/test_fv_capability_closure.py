#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._precision import PrecisionRequest, PrecisionResolution
from phydrax.discretization import CellMesh
from phydrax.discretization.finite_volume import (
    FiniteVolumePrecisionPolicy,
    GeostrophicBalancePlan,
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


def test_polyhedral_cube_geometry_is_closed_and_positive():
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
        dtype=float,
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


def test_polyhedral_flux_vectors_are_owner_oriented_across_shared_face():
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
        dtype=float,
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
    neighbour = np.asarray(finite_volume.neighbour_cells)
    face_centers = np.asarray(finite_volume.face_centers)
    cell_centers = np.asarray(finite_volume.cell_centers)
    area_vectors = np.asarray(finite_volume.area_vectors)
    interior = neighbour >= 0
    assert np.count_nonzero(interior) == 1
    assert np.all(
        np.sum((face_centers - cell_centers[owner]) * area_vectors, axis=-1) > 0.0
    )
    assert np.all(
        np.sum(
            (face_centers[interior] - cell_centers[neighbour[interior]])
            * area_vectors[interior],
            axis=-1,
        )
        < 0.0
    )

    face_flux = area_vectors[:, 0]
    cell_flux = np.zeros((finite_volume.cell_count,))
    np.add.at(cell_flux, owner, face_flux)
    np.add.at(cell_flux, neighbour[interior], -face_flux[interior])
    np.testing.assert_allclose(cell_flux, 0.0, atol=1e-14)


def test_equilibrium_weno_z_preserves_constant_surface_and_reports_dry_fallback():
    bed = jnp.asarray((0.0, 0.2, 0.5, 0.9, 1.0, 0.7, 0.3, 0.1))
    surface = jnp.ones_like(bed)
    state = jnp.stack((surface - bed, jnp.zeros_like(bed)), axis=-1)
    plan = ShallowWaterEquilibriumWENOZPlan(characteristic=True)
    left, right, bed_left, bed_right, evidence = jax.jit(plan.reconstruct)(state, bed)
    np.testing.assert_allclose(left[..., 0] + bed_left, 1.0, atol=2e-6)
    np.testing.assert_allclose(right[..., 0] + bed_right, 1.0, atol=2e-6)
    assert np.asarray(evidence.dry_stencil_fallback).any()


@pytest.mark.parametrize(
    "lower",
    (
        lower_triangle_unstructured_shallow_water,
        lower_sbp_shallow_water,
        lower_global_spectral_shallow_water,
        lower_dgsem_shallow_water,
    ),
)
def test_method_wide_shallow_water_lowerings_preserve_lake_at_rest(lower):
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


def test_arbitrary_normal_hydrostatic_flux_has_zero_lake_mass_and_ale_gcl_flux():
    system = ShallowWaterSystem(2)
    plan = ShallowWaterHydrostaticHLLPlan()
    normal = jnp.asarray((3.0, 4.0)) / 5.0
    left = jnp.asarray((1.0, 0.0, 0.0))
    right = jnp.asarray((0.5, 0.0, 0.0))
    result = plan.normal_face_contribution(system, left, right, 0.0, 0.5, normal)
    np.testing.assert_allclose(result.normal_flux[0], 0.0, atol=1e-7)
    np.testing.assert_allclose(result.left_correction[0], 0.0)
    np.testing.assert_allclose(result.right_correction[0], 0.0)


def test_typed_open_boundaries_and_declared_geostrophic_reference():
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


def test_multilayer_exner_les_and_subfloat_precision_contracts():
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
