#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import functools
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples._native_surface_sources import sphere
from phydrax.geometry import MeshingDomain
from phydrax.geometry._sphere_material_atlas import (
    sphere_material_atlas_from_result,
    SphereMaterialCellAtlas,
)
from phydrax.linalg._small_batched import prepare_exact_small_linear_actions
from phydrax.meshing._result import CellMeshingResult


@functools.cache
def _accepted(
    size: float,
) -> tuple[MeshingDomain, CellMeshingResult, SphereMaterialCellAtlas]:
    domain = sphere().domain
    meshing = phx.meshing
    scope = meshing.MeshingScope(
        domain.source_id,
        domain.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.asarray(domain.scope_indices(2), dtype=np.int64),
    )
    deviation = 0.25 * size * size
    specification = meshing.SurfaceMeshingSpec(
        meshing.CellMeshingTarget(2, 3, meshing.CellFamilyPolicy(required=("triangle",))),
        scope,
        size_controls=(
            meshing.UniformSizeControl(
                scope,
                size,
                maximum_size=1.5 * size,
                strength=meshing.SizeControlStrength.HARD,
            ),
        ),
        size_compliance=meshing.SizeCompliancePolicy(
            relative_tolerance=0.5, target_statistics=("p50",)
        ),
        protected_features=(
            meshing.ProtectedFeature(
                scope,
                meshing.FeatureKind.SURFACE,
                maximum_deviation=deviation,
            ),
        ),
        limits=meshing.MeshingLimits(
            maximum_vertices=20_000,
            maximum_edges=160_000,
            maximum_faces=160_000,
            maximum_cells=160_000,
            maximum_work_units=640_000,
            maximum_geometry_queries=1_280_000,
            maximum_scratch_bytes=81_920_000,
        ),
    )
    result = (
        meshing.NativeMeshingProvider(meshing.NativeMeshingOptions("parametric_surface"))
        .plan(
            meshing.NativeSurfaceSource(domain),
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    return (
        domain,
        result,
        sphere_material_atlas_from_result(domain, result, maximum_fidelity=deviation),
    )


def test_actual_sphere_material_cells_cover_poles_with_regular_maps_and_true_area() -> (
    None
):
    domain, result, atlas = _accepted(0.4)
    atlas.require_bound(domain, result.mesh, result.geometry, result.coordinate_contract)
    rows = jnp.arange(atlas.num_charts, dtype=jnp.int32)
    corners = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    corner_rows = jnp.broadcast_to(rows[:, None], (atlas.num_charts, 3))
    references = jnp.broadcast_to(corners, (atlas.num_charts, 3, 2))
    densities = np.asarray(atlas.jacobian(corner_rows, references))
    assert np.all(densities >= np.asarray(atlas.jacobian_lower)[:, None])
    points = np.asarray(atlas.map(corner_rows, references))
    np.testing.assert_allclose(
        np.linalg.norm(points, axis=-1), 1.0, rtol=0.0, atol=2.0e-13
    )
    vertex_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(result.mesh.vertex_global_ids))
    }
    local_corners = np.asarray(
        [
            [vertex_rows[int(identifier)] for identifier in corners]
            for corners in np.asarray(atlas.physical_corner_global_ids)
        ],
        dtype=np.int64,
    )
    old_corners = np.asarray(result.mesh.coordinates)[local_corners]
    # The old affine chord at an interior reference point is distinct from the
    # authoritative radial surface; the complete interpolation bound dominates it.
    reference = jnp.broadcast_to(jnp.asarray((0.2, 0.3)), (atlas.num_charts, 2))
    actual = np.asarray(atlas.map(rows, reference))
    old = 0.5 * old_corners[:, 0] + 0.2 * old_corners[:, 1] + 0.3 * old_corners[:, 2]
    assert np.all(
        np.linalg.norm(actual - old, axis=1) <= np.asarray(atlas.source_fidelity_bounds)
    )
    assert np.max(np.asarray(atlas.source_fidelity_bounds)) <= 0.04
    inverse = atlas.inverse(rows, jnp.asarray(actual))
    assert np.all(np.asarray(inverse.successful))
    np.testing.assert_allclose(
        np.asarray(inverse.reference), np.asarray(reference), rtol=0.0, atol=2.0e-11
    )
    nodes, weights = np.polynomial.legendre.leggauss(16)
    unit = 0.5 * (nodes + 1.0)
    first, second = np.meshgrid(unit, unit, indexing="ij")
    quadrature = np.stack((first.ravel(), ((1.0 - first) * second).ravel()), axis=-1)
    quadrature_weights = (
        0.25 * weights[:, None] * weights[None, :] * (1.0 - first)
    ).ravel()
    values = np.asarray(
        atlas.jacobian(
            jnp.broadcast_to(rows[:, None], (atlas.num_charts, quadrature.shape[0])),
            jnp.broadcast_to(
                jnp.asarray(quadrature), (atlas.num_charts, *quadrature.shape)
            ),
        )
    )
    assert abs(float(np.sum(values * quadrature_weights)) - 4.0 * np.pi) < 2.0e-11
    changed = eqx.tree_at(
        lambda value: value.directions, atlas, atlas.directions.at[0, 0, 0].add(0.01)
    )
    with pytest.raises(ValueError):
        changed.require_bound(
            domain, result.mesh, result.geometry, result.coordinate_contract
        )


def test_different_sphere_material_cells_need_actual_projective_reference_maps() -> None:
    _, _, source = _accepted(0.4)
    _, _, target = _accepted(0.32)
    reference = jnp.asarray((0.23, 0.31))
    point = source.map(jnp.asarray(0, dtype=jnp.int32), reference)
    inverse = target.inverse(
        jnp.arange(target.num_charts, dtype=jnp.int32),
        jnp.broadcast_to(point, (target.num_charts, 3)),
    )
    coordinates = np.asarray(inverse.reference)
    candidates = np.flatnonzero(
        np.asarray(inverse.successful)
        & np.all(coordinates >= 0.0, axis=1)
        & (np.sum(coordinates, axis=1) <= 1.0),
    )
    target_row = int(candidates[0])
    mapping = source.projective_reference_map(
        int(np.asarray(source.cell_global_ids)[0]),
        target,
        int(np.asarray(target.cell_global_ids)[target_row]),
    )
    old = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(source.directions)[0]
    )
    matrix = tuple(
        tuple(Fraction(float(value)) for value in row)
        for row in np.asarray(target.directions)[target_row].T
    )
    frame = tuple(
        (old[0][axis], old[1][axis] - old[0][axis], old[2][axis] - old[0][axis])
        for axis in range(3)
    )
    direct = prepare_exact_small_linear_actions(matrix, frame)
    assert direct.actions is not None and mapping.exact_coefficients == direct.actions
    assert mapping.orientation_ratio == source.exact_determinants[0] / direct.determinant
    origin, epsilon = (Fraction(23, 100), Fraction(31, 100)), Fraction(1, 1000)
    piece = (origin, (origin[0] + epsilon, origin[1]), (origin[0], origin[1] + epsilon))
    bounds = mapping.certify_triangle(piece)
    assert bounds.denominator_lower > 0 and bounds.jacobian_lower > 0
    projected = mapping.map(reference)
    np.testing.assert_allclose(
        np.asarray(projected), coordinates[target_row], rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        np.asarray(target.map(jnp.asarray(target_row, dtype=jnp.int32), projected)),
        np.asarray(point),
        rtol=0.0,
        atol=2.0e-12,
    )
    ends = jnp.asarray(
        (
            (float(origin[0]), float(origin[1])),
            (float(origin[0] + epsilon), float(origin[1])),
        )
    )
    midpoint = np.asarray(mapping.map(jnp.mean(ends, axis=0)))
    corner_interpolant = np.mean(np.asarray(mapping.map(ends)), axis=0)
    assert np.max(np.abs(midpoint - corner_interpolant)) > 1.0e-12
    with pytest.raises(ValueError):
        mapping.certify_triangle(((Fraction(-1), Fraction(0)), *piece[1:]))
    inconsistent = eqx.tree_at(
        lambda value: value.coefficients, mapping, jnp.zeros_like(mapping.coefficients)
    )
    with pytest.raises(ValueError):
        inconsistent.certify_triangle(piece)
    with pytest.raises(ValueError):
        inconsistent.require_bound(source, target)
    original_id = int(np.asarray(source.cell_global_ids)[0])
    same_source = source.projective_reference_map(original_id, source, original_id)
    assert same_source.exact_coefficients == (
        (Fraction(1), Fraction(-1), Fraction(-1)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    np.testing.assert_allclose(
        same_source.map(reference), reference, rtol=0.0, atol=4 * np.finfo(np.float64).eps
    )


def test_exact_small_projective_actions_preserve_binary_cancellation_and_singularity() -> (
    None
):
    epsilon = Fraction(float(np.nextafter(1.0, 2.0))) - 1
    matrix = ((Fraction(1), Fraction(1)), (Fraction(1), 1 + epsilon))
    right = ((Fraction(1), Fraction(0)), (Fraction(1), epsilon))
    prepared = prepare_exact_small_linear_actions(matrix, right)
    assert prepared.successful and prepared.rank == 2
    assert prepared.actions == ((Fraction(1), Fraction(-1)), (Fraction(0), Fraction(1)))
    singular = prepare_exact_small_linear_actions(
        ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))),
        ((Fraction(1),), (Fraction(2),)),
    )
    assert (
        singular.status == "singular" and singular.rank == 1 and singular.actions is None
    )
    determinant_only = prepare_exact_small_linear_actions(matrix, ((), ()))
    assert determinant_only.successful and determinant_only.rank == 2
    assert determinant_only.actions == ((), ())
    assert determinant_only.determinant == prepared.determinant == epsilon
    assert determinant_only.operation_count < prepared.operation_count
    singular_determinant = prepare_exact_small_linear_actions(
        ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))), ((), ())
    )
    assert singular_determinant.status == "singular"
    assert singular_determinant.rank == 1 and singular_determinant.determinant == 0
    assert singular_determinant.actions is None


def test_polynomial_budget_counts_dense_conversion_and_retained_basis_before_expansion() -> (
    None
):
    from phydrax.discretization._coordinate_enclosure import (
        bernstein_coefficients,
        constant,
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
        lattice_basis,
        Polynomial,
    )

    polynomial: Polynomial = {
        (first, second): Fraction(1)
        for first in range(11)
        for second in range(11 - first)
    }
    insufficient = CoordinateEnclosureBudget(198, 16 * 1024**2)
    with (
        insufficient.activate(),
        pytest.raises(CoordinateEnclosureResourceError) as refused,
    ):
        bernstein_coefficients(polynomial, "simplex", 2)
    assert refused.value.resource == "coefficient_work"
    assert refused.value.requested > refused.value.limit
    assert refused.value.completed <= refused.value.limit
    complete = CoordinateEnclosureBudget(20_000, 16 * 1024**2)
    with complete.activate():
        coefficients = bernstein_coefficients(polynomial, "simplex", 2)
    assert complete.work_units >= 66 * 66
    # Independent endpoint values are exact source values of this dense degree-ten polynomial.
    assert min(coefficients) == 1 and max(coefficients) == 11
    lattice_basis("triangle", 3)
    # Admit the dense construction workspace, then refuse the complete live basis.
    limited_storage = CoordinateEnclosureBudget(100_000, 300_000)
    with (
        limited_storage.activate(),
        pytest.raises(CoordinateEnclosureResourceError) as retained,
    ):
        lattice_basis("triangle", 3)
    assert retained.value.resource == "retained_basis"
    assert retained.value.requested > retained.value.limit
    empty = CoordinateEnclosureBudget(0, 0)
    with empty.activate(), pytest.raises(CoordinateEnclosureResourceError) as zero:
        constant(1, 2)
    assert zero.value.limit == zero.value.completed == 0
    assert zero.value.requested > 0
    assert empty.work_units == empty.temporary_bytes_upper == 0
