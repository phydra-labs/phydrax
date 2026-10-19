#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Iterator
from contextlib import contextmanager
from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._model._structure import (
    model_from_logical_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from phydrax.geometry import MeshingDomain, SphereMaterialCellAtlas
from phydrax.geometry._domain_numerics import interval_vector_cross, interval_vector_norm
from phydrax.geometry._interval_enclosure import interval_subtract
from phydrax.geometry._sphere_material_atlas import (
    float_enclosure,
    sphere_material_strata,
    sphere_radial_triangle_bounds,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.lifecycle import register_meshing_source_artifacts
from phydrax.linalg._small_batched import (
    exact_small_linear_action_bounds,
    prepare_exact_small_linear_actions,
)
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.recipes import recipe_field_position
from tests._support.sphere_material import accepted_sphere_material


# Source: the pre-remediation native-material regression acceptance limits.
# These empirical budgets remain unchanged through the phase and fixture cutover.
_SPHERE_RADIUS_ACCEPTANCE = 2.0e-13
_SPHERE_INVERSE_ACCEPTANCE = 2.0e-11
_SPHERE_AREA_ACCEPTANCE = 2.0e-11
_PROJECTIVE_REFERENCE_ACCEPTANCE = 2.0e-12
_PROJECTIVE_NONLINEARITY_FLOOR = 1.0e-12


@contextmanager
def _exact_action_budget(
    matrix: tuple[tuple[Fraction, ...], ...], right: tuple[tuple[Fraction, ...], ...], /
) -> Iterator[CoordinateEnclosureBudget]:
    work, storage = exact_small_linear_action_bounds(matrix, right)
    ticket = HostBatchTicket(work, bytes_upper=storage)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=work, maximum_memory_bytes=storage
    )
    with ticket, budget.activate():
        yield budget


def test_sphere_material_cells_cover_poles_with_regular_maps_and_true_area() -> None:
    domain, result, atlas = accepted_sphere_material(0.4)
    atlas.require_bound(domain, result.mesh, result.geometry, result.coordinate_contract)
    rows = jnp.arange(atlas.num_charts, dtype=jnp.int32)
    corners = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    corner_rows = jnp.broadcast_to(rows[:, None], (atlas.num_charts, 3))
    references = jnp.broadcast_to(corners, (atlas.num_charts, 3, 2))
    densities = np.asarray(atlas.jacobian(corner_rows, references))
    assert np.all(densities >= np.asarray(atlas.jacobian_lower)[:, None])
    points = np.asarray(atlas.map(corner_rows, references))
    np.testing.assert_allclose(
        np.linalg.norm(points, axis=-1), 1.0, rtol=0.0, atol=_SPHERE_RADIUS_ACCEPTANCE
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
    # The radial surface differs from the affine chord; the interpolation bound
    # contains that difference on the whole reference triangle.
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
        np.asarray(inverse.reference),
        np.asarray(reference),
        rtol=0.0,
        atol=_SPHERE_INVERSE_ACCEPTANCE,
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
    assert (
        abs(float(np.sum(values * quadrature_weights)) - 4.0 * np.pi)
        < _SPHERE_AREA_ACCEPTANCE
    )
    changed = eqx.tree_at(
        lambda value: value.directions, atlas, atlas.directions.at[0, 0, 0].add(0.01)
    )
    with pytest.raises(ValueError, match="sphere_material.atlas_id"):
        changed.require_bound(
            domain, result.mesh, result.geometry, result.coordinate_contract
        )


def test_constructor_and_result_preparation_retain_material_identity() -> None:
    domain, result, atlas = accepted_sphere_material(0.4)
    strata = sphere_material_strata(domain, result)
    direct = SphereMaterialCellAtlas(
        domain,
        result.mesh,
        result.geometry,
        result.coordinate_contract,
        cell_patches=strata.patches,
        corner_source_dimensions=strata.dimensions,
        corner_source_indices=strata.indices,
        corner_source_parameters=strata.parameters,
        corner_source_occurrence_paths=strata.occurrence_paths,
        corner_source_entity_ids=strata.entity_ids,
        maximum_fidelity=0.25 * 0.4 * 0.4,
    )
    direct.require_bound(domain, result.mesh, result.geometry, result.coordinate_contract)
    assert direct.atlas_id == atlas.atlas_id
    assert direct.work_units == atlas.work_units
    assert (
        direct.coordinate_preparation_work_units
        == atlas.coordinate_preparation_work_units
    )
    assert direct.coordinate_retained_basis_bytes == atlas.coordinate_retained_basis_bytes
    assert direct.coordinate_storage_upper_bytes == atlas.coordinate_storage_upper_bytes
    np.testing.assert_array_equal(direct.cell_global_ids, atlas.cell_global_ids)
    np.testing.assert_array_equal(
        direct.physical_corner_global_ids, atlas.physical_corner_global_ids
    )
    np.testing.assert_array_equal(direct.directions, atlas.directions)
    np.testing.assert_array_equal(
        direct.source_fidelity_bounds, atlas.source_fidelity_bounds
    )


def test_material_binding_refuses_changed_domain_content_with_stale_domain_id() -> None:
    domain, result, atlas = accepted_sphere_material(0.4)
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(domain)
    arrays = model_recipe_array_values(domain, recipe, prefix="sphere-material-domain")
    recipe["items"][recipe_field_position(recipe, "source_revision")] = (
        model_structure_recipe(domain.source_revision + "-changed")
    )
    changed = model_from_logical_array_recipe(
        recipe, arrays, prefix="sphere-material-domain"
    )
    assert isinstance(changed, MeshingDomain)
    assert changed.domain_id == domain.domain_id
    assert changed.content_domain_id() != domain.content_domain_id()
    with pytest.raises(ValueError, match="sphere_material.binding"):
        atlas.require_bound(
            changed, result.mesh, result.geometry, result.coordinate_contract
        )


def test_stored_material_validation_refuses_changed_direction_evidence() -> None:
    _, _, atlas = accepted_sphere_material(0.4)
    assert SphereMaterialCellAtlas.from_stored_fields(atlas) is atlas
    changed = eqx.tree_at(
        lambda value: value.directions,
        atlas,
        atlas.directions.at[0, 0, 0].add(0.01),
    )
    with pytest.raises(ValueError, match="sphere_material.atlas_id"):
        SphereMaterialCellAtlas.from_stored_fields(changed)


def test_distinct_sphere_material_cells_need_projective_reference_maps() -> None:
    _, _, source = accepted_sphere_material(0.4)
    _, _, target = accepted_sphere_material(0.32)
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
    with coordinate_reference_budget() as budget:
        direct = prepare_exact_small_linear_actions(matrix, frame, budget=budget)
    assert direct.actions is not None and mapping.exact_coefficients == direct.actions
    assert mapping.orientation_ratio == source.exact_determinants[0] / direct.determinant
    origin, epsilon = (Fraction(23, 100), Fraction(31, 100)), Fraction(1, 1000)
    piece = (origin, (origin[0] + epsilon, origin[1]), (origin[0], origin[1] + epsilon))
    bounds = mapping.certify_triangle(piece)
    assert bounds.denominator_lower > 0 and bounds.jacobian_lower > 0
    projected = mapping.map(reference)
    np.testing.assert_allclose(
        np.asarray(projected),
        coordinates[target_row],
        rtol=0.0,
        atol=_PROJECTIVE_REFERENCE_ACCEPTANCE,
    )
    np.testing.assert_allclose(
        np.asarray(target.map(jnp.asarray(target_row, dtype=jnp.int32), projected)),
        np.asarray(point),
        rtol=0.0,
        atol=_PROJECTIVE_REFERENCE_ACCEPTANCE,
    )
    ends = jnp.asarray(
        (
            (float(origin[0]), float(origin[1])),
            (float(origin[0] + epsilon), float(origin[1])),
        )
    )
    midpoint = np.asarray(mapping.map(jnp.mean(ends, axis=0)))
    corner_interpolant = np.mean(np.asarray(mapping.map(ends)), axis=0)
    assert np.max(np.abs(midpoint - corner_interpolant)) > _PROJECTIVE_NONLINEARITY_FLOOR
    with pytest.raises(ValueError, match="sphere_projective.vertices"):
        mapping.certify_triangle(((Fraction(-1), Fraction(0)), *piece[1:]))
    inconsistent = eqx.tree_at(
        lambda value: value.coefficients, mapping, jnp.zeros_like(mapping.coefficients)
    )
    with pytest.raises(ValueError, match="sphere_projective.coefficients"):
        inconsistent.certify_triangle(piece)
    with pytest.raises(ValueError, match="sphere_projective.coefficients"):
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
    with _exact_action_budget(matrix, right) as budget:
        prepared = prepare_exact_small_linear_actions(matrix, right, budget=budget)
    assert prepared.successful and prepared.rank == 2
    assert prepared.actions == ((Fraction(1), Fraction(-1)), (Fraction(0), Fraction(1)))
    with _exact_action_budget(
        ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))),
        ((Fraction(1),), (Fraction(2),)),
    ) as budget:
        singular = prepare_exact_small_linear_actions(
            ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))),
            ((Fraction(1),), (Fraction(2),)),
            budget=budget,
        )
    assert (
        singular.status == "singular" and singular.rank == 1 and singular.actions is None
    )
    with _exact_action_budget(matrix, ((), ())) as budget:
        determinant_only = prepare_exact_small_linear_actions(
            matrix, ((), ()), budget=budget
        )
    assert determinant_only.successful and determinant_only.rank == 2
    assert determinant_only.actions == ((), ())
    assert determinant_only.determinant == prepared.determinant == epsilon
    assert determinant_only.operation_count < prepared.operation_count
    with _exact_action_budget(
        ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))), ((), ())
    ) as budget:
        singular_determinant = prepare_exact_small_linear_actions(
            ((Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))),
            ((), ()),
            budget=budget,
        )
    assert singular_determinant.status == "singular"
    assert singular_determinant.rank == 1 and singular_determinant.determinant == 0
    assert singular_determinant.actions is None


@pytest.mark.parametrize(
    "rays",
    [
        np.eye(3, dtype=np.float64),
        np.eye(3, dtype=np.float64)[[1, 0, 2]],
        np.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        np.asarray(((1.0, 0.0, 0.0), (1.0, 2.0**-50, 0.0), (1.0, 0.0, 2.0**-50))),
    ],
    ids=["positive", "negative", "singular", "narrow"],
)
def test_radial_cone_bounds_preserve_exact_elimination_without_unused_actions(
    rays: np.ndarray,
) -> None:
    # The independent rational elimination owner is the prior determinant oracle.
    matrix = tuple(tuple(Fraction(float(value)) for value in row) for row in rays.T)
    with coordinate_reference_budget() as reference:
        reference.reserve(9)
        prepared = prepare_exact_small_linear_actions(
            matrix, ((), (), ()), budget=reference
        )
        expected = None
        if prepared.successful:
            first = interval_subtract((rays[1], rays[1]), (rays[0], rays[0]))
            second = interval_subtract((rays[2], rays[2]), (rays[0], rays[0]))
            first = (np.asarray(first[0]), np.asarray(first[1]))
            second = (np.asarray(second[0]), np.asarray(second[1]))
            normal_upper = float(
                interval_vector_norm(interval_vector_cross(first, second))[1]
            )
            floor = float_enclosure(abs(prepared.determinant) / Fraction(normal_upper))[0]
            lows, highs = interval_vector_norm((rays, rays))
            unit_error = max(
                max(Fraction(float(high)) - 1, 1 - Fraction(float(low)), Fraction(0))
                for low, high in zip(lows, highs, strict=True)
            )
            expected = prepared.determinant, floor, float(np.max(highs)), unit_error
    with coordinate_reference_budget() as current:
        actual = sphere_radial_triangle_bounds(rays, budget=current)
    assert actual == expected
    if prepared.successful:
        assert current.work_units < reference.work_units
    print(f"Radial determinant work: {reference.work_units} -> {current.work_units}")


def test_polynomial_budget_counts_dense_conversion_and_retained_basis_before_expansion() -> (
    None
):
    from hashlib import sha256
    from json import dumps

    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.discretization._coordinate_enclosure import (
        bernstein_coefficients,
        constant,
        lattice_basis,
        Polynomial,
    )

    polynomial: Polynomial = {
        (first, second): Fraction(1)
        for first in range(11)
        for second in range(11 - first)
    }
    insufficient_ticket = HostBatchTicket(198, bytes_upper=16 * 1024**2)
    insufficient = CoordinateEnclosureBudget(
        insufficient_ticket, maximum_work_units=198, maximum_memory_bytes=16 * 1024**2
    )
    with (
        insufficient_ticket,
        insufficient.activate(),
        pytest.raises(CoordinateEnclosureResourceError) as refused,
    ):
        bernstein_coefficients(polynomial, "simplex", 2, budget=insufficient)
    assert refused.value.resource == "coefficient_work"
    assert refused.value.requested > refused.value.limit
    assert refused.value.completed <= refused.value.limit
    complete_ticket = HostBatchTicket(20_000, bytes_upper=16 * 1024**2)
    complete = CoordinateEnclosureBudget(
        complete_ticket, maximum_work_units=20_000, maximum_memory_bytes=16 * 1024**2
    )
    with complete_ticket, complete.activate():
        coefficients = bernstein_coefficients(polynomial, "simplex", 2, budget=complete)
    assert complete.work_units >= 66 * 66
    # Independent endpoint values come from this dense degree-ten polynomial.
    assert min(coefficients) == 1 and max(coefficients) == 11

    def coefficient_bank_digest(basis: tuple[Polynomial, ...]) -> str:
        payload = [
            [
                [list(index), str(rational.numerator), str(rational.denominator)]
                for index, value in polynomial.items()
                for rational in (rational_coefficient(value, role="fixture coefficient"),)
            ]
            for polynomial in basis
        ]
        return sha256(dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()

    with coordinate_source_request(("triangle", 3)) as budget:
        expected_basis = lattice_basis("triangle", 3, budget=budget)
    expected_digest = coefficient_bank_digest(expected_basis)
    retention_requirement = budget.peak_bytes_upper
    assert retention_requirement > budget.retained_basis_bytes > 0
    assert budget.temporary_bytes_upper == 0
    # Retention must admit the basis and its live construction/index workspace,
    # not the discarded products of earlier basis terms.
    limited_storage_ticket = HostBatchTicket(
        100_000, bytes_upper=retention_requirement - 1
    )
    limited_storage = CoordinateEnclosureBudget(
        limited_storage_ticket,
        maximum_work_units=100_000,
        maximum_memory_bytes=retention_requirement - 1,
    )
    with (
        limited_storage_ticket,
        limited_storage.activate(),
        pytest.raises(CoordinateEnclosureResourceError) as retained,
    ):
        lattice_basis("triangle", 3, budget=limited_storage)
    assert retained.value.resource == "retained_basis"
    assert retained.value.requested > retained.value.limit
    assert retained.value.requested == retention_requirement
    assert retained.value.completed == 0
    assert limited_storage.retained_basis_bytes == 0
    assert limited_storage.retained_objects == {}
    assert limited_storage.lattice_cache == {}
    assert limited_storage.temporary_bytes_upper == 0
    boundary_ticket = HostBatchTicket(100_000, bytes_upper=retention_requirement)
    boundary = CoordinateEnclosureBudget(
        boundary_ticket,
        maximum_work_units=100_000,
        maximum_memory_bytes=retention_requirement,
    )
    with boundary_ticket, boundary.activate():
        boundary_basis = lattice_basis("triangle", 3, budget=boundary)
    assert coefficient_bank_digest(boundary_basis) == expected_digest
    assert boundary.peak_bytes_upper == retention_requirement
    assert boundary.retained_basis_bytes == budget.retained_basis_bytes
    assert boundary.work_units == budget.work_units
    assert boundary.temporary_bytes_upper == 0
    empty_ticket = HostBatchTicket(0, bytes_upper=0)
    empty = CoordinateEnclosureBudget(
        empty_ticket, maximum_work_units=0, maximum_memory_bytes=0
    )
    with (
        empty_ticket,
        empty.activate(),
        pytest.raises(CoordinateEnclosureResourceError) as zero,
    ):
        constant(1, 2, budget=empty)
    assert zero.value.limit == zero.value.completed == 0
    assert zero.value.requested > 0
    assert empty.work_units == empty.temporary_bytes_upper == 0
