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

import phydrax.discretization._sphere_chart_deformation as sphere_chart_deformation_owner
from phydrax._model._structure import (
    model_from_logical_array_recipe,
    model_recipe_array_values,
    model_structure_recipe,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.discretization._exact_algebraic_geometry import rational_coefficient
from phydrax.geometry import MeshingDomain, SphereMaterialCellAtlas
from phydrax.geometry._domain_numerics import interval_vector_cross, interval_vector_norm
from phydrax.geometry._interval_enclosure import interval_subtract
from phydrax.geometry._sphere_material_atlas import (
    float_enclosure,
    sphere_material_strata,
    sphere_radial_triangle_bounds,
    SphereProjectiveActionBank,
)
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.lifecycle import register_meshing_source_artifacts
from phydrax.linalg._small_batched import (
    confirm_exact_action_3,
    exact_adjugate_action,
    exact_column_height_3,
    exact_small_linear_action_bounds,
    prepare_exact_adjugate_3,
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


def test_material_source_enclosures_are_batched_without_changing_bounds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.geometry._sphere_material_atlas as owner

    domain, result, prior = accepted_sphere_material(0.4)
    original = owner.source_image_enclosure
    sizes: list[int] = []

    def checked_batch(source, directions, *, budget):
        sizes.append(directions.shape[0])
        actual = original(source, directions, budget=budget)
        # Compare to the old per-triangle interval operation, including all
        # directed-rounding endpoints rather than only a sampled tolerance.
        with coordinate_reference_budget() as reference_budget:
            reference = [
                original(source, triangle, budget=reference_budget)
                for triangle in directions.reshape((-1, 3, 3))
            ]
        for endpoint in range(2):
            np.testing.assert_array_equal(
                actual[endpoint],
                np.concatenate([bounds[endpoint] for bounds in reference]),
            )
        return actual

    monkeypatch.setattr(owner, "source_image_enclosure", checked_batch)
    atlas = owner.sphere_material_atlas_from_result(domain, result, maximum_fidelity=0.04)
    assert len(sizes) == np.unique(np.asarray(atlas.patches)).size
    assert sum(sizes) == 3 * atlas.num_charts
    np.testing.assert_array_equal(
        atlas.source_corner_enclosure_bounds, prior.source_corner_enclosure_bounds
    )
    np.testing.assert_array_equal(
        atlas.source_fidelity_bounds, prior.source_fidelity_bounds
    )


def test_material_atlas_shares_exact_geometry_bank_with_later_certificates() -> None:
    from phydrax.discretization._coordinate_enclosure import (
        prepared_coordinate_source_bank,
    )
    from phydrax.geometry._sphere_material_atlas import sphere_material_atlas_from_result

    domain, source, _ = accepted_sphere_material(0.4)
    queries = 3 * sum(block.cell_count for block in source.mesh.blocks)
    with coordinate_reference_budget(queries_upper=queries) as budget:
        atlas = sphere_material_atlas_from_result(
            domain, source, maximum_fidelity=0.04, budget=budget
        )
        before = budget.work_units
        bank = prepared_coordinate_source_bank(source.geometry, budget=budget)
        assert budget.work_units == before
        assert bank is prepared_coordinate_source_bank(
            atlas.source_geometry, budget=budget
        )
        changed = eqx.tree_at(
            lambda geometry: geometry.coordinates,
            source.geometry,
            source.geometry.coordinates.at[0, 0].add(0.01),
        )
        replacement = prepared_coordinate_source_bank(changed, budget=budget)
        assert replacement is not bank
        assert budget.work_units > before


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


def test_stored_material_coordinate_validation_uses_actual_caller_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._cell_geometry as geometry_owner

    domain, result, atlas = accepted_sphere_material(0.4)
    original = geometry_owner.CellGeometrySpec.resolve
    seen = []

    def resolve(geometry, mesh, /, *, exact_source_prepared=False, budget=None):
        assert budget is caller
        seen.append(geometry)
        return original(
            geometry, mesh, exact_source_prepared=exact_source_prepared, budget=budget
        )

    def forbid_preparation(*args, **kwargs):
        raise AssertionError("explicit restoration must not open a helper preparation")

    monkeypatch.setattr(geometry_owner.CellGeometrySpec, "resolve", resolve)
    monkeypatch.setattr(
        geometry_owner, "owned_coordinate_preparation", forbid_preparation
    )
    with coordinate_reference_budget() as caller:
        assert SphereMaterialCellAtlas.from_stored_fields(atlas, budget=caller) is atlas
        atlas.require_bound(
            domain, result.mesh, result.geometry, result.coordinate_contract,
            budget=caller,
        )
        assert caller.work_units >= 4 * atlas.num_charts
    assert len(seen) == 2
    assert seen[0] is atlas.source_geometry
    assert seen[1] is result.geometry


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
    with pytest.raises(TypeError, match="sphere_projective.budget"):
        source.projective_reference_map(
            int(np.asarray(source.cell_global_ids)[0]),
            target,
            int(np.asarray(target.cell_global_ids)[target_row]),
        )
    with coordinate_reference_budget() as budget:
        mapping = source.projective_reference_map(
            int(np.asarray(source.cell_global_ids)[0]),
            target,
            int(np.asarray(target.cell_global_ids)[target_row]),
            budget=budget,
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
    # Projective target areas follow the homogeneous determinant identity
    # exactly, including a fan with a shared collinear edge subdivision.
    midpoint = (piece[1][0] + piece[2][0]) / 2, (piece[1][1] + piece[2][1]) / 2
    polygon = piece[0], piece[1], midpoint, piece[2]
    images = mapping.reference_images(polygon)
    source_areas = sphere_chart_deformation_owner._fan_areas(polygon)
    assert sphere_chart_deformation_owner._projective_fan_areas(
        mapping, images, source_areas
    ) == sphere_chart_deformation_owner._fan_areas(tuple(image[1] for image in images))
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
    with coordinate_reference_budget() as budget:
        same_source = source.projective_reference_map(
            original_id, source, original_id, budget=budget
        )
        same_source.require_bound(source, source, budget=budget)
    assert same_source.exact_coefficients == (
        (Fraction(1), Fraction(-1), Fraction(-1)),
        (Fraction(0), Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(0), Fraction(1)),
    )
    np.testing.assert_allclose(
        same_source.map(reference), reference, rtol=0.0, atol=4 * np.finfo(np.float64).eps
    )


def test_authenticated_projective_action_bank_reproduces_elimination_maps() -> None:
    _, _, source = accepted_sphere_material(0.4)
    _, _, target = accepted_sphere_material(0.32)
    source_ids = np.asarray(source.cell_global_ids)
    target_ids = np.asarray(target.cell_global_ids)
    with coordinate_reference_budget() as budget:
        sources, targets, _ = sphere_chart_deformation_owner._candidates(
            source, target, 10**6, 10**9, budget=budget
        )
        unknown_columns = 0
        with budget.live_storage() as storage:
            actions = SphereProjectiveActionBank.authenticate(
                source, target, budget=budget, storage=storage
            )
            for slot, (source_row, target_row) in enumerate(
                zip(sources, targets, strict=True)
            ):
                source_id = int(source_ids[source_row])
                target_id = int(target_ids[target_row])
                banked = source.projective_reference_map(
                    source_id, target, target_id, budget=budget, actions=actions
                )
                shared = set(target.corner_tokens[int(target_row)])
                unknown_columns += sum(
                    token not in shared for token in source.corner_tokens[int(source_row)]
                )
                # Compare a fixed deterministic stride of the banked maps
                # exactly with an independent Gaussian elimination.
                if slot % 6:
                    continue
                old = tuple(
                    tuple(Fraction(float(value)) for value in row)
                    for row in np.asarray(source.directions)[source_row]
                )
                matrix = tuple(
                    tuple(Fraction(float(value)) for value in row)
                    for row in np.asarray(target.directions)[target_row].T
                )
                frame = tuple(
                    (
                        old[0][axis],
                        old[1][axis] - old[0][axis],
                        old[2][axis] - old[0][axis],
                    )
                    for axis in range(3)
                )
                eliminated = prepare_exact_small_linear_actions(
                    matrix, frame, budget=budget
                )
                assert eliminated.actions == banked.exact_coefficients
                assert (
                    banked.orientation_ratio
                    == source.exact_determinants[int(source_row)] / eliminated.determinant
                    == source.exact_determinants[int(source_row)]
                    / target.exact_determinants[int(target_row)]
                )
                # A separately owned bank derives the same identity, and the
                # stored columns verify against the shared bank.
                standalone = source.projective_reference_map(
                    source_id, target, target_id, budget=budget
                )
                assert standalone.map_id == banked.map_id
                standalone.require_bound(source, target, budget=budget, actions=actions)
                standalone.require_bound(source, target, budget=budget)
            # Every requested column is shared by all source cells at its ray.
            assert 0 < len(actions.columns) < unknown_columns
            assert set(actions.inverses) <= {int(row) for row in targets}
            (row, key), column = next(iter(actions.columns.items()))
            ray = next(
                actions.cell_corners(False, int(source_row))[slot]
                for source_row in sources
                for slot in range(3)
                if actions.corner_key(False, int(source_row), slot) == key
            )
            assert actions.confirm_column(row, ray, key, column)
            fresh_bank = SphereProjectiveActionBank.authenticate(
                source, target, budget=budget, storage=storage
            )
            wrong = (column[0] + Fraction(1, 2**80), *column[1:])
            assert not fresh_bank.confirm_column(row, ray, key, wrong)
            assert fresh_bank.confirm_column(row, ray, key, column)
            with pytest.raises(ValueError, match="sphere_projective.actions"):
                target.projective_reference_map(
                    int(target_ids[0]), source, int(source_ids[0]), actions=actions
                )


def test_exact_adjugate_actions_prove_and_confirm_unique_solutions() -> None:
    columns = (
        (Fraction(2), Fraction(0), Fraction(1)),
        (Fraction(1), Fraction(3), Fraction(0)),
        (Fraction(0), Fraction(1), Fraction(1, 2)),
    )
    right = (Fraction(1), Fraction(-2), Fraction(5, 3))
    matrix = tuple(zip(*columns, strict=True))
    with coordinate_reference_budget() as budget:
        adjugate = prepare_exact_adjugate_3(columns, budget=budget)
        assert adjugate is not None
        solved = exact_adjugate_action(adjugate, right, budget=budget)
        eliminated = prepare_exact_small_linear_actions(
            matrix, tuple((value,) for value in right), budget=budget
        )
        assert eliminated.actions is not None
        assert solved == tuple(row[0] for row in eliminated.actions)
        assert adjugate.determinant == eliminated.determinant
        height = exact_column_height_3(columns, budget=budget)
        assert confirm_exact_action_3(columns, height, solved, right, budget=budget)
        changed = (solved[0], solved[1] + Fraction(1, 7), solved[2])
        assert not confirm_exact_action_3(columns, height, changed, right, budget=budget)
        before = budget.work_units
        assert (
            prepare_exact_adjugate_3((columns[0], columns[1], columns[0]), budget=budget)
            is None
        )
        # The singular attempt charges the same profile and adjugate visits.
        assert budget.work_units - before == 9 + 72


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


def test_radial_batch_shares_only_identical_current_direction_coefficients() -> None:
    from phydrax.geometry._sphere_material_atlas import _sphere_radial_bound_rows

    points = np.asarray(
        ((1.0, 0.0, 0.0), (0.9, 0.3, 0.0), (0.9, 0.0, 0.3), (0.9, -0.3, 0.0))
    )
    rays = points[np.asarray(((0, 1, 2), (0, 2, 3), (0, 1, 2)))]
    with coordinate_reference_budget() as reference:
        expected = [
            sphere_radial_triangle_bounds(triangle, budget=reference) for triangle in rays
        ]
    with coordinate_reference_budget() as current:
        actual = list(_sphere_radial_bound_rows(rays, budget=current))
        assert actual == expected
        assert current.work_units == 3 * 4 + 14 * 3
        assert current.work_units < reference.work_units
        assert current.temporary_bytes_upper == 0
        rays[2, 0, 0] = np.nextafter(rays[2, 0, 0], np.inf)
        changed = list(_sphere_radial_bound_rows(rays, budget=current))
    with coordinate_reference_budget() as reference:
        assert changed == [
            sphere_radial_triangle_bounds(triangle, budget=reference) for triangle in rays
        ]
    assert changed[:2] == actual[:2]
    assert changed[2] != actual[2]


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


def test_same_ray_chord_bound_replaces_coarse_triangle_displacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.discretization._sphere_chart_deformation as deformation_owner
    from phydrax.discretization._sphere_chart_deformation import (
        prepare_sphere_chart_deformation,
    )

    _, _, source = accepted_sphere_material(1.2)
    _, _, target = accepted_sphere_material(1.0)
    fidelities = [np.asarray(atlas.source_fidelity_bounds) for atlas in (source, target)]
    chords = max(
        float(
            np.max(
                np.asarray(atlas.source_polynomial_chord_bounds)
                + np.asarray(atlas.source_corner_enclosure_bounds)
            )
        )
        for atlas in (source, target)
    )
    # Both coordinate charts lie on one projective ray, so their chord radii do
    # not add; the coarse triangle-inequality sum would need coefficient maps.
    tolerance = max(float(np.max(value)) for value in fidelities) + 2 * chords + 1e-9

    def unavailable(*args: object, **kwargs: object) -> None:
        raise AssertionError("same-ray chords must not need full coefficient maps")

    monkeypatch.setattr(
        deformation_owner, "_sphere_full_coordinate_expressions", unavailable
    )
    chart = prepare_sphere_chart_deformation(
        source, target, maximum_fidelity=0.3, maximum_displacement=tolerance
    )
    source_bounds, target_bounds = (
        np.asarray(chart.source_fidelity_bounds),
        np.asarray(chart.target_fidelity_bounds),
    )
    assert any(
        source_bounds[piece.source_row] + target_bounds[piece.target_row] > tolerance
        for piece in chart.pieces
    )
    assert chart.maximum_displacement_bound <= tolerance
    corners = [
        np.concatenate(
            [
                np.asarray(mesh.coordinates)[np.asarray(block.vertices)]
                for block in mesh.blocks
            ]
        )
        for mesh in (chart.source_mesh, chart.target_mesh)
    ]
    for piece in chart.pieces:
        vertices = np.asarray(piece.source_reference_vertices)
        references = np.concatenate(
            (
                vertices,
                np.mean(vertices, axis=0)[None],
                (np.asarray((0.2, 0.3, 0.5)) @ vertices)[None],
            )
        )
        homogeneous = (
            np.column_stack((np.ones(references.shape[0]), references))
            @ np.asarray(piece.projective_map.coefficients).T
        )
        mapped = homogeneous[:, 1:] / np.sum(homogeneous, axis=1)[:, None]
        old = np.column_stack((1.0 - np.sum(references, axis=1), references))
        new = np.column_stack((1.0 - np.sum(mapped, axis=1), mapped))
        displacement = np.linalg.norm(
            old @ corners[0][piece.source_row] - new @ corners[1][piece.target_row],
            axis=1,
        )
        assert np.max(displacement) <= piece.displacement_bound + 1e-12


def test_projective_polygon_images_bound_actual_expression_growth_and_resources() -> None:
    _, _, source = accepted_sphere_material(0.4)
    _, _, target = accepted_sphere_material(0.32)
    owner = sphere_chart_deformation_owner
    with coordinate_reference_budget() as budget, budget.live_storage() as storage:
        actions = SphereProjectiveActionBank.authenticate(
            source, target, budget=budget, storage=storage
        )
        source_rows, target_rows, _ = owner._candidates(
            source, target, 10**6, 10**9, budget=budget
        )
        for source_row, target_row in zip(source_rows, target_rows, strict=True):
            mapping = source.projective_reference_map(
                int(source.cell_global_ids[source_row]),
                target,
                int(target.cell_global_ids[target_row]),
                budget=budget,
                actions=actions,
            )
            polygon, _, keys, corners, _ = owner._native_polygon(
                mapping,
                10**8,
                10**7,
                owner._support_edges(actions, int(source_row), int(target_row)),
                {},
                storage,
                source_row=int(source_row),
                target_corner_keys=tuple(
                    actions.corner_key(True, int(target_row), slot) for slot in range(3)
                ),
                target_corner_points={},
                budget=budget,
            )
            if len(polygon) >= 3:
                break
        else:
            raise AssertionError("actual native source/target cells have no overlap")
    expected = mapping.reference_images(polygon)
    operands = (
        *mapping.exact_coefficients,
        *polygon,
        mapping.exact_denominator,
        (mapping.orientation_ratio,),
    )
    bits = max(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        for row in operands for value in row
    )
    upper_bits = 54 * bits + 32
    source_areas = owner._fan_areas(polygon)
    target_areas = owner._projective_fan_areas(mapping, expected, source_areas)
    final_values = [
        *(value for scale, point in expected for value in (scale, *point)),
        *source_areas,
        *target_areas,
    ]
    for slot in range(1, len(polygon) - 1):
        bounds = mapping.triangle_bounds(
            (expected[0][0], expected[slot][0], expected[slot + 1][0])
        )
        final_values.extend(
            (bounds.denominator_lower, bounds.denominator_upper,
             bounds.jacobian_lower, bounds.jacobian_upper)
        )
    assert all(
        max(abs(value.numerator).bit_length(), value.denominator.bit_length())
        <= upper_bits
        for value in final_values
    )
    assert target_areas == owner._fan_areas(tuple(point for _, point in expected))

    def images(work: int, memory: int) -> tuple[int, int]:
        ticket = HostBatchTicket(work, bytes_upper=memory)
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory
        )
        with ticket, ledger.activate(), ledger.live_storage() as retained:
            actual = owner._sphere_polygon_images(
                mapping, polygon, keys, corners, int(target_row), {},
                retained, budget=ledger,
            )
            assert actual == expected
        return ledger.work_units, ledger.peak_bytes_upper

    work, memory = images(10**9, 10**9)
    assert images(work, memory) == (work, memory)
    with pytest.raises(CoordinateEnclosureResourceError) as work_refused:
        images(work - 1, memory)
    assert work_refused.value.resource == "coefficient_work"
    with pytest.raises(CoordinateEnclosureResourceError) as memory_refused:
        images(work, memory - 1)
    assert memory_refused.value.resource == "polynomial_storage"
