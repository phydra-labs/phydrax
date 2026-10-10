#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

import numpy as np
import pytest

from phydrax.geometry import DelaunayTriangulation


@pytest.mark.parametrize("dimension", [2, 3])
def test_qhull_delaunay_covers_the_hull_with_oriented_canonical_simplices(
    dimension: int,
) -> None:
    rng = np.random.default_rng(5)
    corners = np.stack(
        np.meshgrid(*([0.0, 1.0],) * dimension, indexing="ij"), axis=-1
    ).reshape(-1, dimension)
    points = np.concatenate((corners, rng.uniform(0.05, 0.95, (40, dimension))))
    triangulation = DelaunayTriangulation(points, provider="qhull")
    simplices = np.asarray(triangulation.simplices)
    vertices = points[simplices]
    determinants = np.linalg.det(np.swapaxes(vertices[:, 1:] - vertices[:, :1], 1, 2))
    evidence = triangulation.evidence

    assert np.all(determinants > 0.0)
    # Unit-cube convex hull volume, independent of the triangulation.
    np.testing.assert_allclose(
        np.sum(determinants) / np.prod(np.arange(1, dimension + 1)), 1.0
    )
    np.testing.assert_array_equal(np.unique(simplices), np.arange(points.shape[0]))
    np.testing.assert_array_equal(simplices, simplices[np.lexsort(simplices.T[::-1])])
    assert evidence.provider == "qhull"
    assert evidence.predicate_mode == "filtered"
    assert evidence.duplicate_count == 0 and evidence.redundant_count == 0
    repeated = DelaunayTriangulation(points, provider="qhull")
    np.testing.assert_array_equal(repeated.simplices, simplices)
    assert repeated.evidence.evidence_id == evidence.evidence_id


def test_qhull_delaunay_maps_duplicates_and_refuses_invalid_selectors() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)))
    triangulation = DelaunayTriangulation(points, provider="qhull")

    assert int(triangulation.vertex_map[4]) == 1
    assert triangulation.evidence.duplicate_count == 1
    assert 4 not in np.asarray(triangulation.simplices)
    with pytest.raises(ValueError, match="provider"):
        DelaunayTriangulation(points, provider="exact")  # ty: ignore[invalid-argument-type]


def test_periodic_power_preparation_preserves_original_source_and_complete_weighted_images() -> (
    None
):
    from itertools import product

    from phydrax.discretization import PeriodicCell
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    sites = np.asarray(((0.125, 0.25, 0.5), (2.0, -1.0, 0.25)))
    weights = np.asarray((0.0, 8.0))
    domain = np.asarray(tuple(product((0.0, 1.0), repeat=3)))
    cell = PeriodicCell(np.eye(3))
    prepared = PeriodicPowerPreparation(
        sites,
        weights,
        domain,
        cell,
        maximum_images=100_000,
    )
    np.testing.assert_array_equal(prepared.points, sites)
    np.testing.assert_array_equal(prepared.weights, weights)
    bank = {
        (int(owner), tuple(action))
        for owner, action in zip(
            prepared.image_sites, prepared.image_exponents, strict=True
        )
    }
    # Independent original-site power oracle; do not use preparation radius or
    # image coordinates to decide whether an image must have been retained.
    for point in domain:
        reference = np.sum((point - sites[0]) ** 2) - weights[0]
        for action in product(range(-6, 7), repeat=3):
            for owner, site in enumerate(sites):
                if np.sum((point - site - action) ** 2) - weights[owner] <= reference:
                    assert (owner, action) in bank
    for array in (
        prepared.points,
        prepared.weights,
        prepared.domain_points,
        prepared.generators,
        prepared.image_sites,
        prepared.image_exponents,
    ):
        assert not array.flags.writeable
    repeated = PeriodicPowerPreparation(
        prepared.points,
        prepared.weights,
        prepared.domain_points,
        prepared.periodic_group,
        maximum_images=prepared.maximum_images,
        maximum_work_units=prepared.maximum_work_units,
    )
    assert prepared.preparation_id == repeated.preparation_id
    assert prepared.spent_work > 0
    assert prepared.spent_work == repeated.spent_work
    assert prepared.native_charged_work == 0
    changed = PeriodicPowerPreparation(
        sites,
        weights + 1.0,
        domain,
        cell,
        maximum_images=100_000,
    )
    assert prepared.source_id != changed.source_id
    historical_receipt = (prepared.spent_work, prepared.native_charged_work)
    prepared.validate_restored()
    assert (prepared.spent_work, prepared.native_charged_work) == historical_receipt


def test_periodic_power_preparation_refuses_image_capacity_without_source_mutation() -> (
    None
):
    from phydrax.discretization import PeriodicCell
    from phydrax.geometry._triangulation import (
        PeriodicPowerImageCapacityRefusal,
        PeriodicPowerPreparation,
    )

    points = np.asarray(((0.25, 0.25, 0.25),))
    weights = np.asarray((0.0,))
    carrier = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    original = points.copy()
    with pytest.raises(PeriodicPowerImageCapacityRefusal) as caught:
        PeriodicPowerPreparation(
            points,
            weights,
            carrier,
            PeriodicCell(np.eye(3)),
            maximum_images=1,
        )
    assert caught.value.requested_images > caught.value.maximum_images
    assert caught.value.completed_images == 0
    assert caught.value.maximum_images == 1
    assert caught.value.stage == "power-images"
    np.testing.assert_array_equal(caught.value.points, original)
    np.testing.assert_array_equal(caught.value.weights, weights)
    np.testing.assert_array_equal(points, original)
    with pytest.raises(ValueError, match="weights"):
        PeriodicPowerPreparation(
            points,
            np.asarray((np.inf,)),
            carrier,
            PeriodicCell(np.eye(3)),
            maximum_images=100,
        )


def test_periodic_power_preparation_finite_group_keeps_action_and_site_axes_distinct() -> (
    None
):
    from phydrax.discretization._periodic_topology import PeriodicIsometryGroup
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    generator = np.diag((-1.0, -1.0, 1.0, 1.0))[None]
    group = PeriodicIsometryGroup(generator, tolerance=0.0)
    sites = np.asarray(((0.5, 0.0, 0.0), (0.25, 0.25, 0.5)))
    preparation = PeriodicPowerPreparation(
        sites,
        np.asarray((0.0, 0.25)),
        np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))),
        group,
        maximum_images=4,
    )
    np.testing.assert_array_equal(preparation.image_sites, (0, 1, 0, 1))
    np.testing.assert_array_equal(preparation.image_exponents, ((0,), (0,), (1,), (1,)))
    assert preparation.orders == (2,)


def test_periodic_power_image_expansions_retain_unrepresentable_translated_source_bits() -> (
    None
):
    from fractions import Fraction

    from phydrax.discretization import PeriodicCell
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    sites = np.asarray(((2.0**-54, 0.25, 0.25),))
    preparation = PeriodicPowerPreparation(
        sites,
        np.zeros(1),
        np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
        PeriodicCell(np.eye(3)),
        maximum_images=10_000,
    )
    offsets = preparation.image_coordinate_offsets
    components = preparation.image_coordinate_components
    for row, action in enumerate(preparation.image_exponents):
        for axis in range(3):
            coordinate = 3 * row + axis
            exact = sum(
                (
                    Fraction(float(value))
                    for value in components[offsets[coordinate] : offsets[coordinate + 1]]
                ),
                Fraction(0),
            )
            assert exact == Fraction(float(sites[0, axis])) + int(action[axis])
    assert np.max(np.diff(offsets)) > 1


def test_periodic_power_mixed_group_uses_complete_bounded_exact_actions() -> None:
    from fractions import Fraction
    from itertools import product

    from phydrax.discretization._periodic_topology import PeriodicIsometryGroup
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    generators = np.repeat(np.eye(4)[None], 2, axis=0)
    generators[0] = np.diag((-1.0, -1.0, 1.0, 1.0))
    generators[1, 2, 3] = 1.0
    group = PeriodicIsometryGroup(generators, tolerance=0.0)
    sites = np.asarray(((0.25, 0.5, 2.0**-54),))
    domain = np.asarray(tuple(product((-1.0, 1.0), repeat=3)))
    preparation = PeriodicPowerPreparation(
        sites,
        np.zeros(1),
        domain,
        group,
        maximum_images=1000,
    )
    bank = {tuple(action) for action in preparation.image_exponents}
    for point in domain:
        reference = np.sum((point - sites[0]) ** 2)
        for rotation, translation in product(range(2), range(-5, 6)):
            image = sites[0].copy()
            image[:2] *= (-1) ** rotation
            image[2] += translation
            if np.sum((point - image) ** 2) <= reference:
                assert (rotation, translation) in bank
    offsets, components = (
        preparation.image_coordinate_offsets,
        preparation.image_coordinate_components,
    )
    for image, action in enumerate(preparation.image_exponents):
        first, last = offsets[3 * image + 2 : 3 * image + 4]
        exact_z = sum((Fraction(float(x)) for x in components[first:last]), Fraction(0))
        assert exact_z == Fraction(float(sites[0, 2])) + int(action[1])
    assert preparation.orders == (2, 0)


def test_exact_periodic_image_clipping_preserves_volume_reciprocity_and_original_sites() -> (
    None
):
    from itertools import permutations, product

    from phydrax.discretization import PeriodicCell
    from phydrax.geometry._triangulation import (
        PeriodicPowerPreparation,
        RestrictedPowerDiagram,
    )

    carrier = np.asarray(tuple(product((0.0, 1.0), repeat=3)))
    vertex = {tuple(point): index for index, point in enumerate(carrier)}
    tetrahedra = []
    for order in permutations(range(3)):
        corner = np.zeros(3)
        cell = [vertex[tuple(corner)]]
        for axis in order:
            corner[axis] = 1.0
            cell.append(vertex[tuple(corner)])
        coordinates = carrier[cell]
        if np.linalg.det(coordinates[1:] - coordinates[0]) < 0:
            cell[0], cell[1] = cell[1], cell[0]
        tetrahedra.append(cell)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
    facets = np.full((6, 4), -1, dtype=np.int32)
    for cell, corners in enumerate(tetrahedra):
        for opposite in range(4):
            face = carrier[np.delete(corners, opposite)]
            for axis, side in product(range(3), range(2)):
                if np.all(face[:, axis] == side):
                    facets[cell, opposite] = 2 * axis + side
    sites = np.asarray(((0.25, 0.5, 0.5), (0.75, 0.5, 0.5)))
    weights = np.zeros(2)
    group = PeriodicCell(np.eye(3))
    preparation = PeriodicPowerPreparation(
        sites,
        weights,
        carrier,
        group,
        maximum_images=1000,
    )
    diagram = RestrictedPowerDiagram(
        sites,
        weights,
        carrier,
        tetrahedra,
        np.zeros(6, dtype=np.int32),
        facets,
        periodic_group=group,
        periodic_preparation=preparation,
        maximum_images=1000,
        work_limit=1_000_000,
    )
    data = diagram.construction
    np.testing.assert_array_equal(diagram.points, sites)
    np.testing.assert_array_equal(diagram.weights, weights)
    np.testing.assert_array_equal(
        diagram.cell_original_sites,
        preparation.image_sites[data.cell_sites],
    )
    np.testing.assert_array_equal(
        diagram.piece_original_sites,
        preparation.image_sites[data.piece_sites],
    )
    np.testing.assert_allclose(
        np.bincount(
            diagram.cell_original_sites,
            weights=data.cell_volumes,
            minlength=2,
        ),
        (0.5, 0.5),
        rtol=0.0,
        atol=1.0e-13,
    )
    # Independently integrate every outward loop by the divergence theorem.
    # Interior faces enter its two cells with opposite orientation, rather than
    # being counted as boundary faces of a silently clipped nonperiodic mesh.
    face_volume = np.zeros(len(data.cell_sites))
    internal_count = 0
    face_cells = np.asarray(data.face_cells, dtype=np.int64)
    for face in range(face_cells.shape[0]):
        first = int(face_cells[face, 0])
        second = int(face_cells[face, 1])
        loop = data.vertices[
            data.face_vertices[data.face_offsets[face] : data.face_offsets[face + 1]]
        ]
        contribution = sum(
            np.dot(loop[0], np.cross(loop[index], loop[index + 1])) / 6.0
            for index in range(1, len(loop) - 1)
        )
        face_volume[first] += contribution
        if second >= 0:
            internal_count += 1
            face_volume[second] -= contribution
    assert internal_count > 0
    np.testing.assert_allclose(face_volume, data.cell_volumes, rtol=0.0, atol=1.0e-13)
    assert np.all(data.piece_tets < len(tetrahedra))
    assert np.all(data.vertex_sites < len(preparation.image_sites))


def test_periodic_power_adjacency_refusal_retains_source_and_reuse_rejects_foreign_bytes() -> (
    None
):
    from phydrax._meshcore import MeshcoreStatus
    from phydrax.discretization import PeriodicCell
    from phydrax.geometry._triangulation import (
        PeriodicPowerPreparation,
        PeriodicPowerSourceRefusal,
        RestrictedPowerDiagram,
    )

    sites = np.asarray(((0.25, 0.25, 0.25),))
    weights = np.zeros(1)
    carrier = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )
    group = PeriodicCell(np.eye(3))
    preparation = PeriodicPowerPreparation(
        sites,
        weights,
        carrier,
        group,
        maximum_images=1000,
    )
    arguments = (
        carrier,
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )
    with pytest.raises(PeriodicPowerSourceRefusal) as caught:
        RestrictedPowerDiagram(
            sites,
            weights,
            *arguments,
            periodic_preparation=preparation,
            work_limit=1,
        )
    assert caught.value.status == MeshcoreStatus.CAPACITY_EXCEEDED
    assert caught.value.preparation.source_id == preparation.source_id
    assert caught.value.maximum_work == 1
    assert caught.value.requested_work is not None and caught.value.requested_work > 1
    assert caught.value.completed_work == 0
    np.testing.assert_array_equal(caught.value.preparation.points, sites)
    with pytest.raises(ValueError, match="original source"):
        RestrictedPowerDiagram(
            sites + 0.125,
            weights,
            *arguments,
            periodic_preparation=preparation,
        )


def test_periodic_power_preparation_uses_original_coefficient_work_ledger() -> None:
    from phydrax.discretization import PeriodicCell
    from phydrax.discretization._coordinate_enclosure import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    sites = np.asarray(((0.25, 0.25, 0.25),))
    weights = np.zeros(1)
    carrier = np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)))
    group = PeriodicCell(np.eye(3))
    ledger = CoordinateEnclosureBudget(1_000_000, 1_000_000)
    with ledger.activate():
        preparation = PeriodicPowerPreparation(
            sites,
            weights,
            carrier,
            group,
            maximum_images=1000,
            maximum_work_units=1_000_000,
        )
    assert preparation.spent_work == ledger.work_units
    assert ledger.native_charged_work_units == 0
    refused = CoordinateEnclosureBudget(1, 1_000_000)
    with refused.activate(), pytest.raises(CoordinateEnclosureResourceError) as caught:
        PeriodicPowerPreparation(
            sites,
            weights,
            carrier,
            group,
            maximum_images=1000,
            maximum_work_units=1_000_000,
        )
    assert caught.value.resource == "coefficient_work"
    assert caught.value.limit == 1
    assert caught.value.requested > caught.value.completed
    assert caught.value.completed == refused.work_units
    np.testing.assert_array_equal(sites, ((0.25, 0.25, 0.25),))


def test_periodic_power_refuses_numerically_commuting_but_nonclosed_scientific_group() -> (
    None
):
    from phydrax.discretization._periodic_topology import (
        PeriodicIsometryGroup,
        PeriodicIsometryIdentityError,
    )
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    generators = np.repeat(np.eye(4)[None], 2, axis=0)
    generators[0] = np.diag((-1.0, -1.0, 1.0, 1.0))
    generators[1, 0, 3] = 2.0**-40
    generators[1, 2, 3] = 1.0
    sites = np.asarray(((0.25, 0.25, 0.25),))
    with pytest.raises(PeriodicIsometryIdentityError, match="commute exactly"):
        group = PeriodicIsometryGroup(generators, tolerance=1.0e-10)
        PeriodicPowerPreparation(
            sites,
            np.zeros(1),
            np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))),
            group,
            maximum_images=1000,
        )
    assert generators[1, 0, 3] == 2.0**-40


def test_periodic_power_screw_images_keep_original_generator_axis_and_exact_source() -> (
    None
):
    from fractions import Fraction
    from itertools import product

    from phydrax.discretization._periodic_topology import PeriodicIsometryGroup
    from phydrax.geometry._triangulation import PeriodicPowerPreparation

    generator = np.diag((1.0, -1.0, -1.0, 1.0))
    generator[:3, 3] = 1.0
    group = PeriodicIsometryGroup(generator[None], tolerance=0.0)
    sites = np.asarray(((0.25, 0.25, 0.25),))
    carrier = np.asarray(tuple(product((0.0, 1.0), repeat=3)))
    preparation = PeriodicPowerPreparation(
        sites,
        np.zeros(1),
        carrier,
        group,
        maximum_images=1000,
    )
    assert preparation.orders == (0,)
    assert group.linear_orders == (2,)
    assert group.translation_periods == (2,)
    bank = {int(action[0]) for action in preparation.image_exponents}
    for point in carrier:
        reference = np.sum((point - sites[0]) ** 2)
        for n in range(-10, 11):
            yz = 0.25 if n % 2 == 0 else 0.75
            image = np.asarray((0.25 + n, yz, yz))
            if np.sum((point - image) ** 2) <= reference:
                assert n in bank
    offsets, components = (
        preparation.image_coordinate_offsets,
        preparation.image_coordinate_components,
    )
    for image, action in enumerate(preparation.image_exponents):
        exact = tuple(
            sum(
                (
                    Fraction(float(value))
                    for value in components[
                        offsets[3 * image + axis] : offsets[3 * image + axis + 1]
                    ]
                ),
                Fraction(0),
            )
            for axis in range(3)
        )
        n = int(action[0])
        yz = Fraction(1, 4) if n % 2 == 0 else Fraction(3, 4)
        assert exact == (Fraction(1, 4) + n, yz, yz)
    np.testing.assert_array_equal(preparation.generators, generator[None])
    np.testing.assert_array_equal(preparation.points, sites)
