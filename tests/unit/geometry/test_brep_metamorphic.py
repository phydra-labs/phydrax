"""Binary similarities preserve certified B-Rep decisions and dimensions."""

import numpy as np
from hypothesis import example, given, settings

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.brep import (
    brep_box,
    brep_edge_classification,
    brep_mass_properties,
    BRepTessellationPolicy,
    prepare_brep_query,
)
from tests._support.brep_metamorphic import (
    assert_scaled_bounds,
    binary_similarities,
    BinarySimilarity,
)
from tests._support.coordinate_references import coordinate_reference_budget


@settings(max_examples=8, deadline=None)
@given(transform=binary_similarities())
@example(transform=BinarySimilarity((0, 1, 2), (1, 1, 1), -20))
@example(transform=BinarySimilarity((2, 0, 1), (-1, 1, -1), 20))
def test_binary_similarity_preserves_brep_queries(transform: BinarySimilarity) -> None:
    # Cubes are invariant under all signed axis permutations, so corresponding
    # moment tensor entries need no representative or topology tie-breaking.
    for half_width in (0.5, 1.0):
        models = []
        for exponent in (0, transform.exponent):
            radius = np.ldexp(half_width, exponent)
            models.append(
                brep_box(
                    (-radius, -radius, -radius),
                    (radius, radius, radius),
                    coordinate_contract=SpatialCoordinateContract.si(),
                    tessellation=BRepTessellationPolicy(realize=False),
                )
            )
        source, target = models
        source_query, target_query = (prepare_brep_query(model) for model in models)
        points = half_width * np.asarray(
            ((0.0, 0.0, 0.0), (0.5, 0.25, -0.5), (2.0, 0.0, 0.0)),
            dtype=np.float64,
        )
        before = source_query.contains(points)
        after = target_query.contains(transform.apply(points))
        for left, right in (
            (before.inside, after.inside),
            (before.unresolved, after.unresolved),
            (before.resource_exhausted, after.resource_exhausted),
        ):
            np.testing.assert_array_equal(left, right)
        assert_scaled_bounds(
            np.asarray(before.distance_lower_bounds),
            np.asarray(after.distance_lower_bounds),
            transform.exponent,
        )
        assert_scaled_bounds(
            np.asarray(before.distance_upper_bounds),
            np.asarray(after.distance_upper_bounds),
            transform.exponent,
        )
        source_geometry, target_geometry = source.geometry, target.geometry
        assert source_geometry is not None and target_geometry is not None
        with coordinate_reference_budget() as budget:
            edges = brep_edge_classification(
                source, source_geometry, maximum_cells=1, coordinate_budget=budget
            )
            transformed_edges = brep_edge_classification(
                target, target_geometry, maximum_cells=1, coordinate_budget=budget
            )
            mass = brep_mass_properties(
                source, source_geometry, subdivisions=2, coordinate_budget=budget
            )
            transformed_mass = brep_mass_properties(
                target, target_geometry, subdivisions=2, coordinate_budget=budget
            )
        assert [edge.status for edge in edges] == [
            edge.status for edge in transformed_edges
        ]
        for left_edge, right_edge in zip(edges, transformed_edges, strict=True):
            assert_scaled_bounds(
                np.asarray(left_edge.length_enclosure),
                np.asarray(right_edge.length_enclosure),
                transform.exponent,
            )
        for left_bounds, right_bounds, degree in (
            (mass.face_area_enclosures, transformed_mass.face_area_enclosures, 2),
            (mass.volume_enclosures, transformed_mass.volume_enclosures, 3),
            (mass.inertia_enclosures, transformed_mass.inertia_enclosures, 5),
        ):
            assert_scaled_bounds(
                np.asarray(left_bounds),
                np.asarray(right_bounds),
                degree * transform.exponent,
            )
