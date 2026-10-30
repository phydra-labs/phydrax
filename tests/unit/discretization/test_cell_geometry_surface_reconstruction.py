from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax.discretization import CellBlock, CellGeometrySpec, CellMesh
from phydrax.discretization._cell_geometry_transition_types import CellGeometryTransition
from phydrax.discretization._coordinate_enclosure import (
    add,
    axes,
    ExactScalar,
    scale,
)
from phydrax.discretization._nested_reference_types import (
    evaluate_reference_chart_arguments,
    freeze_reference_chart_arguments,
    ReferenceCellMapWitnesses,
)
from phydrax.discretization.fem import lagrange_element


def _rational_point(point: tuple[ExactScalar, ...], /) -> tuple[Fraction, ...]:
    if not all(isinstance(value, Fraction) for value in point):
        raise TypeError("Affine chart corners of rational vertices must stay rational.")
    return tuple(value for value in point if isinstance(value, Fraction))


def _surface_reconstruction_case(
    *, explicit_charts: bool = True
) -> tuple[CellMesh, CellGeometrySpec, CellMesh, ReferenceCellMapWitnesses]:
    source_element = lagrange_element("triangle", 2)
    source_nodes = np.asarray(source_element.reference_nodes, dtype=np.float64)
    source_coordinates = np.column_stack(
        (
            source_nodes[:, 0],
            source_nodes[:, 1],
            source_nodes[:, 0] * source_nodes[:, 1],
        )
    )
    source_mesh = CellMesh(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))),
        (
            CellBlock(
                "surface",
                "triangle",
                np.asarray(((0, 1, 2),), dtype=np.int32),
                global_ids=np.asarray((101,), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((11, 12, 13), dtype=np.int64),
    )
    source_geometry = CellGeometrySpec(
        {"surface": source_element},
        {"surface": np.arange(source_element.local_dof_count, dtype=np.int32)[None, :]},
        source_coordinates,
    )

    target_uv = np.asarray(
        ((0.0, 0.0), (1.0, 0.0), (0.5, 0.5), (0.0, 1.0)), dtype=np.float64
    )
    target_coordinates = np.column_stack(
        (
            target_uv[:, 0],
            target_uv[:, 1],
            target_uv[:, 0] * target_uv[:, 1],
        )
    )
    target_topology = CellMesh(
        target_coordinates,
        (
            CellBlock(
                "surface",
                "triangle",
                np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
                global_ids=np.asarray((201, 202), dtype=np.int64),
            ),
        ),
        vertex_global_ids=np.asarray((21, 22, 23, 24), dtype=np.int64),
    )

    x, y = axes(2)
    half = Fraction(1, 2)
    chart_arguments = (
        (add(x, scale(y, half)), scale(y, half)),
        (scale(x, half), add(scale(x, half), y)),
    )
    target_reference_vertices = (
        (Fraction(0), Fraction(0)),
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
    )
    exact_corners = tuple(
        tuple(
            _rational_point(
                evaluate_reference_chart_arguments("triangle", arguments, point)
            )
            for point in target_reference_vertices
        )
        for arguments in chart_arguments
    )
    execution_corners = np.asarray(
        [[[float(value) for value in point] for point in cell] for cell in exact_corners],
        dtype=np.float64,
    )
    chart_bank = (
        freeze_reference_chart_arguments(chart_arguments) if explicit_charts else None
    )
    reference_map = ReferenceCellMapWitnesses(
        np.asarray((201, 202), dtype=np.int64),
        np.asarray((101, 101), dtype=np.int64),
        "triangle",
        "triangle",
        execution_corners,
        exact_corners,
        chart_bank,
    )
    return source_mesh, source_geometry, target_topology, reference_map


def test_nonnested_surface_reconstruction_uses_source_map_and_bounds_error() -> None:
    source_mesh, source_geometry, target_topology, reference_map = (
        _surface_reconstruction_case()
    )

    transition = CellGeometryTransition.reconstruct_surface(
        source_mesh,
        source_geometry,
        target_topology,
        reference_map=reference_map,
        maximum_displacement=0.2,
        maximum_continuity_residual=0.0,
    )

    assert transition.evidence.kind == "bounded_reconstruction"
    assert transition.evidence.reference_map_count == 2
    assert transition.evidence.continuity_residual == 0.0
    assert np.asarray(transition.target_mesh.coordinates)[2, 2] == 0.25

    # The exact source-target z residuals are
    # y(2x + y - 1)/4 and x(x + 2y - 1)/4. Their exact maximum
    # absolute value on either reference triangle is 1/16.
    samples = np.asarray(
        [
            (first / 40.0, second / 40.0)
            for first in range(41)
            for second in range(41 - first)
        ],
        dtype=np.float64,
    )
    oracle_error = 0.0
    for cell_id in (201, 202):
        x_values, y_values = samples[:, 0], samples[:, 1]
        if cell_id == 201:
            source_u = x_values + 0.5 * y_values
            source_v = 0.5 * y_values
        else:
            source_u = 0.5 * x_values
            source_v = 0.5 * x_values + y_values
        oracle = np.column_stack((source_u, source_v, source_u * source_v))
        represented = transition.target_geometry.evaluate(
            transition.target_mesh,
            np.asarray((cell_id,), dtype=np.int64),
            samples[None, :, :],
        )[0]
        oracle_error = max(
            oracle_error,
            float(np.max(np.linalg.norm(oracle - represented, axis=1))),
        )
    assert oracle_error == pytest.approx(1.0 / 16.0, abs=1e-14)
    assert transition.evidence.approximation_bound >= oracle_error
    assert transition.evidence.approximation_bound <= 1.0 / 8.0 + 1e-12

    transition.require_bound(
        source_mesh,
        source_geometry,
        transition.target_mesh,
        transition.target_geometry,
    )
    restored = CellGeometryTransition.from_stored_fields(transition)
    assert restored.transition_id == transition.transition_id
    stale_target = CellMesh(
        np.asarray(transition.target_mesh.coordinates)
        + np.asarray((0.0, 0.0, 0.01), dtype=np.float64),
        transition.target_mesh.blocks,
        vertex_global_ids=np.asarray(transition.target_mesh.vertex_global_ids),
    )
    with pytest.raises(ValueError, match="stale for these endpoint maps"):
        transition.require_bound(
            source_mesh,
            source_geometry,
            stale_target,
            transition.target_geometry,
        )


def test_nonnested_surface_reconstruction_refuses_over_tolerance() -> None:
    source_mesh, source_geometry, target_topology, reference_map = (
        _surface_reconstruction_case()
    )

    with pytest.raises(
        ValueError,
        match="Certified source-target displacement bound .* exceeds requested maximum_displacement",
    ):
        CellGeometryTransition.reconstruct_surface(
            source_mesh,
            source_geometry,
            target_topology,
            reference_map=reference_map,
            maximum_displacement=0.05,
            maximum_continuity_residual=0.0,
        )


def test_nonnested_surface_reconstruction_refuses_corner_only_chart() -> None:
    source_mesh, source_geometry, target_topology, reference_map = (
        _surface_reconstruction_case(explicit_charts=False)
    )

    with pytest.raises(
        ValueError,
        match="unresolved without explicit target-to-source reference chart arguments",
    ):
        CellGeometryTransition.reconstruct_surface(
            source_mesh,
            source_geometry,
            target_topology,
            reference_map=reference_map,
            maximum_displacement=0.2,
            maximum_continuity_residual=0.0,
        )
