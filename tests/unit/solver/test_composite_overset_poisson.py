from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from phydrax.geometry._overset_partition import (
    OversetPartition,
    partition_overset_triangles,
)
from phydrax.solver._composite_overset_poisson import (
    CompositeOversetPoissonResult,
    solve_composite_poisson,
)


type Part = tuple[str, np.ndarray, np.ndarray]
type PreparedPart = tuple[Part, FiniteElementDiscretization]

pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="mesh certification requires meshcore"
)


def _part(name: str, bounds: tuple[float, float, float, float]) -> PreparedPart:
    x0, y0, x1, y1 = bounds
    points = np.asarray(((x0, y0), (x1, y0), (x1, y1), (x0, y1)), dtype=np.float64)
    triangles = np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64)
    mesh = CellMesh(
        points, (CellBlock(name, "triangle", triangles),), numeric_version=name
    )
    certified = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    disc = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
        coordinate_spec=certified.geometry,
    ).prepare()
    return (name, points, triangles), disc


def _fixture(
    cavities: tuple[np.ndarray, ...] = (),
) -> tuple[OversetPartition, dict[str, FiniteElementDiscretization]]:
    patch, patch_disc = _part("patch", (0.25, 0.2, 0.75, 0.8))
    background, background_disc = _part("background", (0.0, 0.0, 1.0, 1.0))
    partition = partition_overset_triangles((patch, background), cavities)
    return partition, {"patch": patch_disc, "background": background_disc}


def _affine(points: Array) -> Array:
    return 1.0 + points[:, 0] - 2.0 * points[:, 1]


def test_coupled_affine_field_and_no_previous_state_dependency() -> None:
    partition, spaces = _fixture()

    def zero(points: Array) -> Array:
        return jnp.zeros(points.shape[0])

    result = solve_composite_poisson(
        partition, spaces, "u", _affine, solve_tolerance=1e-8, source=zero
    )
    again = solve_composite_poisson(
        partition, spaces, "u", _affine, solve_tolerance=1e-8, source=zero
    )
    assert any(edge.kind == "inter_part" for edge in result.interfaces)
    for name, disc in spaces.items():
        dofs = np.asarray(disc.dof_maps[0].cell_dofs[0])
        vertices = np.asarray(disc.mesh.blocks[0].vertices)
        coefficients = np.asarray(result.coefficients_by_name[name])
        np.testing.assert_allclose(
            coefficients[dofs],
            np.asarray(_affine(jnp.asarray(disc.mesh.coordinates)))[vertices],
            atol=1e-7,
        )
        np.testing.assert_allclose(
            coefficients, again.coefficients_by_name[name], atol=1e-10
        )
    assert result.linear_residual < 1e-8
    assert result.flux_defect < 1e-8
    assert abs(sum(result.interface_flux_by_name.values())) < 1e-12


def test_cut_cavity_unit_source_and_local_conservation() -> None:
    cavity = np.asarray(((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)))
    partition, spaces = _fixture((cavity,))
    result = solve_composite_poisson(
        partition, spaces, "u", _affine, solve_tolerance=1e-8
    )
    assert float(partition.total_area) == pytest.approx(0.96)
    assert result.source_content == pytest.approx(0.96)
    assert result.boundary_flux == pytest.approx(-0.96, abs=1e-8)
    for name in spaces:
        assert result.source_area_by_name[name] + result.physical_flux_by_name[
            name
        ] + result.interface_flux_by_name[name] == pytest.approx(0.0, abs=1e-8)


def test_invalid_data_and_tolerance_rejected() -> None:
    partition, spaces = _fixture()
    with pytest.raises(ValueError, match="tolerance"):
        solve_composite_poisson(partition, spaces, "u", _affine, solve_tolerance=0)
    with pytest.raises(ValueError, match="finite scalar"):
        solve_composite_poisson(
            partition,
            spaces,
            "u",
            lambda p: jnp.full((p.shape[0],), jnp.nan),
            solve_tolerance=1e-8,
        )


def test_source_changes_both_coupled_fields() -> None:
    partition, spaces = _fixture()
    zero = lambda points: jnp.zeros(points.shape[0])
    baseline = solve_composite_poisson(
        partition, spaces, "u", _affine, solve_tolerance=1e-8, source=zero
    )
    forced = solve_composite_poisson(
        partition, spaces, "u", _affine, solve_tolerance=1e-8
    )
    for name in spaces:
        assert (
            np.linalg.norm(
                np.asarray(forced.coefficients_by_name[name])
                - np.asarray(baseline.coefficients_by_name[name])
            )
            > 1e-6
        )


def test_fully_covered_part_has_no_artificial_unknowns() -> None:
    first, first_disc = _part("first", (0.0, 0.0, 1.0, 1.0))
    hidden, hidden_disc = _part("hidden", (0.0, 0.0, 1.0, 1.0))
    partition = partition_overset_triangles((first, hidden), ())
    result = solve_composite_poisson(
        partition,
        {"first": first_disc, "hidden": hidden_disc},
        "u",
        _affine,
        solve_tolerance=1e-8,
    )
    np.testing.assert_array_equal(result.coefficients_by_name["hidden"], 0)
    assert result.source_area_by_name["hidden"] == 0
    assert result.source_content == pytest.approx(1.0)


def test_three_part_manufactured_unit_source_and_cross_part_response() -> None:
    first, first_disc = _part("first", (0.15, 0.15, 0.65, 0.75))
    second, second_disc = _part("second", (0.4, 0.25, 0.85, 0.85))
    background, background_disc = _part("background", (0.0, 0.0, 1.0, 1.0))
    spaces = {"first": first_disc, "second": second_disc, "background": background_disc}
    partition = partition_overset_triangles((first, second, background), ())

    # -Δu=1, with a genuinely non-affine physical boundary trace.
    def exact(points: Array) -> Array:
        x, y = points[:, 0], points[:, 1]
        return 5 + x + y - (x * x + y * y) / 4

    result = solve_composite_poisson(partition, spaces, "u", exact, solve_tolerance=1e-9)

    def perturbed_source(points: Array) -> Array:
        return 1.0 + jnp.exp(
            -100 * ((points[:, 0] - 0.25) ** 2 + (points[:, 1] - 0.3) ** 2)
        )

    perturbed = solve_composite_poisson(
        partition, spaces, "u", exact, solve_tolerance=1e-9, source=perturbed_source
    )

    assert float(partition.total_area) == pytest.approx(1.0)
    assert result.source_content == pytest.approx(1.0)
    assert result.normalized_residual <= 1e-9
    assert result.flux_defect <= 1e-8
    assert bool(jnp.all(result.solve_result.successful))
    assert abs(sum(result.interface_flux_by_name.values())) <= 1e-10
    for name in spaces:
        assert result.source_area_by_name[name] > 0
        assert (
            abs(
                result.source_area_by_name[name]
                + result.physical_flux_by_name[name]
                + result.interface_flux_by_name[name]
            )
            <= 1e-8
        )
        delta = np.asarray(perturbed.coefficients_by_name[name]) - np.asarray(
            result.coefficients_by_name[name]
        )
        assert np.linalg.norm(delta) > 1e-9

    # Integrate the P1 solution QoI and a degree-two quadrature estimate of
    # its manufactured L2 error over the exact disjoint owned pieces.
    qoi = 0.0
    squared_error = 0.0
    for piece in partition.pieces:
        disc = spaces[piece.part_name]
        cell = piece.cell_index
        vertices = np.asarray(disc.mesh.blocks[0].vertices)[cell]
        coordinates = np.asarray(disc.default_runtime.coordinates)[vertices]
        dofs = np.asarray(disc.dof_maps[0].cell_dofs[0])[cell]
        coefficients = np.asarray(result.coefficients_by_name[piece.part_name])[dofs]
        inverse = np.linalg.inv(
            np.column_stack(
                (coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
            )
        )
        polygon = np.asarray(piece.vertices, dtype=float)
        for i in range(1, len(polygon) - 1):
            triangle = polygon[[0, i, i + 1]]
            a, b = triangle[1] - triangle[0], triangle[2] - triangle[0]
            area = abs(a[0] * b[1] - a[1] * b[0]) / 2
            points = (triangle.sum(axis=0) + 3 * triangle) / 6
            uv = (points - coordinates[0]) @ inverse.T
            basis = np.column_stack((1 - uv.sum(axis=1), uv))
            values = basis @ coefficients
            qoi += area * float(np.mean(values))
            squared_error += area * float(
                np.mean((values - np.asarray(exact(jnp.asarray(points)))) ** 2)
            )
    assert np.isfinite(qoi)
    assert np.isfinite(squared_error) and squared_error > 0


def _refined_part(
    name: str, bounds: tuple[float, float, float, float], subdivisions: int
) -> PreparedPart:
    x0, y0, x1, y1 = bounds
    xs = np.linspace(x0, x1, subdivisions + 1)
    ys = np.linspace(y0, y1, subdivisions + 1)
    points = np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)
    cells = []
    for j in range(subdivisions):
        for i in range(subdivisions):
            a = j * (subdivisions + 1) + i
            b, c, d = a + 1, a + subdivisions + 2, a + subdivisions + 1
            cells.extend(((a, b, c), (a, c, d)))
    triangles = np.asarray(cells, dtype=np.int64)
    mesh = CellMesh(
        points,
        (CellBlock(name, "triangle", triangles),),
        numeric_version=f"{name}-{subdivisions}",
    )
    certified = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    disc = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec("u", lagrange_element("triangle", 1)),
        coordinate_spec=certified.geometry,
    ).prepare()
    return (name, points, triangles), disc


def _manufactured_metrics(
    result: CompositeOversetPoissonResult, spaces: dict[str, FiniteElementDiscretization]
) -> tuple[float, float]:
    # Six-point degree-four triangle rule exactly integrates the square of
    # P1 minus the quadratic manufactured field, up to floating arithmetic.
    a, b = 0.816847572980459, 0.091576213509771
    c, d = 0.108103018168070, 0.445948490915965
    barycentric = np.asarray(
        ((a, b, b), (b, a, b), (b, b, a), (c, d, d), (d, c, d), (d, d, c))
    )
    weights = np.asarray((0.109951743655322,) * 3 + (0.223381589678011,) * 3)
    error = 0.0
    qoi = 0.0
    for piece in result.partition.pieces:
        disc = spaces[piece.part_name]
        vertices = np.asarray(disc.mesh.blocks[0].vertices)[piece.cell_index]
        coordinates = np.asarray(disc.default_runtime.coordinates)[vertices]
        dofs = np.asarray(disc.dof_maps[0].cell_dofs[0])[piece.cell_index]
        values = np.asarray(result.coefficients_by_name[piece.part_name])[dofs]
        inverse = np.linalg.inv(
            np.column_stack(
                (coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])
            )
        )
        polygon = np.asarray(piece.vertices, dtype=np.float64)
        for i in range(1, len(polygon) - 1):
            triangle = polygon[[0, i, i + 1]]
            e1, e2 = triangle[1] - triangle[0], triangle[2] - triangle[0]
            area = abs(e1[0] * e2[1] - e1[1] * e2[0]) / 2
            points = barycentric @ triangle
            uv = (points - coordinates[0]) @ inverse.T
            numerical = np.column_stack((1 - uv.sum(axis=1), uv)) @ values
            exact = 5 + points[:, 0] + points[:, 1] - np.sum(points * points, axis=1) / 4
            qoi += area * float(weights @ numerical)
            error += area * float(weights @ ((numerical - exact) ** 2))
    return float(np.sqrt(error)), float(qoi)


def test_three_part_manufactured_refinement_convergence(
    record_property: Callable[[str, float], None],
) -> None:
    metrics = []
    for subdivisions in (2, 4):
        parts_and_spaces = (
            _refined_part("first", (0.15, 0.15, 0.65, 0.75), subdivisions),
            _refined_part("second", (0.4, 0.25, 0.85, 0.85), subdivisions),
            _refined_part("background", (0.0, 0.0, 1.0, 1.0), subdivisions),
        )
        parts = tuple(item[0] for item in parts_and_spaces)
        spaces = {part[0]: disc for part, disc in parts_and_spaces}
        partition = partition_overset_triangles(parts, ())

        def boundary(points: Array) -> Array:
            x, y = points[:, 0], points[:, 1]
            return 5 + x + y - (x * x + y * y) / 4

        result = solve_composite_poisson(
            partition, spaces, "u", boundary, solve_tolerance=1e-9
        )
        assert float(partition.total_area) == pytest.approx(1.0)
        assert result.normalized_residual <= 1e-9
        assert result.flux_defect <= 1e-8
        assert bool(jnp.all(result.solve_result.successful))
        l2, qoi = _manufactured_metrics(result, spaces)
        assert np.isfinite(l2) and np.isfinite(qoi)
        record_property(f"l2_{subdivisions}", l2)
        record_property(f"qoi_{subdivisions}", qoi)
        metrics.append((l2, qoi))
    record_property("l2_reduction", metrics[1][0] / metrics[0][0])
    assert metrics[1][0] < metrics[0][0]
