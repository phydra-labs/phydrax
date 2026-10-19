from __future__ import annotations

import logging

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax import config as pxconfig
from phydrax._resource_ticket import HostBatchEstimateError, HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization.fem._reference_rules import (
    degree_aware_reference_rule,
    host_degree_aware_reference_rule,
)
from phydrax.meshing._quality import (
    _masked_quality_workset,
    _standard_block_quality,
    _vertex_metric,
    _zero_face_warpage,
    BlockQuality,
    measure_rule,
)
from tests._support.cell_meshes import unit_square
from tests._support.quality_fixtures import (
    curved_quality_tetrahedron,
    reference_quality_mesh,
    sheared_quality_polyhedron,
)


# Retains the cold-trace parity bound from test_audit_quality (about 45 binary64 ulps).
_TRACE_PARITY_RTOL = 1.0e-14
# A conservative 256-epsilon envelope for quadrature, determinants, and AD reductions.
_REDUCTION_ATOL = 256.0 * np.finfo(np.float64).eps


def test_absent_metric_and_constant_warpage_do_not_compile(
    caplog: pytest.LogCaptureFixture,
) -> None:
    mesh = unit_square()
    corners = jax.device_put(
        np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]
    )
    caplog.clear()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING):
        assert _vertex_metric(mesh, None) is None
        warpage = _zero_face_warpage(corners)

    assert not any(
        record.getMessage().startswith("Finished XLA compilation")
        for record in caplog.records
    )
    np.testing.assert_array_equal(warpage, np.zeros((2,), dtype=np.float64))
    caplog.clear()
    with jax.log_compiles(True), caplog.at_level(logging.WARNING):
        quality = phx.meshing.evaluate_cell_quality(mesh)

    np.testing.assert_array_equal(quality.measures, (0.5, 0.5))
    assert not any(
        record.getMessage().startswith(
            ("Compiling jit(_cell_metric)", "Compiling jit(_vertex_metric)")
        )
        for record in caplog.records
    )


def _tetrahedral_workset(points: Array, metric: Array | None) -> BlockQuality:
    return _standard_block_quality("tetrahedron", points, metric)


@jax.jit
def _masked_tetrahedral_workset(
    points: Array, metric: Array, valid: Array
) -> BlockQuality:
    return _masked_quality_workset(_tetrahedral_workset, points, metric, valid)


@pytest.mark.parametrize("inactive", ("large-finite", "nan"))
def test_inactive_quality_lanes_cannot_change_results_or_coordinate_metric_gradients(
    inactive: str,
) -> None:
    reference = np.asarray(
        ((0.0, 0.0, 0.0), (1.1, 0.03, 0.07), (0.12, 0.91, -0.02), (0.08, 0.09, 0.83)),
        dtype=np.float64,
    )
    points = np.broadcast_to(reference, (8, 4, 3)).copy()
    points[1] += 2.0
    points[2] += 4.0
    tensors = np.broadcast_to(np.eye(3, dtype=np.float64), (8, 3, 3)).copy()
    valid = jnp.asarray(
        (True, True, True, False, False, False, False, False), dtype=jnp.bool_
    )
    changed_points, changed_tensors = points.copy(), tensors.copy()
    fill = 1.0e100 if inactive == "large-finite" else np.nan
    changed_points[3:] = fill
    changed_tensors[3:] = fill
    coordinates, metric = jnp.asarray(points), jnp.asarray(tensors)
    altered_coordinates, altered_metric = (
        jnp.asarray(changed_points),
        jnp.asarray(changed_tensors),
    )
    expected = _masked_tetrahedral_workset(coordinates, metric, valid)
    observed = _masked_tetrahedral_workset(altered_coordinates, altered_metric, valid)
    for actual, baseline in zip(observed, expected, strict=True):
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint8), np.asarray(baseline).view(np.uint8)
        )

    def objective(values: Array, tensor: Array) -> Array:
        quality = _masked_tetrahedral_workset(values, tensor, valid)
        return jnp.sum(quality.measure + quality.mean_ratio + quality.metric_quality)

    reference_value, reference_gradients = jax.value_and_grad(objective, argnums=(0, 1))(
        coordinates, metric
    )
    changed_value, changed_gradients = jax.value_and_grad(objective, argnums=(0, 1))(
        altered_coordinates, altered_metric
    )
    np.testing.assert_array_equal(
        np.asarray(changed_value).view(np.uint64),
        np.asarray(reference_value).view(np.uint64),
    )
    for actual, baseline in zip(changed_gradients, reference_gradients, strict=True):
        assert np.all(np.isfinite(np.asarray(actual)))
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint8), np.asarray(baseline).view(np.uint8)
        )
        np.testing.assert_array_equal(
            np.asarray(actual)[3:], np.zeros_like(np.asarray(actual)[3:])
        )


@pytest.mark.parametrize("cell_kind", ("prism", "pyramid", "hexahedron"))
def test_cold_traced_quality_does_not_cache_tracers(cell_kind: str) -> None:
    jax.clear_caches()
    measure_rule.cache_clear()
    mesh = reference_quality_mesh(cell_kind)
    compiled = jax.jit(
        lambda coordinates: phx.meshing.evaluate_cell_quality(mesh, coordinates)
    )(mesh.coordinates)
    eager = phx.meshing.evaluate_cell_quality(mesh)
    np.testing.assert_allclose(compiled.measures, eager.measures, rtol=_TRACE_PARITY_RTOL)
    assert float(compiled.measures[0]) > 0.0


@pytest.mark.parametrize(
    "cell_kind,measure",
    (
        ("interval", 1.0),
        ("triangle", 0.5),
        ("quadrilateral", 1.0),
        ("tetrahedron", 1.0 / 6.0),
        ("prism", 0.5),
        ("pyramid", 1.0 / 3.0),
        ("hexahedron", 1.0),
    ),
)
def test_reference_cell_quality_retains_measure_and_sampled_status(
    cell_kind: str, measure: float
) -> None:
    mesh = reference_quality_mesh(cell_kind)
    quality = phx.meshing.evaluate_cell_quality(mesh)
    np.testing.assert_allclose(quality.measures, [measure], rtol=_TRACE_PARITY_RTOL)
    assert bool(quality.sampled_valid[0])
    assert float(quality.mean_ratios[0]) > 0.0
    assert float(quality.aspect_ratios[0]) >= 1.0


@pytest.mark.parametrize(
    "count",
    (1, 3, 65, 257),
    ids=("one-lane", "remainder", "second-bucket", "chunk-and-remainder"),
)
def test_quality_worksets_preserve_cell_order_and_adjoint_measure(count: int) -> None:
    mesh = reference_quality_mesh("triangle", count)

    def measures(points: Array) -> Array:
        return phx.meshing.evaluate_cell_quality(mesh, points).measures

    result, tangent = jax.jvp(measures, (mesh.coordinates,), (mesh.coordinates,))
    _, pullback = jax.vjp(measures, mesh.coordinates)
    cotangent = jnp.arange(1, count + 1, dtype=jnp.float64)
    (gradient,) = pullback(cotangent)
    np.testing.assert_allclose(
        result, np.full(count, 0.5, dtype=np.float64), rtol=_TRACE_PARITY_RTOL
    )
    np.testing.assert_allclose(tangent, 2.0 * result, rtol=_TRACE_PARITY_RTOL)
    np.testing.assert_allclose(
        jnp.sum(gradient * mesh.coordinates),
        jnp.sum(cotangent * tangent),
        rtol=_TRACE_PARITY_RTOL,
        atol=_REDUCTION_ATOL,
    )
    quality = phx.meshing.evaluate_cell_quality(mesh)
    np.testing.assert_array_equal(quality.cell_global_ids, mesh.blocks[0].global_ids)
    assert quality.measures.shape == (count,)
    assert quality.block_offsets == (0, count)
    assert np.all(np.asarray(quality.sampled_valid))


@pytest.mark.parametrize(
    "cell_kind,vertices",
    (
        ("triangle", ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        ("quadrilateral", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ("polygon", ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 1.0))),
        ("triangle", ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.25))),
        (
            "quadrilateral",
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.25), (0.0, 1.0, 0.0)),
        ),
    ),
    ids=(
        "planar-triangle",
        "planar-quad",
        "planar-polygon",
        "embedded-triangle",
        "warped-quad",
    ),
)
def test_cyclic_quality_and_measure_gradients_are_bit_invariant_under_exact_translation(
    cell_kind: str,
    vertices: tuple[tuple[float, ...], ...],
) -> None:
    points = np.asarray(vertices, dtype=np.float64)
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells",
                cell_kind,
                np.arange(points.shape[0], dtype=np.int32)[None],
            ),
        ),
    )
    # Dyadic vertices and integer powers-of-two translations are represented
    # exactly; any difference comes from execution rather than rounded inputs.
    offset = np.asarray((2**20, -(2**21), 2**18), dtype=np.float64)[: points.shape[1]]
    translated = jnp.asarray(points + offset, dtype=jnp.float64)
    reference = phx.meshing.evaluate_cell_quality(mesh, mesh.coordinates)
    shifted = phx.meshing.evaluate_cell_quality(mesh, translated)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(eqx.filter(shifted, eqx.is_array)),
        jax.tree_util.tree_leaves(eqx.filter(reference, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.asarray(actual).view(np.uint8),
            np.asarray(expected).view(np.uint8),
        )

    def total_measure(coordinates: Array) -> Array:
        return jnp.sum(phx.meshing.evaluate_cell_quality(mesh, coordinates).measures)

    reference_gradient = jax.grad(total_measure)(mesh.coordinates)
    shifted_gradient = jax.grad(total_measure)(translated)
    np.testing.assert_array_equal(
        np.asarray(shifted_gradient).view(np.uint8),
        np.asarray(reference_gradient).view(np.uint8),
    )


@pytest.mark.parametrize("family", ("standard", "mapped", "polyhedral"))
def test_quality_compiled_and_uncompiled_values_status_and_semantic_identity(
    family: str,
) -> None:
    if family == "mapped":
        mesh, geometry = curved_quality_tetrahedron()
    elif family == "polyhedral":
        mesh, geometry = sheared_quality_polyhedron(), None
    else:
        mesh, geometry = reference_quality_mesh("tetrahedron"), None
    compiled = phx.meshing.evaluate_cell_quality(mesh, geometry=geometry)
    with pxconfig.context(pxconfig.NumericsConfig(disable_jit=True)):
        reference = phx.meshing.evaluate_cell_quality(mesh, geometry=geometry)
    compiled_arrays = eqx.filter(compiled, eqx.is_array)
    reference_arrays = eqx.filter(reference, eqx.is_array)
    for actual, expected in zip(
        jax.tree_util.tree_leaves(compiled_arrays),
        jax.tree_util.tree_leaves(reference_arrays),
        strict=True,
    ):
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=_TRACE_PARITY_RTOL,
            atol=_REDUCTION_ATOL,
            equal_nan=True,
        )
    np.testing.assert_array_equal(compiled.sampled_valid, reference.sampled_valid)
    assert compiled.evaluation_id == reference.evaluation_id
    assert compiled.topology_id == reference.topology_id
    assert compiled.geometry_layout_id == reference.geometry_layout_id
    assert compiled.block_names == reference.block_names
    assert compiled.block_offsets == reference.block_offsets
    report = phx.meshing.summarize_cell_quality(compiled, mesh=mesh)
    raw_report = phx.meshing.summarize_cell_quality(reference, mesh=mesh)
    assert (
        report.sampled_invalid_cell_global_ids
        == raw_report.sampled_invalid_cell_global_ids
    )
    assert report.worst_cell_global_ids == raw_report.worst_cell_global_ids
    # Report identity hashes numerical evidence, not just evaluation identity.
    # Compiled transcendental/fused arithmetic can round differently; every
    # differing report must correspond to differing hashed binary64 evidence.
    compiled_report_fields = (
        compiled.measures,
        compiled.mean_ratios,
        compiled.aspect_ratios,
        compiled.scaled_jacobian,
        compiled.condition_number,
        compiled.minimum_angle,
        compiled.maximum_angle,
        compiled.warpage,
        compiled.metric_quality,
        compiled.sampled_valid,
    )
    raw_report_fields = (
        reference.measures,
        reference.mean_ratios,
        reference.aspect_ratios,
        reference.scaled_jacobian,
        reference.condition_number,
        reference.minimum_angle,
        reference.maximum_angle,
        reference.warpage,
        reference.metric_quality,
        reference.sampled_valid,
    )
    same_bits = all(
        np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
        for actual, expected in zip(
            compiled_report_fields, raw_report_fields, strict=True
        )
    )
    assert (report.report_id == raw_report.report_id) == same_bits
    replay = phx.meshing.summarize_cell_quality(
        phx.meshing.evaluate_cell_quality(mesh, geometry=geometry), mesh=mesh
    )
    assert replay.report_id == report.report_id
    with pxconfig.context(pxconfig.NumericsConfig(disable_jit=True)):
        raw_replay = phx.meshing.evaluate_cell_quality(mesh, geometry=geometry)
    assert (
        phx.meshing.summarize_cell_quality(raw_replay, mesh=mesh).report_id
        == raw_report.report_id
    )


def test_finite_volume_quality_angle_has_the_analytic_shear_derivative() -> None:
    mesh = phx.discretization.CellMesh(
        np.asarray(
            ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (0.0, 1.0), (1.0, 1.0), (2.0, 1.0)),
            dtype=np.float64,
        ),
        (
            phx.discretization.CellBlock(
                "cells",
                "quadrilateral",
                np.asarray(((0, 1, 4, 3), (1, 2, 5, 4)), dtype=np.int32),
                global_ids=np.asarray((7, 9), dtype=np.int64),
            ),
        ),
    )
    direction = jnp.asarray(
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (1.0, 0.0), (1.0, 0.0), (1.0, 0.0)),
        dtype=jnp.float64,
    )

    def angle(shear: Array) -> Array:
        runtime = eqx.tree_at(
            lambda value: value.coordinates, mesh, mesh.coordinates + shear * direction
        )
        return phx.meshing.evaluate_finite_volume_quality(runtime).non_orthogonality[0]

    value, tangent = jax.jvp(
        angle,
        (jnp.asarray(0.5, dtype=jnp.float64),),
        (jnp.asarray(1.0, dtype=jnp.float64),),
    )
    np.testing.assert_allclose(value, np.arctan(0.5), rtol=_TRACE_PARITY_RTOL)
    np.testing.assert_allclose(tangent, 1.0 / (1.0 + 0.5**2), rtol=_TRACE_PARITY_RTOL)


@pytest.mark.parametrize(
    "allowance,completed,requested",
    ((26, 0, 27), (27, 27, 36), (63, 63, 4), (67, 67, 4)),
    ids=(
        "quadrature-refusal",
        "boundary-refusal",
        "first-face-refusal",
        "second-face-refusal",
    ),
)
def test_mapped_quality_refusal_preserves_query_phase_charge_order(
    allowance: int, completed: int, requested: int
) -> None:
    mesh, geometry = curved_quality_tetrahedron()
    ticket = HostBatchTicket(
        1_000_000, queries_upper=allowance, bytes_upper=64 * 1024 * 1024
    )
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1_000_000, maximum_memory_bytes=64 * 1024 * 1024
    )
    # P2 tetrahedron: 3^3 volume probes, 12 directed edges times 3 probes,
    # then four faces with one centroid and three edge midpoints each.
    with ticket, budget.activate():
        with pytest.raises(HostBatchEstimateError) as caught:
            phx.meshing.evaluate_cell_quality(mesh, geometry=geometry, budget=budget)
        assert ticket.geometry_queries == completed
        assert caught.value.batch_requested_upper[1] == requested
        assert caught.value.batch_refusal_evidence[5] == completed


def test_mapped_quality_success_charges_the_same_prepared_query_plan() -> None:
    mesh, geometry = curved_quality_tetrahedron()
    ticket = HostBatchTicket(1_000_000, queries_upper=79, bytes_upper=64 * 1024 * 1024)
    budget = CoordinateEnclosureBudget(
        ticket, maximum_work_units=1_000_000, maximum_memory_bytes=64 * 1024 * 1024
    )
    with ticket, budget.activate():
        quality = phx.meshing.evaluate_cell_quality(
            mesh, geometry=geometry, budget=budget
        )
        assert ticket.geometry_queries == 79
    np.testing.assert_allclose(quality.measures, [1.0 / 60.0], rtol=_TRACE_PARITY_RTOL)
    assert bool(quality.sampled_valid[0])


@pytest.mark.parametrize(
    "cell_kind",
    (
        "interval",
        "triangle",
        "quadrilateral",
        "tetrahedron",
        "prism",
        "pyramid",
        "hexahedron",
        "simplex:3",
        "tensor:3",
    ),
)
@pytest.mark.strict_jax
def test_host_and_device_reference_quadrature_share_binary64_node_order(
    cell_kind: str,
) -> None:
    host_points, host_weights = host_degree_aware_reference_rule(cell_kind, 2)
    points, weights = degree_aware_reference_rule(cell_kind, 2)
    assert host_points.dtype == np.dtype(np.float64)
    assert host_weights.dtype == np.dtype(np.float64)
    np.testing.assert_array_equal(
        np.asarray(points).view(np.uint64), host_points.view(np.uint64)
    )
    np.testing.assert_array_equal(
        np.asarray(weights).view(np.uint64), host_weights.view(np.uint64)
    )
    assert np.all(host_weights > 0.0)


def test_quality_checked_boundary_refuses_wrong_mesh_kind() -> None:
    with pytest.raises(TypeError, match="CellMesh"):
        phx.meshing.evaluate_cell_quality("not-a-mesh")  # ty: ignore[invalid-argument-type]


def test_finite_volume_checked_boundary_refuses_wrong_mesh_kind() -> None:
    with pytest.raises(TypeError, match="CellMesh"):
        phx.meshing.evaluate_finite_volume_quality("not-a-mesh")  # ty: ignore[invalid-argument-type]


def test_positive_hexahedral_samples_do_not_certify_the_whole_map() -> None:
    points = np.asarray(
        (
            (-0.62, 0.9, 0.43),
            (0.83, -0.37, -0.44),
            (1.06, 0.71, -0.34),
            (0.37, 1.16, -0.18),
            (0.33, 0.62, 0.51),
            (0.73, 0.42, 1.32),
            (1.1, 1.52, 0.51),
            (-0.67, 0.61, 1.06),
        )
    )
    mesh = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("cells", "hexahedron", np.arange(8)[None, :]),),
    )
    quality = phx.meshing.evaluate_cell_quality(mesh)
    certificate = phx.discretization.certify_cell_geometry_validity(mesh)

    assert bool(quality.sampled_valid[0])
    assert float(quality.scaled_jacobian[0]) > 0.0
    assert int(certificate.status[0]) == phx.discretization.CellValidityStatus.INVALID
    assert float(certificate.determinant_lower[0]) < 0.0
    assert int(certificate.depth[0]) > 0


@pytest.mark.parametrize(
    "reflex", range(4), ids=("corner-0", "corner-1", "corner-2", "corner-3")
)
def test_reflex_quadrilateral_quality_retains_positive_area_and_invalid_frames(
    reflex: int,
) -> None:
    points = np.asarray(
        phx.discretization.reference_cell_topology("quadrilateral").vertices
    )
    points[reflex] = 0.75 * points[(reflex + 2) % 4] + 0.25 * points[reflex]
    mesh = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("cells", "quadrilateral", np.arange(4)[None, :]),),
    )
    quality = phx.meshing.evaluate_cell_quality(mesh)
    certificate = phx.discretization.certify_cell_geometry_validity(mesh)
    assert float(quality.measures[0]) > 0.0
    assert not bool(quality.sampled_valid[0])
    assert float(quality.scaled_jacobian[0]) < 0.0
    assert float(quality.maximum_angle[0]) > np.pi
    assert int(certificate.status[0]) == phx.discretization.CellValidityStatus.INVALID


def test_tetrahedral_quality_distinguishes_slivers_from_regular_shapes() -> None:
    height = 1.0e-3
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, height), (1.0, 0.0, height))
    )
    mesh = phx.discretization.CellMesh(
        points,
        (phx.discretization.CellBlock("cells", "tetrahedron", np.arange(4)[None, :]),),
    )
    sliver = phx.meshing.evaluate_cell_quality(mesh)
    regular_points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.5, 0.5 * np.sqrt(3.0), 0.0),
            (0.5, np.sqrt(3.0) / 6.0, np.sqrt(2.0 / 3.0)),
        )
    )
    regular = phx.meshing.evaluate_cell_quality(
        phx.discretization.CellMesh(
            regular_points,
            (
                phx.discretization.CellBlock(
                    "cells", "tetrahedron", np.arange(4)[None, :]
                ),
            ),
        )
    )

    assert bool(sliver.sampled_valid[0])
    assert float(sliver.aspect_ratios[0]) < 1.5
    assert float(sliver.minimum_angle[0]) < 0.01
    assert float(sliver.maximum_angle[0]) > np.pi - 0.01
    assert float(sliver.radius_ratios[0]) < 0.01
    assert float(sliver.sliver_measures[0]) < 0.01
    assert float(regular.minimum_angle[0]) == pytest.approx(np.arccos(1.0 / 3.0))
    assert float(regular.maximum_angle[0]) == pytest.approx(np.arccos(1.0 / 3.0))
    assert float(regular.radius_ratios[0]) == pytest.approx(1.0)
    assert float(regular.mean_ratios[0]) == pytest.approx(1.0)
