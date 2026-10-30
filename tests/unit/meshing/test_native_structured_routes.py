from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax._physical import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax._trainable import NonTrainableState
from phydrax.discretization.fem._reference import FiniteElementSpec
from phydrax.discretization.finite_volume import UnstructuredFiniteVolumePlan
from phydrax.geometry._domain_coverage import PiecewiseLinearDomain
from phydrax.geometry._piecewise_linear_boundary import PiecewiseLinearBoundarySource
from phydrax.geometry._planar_embedding import PlanarEmbedding
from phydrax.geometry._source_boundary_records import (
    SourceBoundaryDistance,
    SourceBoundarySamples,
)
from phydrax.geometry.brep._patches import PlanePatch
from phydrax.meshing import NativeMeshingProvider
from phydrax.meshing._contracts import (
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingFailure,
    SurfaceMeshingSpec,
    VolumeFillStrategy,
    VolumeMeshingSpec,
)
from phydrax.meshing._controls import (
    BlockInterfaceControl,
    LayerSchedule,
    TransfiniteCurveControl,
    TransfiniteSurfaceControl,
)
from phydrax.meshing._multiblock import glue_structured_blocks
from phydrax.meshing._scope import MeshingEntityKind, MeshingScope
from phydrax.meshing._structured import (
    generate_structured_block,
    logical_cell_bank,
    TransfiniteBlock,
)
from phydrax.meshing._sweep import (
    column_connectivity,
    generate_sweep,
    SweepControl,
    SweepMapKind,
)
from phydrax.meshing.providers._native_options import NativeMeshingOptions
from phydrax.meshing.providers._native_schedules import NativeStructuredSchedule
from phydrax.meshing.providers._native_sources import (
    NativeStructuredSource,
    NativeSweepSource,
)
from phydrax.meshing.providers._native_structured import prepare_native_structured


def _curve(
    points: tuple[tuple[float, float], ...],
    /,
    *,
    source_id: str,
    intervals: int,
) -> TransfiniteCurveControl:
    degree = len(points) - 1
    knots = (0.0,) * (degree + 1) + (1.0,) * (degree + 1)
    curve = phx.geometry.BSplineCurve(
        np.asarray(points, dtype=np.float64),
        np.ones((len(points),), dtype=np.float64),
        np.asarray(knots, dtype=np.float64),
        degree,
    )
    return TransfiniteCurveControl(curve, source_id, (0.0, 1.0), intervals)


def _rectangle(
    name: str,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    /,
    *,
    x_intervals: int = 2,
    y_intervals: int = 2,
    curved_bottom: bool = False,
) -> TransfiniteBlock:
    bottom_points = (
        ((x0, y0), ((x0 + x1) / 2.0, y0 - 0.2), (x1, y0))
        if curved_bottom
        else ((x0, y0), (x1, y0))
    )
    return TransfiniteBlock(
        name,
        (
            _curve(((x0, y0), (x0, y1)), source_id=f"{name}:left", intervals=y_intervals),
            _curve(
                ((x1, y0), (x1, y1)), source_id=f"{name}:right", intervals=y_intervals
            ),
            _curve(bottom_points, source_id=f"{name}:bottom", intervals=x_intervals),
            _curve(((x0, y1), (x1, y1)), source_id=f"{name}:top", intervals=x_intervals),
        ),
    )


def _box(
    name: str,
    origin: tuple[float, float, float],
    axes: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ],
    /,
    *,
    intervals: tuple[int, int, int] = (2, 2, 2),
) -> TransfiniteBlock:
    faces = []
    for face in range(6):
        normal, side = divmod(face, 2)
        free = tuple(axis for axis in range(3) if axis != normal)
        start = np.asarray(origin, dtype=np.float64) + side * np.asarray(axes[normal])
        surface = PlanePatch(start, axes[free[0]], axes[free[1]])
        faces.append(
            TransfiniteSurfaceControl(
                surface,
                f"{name}:face:{face}",
                np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64),
                (intervals[free[0]], intervals[free[1]]),
            )
        )
    return TransfiniteBlock(name, tuple(faces))


def test_logical_cell_bank_uses_c_order_quad_and_hex_reference_vertices() -> None:
    np.testing.assert_array_equal(
        logical_cell_bank((2, 2)),
        np.asarray(((0, 2, 3, 1),), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        logical_cell_bank((2, 2, 2)),
        np.asarray(((0, 4, 6, 2, 1, 5, 7, 3),), dtype=np.int32),
    )


@pytest.mark.parametrize("shape", ((), (2,), (2, 2, 2, 2), (0, 2), (2, 1)))
def test_logical_cell_bank_rejects_invalid_logical_dimensions(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="logical_shape"):
        logical_cell_bank(shape)


def test_logical_cell_bank_rejects_int32_identity_overflow_before_allocation() -> None:
    with pytest.raises(OverflowError, match="int32"):
        logical_cell_bank((2**31 + 1, 2))


def test_opposite_curve_count_mismatch_is_rejected() -> None:
    block = _rectangle("counts", 0.0, 1.0, 0.0, 1.0)
    boundaries = list(block.boundaries)
    boundaries[1] = _curve(
        ((1.0, 0.0), (1.0, 1.0)), source_id="different-count", intervals=3
    )
    with pytest.raises(ValueError, match="opposing curve interval counts"):
        TransfiniteBlock("mismatch", tuple(boundaries))


def test_curved_transfinite_boundary_is_realized_in_the_high_order_map() -> None:
    block = _rectangle("curved", 0.0, 1.0, 0.0, 1.0, curved_bottom=True)
    construction = generate_structured_block(block, geometry_degree=2)
    assert construction.mesh.blocks[0].cell_kind == "quadrilateral"
    elements, routes, coordinates = construction.geometry.resolve(construction.mesh)
    element = elements[0]
    assert isinstance(element, FiniteElementSpec)
    assert element.degree == 2
    assert routes[0].shape == (4, element.local_dof_count)
    grid = np.asarray(coordinates).reshape((5, 5, 2))
    bottom = block.boundaries[2]
    assert isinstance(bottom, TransfiniteCurveControl)
    expected = np.asarray(
        bottom.curve.evaluate(jnp.asarray(np.linspace(0.0, 1.0, 5), dtype=jnp.float64))
    )
    np.testing.assert_allclose(grid[:, 0], expected, rtol=0.0, atol=2e-15)


def test_transfinite_source_charts_must_agree_at_shared_vertices() -> None:
    block = _rectangle("cracked", 0.0, 1.0, 0.0, 1.0)
    boundaries = list(block.boundaries)
    boundaries[1] = _curve(
        ((1.0, 0.1), (1.0, 1.0)), source_id="cracked-right", intervals=2
    )
    with pytest.raises(ValueError, match="boundaries disagree"):
        generate_structured_block(TransfiniteBlock("cracked", tuple(boundaries)))


def test_elliptic_improvement_keeps_every_sampled_boundary_node_fixed() -> None:
    block = _rectangle("elliptic", 0.0, 1.0, 0.0, 1.0, x_intervals=3, y_intervals=2)
    initial = generate_structured_block(block)
    optimized = generate_structured_block(block, optimization_steps=1)
    initial_points = np.asarray(initial.mesh.coordinates).reshape((4, 3, 2))
    optimized_points = np.asarray(optimized.mesh.coordinates).reshape((4, 3, 2))
    boundary = np.zeros((4, 3), dtype=np.bool_)
    boundary[[0, -1], :] = True
    boundary[:, [0, -1]] = True
    np.testing.assert_array_equal(initial_points[boundary], optimized_points[boundary])
    assert optimized.optimization is not None


def test_transfinite_surface_faces_construct_a_positive_structured_hex() -> None:
    block = _box(
        "box",
        (0.0, 0.0, 0.0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    construction = generate_structured_block(block, geometry_degree=2)
    assert construction.mesh.blocks[0].cell_kind == "hexahedron"
    assert construction.mesh.blocks[0].cell_count == 8
    elements, _, _ = construction.geometry.resolve(construction.mesh)
    element = elements[0]
    assert isinstance(element, FiniteElementSpec)
    values, gradients = element.tabulate(
        jnp.asarray(((0.17, 0.31, 0.43),), dtype=jnp.float64)
    )
    np.testing.assert_allclose(np.sum(values, axis=1), 1.0, atol=2e-14)
    np.testing.assert_allclose(np.sum(gradients, axis=1), 0.0, atol=2e-13)


def test_triangle_and_quad_sweeps_preserve_family_and_source_cell_ids() -> None:
    schedule = LayerSchedule((0.25, 0.75))
    control = SweepControl(SweepMapKind.EXTRUSION, schedule, "z-axis")
    quad = phx.discretization.CellMesh(
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        (
            phx.discretization.CellBlock(
                "quad", "quadrilateral", ((0, 1, 2, 3),), global_ids=(91,)
            ),
        ),
    )
    triangle = phx.discretization.CellMesh.from_triangles(
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ((0, 1, 2),), cell_global_ids=(37,)
    )
    swept_quad = generate_sweep(quad, control)
    swept_triangle = generate_sweep(triangle, control)
    assert swept_quad.mesh.blocks[0].cell_kind == "hexahedron"
    assert swept_triangle.mesh.blocks[0].cell_kind == "prism"
    np.testing.assert_array_equal(swept_quad.cell_parent_ids[0][1], (91, 91))
    np.testing.assert_array_equal(swept_triangle.cell_parent_ids[0][1], (37, 37))
    triangle_map = swept_triangle.reference_map[0]
    assert triangle_map.target_cell_kind == "prism"
    assert triangle_map.source_cell_kind == "triangle"
    np.testing.assert_array_equal(triangle_map.source_cell_ids, (37, 37))
    np.testing.assert_array_equal(
        triangle_map.source_reference_vertices[:, :3],
        triangle_map.source_reference_vertices[:, 3:],
    )
    np.testing.assert_allclose(swept_quad.stations, (0.0, 0.25, 1.0))
    np.testing.assert_allclose(
        swept_quad.measured_layer_thicknesses, ((0.25,) * 4, (0.75,) * 4)
    )


def test_closed_revolution_has_an_explicit_periodic_seam_map() -> None:
    profile = phx.discretization.CellMesh.from_triangles(
        ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
        ((0, 1, 2),),
    )
    control = SweepControl(
        SweepMapKind.REVOLUTION,
        LayerSchedule((np.pi / 2.0,) * 4),
        "full-turn",
        closed=True,
    )
    swept = generate_sweep(profile, control)
    assert swept.mesh.blocks[0].cell_kind == "prism"
    np.testing.assert_array_equal(swept.layer_vertex_ids[0], swept.layer_vertex_ids[-1])
    np.testing.assert_array_equal(
        swept.mesh.blocks[0].vertices[-1, 3:], swept.layer_vertex_ids[0]
    )


def test_column_connectivity_closes_only_the_declared_last_station() -> None:
    profile = np.asarray(((0, 2, 3, 1),), dtype=np.int32)
    columns = column_connectivity(profile, 3, 4, True)
    np.testing.assert_array_equal(columns[-1, 4:], profile[0])
    assert int(np.max(columns)) == 11


def test_two_dimensional_blocks_glue_only_declared_oriented_edges() -> None:
    left = generate_structured_block(_rectangle("left", 0.0, 0.5, 0.0, 1.0))
    right = generate_structured_block(_rectangle("right", 0.5, 1.0, 0.0, 1.0))
    interface = BlockInterfaceControl("left", 1, "right", 0, (0,), (False,))
    glued = glue_structured_blocks((left, right), (interface,))
    assert glued.mesh.coordinates.shape[0] == 15
    assert glued.maximum_gluing_residual == 0.0
    left_map = dict(glued.block_vertex_ids)["left"]
    right_map = dict(glued.block_vertex_ids)["right"]
    np.testing.assert_array_equal(left_map[[6, 7, 8]], right_map[[0, 1, 2]])
    assert dict(glued.block_cell_parent_ids)["left"].shape == (
        left.mesh.blocks[0].cell_count,
    )


def test_coincident_blocks_are_not_welded_without_an_interface() -> None:
    first = generate_structured_block(_rectangle("first", 0.0, 1.0, 0.0, 1.0))
    second = generate_structured_block(_rectangle("second", 0.0, 1.0, 0.0, 1.0))
    construction = glue_structured_blocks((first, second), ())
    assert construction.mesh.coordinates.shape[0] == 18


def test_wrong_interface_permutation_and_nonconforming_counts_are_rejected() -> None:
    left = generate_structured_block(_rectangle("left", 0.0, 0.5, 0.0, 1.0))
    right = generate_structured_block(_rectangle("right", 0.5, 1.0, 0.0, 1.0))
    wrong = BlockInterfaceControl("left", 1, "right", 0, (0,), (True,))
    with pytest.raises(ValueError, match="residual"):
        glue_structured_blocks((left, right), (wrong,))
    nonconforming_right = generate_structured_block(
        _rectangle("other", 0.5, 1.0, 0.0, 1.0, y_intervals=3)
    )
    mismatch = BlockInterfaceControl("left", 1, "other", 0, (0,), (False,))
    with pytest.raises(ValueError, match="counts differ"):
        glue_structured_blocks((left, nonconforming_right), (mismatch,))


def test_three_dimensional_face_permutation_and_flip_are_applied_explicitly() -> None:
    left = generate_structured_block(
        _box(
            "left3d", (0.0, 0.0, 0.0), ((0.5, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
    )
    right = generate_structured_block(
        _box(
            "right3d",
            (0.5, 0.0, 1.0),
            ((0.5, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        )
    )
    interface = BlockInterfaceControl("left3d", 1, "right3d", 0, (1, 0), (False, True))
    glued = glue_structured_blocks((left, right), (interface,))
    assert glued.mesh.blocks[0].cell_kind == "hexahedron"
    assert glued.maximum_gluing_residual == 0.0


def test_sweep_connectivity_carries_high_order_geometry_rows_and_refuses_wraparound() -> (
    None
):
    profile_geometry_rows = np.asarray(((0, 1, 2, 3, 4, 5),), dtype=np.int32)
    swept = column_connectivity(profile_geometry_rows, 2, 6, False)
    np.testing.assert_array_equal(swept[0], (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
    with pytest.raises(OverflowError, match="int32"):
        column_connectivity(np.asarray(((0, 2**31, 2),), dtype=np.int64), 1, 3, False)


def test_closed_frame_sweep_retains_the_declared_periodic_seam() -> None:
    profile = phx.discretization.CellMesh.from_triangles(
        ((1.0, 0.0, 0.0), (1.0, 0.0, 1.0), (2.0, 0.0, 0.0)),
        ((0, 1, 2),),
    )
    identity = np.eye(3, dtype=np.float64)
    quarter_turn = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    half_turn = np.asarray(
        ((-1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    three_quarter_turn = np.asarray(
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    frames = np.stack((identity, quarter_turn, half_turn, three_quarter_turn, identity))
    control = SweepControl(
        SweepMapKind.FRAMES,
        LayerSchedule((0.25, 0.25, 0.25, 0.25)),
        "closed-frame-cycle",
        frames=frames,
        translations=np.zeros((5, 3), dtype=np.float64),
        closed=True,
    )
    swept = generate_sweep(profile, control)
    assert swept.mesh.blocks[0].cell_kind == "prism"
    np.testing.assert_array_equal(swept.layer_vertex_ids[0], swept.layer_vertex_ids[-1])
    np.testing.assert_array_equal(
        swept.mesh.blocks[0].vertices[-1, 3:], swept.layer_vertex_ids[0]
    )


class _SquareBoundaryQuery(StrictModule, NonTrainableState):
    source_id: str = eqx.field(static=True)
    source_revision: str = eqx.field(static=True)
    ambient_dimension: int = eqx.field(static=True)

    def __init__(self) -> None:
        self.source_id = "square-source"
        self.source_revision = "square-revision"
        self.ambient_dimension = 2

    def boundary_distance(self, points: np.ndarray, /) -> SourceBoundaryDistance:
        coordinates = np.asarray(points, dtype=np.float64)
        boundaries = (
            np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float64),
            np.asarray(((1.0, 0.0), (1.0, 1.0)), dtype=np.float64),
            np.asarray(((1.0, 1.0), (0.0, 1.0)), dtype=np.float64),
            np.asarray(((0.0, 1.0), (0.0, 0.0)), dtype=np.float64),
        )
        distances = np.full((coordinates.shape[0],), np.inf, dtype=np.float64)
        for edge in boundaries:
            direction = edge[1] - edge[0]
            parameter = np.clip(
                ((coordinates - edge[0]) @ direction) / np.dot(direction, direction),
                0.0,
                1.0,
            )
            closest = edge[0] + parameter[:, None] * direction
            distances = np.minimum(
                distances, np.linalg.norm(coordinates - closest, axis=-1)
            )
        return SourceBoundaryDistance(distances, distances, "certified")

    def boundary_cover(self, maximum_radius: float, /) -> SourceBoundarySamples:
        if isinstance(maximum_radius, (bool, np.bool_)):
            raise TypeError("maximum_radius must be a positive physical length.")
        requested = float(maximum_radius)
        if not math.isfinite(requested) or requested <= 0.0:
            raise ValueError("maximum_radius must be finite and positive.")
        intervals = max(1, math.ceil(1.0 / (2.0 * requested)))
        radius = float(np.nextafter(0.5 / intervals, np.inf))
        while radius > requested:
            intervals += 1
            radius = float(np.nextafter(0.5 / intervals, np.inf))
        parameter = np.arange(intervals, dtype=np.float64) / intervals
        points = np.concatenate(
            (
                np.column_stack((parameter, np.zeros_like(parameter))),
                np.column_stack((np.ones_like(parameter), parameter)),
                np.column_stack((1.0 - parameter, np.ones_like(parameter))),
                np.column_stack((np.zeros_like(parameter), 1.0 - parameter)),
            ),
            axis=0,
        )
        return SourceBoundarySamples(
            points,
            np.full((points.shape[0],), radius, dtype=np.float64),
            np.zeros((points.shape[0],), dtype=np.float64),
            "certified",
            complete=True,
        )


class _SweepPwlDomainOwner:
    domain: PiecewiseLinearDomain
    source_id: str
    source_revision: str
    envelope_id: str
    edges: np.ndarray
    scope_entity_ids_by_mesh_row: tuple[np.ndarray, ...]
    _scope_entity_set_ids: tuple[str, ...]
    _entity_keys: tuple[tuple[tuple[str, tuple[str, ...]], ...], ...]
    _domain_id: str

    def __init__(self, domain: PiecewiseLinearDomain, /) -> None:
        self.domain = domain
        self.source_id = domain.source_id
        self.source_revision = domain.source_revision
        self.envelope_id = domain.domain_id
        self.edges = np.asarray(
            sorted(
                {
                    tuple(sorted((int(face[index]), int(face[(index + 1) % 3]))))
                    for face in np.asarray(domain.facets, dtype=np.int64)
                    for index in range(3)
                }
            ),
            dtype=np.int64,
        ).reshape((-1, 2))
        entity_rows = (
            np.arange(domain.vertices.shape[0], dtype=np.int64),
            np.arange(self.edges.shape[0], dtype=np.int64),
            np.arange(domain.facets.shape[0], dtype=np.int64),
            np.arange(len(domain.region_ids), dtype=np.int64),
        )
        self.scope_entity_ids_by_mesh_row = entity_rows
        self._scope_entity_set_ids = tuple(
            f"{domain.domain_id}:pwl:{dimension}" for dimension in range(len(entity_rows))
        )
        self._entity_keys = (
            tuple(
                (f"{domain.domain_id}:vertex:{row}", ())
                for row in range(entity_rows[0].size)
            ),
            tuple(
                (
                    f"{domain.domain_id}:edge:{int(edge[0])}:{int(edge[1])}",
                    (),
                )
                for edge in self.edges
            ),
            tuple(
                (f"{domain.domain_id}:facet:{row}", ())
                for row in range(entity_rows[2].size)
            ),
            tuple(
                (name, path)
                for name, path in zip(
                    domain.region_ids, domain.region_occurrence_paths, strict=True
                )
            ),
        )
        self._domain_id = domain.domain_id

    def require_current(self, /) -> None:
        if self.domain.domain_id != self._domain_id or (
            self.domain.source_id,
            self.domain.source_revision,
        ) != (self.source_id, self.source_revision):
            raise ValueError("The sweep PWL source domain changed after preparation.")

    def scope_entity_ids(self, dimension: int, /) -> np.ndarray:
        return self.scope_entity_ids_by_mesh_row[dimension]

    def scope_entity_set_id(self, dimension: int, /) -> str:
        return self._scope_entity_set_ids[dimension]

    def source_entity_key_for_scope(
        self, dimension: int, entity_id: int, /
    ) -> tuple[str, tuple[str, ...]]:
        identifiers = self.scope_entity_ids_by_mesh_row[dimension]
        position = int(np.searchsorted(identifiers, entity_id))
        if position >= identifiers.size or int(identifiers[position]) != entity_id:
            raise ValueError("The sweep PWL source entity is outside its retained bank.")
        return self._entity_keys[dimension][position]


def _native_square_request(
    required_families: tuple[str, ...],
    /,
    *,
    outside_source_bank: bool = False,
) -> tuple[
    NativeStructuredSource,
    SurfaceMeshingSpec,
    NativeMeshingOptions,
    SpatialCoordinateContract,
]:
    source_id = "square-source"
    revision = "square-revision"
    source = NativeStructuredSource(
        (_rectangle("square-block", 0.0, 1.0, 0.0, 1.0),),
        (),
        source_id,
        revision,
        fidelity_source=_SquareBoundaryQuery(),
        maximum_deviation=1e-12,
    )
    scope = MeshingScope.whole(source, 2)
    if outside_source_bank:
        bank_ids = np.asarray(source.scope_entity_ids(2), dtype=np.int64)
        bank_id_set = {int(identifier) for identifier in bank_ids}
        invalid_id = 0
        while invalid_id in bank_id_set:
            invalid_id += 1
        scope = MeshingScope(
            source_id,
            revision,
            MeshingEntityKind.GEOMETRY,
            2,
            source.scope_entity_set_id(2),
            np.asarray((invalid_id,), dtype=np.int64),
        )
    target = CellMeshingTarget(
        2,
        2,
        CellFamilyPolicy(required=required_families),
        geometry_order=1,
    )
    specification = SurfaceMeshingSpec(
        target,
        scope,
        size_controls=(),
        planar_embedding=PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
    )
    options = NativeMeshingOptions(
        "structured_transfinite",
        structured_schedule=NativeStructuredSchedule(),
    )
    return source, specification, options, SpatialCoordinateContract.si()


def test_native_structured_provider_returns_canonical_construction_data() -> None:
    source, specification, options, coordinate_contract = _native_square_request(
        ("quadrilateral",)
    )
    prepared = prepare_native_structured(
        source, specification, options, coordinate_contract
    )
    construction = prepared.execute()
    assert construction.mesh.blocks[0].cell_kind == "quadrilateral"
    elements, _, _ = construction.geometry.resolve(construction.mesh)
    assert isinstance(elements[0], FiniteElementSpec)
    assert elements[0].degree == 1
    assert construction.compliance.passed
    assert construction.certification.schedule.route == "surface"
    assert construction.certification.fidelity_tolerance == 1e-12
    assert any(
        label.name.startswith("structured-boundary:")
        for label in construction.organization.labels
    )


def test_native_sweep_provider_preserves_domain_and_profile_field_ancestry() -> None:
    source_id = "sweep-prism"
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float64,
    )
    facets = np.asarray(
        (
            (0, 2, 1),
            (3, 4, 5),
            (0, 1, 4),
            (0, 4, 3),
            (1, 2, 5),
            (1, 5, 4),
            (2, 0, 3),
            (2, 3, 5),
        ),
        dtype=np.int64,
    )
    domain = PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(
            np.asarray(((0, -1),), dtype=np.int64),
            (facets.shape[0], 1),
        ),
        ("solid",),
        source_id=source_id,
    )
    profile = phx.discretization.CellMesh.from_triangles(
        vertices[:3],
        np.asarray(((0, 1, 2),), dtype=np.int64),
        cell_global_ids=(37,),
    )
    control = SweepControl(
        SweepMapKind.EXTRUSION,
        LayerSchedule((1.0,)),
        "unit-prism-extrusion",
    )
    source = NativeSweepSource(
        profile,
        control,
        source_id,
        domain.source_revision,
        fidelity_source=PiecewiseLinearBoundarySource(_SweepPwlDomainOwner(domain), 2),
        maximum_deviation=0.0,
        domain=domain,
        domain_vertex_map=np.asarray(
            ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)),
            dtype=np.int64,
        ),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(
            3,
            3,
            CellFamilyPolicy(required=("prism",)),
            geometry_order=1,
        ),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.SWEEP,
        size_controls=(),
    )

    unmapped_source = NativeSweepSource(
        profile,
        control,
        source_id,
        domain.source_revision,
        fidelity_source=source.fidelity_source,
        maximum_deviation=0.0,
        domain=domain,
    )
    unmapped_specification = VolumeMeshingSpec(
        specification.target,
        MeshingScope.whole(unmapped_source, 2),
        specification.fill_strategy,
        size_controls=(),
    )
    with pytest.raises(MeshingFailure, match="explicit source-domain vertex"):
        NativeMeshingProvider(NativeMeshingOptions("sweep")).plan(
            unmapped_source,
            unmapped_specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )

    result = (
        NativeMeshingProvider(NativeMeshingOptions("sweep"))
        .plan(
            source,
            specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        .execute()
    )

    assert result.mesh.blocks[0].cell_kind == "prism"
    assert result.certification is not None and result.certification.passed
    fidelity = result.certification.fidelity
    assert fidelity is not None
    assert fidelity.mesh_to_source_upper == 0.0
    assert fidelity.source_to_mesh_upper == 0.0
    assert result.audit.passed
    assert result.compliance.passed
    assert result.certification.inputs.domain is domain
    assert result.certification.inputs.source is source.fidelity_source
    np.testing.assert_array_equal(
        result.mesh.coordinates,
        np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
                (1.0, 0.0, 1.0),
                (0.0, 1.0, 1.0),
            ),
            dtype=np.float64,
        ),
    )
    attributes = {attribute.name: attribute for attribute in result.attributes}
    np.testing.assert_array_equal(
        attributes[f"{result.mesh.blocks[0].name}:source-profile-cell-id"].values,
        (37,),
    )
    assert {zone.name for zone in result.zones} == {"solid"}

    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare()
    cell_volumes = np.asarray(finite_volume.cell_volumes, dtype=np.float64)
    np.testing.assert_allclose(cell_volumes, (0.5,), rtol=0.0, atol=1.0e-12)
    source_field = {37: 2.0}
    parent_ids = np.asarray(
        attributes[f"{result.mesh.blocks[0].name}:source-profile-cell-id"].values,
        dtype=np.int64,
    )
    transported_field = np.asarray(
        [source_field[int(parent_id)] for parent_id in parent_ids],
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        np.dot(transported_field, cell_volumes),
        1.0,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_native_graded_whole_slab_sweep_keeps_the_physical_layer_schedule() -> None:
    # Native counterpart of the CAD whole-solid layer sweep: the full slab is the
    # layer domain, with no core fill and an exact first-cell height.
    source_id = "graded-slab"
    thicknesses = (0.08, 0.10, 0.12)
    stations = np.concatenate(((0.0,), np.cumsum(thicknesses)))
    triangle = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64)
    # The PWL boundary carries every station so each wall facet is one swept face.
    vertices = np.asarray(
        [(x, y, z) for z in stations for x, y in triangle], dtype=np.float64
    )
    top = 3 * (len(thicknesses))
    facet_rows = [(0, 2, 1), (top, top + 1, top + 2)]
    for layer in range(len(thicknesses)):
        base = 3 * layer
        for first, second in ((0, 1), (1, 2), (2, 0)):
            facet_rows.append((base + first, base + second, base + second + 3))
            facet_rows.append((base + first, base + second + 3, base + first + 3))
    facets = np.asarray(facet_rows, dtype=np.int64)
    domain = PiecewiseLinearDomain(
        vertices,
        facets,
        np.tile(np.asarray(((0, -1),), dtype=np.int64), (facets.shape[0], 1)),
        ("slab",),
        source_id=source_id,
    )
    profile = phx.discretization.CellMesh.from_triangles(
        vertices[:3],
        np.asarray(((0, 1, 2),), dtype=np.int64),
        cell_global_ids=(5,),
    )
    source = NativeSweepSource(
        profile,
        SweepControl(
            SweepMapKind.EXTRUSION,
            LayerSchedule(thicknesses),
            "graded-slab-extrusion",
        ),
        source_id,
        domain.source_revision,
        fidelity_source=PiecewiseLinearBoundarySource(_SweepPwlDomainOwner(domain), 2),
        maximum_deviation=0.0,
        domain=domain,
        domain_vertex_map=np.asarray(
            [
                (profile_row, station)
                for station in range(stations.size)
                for profile_row in range(3)
            ],
            dtype=np.int64,
        ),
    )
    specification = VolumeMeshingSpec(
        CellMeshingTarget(3, 3, CellFamilyPolicy(required=("prism",))),
        MeshingScope.whole(source, 2),
        VolumeFillStrategy.SWEEP,
        size_controls=(),
    )

    result = (
        NativeMeshingProvider(NativeMeshingOptions("sweep"))
        .plan(
            source,
            specification,
            coordinate_contract=SpatialCoordinateContract.si(),
        )
        .execute()
    )

    assert {block.cell_kind for block in result.mesh.blocks} == {"prism"}
    assert sum(block.cell_count for block in result.mesh.blocks) == len(thicknesses)
    assert result.certification is not None and result.certification.passed
    assert result.audit.passed and result.compliance.passed
    stations = np.unique(np.asarray(result.mesh.coordinates, dtype=np.float64)[:, 2])
    np.testing.assert_allclose(
        stations, np.concatenate(((0.0,), np.cumsum(thicknesses))), rtol=0.0, atol=1e-15
    )
    assert stations[1] - stations[0] == pytest.approx(thicknesses[0], rel=0.0, abs=1e-15)
    finite_volume = UnstructuredFiniteVolumePlan.from_cell_mesh(result.mesh).prepare()
    cell_volumes = np.sort(np.asarray(finite_volume.cell_volumes, dtype=np.float64))
    # Independent oracle: each layer is a right prism over the unit right triangle.
    np.testing.assert_allclose(
        cell_volumes, np.sort(0.5 * np.asarray(thicknesses)), rtol=0.0, atol=1e-14
    )


def test_native_structured_provider_refuses_a_required_nonquad_target() -> None:
    source, specification, options, coordinate_contract = _native_square_request(
        ("triangle",)
    )
    with pytest.raises(MeshingFailure, match="required_cell_families"):
        prepare_native_structured(source, specification, options, coordinate_contract)


def test_native_structured_provider_refuses_scope_ids_outside_the_source_bank() -> None:
    source, specification, options, coordinate_contract = _native_square_request(
        ("quadrilateral",), outside_source_bank=True
    )
    with pytest.raises(MeshingFailure, match="source-owned entity bank"):
        prepare_native_structured(source, specification, options, coordinate_contract)
