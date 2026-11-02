from __future__ import annotations

from fractions import Fraction
from typing import final

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax._interpolation import TMesh2D
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._meshing_domain import MeshingDomain, SourcePreparation
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_planar_face,
    brep_sphere,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._model import BRepGeometry, BRepModel, BRepTessellation
from phydrax.geometry.brep._patches import (
    AbstractCurve,
    BSplineSurfacePatch,
    ExactLineCurve,
    LineCurve,
    OffsetCurve,
    RuledSurface,
)
from phydrax.geometry.brep._placement import edit_brep_occurrence_placement
from phydrax.geometry.brep._tessellation import tessellate_brep_model
from phydrax.geometry.brep._tspline import TSplineSurfacePatch


@final
class _UnqualifiedLineCurve(AbstractCurve):
    """A straight source curve with no declared continuity class."""

    @property
    def ambient_dimension(self) -> int:
        return 3

    def evaluate(self, parameters: jax.Array, /) -> jax.Array:
        return jnp.stack(
            (parameters, jnp.zeros_like(parameters), jnp.zeros_like(parameters)),
            axis=-1,
        )


def _policy(*, linear: float = 0.02, angular: float = 0.2) -> BRepTessellationPolicy:
    return BRepTessellationPolicy(
        linear_deflection=linear,
        angular_deflection=angular,
        trim_samples_per_edge=5,
    )


def _attach_tessellation(model: BRepModel, policy: BRepTessellationPolicy) -> BRepModel:
    tessellation, certificate = tessellate_brep_model(model, policy)
    return BRepModel(
        geometry=model.geometry,
        patches=model.patches,
        parameter_bounds=np.asarray(model.parameter_bounds),
        orientation=np.asarray(model.orientation),
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
        tessellation=tessellation,
        tessellation_certificate=certificate,
    )


def _spline_bump_model(contract: SpatialCoordinateContract) -> BRepModel:
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
        source_id="tessellation-spline-bump",
    )
    mesh = TMesh2D.tensor_product((Fraction(0), Fraction(1)), (Fraction(0), Fraction(1)))
    exact_controls = tuple(
        (
            *anchor.local_knots.greville_parameter,
            Fraction(1, 4)
            if all(0 < value < 1 for value in anchor.local_knots.greville_parameter)
            else Fraction(0),
        )
        for anchor in mesh.anchors
    )
    controls = np.asarray(
        [[float(value) for value in point] for point in exact_controls], dtype=np.float64
    )
    patch = TSplineSurfacePatch(
        mesh,
        controls,
        np.ones(mesh.coefficient_count, dtype=np.float64),
        exact_control_points=exact_controls,
    )
    return BRepModel(
        geometry=source.geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def _bspline_bump_model(contract: SpatialCoordinateContract) -> BRepModel:
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
        source_id="tessellation-bspline-bump",
    )
    exact_controls = tuple(
        tuple(
            (
                Fraction(u, 2),
                Fraction(v, 2),
                Fraction(1, 4) if (u, v) == (1, 1) else Fraction(0),
            )
            for v in range(3)
        )
        for u in range(3)
    )
    knots = (
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(1),
        Fraction(1),
        Fraction(1),
    )
    control_points = np.asarray(
        [[[float(value) for value in point] for point in row] for row in exact_controls],
        dtype=np.float64,
    )
    exact_weights = tuple(tuple(Fraction(1) for _ in range(3)) for _ in range(3))
    patch = BSplineSurfacePatch(
        control_points,
        np.ones((3, 3), dtype=np.float64),
        np.asarray([float(value) for value in knots], dtype=np.float64),
        np.asarray([float(value) for value in knots], dtype=np.float64),
        2,
        2,
        exact_control_points=exact_controls,
        exact_weights=exact_weights,
        exact_u_knots=knots,
        exact_v_knots=knots,
        control_point_ids=tuple(tuple(3 * u + v for v in range(3)) for u in range(3)),
    )
    return BRepModel(
        geometry=source.geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def test_local_span_reuse_requires_unchanged_brep_source() -> None:
    model = _bspline_bump_model(SpatialCoordinateContract.si())
    domain = MeshingDomain.from_brep(model)
    preparation = SourcePreparation(domain)
    original = preparation.surface_spans(model.patches[0])
    same = preparation.for_placement_domain(MeshingDomain.from_brep(model))
    assert original is not None
    assert same.surface_spans(model.patches[0]) is original

    changed = BRepModel(
        geometry=model.geometry,
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=model.orientation,
        physical_tags=("changed-tag",),
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
    )
    cold = preparation.for_placement_domain(MeshingDomain.from_brep(changed))
    assert cold.surface_spans(changed.patches[0]) is not original
    reversed_incidence = BRepModel(
        geometry=model.geometry,
        patches=model.patches,
        parameter_bounds=model.parameter_bounds,
        orientation=-model.orientation,
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
    )
    cold = preparation.for_placement_domain(MeshingDomain.from_brep(reversed_incidence))
    assert cold.surface_spans(reversed_incidence.patches[0]) is not original


def _offset_ruled_model(contract: SpatialCoordinateContract) -> BRepModel:
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=contract,
        source_id="tessellation-offset-ruled",
    )
    # The base runs along +x, so the +z direction offsets it toward -y.
    top = OffsetCurve(
        LineCurve((0.0, 1.0, 0.0), (0.5, 0.0, 0.0)),
        Fraction(1, 10),
        (0.0, 0.0, 1.0),
    )
    curves = (
        ExactLineCurve((0, 0, 0), (1, 0, 0)),
        ExactLineCurve((1, 0, 0), np.asarray((1, Fraction(9, 10), 0), dtype=object)),
        top,
        ExactLineCurve((0, 0, 0), np.asarray((0, Fraction(9, 10), 0), dtype=object)),
    )
    geometry = BRepGeometry(
        vertex_points=np.asarray(
            (
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 0.9, 0.0),
                (0.0, 0.9, 0.0),
            ),
            dtype=np.float64,
        ),
        vertex_exact_points=(
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(9, 10), Fraction(0)),
            (Fraction(0), Fraction(9, 10), Fraction(0)),
        ),
        curves=curves,
        edge_curves=(0, 1, 2, 3),
        edge_ranges=np.asarray(((0.0, 1.0), (0.0, 1.0), (0.0, 2.0), (0.0, 1.0))),
        edge_vertices=((0, 1), (1, 2), (3, 2), (0, 3)),
        pcurves=(
            LineCurve((0.0, 0.0), (1.0, 0.0)),
            LineCurve((1.0, 0.0), (0.0, 1.0)),
            LineCurve((0.0, 1.0), (0.5, 0.0)),
            LineCurve((0.0, 0.0), (0.0, 1.0)),
        ),
        coedge_edges=(0, 1, 2, 3),
        coedge_senses=(1, 1, -1, -1),
        face_loops=(((0, 1, 2, 3),),),
        shell_faces=(),
        shell_orientations=(),
        solid_shells=(),
    )
    # The ruling pairs both rails at one chart parameter u in [0, 1]; the top edge
    # runs at half that speed over its own range [0, 2].
    patch = RuledSurface(
        curves[0],
        OffsetCurve(
            LineCurve((0.0, 1.0, 0.0), (1.0, 0.0, 0.0)),
            Fraction(1, 10),
            (0.0, 0.0, 1.0),
        ),
    )
    return BRepModel(
        geometry=geometry,
        patches=(patch,),
        parameter_bounds=np.asarray(source.parameter_bounds),
        orientation=np.asarray(source.orientation),
        physical_tags=source.physical_tags,
        coordinate_contract=contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )


def _model_with_assembly(model: BRepModel, graph: BRepAssemblyGraph) -> BRepModel:
    source = model.geometry
    geometry = BRepGeometry(
        vertex_points=source.vertex_points,
        curves=source.curves,
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=graph,
    )
    return BRepModel(
        geometry=geometry,
        patches=model.patches,
        parameter_bounds=np.asarray(model.parameter_bounds),
        orientation=np.asarray(model.orientation),
        physical_tags=model.physical_tags,
        coordinate_contract=model.coordinate_contract,
        report=model.report,
        trim_domains=model.trim_domains,
    )


def test_direct_native_tessellation_smoke_and_source_identity() -> None:
    contract = SpatialCoordinateContract.si()
    source = brep_box((0.0, 0.0, 0.0), (1.0, 1.5, 2.0), coordinate_contract=contract)
    source_revision = source.source_revision
    geometry_id = source.geometry.geometry_id

    realized = _attach_tessellation(source, _policy())

    assert source.tessellation is None
    assert source.source_revision == source_revision
    assert source.geometry.geometry_id == geometry_id
    assert realized.source_revision == source_revision
    # The physical source keeps its identity; a tessellated model is a distinct
    # query representation of it (source revision plus tessellation identity).
    assert source.model_id == source_revision
    assert realized.model_id != source.model_id
    assert realized.model_id == canonical_fingerprint(
        {
            "kind": "brep-model",
            "source_revision": source_revision,
            "query_representation": realized.require_tessellation().tessellation_id,
        }
    )
    assert realized.tessellation is not None
    assert np.max(np.asarray(realized.tessellation.deviation_bounds)) <= 0.02
    assert np.max(np.asarray(realized.tessellation.normal_bounds)) <= 0.2
    realization = realized.require_tessellation()
    witness = realization.source_chart_witness
    root_cell_ids = np.asarray(witness.cell_global_ids, dtype=np.int64)
    root_bindings = witness.root_mesh_cell_bindings
    assert all(binding is not None for binding in root_bindings)
    bound_root_bindings = tuple(
        binding for binding in root_bindings if binding is not None
    )
    assert len(bound_root_bindings) == len(root_bindings)
    assert all(
        binding.root_geometry_id == witness.geometry_id for binding in bound_root_bindings
    )
    assert all(
        binding.root_topology_id == witness.topology_id for binding in bound_root_bindings
    )
    assert [
        binding.root_cell_global_id for binding in bound_root_bindings
    ] == root_cell_ids.tolist()
    witness_rows = {
        int(identifier): row
        for row, identifier in enumerate(np.asarray(witness.cell_global_ids))
    }
    for row, cell_id in enumerate(np.asarray(realization.triangle_global_ids)):
        exact_chart = witness.exact_chart(witness_rows[int(cell_id)])
        np.testing.assert_array_equal(
            np.asarray([[float(value) for value in point] for point in exact_chart]),
            np.asarray(realization.parameters[row]),
        )
    assert set(np.asarray(realized.triangle_face_ids).tolist()) == set(
        range(source.topology.num_faces)
    )


def test_model_reuses_only_the_certificate_bound_to_its_tessellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry import _source_fidelity

    contract = SpatialCoordinateContract.si()
    source = brep_box((0.0, 0.0, 0.0), (1.0, 1.5, 2.0), coordinate_contract=contract)
    tessellation, certificate = tessellate_brep_model(source, _policy())
    _, other_certificate = tessellate_brep_model(source, _policy(linear=0.01))

    def model(**realization: object) -> BRepModel:
        return BRepModel(
            geometry=source.geometry,
            patches=source.patches,
            parameter_bounds=np.asarray(source.parameter_bounds),
            orientation=np.asarray(source.orientation),
            physical_tags=source.physical_tags,
            coordinate_contract=contract,
            report=source.report,
            trim_domains=source.trim_domains,
            **realization,  # ty: ignore[invalid-argument-type]
        )

    def recomputed(*args: object, **kwargs: object) -> None:
        raise AssertionError("An in-hand bound certificate must not be recomputed.")

    with monkeypatch.context() as patch:
        patch.setattr(_source_fidelity, "certify_source_fidelity", recomputed)
        reused = model(tessellation=tessellation, tessellation_certificate=certificate)
    assert reused.tessellation is tessellation
    with pytest.raises(ValueError, match="another"):
        model(tessellation=tessellation, tessellation_certificate=other_certificate)
    with pytest.raises(ValueError, match="requires its tessellation"):
        model(tessellation_certificate=certificate)
    with pytest.raises(TypeError, match="tessellation_certificate"):
        model(tessellation=tessellation, tessellation_certificate=tessellation)


def test_trimmed_planar_profile_excludes_hole_and_meets_physical_bounds() -> None:
    contract = SpatialCoordinateContract.si()
    profile = PlanarProfile(
        ProfilePlane(),
        ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        holes=(ProfileLoop.polygon(((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6))),),
    )
    model = brep_planar_face(
        profile,
        coordinate_contract=contract,
        tessellation=_policy(linear=0.01, angular=0.1),
        source_id="trimmed-native-face",
    )
    tessellation = model.require_tessellation()
    charts = np.asarray(model.triangle_parameters)
    centroids = np.mean(charts, axis=1)
    in_hole = (
        (centroids[:, 0] > 0.4)
        & (centroids[:, 0] < 0.6)
        & (centroids[:, 1] > 0.4)
        & (centroids[:, 1] < 0.6)
    )
    assert not np.any(in_hole)
    assert np.all(np.asarray(tessellation.deviation_bounds) <= 0.01)
    uv_edges = charts[:, 1:] - charts[:, :1]
    areas = 0.5 * np.abs(
        uv_edges[:, 0, 0] * uv_edges[:, 1, 1] - uv_edges[:, 0, 1] * uv_edges[:, 1, 0]
    )
    assert np.sum(areas) == pytest.approx(0.96)
    assert np.all(np.asarray(tessellation.normal_bounds) <= 0.1)


def test_t_spline_trimmed_carrier_has_continuous_chord_and_normal_bounds() -> None:
    contract = SpatialCoordinateContract.si()
    source = _spline_bump_model(contract)
    policy = _policy(linear=0.025, angular=0.25)
    model = _attach_tessellation(source, policy)
    tessellation = model.require_tessellation()
    assert np.max(np.asarray(tessellation.deviation_bounds)) > 0.0
    assert np.max(np.asarray(tessellation.normal_bounds)) > 0.0
    for triangle_row, face in enumerate(np.asarray(tessellation.face_ids)):
        uv = np.asarray(tessellation.parameters[triangle_row])
        world = np.asarray(tessellation.vertices)[
            np.asarray(tessellation.triangles[triangle_row])
        ]
        uv_center = np.mean(uv, axis=0)
        source_center = np.asarray(
            model.patches[int(face)].evaluate(jnp.asarray(uv_center)), dtype=np.float64
        )
        target_center = np.mean(world, axis=0)
        assert np.linalg.norm(source_center - target_center) <= float(
            tessellation.deviation_bounds[triangle_row]
        )
        differential = np.asarray(
            jax.jacfwd(model.patches[int(face)].evaluate)(jnp.asarray(uv_center)),
            dtype=np.float64,
        )
        source_normal = np.cross(differential[:, 0], differential[:, 1]) * float(
            np.asarray(model.orientation)[int(face)]
        )
        target_normal = np.cross(world[1] - world[0], world[2] - world[0])
        cosine = np.dot(source_normal, target_normal) / (
            np.linalg.norm(source_normal) * np.linalg.norm(target_normal)
        )
        angle = float(np.arccos(np.clip(cosine, -1.0, 1.0)))
        assert angle <= float(tessellation.normal_bounds[triangle_row])
    assert np.all(np.asarray(tessellation.deviation_bounds) <= policy.linear_deflection)
    assert np.all(np.asarray(tessellation.normal_bounds) <= policy.angular_deflection)


def test_bspline_trimmed_carrier_has_continuous_chord_and_normal_bounds() -> None:
    contract = SpatialCoordinateContract.si()
    policy = _policy(linear=0.025, angular=0.25)
    model = _attach_tessellation(_bspline_bump_model(contract), policy)
    tessellation = model.require_tessellation()
    deviations = np.asarray(tessellation.deviation_bounds)
    normal_bounds = np.asarray(tessellation.normal_bounds)
    assert np.max(deviations) > 0.0
    assert np.max(normal_bounds) > 0.0
    assert np.all(deviations <= policy.linear_deflection)
    assert np.all(normal_bounds <= policy.angular_deflection)
    for row, face in enumerate(np.asarray(tessellation.face_ids)):
        uv_center = np.mean(np.asarray(tessellation.parameters[row]), axis=0)
        source_center = np.asarray(
            model.patches[int(face)].evaluate(jnp.asarray(uv_center)),
            dtype=np.float64,
        )
        target_center = np.mean(
            np.asarray(tessellation.vertices)[np.asarray(tessellation.triangles[row])],
            axis=0,
        )
        assert np.linalg.norm(source_center - target_center) <= deviations[row]


def test_native_surface_tessellation_is_conforming_on_shared_source_edges() -> None:
    contract = SpatialCoordinateContract.si()
    policy = _policy()
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.5, 2.0),
        coordinate_contract=contract,
        tessellation=policy,
    )
    tessellation = model.require_tessellation()
    triangles = np.asarray(tessellation.triangles, dtype=np.int64)
    face_ids = np.asarray(tessellation.face_ids, dtype=np.int64)
    triangle_occurrences = np.asarray(
        tessellation.triangle_occurrence_ids, dtype=np.int64
    )
    dimensions = np.asarray(tessellation.vertex_source_dimensions, dtype=np.int64)
    source_indices = np.asarray(tessellation.vertex_source_indices, dtype=np.int64)
    vertex_occurrences = np.asarray(tessellation.vertex_occurrence_ids, dtype=np.int64)
    vertex_parameters = np.asarray(tessellation.vertex_parameters, dtype=np.float64)
    shared_edges = 0
    for edge, endpoints in enumerate(model.geometry.edge_vertices):
        incident_faces = tuple(
            face
            for face, loops in enumerate(model.geometry.face_loops)
            if any(
                model.geometry.coedge_edges[coedge] == edge
                for loop in loops
                for coedge in loop
            )
        )
        if len(incident_faces) < 2:
            continue
        shared_edges += 1
        expected_vertices = {
            row
            for row, (dimension, source_index) in enumerate(
                zip(dimensions.tolist(), source_indices.tolist(), strict=True)
            )
            if (dimension == 1 and source_index == edge)
            or (dimension == 0 and source_index in endpoints)
        }
        for occurrence in set(triangle_occurrences.tolist()):
            face_vertex_sets = []
            for face in incident_faces:
                rows = np.flatnonzero(
                    (face_ids == face) & (triangle_occurrences == occurrence)
                )
                face_vertices = set(triangles[rows].reshape((-1,)).tolist())
                face_vertex_sets.append(face_vertices & expected_vertices)
            assert all(rows == expected_vertices for rows in face_vertex_sets)
            assert all(
                vertex_occurrences[row] == occurrence for row in face_vertex_sets[0]
            )
        edge_parameters = np.sort(
            vertex_parameters[(dimensions == 1) & (source_indices == edge), 0]
        )
        lower, upper = np.asarray(model.geometry.edge_ranges)[edge]
        endpoint_parameters = tuple(
            lower if vertex == endpoints[0] else upper for vertex in endpoints
        )
        assert (
            np.unique(np.concatenate((edge_parameters, endpoint_parameters))).size
            >= policy.trim_samples_per_edge
        )
    assert shared_edges > 0


def test_reflected_reused_solid_occurrences_remain_distinct_in_tessellation() -> None:
    contract = SpatialCoordinateContract.si()
    source = brep_box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), coordinate_contract=contract)
    reflection = BRepAffinePlacement.from_matrix(
        (
            (-1, 0, 0, 3),
            (0, 1, 0, 0),
            (0, 0, 1, 0),
            (0, 0, 0, 1),
        )
    )
    graph = BRepAssemblyGraph(
        (
            BRepAssemblyUse("definition-use", solid_index=0),
            BRepAssemblyUse("reflected-use", placement=reflection, solid_index=0),
        ),
        (),
        1,
    )
    model = _model_with_assembly(source, graph)
    realized = _attach_tessellation(model, _policy())
    tessellation = realized.require_tessellation()
    assert set(np.asarray(tessellation.triangle_occurrence_ids).tolist()) == {0, 1}
    assert set(np.asarray(tessellation.vertex_occurrence_ids).tolist()) == {0, 1}
    assert set(tessellation.source_chart_witness.occurrence_paths) == {
        ("definition-use",),
        ("reflected-use",),
    }
    assert np.unique(np.asarray(tessellation.vertex_global_ids)).size == len(
        tessellation.vertex_global_ids
    )
    assert realized.source_revision == tessellation.source_revision


def test_repeated_spline_occurrence_motion_reuses_local_spans() -> None:
    source = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    bounds = np.asarray(source.parameter_bounds[0], dtype=np.float64)
    parameters = np.asarray(
        [[(u, v) for v in bounds[:, 1]] for u in bounds[:, 0]], dtype=np.float64
    )
    control = np.asarray(source.patches[0].evaluate(jnp.asarray(parameters)))
    exact = tuple(
        tuple(
            tuple(Fraction.from_float(float(value)) for value in point) for point in row
        )
        for row in control
    )
    uknots = tuple(Fraction.from_float(float(value)) for value in bounds[:, 0])
    vknots = tuple(Fraction.from_float(float(value)) for value in bounds[:, 1])
    patch = BSplineSurfacePatch(
        control,
        np.ones((2, 2), dtype=np.float64),
        np.asarray((uknots[0], uknots[0], uknots[1], uknots[1]), dtype=np.float64),
        np.asarray((vknots[0], vknots[0], vknots[1], vknots[1]), dtype=np.float64),
        1,
        1,
        exact_control_points=exact,
        exact_weights=((Fraction(1), Fraction(1)), (Fraction(1), Fraction(1))),
        exact_u_knots=(uknots[0], uknots[0], uknots[1], uknots[1]),
        exact_v_knots=(vknots[0], vknots[0], vknots[1], vknots[1]),
        control_point_ids=((0, 1), (2, 3)),
    )
    source = BRepModel(
        geometry=source.geometry,
        patches=(patch, *source.patches[1:]),
        parameter_bounds=source.parameter_bounds,
        orientation=source.orientation,
        physical_tags=source.physical_tags,
        coordinate_contract=source.coordinate_contract,
        report=source.report,
        trim_domains=source.trim_domains,
    )
    graph = BRepAssemblyGraph(
        (
            BRepAssemblyUse("first", definition_id="part"),
            BRepAssemblyUse("second", definition_id="part"),
        ),
        (BRepAssemblyDefinition("part", (BRepAssemblyUse("leaf", solid_index=0),)),),
        1,
    )
    model = _model_with_assembly(source, graph)
    preparation = SourcePreparation(MeshingDomain.from_brep(model))
    local = preparation.surface_spans(patch)
    assert local is not None
    placement = BRepAffinePlacement.from_matrix(
        ((1, 0, 0, 2), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1))
    )
    moved = edit_brep_occurrence_placement(model, ("first", "leaf"), placement).model
    domain = MeshingDomain.from_brep(moved)
    reused = preparation.for_placement_domain(domain)
    assert reused.surface_spans(patch) is local
    cold = SourcePreparation(domain)
    for surface in (item.surface for item in domain.patches):
        warm_spans = reused.surface_spans(surface)
        cold_spans = cold.surface_spans(surface)
        assert (warm_spans is None) == (cold_spans is None)
        if warm_spans is not None and cold_spans is not None:
            assert warm_spans.source_id == cold_spans.source_id
    warm_mesh, warm_proof = tessellate_brep_model(moved, _policy(), preparation=reused)
    cold_mesh, cold_proof = tessellate_brep_model(moved, _policy(), preparation=cold)
    np.testing.assert_array_equal(warm_mesh.vertices, cold_mesh.vertices)
    np.testing.assert_array_equal(warm_mesh.triangles, cold_mesh.triangles)
    assert warm_proof.certificate_id == cold_proof.certificate_id


def test_singular_pole_trim_certifies_a_tessellation_with_bounded_evidence() -> None:
    source = brep_sphere(1.0, coordinate_contract=SpatialCoordinateContract.si())
    policy = _policy()
    tessellation, _ = tessellate_brep_model(source, policy)

    vertices = np.asarray(tessellation.vertices, dtype=np.float64)
    triangles = np.asarray(tessellation.triangles, dtype=np.int64)
    deviations = np.asarray(tessellation.deviation_bounds, dtype=np.float64)
    normals = np.asarray(tessellation.normal_bounds, dtype=np.float64)
    assert triangles.shape[0] > 0
    assert deviations.shape == normals.shape == (triangles.shape[0],)
    assert np.all(np.isfinite(deviations)) and np.all(np.isfinite(normals))
    assert np.all(deviations <= policy.linear_deflection)
    assert np.all(normals <= policy.angular_deflection)
    assert set(np.asarray(tessellation.face_ids).tolist()) == {0}
    # Every vertex lies on the unit sphere, and every facet faces outward.
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), 1.0, atol=1e-12)
    corners = vertices[triangles]
    facets = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    assert np.all(np.sum(facets * corners.mean(axis=1), axis=1) > 0.0)
    # Every source vertex, including the collapsed pole trims' vertices, is a
    # tessellation vertex at its source point.
    dimensions = np.asarray(tessellation.vertex_source_dimensions, dtype=np.int64)
    source_indices = np.asarray(tessellation.vertex_source_indices, dtype=np.int64)
    pole_rows = np.flatnonzero(dimensions == 0)
    source_vertices = np.asarray(source.geometry.vertex_points, dtype=np.float64)
    assert set(source_indices[pole_rows].tolist()) == set(range(len(source_vertices)))
    np.testing.assert_allclose(
        vertices[pole_rows], source_vertices[source_indices[pole_rows]], atol=1e-12
    )
    assert source.tessellation is None


def test_offset_curve_ruled_surface_carrier_tessellates_with_bounds() -> None:
    contract = SpatialCoordinateContract.si()
    source = _offset_ruled_model(contract)
    patch = source.patches[0]
    assert patch.is_c1_on(np.asarray(((0.0, 0.0), (1.0, 1.0))))

    model = _attach_tessellation(source, _policy(linear=0.01, angular=0.1))
    tessellation = model.require_tessellation()
    assert set(np.asarray(tessellation.face_ids).tolist()) == {0}
    assert np.all(np.asarray(tessellation.deviation_bounds) <= 0.01)
    assert np.all(np.asarray(tessellation.normal_bounds) <= 0.1)
    dimensions = np.asarray(tessellation.vertex_source_dimensions, dtype=np.int64)
    source_indices = np.asarray(tessellation.vertex_source_indices, dtype=np.int64)
    edge_parameters = np.asarray(tessellation.vertex_parameters, dtype=np.float64)[
        (dimensions == 1) & (source_indices == 2), 0
    ]
    assert edge_parameters.size >= 3
    assert np.all((edge_parameters > 0.0) & (edge_parameters < 2.0))


def test_custom_curve_without_c1_claim_has_no_tessellatable_source_model() -> None:
    contract = SpatialCoordinateContract.si()
    source = _offset_ruled_model(contract)
    assert isinstance(source.patches[0], RuledSurface)
    patch = RuledSurface(
        _UnqualifiedLineCurve(),
        source.patches[0].second,
    )
    assert not patch.is_c1_on(np.asarray(((0.0, 0.0), (1.0, 1.0))))
    # A carrier without declared structure has no exact edge-on-face proof, so
    # the source model refuses it before any tessellation could be requested.
    with pytest.raises(ValueError, match="exact source face-gap proof"):
        BRepModel(
            geometry=source.geometry,
            patches=(patch,),
            parameter_bounds=np.asarray(source.parameter_bounds),
            orientation=np.asarray(source.orientation),
            physical_tags=source.physical_tags,
            coordinate_contract=contract,
            report=source.report,
            trim_domains=source.trim_domains,
        )


def test_brep_model_rejects_unproven_bounds_and_forged_root_binding() -> None:
    contract = SpatialCoordinateContract.si()
    source = _spline_bump_model(contract)
    policy = _policy(linear=0.025, angular=0.25)
    certified = _attach_tessellation(source, policy).require_tessellation()
    forged = BRepTessellation(
        linear_deflection=certified.linear_deflection,
        angular_deflection=certified.angular_deflection,
        source_revision=certified.source_revision,
        policy_id=certified.policy_id,
        vertices=certified.vertices,
        triangles=certified.triangles,
        face_ids=certified.face_ids,
        parameters=certified.parameters,
        deviation_bounds=np.zeros_like(np.asarray(certified.deviation_bounds)),
        normal_bounds=np.zeros_like(np.asarray(certified.normal_bounds)),
        vertex_source_dimensions=certified.vertex_source_dimensions,
        vertex_source_indices=certified.vertex_source_indices,
        vertex_parameters=certified.vertex_parameters,
        triangle_occurrence_ids=certified.triangle_occurrence_ids,
        vertex_occurrence_ids=certified.vertex_occurrence_ids,
        vertex_global_ids=certified.vertex_global_ids,
        triangle_global_ids=certified.triangle_global_ids,
        source_chart_witness=certified.source_chart_witness,
    )
    with pytest.raises(ValueError, match="Stored tessellation bounds"):
        BRepModel(
            geometry=source.geometry,
            patches=source.patches,
            parameter_bounds=np.asarray(source.parameter_bounds),
            orientation=np.asarray(source.orientation),
            physical_tags=source.physical_tags,
            coordinate_contract=contract,
            report=source.report,
            trim_domains=source.trim_domains,
            tessellation=forged,
        )
    witness = certified.source_chart_witness
    from phydrax.discretization._surface_chart_records import (
        SurfaceChartWitness,
        SurfaceRootMeshCellBinding,
    )

    bindings = list(witness.root_mesh_cell_bindings)
    first_binding = bindings[0]
    assert first_binding is not None
    bindings[0] = SurfaceRootMeshCellBinding(
        first_binding.root_geometry_id,
        first_binding.root_topology_id,
        first_binding.root_cell_global_id + 1,
    )
    forged_witness = SurfaceChartWitness(
        geometry_id=witness.geometry_id,
        topology_id=witness.topology_id,
        domain_id=witness.domain_id,
        cell_global_ids=witness.cell_global_ids,
        patches=witness.patches,
        charts=witness.charts,
        source_root_cell_ids=witness.source_root_cell_ids,
        source_piece_ids=witness.source_piece_ids,
        geometry_entity_ids=witness.geometry_entity_ids,
        occurrence_paths=witness.occurrence_paths,
        chart_rational_tokens=witness.chart_rational_tokens,
        root_mesh_cell_bindings=bindings,
    )
    forged_root = BRepTessellation(
        linear_deflection=certified.linear_deflection,
        angular_deflection=certified.angular_deflection,
        source_revision=certified.source_revision,
        policy_id=certified.policy_id,
        vertices=certified.vertices,
        triangles=certified.triangles,
        face_ids=certified.face_ids,
        parameters=certified.parameters,
        deviation_bounds=certified.deviation_bounds,
        normal_bounds=certified.normal_bounds,
        vertex_source_dimensions=certified.vertex_source_dimensions,
        vertex_source_indices=certified.vertex_source_indices,
        vertex_parameters=certified.vertex_parameters,
        triangle_occurrence_ids=certified.triangle_occurrence_ids,
        vertex_occurrence_ids=certified.vertex_occurrence_ids,
        vertex_global_ids=certified.vertex_global_ids,
        triangle_global_ids=certified.triangle_global_ids,
        source_chart_witness=forged_witness,
    )
    with pytest.raises(ValueError, match="root-cell binding"):
        BRepModel(
            geometry=source.geometry,
            patches=source.patches,
            parameter_bounds=np.asarray(source.parameter_bounds),
            orientation=np.asarray(source.orientation),
            physical_tags=source.physical_tags,
            coordinate_contract=contract,
            report=source.report,
            trim_domains=source.trim_domains,
            tessellation=forged_root,
        )
