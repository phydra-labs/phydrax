from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep import brep_box
from phydrax.geometry.brep._patches import CircleCurve
from phydrax.geometry.brep._source_shell_admissibility import (
    _convex_source_face,
    _source_region_contacts,
    _source_region_shells,
    _source_region_sides,
    _trim_enclosure,
    certify_source_regions,
)


if TYPE_CHECKING:
    from phydrax.geometry.brep._model import BRepModel
    from phydrax.geometry.brep._source_shell_admissibility import (
        SourceRegionAdmissibility,
    )


def test_polygon_trim_bounds_use_each_native_face_chart() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (2.0, 3.0, 4.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    bounds_seen = set()
    for face, trim in enumerate(model.trim_domains):
        assert trim is not None and trim.outer is not None and not trim.curve_loops
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        points = np.asarray(trim.outer, dtype=np.float64)
        lower = tuple(Fraction.from_float(float(value)) for value in bounds[0])
        upper = tuple(Fraction.from_float(float(value)) for value in bounds[1])
        expected = tuple(
            tuple(
                lower[axis]
                + Fraction.from_float(float(reduce(points[:, axis])))
                * (upper[axis] - lower[axis])
                for axis in range(2)
            )
            for reduce in (np.min, np.max)
        )
        assert _trim_enclosure(model, face) == expected
        bounds_seen.add((lower, upper))
    assert len(bounds_seen) > 1


def test_iges_quadric_source_face_pair_replay(tmp_path: Path) -> None:
    from phydrax.geometry.brep._source_shell_admissibility import (
        certify_retained_branch_pair,
        certify_same_chart_face_separation,
    )
    from tools.meshing_qualification.scenarios.native_cad_profiles import _quadric_case

    model, _, _, _ = _quadric_case("iges", tmp_path)
    separation = certify_same_chart_face_separation(model, 0, 1)
    assert separation is not None
    assert separation.replay(model)
    for first, second in ((0, 2), (1, 2)):
        proof = certify_retained_branch_pair(model, first, second)
        assert proof is not None
        assert proof.shared_edges
        assert proof.replay(model)


def test_original_step_source_pair_certificates_and_stale_refusal(tmp_path: Path) -> None:
    import equinox as eqx

    from phydrax.geometry.brep._source_shell_admissibility import (
        _opposed_source_uses,
        _shared_two_arc_circle,
        certify_coaxial_radial_separation,
        certify_plane_torus_pair,
        certify_source_ring_face_pair,
        certify_tangent_cylinder_torus_pair,
    )
    from tools.meshing_qualification.scenarios.native_cad_profiles import _external_case

    model, _, _, _ = _external_case("external-routine", "step", tmp_path)
    for certify, pairs in (
        (certify_source_ring_face_pair, ((0, 1), (1, 2))),
        (certify_tangent_cylinder_torus_pair, ((0, 5), (2, 4))),
        (certify_plane_torus_pair, ((3, 4), (3, 5))),
    ):
        for first, second in pairs:
            proof = certify(model, first, second)
            assert proof is not None
            assert proof.replay(model)
            assert certify(model, second, first) is None
            assert certify(model, first, first) is None
    assert certify_tangent_cylinder_torus_pair(model, 0, 4) is None
    assert certify_plane_torus_pair(model, 1, 4) is None
    radial = certify_coaxial_radial_separation(model, 0, 4)
    assert radial is not None and radial.replay(model)
    assert certify_coaxial_radial_separation(model, 0, 5) is None
    assert _opposed_source_uses(model, (0, 1), (0, 6))
    assert not _opposed_source_uses(model, (0, 1), (0, 5))
    assert _shared_two_arc_circle(model, 0, 5) is not None
    shortened_ranges = model.geometry.edge_ranges.at[1, 1].set(1.0)
    shortened = eqx.tree_at(
        lambda value: value.geometry.edge_ranges, model, shortened_ranges
    )
    assert _shared_two_arc_circle(shortened, 0, 5) is None

    # A forged numeric circle shadow is not a current source edge, even if its
    # retained root or a mesh sample still resembles the old boundary.
    circle = model.geometry.curves[0]
    assert isinstance(circle, CircleCurve)
    stale = eqx.tree_at(
        lambda value: value.geometry.curves[0].radius,
        model,
        circle.radius * 1.01,
    )
    assert certify_source_ring_face_pair(stale, 0, 1) is None


def test_boolean_cut_washer_source_pair_certificates(tmp_path: Path) -> None:
    import equinox as eqx

    from phydrax.geometry.brep._intersection import ConicParameterEndpoint
    from phydrax.geometry.brep._source_shell_admissibility import (
        _shared_two_arc_circle,
        certify_plane_torus_pair,
        certify_retained_branch_pair,
        certify_tangent_cylinder_torus_pair,
    )
    from tools.meshing_qualification.scenarios.native_cad_profiles import _external_case

    # The radial-hole difference rebinds every washer circle endpoint to its
    # published carrier with the retained raw parameter; the tangent and planar
    # torus circles must still certify through their exact conic roots.
    model, _, _, _ = _external_case("external-combined", "step", tmp_path)
    for certify, pairs in (
        (certify_tangent_cylinder_torus_pair, ((0, 5), (2, 4))),
        (certify_plane_torus_pair, ((3, 4), (3, 5))),
    ):
        for first, second in pairs:
            proof = certify(model, first, second)
            assert proof is not None
            assert proof.replay(model)
    # Each wall face meets both hole sides; the branch on the other side lies
    # outside the hole face's full-turn trim and is excluded exactly.
    for first, second in ((0, 6), (0, 7), (2, 6), (2, 7)):
        common = set(model.geometry.topology.face_edges[first]) & set(
            model.geometry.topology.face_edges[second]
        )
        proof = certify_retained_branch_pair(model, first, second)
        assert proof is not None
        assert set(proof.shared_edges) == common
        assert len(proof.excluded_branches) == 1
        assert proof.replay(model)

    edge = 3
    first_root, second_root = model.geometry.edge_endpoint_roots[edge]
    assert isinstance(first_root, ConicParameterEndpoint)
    assert first_root.parameter_exact is not None
    moved = ConicParameterEndpoint(
        first_root.root,
        first_root.carrier,
        Fraction(1, 1000),
        first_root.point_exact,
    )
    roots = list(model.geometry.edge_endpoint_roots)
    roots[edge] = (moved, second_root)
    outside = eqx.tree_at(
        lambda value: value.geometry.edge_endpoint_roots, model, tuple(roots)
    )
    assert _shared_two_arc_circle(model, 0, 5) is not None
    assert _shared_two_arc_circle(outside, 0, 5) is None


def test_revolution_rims_bind_exact_full_turn_circle_edges() -> None:
    import equinox as eqx

    from phydrax.geometry.brep import brep_cone, brep_cylinder
    from phydrax.geometry.brep._source_shell_admissibility import (
        certify_revolution_rim_pair,
    )

    contract = SpatialCoordinateContract.si()
    # Faces 0 and 2 are the radial-profile caps, face 1 the lateral face; each
    # rim is stored as an exact circle edge, not an intersection branch.
    for model in (
        brep_cylinder(0.5, 1.0, coordinate_contract=contract),
        brep_cone(0.5, 0.25, 1.0, coordinate_contract=contract),
    ):
        for first, second in ((0, 1), (1, 2)):
            common = set(model.geometry.topology.face_edges[first]) & set(
                model.geometry.topology.face_edges[second]
            )
            proof = certify_revolution_rim_pair(model, first, second)
            assert proof is not None
            assert set(proof.shared_edges) == common
            assert proof.replay(model)
            assert certify_revolution_rim_pair(model, second, first) is None
        assert certify_revolution_rim_pair(model, 0, 2) is None

    model = brep_cylinder(0.5, 1.0, coordinate_contract=contract)
    (edge,) = set(model.geometry.topology.face_edges[0]) & set(
        model.geometry.topology.face_edges[1]
    )
    # An edge that stops short of the full native turn leaves part of the
    # exact rim undeclared.
    shortened = eqx.tree_at(
        lambda value: value.geometry.edge_ranges,
        model,
        model.geometry.edge_ranges.at[edge, 1].set(1.0),
    )
    assert certify_revolution_rim_pair(shortened, 0, 1) is None


def test_face_pairs_bind_exact_rims_without_interval_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.geometry.brep import (
        _source_shell_admissibility as admissibility,
        brep_cylinder,
    )
    from phydrax.geometry.brep._source_shell_admissibility import (
        SourceFaceBranchCoverage,
    )

    model = brep_cylinder(0.5, 1.0, coordinate_contract=SpatialCoordinateContract.si())
    traced: list[tuple[int, int]] = []
    branch_pair = admissibility.certify_retained_branch_pair

    def counted(
        model: BRepModel, first: int, second: int, /
    ) -> SourceFaceBranchCoverage | None:
        traced.append((first, second))
        return branch_pair(model, first, second)

    monkeypatch.setattr(admissibility, "certify_retained_branch_pair", counted)
    certificates = admissibility.certify_source_face_pairs(model)
    # Both side/cap rims bind through the exact meridian proof; the interval
    # branch route (Krawczyk tracing of the rim circle) is never entered.
    assert certificates is not None
    for first, second in ((0, 1), (1, 2)):
        rim = admissibility.certify_revolution_rim_pair(model, first, second)
        assert rim is not None and rim.certificate_id in certificates
    assert traced == []


def _square(
    corners: tuple[tuple[float, float, float], ...], /
) -> tuple[dict[int, tuple[Fraction, ...]], dict[int, tuple[int, int]]]:
    vertices = {
        index: tuple(Fraction(value) for value in corner)
        for index, corner in enumerate(corners)
    }
    edges = {index: (index, (index + 1) % len(corners)) for index in range(len(corners))}
    return vertices, edges


def _plane_of(
    polygon: tuple[dict[int, tuple[Fraction, ...]], dict[int, tuple[int, int]]], /
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    from phydrax.geometry._rational import rational_cross

    points = polygon[0]
    first = tuple(b - a for a, b in zip(points[0], points[1], strict=True))
    second = tuple(b - a for a, b in zip(points[0], points[2], strict=True))
    return points[0], rational_cross(first, second)


def test_plane_pair_rule_refuses_crossing_and_coplanar_faces() -> None:
    from phydrax.geometry.brep._source_shell_admissibility import (
        _plane_faces_meet_only_shared,
    )

    floor = _square(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))

    def meet_only_shared(
        other: tuple[dict[int, tuple[Fraction, ...]], dict[int, tuple[int, int]]], /
    ) -> bool:
        return _plane_faces_meet_only_shared(
            (_plane_of(floor), _plane_of(other)), (floor, other), set(), set()
        )

    # A wall pushed through the floor's interior crosses it off any shared edge.
    through = _square(((1, 0.5, -1), (1, 1.5, -1), (1, 1.5, 1), (1, 0.5, 1)))
    assert not meet_only_shared(through)
    # The same wall standing beside the floor, and one above it, are disjoint.
    beside = _square(((3, 0.5, -1), (3, 1.5, -1), (3, 1.5, 1), (3, 0.5, 1)))
    above = _square(((1, 0.5, 1), (1, 1.5, 1), (1, 1.5, 2), (1, 0.5, 2)))
    assert meet_only_shared(beside) and meet_only_shared(above)
    # A wall touching the floor in a segment it does not share meets it there.
    standing = _square(((1, 0.5, 0), (1, 1.5, 0), (1, 1.5, 1), (1, 0.5, 1)))
    assert not meet_only_shared(standing)
    # Coplanar faces are refused, even when disjoint; parallel ones are not.
    coplanar = _square(((5, 0, 0), (6, 0, 0), (6, 1, 0), (5, 1, 0)))
    parallel = _square(((0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1)))
    assert not meet_only_shared(coplanar)
    assert meet_only_shared(parallel)


def test_concave_plane_pairs_of_an_l_prism_certify() -> None:
    # The L-shaped caps lie on both sides of the inner walls' planes and touch
    # them only on the shared edges: the line test certifies those pairs.
    from phydrax.geometry.brep._constructors import (
        brep_extrusion,
        PlanarProfile,
        ProfileLoop,
        ProfilePlane,
    )
    from phydrax.geometry.brep._source_shell_admissibility import (
        certify_plane_plane_pair,
    )
    from phydrax.meshing._native_cad_shell import prove_source_shell

    model = brep_extrusion(
        PlanarProfile(
            ProfilePlane((0.0, 0.0, 0.0)),
            ProfileLoop.polygon(
                ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0))
            ),
        ),
        (0.0, 0.0, 1.0),
        coordinate_contract=SpatialCoordinateContract.si(),
    )
    topology = model.geometry.topology
    adjacent = {
        tuple(sorted(faces)) for faces in topology.edge_faces if len(set(faces)) == 2
    }
    assert len(adjacent) == 18
    for first, second in sorted(adjacent):
        assert certify_plane_plane_pair(model, first, second) is not None
    assert prove_source_shell(MeshingDomain.from_brep(model)).refusal is None


_BoxPoints = tuple[tuple[float, float, float], tuple[float, float, float]]


def _partition_region_boxes(*boxes: _BoxPoints) -> BRepModel:
    from phydrax.geometry.brep._partition import (
        all_brep_solids,
        BRepPartitionOperand,
        BRepPartitionPlan,
        BRepPartitionPolicy,
        BRepPartitionRole,
        partition_brep,
    )

    coordinates = SpatialCoordinateContract.si()
    operands = []
    for index, (lower, upper) in enumerate(boxes):
        source = brep_box(lower, upper, coordinate_contract=coordinates)
        operands.append(
            BRepPartitionOperand(
                f"region-{index}",
                source,
                BRepPartitionRole.REGION,
                all_brep_solids(source),
            )
        )
    return partition_brep(
        BRepPartitionPlan(
            coordinates,
            tuple(operands),
            BRepPartitionPolicy(tuple(f"region-{index}" for index in range(len(boxes)))),
        )
    ).model


def _assemble_region_shells(
    models: tuple[BRepModel, ...],
    regions: tuple[tuple[tuple[int, int], ...], ...],
    /,
) -> BRepModel:
    """Join exact source banks without sewing or changing any incidence identities."""
    from phydrax.geometry.brep._constructors import _Builder, assemble_brep_model

    builder = _Builder()
    shell_faces = []
    shell_orientations = []
    for model in models:
        geometry = model.geometry
        vertex_start = len(builder.vertices)
        edge_start = len(builder.edge_vertices)
        coedge_start = len(builder.coedge_edges)
        face_start = len(builder.faces)
        for point, exact in zip(
            np.asarray(geometry.vertex_points), geometry.vertex_exact_points, strict=True
        ):
            assert exact is not None
            builder.vertex(point, exact_point=exact)
        ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
        for edge, (first, second) in enumerate(geometry.edge_vertices):
            curve = geometry.edge_curves[edge]
            start, stop = ranges[edge]
            builder.edge(
                geometry.curves[curve],
                float(start),
                float(stop),
                first + vertex_start,
                second + vertex_start,
            )
        for coedge, edge in enumerate(geometry.coedge_edges):
            builder.coedge(
                edge + edge_start,
                geometry.coedge_senses[coedge],
                geometry.pcurves[coedge],
            )
        bounds = np.asarray(model.parameter_bounds)
        orientation = np.asarray(model.orientation)
        for face, patch in enumerate(model.patches):
            builder.face(
                patch,
                bounds[face],
                int(orientation[face]),
                model.physical_tags[face],
                [
                    [coedge + coedge_start for coedge in loop]
                    for loop in geometry.face_loops[face]
                ],
            )
        shell_faces.extend(
            tuple(face + face_start for face in faces) for faces in geometry.shell_faces
        )
        shell_orientations.extend(geometry.shell_orientations)
    shell_signs = {shell: sign for region in regions for shell, sign in region}
    geometry, patches, bounds, orientation, tags, trims = builder.geometry(
        solid=True,
        shell_faces=tuple(shell_faces),
        shell_orientations=tuple(
            tuple(sign * shell_signs[shell] for sign in orientations)
            for shell, orientations in enumerate(shell_orientations)
        ),
        solid_shells=tuple(tuple(shell for shell, _ in region) for region in regions),
    )
    return assemble_brep_model(
        geometry,
        patches,
        bounds,
        orientation,
        tags,
        trim_domains=trims,
        coordinate_contract=SpatialCoordinateContract.si(),
        source_id="authored-exact-region-shells",
    )


def _assert_region_contact_premises(domain: MeshingDomain, model: BRepModel, /) -> None:
    faces = tuple(_convex_source_face(model, i) for i in range(len(model.patches)))
    exact_faces = tuple(face for face in faces if face is not None)
    assert len(exact_faces) == len(faces)
    sides = _source_region_sides(model)
    assert sides is not None
    assert _source_region_shells(model, domain, exact_faces)
    assert _source_region_contacts(model, exact_faces, sides)


def _material_at(model: BRepModel, point: tuple[float, float, float], /) -> set[int]:
    from phydrax.geometry.brep._projection import prepare_brep_projection

    projection = prepare_brep_projection(model)
    query = np.asarray(point, dtype=np.float64)
    labels = tuple(
        projection._contains_solid(query, solid)
        for solid in range(model.topology.num_solids)
    )
    assert None not in labels
    return {solid for solid, inside in enumerate(labels) if inside}


def _check_face_sides(
    model: BRepModel, proof: SourceRegionAdmissibility, faces: tuple[int, ...], /
) -> None:
    from phydrax.geometry.brep._patches import PlanePatch
    from phydrax.geometry.brep._projection import prepare_brep_projection

    projection = prepare_brep_projection(model)
    points = np.asarray(model.geometry.vertex_points, dtype=np.float64)
    orientations = np.asarray(model.orientation, dtype=np.int8)
    for face in faces:
        vertices = sorted(
            {
                vertex
                for edge in model.topology.face_edges[face]
                for vertex in model.geometry.edge_vertices[edge]
            }
        )
        center = np.mean(points[vertices], axis=0)
        patch = model.patches[face]
        assert isinstance(patch, PlanePatch)
        normal = np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
        # These box fixtures have axis normals and dyadic centroids; this exact
        # offset is smaller than the clearance to any other distinct boundary.
        offset = normal * (orientations[face] / 16.0)
        for query, owner in zip(
            (center - offset, center + offset), proof.face_sides[face], strict=True
        ):
            for solid in range(model.topology.num_solids):
                assert projection._contains_solid(query, solid) is (solid == owner)


def test_source_regions_refuse_nested_outward_material_despite_pair_separation() -> None:
    outer = _partition_region_boxes(((0.0, 0.0, 0.0), (4.0, 4.0, 4.0)))
    inner = _partition_region_boxes(((1.0, 1.0, 1.0), (2.0, 2.0, 2.0)))
    nested = _assemble_region_shells((outer, inner), (((0, 1),), ((1, 1),)))
    domain = MeshingDomain.from_brep(nested)
    _assert_region_contact_premises(domain, nested)
    assert _material_at(nested, (1.5, 1.5, 1.5)) == {0, 1}
    assert certify_source_regions(domain) is None

    separated = _partition_region_boxes(
        ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0)), ((5.0, 1.0, 1.0), (6.0, 2.0, 2.0))
    )
    assert certify_source_regions(MeshingDomain.from_brep(separated)) is not None


def test_source_regions_admit_a_solid_in_a_declared_inward_cavity() -> None:
    outer = _partition_region_boxes(((0.0, 0.0, 0.0), (6.0, 6.0, 6.0)))
    void = _partition_region_boxes(((1.0, 1.0, 1.0), (5.0, 5.0, 5.0)))
    island = _partition_region_boxes(((2.0, 2.0, 2.0), (3.0, 3.0, 3.0)))
    model = _assemble_region_shells((outer, void, island), (((0, 1), (1, -1)), ((2, 1),)))
    domain = MeshingDomain.from_brep(model)
    proof = certify_source_regions(domain)
    assert proof is not None and proof.replay(domain)
    assert _material_at(model, (0.5, 0.5, 0.5)) == {0}
    assert _material_at(model, (1.5, 1.5, 1.5)) == set()
    assert _material_at(model, (2.5, 2.5, 2.5)) == {1}
    _check_face_sides(model, proof, tuple(range(len(model.patches))))


def test_source_regions_check_all_disconnected_shell_components() -> None:
    first = _partition_region_boxes(((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)))
    second = _partition_region_boxes(((4.0, 0.0, 0.0), (6.0, 2.0, 2.0)))
    far = _partition_region_boxes(((8.0, 0.0, 0.0), (9.0, 1.0, 1.0)))
    near = _partition_region_boxes(((4.5, 0.5, 0.5), (5.5, 1.5, 1.5)))
    regions = (((0, 1), (1, 1)), ((2, 1),))
    separated = _assemble_region_shells((first, second, far), regions)
    assert sorted(map(len, separated.geometry.solid_shells)) == [1, 2]
    assert certify_source_regions(MeshingDomain.from_brep(separated)) is not None

    nested = _assemble_region_shells((first, second, near), regions)
    domain = MeshingDomain.from_brep(nested)
    _assert_region_contact_premises(domain, nested)
    assert certify_source_regions(domain) is None


def test_source_regions_require_declared_opposite_interface_incidence() -> None:
    first = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    second = ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    unsplit = _assemble_region_shells(
        (_partition_region_boxes(first), _partition_region_boxes(second)),
        (((0, 1),), ((1, 1),)),
    )
    assert all(len(owners) == 1 for owners in unsplit.topology.face_solids)
    assert certify_source_regions(MeshingDomain.from_brep(unsplit)) is None

    model = _partition_region_boxes(first, second)
    domain = MeshingDomain.from_brep(model)
    (interface,) = tuple(
        face for face, owners in enumerate(model.topology.face_solids) if len(owners) == 2
    )
    proof = certify_source_regions(domain)
    assert proof is not None
    assert set(proof.face_sides[interface]) == {0, 1}
    _check_face_sides(model, proof, (interface,))

    reversed_interface = _assemble_region_shells((model,), (((0, 1),), ((1, -1),)))
    topology = reversed_interface.topology
    interface_signs = {
        signs[faces.index(interface)]
        for faces, signs in zip(
            topology.solid_faces, topology.solid_face_orientations, strict=True
        )
    }
    assert len(interface_signs) == 1
    assert certify_source_regions(MeshingDomain.from_brep(reversed_interface)) is None

    # Reversing both regions preserves opposite interface signs and every
    # manifold link; exact material-side orientation must still refuse.
    inward = _assemble_region_shells((model,), (((0, -1),), ((1, -1),)))
    inward_domain = MeshingDomain.from_brep(inward)
    _assert_region_contact_premises(inward_domain, inward)
    assert not proof.replay(inward_domain)
    assert certify_source_regions(inward_domain) is None


def test_source_regions_refuse_imported_interface_without_exact_vertex_bindings() -> None:
    pytest.importorskip("OCP")
    from OCP.BOPAlgo import BOPAlgo_Splitter
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCP.gp import gp_Pnt

    from phydrax.interchange import import_occt_shape

    splitter = BOPAlgo_Splitter()
    for x in (0.0, 1.0):
        splitter.AddArgument(
            BRepPrimAPI_MakeBox(gp_Pnt(x, 0.0, 0.0), 1.0, 1.0, 1.0).Shape()
        )
    splitter.Perform()
    assert not splitter.HasErrors()
    model = import_occt_shape(
        splitter.Shape(), coordinate_contract=SpatialCoordinateContract.si()
    ).model
    assert sum(len(owners) == 2 for owners in model.topology.face_solids) == 1
    assert all(point is None for point in model.geometry.vertex_exact_points)
    assert certify_source_regions(MeshingDomain.from_brep(model)) is None


def test_exact_split_box_matches_source_regions_through_labeled_coverage() -> None:
    from phydrax._meshcore import (
        cad_mesh_solid,
        CadFidelityMesh,
        MeshcoreStatus,
        VerdictStatus,
        verify_cad_fidelity,
        verify_chart_partition,
        verify_surface_embedding,
        verify_tetrahedra,
        verify_volume,
    )
    from phydrax.discretization._cell_mesh import CellMesh
    from phydrax.geometry._meshing_domain import SourcePreparation
    from phydrax.geometry.brep._curve_approximation_topology import (
        certify_trim_reference_homotopy,
    )
    from phydrax.geometry.brep._source_embedding import (
        certify_brep_trim_sheet,
        certify_reference_trim_sheet,
    )
    from phydrax.meshing._native_cad_certificate import (
        FIDELITY_DEPTH,
        NativeCadSurfaceWitness,
    )
    from phydrax.meshing._native_cad_records import _native_cad_regional_surface
    from phydrax.meshing._native_cad_shell import _reference_loops
    from phydrax.meshing.providers._native_cad_bundle import (
        cad_geometry_bundle,
        coefficient_bounds,
    )

    source = _partition_region_boxes(
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
        ((1.0, 0.0, 0.0), (2.0, 1.0, 1.0)),
    )
    domain = MeshingDomain.from_brep(source)
    material = certify_source_regions(domain)
    assert material is not None
    bundle = cad_geometry_bundle(domain)
    model = bundle.model
    tolerance = 0.05
    solid = cad_mesh_solid(
        model,
        np.full(model.surface_nodes.size, 0.4, dtype=np.float64),
        np.full(model.surface_nodes.size, tolerance, dtype=np.float64),
        np.full(model.curve_nodes.size, tolerance, dtype=np.float64),
        radius_edge_bound=8.0,
        min_dihedral_degrees=0.0,
    )
    assert solid.status is MeshcoreStatus.OK
    points, cells = solid.volume.points, solid.volume.tetrahedra
    mesh = CellMesh.from_simplices(points, cells, dimension=3)
    np.testing.assert_array_equal(mesh.coordinates, points)
    np.testing.assert_array_equal(mesh.blocks[0].vertices, cells)
    witness = NativeCadSurfaceWitness(mesh, solid.surface, model.loop_surfaces.size)
    rows = _native_cad_regional_surface(
        mesh, domain, witness, native_surface=solid.surface
    )
    expected = np.asarray(
        [material.face_sides[domain.source_indices[2][patch]] for patch in rows.patches],
        dtype=np.int64,
    )
    # Source solid indices are not the region rows supplied by the mesh.
    source_solid = np.asarray((*domain.region_source_indices, -1), dtype=np.int64)
    np.testing.assert_array_equal(source_solid[rows.regions], expected)
    assert np.any(np.all(rows.regions >= 0, axis=1))
    chords = witness.chords()
    assert not verify_chart_partition(
        len(domain.patches), witness.chart_patches, witness.chart_uv, chords
    ).size
    for region in (None, *range(len(domain.regions))):
        _, triangles = rows.outward(region)
        assert (
            verify_surface_embedding(points, triangles).status is VerdictStatus.CERTIFIED
        )
    preparation = SourcePreparation(domain)
    for patch in range(len(domain.patches)):
        native = certify_brep_trim_sheet(domain, patch)
        loops = _reference_loops(domain, model, witness, patch)
        assert native is not None and loops is not None, patch
        certify_trim_reference_homotopy(loops, preparation)
        assert (
            certify_reference_trim_sheet(domain, patch, native, loops, preparation)
            is not None
        )
    lower, upper = coefficient_bounds(bundle, domain)
    fidelity = verify_cad_fidelity(
        model,
        lower,
        upper,
        CadFidelityMesh(
            points,
            witness.chart_triangles,
            witness.chart_patches,
            witness.chart_uv,
            witness.surface_triangles,
            witness.chart_to_mesh,
            witness.edge_chord,
        ),
        chords,
        tolerance,
        FIDELITY_DEPTH,
    )
    assert fidelity.status is MeshcoreStatus.OK and not fidelity.findings.size
    assert np.all(fidelity.item_status == int(MeshcoreStatus.OK))
    assert fidelity.maximum_upper <= tolerance
    assert verify_tetrahedra(points, cells).status is VerdictStatus.CERTIFIED
    np.testing.assert_array_equal(np.unique(solid.volume.tetrahedron_regions), [0, 1])
    coverage = verify_volume(
        points,
        cells,
        rows.triangles,
        cell_regions=solid.volume.tetrahedron_regions,
        surface_regions=rows.regions,
    )
    assert coverage.certified, coverage
