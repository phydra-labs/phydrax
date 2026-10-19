from collections.abc import Iterator
from contextlib import contextmanager
from itertools import permutations, product
from math import pi

import numpy as np
import pytest

import phydrax.meshing as M
from phydrax._fingerprint import canonical_fingerprint
from phydrax._physical import SpatialCoordinateContract
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._meshing_domain import MeshingDomain
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_extrusion,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._model import BRepModel
from phydrax.geometry.brep._normal_range import brep_normal_ranges
from phydrax.geometry.brep._patches import (
    BSplineCurve,
    CylinderPatch,
    ExtrusionSurface,
    PlanePatch,
)
from phydrax.geometry.brep._selection import (
    BRepPredicate,
    BRepSelection,
    select_brep_entities,
)
from phydrax.geometry.design._naming import (
    RecipeEntityName,
    RecipeEntityNames,
    RecipeNamingError,
)
from tests._support.cad_models import native_solid


SI = SpatialCoordinateContract.si()


@contextmanager
def _budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100_000_000, bytes_upper=1_000_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000_000, maximum_memory_bytes=1_000_000_000
        )
        with budget.activate():
            yield budget


def _names(model: BRepModel) -> RecipeEntityNames:
    return RecipeEntityNames(
        model,
        tuple(
            RecipeEntityName("face", "material", f"face-{i}", ("extrusion",))
            for i in range(len(model.face_ids))
        ),
        tuple(
            RecipeEntityName("edge", "material", f"edge-{i}", ("extrusion",))
            for i in range(len(model.edge_ids))
        ),
    )


def _select(
    model: BRepModel, predicate: BRepPredicate, *, kind: str = "face"
) -> BRepSelection:
    with _budget() as budget:
        if kind == "edge":
            return select_brep_entities(
                model, predicate, kind="edge", maximum_cells=32, coordinate_budget=budget
            )
        return select_brep_entities(
            model, predicate, maximum_cells=32, coordinate_budget=budget
        )


@pytest.mark.parametrize("name", ("box", "cylinder"))
def test_closed_form_box_cylinder_normal_angles(name: str) -> None:
    model = native_solid(name, catalog="geometry")
    with _budget() as budget:
        records = brep_normal_ranges(
            model,
            (0.0, 0.0, 1.0),
            draft_angle=0.1,
            maximum_cells=16,
            coordinate_budget=budget,
        )
    assert tuple(record.face_id for record in records) == model.face_ids
    for face, (patch, record) in enumerate(zip(model.patches, records, strict=True)):
        if isinstance(patch, PlanePatch):
            normal = (
                np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
                * np.asarray(model.orientation)[face]
            )
            reference = np.arccos(normal[2] / np.sqrt(np.sum(normal**2)))
        else:
            assert isinstance(patch, CylinderPatch)
            reference = pi / 2
            assert record.status == "vertical"
        lower, upper = np.asarray(record.angle_enclosure)
        assert lower <= reference <= upper
        assert record.singular_points.shape == (0, 2, 3)


def test_drafted_prism_closed_form_and_exact_limit() -> None:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop.polygon(((0.0, 0.0), (2.0, 0.0), (1.5, 1.0), (0.5, 1.0))),
    )
    model = brep_extrusion(profile, (0.0, 2.0, 0.0), coordinate_contract=SI)
    taper = np.arctan(0.5)
    with _budget() as budget:
        records = brep_normal_ranges(
            model,
            (0.0, 0.0, 1.0),
            draft_angle=float(taper),
            maximum_cells=8,
            coordinate_budget=budget,
        )
    sloped = []
    for face, (patch, record) in enumerate(zip(model.patches, records, strict=True)):
        if isinstance(patch, PlanePatch):
            first, second = np.asarray(patch.first_axis), np.asarray(patch.second_axis)
        else:
            assert isinstance(patch, ExtrusionSurface)
            assert isinstance(patch.curve, BSplineCurve)
            controls = np.asarray(patch.curve.control_points)
            assert controls.shape == (2, 3)
            first, second = controls[1] - controls[0], np.asarray(patch.direction)
        normal = np.cross(first, second) * np.asarray(model.orientation)[face]
        normal /= np.sqrt(np.sum(normal**2))
        angle = np.arccos(normal[2])
        lower, upper = np.asarray(record.angle_enclosure)
        assert lower <= angle <= upper
        if 0 < normal[2] < 1:
            sloped.append(record)
            assert lower <= pi / 2 - taper <= upper
            assert record.status == "unresolved"
            assert record.unresolved_cells.shape[0] > 0
    assert len(sloped) == 2


def test_sphere_reports_both_poles_without_nudging() -> None:
    model = native_solid("sphere", catalog="geometry")
    with _budget() as budget:
        records = brep_normal_ranges(
            model,
            (0.0, 0.0, 1.0),
            draft_angle=0.2,
            maximum_cells=16,
            coordinate_budget=budget,
        )
    points = np.concatenate([np.asarray(record.singular_points) for record in records])
    assert points.shape == (2, 2, 3)
    for reference in (np.asarray((0.0, 0.0, -1.0)), np.asarray((0.0, 0.0, 1.0))):
        assert any(
            np.all(box[0] <= reference) and np.all(reference <= box[1]) for box in points
        )
    assert all(record.status == "unresolved" for record in records)


def test_signed_axis_permutations_and_power_two_scaling_preserve_decisions() -> None:
    expected = ["positive", "undercut", "vertical", "vertical", "vertical", "vertical"]
    for index, (permutation, signs) in enumerate(
        product(permutations(range(3)), product((-1, 1), repeat=3))
    ):
        exponent = (-20, 0, 20)[index % 3]
        scale = 2.0**exponent
        matrix = np.eye(3)[np.asarray(permutation)] * np.asarray(signs)[:, None]
        corners = np.stack((np.zeros(3), matrix @ (scale * np.asarray((1.0, 2.0, 3.0)))))
        model = brep_box(
            np.min(corners, axis=0), np.max(corners, axis=0), coordinate_contract=SI
        )
        with _budget() as budget:
            records = brep_normal_ranges(
                model,
                matrix @ np.asarray((0.0, 0.0, 1.0)),
                draft_angle=0.1,
                maximum_cells=1,
                coordinate_budget=budget,
            )
        assert sorted(record.status for record in records) == expected


def test_box_with_hole_selections_match_analytic_oracles() -> None:
    model = native_solid("extruded_plate", catalog="geometry")
    planar = BRepPredicate("analytic_family", family="plane")
    top = _select(model, planar & BRepPredicate("facing", angle=0.1))
    hole = _select(model, BRepPredicate("analytic_family", family="cylinder"))
    far = _select(model, BRepPredicate("extremal", direction=(1.0, 0.0, 0.0)))
    assert len(top.selected) == len(hole.selected) == len(far.selected) == 1
    assert not top.unresolved and not hole.unresolved and not far.unresolved
    assert isinstance(model.patches[hole.selected[0].index], CylinderPatch)
    patch = model.patches[top.selected[0].index]
    assert isinstance(patch, PlanePatch)
    assert np.asarray(patch.origin)[2] == 0.5
    patch = model.patches[far.selected[0].index]
    assert isinstance(patch, PlanePatch)
    assert np.asarray(patch.origin)[0] == 2.0
    assert np.asarray(patch.first_axis)[0] == np.asarray(patch.second_axis)[0] == 0.0


def test_straddling_normals_and_kleene_combinators() -> None:
    model = native_solid("cylinder", catalog="geometry")
    facing = BRepPredicate("facing", direction=(1.0, 0.0, 0.0), angle=pi / 4)
    family = BRepPredicate("analytic_family", family="cylinder")
    direct = _select(model, facing)
    wall = next(
        entity
        for entity in direct.unresolved
        if isinstance(model.patches[entity.index], CylinderPatch)
    )
    conjunction = _select(model, family & facing)
    disjunction = _select(model, (~family) | facing)
    assert conjunction.unresolved == (wall,)
    assert disjunction.unresolved == (wall,)
    assert set(_select(model, ~facing).unresolved) == set(direct.unresolved)
    parallel = _select(model, BRepPredicate("parallel", angle=0.1))
    perpendicular = _select(model, BRepPredicate("perpendicular", angle=0.1))
    assert len(parallel.selected) == 2
    assert perpendicular.selected == (wall,)


def test_adjacency_neighbors_and_edge_convexity() -> None:
    model = native_solid("box", catalog="geometry")
    top = _select(model, BRepPredicate("facing", angle=0.1))
    adjacent = _select(model, BRepPredicate("adjacent", entities=top.selected))
    neighbors = _select(model, BRepPredicate("facing", angle=0.1).neighbors())
    assert len(adjacent.selected) == 4
    assert adjacent.selected == neighbors.selected
    convex = _select(model, BRepPredicate("convexity"), kind="edge")
    assert convex.selected == model.edge_ids
    assert not convex.unresolved


def test_label_producer_binding_parameter_edit_and_split_refusal() -> None:
    model = native_solid("box", catalog="geometry")
    selection = BRepSelection(
        model, (model.face_ids[0],), model.face_ids[1:], (), provenance=("label:inlet",)
    )
    domain = MeshingDomain.from_brep(model)
    control = selection.bind_region("inlet", _names(model), domain, ("material",))
    changed = brep_box((0.0, 0.0, 0.0), (2.0, 3.0, 4.0), coordinate_contract=SI)
    rebound = control.rebind(_names(changed), MeshingDomain.from_brep(changed))
    np.testing.assert_array_equal(rebound.scope.entity_ids, control.scope.entity_ids)
    assert rebound.scope.source_revision != control.scope.source_revision
    split = brep_extrusion(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(
                ((0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 2.0), (0.0, 2.0))
            ),
        ),
        (0.0, 0.0, 1.0),
        coordinate_contract=SI,
    )
    with pytest.raises(RecipeNamingError, match="split, merge"):
        control.rebind(_names(split), MeshingDomain.from_brep(split))
    with pytest.raises(RecipeNamingError, match="source revision"):
        control.rebind(_names(model), MeshingDomain.from_brep(changed))
    split_selection = BRepSelection(
        split,
        (split.face_ids[0],),
        split.face_ids[1:],
        (),
        provenance=("label:split-inlet",),
    )
    split_control = split_selection.bind_region(
        "inlet", _names(split), MeshingDomain.from_brep(split), ("material",)
    )
    with pytest.raises(RecipeNamingError, match="split, merge"):
        split_control.rebind(_names(model), domain)


def test_unresolved_binding_requires_explicit_inclusion() -> None:
    model = native_solid("box", catalog="geometry")
    selection = BRepSelection(
        model,
        (model.face_ids[0],),
        model.face_ids[2:],
        (model.face_ids[1],),
        provenance=("label:uncertain",),
    )
    domain = MeshingDomain.from_brep(model)
    with pytest.raises(ValueError, match="explicit inclusion"):
        selection.bind_region("inlet", _names(model), domain, ("material",))
    included = selection.bind_region(
        "inlet", _names(model), domain, ("material",), include_unresolved=True
    )
    excluded = selection.bind_region(
        "inlet", _names(model), domain, ("material",), include_unresolved=False
    )
    assert included.scope.entity_ids.shape == (2,)
    assert excluded.scope.entity_ids.shape == (1,)


def test_absent_recipe_binding_preserves_two_control_identity_payloads() -> None:
    domain = MeshingDomain.from_brep(native_solid("box", catalog="geometry"))
    for index, required in ((0, True), (1, False)):
        scope = M.MeshingScope(
            domain.source_id,
            domain.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            2,
            domain.entity_set_id(2),
            np.asarray((index,), dtype=np.int64),
        )
        # This is the pre-field constructor payload, not a re-pinned ID.
        before = canonical_fingerprint(
            {
                "kind": "patch-control",
                "name": f"wall-{index}",
                "scope": scope.scope_id,
                "adjacent_region_names": ("material",),
                "required": required,
            }
        )
        after = M.PatchControl(f"wall-{index}", scope, ("material",), required=required)
        assert before == after.control_id
        assert after.recipe_binding is None


def test_native_mesh_patch_carries_selected_source_faces() -> None:
    model = native_solid("box", catalog="geometry")
    domain = MeshingDomain.from_brep(model)
    selection = _select(model, BRepPredicate("facing", angle=0.1))
    control = selection.bind_region("outlet", _names(model), domain, ("material",))
    boundary = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        domain.entity_set_id(2),
        np.arange(len(domain.patches), dtype=np.int64),
    )
    solid = M.MeshingScope(
        domain.source_id,
        domain.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        3,
        domain.entity_set_id(3),
        np.asarray((0,), dtype=np.int64),
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 3, M.CellFamilyPolicy(required=("triangle",))),
        boundary,
        size_controls=(
            M.UniformSizeControl(boundary, 1.0, strength=M.SizeControlStrength.SOFT),
        ),
        region_controls=(
            M.RegionControl(solid, "material", "steel", M.RegionRole.SOLID),
        ),
        patch_controls=(control,),
    )
    provider = M.NativeMeshingProvider(M.NativeMeshingOptions("parametric_surface"))
    result = provider.plan(
        M.NativeSurfaceSource(domain), specification, coordinate_contract=SI
    ).execute()
    outlet = next(patch for patch in result.patches if patch.name == "outlet")
    faces = np.asarray(result.mesh.blocks[0].vertices)
    coordinates = np.asarray(result.mesh.coordinates)
    face_ids = np.asarray(result.mesh.entity_set(2).entity_ids)
    top = np.all(coordinates[faces, 2] == 3.0, axis=1)
    np.testing.assert_array_equal(outlet.scope.entity_ids, face_ids[top])
    assert outlet.scope.entity_ids.shape[0] > 0


def test_normal_budget_refuses_before_evaluation() -> None:
    model = native_solid("box", catalog="geometry")
    with HostBatchTicket(1, bytes_upper=1024) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1, maximum_memory_bytes=1024
        )
        with pytest.raises(ValueError, match="budget|work|reserve|resource"):
            brep_normal_ranges(
                model,
                (0.0, 0.0, 1.0),
                draft_angle=0.1,
                maximum_cells=16,
                coordinate_budget=budget,
            )


def test_drafted_prism_positive_and_insufficient_limits() -> None:
    profile = PlanarProfile(
        ProfilePlane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ProfileLoop.polygon(((0.0, 0.0), (2.0, 0.0), (1.5, 1.0), (0.5, 1.0))),
    )
    model = brep_extrusion(profile, (0.0, 2.0, 0.0), coordinate_contract=SI)
    reference = np.pi / 2 - np.arctan(0.5)
    for draft, expected in ((0.1, "positive"), (0.6, "insufficient")):
        with _budget() as budget:
            records = brep_normal_ranges(
                model,
                (0.0, 0.0, 1.0),
                draft_angle=draft,
                maximum_cells=1,
                coordinate_budget=budget,
            )
        slopes = tuple(
            record
            for record in records
            if np.asarray(record.angle_enclosure)[0]
            <= reference
            <= np.asarray(record.angle_enclosure)[1]
        )
        assert len(slopes) == 2
        assert all(record.status == expected for record in slopes)
