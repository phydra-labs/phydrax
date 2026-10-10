#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable
from dataclasses import dataclass

import jax.random as jr
import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakePolygon
from OCP.gp import gp_Pnt
from OCP.TopoDS import TopoDS_Face

import phydrax as phx
from phydrax.geometry.multiregion_surface import (
    multiregion_sheet_views,
    PreparedMultiRegionSurface,
    seed_double_bubble,
)
from phydrax.geometry.surface import InterfaceSide
from phydrax.meshing import MeshInterfaceAttachment, MeshPart
from phydrax.solver.coupling import (
    InterfaceBinding,
    InterfaceEndpoint,
    InterfaceSource,
    PairedSupportAttachment,
    SheetViewAttachment,
)


_CONTRACT = phx.SpatialCoordinateContract.si()
_POLICY = phx.meshing.AssociationPropagationPolicy(classification_tolerance=1e-8)
_EMBEDDING = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
_TOLERANCE = 1e-9


def _rectangle(x0: float, x1: float) -> TopoDS_Face:
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((x0, 0.0), (x1, 0.0), (x1, 1.0), (x0, 1.0)):
        polygon.Add(gp_Pnt(x, y, 0.0))
    polygon.Close()
    return BRepBuilderAPI_MakeFace(polygon.Wire()).Face()


def _plate_projection(width: float) -> phx.interchange.PreparedOcctProjection:
    """Two rectangles sharing the B-Rep edge x = 0; ``width`` sets the right one."""
    shape = BRepAlgoAPI_Fuse(_rectangle(-1.0, 0.0), _rectangle(0.0, width)).Shape()
    model = phx.interchange.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    return phx.interchange.prepare_occt_projection(model, shape, embedding=_EMBEDDING)


def _entity_at(
    projection: phx.interchange.PreparedOcctProjection,
    dimension: int,
    point: tuple[float, float],
) -> str:
    """The unique B-Rep entity of ``dimension`` passing through ``point``."""
    count = int(np.asarray(projection.entity_counts)[dimension])
    projected = projection.project(
        np.repeat(np.asarray([point], dtype=np.float64), count, axis=0),
        np.full((count,), dimension, dtype=np.int64),
        np.arange(count, dtype=np.int64),
    )
    matches = np.flatnonzero(np.asarray(projected.residuals) < 1e-12)
    assert matches.size == 1
    return phx.geometry.brep_entity_id(
        projection.source_revision, dimension, int(matches[0])
    )


def _grid_mesh(x0: float, x1: float, count: int) -> phx.discretization.CellMesh:
    xs = np.linspace(x0, x1, count)
    points = np.asarray([[x, y] for y in np.linspace(0.0, 1.0, count) for x in xs])
    triangles = []
    for row in range(count - 1):
        for column in range(count - 1):
            a = row * count + column
            triangles += [[a, a + 1, a + count + 1], [a, a + count + 1, a + count]]
    return phx.discretization.CellMesh.from_triangles(
        points, np.asarray(triangles, dtype=np.int64)
    )


@dataclass(frozen=True, slots=True)
class PlateSide:
    """One mesh part of the plate with its certified edge and cell associations."""

    part: MeshPart
    edges: phx.meshing.GeometryAssociation
    cells: phx.meshing.GeometryAssociation
    wall: phx.meshing.MeshPatch
    region: phx.meshing.MeshZone


def _plate_side(
    projection: phx.interchange.PreparedOcctProjection,
    name: str,
    bounds: tuple[float, float],
    count: int,
    wall_entity: str,
) -> PlateSide:
    mesh = _grid_mesh(*bounds, count)
    vertex = phx.meshing.associate_mesh_vertices(mesh, projection, policy=_POLICY)
    edges = phx.meshing.associate_mesh_entities(
        mesh, vertex, projection, 1, policy=_POLICY
    )
    cells = phx.meshing.associate_mesh_entities(
        mesh, vertex, projection, 2, policy=_POLICY
    )
    rows = [
        row for row, value in enumerate(edges.source_entity_ids) if value == wall_entity
    ]
    wall = phx.meshing.MeshPatch(
        "wall",
        phx.meshing.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            1,
            mesh.entity_set(1).entity_set_id,
            np.asarray(edges.target_global_ids)[rows],
        ),
    )
    region = phx.meshing.MeshZone(
        name,
        phx.meshing.MeshZoneRole.REGION,
        phx.meshing.MeshingScope(
            mesh.mesh_id,
            mesh.numeric_version,
            phx.meshing.MeshingEntityKind.MESH,
            2,
            mesh.entity_set(2).entity_set_id,
            np.asarray(mesh.entity_set(2).entity_ids),
        ),
    )
    result = phx.meshing.certify_cell_mesh(
        mesh,
        _CONTRACT,
        associations=(vertex, edges, cells),
        patches=(wall,),
        zones=(region,),
    )
    return PlateSide(MeshPart(name, result), edges, cells, wall, region)


def _wall(side: PlateSide, wall_entity: str) -> MeshInterfaceAttachment:
    return MeshInterfaceAttachment(
        side.part,
        side.wall,
        side.edges,
        (wall_entity,),
        tolerance=_TOLERANCE,
        region=side.region,
    )


@dataclass(frozen=True, slots=True)
class Plate:
    """Two independently meshed parts of one two-face B-Rep revision."""

    projection: phx.interchange.PreparedOcctProjection
    wall_entity: str
    left_face: str
    source: InterfaceSource
    left: PlateSide
    right: PlateSide
    left_wall: MeshInterfaceAttachment
    right_wall: MeshInterfaceAttachment

    def binding(
        self, minus: MeshInterfaceAttachment, plus: MeshInterfaceAttachment
    ) -> InterfaceBinding:
        return InterfaceBinding(
            "wall",
            self.source,
            "two-sided",
            (
                InterfaceEndpoint("sem", minus, fields={"value": "u-sem"}),
                InterfaceEndpoint("vem", plus, fields={"value": "u-vem"}),
            ),
        )


@pytest.fixture(scope="module")
def plate() -> Plate:
    projection = _plate_projection(1.0)
    wall_entity = _entity_at(projection, 1, (0.0, 0.5))
    left = _plate_side(projection, "left", (-1.0, 0.0), 3, wall_entity)
    right = _plate_side(projection, "right", (0.0, 1.0), 4, wall_entity)
    return Plate(
        projection,
        wall_entity,
        _entity_at(projection, 2, (-0.5, 0.5)),
        InterfaceSource(projection.source_id, projection.source_revision, (wall_entity,)),
        left,
        right,
        _wall(left, wall_entity),
        _wall(right, wall_entity),
    )


def test_two_sided_mesh_binding_orders_sides_by_authoritative_normal(
    plate: Plate,
) -> None:
    # Independent oracle: the owner's projected edge tangent t at the wall gives
    # the normal (t_y, -t_x); the left part lies on its -x side, so the left
    # part's outward normal (+x) agrees with it exactly when t_y > 0.
    projected = plate.projection.project(
        np.asarray([[0.0, 0.5]]),
        np.asarray([1], dtype=np.int64),
        np.asarray([int(plate.wall_entity.rsplit(":", 1)[1])], dtype=np.int64),
    )
    tangent = np.asarray(projected.tangents)[0, 0]
    expected = int(np.sign(tangent[1]))
    assert expected != 0
    assert plate.left_wall.orientation == expected
    assert plate.right_wall.orientation == -expected
    minus, plus = (
        (plate.left_wall, plate.right_wall)
        if expected > 0
        else (plate.right_wall, plate.left_wall)
    )
    binding = plate.binding(minus, plus)
    assert binding.source.source_revision == plate.projection.source_revision
    assert binding.side_of("sem") is InterfaceSide.MINUS
    assert binding.side(InterfaceSide.PLUS).attachment is plus
    assert binding.endpoint("vem").field_id("value") == "u-vem"
    assert binding.witness_ids == (minus.association_id, plus.association_id)
    assert plate.left_wall.organization_ids == tuple(
        sorted((plate.left.wall.patch_id, plate.left.region.zone_id))
    )
    binding.require_current(phx.meshing.MeshAssembly((plate.left.part, plate.right.part)))


def _oriented(plate: Plate) -> tuple[MeshInterfaceAttachment, MeshInterfaceAttachment]:
    if plate.left_wall.orientation > 0:
        return plate.left_wall, plate.right_wall
    return plate.right_wall, plate.left_wall


def _uncertified_association(plate: Plate) -> MeshInterfaceAttachment:
    # The left carrier's association is not evidence of the right carrier.
    return MeshInterfaceAttachment(
        plate.right.part,
        plate.right.wall,
        plate.left.edges,
        (plate.wall_entity,),
        tolerance=_TOLERANCE,
        region=plate.right.region,
    )


def _other_revision(plate: Plate) -> InterfaceBinding:
    # Same topology, same wall geometry, same entity index: only the revision differs.
    other = _plate_projection(1.5)
    entity = _entity_at(other, 1, (0.0, 0.5))
    assert entity.rsplit(":", 2)[1:] == plate.wall_entity.rsplit(":", 2)[1:]
    assert other.source_revision != plate.projection.source_revision
    source = InterfaceSource(other.source_id, other.source_revision, (entity,))
    minus, plus = _oriented(plate)
    return InterfaceBinding(
        "wall",
        source,
        "two-sided",
        (
            InterfaceEndpoint("sem", minus, fields={}),
            InterfaceEndpoint("vem", plus, fields={}),
        ),
    )


def _reversed(plate: Plate) -> InterfaceBinding:
    minus, plus = _oriented(plate)
    return plate.binding(plus, minus)


def _unsided(plate: Plate) -> InterfaceBinding:
    minus, _ = _oriented(plate)
    right = plate.right
    unsided = MeshInterfaceAttachment(
        right.part, right.wall, right.edges, (plate.wall_entity,), tolerance=_TOLERANCE
    )
    return plate.binding(minus, unsided)


def _same_side(plate: Plate) -> InterfaceBinding:
    carrier = plate.left.part.carrier
    assert isinstance(carrier, phx.meshing.CellMeshingResult)
    twin = MeshPart("left-twin", carrier)
    wall = MeshInterfaceAttachment(
        twin,
        plate.left.wall,
        plate.left.edges,
        (plate.wall_entity,),
        tolerance=_TOLERANCE,
        region=plate.left.region,
    )
    return plate.binding(plate.left_wall, wall)


def _foreign_patch(plate: Plate) -> MeshInterfaceAttachment:
    return MeshInterfaceAttachment(
        plate.right.part,
        plate.left.wall,
        plate.right.edges,
        (plate.wall_entity,),
        tolerance=_TOLERANCE,
    )


def _wrong_entities(plate: Plate) -> MeshInterfaceAttachment:
    return MeshInterfaceAttachment(
        plate.right.part,
        plate.right.wall,
        plate.right.edges,
        (plate.left_face,),
        tolerance=_TOLERANCE,
    )


def _wrong_entity_set(plate: Plate) -> MeshInterfaceAttachment:
    # Same part, same patch, but a cell association: equal shapes are no witness.
    return MeshInterfaceAttachment(
        plate.right.part,
        plate.right.wall,
        plate.right.cells,
        (plate.wall_entity,),
        tolerance=_TOLERANCE,
    )


def _junction_of_two(plate: Plate) -> InterfaceBinding:
    minus, plus = _oriented(plate)
    return InterfaceBinding(
        "wall",
        plate.source,
        "junction",
        (
            InterfaceEndpoint("sem", minus, fields={}),
            InterfaceEndpoint("vem", plus, fields={}),
        ),
    )


def _sided_overlap(plate: Plate) -> InterfaceBinding:
    minus, plus = _oriented(plate)
    return InterfaceBinding(
        "wall",
        plate.source,
        "overlap",
        (
            InterfaceEndpoint("sem", minus, fields={}),
            InterfaceEndpoint("vem", plus, fields={}),
        ),
    )


@dataclass(frozen=True, slots=True)
class RefusalCase:
    """One declaration that must be refused, with the owner of the refusal."""

    build: Callable[[Plate], object]
    error: type[Exception]
    match: str


_REFUSALS = {
    "reversed-normals": RefusalCase(_reversed, ValueError, "normals are reversed"),
    "unsided-two-sided": RefusalCase(_unsided, ValueError, "require sided endpoints"),
    "same-side": RefusalCase(_same_side, ValueError, "same side"),
    "other-source-revision": RefusalCase(_other_revision, ValueError, "source revision"),
    "uncertified-witness": RefusalCase(
        _uncertified_association, ValueError, "not certified evidence"
    ),
    "foreign-patch": RefusalCase(_foreign_patch, ValueError, "not organization evidence"),
    "undeclared-geometry": RefusalCase(
        _wrong_entities, ValueError, "outside the declared"
    ),
    "shape-without-witness": RefusalCase(
        _wrong_entity_set, ValueError, "another entity set"
    ),
    "junction-of-two": RefusalCase(_junction_of_two, ValueError, "at least three"),
    "sided-overlap": RefusalCase(_sided_overlap, ValueError, "does not acquire a normal"),
}


@pytest.mark.parametrize("case", tuple(_REFUSALS.values()), ids=tuple(_REFUSALS))
def test_interface_declarations_without_exact_witness_are_refused(
    plate: Plate, case: RefusalCase
) -> None:
    with pytest.raises(case.error, match=case.match):
        case.build(plate)


def test_moved_endpoint_invalidates_binding_but_not_interface_identity(
    plate: Plate,
) -> None:
    binding = plate.binding(*_oriented(plate))
    refined = _plate_side(plate.projection, "right", (0.0, 1.0), 5, plate.wall_entity)
    with pytest.raises(ValueError, match="stale"):
        binding.require_current(plate.left.part, refined.part)
    with pytest.raises(ValueError, match="No owner"):
        binding.require_current(plate.left.part)
    carrier = plate.right.part.carrier
    assert isinstance(carrier, phx.meshing.CellMeshingResult)
    moved = carrier.mesh.with_coordinates(
        carrier.mesh.coordinates + 0.1, numeric_version="moved"
    )
    moved_part = MeshPart("right", phx.meshing.certify_cell_mesh(moved, _CONTRACT))
    with pytest.raises(ValueError, match="stale"):
        binding.require_current(plate.left.part, moved_part)
    rebound = InterfaceBinding(
        "wall",
        plate.source,
        "two-sided",
        tuple(
            InterfaceEndpoint(
                endpoint.role,
                endpoint.attachment
                if endpoint.attachment is not plate.right_wall
                else _wall(refined, plate.wall_entity),
                fields=dict(endpoint.fields),
            )
            for endpoint in binding.endpoints
        ),
    )
    rebound.require_current(plate.left.part, refined.part)
    assert rebound.interface_id == binding.interface_id
    assert rebound.binding_id != binding.binding_id


def test_ownership_redistribution_keeps_the_bound_interface(plate: Plate) -> None:
    binding = plate.binding(*_oriented(plate))
    policy = phx.meshing.MeshPartitionPolicy(phx.meshing.MeshPartitionKind.PROVIDER, 2)
    cells = np.asarray(plate.left.region.scope.entity_ids).size
    first = phx.meshing.prepare_mesh_distribution(
        plate.left.part, policy=policy, ownership=np.arange(cells) % 2
    )
    second = phx.meshing.prepare_mesh_distribution(
        plate.left.part, policy=policy, ownership=(np.arange(cells) // (cells // 2)) % 2
    )
    assert first.distribution_id != second.distribution_id
    first.require_current(plate.left.part)
    second.require_current(plate.left.part)
    binding.require_current(plate.left.part, plate.right.part)
    assert plate.binding(*_oriented(plate)).binding_id == binding.binding_id
    assert binding.interface_id == "wall"


def test_overlap_is_unsided_and_embedded_incidence_declares_its_meaning(
    plate: Plate,
) -> None:
    source = InterfaceSource(
        plate.projection.source_id,
        plate.projection.source_revision,
        (plate.left_face, plate.wall_entity),
    )
    left = plate.left
    coarse = MeshInterfaceAttachment(
        left.part, left.region, left.cells, (plate.left_face,), tolerance=_TOLERANCE
    )
    fine_side = _plate_side(
        plate.projection, "left-fine", (-1.0, 0.0), 5, plate.wall_entity
    )
    fine = MeshInterfaceAttachment(
        fine_side.part,
        fine_side.region,
        fine_side.cells,
        (plate.left_face,),
        tolerance=_TOLERANCE,
    )
    overlap = InterfaceBinding(
        "left-overlap",
        source,
        "overlap",
        (
            InterfaceEndpoint("fine", fine, fields={}),
            InterfaceEndpoint("coarse", coarse, fields={}),
        ),
    )
    assert overlap.roles == ("coarse", "fine")
    with pytest.raises(ValueError, match="minus and plus sides"):
        overlap.side_of("fine")
    trace = MeshInterfaceAttachment(
        left.part, left.wall, left.edges, (plate.wall_entity,), tolerance=_TOLERANCE
    )
    endpoints = (
        InterfaceEndpoint("host", coarse, fields={"value": "u"}),
        InterfaceEndpoint("line", trace, fields={"value": "lambda"}),
    )
    with pytest.raises(ValueError, match="traced, averaged, or integrated"):
        InterfaceBinding("inclusion", source, "embedded", endpoints)
    with pytest.raises(ValueError, match="identity of its measure"):
        InterfaceBinding("inclusion", source, "embedded", endpoints, meaning="average")
    with pytest.raises(ValueError, match="higher codimension"):
        InterfaceBinding(
            "inclusion", source, "embedded", endpoints[::-1], meaning="trace"
        )
    embedded = InterfaceBinding(
        "inclusion",
        source,
        "embedded",
        endpoints,
        meaning="average",
        measure_id="wall-arc-length",
    )
    assert embedded.roles == ("host", "line")
    assert (embedded.meaning, embedded.measure_id) == ("average", "wall-arc-length")


def test_multiregion_sheets_bind_two_sided_walls_and_ordered_junctions() -> None:
    seed = seed_double_bubble(1.0, 0.8, ring_points=12)
    topology = seed.topology(seed.capacity_plan(resource_id="interface", headroom=1.25))
    state = seed.state(topology)
    prepared = PreparedMultiRegionSurface(topology, state)
    views = multiregion_sheet_views(prepared, state)

    wall = ("bubble-1", "bubble-2")
    source = InterfaceSource.sheet_views(views, (wall,))
    minus = SheetViewAttachment(views, wall, side=InterfaceSide.MINUS)
    plus = SheetViewAttachment(views, wall, side=InterfaceSide.PLUS)
    binding = InterfaceBinding(
        "film",
        source,
        "two-sided",
        (
            InterfaceEndpoint("inner", minus, fields={}),
            InterfaceEndpoint("outer", plus, fields={}),
        ),
    )
    assert binding.side(InterfaceSide.MINUS).attachment is minus
    assert (minus.region_id, plus.region_id) == wall
    with pytest.raises(ValueError, match="normals are reversed"):
        InterfaceBinding(
            "film",
            source,
            "two-sided",
            (
                InterfaceEndpoint("outer", plus, fields={}),
                InterfaceEndpoint("inner", minus, fields={}),
            ),
        )
    pairs = (("bubble-1", "ambient"), ("bubble-2", "ambient"), wall)
    junction = InterfaceBinding(
        "plateau-border",
        InterfaceSource.sheet_views(views, pairs),
        "junction",
        tuple(
            InterfaceEndpoint(
                f"sheet-{index}", SheetViewAttachment(views, pair), fields={}
            )
            for index, pair in enumerate(pairs)
        ),
    )
    assert junction.roles == ("sheet-0", "sheet-1", "sheet-2")
    with pytest.raises(ValueError, match="minus and plus sides"):
        junction.side_of("sheet-0")
    junction.require_current(views)
    moved = multiregion_sheet_views(
        prepared, state.with_positions(state.positions * 1.01)
    )
    assert moved.views_id == views.views_id
    with pytest.raises(ValueError, match="stale"):
        junction.require_current(moved)


def test_analytic_paired_support_remains_its_own_authority() -> None:
    domain = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    points = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(1))
    left = PairedSupportAttachment(
        cover, pairing.pairing_id, pairing.left_patch_id, points
    )
    right = PairedSupportAttachment(
        cover, pairing.pairing_id, pairing.right_patch_id, points
    )
    binding = InterfaceBinding(
        "cut",
        InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        (
            InterfaceEndpoint("left", left, fields={}),
            InterfaceEndpoint("right", right, fields={}),
        ),
    )
    assert (left.side, right.side) == ("left", "right")
    assert left.evidence.verified and left.evidence.pairing_id == pairing.pairing_id
    binding.require_current(cover)
    moved = phx.domain.cartesian_subdomain_cover(
        phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.5, 1.0])),
        "x",
        (2, 1),
        cover_id="plate",
    )
    assert moved.pairing_ids == cover.pairing_ids
    with pytest.raises(ValueError, match="stale"):
        binding.require_current(moved)
    with pytest.raises(ValueError, match="source revision"):
        InterfaceBinding(
            "cut",
            InterfaceSource.paired_support(moved, pairing.pairing_id),
            "two-sided",
            binding.endpoints,
        )


def test_flipped_paired_support_normal_is_refused_at_attachment() -> None:
    """The cut ``x = 1`` of ``[0, 2] x [0, 1]`` separates the left patch
    ``x <= 1`` from the right ``x >= 1``; its normal out of the left patch is
    ``+e_x``. A pairing declaring ``-e_x`` is refused on either side."""
    domain = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", (2, 1), cover_id="plate")
    (pairing,) = cover.pairings
    points = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(1))
    assert pairing.normal is not None
    np.testing.assert_array_equal(
        np.asarray(pairing.normal(points).data), np.tile([1.0, 0.0], (8, 1))
    )
    flipped = phx.domain.SubdomainCover(
        cover.ambient,
        cover.patches,
        (
            phx.domain.PairedSupport(
                pairing.component,
                dict(pairing.left_coordinates),
                dict(pairing.right_coordinates),
                pairing_id=pairing.pairing_id,
                left_patch_id=pairing.left_patch_id,
                right_patch_id=pairing.right_patch_id,
                normal=-pairing.normal,
                topology=pairing.topology,
            ),
        ),
        cover_id="flipped",
    )
    for patch in (pairing.left_patch_id, pairing.right_patch_id):
        PairedSupportAttachment(cover, pairing.pairing_id, patch, points)
        with pytest.raises(ValueError, match="does not point out of"):
            PairedSupportAttachment(flipped, pairing.pairing_id, patch, points)


def test_periodic_self_seam_is_refused_as_a_two_sided_physical_interface() -> None:
    domain = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(
        domain, "x", (1, 1), periodic=(True, False), cover_id="ring"
    )
    (seam,) = cover.pairings
    points = seam.component.sample(phx.domain.PointSampling(4), key=jr.key(1))

    assert seam.self_seam
    with pytest.raises(ValueError, match="periodic self-seam"):
        PairedSupportAttachment(cover, seam.pairing_id, seam.left_patch_id, points)


def test_law_sides_must_lie_on_the_attached_mesh_wall(plate: Plate) -> None:
    """Independent geometry: the wall is ``x = 0, 0 <= y <= 1`` and the left
    part (``x <= 0``) has outward normal ``+e_x`` on it."""
    minus, plus = _oriented(plate)
    endpoint = plate.binding(minus, plus).endpoint(
        "sem" if minus is plate.left_wall else "vem"
    )
    owners = (phx.meshing.MeshAssembly((plate.left.part, plate.right.part)),)
    sites = np.asarray([[[0.0, 0.1], [0.0, 0.4]], [[0.0, 0.6], [0.0, 1.0]]])
    outward = np.broadcast_to(np.asarray([1.0, 0.0]), sites.shape)
    endpoint.require_side(owners, sites, outward)
    with pytest.raises(ValueError, match="off the attached support"):
        endpoint.require_side(owners, sites + np.asarray([1.0e-3, 0.0]), outward)
    with pytest.raises(ValueError, match="across the interface"):
        endpoint.require_side(owners, sites, -outward)
    carrier = plate.left.part.carrier
    assert isinstance(carrier, phx.meshing.CellMeshingResult)
    edges = carrier.mesh.entity_set(1)
    rows = np.flatnonzero(
        np.isin(
            np.asarray(edges.entity_ids), np.asarray(plate.left.wall.scope.entity_ids)
        )
    )
    assert rows.size == 2
    with pytest.raises(ValueError, match="misses 1"):
        endpoint.require_side(
            owners, sites[:1], outward[:1], facets=(edges.entity_set_id, rows[:1])
        )
