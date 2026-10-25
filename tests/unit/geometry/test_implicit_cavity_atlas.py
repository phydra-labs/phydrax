# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Per-edit chart coverage, chronology, and fail-closed geometry contracts."""
from collections.abc import Callable, Sequence
from dataclasses import replace
from fractions import Fraction as F
from itertools import combinations
from typing import TypedDict

import pytest

from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._exact_polyhedral_geometry import (
    ExactOverlapRefusal,
    star_determinant,
    tetrahedra_disjoint,
    tetrahedron_overlap,
    TetrahedronPlanePreparation,
)
from phydrax.geometry._implicit_cavity_atlas import (
    _point,
    _tet,
    build_implicit_cavity_atlas,
    CavityAtlasRefusal,
    CavityAtlasStatus,
    CavityFacetParent,
    CavitySnapshot,
    CHART_NOTE_PREFIX,
    ImplicitCavityAtlas,
    Point,
    prepare_implicit_cavity_edit,
    SNAPSHOT_NOTE_PREFIX,
    Tet,
)
from tests._support.coordinate_references import coordinate_reference_budget


def _volume_coverage_status(
    actual: Sequence[Tet], virtual: Sequence[Tet], budget: CoordinateEnclosureBudget,
) -> CavityAtlasStatus | None:
    """Independent oracle: exact clipped volumes prove coverage both ways."""
    preparation = TetrahedronPlanePreparation(budget)

    def overlap(first: Tet, second: Tet) -> F:
        if tetrahedra_disjoint(first, second):
            return F(0)
        result = tetrahedron_overlap(first, second, budget=budget, preparation=preparation)
        assert not isinstance(result, ExactOverlapRefusal)
        return result.volume

    for bank in (actual, virtual):
        if any(overlap(first, second) for first, second in combinations(bank, 2)):
            return CavityAtlasStatus.DOUBLE_COVERAGE
    actual_covered = [F(0)] * len(actual)
    virtual_covered = [F(0)] * len(virtual)
    for i, first in enumerate(actual):
        for j, second in enumerate(virtual):
            volume = overlap(first, second)
            actual_covered[i] += volume
            virtual_covered[j] += volume
    for bank, covered in ((actual, actual_covered), (virtual, virtual_covered)):
        for tet, volume in zip(bank, covered, strict=True):
            if volume != star_determinant(tet, budget=budget) / 6:
                return CavityAtlasStatus.COVERAGE_GAP
    return None


def test_typed_cavity_refusal_retains_exact_replay_note() -> None:
    import json

    before, after = _snapshots()
    anchor = (F(2), F(2), F(2))
    with coordinate_reference_budget() as budget, pytest.raises(CavityAtlasRefusal) as first:
        _build(before, after, budget, anchor=anchor)
    assert first.value.status == CavityAtlasStatus.NO_COMMON_KERNEL
    snapshot, chart = first.value.__notes__
    assert snapshot.startswith(SNAPSHOT_NOTE_PREFIX)
    assert chart.startswith(CHART_NOTE_PREFIX)
    payload = {
        **json.loads(snapshot.removeprefix(SNAPSHOT_NOTE_PREFIX)),
        **json.loads(chart.removeprefix(CHART_NOTE_PREFIX)),
    }

    def restore(value: dict) -> CavitySnapshot:
        return CavitySnapshot(
            tuple(_point(tuple(F(x) for x in point)) for point in value["vertices"]),
            tuple(tuple(row) for row in value["cells"]),
            tuple(tuple(row) for row in value["constrained_faces"]),
            tuple(value["source_labels"]), tuple(value["cell_ids"]),
            tuple(tuple(row) for row in value["rim_ancestry"]),
        )

    assert restore(payload["before"]) == before
    assert restore(payload["after"]) == after
    with coordinate_reference_budget() as replay_budget, pytest.raises(CavityAtlasRefusal) as replay:
        assert payload["maximum_work_units"] == replay_budget.maximum_work_units
        assert payload["maximum_memory_bytes"] == replay_budget.maximum_memory_bytes
        edit = prepare_implicit_cavity_edit(
            restore(payload["before"]), restore(payload["after"]),
            source_id=payload["source_id"], discovery_id=payload["discovery_id"],
            chronology_index=payload["chronology_index"],
            predecessor_id=payload["predecessor_id"], budget=replay_budget,
        )
        build_implicit_cavity_atlas(
            edit, chart_axes=tuple(payload["chart_axes"]),
            anchor=tuple(F(x) for x in payload["anchor"]),
            chart_direction=None if payload["chart_direction"] is None else tuple(
                F(x) for x in payload["chart_direction"]),
            budget=replay_budget,
        )
    assert replay.value.status == first.value.status
    assert replay.value.__notes__ == first.value.__notes__


@pytest.mark.parametrize(("duplicate", "status"), [
    (False, CavityAtlasStatus.INVALID_ANCESTRY),
    (True, CavityAtlasStatus.DOUBLE_COVERAGE),
])
def test_actual_disjointness_preparation_matches_exhaustive_exact_work(
    monkeypatch: pytest.MonkeyPatch, duplicate: bool, status: CavityAtlasStatus,
) -> None:
    from phydrax.geometry import _implicit_cavity_atlas as owner
    from phydrax.geometry._exact_polyhedral_geometry import ExactPolyhedralRefusal

    # Consecutive boxes touch exactly. These contacts have zero volume, but a
    # duplicated positive tetrahedron still has to be clipped and refused.
    bank = tuple(_tet(tuple(_point(tuple(F(v) for v in point)) for point in (
        (i, 0, 0), (i + 1, 0, 0), (i, 1, 0), (i, 0, 1),
    ))) for i in range(4))
    actual = (*bank, bank[0]) if duplicate else bank
    cell_ids = tuple(range(1, len(actual) + 1))
    anchor = _point((F(1, 4), F(1, 4), F(1, 4)))
    # A foreign witness is examined only after actual disjointness is proven.
    witness = (_tet((anchor, *bank[0][1:])),)
    foreign = (owner.CavityFacetParent(0, tuple(sorted(bank[0][1:])), 1),)

    def outcome() -> tuple[CavityAtlasStatus, int]:
        with coordinate_reference_budget() as budget:
            with pytest.raises(CavityAtlasRefusal) as failure:
                owner._cover_coned_boundary(actual, cell_ids, witness, foreign, anchor, budget)
            return failure.value.status, budget.work_units

    prepared_status, prepared_work = outcome()

    def exhaustive(
        a: Tet, b: Tet, budget: CoordinateEnclosureBudget,
        preparation: TetrahedronPlanePreparation,
    ) -> F:
        result = tetrahedron_overlap(a, b, budget=budget)
        if isinstance(result, ExactOverlapRefusal):
            raise ExactPolyhedralRefusal(result)
        return result.volume

    monkeypatch.setattr(owner, "_overlap", exhaustive)
    exhaustive_status, exhaustive_work = outcome()
    assert prepared_status == exhaustive_status == status
    assert prepared_work < exhaustive_work


def _snapshots() -> tuple[CavitySnapshot, CavitySnapshot]:
    points = ((F(0), F(0), F(-1)), (F(0), F(0), F(0)),
              (F(1), F(0), F(0)), (F(0), F(1), F(0)))
    rim = ((1, 2, "ab"), (2, 3, "bc"), (3, 1, "ca"))
    before = CavitySnapshot(points, ((0, 1, 2, 3),), ((1, 2, 3),),
                            ("source-face",), (17,), rim)
    after = CavitySnapshot((*points, (F(1, 3), F(1, 3), F(1, 6))),
                           ((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4)),
                           ((1, 2, 4), (2, 3, 4), (3, 1, 4)),
                           ("source-face",) * 3, (23, 24, 25), rim)
    return before, after


class _BuildChanges(TypedDict, total=False):
    chart_axes: tuple[int, int]
    anchor: Point
    chronology_index: int


def _build(
    before: CavitySnapshot, after: CavitySnapshot, budget: CoordinateEnclosureBudget, *,
    chart_axes: tuple[int, int] = (0, 1),
    anchor: Point = (F(1, 4), F(1, 4), F(-1, 4)),
    source_id: str = "original-source", discovery_id: str = "original-discovery",
    chronology_index: int = 0, predecessor_id: str = "",
    chart_direction: Point | None = None,
) -> ImplicitCavityAtlas:
    edit = prepare_implicit_cavity_edit(
        before, after, source_id=source_id, discovery_id=discovery_id,
        chronology_index=chronology_index, predecessor_id=predecessor_id, budget=budget,
    )
    return build_implicit_cavity_atlas(
        edit, chart_axes=chart_axes, anchor=anchor, budget=budget,
        chart_direction=chart_direction,
    )


@pytest.mark.parametrize("collapse", [False, True])
def test_projected_insertion_and_collapse_cover_both_actual_cavities(collapse: bool) -> None:
    before, after = _snapshots()
    if collapse:
        before, after = after, before
    with coordinate_reference_budget() as budget:
        atlas = _build(before, after, budget)
        assert atlas.source_cell_ids == before.cell_ids
        assert atlas.target_cell_ids == after.cell_ids
        assert len(atlas.old_tets) == len(atlas.new_tets)
        for bank, snapshot in ((atlas.old_tets, before), (atlas.new_tets, after)):
            determinants = tuple(star_determinant(t, budget=budget) for t in bank)
            assert all(d > 0 for d in determinants)
            assert sum(determinants) == sum(star_determinant(
                _tet(tuple(snapshot.vertices[i] for i in row)), budget=budget)
                for row in snapshot.cells)
        assert atlas.old_tets != atlas.new_tets


def test_chart_identity_binds_chronology_and_original_discovery() -> None:
    before, after = _snapshots()
    with coordinate_reference_budget() as budget:
        first = _build(before, after, budget)
        replay = _build(before, after, budget)
        later = _build(before, after, budget, chronology_index=1, predecessor_id=first.chart_id)
        other = _build(before, after, budget, discovery_id="different-discovery")
    assert first.chart_id == replay.chart_id
    assert len({first.chart_id, later.chart_id, other.chart_id}) == 3


@pytest.mark.parametrize("changes,status", [
    ({"chart_axes": (0, 2)}, CavityAtlasStatus.NOT_A_GRAPH),
    ({"anchor": (F(0), F(0), F(-1))}, CavityAtlasStatus.NO_COMMON_KERNEL),
    ({"anchor": (F(2), F(2), F(2))}, CavityAtlasStatus.NO_COMMON_KERNEL),
    ({"chronology_index": 1}, CavityAtlasStatus.INVALID_ANCESTRY),
], ids=("vertical-chart", "boundary-anchor", "exterior-anchor", "wrong-epoch"))
def test_unsupported_chart_and_kernel_are_typed(changes: _BuildChanges, status: CavityAtlasStatus) -> None:
    before, after = _snapshots()
    with coordinate_reference_budget() as budget, pytest.raises(CavityAtlasRefusal) as error:
        _build(before, after, budget, **changes)
    assert error.value.status == status
    assert error.value.recoverable == (status in (
        CavityAtlasStatus.NOT_A_GRAPH, CavityAtlasStatus.NO_COMMON_KERNEL))


@pytest.mark.parametrize("mutation,status", [
    (lambda s: replace(s, source_labels=("foreign",) * 3), CavityAtlasStatus.INVALID_ANCESTRY),
    (lambda s: replace(s, rim_ancestry=s.rim_ancestry[:-1]), CavityAtlasStatus.INVALID_ANCESTRY),
    (lambda s: replace(s, rim_ancestry=((1, 2, "wrong"), *s.rim_ancestry[1:])), CavityAtlasStatus.RIM_MISMATCH),
    (lambda s: replace(s, cells=(s.cells[0],) * 3), CavityAtlasStatus.INVALID_CAVITY),
    (lambda s: replace(s, constrained_faces=(s.constrained_faces[0],) * 3), CavityAtlasStatus.INVALID_CAVITY),
], ids=("foreign-source", "missing-rim", "changed-rim", "duplicate-cell", "duplicate-face"))
def test_malformed_snapshots_never_publish_an_atlas(
    mutation: Callable[[CavitySnapshot], CavitySnapshot], status: CavityAtlasStatus,
) -> None:
    before, after = _snapshots()
    with coordinate_reference_budget() as budget, pytest.raises(CavityAtlasRefusal) as error:
        _build(before, mutation(after), budget)
    assert error.value.status == status
    assert not error.value.recoverable


def test_oblique_insertion_preserves_nonbinary_original_edge_preimage() -> None:
    points = tuple(_point(tuple(F(x) for x in p)) for p in (
        (0, 0, -1), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    ))
    rim = ((1, 2, "ab"), (2, 3, "bc"), (3, 4, "cd"), (4, 1, "da"))
    before = CavitySnapshot(
        points, ((0, 1, 2, 3), (0, 1, 3, 4)), ((1, 2, 3), (1, 3, 4)),
        ("patch",) * 2, (11, 12), rim,
    )
    fraction = F(1, 3)
    preimage = _point(tuple((1 - fraction) * a + fraction * b
                           for a, b in zip(points[1], points[3], strict=True)))
    direction = (F(1, 12), F(-1, 24), F(1, 6))
    inserted = _point(tuple(a + d for a, d in zip(preimage, direction, strict=True)))
    after = CavitySnapshot(
        (*points, inserted),
        ((0, 1, 2, 5), (0, 2, 3, 5), (0, 1, 5, 4), (0, 5, 3, 4)),
        ((1, 2, 5), (2, 3, 5), (1, 5, 4), (5, 3, 4)),
        ("patch",) * 4, (21, 22, 23, 24), rim,
    )
    with coordinate_reference_budget() as budget:
        chart = _build(before, after, budget, chart_direction=direction)
        axis_chart = _build(before, after, budget)
    preimages = {a for old, new in zip(chart.old_tets, chart.new_tets, strict=True)
                 for a, b in zip(old, new, strict=True) if b == inserted}
    assert preimages == {preimage}
    assert chart.chart_direction == direction
    assert chart.chart_id != axis_chart.chart_id
    assert {a for old, new in zip(axis_chart.old_tets, axis_chart.new_tets, strict=True)
            for a, b in zip(old, new, strict=True) if b == inserted} != {preimage}


def _oblique_insertion() -> tuple[CavitySnapshot, CavitySnapshot, Point, Point]:
    """A two-face patch whose oblique preimage lies on the shared diagonal."""
    points = tuple(_point(tuple(F(x) for x in p)) for p in (
        (0, 0, -1), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    ))
    rim = ((1, 2, "ab"), (2, 3, "bc"), (3, 4, "cd"), (4, 1, "da"))
    before = CavitySnapshot(
        points, ((0, 1, 2, 3), (0, 1, 3, 4)), ((1, 2, 3), (1, 3, 4)),
        ("patch",) * 2, (11, 12), rim,
    )
    preimage = _point(tuple(F(2, 3) * a + F(1, 3) * b for a, b in zip(points[1], points[3], strict=True)))
    direction = _point((F(1, 12), F(-1, 24), F(1, 6)))
    after = CavitySnapshot(
        (*points, _point(tuple(a + d for a, d in zip(preimage, direction, strict=True)))),
        ((0, 1, 2, 5), (0, 2, 3, 5), (0, 1, 5, 4), (0, 5, 3, 4)),
        ((1, 2, 5), (2, 3, 5), (1, 5, 4), (5, 3, 4)),
        ("patch",) * 4, (21, 22, 23, 24), rim,
    )
    return before, after, preimage, direction


def _affine(origin: Tet, image: Tet) -> tuple[tuple[tuple[F, ...], ...], tuple[F, ...]]:
    """Independent exact affine map ``x = L y + t`` taking ``origin`` onto ``image``."""
    a = [[origin[k + 1][r] - origin[0][r] for k in range(3)] for r in range(3)]
    b = [[image[k + 1][r] - image[0][r] for k in range(3)] for r in range(3)]
    det = (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )
    inverse = [[(
        a[(c + 1) % 3][(r + 1) % 3] * a[(c + 2) % 3][(r + 2) % 3]
        - a[(c + 1) % 3][(r + 2) % 3] * a[(c + 2) % 3][(r + 1) % 3]
    ) / det for c in range(3)] for r in range(3)]
    linear = tuple(tuple(sum((b[r][m] * inverse[m][c] for m in range(3)), F(0)) for c in range(3)) for r in range(3))
    offset = tuple(image[0][r] - sum((linear[r][c] * origin[0][c] for c in range(3)), F(0)) for r in range(3))
    return linear, offset


@pytest.mark.parametrize("oblique", [False, True], ids=("axis-chart", "oblique-collinear"))
def test_overlay_cones_share_one_exact_map_and_keep_every_overlay_vertex(oblique: bool) -> None:
    from phydrax.geometry._implicit_cavity_atlas import _chart_project
    from phydrax.geometry._planar_primitives import intersection_polygon

    before, after, preimage, direction = _oblique_insertion()
    chart_direction = direction if oblique else None
    with coordinate_reference_budget() as budget:
        atlas = _build(before, after, budget, chart_direction=chart_direction)
        # Independent exact clipping proves both cone banks partition their cavities.
        for snapshot, virtual in ((before, atlas.old_tets), (after, atlas.new_tets)):
            actual = tuple(_tet(tuple(snapshot.vertices[i] for i in row)) for row in snapshot.cells)
            assert _volume_coverage_status(actual, virtual, budget) is None
        groups: dict[tuple[CavityFacetParent, CavityFacetParent], list[tuple[Tet, Tet]]] = {}
        for old, new, source, target in zip(
            atlas.old_tets, atlas.new_tets,
            atlas.source_boundary_parents, atlas.target_boundary_parents, strict=True,
        ):
            groups.setdefault((source, target), []).append((old, new))
        subdivided = set()
        for (source, target), cones in groups.items():
            # One exact affine map interpolates every cone of the face pair: the
            # first cone's map sends every old corner to its paired new corner.
            linear, offset = _affine(*cones[0])
            for old, new in cones:
                assert old[0] == new[0] == atlas.anchor
                for a, b in zip(old, new, strict=True):
                    assert tuple(
                        sum((linear[r][c] * a[c] for c in range(3)), offset[r]) for r in range(3)
                    ) == b
            if source.vertices == target.vertices:
                # An unchanged facet has the identity map on every cone.
                assert all(old == new for old, new in cones)
                continue
            # Every original overlay vertex, including collinear subdivisions,
            # remains represented in the cone bases.
            polygon = intersection_polygon(
                _chart_project(source.vertices, atlas.chart_axes, chart_direction, budget),
                _chart_project(target.vertices, atlas.chart_axes, chart_direction, budget),
                budget=budget,
            )
            bases = {
                vertex
                for old, _ in cones
                for vertex in _chart_project(old[1:], atlas.chart_axes, chart_direction, budget)
            }
            assert set(polygon) <= bases
            if any(preimage in old[1:] for old, _ in cones):
                subdivided.add(source.vertices)
    if oblique:
        # The preimage subdivides the shared diagonal inside both source facets.
        assert len(subdivided) == 2


def test_oblique_chart_refuses_zero_dropped_direction_component() -> None:
    before, after = _snapshots()
    with coordinate_reference_budget() as budget, pytest.raises(CavityAtlasRefusal) as error:
        _build(before, after, budget, chart_direction=(F(1), F(0), F(0)))
    assert error.value.status == CavityAtlasStatus.NOT_A_GRAPH


def test_native_resource_refusal_retains_prepaid_cavity_snapshot() -> None:
    import json
    import math

    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionLimits,
        NativeExecutionScope,
    )
    from phydrax.geometry._implicit_cavity_atlas import _cavity_chart_note

    before, after = _snapshots()
    anchor = (F(1, 4), F(1, 4), F(-1, 4))
    with coordinate_reference_budget() as measured:
        edit = prepare_implicit_cavity_edit(
            before, after, source_id="original-source", discovery_id="original-discovery",
            chronology_index=0, predecessor_id="", budget=measured,
        )
        chart = _cavity_chart_note(
            chart_axes=(0, 1), anchor=anchor, chart_direction=None, budget=measured,
        )
        diagnostic_work = measured.work_units
        maximum_work = measured.maximum_work_units
        maximum_bytes = measured.maximum_memory_bytes
    # The root has only one unit after the real diagnostic preparation. The
    # coordinate policy remains unchanged, so refusal comes from the native
    # remaining allowance, as in the recorded public torus failure.
    limits = NativeExecutionLimits(
        diagnostic_work + 1, 0, 0, maximum_bytes, math.inf
    )
    with pytest.raises(MeshcoreError) as failure:
        with NativeExecutionScope(limits) as scope:
            with scope.batch(diagnostic_work + 1, bytes_upper=maximum_bytes) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=maximum_work,
                    maximum_memory_bytes=maximum_bytes,
                )
                with budget.activate():
                    _build(before, after, budget)
    assert failure.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    notes = failure.value.__notes__
    assert notes.index(chart) == notes.index(edit.note) + 1
    payload = {
        **json.loads(edit.note.removeprefix(SNAPSHOT_NOTE_PREFIX)),
        **json.loads(chart.removeprefix(CHART_NOTE_PREFIX)),
    }
    assert payload["before"]["vertices"] == [
        [str(value) for value in point] for point in before.vertices
    ]
    assert payload["after"]["cells"] == [list(cell) for cell in after.cells]
    with coordinate_reference_budget() as replay_budget:
        replay = _build(
            before, after, replay_budget,
            chart_axes=(int(payload["chart_axes"][0]), int(payload["chart_axes"][1])),
            anchor=_point(tuple(F(value) for value in payload["anchor"])),
        )
    assert replay.source_id == payload["source_id"]
    assert replay.discovery_id == payload["discovery_id"]


@pytest.mark.parametrize("side", ("source", "target"))
@pytest.mark.parametrize(
    ("mutation", "status"),
    (
        ("none", None),
        ("gap", CavityAtlasStatus.COVERAGE_GAP),
        ("overlap", CavityAtlasStatus.DOUBLE_COVERAGE),
    ),
)
def test_coned_boundary_chain_matches_independent_volume_coverage(
    side: str, mutation: str, status: CavityAtlasStatus | None,
) -> None:
    from phydrax.geometry import _implicit_cavity_atlas as owner

    before, after = _snapshots()
    with coordinate_reference_budget() as preparation:
        atlas = _build(before, after, preparation)
    snapshot = before if side == "source" else after
    actual = tuple(_tet(tuple(snapshot.vertices[index] for index in cell))
                   for cell in snapshot.cells)
    virtual = atlas.old_tets if side == "source" else atlas.new_tets
    parents = atlas.source_boundary_parents if side == "source" else atlas.target_boundary_parents
    if mutation == "gap":
        virtual, parents = virtual[:-1], parents[:-1]
    elif mutation == "overlap":
        virtual, parents = (*virtual, virtual[0]), (*parents, parents[0])
    with coordinate_reference_budget() as chain_budget:
        if status is None:
            owner._cover_coned_boundary(
                actual, snapshot.cell_ids, virtual, parents, atlas.anchor, chain_budget,
            )
        else:
            with pytest.raises(CavityAtlasRefusal) as chain_failure:
                owner._cover_coned_boundary(
                    actual, snapshot.cell_ids, virtual, parents, atlas.anchor, chain_budget,
                )
            assert chain_failure.value.status == status
        chain_work = chain_budget.work_units
    with coordinate_reference_budget() as volume_budget:
        assert _volume_coverage_status(actual, virtual, volume_budget) == status
        if status is None:
            assert chain_work < volume_budget.work_units


@pytest.mark.parametrize(
    ("mutation", "status"),
    (
        ("missing-parent", CavityAtlasStatus.INVALID_ANCESTRY),
        ("foreign-parent", CavityAtlasStatus.INVALID_ANCESTRY),
        ("parent-orientation", CavityAtlasStatus.INVALID_ANCESTRY),
        ("cone-orientation", CavityAtlasStatus.INVALID_CAVITY),
        ("different-anchor", CavityAtlasStatus.INVALID_CAVITY),
    ),
)
def test_coned_boundary_chain_rejects_unbound_or_reversed_witnesses(
    mutation: str, status: CavityAtlasStatus,
) -> None:
    from phydrax.geometry import _implicit_cavity_atlas as owner

    before, after = _snapshots()
    with coordinate_reference_budget() as preparation:
        atlas = _build(before, after, preparation)
    actual = tuple(_tet(tuple(before.vertices[index] for index in cell))
                   for cell in before.cells)
    virtual, parents, anchor = atlas.old_tets, atlas.source_boundary_parents, atlas.anchor
    if mutation == "missing-parent":
        parents = parents[:-1]
    elif mutation == "foreign-parent":
        parents = (replace(parents[0], cell_id=999), *parents[1:])
    elif mutation == "parent-orientation":
        parents = (replace(parents[0], orientation=-parents[0].orientation), *parents[1:])
    elif mutation == "cone-orientation":
        first = virtual[0]
        virtual = ((first[0], first[1], first[3], first[2]), *virtual[1:])
    else:
        anchor = (anchor[0], anchor[1], anchor[2] - F(1, 16))
    with coordinate_reference_budget() as budget, pytest.raises(CavityAtlasRefusal) as failure:
        owner._cover_coned_boundary(
            actual, before.cell_ids, virtual, parents, anchor, budget,
        )
    assert failure.value.status == status


def test_actual_cavity_signed_interior_faces_must_cancel() -> None:
    from phydrax.geometry import _implicit_cavity_atlas as owner

    shared = ((F(0), F(0), F(0)), (F(1), F(0), F(0)), (F(0), F(1), F(0)))
    actual = (
        _tet((*shared, (F(0), F(0), F(1)))),
        _tet((*shared, (F(0), F(0), F(2)))),
    )
    with coordinate_reference_budget() as budget:
        assert all(star_determinant(tet, budget=budget) > 0 for tet in actual)
        with pytest.raises(CavityAtlasRefusal) as failure:
            owner._oriented_cavity_boundary(actual, (1, 2), budget)
    assert failure.value.status is CavityAtlasStatus.INVALID_CAVITY
    assert "opposite signs" in str(failure.value)
