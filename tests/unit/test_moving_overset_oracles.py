from fractions import Fraction

import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.geometry._overset_interfaces import AtomicEdge, EdgeOwner
from tools.meshing_qualification.scenarios.moving_overset_matrix import (
    _annulus_region,
    _boundary_flux_oracle,
    _entry_with_value,
    _native_part,
    _part_source_measures,
    _PROFILES,
    _require_refinement_convergence,
    _resume_cold_profile,
    _run_cold_profile,
    _source_center,
)


def _boundary(
    points: tuple[tuple[Fraction, Fraction], ...],
) -> list[tuple[AtomicEdge, float]]:
    result = []
    for start, end in zip(points, points[1:] + points[:1], strict=True):
        dx, dy = end[0] - start[0], end[1] - start[1]
        normal = (dy, -dx)
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        exact = (1 - midpoint[0] / 2) * normal[0] + (1 - midpoint[1] / 2) * normal[1]
        owner = EdgeOwner("part", 0, 0, 1, 0)
        result.append(
            (
                AtomicEdge(
                    start, end, (owner,), "physical_boundary", normal, dx * dx + dy * dy
                ),
                float(exact),
            )
        )
    return result


def test_current_source_center_uses_the_accepted_successor() -> None:
    from phydrax.meshing._motion_source import NativePlanarMotionSource
    from phydrax.meshing.providers._native_sources import NativePlanarSource

    body = _PROFILES[1].bodies[0]
    region = _annulus_region(body)
    source = NativePlanarSource(
        region,
        "accepted-source-revision",
        vertex_entity_ids=np.arange(len(region.vertices), dtype=np.int64),
        edge_entity_ids=np.arange(len(region.edges), dtype=np.int64),
        region_scope_entity_ids=np.asarray((0,), dtype=np.int64),
    )
    successor = NativePlanarMotionSource(source, np.asarray((0.125, -0.375))).successor
    np.testing.assert_allclose(
        _source_center(successor), np.asarray(body.center) + (0.125, -0.375)
    )
    assert not np.array_equal(_source_center(successor), _source_center(source))


def test_grouped_flux_uses_actual_exterior_and_cavity_geometry() -> None:
    f = Fraction
    exterior = ((f(-3), f(-3)), (f(3), f(-3)), (f(3), f(3)), (f(-3), f(3)))
    cavity = (
        (f(-1, 4), f(-1, 4)),
        (f(-1, 4), f(1, 4)),
        (f(1, 4), f(1, 4)),
        (f(1, 4), f(-1, 4)),
    )
    edges = _boundary(exterior) + _boundary(cavity)
    wall = np.asarray(cavity, dtype=np.float64)
    evidence = _boundary_flux_oracle(_PROFILES[0], (wall,), tuple(edges))
    groups = evidence["groups"]
    assert isinstance(groups, dict)
    assert isinstance(groups["exterior"], dict)
    assert isinstance(groups["moving"], dict)
    assert groups["exterior"]["manufactured_outward_flux"] == pytest.approx(-36.0)
    assert groups["moving"]["manufactured_outward_flux"] == pytest.approx(0.25)
    assert evidence["group_flux_error_l2"] == pytest.approx(0.0)
    edge, value = edges[0]
    edges[0] = edge, value + 0.125
    changed = _boundary_flux_oracle(_PROFILES[0], (wall,), tuple(edges))
    assert changed["group_flux_error_l2"] == pytest.approx(0.125)


def _record(l2: float, qoi: float, flux: float) -> dict[str, object]:
    return {
        "epochs": [
            {
                "epoch": epoch,
                "composite_pde": {
                    "independent_physical_oracle": {
                        "manufactured_l2_error": l2,
                        "qoi_absolute_error": qoi,
                    },
                    "independent_boundary_flux_oracle": {"group_flux_error_l2": flux},
                },
            }
            for epoch in (1, 2)
        ]
    }


def test_each_refinement_oracle_is_a_required_gate() -> None:
    original = _record(2.0, 2.0, 2.0)
    assert len(_require_refinement_convergence(original, _record(1.0, 1.0, 1.0))) == 2
    for refined in (
        _record(3.0, 1.0, 1.0),
        _record(1.0, 3.0, 1.0),
        _record(1.0, 1.0, 3.0),
    ):
        with pytest.raises(RuntimeError, match="did not decrease"):
            _require_refinement_convergence(original, refined)


def test_refinement_evidence_serializes_numpy_scalar_comparisons() -> None:
    import json

    comparisons = _require_refinement_convergence(
        _record(2.0, 2.0, 2.0),
        _record(np.float64(1.0), np.float64(1.0), np.float64(1.0)),
    )
    assert (
        json.loads(json.dumps(comparisons))[0]["metrics"]["qoi_absolute_error"][
            "decreased"
        ]
        is True
    )


def test_cold_profile_requires_a_distinct_consumer_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_qualification.scenarios import cold_restore

    replies = iter(
        (
            {"pid": 17, "checkpoint": {"composition_id": "accepted"}},
            {"cold_continuation": {"consumer_pid": 17}},
        )
    )
    monkeypatch.setattr(cold_restore, "_run_child", lambda *args, **kwargs: next(replies))
    with pytest.raises(RuntimeError, match="fresh process"):
        _run_cold_profile(_PROFILES[0])


def test_cold_profile_records_separate_producer_and_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tools.meshing_qualification.scenarios import cold_restore

    calls: list[str] = []

    def child(code: str, /, *, environment: dict[str, str]) -> dict[str, object]:
        calls.append(code)
        if len(calls) == 1:
            return {"pid": 17, "checkpoint": {"composition_id": "accepted"}}
        return {"cold_continuation": {"consumer_pid": 23, "status": "passed"}}

    monkeypatch.setattr(cold_restore, "_run_child", child)
    result = _run_cold_profile(_PROFILES[0])
    assert len(calls) == 2
    assert "_run_profile" in calls[0] and "_resume_cold_profile" in calls[1]
    continuation = result["cold_continuation"]
    assert isinstance(continuation, dict)
    assert continuation["producer_process_exited_before_consumer"]
    assert continuation["producer_pid"] == 17
    assert continuation["consumer_pid"] == 23


def test_cold_resume_rejects_a_changed_producer_receipt() -> None:
    with pytest.raises(ValueError, match="trusted receipt"):
        _resume_cold_profile(
            _PROFILES[0],
            {"profile": "routine", "cold_checkpoint": {"epoch": 0}},
            "unused.phx",
            {"epoch": 1},
        )


def test_checkpoint_projection_preserves_composition_dependencies() -> None:
    from phydrax.lifecycle import Composition, CompositionEntry

    source = CompositionEntry(
        np.asarray([1.0]),
        entry_id="source",
        role="topology",
        owner_id="test",
        structure_id="source-structure",
        revision_id="source-revision",
        semantics_id="source-semantics",
    )
    state = CompositionEntry(
        np.asarray([2.0]),
        entry_id="state",
        role="physical-state",
        owner_id="test",
        structure_id="state-structure",
        revision_id="state-revision",
        semantics_id="state-semantics",
        dependencies=(source.binding("revision"),),
    )
    original = Composition((source, state), boundary_id="accepted")
    projected = Composition(
        (_entry_with_value(source, np.asarray([3.0])), state), boundary_id="accepted"
    )
    assert projected.composition_id == original.composition_id
    assert projected.entry("state").dependencies == state.dependencies


@pytest.mark.skipif(not meshcore_available(), reason="native meshing requires meshcore")
def test_authored_decimal_body_preserves_boundary_and_cavity() -> None:
    body = _PROFILES[1].bodies[0]
    part, source, _, _ = _native_part(body.name, _annulus_region(body), body=body)
    measures = _part_source_measures(part, body, source)
    assert measures["independent_triangle_area"] == pytest.approx(
        4 * (body.half_width**2 - body.cavity_half_width**2)
    )
    assert measures["associated_cavity_wall_length"] == pytest.approx(
        8 * body.cavity_half_width
    )


@pytest.mark.skipif(not meshcore_available(), reason="native meshing requires meshcore")
def test_partition_rejects_a_different_authored_wall_source() -> None:
    import phydrax as phx
    from phydrax.meshing._motion_source import NativePlanarMotionSource
    from tools.meshing_qualification.scenarios.moving_overset_matrix import (
        _source_edge_ids,
    )

    body = _PROFILES[0].bodies[0]
    part, source, _, _ = _native_part(body.name, _annulus_region(body), body=body)
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((-10.0, -10.0), (20.0, -10.0), (-10.0, 20.0))),
        np.asarray(((0, 1, 2),), dtype=np.int32),
    )
    background = phx.meshing.MeshPart(
        "background",
        phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si()),
    )
    specs = (
        phx.meshing.OversetPartSpec("background"),
        phx.meshing.OversetPartSpec(
            body.name,
            wall=part.scope(1, _source_edge_ids(part, source, (4, 5, 6, 7))),
            boundary=part.scope(1, _source_edge_ids(part, source, (0, 1, 2, 3))),
        ),
    )
    assembly = phx.meshing.MeshAssembly((background, part))
    shifted = NativePlanarMotionSource(source, np.asarray((0.125, 0.0))).successor
    with pytest.raises(ValueError, match="source does not bind"):
        phx.meshing.prepare_overset_connectivity(
            assembly,
            specs,
            partition_priority=(body.name, "background"),
            partition_sources={body.name: shifted},
        )
