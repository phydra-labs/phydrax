"""White-box contracts for alias classes and shared trim intersection phases."""

import numpy as np
import pytest

from phydrax.geometry.brep import (
    boolean_brep,
    BRepBooleanFailure,
    BRepBooleanPolicy,
    CurveTrimSegment,
    LineCurve,
    ParametricIntersectionPolicy,
)
from phydrax.geometry.brep._boolean_arrangement import (
    Arc,
    Arrangement,
    intersect_arc_pair,
)
from phydrax.geometry.brep._boolean_core import merge_node_aliases, resolve_node
from phydrax.geometry.brep._coordinate_request import coordinate_source_request


def _arc(token: str, origin: tuple[float, float], direction: tuple[float, float]) -> Arc:
    pcurve = LineCurve(origin, direction)
    return Arc(
        token=token,
        owner=(0, 0),
        trim=CurveTrimSegment(pcurve, 0.0, 1.0),
        curve=None,
        pcurve=pcurve,
        offset=0.0,
        scale=1.0,
        first_node=f"{token}:first",
        last_node=f"{token}:last",
        first_vertex=f"{token}:first",
        last_vertex=f"{token}:last",
        source_edge=None,
        branch=None,
    )


def _exhausted_arrangement() -> Arrangement:
    return Arrangement(
        models=(),
        selection="union",
        policy=BRepBooleanPolicy(
            intersection=ParametricIntersectionPolicy(maximum_boxes=1)
        ),
        queries=(),
        boxes_used=1,
    )


def test_alias_merge_resolves_both_transitive_classes() -> None:
    aliases = {"z": "m", "m": "b", "y": "a"}
    merge_node_aliases(aliases, "z", "y")
    assert resolve_node("z", aliases) == "a"
    assert resolve_node("m", aliases) == "a"
    assert resolve_node("y", aliases) == "a"
    assert resolve_node("unrelated", aliases) == "unrelated"
    before = dict(aliases)
    merge_node_aliases(aliases, "z", "a")
    assert aliases == before


@pytest.mark.parametrize(
    "aliases",
    [
        pytest.param({"node": "node"}, id="self-cycle"),
        pytest.param({"node": "other", "other": "node"}, id="two-node-cycle"),
    ],
)
def test_alias_cycle_refuses_without_rewriting_input(aliases: dict[str, str]) -> None:
    before = dict(aliases)
    with pytest.raises(BRepBooleanFailure, match=r"BRepBoolean\.node_aliases"):
        resolve_node("node", aliases)
    assert aliases == before


def test_disjoint_trim_pair_does_not_refuse_an_exhausted_root_budget() -> None:
    first = _arc("first", (0.0, 0.0), (1.0, 0.0))
    second = _arc("second", (0.0, 2.0), (1.0, 0.0))
    state = _exhausted_arrangement()
    with coordinate_source_request((first.trim, second.trim)) as budget:
        phase = intersect_arc_pair(first, second, state, budget=budget)
    assert phase.result is None
    assert state.boxes_used == 1


def test_overlapping_trim_pair_refuses_before_unbudgeted_root_discovery() -> None:
    first = _arc("first", (0.0, 0.0), (1.0, 0.0))
    second = _arc("second", (0.5, -0.5), (0.0, 1.0))
    state = _exhausted_arrangement()
    with coordinate_source_request((first.trim, second.trim)) as budget:
        with pytest.raises(BRepBooleanFailure, match=r"BRepBoolean\.root_work"):
            intersect_arc_pair(first, second, state, budget=budget)
    assert state.boxes_used == 1


def test_shared_trim_phase_retains_transversal_root_enclosures() -> None:
    first = _arc("first", (0.0, 0.0), (1.0, 0.0))
    second = _arc("second", (0.5, -0.5), (0.0, 1.0))
    policy = ParametricIntersectionPolicy()
    with coordinate_source_request((first.trim, second.trim)) as budget:
        result = intersect_arc_pair(first, second, policy, budget=budget).result
    assert result is not None
    assert result.complete
    assert not result.coincident
    assert len(result.points) == 1
    root = result.points[0]
    assert root.kind == "crossing"
    assert np.all(root.parameter_lower <= 0.5)
    assert np.all(root.parameter_upper >= 0.5)


def test_boolean_nominal_operands_refuse_at_the_public_boundary() -> None:
    with pytest.raises(TypeError, match="first"):
        boolean_brep(True, True, "union")  # ty: ignore[invalid-argument-type]


def test_boolean_policy_nominal_sewing_refuses_at_construction() -> None:
    with pytest.raises(TypeError, match="sewing"):
        BRepBooleanPolicy(sewing="invalid")  # ty: ignore[invalid-argument-type]
