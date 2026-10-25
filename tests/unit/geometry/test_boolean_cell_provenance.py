#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""White-box contracts for quotient ancestry and face-work accounting."""

import math
from collections.abc import Callable
from fractions import Fraction
from typing import Concatenate

import numpy as np
import pytest

from phydrax.geometry._atlas import CurveTrimLoop, TrimDomain
from phydrax.geometry.brep._boolean_arrangement import Arc, Arrangement, Cut, Piece
from phydrax.geometry.brep._boolean_cells import (
    _box_status,
    _exact_loop,
    _FaceCellWork,
    _quotient_workset,
)
from phydrax.geometry.brep._boolean_core import BRepBooleanFailure, BRepBooleanPolicy
from phydrax.geometry.brep._coordinate_request import coordinate_source_request
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment
from phydrax.geometry.brep._patches import CircleCurve, LineCurve
from phydrax.geometry.brep._request_preparations import TrimRequestPreparations


def _arc(token: str, first: str, last: str) -> tuple[Arc, list[Cut]]:
    carrier = LineCurve((0.0, 0.0), (1.0, 0.0))
    arc = Arc(
        token=token,
        owner=(0, 0),
        trim=CurveTrimSegment(carrier, 0.0, 1.0),
        curve=None,
        pcurve=carrier,
        offset=0.0,
        scale=1.0,
        first_node=first,
        last_node=last,
        first_vertex=first,
        last_vertex=last,
        source_edge=None,
        branch=None,
    )
    cuts = [
        Cut(value, value, value, node, node, None, None, np.zeros(3, dtype=np.float64))
        for value, node in ((0.0, first), (1.0, last))
    ]
    return arc, cuts


def test_authored_atom_text_does_not_define_quotient_parent() -> None:
    source, cuts = _arc("source:atom:authored", "a", "b")
    first = _quotient_workset(
        (source,), {source.token: cuts}, {}, [], [], 0, _FaceCellWork(100)
    )
    atom = first.arcs[0]
    assert atom.parent_token == source.token
    assert first.parent_tokens[atom.token] == source.token
    assert atom.pcurve is source.pcurve
    assert atom.trim is source.trim
    second = _quotient_workset(
        first.arcs,
        {atom.token: list(first.cuts[atom.token])},
        {},
        [],
        [],
        0,
        _FaceCellWork(100),
    )
    assert second.arcs[0].parent_token == source.token
    assert second.parent_tokens[second.arcs[0].token] == source.token


def test_overlap_aliases_preserve_charge_order_and_source_lineage() -> None:
    first, first_cuts = _arc("first", "a", "b")
    second, second_cuts = _arc("second", "c", "d")
    charges: list[int] = []
    work = _FaceCellWork(10, charge=charges.append)
    result = _quotient_workset(
        (first, second),
        {first.token: first_cuts, second.token: second_cuts},
        {},
        [(first, second, Fraction(1), Fraction(0))],
        [],
        0,
        work,
    )
    assert result.aliases == {"c": "a", "d": "b"}
    assert len(result.arcs) == 1
    assert tuple(result.quotient_sources.values()) == ((result.arcs[0].token, "second"),)
    assert work.used == 10
    assert charges == [1] * 10
    assert not result.unresolved


def test_overlap_refusal_does_not_charge_unadmitted_work() -> None:
    first, first_cuts = _arc("first", "a", "b")
    second, second_cuts = _arc("second", "c", "d")
    charges: list[int] = []
    work = _FaceCellWork(9, charge=charges.append)
    with pytest.raises(BRepBooleanFailure, match=r"FaceCellWork\.maximum"):
        _quotient_workset(
            (first, second),
            {first.token: first_cuts, second.token: second_cuts},
            {},
            [(first, second, Fraction(1), Fraction(0))],
            [],
            0,
            work,
        )
    assert work.used == 9
    assert charges == [1] * 9


def test_face_loop_and_classification_share_the_source_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    carrier = CircleCurve((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), 1.0)
    trim = CurveTrimSegment(carrier, 0.0, 2 * math.pi)
    arc = Arc(
        token="circle",
        owner=(0, 0),
        trim=trim,
        curve=None,
        pcurve=carrier,
        offset=0.0,
        scale=2 * math.pi,
        first_node="seam",
        last_node="seam",
        first_vertex="seam",
        last_vertex="seam",
        source_edge=None,
        branch=None,
    )
    cuts = tuple(
        Cut(value, value, value, "seam", "seam", None, None, np.zeros(3))
        for value in (0.0, 1.0)
    )
    state = Arrangement(
        (), "intersection", BRepBooleanPolicy(trim_resolution=0.25), ()
    )
    request = TrimRequestPreparations()
    observed = []
    with coordinate_source_request((trim,)) as budget:

        def checked_context[S, **P, R](
            original: Callable[Concatenate[S, P], R],
        ) -> Callable[Concatenate[S, P], R]:
            def checked(self: S, /, *args: P.args, **kwargs: P.kwargs) -> R:
                assert kwargs["preparations"] is request
                assert kwargs["budget"] is budget
                observed.append(original)
                return original(self, *args, **kwargs)

            return checked

        for owner, name in (
            (CurveTrimSegment, "__init__"),
            (CurveTrimLoop, "__init__"),
            (TrimDomain, "classify_boxes"),
        ):
            monkeypatch.setattr(owner, name, checked_context(getattr(owner, name)))
        work = _FaceCellWork(256)
        loop = _exact_loop(
            (Piece(arc, cuts[0], cuts[1], 1),),
            state,
            work,
            preparations=request,
            budget=budget,
        )
        assert _box_status(
            TrimDomain(loop),
            np.asarray(((-0.1, -0.1), (0.1, 0.1))),
            work,
            preparations=request,
            budget=budget,
        ) == 1
    assert len(observed) == 3
    assert request.trim_source_arcs
