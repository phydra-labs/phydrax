#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""White-box contracts for quotient ancestry and face-work accounting."""

from fractions import Fraction

import numpy as np
import pytest

from phydrax.geometry.brep._boolean_arrangement import Arc, Cut
from phydrax.geometry.brep._boolean_cells import _FaceCellWork, _quotient_workset
from phydrax.geometry.brep._boolean_core import BRepBooleanFailure
from phydrax.geometry.brep._intersection_curve import CurveTrimSegment
from phydrax.geometry.brep._patches import LineCurve


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
