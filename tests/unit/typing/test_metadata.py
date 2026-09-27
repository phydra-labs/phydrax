import enum
from typing import Literal, TypeAlias

import numpy as np
import pytest

import phydrax.typing as pt


class ComponentDim(pt.Dim, minimum=1):
    pass


Basis: TypeAlias = Literal["nodal", "modal"]


class Mode(enum.Enum):
    DENSE = "dense"
    SPARSE = "sparse"


class Flag(enum.IntEnum):
    ON = 1


def test_size_accepts_only_exact_ints_at_or_above_the_minimum() -> None:
    assert pt.parse(3, pt.Size[ComponentDim], "count") == 3
    for wrong_kind in (True, np.int64(3), Flag.ON, 3.0):
        with pytest.raises(TypeError):
            pt.parse(wrong_kind, pt.Size[ComponentDim], "count")
    with pytest.raises(ValueError):
        pt.parse(0, pt.Size[ComponentDim], "count")


def test_identifier_accepts_canonical_strings() -> None:
    assert pt.parse("h2o", pt.Identifier, "name") == "h2o"
    with pytest.raises(TypeError):
        pt.parse(b"h2o", pt.Identifier, "name")
    for invalid in ("", " h2o", "h2o\n"):
        with pytest.raises(ValueError):
            pt.parse(invalid, pt.Identifier, "name")


def test_identifiers_are_unique_canonical_tuples_that_bind_their_dimension() -> None:
    scope = pt.Scope()
    names = ("h2", "o2")

    assert pt.parse(names, pt.Identifiers[ComponentDim], "names", scope=scope) is names
    assert scope.size(ComponentDim) == 2
    with pytest.raises(TypeError):
        pt.parse(["h2", "o2"], pt.Identifiers[ComponentDim], "names")
    with pytest.raises(TypeError):
        pt.parse(("h2", 2), pt.Identifiers[ComponentDim], "names")
    with pytest.raises(ValueError):
        pt.parse(("h2", "h2"), pt.Identifiers[ComponentDim], "names")
    with pytest.raises(ValueError):
        pt.parse((), pt.Identifiers[ComponentDim], "names")


def test_literal_parse_returns_the_declared_literal() -> None:
    parsed = pt.parse(np.str_("nodal"), Basis, "basis")

    assert parsed == "nodal" and type(parsed) is str
    assert pt.parse("modal", Basis, "basis") == "modal"
    with pytest.raises(ValueError):
        pt.parse("spectral", Basis, "basis")
    with pytest.raises(TypeError):
        pt.parse(np.asarray(["nodal"]), Basis, "basis")


def test_literal_parse_prefers_exact_types_before_equality() -> None:
    assert type(pt.parse(True, Literal[1, True], "flag")) is bool
    parsed = pt.parse(True, Literal[1], "flag")
    assert parsed == 1 and type(parsed) is int
    assert pt.parse(Mode.DENSE, Literal["dense"] | Mode, "mode") is Mode.DENSE


def test_enums_are_matched_by_membership_only() -> None:
    assert pt.parse(Mode.SPARSE, Mode, "mode") is Mode.SPARSE
    with pytest.raises(TypeError):
        pt.parse("sparse", Mode, "mode")


def test_optional_and_fixed_tuple_compositions() -> None:
    assert pt.parse(None, pt.Identifier | None, "parent") is None
    pair = pt.parse((2, "h2"), tuple[pt.Size[ComponentDim], pt.Identifier], "pair")
    assert pair == (2, "h2")
    canonical = pt.parse(
        (np.str_("modal"), 2), tuple[Basis, pt.Size[ComponentDim]], "pair"
    )
    assert type(canonical[0]) is str
    with pytest.raises(ValueError):
        pt.parse((2,), tuple[pt.Size[ComponentDim], pt.Identifier], "pair")
