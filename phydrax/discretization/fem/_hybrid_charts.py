from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from .._coordinate_budget import CoordinateEnclosureBudget
from .._reference_cell import reference_cell_topology
from ._form_basis_types import FormElementFamily


if TYPE_CHECKING:
    from . import _form_elements as forms


def hybrid_kind(family: FormElementFamily) -> str:
    if family == "prism-trimmed":
        return "prism"
    if family == "pyramid-trimmed":
        return "pyramid"
    raise ValueError(f"hybrid_kind.family must be a hybrid form family; got {family!r}")


def entity_kind(family: FormElementFamily, face: tuple[int, ...]) -> str:
    if len(face) == 1:
        return "simplex:0"
    if len(face) == 2:
        return "interval"
    if len(face) == 3:
        return "triangle"
    if len(face) == 4:
        return "quadrilateral"
    return hybrid_kind(family)


def entity_chart(
    family: FormElementFamily, face: tuple[int, ...]
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    topology = reference_cell_topology(hybrid_kind(family))
    vertices = np.asarray(topology.vertices, dtype=np.float64)
    if len(face) == len(vertices):
        return np.zeros(3, dtype=np.float64), np.eye(3, dtype=np.float64)
    origin = vertices[face[0]]
    if len(face) == 4:
        return origin, np.stack(
            (vertices[face[1]] - origin, vertices[face[3]] - origin), axis=1
        )
    return origin, (vertices[list(face[1:])] - origin).T


def trace_basis(
    family: FormElementFamily,
    k: int,
    r: int,
    face: tuple[int, ...],
    *,
    budget: CoordinateEnclosureBudget | None = None,
) -> forms.FormBasis:
    # The form factory consumes this chart leaf; defer only factory construction.
    from . import _form_elements as forms

    kind = entity_kind(family, face)
    n = reference_cell_topology(kind).dimension
    if n == 3:
        return forms.FormBasis(3, k, r, family, "untwisted", budget=budget)
    return forms.FormBasis(
        n,
        k,
        r,
        "tensor-trimmed" if kind == "quadrilateral" else "trimmed",
        "untwisted",
        budget=budget,
    )
