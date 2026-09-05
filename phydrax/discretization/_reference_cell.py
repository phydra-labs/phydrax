from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp

from .._fingerprint import canonical_fingerprint


FacetShape = Literal["point", "edge", "triangle", "quadrilateral"]


@dataclass(frozen=True)
class FacetOrientationAction:
    shape: FacetShape
    permutation: tuple[int, ...]

    def __post_init__(self):
        sizes = {"point": 1, "edge": 2, "triangle": 3, "quadrilateral": 4}
        size = sizes[self.shape]
        if tuple(sorted(self.permutation)) != tuple(range(size)):
            raise ValueError("Facet orientation permutation is invalid.")

    @property
    def orientation_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "facet-orientation-action",
                "shape": self.shape,
                "permutation": self.permutation,
            }
        )

    @property
    def inverse(self) -> FacetOrientationAction:
        inverse = [0] * len(self.permutation)
        for index, value in enumerate(self.permutation):
            inverse[value] = index
        return FacetOrientationAction(self.shape, tuple(inverse))

    def compose(self, right: FacetOrientationAction, /) -> FacetOrientationAction:
        if self.shape != right.shape:
            raise ValueError("Facet orientation composition requires equal shapes.")
        return FacetOrientationAction(
            self.shape,
            tuple(right.permutation[index] for index in self.permutation),
        )

    def apply(self, values, /, *, axis: int = 0):
        return jnp.take(jnp.asarray(values), jnp.asarray(self.permutation), axis=axis)


def facet_orientation_actions(shape: FacetShape, /) -> tuple[FacetOrientationAction, ...]:
    if shape == "point":
        permutations = ((0,),)
    elif shape == "edge":
        permutations = ((0, 1), (1, 0))
    elif shape == "triangle":
        rotations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
        reflections = ((0, 2, 1), (2, 1, 0), (1, 0, 2))
        permutations = rotations + reflections
    elif shape == "quadrilateral":
        rotations = (
            (0, 1, 2, 3),
            (1, 2, 3, 0),
            (2, 3, 0, 1),
            (3, 0, 1, 2),
        )
        reflections = (
            (0, 3, 2, 1),
            (3, 2, 1, 0),
            (2, 1, 0, 3),
            (1, 0, 3, 2),
        )
        permutations = rotations + reflections
    else:
        raise ValueError("Unknown facet shape.")
    return tuple(FacetOrientationAction(shape, value) for value in permutations)


def facet_orientation_between(
    canonical_vertices: tuple[int, ...],
    local_vertices: tuple[int, ...],
    /,
) -> FacetOrientationAction:
    shape: FacetShape = {
        1: "point",
        2: "edge",
        3: "triangle",
        4: "quadrilateral",
    }[len(canonical_vertices)]
    if set(canonical_vertices) != set(local_vertices):
        raise ValueError("Facet orientations require identical vertex sets.")
    local_positions = tuple(local_vertices.index(value) for value in canonical_vertices)
    for action in facet_orientation_actions(shape):
        if action.permutation == local_positions:
            return action
    raise ValueError("Facet vertex order is not a valid orientation-group action.")


@dataclass(frozen=True)
class ReferenceCellTopology:
    name: str
    dimension: int
    vertices: tuple[tuple[float, ...], ...]
    entities: tuple[tuple[tuple[int, ...], ...], ...]


REFERENCE_TOPOLOGIES = {
    "interval": ReferenceCellTopology(
        "interval", 1, ((0.0,), (1.0,)), (((0,), (1,)), ((0, 1),))
    ),
    "triangle": ReferenceCellTopology(
        "triangle",
        2,
        ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
        (((0,), (1,), (2,)), ((0, 1), (1, 2), (2, 0)), ((0, 1, 2),)),
    ),
    "quadrilateral": ReferenceCellTopology(
        "quadrilateral",
        2,
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        (((0,), (1,), (2,), (3,)), ((0, 1), (1, 2), (2, 3), (3, 0)), ((0, 1, 2, 3),)),
    ),
    "tetrahedron": ReferenceCellTopology(
        "tetrahedron",
        3,
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        (
            ((0,), (1,), (2,), (3,)),
            ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)),
            ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)),
            ((0, 1, 2, 3),),
        ),
    ),
    "prism": ReferenceCellTopology(
        "prism",
        3,
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        (
            ((0,), (1,), (2,), (3,), (4,), (5,)),
            (
                (0, 1),
                (1, 2),
                (2, 0),
                (3, 4),
                (4, 5),
                (5, 3),
                (0, 3),
                (1, 4),
                (2, 5),
            ),
            (
                (0, 2, 1),
                (3, 4, 5),
                (0, 1, 4, 3),
                (1, 2, 5, 4),
                (2, 0, 3, 5),
            ),
            ((0, 1, 2, 3, 4, 5),),
        ),
    ),
    "pyramid": ReferenceCellTopology(
        "pyramid",
        3,
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.5, 1.0),
        ),
        (
            ((0,), (1,), (2,), (3,), (4,)),
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (0, 4),
                (1, 4),
                (2, 4),
                (3, 4),
            ),
            (
                (0, 3, 2, 1),
                (0, 1, 4),
                (1, 2, 4),
                (2, 3, 4),
                (3, 0, 4),
            ),
            ((0, 1, 2, 3, 4),),
        ),
    ),
    "hexahedron": ReferenceCellTopology(
        "hexahedron",
        3,
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 1.0),
            (1.0, 1.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        (
            ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)),
            (
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ),
            (
                (0, 3, 2, 1),
                (4, 5, 6, 7),
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
            ),
            ((0, 1, 2, 3, 4, 5, 6, 7),),
        ),
    ),
}


def reference_cell_topology(name: str) -> ReferenceCellTopology:
    if name not in REFERENCE_TOPOLOGIES:
        raise KeyError(f"Unknown reference topology {name!r}.")
    return REFERENCE_TOPOLOGIES[name]


__all__ = [
    "FacetOrientationAction",
    "FacetShape",
    "REFERENCE_TOPOLOGIES",
    "ReferenceCellTopology",
    "facet_orientation_actions",
    "facet_orientation_between",
    "reference_cell_topology",
]
