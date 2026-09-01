from __future__ import annotations

from dataclasses import dataclass


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


__all__ = ["REFERENCE_TOPOLOGIES", "ReferenceCellTopology", "reference_cell_topology"]
