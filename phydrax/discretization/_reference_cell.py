from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations, permutations, product
from math import factorial
from typing import assert_never, Literal, TypeAlias

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._fingerprint import canonical_fingerprint
from ..typing import parse


@dataclass(frozen=True)
class ReferenceCellTopology:
    name: str
    dimension: int
    vertices: tuple[tuple[float, ...], ...]
    entities: tuple[tuple[tuple[int, ...], ...], ...]

    @property
    def topology_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "reference-cell",
                "name": self.name,
                "dimension": self.dimension,
                "vertices": self.vertices,
                "entities": self.entities,
            }
        )


_NamedFacetShape: TypeAlias = Literal["point", "edge", "triangle", "quadrilateral"]
FacetShape: TypeAlias = _NamedFacetShape | ReferenceCellTopology


@dataclass(frozen=True)
class FacetOrientationAction:
    shape: FacetShape
    permutation: tuple[int, ...]

    def __post_init__(self) -> None:
        if isinstance(self.shape, ReferenceCellTopology):
            if self.shape != reference_cell_topology(self.shape.name):
                raise ValueError(
                    "Facet reference topology must have its canonical field set."
                )
            size = len(self.shape.vertices)
        else:
            named = parse(self.shape, _NamedFacetShape, "shape")
            sizes = {"point": 1, "edge": 2, "triangle": 3, "quadrilateral": 4}
            size = sizes[named]
        if tuple(sorted(self.permutation)) != tuple(range(size)):
            raise ValueError("Facet orientation permutation is invalid.")

    def _shape_identity(self) -> str:
        return (
            self.shape.topology_id
            if isinstance(self.shape, ReferenceCellTopology)
            else self.shape
        )

    @property
    def orientation_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "facet-orientation-action",
                "shape": self._shape_identity(),
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

    def apply(self, values: ArrayLike, /, *, axis: int = 0) -> Array:
        return jnp.take(jnp.asarray(values), jnp.asarray(self.permutation), axis=axis)


def facet_orientation_actions(
    shape: FacetShape, /, *, maximum_actions: int = 1 << 16
) -> tuple[FacetOrientationAction, ...]:
    if isinstance(shape, ReferenceCellTopology):
        return _generic_facet_actions(shape, maximum_actions)
    shape = parse(shape, _NamedFacetShape, "shape")
    match shape:
        case "point":
            permutations = ((0,),)
        case "edge":
            permutations = ((0, 1), (1, 0))
        case "triangle":
            rotations = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
            reflections = ((0, 2, 1), (2, 1, 0), (1, 0, 2))
            permutations = rotations + reflections
        case "quadrilateral":
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
        case _:
            assert_never(shape)
    return tuple(FacetOrientationAction(shape, value) for value in permutations)


def facet_orientation_between(
    canonical_vertices: tuple[int, ...],
    local_vertices: tuple[int, ...],
    /,
    *,
    shape: FacetShape | None = None,
) -> FacetOrientationAction:
    if shape is None:
        shapes: dict[int, _NamedFacetShape] = {
            1: "point",
            2: "edge",
            3: "triangle",
            4: "quadrilateral",
        }
        if len(canonical_vertices) not in shapes:
            raise ValueError(
                "Higher-dimensional facet orientation requires an explicit reference topology."
            )
        shape = shapes[len(canonical_vertices)]
    if set(canonical_vertices) != set(local_vertices):
        raise ValueError("Facet orientations require identical vertex sets.")
    local_positions = tuple(local_vertices.index(value) for value in canonical_vertices)
    for action in facet_orientation_actions(shape):
        if action.permutation == local_positions:
            return action
    raise ValueError("Facet vertex order is not a valid orientation-group action.")


def _generic_facet_actions(
    shape: ReferenceCellTopology, maximum_actions: int, /
) -> tuple[FacetOrientationAction, ...]:
    if shape != reference_cell_topology(shape.name):
        raise ValueError("Facet reference topology must have its canonical field set.")
    if not isinstance(maximum_actions, int) or isinstance(maximum_actions, bool):
        raise TypeError("maximum_actions must be an integer.")
    if maximum_actions < 1:
        raise ValueError("maximum_actions must be positive.")
    simplex = shape.name.startswith("simplex:") or shape.name in (
        "interval",
        "triangle",
        "tetrahedron",
    )
    count = (
        factorial(len(shape.vertices))
        if simplex
        else 2**shape.dimension * factorial(shape.dimension)
    )
    if count > maximum_actions:
        raise ValueError("Facet orientation symmetry exceeds maximum_actions.")
    if simplex:
        return tuple(
            FacetOrientationAction(shape, permutation)
            for permutation in permutations(range(len(shape.vertices)))
        )
    positions = {vertex: index for index, vertex in enumerate(shape.vertices)}
    actions: list[FacetOrientationAction] = []
    for axes in permutations(range(shape.dimension)):
        for flips in product((False, True), repeat=shape.dimension):
            permutation = tuple(
                positions[
                    tuple(
                        1.0 - vertex[axis] if flip else vertex[axis]
                        for axis, flip in zip(axes, flips, strict=True)
                    )
                ]
                for vertex in shape.vertices
            )
            actions.append(FacetOrientationAction(shape, permutation))
    return tuple(actions)


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
            # Match tetrahedral connectivity: face i is opposite vertex i.
            ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)),
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


@lru_cache(maxsize=64)
def reference_cell_topology(
    name: str, /, *, maximum_entities: int = 1 << 20
) -> ReferenceCellTopology:
    """Resolve an explicit named or dimension-qualified reference-cell identity."""
    if not isinstance(name, str):
        raise TypeError("Reference-cell identity must be a string.")
    if name in REFERENCE_TOPOLOGIES:
        return REFERENCE_TOPOLOGIES[name]
    family, separator, dimension_text = name.partition(":")
    if (
        not separator
        or family not in ("simplex", "tensor")
        or not dimension_text.isdecimal()
    ):
        raise KeyError(f"Unknown reference topology {name!r}.")
    dimension = int(dimension_text)
    if dimension < 1 or dimension_text != str(dimension):
        raise ValueError(
            "Dimension-qualified cells require a canonical positive dimension."
        )
    if not isinstance(maximum_entities, int) or isinstance(maximum_entities, bool):
        raise TypeError("maximum_entities must be an integer.")
    if maximum_entities < 1:
        raise ValueError("maximum_entities must be positive.")
    # Bound the exponent before computing it or constructing any reference tables.
    base = 2 if family == "simplex" else 3
    exponent = dimension + 1 if family == "simplex" else dimension
    limit = maximum_entities + 1 if family == "simplex" else maximum_entities
    if exponent > maximum_entities.bit_length() or base**exponent > limit:
        raise ValueError("Reference-cell entity count exceeds maximum_entities.")
    native_names = {
        ("simplex", 1): "interval",
        ("simplex", 2): "triangle",
        ("simplex", 3): "tetrahedron",
        ("tensor", 1): "interval",
        ("tensor", 2): "quadrilateral",
        ("tensor", 3): "hexahedron",
    }
    native_name = native_names.get((family, dimension))
    if native_name is not None:
        native = REFERENCE_TOPOLOGIES[native_name]
        return ReferenceCellTopology(name, dimension, native.vertices, native.entities)
    if family == "simplex":
        vertices = ((0.0,) * dimension,) + tuple(
            tuple(1.0 if axis == vertex else 0.0 for axis in range(dimension))
            for vertex in range(dimension)
        )
        entities = tuple(
            tuple(combinations(range(dimension + 1), degree + 1))
            for degree in range(dimension + 1)
        )
    else:
        vertices = tuple(product((0.0, 1.0), repeat=dimension))
        entities = _tensor_reference_entities(vertices, dimension)
    return ReferenceCellTopology(name, dimension, vertices, entities)


def _tensor_reference_entities(
    vertices: tuple[tuple[float, ...], ...], dimension: int, /
) -> tuple[tuple[tuple[int, ...], ...], ...]:
    positions = {vertex: index for index, vertex in enumerate(vertices)}
    levels: list[tuple[tuple[int, ...], ...]] = []
    for degree in range(dimension + 1):
        cells: list[tuple[int, ...]] = []
        for axes in combinations(range(dimension), degree):
            fixed_axes = tuple(axis for axis in range(dimension) if axis not in axes)
            for fixed_values in product((0.0, 1.0), repeat=len(fixed_axes)):
                cells.append(
                    _tensor_face_vertices(
                        positions, dimension, axes, fixed_axes, fixed_values
                    )
                )
        levels.append(tuple(cells))
    return tuple(levels)


def _tensor_face_vertices(
    positions: dict[tuple[float, ...], int],
    dimension: int,
    axes: tuple[int, ...],
    fixed_axes: tuple[int, ...],
    fixed_values: tuple[float, ...],
    /,
) -> tuple[int, ...]:
    face: list[int] = []
    for varying in product((0.0, 1.0), repeat=len(axes)):
        point = [0.0] * dimension
        for axis, value in zip(fixed_axes, fixed_values, strict=True):
            point[axis] = value
        for axis, value in zip(axes, varying, strict=True):
            point[axis] = value
        face.append(positions[tuple(point)])
    return tuple(face)


__all__ = [
    "FacetOrientationAction",
    "FacetShape",
    "REFERENCE_TOPOLOGIES",
    "ReferenceCellTopology",
    "facet_orientation_actions",
    "facet_orientation_between",
    "reference_cell_topology",
]
