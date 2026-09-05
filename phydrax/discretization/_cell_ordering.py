#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


MESHIO_CELL_TYPES = {
    "line": ("interval", 1, 2),
    "triangle": ("triangle", 1, 3),
    "triangle6": ("triangle", 2, 3),
    "quad": ("quadrilateral", 1, 4),
    "quad9": ("quadrilateral", 2, 4),
    "tetra": ("tetrahedron", 1, 4),
    "tetra10": ("tetrahedron", 2, 4),
    "hexahedron": ("hexahedron", 1, 8),
    "hexahedron27": ("hexahedron", 2, 8),
    "wedge": ("prism", 1, 6),
    "wedge18": ("prism", 2, 6),
    "pyramid": ("pyramid", 1, 5),
    "pyramid14": ("pyramid", 2, 5),
}


_MESHIO_REFERENCE_NODES = {
    "line": ((0.0,), (1.0,)),
    "triangle": ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)),
    "triangle6": (
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.0, 0.5),
    ),
    "quad": ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
    "quad9": (
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
        (0.5, 0.0),
        (1.0, 0.5),
        (0.5, 1.0),
        (0.0, 0.5),
        (0.5, 0.5),
    ),
    "tetra": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    "tetra10": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
    ),
    "hexahedron": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
    ),
    "hexahedron27": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (1.0, 1.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 1.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
        (1.0, 0.5, 1.0),
        (0.5, 1.0, 1.0),
        (0.0, 0.5, 1.0),
        (0.0, 0.0, 0.5),
        (1.0, 0.0, 0.5),
        (1.0, 1.0, 0.5),
        (0.0, 1.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.5, 1.0),
        (0.5, 0.0, 0.5),
        (1.0, 0.5, 0.5),
        (0.5, 1.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    ),
    "wedge": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
    ),
    "wedge18": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (0.5, 0.0, 0.0),
        (0.5, 0.5, 0.0),
        (0.0, 0.5, 0.0),
        (0.5, 0.0, 1.0),
        (0.5, 0.5, 1.0),
        (0.0, 0.5, 1.0),
        (0.0, 0.0, 0.5),
        (1.0, 0.0, 0.5),
        (0.0, 1.0, 0.5),
        (0.5, 0.0, 0.5),
        (0.5, 0.5, 0.5),
        (0.0, 0.5, 0.5),
    ),
    "pyramid": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 1.0),
    ),
    "pyramid14": (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.5, 0.5, 1.0),
        (0.5, 0.0, 0.0),
        (1.0, 0.5, 0.0),
        (0.5, 1.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.25, 0.25, 0.5),
        (0.75, 0.25, 0.5),
        (0.75, 0.75, 0.5),
        (0.25, 0.75, 0.5),
        (0.5, 0.5, 0.0),
    ),
}


def meshio_reference_nodes(cell_type: str, /) -> np.ndarray:
    value = str(cell_type)
    if value not in _MESHIO_REFERENCE_NODES:
        raise ValueError(f"Unsupported high-order mesh cell type {value!r}.")
    return np.asarray(_MESHIO_REFERENCE_NODES[value], dtype=float)


def reference_node_permutation(cell_type: str, target_nodes: ArrayLike, /) -> np.ndarray:
    source = meshio_reference_nodes(cell_type)
    target = np.asarray(target_nodes, dtype=float)
    if source.shape != target.shape:
        raise ValueError("Imported and target geometry node counts differ.")
    permutation = []
    for point in target:
        matches = np.flatnonzero(np.max(np.abs(source - point), axis=1) <= 2.0e-12)
        if matches.size != 1:
            raise ValueError("High-order geometry node ordering is ambiguous.")
        permutation.append(int(matches[0]))
    return np.asarray(permutation, dtype=np.int32)


__all__ = ["MESHIO_CELL_TYPES", "meshio_reference_nodes", "reference_node_permutation"]
