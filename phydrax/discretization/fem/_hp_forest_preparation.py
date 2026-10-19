# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Prepare isolated host topology buffers for one hp forest edit."""

from typing import final

import numpy as np
from numpy.typing import NDArray

from ..._strict import StrictModule
from ...typing import checked
from ._hp import FiniteElementHPTopology


@final
class HPForestPreparation(StrictModule):
    identifiers: NDArray[np.int64]
    allocated: NDArray[np.bool_]
    active: NDArray[np.bool_]
    degrees: NDArray[np.int32]
    roots: NDArray[np.int64]
    paths: NDArray[np.int64]
    levels: NDArray[np.int32]
    parents: NDArray[np.int32]
    children: NDArray[np.int32]
    child_valid: NDArray[np.bool_]


@checked
def prepare_hp_forest_edit(topology: FiniteElementHPTopology) -> HPForestPreparation:
    """Copy topology buffers in field order without mutating the accepted forest."""
    return HPForestPreparation(
        np.array(topology.cell_global_ids, dtype=np.int64, copy=True),
        np.array(topology.allocated, dtype=np.bool_, copy=True),
        np.array(topology.active, dtype=np.bool_, copy=True),
        np.array(topology.cell_degrees, dtype=np.int32, copy=True),
        np.array(topology.root_cell_ids, dtype=np.int64, copy=True),
        np.array(topology.path_codes, dtype=np.int64, copy=True),
        np.array(topology.levels, dtype=np.int32, copy=True),
        np.array(topology.parent_slots, dtype=np.int32, copy=True),
        np.array(topology.child_slots, dtype=np.int32, copy=True),
        np.array(topology.child_valid, dtype=np.bool_, copy=True),
    )


def tensor_corner_points(dimension: int, /) -> np.ndarray:
    """Return tensor corners in the quadrilateral/hexahedron vertex ordering."""
    match dimension:
        case 1:
            return np.asarray(((0.0,), (1.0,)), dtype=np.float64)
        case 2:
            return np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
            )
        case 3:
            return np.asarray(
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
                dtype=np.float64,
            )
        case _:
            raise ValueError(
                f"hp_forest.dimension must be one, two, or three; got {dimension}"
            )


def multilinear_cell_map(vertices: np.ndarray, points: np.ndarray, /) -> np.ndarray:
    """Evaluate the vertex shape functions at host tensor coordinates."""
    corners = tensor_corner_points(points.shape[1])
    factors = np.where(
        corners[None, :, :] == 0.0,
        1.0 - points[:, None, :],
        points[:, None, :],
    )
    basis = np.prod(factors, axis=-1)
    return basis @ vertices
