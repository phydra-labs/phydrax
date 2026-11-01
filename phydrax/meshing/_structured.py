#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Transfinite structured-block construction and tensor-cell topology."""

from __future__ import annotations

from itertools import combinations, product
from math import prod
from typing import final, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization.fem._high_order import ReferenceNodalFamily
from ..discretization.fem._reference import FiniteElementSpec, lagrange_element
from ..optim import (
    MinimizationResult,
    minimize,
    NewtonTrustRegion,
    OptimizationTermination,
)
from ..typing import checked
from ._controls import TransfiniteCurveControl, TransfiniteSurfaceControl


@final
class TransfiniteBlock(StrictModule, NonTrainableState):
    """Four oriented source curves or six oriented source faces for a block.

    Face indices are ``2 * normal_axis + side``. Curves increase along their
    remaining logical axis; face controls list intervals in increasing order of
    the remaining logical axes.
    """

    boundaries: tuple[TransfiniteCurveControl | TransfiniteSurfaceControl, ...]
    name: str = eqx.field(static=True)
    intervals: tuple[int, int] | tuple[int, int, int] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        boundaries: tuple[TransfiniteCurveControl | TransfiniteSurfaceControl, ...],
        /,
        *,
        tolerance: float = 1.0e-12,
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"TransfiniteBlock.name must be nonempty; got {name!r}")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                f"TransfiniteBlock.tolerance must be finite and nonnegative; got {tolerance!r}"
            )
        intervals: tuple[int, int] | tuple[int, int, int]
        if len(boundaries) == 4:
            curve0, curve1, curve2, curve3 = (
                boundaries[0],
                boundaries[1],
                boundaries[2],
                boundaries[3],
            )
            if (
                not isinstance(curve0, TransfiniteCurveControl)
                or not isinstance(curve1, TransfiniteCurveControl)
                or not isinstance(curve2, TransfiniteCurveControl)
                or not isinstance(curve3, TransfiniteCurveControl)
            ):
                raise TypeError(
                    "TransfiniteBlock.boundaries must contain four curve controls or six surface controls of one kind."
                )
            curves = (curve0, curve1, curve2, curve3)
            if (
                curves[0].intervals != curves[1].intervals
                or curves[2].intervals != curves[3].intervals
            ):
                raise ValueError(
                    "TransfiniteBlock.boundaries must have matching opposing curve interval counts."
                )
            intervals = (curves[2].intervals, curves[0].intervals)
        elif len(boundaries) == 6:
            face0, face1, face2, face3, face4, face5 = (
                boundaries[0],
                boundaries[1],
                boundaries[2],
                boundaries[3],
                boundaries[4],
                boundaries[5],
            )
            if (
                not isinstance(face0, TransfiniteSurfaceControl)
                or not isinstance(face1, TransfiniteSurfaceControl)
                or not isinstance(face2, TransfiniteSurfaceControl)
                or not isinstance(face3, TransfiniteSurfaceControl)
                or not isinstance(face4, TransfiniteSurfaceControl)
                or not isinstance(face5, TransfiniteSurfaceControl)
            ):
                raise TypeError(
                    "TransfiniteBlock.boundaries must contain four curve controls or six surface controls of one kind."
                )
            faces = (face0, face1, face2, face3, face4, face5)
            intervals = (
                faces[2].intervals[0],
                faces[0].intervals[0],
                faces[0].intervals[1],
            )
            for axis in range(3):
                expected = tuple(intervals[index] for index in range(3) if index != axis)
                if any(faces[2 * axis + side].intervals != expected for side in range(2)):
                    raise ValueError(
                        "TransfiniteBlock.boundaries must have matching face-edge interval counts."
                    )
        else:
            raise TypeError(
                "TransfiniteBlock.boundaries must contain four curve controls or six surface controls of one kind."
            )
        self.boundaries = boundaries
        self.name = name
        self.intervals = intervals
        self.tolerance = float(tolerance)
        self.block_id = canonical_fingerprint(
            {
                "kind": "transfinite-block",
                "name": name,
                "boundaries": tuple(boundary.control_id for boundary in boundaries),
                "tolerance": float(tolerance),
            }
        )


@final
class StructuredConstruction(StrictModule):
    """Mesh and coordinate-map construction data for canonical acceptance."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    logical_shape: tuple[int, ...] = eqx.field(static=True)
    boundary_residual: float = eqx.field(static=True)
    boundary_source_ids: tuple[str, ...] = eqx.field(static=True)
    block_id: str = eqx.field(static=True)
    optimization: MinimizationResult | None

    def __init__(
        self,
        mesh: CellMesh,
        geometry: CellGeometrySpec,
        logical_shape: tuple[int, ...],
        boundary_residual: float,
        boundary_source_ids: tuple[str, ...],
        block_id: str,
        optimization: MinimizationResult | None,
    ) -> None:
        self.mesh = mesh
        self.geometry = geometry
        self.logical_shape = logical_shape
        self.boundary_residual = boundary_residual
        self.boundary_source_ids = boundary_source_ids
        self.block_id = block_id
        self.optimization = optimization


def _checked_shape(shape: tuple[int, ...], /) -> tuple[int, ...]:
    if len(shape) not in (2, 3) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 2 for size in shape
    ):
        raise ValueError(
            f"structured.logical_shape must have two or three axes with at least two nodes; got {shape!r}"
        )
    node_count = prod(shape)
    if node_count - 1 > np.iinfo(np.int32).max:
        raise OverflowError(
            f"structured node identity exceeds int32 connectivity range for shape {shape!r}"
        )
    return shape


def logical_cell_bank(shape: tuple[int, ...], /) -> np.ndarray:
    """Build C-order quad/hex connectivity with the reference corner ordering."""
    _checked_shape(shape)
    if len(shape) == 2:
        offsets = ((0, 0), (1, 0), (1, 1), (0, 1))
    else:
        offsets = (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1),
            (0, 1, 1),
        )
    cell_shape = tuple(size - 1 for size in shape)
    cell_count = prod(cell_shape)
    node_strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
    cell_strides = tuple(prod(cell_shape[axis + 1 :]) for axis in range(len(shape)))
    flat = np.arange(cell_count, dtype=np.int64)
    base = np.zeros(cell_count, dtype=np.int64)
    coordinate = np.empty(cell_count, dtype=np.int64)
    for axis in range(len(shape)):
        np.floor_divide(flat, cell_strides[axis], out=coordinate)
        np.remainder(coordinate, cell_shape[axis], out=coordinate)
        np.multiply(coordinate, node_strides[axis], out=coordinate)
        np.add(base, coordinate, out=base)
    cells = np.empty((cell_count, len(offsets)), dtype=np.int32)
    for corner, offset in enumerate(offsets):
        shift = sum(
            value * stride for value, stride in zip(offset, node_strides, strict=True)
        )
        np.add(base, shift, out=cells[:, corner], casting="unsafe")
    return cells


def curve_values(
    control: TransfiniteCurveControl,
    /,
    *,
    subdivisions: int = 1,
) -> np.ndarray:
    """Sample an oriented source curve at every logical geometry node."""
    if (
        isinstance(subdivisions, bool)
        or not isinstance(subdivisions, int)
        or subdivisions < 1
    ):
        raise ValueError("subdivisions must be a positive integer.")
    first, last = control.parameter_range
    parameters = np.linspace(
        first, last, control.intervals * subdivisions + 1, dtype=np.float64
    )
    if control.reversed:
        parameters = parameters[::-1]
    values = np.asarray(
        control.curve.evaluate(jnp.asarray(parameters, dtype=jnp.float64)),
        dtype=np.float64,
    )
    if (
        values.ndim != 2
        or values.shape[0] != parameters.size
        or not np.all(np.isfinite(values))
    ):
        raise ValueError(
            "A transfinite source curve must return finite coordinates for every parameter."
        )
    return values


def _face_values(
    control: TransfiniteSurfaceControl,
    /,
    *,
    subdivisions: int,
) -> np.ndarray:
    shape = tuple(intervals * subdivisions + 1 for intervals in control.intervals)
    logical = np.empty((*shape, 2), dtype=np.float64)
    logical[..., 0] = np.linspace(0.0, 1.0, shape[0], dtype=np.float64)[:, None]
    logical[..., 1] = np.linspace(0.0, 1.0, shape[1], dtype=np.float64)[None, :]
    parameters = np.empty_like(logical)
    box = np.asarray(control.parameter_box, dtype=np.float64)
    for logical_axis, parameter_axis in enumerate(control.permutation):
        coordinate = (
            1.0 - logical[..., logical_axis]
            if control.flips[logical_axis]
            else logical[..., logical_axis]
        )
        parameters[..., parameter_axis] = box[0, parameter_axis] + coordinate * (
            box[1, parameter_axis] - box[0, parameter_axis]
        )
    values = np.asarray(
        control.surface.evaluate(jnp.asarray(parameters, dtype=jnp.float64)),
        dtype=np.float64,
    )
    if values.ndim != 3 or values.shape[:2] != shape or not np.all(np.isfinite(values)):
        raise ValueError(
            "A transfinite source face must return finite coordinates on its parameter grid."
        )
    return values


def boundary_lattice(
    block: TransfiniteBlock,
    /,
    *,
    subdivisions: int = 1,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Sample and reconcile all declared oriented source boundaries."""
    if (
        isinstance(subdivisions, bool)
        or not isinstance(subdivisions, int)
        or subdivisions < 1
    ):
        raise ValueError("subdivisions must be a positive integer.")
    shape = tuple(intervals * subdivisions + 1 for intervals in block.intervals)
    sampled = tuple(
        curve_values(boundary, subdivisions=subdivisions)
        if isinstance(boundary, TransfiniteCurveControl)
        else _face_values(boundary, subdivisions=subdivisions)
        for boundary in block.boundaries
    )
    ambient_dimensions = {value.shape[-1] for value in sampled}
    if len(ambient_dimensions) != 1:
        raise ValueError("All transfinite boundaries must share one ambient dimension.")
    ambient = sampled[0].shape[-1]
    if ambient < len(shape):
        raise ValueError("The ambient dimension must be at least the block dimension.")
    lattice = np.zeros((*shape, ambient), dtype=np.float64)
    assigned = np.zeros(shape, dtype=np.bool_)
    residual = 0.0
    for face, values in enumerate(sampled):
        axis, side = divmod(face, 2)
        selector = tuple(
            (0 if side == 0 else -1) if logical_axis == axis else slice(None)
            for logical_axis in range(len(shape))
        )
        target = lattice[selector]
        existing = assigned[selector]
        if values.shape != target.shape:
            raise ValueError(
                f"Boundary chart {face} has shape {values.shape}; expected {target.shape}."
            )
        mismatch = float(
            np.max(
                np.linalg.norm(target[existing] - values[existing], axis=-1), initial=0.0
            )
        )
        residual = max(residual, mismatch)
        if mismatch > block.tolerance:
            raise ValueError(
                f"Transfinite boundaries disagree at shared edges or vertices; residual {mismatch} exceeds {block.tolerance}."
            )
        target[~existing] = values[~existing]
        existing[...] = True
    return lattice, assigned, residual


def _transfinite(lattice: np.ndarray, /) -> np.ndarray:
    """Extend a compatible boundary lattice by the Gordon–Hall Boolean sum."""
    shape = lattice.shape[:-1]
    dimension = len(shape)
    if dimension not in (2, 3) or any(size < 2 for size in shape):
        raise ValueError("A transfinite lattice must have two or three nontrivial axes.")
    parameters = tuple(np.linspace(0.0, 1.0, size, dtype=np.float64) for size in shape)
    result = np.zeros_like(lattice)
    for codimension in range(1, dimension + 1):
        sign = 1.0 if codimension % 2 else -1.0
        for fixed_axes in combinations(range(dimension), codimension):
            for sides in product((0, 1), repeat=codimension):
                selector = tuple(
                    (0 if sides[fixed_axes.index(axis)] == 0 else -1)
                    if axis in fixed_axes
                    else slice(None)
                    for axis in range(dimension)
                )
                values = lattice[selector]
                expanded = [size for size in shape]
                for axis in fixed_axes:
                    expanded[axis] = 1
                term = values.reshape((*expanded, lattice.shape[-1]))
                for axis, side in zip(fixed_axes, sides, strict=True):
                    weights = parameters[axis] if side else 1.0 - parameters[axis]
                    weight_shape = [1] * (dimension + 1)
                    weight_shape[axis] = shape[axis]
                    term = term * weights.reshape(weight_shape)
                result += sign * term
    return result


def _tensor_coordinate_element(
    cell_kind: Literal["quadrilateral", "hexahedron"], degree: int, /
) -> FiniteElementSpec:
    """Return the existing tensor Lagrange basis on equispaced geometry nodes."""
    if degree < 1:
        raise ValueError("Tensor coordinate elements require degree >= 1.")
    return ReferenceNodalFamily(cell_kind, degree, node_set="equispaced").finite_element()


def _geometry_routes(
    intervals: tuple[int, int] | tuple[int, int, int],
    degree: int,
    reference_nodes: np.ndarray,
    /,
) -> np.ndarray:
    local = np.rint(reference_nodes * degree).astype(np.int64)
    route = np.empty((prod(intervals), reference_nodes.shape[0]), dtype=np.int32)
    if len(intervals) == 2:
        shape = (
            intervals[0] * degree + 1,
            intervals[1] * degree + 1,
        )
        for cell, index in enumerate(np.ndindex(intervals)):
            for local_index in range(local.shape[0]):
                route[cell, local_index] = np.ravel_multi_index(
                    (
                        index[0] * degree + int(local[local_index, 0]),
                        index[1] * degree + int(local[local_index, 1]),
                    ),
                    shape,
                )
        return route
    if len(intervals) == 3:
        shape = (
            intervals[0] * degree + 1,
            intervals[1] * degree + 1,
            intervals[2] * degree + 1,
        )
        for cell, index in enumerate(np.ndindex(intervals)):
            for local_index in range(local.shape[0]):
                route[cell, local_index] = np.ravel_multi_index(
                    (
                        index[0] * degree + int(local[local_index, 0]),
                        index[1] * degree + int(local[local_index, 1]),
                        index[2] * degree + int(local[local_index, 2]),
                    ),
                    shape,
                )
        return route
    raise ValueError(
        "Structured geometry routes require two or three logical dimensions."
    )


def _elliptic_objective(
    free: Array, args: tuple[Array, Array, tuple[int, ...]], /
) -> Array:
    base, rows, shape = args
    lattice = base.at[rows].set(free).reshape((*shape, base.shape[-1]))
    energy = jnp.asarray(0.0, dtype=jnp.float64)
    for axis, size in enumerate(shape):
        differences = jnp.diff(lattice, axis=axis)
        energy = energy + jnp.sum(differences * differences) * (size - 1) ** 2
    return energy


@checked
def construct_structured_block(
    block: TransfiniteBlock,
    /,
    *,
    geometry_degree: int = 1,
    optimization_steps: int = 0,
) -> StructuredConstruction:
    """Build a transfinite quad/hex block and its shared high-order map."""
    if (
        isinstance(geometry_degree, bool)
        or not isinstance(geometry_degree, int)
        or geometry_degree < 1
    ):
        raise ValueError("geometry_degree must be a positive integer.")
    if (
        isinstance(optimization_steps, bool)
        or not isinstance(optimization_steps, int)
        or optimization_steps < 0
    ):
        raise ValueError("optimization_steps must be a nonnegative integer.")
    lattice, assigned, residual = boundary_lattice(block, subdivisions=geometry_degree)
    lattice = _transfinite(lattice)
    high_shape = lattice.shape[:-1]
    if prod(high_shape) - 1 > np.iinfo(np.int32).max:
        raise OverflowError(
            "Structured coordinate-node identities exceed int32 route range."
        )
    optimization = None
    if optimization_steps:
        free = np.flatnonzero(~assigned.reshape(-1))
        if free.size:
            base = jnp.asarray(lattice.reshape(-1, lattice.shape[-1]), dtype=jnp.float64)
            rows = jnp.asarray(free, dtype=jnp.int32)
            optimization = minimize(
                _elliptic_objective,
                base[rows],
                method=NewtonTrustRegion(),
                termination=OptimizationTermination(maximum_steps=optimization_steps),
                args=(base, rows, high_shape),
            )
            lattice.reshape(-1, lattice.shape[-1])[free] = np.asarray(
                jax.device_get(optimization.parameters), dtype=np.float64
            )
    ambient = lattice.shape[-1]
    node_slice = tuple(slice(None, None, geometry_degree) for _ in block.intervals)
    mesh_points = lattice[node_slice].reshape(-1, ambient)
    cells = logical_cell_bank(tuple(intervals + 1 for intervals in block.intervals))
    kind: Literal["quadrilateral", "hexahedron"] = (
        "quadrilateral" if len(block.intervals) == 2 else "hexahedron"
    )
    mesh = CellMesh(mesh_points, (CellBlock(block.name, kind, cells),))
    element = (
        lagrange_element(kind, 1)
        if geometry_degree == 1
        else _tensor_coordinate_element(kind, geometry_degree)
    )
    geometry = CellGeometrySpec(
        {block.name: element},
        {
            block.name: _geometry_routes(
                block.intervals, geometry_degree, np.asarray(element.reference_nodes)
            )
        },
        lattice.reshape(-1, ambient),
    )
    return StructuredConstruction(
        mesh,
        geometry,
        tuple(size + 1 for size in block.intervals),
        residual,
        tuple(boundary.source_id for boundary in block.boundaries),
        block.block_id,
        optimization,
    )


def generate_structured_block(
    block: TransfiniteBlock,
    /,
    *,
    geometry_degree: int = 1,
    optimization_steps: int = 0,
) -> StructuredConstruction:
    """Construct a structured block for canonical native acceptance."""
    return construct_structured_block(
        block,
        geometry_degree=geometry_degree,
        optimization_steps=optimization_steps,
    )


__all__ = [
    "TransfiniteBlock",
    "StructuredConstruction",
    "logical_cell_bank",
    "boundary_lattice",
    "construct_structured_block",
    "generate_structured_block",
]
