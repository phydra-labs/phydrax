#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Construct Coons/Gordon–Hall coordinate maps on cell meshes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.discretization._cell_geometry_validity as _dependency_phydrax_discretization__cell_geometry_validity
import phydrax.discretization._coordinate_budget as _dependency_phydrax_discretization__coordinate_budget
import phydrax.geometry._embedding_state as _dependency_phydrax_geometry__embedding_state
import phydrax.geometry._global_embedding as _dependency_phydrax_geometry__global_embedding
import phydrax.meshing._controls as _dependency_phydrax_meshing__controls
import phydrax.meshing._failures as _leaf_phydrax_meshing__failures
import phydrax.meshing._family_execution as family_execution_owner
import phydrax.meshing._host_banks as _dependency_phydrax_meshing__host_banks
import phydrax.meshing._volume_generation as _dependency_phydrax_meshing__volume_generation

from .._fingerprint import canonical_fingerprint
from .._meshcore import NativeExecutionScope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import nonnegative_integer
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry import affine_geometry_construction_bounds
from ..discretization._cell_geometry_predecessor import CellGeometryPredecessor
from ..discretization._cell_mesh import (
    cell_block_construction_bounds,
    cell_mesh_construction_bounds,
)
from ..optim import (
    MinimizationResult,
    minimize,
    NewtonTrustRegion,
    OptimizationTermination,
)
from ..typing import checked
from ._contracts import MeshingLimits


@final
class TransfiniteBlock(StrictModule, NonTrainableState):
    """Four curves (u-low, u-high, v-low, v-high) or six logical faces.

    Every curve increases in its remaining logical axis. Six faces are ordered
    ``2 * normal_axis + side`` and increase in the remaining sorted axes.
    Interval counts and source chart orientation follow the declared controls.
    """

    boundaries: tuple[
        _dependency_phydrax_meshing__controls.TransfiniteCurveControl
        | _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl,
        ...,
    ]
    name: str = eqx.field(static=True)
    intervals: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        boundaries: tuple[
            _dependency_phydrax_meshing__controls.TransfiniteCurveControl
            | _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl,
            ...,
        ],
        /,
        *,
        tolerance: float = 1e-12,
    ) -> None:
        if not name.strip():
            raise ValueError(f"TransfiniteBlock.name must be nonempty; got {name!r}")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                f"TransfiniteBlock.tolerance must be finite and nonnegative; got {tolerance}"
            )
        match boundaries:
            case (
                _dependency_phydrax_meshing__controls.TransfiniteCurveControl() as c0,
                _dependency_phydrax_meshing__controls.TransfiniteCurveControl() as c1,
                _dependency_phydrax_meshing__controls.TransfiniteCurveControl() as c2,
                _dependency_phydrax_meshing__controls.TransfiniteCurveControl() as c3,
            ):
                curves = (c0, c1, c2, c3)
                counts = (curves[2].intervals, curves[0].intervals)
                if (
                    curves[0].intervals != curves[1].intervals
                    or curves[2].intervals != curves[3].intervals
                ):
                    raise ValueError(
                        f"TransfiniteBlock.boundaries must have matching opposing curve interval counts; got {tuple(curve.intervals for curve in curves)}"
                    )
            case (
                _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl() as f0,
                _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl() as f1,
                _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl() as f2,
                _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl() as f3,
                _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl() as f4,
                _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl() as f5,
            ):
                faces = (f0, f1, f2, f3, f4, f5)
                counts = (
                    faces[2].intervals[0],
                    faces[0].intervals[0],
                    faces[0].intervals[1],
                )
                for axis in range(3):
                    expected = tuple(counts[i] for i in range(3) if i != axis)
                    if any(
                        faces[2 * axis + side].intervals != expected for side in range(2)
                    ):
                        raise ValueError(
                            f"TransfiniteBlock.boundaries must have matching face and edge interval counts; got {tuple(face.intervals for face in faces)}"
                        )
            case _:
                raise TypeError(
                    f"TransfiniteBlock.boundaries must contain four curve controls or six surface controls; got {len(boundaries)} controls with incompatible kinds"
                )
        self.boundaries = boundaries
        self.name = name
        self.intervals = counts
        self.tolerance = float(tolerance)
        self.block_id = canonical_fingerprint(
            {
                "kind": "transfinite-block",
                "name": name,
                "boundaries": tuple(b.control_id for b in boundaries),
                "tolerance": tolerance,
            }
        )


@dataclass(frozen=True, slots=True)
class StructuredConstruction:
    """Construction evidence, not a second computational mesh carrier."""

    mesh: CellMesh
    geometry: CellGeometrySpec
    logical_shape: tuple[int, ...]
    validity: (
        _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate
    )
    embedding: _dependency_phydrax_geometry__global_embedding.GlobalEmbeddingCertificate
    optimization: MinimizationResult | None
    boundary_residual: float
    block_id: str


def _fill_tensor_cells(
    nodes: np.ndarray,
    cells: np.ndarray,
    /,
    *,
    scope: NativeExecutionScope,
) -> None:
    """Gather C-order tensor corners directly, without a local connectivity bank."""
    if nodes.ndim != 3 or any(size < 2 for size in nodes.shape):
        raise ValueError("tensor_grid.nodes must be a three-axis node lattice")
    shape = tuple(size - 1 for size in nodes.shape)
    if cells.shape != (prod(shape), 8):
        raise ValueError("tensor_grid.cells must cover every lattice cell")
    corners = (
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    )
    with scope.batch(cells.size) as ticket:
        target = cells.reshape((*shape, 8))
        for corner, offset in enumerate(corners):
            ticket.consume(prod(shape))
            target[..., corner] = nodes[
                tuple(slice(start, start + size) for start, size in zip(offset, shape, strict=True))
            ]


@checked
def logical_cell_bank(
    shape: tuple[int, ...], /, *, scope: NativeExecutionScope
) -> np.ndarray:
    """Construct cyclic tensor-cell connectivity for a C-order node lattice."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in shape):
        raise ValueError(
            f"structured.logical_shape must have at least two integer nodes per axis; got {shape}"
        )
    if len(shape) == 2:
        offsets = ((0, 0), (1, 0), (1, 1), (0, 1))
    elif len(shape) == 3:
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
    else:
        raise ValueError(
            f"structured.logical_shape must have two or three axes; got {shape}"
        )
    if prod(shape) - 1 > np.iinfo(np.int32).max:
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.RESOURCE_EXHAUSTED,
            f"structured.logical_shape must fit the int32 node range; got {shape}",
            stage="structured_topology",
        )
    cell_shape = tuple(n - 1 for n in shape)
    count = prod(cell_shape)
    row_count = min(count, 4096)
    # Each axis decodes, scales and accumulates one base index per cell.
    # Corner offsets are immutable grid operands, not per-cell arithmetic.
    corner_work = 2 * len(shape) * len(offsets)
    work_upper = (
        (4 * len(shape) + len(offsets)) * count
        + count - row_count + corner_work
    )
    with scope.batch(work_upper) as ticket, scope.host_workspace() as workspace:
        # Stream C-order rows through a fixed-size index bank. Connectivity is
        # retained in full, but temporary indices do not scale with the mesh.
        workspace.set_bound(row_count * 8 + 4096)
        strides = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
        ticket.consume(corner_work)
        corner_strides = tuple(
            sum(offset * stride for offset, stride in zip(corner, strides, strict=True))
            for corner in offsets
        )
        flat = np.arange(row_count, dtype=np.int64)
        coordinate = _dependency_phydrax_meshing__host_banks.family_host_array(
            (row_count,), np.int64, scope=scope
        )
        base = _dependency_phydrax_meshing__host_banks.family_host_array(
            (row_count,), np.int64, scope=scope
        )
        cells = _dependency_phydrax_meshing__host_banks.family_host_array(
            (count, len(offsets)), np.int32, scope=scope
        )
        for start in range(0, count, row_count):
            stop = min(start + row_count, count)
            width = stop - start
            base[:width].fill(0)
            if start:
                ticket.consume(width)
                np.add(flat[:width], row_count, out=flat[:width])
            for axis in range(len(shape)):
                ticket.consume(4 * width)
                np.floor_divide(
                    flat[:width], prod(cell_shape[axis + 1 :]),
                    out=coordinate[:width],
                )
                np.remainder(
                    coordinate[:width], cell_shape[axis], out=coordinate[:width]
                )
                np.multiply(
                    coordinate[:width], strides[axis], out=coordinate[:width]
                )
                np.add(base[:width], coordinate[:width], out=base[:width])
            for corner, stride in enumerate(corner_strides):
                ticket.consume(width)
                np.add(base[:width], stride, out=cells[start:stop, corner])
        return cells


@checked
def curve_values(
    control: _dependency_phydrax_meshing__controls.TransfiniteCurveControl,
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    curve = control.curve
    first, last = control.parameter_range
    count = control.intervals + 1
    # One curve evaluation per sampled node.
    with (
        scope.batch(count, queries_upper=count) as ticket,
        scope.host_workspace() as workspace,
    ):
        # The parameter vector is the one host temporary; sampled values remain
        # in their JAX owner.
        workspace.set_bound(8 * count + 4096)
        values = np.linspace(first, last, count, dtype=np.float64)
        if control.reversed:
            values = values[::-1]
        ticket.consume(count, geometry_queries=count)
        return np.asarray(
            curve.evaluate(jnp.asarray(values, dtype=jnp.float64)), dtype=np.float64
        )


def _face_values(
    control: _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl,
    /,
    *,
    scope: NativeExecutionScope,
) -> np.ndarray:
    surface = control.surface
    shape = tuple(n + 1 for n in control.intervals)
    count = shape[0] * shape[1]
    # One surface evaluation per node.
    with (
        scope.batch(count, queries_upper=count) as ticket,
        scope.host_workspace() as workspace,
    ):
        # The two logical spacing vectors are the only unpooled temporaries.
        workspace.set_bound(8 * sum(shape) + 4096)
        logical = _dependency_phydrax_meshing__host_banks.family_host_array(
            (*shape, 2), np.float64, scope=scope
        )
        logical[..., 0] = np.linspace(0.0, 1.0, shape[0], dtype=np.float64)[:, None]
        logical[..., 1] = np.linspace(0.0, 1.0, shape[1], dtype=np.float64)[None, :]
        parameters = _dependency_phydrax_meshing__host_banks.family_host_array(
            logical.shape, np.float64, scope=scope
        )
        box = np.asarray(control.parameter_box, dtype=np.float64)
        for axis, parameter_axis in enumerate(control.permutation):
            coordinate = parameters[..., parameter_axis]
            if control.flips[axis]:
                np.subtract(1.0, logical[..., axis], out=coordinate)
            else:
                np.copyto(coordinate, logical[..., axis])
            np.multiply(
                coordinate,
                box[1, parameter_axis] - box[0, parameter_axis],
                out=coordinate,
            )
            np.add(coordinate, box[0, parameter_axis], out=coordinate)
        ticket.consume(count, geometry_queries=count)
        return np.asarray(
            surface.evaluate(jnp.asarray(parameters, dtype=jnp.float64)), dtype=np.float64
        )


@checked
def boundary_lattice(
    block: TransfiniteBlock, /, *, scope: NativeExecutionScope
) -> tuple[np.ndarray, np.ndarray, float]:
    shape = tuple(n + 1 for n in block.intervals)
    sampled = []
    for boundary in block.boundaries:
        match boundary:
            case _dependency_phydrax_meshing__controls.TransfiniteCurveControl():
                sampled.append(curve_values(boundary, scope=scope))
            case _dependency_phydrax_meshing__controls.TransfiniteSurfaceControl():
                sampled.append(_face_values(boundary, scope=scope))
            case _:
                raise TypeError(
                    f"TransfiniteBlock.boundaries must contain curve or surface controls; got {type(boundary).__name__}"
                )
    dimensions = {value.shape[-1] for value in sampled}
    if len(dimensions) != 1:
        raise ValueError(
            f"structured.boundaries must share one ambient dimension; got {sorted(dimensions)}"
        )
    ambient = sampled[0].shape[-1]
    if ambient < len(shape):
        raise ValueError(
            f"structured.boundaries must have ambient dimension at least the block dimension; got {ambient} for {len(shape)} logical axes"
        )
    sizes = tuple(prod(values.shape[:-1]) for values in sampled)
    largest = max(sizes)
    lattice = _dependency_phydrax_meshing__host_banks.family_host_array(
        (*shape, ambient), np.float64, scope=scope
    )
    lattice.fill(0.0)
    assigned = _dependency_phydrax_meshing__host_banks.family_host_array(
        shape, np.bool_, scope=scope
    )
    assigned.fill(False)
    # Every face reuses one pooled difference, distance and vacancy bank.
    differences = _dependency_phydrax_meshing__host_banks.family_host_array(
        (largest, ambient), np.float64, scope=scope
    )
    distances = _dependency_phydrax_meshing__host_banks.family_host_array(
        (largest,), np.float64, scope=scope
    )
    vacancies = _dependency_phydrax_meshing__host_banks.family_host_array(
        (largest,), np.bool_, scope=scope
    )
    residual = 0.0
    for face, (values, count) in enumerate(zip(sampled, sizes, strict=True)):
        axis, side = divmod(face, 2)
        selector = tuple(
            (0 if side == 0 else -1) if i == axis else slice(None)
            for i in range(len(shape))
        )
        target = lattice[selector]
        existing = assigned[selector]
        difference = differences[:count].reshape(values.shape)
        distance = distances[:count].reshape(values.shape[:-1])
        vacant = vacancies[:count].reshape(values.shape[:-1])
        np.subtract(target, values, out=difference)
        np.square(difference, out=difference)
        np.sum(difference, axis=-1, out=distance)
        np.sqrt(distance, out=distance)
        mismatch = float(np.max(distance, where=existing, initial=0.0))
        residual = max(residual, mismatch)
        if mismatch > block.tolerance:
            raise ValueError(
                f"structured.boundaries must agree within tolerance {block.tolerance}; got incompatible boundary charts at face {face} of {block.name!r}, residual {mismatch}"
            )
        np.logical_not(existing, out=vacant)
        np.copyto(target, values, where=vacant[..., None])
        existing.fill(True)
    return lattice, assigned, residual


def _transfinite(lattice: np.ndarray, /, *, scope: NativeExecutionScope) -> np.ndarray:
    """Boolean-sum extension: sum faces minus edges plus vertices."""
    shape = lattice.shape[:-1]
    dimension = len(shape)
    work_upper = lattice.size * (3**dimension - 1)
    with scope.batch(work_upper) as ticket, scope.host_workspace() as workspace:
        # At most the old/new broadcast term coexist. Parameter vectors and
        # their reversed weights are separately included; this is an admitted
        # temporary upper bound, never reported as measured array payload.
        workspace.set_bound(2 * lattice.nbytes + 16 * sum(shape) + 4096)
        parameters = tuple(np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape)
        result = _dependency_phydrax_meshing__host_banks.family_host_array(
            lattice.shape, np.float64, scope=scope
        )
        result.fill(0.0)
        for codimension in range(1, dimension + 1):
            for fixed_axes in combinations(range(dimension), codimension):
                for sides in product((0, 1), repeat=codimension):
                    selector = tuple(
                        (0 if sides[fixed_axes.index(axis)] == 0 else -1)
                        if axis in fixed_axes
                        else slice(None)
                        for axis in range(dimension)
                    )
                    values = lattice[selector]
                    expanded = list(shape)
                    for axis in fixed_axes:
                        expanded[axis] = 1
                    term = values.reshape((*expanded, lattice.shape[-1]))
                    ticket.consume(result.size)
                    for axis, side in zip(fixed_axes, sides, strict=True):
                        weights = parameters[axis] if side else 1.0 - parameters[axis]
                        weight_shape = [1] * (dimension + 1)
                        weight_shape[axis] = shape[axis]
                        term = term * weights.reshape(weight_shape)
                    result += (1 if codimension % 2 else -1) * term
        return result


def _elliptic_objective(
    free: Array, args: tuple[Array, Array, tuple[int, ...]], /
) -> Array:
    base, rows, shape = args
    lattice = base.at[rows].set(free).reshape((*shape, base.shape[-1]))
    # Discrete Dirichlet energy has the harmonic/elliptic extension as minimizer.
    energy = jnp.asarray(0.0, dtype=jnp.float64)
    for axis in range(len(shape)):
        differences = jnp.diff(lattice, axis=axis)
        energy = energy + jnp.sum(differences * differences) * (shape[axis] - 1) ** 2
    return energy


@checked
def certify_constructed_mesh(
    mesh: CellMesh,
    /,
    *,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy
    | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    geometry: CellGeometrySpec | None = None,
    predecessor: CellGeometryPredecessor | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> tuple[
    _dependency_phydrax_discretization__cell_geometry_validity.CellValidityCertificate,
    _dependency_phydrax_geometry__global_embedding.GlobalEmbeddingCertificate,
]:
    """Independent positivity and whole-complex collision acceptance.

    A request-owned ``predecessor`` lets both certificates import its
    authenticated unchanged rows and root theorem (see their owners).
    """
    if geometry is None:
        mesh_shapes = tuple(
            (cell_block.cell_kind, cell_block.cell_count, cell_block.arity)
            for cell_block in mesh.blocks
        )
        with scope.host_workspace() as workspace:
            workspace.set_bound(
                affine_geometry_construction_bounds(
                    mesh.coordinates.shape[0],
                    mesh.ambient_dimension,
                    mesh_shapes,
                )
            )
            geometry = CellGeometrySpec.affine(mesh)
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.polynomial_preparation_scope(),
    ):
        validity = _dependency_phydrax_discretization__cell_geometry_validity.certify_cell_geometry_validity(
            geometry,
            mesh=mesh,
            policy=validity_policy,
            budget=budget,
            predecessor=predecessor,
        )
    bad = (
        np.asarray(validity.status)
        != _dependency_phydrax_discretization__cell_geometry_validity.CellValidityStatus.CERTIFIED_VALID
    )
    if np.any(bad):
        ids = np.concatenate([np.asarray(b.global_ids) for b in mesh.blocks])
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED,
            f"structured.validity must certify mapped Jacobians; got {validity.invalid_count} invalid and {validity.unresolved_count} unresolved",
            stage="structured_validity",
            entity_ids=tuple(int(i) for i in ids[bad]),
        )
    with (
        family_execution_owner.coordinate_phase(scope=scope, budget=budget),
        budget.polynomial_preparation_scope(),
    ):
        embedding = (
            _dependency_phydrax_geometry__global_embedding.certify_global_embedding(
                mesh,
                geometry,
                validity,
                limits=certificate_limits,
                budget=budget,
                predecessor=predecessor,
            )
        )
    if embedding.status != "certified":
        raise _leaf_phydrax_meshing__failures.MeshingFailure(
            _leaf_phydrax_meshing__failures.MeshingFailureCategory.AUDIT_FAILED,
            "structured.embedding must certify the global complex; got "
            + "; ".join(finding.check for finding in embedding.findings),
            stage="structured_embedding",
        )
    return validity, embedding


@checked
def construct_structured_block(
    block: TransfiniteBlock,
    /,
    *,
    optimization_steps: int = 0,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy
    | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    scope: NativeExecutionScope,
    budget: _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget,
) -> StructuredConstruction:
    """Build a pure quad/hex block, with bounded optional native elliptic solve.

    Boundaries never move. Optimizer status reaches the construction evidence;
    regardless of termination, its returned coordinates must independently
    certify. A failed candidate is not silently replaced by the initial map.
    """

    optimization_steps = nonnegative_integer(optimization_steps, "optimization_steps")
    boundary, assigned, residual = boundary_lattice(block, scope=scope)
    lattice = _transfinite(boundary, scope=scope)
    shape = lattice.shape[:-1]
    ambient = lattice.shape[-1]
    # Declared boundary nodes keep their sampled coordinates exactly.
    np.copyto(lattice, boundary, where=assigned[..., None])
    points = lattice.reshape(-1, ambient)
    optimization = None
    if optimization_steps:
        nodes = assigned.size
        with scope.host_workspace() as workspace:
            # The vacancy mask and the free-row index vector are the phase's
            # unpooled host banks. JAX copies of the lattice and rows are device
            # memory outside managed bytes.
            workspace.set_bound(9 * nodes + 4096)
            free = np.flatnonzero(~assigned.reshape(-1))
            if free.size:
                method = NewtonTrustRegion()
                edge_count = sum(
                    (n - 1) * prod(shape[:axis] + shape[axis + 1 :])
                    for axis, n in enumerate(shape)
                )
                # Two objective/three gradient traversals and at most the
                # declared CG actions plus its final model action per Newton
                # step. Final value/gradient publication is separately included
                # in admission.
                work_upper = edge_count * (
                    optimization_steps * (5 + method.subproblem.maximum_steps + 1) + 2
                )
                with scope.batch(work_upper) as ticket:
                    base = jnp.asarray(points, dtype=jnp.float64)
                    rows = jnp.asarray(free, dtype=jnp.int32)
                    optimization = minimize(
                        _elliptic_objective,
                        base[rows],
                        method=method,
                        termination=OptimizationTermination(
                            maximum_steps=optimization_steps
                        ),
                        args=(base, rows, shape),
                    )
                    if not optimization.diagnostics.counts_complete:
                        raise RuntimeError(
                            "structured.optimization.diagnostics must report measured work counts; got counts_complete=False"
                        )
                    objectives, gradients, actions = jax.device_get(
                        (
                            optimization.diagnostics.objective_evaluations,
                            optimization.diagnostics.gradient_evaluations,
                            optimization.diagnostics.hvp_evaluations,
                        )
                    )
                    ticket.consume(
                        edge_count * (int(objectives) + int(gradients) + int(actions))
                    )
                    # Interior rows return into the pooled lattice; declared
                    # boundary rows never move.
                    points[free] = np.asarray(optimization.parameters, dtype=np.float64)
    kind = "quadrilateral" if len(shape) == 2 else "hexahedron"
    cells = logical_cell_bank(shape, scope=scope)
    vertex_count = points.shape[0]
    shapes = ((kind, cells.shape[0], cells.shape[1]),)
    # Each constructor's host temporaries end before the next one starts.
    with scope.host_workspace() as workspace:
        workspace.set_bound(cell_block_construction_bounds(*cells.shape))
        cell_block = CellBlock(block.name, kind, cells)
        workspace.set_bound(cell_mesh_construction_bounds(vertex_count, ambient, shapes))
        mesh = CellMesh(points, (cell_block,))
        workspace.set_bound(
            affine_geometry_construction_bounds(vertex_count, ambient, shapes)
        )
        geometry = CellGeometrySpec.affine(mesh)
    validity, embedding = certify_constructed_mesh(
        mesh,
        geometry=geometry,
        validity_policy=validity_policy,
        certificate_limits=certificate_limits,
        scope=scope,
        budget=budget,
    )
    return StructuredConstruction(
        mesh, geometry, shape, validity, embedding, optimization, residual, block.block_id
    )


@checked
def logical_cells(
    shape: tuple[int, ...], /, *, limits: MeshingLimits = MeshingLimits()
) -> np.ndarray:
    """Construct logical connectivity within the declared meshing limits."""
    with _dependency_phydrax_meshing__volume_generation.native_volume_execution_budget(
        limits
    ) as scope:
        return logical_cell_bank(shape, scope=scope)


@checked
def generate_structured_block(
    block: TransfiniteBlock,
    /,
    *,
    optimization_steps: int = 0,
    validity_policy: _dependency_phydrax_discretization__cell_geometry_validity.CellValidityPolicy
    | None = None,
    certificate_limits: _dependency_phydrax_geometry__embedding_state.MeshCertificateLimits
    | None = None,
    limits: MeshingLimits = MeshingLimits(),
) -> StructuredConstruction:
    """Construct and certify a block within the declared meshing limits."""
    with _dependency_phydrax_meshing__volume_generation.native_volume_execution_budget(
        limits
    ) as scope:
        ticket = scope.batch(0)
        ticket.settle()
        budget = _dependency_phydrax_discretization__coordinate_budget.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.retained_storage(lambda n: scope.batch(0, bytes_upper=n)):
            return construct_structured_block(
                block,
                optimization_steps=optimization_steps,
                validity_policy=validity_policy,
                certificate_limits=certificate_limits,
                scope=scope,
                budget=budget,
            )


__all__ = [
    "TransfiniteBlock",
    "StructuredConstruction",
    "logical_cells",
    "generate_structured_block",
]
