#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Boundary-owned Coons/Gordon–Hall construction on canonical cell meshes."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import combinations, product
from math import prod
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._fingerprint import canonical_fingerprint
from .._meshcore import charge_native_geometry_queries, current_native_execution_budget
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import CellBlock, CellGeometrySpec, CellMesh
from ..discretization._cell_geometry_validity import (
    CellValidityCertificate,
    CellValidityPolicy,
    CellValidityStatus,
    certify_cell_geometry_validity,
)
from ..geometry._mesh_certificates import (
    certify_global_embedding,
    GlobalEmbeddingCertificate,
    MeshCertificateLimits,
)
from ..geometry.brep._patches import AbstractCurve, AbstractSurfacePatch
from ..optim import (
    MinimizationResult,
    minimize,
    NewtonTrustRegion,
    OptimizationTermination,
)
from ._contracts import MeshingFailure, MeshingFailureCategory
from ._controls import TransfiniteCurveControl, TransfiniteSurfaceControl
from ._quad_generation import _family_host_array


@final
class TransfiniteBlock(StrictModule, NonTrainableState):
    """Four curves (u-low, u-high, v-low, v-high) or six logical faces.

    Every curve increases in its remaining logical axis. Six faces are ordered
    ``2 * normal_axis + side`` and increase in the remaining sorted axes.
    Counts and source chart orientation are authoritative, not guessed.
    """

    boundaries: tuple[TransfiniteCurveControl | TransfiniteSurfaceControl, ...]
    name: str = eqx.field(static=True)
    intervals: tuple[int, ...] = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    block_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        boundaries: tuple[TransfiniteCurveControl | TransfiniteSurfaceControl, ...],
        /,
        *,
        tolerance: float = 1e-12,
    ) -> None:
        if not name.strip():
            raise ValueError("Blocks require an explicit name.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("tolerance must be finite and nonnegative.")
        if len(boundaries) == 4 and all(
            isinstance(b, TransfiniteCurveControl) for b in boundaries
        ):
            curves = tuple(
                b for b in boundaries if isinstance(b, TransfiniteCurveControl)
            )
            counts = (curves[2].intervals, curves[0].intervals)
            if (
                curves[0].intervals != curves[1].intervals
                or curves[2].intervals != curves[3].intervals
            ):
                raise ValueError("Opposing curve interval counts must agree exactly.")
        elif len(boundaries) == 6 and all(
            isinstance(b, TransfiniteSurfaceControl) for b in boundaries
        ):
            faces = tuple(
                b for b in boundaries if isinstance(b, TransfiniteSurfaceControl)
            )
            counts = (faces[2].intervals[0], faces[0].intervals[0], faces[0].intervals[1])
            for axis in range(3):
                expected = tuple(counts[i] for i in range(3) if i != axis)
                if any(faces[2 * axis + side].intervals != expected for side in range(2)):
                    raise ValueError(
                        "All logical face/edge interval counts must agree exactly."
                    )
        else:
            raise TypeError("Blocks require four curve controls or six surface controls.")
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
    validity: CellValidityCertificate
    embedding: GlobalEmbeddingCertificate
    optimization: MinimizationResult | None
    boundary_residual: float
    block_id: str


def logical_cells(shape: tuple[int, ...], /) -> np.ndarray:
    """Canonical cyclic tensor-cell connectivity for a C-order node lattice."""
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in shape):
        raise ValueError("Each logical axis requires at least two nodes.")
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
        raise ValueError("Logical cells require two or three axes.")
    if prod(shape) - 1 > np.iinfo(np.int32).max:
        raise MeshingFailure(
            MeshingFailureCategory.RESOURCE_EXHAUSTED,
            "Structured connectivity exceeds the canonical int32 node range.",
            stage="structured_topology",
        )
    cell_shape = tuple(n - 1 for n in shape)
    count = prod(cell_shape)
    budget = current_native_execution_budget()
    with budget.host_workspace() if budget is not None else nullcontext() as workspace:
        # NumPy's arange owns one temporary index vector. All remaining large
        # connectivity banks are admitted native allocations and use out=.
        if workspace is not None:
            workspace.set_bound(count * 8 + 4096)
        flat = np.arange(count, dtype=np.int64)
        coordinate = _family_host_array((count,), np.int64)
        contribution = _family_host_array((count,), np.int64)
        cells = _family_host_array((count, len(offsets)), np.int32)
        cells.fill(0)
        for axis in range(len(shape)):
            if budget is not None:
                budget.charge(work=2 * count)
            np.floor_divide(flat, prod(cell_shape[axis + 1 :]), out=coordinate)
            np.remainder(coordinate, cell_shape[axis], out=coordinate)
            for corner, offset in enumerate(offsets):
                if budget is not None:
                    budget.charge(work=3 * count)
                np.add(coordinate, offset[axis], out=contribution)
                np.multiply(contribution, prod(shape[axis + 1 :]), out=contribution)
                np.add(cells[:, corner], contribution, out=cells[:, corner])
        return cells


def _curve_values(control: TransfiniteCurveControl, /) -> np.ndarray:
    curve = control.curve
    if not isinstance(curve, AbstractCurve):
        raise TypeError("A transfinite curve lost its source carrier.")
    first, last = control.parameter_range
    values = np.linspace(first, last, control.intervals + 1, dtype=np.float64)
    if control.reversed:
        values = values[::-1]
    charge_native_geometry_queries(values.size, work_units=values.size)
    return np.asarray(
        curve.evaluate(jnp.asarray(values, dtype=jnp.float64)), dtype=np.float64
    )


def _face_values(control: TransfiniteSurfaceControl, /) -> np.ndarray:
    surface = control.surface
    if not isinstance(surface, AbstractSurfacePatch):
        raise TypeError("A transfinite face lost its source carrier.")
    shape = tuple(n + 1 for n in control.intervals)
    logical = _family_host_array((*shape, 2), np.float64)
    logical[..., 0] = np.linspace(0.0, 1.0, shape[0], dtype=np.float64)[:, None]
    logical[..., 1] = np.linspace(0.0, 1.0, shape[1], dtype=np.float64)[None, :]
    parameters = _family_host_array(logical.shape, np.float64)
    box = np.asarray(control.parameter_box, dtype=np.float64)
    for axis, parameter_axis in enumerate(control.permutation):
        coordinate = (
            1.0 - logical[..., axis] if control.flips[axis] else logical[..., axis]
        )
        parameters[..., parameter_axis] = box[0, parameter_axis] + coordinate * (
            box[1, parameter_axis] - box[0, parameter_axis]
        )
    count = shape[0] * shape[1]
    charge_native_geometry_queries(count, work_units=count)
    return np.asarray(
        surface.evaluate(jnp.asarray(parameters, dtype=jnp.float64)), dtype=np.float64
    )


def _boundary_lattice(block: TransfiniteBlock, /) -> tuple[np.ndarray, np.ndarray, float]:
    shape = tuple(n + 1 for n in block.intervals)
    sampled = []
    for boundary in block.boundaries:
        if isinstance(boundary, TransfiniteCurveControl):
            sampled.append(_curve_values(boundary))
        else:
            sampled.append(_face_values(boundary))
    dimensions = {value.shape[-1] for value in sampled}
    if len(dimensions) != 1:
        raise ValueError("Block boundary ambient dimensions disagree.")
    ambient = sampled[0].shape[-1]
    if ambient < len(shape):
        raise ValueError(
            "Boundary ambient dimension is smaller than the block dimension."
        )
    lattice = _family_host_array((*shape, ambient), np.float64)
    lattice.fill(0.0)
    assigned = _family_host_array(shape, np.bool_)
    assigned.fill(False)
    residual = 0.0
    for face, values in enumerate(sampled):
        axis, side = divmod(face, 2)
        selector = tuple(
            (0 if side == 0 else -1) if i == axis else slice(None)
            for i in range(len(shape))
        )
        existing = assigned[selector]
        differences = np.linalg.norm(lattice[selector] - values, axis=-1)
        mismatch = float(np.max(differences[existing], initial=0.0))
        residual = max(residual, mismatch)
        if mismatch > block.tolerance:
            raise ValueError(
                f"Block {block.name!r} has incompatible boundary charts at face {face}: {mismatch}."
            )
        lattice[selector] = np.where(existing[..., None], lattice[selector], values)
        assigned[selector] = True
    return lattice, assigned, residual


def _transfinite(lattice: np.ndarray, /) -> np.ndarray:
    """Boolean-sum extension: sum faces minus edges plus vertices."""
    shape = lattice.shape[:-1]
    dimension = len(shape)
    budget = current_native_execution_budget()
    with budget.host_workspace() if budget is not None else nullcontext() as workspace:
        # At most the old/new broadcast term coexist. Parameter vectors and
        # their reversed weights are separately included; this is an admitted
        # temporary upper bound, never reported as measured array payload.
        if workspace is not None:
            workspace.set_bound(2 * lattice.nbytes + 16 * sum(shape) + 4096)
        parameters = tuple(np.linspace(0.0, 1.0, n, dtype=np.float64) for n in shape)
        result = _family_host_array(lattice.shape, np.float64)
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
                    if budget is not None:
                        budget.charge(work=result.size)
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


def certify_constructed_mesh(
    mesh: CellMesh,
    /,
    *,
    validity_policy: CellValidityPolicy | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
    geometry: CellGeometrySpec | None = None,
) -> tuple[CellValidityCertificate, GlobalEmbeddingCertificate]:
    """Independent positivity and whole-complex collision acceptance."""
    geometry = CellGeometrySpec.affine(mesh) if geometry is None else geometry
    validity = certify_cell_geometry_validity(geometry, mesh=mesh, policy=validity_policy)
    bad = np.asarray(validity.status) != CellValidityStatus.CERTIFIED_VALID
    if np.any(bad):
        ids = np.concatenate([np.asarray(b.global_ids) for b in mesh.blocks])
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Structured cells have invalid or unresolved mapped Jacobians.",
            stage="structured_validity",
            entity_ids=tuple(int(i) for i in ids[bad]),
        )
    embedding = certify_global_embedding(
        mesh, geometry, validity, limits=certificate_limits
    )
    if embedding.status != "certified":
        raise MeshingFailure(
            MeshingFailureCategory.AUDIT_FAILED,
            "Structured global embedding failed: "
            + "; ".join(finding.check for finding in embedding.findings),
            stage="structured_embedding",
        )
    return validity, embedding


def generate_structured_block(
    block: TransfiniteBlock,
    /,
    *,
    optimization_steps: int = 0,
    validity_policy: CellValidityPolicy | None = None,
    certificate_limits: MeshCertificateLimits | None = None,
) -> StructuredConstruction:
    """Build a pure quad/hex block, with bounded optional native elliptic solve.

    Boundaries never move. Optimizer status reaches the construction evidence;
    regardless of termination, its returned coordinates must independently
    certify. A failed candidate is not silently replaced by the initial map.
    """
    if not isinstance(block, TransfiniteBlock):
        raise TypeError("block must be TransfiniteBlock.")
    if (
        isinstance(optimization_steps, bool)
        or not isinstance(optimization_steps, int)
        or optimization_steps < 0
    ):
        raise ValueError("optimization_steps must be a nonnegative integer.")
    boundary, assigned, residual = _boundary_lattice(block)
    lattice = _transfinite(boundary)
    lattice[assigned] = boundary[assigned]
    shape = lattice.shape[:-1]
    points = lattice.reshape(-1, lattice.shape[-1])
    optimization = None
    free = np.flatnonzero(~assigned.reshape(-1)).astype(np.int32)
    if optimization_steps and free.size:
        method = NewtonTrustRegion()
        budget = current_native_execution_budget()
        edge_count = sum(
            (n - 1) * prod(shape[:axis] + shape[axis + 1 :])
            for axis, n in enumerate(shape)
        )
        if budget is not None:
            # Two objective/three gradient traversals and at most the declared
            # CG actions plus its final model action per Newton step. Final
            # value/gradient publication is separately included in admission.
            budget.admit_work_bound(
                edge_count
                * (optimization_steps * (5 + method.subproblem.maximum_steps + 1) + 2)
            )
        base = jnp.asarray(points, dtype=jnp.float64)
        rows = jnp.asarray(free, dtype=jnp.int32)
        optimization = minimize(
            _elliptic_objective,
            base[rows],
            method=method,
            termination=OptimizationTermination(maximum_steps=optimization_steps),
            args=(base, rows, shape),
        )
        if budget is not None:
            if not optimization.diagnostics.counts_complete:
                raise RuntimeError(
                    "Native structured optimization requires complete measured work counts."
                )
            objectives, gradients, actions = jax.device_get(
                (
                    optimization.diagnostics.objective_evaluations,
                    optimization.diagnostics.gradient_evaluations,
                    optimization.diagnostics.hvp_evaluations,
                )
            )
            budget.charge(
                work=edge_count * (int(objectives) + int(gradients) + int(actions))
            )
        points = np.asarray(base.at[rows].set(optimization.parameters), dtype=np.float64)
    kind = "quadrilateral" if len(shape) == 2 else "hexahedron"
    mesh = CellMesh(points, (CellBlock(block.name, kind, logical_cells(shape)),))
    geometry = CellGeometrySpec.affine(mesh)
    validity, embedding = certify_constructed_mesh(
        mesh,
        geometry=geometry,
        validity_policy=validity_policy,
        certificate_limits=certificate_limits,
    )
    return StructuredConstruction(
        mesh, geometry, shape, validity, embedding, optimization, residual, block.block_id
    )


__all__ = [
    "TransfiniteBlock",
    "StructuredConstruction",
    "logical_cells",
    "generate_structured_block",
]
