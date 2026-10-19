# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Consumer-owned manufactured reaction solves and independent physical oracles."""

from __future__ import annotations

import resource
import sys
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax import NonTrainableState, StrictModule
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementHPTransferPlan,
)
from phydrax.meshing import PhysicalErrorEvidence


# Historical manufactured-reaction conservation and derivative gates; no widening.
CONSERVATION_TOLERANCE = 1.0e-10
TRANSFER_DERIVATIVE_TOLERANCE = 1.0e-10


def exact(points: jax.Array) -> jax.Array:
    return points[..., 0] ** 2


def source(points: jax.Array, args: object) -> jax.Array:
    del args
    return exact(points)


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


@eqx.filter_jit
def solve_problem(compiled: Any) -> tuple[jax.Array, Any]:
    operator, rhs = compiled.linear_system()
    solved = phx.linalg.solve(
        operator, rhs, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    return compiled.expand(solved.value), solved


def compile_and_solve(
    epoch: Any,
) -> tuple[Any, jax.Array, float, float, float]:
    start = perf_counter()
    form = phx.equations.FiniteElementForm(
        "manufactured-reaction",
        "u",
        (
            phx.equations.MassAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(source, coefficient_id="reaction-x-squared"),
            ),
        ),
    )
    space = (
        epoch if isinstance(epoch, FiniteElementDiscretization) else epoch.discretization
    )
    compiled = phx.equations.compile_finite_element_problem(form, space)
    executable = solve_problem.lower(compiled).compile()  # ty: ignore[unresolved-attribute]
    compilation_seconds = perf_counter() - start
    start = perf_counter()
    values, solved = jax.block_until_ready(executable(compiled))
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError(
            "solver_campaign.solution must report native solve success; got unsuccessful status"
        )
    operator, _ = compiled.linear_system()
    matrix = np.asarray(
        jax.vmap(operator.operator.mv)(jnp.eye(values.size, dtype=jnp.float64))
    ).T
    condition = float(np.linalg.cond(matrix))
    return compiled, values, condition, compilation_seconds, perf_counter() - start


def error_and_integral(epoch: Any, values: jax.Array, count: int) -> tuple[float, float]:
    # Independent Gauss/Duffy integration, not the compiler's quadrature ledger.
    axis, weights = np.polynomial.legendre.leggauss(count)
    axis, weights = (axis + 1.0) / 2.0, weights / 2.0
    tensor_points = np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape(
        -1, 2
    )
    tensor_weights = np.outer(weights, weights).reshape(-1)
    squared, integral = jnp.asarray(0.0), jnp.asarray(0.0)
    space = (
        epoch
        if isinstance(epoch, phx.discretization.FiniteElementDiscretization)
        else epoch.discretization
    )
    for block, cell_block in enumerate(space.mesh.blocks):
        if cell_block.cell_kind == "triangle":
            points = jnp.asarray(
                np.column_stack(
                    (
                        tensor_points[:, 0],
                        (1.0 - tensor_points[:, 0]) * tensor_points[:, 1],
                    )
                )
            )
            quadrature = jnp.asarray(tensor_weights * (1.0 - tensor_points[:, 0]))
        elif cell_block.cell_kind == "quadrilateral":
            points, quadrature = jnp.asarray(tensor_points), jnp.asarray(tensor_weights)
        else:
            raise ValueError(
                "solver_campaign.cell_kind must be triangle or quadrilateral; got unsupported cell kind"
            )
        geometry = space.evaluate_block_geometry(
            "u", block, space.default_runtime.coordinates, points, quadrature
        )
        dofs = space.dof_maps[0].cell_dofs[block]
        discrete = phx.ein.contract("ql,cl->cq", geometry.basis_values, values[dofs])
        squared += jnp.sum(
            geometry.physical_weights * (exact(geometry.physical_points) - discrete) ** 2
        )
        integral += jnp.sum(geometry.physical_weights * discrete)
    return float(jnp.sqrt(squared)), float(integral)


def pack(epoch: Any, values: jax.Array, width: int) -> jax.Array:
    result = jnp.zeros((epoch.topology.capacity, width), dtype=jnp.float64)
    offset = 0
    for dofs in epoch.discretization.dof_maps[0].cell_dofs:
        for row in range(dofs.shape[0]):
            slot = int(epoch.active_cell_slots[offset + row])
            result = result.at[slot, : dofs.shape[1]].set(values[dofs[row]])
        offset += dofs.shape[0]
    return result


def unpack(epoch: Any, values: jax.Array) -> jax.Array:
    result = jnp.zeros(
        (epoch.discretization.field_spaces[0].layout.size,), dtype=jnp.float64
    )
    offset = 0
    for dofs in epoch.discretization.dof_maps[0].cell_dofs:
        for row in range(dofs.shape[0]):
            slot = int(epoch.active_cell_slots[offset + row])
            result = result.at[dofs[row]].set(values[slot, : dofs.shape[1]])
        offset += dofs.shape[0]
    return result


def physical(
    epoch: Any, values: jax.Array, estimator: str, count: int
) -> PhysicalErrorEvidence:
    error, _ = error_and_integral(epoch, values, count)
    # This measured total physical error already includes algebraic and refresh
    # defects; adding either again would double-count the same manufactured error.
    return PhysicalErrorEvidence(
        epoch.epoch_id,
        "reaction-L2-error",
        estimator,
        field_error=error,
        geometry_error=0.0,
        algebraic_error=0.0,
        transfer_error=0.0,
    )


class SolvedFieldRefresh(StrictModule, NonTrainableState):
    values: jax.Array

    def __call__(
        self, fields: Any, candidate: Any, args: object
    ) -> tuple[jax.Array, ...]:
        del fields, candidate, args
        return (self.values,)


class HPHistoryTransfer(StrictModule, NonTrainableState):
    transfer: FiniteElementHPTransferPlan

    def __call__(
        self, auxiliary: Any, integrator: Any, candidate: Any, args: object
    ) -> tuple[Any, Any]:
        del candidate, args
        carried_auxiliary = tuple(
            (name, self.transfer.apply_l2_projection(value)) for name, value in auxiliary
        )
        carried_integrator = tuple(
            (name, self.transfer.apply_l2_projection(value)) for name, value in integrator
        )
        return carried_auxiliary, carried_integrator


class HPPhysicalCertifier(StrictModule, NonTrainableState):
    carried: jax.Array
    source_integral: jax.Array
    tolerance: float = eqx.field(static=True)

    def __call__(
        self, epoch: Any, fields: Any, materials: Any, candidate: Any, args: object
    ) -> bool:
        del materials, candidate, args
        carried_integral = error_and_integral(epoch, unpack(epoch, self.carried), 7)[1]
        error, integral = error_and_integral(epoch, unpack(epoch, fields[0]), 7)
        reference = float(self.source_integral)
        return (
            bool(jnp.all(jnp.isfinite(fields[0])))
            and max(
                abs(carried_integral - reference),
                abs(integral - reference),
            )
            < CONSERVATION_TOLERANCE
            and error < self.tolerance
        )


class MetricPhysicalCertifier(StrictModule, NonTrainableState):
    space: FiniteElementDiscretization
    source_integral: jax.Array
    tolerance: float = eqx.field(static=True)

    def __call__(
        self, mesh: Any, fields: Any, materials: Any, lineage: Any, args: object
    ) -> bool:
        del mesh, materials, lineage, args
        error, integral = error_and_integral(self.space, fields[0], 7)
        return (
            error < self.tolerance
            and abs(integral - float(self.source_integral)) < CONSERVATION_TOLERANCE
        )


class MetricPhysicalReanalysis(StrictModule, NonTrainableState):
    space: FiniteElementDiscretization
    revision_id: str = eqx.field(static=True)
    objective_id: str = eqx.field(static=True)

    def __call__(
        self, mesh: Any, fields: Any, materials: Any, args: object
    ) -> PhysicalErrorEvidence:
        del mesh, materials, args
        error, _ = error_and_integral(self.space, fields[0], 7)
        return PhysicalErrorEvidence(
            self.revision_id,
            self.objective_id,
            "independent-accepted-Duffy-7",
            field_error=error,
            geometry_error=0.0,
            algebraic_error=0.0,
            transfer_error=0.0,
        )


def hp_transfer_derivatives(
    transfer: FiniteElementHPTransferPlan, source_values: jax.Array
) -> dict[str, Any]:
    """Check the owner action against an independent NumPy stencil assembly."""
    if transfer.l2_projection is None:
        raise RuntimeError(
            "solver_campaign.projection must retain prepared L2 weights; got None"
        )
    projections = np.asarray(transfer.l2_projection)
    target_width, source_width = projections.shape[1:]
    matrix = np.zeros(
        (
            transfer.target_capacity * target_width,
            transfer.source_capacity * source_width,
        ),
        dtype=np.float64,
    )
    for route in np.flatnonzero(np.asarray(transfer.valid)):
        source_count = int(transfer.source_dof_count[route])
        target_count = int(transfer.target_dof_count[route])
        rows = int(transfer.target_slots[route]) * target_width + np.arange(target_count)
        columns = int(transfer.source_slots[route]) * source_width + np.arange(
            source_count
        )
        matrix[np.ix_(rows, columns)] += projections[route, :target_count, :source_count]
    direction = jnp.linspace(-0.2, 0.3, source_values.size).reshape(source_values.shape)
    dual = jnp.linspace(-0.4, 0.7, matrix.shape[0]).reshape(
        transfer.target_capacity, target_width
    )
    _, tangent = jax.jvp(transfer.apply_l2_projection, (source_values,), (direction,))
    _, pullback = jax.vjp(transfer.apply_l2_projection, source_values)
    adjoint = pullback(dual)[0]
    jvp_error = float(
        np.max(
            np.abs(
                np.asarray(tangent).reshape(-1)
                - matrix @ np.asarray(direction).reshape(-1)
            )
        )
    )
    vjp_error = float(
        np.max(
            np.abs(
                np.asarray(adjoint).reshape(-1) - matrix.T @ np.asarray(dual).reshape(-1)
            )
        )
    )
    duality_error = float(jnp.abs(jnp.vdot(dual, tangent) - jnp.vdot(adjoint, direction)))
    if max(jvp_error, vjp_error, duality_error) > TRANSFER_DERIVATIVE_TOLERANCE:
        raise RuntimeError(
            "solver_campaign.derivatives must agree with the independent stencil; got excessive derivative defect"
        )
    return {
        "JVP_error": jvp_error,
        "VJP_error": vjp_error,
        "duality_error": duality_error,
        "scope": "fixed prepared L2 field-transfer action; p/h acceptance remains a stopped event",
    }
