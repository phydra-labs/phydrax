#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    AbstractLinearOperator,
    AbstractPreconditioner,
    AbstractPreconditionerBuilder,
    AbstractVectorSpace,
    ArraySpace,
    DenseInversePreconditionerBuilder,
    DenseLinearOperator,
    MaterializationPolicy,
    PreconditionerCostEstimate,
    PreconditionerProperties,
)
from ._generic import FiniteElementDiscretization


class FiniteElementPatchPlan(StrictModule, NonTrainableState):
    gathers: Array
    valid: Array
    partition_weights: Array
    global_size: int = eqx.field(static=True)
    patch_id: str = eqx.field(static=True)
    partition_residual: float = eqx.field(static=True)

    def __init__(
        self,
        gathers: ArrayLike,
        valid: ArrayLike,
        partition_weights: ArrayLike,
        global_size: int,
        /,
    ):
        routes = jnp.asarray(gathers, dtype=jnp.int32)
        valid_ = jnp.asarray(valid, dtype=bool)
        weights = jnp.asarray(partition_weights)
        size = int(global_size)
        if (
            routes.ndim != 2
            or valid_.shape != routes.shape
            or weights.shape != routes.shape
            or routes.shape[0] == 0
            or routes.shape[1] == 0
        ):
            raise ValueError(
                "Patch gather, validity, and weight layouts are incompatible."
            )
        routes_host = np.asarray(routes)
        valid_host = np.asarray(valid_)
        weights_host = np.asarray(weights)
        if not np.issubdtype(weights_host.dtype, np.inexact):
            raise TypeError("Patch partition weights must have an inexact dtype.")
        if size <= 0 or np.any(valid_host & ((routes_host < 0) | (routes_host >= size))):
            raise ValueError("Patch routes/global size are invalid.")
        if np.any(
            np.where(
                valid_host,
                ~np.isfinite(weights_host) | (weights_host <= 0.0),
                weights_host != 0.0,
            )
        ):
            raise ValueError(
                "Patch partition weights must be finite and positive only on valid routes."
            )
        partition = np.zeros((size,), dtype=weights_host.dtype)
        np.add.at(partition, routes_host[valid_host], weights_host[valid_host])
        residual = float(np.max(np.abs(partition - 1.0)))
        tolerance = 32.0 * np.finfo(weights_host.dtype).eps
        if not np.isfinite(residual) or residual > tolerance:
            raise ValueError(
                "Patch weights do not form a partition of unity over the global space."
            )
        self.gathers = routes
        self.valid = valid_
        self.partition_weights = weights
        self.global_size = size
        self.partition_residual = residual
        self.patch_id = canonical_fingerprint(
            {
                "kind": "finite-element-one-ring-patch-plan",
                "gathers": array_tree_fingerprint(routes_host),
                "valid": array_tree_fingerprint(valid_host),
                "partition_weights": array_tree_fingerprint(weights_host),
                "global_size": size,
            }
        )


def one_ring_patch_plan(
    discretization: FiniteElementDiscretization,
    field_name: str,
    /,
) -> FiniteElementPatchPlan:
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    field_index = discretization._field_index(field_name)
    dof_map = discretization.dof_maps[field_index]
    cell_count = sum(block.cell_count for block in discretization.mesh.blocks)
    block_by_cell = []
    local_by_cell = []
    for block_index, block in enumerate(discretization.mesh.blocks):
        block_by_cell.extend((block_index,) * block.cell_count)
        local_by_cell.extend(range(block.cell_count))
    neighbours = [set((cell,)) for cell in range(cell_count)]
    for owner, neighbour in zip(
        np.asarray(discretization.interior_facet_domain.owner_cells),
        np.asarray(discretization.interior_facet_domain.neighbour_cells),
        strict=True,
    ):
        neighbours[int(owner)].add(int(neighbour))
        neighbours[int(neighbour)].add(int(owner))
    patches = []
    for cells in neighbours:
        dofs = set()
        for cell in sorted(cells):
            block = block_by_cell[cell]
            local = local_by_cell[cell]
            dofs.update(np.asarray(dof_map.cell_dofs[block][local]).tolist())
        patches.append(tuple(sorted(dofs)))
    width = max(len(patch) for patch in patches)
    routes = np.zeros((cell_count, width), dtype=np.int32)
    valid = np.zeros((cell_count, width), dtype=bool)
    for patch, values in enumerate(patches):
        routes[patch, : len(values)] = values
        valid[patch, : len(values)] = True
    coverage = np.zeros((dof_map.global_dof_count,), dtype=np.int32)
    np.add.at(coverage, routes[valid], 1)
    weights = np.zeros_like(
        routes, dtype=np.asarray(discretization.mesh.coordinates).dtype
    )
    weights[valid] = 1.0 / coverage[routes[valid]]
    return FiniteElementPatchPlan(
        routes,
        valid,
        weights,
        dof_map.global_dof_count,
    )


class FiniteElementPatchPreconditioner(AbstractPreconditioner):
    plan: FiniteElementPatchPlan
    local_inverse: Array | None
    local_solvers: tuple[AbstractPreconditioner, ...]
    builder_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        plan: FiniteElementPatchPlan,
        local_action: ArrayLike | tuple[AbstractPreconditioner, ...],
        space: AbstractVectorSpace,
        /,
        *,
        builder_id: str | None = None,
    ):
        if not isinstance(plan, FiniteElementPatchPlan):
            raise TypeError("plan must be FiniteElementPatchPlan.")
        if not isinstance(space, AbstractVectorSpace):
            raise TypeError("space must be an AbstractVectorSpace.")
        width = plan.gathers.shape[1]
        if isinstance(local_action, tuple):
            solvers = tuple(local_action)
            if len(solvers) != plan.gathers.shape[0] or not all(
                isinstance(solver, AbstractPreconditioner) for solver in solvers
            ):
                raise ValueError(
                    "Patch local actions must contain one preconditioner per patch."
                )
            patch_sizes = tuple(
                int(value) for value in np.sum(np.asarray(plan.valid), axis=1)
            )
            if any(
                solver.space.size != patch_size
                for solver, patch_size in zip(solvers, patch_sizes, strict=True)
            ):
                raise ValueError("Patch local preconditioners act on the wrong spaces.")
            inverse = None
        else:
            inverse = jnp.asarray(local_action)
            if inverse.shape != plan.gathers.shape + (width,):
                raise ValueError("Patch inverse matrices have an incompatible shape.")
            solvers = ()
        if space.size != plan.global_size:
            raise ValueError("Patch preconditioner space does not match patch plan.")
        identifier = None if builder_id is None else str(builder_id)
        if identifier is not None and not identifier:
            raise ValueError("builder_id must be non-empty when supplied.")
        self.plan = plan
        self.local_inverse = inverse
        self.local_solvers = solvers
        self.builder_id = identifier
        self.space = space
        linear = (
            all(solver.properties.certifies("linear") for solver in solvers)
            if solvers
            else True
        )
        stationary = (
            all(solver.properties.certifies("stationary") for solver in solvers)
            if solvers
            else True
        )
        self.properties = PreconditionerProperties(
            linear=linear,
            stationary=stationary,
            evidence={
                name: "transformed"
                for name, certified in {
                    "linear": linear,
                    "stationary": stationary,
                }.items()
                if certified
            },
        )
        local_identity = (
            array_tree_fingerprint(np.asarray(inverse))
            if inverse is not None
            else {
                "identifiers": [solver.preconditioner_id for solver in solvers],
                "numeric": array_tree_fingerprint(solvers),
            }
        )
        self.preconditioner_id = canonical_fingerprint(
            {
                "kind": "finite-element-one-ring-preconditioner",
                "patch": plan.patch_id,
                "space": space.space_id,
                "local_actions": local_identity,
                "builder": identifier,
            }
        )

    def apply(
        self,
        residual: PyTree,
        /,
        *,
        iteration: ArrayLike | None = None,
    ):
        coordinates = self.space.flatten(self.space.validate(residual))
        safe_routes = jnp.maximum(self.plan.gathers, 0)
        local = coordinates[safe_routes]
        local = jnp.where(self.plan.valid, local, 0.0)
        if self.local_inverse is not None:
            correction = ein.contract("pij,pj->pi", self.local_inverse, local)
        else:
            local_corrections = []
            valid_host = np.asarray(self.plan.valid)
            for index, solver in enumerate(self.local_solvers):
                active = valid_host[index]
                solved = solver.apply(local[index, active], iteration=iteration)
                padded = (
                    jnp.zeros(
                        (self.plan.gathers.shape[1],),
                        dtype=coordinates.dtype,
                    )
                    .at[active]
                    .set(solved)
                )
                local_corrections.append(padded)
            correction = jnp.stack(tuple(local_corrections))
        correction = jnp.where(
            self.plan.valid,
            correction * self.plan.partition_weights,
            0.0,
        )
        assembled = (
            jnp.zeros((self.plan.global_size,), dtype=correction.dtype)
            .at[safe_routes]
            .add(correction)
        )
        return self.space.unflatten(assembled)


def _setup_operator(
    setup_operator: AbstractLinearOperator,
    plan: FiniteElementPatchPlan,
    /,
) -> AbstractLinearOperator:
    if not isinstance(setup_operator, AbstractLinearOperator):
        raise TypeError("setup_operator must be an AbstractLinearOperator.")
    if (
        setup_operator.batch_shape
        or not setup_operator.source.compatible(setup_operator.target)
        or setup_operator.source.size != plan.global_size
    ):
        raise ValueError(
            "Patch setup requires an unbatched endomorphism on the plan space."
        )
    return setup_operator


def _patch_local_operators(
    plan: FiniteElementPatchPlan,
    setup_operator: AbstractLinearOperator,
    /,
) -> tuple[AbstractLinearOperator, ...]:
    operator = _setup_operator(setup_operator, plan)
    dtype = jnp.result_type(
        *[leaf.dtype for leaf in jax.tree.leaves(operator.source.structure())]
    )
    routes = np.asarray(plan.gathers)
    valid = np.asarray(plan.valid)
    operators = []
    for patch_index in range(routes.shape[0]):
        patch_routes = routes[patch_index, valid[patch_index]]
        local_space = ArraySpace((patch_routes.size,), dtype=dtype)
        restriction = np.zeros(
            (patch_routes.size, plan.global_size),
            dtype=np.dtype(dtype),
        )
        restriction[np.arange(patch_routes.size), patch_routes] = 1.0
        restriction_operator = DenseLinearOperator(
            restriction,
            source=operator.source,
            target=local_space,
            operator_id=f"patch-restriction/{plan.patch_id}/{patch_index}",
        )
        extension_operator = DenseLinearOperator(
            restriction.T,
            source=local_space,
            target=operator.source,
            operator_id=f"patch-extension/{plan.patch_id}/{patch_index}",
        )
        operators.append(restriction_operator @ operator @ extension_operator)
    return tuple(operators)


class FiniteElementPatchPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare weighted one-ring restricted additive Schwarz from one operator."""

    plan: FiniteElementPatchPlan
    local_solver: AbstractPreconditionerBuilder
    properties: PreconditionerProperties | None
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FiniteElementPatchPlan,
        /,
        *,
        local_solver: AbstractPreconditionerBuilder | None = None,
        properties: PreconditionerProperties | None = None,
    ):
        if not isinstance(plan, FiniteElementPatchPlan):
            raise TypeError("plan must be a FiniteElementPatchPlan.")
        solver = (
            DenseInversePreconditionerBuilder() if local_solver is None else local_solver
        )
        if not isinstance(solver, AbstractPreconditionerBuilder):
            raise TypeError("local_solver must be an AbstractPreconditionerBuilder.")
        if properties is not None and not isinstance(
            properties, PreconditionerProperties
        ):
            raise TypeError("properties must be PreconditionerProperties or None.")
        self.plan = plan
        self.local_solver = solver
        self.properties = properties
        self._builder_id = canonical_fingerprint(
            {
                "kind": "finite-element-one-ring-preconditioner-builder",
                "patch": plan.patch_id,
                "local_solver": solver.builder_id,
                "properties": None
                if properties is None
                else {
                    "linear": properties.linear,
                    "stationary": properties.stationary,
                    "self_adjoint": properties.self_adjoint,
                    "positive_definite": properties.positive_definite,
                },
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> str:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        local_operators = _patch_local_operators(self.plan, setup_operator)
        local_properties = tuple(
            self.local_solver.properties_for(operator) for operator in local_operators
        )
        linear = all(value.certifies("linear") for value in local_properties)
        stationary = all(value.certifies("stationary") for value in local_properties)
        if self.properties is not None:
            if (
                (self.properties.linear and not linear)
                or (self.properties.stationary and not stationary)
                or self.properties.self_adjoint
                or self.properties.positive_definite
            ):
                raise ValueError(
                    "Weighted restricted Schwarz cannot certify the supplied "
                    "preconditioner properties."
                )
            return self.properties
        return PreconditionerProperties(
            linear=linear,
            stationary=stationary,
            evidence={
                name: "transformed"
                for name, certified in {
                    "linear": linear,
                    "stationary": stationary,
                }.items()
                if certified
            },
        )

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        local_operators = _patch_local_operators(self.plan, setup_operator)
        self.properties_for(setup_operator)
        estimates = tuple(
            self.local_solver.cost_for(
                operator,
                materialization=materialization,
            )
            for operator in local_operators
        )
        rejected = tuple(
            f"patch {index}: {estimate.reason}"
            for index, estimate in enumerate(estimates)
            if not estimate.accepted
        )
        itemsize = np.dtype(self.plan.partition_weights.dtype).itemsize
        route_storage = int(
            self.plan.gathers.nbytes
            + self.plan.valid.nbytes
            + self.plan.partition_weights.nbytes
        )
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=route_storage
            + sum(estimate.storage_bytes for estimate in estimates),
            preparation_workspace_bytes=sum(
                estimate.preparation_workspace_bytes for estimate in estimates
            ),
            apply_workspace_bytes_per_rhs=(
                2 * self.plan.gathers.size * itemsize
                + self.plan.global_size * itemsize
                + sum(estimate.apply_workspace_bytes_per_rhs for estimate in estimates)
            ),
            setup_matvec_count=sum(estimate.setup_matvec_count for estimate in estimates),
            accepted=not rejected,
            reason=(
                "weighted one-ring Schwarz local actions"
                if not rejected
                else "; ".join(rejected)
            ),
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        estimate = self.cost_for(
            setup_operator,
            materialization=materialization,
        )
        if not estimate.accepted:
            raise ValueError(f"Patch preconditioner is ineligible: {estimate.reason}.")
        local_operators = _patch_local_operators(self.plan, setup_operator)
        solvers = tuple(
            self.local_solver.prepare(operator, materialization=materialization)
            for operator in local_operators
        )
        return FiniteElementPatchPreconditioner(
            self.plan,
            solvers,
            setup_operator.source,
            builder_id=self.builder_id,
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, FiniteElementPatchPreconditioner):
            raise TypeError("Patch refresh requires a FiniteElementPatchPreconditioner.")
        if preconditioner.builder_id != self.builder_id:
            raise ValueError("Patch refresh must preserve the builder identity.")
        if not preconditioner.local_solvers:
            raise ValueError("Patch refresh requires builder-prepared local actions.")
        local_operators = _patch_local_operators(self.plan, setup_operator)
        solvers = tuple(
            self.local_solver.refresh(
                solver,
                operator,
                materialization=materialization,
            )
            for solver, operator in zip(
                preconditioner.local_solvers,
                local_operators,
                strict=True,
            )
        )
        return FiniteElementPatchPreconditioner(
            self.plan,
            solvers,
            setup_operator.source,
            builder_id=self.builder_id,
        )


__all__ = [
    "FiniteElementPatchPlan",
    "FiniteElementPatchPreconditioner",
    "FiniteElementPatchPreconditionerBuilder",
    "one_ring_patch_plan",
]
