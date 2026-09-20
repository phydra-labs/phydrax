#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._fd_precision import FDExecutionPrecisionPolicy
from ._core import (
    BlockHierarchyPlan,
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockLevelState,
)
from ._fd_halo import (
    FDAMRFillPatchPlan,
    FDAMRFillPatchResult,
    FDAMRFillPatchWorkspace,
    FDAMRPhysicalBoundaryRequest,
)
from ._fd_transfer import AMREntityTransferPlan
from ._topology_compiler import (
    BlockTopologyCompiler,
    BlockTopologyCompileResult,
)
from ._topology_transfer import BlockFieldTopologyTransition


def _validate_level_state_precision(
    state: BlockLevelState,
    precision: FDExecutionPrecisionPolicy,
    /,
) -> None:
    expected = jnp.dtype(precision.field_dtype)
    if state.values.dtype != expected:
        raise TypeError(
            f"FD AMR state has dtype {state.values.dtype}; expected {expected}."
        )


class FDAMRHierarchyPlan(StrictModule, NonTrainableState):
    """Cell-centered FD topology, transition, and FillPatch preparation policy."""

    hierarchy: BlockHierarchyPlan
    transfers: tuple[AMREntityTransferPlan, ...]
    tag_buffer: int = eqx.field(static=True)
    proper_nesting: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    precision: FDExecutionPrecisionPolicy

    def __init__(
        self,
        hierarchy: BlockHierarchyPlan,
        transfers: Sequence[AMREntityTransferPlan] | None = None,
        /,
        *,
        tag_buffer: int = 0,
        proper_nesting: int = 0,
        precision: FDExecutionPrecisionPolicy | None = None,
    ):
        precision_ = FDExecutionPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, FDExecutionPrecisionPolicy):
            raise TypeError("precision must be an FDExecutionPrecisionPolicy.")
        if not isinstance(hierarchy, BlockHierarchyPlan):
            raise TypeError("FD AMR hierarchy requires BlockHierarchyPlan.")
        dimension = len(hierarchy.grid.shape)
        transfers_ = (
            tuple(
                AMREntityTransferPlan.cells(
                    dimension, hierarchy.levels[level].refinement_ratio
                )
                for level in range(len(hierarchy.levels) - 1)
            )
            if transfers is None
            else tuple(transfers)
        )
        if len(transfers_) != len(hierarchy.levels) - 1 or not all(
            isinstance(transfer, AMREntityTransferPlan) for transfer in transfers_
        ):
            raise ValueError("FD AMR requires one entity transfer per level transition.")
        if any(
            transfer.axis_entities != ("interval",) * dimension for transfer in transfers_
        ):
            raise NotImplementedError(
                "Prepared FD AMR is cell-centered; AMREntityTransferPlan remains the explicit non-cell extension seam."
            )
        if any(
            transfer.refinement_ratio != hierarchy.levels[level].refinement_ratio
            for level, transfer in enumerate(transfers_)
        ):
            raise ValueError("FD AMR transfer ratios must match hierarchy levels.")
        buffer_ = int(tag_buffer)
        nesting = int(proper_nesting)
        if buffer_ < 0 or nesting < 0:
            raise ValueError("Tag buffer and proper nesting must be non-negative.")
        self.hierarchy = hierarchy
        self.transfers = transfers_
        self.tag_buffer = buffer_
        self.proper_nesting = nesting
        self.precision = precision_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fd-amr-hierarchy-plan",
                "hierarchy": hierarchy.plan_id,
                "transfers": [value.transfer_id for value in transfers_],
                "tag_buffer": buffer_,
                "proper_nesting": nesting,
                "precision": precision_.policy_id,
            }
        )

    def prepare(self, /) -> "PreparedFDAMRHierarchy":
        return PreparedFDAMRHierarchy(self)


class PreparedFDAMRHierarchy(StrictModule, NonTrainableState):
    """Prepared AMR topology compiler, conservative transition, and FillPatch runtime.

    Solver time advancement is deliberately absent.  Time values enter only to
    interpolate already supplied coarse old/new states during FillPatch.
    """

    plan: FDAMRHierarchyPlan
    topology_compiler: BlockTopologyCompiler
    prepared_id: str = eqx.field(static=True)

    def __init__(self, plan: FDAMRHierarchyPlan, /):
        if not isinstance(plan, FDAMRHierarchyPlan):
            raise TypeError("plan must be FDAMRHierarchyPlan.")
        compiler = BlockTopologyCompiler(
            plan.hierarchy,
            tag_buffer=plan.tag_buffer,
            proper_nesting=plan.proper_nesting,
        )
        self.plan = plan
        self.topology_compiler = compiler
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-fd-amr-hierarchy",
                "plan": plan.plan_id,
                "topology_compiler": compiler.compiler_id,
            }
        )

    @property
    def precision_evidence(self):
        return self.plan.precision.evidence()

    def initial_topology(self, /) -> BlockHierarchyTopology:
        return self.topology_compiler.initial_topology()

    def compile_topology(
        self,
        source: BlockHierarchyTopology,
        block_tags: Sequence[ArrayLike],
        /,
    ) -> BlockTopologyCompileResult:
        return self.topology_compiler.compile(source, block_tags)

    def field_transition(
        self,
        source: BlockHierarchyTopology,
        target: BlockHierarchyTopology,
        field_name: str,
        /,
        *,
        component_shape: Sequence[int] = (),
        dtype=None,
    ) -> BlockFieldTopologyTransition:
        dtype_ = self.plan.precision.field_dtype if dtype is None else dtype
        return BlockFieldTopologyTransition(
            source,
            target,
            field_name,
            component_shape=component_shape,
            dtype=dtype_,
        )

    def prepare_fill_patch(
        self,
        topology: BlockHierarchyTopology,
        /,
    ) -> tuple[FDAMRFillPatchPlan, ...]:
        if not isinstance(topology, BlockHierarchyTopology) or (
            topology.plan.plan_id != self.plan.hierarchy.plan_id
        ):
            raise ValueError("FillPatch topology does not match the prepared hierarchy.")
        return tuple(
            FDAMRFillPatchPlan(
                topology,
                level,
                None if level == 0 else self.plan.transfers[level - 1],
            )
            for level in range(len(self.plan.hierarchy.levels))
        )

    def fill_patch(
        self,
        state: BlockHierarchyState,
        /,
        *,
        coarse_old: BlockHierarchyState | None = None,
        coarse_new: BlockHierarchyState | None = None,
        coarse_old_time: ArrayLike = 0.0,
        coarse_new_time: ArrayLike = 0.0,
        fill_time: ArrayLike = 0.0,
        physical_boundary_values: Sequence[ArrayLike | None] | None = None,
    ) -> FDAMRFillPatchResult:
        if not isinstance(state, BlockHierarchyState) or (
            state.topology.plan.plan_id != self.plan.hierarchy.plan_id
        ):
            raise ValueError("FD AMR hierarchy state does not match the prepared plan.")
        old = state if coarse_old is None else coarse_old
        new = state if coarse_new is None else coarse_new
        for hierarchy_state in (state, old, new):
            for level in hierarchy_state.levels:
                _validate_level_state_precision(level, self.plan.precision)
        boundaries = (
            (None,) * len(state.levels)
            if physical_boundary_values is None
            else tuple(physical_boundary_values)
        )
        if len(boundaries) != len(state.levels):
            raise ValueError("Physical boundary values require one entry per AMR level.")
        plans = self.prepare_fill_patch(state.topology)
        executed = tuple(
            fill.execute(
                state,
                old,
                new,
                coarse_old_time,
                coarse_new_time,
                fill_time,
                boundary,
            )
            for fill, boundary in zip(plans, boundaries, strict=True)
        )
        workspaces: tuple[FDAMRFillPatchWorkspace, ...] = tuple(
            value[0] for value in executed
        )
        requests: tuple[FDAMRPhysicalBoundaryRequest, ...] = tuple(
            value[1] for value in executed
        )
        complete_by_level = []
        for level, workspace in zip(state.topology.levels, workspaces, strict=True):
            active = level.active.reshape(
                (level.active.shape[0],) + (1,) * (workspace.valid.ndim - 1)
            )
            complete_by_level.append(jnp.all(workspace.valid | ~active))
        complete = jnp.all(jnp.stack(tuple(complete_by_level)))
        return FDAMRFillPatchResult(
            workspaces=workspaces,
            physical_boundary_requests=requests,
            complete=complete,
            result_id=canonical_fingerprint(
                {
                    "kind": "fd-amr-fill-patch-result",
                    "prepared": self.prepared_id,
                    "epoch": state.topology.epoch.epoch_id,
                    "plans": [value.plan_id for value in plans],
                    "physical_values_supplied": [
                        value is not None for value in boundaries
                    ],
                }
            ),
        )

    def validate_stencil_footprints(
        self,
        footprints: Sequence[Any],
        /,
    ) -> tuple[tuple[int, ...], ...]:
        """Require every level halo to contain its exact stencil read reach."""
        from ..finite_difference._stencil import StencilFootprint

        values = tuple(footprints)
        if len(values) != len(self.plan.hierarchy.levels) or not all(
            isinstance(value, StencilFootprint) for value in values
        ):
            raise ValueError("One StencilFootprint is required per AMR level.")
        required = []
        for level, footprint in zip(self.plan.hierarchy.levels, values, strict=True):
            if footprint.axis_names != self.plan.hierarchy.grid.axis_names:
                raise ValueError("Stencil footprint axes do not match AMR geometry.")
            reach = tuple(
                max(lower, upper)
                for lower, upper in zip(footprint.lower, footprint.upper, strict=True)
            )
            if any(
                available < needed
                for available, needed in zip(level.halo_width, reach, strict=True)
            ):
                raise ValueError(
                    "AMR block halo is smaller than the stencil read footprint."
                )
            required.append(reach)
        return tuple(required)


__all__ = [
    "FDAMRHierarchyPlan",
    "PreparedFDAMRHierarchy",
]
