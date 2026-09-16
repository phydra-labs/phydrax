#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Static-capacity AMR transfers and atomic topology epochs for NR systems."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import StructuredCochainBridge
from ...discretization.amr import (
    AMREntityTransferPlan,
    BlockFieldTopologyTransition,
    BlockHierarchyState,
    BlockHierarchyTopology,
    BlockLevelState,
    BlockTopologyCompileResult,
    FDAMRFillPatchPlan,
    FDAMRPhysicalBoundaryRequest,
    FluxRegister,
    PreparedFDAMRHierarchy,
)
from ...discretization.finite_volume import BlockAMRConservationPlan
from ...solver._block_amr_runtime import AMRTimeSchedulePlan
from ...solver._mhd_amr import (
    ConstrainedMHDAMRSynchronizationPlan,
    DivergenceFreeMagneticTransferPlan,
    ElectromotiveForceRegister,
    MagneticAMRTransferDiagnostics,
)
from ._distributed import (
    _formulation,
    formulation_field_names,
    NumericalRelativityFormulation,
    NumericalRelativityOwnership,
    PreparedNumericalRelativityAMRDistribution,
)
from ._state import pack_symmetric, Z4cState


class AMRTransferBinding(StrictModule, NonTrainableState):
    """Exact content/topology/field authority for one AMR transfer result."""

    source_content_id: str = eqx.field(static=True)
    target_content_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    source_epoch_id: str = eqx.field(static=True)
    target_epoch_id: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    formulation: NumericalRelativityFormulation = eqx.field(static=True)
    transfer_plan_id: str = eqx.field(static=True)
    transition_plan_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_content: Any,
        target_content: Any,
        /,
        *,
        source_topology_id: str,
        target_topology_id: str,
        source_epoch_id: str,
        target_epoch_id: str,
        field_name: str,
        formulation: NumericalRelativityFormulation,
        transfer_plan_id: str,
        transition_plan_id: str,
    ):
        identifiers = tuple(
            str(value).strip()
            for value in (
                source_topology_id,
                target_topology_id,
                source_epoch_id,
                target_epoch_id,
                field_name,
                transfer_plan_id,
                transition_plan_id,
            )
        )
        formulation_ = _formulation(formulation)
        if any(not value for value in identifiers):
            raise ValueError("AMR transfer binding identities must be non-empty.")
        source_id = array_tree_fingerprint(source_content)
        target_id = array_tree_fingerprint(target_content)
        (
            source_topology,
            target_topology,
            source_epoch,
            target_epoch,
            field,
            transfer_plan,
            transition_plan,
        ) = identifiers
        self.source_content_id = source_id
        self.target_content_id = target_id
        self.source_topology_id = source_topology
        self.target_topology_id = target_topology
        self.source_epoch_id = source_epoch
        self.target_epoch_id = target_epoch
        self.field_name = field
        self.formulation = formulation_
        self.transfer_plan_id = transfer_plan
        self.transition_plan_id = transition_plan
        self.binding_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-amr-transfer-binding",
                "source_content": source_id,
                "target_content": target_id,
                "source_topology": source_topology,
                "target_topology": target_topology,
                "source_epoch": source_epoch,
                "target_epoch": target_epoch,
                "field": field,
                "formulation": formulation_,
                "transfer_plan": transfer_plan,
                "transition_plan": transition_plan,
            }
        )


class Z4cAMRTransferEvidence(StrictModule):
    determinant_constraint_before: Array
    determinant_constraint_after: Array
    trace_free_constraint_before: Array
    trace_free_constraint_after: Array
    finite: Array
    physically_valid: Array
    constraint_valid: Array
    qualified: Array
    derivative_valid: Array
    transfer_id: str = eqx.field(static=True)
    binding: AMRTransferBinding | None = None


class RelativisticMaterialTransferEvidence(StrictModule):
    source_content: Array
    target_content: Array
    conservation_residual: Array
    finite: Array
    physically_valid: Array
    conservation_valid: Array
    qualified: Array
    derivative_valid: Array
    transfer_id: str = eqx.field(static=True)
    binding: AMRTransferBinding | None = None


class RelativisticRadiationTransferEvidence(StrictModule):
    source_content: Array
    target_content: Array
    conservation_residual: Array
    finite: Array
    physically_valid: Array
    conservation_valid: Array
    qualified: Array
    derivative_valid: Array
    transfer_id: str = eqx.field(static=True)
    binding: AMRTransferBinding | None = None


class RelativisticMagneticTransferEvidence(StrictModule):
    divergence_before: Array
    divergence_after: Array
    divergence_change: Array
    flux_defect: Array
    finite: Array
    divergence_valid: Array
    qualified: Array
    derivative_valid: Array
    transfer_id: str = eqx.field(static=True)
    binding: AMRTransferBinding | None = None


class NumericalRelativityAMRHaloWorkspace(StrictModule):
    values: Array
    valid: Array
    source_class: Array
    finite: Array
    field_name: str = eqx.field(static=True)
    workspace_id: str = eqx.field(static=True)


class NumericalRelativityAMRHaloPlan(StrictModule, NonTrainableState):
    """Formulation-aware view over topology-prepared fixed-capacity FillPatch."""

    fill_patch: FDAMRFillPatchPlan
    field_name: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: BlockHierarchyTopology,
        level: int,
        field_name: str,
        /,
        *,
        transfer: AMREntityTransferPlan | None = None,
    ):
        if not isinstance(topology, BlockHierarchyTopology):
            raise TypeError("NR AMR halo planning requires BlockHierarchyTopology.")
        name = str(field_name)
        if name == "z4c":
            components = (25,)
        elif name == "material":
            components = (5,)
        elif name == "radiation":
            components = (4,)
        else:
            raise ValueError(
                "NR cell FillPatch supports 'z4c', 'material', or 'radiation'."
            )
        level_ = int(level)
        transfer_ = transfer
        if level_ > 0 and transfer_ is None:
            transfer_ = AMREntityTransferPlan.cells(
                len(topology.plan.grid.shape),
                topology.plan.levels[level_ - 1].refinement_ratio,
            )
        fill_patch = FDAMRFillPatchPlan(topology, level_, transfer_)
        self.fill_patch = fill_patch
        self.field_name = name
        self.component_shape = components
        self.plan_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-amr-fill-patch",
                "field": name,
                "fill_patch": fill_patch.plan_id,
            }
        )

    def execute(
        self,
        state: BlockHierarchyState,
        coarse_old: BlockHierarchyState,
        coarse_new: BlockHierarchyState,
        coarse_old_time: ArrayLike,
        coarse_new_time: ArrayLike,
        fill_time: ArrayLike,
        physical_boundary_values: ArrayLike | None = None,
        /,
    ) -> tuple[
        NumericalRelativityAMRHaloWorkspace,
        FDAMRPhysicalBoundaryRequest,
    ]:
        for value in (state, coarse_old, coarse_new):
            if not isinstance(value, BlockHierarchyState):
                raise TypeError("NR FillPatch inputs must be BlockHierarchyState.")
            level = value.levels[self.fill_patch.level]
            rank = 1 + len(level.plan.block_shape)
            if level.values.shape[rank:] != self.component_shape:
                raise ValueError(
                    "NR FillPatch hierarchy payload has the wrong field components."
                )
        workspace, request = self.fill_patch.execute(
            state,
            coarse_old,
            coarse_new,
            coarse_old_time,
            coarse_new_time,
            fill_time,
            physical_boundary_values,
        )
        result = NumericalRelativityAMRHaloWorkspace(
            workspace.values,
            workspace.valid,
            workspace.source_class,
            jnp.all(jnp.isfinite(workspace.values)),
            self.field_name,
            canonical_fingerprint(
                {
                    "kind": "numerical-relativity-amr-fill-patch-workspace",
                    "plan": self.plan_id,
                    "workspace": workspace.workspace_id,
                }
            ),
        )
        return result, request


TransferEvidence = (
    Z4cAMRTransferEvidence
    | RelativisticMaterialTransferEvidence
    | RelativisticRadiationTransferEvidence
    | RelativisticMagneticTransferEvidence
)


def _determinant_3x3(metric: Array, /) -> Array:
    a, b, c = metric[0, 0], metric[0, 1], metric[0, 2]
    d, e, f = metric[1, 1], metric[1, 2], metric[2, 2]
    return a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d)


def _inverse_symmetric_3x3(metric: Array, determinant: Array, /) -> Array:
    a, b, c = metric[0, 0], metric[0, 1], metric[0, 2]
    d, e, f = metric[1, 1], metric[1, 2], metric[2, 2]
    safe = jnp.where(
        jnp.abs(determinant) > jnp.finfo(metric.dtype).tiny, determinant, 1.0
    )
    inverse = jnp.stack(
        (
            d * f - e * e,
            c * e - b * f,
            b * e - c * d,
            c * e - b * f,
            a * f - c * c,
            b * c - a * e,
            b * e - c * d,
            b * c - a * e,
            a * d - b * b,
        ),
        axis=0,
    ).reshape((3, 3) + metric.shape[2:])
    return inverse / safe[None, None, ...]


def _z4c_algebraic_residuals(state: Z4cState, /) -> tuple[Array, Array, Array]:
    metric = state.conformal_metric
    determinant = _determinant_3x3(metric)
    inverse = _inverse_symmetric_3x3(metric, determinant)
    trace = ein.contract(
        "ij...,ij...->...",
        inverse,
        state.conformal_extrinsic_curvature,
        backend="jax",
    )
    determinant_residual = jnp.max(jnp.abs(determinant - 1.0), initial=0.0)
    trace_residual = jnp.max(jnp.abs(trace), initial=0.0)
    positive = (
        jnp.all(determinant > 0.0) & jnp.all(state.chi > 0.0) & jnp.all(state.lapse > 0.0)
    )
    return determinant_residual, trace_residual, positive


def _project_z4c_algebraic_constraints(values: Array, /) -> Array:
    metric = Z4cState(values, grid_id="amr:projection").conformal_metric
    determinant = _determinant_3x3(metric)
    safe_determinant = jnp.where(
        jnp.isfinite(determinant) & (determinant > jnp.finfo(values.dtype).tiny),
        determinant,
        1.0,
    )
    normalized_metric = metric / jnp.cbrt(safe_determinant)[None, None, ...]
    inverse = _inverse_symmetric_3x3(normalized_metric, jnp.ones_like(determinant))
    extrinsic = Z4cState(values, grid_id="amr:projection").conformal_extrinsic_curvature
    trace = ein.contract("ij...,ij...->...", inverse, extrinsic, backend="jax")
    trace_free = extrinsic - normalized_metric * trace[None, None, ...] / 3.0
    return (
        values.at[1:7]
        .set(pack_symmetric(normalized_metric))
        .at[8:14]
        .set(pack_symmetric(trace_free))
    )


class Z4cAMRTransferPlan(StrictModule, NonTrainableState):
    """Constraint-projecting cell transfer for the canonical 25-channel Z4c state."""

    transfer: AMREntityTransferPlan
    constraint_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement_ratio: int = 2,
        /,
        *,
        constraint_tolerance: float = 1.0e-10,
    ):
        tolerance = float(constraint_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "Z4c AMR constraint tolerance must be finite and nonnegative."
            )
        transfer = AMREntityTransferPlan.cells(3, refinement_ratio)
        self.transfer = transfer
        self.constraint_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "z4c-amr-transfer",
                "transfer": transfer.transfer_id,
                "constraint_tolerance": tolerance,
            }
        )

    def prolong(
        self, state: Z4cState, /, *, target_grid_id: str
    ) -> tuple[Z4cState, Z4cAMRTransferEvidence]:
        return self._apply(state, target_grid_id, prolong=True)

    def restrict(
        self, state: Z4cState, /, *, target_grid_id: str
    ) -> tuple[Z4cState, Z4cAMRTransferEvidence]:
        return self._apply(state, target_grid_id, prolong=False)

    def _apply(
        self,
        state: Z4cState,
        target_grid_id: str,
        /,
        *,
        prolong: bool,
    ) -> tuple[Z4cState, Z4cAMRTransferEvidence]:
        if not isinstance(state, Z4cState):
            raise TypeError("Z4c AMR transfer requires Z4cState.")
        target = str(target_grid_id)
        if not target or target == state.grid_id:
            raise ValueError("Z4c AMR transfer requires a distinct target grid identity.")
        component_last = jnp.moveaxis(state.values, 0, -1)
        transferred = (
            self.transfer.prolong(component_last)
            if prolong
            else self.transfer.restrict(component_last)
        )
        projected = _project_z4c_algebraic_constraints(jnp.moveaxis(transferred, -1, 0))
        result = Z4cState(projected, grid_id=target)
        determinant_before, trace_before, _ = _z4c_algebraic_residuals(state)
        determinant_after, trace_after, physical = _z4c_algebraic_residuals(result)
        finite = jnp.all(jnp.isfinite(projected))
        constraints = (determinant_after <= self.constraint_tolerance) & (
            trace_after <= self.constraint_tolerance
        )
        qualified = finite & physical & constraints
        evidence = Z4cAMRTransferEvidence(
            determinant_before,
            determinant_after,
            trace_before,
            trace_after,
            finite,
            physical,
            constraints,
            qualified,
            jnp.asarray(False),
            self.plan_id,
        )
        return result, evidence


class RelativisticMaterialTransferPlan(StrictModule, NonTrainableState):
    """Volume-conservative transfer for GRHD/GRMHD ``(D,S_i,tau)`` states."""

    transfer: AMREntityTransferPlan
    conservation_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement_ratio: int = 2,
        /,
        *,
        conservation_tolerance: float = 1.0e-10,
    ):
        tolerance = float(conservation_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError(
                "Material transfer tolerance must be finite and nonnegative."
            )
        transfer = AMREntityTransferPlan.cells(3, refinement_ratio)
        self.transfer = transfer
        self.conservation_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-material-amr-transfer",
                "transfer": transfer.transfer_id,
                "conservation_tolerance": tolerance,
            }
        )

    @staticmethod
    def _state(value: ArrayLike, /) -> Array:
        state = jnp.asarray(value)
        if state.ndim != 4 or state.shape[-1] != 5:
            raise ValueError("Relativistic material state must have shape (nx,ny,nz,5).")
        if not jnp.issubdtype(state.dtype, jnp.floating):
            raise TypeError("Relativistic material state must be floating point.")
        return state

    @staticmethod
    def _volume(value: ArrayLike, shape: tuple[int, int, int], dtype: Any, /) -> Array:
        volume = jnp.asarray(value, dtype=dtype)
        if volume.shape == ():
            volume = jnp.broadcast_to(volume, shape)
        if volume.shape != shape:
            raise ValueError("Cell volumes must be scalar or match the material grid.")
        return eqx.error_if(
            volume,
            jnp.any(~jnp.isfinite(volume) | (volume <= 0.0)),
            "Cell volumes must be finite and positive.",
        )

    @staticmethod
    def _content(state: Array, volume: Array, /) -> Array:
        return jnp.sum(state * volume[..., None], axis=(0, 1, 2))

    def _evidence(
        self,
        source: Array,
        target: Array,
        source_volume: Array,
        target_volume: Array,
        /,
    ) -> RelativisticMaterialTransferEvidence:
        source_content = self._content(source, source_volume)
        target_content = self._content(target, target_volume)
        residual = target_content - source_content
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(source_content), jnp.abs(target_content)), 1.0
        )
        conservation = jnp.all(jnp.abs(residual) <= self.conservation_tolerance * scale)
        finite = (
            jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(target))
            & jnp.all(jnp.isfinite(residual))
        )
        physical = (
            jnp.all(source[..., 0] > 0.0)
            & jnp.all(source[..., 4] >= 0.0)
            & jnp.all(target[..., 0] > 0.0)
            & jnp.all(target[..., 4] >= 0.0)
        )
        return RelativisticMaterialTransferEvidence(
            source_content,
            target_content,
            residual,
            finite,
            physical,
            conservation,
            finite & physical & conservation,
            jnp.asarray(False),
            self.plan_id,
        )

    def prolong(
        self,
        coarse_state: ArrayLike,
        coarse_cell_volume: ArrayLike,
        /,
        *,
        fine_cell_volume: ArrayLike | None = None,
    ) -> tuple[Array, RelativisticMaterialTransferEvidence]:
        coarse = self._state(coarse_state)
        coarse_volume = self._volume(coarse_cell_volume, coarse.shape[:3], coarse.dtype)
        fine = self.transfer.prolong(coarse)
        derived_fine_volume = self.transfer.prolong(coarse_volume) / (
            self.transfer.refinement_ratio**3
        )
        fine_volume = self._volume(
            derived_fine_volume if fine_cell_volume is None else fine_cell_volume,
            fine.shape[:3],
            coarse.dtype,
        )
        return fine, self._evidence(coarse, fine, coarse_volume, fine_volume)

    def restrict(
        self,
        fine_state: ArrayLike,
        fine_cell_volume: ArrayLike,
        /,
        *,
        coarse_cell_volume: ArrayLike | None = None,
    ) -> tuple[Array, RelativisticMaterialTransferEvidence]:
        fine = self._state(fine_state)
        fine_volume = self._volume(fine_cell_volume, fine.shape[:3], fine.dtype)
        ratio_volume = self.transfer.refinement_ratio**3
        coarse_content = (
            self.transfer.restrict(fine * fine_volume[..., None]) * ratio_volume
        )
        derived_coarse_volume = self.transfer.restrict(fine_volume) * ratio_volume
        coarse_shape = tuple(
            size // self.transfer.refinement_ratio for size in fine.shape[:3]
        )
        coarse_volume = self._volume(
            derived_coarse_volume if coarse_cell_volume is None else coarse_cell_volume,
            coarse_shape,
            fine.dtype,
        )
        coarse = coarse_content / coarse_volume[..., None]
        return coarse, self._evidence(fine, coarse, fine_volume, coarse_volume)


class RelativisticRadiationTransferPlan(StrictModule, NonTrainableState):
    """Conservative, realizability-checked AMR transfer for ``(E,F_i)``."""

    transfer: AMREntityTransferPlan
    conservation_tolerance: float = eqx.field(static=True)
    realizability_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        refinement_ratio: int = 2,
        /,
        *,
        conservation_tolerance: float = 1.0e-10,
        realizability_tolerance: float = 1.0e-10,
    ) -> None:
        conservation = float(conservation_tolerance)
        realizability = float(realizability_tolerance)
        if (
            not np.isfinite(conservation)
            or conservation < 0.0
            or not np.isfinite(realizability)
            or realizability < 0.0
        ):
            raise ValueError("Radiation AMR tolerances must be finite and nonnegative.")
        self.transfer = AMREntityTransferPlan.cells(3, refinement_ratio)
        self.conservation_tolerance = conservation
        self.realizability_tolerance = realizability
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-radiation-amr-transfer",
                "transfer": self.transfer.transfer_id,
                "conservation_tolerance": conservation,
                "realizability_tolerance": realizability,
            }
        )

    @staticmethod
    def _state(value: ArrayLike, /) -> Array:
        state = jnp.asarray(value)
        if state.ndim != 4 or state.shape[-1] != 4:
            raise ValueError("Relativistic radiation state must have shape (nx,ny,nz,4).")
        if not jnp.issubdtype(state.dtype, jnp.floating):
            raise TypeError("Relativistic radiation state must be floating point.")
        return state

    @staticmethod
    def _volume(value: ArrayLike, shape: tuple[int, int, int], dtype, /) -> Array:
        volume = jnp.asarray(value, dtype=dtype)
        if volume.shape == ():
            volume = jnp.broadcast_to(volume, shape)
        if volume.shape != shape:
            raise ValueError("Cell volumes must be scalar or match the radiation grid.")
        return eqx.error_if(
            volume,
            jnp.any(~jnp.isfinite(volume) | (volume <= 0.0)),
            "Cell volumes must be finite and positive.",
        )

    def _evidence(
        self,
        source: Array,
        target: Array,
        source_volume: Array,
        target_volume: Array,
        /,
    ) -> RelativisticRadiationTransferEvidence:
        source_content = jnp.sum(source * source_volume[..., None], axis=(0, 1, 2))
        target_content = jnp.sum(target * target_volume[..., None], axis=(0, 1, 2))
        residual = target_content - source_content
        scale = jnp.maximum(
            jnp.maximum(jnp.abs(source_content), jnp.abs(target_content)), 1.0
        )
        conservation = jnp.all(jnp.abs(residual) <= self.conservation_tolerance * scale)

        def realizable(value: Array) -> Array:
            energy = value[..., 0]
            flux_norm = jnp.sqrt(jnp.sum(value[..., 1:] ** 2, axis=-1))
            return jnp.all(
                (energy > 0.0)
                & (
                    flux_norm
                    <= energy
                    * (1.0 + jnp.asarray(self.realizability_tolerance, energy.dtype))
                )
            )

        finite = (
            jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(target))
            & jnp.all(jnp.isfinite(residual))
        )
        physical = realizable(source) & realizable(target)
        qualified = finite & physical & conservation
        return RelativisticRadiationTransferEvidence(
            source_content,
            target_content,
            residual,
            finite,
            physical,
            conservation,
            qualified,
            jnp.asarray(False),
            self.plan_id,
        )

    def prolong(
        self,
        coarse_state: ArrayLike,
        coarse_cell_volume: ArrayLike,
        /,
        *,
        fine_cell_volume: ArrayLike | None = None,
    ) -> tuple[Array, RelativisticRadiationTransferEvidence]:
        coarse = self._state(coarse_state)
        coarse_volume = self._volume(coarse_cell_volume, coarse.shape[:3], coarse.dtype)
        fine = self.transfer.prolong(coarse)
        derived_fine_volume = self.transfer.prolong(coarse_volume) / (
            self.transfer.refinement_ratio**3
        )
        fine_volume = self._volume(
            derived_fine_volume if fine_cell_volume is None else fine_cell_volume,
            fine.shape[:3],
            coarse.dtype,
        )
        return fine, self._evidence(coarse, fine, coarse_volume, fine_volume)

    def restrict(
        self,
        fine_state: ArrayLike,
        fine_cell_volume: ArrayLike,
        /,
        *,
        coarse_cell_volume: ArrayLike | None = None,
    ) -> tuple[Array, RelativisticRadiationTransferEvidence]:
        fine = self._state(fine_state)
        fine_volume = self._volume(fine_cell_volume, fine.shape[:3], fine.dtype)
        ratio_volume = self.transfer.refinement_ratio**3
        coarse_content = (
            self.transfer.restrict(fine * fine_volume[..., None]) * ratio_volume
        )
        derived_coarse_volume = self.transfer.restrict(fine_volume) * ratio_volume
        coarse_shape = tuple(
            size // self.transfer.refinement_ratio for size in fine.shape[:3]
        )
        coarse_volume = self._volume(
            derived_coarse_volume if coarse_cell_volume is None else coarse_cell_volume,
            coarse_shape,
            fine.dtype,
        )
        coarse = coarse_content / coarse_volume[..., None]
        return coarse, self._evidence(fine, coarse, fine_volume, coarse_volume)


class RelativisticMaterialRefluxResult(StrictModule):
    material_state: Array
    correction: Array
    conservation_residual: Array
    finite: Array
    conservation_valid: Array
    qualified: Array
    register_id: str = eqx.field(static=True)


def reflux_relativistic_material(
    coarse_material_state: ArrayLike,
    coarse_cell_volume: ArrayLike,
    register: FluxRegister,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> RelativisticMaterialRefluxResult:
    """Apply only the accepted conservative material-flux mismatch."""

    if not isinstance(register, FluxRegister):
        raise TypeError("Material reflux requires a FluxRegister.")
    state = RelativisticMaterialTransferPlan._state(coarse_material_state)
    volume = RelativisticMaterialTransferPlan._volume(
        coarse_cell_volume, state.shape[:3], state.dtype
    )
    tolerance_ = float(tolerance)
    if not np.isfinite(tolerance_) or tolerance_ < 0.0:
        raise ValueError("Material reflux tolerance must be finite and nonnegative.")
    updated = register.apply(state, volume)
    correction = updated - state
    expected_content = jnp.sum(register.mismatch(), axis=(0, 1, 2))
    realized_content = jnp.sum(correction * volume[..., None], axis=(0, 1, 2))
    residual = realized_content - expected_content
    scale = jnp.maximum(
        jnp.maximum(jnp.abs(realized_content), jnp.abs(expected_content)), 1.0
    )
    conservation = jnp.all(jnp.abs(residual) <= tolerance_ * scale)
    finite = jnp.all(jnp.isfinite(updated)) & jnp.all(jnp.isfinite(residual))
    physical = (
        jnp.all(state[..., 0] > 0.0)
        & jnp.all(state[..., 4] >= 0.0)
        & jnp.all(updated[..., 0] > 0.0)
        & jnp.all(updated[..., 4] >= 0.0)
    )
    return RelativisticMaterialRefluxResult(
        updated,
        correction,
        residual,
        finite,
        conservation,
        finite & physical & conservation,
        register.register_id,
    )


class RelativisticMagneticAMRTransferPlan(StrictModule, NonTrainableState):
    """Bridge-derived transfer of packed densitized magnetic face flux."""

    transfer: DivergenceFreeMagneticTransferPlan
    divergence_tolerance: float = eqx.field(static=True)
    coarse_bridge_id: str = eqx.field(static=True)
    fine_bridge_id: str = eqx.field(static=True)
    coarse_grid_topology_id: str = eqx.field(static=True)
    fine_grid_topology_id: str = eqx.field(static=True)
    direction: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse: StructuredCochainBridge,
        fine: StructuredCochainBridge,
        /,
        *,
        direction: str,
        refinement_ratio: int = 2,
        divergence_tolerance: float = 1.0e-10,
    ):
        tolerance = float(divergence_tolerance)
        direction_ = str(direction)
        if (
            not np.isfinite(tolerance)
            or tolerance < 0.0
            or direction_ not in ("prolong", "restrict")
        ):
            raise ValueError("Magnetic AMR tolerance or direction is invalid.")
        transfer = DivergenceFreeMagneticTransferPlan(
            coarse, fine, refinement_ratio=refinement_ratio
        )
        self.transfer = transfer
        self.divergence_tolerance = tolerance
        self.coarse_bridge_id = coarse.bridge_id
        self.fine_bridge_id = fine.bridge_id
        self.coarse_grid_topology_id = coarse.grid.topology.topology_id
        self.fine_grid_topology_id = fine.grid.topology.topology_id
        self.direction = direction_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-magnetic-amr-transfer",
                "transfer": transfer.plan_id,
                "divergence_tolerance": tolerance,
                "coarse_bridge": self.coarse_bridge_id,
                "fine_bridge": self.fine_bridge_id,
                "coarse_grid_topology": self.coarse_grid_topology_id,
                "fine_grid_topology": self.fine_grid_topology_id,
                "direction": direction_,
            }
        )

    def _evidence(
        self, diagnostics: MagneticAMRTransferDiagnostics, result: Array, /
    ) -> RelativisticMagneticTransferEvidence:
        before = jnp.max(jnp.abs(diagnostics.divergence_before), initial=0.0)
        after = jnp.max(jnp.abs(diagnostics.divergence_after), initial=0.0)
        divergence_valid = (before <= self.divergence_tolerance) & (
            after <= self.divergence_tolerance
        )
        finite = jnp.all(jnp.isfinite(result)) & jnp.all(
            jnp.isfinite(diagnostics.divergence_after)
        )
        flux_valid = jnp.abs(diagnostics.flux_defect) <= self.divergence_tolerance
        return RelativisticMagneticTransferEvidence(
            diagnostics.divergence_before,
            diagnostics.divergence_after,
            diagnostics.divergence_change,
            diagnostics.flux_defect,
            finite,
            divergence_valid,
            finite & divergence_valid & flux_valid,
            jnp.asarray(False),
            self.plan_id,
        )

    def prolong(
        self, coarse_flux: ArrayLike, /
    ) -> tuple[Array, RelativisticMagneticTransferEvidence]:
        if self.direction != "prolong":
            raise ValueError("This magnetic transfer plan is bound to restriction.")
        result, diagnostics = self.transfer.prolong(coarse_flux)
        return result, self._evidence(diagnostics, result)

    def restrict(
        self, fine_flux: ArrayLike, /
    ) -> tuple[Array, RelativisticMagneticTransferEvidence]:
        if self.direction != "restrict":
            raise ValueError("This magnetic transfer plan is bound to prolongation.")
        result, diagnostics = self.transfer.restrict(fine_flux)
        return result, self._evidence(diagnostics, result)


class RelativisticMagneticRefluxPlan(StrictModule, NonTrainableState):
    """Apply an accepted edge-EMF mismatch through the exact cochain curl."""

    synchronization: ConstrainedMHDAMRSynchronizationPlan
    divergence_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        coarse_bridge: StructuredCochainBridge,
        /,
        *,
        divergence_tolerance: float = 1.0e-10,
    ):
        tolerance = float(divergence_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Magnetic reflux tolerance must be finite and nonnegative.")
        synchronization = ConstrainedMHDAMRSynchronizationPlan(coarse_bridge)
        self.synchronization = synchronization
        self.divergence_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-magnetic-emf-reflux",
                "synchronization": synchronization.plan_id,
                "divergence_tolerance": tolerance,
            }
        )

    def reflux(
        self,
        coarse_magnetic_flux: ArrayLike,
        coarse_emf_integral: ArrayLike,
        restricted_fine_emf_integral: ArrayLike,
        /,
    ) -> tuple[Array, RelativisticMagneticTransferEvidence]:
        register = ElectromotiveForceRegister(
            coarse_emf_integral,
            restricted_fine_emf_integral,
            register_id=canonical_fingerprint(
                {
                    "kind": "relativistic-magnetic-emf-register",
                    "plan": self.plan_id,
                }
            ),
        )
        result, diagnostics = self.synchronization.reflux_curl(
            coarse_magnetic_flux, register
        )
        finite = jnp.all(jnp.isfinite(result)) & jnp.all(
            jnp.isfinite(diagnostics.divergence_after)
        )
        before = jnp.max(jnp.abs(diagnostics.divergence_before), initial=0.0)
        after = jnp.max(jnp.abs(diagnostics.divergence_after), initial=0.0)
        divergence_change_valid = (
            jnp.max(
                jnp.abs(diagnostics.divergence_after - diagnostics.divergence_before),
                initial=0.0,
            )
            <= self.divergence_tolerance
        )
        divergence_valid = (
            (before <= self.divergence_tolerance)
            & (after <= self.divergence_tolerance)
            & divergence_change_valid
        )
        flux_valid = jnp.abs(diagnostics.flux_defect) <= self.divergence_tolerance
        evidence = RelativisticMagneticTransferEvidence(
            diagnostics.divergence_before,
            diagnostics.divergence_after,
            diagnostics.divergence_change,
            diagnostics.flux_defect,
            finite,
            divergence_valid,
            finite & divergence_valid & flux_valid,
            jnp.asarray(False),
            self.plan_id,
        )
        return result, evidence


class RelativisticMaterialSubcyclingPlan(StrictModule, NonTrainableState):
    """Topology-bound AMR time schedule plus accepted-ledger synchronization."""

    schedule: AMRTimeSchedulePlan
    conservation: BlockAMRConservationPlan
    formulation: NumericalRelativityFormulation = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: NumericalRelativityFormulation,
        hierarchy: PreparedFDAMRHierarchy,
        topology: BlockHierarchyTopology,
        /,
        *,
        subcycling: bool = True,
        temporal_method_id: str = "temporal:ssprk33",
    ):
        formulation_ = _formulation(formulation)
        if not any(name in formulation_ for name in ("grhd", "grmhd", "grrmhd")):
            raise ValueError(
                "Conservative material subcycling requires relativistic matter."
            )
        schedule = AMRTimeSchedulePlan(
            hierarchy,
            subcycling=subcycling,
            temporal_method_id=temporal_method_id,
        )
        conservation = BlockAMRConservationPlan(hierarchy, topology)
        self.schedule = schedule
        self.conservation = conservation
        self.formulation = formulation_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "relativistic-material-subcycling",
                "formulation": formulation_,
                "schedule": schedule.schedule_id,
                "conservation": conservation.plan_id,
            }
        )

    def reflux(
        self,
        level_values: Sequence[ArrayLike],
        register: FluxRegister,
        /,
    ) -> tuple[Array, ...]:
        return self.conservation.reflux(level_values, register)

    def restrict_covered(
        self,
        coarse_values: ArrayLike,
        fine_values: ArrayLike,
        coarse_level: int,
        /,
    ) -> Array:
        return self.conservation.restrict_covered(
            coarse_values, fine_values, coarse_level
        )


class NumericalRelativityAMRTopologyEpoch(StrictModule, NonTrainableState):
    """One canonical block epoch plus its authoritative prepared ownership."""

    index: int = eqx.field(static=True)
    formulation: NumericalRelativityFormulation = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    hierarchy: BlockHierarchyState
    distribution: PreparedNumericalRelativityAMRDistribution
    ownership: tuple[NumericalRelativityOwnership, ...]
    magnetic_bridge_id: str | None = eqx.field(static=True)
    magnetic_grid_topology_id: str | None = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        formulation: NumericalRelativityFormulation,
        hierarchy: BlockHierarchyState,
        distribution: PreparedNumericalRelativityAMRDistribution,
        /,
        *,
        magnetic_bridge: StructuredCochainBridge | None = None,
    ):
        formulation_ = _formulation(formulation)
        if not isinstance(hierarchy, BlockHierarchyState) or not isinstance(
            distribution, PreparedNumericalRelativityAMRDistribution
        ):
            raise TypeError(
                "NR AMR epoch requires a hierarchy state and prepared block distribution."
            )
        if (
            hierarchy.topology.epoch.epoch_id
            != distribution.distribution.topology.epoch.epoch_id
        ):
            raise ValueError(
                "NR AMR hierarchy and prepared ownership must share one topology epoch."
            )
        uses_magnetic = formulation_ in (
            "grmhd",
            "grrmhd",
            "z4c-grmhd",
            "z4c-grrmhd",
        )
        if uses_magnetic != isinstance(magnetic_bridge, StructuredCochainBridge):
            raise TypeError(
                "GRMHD AMR epochs require their exact magnetic cochain bridge."
            )
        magnetic_bridge_id = (
            None if magnetic_bridge is None else magnetic_bridge.bridge_id
        )
        magnetic_grid_topology_id = (
            None if magnetic_bridge is None else magnetic_bridge.grid.topology.topology_id
        )
        topology = hierarchy.topology
        self.index = topology.epoch.index
        self.formulation = formulation_
        self.geometry_id = topology.plan.geometry_id
        self.hierarchy = hierarchy
        self.distribution = distribution
        self.ownership = distribution.ownership
        self.magnetic_bridge_id = magnetic_bridge_id
        self.magnetic_grid_topology_id = magnetic_grid_topology_id
        self.topology_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-amr-topology-epoch",
                "formulation": formulation_,
                "block_epoch": topology.epoch.epoch_id,
                "distribution": distribution.prepared_id,
                "magnetic_bridge": magnetic_bridge_id,
                "magnetic_grid_topology": magnetic_grid_topology_id,
            }
        )

    @property
    def total_block_capacity(self) -> int:
        return sum(level.maximum_blocks for level in self.hierarchy.plan.levels)


class NumericalRelativityAMRState(StrictModule):
    """Fixed-capacity packed block fields bound to one immutable topology epoch."""

    fields: tuple[Array, ...]
    epoch: NumericalRelativityAMRTopologyEpoch
    field_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        epoch: NumericalRelativityAMRTopologyEpoch,
        fields: Sequence[ArrayLike],
        /,
    ):
        if not isinstance(epoch, NumericalRelativityAMRTopologyEpoch):
            raise TypeError("NR AMR state requires NumericalRelativityAMRTopologyEpoch.")
        values = tuple(jnp.asarray(value) for value in fields)
        names = formulation_field_names(epoch.formulation)
        capacity = epoch.total_block_capacity
        block_shapes = {level.block_shape for level in epoch.hierarchy.plan.levels}
        if len(block_shapes) != 1:
            raise ValueError(
                "Packed NR AMR state requires one static block shape across levels."
            )
        block_shape = next(iter(block_shapes))
        if len(values) != len(names):
            raise ValueError("NR AMR fields must match the formulation.")
        for name, value in zip(names, values, strict=True):
            if name == "z4c":
                expected = (capacity, 25, *block_shape)
                valid = value.shape == expected
            elif name == "material":
                expected = (capacity, *block_shape, 5)
                valid = value.shape == expected
            elif name == "radiation":
                expected = (capacity, *block_shape, 4)
                valid = value.shape == expected
            else:
                valid = value.ndim == 1 and value.size > 0
            if not valid:
                raise ValueError(
                    f"NR AMR field {name!r} has an invalid fixed-capacity block shape."
                )
        self.fields = values
        self.epoch = epoch
        self.field_names = names

    def field(self, name: str, /) -> Array:
        identifier = str(name)
        if identifier not in self.field_names:
            raise KeyError(f"NR AMR state has no field {identifier!r}.")
        return self.fields[self.field_names.index(identifier)]


def _active_block_slots(epoch: NumericalRelativityAMRTopologyEpoch, /) -> np.ndarray:
    return np.flatnonzero(
        np.concatenate(
            tuple(
                np.asarray(metadata.active, dtype=bool)
                for metadata in epoch.hierarchy.topology.levels
            )
        )
    )


def _packed_z4c_constraints(
    values: Array, epoch: NumericalRelativityAMRTopologyEpoch, /
) -> tuple[Array, Array, Array, Array]:
    determinant = jnp.asarray(0.0, dtype=values.dtype)
    trace = jnp.asarray(0.0, dtype=values.dtype)
    physical = jnp.asarray(True)
    finite = jnp.asarray(True)
    for slot in _active_block_slots(epoch):
        state = Z4cState(values[int(slot)], grid_id=epoch.topology_id)
        determinant_slot, trace_slot, physical_slot = _z4c_algebraic_residuals(state)
        determinant = jnp.maximum(determinant, determinant_slot)
        trace = jnp.maximum(trace, trace_slot)
        physical = physical & physical_slot
        finite = finite & jnp.all(jnp.isfinite(state.values))
    return determinant, trace, physical, finite


def _packed_material_valid(
    values: Array, epoch: NumericalRelativityAMRTopologyEpoch, /
) -> Array:
    valid = jnp.asarray(True)
    for slot in _active_block_slots(epoch):
        block = values[int(slot)]
        valid = (
            valid
            & jnp.all(jnp.isfinite(block))
            & jnp.all(block[..., 0] > 0.0)
            & jnp.all(block[..., 4] >= 0.0)
        )
    return valid


def _with_transfer_binding(
    evidence: TransferEvidence,
    binding: AMRTransferBinding,
    /,
) -> TransferEvidence:
    if isinstance(evidence, Z4cAMRTransferEvidence):
        return Z4cAMRTransferEvidence(
            evidence.determinant_constraint_before,
            evidence.determinant_constraint_after,
            evidence.trace_free_constraint_before,
            evidence.trace_free_constraint_after,
            evidence.finite,
            evidence.physically_valid,
            evidence.constraint_valid,
            evidence.qualified,
            evidence.derivative_valid,
            evidence.transfer_id,
            binding,
        )
    if isinstance(evidence, RelativisticMaterialTransferEvidence):
        return RelativisticMaterialTransferEvidence(
            evidence.source_content,
            evidence.target_content,
            evidence.conservation_residual,
            evidence.finite,
            evidence.physically_valid,
            evidence.conservation_valid,
            evidence.qualified,
            evidence.derivative_valid,
            evidence.transfer_id,
            binding,
        )
    if isinstance(evidence, RelativisticMagneticTransferEvidence):
        return RelativisticMagneticTransferEvidence(
            evidence.divergence_before,
            evidence.divergence_after,
            evidence.divergence_change,
            evidence.flux_defect,
            evidence.finite,
            evidence.divergence_valid,
            evidence.qualified,
            evidence.derivative_valid,
            evidence.transfer_id,
            binding,
        )
    raise TypeError("Unknown numerical-relativity AMR transfer evidence.")


class NumericalRelativityAMRTransitionResult(StrictModule):
    predecessor: NumericalRelativityAMRState
    candidate: NumericalRelativityAMRState
    accepted_state: NumericalRelativityAMRState
    committed: Array
    overflow: Array
    finite: Array
    qualified: Array
    derivative_valid: Array
    compile_status: str = eqx.field(static=True)
    transition_id: str = eqx.field(static=True)


def _cell_hierarchy_from_packed(
    values: Array,
    epoch: NumericalRelativityAMRTopologyEpoch,
    field_name: str,
    /,
) -> BlockHierarchyState:
    levels = []
    offset = 0
    for plan, metadata in zip(
        epoch.hierarchy.plan.levels,
        epoch.hierarchy.topology.levels,
        strict=True,
    ):
        chunk = values[offset : offset + plan.maximum_blocks]
        offset += plan.maximum_blocks
        if field_name == "z4c":
            chunk = jnp.moveaxis(chunk, 1, -1)
        levels.append(BlockLevelState(plan, metadata, chunk))
    return BlockHierarchyState(epoch.hierarchy.topology, tuple(levels))


def _packed_from_cell_hierarchy(state: BlockHierarchyState, field_name: str, /) -> Array:
    values = tuple(
        jnp.moveaxis(level.values, -1, 1) if field_name == "z4c" else level.values
        for level in state.levels
    )
    return jnp.concatenate(values, axis=0)


def _project_packed_z4c(
    values: Array, epoch: NumericalRelativityAMRTopologyEpoch, /
) -> Array:
    result = values
    for slot in _active_block_slots(epoch):
        result = result.at[int(slot)].set(
            _project_z4c_algebraic_constraints(result[int(slot)])
        )
    return result


def _required_transfer_plans(
    formulation: NumericalRelativityFormulation, /
) -> tuple[type[Any], ...]:
    if formulation == "z4c":
        return (Z4cAMRTransferPlan,)
    if formulation == "grhd":
        return (RelativisticMaterialTransferPlan,)
    if formulation == "grmhd":
        return (
            RelativisticMaterialTransferPlan,
            RelativisticMagneticAMRTransferPlan,
        )
    if formulation == "grrmhd":
        return (
            RelativisticMaterialTransferPlan,
            RelativisticRadiationTransferPlan,
            RelativisticMagneticAMRTransferPlan,
        )
    if formulation == "z4c-grhd":
        return (Z4cAMRTransferPlan, RelativisticMaterialTransferPlan)
    if formulation == "z4c-grmhd":
        return (
            Z4cAMRTransferPlan,
            RelativisticMaterialTransferPlan,
            RelativisticMagneticAMRTransferPlan,
        )
    return (
        Z4cAMRTransferPlan,
        RelativisticMaterialTransferPlan,
        RelativisticRadiationTransferPlan,
        RelativisticMagneticAMRTransferPlan,
    )


class NumericalRelativityAMRTopologyTransition(StrictModule, NonTrainableState):
    """Execute field transfers and atomically adopt their compiled target epoch."""

    source: NumericalRelativityAMRTopologyEpoch
    compilation: BlockTopologyCompileResult
    target: NumericalRelativityAMRTopologyEpoch | None
    transfer_plans: tuple[Any, ...]
    transition_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: NumericalRelativityAMRTopologyEpoch,
        compilation: BlockTopologyCompileResult,
        target: NumericalRelativityAMRTopologyEpoch | None = None,
        /,
        *,
        transfer_plans: Sequence[Any] = (),
    ):
        if not isinstance(source, NumericalRelativityAMRTopologyEpoch) or not isinstance(
            compilation, BlockTopologyCompileResult
        ):
            raise TypeError(
                "NR AMR transition requires a source epoch and topology compilation."
            )
        plans = tuple(transfer_plans)
        changed = compilation.status.successful and compilation.status.changed
        if changed:
            if not isinstance(target, NumericalRelativityAMRTopologyEpoch):
                raise TypeError(
                    "A successful changed topology compilation requires its target epoch."
                )
            if (
                target.index != source.index + 1
                or target.formulation != source.formulation
                or target.hierarchy.plan.plan_id != source.hierarchy.plan.plan_id
                or target.total_block_capacity != source.total_block_capacity
                or target.hierarchy.topology.epoch.epoch_id
                != compilation.topology.epoch.epoch_id
            ):
                raise ValueError(
                    "Compiled NR AMR target must be the consecutive fixed-capacity epoch."
                )
            required = _required_transfer_plans(source.formulation)
            if len(plans) != len(required) or any(
                not isinstance(plan, expected)
                for plan, expected in zip(plans, required, strict=True)
            ):
                raise TypeError(
                    "Changed NR AMR topology requires one exact transfer plan per field."
                )
            for plan in plans:
                if isinstance(plan, RelativisticMagneticAMRTransferPlan):
                    if plan.direction == "prolong":
                        source_bridge = plan.coarse_bridge_id
                        source_grid = plan.coarse_grid_topology_id
                        target_bridge = plan.fine_bridge_id
                        target_grid = plan.fine_grid_topology_id
                    else:
                        source_bridge = plan.fine_bridge_id
                        source_grid = plan.fine_grid_topology_id
                        target_bridge = plan.coarse_bridge_id
                        target_grid = plan.coarse_grid_topology_id
                    if (
                        source_bridge != source.magnetic_bridge_id
                        or source_grid != source.magnetic_grid_topology_id
                        or target_bridge != target.magnetic_bridge_id
                        or target_grid != target.magnetic_grid_topology_id
                    ):
                        raise ValueError(
                            "Magnetic transfer bridges do not match the AMR epoch layouts."
                        )
        else:
            if target is not None or plans:
                raise ValueError(
                    "Failed or unchanged topology compilation has no target transfer."
                )
            if (
                compilation.topology.epoch.epoch_id
                != source.hierarchy.topology.epoch.epoch_id
            ):
                raise ValueError(
                    "Rejected topology compilation must retain the source topology."
                )
        self.source = source
        self.compilation = compilation
        self.target = target
        self.transfer_plans = plans
        self.transition_id = canonical_fingerprint(
            {
                "kind": "numerical-relativity-amr-topology-transition",
                "source": source.topology_id,
                "compilation": compilation.result_id,
                "target": None if target is None else target.topology_id,
                "transfer_plans": [plan.plan_id for plan in plans],
            }
        )

    def _binding(
        self,
        source: Array,
        target: Array,
        field_name: str,
        transfer_plan_id: str,
        /,
    ) -> AMRTransferBinding:
        if self.target is None:
            raise RuntimeError("Rejected topology transition has no binding target.")
        return AMRTransferBinding(
            source,
            target,
            source_topology_id=self.source.topology_id,
            target_topology_id=self.target.topology_id,
            source_epoch_id=self.source.hierarchy.topology.epoch.epoch_id,
            target_epoch_id=self.target.hierarchy.topology.epoch.epoch_id,
            field_name=field_name,
            formulation=self.source.formulation,
            transfer_plan_id=transfer_plan_id,
            transition_plan_id=self.transition_id,
        )

    def _transfer_z4c(
        self,
        source: Array,
        plan: Z4cAMRTransferPlan,
        /,
    ) -> tuple[Array, Z4cAMRTransferEvidence]:
        if self.target is None:
            raise RuntimeError("Z4c topology transfer requires a target.")
        transition = BlockFieldTopologyTransition(
            self.source.hierarchy.topology,
            self.target.hierarchy.topology,
            "z4c",
            component_shape=(25,),
            dtype=source.dtype,
        )
        transferred = transition.apply(
            _cell_hierarchy_from_packed(source, self.source, "z4c")
        )
        target = _project_packed_z4c(
            _packed_from_cell_hierarchy(transferred.state, "z4c"),
            self.target,
        )
        source_det, source_trace, source_physical, source_finite = (
            _packed_z4c_constraints(source, self.source)
        )
        target_det, target_trace, target_physical, target_finite = (
            _packed_z4c_constraints(target, self.target)
        )
        constraints = (
            (source_det <= plan.constraint_tolerance)
            & (source_trace <= plan.constraint_tolerance)
            & (target_det <= plan.constraint_tolerance)
            & (target_trace <= plan.constraint_tolerance)
        )
        finite = source_finite & target_finite
        physical = source_physical & target_physical
        evidence = Z4cAMRTransferEvidence(
            source_det,
            target_det,
            source_trace,
            target_trace,
            finite,
            physical,
            constraints,
            finite & physical & constraints & transferred.successful,
            jnp.asarray(False),
            plan.plan_id,
            self._binding(source, target, "z4c", plan.plan_id),
        )
        return target, evidence

    def _transfer_material(
        self,
        source: Array,
        plan: RelativisticMaterialTransferPlan,
        /,
    ) -> tuple[Array, RelativisticMaterialTransferEvidence]:
        if self.target is None:
            raise RuntimeError("Material topology transfer requires a target.")
        transition = BlockFieldTopologyTransition(
            self.source.hierarchy.topology,
            self.target.hierarchy.topology,
            "material",
            component_shape=(5,),
            dtype=source.dtype,
        )
        transferred = transition.apply(
            _cell_hierarchy_from_packed(source, self.source, "material")
        )
        target = _packed_from_cell_hierarchy(transferred.state, "material")
        scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(transferred.source_content),
                jnp.abs(transferred.target_content),
            ),
            1.0,
        )
        conservation = jnp.all(
            jnp.abs(transferred.conservation_residual)
            <= plan.conservation_tolerance * scale
        )
        source_valid = _packed_material_valid(source, self.source)
        target_valid = _packed_material_valid(target, self.target)
        finite = jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(target))
        evidence = RelativisticMaterialTransferEvidence(
            transferred.source_content,
            transferred.target_content,
            transferred.conservation_residual,
            finite,
            source_valid & target_valid,
            conservation,
            finite & source_valid & target_valid & conservation & transferred.successful,
            jnp.asarray(False),
            plan.plan_id,
            self._binding(source, target, "material", plan.plan_id),
        )
        return target, evidence

    def _transfer_radiation(
        self,
        source: Array,
        plan: RelativisticRadiationTransferPlan,
        /,
    ) -> tuple[Array, RelativisticRadiationTransferEvidence]:
        if self.target is None:
            raise RuntimeError("Radiation topology transfer requires a target.")
        transition = BlockFieldTopologyTransition(
            self.source.hierarchy.topology,
            self.target.hierarchy.topology,
            "radiation",
            component_shape=(4,),
            dtype=source.dtype,
        )
        transferred = transition.apply(
            _cell_hierarchy_from_packed(source, self.source, "radiation")
        )
        target = _packed_from_cell_hierarchy(transferred.state, "radiation")
        scale = jnp.maximum(
            jnp.maximum(
                jnp.abs(transferred.source_content),
                jnp.abs(transferred.target_content),
            ),
            1.0,
        )
        conservation = jnp.all(
            jnp.abs(transferred.conservation_residual)
            <= plan.conservation_tolerance * scale
        )
        source_energy = source[..., 0]
        target_energy = target[..., 0]
        source_flux = jnp.sqrt(jnp.sum(source[..., 1:] ** 2, axis=-1))
        target_flux = jnp.sqrt(jnp.sum(target[..., 1:] ** 2, axis=-1))
        physical = (
            jnp.all(source_energy > 0.0)
            & jnp.all(target_energy > 0.0)
            & jnp.all(source_flux <= source_energy * (1.0 + plan.realizability_tolerance))
            & jnp.all(target_flux <= target_energy * (1.0 + plan.realizability_tolerance))
        )
        finite = jnp.all(jnp.isfinite(source)) & jnp.all(jnp.isfinite(target))
        evidence = RelativisticRadiationTransferEvidence(
            transferred.source_content,
            transferred.target_content,
            transferred.conservation_residual,
            finite,
            physical,
            conservation,
            finite & physical & conservation & transferred.successful,
            jnp.asarray(False),
            plan.plan_id,
            self._binding(source, target, "radiation", plan.plan_id),
        )
        return target, evidence

    def _transfer_magnetic(
        self,
        source: Array,
        plan: RelativisticMagneticAMRTransferPlan,
        /,
    ) -> tuple[Array, RelativisticMagneticTransferEvidence]:
        degree = plan.transfer.coarse.dimension - 1
        coarse_size = plan.transfer.coarse.cochain.cell_counts[degree]
        fine_size = plan.transfer.fine.cochain.cell_counts[degree]
        if plan.direction == "prolong":
            if source.shape != (coarse_size,):
                raise ValueError(
                    "Magnetic prolongation source does not match its bound coarse bridge."
                )
            target, evidence = plan.prolong(source)
        else:
            if source.shape != (fine_size,):
                raise ValueError(
                    "Magnetic restriction source does not match its bound fine bridge."
                )
            target, evidence = plan.restrict(source)
        return target, _with_transfer_binding(
            evidence,
            self._binding(source, target, "magnetic_flux", plan.plan_id),
        )

    def apply(
        self,
        predecessor: NumericalRelativityAMRState,
        /,
    ) -> NumericalRelativityAMRTransitionResult:
        if not isinstance(predecessor, NumericalRelativityAMRState):
            raise TypeError("NR AMR transition requires its predecessor state.")
        if predecessor.epoch.topology_id != self.source.topology_id:
            raise ValueError("NR AMR predecessor is not bound to the source topology.")
        overflow = jnp.asarray(
            self.compilation.status.code == "capacity_exceeded"
            or self.compilation.evidence.overflow_level is not None
        )
        if self.target is None:
            finite = jnp.all(
                jnp.stack(
                    tuple(jnp.all(jnp.isfinite(value)) for value in predecessor.fields)
                )
            )
            return NumericalRelativityAMRTransitionResult(
                predecessor,
                predecessor,
                predecessor,
                jnp.asarray(False),
                overflow,
                finite,
                jnp.asarray(self.compilation.status.successful),
                jnp.asarray(False),
                self.compilation.status.code,
                self.transition_id,
            )
        candidate_fields = []
        evidence: list[TransferEvidence] = []
        for field_name, source, plan in zip(
            predecessor.field_names,
            predecessor.fields,
            self.transfer_plans,
            strict=True,
        ):
            if isinstance(plan, Z4cAMRTransferPlan):
                target, record = self._transfer_z4c(source, plan)
            elif isinstance(plan, RelativisticMaterialTransferPlan):
                target, record = self._transfer_material(source, plan)
            elif isinstance(plan, RelativisticRadiationTransferPlan):
                target, record = self._transfer_radiation(source, plan)
            elif isinstance(plan, RelativisticMagneticAMRTransferPlan):
                target, record = self._transfer_magnetic(source, plan)
            else:
                raise TypeError(f"No AMR transfer executor exists for {field_name!r}.")
            candidate_fields.append(target)
            evidence.append(record)
        candidate = NumericalRelativityAMRState(self.target, tuple(candidate_fields))
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in candidate.fields))
        )
        qualified = jnp.all(jnp.stack(tuple(value.qualified for value in evidence)))
        committed = finite & qualified & ~overflow
        accepted = candidate if bool(jax.device_get(committed)) else predecessor
        return NumericalRelativityAMRTransitionResult(
            predecessor,
            candidate,
            accepted,
            committed,
            overflow,
            finite,
            qualified,
            jnp.asarray(False),
            self.compilation.status.code,
            self.transition_id,
        )


__all__ = [
    "AMRTransferBinding",
    "NumericalRelativityAMRState",
    "NumericalRelativityAMRHaloPlan",
    "NumericalRelativityAMRHaloWorkspace",
    "NumericalRelativityAMRTopologyEpoch",
    "NumericalRelativityAMRTopologyTransition",
    "NumericalRelativityAMRTransitionResult",
    "RelativisticMagneticAMRTransferPlan",
    "RelativisticMagneticRefluxPlan",
    "RelativisticMagneticTransferEvidence",
    "RelativisticMaterialRefluxResult",
    "RelativisticMaterialSubcyclingPlan",
    "RelativisticMaterialTransferEvidence",
    "RelativisticMaterialTransferPlan",
    "RelativisticRadiationTransferEvidence",
    "RelativisticRadiationTransferPlan",
    "TransferEvidence",
    "Z4cAMRTransferEvidence",
    "Z4cAMRTransferPlan",
    "reflux_relativistic_material",
]
