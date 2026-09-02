#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._conservation_ledger import (
    AcceptedConservationFluxIntegralBlock,
    AcceptedConservationIntegralLedger,
)
from ..discretization.finite_volume import (
    PreparedUnstructuredFiniteVolumeDynamics,
    UnstructuredAMRFluxRegister,
    UnstructuredAMRHierarchyPlan,
    UnstructuredAMRSelection,
)
from ._finite_volume import FiniteVolumeStageStateProvider
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._finite_volume_runtime import (
    FiniteVolumeAdvanceResult,
    FiniteVolumeRuntimeState,
    FiniteVolumeStepPolicy,
    PreparedFiniteVolumeRuntime,
)
from ._finite_volume_topology_events import (
    FiniteVolumeTopologyEpoch,
    FiniteVolumeTopologyEventRequest,
    FiniteVolumeTopologyEventTransaction,
    TopologyEventKind,
    TopologyEventStatus,
)


class UnstructuredAMRRuntimeState(StrictModule):
    """Atomic two-level state and the selection frozen for the current step."""

    coarse_state: FiniteVolumeRuntimeState
    fine_state: FiniteVolumeRuntimeState
    selection: UnstructuredAMRSelection

    def __init__(
        self,
        coarse_state: FiniteVolumeRuntimeState,
        fine_state: FiniteVolumeRuntimeState,
        selection: UnstructuredAMRSelection,
        /,
    ):
        if not isinstance(coarse_state, FiniteVolumeRuntimeState):
            raise TypeError("coarse_state must be FiniteVolumeRuntimeState.")
        if not isinstance(fine_state, FiniteVolumeRuntimeState):
            raise TypeError("fine_state must be FiniteVolumeRuntimeState.")
        if not isinstance(selection, UnstructuredAMRSelection):
            raise TypeError("selection must be UnstructuredAMRSelection.")
        if (
            coarse_state.content_state.time.shape != ()
            or fine_state.content_state.time.shape != ()
        ):
            raise ValueError("AMR level times must be scalar.")
        self.coarse_state = coarse_state
        self.fine_state = fine_state
        self.selection = selection

    @property
    def time(self) -> Array:
        return self.coarse_state.time

    @property
    def coarse(self) -> FiniteVolumeRuntimeState:
        return self.coarse_state

    @property
    def fine(self) -> FiniteVolumeRuntimeState:
        return self.fine_state


class UnstructuredAMRRefluxReport(StrictModule):
    """Budget evidence for the one coarse/fine synchronization correction."""

    coarse_integral: Array
    fine_integral: Array
    correction: Array
    maximum_budget_defect: Array
    covered_cell_mask: Array

    def __init__(
        self,
        coarse_integral: ArrayLike,
        fine_integral: ArrayLike,
        correction: ArrayLike,
        covered_cell_mask: ArrayLike,
        /,
    ):
        coarse = jnp.asarray(coarse_integral)
        fine = jnp.asarray(fine_integral)
        delta = jnp.asarray(correction)
        covered = jnp.asarray(covered_cell_mask)
        if coarse.shape != fine.shape or coarse.shape != delta.shape:
            raise ValueError("AMR reflux budget arrays must have identical shapes.")
        if covered.shape != (coarse.shape[0],) or covered.dtype != jnp.bool_:
            raise ValueError("covered_cell_mask must match the coarse cell axis.")
        mask = covered.reshape((covered.shape[0],) + (1,) * (coarse.ndim - 1))
        defect = jnp.where(mask, fine - coarse - delta, jnp.zeros_like(delta))
        self.coarse_integral = coarse
        self.fine_integral = fine
        self.correction = delta
        self.maximum_budget_defect = jnp.max(jnp.abs(defect))
        self.covered_cell_mask = covered


class UnstructuredAMRAdvanceResult(StrictModule):
    """Result of one atomic coarse step and exactly-r fine substeps."""

    runtime_state: UnstructuredAMRRuntimeState
    accepted: Array
    retries: Array
    attempted_step_size: Array
    accepted_step_size: Array
    coarse_advance: FiniteVolumeAdvanceResult | None
    fine_advances: tuple[FiniteVolumeAdvanceResult, ...]
    coarse_accepted_flux_integrals: AcceptedConservationIntegralLedger | None
    fine_accepted_flux_integrals: AcceptedConservationIntegralLedger | None
    fine_substep_ledgers: tuple[AcceptedConservationIntegralLedger, ...]
    reflux_register: UnstructuredAMRFluxRegister
    reflux_report: UnstructuredAMRRefluxReport
    composite_state: Array
    composite_integral: Array
    selection: UnstructuredAMRSelection
    topology_event_request: FiniteVolumeTopologyEventRequest | None
    successor_runtime: Any = eqx.field(default=None)
    regrid_committed: Array = eqx.field(default=False)
    orchestration_failure: str | None = eqx.field(static=True, default=None)

    @property
    def coarse_state(self) -> FiniteVolumeRuntimeState:
        return self.runtime_state.coarse_state

    @property
    def fine_state(self) -> FiniteVolumeRuntimeState:
        return self.runtime_state.fine_state


class UnstructuredAMRRegridArtifact(StrictModule, NonTrainableState):
    """Typed active-set transfer artifact for an accepted AMR regrid."""

    content_state: FiniteVolumeConservativeContentState
    passed: Array
    status: Array
    coverage_error: Array
    conservation_defect: Array
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        content_state: FiniteVolumeConservativeContentState,
        conservation_defect: ArrayLike,
        selection: UnstructuredAMRSelection,
        hierarchy_id: str,
        /,
    ):
        defect = jnp.asarray(conservation_defect)
        passed = jnp.all(jnp.isfinite(defect)) & jnp.all(jnp.abs(defect) <= 1e-10)
        self.content_state = content_state
        self.passed = passed
        self.status = jnp.where(
            passed,
            jnp.asarray(int(TopologyEventStatus.SUCCESS), dtype=jnp.int32),
            jnp.asarray(int(TopologyEventStatus.FAILED_COVERAGE), dtype=jnp.int32),
        )
        self.coverage_error = jnp.asarray(0.0)
        self.conservation_defect = defect
        self.result_id = canonical_fingerprint(
            {
                "kind": "unstructured-amr-active-set-transfer",
                "hierarchy": hierarchy_id,
                "selection": {
                    "coarse": array_tree_fingerprint(selection.coarse_refined),
                    "fine": array_tree_fingerprint(selection.fine_active),
                },
                "topology_epoch_id": content_state.topology_epoch_id,
                "geometry_family_id": content_state.geometry_family_id,
                "content": array_tree_fingerprint(content_state.conservative_content),
                "defect": array_tree_fingerprint(defect),
            }
        )


class _AMRCoarseTemporalStageTrace(StrictModule, NonTrainableState):
    """Prescribe non-owned fine averages from one accepted coarse trace."""

    fine_start_average: Array
    fine_end_average: Array
    fine_owned_mask: Array
    coarse_start_time: Array
    coarse_end_time: Array

    def __call__(self, stage_time: Array, state: Array, /) -> Array:
        interval = self.coarse_end_time - self.coarse_start_time
        fraction = (stage_time - self.coarse_start_time) / interval
        fraction = eqx.error_if(
            fraction,
            ~jnp.isfinite(fraction) | (fraction < 0.0) | (fraction > 1.0),
            "Fine SSPRK stage time lies outside the accepted coarse trace.",
        )
        trace = (
            1.0 - fraction
        ) * self.fine_start_average + fraction * self.fine_end_average
        mask = self.fine_owned_mask.reshape(
            self.fine_owned_mask.shape + (1,) * (state.ndim - 1)
        )
        return jnp.where(mask, state, trace)


class PreparedUnstructuredAMRRuntime(StrictModule, NonTrainableState):
    """Single-device fixed-hierarchy AMR orchestration over prepared FV runtimes.

    The level runtimes continue to own their SSPRK stages, positivity and retry
    logic. This class owns coarse/fine subcycling, ledgers, reflux, and events.
    """

    hierarchy: UnstructuredAMRHierarchyPlan
    coarse_runtime: PreparedFiniteVolumeRuntime
    fine_runtime: PreparedFiniteVolumeRuntime
    refinement_ratio: int = eqx.field(static=True)
    policy: FiniteVolumeStepPolicy
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        hierarchy: UnstructuredAMRHierarchyPlan,
        coarse_runtime: PreparedFiniteVolumeRuntime,
        fine_runtime: PreparedFiniteVolumeRuntime,
        /,
        *,
        refinement_ratio: int = 2,
        policy: FiniteVolumeStepPolicy | None = None,
    ):
        if not isinstance(hierarchy, UnstructuredAMRHierarchyPlan):
            raise TypeError("hierarchy must be UnstructuredAMRHierarchyPlan.")
        if not isinstance(coarse_runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("coarse_runtime must be PreparedFiniteVolumeRuntime.")
        if not isinstance(fine_runtime, PreparedFiniteVolumeRuntime):
            raise TypeError("fine_runtime must be PreparedFiniteVolumeRuntime.")
        if not isinstance(
            coarse_runtime.dynamics, PreparedUnstructuredFiniteVolumeDynamics
        ) or not isinstance(
            fine_runtime.dynamics, PreparedUnstructuredFiniteVolumeDynamics
        ):
            raise TypeError("AMR levels require prepared unstructured FV dynamics.")
        if not isinstance(refinement_ratio, (int, np.integer)) or isinstance(
            refinement_ratio, (bool, np.bool_)
        ):
            raise TypeError("refinement_ratio must be an integer greater than one.")
        ratio = int(refinement_ratio)
        if ratio <= 1:
            raise ValueError("refinement_ratio must be an integer greater than one.")
        if ratio != hierarchy.refinement_ratio:
            raise ValueError("refinement_ratio must match the prepared hierarchy.")
        coarse_disc = coarse_runtime.dynamics.discretization
        fine_disc = fine_runtime.dynamics.discretization
        if coarse_disc.prepared_id != hierarchy.coarse.prepared_id:
            raise ValueError("coarse_runtime does not match the AMR coarse geometry.")
        if fine_disc.prepared_id != hierarchy.fine.prepared_id:
            raise ValueError("fine_runtime does not match the AMR fine geometry.")
        if (
            coarse_runtime.dynamics.system.component_count
            != fine_runtime.dynamics.system.component_count
        ):
            raise ValueError("AMR levels must have the same component count.")
        coarse_event_policy = coarse_runtime.dynamics.coupling.topology_event_policy
        fine_event_policy = fine_runtime.dynamics.coupling.topology_event_policy
        if (coarse_event_policy == "accepted_step") != (
            fine_event_policy == "accepted_step"
        ):
            raise ValueError(
                "AMR levels must both use accepted-step topology events or both "
                "disable them."
            )
        policy_ = coarse_runtime.policy if policy is None else policy
        if not isinstance(policy_, FiniteVolumeStepPolicy):
            raise TypeError("policy must be FiniteVolumeStepPolicy.")
        self.hierarchy = hierarchy
        self.coarse_runtime = coarse_runtime
        self.fine_runtime = fine_runtime
        self.refinement_ratio = ratio
        self.policy = policy_
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-unstructured-amr-runtime",
                "hierarchy": hierarchy.plan_id,
                "coarse_runtime": coarse_runtime.runtime_id,
                "fine_runtime": fine_runtime.runtime_id,
                "refinement_ratio": ratio,
                "policy": policy_.policy_id,
                "fine_stage_state_provider": "coarse-linear-accepted-trace-v1",
            }
        )

    @property
    def coarse_dynamics(self) -> PreparedUnstructuredFiniteVolumeDynamics:
        dynamics = self.coarse_runtime.dynamics
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("AMR coarse dynamics are not unstructured.")
        return dynamics

    @property
    def fine_dynamics(self) -> PreparedUnstructuredFiniteVolumeDynamics:
        dynamics = self.fine_runtime.dynamics
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("AMR fine dynamics are not unstructured.")
        return dynamics

    def _accepted_step_regrids_enabled(self) -> bool:
        return (
            self.coarse_dynamics.coupling.topology_event_policy == "accepted_step"
            and self.fine_dynamics.coupling.topology_event_policy == "accepted_step"
        )

    def _require_accepted_step_regrids(self, source: str, /) -> None:
        if not self._accepted_step_regrids_enabled():
            raise ValueError(
                f"{source} requires topology_event_policy='accepted_step' on "
                "both AMR levels."
            )

    @staticmethod
    def _selection_changed(
        current: UnstructuredAMRSelection,
        successor: UnstructuredAMRSelection,
        /,
    ) -> bool:
        return not np.array_equal(
            np.asarray(successor.coarse_refined),
            np.asarray(current.coarse_refined),
        ) or not np.array_equal(
            np.asarray(successor.fine_active),
            np.asarray(current.fine_active),
        )

    def _selection(
        self,
        indicator: ArrayLike | None,
        threshold: ArrayLike,
        /,
    ) -> UnstructuredAMRSelection:
        values = (
            jnp.zeros((self.hierarchy.coarse.cell_count,), dtype=jnp.float64)
            if indicator is None
            else jnp.asarray(indicator)
        )
        return self.hierarchy.select(values, threshold)

    def _validate_selection(self, selection: UnstructuredAMRSelection, /) -> None:
        if not isinstance(selection, UnstructuredAMRSelection):
            raise TypeError("selection must be UnstructuredAMRSelection.")
        if selection.coarse_refined.shape != (self.hierarchy.coarse.cell_count,):
            raise ValueError("Selection coarse mask does not match the hierarchy.")
        if selection.fine_active.shape != (self.hierarchy.fine.cell_count,):
            raise ValueError("Selection fine mask does not match the hierarchy.")
        if bool(np.asarray(selection.capacity_overflow)):
            raise ValueError("AMR tag capacity overflow requires an explicit policy.")
        # Refining a cell whose prepared EB geometry is inactive is undefined:
        # the hierarchy must be re-tagged rather than silently creating content.
        coarse_active = self.coarse_runtime.active_cell_mask
        fine_active = self.fine_runtime.active_cell_mask
        if bool(
            np.asarray(
                jnp.any(selection.coarse_refined & ~coarse_active)
                | jnp.any(selection.fine_active & ~fine_active)
            )
        ):
            raise ValueError("AMR selection includes an inactive EB cell.")

    def initialize_state(
        self,
        coarse_or_state: ArrayLike | FiniteVolumeRuntimeState,
        *values: Any,
        fine_state: FiniteVolumeRuntimeState | None = None,
        fine_cell_average: ArrayLike | None = None,
        selection: UnstructuredAMRSelection | None = None,
        indicator: ArrayLike | None = None,
        threshold: ArrayLike = 0.0,
        **kwargs: Any,
    ) -> UnstructuredAMRRuntimeState:
        """Initialize one AMR transaction.

        ``coarse_or_state, time, step_size`` is the compact form.  A fine
        average may be supplied as ``fine_cell_average=...`` or by using the
        positional form ``coarse_average, fine_average, time, step_size``.
        Existing coarse/fine runtime states are accepted for restart/checkpoint
        paths and are never mutated.
        """
        keyword_time = kwargs.pop("time", None)
        keyword_step_size = kwargs.pop("step_size", None)
        if len(values) == 2:
            time, step_size = values
        elif len(values) == 3:
            if fine_state is not None or fine_cell_average is not None:
                raise TypeError("Fine state/average was supplied twice.")
            fine_candidate, time, step_size = values
            if isinstance(fine_candidate, FiniteVolumeRuntimeState):
                fine_state = fine_candidate
            else:
                fine_cell_average = fine_candidate
        elif len(values) == 0 and isinstance(coarse_or_state, FiniteVolumeRuntimeState):
            time = coarse_or_state.time
            step_size = coarse_or_state.step_size
        elif (
            len(values) == 0
            and keyword_time is not None
            and keyword_step_size is not None
        ):
            time = keyword_time
            step_size = keyword_step_size
        else:
            raise TypeError(
                "initialize_state expects coarse,time,step_size or "
                "coarse,fine,time,step_size."
            )
        if keyword_time is not None and len(values) > 0:
            raise TypeError("time was supplied both positionally and by keyword.")
        if keyword_step_size is not None and len(values) > 0:
            raise TypeError("step_size was supplied both positionally and by keyword.")
        if kwargs:
            allowed = {
                "motion_args",
                "accepted_step",
                "last_status",
                "controller_state",
                "integrator_state",
                "output_cursor",
            }
            unknown = set(kwargs) - allowed
            if unknown:
                raise TypeError(f"Unknown initialization options: {sorted(unknown)}")
        if selection is None:
            selection = self._selection(indicator, threshold)
        self._validate_selection(selection)

        if isinstance(coarse_or_state, FiniteVolumeRuntimeState):
            coarse = coarse_or_state
            if coarse.content_state.time != self.coarse_runtime.precision.decision(time):
                raise ValueError("Restart coarse state time does not match time.")
        else:
            coarse = self.coarse_runtime.initialize_state(
                coarse_or_state,
                time,
                step_size,
                **kwargs,
            )

        if fine_state is not None:
            if not isinstance(fine_state, FiniteVolumeRuntimeState):
                raise TypeError("fine_state must be FiniteVolumeRuntimeState.")
            fine = fine_state
        else:
            if fine_cell_average is None:
                coarse_average = coarse.content_state.cell_average()
                fine_average = self.hierarchy.prolong(coarse_average)
            else:
                fine_average = jnp.asarray(fine_cell_average)
            fine = self.fine_runtime.initialize_state(
                fine_average,
                coarse.time,
                self.fine_runtime.precision.decision(
                    self.coarse_runtime.precision.reduction(step_size)
                    / self.refinement_ratio
                ),
                **kwargs,
            )
        self._validate_level_state(coarse, self.coarse_runtime)
        self._validate_level_state(fine, self.fine_runtime)
        if coarse.time != fine.time:
            raise ValueError("AMR levels must start at exactly the same time.")
        return UnstructuredAMRRuntimeState(coarse, fine, selection)

    def _validate_level_state(
        self,
        state: FiniteVolumeRuntimeState,
        runtime: PreparedFiniteVolumeRuntime,
        /,
    ) -> None:
        content = state.content_state
        dynamics = runtime.dynamics
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("AMR level dynamics must be prepared and unstructured.")
        if content.cell_count != dynamics.discretization.cell_count:
            raise ValueError("AMR state cell count does not match its dynamics.")
        if content.precision.policy_id != dynamics.precision.policy_id:
            raise ValueError("AMR state precision does not match its dynamics.")
        if content.geometry_family_id != runtime.geometry_family_id:
            raise ValueError("AMR state geometry family does not match its runtime.")
        if content.topology_epoch_id != runtime.topology_epoch_id:
            raise ValueError("AMR state topology epoch does not match its runtime.")
        average = content.cell_average().reshape((-1, dynamics.system.component_count))
        active = np.asarray(content.active_cell_mask)
        admissible = dynamics.system.admissible(average)
        if not bool(np.asarray(jnp.all(jnp.where(active, admissible, True)))):
            raise ValueError("AMR state contains a non-admissible EOS state.")
        coupling = dynamics.coupling
        embedded = coupling.embedded_metrics
        if embedded is not None and bool(
            np.asarray(jnp.any(content.active_cell_mask != embedded.active_fluid_cells))
        ):
            raise ValueError("AMR state activity does not match prepared EB metrics.")

    def _state_with_content(
        self,
        state: FiniteVolumeRuntimeState,
        content: Array,
        /,
        *,
        time: ArrayLike | None = None,
        step_size: ArrayLike | None = None,
    ) -> FiniteVolumeRuntimeState:
        original = state.content_state
        next_content = original.with_content(
            content,
            time=original.time if time is None else time,
        )
        return FiniteVolumeRuntimeState(
            next_content,
            state.topology_journal,
            state.step_size if step_size is None else step_size,
            accepted_step=state.accepted_step,
            last_status=state.last_status,
            controller_state=state.controller_state,
            integrator_state=state.integrator_state,
            output_cursor=state.output_cursor,
        )

    def fill_fine_ghost(
        self,
        fine_state: FiniteVolumeRuntimeState,
        coarse_start: FiniteVolumeRuntimeState,
        coarse_end: FiniteVolumeRuntimeState,
        selection: UnstructuredAMRSelection,
        fraction: ArrayLike,
        /,
    ) -> FiniteVolumeRuntimeState:
        """Fill non-owned fine cells from linearly interpolated coarse content."""
        self._validate_selection(selection)
        if coarse_start.time == coarse_end.time:
            raise ValueError("Coarse temporal ghost interval must be nonzero.")
        fraction_ = jnp.asarray(
            fraction, dtype=coarse_start.content_state.time.dtype
        ).reshape(())
        fraction_ = eqx.error_if(
            fraction_,
            ~jnp.isfinite(fraction_) | (fraction_ < 0.0) | (fraction_ > 1.0),
            "AMR ghost interpolation fraction must lie in [0, 1].",
        )
        q0 = coarse_start.content_state.cell_average()
        q1 = coarse_end.content_state.cell_average()
        coarse_average = (1.0 - fraction_) * q0 + fraction_ * q1
        fine_ghost_average = self.hierarchy.prolong(coarse_average)
        current_average = fine_state.content_state.cell_average()
        mask = selection.fine_active.reshape(
            selection.fine_active.shape + (1,) * (current_average.ndim - 1)
        )
        filled_average = jnp.where(mask, current_average, fine_ghost_average)
        content = (
            filled_average
            * fine_state.content_state.effective_cell_volumes.reshape(
                (-1,) + (1,) * (filled_average.ndim - 1)
            )
        )
        return self._state_with_content(fine_state, content)

    def _fine_stage_state_provider(
        self,
        coarse_start: FiniteVolumeRuntimeState,
        coarse_end: FiniteVolumeRuntimeState,
        selection: UnstructuredAMRSelection,
        substep: int,
        /,
    ) -> FiniteVolumeStageStateProvider:
        """Bind one fine substep to the accepted coarse temporal trace."""
        self._validate_selection(selection)
        if coarse_start.time == coarse_end.time:
            raise ValueError("Coarse temporal stage-trace interval must be nonzero.")
        fine_start_average = self.hierarchy.prolong(
            coarse_start.content_state.cell_average()
        )
        fine_end_average = self.hierarchy.prolong(coarse_end.content_state.cell_average())
        interval = coarse_end.time - coarse_start.time
        substep_start = coarse_start.time + (
            jnp.asarray(substep, dtype=interval.dtype) * interval / self.refinement_ratio
        )
        substep_end = coarse_start.time + (
            jnp.asarray(substep + 1, dtype=interval.dtype)
            * interval
            / self.refinement_ratio
        )
        provider_id = canonical_fingerprint(
            {
                "kind": "amr-coarse-temporal-stage-trace",
                "hierarchy": self.hierarchy.plan_id,
                "coarse_runtime": self.coarse_runtime.runtime_id,
                "fine_runtime": self.fine_runtime.runtime_id,
                "coarse_start_content": array_tree_fingerprint(
                    np.asarray(coarse_start.content_state.conservative_content)
                ),
                "coarse_end_content": array_tree_fingerprint(
                    np.asarray(coarse_end.content_state.conservative_content)
                ),
                "coarse_start_time": array_tree_fingerprint(
                    np.asarray(coarse_start.time)
                ),
                "coarse_end_time": array_tree_fingerprint(np.asarray(coarse_end.time)),
                "fine_owned_mask": array_tree_fingerprint(
                    np.asarray(selection.fine_active)
                ),
                "substep": int(substep),
                "substep_start": array_tree_fingerprint(np.asarray(substep_start)),
                "substep_end": array_tree_fingerprint(np.asarray(substep_end)),
            }
        )
        return FiniteVolumeStageStateProvider(
            _AMRCoarseTemporalStageTrace(
                fine_start_average,
                fine_end_average,
                selection.fine_active,
                coarse_start.time,
                coarse_end.time,
            ),
            provider_id=provider_id,
        )

    def transfer_volume_fraction(
        self,
        coarse_volume_fraction: ArrayLike,
        /,
    ) -> Array:
        """Conservative bounded prolongation of a VOF cell fraction."""
        alpha = jnp.asarray(coarse_volume_fraction)
        coupling = self.coarse_dynamics.coupling
        if coupling.vof is not None:
            coupling.vof.validate_volume_fraction(alpha)
        fine = self.hierarchy.prolong(alpha)
        coupling_fine = self.fine_dynamics.coupling
        if coupling_fine.vof is not None:
            coupling_fine.vof.validate_volume_fraction(fine)
        return eqx.error_if(
            fine,
            jnp.any(~jnp.isfinite(fine) | (fine < 0.0) | (fine > 1.0)),
            "AMR VOF transfer violated [0, 1].",
        )

    def restrict_volume_fraction(
        self,
        fine_volume_fraction: ArrayLike,
        /,
    ) -> Array:
        """Conservative bounded restriction of a fine VOF fraction."""
        alpha = jnp.asarray(fine_volume_fraction)
        if self.fine_dynamics.coupling.vof is not None:
            self.fine_dynamics.coupling.vof.validate_volume_fraction(alpha)
        coarse = self.hierarchy.restrict(alpha)
        if self.coarse_dynamics.coupling.vof is not None:
            self.coarse_dynamics.coupling.vof.validate_volume_fraction(coarse)
        return eqx.error_if(
            coarse,
            jnp.any(~jnp.isfinite(coarse) | (coarse < 0.0) | (coarse > 1.0)),
            "AMR VOF restriction violated [0, 1].",
        )

    @staticmethod
    def _close(first: ArrayLike, second: ArrayLike, /) -> bool:
        left = np.asarray(first)
        right = np.asarray(second)
        if left.shape != right.shape:
            return False
        if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            return False
        dtype = np.result_type(left.dtype, right.dtype, np.float64)
        eps = np.finfo(dtype).eps
        scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
        return bool(np.all(np.abs(left - right) <= 128.0 * eps * scale))

    def _validate_ledger_interval(
        self,
        ledger: AcceptedConservationIntegralLedger,
        start: FiniteVolumeRuntimeState,
        end: FiniteVolumeRuntimeState,
        /,
    ) -> None:
        if not isinstance(ledger, AcceptedConservationIntegralLedger):
            raise TypeError("Accepted level output must contain an accepted ledger.")
        if ledger.units != "content":
            raise ValueError("AMR synchronization requires content-unit ledgers.")
        if ledger.start_topology_epoch_id != ledger.end_topology_epoch_id:
            raise ValueError("An AMR interface ledger cannot cross a topology epoch.")
        if ledger.start_topology_epoch_id != start.content_state.topology_epoch_id:
            raise ValueError("Accepted ledger starts in the wrong topology epoch.")
        if ledger.end_topology_epoch_id != end.content_state.topology_epoch_id:
            raise ValueError("Accepted ledger ends in the wrong topology epoch.")
        if start.content_state.geometry_family_id != end.content_state.geometry_family_id:
            raise ValueError("AMR level states changed geometry family.")
        if ledger.geometry_family_id != start.content_state.geometry_family_id:
            raise ValueError("Accepted ledger belongs to the wrong geometry family.")
        if not self._close(ledger.start_time, start.time) or not self._close(
            ledger.end_time, end.time
        ):
            raise ValueError("Accepted ledger interval does not match the level state.")
        if not self._close(
            ledger.end_time - ledger.start_time,
            end.time - start.time,
        ):
            raise ValueError("Accepted ledger has an inconsistent time interval.")

    def _aggregate_fine_ledgers(
        self,
        ledgers: tuple[AcceptedConservationIntegralLedger, ...],
        start: FiniteVolumeRuntimeState,
        end: FiniteVolumeRuntimeState,
        /,
    ) -> AcceptedConservationIntegralLedger:
        if len(ledgers) != self.refinement_ratio:
            raise ValueError("AMR must aggregate exactly refinement_ratio ledgers.")
        if start.content_state.geometry_family_id != end.content_state.geometry_family_id:
            raise ValueError("Fine AMR level states changed geometry family.")
        for ledger in ledgers:
            if not isinstance(ledger, AcceptedConservationIntegralLedger):
                raise TypeError("Accepted fine output must contain an accepted ledger.")
            if ledger.units != "content":
                raise ValueError("AMR synchronization requires content-unit ledgers.")
            if ledger.geometry_family_id != start.content_state.geometry_family_id:
                raise ValueError(
                    "Fine accepted ledger belongs to the wrong geometry family."
                )
            if (
                ledger.start_topology_epoch_id != start.content_state.topology_epoch_id
                or ledger.end_topology_epoch_id != end.content_state.topology_epoch_id
            ):
                raise ValueError("Fine accepted ledgers changed topology epoch.")
        first = ledgers[0]
        for previous, current in zip(ledgers[:-1], ledgers[1:], strict=True):
            if not self._close(previous.end_time, current.start_time):
                raise ValueError("Fine accepted ledger intervals are not contiguous.")
            if previous.geometry_family_id != current.geometry_family_id:
                raise ValueError("Fine accepted ledgers changed geometry family.")
            if previous.geometry_layout_id != current.geometry_layout_id:
                raise ValueError("Fine accepted ledgers changed geometry layout.")
            if previous.evidence_policy_id != current.evidence_policy_id:
                raise ValueError("Fine accepted ledgers changed evidence policy.")
            if previous.start_topology_epoch_id != current.start_topology_epoch_id:
                raise ValueError("Fine accepted ledgers changed topology epoch.")
        if not self._close(first.start_time, start.time) or not self._close(
            ledgers[-1].end_time, end.time
        ):
            raise ValueError(
                "Fine accepted ledger union does not equal the coarse interval."
            )
        if any(len(item.blocks) != len(first.blocks) for item in ledgers[1:]):
            raise ValueError("Fine accepted ledgers changed route count.")
        blocks = []
        for index, reference in enumerate(first.blocks):
            integral = jnp.zeros_like(reference.flux_integral)
            for ledger in ledgers:
                block = ledger.blocks[index]
                if (
                    block.block_id != reference.block_id
                    or block.block_kind != reference.block_kind
                    or block.route_id != reference.route_id
                ):
                    raise ValueError("Fine accepted ledgers changed a flux route.")
                integral = integral + block.flux_integral
            blocks.append(
                AcceptedConservationFluxIntegralBlock(
                    integral,
                    reference.owner_cells,
                    reference.neighbour_cells,
                    reference.active_mask,
                    reference.block_id,
                    reference.block_kind,
                )
            )
        source = jnp.zeros_like(first.source_integral)
        for ledger in ledgers:
            source = source + ledger.source_integral
        return AcceptedConservationIntegralLedger(
            tuple(blocks),
            source,
            first.active_cell_mask,
            geometry_family_id=first.geometry_family_id,
            geometry_layout_id=first.geometry_layout_id,
            stage_geometry_versions=first.stage_geometry_versions,
            start_geometry_version=first.start_geometry_version,
            end_geometry_version=ledgers[-1].end_geometry_version,
            evidence_policy_id=first.evidence_policy_id,
            stage_evidence_versions=first.stage_evidence_versions,
            start_evidence_version=first.start_evidence_version,
            end_evidence_version=ledgers[-1].end_evidence_version,
            start_topology_epoch_id=first.start_topology_epoch_id,
            end_topology_epoch_id=ledgers[-1].end_topology_epoch_id,
            start_time=first.start_time,
            end_time=ledgers[-1].end_time,
            accepted_step=ledgers[-1].accepted_step,
        )

    def _interface_scatter(
        self,
        ledger: AcceptedConservationIntegralLedger,
        endpoint_mask: Array,
        route_ids: tuple[str, ...] | None = None,
        allowed_cells: Array | None = None,
        /,
    ) -> Array:
        """Scatter only certified coarse/fine interface routes.

        The hierarchy's explicit map is the certificate that ownership-derived
        crossing routes are meaningful. Physical boundaries and same-ownership
        routes are excluded; their accepted integrals cannot participate in
        reflux.
        """
        try:
            interface = self.hierarchy.coarse_fine_interface_map
        except AttributeError as error:
            raise ValueError(
                "AMR reflux requires an explicit coarse_fine_interface_map; "
                "whole-cell ledger correction is unsupported."
            ) from error
        interface_array = np.asarray(interface)
        if (
            interface_array.ndim != 2
            or interface_array.shape[1] != 2
            or interface_array.shape[0] == 0
        ):
            raise ValueError("AMR coarse_fine_interface_map is not certified.")
        if ledger.units != "content":
            raise ValueError("AMR interface filtering requires content-unit ledgers.")
        if ledger.start_topology_epoch_id != ledger.end_topology_epoch_id:
            raise ValueError("AMR interface ledger crosses a topology epoch.")
        if route_ids is not None:
            present_routes = {block.route_id for block in ledger.blocks}
            missing_routes = set(route_ids) - present_routes
            if missing_routes:
                raise ValueError(
                    "AMR interface route IDs are absent from the accepted ledger."
                )
        scattered = jnp.zeros(
            (ledger.cell_count,) + ledger.component_shape,
            dtype=ledger.source_integral.dtype,
        )
        for block in ledger.blocks:
            if route_ids is not None and block.route_id not in route_ids:
                continue
            owner = block.owner_cells
            neighbour = block.neighbour_cells
            safe_neighbour = jnp.maximum(neighbour, 0)
            crosses = (
                block.active_mask
                & (neighbour >= 0)
                & (endpoint_mask[owner] != endpoint_mask[safe_neighbour])
            )
            values = jnp.where(
                crosses.reshape(crosses.shape + (1,) * (block.flux_integral.ndim - 1)),
                block.flux_integral,
                jnp.zeros_like(block.flux_integral),
            )
            scattered = scattered.at[owner].add(-values)
            scattered = scattered.at[safe_neighbour].add(
                jnp.where(
                    (neighbour >= 0).reshape(neighbour.shape + (1,) * (values.ndim - 1)),
                    values,
                    jnp.zeros_like(values),
                )
            )
        if allowed_cells is not None:
            allowed_host = np.asarray(allowed_cells)
            if np.any(allowed_host < 0) or np.any(allowed_host >= ledger.cell_count):
                raise ValueError("AMR interface endpoint cell is out of range.")
            allowed = (
                jnp.zeros((ledger.cell_count,), dtype=jnp.bool_)
                .at[allowed_cells]
                .set(True)
            )
            scattered = jnp.where(
                allowed.reshape(allowed.shape + (1,) * (scattered.ndim - 1)),
                scattered,
                jnp.zeros_like(scattered),
            )
        return scattered

    def _interface_cells(
        self,
        level: str,
        /,
    ) -> Array:
        """Return the explicit hierarchy interface endpoint cells."""
        cells = (
            self.hierarchy.coarse_fine_interface_coarse_cells
            if level == "coarse"
            else self.hierarchy.coarse_fine_interface_fine_cells
        )
        values = jnp.asarray(cells, dtype=jnp.int32).reshape(-1)
        if values.size == 0:
            raise ValueError("AMR interface map contains no endpoint cells.")
        return values

    def _interface_route_ids(
        self,
        level: str,
        /,
    ) -> tuple[str, ...]:
        """Return the explicit per-level interface route IDs."""
        route_ids = (
            self.hierarchy.coarse_interface_route_ids
            if level == "coarse"
            else self.hierarchy.fine_interface_route_ids
        )
        if not route_ids:
            raise ValueError("AMR interface map contains no route IDs.")
        return tuple(route_ids)

    def _reflux(
        self,
        coarse_ledger: AcceptedConservationIntegralLedger,
        fine_ledger: AcceptedConservationIntegralLedger,
        selection: UnstructuredAMRSelection,
        fine_ledgers: tuple[AcceptedConservationIntegralLedger, ...],
        /,
    ) -> tuple[UnstructuredAMRFluxRegister, UnstructuredAMRRefluxReport]:
        """Build one interface-only coarse/fine accepted-flux mismatch."""
        try:
            interface_route_id = self.hierarchy.interface_route_id
            interface_layout_id = self.hierarchy.interface_layout_id
        except AttributeError as error:
            raise ValueError(
                "AMR reflux requires certified interface route/layout identities; "
                "whole-cell ledger correction is unsupported."
            ) from error
        coarse_route_ids = self._interface_route_ids("coarse")
        fine_route_ids = self._interface_route_ids("fine")
        if not coarse_route_ids or not fine_route_ids:
            raise ValueError("whole-cell correction is unsupported.")
        coarse_interface = self._interface_scatter(
            coarse_ledger,
            selection.coarse_refined,
            coarse_route_ids,
            self._interface_cells("coarse"),
        )
        fine_interface = self._interface_scatter(
            fine_ledger,
            selection.fine_active,
            fine_route_ids,
            self._interface_cells("fine"),
        )
        fine_interface = self.hierarchy.restrict_content(
            fine_interface,
            fine_active_mask=jnp.ones((self.hierarchy.fine.cell_count,), dtype=jnp.bool_),
            coarse_active_mask=jnp.ones(
                (self.hierarchy.coarse.cell_count,), dtype=jnp.bool_
            ),
            fine_volumes=self.hierarchy.fine.cell_volumes,
        )
        active_fine_ledgers = fine_ledgers
        register = UnstructuredAMRFluxRegister(
            fine_interface - coarse_interface,
            fine_interface,
            coarse_flux_integral=coarse_interface,
            coarse_interval=(
                coarse_ledger.start_time,
                coarse_ledger.end_time,
            ),
            fine_intervals=tuple(
                (ledger.start_time, ledger.end_time) for ledger in active_fine_ledgers
            )
            or ((fine_ledger.start_time, fine_ledger.end_time),),
            accepted_steps=tuple(ledger.accepted_step for ledger in active_fine_ledgers)
            or (fine_ledger.accepted_step,),
            route_id=interface_route_id,
            layout_id=interface_layout_id,
            coarse_topology_id=coarse_ledger.start_topology_epoch_id,
            fine_topology_id=fine_ledger.start_topology_epoch_id,
        )
        report = UnstructuredAMRRefluxReport(
            coarse_interface,
            fine_interface,
            register.integrated_correction,
            selection.coarse_refined,
        )
        return register, report

    def _event_request(
        self,
        state: FiniteVolumeRuntimeState,
        selection: UnstructuredAMRSelection,
        /,
    ) -> FiniteVolumeTopologyEventRequest:
        self._require_accepted_step_regrids("AMR regrid")
        payload_id = canonical_fingerprint(
            {
                "kind": "unstructured-amr-selection",
                "hierarchy": self.hierarchy.plan_id,
                "coarse_refined": array_tree_fingerprint(
                    np.asarray(selection.coarse_refined)
                ),
                "fine_active": array_tree_fingerprint(np.asarray(selection.fine_active)),
                "selected_count": int(np.asarray(selection.selected_count)),
                "overflow": bool(np.asarray(selection.capacity_overflow)),
                "accepted_step": int(np.asarray(state.accepted_step)),
            }
        )
        return FiniteVolumeTopologyEventRequest(
            TopologyEventKind.AMR_REGRID,
            state.content_state.topology_epoch_id,
            self.hierarchy.plan_id,
            payload_id=payload_id,
            reason="accepted two-level AMR tag",
        )

    def _append_event(
        self,
        state: FiniteVolumeRuntimeState,
        request: FiniteVolumeTopologyEventRequest | None,
        /,
    ) -> FiniteVolumeRuntimeState:
        if request is None:
            return state
        journal = state.topology_journal.append_requested(
            request,
            state.accepted_step,
            state.time,
        )
        if bool(np.asarray(journal.overflowed)):
            raise ValueError("AMR topology-event journal capacity overflowed.")
        return FiniteVolumeRuntimeState(
            state.content_state,
            journal,
            state.step_size,
            accepted_step=state.accepted_step,
            last_status=state.last_status,
            controller_state=state.controller_state,
            integrator_state=state.integrator_state,
            output_cursor=state.output_cursor,
        )

    def advance(
        self,
        runtime_state: UnstructuredAMRRuntimeState | FiniteVolumeRuntimeState,
        args: Any = None,
        /,
        *,
        indicator: ArrayLike | None = None,
        threshold: ArrayLike = 0.0,
        selection: UnstructuredAMRSelection | None = None,
    ) -> UnstructuredAMRAdvanceResult:
        """Advance one accepted coarse interval, or return the original state.

        The fine level is accepted only when all ``r`` substeps succeed and
        their accepted intervals form exactly the coarse accepted interval.
        No partial level state or topology journal is published on failure.
        """
        requested_selection: UnstructuredAMRSelection | None = None
        if isinstance(runtime_state, FiniteVolumeRuntimeState):
            if selection is None:
                selection = self._selection(indicator, threshold)
            self._validate_selection(selection)
            runtime_state = UnstructuredAMRRuntimeState(
                runtime_state,
                self.fine_runtime.initialize_state(
                    self.hierarchy.prolong(runtime_state.content_state.cell_average()),
                    runtime_state.time,
                    runtime_state.step_size / self.refinement_ratio,
                ),
                selection,
            )
        elif selection is not None:
            self._validate_selection(selection)
            requested_selection = selection
        if not isinstance(runtime_state, UnstructuredAMRRuntimeState):
            raise TypeError("runtime_state must be UnstructuredAMRRuntimeState.")
        selection = runtime_state.selection
        self._validate_selection(selection)
        if requested_selection is not None and self._selection_changed(
            selection, requested_selection
        ):
            self._require_accepted_step_regrids("Explicit AMR selection change")
        fine_original = runtime_state.fine_state
        coarse_original = runtime_state.coarse_state
        if not self._close(coarse_original.time, fine_original.time):
            raise ValueError("AMR levels must have a common start time.")

        coarse_result = self.coarse_runtime.advance(coarse_original, args)
        coarse_accepted = bool(np.asarray(coarse_result.accepted))
        if not coarse_accepted:
            empty_register = UnstructuredAMRFluxRegister(
                jnp.zeros_like(coarse_original.content_state.conservative_content)
            )
            empty_report = UnstructuredAMRRefluxReport(
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                selection.coarse_refined,
            )
            return UnstructuredAMRAdvanceResult(
                runtime_state,
                jnp.asarray(False),
                coarse_result.retries,
                coarse_result.attempted_step_size,
                jnp.asarray(0.0, dtype=coarse_original.time.dtype),
                coarse_result,
                tuple(),
                None,
                None,
                tuple(),
                empty_register,
                empty_report,
                self.hierarchy.composite_integral(
                    coarse_original.content_state.cell_average(),
                    fine_original.content_state.cell_average(),
                    selection,
                ),
                self.hierarchy.composite_integral(
                    coarse_original.content_state.cell_average(),
                    fine_original.content_state.cell_average(),
                    selection,
                ),
                selection,
                None,
            )
        coarse_next = coarse_result.runtime_state
        coarse_ledger = coarse_result.accepted_flux_integrals
        self._validate_ledger_interval(coarse_ledger, coarse_original, coarse_next)
        dt = coarse_next.time - coarse_original.time
        fine_dt = dt / self.refinement_ratio
        fine_work = self._state_with_content(
            fine_original,
            fine_original.content_state.conservative_content,
            step_size=fine_dt,
        )
        fine_advances: list[FiniteVolumeAdvanceResult] = []
        fine_ledgers: list[AcceptedConservationIntegralLedger] = []
        fine_start = fine_work
        fine_failed = False
        for substep in range(self.refinement_ratio):
            stage_state_provider = self._fine_stage_state_provider(
                coarse_original,
                coarse_next,
                selection,
                substep,
            )
            fine_attempt_runtime = self.fine_runtime.with_stage_state_provider(
                stage_state_provider
            )
            fine_result = fine_attempt_runtime.advance(fine_work, args)
            fine_advances.append(fine_result)
            if not bool(np.asarray(fine_result.accepted)):
                fine_failed = True
                break
            accepted_dt = fine_result.accepted_step_size
            if not self._close(accepted_dt, fine_dt):
                raise ValueError("Fine accepted substep does not match dt/r exactly.")
            fine_next = fine_result.runtime_state
            ledger = fine_result.accepted_flux_integrals
            self._validate_ledger_interval(ledger, fine_start, fine_next)
            fine_ledgers.append(ledger)
            fine_start = fine_next
            fine_work = fine_next
        if fine_failed or len(fine_ledgers) != self.refinement_ratio:
            # The coarse prediction and every fine attempt are intentionally
            # discarded.  In particular, no event request is appended.
            empty_register = UnstructuredAMRFluxRegister(
                jnp.zeros_like(coarse_original.content_state.conservative_content)
            )
            empty_report = UnstructuredAMRRefluxReport(
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                selection.coarse_refined,
            )
            return UnstructuredAMRAdvanceResult(
                runtime_state,
                jnp.asarray(False),
                coarse_result.retries,
                coarse_result.attempted_step_size,
                jnp.asarray(0.0, dtype=coarse_original.time.dtype),
                coarse_result,
                tuple(fine_advances),
                None,
                None,
                tuple(fine_ledgers),
                empty_register,
                empty_report,
                self.hierarchy.composite_integral(
                    coarse_original.content_state.cell_average(),
                    fine_original.content_state.cell_average(),
                    selection,
                ),
                self.hierarchy.composite_integral(
                    coarse_original.content_state.cell_average(),
                    fine_original.content_state.cell_average(),
                    selection,
                ),
                selection,
                None,
            )
        fine_final = fine_work
        if not self._close(fine_final.time, coarse_next.time):
            raise ValueError("Fine substep interval union does not end at coarse time.")
        fine_ledger = self._aggregate_fine_ledgers(
            tuple(fine_ledgers), fine_original, fine_final
        )
        register, report = self._reflux(
            coarse_ledger, fine_ledger, selection, tuple(fine_ledgers)
        )
        coarse_average = self.hierarchy.reflux(
            coarse_next.content_state.cell_average(), register
        )
        synchronized_average = self.hierarchy.synchronize(
            coarse_average,
            fine_final.content_state.cell_average(),
            selection,
        )
        coarse_content = (
            synchronized_average
            * coarse_next.content_state.effective_cell_volumes.reshape(
                (-1,) + (1,) * (synchronized_average.ndim - 1)
            )
        )
        coarse_synced = self._state_with_content(
            coarse_next,
            coarse_content,
            time=coarse_next.time,
        )
        self._validate_level_state(coarse_synced, self.coarse_runtime)
        self._validate_level_state(fine_final, self.fine_runtime)
        successor_selection = (
            requested_selection
            if requested_selection is not None
            else (
                self._selection(indicator, threshold)
                if indicator is not None
                else selection
            )
        )
        self._validate_selection(successor_selection)
        selection_changed = self._selection_changed(selection, successor_selection)
        if (
            selection_changed
            and indicator is not None
            and not self._accepted_step_regrids_enabled()
        ):
            # The indicator is evaluated only after the two-level interval has
            # accepted. Without solver-owned accepted-step events, ownership
            # cannot change atomically, so discard every tentative level update.
            empty_register = UnstructuredAMRFluxRegister(
                jnp.zeros_like(coarse_original.content_state.conservative_content)
            )
            empty_report = UnstructuredAMRRefluxReport(
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                jnp.zeros_like(coarse_original.content_state.conservative_content),
                selection.coarse_refined,
            )
            original_composite = self.hierarchy.composite_integral(
                coarse_original.content_state.cell_average(),
                fine_original.content_state.cell_average(),
                selection,
            )
            return UnstructuredAMRAdvanceResult(
                runtime_state,
                jnp.asarray(False),
                coarse_result.retries,
                coarse_result.attempted_step_size,
                jnp.asarray(0.0, dtype=coarse_original.time.dtype),
                coarse_result,
                tuple(fine_advances),
                None,
                None,
                tuple(fine_ledgers),
                empty_register,
                empty_report,
                original_composite,
                original_composite,
                selection,
                None,
                orchestration_failure=(
                    "indicator-driven AMR selection change requires accepted-step "
                    "topology events; the accepted level updates were rolled back"
                ),
            )
        request = (
            self._event_request(coarse_synced, successor_selection)
            if selection_changed
            else None
        )
        successor_runtime = None
        regrid_committed = jnp.asarray(False)
        if request is not None:
            coarse_epoch = FiniteVolumeTopologyEpoch(
                self.hierarchy.coarse.prepared_id,
                self.hierarchy.coarse.topology_id,
                self.hierarchy.coarse.geometry_id,
                parent_epoch_id=coarse_synced.content_state.topology_epoch_id,
                topology_artifact_id=self.hierarchy.plan_id,
                metrics_artifact_id=self.hierarchy.plan_id,
                operators_artifact_id=request.request_id,
            )
            fine_epoch = FiniteVolumeTopologyEpoch(
                self.hierarchy.fine.prepared_id,
                self.hierarchy.fine.topology_id,
                self.hierarchy.fine.geometry_id,
                parent_epoch_id=fine_final.content_state.topology_epoch_id,
                topology_artifact_id=self.hierarchy.plan_id,
                metrics_artifact_id=self.hierarchy.plan_id,
                operators_artifact_id=request.request_id,
            )
            coarse_reprepared = self.coarse_runtime.reprepare_for_epoch(coarse_epoch)
            fine_reprepared = self.fine_runtime.reprepare_for_epoch(fine_epoch)
            old_coarse_average = coarse_synced.content_state.cell_average()
            old_fine_average = fine_final.content_state.cell_average()
            prolongated = self.hierarchy.prolong(old_coarse_average)
            newly_active = (
                successor_selection.fine_active & ~runtime_state.selection.fine_active
            )
            new_fine_average = jnp.where(
                newly_active[:, None], prolongated, old_fine_average
            )
            # Fine content remains physically valid on inactive composite cells;
            # ownership is represented only by the successor selection mask.
            restricted_old = self.hierarchy.restrict(old_fine_average)
            newly_unrefined = (
                ~successor_selection.coarse_refined
                & runtime_state.selection.coarse_refined
            )
            new_coarse_average = jnp.where(
                newly_unrefined[:, None], restricted_old, old_coarse_average
            )
            old_composite = self.hierarchy.composite_integral(
                old_coarse_average, old_fine_average, runtime_state.selection
            )
            new_composite = self.hierarchy.composite_integral(
                new_coarse_average,
                new_fine_average,
                successor_selection,
            )
            composite_defect = new_composite - old_composite

            def rebind(
                state: FiniteVolumeRuntimeState,
                average: Array,
                epoch: FiniteVolumeTopologyEpoch,
                source_runtime: PreparedFiniteVolumeRuntime,
                target_runtime: PreparedFiniteVolumeRuntime,
            ) -> FiniteVolumeConservativeContentState:
                source = state.content_state
                if epoch.parent_epoch_id != source.topology_epoch_id:
                    raise ValueError(
                        "AMR content rebind received a stale successor epoch."
                    )
                if source.geometry_family_id != source_runtime.geometry_family_id:
                    raise ValueError("AMR source content geometry family is stale.")
                if target_runtime.topology_epoch_id != epoch.epoch_id:
                    raise ValueError(
                        "AMR target runtime does not own the successor epoch."
                    )
                if source_runtime.geometry_family_id != target_runtime.geometry_family_id:
                    raise ValueError(
                        "AMR successor runtime changed the canonical geometry family."
                    )
                content = average * source.effective_cell_volumes.reshape(
                    (-1,) + (1,) * (average.ndim - 1)
                )
                return FiniteVolumeConservativeContentState(
                    content,
                    source.effective_cell_volumes,
                    source.active_cell_mask,
                    state.time,
                    topology_epoch_id=epoch.epoch_id,
                    geometry_family_id=target_runtime.geometry_family_id,
                    geometry_layout_id=source.geometry_layout_id,
                    geometry_version=source.geometry_version,
                    evidence_policy_id=source.evidence_policy_id,
                    evidence_version=source.evidence_version,
                    precision=source.precision,
                )

            new_coarse_content = rebind(
                coarse_synced,
                new_coarse_average,
                coarse_epoch,
                self.coarse_runtime,
                coarse_reprepared,
            )
            new_fine_content = rebind(
                fine_final,
                new_fine_average,
                fine_epoch,
                self.fine_runtime,
                fine_reprepared,
            )
            coarse_artifact = UnstructuredAMRRegridArtifact(
                new_coarse_content,
                composite_defect,
                successor_selection,
                self.hierarchy.plan_id,
            )
            fine_artifact = UnstructuredAMRRegridArtifact(
                new_fine_content,
                composite_defect,
                successor_selection,
                self.hierarchy.plan_id,
            )
            fine_request = FiniteVolumeTopologyEventRequest(
                request.kind,
                fine_final.content_state.topology_epoch_id,
                request.requested_spec_id,
                payload_id=request.payload_id,
                reason=request.reason,
            )
            coarse_tx = FiniteVolumeTopologyEventTransaction(
                coarse_synced.topology_journal,
                (request,),
                coarse_synced.accepted_step,
                coarse_synced.time,
                accepted=True,
                artifact=coarse_artifact,
                candidate_epoch=coarse_epoch,
                remap=coarse_artifact,
                metrics=coarse_artifact,
                evidence=coarse_artifact,
                status=TopologyEventStatus.SUCCESS,
            )
            fine_tx = FiniteVolumeTopologyEventTransaction(
                fine_final.topology_journal,
                (fine_request,),
                fine_final.accepted_step,
                fine_final.time,
                accepted=True,
                artifact=fine_artifact,
                candidate_epoch=fine_epoch,
                remap=fine_artifact,
                metrics=fine_artifact,
                evidence=fine_artifact,
                status=TopologyEventStatus.SUCCESS,
            )
            coarse_event_result = coarse_tx.execute(
                coarse_synced.content_state,
                artifact=coarse_artifact,
                candidate_epoch=coarse_epoch,
                remap=coarse_artifact,
                metrics=coarse_artifact,
                evidence=coarse_artifact,
                status=TopologyEventStatus.SUCCESS,
                result_id=coarse_epoch.epoch_id,
                payload_ids=(request.payload_id,),
            )
            fine_event_result = fine_tx.execute(
                fine_final.content_state,
                artifact=fine_artifact,
                candidate_epoch=fine_epoch,
                remap=fine_artifact,
                metrics=fine_artifact,
                evidence=fine_artifact,
                status=TopologyEventStatus.SUCCESS,
                result_id=fine_epoch.epoch_id,
                payload_ids=(request.payload_id,),
            )
            if not coarse_event_result.committed or not fine_event_result.committed:
                raise ValueError(
                    "AMR regrid transaction failed atomically: "
                    f"coarse={coarse_event_result.failure!r}, "
                    f"fine={fine_event_result.failure!r}, "
                    f"passed={np.asarray(coarse_artifact.passed)!r}, "
                    f"status={np.asarray(coarse_artifact.status)!r}."
                )
            journal_coarse = coarse_event_result.journal
            journal_fine = fine_event_result.journal
            coarse_synced = FiniteVolumeRuntimeState(
                new_coarse_content,
                journal_coarse,
                coarse_synced.step_size,
                accepted_step=coarse_synced.accepted_step,
                last_status=coarse_synced.last_status,
                controller_state=coarse_synced.controller_state,
                integrator_state=coarse_synced.integrator_state,
                output_cursor=coarse_synced.output_cursor,
            )
            fine_final = FiniteVolumeRuntimeState(
                new_fine_content,
                journal_fine,
                fine_final.step_size,
                accepted_step=fine_final.accepted_step,
                last_status=fine_final.last_status,
                controller_state=fine_final.controller_state,
                integrator_state=fine_final.integrator_state,
                output_cursor=fine_final.output_cursor,
            )
            self._validate_level_state(coarse_synced, coarse_reprepared)
            self._validate_level_state(fine_final, fine_reprepared)
            successor_runtime = PreparedUnstructuredAMRRuntime(
                self.hierarchy,
                coarse_reprepared,
                fine_reprepared,
                refinement_ratio=self.refinement_ratio,
                policy=self.policy,
            )
            selection = successor_selection
            regrid_committed = jnp.asarray(True)
        composite = self.hierarchy.composite_integral(
            coarse_synced.content_state.cell_average(),
            fine_final.content_state.cell_average(),
            selection,
        )
        accepted_state = UnstructuredAMRRuntimeState(coarse_synced, fine_final, selection)
        return UnstructuredAMRAdvanceResult(
            accepted_state,
            jnp.asarray(True),
            coarse_result.retries,
            coarse_result.attempted_step_size,
            coarse_next.time - coarse_original.time,
            coarse_result,
            tuple(fine_advances),
            coarse_ledger,
            fine_ledger,
            tuple(fine_ledgers),
            register,
            report,
            composite,
            composite,
            selection,
            request,
            successor_runtime,
            regrid_committed,
        )


__all__ = [
    "PreparedUnstructuredAMRRuntime",
    "UnstructuredAMRAdvanceResult",
    "UnstructuredAMRRefluxReport",
    "UnstructuredAMRRuntimeState",
]
