#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.finite_volume import (
    AbstractWavePropagationPlan,
    ConservativeSmallCellRedistributionPlan,
    ConservativeSmallCellRedistributionReport,
    FiniteVolumeAdmissibilityReport,
    FiniteVolumeMethodPlan,
    FiniteVolumePrecisionPolicy,
    FluxPositivityPlan,
    lower_embedded_stage_metrics,
    lower_static_unstructured_stage_metrics,
    PeriodicSlidingCoupling,
    PeriodicSlidingInterfacePlan,
    PeriodicSlidingRefreshArtifact,
    PiecewiseConstantReconstruction,
    PreparedFiniteVolumeDynamics,
    PreparedTriangleFiniteVolumeDynamics,
    PreparedUnstructuredFiniteVolumeDynamics,
    TriangleFiniteVolumeDiscretization,
    UnstructuredEmbeddedBoundarySet,
    UnstructuredFiniteVolumeDiscretization,
)
from ..discretization.finite_volume._flux_ledger import (
    FiniteVolumeAcceptedFluxIntegralBlock,
    FiniteVolumeAcceptedFluxIntegralLedger,
    FiniteVolumeStageFluxRateBlock,
    FiniteVolumeStageFluxRateLedger,
)
from ..discretization.finite_volume._geometry_protocol import (
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageMetrics,
)
from ..discretization.finite_volume._positivity import (
    BalancedPositivityBlendResult,
)
from ..discretization.finite_volume._shallow_water import (
    PreparedShallowWaterBathymetry,
    ShallowWaterAcceptedFaceIntegrals,
    ShallowWaterBalancedFaceResult,
    ShallowWaterHydrostaticHLLPlan,
)
from ..discretization.finite_volume._unstructured_motion import (
    UnstructuredALEStepGeometry,
)
from ._finite_volume import (
    FiniteVolumeStageStateProvider,
    unstructured_ale_ssprk33_candidate,
    unstructured_ssprk33_content_candidate,
    zero_unstructured_ale_ssprk33_candidate,
    zero_unstructured_ssprk33_content_candidate,
)
from ._finite_volume_content import FiniteVolumeConservativeContentState
from ._finite_volume_topology_events import (
    FiniteVolumeTopologyEpoch,
    FiniteVolumeTopologyEventJournal,
    FiniteVolumeTopologyEventRequest,
    FiniteVolumeTopologyEventScheduler,
    TopologyEventKind,
    TopologyEventStatus,
)


class FiniteVolumeRunStatus(IntEnum):
    SUCCESS = 0
    RECOVERED_REJECTION = 1
    INVALID_INITIAL_STATE = 2
    RETRY_LIMIT_REACHED = 3
    MINIMUM_STEP_REACHED = 4
    NONFINITE_STATE = 5
    STABILITY_LIMIT_EXCEEDED = 6
    PRESCRIBED_STEP_REJECTED = 7


class FiniteVolumeStepPolicy(StrictModule, NonTrainableState):
    cfl: float = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    reduction_factor: float = eqx.field(static=True)
    minimum_step_size: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        cfl: float = 0.45,
        maximum_retries: int = 4,
        reduction_factor: float = 0.5,
        minimum_step_size: float = 1e-12,
    ):
        cfl_ = float(cfl)
        retries = int(maximum_retries)
        reduction = float(reduction_factor)
        minimum = float(minimum_step_size)
        if (
            not np.isfinite(cfl_)
            or cfl_ <= 0.0
            or retries < 0
            or not 0.0 < reduction < 1.0
            or not np.isfinite(minimum)
            or minimum <= 0.0
        ):
            raise ValueError("Finite-volume step policy is invalid.")
        self.cfl = cfl_
        self.maximum_retries = retries
        self.reduction_factor = reduction
        self.minimum_step_size = minimum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-volume-step-policy",
                "cfl": cfl_,
                "maximum_retries": retries,
                "reduction_factor": reduction,
                "minimum_step_size": minimum,
            }
        )


class FiniteVolumeRuntimeState(StrictModule):
    content_state: FiniteVolumeConservativeContentState
    topology_journal: FiniteVolumeTopologyEventJournal
    accepted_step: Array
    step_size: Array
    last_status: Array
    controller_state: Array
    integrator_state: Array
    output_cursor: Array
    sliding_coupling: PeriodicSlidingCoupling | None
    sliding_shift: Array
    sliding_coupling_id: str | None = eqx.field(static=True)
    sliding_event_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        content_state: FiniteVolumeConservativeContentState,
        topology_journal: FiniteVolumeTopologyEventJournal,
        step_size: ArrayLike,
        /,
        *,
        accepted_step: ArrayLike = 0,
        last_status: ArrayLike = FiniteVolumeRunStatus.SUCCESS,
        controller_state: ArrayLike | None = None,
        integrator_state: ArrayLike | None = None,
        output_cursor: ArrayLike = 0,
        sliding_coupling: PeriodicSlidingCoupling | None = None,
        sliding_shift: ArrayLike = 0.0,
        sliding_event_id: str | None = None,
    ):
        if not isinstance(content_state, FiniteVolumeConservativeContentState):
            raise TypeError("content_state must be FiniteVolumeConservativeContentState.")
        if not isinstance(topology_journal, FiniteVolumeTopologyEventJournal):
            raise TypeError("topology_journal must be FiniteVolumeTopologyEventJournal.")
        if topology_journal.current_epoch_id != content_state.topology_epoch_id:
            raise ValueError(
                "Runtime content and topology journal must own the same current epoch."
            )
        if sliding_coupling is not None and not isinstance(
            sliding_coupling, PeriodicSlidingCoupling
        ):
            raise TypeError("sliding_coupling must be PeriodicSlidingCoupling or None.")
        shift = jnp.asarray(
            sliding_shift, dtype=content_state.precision.reduction_dtype
        ).reshape(())
        shift = eqx.error_if(
            shift,
            ~jnp.isfinite(shift),
            "sliding_shift must be finite.",
        )
        if sliding_event_id is not None and (
            not isinstance(sliding_event_id, str) or not sliding_event_id
        ):
            raise ValueError("sliding_event_id must be nonempty or None.")
        self.content_state = content_state
        self.topology_journal = topology_journal
        self.accepted_step = jnp.asarray(accepted_step, dtype=jnp.int32).reshape(())
        self.step_size = jnp.asarray(step_size).reshape(())
        self.last_status = jnp.asarray(last_status, dtype=jnp.int32).reshape(())
        self.controller_state = jnp.asarray(
            () if controller_state is None else controller_state
        )
        self.integrator_state = jnp.asarray(
            () if integrator_state is None else integrator_state
        )
        self.output_cursor = jnp.asarray(output_cursor, dtype=jnp.int32).reshape(())
        self.sliding_coupling = sliding_coupling
        self.sliding_shift = shift
        self.sliding_coupling_id = (
            None if sliding_coupling is None else sliding_coupling.coupling_id
        )
        self.sliding_event_id = sliding_event_id

    @property
    def time(self) -> Array:
        return self.content_state.time

    def cell_average(self) -> Array:
        """Derive cell averages from the authoritative conservative content."""

        return self.content_state.cell_average()


class FiniteVolumeALEAdvanceEvidence(StrictModule):
    """ALE geometry and stage-rate evidence for the public accepted ledger."""

    accepted: Array
    stage_rate_ledgers: tuple[
        FiniteVolumeStageFluxRateLedger,
        FiniteVolumeStageFluxRateLedger,
        FiniteVolumeStageFluxRateLedger,
    ]
    geometry: UnstructuredALEStepGeometry
    maximum_relative_rate: Array
    relative_cfl_step: Array
    geometry_reduction_factor: Array


class FiniteVolumeEmbeddedAdvanceEvidence(StrictModule):
    """Stationary embedded metrics, redistribution, and effective CFL evidence."""

    accepted: Array
    stage_rate_ledgers: tuple[
        FiniteVolumeStageFluxRateLedger,
        FiniteVolumeStageFluxRateLedger,
        FiniteVolumeStageFluxRateLedger,
    ]
    stage_metrics: tuple[
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
    ]
    accepted_metrics: FiniteVolumeStageMetrics
    redistribution: ConservativeSmallCellRedistributionReport
    stage_maximum_relative_rates: tuple[Array, Array, Array]
    maximum_relative_rate: Array
    relative_cfl_step: Array


class FiniteVolumeAdvanceResult(StrictModule):
    runtime_state: FiniteVolumeRuntimeState
    accepted: Array
    retries: Array
    attempted_step_size: Array
    accepted_step_size: Array
    positivity: FiniteVolumeAdmissibilityReport
    accepted_flux_integrals: FiniteVolumeAcceptedFluxIntegralLedger
    precision_evidence: PrecisionEvidenceEnvelope = eqx.field(static=True)
    ale: FiniteVolumeALEAdvanceEvidence | None
    embedded: FiniteVolumeEmbeddedAdvanceEvidence | None
    successor_runtime: PreparedFiniteVolumeRuntime | None = None
    shallow_water_integrals: ShallowWaterAcceptedFaceIntegrals | None = None


class FiniteVolumeScheduledAdvanceResult(StrictModule):
    """One exact prescribed-step attempt with explicit stability evidence."""

    runtime_state: FiniteVolumeRuntimeState
    attempted: FiniteVolumeAdvanceResult
    accepted: Array
    requested_step_size: Array
    stable_step_size: Array
    stability_margin: Array


class PreparedFiniteVolumeRuntime(StrictModule, NonTrainableState):
    """Content-authoritative SSPRK3 runtime with bounded positivity retries."""

    dynamics: (
        PreparedFiniteVolumeDynamics
        | PreparedTriangleFiniteVolumeDynamics
        | PreparedUnstructuredFiniteVolumeDynamics
    )
    fallback_dynamics: (
        PreparedFiniteVolumeDynamics
        | PreparedTriangleFiniteVolumeDynamics
        | PreparedUnstructuredFiniteVolumeDynamics
    )
    positivity: FluxPositivityPlan
    policy: FiniteVolumeStepPolicy
    stage_state_provider: FiniteVolumeStageStateProvider | None
    effective_cell_volumes: Array
    active_cell_mask: Array
    precision: FiniteVolumePrecisionPolicy
    static_flux_rate_block_templates: tuple[FiniteVolumeStageFluxRateBlock, ...]
    embedded_redistribution: ConservativeSmallCellRedistributionPlan | None
    embedded_stage_template: FiniteVolumeStageMetrics | None
    sliding_plan: PeriodicSlidingInterfacePlan | None
    sliding_initial_coupling: PeriodicSlidingCoupling | None
    initial_topology_epoch: FiniteVolumeTopologyEpoch = eqx.field(static=True)
    topology_epoch_id: str = eqx.field(static=True)
    geometry_family_id: str = eqx.field(static=True)
    geometry_layout_id: str = eqx.field(static=True)
    evidence_policy_id: str = eqx.field(static=True)
    runtime_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: (
            PreparedFiniteVolumeDynamics
            | PreparedTriangleFiniteVolumeDynamics
            | PreparedUnstructuredFiniteVolumeDynamics
        ),
        positivity: FluxPositivityPlan,
        policy: FiniteVolumeStepPolicy | None = None,
        *,
        topology_epoch: FiniteVolumeTopologyEpoch | None = None,
        stage_state_provider: FiniteVolumeStageStateProvider | None = None,
    ):
        if not isinstance(
            dynamics,
            (
                PreparedFiniteVolumeDynamics,
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            raise TypeError("dynamics must be prepared finite-volume dynamics.")
        if isinstance(dynamics, PreparedFiniteVolumeDynamics) and isinstance(
            dynamics.method.interface_solver, AbstractWavePropagationPlan
        ):
            raise ValueError(
                "Wave-propagation dynamics do not expose the face fluxes required "
                "by PreparedFiniteVolumeRuntime."
            )
        if not isinstance(positivity, FluxPositivityPlan):
            raise TypeError("positivity must be FluxPositivityPlan.")
        policy_ = FiniteVolumeStepPolicy() if policy is None else policy
        if not isinstance(policy_, FiniteVolumeStepPolicy):
            raise TypeError("policy must be FiniteVolumeStepPolicy.")
        if stage_state_provider is not None:
            if not isinstance(stage_state_provider, FiniteVolumeStageStateProvider):
                raise TypeError(
                    "stage_state_provider must be FiniteVolumeStageStateProvider or None."
                )
            if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
                raise ValueError(
                    "Stage-state providers are supported only by unstructured "
                    "finite-volume runtimes."
                )
        sliding_plan = (
            dynamics.coupling.sliding
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        sliding_initial_coupling = (
            dynamics.coupling.sliding_coupling
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        if isinstance(
            dynamics,
            (
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            fallback = dynamics.make_fallback_dynamics(positivity.fallback_flux)
        elif isinstance(dynamics.method.interface_solver, ShallowWaterHydrostaticHLLPlan):
            if not isinstance(dynamics.bathymetry, PreparedShallowWaterBathymetry):
                raise TypeError(
                    "Hydrostatic shallow-water runtime requires prepared bathymetry."
                )
            fallback_method = FiniteVolumeMethodPlan(
                PiecewiseConstantReconstruction(),
                dynamics.method.interface_solver,
                differentiability="branchwise",
            )
            fallback = PreparedFiniteVolumeDynamics(
                dynamics.system,
                dynamics.discretization,
                fallback_method,
                dynamics.boundaries,
                bathymetry=dynamics.bathymetry.values,
                precision=dynamics.precision,
                source=dynamics.source,
                source_id=dynamics.source_id,
            )
        else:
            fallback_method = FiniteVolumeMethodPlan(
                PiecewiseConstantReconstruction(),
                positivity.fallback_flux,
                viscous=dynamics.method.viscous,
                differentiability="branchwise",
            )
            fallback = PreparedFiniteVolumeDynamics(
                dynamics.system,
                dynamics.discretization,
                fallback_method,
                dynamics.boundaries,
                capacity=dynamics.capacity,
                bathymetry=dynamics.bathymetry,
                precision=dynamics.precision,
                source=dynamics.source,
                source_id=dynamics.source_id,
            )
        discretization = dynamics.discretization
        embedded_redistribution = None
        embedded_stage_template = None
        embedded_boundaries: UnstructuredEmbeddedBoundarySet | None = None
        embedded_metrics = (
            dynamics.coupling.embedded_metrics
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        overset_effective_volumes = (
            dynamics.overset_effective_cell_volumes
            if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and dynamics.overset_mapping_id is not None
            else None
        )
        if isinstance(dynamics, PreparedFiniteVolumeDynamics):
            effective_volumes = dynamics.effective_volumes
        elif isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            effective_volumes = (
                dynamics.overset_effective_cell_volumes
                if overset_effective_volumes is not None
                else (
                    embedded_metrics.fluid_cell_volumes
                    if embedded_metrics is not None
                    else discretization.cell_volumes
                )
            )
        else:
            effective_volumes = discretization.cell_volumes
        bound_effective_volumes = dynamics.precision.reduction(effective_volumes).reshape(
            (-1,)
        )
        if isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            topology_id = discretization.topology_id
            geometry_id = discretization.geometry_id
            geometry_family_id = geometry_id
            motion = (
                dynamics.coupling.motion
                if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                else None
            )
            if embedded_metrics is not None:
                if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
                    raise TypeError("Embedded metrics require unstructured dynamics.")
                embedded_boundaries = dynamics.coupling.embedded_boundaries
                stabilization = dynamics.coupling.embedded_stabilization_policy
                if (
                    not isinstance(
                        embedded_boundaries,
                        UnstructuredEmbeddedBoundarySet,
                    )
                    or stabilization is None
                ):
                    raise TypeError(
                        "Prepared embedded coupling requires wall ownership and "
                        "stabilization."
                    )
                embedded_stage_template = lower_embedded_stage_metrics(
                    discretization,
                    embedded_metrics,
                    embedded_boundaries,
                    discretization.topology_id,
                    0,
                    0,
                    time=0.0,
                )
                geometry_layout_id = embedded_stage_template.geometry_layout_id
                evidence_policy_id = embedded_stage_template.evidence.policy_id
                embedded_redistribution = ConservativeSmallCellRedistributionPlan(
                    discretization,
                    embedded_metrics,
                    stabilization,
                )
            elif motion is None:
                static_metrics = lower_static_unstructured_stage_metrics(discretization)
                geometry_layout_id = static_metrics.geometry_layout_id
                evidence_policy_id = static_metrics.evidence.policy_id
            else:
                geometry_family_id = motion.plan_id
                geometry_layout_id = motion.geometry_layout_id
                evidence_policy_id = motion.consistency_policy.policy_id
        else:
            topology_id = discretization.prepared_id
            geometry_id = discretization.prepared_id
            geometry_family_id = geometry_id
            geometry_layout_id = canonical_fingerprint(
                {
                    "kind": "static-finite-volume-runtime-geometry-layout",
                    "topology": topology_id,
                    "support": discretization.support.support_id,
                    "cell_layout": discretization.cell_space.layout.layout_id,
                    "effective_cell_volumes": array_tree_fingerprint(
                        np.asarray(bound_effective_volumes)
                    ),
                }
            )
            evidence_policy_id = canonical_fingerprint(
                {
                    "kind": "static-finite-volume-runtime-geometry-evidence",
                    "geometry_layout": geometry_layout_id,
                    "precision": dynamics.precision.policy_id,
                    "requirements": (
                        "all-cells-active",
                        "finite-positive-effective-cell-volumes",
                        "fixed-topology-and-geometry",
                    ),
                }
            )
        if isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            static_flux_rate_block_templates = dynamics.stage_rate_block_templates
        elif isinstance(discretization, TriangleFiniteVolumeDiscretization):
            face_block = discretization.face_block
            static_flux_rate_block_templates = (
                FiniteVolumeStageFluxRateBlock(
                    jnp.zeros(
                        (
                            discretization.face_measures.size,
                            discretization.component_count,
                        ),
                        dtype=jnp.dtype(dynamics.precision.reduction_dtype),
                    ),
                    face_block.owner_cells,
                    face_block.neighbour_cells,
                    face_block.active_mask,
                    face_block.block_id,
                    "static-explicit-face",
                ),
            )
        else:
            if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
                raise TypeError("Structured templates require prepared FV dynamics.")
            discretization = dynamics.discretization
            cell_shape = tuple(discretization.state_shape[:-1])
            cell_ids = np.arange(np.prod(cell_shape), dtype=np.int32).reshape(cell_shape)
            templates = []
            for axis, layout in enumerate(discretization.face_layouts):
                face_shape = tuple(layout.shape)
                periodic = discretization.grid.structured_axes[axis].periodic
                if periodic:
                    owners = np.roll(cell_ids, 1, axis=axis)
                    neighbours = cell_ids
                else:
                    owners = np.empty(face_shape, dtype=np.int32)
                    neighbours = np.full(face_shape, -1, dtype=np.int32)
                    lower_face: list[slice | int] = [slice(None)] * len(face_shape)
                    upper_face: list[slice | int] = [slice(None)] * len(face_shape)
                    interior_faces = [slice(None)] * len(face_shape)
                    lower_cells = [slice(None)] * len(cell_shape)
                    upper_cells = [slice(None)] * len(cell_shape)
                    lower_face[axis] = 0
                    upper_face[axis] = -1
                    interior_faces[axis] = slice(1, -1)
                    lower_cells[axis] = slice(0, -1)
                    upper_cells[axis] = slice(1, None)
                    owners[tuple(lower_face)] = cell_ids[tuple(lower_face)]
                    owners[tuple(upper_face)] = cell_ids[tuple(upper_face)]
                    owners[tuple(interior_faces)] = cell_ids[tuple(lower_cells)]
                    neighbours[tuple(interior_faces)] = cell_ids[tuple(upper_cells)]
                active = owners != neighbours
                block_id = canonical_fingerprint(
                    {
                        "kind": "static-structured-face-route",
                        "geometry_layout": geometry_layout_id,
                        "axis": axis,
                        "periodic": periodic,
                        "owner_cells": array_tree_fingerprint(owners),
                        "neighbour_cells": array_tree_fingerprint(neighbours),
                    }
                )
                templates.append(
                    FiniteVolumeStageFluxRateBlock(
                        jnp.zeros(
                            (int(np.prod(face_shape)), discretization.component_count),
                            dtype=jnp.dtype(dynamics.precision.reduction_dtype),
                        ),
                        owners.reshape((-1,)),
                        neighbours.reshape((-1,)),
                        active.reshape((-1,)),
                        block_id,
                        "static-structured-face",
                    )
                )
            static_flux_rate_block_templates = tuple(templates)
        default_topology_epoch = FiniteVolumeTopologyEpoch(
            discretization.prepared_id,
            topology_id,
            geometry_id,
        )
        initial_topology_epoch = (
            default_topology_epoch if topology_epoch is None else topology_epoch
        )
        if (
            initial_topology_epoch.prepared_id != discretization.prepared_id
            or initial_topology_epoch.topology_id != topology_id
            or initial_topology_epoch.geometry_id != geometry_id
        ):
            raise ValueError("Reprepared topology epoch does not match runtime geometry.")
        topology_epoch_id = initial_topology_epoch.epoch_id
        if (
            embedded_metrics is not None
            and isinstance(discretization, UnstructuredFiniteVolumeDiscretization)
            and isinstance(embedded_boundaries, UnstructuredEmbeddedBoundarySet)
        ):
            embedded_stage_template = lower_embedded_stage_metrics(
                discretization,
                embedded_metrics,
                embedded_boundaries,
                topology_epoch_id,
                0,
                0,
                time=0.0,
            )
            geometry_layout_id = embedded_stage_template.geometry_layout_id
        self.dynamics = dynamics
        self.fallback_dynamics = fallback
        self.positivity = positivity
        self.precision = dynamics.precision
        self.effective_cell_volumes = bound_effective_volumes
        if (
            isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and overset_effective_volumes is not None
        ):
            active_mask = dynamics.overset_active_cell_mask
        elif embedded_metrics is not None:
            active_mask = embedded_metrics.active_fluid_cells
        else:
            active_mask = jnp.ones(self.effective_cell_volumes.shape, dtype=jnp.bool_)
        self.active_cell_mask = jnp.asarray(active_mask, dtype=jnp.bool_)

        self.policy = policy_
        self.stage_state_provider = stage_state_provider
        self.topology_epoch_id = topology_epoch_id
        self.geometry_family_id = geometry_family_id
        self.geometry_layout_id = geometry_layout_id
        self.evidence_policy_id = evidence_policy_id
        self.static_flux_rate_block_templates = static_flux_rate_block_templates
        self.embedded_redistribution = embedded_redistribution
        self.embedded_stage_template = embedded_stage_template
        self.sliding_plan = sliding_plan
        self.sliding_initial_coupling = sliding_initial_coupling
        self.initial_topology_epoch = initial_topology_epoch
        self.runtime_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-volume-runtime",
                "dynamics": dynamics.dynamics_id,
                "fallback": fallback.dynamics_id,
                "positivity": positivity.plan_id,
                "policy": policy_.policy_id,
                "precision": dynamics.precision.policy_id,
                "topology_epoch": topology_epoch_id,
                "geometry_family": geometry_family_id,
                "geometry_layout": geometry_layout_id,
                "evidence_policy": evidence_policy_id,
                "stage_state_provider": (
                    None
                    if stage_state_provider is None
                    else stage_state_provider.provider_id
                ),
                "embedded_redistribution": (
                    None
                    if embedded_redistribution is None
                    else embedded_redistribution.plan_id
                ),
                "sliding_plan": None if sliding_plan is None else sliding_plan.plan_id,
                "sliding_coupling": (
                    None
                    if sliding_initial_coupling is None
                    else sliding_initial_coupling.coupling_id
                ),
                "overset": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    or dynamics.coupling.overset is None
                    else dynamics.coupling.overset.plan_id
                ),
                "overset_policy": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    else dynamics.overset_policy_id
                ),
                "overset_mapping": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    else dynamics.overset_mapping_id
                ),
                "overset_epoch": (
                    None
                    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics)
                    else dynamics.overset_epoch_id
                ),
            }
        )

    def with_stage_state_provider(
        self,
        provider: FiniteVolumeStageStateProvider | None,
        /,
    ) -> PreparedFiniteVolumeRuntime:
        """Return an immutable runtime bound to one stage-state trace."""
        if provider is not None and not isinstance(
            provider, FiniteVolumeStageStateProvider
        ):
            raise TypeError("provider must be FiniteVolumeStageStateProvider or None.")
        if provider is self.stage_state_provider:
            return self
        return PreparedFiniteVolumeRuntime(
            self.dynamics,
            self.positivity,
            self.policy,
            topology_epoch=self.initial_topology_epoch,
            stage_state_provider=provider,
        )

    def _provide_stage_state(self, time: Array, state: Array, /) -> Array:
        provider = self.stage_state_provider
        if provider is None:
            return state
        return self.precision.storage(
            provider(
                self.precision.decision(time),
                self.precision.storage(state),
            )
        )

    def _effective_cell_volumes(self) -> Array:
        return self.effective_cell_volumes

    def _state_shape(self) -> tuple[int, ...]:
        return tuple(self.dynamics.discretization.state_shape)

    def _topology_journal_capacity(self) -> int:
        coupling = (
            self.dynamics.coupling
            if isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            else None
        )
        return (
            coupling.topology_event_capacity
            if coupling is not None and coupling.topology_event_policy == "accepted_step"
            else 1
        )

    def _dynamics_cell_average(
        self,
        content_state: FiniteVolumeConservativeContentState,
        /,
    ) -> Array:
        return content_state.cell_average().reshape(self._state_shape())

    def _active_state_is_admissible(
        self,
        cell_average: Array,
        active_cell_mask: Array,
        /,
    ) -> Array:
        average = cell_average.reshape((-1, cell_average.shape[-1]))
        active = active_cell_mask.reshape((-1,))
        first_active = jnp.argmax(active.astype(jnp.int32))

        def evaluate(_):
            seed = average[first_active]
            safe_average = jnp.where(active[:, None], average, seed[None, :])
            return jnp.all(
                jnp.where(
                    active,
                    self.dynamics.system.admissible(safe_average),
                    True,
                )
            )

        return jax.lax.cond(
            jnp.any(active),
            evaluate,
            lambda _: jnp.asarray(True),
            operand=None,
        )

    def reprepare_for_epoch(
        self,
        topology_epoch: FiniteVolumeTopologyEpoch,
        /,
    ) -> PreparedFiniteVolumeRuntime:
        """Rebind the fixed geometry runtime to a validated successor epoch."""
        return PreparedFiniteVolumeRuntime(
            self.dynamics,
            self.positivity,
            self.policy,
            topology_epoch=topology_epoch,
        )

    def initialize_state(
        self,
        cell_average: ArrayLike,
        time: ArrayLike,
        step_size: ArrayLike,
        /,
        *,
        motion_args: Any = None,
        accepted_step: ArrayLike = 0,
        last_status: ArrayLike = FiniteVolumeRunStatus.SUCCESS,
        controller_state: ArrayLike | None = None,
        integrator_state: ArrayLike | None = None,
        output_cursor: ArrayLike = 0,
    ) -> FiniteVolumeRuntimeState:
        time_value = self.precision.decision(time)
        volumes = self._effective_cell_volumes()
        if (
            isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.motion is not None
        ):
            if time_value.shape != () or float(np.asarray(time_value)) != 0.0:
                raise ValueError(
                    "Fresh moving finite-volume initialization requires time exactly "
                    "t=0; use checkpoint/restart for nonzero starts."
                )
            motion = self.dynamics.coupling.motion
            evaluated_plan = motion.evaluate_plan(time_value, motion_args)
            evaluated_discretization = evaluated_plan.prepare()
            base_discretization = self.dynamics.discretization
            if evaluated_discretization.topology_id != base_discretization.topology_id:
                raise ValueError(
                    "Motion evaluated at t=0 must match the compiled base topology."
                )

            absolute_tolerance = motion.consistency_policy.absolute_tolerance
            relative_tolerance = motion.consistency_policy.relative_tolerance

            def matches_base_geometry(evaluated, compiled, /):
                evaluated_array = np.asarray(evaluated)
                compiled_array = np.asarray(compiled)
                if evaluated_array.shape != compiled_array.shape:
                    return False
                reference = np.maximum(
                    np.abs(evaluated_array),
                    np.abs(compiled_array),
                )
                tolerance = absolute_tolerance + relative_tolerance * reference
                return bool(
                    np.all(np.isfinite(evaluated_array))
                    and np.all(np.isfinite(compiled_array))
                    and np.all(np.abs(evaluated_array - compiled_array) <= tolerance)
                )

            if not matches_base_geometry(
                evaluated_discretization.vertices,
                base_discretization.vertices,
            ):
                raise ValueError(
                    "Motion evaluated at t=0 must match the compiled base geometry "
                    "within ALE precision evidence."
                )
            if not matches_base_geometry(
                evaluated_discretization.cell_volumes,
                volumes,
            ):
                raise ValueError(
                    "Motion evaluated at t=0 must reproduce the compiled base geometry "
                    "cell volumes within ALE precision evidence."
                )

        journal_capacity = self._topology_journal_capacity()
        topology_journal = FiniteVolumeTopologyEventJournal.allocate(
            self.initial_topology_epoch,
            capacity=journal_capacity,
            time=time_value,
        )
        average = self.precision.storage(cell_average)
        state_shape = self._state_shape()
        if average.shape != state_shape:
            raise ValueError(
                f"Finite-volume cell average must have exact shape {state_shape}."
            )
        flattened = average.reshape((-1, state_shape[-1]))
        active = self.active_cell_mask
        content_state = FiniteVolumeConservativeContentState.from_cell_average(
            flattened,
            volumes,
            active,
            time_value,
            topology_epoch_id=self.topology_epoch_id,
            geometry_family_id=self.geometry_family_id,
            geometry_layout_id=self.geometry_layout_id,
            geometry_version=jnp.asarray(0, dtype=jnp.int32),
            evidence_policy_id=self.evidence_policy_id,
            evidence_version=jnp.asarray(0, dtype=jnp.int32),
            precision=self.precision,
        )
        return FiniteVolumeRuntimeState(
            content_state,
            topology_journal,
            self.precision.decision(step_size),
            accepted_step=accepted_step,
            last_status=last_status,
            controller_state=controller_state,
            integrator_state=integrator_state,
            output_cursor=output_cursor,
            sliding_coupling=self.sliding_initial_coupling,
            sliding_shift=(
                0.0
                if self.sliding_initial_coupling is None
                else self.sliding_initial_coupling.normalized_shift
            ),
        )

    def _face_measures(self) -> tuple[Array, ...]:
        discretization = self.dynamics.discretization
        if isinstance(
            discretization,
            (
                TriangleFiniteVolumeDiscretization,
                UnstructuredFiniteVolumeDiscretization,
            ),
        ):
            return (discretization.face_measures,)
        return tuple(discretization.face_measures)

    def _face_fluxes(
        self,
        dynamics: (
            PreparedFiniteVolumeDynamics
            | PreparedTriangleFiniteVolumeDynamics
            | PreparedUnstructuredFiniteVolumeDynamics
        ),
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:
        if isinstance(
            dynamics,
            (
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            flux, speed = dynamics.face_fluxes(time, state, args)
            return (flux,), (speed,)
        fluxes, speeds = dynamics.face_fluxes(time, state, args)
        return fluxes, speeds

    def _flux_residual(
        self,
        dynamics: (
            PreparedFiniteVolumeDynamics
            | PreparedTriangleFiniteVolumeDynamics
            | PreparedUnstructuredFiniteVolumeDynamics
        ),
        fluxes: tuple[Array, ...],
        /,
    ) -> Array:
        if isinstance(
            dynamics,
            (
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            if len(fluxes) != 1:
                raise ValueError("Explicit-face runtime requires one face-flux block.")
            return dynamics.residual_from_fluxes(fluxes[0])
        return dynamics._flux_residual(fluxes)

    def _zero_static_flux_rates(self) -> tuple[Array, ...]:
        discretization = self.dynamics.discretization
        dtype = jnp.dtype(self.precision.reduction_dtype)
        if isinstance(
            discretization,
            (
                TriangleFiniteVolumeDiscretization,
                UnstructuredFiniteVolumeDiscretization,
            ),
        ):
            return (
                jnp.zeros(
                    (
                        discretization.face_measures.size,
                        discretization.component_count,
                    ),
                    dtype=dtype,
                ),
            )
        return tuple(
            jnp.zeros(
                layout.shape + (discretization.component_count,),
                dtype=dtype,
            )
            for layout in discretization.face_layouts
        )

    def _zero_shallow_water_integrals(
        self,
    ) -> ShallowWaterAcceptedFaceIntegrals | None:
        if not isinstance(self.dynamics, PreparedFiniteVolumeDynamics) or not isinstance(
            self.dynamics.method.interface_solver,
            ShallowWaterHydrostaticHLLPlan,
        ):
            return None
        if not isinstance(self.dynamics.bathymetry, PreparedShallowWaterBathymetry):
            raise TypeError(
                "Hydrostatic shallow-water runtime requires prepared bathymetry."
            )
        dtype = jnp.dtype(self.precision.reduction_dtype)
        contributions = []
        for layout in self.dynamics.discretization.face_layouts:
            shape = layout.shape + (self.dynamics.discretization.component_count,)
            state = jnp.zeros(shape, dtype=dtype)
            speed = jnp.zeros(layout.shape, dtype=dtype)
            contributions.append(
                ShallowWaterBalancedFaceResult(
                    state,
                    state,
                    state,
                    speed,
                    state,
                    state,
                    jnp.ones(layout.shape, dtype=bool),
                )
            )
        return ShallowWaterAcceptedFaceIntegrals(
            tuple(contributions),
            tuple(self.dynamics.discretization.face_measures),
            jnp.asarray(0.0, dtype=dtype),
            axis_names=self.dynamics.discretization.grid.axis_names,
            bed_id=self.dynamics.bathymetry.bed_id,
            plan_id=self.dynamics.method.interface_solver.plan_id,
        )

    def _static_accepted_flux_integral_ledger(
        self,
        original_content: FiniteVolumeConservativeContentState,
        next_content: FiniteVolumeConservativeContentState,
        integrated_flux_rates: tuple[Array, ...],
        step_size: Array,
        end_time: Array,
        accepted_step: Array,
        /,
    ) -> FiniteVolumeAcceptedFluxIntegralLedger:
        if len(integrated_flux_rates) != len(self.static_flux_rate_block_templates):
            raise ValueError("Static flux rates must match the prepared ledger routes.")
        discretization = self.dynamics.discretization
        step = self.precision.reduction(step_size)
        blocks = []
        for axis, (integrated_rate, template) in enumerate(
            zip(
                integrated_flux_rates,
                self.static_flux_rate_block_templates,
                strict=True,
            )
        ):
            integral = step * self.precision.reduction(integrated_rate)
            if (
                not isinstance(
                    discretization,
                    (
                        TriangleFiniteVolumeDiscretization,
                        UnstructuredFiniteVolumeDiscretization,
                    ),
                )
                and not discretization.grid.structured_axes[axis].periodic
            ):
                lower_face = [slice(None)] * integral.ndim
                lower_face[axis] = 0
                integral = integral.at[tuple(lower_face)].multiply(-1)
            blocks.append(
                FiniteVolumeAcceptedFluxIntegralBlock._from_stage_rate_block(
                    integral.reshape(template.flux_rate.shape),
                    template,
                )
            )
        content_change = self.precision.reduction(
            next_content.conservative_content
        ) - self.precision.reduction(original_content.conservative_content)
        face_change = jnp.zeros_like(content_change)
        for block in blocks:
            safe_neighbour = jnp.maximum(block.neighbour_cells, 0)
            face_change = face_change.at[block.owner_cells].add(-block.flux_integral)
            face_change = face_change.at[safe_neighbour].add(
                jnp.where(
                    (block.neighbour_cells >= 0)[:, None],
                    block.flux_integral,
                    jnp.zeros((), dtype=block.flux_integral.dtype),
                )
            )
        source_integral = content_change - face_change
        return FiniteVolumeAcceptedFluxIntegralLedger(
            tuple(blocks),
            source_integral,
            original_content.active_cell_mask,
            geometry_family_id=original_content.geometry_family_id,
            geometry_layout_id=original_content.geometry_layout_id,
            stage_geometry_versions=(
                original_content.geometry_version,
                original_content.geometry_version,
                original_content.geometry_version,
            ),
            start_geometry_version=original_content.geometry_version,
            end_geometry_version=next_content.geometry_version,
            evidence_policy_id=original_content.evidence_policy_id,
            stage_evidence_versions=(
                original_content.evidence_version,
                original_content.evidence_version,
                original_content.evidence_version,
            ),
            start_evidence_version=original_content.evidence_version,
            end_evidence_version=next_content.evidence_version,
            start_topology_epoch_id=original_content.topology_epoch_id,
            end_topology_epoch_id=next_content.topology_epoch_id,
            start_time=original_content.time,
            end_time=end_time,
            accepted_step=accepted_step,
        )

    def _limited_euler(
        self,
        time: Array,
        evaluation_state: Array,
        combination_base: Array,
        step_size: Array,
        args: Any,
        /,
    ):
        if isinstance(
            self.dynamics,
            PreparedFiniteVolumeDynamics,
        ) and isinstance(
            self.dynamics.method.interface_solver,
            ShallowWaterHydrostaticHLLPlan,
        ):
            if not isinstance(self.fallback_dynamics, PreparedFiniteVolumeDynamics):
                raise TypeError(
                    "Hydrostatic shallow-water fallback must use structured dynamics."
                )
            high_contributions = self.dynamics.balanced_face_contributions(
                time, evaluation_state, args
            )
            fallback_contributions = self.fallback_dynamics.balanced_face_contributions(
                time, evaluation_state, args
            )
            high_residual = self.precision.reduction(
                self.dynamics(time, evaluation_state, args)
            )
            common_residual = self.precision.storage(
                high_residual
                - self.precision.reduction(
                    self.dynamics._balanced_residual(high_contributions)
                )
            )
            return self.positivity.limit_balanced_face_contributions(
                self.dynamics.system,
                combination_base,
                high_contributions,
                fallback_contributions,
                common_residual,
                step_size,
                self.dynamics.discretization,
            )
        high_fluxes, _ = self._face_fluxes(self.dynamics, time, evaluation_state, args)
        fallback_fluxes, _ = self._face_fluxes(
            self.fallback_dynamics, time, evaluation_state, args
        )
        high_residual = self.precision.reduction(
            self.dynamics(time, evaluation_state, args)
        )
        common_residual = self.precision.storage(
            high_residual
            - self.precision.reduction(self._flux_residual(self.dynamics, high_fluxes))
        )
        return self.positivity.limit_face_fluxes(
            self.dynamics.system,
            combination_base,
            high_fluxes,
            fallback_fluxes,
            common_residual,
            step_size,
            self.dynamics.discretization,
        )

    def _precision_report(
        self,
        report: FiniteVolumeAdmissibilityReport,
        /,
    ) -> FiniteVolumeAdmissibilityReport:
        return jax.tree.map(
            lambda value: (
                self.precision.reduction(value) if eqx.is_inexact_array(value) else value
            ),
            report,
        )

    def _candidate(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        args: Any,
        /,
    ):
        stage_initial = self._provide_stage_state(time, state)
        first = self._limited_euler(time, stage_initial, stage_initial, step_size, args)
        second_base = self.precision.storage(
            0.75 * self.precision.reduction(stage_initial)
            + 0.25 * self.precision.reduction(first.state)
        )
        second_time = self.precision.decision(time + step_size)
        second_state = self._provide_stage_state(
            second_time,
            self.precision.storage(first.state),
        )
        second = self._limited_euler(
            second_time,
            second_state,
            second_base,
            self.precision.decision(0.25 * step_size),
            args,
        )
        third_base = self.precision.storage(
            (1.0 / 3.0) * self.precision.reduction(stage_initial)
            + (2.0 / 3.0) * self.precision.reduction(second.state)
        )
        third_time = self.precision.decision(time + 0.5 * step_size)
        third_state = self._provide_stage_state(
            third_time,
            self.precision.storage(second.state),
        )
        third = self._limited_euler(
            third_time,
            third_state,
            third_base,
            self.precision.decision((2.0 / 3.0) * step_size),
            args,
        )
        integrated_flux_rates = tuple(
            self.precision.reduction(
                (1.0 / 6.0) * self.precision.reduction(first_flux)
                + (1.0 / 6.0) * self.precision.reduction(second_flux)
                + (2.0 / 3.0) * self.precision.reduction(third_flux)
            )
            for first_flux, second_flux, third_flux in zip(
                first.integrated_fluxes,
                second.integrated_fluxes,
                third.integrated_fluxes,
                strict=True,
            )
        )
        normal_fluxes = tuple(
            self.precision.flux(
                integrated_rate / self.precision.reduction(measure[..., None])
            )
            for integrated_rate, measure in zip(
                integrated_flux_rates,
                self._face_measures(),
                strict=True,
            )
        )
        if isinstance(third, BalancedPositivityBlendResult):
            if not isinstance(first, BalancedPositivityBlendResult) or not isinstance(
                second, BalancedPositivityBlendResult
            ):
                raise TypeError(
                    "All hydrostatic shallow-water stages must use balanced positivity."
                )
            accepted_contributions = tuple(
                ShallowWaterBalancedFaceResult(
                    self.precision.flux(
                        (1.0 / 6.0)
                        * self.precision.reduction(first_contribution.normal_flux)
                        + (1.0 / 6.0)
                        * self.precision.reduction(second_contribution.normal_flux)
                        + (2.0 / 3.0)
                        * self.precision.reduction(third_contribution.normal_flux)
                    ),
                    self.precision.flux(
                        (1.0 / 6.0)
                        * self.precision.reduction(first_contribution.left_correction)
                        + (1.0 / 6.0)
                        * self.precision.reduction(second_contribution.left_correction)
                        + (2.0 / 3.0)
                        * self.precision.reduction(third_contribution.left_correction)
                    ),
                    self.precision.flux(
                        (1.0 / 6.0)
                        * self.precision.reduction(first_contribution.right_correction)
                        + (1.0 / 6.0)
                        * self.precision.reduction(second_contribution.right_correction)
                        + (2.0 / 3.0)
                        * self.precision.reduction(third_contribution.right_correction)
                    ),
                    third_contribution.max_speed,
                    third_contribution.reconstructed_left,
                    third_contribution.reconstructed_right,
                    third_contribution.dry_face,
                )
                for (
                    first_contribution,
                    second_contribution,
                    third_contribution,
                ) in zip(
                    first.contributions,
                    second.contributions,
                    third.contributions,
                    strict=True,
                )
            )
            return BalancedPositivityBlendResult(
                state=self.precision.storage(third.state),
                report=self._precision_report(third.report),
                contributions=accepted_contributions,
                normal_fluxes=normal_fluxes,
                integrated_fluxes=integrated_flux_rates,
                face_blend_factors=third.face_blend_factors,
            )
        return type(third)(
            state=self.precision.storage(third.state),
            report=self._precision_report(third.report),
            normal_fluxes=normal_fluxes,
            integrated_fluxes=integrated_flux_rates,
            face_blend_factors=third.face_blend_factors,
        )

    def _validate_sliding_state(self, runtime_state: FiniteVolumeRuntimeState, /) -> None:
        """Reject a stale or absent map before any stage evaluates physics."""
        if self.sliding_plan is None:
            if runtime_state.sliding_coupling is not None:
                raise ValueError("Static runtime cannot carry a sliding coupling.")
            return
        coupling = runtime_state.sliding_coupling
        if not isinstance(coupling, PeriodicSlidingCoupling):
            raise ValueError("Sliding runtime state has no certified coupling.")
        if coupling.evidence_id == "" or not bool(np.asarray(coupling.coverage_passed)):
            raise ValueError("Sliding runtime state carries failed overlap evidence.")
        if runtime_state.sliding_coupling_id != coupling.coupling_id:
            raise ValueError("Sliding runtime coupling identity is stale.")
        if coupling.shift_precision != self.sliding_plan.shift_precision:
            raise ValueError("Sliding runtime shift precision is stale.")
        prepared_coupling = self.sliding_initial_coupling
        if (
            prepared_coupling is None
            or coupling.coupling_id != prepared_coupling.coupling_id
        ):
            raise ValueError(
                "Sliding runtime state does not match this prepared runtime; "
                "advance with the accepted successor runtime."
            )

    def _refresh_sliding_after_accept(
        self,
        prior_state: FiniteVolumeRuntimeState,
        result: FiniteVolumeAdvanceResult,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        """Run exactly one host-side sliding transaction after acceptance."""
        if self.sliding_plan is None or not bool(np.asarray(result.accepted)):
            return result
        plan = self.sliding_plan
        accepted_state = result.runtime_state
        shift = plan.evaluate_shift(accepted_state.time, args)
        try:
            coupling = plan.coupling(shift)
        except (TypeError, ValueError, FloatingPointError) as error:
            raise RuntimeError(
                "Accepted sliding refresh failed overlap coverage; "
                "the predecessor state remains the only valid runtime state."
            ) from error

        # One accepted boundary produces one request and one transaction.  The
        # scheduler owns append/commit ordering and keeps failed artifacts out of
        # the runtime state.
        event_kind = getattr(
            TopologyEventKind,
            "SLIDING_REFRESH",
            TopologyEventKind.OVERSET_DONOR_REBUILD,
        )
        request = FiniteVolumeTopologyEventRequest(
            event_kind,
            prior_state.content_state.topology_epoch_id,
            plan.plan_id,
            payload_id=coupling.evidence_id,
            reason="accepted-step sliding coupling refresh",
        )
        accepted_step = int(np.asarray(accepted_state.accepted_step))
        event_time = float(np.asarray(accepted_state.time))
        predecessor = prior_state.content_state.topology_epoch_id
        discretization = self.dynamics.discretization
        if not isinstance(discretization, UnstructuredFiniteVolumeDiscretization):
            raise ValueError("Sliding refresh requires unstructured geometry.")
        successor = FiniteVolumeTopologyEpoch(
            discretization.prepared_id,
            discretization.topology_id,
            discretization.geometry_id,
            parent_epoch_id=predecessor,
            topology_artifact_id=coupling.coupling_id,
            metrics_artifact_id=coupling.evidence_id,
            operators_artifact_id=plan.plan_id,
        )
        remapped_content = FiniteVolumeConservativeContentState(
            accepted_state.content_state.conservative_content,
            accepted_state.content_state.effective_cell_volumes,
            accepted_state.content_state.active_cell_mask,
            accepted_state.content_state.time,
            topology_epoch_id=successor.epoch_id,
            geometry_family_id=accepted_state.content_state.geometry_family_id,
            geometry_layout_id=accepted_state.content_state.geometry_layout_id,
            geometry_version=accepted_state.content_state.geometry_version,
            evidence_policy_id=accepted_state.content_state.evidence_policy_id,
            evidence_version=accepted_state.content_state.evidence_version,
            precision=accepted_state.content_state.precision,
        )
        scheduler = FiniteVolumeTopologyEventScheduler(prior_state.topology_journal)
        scheduler.submit(request, accepted_step, event_time, accepted=True)
        artifact = PeriodicSlidingRefreshArtifact(
            content_state=remapped_content,
            remap=coupling,
            metrics=coupling,
            evidence=coupling,
            status=jnp.asarray(0, dtype=jnp.int32),
            result_id=successor.epoch_id,
        )
        transaction = scheduler.transact(
            accepted=True,
            source_content=accepted_state.content_state,
            artifact=artifact,
            candidate_epoch=successor,
            remap=coupling,
            metrics=coupling,
            evidence=coupling,
            status=TopologyEventStatus.SUCCESS,
            coverage_tolerance=plan.coverage_tolerance,
        )
        if not transaction.committed or transaction.result_epoch is None:
            raise RuntimeError(
                "Accepted sliding refresh transaction failed; "
                "the predecessor state remains the only valid runtime state."
            )
        journal = transaction.journal
        if not transaction.events:
            raise RuntimeError("Sliding refresh committed without an event successor.")
        event_id = transaction.events[-1].event_id
        refreshed_state = FiniteVolumeRuntimeState(
            remapped_content,
            journal,
            accepted_state.step_size,
            accepted_step=accepted_state.accepted_step,
            last_status=accepted_state.last_status,
            controller_state=accepted_state.controller_state,
            integrator_state=accepted_state.integrator_state,
            output_cursor=accepted_state.output_cursor,
            sliding_coupling=coupling,
            sliding_shift=shift,
            sliding_event_id=event_id,
        )
        if not isinstance(
            self.dynamics,
            PreparedUnstructuredFiniteVolumeDynamics,
        ):
            raise RuntimeError("Sliding refresh requires unstructured dynamics.")
        successor_dynamics = self.dynamics.with_sliding_coupling(coupling)
        successor_runtime = PreparedFiniteVolumeRuntime(
            successor_dynamics,
            self.positivity,
            self.policy,
            topology_epoch=transaction.result_epoch,
            stage_state_provider=self.stage_state_provider,
        )
        refreshed_result = eqx.tree_at(
            lambda answer: answer.runtime_state,
            result,
            refreshed_state,
        )
        return eqx.tree_at(
            lambda answer: answer.successor_runtime,
            refreshed_result,
            successor_runtime,
            is_leaf=lambda node: node is None,
        )

    def advance(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        args: Any = None,
        /,
    ) -> FiniteVolumeAdvanceResult:
        if not isinstance(runtime_state, FiniteVolumeRuntimeState):
            raise TypeError("runtime_state must be FiniteVolumeRuntimeState.")
        content_state = runtime_state.content_state
        if content_state.precision.policy_id != self.precision.policy_id:
            raise ValueError("Runtime content precision does not match the runtime.")
        if (
            runtime_state.topology_journal.capacity < self._topology_journal_capacity()
            or runtime_state.topology_journal.current_epoch_id
            != self.initial_topology_epoch.epoch_id
        ):
            raise ValueError("Runtime topology journal does not match the runtime.")
        if self.sliding_plan is None:
            if content_state.topology_epoch_id != self.topology_epoch_id:
                raise ValueError(
                    "Runtime content topology epoch does not match the runtime."
                )
        elif (
            content_state.topology_epoch_id
            != runtime_state.topology_journal.current_epoch_id
        ):
            raise ValueError(
                "Sliding runtime content epoch does not match the journal tip."
            )
        if content_state.geometry_family_id != self.geometry_family_id:
            raise ValueError(
                "Runtime content geometry family does not match the runtime."
            )
        if content_state.geometry_layout_id != self.geometry_layout_id:
            raise ValueError(
                "Runtime content geometry layout does not match the runtime."
            )
        if content_state.evidence_policy_id != self.evidence_policy_id:
            raise ValueError(
                "Runtime content evidence policy does not match the runtime."
            )
        expected_content_shape = (
            int(np.prod(self._state_shape()[:-1])),
            self._state_shape()[-1],
        )
        if content_state.conservative_content.shape != expected_content_shape:
            raise ValueError(
                "Runtime content coordinates do not match the finite-volume layout."
            )
        if self.sliding_plan is not None:
            self._validate_sliding_state(runtime_state)
        average = self._dynamics_cell_average(content_state)
        stage_average = self._provide_stage_state(content_state.time, average)
        if (
            isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.overset is not None
            and self.embedded_redistribution is None
            and self.dynamics.coupling.motion is None
        ):
            self.precision.validate_state(stage_average)
            valid = self._active_state_is_admissible(
                stage_average,
                content_state.active_cell_mask,
            )
            return self._refresh_sliding_after_accept(
                runtime_state,
                self._advance_overset(runtime_state, average, valid, args),
                args,
            )
        if self.embedded_redistribution is not None:
            average = eqx.error_if(
                average,
                jnp.any(content_state.active_cell_mask != self.active_cell_mask)
                | jnp.any(
                    content_state.effective_cell_volumes != self._effective_cell_volumes()
                ),
                "Embedded runtime content must retain certified fluid ownership.",
            )
            self.precision.validate_state(stage_average)
            valid = self._active_state_is_admissible(
                stage_average,
                content_state.active_cell_mask,
            )
            return self._refresh_sliding_after_accept(
                runtime_state,
                self._advance_embedded(runtime_state, average, valid, args),
                args,
            )
        if (
            isinstance(self.dynamics, PreparedUnstructuredFiniteVolumeDynamics)
            and self.dynamics.coupling.motion is not None
        ):
            average = eqx.error_if(
                average,
                ~jnp.all(content_state.active_cell_mask),
                "Fixed-topology ALE runtimes require every cell to remain active.",
            )
            self.precision.validate_state(stage_average)
            valid = jnp.all(self.dynamics.system.admissible(stage_average))
            return self._refresh_sliding_after_accept(
                runtime_state,
                self._advance_ale(runtime_state, average, valid, args),
                args,
            )
        average = eqx.error_if(
            average,
            ~jnp.all(content_state.active_cell_mask),
            "Static finite-volume runtimes require every cell to remain active.",
        )
        average = eqx.error_if(
            average,
            ~jnp.all(
                content_state.effective_cell_volumes == self._effective_cell_volumes()
            ),
            "Runtime effective cell volumes do not match the fixed geometry.",
        )
        self.precision.validate_state(stage_average)
        valid = jnp.all(self.dynamics.system.admissible(stage_average))

        def valid_branch(_):
            return self._advance_valid(runtime_state, average, args)

        def invalid_branch(_):
            state = FiniteVolumeRuntimeState(
                content_state,
                runtime_state.topology_journal,
                self.precision.decision(runtime_state.step_size),
                accepted_step=runtime_state.accepted_step,
                last_status=int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
                controller_state=runtime_state.controller_state,
                integrator_state=runtime_state.integrator_state,
                output_cursor=runtime_state.output_cursor,
                sliding_coupling=runtime_state.sliding_coupling,
                sliding_shift=runtime_state.sliding_shift,
                sliding_event_id=runtime_state.sliding_event_id,
            )
            report = FiniteVolumeAdmissibilityReport(
                high_order_valid=jnp.asarray(False),
                fallback_valid=jnp.asarray(False),
                blend_factor=jnp.asarray(
                    0.0,
                    dtype=jnp.dtype(self.precision.reduction_dtype),
                ),
                activated=jnp.asarray(False),
                minimum_density=self.precision.decision(jnp.min(stage_average[..., 0])),
                limited_state_valid=jnp.asarray(False),
                secondary_reduction_applied=jnp.asarray(False),
                secondary_reduction_factor=jnp.asarray(
                    0.0,
                    dtype=jnp.dtype(self.precision.reduction_dtype),
                ),
            )
            return FiniteVolumeAdvanceResult(
                runtime_state=state,
                accepted=jnp.asarray(False),
                retries=jnp.asarray(0, dtype=jnp.int32),
                attempted_step_size=self.precision.decision(runtime_state.step_size),
                accepted_step_size=jnp.asarray(
                    0.0, dtype=jnp.dtype(self.precision.reduction_dtype)
                ),
                positivity=report,
                accepted_flux_integrals=self._static_accepted_flux_integral_ledger(
                    content_state,
                    content_state,
                    self._zero_static_flux_rates(),
                    jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype)),
                    self.precision.decision(runtime_state.time + runtime_state.step_size),
                    runtime_state.accepted_step,
                ),
                precision_evidence=self.precision.evidence(),
                ale=None,
                embedded=None,
                shallow_water_integrals=self._zero_shallow_water_integrals(),
            )

        return self._refresh_sliding_after_accept(
            runtime_state,
            jax.lax.cond(valid, valid_branch, invalid_branch, operand=None),
            args,
        )

    def advance_prescribed(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FiniteVolumeScheduledAdvanceResult:
        """Attempt exactly ``step_size`` without accepting a clamp or retry."""

        if not isinstance(self.dynamics, PreparedFiniteVolumeDynamics):
            raise ValueError(
                "Prescribed finite-volume replay currently requires stationary "
                "structured dynamics."
            )
        if self.embedded_redistribution is not None or self.sliding_plan is not None:
            raise ValueError(
                "Prescribed finite-volume replay does not support embedded or "
                "sliding topology."
            )
        requested = self.precision.decision(jnp.asarray(step_size).reshape(()))
        requested = eqx.error_if(
            requested,
            ~jnp.isfinite(requested) | (requested <= 0.0),
            "Prescribed finite-volume step_size must be positive and finite.",
        )
        replay_state = FiniteVolumeRuntimeState(
            runtime_state.content_state,
            runtime_state.topology_journal,
            requested,
            accepted_step=runtime_state.accepted_step,
            last_status=runtime_state.last_status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        stage_state = self._provide_stage_state(
            runtime_state.time, runtime_state.cell_average()
        )
        stable = self.precision.decision(
            self.dynamics.stable_step(stage_state, args, cfl=self.policy.cfl)
        )
        attempted = self.advance(replay_state, args)
        tolerance = (
            32.0 * jnp.finfo(requested.dtype).eps * jnp.maximum(jnp.abs(requested), 1.0)
        )
        exact = (
            attempted.accepted
            & (attempted.retries == 0)
            & (jnp.abs(attempted.accepted_step_size - requested) <= tolerance)
        )
        status = jnp.where(
            exact,
            attempted.runtime_state.last_status,
            jnp.where(
                requested > stable + tolerance,
                int(FiniteVolumeRunStatus.STABILITY_LIMIT_EXCEEDED),
                int(FiniteVolumeRunStatus.PRESCRIBED_STEP_REJECTED),
            ),
        )
        content = jax.lax.cond(
            exact,
            lambda _: attempted.runtime_state.content_state,
            lambda _: runtime_state.content_state,
            operand=None,
        )
        final_state = FiniteVolumeRuntimeState(
            content,
            runtime_state.topology_journal,
            self.precision.decision(jnp.where(exact, requested, runtime_state.step_size)),
            accepted_step=jnp.where(
                exact,
                attempted.runtime_state.accepted_step,
                runtime_state.accepted_step,
            ),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        margin = self.precision.decision(stable / requested - 1.0)
        return FiniteVolumeScheduledAdvanceResult(
            runtime_state=final_state,
            attempted=attempted,
            accepted=exact,
            requested_step_size=requested,
            stable_step_size=stable,
            stability_margin=margin,
        )

    def _advance_overset(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        initial_valid: Array,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        """Advance a stationary certified overset map with frozen stage routes."""
        dynamics = self.dynamics
        fallback = self.fallback_dynamics
        if not isinstance(
            dynamics, PreparedUnstructuredFiniteVolumeDynamics
        ) or not isinstance(fallback, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("Overset advancement requires unstructured dynamics.")

        original_content = runtime_state.content_state
        attempted = self.precision.decision(runtime_state.step_size)
        zero = jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype))
        active_density = jnp.where(
            original_content.active_cell_mask,
            original_average.reshape((-1, original_average.shape[-1]))[..., 0],
            jnp.asarray(jnp.inf, dtype=original_average.dtype),
        )
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=zero,
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(jnp.min(active_density)),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=zero,
        )
        accepted = jnp.asarray(False)
        accepted_content = original_content
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        current_dt = attempted
        accepted_flux_ledger = None
        accepted_stage_ledgers = None

        def stage_at(stage_time: Array, version: Array) -> FiniteVolumeStageMetrics:
            base = lower_static_unstructured_stage_metrics(
                dynamics.discretization,
                time=self.precision.decision(stage_time),
                topology_epoch_id=original_content.topology_epoch_id,
            )
            face_blocks = tuple(
                eqx.tree_at(
                    lambda block: block.layout.active_mask,
                    block,
                    block.layout.active_mask
                    & self.active_cell_mask[block.layout.owner_cells]
                    & (
                        (block.layout.neighbour_cells < 0)
                        | self.active_cell_mask[
                            jnp.maximum(block.layout.neighbour_cells, 0)
                        ]
                    ),
                )
                for block in base.face_blocks
            )
            # ownership through the pytree avoids re-running constructor checks
            # while preserving the certified route identity for JIT stages.
            return eqx.tree_at(
                lambda stage: (
                    stage.geometry_version,
                    stage.time,
                    stage.effective_cell_volumes,
                    stage.coordinate_effective_cell_volumes,
                    stage.mesh_volume_rate,
                    stage.active_cell_mask,
                    stage.face_blocks,
                ),
                base,
                (
                    version,
                    self.precision.decision(stage_time),
                    self.effective_cell_volumes,
                    self.effective_cell_volumes,
                    jnp.zeros_like(self.effective_cell_volumes),
                    self.active_cell_mask,
                    face_blocks,
                ),
            )

        def select_tree(condition: Array, new: Any, old: Any, /):
            return jax.tree.map(
                lambda new_value, old_value: jnp.where(condition, new_value, old_value),
                new,
                old,
            )

        for retry in range(self.policy.maximum_retries + 1):
            stage_metrics = (
                stage_at(runtime_state.time, original_content.geometry_version),
                stage_at(
                    runtime_state.time + current_dt,
                    original_content.geometry_version + 1,
                ),
                stage_at(
                    runtime_state.time + 0.5 * current_dt,
                    original_content.geometry_version + 2,
                ),
            )
            zero_candidate = zero_unstructured_ssprk33_content_candidate(
                dynamics,
                original_content,
                stage_metrics,
                stage_metrics[1],
                current_dt,
                runtime_state.accepted_step,
                stage_state_provider=self.stage_state_provider,
            )
            candidate = jax.lax.cond(
                initial_valid,
                lambda _: unstructured_ssprk33_content_candidate(
                    dynamics,
                    fallback,
                    self.positivity,
                    original_content,
                    stage_metrics,
                    stage_metrics[1],
                    current_dt,
                    runtime_state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
                    args,
                    stage_state_provider=self.stage_state_provider,
                    cfl=self.policy.cfl,
                ),
                lambda _: zero_candidate,
                operand=None,
            )
            if retry == 0:
                accepted_flux_ledger = zero_candidate.accepted_flux_integrals
                accepted_stage_ledgers = zero_candidate.stage_rate_ledgers
            record = ~accepted
            last_report = select_tree(record, candidate.positivity, last_report)
            accepted_flux_ledger = select_tree(
                record, zero_candidate.accepted_flux_integrals, accepted_flux_ledger
            )
            accepted_stage_ledgers = select_tree(
                record, candidate.stage_rate_ledgers, accepted_stage_ledgers
            )
            finite = jnp.all(jnp.isfinite(candidate.content_state.conservative_content))
            cfl_valid = current_dt <= candidate.relative_cfl_step
            candidate_valid = (
                initial_valid
                & finite
                & cfl_valid
                & candidate.positivity.fallback_valid
                & candidate.positivity.limited_state_valid
            )
            take = (~accepted) & candidate_valid
            accepted_content = select_tree(
                take, candidate.content_state, accepted_content
            )
            accepted_flux_ledger = select_tree(
                take, candidate.accepted_flux_integrals, accepted_flux_ledger
            )
            accepted_stage_ledgers = select_tree(
                take, candidate.stage_rate_ledgers, accepted_stage_ledgers
            )
            accepted_dt = jnp.where(take, current_dt, accepted_dt)
            retries = jnp.where(take, retry, retries)
            accepted = accepted | take
            current_dt = self.precision.decision(
                current_dt * self.policy.reduction_factor
            )

        if accepted_flux_ledger is None or accepted_stage_ledgers is None:
            raise RuntimeError("Overset retry loop did not form its evidence envelope.")
        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            ~initial_valid,
            int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
            jnp.where(
                accepted & (retries > 0),
                int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
                jnp.where(
                    accepted,
                    int(FiniteVolumeRunStatus.SUCCESS),
                    jnp.where(
                        minimum_reached,
                        int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                        int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                    ),
                ),
            ),
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(jnp.where(accepted, accepted_dt, attempted)),
            accepted_step=runtime_state.accepted_step + accepted.astype(jnp.int32),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=accepted_flux_ledger,
            precision_evidence=self.precision.evidence(),
            ale=None,
            embedded=None,
        )

    def _advance_embedded(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        initial_valid: Array,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        dynamics = self.dynamics
        fallback = self.fallback_dynamics
        redistribution = self.embedded_redistribution
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("Embedded advancement requires unstructured dynamics.")
        if not isinstance(fallback, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError(
                "Embedded fallback advancement requires unstructured dynamics."
            )
        if not isinstance(
            redistribution,
            ConservativeSmallCellRedistributionPlan,
        ):
            raise TypeError("Embedded advancement requires prepared redistribution.")
        embedded_metrics = dynamics.coupling.embedded_metrics
        embedded_boundaries = dynamics.coupling.embedded_boundaries
        if embedded_metrics is None or not isinstance(
            embedded_boundaries,
            UnstructuredEmbeddedBoundarySet,
        ):
            raise ValueError("Embedded advancement requires certified stationary cuts.")
        embedded_stage_template = self.embedded_stage_template
        if not isinstance(embedded_stage_template, FiniteVolumeStageMetrics):
            raise TypeError("Embedded advancement requires a prepared stage template.")

        original_content = runtime_state.content_state

        def stage_at(stage_time: Array) -> FiniteVolumeStageMetrics:
            return eqx.tree_at(
                lambda stage: (
                    stage.time,
                    stage.geometry_version,
                    stage.evidence.evidence_version,
                ),
                embedded_stage_template,
                (
                    self.precision.decision(stage_time),
                    original_content.geometry_version,
                    original_content.evidence_version,
                ),
            )

        attempted = self.precision.decision(runtime_state.step_size)
        accepted = jnp.asarray(False)
        accepted_content = original_content
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype))
        active_density = jnp.where(
            original_content.active_cell_mask,
            original_average.reshape((-1, original_average.shape[-1]))[..., 0],
            jnp.asarray(jnp.inf, dtype=original_average.dtype),
        )
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=zero,
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(jnp.min(active_density)),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=zero,
        )
        current_dt = self.precision.decision(attempted)
        accepted_flux_ledger = None
        accepted_stage_ledgers = None
        recorded_stage_metrics = None
        recorded_stage_rates: tuple[Array, Array, Array] = (
            zero,
            zero,
            zero,
        )
        recorded_maximum_rate = zero
        recorded_cfl_step = jnp.asarray(jnp.inf, dtype=zero.dtype)

        def select_tree(condition: Array, new: Any, old: Any, /):
            return jax.tree.map(
                lambda new_value, old_value: jnp.where(
                    condition,
                    new_value,
                    old_value,
                ),
                new,
                old,
            )

        for retry in range(self.policy.maximum_retries + 1):
            stage_1 = stage_at(runtime_state.time)
            stage_2 = stage_at(self.precision.decision(runtime_state.time + current_dt))
            stage_3 = stage_at(
                self.precision.decision(runtime_state.time + 0.5 * current_dt)
            )
            stage_metrics = (stage_1, stage_2, stage_3)
            zero_candidate = zero_unstructured_ssprk33_content_candidate(
                dynamics,
                original_content,
                stage_metrics,
                stage_2,
                current_dt,
                runtime_state.accepted_step,
                stage_state_provider=self.stage_state_provider,
                redistribution=redistribution,
            )
            metrics_success = jnp.all(
                jnp.stack(
                    tuple(
                        stage.evidence.passed
                        & (
                            stage.evidence.status
                            == int(FiniteVolumeGeometryStatus.SUCCESS)
                        )
                        for stage in stage_metrics
                    )
                )
            )
            precheck_state = (
                original_content
                if self.stage_state_provider is None
                else original_content.with_content(
                    self.stage_state_provider(
                        stage_1.time, original_content.cell_average()
                    )
                    * stage_1.effective_cell_volumes.reshape(
                        (-1,) + (1,) * (original_content.conservative_content.ndim - 1)
                    )
                )
            )
            precheck = dynamics.evaluate_stage(
                precheck_state,
                stage_1,
                args,
                cfl=self.policy.cfl,
                redistribution=redistribution,
            )
            precheck_cfl_valid = jnp.isinf(precheck.relative_cfl_step) | (
                current_dt <= precheck.relative_cfl_step
            )
            evaluate_physics = initial_valid & metrics_success & precheck_cfl_valid
            candidate = jax.lax.cond(
                evaluate_physics,
                lambda _: unstructured_ssprk33_content_candidate(
                    dynamics,
                    fallback,
                    self.positivity,
                    original_content,
                    stage_metrics,
                    stage_2,
                    current_dt,
                    runtime_state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
                    args,
                    cfl=self.policy.cfl,
                    redistribution=redistribution,
                    stage_state_provider=self.stage_state_provider,
                ),
                lambda _: zero_candidate,
                operand=None,
            )
            if retry == 0:
                accepted_flux_ledger = zero_candidate.accepted_flux_integrals
                accepted_stage_ledgers = zero_candidate.stage_rate_ledgers
                recorded_stage_metrics = stage_metrics
                recorded_stage_rates = candidate.stage_maximum_relative_rates
            record_attempt = ~accepted
            recorded_stage_metrics = select_tree(
                record_attempt,
                stage_metrics,
                recorded_stage_metrics,
            )
            recorded_stage_rates = tuple(
                jnp.where(record_attempt, new, old)
                for new, old in zip(
                    candidate.stage_maximum_relative_rates,
                    recorded_stage_rates,
                    strict=True,
                )
            )
            recorded_maximum_rate = jnp.where(
                record_attempt,
                candidate.maximum_relative_rate,
                recorded_maximum_rate,
            )
            recorded_cfl_step = jnp.where(
                record_attempt,
                candidate.relative_cfl_step,
                recorded_cfl_step,
            )
            last_report = select_tree(
                record_attempt,
                self._precision_report(candidate.positivity),
                last_report,
            )
            accepted_flux_ledger = select_tree(
                record_attempt,
                zero_candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                record_attempt,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )

            finite = jnp.all(jnp.isfinite(candidate.content_state.conservative_content))
            cfl_valid = current_dt <= candidate.relative_cfl_step
            candidate_valid = (
                evaluate_physics
                & finite
                & cfl_valid
                & candidate.positivity.fallback_valid
                & candidate.positivity.limited_state_valid
            )
            take = (~accepted) & candidate_valid
            accepted_content = select_tree(
                take,
                candidate.content_state,
                accepted_content,
            )
            accepted_flux_ledger = select_tree(
                take,
                candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                take,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )
            accepted_dt = jnp.where(take, current_dt, accepted_dt)
            retries = jnp.where(take, retry, retries)
            accepted = accepted | take
            current_dt = self.precision.decision(
                current_dt * self.policy.reduction_factor
            )

        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            ~initial_valid,
            int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
            jnp.where(
                accepted & (retries > 0),
                int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
                jnp.where(
                    accepted,
                    int(FiniteVolumeRunStatus.SUCCESS),
                    jnp.where(
                        minimum_reached,
                        int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                        int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                    ),
                ),
            ),
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(
                jnp.where(accepted, accepted_dt, runtime_state.step_size)
            ),
            accepted_step=(runtime_state.accepted_step + accepted.astype(jnp.int32)),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        if (
            accepted_flux_ledger is None
            or accepted_stage_ledgers is None
            or recorded_stage_metrics is None
            or recorded_stage_rates is None
        ):
            raise RuntimeError("Embedded retry loop did not form its evidence envelope.")
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=accepted_flux_ledger,
            precision_evidence=self.precision.evidence(),
            ale=None,
            embedded=FiniteVolumeEmbeddedAdvanceEvidence(
                accepted=accepted,
                stage_rate_ledgers=accepted_stage_ledgers,
                stage_metrics=recorded_stage_metrics,
                accepted_metrics=recorded_stage_metrics[1],
                redistribution=redistribution.report,
                stage_maximum_relative_rates=recorded_stage_rates,
                maximum_relative_rate=recorded_maximum_rate,
                relative_cfl_step=recorded_cfl_step,
            ),
        )

    def _advance_ale(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        initial_valid: Array,
        args: Any,
        /,
    ) -> FiniteVolumeAdvanceResult:
        dynamics = self.dynamics
        fallback = self.fallback_dynamics
        if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("ALE advancement requires unstructured dynamics.")
        if not isinstance(fallback, PreparedUnstructuredFiniteVolumeDynamics):
            raise TypeError("ALE fallback advancement requires unstructured dynamics.")
        motion = dynamics.coupling.motion
        if motion is None:
            raise ValueError(
                "ALE advancement requires prepared fixed-connectivity motion."
            )

        original_content = runtime_state.content_state
        attempted = self.precision.decision(runtime_state.step_size)
        accepted = jnp.asarray(False)
        accepted_content = original_content
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        zero = jnp.asarray(0.0, dtype=jnp.dtype(self.precision.reduction_dtype))
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=zero,
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(jnp.min(original_average[..., 0])),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=zero,
        )
        current_dt = self.precision.decision(attempted)
        accepted_flux_ledger = None
        accepted_stage_ledgers = None
        recorded_geometry = None
        recorded_maximum_rate = zero
        recorded_cfl_step = jnp.asarray(jnp.inf, dtype=zero.dtype)
        recorded_geometry_reduction = jnp.asarray(1.0, dtype=zero.dtype)
        minimum_geometry_reduction = jnp.asarray(1.0, dtype=zero.dtype)

        def select_tree(condition: Array, new: Any, old: Any, /):
            return jax.tree.map(
                lambda new_value, old_value: jnp.where(
                    condition,
                    new_value,
                    old_value,
                ),
                new,
                old,
            )

        for retry in range(self.policy.maximum_retries + 1):
            geometry = motion.prepare_ssprk33_step(
                runtime_state.time,
                current_dt,
                original_content.topology_epoch_id,
                original_content.geometry_version,
                original_content.evidence_version,
                args,
                prior_effective_cell_volumes=(original_content.effective_cell_volumes),
            )
            zero_candidate = zero_unstructured_ale_ssprk33_candidate(
                dynamics,
                original_content,
                geometry,
                current_dt,
                runtime_state.accepted_step,
                stage_state_provider=self.stage_state_provider,
            )
            geometry_success = geometry.passed & (
                geometry.status == int(FiniteVolumeGeometryStatus.SUCCESS)
            )
            evaluate_physics = initial_valid & geometry_success
            candidate = jax.lax.cond(
                evaluate_physics,
                lambda _: unstructured_ale_ssprk33_candidate(
                    dynamics,
                    fallback,
                    self.positivity,
                    original_content,
                    geometry,
                    current_dt,
                    runtime_state.accepted_step + jnp.asarray(1, dtype=jnp.int32),
                    args,
                    cfl=self.policy.cfl,
                    stage_state_provider=self.stage_state_provider,
                ),
                lambda _: zero_candidate,
                operand=None,
            )
            if retry == 0:
                accepted_flux_ledger = zero_candidate.accepted_flux_integrals
                recorded_geometry = geometry
                accepted_stage_ledgers = zero_candidate.stage_rate_ledgers
            record_attempt = ~accepted
            recorded_geometry = select_tree(
                record_attempt,
                geometry,
                recorded_geometry,
            )
            recorded_maximum_rate = jnp.where(
                record_attempt,
                candidate.maximum_relative_rate,
                recorded_maximum_rate,
            )
            recorded_cfl_step = jnp.where(
                record_attempt,
                candidate.relative_cfl_step,
                recorded_cfl_step,
            )
            recorded_geometry_reduction = jnp.where(
                record_attempt,
                geometry.proposed_reduction_factor,
                recorded_geometry_reduction,
            )
            minimum_geometry_reduction = jnp.where(
                record_attempt,
                jnp.minimum(
                    minimum_geometry_reduction,
                    geometry.proposed_reduction_factor,
                ),
                minimum_geometry_reduction,
            )
            last_report = select_tree(
                record_attempt,
                self._precision_report(candidate.positivity),
                last_report,
            )
            accepted_flux_ledger = select_tree(
                record_attempt,
                zero_candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                record_attempt,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )

            finite = jnp.all(jnp.isfinite(candidate.content_state.conservative_content))
            cfl_valid = current_dt <= candidate.relative_cfl_step
            candidate_valid = (
                evaluate_physics
                & finite
                & cfl_valid
                & candidate.positivity.fallback_valid
                & candidate.positivity.limited_state_valid
            )
            take = (~accepted) & candidate_valid
            accepted_content = select_tree(
                take,
                candidate.content_state,
                accepted_content,
            )
            accepted_flux_ledger = select_tree(
                take,
                candidate.accepted_flux_integrals,
                accepted_flux_ledger,
            )
            accepted_stage_ledgers = select_tree(
                take,
                candidate.stage_rate_ledgers,
                accepted_stage_ledgers,
            )
            accepted_dt = jnp.where(take, current_dt, accepted_dt)
            retries = jnp.where(take, retry, retries)
            accepted = accepted | take
            retry_factor = jnp.where(
                geometry_success,
                jnp.asarray(self.policy.reduction_factor, dtype=current_dt.dtype),
                geometry.proposed_reduction_factor.astype(current_dt.dtype),
            )
            current_dt = self.precision.decision(current_dt * retry_factor)

        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            ~initial_valid,
            int(FiniteVolumeRunStatus.INVALID_INITIAL_STATE),
            jnp.where(
                accepted & (retries > 0),
                int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
                jnp.where(
                    accepted,
                    int(FiniteVolumeRunStatus.SUCCESS),
                    jnp.where(
                        minimum_reached,
                        int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                        int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                    ),
                ),
            ),
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(
                jnp.where(accepted, accepted_dt, runtime_state.step_size)
            ),
            accepted_step=(runtime_state.accepted_step + accepted.astype(jnp.int32)),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        if (
            accepted_flux_ledger is None
            or accepted_stage_ledgers is None
            or recorded_geometry is None
        ):
            raise RuntimeError("ALE retry loop did not form its evidence envelope.")
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=accepted_flux_ledger,
            precision_evidence=self.precision.evidence(),
            ale=FiniteVolumeALEAdvanceEvidence(
                accepted=accepted,
                geometry=recorded_geometry,
                maximum_relative_rate=recorded_maximum_rate,
                stage_rate_ledgers=accepted_stage_ledgers,
                relative_cfl_step=recorded_cfl_step,
                geometry_reduction_factor=jnp.where(
                    accepted,
                    minimum_geometry_reduction,
                    recorded_geometry_reduction,
                ),
            ),
            embedded=None,
        )

    def _advance_valid(
        self,
        runtime_state: FiniteVolumeRuntimeState,
        original_average: Array,
        args: Any = None,
        /,
    ) -> FiniteVolumeAdvanceResult:
        initial_stage_average = self._provide_stage_state(
            runtime_state.time, original_average
        )
        self.precision.validate_state(initial_stage_average)
        stable = self.precision.decision(
            self.dynamics.stable_step(
                initial_stage_average,
                args,
                cfl=self.policy.cfl,
            )
        )
        attempted = self.precision.decision(
            jnp.minimum(self.precision.decision(runtime_state.step_size), stable)
        )
        accepted = jnp.asarray(False)
        accepted_average = original_average
        accepted_dt = jnp.asarray(0.0, dtype=attempted.dtype)
        retries = jnp.asarray(0, dtype=jnp.int32)
        last_report = FiniteVolumeAdmissibilityReport(
            high_order_valid=jnp.asarray(False),
            fallback_valid=jnp.asarray(False),
            blend_factor=jnp.asarray(
                0.0,
                dtype=jnp.dtype(self.precision.reduction_dtype),
            ),
            activated=jnp.asarray(False),
            minimum_density=self.precision.decision(
                jnp.min(initial_stage_average[..., 0])
            ),
            limited_state_valid=jnp.asarray(False),
            secondary_reduction_applied=jnp.asarray(False),
            secondary_reduction_factor=jnp.asarray(
                0.0,
                dtype=jnp.dtype(self.precision.reduction_dtype),
            ),
        )
        accepted_flux_rates = self._zero_static_flux_rates()
        accepted_balanced_contributions: (
            tuple[ShallowWaterBalancedFaceResult, ...] | None
        ) = None
        current_dt = self.precision.decision(attempted)
        for retry in range(self.policy.maximum_retries + 1):
            candidate = self._candidate(
                runtime_state.time, original_average, current_dt, args
            )
            finite = jnp.all(jnp.isfinite(candidate.state))
            valid = (
                finite
                & candidate.report.fallback_valid
                & candidate.report.limited_state_valid
            )
            take = (~accepted) & valid
            accepted_average = jnp.where(
                take,
                self.precision.storage(candidate.state),
                accepted_average,
            )
            accepted_dt = jnp.where(
                take,
                self.precision.decision(current_dt),
                accepted_dt,
            )
            retries = jnp.where(take, retry, retries)
            last_report = jax.tree.map(
                lambda new, old: jnp.where(take, new, old),
                candidate.report,
                last_report,
            )
            accepted_flux_rates = tuple(
                jnp.where(
                    take,
                    self.precision.reduction(new),
                    old,
                )
                for new, old in zip(
                    candidate.integrated_fluxes,
                    accepted_flux_rates,
                    strict=True,
                )
            )
            if isinstance(candidate, BalancedPositivityBlendResult):
                if accepted_balanced_contributions is None:
                    accepted_balanced_contributions = jax.tree.map(
                        jnp.zeros_like, candidate.contributions
                    )
                accepted_balanced_contributions = jax.tree.map(
                    lambda new, old: jnp.where(take, new, old),
                    candidate.contributions,
                    accepted_balanced_contributions,
                )
            accepted = accepted | take
            current_dt = self.precision.decision(
                current_dt * self.policy.reduction_factor
            )

        minimum_reached = current_dt < self.precision.decision(
            self.policy.minimum_step_size
        )
        status = jnp.where(
            accepted & (retries > 0),
            int(FiniteVolumeRunStatus.RECOVERED_REJECTION),
            jnp.where(
                accepted,
                int(FiniteVolumeRunStatus.SUCCESS),
                jnp.where(
                    minimum_reached,
                    int(FiniteVolumeRunStatus.MINIMUM_STEP_REACHED),
                    int(FiniteVolumeRunStatus.RETRY_LIMIT_REACHED),
                ),
            ),
        )
        original_content = runtime_state.content_state
        accepted_content = FiniteVolumeConservativeContentState.from_cell_average(
            accepted_average.reshape(original_content.conservative_content.shape),
            original_content.effective_cell_volumes,
            original_content.active_cell_mask,
            self.precision.decision(runtime_state.time + accepted_dt),
            topology_epoch_id=original_content.topology_epoch_id,
            geometry_family_id=original_content.geometry_family_id,
            geometry_layout_id=original_content.geometry_layout_id,
            geometry_version=original_content.geometry_version,
            evidence_policy_id=original_content.evidence_policy_id,
            evidence_version=original_content.evidence_version,
            precision=original_content.precision,
        )
        next_content = jax.lax.cond(
            accepted,
            lambda _: accepted_content,
            lambda _: original_content,
            operand=None,
        )
        next_state = FiniteVolumeRuntimeState(
            next_content,
            runtime_state.topology_journal,
            self.precision.decision(
                jnp.where(accepted, accepted_dt, runtime_state.step_size)
            ),
            accepted_step=(runtime_state.accepted_step + accepted.astype(jnp.int32)),
            last_status=status,
            controller_state=runtime_state.controller_state,
            integrator_state=runtime_state.integrator_state,
            output_cursor=runtime_state.output_cursor,
            sliding_coupling=runtime_state.sliding_coupling,
            sliding_shift=runtime_state.sliding_shift,
            sliding_event_id=runtime_state.sliding_event_id,
        )
        shallow_water_integrals = None
        if isinstance(self.dynamics, PreparedFiniteVolumeDynamics) and isinstance(
            self.dynamics.method.interface_solver,
            ShallowWaterHydrostaticHLLPlan,
        ):
            if accepted_balanced_contributions is None or not isinstance(
                self.dynamics.bathymetry, PreparedShallowWaterBathymetry
            ):
                raise RuntimeError(
                    "Hydrostatic shallow-water acceptance lacks face evidence."
                )
            shallow_water_integrals = ShallowWaterAcceptedFaceIntegrals(
                accepted_balanced_contributions,
                tuple(self.dynamics.discretization.face_measures),
                accepted_dt,
                axis_names=self.dynamics.discretization.grid.axis_names,
                bed_id=self.dynamics.bathymetry.bed_id,
                plan_id=self.dynamics.method.interface_solver.plan_id,
            )
        return FiniteVolumeAdvanceResult(
            runtime_state=next_state,
            accepted=accepted,
            retries=retries,
            attempted_step_size=attempted,
            accepted_step_size=accepted_dt,
            positivity=last_report,
            accepted_flux_integrals=self._static_accepted_flux_integral_ledger(
                original_content,
                next_content,
                accepted_flux_rates,
                accepted_dt,
                self.precision.decision(
                    runtime_state.time + jnp.where(accepted, accepted_dt, attempted)
                ),
                runtime_state.accepted_step + accepted.astype(jnp.int32),
            ),
            precision_evidence=self.precision.evidence(),
            ale=None,
            embedded=None,
            shallow_water_integrals=shallow_water_integrals,
        )


__all__ = [
    "FiniteVolumeALEAdvanceEvidence",
    "FiniteVolumeEmbeddedAdvanceEvidence",
    "FiniteVolumeAdvanceResult",
    "FiniteVolumeScheduledAdvanceResult",
    "FiniteVolumeRunStatus",
    "FiniteVolumeRuntimeState",
    "FiniteVolumeStepPolicy",
    "PreparedFiniteVolumeRuntime",
]
