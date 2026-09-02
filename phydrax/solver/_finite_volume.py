#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._numerics._ssp_runge_kutta import ssprk33_step
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._conservation_ledger import (
    AcceptedConservationIntegralLedger,
    ConservationStageLedger,
)
from ..discretization.finite_volume import (
    AbstractNumericalFluxPlan,
    PreparedFiniteVolumeDynamics,
    PreparedTriangleFiniteVolumeDynamics,
    PreparedUnstructuredFiniteVolumeDynamics,
    ShallowWaterHydrostaticHLLPlan,
)
from ..discretization.finite_volume._geometry_protocol import (
    FiniteVolumeGeometryStatus,
    FiniteVolumeStageMetrics,
)
from ..discretization.finite_volume._positivity import (
    FiniteVolumeAdmissibilityReport,
    FluxPositivityPlan,
)
from ..discretization.finite_volume._small_cell import (
    ConservativeSmallCellRedistributionPlan,
)
from ..discretization.finite_volume._unstructured_motion import (
    UnstructuredALEStepGeometry,
)
from ._finite_volume_content import FiniteVolumeConservativeContentState


PreparedFVDynamics: TypeAlias = (
    PreparedFiniteVolumeDynamics
    | PreparedTriangleFiniteVolumeDynamics
    | PreparedUnstructuredFiniteVolumeDynamics
)


FiniteVolumeStageStateCallback: TypeAlias = Callable[[Array, Array], ArrayLike]


class FiniteVolumeStageStateProvider(StrictModule, NonTrainableState):
    """Immutable stage-state callback with an explicit numerical identity."""

    callback: FiniteVolumeStageStateCallback
    provider_id: str = eqx.field(static=True)

    def __init__(
        self,
        callback: FiniteVolumeStageStateCallback,
        /,
        *,
        provider_id: str,
    ):
        identity = str(provider_id)
        if not callable(callback):
            raise TypeError("callback must be callable.")
        if not identity:
            raise ValueError("provider_id must be non-empty.")
        self.callback = callback
        self.provider_id = identity

    def __call__(self, time: Array, state: Array, /) -> Array:
        provided = jnp.asarray(self.callback(time, state), dtype=state.dtype)
        if provided.shape != state.shape:
            raise ValueError("Stage-state provider output must match the state shape.")
        return provided


SplittingKind: TypeAlias = Literal["godunov", "strang"]


class FiniteVolumeStepResult(StrictModule):
    state: Array
    time: Array
    step_size: Array
    temporal_method_id: str = eqx.field(static=True)


class FiniteVolumeSSPRK3ContentCandidate(StrictModule):
    """Geometry-agnostic conservative-content SSPRK(3,3) candidate."""

    content_state: FiniteVolumeConservativeContentState
    positivity: FiniteVolumeAdmissibilityReport
    stage_rate_ledgers: tuple[
        ConservationStageLedger,
        ConservationStageLedger,
        ConservationStageLedger,
    ]
    accepted_flux_integrals: AcceptedConservationIntegralLedger
    stage_maximum_relative_rates: tuple[Array, Array, Array]
    maximum_relative_rate: Array
    relative_cfl_step: Array
    candidate_id: str = eqx.field(static=True)


class FiniteVolumeALESSPRK3Candidate(StrictModule):
    """Complete content/volume SSPRK(3,3) candidate for one certified ALE step."""

    content_state: FiniteVolumeConservativeContentState
    positivity: FiniteVolumeAdmissibilityReport
    stage_rate_ledgers: tuple[
        ConservationStageLedger,
        ConservationStageLedger,
        ConservationStageLedger,
    ]
    accepted_flux_integrals: AcceptedConservationIntegralLedger
    geometry: UnstructuredALEStepGeometry
    stage_maximum_relative_rates: tuple[Array, Array, Array]
    maximum_relative_rate: Array
    relative_cfl_step: Array
    candidate_id: str = eqx.field(static=True)


def _content_at_metrics(
    reference: FiniteVolumeConservativeContentState,
    conservative_content: Array,
    metrics: FiniteVolumeStageMetrics,
    /,
) -> FiniteVolumeConservativeContentState:
    if reference.geometry_family_id != metrics.geometry_family_id:
        raise ValueError(
            "Target metrics geometry family does not match the source content."
        )
    return FiniteVolumeConservativeContentState(
        conservative_content,
        metrics.effective_cell_volumes,
        metrics.active_cell_mask,
        metrics.time,
        topology_epoch_id=metrics.topology_epoch_id,
        geometry_family_id=metrics.geometry_family_id,
        geometry_layout_id=metrics.geometry_layout_id,
        geometry_version=metrics.geometry_version,
        evidence_policy_id=metrics.evidence.policy_id,
        evidence_version=metrics.evidence.evidence_version,
        precision=reference.precision,
    )


def _provided_content_at_metrics(
    provider: FiniteVolumeStageStateProvider | None,
    state: FiniteVolumeConservativeContentState,
    metrics: FiniteVolumeStageMetrics,
    /,
) -> FiniteVolumeConservativeContentState:
    if provider is None:
        return state
    average = provider(metrics.time, state.cell_average())
    content = average * metrics.effective_cell_volumes.reshape(
        (-1,) + (1,) * (average.ndim - 1)
    )
    return _content_at_metrics(state, content, metrics)


def _combined_admissibility_report(
    reports: tuple[
        FiniteVolumeAdmissibilityReport,
        FiniteVolumeAdmissibilityReport,
        FiniteVolumeAdmissibilityReport,
    ],
    /,
) -> FiniteVolumeAdmissibilityReport:
    return FiniteVolumeAdmissibilityReport(
        high_order_valid=jnp.all(
            jnp.stack(tuple(report.high_order_valid for report in reports))
        ),
        fallback_valid=jnp.all(
            jnp.stack(tuple(report.fallback_valid for report in reports))
        ),
        blend_factor=jnp.min(jnp.stack(tuple(report.blend_factor for report in reports))),
        activated=jnp.any(jnp.stack(tuple(report.activated for report in reports))),
        minimum_density=jnp.min(
            jnp.stack(tuple(report.minimum_density for report in reports))
        ),
        limited_state_valid=jnp.all(
            jnp.stack(tuple(report.limited_state_valid for report in reports))
        ),
        secondary_reduction_applied=jnp.any(
            jnp.stack(tuple(report.secondary_reduction_applied for report in reports))
        ),
        secondary_reduction_factor=jnp.min(
            jnp.stack(tuple(report.secondary_reduction_factor for report in reports))
        ),
    )


def unstructured_ssprk33_content_candidate(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    fallback_dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    positivity: FluxPositivityPlan,
    initial: FiniteVolumeConservativeContentState,
    stage_metrics: tuple[
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
    ],
    accepted_metrics: FiniteVolumeStageMetrics,
    step_size: ArrayLike,
    accepted_step: ArrayLike,
    args: Any = None,
    /,
    *,
    cfl: float,
    redistribution: ConservativeSmallCellRedistributionPlan | None = None,
    stage_state_provider: FiniteVolumeStageStateProvider | None = None,
) -> FiniteVolumeSSPRK3ContentCandidate:
    """Form Shu--Osher content combinations over certified stage metrics."""

    if not isinstance(dynamics, PreparedUnstructuredFiniteVolumeDynamics):
        raise TypeError("dynamics must be PreparedUnstructuredFiniteVolumeDynamics.")
    if not isinstance(
        fallback_dynamics,
        PreparedUnstructuredFiniteVolumeDynamics,
    ):
        raise TypeError(
            "fallback_dynamics must be PreparedUnstructuredFiniteVolumeDynamics."
        )
    if not isinstance(positivity, FluxPositivityPlan):
        raise TypeError("positivity must be FluxPositivityPlan.")
    if not isinstance(initial, FiniteVolumeConservativeContentState):
        raise TypeError("initial must be FiniteVolumeConservativeContentState.")
    if stage_state_provider is not None and not isinstance(
        stage_state_provider, FiniteVolumeStageStateProvider
    ):
        raise TypeError(
            "stage_state_provider must be FiniteVolumeStageStateProvider or None."
        )
    if (
        not isinstance(stage_metrics, tuple)
        or len(stage_metrics) != 3
        or any(not isinstance(item, FiniteVolumeStageMetrics) for item in stage_metrics)
    ):
        raise TypeError("stage_metrics must contain exactly three stage metrics.")
    if not isinstance(accepted_metrics, FiniteVolumeStageMetrics):
        raise TypeError("accepted_metrics must be FiniteVolumeStageMetrics.")
    if any(
        metrics.geometry_family_id != initial.geometry_family_id
        for metrics in (*stage_metrics, accepted_metrics)
    ):
        raise ValueError(
            "SSPRK stage and accepted metrics must match the initial geometry family."
        )
    if (
        dynamics.discretization.prepared_id
        != fallback_dynamics.discretization.prepared_id
        or dynamics.coupling.prepared_id != fallback_dynamics.coupling.prepared_id
        or getattr(dynamics, "overset_policy_id", None)
        != getattr(fallback_dynamics, "overset_policy_id", None)
        or getattr(dynamics, "overset_mapping_id", None)
        != getattr(fallback_dynamics, "overset_mapping_id", None)
        or getattr(dynamics, "overset_epoch_id", None)
        != getattr(fallback_dynamics, "overset_epoch_id", None)
    ):
        raise ValueError(
            "High-order and fallback stage dynamics must be aligned, including "
            "the stationary overset map identity."
        )
    dt = jnp.asarray(step_size, dtype=initial.effective_cell_volumes.dtype)
    if dt.shape != ():
        raise ValueError("step_size must be scalar.")
    stage_1, stage_2, stage_3 = stage_metrics
    candidate_id = canonical_fingerprint(
        {
            "kind": "unstructured-ssprk33-content-candidate",
            "dynamics": dynamics.dynamics_id,
            "stage_state_provider": (
                None if stage_state_provider is None else stage_state_provider.provider_id
            ),
            "redistribution": (
                None if redistribution is None else redistribution.plan_id
            ),
        }
    )
    stage_initial = _provided_content_at_metrics(stage_state_provider, initial, stage_1)

    zero_candidate = zero_unstructured_ssprk33_content_candidate(
        dynamics,
        initial,
        stage_metrics,
        accepted_metrics,
        dt,
        accepted_step,
        redistribution=redistribution,
        stage_state_provider=stage_state_provider,
    )
    high_1 = dynamics.evaluate_stage(
        stage_initial, stage_1, args, cfl=cfl, redistribution=redistribution
    )
    fallback_1 = fallback_dynamics.evaluate_stage(
        stage_initial, stage_1, args, cfl=cfl, redistribution=redistribution
    )
    limited_1 = positivity.limit_stage_rate_ledgers(
        dynamics.system,
        stage_initial.conservative_content,
        high_1.ledger,
        fallback_1.ledger,
        dt,
        stage_2.effective_cell_volumes,
    )
    finite_1 = jnp.all(jnp.isfinite(limited_1.euler_content))
    safe_q1_content = jnp.where(
        finite_1,
        limited_1.euler_content,
        stage_initial.conservative_content,
    )
    q1 = _content_at_metrics(stage_initial, safe_q1_content, stage_2)
    stage_1_valid = (
        limited_1.report.limited_state_valid
        & finite_1
        & (
            jnp.minimum(
                high_1.relative_cfl_step,
                fallback_1.relative_cfl_step,
            )
            >= dt
        )
    )

    def continue_after_stage_1(_):
        stage_q1 = _provided_content_at_metrics(stage_state_provider, q1, stage_2)
        high_2 = dynamics.evaluate_stage(
            stage_q1, stage_2, args, cfl=cfl, redistribution=redistribution
        )
        fallback_2 = fallback_dynamics.evaluate_stage(
            stage_q1, stage_2, args, cfl=cfl, redistribution=redistribution
        )
        initial_reduction_content = stage_initial.precision.reduction(
            stage_initial.conservative_content
        )
        q1_reduction_content = stage_initial.precision.reduction(q1.conservative_content)
        q2_base = stage_initial.precision.storage(
            initial_reduction_content
            + 0.25 * (q1_reduction_content - initial_reduction_content)
        )
        limited_2 = positivity.limit_stage_rate_ledgers(
            dynamics.system,
            q2_base,
            high_2.ledger,
            fallback_2.ledger,
            0.25 * dt,
            stage_3.effective_cell_volumes,
        )
        finite_2 = jnp.all(jnp.isfinite(limited_2.euler_content))
        safe_q2_content = jnp.where(
            finite_2,
            limited_2.euler_content,
            stage_initial.conservative_content,
        )
        q2 = _content_at_metrics(stage_initial, safe_q2_content, stage_3)
        stage_2_valid = (
            limited_2.report.limited_state_valid
            & finite_2
            & (
                jnp.minimum(
                    high_2.relative_cfl_step,
                    fallback_2.relative_cfl_step,
                )
                >= dt
            )
        )

        def continue_after_stage_2(_):
            stage_q2 = _provided_content_at_metrics(stage_state_provider, q2, stage_3)
            high_3 = dynamics.evaluate_stage(
                stage_q2, stage_3, args, cfl=cfl, redistribution=redistribution
            )
            fallback_3 = fallback_dynamics.evaluate_stage(
                stage_q2, stage_3, args, cfl=cfl, redistribution=redistribution
            )
            q2_reduction_content = stage_initial.precision.reduction(
                q2.conservative_content
            )
            qnew_base = stage_initial.precision.storage(
                initial_reduction_content
                + (2.0 / 3.0) * (q2_reduction_content - initial_reduction_content)
            )
            limited_3 = positivity.limit_stage_rate_ledgers(
                dynamics.system,
                qnew_base,
                high_3.ledger,
                fallback_3.ledger,
                (2.0 / 3.0) * dt,
                accepted_metrics.effective_cell_volumes,
            )
            finite_3 = jnp.all(jnp.isfinite(limited_3.euler_content))
            safe_qnew_content = jnp.where(
                finite_3,
                limited_3.euler_content,
                stage_initial.conservative_content,
            )
            accepted_content = _content_at_metrics(
                stage_initial, safe_qnew_content, accepted_metrics
            )
            stage_3_valid = (
                limited_3.report.limited_state_valid
                & finite_3
                & (
                    jnp.minimum(
                        high_3.relative_cfl_step,
                        fallback_3.relative_cfl_step,
                    )
                    >= dt
                )
            )
            ledgers = (
                limited_1.ledger,
                limited_2.ledger,
                limited_3.ledger,
            )
            accepted_flux_integrals = (
                AcceptedConservationIntegralLedger.integrate_ssprk33(
                    ledgers[0],
                    ledgers[1],
                    ledgers[2],
                    dt,
                    start_geometry_version=initial.geometry_version,
                    end_geometry_version=accepted_metrics.geometry_version,
                    start_evidence_version=initial.evidence_version,
                    end_evidence_version=accepted_metrics.evidence.evidence_version,
                    start_topology_epoch_id=initial.topology_epoch_id,
                    end_topology_epoch_id=accepted_metrics.topology_epoch_id,
                    start_time=initial.time,
                    end_time=accepted_metrics.time,
                    accepted_step=accepted_step,
                )
            )
            stage_rates = (
                jnp.maximum(
                    high_1.maximum_relative_rate,
                    fallback_1.maximum_relative_rate,
                ),
                jnp.maximum(
                    high_2.maximum_relative_rate,
                    fallback_2.maximum_relative_rate,
                ),
                jnp.maximum(
                    high_3.maximum_relative_rate,
                    fallback_3.maximum_relative_rate,
                ),
            )
            maximum_rate = jnp.max(jnp.stack(stage_rates))
            relative_cfl_step = jnp.min(
                jnp.stack(
                    (
                        high_1.relative_cfl_step,
                        fallback_1.relative_cfl_step,
                        high_2.relative_cfl_step,
                        fallback_2.relative_cfl_step,
                        high_3.relative_cfl_step,
                        fallback_3.relative_cfl_step,
                    )
                )
            )
            candidate = FiniteVolumeSSPRK3ContentCandidate(
                content_state=accepted_content,
                positivity=_combined_admissibility_report(
                    (limited_1.report, limited_2.report, limited_3.report)
                ),
                stage_rate_ledgers=ledgers,
                accepted_flux_integrals=accepted_flux_integrals,
                stage_maximum_relative_rates=stage_rates,
                maximum_relative_rate=maximum_rate,
                relative_cfl_step=relative_cfl_step,
                candidate_id=candidate_id,
            )
            return jax.lax.cond(
                stage_3_valid,
                lambda _: candidate,
                lambda _: zero_candidate,
                operand=None,
            )

        return jax.lax.cond(
            stage_2_valid,
            continue_after_stage_2,
            lambda _: zero_candidate,
            operand=None,
        )

    return jax.lax.cond(
        stage_1_valid,
        continue_after_stage_1,
        lambda _: zero_candidate,
        operand=None,
    )


def unstructured_ale_ssprk33_candidate(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    fallback_dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    positivity: FluxPositivityPlan,
    initial: FiniteVolumeConservativeContentState,
    geometry: UnstructuredALEStepGeometry,
    step_size: ArrayLike,
    accepted_step: ArrayLike,
    args: Any = None,
    /,
    *,
    cfl: float,
    stage_state_provider: FiniteVolumeStageStateProvider | None = None,
) -> FiniteVolumeALESSPRK3Candidate:
    """Form exact Shu--Osher content and volume combinations for ALE."""

    if not isinstance(geometry, UnstructuredALEStepGeometry):
        raise TypeError("geometry must be UnstructuredALEStepGeometry.")
    certified_content = eqx.error_if(
        initial.conservative_content,
        ~geometry.passed | (geometry.status != int(FiniteVolumeGeometryStatus.SUCCESS)),
        "ALE SSPRK physical stages require a fully certified geometry step.",
    )
    initial = initial.with_content(certified_content)
    content = unstructured_ssprk33_content_candidate(
        dynamics,
        fallback_dynamics,
        positivity,
        initial,
        (geometry.stage_1, geometry.stage_2, geometry.stage_3),
        geometry.accepted_geometry,
        step_size,
        accepted_step,
        args,
        cfl=cfl,
        stage_state_provider=stage_state_provider,
    )
    return FiniteVolumeALESSPRK3Candidate(
        content_state=content.content_state,
        positivity=content.positivity,
        stage_rate_ledgers=content.stage_rate_ledgers,
        accepted_flux_integrals=content.accepted_flux_integrals,
        geometry=geometry,
        stage_maximum_relative_rates=content.stage_maximum_relative_rates,
        maximum_relative_rate=content.maximum_relative_rate,
        relative_cfl_step=content.relative_cfl_step,
        candidate_id=content.candidate_id,
    )


def zero_unstructured_ssprk33_content_candidate(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    initial: FiniteVolumeConservativeContentState,
    stage_metrics: tuple[
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
        FiniteVolumeStageMetrics,
    ],
    accepted_metrics: FiniteVolumeStageMetrics,
    step_size: ArrayLike,
    accepted_step: ArrayLike,
    /,
    *,
    redistribution: ConservativeSmallCellRedistributionPlan | None = None,
    stage_state_provider: FiniteVolumeStageStateProvider | None = None,
) -> FiniteVolumeSSPRK3ContentCandidate:
    """Build a no-physics, zero-integral candidate over fixed stage routes."""

    if (
        not isinstance(stage_metrics, tuple)
        or len(stage_metrics) != 3
        or any(not isinstance(item, FiniteVolumeStageMetrics) for item in stage_metrics)
    ):
        raise TypeError("stage_metrics must contain exactly three stage metrics.")
    if not isinstance(accepted_metrics, FiniteVolumeStageMetrics):
        raise TypeError("accepted_metrics must be FiniteVolumeStageMetrics.")
    if any(
        metrics.geometry_family_id != initial.geometry_family_id
        for metrics in (*stage_metrics, accepted_metrics)
    ):
        raise ValueError(
            "SSPRK stage and accepted metrics must match the initial geometry family."
        )
    if stage_state_provider is not None and not isinstance(
        stage_state_provider, FiniteVolumeStageStateProvider
    ):
        raise TypeError(
            "stage_state_provider must be FiniteVolumeStageStateProvider or None."
        )
    candidate_id = canonical_fingerprint(
        {
            "kind": "unstructured-ssprk33-content-candidate",
            "dynamics": dynamics.dynamics_id,
            "stage_state_provider": (
                None if stage_state_provider is None else stage_state_provider.provider_id
            ),
            "redistribution": (
                None if redistribution is None else redistribution.plan_id
            ),
        }
    )
    dt = jnp.asarray(step_size, dtype=initial.effective_cell_volumes.dtype)
    ledgers = tuple(
        dynamics.zero_stage_ledger(stage, redistribution=redistribution)
        for stage in stage_metrics
    )
    accepted_flux_integrals = AcceptedConservationIntegralLedger.integrate_ssprk33(
        ledgers[0],
        ledgers[1],
        ledgers[2],
        dt,
        start_geometry_version=initial.geometry_version,
        end_geometry_version=accepted_metrics.geometry_version,
        start_evidence_version=initial.evidence_version,
        end_evidence_version=accepted_metrics.evidence.evidence_version,
        start_topology_epoch_id=initial.topology_epoch_id,
        end_topology_epoch_id=accepted_metrics.topology_epoch_id,
        start_time=initial.time,
        end_time=accepted_metrics.time,
        accepted_step=accepted_step,
    )
    dtype = initial.effective_cell_volumes.dtype
    average = initial.cell_average()
    zero = jnp.asarray(0.0, dtype=dtype)
    active_density = jnp.where(
        initial.active_cell_mask,
        average[..., 0],
        jnp.asarray(jnp.inf, dtype=average.dtype),
    )
    report = FiniteVolumeAdmissibilityReport(
        high_order_valid=jnp.asarray(False),
        fallback_valid=jnp.asarray(False),
        blend_factor=zero,
        activated=jnp.asarray(False),
        minimum_density=jnp.min(active_density),
        limited_state_valid=jnp.asarray(False),
        secondary_reduction_applied=jnp.asarray(False),
        secondary_reduction_factor=zero,
    )
    return FiniteVolumeSSPRK3ContentCandidate(
        content_state=initial,
        positivity=report,
        stage_rate_ledgers=ledgers,
        accepted_flux_integrals=accepted_flux_integrals,
        stage_maximum_relative_rates=(zero, zero, zero),
        maximum_relative_rate=zero,
        relative_cfl_step=jnp.asarray(jnp.inf, dtype=dtype),
        candidate_id=candidate_id,
    )


def zero_unstructured_ale_ssprk33_candidate(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    initial: FiniteVolumeConservativeContentState,
    geometry: UnstructuredALEStepGeometry,
    step_size: ArrayLike,
    accepted_step: ArrayLike,
    /,
    *,
    stage_state_provider: FiniteVolumeStageStateProvider | None = None,
) -> FiniteVolumeALESSPRK3Candidate:
    """Build a no-physics, zero-integral candidate for rejected ALE geometry."""

    content = zero_unstructured_ssprk33_content_candidate(
        dynamics,
        initial,
        (geometry.stage_1, geometry.stage_2, geometry.stage_3),
        geometry.accepted_geometry,
        step_size,
        accepted_step,
        stage_state_provider=stage_state_provider,
    )
    return FiniteVolumeALESSPRK3Candidate(
        content_state=content.content_state,
        positivity=content.positivity,
        stage_rate_ledgers=content.stage_rate_ledgers,
        accepted_flux_integrals=content.accepted_flux_integrals,
        geometry=geometry,
        stage_maximum_relative_rates=content.stage_maximum_relative_rates,
        maximum_relative_rate=content.maximum_relative_rate,
        relative_cfl_step=content.relative_cfl_step,
        candidate_id=content.candidate_id,
    )


_COUPLED_UNSTRUCTURED_SSPRK3_ERROR = (
    "Coupled unstructured finite-volume dynamics require "
    "PreparedFiniteVolumeRuntime; UnsplitFiniteVolumeSSPRK3Plan supports only "
    "canonically uncoupled static unstructured dynamics."
)


def _has_unstructured_coupling(
    dynamics: PreparedUnstructuredFiniteVolumeDynamics,
    /,
) -> bool:
    coupling = dynamics.coupling
    return (
        coupling.motion is not None
        or coupling.embedded_boundary is not None
        or coupling.embedded_metrics is not None
        or coupling.embedded_stabilization_policy is not None
        or coupling.embedded_boundaries is not None
        or coupling.cut_boundary_id is not None
        or coupling.vof is not None
        or coupling.capillarity is not None
        or coupling.contact_angles is not None
        or coupling.amr is not None
        or coupling.overset is not None
        or coupling.overset_policy_id is not None
        or coupling.overset_mapping_id is not None
        or coupling.overset_epoch_id is not None
        or coupling.sliding is not None
        or coupling.sliding_coupling is not None
        or coupling.topology_event_capacity != 0
        or coupling.topology_event_policy != "disabled"
        or coupling.topology_event_id is not None
    )


class UnsplitFiniteVolumeSSPRK3Plan(StrictModule, NonTrainableState):
    """Three-stage SSPRK update of the complete FV semidiscretization."""

    dynamics: PreparedFVDynamics
    temporal_method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, dynamics: PreparedFVDynamics, /):
        if not isinstance(
            dynamics,
            (
                PreparedFiniteVolumeDynamics,
                PreparedTriangleFiniteVolumeDynamics,
                PreparedUnstructuredFiniteVolumeDynamics,
            ),
        ):
            raise TypeError("SSPRK3 requires prepared finite-volume dynamics.")
        if isinstance(
            dynamics, PreparedUnstructuredFiniteVolumeDynamics
        ) and _has_unstructured_coupling(dynamics):
            raise ValueError(_COUPLED_UNSTRUCTURED_SSPRK3_ERROR)
        if isinstance(dynamics, PreparedFiniteVolumeDynamics) and isinstance(
            dynamics.method.interface_solver, ShallowWaterHydrostaticHLLPlan
        ):
            raise ValueError(
                "Hydrostatic wet/dry shallow water requires "
                "PreparedFiniteVolumeRuntime stage positivity."
            )
        self.dynamics = dynamics
        self.temporal_method_id = "temporal:ssprk:3:3"
        self.plan_id = canonical_fingerprint(
            {"kind": "unsplit-fv-ssprk3", "dynamics": dynamics.dynamics_id}
        )

    def advance(
        self,
        time: ArrayLike,
        state: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FiniteVolumeStepResult:
        time_ = jnp.asarray(time).reshape(())
        value = jnp.asarray(state)
        dt = jnp.asarray(step_size).reshape(())
        updated = ssprk33_step(self.dynamics, time_, value, dt, args)
        return FiniteVolumeStepResult(updated, time_ + dt, dt, self.temporal_method_id)


class DirectionalSplitFiniteVolumePlan(StrictModule, NonTrainableState):
    """Godunov or symmetric Strang composition of directional FV operators."""

    dynamics: PreparedFiniteVolumeDynamics
    splitting: SplittingKind = eqx.field(static=True)
    temporal_method_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedFiniteVolumeDynamics,
        /,
        *,
        splitting: SplittingKind = "strang",
    ):
        if not isinstance(dynamics, PreparedFiniteVolumeDynamics):
            raise TypeError("Directional splitting requires finite-volume dynamics.")
        if not isinstance(dynamics.method.interface_solver, AbstractNumericalFluxPlan):
            raise ValueError(
                "Directional splitting requires a numerical-flux interface method."
            )
        if splitting not in ("godunov", "strang"):
            raise ValueError("splitting must be 'godunov' or 'strang'.")
        self.dynamics = dynamics
        self.splitting = splitting
        self.temporal_method_id = f"temporal:split:{splitting}"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "directional-split-fv",
                "dynamics": dynamics.dynamics_id,
                "splitting": splitting,
            }
        )

    def _heun_axis(
        self,
        time: Array,
        state: Array,
        step_size: Array,
        axis: int,
        args: Any,
        /,
    ) -> Array:
        first_rate = self.dynamics.axis_residual(time, state, axis, args)
        predictor = state + step_size * first_rate
        second_rate = self.dynamics.axis_residual(time + step_size, predictor, axis, args)
        return state + 0.5 * step_size * (first_rate + second_rate)

    def _auxiliary_rhs(
        self,
        time: Array,
        state: Array,
        args: Any,
        /,
    ) -> Array:
        directional = sum(
            (
                self.dynamics.axis_residual(time, state, axis, args)
                for axis in range(len(self.dynamics.discretization.cell_shape))
            ),
            jnp.zeros_like(state),
        )
        return self.dynamics(time, state, args) - directional

    def advance(
        self,
        time: ArrayLike,
        state: ArrayLike,
        step_size: ArrayLike,
        args: Any = None,
        /,
    ) -> FiniteVolumeStepResult:
        time_ = jnp.asarray(time).reshape(())
        value = jnp.asarray(state)
        dt = jnp.asarray(step_size).reshape(())
        dimension = len(self.dynamics.discretization.cell_shape)
        if self.splitting == "godunov":
            updated = value
            for axis in range(dimension):
                updated = self._heun_axis(time_, updated, dt, axis, args)
        elif dimension == 1:
            updated = self._heun_axis(time_, value, dt, 0, args)
        else:
            updated = value
            half = 0.5 * dt
            for axis in range(dimension):
                updated = self._heun_axis(time_, updated, half, axis, args)
            for axis in reversed(range(dimension)):
                updated = self._heun_axis(time_ + half, updated, half, axis, args)
        first_auxiliary = self._auxiliary_rhs(time_, updated, args)
        auxiliary_predictor = updated + dt * first_auxiliary
        second_auxiliary = self._auxiliary_rhs(time_ + dt, auxiliary_predictor, args)
        updated = updated + 0.5 * dt * (first_auxiliary + second_auxiliary)
        return FiniteVolumeStepResult(updated, time_ + dt, dt, self.temporal_method_id)


__all__ = [
    "DirectionalSplitFiniteVolumePlan",
    "FiniteVolumeALESSPRK3Candidate",
    "FiniteVolumeSSPRK3ContentCandidate",
    "FiniteVolumeStepResult",
    "FiniteVolumeStageStateCallback",
    "FiniteVolumeStageStateProvider",
    "SplittingKind",
    "UnsplitFiniteVolumeSSPRK3Plan",
    "unstructured_ale_ssprk33_candidate",
    "unstructured_ssprk33_content_candidate",
    "zero_unstructured_ale_ssprk33_candidate",
    "zero_unstructured_ssprk33_content_candidate",
]
