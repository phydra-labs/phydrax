#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from operator import index
from typing import Any, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...typing import checked, parse
from ._distributed import (
    DistributedSpectralExecutionPlan,
    SpectralMeshTopology,
    SpectralResourceError,
    SpectralResourceReport,
)


if TYPE_CHECKING:
    from ...equations._les_closures import AlgebraicLESResult, LESFilterScale
    from ...equations._periodic_les import (
        PeriodicAlgebraicLESStage,
        PreparedPeriodicAlgebraicLES,
        PreparedPeriodicFourierGridFilter,
    )


DistributedPeriodicLESSchedule: TypeAlias = Literal["slab", "pencil"]
# Simultaneously live modal gradient, inverse-gradient, modal-stress, and
# stress-collective buffers used by one algebraic LES evaluation.
_PADDED_COMPLEX_WORKSPACE_FIELDS = 36
# Simultaneously live gradient, viscosity, deviatoric-stress, and
# energy-transfer buffers used by one algebraic LES evaluation.
_PADDED_REAL_WORKSPACE_FIELDS = 20
# Three waves, squared wave, inverse squared wave, and admissibility mask.
_RETAINED_REAL_METADATA_FIELDS = 6
_LES_PAYLOAD_SHAPES = ((), (3,), (3, 3))


@final
class _DistributedPeriodicLESResourceEvidence(StrictModule, NonTrainableState):
    """Exact numerical- and checkpoint-phase LES memory evidence."""

    fft_resource: SpectralResourceReport
    padded_complex_workspace_bytes: int = eqx.field(static=True)
    padded_real_workspace_bytes: int = eqx.field(static=True)
    retained_metadata_bytes: int = eqx.field(static=True)
    closure_workspace_bytes: int = eqx.field(static=True)
    retained_state_bytes: int = eqx.field(static=True)
    checkpoint_count: int = eqx.field(static=True)
    checkpoint_bytes: int = eqx.field(static=True)
    numerical_phase_bytes: int = eqx.field(static=True)
    checkpoint_phase_bytes: int = eqx.field(static=True)
    declared_peak_bytes: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    accepted: bool = eqx.field(static=True)
    reasons: tuple[str, ...] = eqx.field(static=True)
    report_id: str = eqx.field(static=True)

    def __init__(
        self,
        fft_resource: SpectralResourceReport,
        /,
        *,
        padded_complex_workspace_bytes: int,
        padded_real_workspace_bytes: int,
        retained_metadata_bytes: int,
        retained_state_bytes: int,
        checkpoint_count: int,
        maximum_bytes: int,
        reasons: Sequence[str] = (),
    ) -> None:
        if not isinstance(fft_resource, SpectralResourceReport):
            raise TypeError("fft_resource must be a SpectralResourceReport.")
        byte_counts = tuple(
            index(value)
            for value in (
                padded_complex_workspace_bytes,
                padded_real_workspace_bytes,
                retained_metadata_bytes,
                retained_state_bytes,
            )
        )
        checkpoints = index(checkpoint_count)
        maximum = index(maximum_bytes)
        if any(value < 0 for value in byte_counts) or checkpoints < 0 or maximum <= 0:
            raise ValueError("LES resource byte counts and limits are invalid.")
        (
            padded_complex,
            padded_real,
            retained_metadata,
            retained_state,
        ) = byte_counts
        closure_workspace = padded_complex + padded_real + retained_metadata
        checkpoint_bytes = retained_state * checkpoints
        numerical_phase = fft_resource.peak_live_bytes + closure_workspace
        checkpoint_phase = retained_state + checkpoint_bytes
        declared_peak = max(numerical_phase, checkpoint_phase)
        refusal_reasons = tuple(fft_resource.reasons) + tuple(
            str(reason) for reason in reasons
        )
        if any(not reason for reason in refusal_reasons):
            raise ValueError("LES resource refusal reasons must be non-empty.")
        if declared_peak > maximum:
            refusal_reasons += (
                f"required LES peak bytes {declared_peak} exceed maximum_bytes {maximum}",
            )
        self.fft_resource = fft_resource
        self.padded_complex_workspace_bytes = padded_complex
        self.padded_real_workspace_bytes = padded_real
        self.retained_metadata_bytes = retained_metadata
        self.closure_workspace_bytes = closure_workspace
        self.retained_state_bytes = retained_state
        self.checkpoint_count = checkpoints
        self.checkpoint_bytes = checkpoint_bytes
        self.numerical_phase_bytes = numerical_phase
        self.checkpoint_phase_bytes = checkpoint_phase
        self.declared_peak_bytes = declared_peak
        self.maximum_bytes = maximum
        self.accepted = not refusal_reasons
        self.reasons = refusal_reasons
        self.report_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-resource-evidence",
                "fft_resource": fft_resource.report_id,
                "padded_complex_workspace_bytes": padded_complex,
                "padded_real_workspace_bytes": padded_real,
                "retained_metadata_bytes": retained_metadata,
                "closure_workspace_bytes": closure_workspace,
                "retained_state_bytes": retained_state,
                "checkpoint_count": checkpoints,
                "checkpoint_bytes": checkpoint_bytes,
                "numerical_phase_bytes": numerical_phase,
                "checkpoint_phase_bytes": checkpoint_phase,
                "declared_peak_bytes": declared_peak,
                "maximum_bytes": maximum,
                "accepted": self.accepted,
                "reasons": list(refusal_reasons),
            }
        )


class _DistributedPeriodicLESResourceError(MemoryError):
    """Typed LES refusal retaining exact consumer-owned resource evidence."""

    report: _DistributedPeriodicLESResourceEvidence

    def __init__(self, report: _DistributedPeriodicLESResourceEvidence, /) -> None:
        self.report = report
        super().__init__(
            "Distributed periodic LES resource preflight refused execution: "
            + "; ".join(report.reasons)
        )


class DistributedPeriodicLESPreparationEvidence(StrictModule, NonTrainableState):
    """Backend-specific support, identity, resource, and sharding evidence."""

    scientific_prepared_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    numerical_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    admitted_payload_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    transform_sequence_ids: tuple[str, str, str, str] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    schedule: DistributedPeriodicLESSchedule = eqx.field(static=True)
    retained_shape: tuple[int, int, int] = eqx.field(static=True)
    evaluation_shape: tuple[int, int, int] = eqx.field(static=True)
    modal_layout_id: str = eqx.field(static=True)
    physical_layout_id: str = eqx.field(static=True)
    padded_modal_layout_id: str = eqx.field(static=True)
    padded_physical_layout_id: str = eqx.field(static=True)
    reduction_axes: tuple[str, ...] = eqx.field(static=True)
    host_gather: bool = eqx.field(static=True)
    differentiable: bool = eqx.field(static=True)
    restart_preserves_sharding: bool = eqx.field(static=True)
    scientific_parity_bound: bool = eqx.field(static=True)
    qualification_inherited: bool = eqx.field(static=True)
    fft_resource: SpectralResourceReport
    resource: _DistributedPeriodicLESResourceEvidence
    report_id: str = eqx.field(static=True)


class PreparedDistributedPeriodicFourierFilter(StrictModule, NonTrainableState):
    """Sharp retained projection placed in the execution plan's modal layout."""

    scientific: Any
    execution: DistributedSpectralExecutionPlan
    live_mask: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        scientific: PreparedPeriodicFourierGridFilter,
        execution: DistributedSpectralExecutionPlan,
        /,
    ) -> None:
        from ...equations._periodic_les import PreparedPeriodicFourierGridFilter

        if not isinstance(scientific, PreparedPeriodicFourierGridFilter):
            raise TypeError("scientific must be a PreparedPeriodicFourierGridFilter.")
        if not isinstance(execution, DistributedSpectralExecutionPlan):
            raise TypeError("execution must be a DistributedSpectralExecutionPlan.")
        discretization = scientific.discretization
        if execution.owner_id != discretization.prepared_id:
            raise ValueError(
                "Distributed filter execution belongs to a foreign scientific owner."
            )
        if execution.precision.policy_id != discretization.plan.precision.policy_id:
            raise ValueError(
                "Distributed filter execution has a foreign precision policy."
            )
        if execution.admitted_payload_shapes != _LES_PAYLOAD_SHAPES:
            raise ValueError(
                "Distributed filter execution does not admit the exact LES payload shapes."
            )
        if execution.spatial_shape != tuple(discretization.modal_shape):
            raise ValueError("Distributed filter and retained discretization disagree.")
        live_mask = execution.place_batched(
            scientific.live_mask,
            representation="modal",
        )
        self.scientific = scientific
        self.execution = execution
        self.live_mask = live_mask
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-periodic-fourier-grid-filter",
                "scientific_filter": scientific.prepared_id,
                "owner": execution.owner_id,
                "precision": execution.precision.policy_id,
                "numerical": execution.numerical_id,
                "execution": execution.execution_id,
                "execution_plan": execution.plan_id,
                "payload_shapes": [
                    list(shape) for shape in execution.admitted_payload_shapes
                ],
                "transform_sequences": [
                    execution.report.forward_sequence_id,
                    execution.report.inverse_sequence_id,
                    execution.report.padded_forward_sequence_id,
                    execution.report.padded_inverse_sequence_id,
                ],
                "topology": execution.topology.topology_id,
                "layout": execution.modal_layout.layout_id,
                "sharding": "retained-modal-layout",
            }
        )

    def apply(self, coefficients: ArrayLike, /) -> Array:
        value = self.execution.place_batched(
            coefficients,
            representation="modal",
        )
        trailing = (1,) * (value.ndim - 3)
        result = value * self.live_mask.reshape(self.live_mask.shape + trailing)
        return self.execution.place_batched(result, representation="modal")


class DistributedPeriodicLESStage(StrictModule):
    """One distributed algebraic SGS action and its global work evidence."""

    velocity_gradient: Array
    filter_scale: LESFilterScale
    model_result: AlgebraicLESResult
    modal_deviatoric_specific_stress: Array
    unprojected_rate: Array
    projected_rate: Array
    modeled_dissipation: Array
    unprojected_energy_rate: Array
    modal_energy_rate: Array
    energy_identity_defect: Array
    projection_energy_defect: Array
    maximum_kinematic_viscosity: Array
    divergence_norm: Array
    imaginary_leakage: Array
    finite: Array
    dissipative: Array
    energy_consistent: Array
    sharding_preserved: bool = eqx.field(static=True)
    reduction_axes: tuple[str, ...] = eqx.field(static=True)
    modal_layout_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)


class DistributedPeriodicLESStepRestriction(StrictModule):
    """Globally reduced state-dependent explicit step evidence."""

    advective: Array
    molecular_diffusive: Array
    algebraic_les_diffusive: Array
    combined_diffusive: Array
    etdrk_selected: Array
    fully_explicit_selected: Array
    maximum_kinematic_viscosity: Array
    finite: Array
    reduction_axes: tuple[str, ...] = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)


class DistributedPeriodicLESRestartEvidence(StrictModule):
    """A restart payload bound to the exact backend topology and modal sharding."""

    state: Array
    finite: Array
    sharding_preserved: bool = eqx.field(static=True)
    owner_id: str = eqx.field(static=True)
    precision_policy_id: str = eqx.field(static=True)
    numerical_id: str = eqx.field(static=True)
    execution_id: str = eqx.field(static=True)
    execution_plan_id: str = eqx.field(static=True)
    admitted_payload_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    transform_sequence_ids: tuple[str, str, str, str] = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    restart_id: str = eqx.field(static=True)


class DistributedPeriodicLESParityEvidence(StrictModule):
    """Measured one-device parity without extending scientific qualification."""

    projected_rate_maximum_error: Array
    stress_maximum_error: Array
    modeled_dissipation_error: Array
    finite: Array
    passed: Array
    absolute_tolerance: float = eqx.field(static=True)
    relative_tolerance: float = eqx.field(static=True)
    scientific_prepared_id: str = eqx.field(static=True)
    backend_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    qualification_inherited: bool = eqx.field(static=True)


class DistributedPeriodicLESPlan(StrictModule, NonTrainableState):
    """Bind periodic LES science to a distinct slab or pencil execution identity."""

    scientific: Any
    topology: SpectralMeshTopology
    schedule: DistributedPeriodicLESSchedule = eqx.field(static=True)
    checkpoint_count: int = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        scientific: PreparedPeriodicAlgebraicLES,
        topology: SpectralMeshTopology,
        /,
        *,
        schedule: DistributedPeriodicLESSchedule = "slab",
        checkpoint_count: int = 0,
        maximum_bytes: int = 2 * 1024**3,
    ) -> None:
        from ...equations._periodic_les import PreparedPeriodicAlgebraicLES

        if not isinstance(scientific, PreparedPeriodicAlgebraicLES):
            raise TypeError("scientific must be a PreparedPeriodicAlgebraicLES.")
        if not isinstance(topology, SpectralMeshTopology):
            raise TypeError("topology must be a SpectralMeshTopology.")
        schedule = parse(schedule, DistributedPeriodicLESSchedule, "schedule")
        checkpoints = index(checkpoint_count)
        maximum = index(maximum_bytes)
        if checkpoints < 0 or maximum <= 0:
            raise ValueError("checkpoint_count and maximum_bytes are invalid.")
        retained = tuple(scientific.grid_filter.discretization.modal_shape)
        evaluation = tuple(scientific.closure_method.dealiasing.evaluation.modal_shape)
        if len(retained) != 3 or len(evaluation) != 3:
            raise ValueError(
                "Distributed periodic LES requires a three-dimensional grid."
            )
        if scientific.closure_method.dealiasing.report.kind != "oversampling":
            raise ValueError(
                "Distributed periodic LES requires oversampled SGS evaluation."
            )
        self.scientific = scientific
        self.topology = topology
        self.schedule = schedule
        self.checkpoint_count = checkpoints
        self.maximum_bytes = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-plan",
                "scientific_prepared": scientific.prepared_id,
                "owner": scientific.grid_filter.discretization.prepared_id,
                "precision": scientific.grid_filter.discretization.plan.precision.policy_id,
                "payload_shapes": [list(shape) for shape in _LES_PAYLOAD_SHAPES],
                "topology": topology.topology_id,
                "schedule": schedule,
                "checkpoint_count": checkpoints,
                "maximum_bytes": maximum,
                "qualification": "backend-specific-not-inherited",
            }
        )

    def prepare(self, /) -> PreparedDistributedPeriodicLES:
        self.topology.require_available()
        return PreparedDistributedPeriodicLES(self)


class PreparedDistributedPeriodicLES(StrictModule, NonTrainableState):
    """Sharding-preserving periodic Fourier LES backend adapter."""

    plan: DistributedPeriodicLESPlan
    scientific: Any
    execution: DistributedSpectralExecutionPlan
    grid_filter: PreparedDistributedPeriodicFourierFilter
    quadrature_weights: Array
    wavenumbers: tuple[Array, Array, Array]
    wavenumber_squared: Array
    inverse_wavenumber_squared: Array
    admissibility_mask: Array
    preparation: DistributedPeriodicLESPreparationEvidence
    prepared_id: str = eqx.field(static=True)

    @checked
    def __init__(self, plan: DistributedPeriodicLESPlan, /) -> None:
        plan.topology.require_available()
        scientific = plan.scientific
        discretization = scientific.grid_filter.discretization
        retained = tuple(discretization.modal_shape)
        evaluation = tuple(scientific.closure_method.dealiasing.evaluation.modal_shape)
        coefficient_dtype = np.dtype(discretization.plan.precision.coefficient_dtype)
        physical_dtype = np.dtype(discretization.plan.precision.physical_dtype)
        evaluation_points = prod(evaluation)
        retained_points = prod(retained)
        padded_complex_workspace = (
            evaluation_points
            * _PADDED_COMPLEX_WORKSPACE_FIELDS
            * coefficient_dtype.itemsize
        )
        padded_real_workspace = (
            evaluation_points * _PADDED_REAL_WORKSPACE_FIELDS * physical_dtype.itemsize
        )
        retained_metadata = (
            retained_points * _RETAINED_REAL_METADATA_FIELDS * physical_dtype.itemsize
        )
        retained_state = retained_points * 3 * coefficient_dtype.itemsize

        def les_resource(
            fft_resource: SpectralResourceReport, /
        ) -> _DistributedPeriodicLESResourceEvidence:
            return _DistributedPeriodicLESResourceEvidence(
                fft_resource,
                padded_complex_workspace_bytes=padded_complex_workspace,
                padded_real_workspace_bytes=padded_real_workspace,
                retained_metadata_bytes=retained_metadata,
                retained_state_bytes=retained_state,
                checkpoint_count=plan.checkpoint_count,
                maximum_bytes=plan.maximum_bytes,
            )

        try:
            execution_plan = DistributedSpectralExecutionPlan.from_discretization(
                plan.topology,
                discretization,
                schedule=plan.schedule,
                padded_shape=evaluation,
                state_shape=(3,),
                admitted_payload_shapes=_LES_PAYLOAD_SHAPES,
                maximum_bytes=plan.maximum_bytes,
            )
        except SpectralResourceError as error:
            raise _DistributedPeriodicLESResourceError(
                les_resource(error.report)
            ) from None
        execution = execution_plan.prepare()
        resource = les_resource(execution.report.resource)
        if not resource.accepted:
            raise _DistributedPeriodicLESResourceError(resource)
        grid_filter = PreparedDistributedPeriodicFourierFilter(
            scientific.grid_filter,
            execution,
        )
        weights = execution.place_batched(
            scientific.closure_method.dealiasing.evaluation.quadrature_weights,
            representation="physical",
            padded=True,
        )
        # The plan validated a three-dimensional grid: one wavenumber array per axis.
        wave_x, wave_y, wave_z = scientific.projector.wavenumbers
        wavenumbers = (
            execution.place_batched(wave_x, representation="modal"),
            execution.place_batched(wave_y, representation="modal"),
            execution.place_batched(wave_z, representation="modal"),
        )
        wavenumber_squared = execution.place_batched(
            scientific.projector.wavenumber_squared,
            representation="modal",
        )
        inverse_wavenumber_squared = execution.place_batched(
            scientific.projector.inverse_wavenumber_squared,
            representation="modal",
        )
        admissibility_mask = execution.place_batched(
            scientific.projector.admissibility_mask,
            representation="modal",
        )
        sequence_ids = (
            execution.report.forward_sequence_id,
            execution.report.inverse_sequence_id,
            execution.report.padded_forward_sequence_id,
            execution.report.padded_inverse_sequence_id,
        )
        backend_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-periodic-les",
                "plan": plan.plan_id,
                "scientific_prepared": scientific.prepared_id,
                "model": scientific.model.prepared_id,
                "filter": grid_filter.prepared_id,
                "owner": execution.owner_id,
                "precision": execution.precision.policy_id,
                "numerical": execution.numerical_id,
                "execution": execution.execution_id,
                "execution_plan": execution.plan_id,
                "payload_shapes": [
                    list(shape) for shape in execution.admitted_payload_shapes
                ],
                "transform_sequences": list(sequence_ids),
                "fft_resource": execution.report.resource.report_id,
                "les_resource": resource.report_id,
                "topology": plan.topology.topology_id,
                "schedule": plan.schedule,
                "runtime_scope": "distributed-3d-unit-density-full-complex-fourier",
                "qualification": "backend-specific-not-inherited",
            }
        )
        report_id = canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-preparation-evidence",
                "backend": backend_id,
                "owner": execution.owner_id,
                "precision": execution.precision.policy_id,
                "numerical": execution.numerical_id,
                "execution": execution.execution_id,
                "execution_plan": execution.plan_id,
                "payload_shapes": [
                    list(shape) for shape in execution.admitted_payload_shapes
                ],
                "transform_sequences": list(sequence_ids),
                "fft_resource": execution.report.resource.report_id,
                "les_resource": resource.report_id,
                "retained_shape": list(retained),
                "evaluation_shape": list(evaluation),
                "layouts": [
                    execution.modal_layout.layout_id,
                    execution.physical_layout.layout_id,
                    execution.padded_modal_layout.layout_id,
                    execution.padded_physical_layout.layout_id,
                ],
                "host_gather": False,
                "restart_preserves_sharding": True,
                "qualification_inherited": False,
            }
        )
        preparation = DistributedPeriodicLESPreparationEvidence(
            scientific_prepared_id=scientific.prepared_id,
            backend_id=backend_id,
            owner_id=execution.owner_id,
            precision_policy_id=execution.precision.policy_id,
            numerical_id=execution.numerical_id,
            execution_id=execution.execution_id,
            execution_plan_id=execution.plan_id,
            admitted_payload_shapes=execution.admitted_payload_shapes,
            transform_sequence_ids=sequence_ids,
            topology_id=plan.topology.topology_id,
            schedule=plan.schedule,
            retained_shape=retained,
            evaluation_shape=evaluation,
            modal_layout_id=execution.modal_layout.layout_id,
            physical_layout_id=execution.physical_layout.layout_id,
            padded_modal_layout_id=execution.padded_modal_layout.layout_id,
            padded_physical_layout_id=execution.padded_physical_layout.layout_id,
            reduction_axes=execution.modal_layout.used_mesh_axes,
            host_gather=False,
            differentiable=True,
            restart_preserves_sharding=True,
            scientific_parity_bound=True,
            qualification_inherited=False,
            fft_resource=execution.report.resource,
            resource=resource,
            report_id=report_id,
        )
        self.plan = plan
        self.scientific = scientific
        self.execution = execution
        self.grid_filter = grid_filter
        self.quadrature_weights = weights
        self.preparation = preparation
        self.prepared_id = backend_id

        self.wavenumbers = wavenumbers
        self.wavenumber_squared = wavenumber_squared
        self.inverse_wavenumber_squared = inverse_wavenumber_squared
        self.admissibility_mask = admissibility_mask

    def _validate_state(self, state: ArrayLike, owner: str, /) -> Array:
        value = self.scientific.projector.validate_state(state, owner=owner)
        return self.execution.place(value, representation="modal")

    def _zero_forbidden_modes(self, state: ArrayLike, /) -> Array:
        value = self._validate_state(state, "Distributed projected velocity")
        mask = jnp.real(self.admissibility_mask).astype("bool")
        return self.execution.place(
            value * mask[..., None],
            representation="modal",
        )

    def _project(self, state: ArrayLike, /) -> Array:
        value = self._zero_forbidden_modes(state)
        longitudinal = jnp.zeros(self.execution.spatial_shape, dtype=value.dtype)
        for component, wave in enumerate(self.wavenumbers):
            longitudinal = (
                longitudinal + jnp.real(wave).astype(value.dtype) * value[..., component]
            )
        inverse = jnp.real(self.inverse_wavenumber_squared).astype(value.real.dtype)
        components = tuple(
            value[..., component]
            - jnp.real(wave).astype(value.dtype) * inverse * longitudinal
            for component, wave in enumerate(self.wavenumbers)
        )
        return self._zero_forbidden_modes(jnp.stack(components, axis=-1))

    def _divergence(self, state: ArrayLike, /) -> Array:
        value = self._zero_forbidden_modes(state)
        result = jnp.zeros(self.execution.spatial_shape, dtype=value.dtype)
        for component, wave in enumerate(self.wavenumbers):
            result = (
                result + 1j * jnp.real(wave).astype(value.dtype) * value[..., component]
            )
        return self.execution.place_batched(result, representation="modal")

    def validate_state(
        self, state: ArrayLike, /, *, owner: str = "Distributed velocity"
    ) -> Array:
        """Validate and place a modal velocity without changing its global value."""
        return self._validate_state(state, owner)

    def zero_forbidden_modes(self, state: ArrayLike, /) -> Array:
        """Apply the retained admissibility mask while preserving modal sharding."""
        return self._zero_forbidden_modes(state)

    def project(self, state: ArrayLike, /) -> Array:
        """Apply the distributed periodic Leray projection."""
        return self._project(state)

    def divergence(self, state: ArrayLike, /) -> Array:
        """Return modal divergence in the execution plan's retained layout."""
        return self._divergence(state)

    def rotational_unprojected_rate(self, state: ArrayLike, /) -> Array:
        """Evaluate the dealiased rotational advective rate before Leray projection."""
        value = self.grid_filter.apply(
            self._validate_state(state, "Distributed rotational velocity")
        )
        rate = self.execution.rotational_nonlinear(value)
        return self.grid_filter.apply(rate)

    def rotational_rate(self, state: ArrayLike, /) -> Array:
        """Evaluate the dealiased rotational advective rate after projection."""
        return self._project(self.rotational_unprojected_rate(state))

    def evaluate(self, state: ArrayLike, /) -> DistributedPeriodicLESStage:
        """Evaluate filtered gradient, SGS stress, divergence, projection, and work."""
        from ...equations._les_closures import AlgebraicLESInputs

        retained = self._validate_state(state, "Distributed periodic LES velocity")
        live = self.grid_filter.apply(retained)
        embedded = self.execution.pad_modal_batched(live)
        gradient_modal = jnp.stack(
            tuple(
                self.execution.modal_derivative_batched(
                    embedded,
                    axis,
                    padded=True,
                )
                for axis in range(3)
            ),
            axis=-1,
        )
        velocity_gradient = jnp.real(
            self.execution.to_physical_batched(gradient_modal, padded=True)
        )
        model_result = self.scientific.model.evaluate(
            AlgebraicLESInputs(
                velocity_gradient,
                self.scientific.grid_filter.filter_scale,
            )
        )
        padded_modal_stress = self.execution.to_modal_batched(
            model_result.specific_deviatoric_stress,
            padded=True,
        )
        modal_stress = self.grid_filter.apply(
            self.execution.unpad_modal_batched(padded_modal_stress)
        )
        rate_components = []
        for component in range(3):
            divergence = jnp.zeros(
                self.execution.spatial_shape,
                dtype=modal_stress.dtype,
            )
            for axis in range(3):
                divergence = divergence + self.execution.modal_derivative_batched(
                    modal_stress[..., component, axis],
                    axis,
                )
            rate_components.append(-divergence)
        unprojected_rate = self.grid_filter.apply(
            jnp.stack(tuple(rate_components), axis=-1)
        )
        projected_rate = self._project(unprojected_rate)
        modeled_dissipation = jnp.real(
            self.execution.diagnostics_batched(
                self.quadrature_weights * model_result.energy_transfer,
                representation="physical",
                padded=True,
            ).total
        )
        unprojected_energy_rate = jnp.real(
            self.execution.global_inner_product(live, unprojected_rate)
        )
        modal_energy_rate = jnp.real(
            self.execution.global_inner_product(live, projected_rate)
        )
        identity_defect = modal_energy_rate + modeled_dissipation
        projection_defect = modal_energy_rate - unprojected_energy_rate
        maximum_viscosity = self.execution.diagnostics_batched(
            model_result.kinematic_viscosity,
            representation="physical",
            padded=True,
        ).maximum_absolute
        divergence = self._divergence(projected_rate)
        divergence_norm = self.execution.diagnostics_batched(divergence).l2_norm
        projected_physical = self.execution.to_physical_batched(projected_rate)
        imaginary_leakage = self.execution.diagnostics_batched(
            jnp.imag(projected_physical),
            representation="physical",
        ).maximum_absolute
        physical_finite = (
            jnp.all(jnp.isfinite(velocity_gradient), axis=(-2, -1))
            & jnp.isfinite(model_result.kinematic_viscosity)
            & jnp.all(
                jnp.isfinite(model_result.specific_deviatoric_stress),
                axis=(-2, -1),
            )
            & jnp.isfinite(model_result.energy_transfer)
        )
        modal_finite = jnp.all(jnp.isfinite(projected_rate), axis=-1)
        finite = (
            self.execution.global_all(
                physical_finite,
                representation="physical",
                padded=True,
            )
            & self.execution.global_all(modal_finite)
            & jnp.isfinite(modeled_dissipation)
            & jnp.isfinite(identity_defect)
            & jnp.isfinite(projection_defect)
        )
        tolerance = jnp.asarray(
            self.scientific.plan.energy_tolerance,
            dtype=modeled_dissipation.dtype,
        )
        nonnegative = self.execution.global_all(
            (model_result.kinematic_viscosity >= 0.0)
            & (model_result.energy_transfer >= -tolerance),
            representation="physical",
            padded=True,
        )
        energy_scale = jnp.maximum(
            jnp.asarray(1.0, dtype=modeled_dissipation.dtype),
            jnp.maximum(jnp.abs(modal_energy_rate), jnp.abs(modeled_dissipation)),
        )
        dissipative = nonnegative & (modal_energy_rate <= tolerance * energy_scale)
        energy_consistent = jnp.abs(identity_defect) <= tolerance * energy_scale
        return DistributedPeriodicLESStage(
            velocity_gradient=velocity_gradient,
            filter_scale=self.scientific.grid_filter.filter_scale,
            model_result=model_result,
            modal_deviatoric_specific_stress=modal_stress,
            unprojected_rate=unprojected_rate,
            projected_rate=projected_rate,
            modeled_dissipation=modeled_dissipation,
            unprojected_energy_rate=unprojected_energy_rate,
            modal_energy_rate=modal_energy_rate,
            energy_identity_defect=identity_defect,
            projection_energy_defect=projection_defect,
            maximum_kinematic_viscosity=maximum_viscosity,
            divergence_norm=divergence_norm,
            imaginary_leakage=imaginary_leakage,
            finite=finite,
            dissipative=dissipative,
            energy_consistent=energy_consistent,
            sharding_preserved=True,
            reduction_axes=self.execution.modal_layout.used_mesh_axes,
            modal_layout_id=self.execution.modal_layout.layout_id,
            topology_id=self.execution.topology.topology_id,
            backend_id=self.prepared_id,
        )

    def step_restriction(
        self,
        state: ArrayLike,
        molecular_viscosity: ArrayLike,
        /,
        *,
        stage: DistributedPeriodicLESStage | None = None,
    ) -> DistributedPeriodicLESStepRestriction:
        """Return conservative global advective and diffusive bounds."""
        value = self._validate_state(state, "Distributed restriction velocity")
        stage_ = self.evaluate(value) if stage is None else stage
        if not isinstance(stage_, DistributedPeriodicLESStage):
            raise TypeError("stage must be a DistributedPeriodicLESStage or None.")
        if stage_.backend_id != self.prepared_id:
            raise ValueError("LES stage was produced by a different distributed backend.")
        live = self.grid_filter.apply(value)
        velocity = jnp.real(self.execution.to_physical_batched(live))
        widths = self.scientific.grid_filter.filter_scale.directional_widths.astype(
            velocity.dtype
        )
        advective_frequency = self.execution.diagnostics_batched(
            jnp.sum(jnp.abs(velocity) / widths, axis=-1),
            representation="physical",
        ).maximum_absolute
        infinity = jnp.asarray(jnp.inf, dtype=advective_frequency.dtype)
        positive_advective = advective_frequency > 0.0
        advective = jnp.where(
            positive_advective,
            1.0
            / jnp.where(
                positive_advective,
                advective_frequency,
                jnp.ones_like(advective_frequency),
            ),
            infinity,
        )
        admissible_k2 = jnp.where(
            jnp.real(self.admissibility_mask).astype("bool"),
            jnp.real(self.wavenumber_squared),
            jnp.zeros_like(jnp.real(self.wavenumber_squared)),
        )
        maximum_k2 = self.execution.diagnostics_batched(admissible_k2).maximum_absolute
        maximum_viscosity = stage_.maximum_kinematic_viscosity.astype(
            advective_frequency.dtype
        )
        molecular = jnp.asarray(molecular_viscosity, dtype=advective_frequency.dtype)
        molecular = molecular.reshape(())
        positive_sgs = (maximum_viscosity > 0.0) & (maximum_k2 > 0.0)
        algebraic_diffusive = jnp.where(
            positive_sgs,
            1.0
            / jnp.where(
                positive_sgs,
                maximum_viscosity * maximum_k2,
                jnp.ones_like(maximum_viscosity),
            ),
            infinity,
        )
        positive_molecular = (molecular > 0.0) & (maximum_k2 > 0.0)
        molecular_diffusive = jnp.where(
            positive_molecular,
            2.0
            / jnp.where(
                positive_molecular,
                molecular * maximum_k2,
                jnp.ones_like(molecular),
            ),
            infinity,
        )
        combined_coefficient = molecular + 2.0 * maximum_viscosity
        positive_combined = (combined_coefficient > 0.0) & (maximum_k2 > 0.0)
        combined_diffusive = jnp.where(
            positive_combined,
            2.0
            / jnp.where(
                positive_combined,
                combined_coefficient * maximum_k2,
                jnp.ones_like(combined_coefficient),
            ),
            infinity,
        )
        state_finite = self.execution.global_all(jnp.all(jnp.isfinite(value), axis=-1))
        velocity_finite = self.execution.global_all(
            jnp.all(jnp.isfinite(velocity), axis=-1),
            representation="physical",
        )
        finite = (
            stage_.finite
            & stage_.dissipative
            & stage_.energy_consistent
            & state_finite
            & velocity_finite
            & jnp.isfinite(advective_frequency)
            & jnp.isfinite(maximum_viscosity)
            & jnp.isfinite(molecular)
            & (molecular >= 0.0)
        )
        return DistributedPeriodicLESStepRestriction(
            advective=advective,
            molecular_diffusive=molecular_diffusive,
            algebraic_les_diffusive=algebraic_diffusive,
            combined_diffusive=combined_diffusive,
            etdrk_selected=jnp.minimum(advective, algebraic_diffusive),
            fully_explicit_selected=jnp.minimum(advective, combined_diffusive),
            maximum_kinematic_viscosity=maximum_viscosity,
            finite=finite,
            reduction_axes=self.execution.modal_layout.used_mesh_axes,
            backend_id=self.prepared_id,
        )

    def _restart_identity(self, /) -> str:
        return canonical_fingerprint(
            {
                "kind": "distributed-periodic-les-restart",
                "backend": self.prepared_id,
                "owner": self.execution.owner_id,
                "precision": self.execution.precision.policy_id,
                "numerical": self.execution.numerical_id,
                "execution": self.execution.execution_id,
                "execution_plan": self.execution.plan_id,
                "payload_shapes": [
                    list(shape) for shape in self.execution.admitted_payload_shapes
                ],
                "transform_sequences": list(self.preparation.transform_sequence_ids),
                "topology": self.execution.topology.topology_id,
                "layout": self.execution.modal_layout.layout_id,
            }
        )

    def restart_evidence(
        self, state: ArrayLike, /
    ) -> DistributedPeriodicLESRestartEvidence:
        """Place one restart payload and bind its exact sharding identity."""
        value = self._validate_state(state, "Distributed restart velocity")
        diagnostics = self.execution.diagnostics(value)
        expected = self.execution.modal_layout.sharding(self.execution.topology)
        restart_id = self._restart_identity()
        return DistributedPeriodicLESRestartEvidence(
            state=value,
            finite=diagnostics.finite,
            sharding_preserved=value.sharding == expected,
            owner_id=self.execution.owner_id,
            precision_policy_id=self.execution.precision.policy_id,
            numerical_id=self.execution.numerical_id,
            execution_id=self.execution.execution_id,
            execution_plan_id=self.execution.plan_id,
            admitted_payload_shapes=self.execution.admitted_payload_shapes,
            transform_sequence_ids=self.preparation.transform_sequence_ids,
            topology_id=self.execution.topology.topology_id,
            layout_id=self.execution.modal_layout.layout_id,
            backend_id=self.prepared_id,
            restart_id=restart_id,
        )

    @checked
    def restore(self, evidence: DistributedPeriodicLESRestartEvidence, /) -> Array:
        """Restore only restart evidence produced for this exact backend identity."""
        if (
            evidence.backend_id != self.prepared_id
            or evidence.owner_id != self.execution.owner_id
            or evidence.precision_policy_id != self.execution.precision.policy_id
            or evidence.numerical_id != self.execution.numerical_id
            or evidence.execution_id != self.execution.execution_id
            or evidence.execution_plan_id != self.execution.plan_id
            or evidence.admitted_payload_shapes != self.execution.admitted_payload_shapes
            or evidence.transform_sequence_ids != self.preparation.transform_sequence_ids
            or evidence.topology_id != self.execution.topology.topology_id
            or evidence.layout_id != self.execution.modal_layout.layout_id
            or evidence.restart_id != self._restart_identity()
            or not evidence.sharding_preserved
        ):
            raise ValueError(
                "Restart evidence does not belong to this distributed backend."
            )
        return self._validate_state(evidence.state, "Distributed restored velocity")

    def parity_evidence(
        self,
        state: ArrayLike,
        /,
        *,
        absolute_tolerance: float = 1e-9,
        relative_tolerance: float = 1e-8,
    ) -> DistributedPeriodicLESParityEvidence:
        """Measure parity with the bound scientific action on a real one-device mesh."""
        if self.execution.topology.device_count != 1:
            raise ValueError("Scientific parity evidence requires a one-device topology.")
        absolute = float(absolute_tolerance)
        relative = float(relative_tolerance)
        if (
            not np.isfinite(absolute)
            or not np.isfinite(relative)
            or absolute < 0.0
            or relative < 0.0
        ):
            raise ValueError("Parity tolerances must be finite and nonnegative.")
        value = self._validate_state(state, "Distributed parity velocity")
        distributed = self.evaluate(value)
        reference: PeriodicAlgebraicLESStage = self.scientific.evaluate(value)
        rate_error = self.execution.diagnostics_batched(
            distributed.projected_rate - reference.projected_rate
        ).maximum_absolute
        stress_error = self.execution.diagnostics_batched(
            distributed.modal_deviatoric_specific_stress
            - reference.modal_deviatoric_specific_stress
        ).maximum_absolute
        dissipation_error = jnp.abs(
            distributed.modeled_dissipation - reference.modeled_dissipation
        )
        rate_scale = self.execution.diagnostics_batched(
            reference.projected_rate
        ).maximum_absolute
        stress_scale = self.execution.diagnostics_batched(
            reference.modal_deviatoric_specific_stress
        ).maximum_absolute
        dissipation_scale = jnp.abs(reference.modeled_dissipation)
        passed = (
            (rate_error <= absolute + relative * rate_scale)
            & (stress_error <= absolute + relative * stress_scale)
            & (dissipation_error <= absolute + relative * dissipation_scale)
        )
        finite = (
            jnp.isfinite(rate_error)
            & jnp.isfinite(stress_error)
            & jnp.isfinite(dissipation_error)
        )
        return DistributedPeriodicLESParityEvidence(
            projected_rate_maximum_error=rate_error,
            stress_maximum_error=stress_error,
            modeled_dissipation_error=dissipation_error,
            finite=finite,
            passed=passed & finite,
            absolute_tolerance=absolute,
            relative_tolerance=relative,
            scientific_prepared_id=self.scientific.prepared_id,
            backend_id=self.prepared_id,
            topology_id=self.execution.topology.topology_id,
            qualification_inherited=False,
        )


__all__ = [
    "DistributedPeriodicLESParityEvidence",
    "DistributedPeriodicLESPlan",
    "DistributedPeriodicLESPreparationEvidence",
    "DistributedPeriodicLESRestartEvidence",
    "DistributedPeriodicLESSchedule",
    "DistributedPeriodicLESStage",
    "DistributedPeriodicLESStepRestriction",
    "PreparedDistributedPeriodicFourierFilter",
    "PreparedDistributedPeriodicLES",
]
