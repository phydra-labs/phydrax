"""Production run of one fuel rod with durable restart and retained outputs."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import PyTree
from numpy.typing import NDArray

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...solver import (
    ByteBoundedAsyncPublisher,
    CheckpointGenerationPolicy,
    DurableCheckpointStore,
    ExactTimeSchedule,
    IncrementLimitedStepController,
    PreparedProductionRun,
    ProductionCaseManifest,
    ProductionRunPlan,
    ProductionRunState,
    RetriedFixedStepResult,
    RobustRetryPolicy,
)
from ._acceptance import FuelAcceptancePolicy, FuelStepEvidence, physical_increments
from ._bundles import FuelModelBundle
from ._coolant import PrescribedWallTemperature
from ._problem import PreparedFuelRod
from ._result import FuelRodResult
from ._state import FuelRodState
from ._step import FuelStepMethod


if TYPE_CHECKING:
    from ._execution import FuelNumericalExecution


# The default maximum step is this multiple of the nominal `time_step`. The
# default minimum step, `time_step / 2**maximum_retries`, is what the retry
# policy's successive halvings reach.
_DEFAULT_MAXIMUM_STEP_FACTOR = 4
# Durable checkpoint generations kept on disk: the latest plus one fallback.
_CHECKPOINT_GENERATIONS = 2
# Published states awaiting the asynchronous writer before the run blocks.
_MAXIMUM_PENDING_PUBLICATIONS = 2


def _increments(
    previous: PyTree[Array],
    candidate: PyTree[Array],
    _dt: Array,
    _args: FuelModelBundle | None,
    /,
) -> Array:
    if not isinstance(previous, FuelRodState) or not isinstance(candidate, FuelRodState):
        raise TypeError(
            "Fuel increment controller requires complete accepted rod states."
        )
    return physical_increments(previous, candidate)


def _output_times(prepared: PreparedFuelRod, /) -> NDArray[np.float64]:
    """Irradiation-history breakpoints plus any prescribed wall-temperature times."""
    history = prepared.admission.history
    targets = np.asarray(history.times)[1:]
    coolant = prepared.admission.coolant
    if isinstance(coolant, PrescribedWallTemperature):
        targets = np.unique(np.concatenate((targets, np.asarray(coolant.times))))
        targets = targets[(targets > 0) & (targets <= history.end_time)]
    return targets


def _step_controller(
    policy: FuelAcceptancePolicy,
    /,
    *,
    time_step: float,
    minimum_time_step: float | None,
    maximum_time_step: float | None,
    maximum_retries: int,
) -> IncrementLimitedStepController:
    # Limits follow the `physical_increments` order.
    limits = np.asarray(
        (
            policy.temperature_increment,
            policy.burnup_increment,
            policy.release_fraction_increment,
            policy.creep_increment,
        ),
        dtype=np.float64,
    )
    minimum = (
        time_step / 2**maximum_retries if minimum_time_step is None else minimum_time_step
    )
    maximum = (
        time_step * _DEFAULT_MAXIMUM_STEP_FACTOR
        if maximum_time_step is None
        else maximum_time_step
    )
    return IncrementLimitedStepController(
        _increments,
        limits,
        minimum_step=minimum,
        maximum_step=maximum,
        controller_id=policy.policy_id,
    )


class FuelRodSimulation:
    """Production run of one prepared rod with durable checkpoints.

    `ProductionRunPlan` advances and commits every accepted state. The publisher
    retains immutable snapshots at output times and the attempt observer records
    the evidence of every executed retry attempt; neither reruns the model. An
    irradiated rod can only resume from an exact checkpoint.
    """

    def __init__(
        self,
        prepared: PreparedFuelRod,
        /,
        *,
        time_step: float,
        checkpoint_directory: str | Path,
        maximum_steps: int = 10000,
        checkpoint_interval: int = 1,
        segment_steps: int = 1,
        acceptance_policy: FuelAcceptancePolicy | None = None,
        minimum_time_step: float | None = None,
        maximum_time_step: float | None = None,
        maximum_retries: int = 8,
        numerical_execution: FuelNumericalExecution | None = None,
    ) -> None:
        if not isinstance(prepared, PreparedFuelRod):
            raise TypeError("FuelRodSimulation requires a prepared physical problem.")
        policy = (
            FuelAcceptancePolicy() if acceptance_policy is None else acceptance_policy
        )
        self.prepared, self.maximum_steps = prepared, maximum_steps
        self.checkpoint_directory = Path(checkpoint_directory)
        self._states: list[FuelRodState] = []
        self._attempts: list[FuelStepEvidence] = []
        self.publications: dict[str, PyTree[np.ndarray]] = {}
        self.method = FuelStepMethod(
            prepared, policy=policy, numerical_execution=numerical_execution
        )
        history = prepared.admission.history
        targets = _output_times(prepared)
        controller = _step_controller(
            policy,
            time_step=time_step,
            minimum_time_step=minimum_time_step,
            maximum_time_step=maximum_time_step,
            maximum_retries=maximum_retries,
        )
        self.plan = ProductionRunPlan(
            self.method,
            RobustRetryPolicy(maximum_retries=maximum_retries),
            step_size=time_step,
            end_time=history.end_time,
            maximum_steps=maximum_steps,
            checkpoint_interval=checkpoint_interval,
            segment_steps=segment_steps,
            output_schedule=ExactTimeSchedule(targets),
            step_controller=controller,
            device_resident=True,
        )
        self.manifest = ProductionCaseManifest(
            problem_id=prepared.prepared_id,
            method_id=self.method.method_id,
            precision_id="fuel-float64",
            topology_id=prepared.geometry.geometry_id,
            geometry_layout_id=canonical_fingerprint(prepared.execution_signature),
            dtype="float64",
        )
        self.store = DurableCheckpointStore(
            self.checkpoint_directory,
            self.manifest,
            CheckpointGenerationPolicy(_CHECKPOINT_GENERATIONS),
        )
        initial = prepared.initial_state()
        state_bytes = sum(leaf.nbytes for leaf in jax.tree.leaves(initial))
        self.retained_trajectory_bytes = (maximum_steps + 1) * state_bytes
        self.retained_publication_bytes = targets.size * state_bytes
        self.publisher = ByteBoundedAsyncPublisher(
            self._publish_state,
            maximum_pending=_MAXIMUM_PENDING_PUBLICATIONS,
            maximum_pending_bytes=max(_MAXIMUM_PENDING_PUBLICATIONS * state_bytes, 1),
        )
        self.runtime = self._bind_runtime(prepared.admission.bundle)

    def _publish_state(self, event_id: str, state: PyTree[Array], /) -> None:
        if not isinstance(state, FuelRodState):
            raise TypeError("Native fuel publication requires a complete physical state.")
        self.publications[event_id] = state

    def _observe_attempts(
        self,
        transition: RetriedFixedStepResult,
        _previous: PyTree[Array],
        _step_index: Array,
        _time: Array,
        /,
    ) -> None:
        mask = np.asarray(transition.attempt_executed, dtype=np.bool_)
        evidence = transition.attempt_evidence
        if evidence is None:
            raise ValueError("Native fuel retry omitted its declared attempt evidence.")
        for attempt in np.flatnonzero(mask):
            item = jax.tree.map(lambda value: jnp.asarray(value[attempt]), evidence)
            if not isinstance(item, FuelStepEvidence):
                raise TypeError(
                    "Native fuel attempt evidence has the wrong scientific identity."
                )
            self._attempts.append(item)
        if bool(np.asarray(transition.successful)):
            accepted = transition.accepted_state
            if not isinstance(accepted, FuelRodState):
                raise TypeError(
                    "Native accepted transition omitted its physical rod state."
                )
            self._states.append(accepted)

    def _bind_runtime(self, bundle: FuelModelBundle, /) -> PreparedProductionRun:
        args_id = canonical_fingerprint(
            {
                "bundle": bundle.bundle_id,
                "numeric": array_tree_fingerprint(tuple(jax.tree.leaves(bundle))),
            }
        )
        return PreparedProductionRun(
            self.manifest,
            self.plan,
            self.store,
            args=bundle,
            args_id=args_id,
            publisher=self.publisher,
            attempt_observer=self._observe_attempts,
            attempt_observer_id="fuel-retain-native-attempt-evidence",
        )

    def run(
        self,
        initial_state: FuelRodState | None = None,
        /,
        *,
        restart: bool = False,
        parameters: FuelModelBundle | None = None,
    ) -> FuelRodResult:
        bundle = self.prepared.admission.bundle if parameters is None else parameters
        if parameters is not None:
            self.runtime = self._bind_runtime(bundle)
        template = self.runtime.initial_state(
            self.prepared.initial_state(parameters=bundle)
        )
        start = self._start_state(template, initial_state, restart=restart)
        if not isinstance(start.accepted_state, FuelRodState):
            raise TypeError(
                "Restored checkpoint omitted the complete fuel physical state."
            )
        self._states = [start.accepted_state]
        self._attempts = []
        result = self.runtime.run(start)
        final = result.state.accepted_state
        if not isinstance(final, FuelRodState):
            raise TypeError(
                "Native production result omitted the complete physical state."
            )
        if float(np.asarray(final.time)) > float(np.asarray(self._states[-1].time)):
            self._states.append(final)
        return FuelRodResult(
            tuple(self._states),
            tuple(self._attempts),
            result,
            self.prepared.prepared_id,
            bundle_id=bundle.bundle_id,
            history_id=self.prepared.admission.history.history_id,
        )

    def _start_state(
        self,
        template: ProductionRunState,
        initial_state: FuelRodState | None,
        /,
        *,
        restart: bool,
    ) -> ProductionRunState:
        """Resume from the latest checkpoint or start from a fresh rod state."""
        if restart:
            if initial_state is not None:
                raise ValueError(
                    "Restart uses an exact checkpoint, not a second initialization."
                )
            return self.runtime.resume(template)
        if initial_state is None:
            return template
        if (
            initial_state.prepared_id != self.prepared.prepared_id
            or float(np.asarray(initial_state.time)) != 0
        ):
            raise ValueError(
                "Irradiated rods require exact native checkpoint restart, not burnup initialization."
            )
        irradiated = (
            jnp.any(initial_state.exposure.fission_density > 0)
            | jnp.any(initial_state.exposure.fast_fluence > 0)
            | jnp.any(initial_state.gas.produced_moles > 0)
        )
        if bool(irradiated):
            raise ValueError(
                "An irradiated initial state requires a complete accepted native checkpoint."
            )
        return self.runtime.initial_state(initial_state)

    def close(self, /) -> None:
        self.publisher.close()
        self.store.close()
