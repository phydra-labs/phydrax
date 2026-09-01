#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations import (
    MaterialSiteId,
    MaterialState,
    MaterialTransaction,
)
from ._finite_element_checkpoint import FiniteElementCheckpoint
from ._schedule import TimeLaw


class FiniteElementAcceptedState(StrictModule, NonTrainableState):
    """Immutable fields and committed material data at one accepted step."""

    fields: tuple[Array, ...]
    materials: MaterialTransaction | None
    time: Array
    step: int = eqx.field(static=True)
    schedule_cursor: int = eqx.field(static=True)
    state_version: int = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    accepted_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: Sequence[ArrayLike],
        time: ArrayLike,
        step: int,
        topology_id: str,
        prepared_id: str,
        compilation_id: str,
        /,
        *,
        materials: MaterialTransaction | None = None,
        schedule_cursor: int = 0,
        state_version: int = 0,
    ):
        fields_ = tuple(jnp.asarray(value) for value in fields)
        time_ = jnp.asarray(time)
        step_ = int(step)
        cursor = int(schedule_cursor)
        version = int(state_version)
        topology = str(topology_id)
        prepared = str(prepared_id)
        compilation = str(compilation_id)
        if not fields_ or any(
            not jnp.issubdtype(value.dtype, jnp.inexact) for value in fields_
        ):
            raise ValueError("Accepted FE state requires one or more inexact fields.")
        if time_.shape != () or not bool(jnp.isfinite(time_)):
            raise ValueError("Accepted FE time must be one finite scalar.")
        if step_ < 0 or cursor < 0 or version < 0:
            raise ValueError("Accepted FE step, cursor, and version must be nonnegative.")
        if not topology or not prepared or not compilation:
            raise ValueError("Accepted FE identities must be non-empty.")
        if materials is not None and not isinstance(materials, MaterialTransaction):
            raise TypeError("materials must be FiniteElementMaterialTransaction or None.")
        self.fields = fields_
        self.materials = materials
        self.time = time_
        self.step = step_
        self.schedule_cursor = cursor
        self.state_version = version
        self.topology_id = topology
        self.prepared_id = prepared
        self.compilation_id = compilation
        self.accepted_id = canonical_fingerprint(
            {
                "kind": "finite-element-accepted-state",
                "topology": topology,
                "prepared": prepared,
                "compilation": compilation,
                "step": step_,
                "cursor": cursor,
                "version": version,
                "field_shapes": [list(value.shape) for value in fields_],
                "materials": (None if materials is None else materials.transaction_id),
            }
        )

    def checkpoint(self, /) -> FiniteElementCheckpoint:
        material_states = () if self.materials is None else self.materials.states
        return FiniteElementCheckpoint(
            self.prepared_id,
            self.compilation_id,
            self.time,
            self.step,
            self.fields,
            material_states=material_states,
        )


class FiniteElementAttemptResult(StrictModule):
    """Candidate state and explicit accept/retry evidence for one attempt."""

    fields: tuple[Array, ...]
    materials: MaterialTransaction | None
    accepted: Array
    retry_requested: Array
    suggested_step: Array
    diagnostics: object

    def __init__(
        self,
        fields: Sequence[ArrayLike],
        accepted: ArrayLike,
        /,
        *,
        materials: MaterialTransaction | None = None,
        retry_requested: ArrayLike = False,
        suggested_step: ArrayLike = 0.0,
        diagnostics: object = None,
    ):
        fields_ = tuple(jnp.asarray(value) for value in fields)
        accepted_ = jnp.asarray(accepted, dtype=bool)
        retry = jnp.asarray(retry_requested, dtype=bool)
        suggested = jnp.asarray(suggested_step)
        if not fields_:
            raise ValueError("Finite-element attempts require candidate fields.")
        if accepted_.shape != () or retry.shape != () or suggested.shape != ():
            raise ValueError("Attempt decision and suggested step must be scalars.")
        if materials is not None and not isinstance(materials, MaterialTransaction):
            raise TypeError("materials must be FiniteElementMaterialTransaction or None.")
        self.fields = fields_
        self.materials = materials
        self.accepted = accepted_
        self.retry_requested = retry
        self.suggested_step = suggested
        self.diagnostics = diagnostics


class FiniteElementStepPolicy(StrictModule, NonTrainableState):
    minimum_step: float = eqx.field(static=True)
    reduction: float = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        minimum_step: float = 1.0e-8,
        reduction: float = 0.5,
        maximum_retries: int = 8,
    ):
        minimum = float(minimum_step)
        reduction_ = float(reduction)
        retries = int(maximum_retries)
        if minimum <= 0.0 or not 0.0 < reduction_ < 1.0 or retries < 0:
            raise ValueError("Accepted-step retry policy is invalid.")
        self.minimum_step = minimum
        self.reduction = reduction_
        self.maximum_retries = retries
        self.policy_id = canonical_fingerprint(
            {
                "kind": "finite-element-step-policy",
                "minimum_step": minimum,
                "reduction": reduction_,
                "maximum_retries": retries,
            }
        )


class FiniteElementStepDiagnostics(StrictModule):
    accepted: Array
    attempts: Array
    rejected_attempts: Array
    attempted_step: Array
    accepted_step: Array
    terminal_reason: str = eqx.field(static=True)


class FiniteElementAcceptedStepSchedule(StrictModule, NonTrainableState):
    """Host transaction that promotes only an explicitly accepted FE candidate."""

    solve_attempt: Callable
    policy: FiniteElementStepPolicy
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        solve_attempt: Callable,
        /,
        *,
        policy: FiniteElementStepPolicy | None = None,
        schedule_id: str = "finite-element-accepted-step",
    ):
        if not callable(solve_attempt):
            raise TypeError("solve_attempt must be callable.")
        policy_ = FiniteElementStepPolicy() if policy is None else policy
        if not isinstance(policy_, FiniteElementStepPolicy):
            raise TypeError("policy must be FiniteElementStepPolicy or None.")
        identifier = str(schedule_id)
        if not identifier:
            raise ValueError("schedule_id must be non-empty.")
        self.solve_attempt = solve_attempt
        self.policy = policy_
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "finite-element-accepted-step-schedule",
                "declared_id": identifier,
                "policy": policy_.policy_id,
            }
        )

    def advance(
        self,
        accepted: FiniteElementAcceptedState,
        end_time: ArrayLike,
        time_law: TimeLaw,
        args: object = None,
        /,
    ) -> tuple[FiniteElementAcceptedState, FiniteElementStepDiagnostics]:
        if not isinstance(accepted, FiniteElementAcceptedState):
            raise TypeError("accepted must be FiniteElementAcceptedState.")
        if not isinstance(time_law, TimeLaw):
            raise TypeError("time_law must be TimeLaw.")
        target = float(jnp.asarray(end_time))
        start = float(accepted.time)
        if not jnp.isfinite(target) or target <= start:
            raise ValueError("Accepted-step target time must be finite and increasing.")
        attempted_step = target - start
        step_size = attempted_step
        rejected = 0
        for attempt in range(self.policy.maximum_retries + 1):
            candidate_end = start + step_size
            result = self.solve_attempt(
                accepted,
                start,
                candidate_end,
                time_law,
                args,
            )
            if not isinstance(result, FiniteElementAttemptResult):
                raise TypeError("solve_attempt must return FiniteElementAttemptResult.")
            if tuple(value.shape for value in result.fields) != tuple(
                value.shape for value in accepted.fields
            ):
                raise ValueError("Candidate fields must preserve accepted field shapes.")
            if bool(result.accepted):
                materials = (
                    None if result.materials is None else result.materials.commit()
                )
                promoted = FiniteElementAcceptedState(
                    result.fields,
                    candidate_end,
                    accepted.step + 1,
                    accepted.topology_id,
                    accepted.prepared_id,
                    accepted.compilation_id,
                    materials=materials,
                    schedule_cursor=accepted.schedule_cursor + 1,
                    state_version=accepted.state_version + 1,
                )
                return promoted, FiniteElementStepDiagnostics(
                    accepted=jnp.asarray(True),
                    attempts=jnp.asarray(attempt + 1, dtype=jnp.int32),
                    rejected_attempts=jnp.asarray(rejected, dtype=jnp.int32),
                    attempted_step=jnp.asarray(attempted_step),
                    accepted_step=jnp.asarray(step_size),
                    terminal_reason="accepted",
                )
            rejected += 1
            if not bool(result.retry_requested):
                break
            suggested = float(result.suggested_step)
            step_size = (
                suggested
                if jnp.isfinite(suggested) and 0.0 < suggested < step_size
                else self.policy.reduction * step_size
            )
            if step_size < self.policy.minimum_step:
                break
        return accepted, FiniteElementStepDiagnostics(
            accepted=jnp.asarray(False),
            attempts=jnp.asarray(rejected, dtype=jnp.int32),
            rejected_attempts=jnp.asarray(rejected, dtype=jnp.int32),
            attempted_step=jnp.asarray(attempted_step),
            accepted_step=jnp.asarray(0.0),
            terminal_reason="rejected",
        )


class FiniteElementRestartManifest(StrictModule, NonTrainableState):
    """Versioned fixed-topology accepted state plus named runtime histories."""

    state: FiniteElementAcceptedState
    auxiliary_state: tuple[tuple[str, Array], ...]
    integrator_state: tuple[tuple[str, Array], ...]
    schema_version: int = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        state: FiniteElementAcceptedState,
        /,
        *,
        auxiliary_state: Sequence[tuple[str, ArrayLike]] = (),
        integrator_state: Sequence[tuple[str, ArrayLike]] = (),
        schema_version: int = 1,
    ):
        if not isinstance(state, FiniteElementAcceptedState):
            raise TypeError("state must be FiniteElementAcceptedState.")
        version = int(schema_version)
        if version != 1:
            raise ValueError("Unsupported finite-element restart schema version.")

        def named(values):
            result = tuple((str(name), jnp.asarray(value)) for name, value in values)
            names = tuple(name for name, _ in result)
            if len(set(names)) != len(names) or any(not name for name in names):
                raise ValueError("Restart state names must be non-empty and unique.")
            return result

        auxiliary = named(auxiliary_state)
        integrator = named(integrator_state)
        self.state = state
        self.auxiliary_state = auxiliary
        self.integrator_state = integrator
        self.schema_version = version
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "finite-element-restart-manifest",
                "schema_version": version,
                "accepted": state.accepted_id,
                "auxiliary": [
                    [name, list(value.shape), str(value.dtype)]
                    for name, value in auxiliary
                ],
                "integrator": [
                    [name, list(value.shape), str(value.dtype)]
                    for name, value in integrator
                ],
            }
        )


def write_finite_element_restart(
    path: str | Path,
    manifest: FiniteElementRestartManifest,
    /,
) -> None:
    if not isinstance(manifest, FiniteElementRestartManifest):
        raise TypeError("manifest must be FiniteElementRestartManifest.")
    target = Path(path)
    material_states = (
        () if manifest.state.materials is None else manifest.state.materials.states
    )
    metadata = {
        "schema_version": manifest.schema_version,
        "manifest_id": manifest.manifest_id,
        "topology_id": manifest.state.topology_id,
        "prepared_id": manifest.state.prepared_id,
        "compilation_id": manifest.state.compilation_id,
        "step": manifest.state.step,
        "schedule_cursor": manifest.state.schedule_cursor,
        "state_version": manifest.state.state_version,
        "field_count": len(manifest.state.fields),
        "materials": [
            [value.site_id.key, value.model_id, value.state_version]
            for value in material_states
        ],
        "auxiliary_names": [name for name, _ in manifest.auxiliary_state],
        "integrator_names": [name for name, _ in manifest.integrator_state],
    }
    arrays = {
        "time": np.asarray(manifest.state.time),
        **{
            f"field_{index}": np.asarray(value)
            for index, value in enumerate(manifest.state.fields)
        },
        **{
            f"material_{index}": np.asarray(value.committed)
            for index, value in enumerate(material_states)
        },
        **{
            f"auxiliary_{index}": np.asarray(value)
            for index, (_, value) in enumerate(manifest.auxiliary_state)
        },
        **{
            f"integrator_{index}": np.asarray(value)
            for index, (_, value) in enumerate(manifest.integrator_state)
        },
    }
    np.savez(target, metadata=np.asarray(json.dumps(metadata)), **arrays)


def read_finite_element_restart(
    path: str | Path,
    /,
) -> FiniteElementRestartManifest:
    archive = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))
    if int(metadata["schema_version"]) != 1:
        raise ValueError("Unsupported finite-element restart schema version.")
    fields = tuple(
        archive[f"field_{index}"] for index in range(int(metadata["field_count"]))
    )
    material_states_: list[MaterialState] = []
    for index, entry in enumerate(metadata["materials"]):
        if len(entry) == 2:
            material_id, version = entry
            model_id = material_id
        elif len(entry) == 3:
            material_id, model_id, version = entry
        else:
            raise ValueError("Finite-element material restart metadata is invalid.")
        material_states_.append(
            MaterialState(
                MaterialSiteId(material_id),
                model_id,
                archive[f"material_{index}"],
                state_version=int(version),
            )
        )
    material_states = tuple(material_states_)
    materials = None if not material_states else MaterialTransaction(material_states)
    state = FiniteElementAcceptedState(
        fields,
        archive["time"],
        int(metadata["step"]),
        metadata["topology_id"],
        metadata["prepared_id"],
        metadata["compilation_id"],
        materials=materials,
        schedule_cursor=int(metadata["schedule_cursor"]),
        state_version=int(metadata["state_version"]),
    )
    auxiliary = tuple(
        (name, archive[f"auxiliary_{index}"])
        for index, name in enumerate(metadata["auxiliary_names"])
    )
    integrator = tuple(
        (name, archive[f"integrator_{index}"])
        for index, name in enumerate(metadata["integrator_names"])
    )
    manifest = FiniteElementRestartManifest(
        state,
        auxiliary_state=auxiliary,
        integrator_state=integrator,
        schema_version=1,
    )
    if manifest.manifest_id != metadata["manifest_id"]:
        raise ValueError("Finite-element restart manifest identity mismatch.")
    return manifest


__all__ = [
    "FiniteElementAcceptedState",
    "FiniteElementAcceptedStepSchedule",
    "FiniteElementAttemptResult",
    "FiniteElementRestartManifest",
    "FiniteElementStepDiagnostics",
    "FiniteElementStepPolicy",
    "read_finite_element_restart",
    "write_finite_element_restart",
]
