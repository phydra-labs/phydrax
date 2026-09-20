#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..lattice_boltzmann._checkpoint import (
    KineticCheckpointPlan,
    read_kinetic_checkpoint,
    write_kinetic_checkpoint,
)
from ._smooth_compressible import SmoothCompressibleKineticState
from ._spatial import PreparedSmoothCompressibleD2V17SpatialDynamics


class _SmoothCompressibleD2VAcceptedPayload(StrictModule):
    accepted_state: SmoothCompressibleKineticState
    boundary_history: tuple[Array, ...]
    source_history: tuple[Array, ...]


class SmoothCompressibleD2VCheckpointPlan(StrictModule, NonTrainableState):
    """Exact restart identity and retained-history inventory for one D2V runtime."""

    kinetic_plan: KineticCheckpointPlan
    runtime_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    frozen_artifact_id: str = eqx.field(static=True)
    numeric_revision_id: str = eqx.field(static=True)
    boundary_history_names: tuple[str, ...] = eqx.field(static=True)
    source_history_names: tuple[str, ...] = eqx.field(static=True)
    population_shape: tuple[int, ...] = eqx.field(static=True)
    population_dtype: str = eqx.field(static=True)
    population_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
        topology_id: str,
        support_id: str,
        frozen_artifact_id: str,
        numeric_revision_id: str,
        /,
        *,
        boundary_history_names: Sequence[str] = (),
        source_history_names: Sequence[str] = (),
    ):
        if not isinstance(dynamics, PreparedSmoothCompressibleD2V17SpatialDynamics):
            raise TypeError(
                "dynamics must be PreparedSmoothCompressibleD2V17SpatialDynamics."
            )
        identities = tuple(
            str(value).strip()
            for value in (
                dynamics.prepared_id,
                topology_id,
                dynamics.method.method_id,
                support_id,
                frozen_artifact_id,
                numeric_revision_id,
            )
        )
        if any(not value for value in identities):
            raise ValueError("D2V checkpoint identities must be non-empty.")
        boundary_names = tuple(str(name).strip() for name in boundary_history_names)
        source_names = tuple(str(name).strip() for name in source_history_names)
        all_names = (*boundary_names, *source_names)
        if (
            any(not name for name in all_names)
            or len(set(boundary_names)) != len(boundary_names)
            or len(set(source_names)) != len(source_names)
            or set(boundary_names) & set(source_names)
        ):
            raise ValueError(
                "D2V boundary/source history names must be non-empty, unique, and disjoint."
            )
        runtime, topology, method, support, artifact, revision = identities
        replay_identity = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v-frozen-runtime",
                "support": support,
                "frozen_artifact": artifact,
                "numeric_revision": revision,
                "boundary_history": boundary_names,
                "source_history": source_names,
            }
        )
        kinetic_plan = KineticCheckpointPlan(
            runtime,
            dynamics.program_manifest,
            geometry_epoch_id=method,
            topology_id=topology,
            execution_id=support,
            replay_policy_id=replay_identity,
        )
        population_shape = (
            *dynamics.transport.spatial_shape,
            dynamics.method.quadrature.population_count,
        )
        population_dtype = np.dtype(dynamics.method.quadrature.velocities.dtype).str
        population_floor = dynamics.population_floor
        self.kinetic_plan = kinetic_plan
        self.runtime_id = runtime
        self.topology_id = topology
        self.method_id = method
        self.support_id = support
        self.frozen_artifact_id = artifact
        self.numeric_revision_id = revision
        self.boundary_history_names = boundary_names
        self.source_history_names = source_names
        self.population_shape = population_shape
        self.population_dtype = population_dtype
        self.population_floor = population_floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v-accepted-checkpoint-plan",
                "kinetic_plan": kinetic_plan.plan_id,
                "runtime": runtime,
                "topology": topology,
                "method": method,
                "support": support,
                "frozen_artifact": artifact,
                "numeric_revision": revision,
                "boundary_history": boundary_names,
                "source_history": source_names,
                "population_shape": population_shape,
                "population_dtype": population_dtype,
                "population_floor": population_floor,
            }
        )


class SmoothCompressibleD2VCheckpoint(StrictModule):
    """Accepted population state and the minimal causal history needed to continue."""

    time: Array
    step_index: Array
    accepted_state: SmoothCompressibleKineticState
    boundary_history: tuple[Array, ...]
    source_history: tuple[Array, ...]
    boundary_history_names: tuple[str, ...] = eqx.field(static=True)
    source_history_names: tuple[str, ...] = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def boundary_value(self, name: str, /) -> Array:
        if name not in self.boundary_history_names:
            raise KeyError(f"D2V checkpoint has no boundary history {name!r}.")
        return self.boundary_history[self.boundary_history_names.index(name)]

    def source_value(self, name: str, /) -> Array:
        if name not in self.source_history_names:
            raise KeyError(f"D2V checkpoint has no source history {name!r}.")
        return self.source_history[self.source_history_names.index(name)]


def _validate_accepted_state(
    plan: SmoothCompressibleD2VCheckpointPlan,
    state: SmoothCompressibleKineticState,
    /,
) -> None:
    if not isinstance(state, SmoothCompressibleKineticState):
        raise TypeError("accepted_state must be SmoothCompressibleKineticState.")
    fields = (state.particle_populations, state.total_energy_populations)
    if any(tuple(field.shape) != plan.population_shape for field in fields):
        raise ValueError("D2V checkpoint population shape does not match the runtime.")
    if any(np.dtype(field.dtype).str != plan.population_dtype for field in fields):
        raise TypeError("D2V checkpoint population dtype does not match the runtime.")
    if any(not bool(jnp.all(jnp.isfinite(field))) for field in fields):
        raise ValueError("D2V checkpoint accepted populations must be finite.")
    if any(not bool(jnp.all(field >= plan.population_floor)) for field in fields):
        raise ValueError(
            "D2V checkpoint populations violate the runtime acceptance floor."
        )


def _ordered_history(
    values: Mapping[str, ArrayLike] | None,
    names: tuple[str, ...],
    kind: str,
    /,
) -> tuple[Array, ...]:
    supplied = {} if values is None else dict(values)
    if set(supplied) != set(names):
        raise ValueError(
            f"D2V checkpoint {kind} history must exactly match the prepared inventory."
        )
    arrays = tuple(jnp.asarray(supplied[name]) for name in names)
    if any(np.dtype(array.dtype).kind not in "biufc" for array in arrays):
        raise TypeError(f"D2V checkpoint {kind} history must be numeric arrays.")
    if any(
        jnp.issubdtype(array.dtype, jnp.inexact)
        and not bool(jnp.all(jnp.isfinite(array)))
        for array in arrays
    ):
        raise ValueError(f"D2V checkpoint {kind} history must be finite.")
    return arrays


def _require_accepted_boundary(accepted: ArrayLike, /) -> None:
    value = jnp.asarray(accepted)
    if value.shape != () or value.dtype != jnp.dtype(jnp.bool_):
        raise TypeError("accepted must be one boolean scalar.")
    if not bool(value):
        raise ValueError("D2V checkpoints may only be written at an accepted boundary.")


def _checkpoint(
    plan: SmoothCompressibleD2VCheckpointPlan,
    time: Array,
    step_index: Array,
    payload: _SmoothCompressibleD2VAcceptedPayload,
    payload_id: str,
    /,
) -> SmoothCompressibleD2VCheckpoint:
    return SmoothCompressibleD2VCheckpoint(
        time=time,
        step_index=step_index,
        accepted_state=payload.accepted_state,
        boundary_history=payload.boundary_history,
        source_history=payload.source_history,
        boundary_history_names=plan.boundary_history_names,
        source_history_names=plan.source_history_names,
        payload_id=payload_id,
        plan_id=plan.plan_id,
    )


def write_smooth_compressible_d2v_checkpoint(
    path: str | Path,
    plan: SmoothCompressibleD2VCheckpointPlan,
    time: ArrayLike,
    step_index: ArrayLike,
    accepted_state: SmoothCompressibleKineticState,
    /,
    *,
    accepted: ArrayLike,
    boundary_history: Mapping[str, ArrayLike] | None = None,
    source_history: Mapping[str, ArrayLike] | None = None,
) -> SmoothCompressibleD2VCheckpoint:
    """Write only a committed D2V state; model weights and derived moments stay external."""

    if not isinstance(plan, SmoothCompressibleD2VCheckpointPlan):
        raise TypeError("plan must be SmoothCompressibleD2VCheckpointPlan.")
    _require_accepted_boundary(accepted)
    _validate_accepted_state(plan, accepted_state)
    payload = _SmoothCompressibleD2VAcceptedPayload(
        accepted_state,
        _ordered_history(boundary_history, plan.boundary_history_names, "boundary"),
        _ordered_history(source_history, plan.source_history_names, "source"),
    )
    written = write_kinetic_checkpoint(
        path,
        plan.kinetic_plan,
        time,
        step_index,
        payload,
    )
    return _checkpoint(
        plan, written.time, written.step_index, written.state, written.payload_id
    )


def read_smooth_compressible_d2v_checkpoint(
    path: str | Path,
    plan: SmoothCompressibleD2VCheckpointPlan,
    accepted_state_template: SmoothCompressibleKineticState,
    /,
    *,
    boundary_history_template: Mapping[str, ArrayLike] | None = None,
    source_history_template: Mapping[str, ArrayLike] | None = None,
) -> SmoothCompressibleD2VCheckpoint:
    """Read a checkpoint only against the exact runtime, model, ABI, and history layout."""

    if not isinstance(plan, SmoothCompressibleD2VCheckpointPlan):
        raise TypeError("plan must be SmoothCompressibleD2VCheckpointPlan.")
    _validate_accepted_state(plan, accepted_state_template)
    template = _SmoothCompressibleD2VAcceptedPayload(
        accepted_state_template,
        _ordered_history(
            boundary_history_template, plan.boundary_history_names, "boundary"
        ),
        _ordered_history(source_history_template, plan.source_history_names, "source"),
    )
    restored = read_kinetic_checkpoint(path, plan.kinetic_plan, template)
    payload = restored.state
    if not isinstance(payload, _SmoothCompressibleD2VAcceptedPayload):
        raise TypeError("Kinetic archive did not restore a D2V accepted-state payload.")
    _validate_accepted_state(plan, payload.accepted_state)
    return _checkpoint(
        plan,
        restored.time,
        restored.step_index,
        payload,
        restored.payload_id,
    )


__all__ = [
    "SmoothCompressibleD2VCheckpoint",
    "SmoothCompressibleD2VCheckpointPlan",
    "read_smooth_compressible_d2v_checkpoint",
    "write_smooth_compressible_d2v_checkpoint",
]
