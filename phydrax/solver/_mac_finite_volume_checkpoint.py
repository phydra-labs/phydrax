#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from .._array_archive import (
    ArrayArchiveCorruptionError,
    read_array_archive,
    write_array_archive,
)
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._mac_adaptive import MACAdaptiveRolloutPlan, MACAdaptiveRuntimeState


_RUNTIME_ARRAYS = (
    "state",
    "time",
    "accepted_step_count",
    "requested_next_step",
    "status",
    "retry_count",
    "grid_times",
    "grid_step_sizes",
    "grid_valid_steps",
    "output_cursor",
    "forcing_state",
)


class MACFiniteVolumeCheckpoint(StrictModule):
    runtime_state: MACAdaptiveRuntimeState
    checkpoint_plan_id: str = eqx.field(static=True)


class MACFiniteVolumeCheckpointPlan(StrictModule, NonTrainableState):
    """Strict archive identity and exact runtime leaf template for MAC restart."""

    adaptive: MACAdaptiveRolloutPlan
    template: MACAdaptiveRuntimeState
    dynamics_id: str = eqx.field(static=True)
    method_id: str = eqx.field(static=True)
    controller_id: str = eqx.field(static=True)
    grid_id: str = eqx.field(static=True)
    precision_dtype: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self, adaptive: MACAdaptiveRolloutPlan, template: MACAdaptiveRuntimeState, /
    ):
        if not isinstance(adaptive, MACAdaptiveRolloutPlan) or not isinstance(
            template, MACAdaptiveRuntimeState
        ):
            raise TypeError(
                "MAC checkpoint plan requires adaptive plan/runtime template."
            )
        if (
            template.dynamics_id != adaptive.dynamics.compilation_id
            or template.method_id != adaptive.method.method_id
            or template.controller_id != adaptive.controller.controller_id
        ):
            raise ValueError("MAC checkpoint template identities differ from the plan.")
        discretization = adaptive.dynamics.momentum.operators.discretization
        dtype = np.dtype(adaptive.dynamics.momentum.operators.pressure_space.dtype).str
        self.adaptive, self.template = adaptive, template
        self.dynamics_id = adaptive.dynamics.compilation_id
        self.method_id = adaptive.method.method_id
        self.controller_id = adaptive.controller.controller_id
        self.grid_id = discretization.prepared_id
        self.precision_dtype = dtype
        template_arrays = _runtime_arrays(template)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mac-finite-volume-checkpoint-plan",
                "dynamics": self.dynamics_id,
                "method": self.method_id,
                "controller": self.controller_id,
                "grid": self.grid_id,
                "precision_dtype": dtype,
                "leaves": {
                    name: {
                        "shape": list(value.shape),
                        "dtype": value.dtype.str,
                    }
                    for name, value in template_arrays.items()
                },
            }
        )


def _runtime_arrays(runtime: MACAdaptiveRuntimeState, /) -> dict[str, np.ndarray]:
    return {
        "state": np.asarray(runtime.state),
        "time": np.asarray(runtime.time),
        "accepted_step_count": np.asarray(runtime.accepted_step_count),
        "requested_next_step": np.asarray(runtime.requested_next_step),
        "status": np.asarray(runtime.status),
        "retry_count": np.asarray(runtime.retry_count),
        "grid_times": np.asarray(runtime.grid_times),
        "grid_step_sizes": np.asarray(runtime.grid_step_sizes),
        "grid_valid_steps": np.asarray(runtime.grid_valid_steps),
        "output_cursor": np.asarray(runtime.output_cursor),
        "forcing_state": np.asarray(runtime.forcing_state),
    }


def write_mac_finite_volume_checkpoint(
    path: str | os.PathLike[str],
    plan: MACFiniteVolumeCheckpointPlan,
    runtime_state: MACAdaptiveRuntimeState,
    /,
) -> Path:
    if not isinstance(plan, MACFiniteVolumeCheckpointPlan) or not isinstance(
        runtime_state, MACAdaptiveRuntimeState
    ):
        raise TypeError("MAC checkpoint write requires a plan and runtime state.")
    if (
        runtime_state.dynamics_id != plan.dynamics_id
        or runtime_state.method_id != plan.method_id
        or runtime_state.controller_id != plan.controller_id
    ):
        raise ValueError("MAC checkpoint runtime identities differ from the plan.")
    arrays = _runtime_arrays(runtime_state)
    template = _runtime_arrays(plan.template)
    for name in _RUNTIME_ARRAYS:
        if (
            arrays[name].shape != template[name].shape
            or arrays[name].dtype != template[name].dtype
        ):
            raise ValueError(f"MAC checkpoint leaf {name!r} changed shape or dtype.")
    return write_array_archive(
        path,
        manifest={
            "kind": "mac-finite-volume-checkpoint",
            "checkpoint_plan_id": plan.plan_id,
            "dynamics_id": plan.dynamics_id,
            "method_id": plan.method_id,
            "controller_id": plan.controller_id,
            "grid_id": plan.grid_id,
            "precision_dtype": plan.precision_dtype,
        },
        arrays=arrays,
    )


def read_mac_finite_volume_checkpoint(
    path: str | os.PathLike[str], plan: MACFiniteVolumeCheckpointPlan, /
) -> MACFiniteVolumeCheckpoint:
    if not isinstance(plan, MACFiniteVolumeCheckpointPlan):
        raise TypeError("MAC checkpoint read requires MACFiniteVolumeCheckpointPlan.")
    manifest, arrays = read_array_archive(path)
    expected_manifest = {
        "kind",
        "checkpoint_plan_id",
        "dynamics_id",
        "method_id",
        "controller_id",
        "grid_id",
        "precision_dtype",
        "arrays",
    }
    if (
        set(manifest) != expected_manifest
        or manifest["kind"] != "mac-finite-volume-checkpoint"
    ):
        raise ArrayArchiveCorruptionError("Archive is not a canonical MAC FV checkpoint.")
    expected_identity = (
        plan.plan_id,
        plan.dynamics_id,
        plan.method_id,
        plan.controller_id,
        plan.grid_id,
        plan.precision_dtype,
    )
    observed_identity = (
        manifest["checkpoint_plan_id"],
        manifest["dynamics_id"],
        manifest["method_id"],
        manifest["controller_id"],
        manifest["grid_id"],
        manifest["precision_dtype"],
    )
    if observed_identity != expected_identity:
        raise ValueError("MAC checkpoint prepared identities do not match the caller.")
    if set(arrays) != set(_RUNTIME_ARRAYS):
        raise ArrayArchiveCorruptionError("MAC checkpoint runtime leaves are incomplete.")
    template = _runtime_arrays(plan.template)
    for name in _RUNTIME_ARRAYS:
        if (
            arrays[name].shape != template[name].shape
            or arrays[name].dtype != template[name].dtype
        ):
            raise ValueError(f"MAC checkpoint leaf {name!r} changed shape or dtype.")
    runtime = MACAdaptiveRuntimeState(
        **{name: jnp.asarray(arrays[name]) for name in _RUNTIME_ARRAYS},
        dynamics_id=plan.dynamics_id,
        method_id=plan.method_id,
        controller_id=plan.controller_id,
    )
    return MACFiniteVolumeCheckpoint(runtime, plan.plan_id)


__all__ = [
    "MACFiniteVolumeCheckpoint",
    "MACFiniteVolumeCheckpointPlan",
    "read_mac_finite_volume_checkpoint",
    "write_mac_finite_volume_checkpoint",
]
