#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from ..backends._types import BackendAvailability
from ..backends.iree import iree_availability
from ..closure_data._kinetic_equilibrium import (
    EnergyEquilibriumSupportEvidence,
    PreparedLearnedEnergyEquilibriumBinding,
)
from ..discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleKineticState,
)
from ..discretization.discrete_velocity._spatial import (
    PreparedSmoothCompressibleD2V17SpatialDynamics,
    SmoothCompressibleD2VStepStatus,
)
from ._iree import IREEExportPolicy, IREEExportResult, save_iree


DiscreteVelocityIREEExportMode: TypeAlias = Literal[
    "frozen-equilibrium", "one-step", "fixed-horizon"
]

_D2V_OUTPUT_NAMES = (
    "accepted_f",
    "accepted_g",
    "successful",
    "rollback_applied",
    "status",
    "first_failure_step",
    "maximum_conservation_residual",
    "maximum_energy_flux_error",
    "minimum_f",
    "minimum_g",
    "minimum_support_margin",
)


@dataclass(frozen=True, slots=True)
class DiscreteVelocityIREEContract:
    """Host-specific fixed ABI for one frozen learned D2V realization."""

    execution_mode: DiscreteVelocityIREEExportMode
    host_id: str
    target_backend: str
    runtime_driver: str
    runtime_id: str
    topology_id: str
    method_id: str
    support_id: str
    frozen_artifact_id: str
    numeric_revision_id: str
    input_names: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtypes: tuple[str, ...]
    output_names: tuple[str, ...]
    output_shapes: tuple[tuple[int, ...], ...]
    output_dtypes: tuple[str, ...]
    step_count: int
    step_size: float
    supports_reverse_mode: bool
    supports_training_export: bool
    contract_id: str

    def pack_inputs(self, *values: Array) -> tuple[Array, ...]:
        if len(values) != len(self.input_names):
            raise ValueError("D2V IREE input count does not match the contract.")
        packed: list[Array] = []
        for value, shape, dtype in zip(
            values, self.input_shapes, self.input_dtypes, strict=True
        ):
            if not eqx.is_array(value):
                raise TypeError("D2V IREE inputs must be explicit arrays.")
            if tuple(value.shape) != shape:
                raise ValueError("D2V IREE input shape does not match the contract.")
            if np.dtype(value.dtype).str != dtype:
                raise TypeError("D2V IREE input dtype does not match the contract.")
            packed.append(value)
        return tuple(packed)

    def require_compatible(
        self,
        dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
        binding: PreparedLearnedEnergyEquilibriumBinding,
        /,
        *,
        host_id: str,
        policy: IREEExportPolicy | None = None,
    ) -> None:
        policy_ = IREEExportPolicy() if policy is None else policy
        if not isinstance(policy_, IREEExportPolicy):
            raise TypeError("policy must be IREEExportPolicy or None.")
        _validate_runtime_binding(dynamics, binding)
        observed = (
            str(host_id).strip(),
            policy_.target_backend,
            policy_.runtime_driver,
            dynamics.prepared_id,
            dynamics.transport.plan_id,
            dynamics.method.method_id,
            binding.plan.support.support_id,
            binding.prepared_id,
            binding.numeric_revision.revision_id,
        )
        required = (
            self.host_id,
            self.target_backend,
            self.runtime_driver,
            self.runtime_id,
            self.topology_id,
            self.method_id,
            self.support_id,
            self.frozen_artifact_id,
            self.numeric_revision_id,
        )
        if observed != required:
            raise ValueError(
                "D2V IREE runtime, host, backend, or artifact identity changed."
            )


@dataclass(frozen=True, slots=True)
class DiscreteVelocityIREEExportBundle:
    """One forward-only executable and its exact D2V deployment contract."""

    forward: IREEExportResult
    contract: DiscreteVelocityIREEContract

    @property
    def path(self) -> Path:
        return self.forward.path

    @property
    def vjp(self) -> None:
        return None


def discrete_velocity_iree_availability() -> BackendAvailability:
    """Return matched IREE compiler/runtime evidence for D2V deployment."""

    return iree_availability()


def _validate_runtime_binding(
    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    /,
) -> None:
    if not isinstance(dynamics, PreparedSmoothCompressibleD2V17SpatialDynamics):
        raise TypeError(
            "D2V IREE export requires PreparedSmoothCompressibleD2V17SpatialDynamics."
        )
    if not isinstance(binding, PreparedLearnedEnergyEquilibriumBinding):
        raise TypeError(
            "D2V IREE export requires PreparedLearnedEnergyEquilibriumBinding."
        )
    method = dynamics.method
    plan = binding.plan
    if (
        method.quadrature.quadrature_id != plan.equilibrium_plan.quadrature.quadrature_id
        or method.material.material_id != plan.material.material_id
        or dynamics.energy_plan.plan_id != plan.equilibrium_plan.plan_id
    ):
        raise ValueError(
            "Frozen equilibrium artifact does not match the D2V method and support."
        )


def _support_margin(evidence: EnergyEquilibriumSupportEvidence, /) -> Array:
    return jnp.min(
        jnp.stack(
            (
                jnp.min(evidence.rho_margin),
                jnp.min(evidence.u_x_margin),
                jnp.min(evidence.u_y_margin),
                jnp.min(evidence.temperature_margin),
                jnp.min(evidence.mach_margin),
                jnp.min(evidence.hull_margin),
                jnp.min(evidence.particle_equilibrium_margin),
            )
        )
    )


def _accepted_outputs(
    accepted: SmoothCompressibleKineticState,
    successful: Array,
    status: Array,
    first_failure_step: Array,
    conservation_residual: Array,
    energy_flux_error: Array,
    support_margin: Array,
    /,
) -> tuple[Array, ...]:
    dtype = accepted.particle_populations.dtype
    success = jnp.asarray(successful, dtype=jnp.bool_)
    return (
        accepted.particle_populations,
        accepted.total_energy_populations,
        success,
        ~success,
        jnp.asarray(status, dtype=jnp.int32),
        jnp.asarray(first_failure_step, dtype=jnp.int32),
        jnp.asarray(conservation_residual, dtype=dtype),
        jnp.asarray(energy_flux_error, dtype=dtype),
        jnp.asarray(jnp.min(accepted.particle_populations), dtype=dtype),
        jnp.asarray(jnp.min(accepted.total_energy_populations), dtype=dtype),
        jnp.asarray(support_margin, dtype=dtype),
    )


def _frozen_equilibrium_outputs(
    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    conserved: Array,
    /,
) -> tuple[Array, ...]:
    dual, support = binding.predict_dual_with_evidence(conserved)
    equilibrium, evidence = dynamics.method.equilibrium_from_energy_dual_with_evidence(
        conserved, dual, dynamics.energy_plan
    )
    successful = jnp.all(support.successful) & evidence.successful
    accepted = SmoothCompressibleKineticState(
        jnp.where(successful, equilibrium.particle_populations, 0.0),
        jnp.where(successful, equilibrium.total_energy_populations, 0.0),
    )
    status = jnp.where(
        successful,
        int(SmoothCompressibleD2VStepStatus.SUCCESS),
        int(SmoothCompressibleD2VStepStatus.ENERGY_EQUILIBRIUM_FAILED),
    ).astype(jnp.int32)
    first_failure = jnp.where(successful, -1, 0).astype(jnp.int32)
    return _accepted_outputs(
        accepted,
        successful,
        status,
        first_failure,
        evidence.maximum_absolute_residual,
        jnp.max(evidence.energy.flux_error_norm),
        _support_margin(support),
    )


def _one_step_outputs(
    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    state: SmoothCompressibleKineticState,
    transport_args: Any,
    /,
) -> tuple[Array, ...]:
    moments = dynamics.method.moments(state)
    dual, support = binding.predict_dual_with_evidence(moments.conserved)
    result = dynamics.step_with_energy_dual(
        state,
        jnp.asarray(dynamics.required_step_size, dtype=state.particle_populations.dtype),
        dual,
        transport_args,
    )
    support_success = jnp.all(support.successful)
    successful = support_success & result.successful
    accepted = SmoothCompressibleKineticState(
        jnp.where(
            successful,
            result.accepted_state.particle_populations,
            state.particle_populations,
        ),
        jnp.where(
            successful,
            result.accepted_state.total_energy_populations,
            state.total_energy_populations,
        ),
    )
    status = jnp.where(
        support_success,
        result.evidence.status,
        int(SmoothCompressibleD2VStepStatus.ENERGY_EQUILIBRIUM_FAILED),
    ).astype(jnp.int32)
    first_failure = jnp.where(successful, -1, 0).astype(jnp.int32)
    return _accepted_outputs(
        accepted,
        successful,
        status,
        first_failure,
        result.evidence.conservation.maximum_absolute_residual,
        jnp.max(result.evidence.equilibrium.energy.flux_error_norm),
        _support_margin(support),
    )


def _fixed_horizon_outputs(
    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    state: SmoothCompressibleKineticState,
    transport_args: Any,
    step_count: int,
    /,
) -> tuple[Array, ...]:
    dtype = state.particle_populations.dtype
    accepted_f = state.particle_populations
    accepted_g = state.total_energy_populations
    successful = jnp.asarray(True)
    status = jnp.asarray(int(SmoothCompressibleD2VStepStatus.SUCCESS), dtype=jnp.int32)
    first_failure = jnp.asarray(-1, dtype=jnp.int32)
    maximum_residual = jnp.zeros((), dtype=dtype)
    maximum_flux_error = jnp.zeros((), dtype=dtype)
    minimum_margin = jnp.asarray(jnp.inf, dtype=dtype)
    for step_index in range(step_count):
        active = successful
        outputs = _one_step_outputs(
            dynamics,
            binding,
            SmoothCompressibleKineticState(accepted_f, accepted_g),
            transport_args,
        )
        (
            next_f,
            next_g,
            step_successful,
            _,
            step_status,
            _,
            residual,
            flux_error,
            _,
            _,
            support_margin,
        ) = outputs
        accepted_f = jnp.where(active, next_f, accepted_f)
        accepted_g = jnp.where(active, next_g, accepted_g)
        failed_now = active & ~step_successful
        first_failure = jnp.where(
            failed_now,
            jnp.asarray(step_index, dtype=jnp.int32),
            first_failure,
        )
        status = jnp.where(failed_now, step_status, status).astype(jnp.int32)
        maximum_residual = jnp.where(
            active, jnp.maximum(maximum_residual, residual), maximum_residual
        )
        maximum_flux_error = jnp.where(
            active,
            jnp.maximum(maximum_flux_error, flux_error),
            maximum_flux_error,
        )
        minimum_margin = jnp.where(
            active, jnp.minimum(minimum_margin, support_margin), minimum_margin
        )
        successful = active & step_successful
    return _accepted_outputs(
        SmoothCompressibleKineticState(accepted_f, accepted_g),
        successful,
        status,
        first_failure,
        maximum_residual,
        maximum_flux_error,
        minimum_margin,
    )


def prepare_discrete_velocity_iree_contract(
    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    example: SmoothCompressibleKineticState | Array,
    /,
    *,
    host_id: str,
    mode: DiscreteVelocityIREEExportMode = "one-step",
    step_count: int = 1,
    runtime_arrays: tuple[Array, ...] = (),
    runtime_array_names: tuple[str, ...] = (),
    policy: IREEExportPolicy | None = None,
) -> DiscreteVelocityIREEContract:
    """Prepare a fixed forward-only ABI without invoking an IREE compiler."""

    _validate_runtime_binding(dynamics, binding)
    if mode not in ("frozen-equilibrium", "one-step", "fixed-horizon"):
        raise ValueError("Unknown D2V IREE export mode.")
    host = str(host_id).strip()
    if not host:
        raise ValueError("D2V IREE export requires a non-empty host_id.")
    policy_ = IREEExportPolicy() if policy is None else policy
    if not isinstance(policy_, IREEExportPolicy):
        raise TypeError("policy must be IREEExportPolicy or None.")
    count = int(step_count)
    if count <= 0:
        raise ValueError("D2V IREE step_count must be positive.")
    if mode != "fixed-horizon" and count != 1:
        raise ValueError(
            "Only fixed-horizon D2V export accepts step_count other than one."
        )

    runtime = tuple(runtime_arrays)
    runtime_names = tuple(str(name).strip() for name in runtime_array_names)
    if any(not eqx.is_array(value) for value in runtime):
        raise TypeError("D2V IREE runtime inputs must be explicit arrays.")
    if (
        len(runtime_names) != len(runtime)
        or any(not name for name in runtime_names)
        or len(set(runtime_names)) != len(runtime_names)
    ):
        raise ValueError("D2V IREE runtime array names must be unique and exhaustive.")
    population_shape = (
        *dynamics.transport.spatial_shape,
        dynamics.method.quadrature.population_count,
    )
    population_dtype = np.dtype(dynamics.method.quadrature.velocities.dtype).str
    if mode == "frozen-equilibrium":
        if not eqx.is_array(example):
            raise TypeError("Frozen-equilibrium D2V export requires a conserved array.")
        conserved = example
        expected_conserved_shape = (*dynamics.transport.spatial_shape, 4)
        if tuple(conserved.shape) != expected_conserved_shape:
            raise ValueError(
                "Frozen-equilibrium conserved input must have spatial_shape + (4,)."
            )
        if np.dtype(conserved.dtype).str != population_dtype:
            raise TypeError("Frozen-equilibrium input dtype must match the D2V runtime.")
        if runtime:
            raise ValueError("Frozen-equilibrium export has no dynamic runtime arrays.")
        inputs = (conserved,)
        input_names = ("conserved",)
    else:
        if not isinstance(example, SmoothCompressibleKineticState):
            raise TypeError("D2V forward export requires a kinetic-state example.")
        fields = (example.particle_populations, example.total_energy_populations)
        if any(tuple(value.shape) != population_shape for value in fields):
            raise ValueError("D2V IREE populations do not match the fixed runtime shape.")
        if any(np.dtype(value.dtype).str != population_dtype for value in fields):
            raise TypeError("D2V IREE populations do not match the runtime dtype.")
        inputs = (*fields, *runtime)
        input_names = ("f", "g", *runtime_names)
    if len(set(input_names)) != len(input_names):
        raise ValueError("D2V IREE input names must be unique.")

    input_shapes = tuple(tuple(value.shape) for value in inputs)
    input_dtypes = tuple(np.dtype(value.dtype).str for value in inputs)
    scalar = ()
    output_shapes = (population_shape, population_shape, *(scalar for _ in range(9)))
    output_dtypes = (
        population_dtype,
        population_dtype,
        np.dtype(np.bool_).str,
        np.dtype(np.bool_).str,
        np.dtype(np.int32).str,
        np.dtype(np.int32).str,
        *(population_dtype for _ in range(5)),
    )
    metadata = {
        "kind": "discrete-velocity-iree-contract",
        "mode": mode,
        "host": host,
        "target_backend": policy_.target_backend,
        "runtime_driver": policy_.runtime_driver,
        "runtime": dynamics.prepared_id,
        "topology": dynamics.transport.plan_id,
        "method": dynamics.method.method_id,
        "support": binding.plan.support.support_id,
        "frozen_artifact": binding.prepared_id,
        "numeric_revision": binding.numeric_revision.revision_id,
        "inputs": tuple(zip(input_names, input_shapes, input_dtypes, strict=True)),
        "outputs": tuple(
            zip(_D2V_OUTPUT_NAMES, output_shapes, output_dtypes, strict=True)
        ),
        "step_count": count,
        "step_size": dynamics.required_step_size,
        "reverse_mode": False,
        "training_export": False,
    }
    return DiscreteVelocityIREEContract(
        mode,
        host,
        policy_.target_backend,
        policy_.runtime_driver,
        dynamics.prepared_id,
        dynamics.transport.plan_id,
        dynamics.method.method_id,
        binding.plan.support.support_id,
        binding.prepared_id,
        binding.numeric_revision.revision_id,
        input_names,
        input_shapes,
        input_dtypes,
        _D2V_OUTPUT_NAMES,
        output_shapes,
        output_dtypes,
        count,
        dynamics.required_step_size,
        False,
        False,
        canonical_fingerprint(metadata),
    )


def save_discrete_velocity_iree(
    dynamics: PreparedSmoothCompressibleD2V17SpatialDynamics,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    path: str | Path,
    /,
    *,
    example: SmoothCompressibleKineticState | Array,
    host_id: str,
    mode: DiscreteVelocityIREEExportMode = "one-step",
    step_count: int = 1,
    runtime_arrays: tuple[Array, ...] = (),
    runtime_array_names: tuple[str, ...] = (),
    policy: IREEExportPolicy | None = None,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-6,
) -> DiscreteVelocityIREEExportBundle:
    """Compile a frozen, forward-only D2V executable with mandatory native parity."""

    policy_ = IREEExportPolicy() if policy is None else policy
    contract = prepare_discrete_velocity_iree_contract(
        dynamics,
        binding,
        example,
        host_id=host_id,
        mode=mode,
        step_count=step_count,
        runtime_arrays=runtime_arrays,
        runtime_array_names=runtime_array_names,
        policy=policy_,
    )
    discrete_velocity_iree_availability().require("compiled-inference")

    if mode == "frozen-equilibrium":
        conserved = example

        def forward(conserved_input, *, key=None):
            if key is not None:
                raise ValueError("D2V IREE export requires key=None.")
            return _frozen_equilibrium_outputs(dynamics, binding, conserved_input)

        primal_inputs = contract.pack_inputs(conserved)
    else:
        state = example
        if not isinstance(state, SmoothCompressibleKineticState):
            raise TypeError("D2V forward export requires a kinetic-state example.")
        runtime = tuple(runtime_arrays)

        def forward(f, g, *runtime_inputs, key=None):
            if key is not None:
                raise ValueError("D2V IREE export requires key=None.")
            transport_args = None if not runtime_inputs else tuple(runtime_inputs)
            kinetic_state = SmoothCompressibleKineticState(f, g)
            if mode == "one-step":
                return _one_step_outputs(dynamics, binding, kinetic_state, transport_args)
            return _fixed_horizon_outputs(
                dynamics,
                binding,
                kinetic_state,
                transport_args,
                contract.step_count,
            )

        primal_inputs = contract.pack_inputs(
            state.particle_populations,
            state.total_energy_populations,
            *runtime,
        )
    artifact = save_iree(
        forward,
        path,
        inputs=primal_inputs,
        input_names=contract.input_names,
        output_names=contract.output_names,
        policy=policy_,
        key=None,
        validate=True,
        rtol=rtol,
        atol=atol,
    )
    if (
        artifact.manifest.target_backend != contract.target_backend
        or artifact.manifest.runtime_driver != contract.runtime_driver
        or artifact.manifest.input_names != contract.input_names
        or artifact.manifest.input_shapes != contract.input_shapes
        or artifact.manifest.input_dtypes != contract.input_dtypes
        or artifact.manifest.output_names != contract.output_names
        or artifact.manifest.output_shapes != contract.output_shapes
        or artifact.manifest.output_dtypes != contract.output_dtypes
        or artifact.manifest.validation_ok is not True
    ):
        raise RuntimeError("D2V IREE artifact backend, ABI, or native parity changed.")
    return DiscreteVelocityIREEExportBundle(artifact, contract)


__all__ = [
    "DiscreteVelocityIREEContract",
    "DiscreteVelocityIREEExportBundle",
    "DiscreteVelocityIREEExportMode",
    "discrete_velocity_iree_availability",
    "prepare_discrete_velocity_iree_contract",
    "save_discrete_velocity_iree",
]
