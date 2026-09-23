#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._io import write_json_atomic
from benchmarks._runtime import capture_environment
from phydrax._fingerprint import canonical_fingerprint, canonical_json
from phydrax.backends._types import BackendAvailability
from phydrax.closure_data._kinetic_equilibrium_artifact import (
    LearnedEnergyEquilibriumArtifact,
    read_learned_energy_equilibrium_artifact,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
    SmoothCompressibleD2VStepStatus,
)
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.export._discrete_velocity_iree import (
    _fixed_horizon_outputs,
    _frozen_equilibrium_outputs,
    _one_step_outputs,
    discrete_velocity_iree_availability,
    DiscreteVelocityIREEContract,
    prepare_discrete_velocity_iree_contract,
    save_discrete_velocity_iree,
)
from phydrax.export._iree import IREEExecutable, IREEExportPolicy, load_iree


DEFAULT_MODEL_ARTIFACT = Path("benchmarks/learned_energy_equilibrium.phxml")
DEFAULT_PARENT_SPATIAL = Path("benchmarks/learned_energy_spatial.json")
DEFAULT_OUTPUT = Path("benchmarks/learned_energy_export.json")
FIXED_HORIZON_STEPS = 4
PARTICLE_RELAXATION_TIME = 0.03
TOTAL_ENERGY_RELAXATION_TIME = 0.04
PARITY_RTOL = 1.0e-10
PARITY_ATOL = 1.0e-11

OUTPUT_NAMES = (
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


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _strict_bool(value: Any, owner: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{owner} must be a JSON boolean.")
    return value


def _identifier(value: Any, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _finite_float(value: Any, owner: str, /, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{owner} must be a JSON number.")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        qualifier = " finite and positive" if positive else " finite"
        raise ValueError(f"{owner} must be{qualifier}.")
    return result


def _positive_int(value: Any, owner: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{owner} must be a positive integer.")
    return value


def _mapping(value: Any, owner: str, /) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a JSON object.")
    return value


def _read_parent_spatial(path: Path, /) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    report = _mapping(report, "Parent spatial qualification")
    required = {
        "configuration",
        "environment",
        "gates",
        "identities",
        "passed",
        "qualification_id",
        "thresholds",
        "tool",
    }
    if not required.issubset(report):
        raise ValueError("Parent spatial qualification is missing required fields.")
    if report["tool"] != "learned_energy_spatial_qualification":
        raise ValueError("Parent report is not the learned-energy spatial qualification.")
    _strict_bool(report["passed"], "Parent spatial passed state")
    qualification_id = _identifier(
        report["qualification_id"], "Parent spatial qualification identity"
    )
    fingerprinted = dict(report)
    del fingerprinted["qualification_id"]
    if canonical_fingerprint(fingerprinted) != qualification_id:
        raise ValueError("Parent spatial qualification fingerprint is inconsistent.")

    identities = _mapping(report["identities"], "Parent spatial identities")
    required_identities = {
        "artifact",
        "energy_plan",
        "kinetic_method",
        "material",
        "model_numeric_revision",
        "prepared_binding",
        "prepared_spatial",
        "quadrature",
        "spatial_plan",
        "support",
        "transport_closure",
    }
    if not required_identities.issubset(identities):
        raise ValueError("Parent spatial qualification has an incomplete identity chain.")
    _mapping(report["configuration"], "Parent spatial configuration")
    _mapping(report["thresholds"], "Parent spatial thresholds")
    _mapping(report["environment"], "Parent spatial environment")
    return report


def _pair_of_positive_ints(value: Any, owner: str, /) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{owner} must contain exactly two integers.")
    return (
        _positive_int(value[0], f"{owner}[0]"),
        _positive_int(value[1], f"{owner}[1]"),
    )


def _pair_of_positive_floats(value: Any, owner: str, /) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{owner} must contain exactly two numbers.")
    return (
        _finite_float(value[0], f"{owner}[0]", positive=True),
        _finite_float(value[1], f"{owner}[1]", positive=True),
    )


def _build_runtime(
    artifact: LearnedEnergyEquilibriumArtifact,
    parent: Mapping[str, Any],
    /,
) -> PreparedSmoothCompressibleD2V17SpatialDynamics:
    configuration = _mapping(parent["configuration"], "Parent spatial configuration")
    thresholds = _mapping(parent["thresholds"], "Parent spatial thresholds")
    spatial_shape = _pair_of_positive_ints(
        configuration["coarse_resolution"], "Parent coarse resolution"
    )
    cell_spacing = _pair_of_positive_floats(
        configuration["coarse_cell_spacing"], "Parent coarse cell spacing"
    )
    step_size = _finite_float(
        configuration["coarse_time_step"],
        "Parent coarse time step",
        positive=True,
    )
    conservation_tolerance = _finite_float(
        thresholds["maximum_conservation_residual"],
        "Parent conservation tolerance",
        positive=True,
    )
    binding = artifact.binding
    method = SmoothCompressibleD2VKineticMethod(
        binding.plan.equilibrium_plan.quadrature,
        binding.plan.material,
        ConstantTransport(
            PARTICLE_RELAXATION_TIME,
            TOTAL_ENERGY_RELAXATION_TIME,
        ),
    )
    transport = D2V17PeriodicTransportPlan(
        method.quadrature,
        spatial_shape,
        cell_spacing,
        step_size,
    )
    return PreparedSmoothCompressibleD2V17SpatialDynamics(
        method,
        binding.plan.equilibrium_plan,
        transport,
        conservation_tolerance=conservation_tolerance,
    )


def _interior_field(
    bounds: tuple[float, float], pattern: jax.Array, amplitude: float, /
) -> jax.Array:
    lower, upper = bounds
    midpoint = 0.5 * (lower + upper)
    return midpoint + amplitude * (upper - lower) * pattern


def _accepted_conserved(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    artifact: LearnedEnergyEquilibriumArtifact,
    /,
) -> jax.Array:
    shape = runtime.transport.spatial_shape
    dtype = runtime.method.quadrature.velocities.dtype
    x = jnp.arange(shape[0], dtype=dtype)[:, None]
    y = jnp.arange(shape[1], dtype=dtype)[None, :]
    phase_x = 2.0 * jnp.pi * x / shape[0]
    phase_y = 2.0 * jnp.pi * y / shape[1]
    support = artifact.binding.plan.support
    density = _interior_field(
        support.rho_bounds,
        jnp.sin(phase_x) * jnp.cos(phase_y),
        0.02,
    )
    velocity_x = _interior_field(
        support.u_x_bounds,
        jnp.cos(phase_x) * jnp.sin(phase_y),
        0.02,
    )
    velocity_y = _interior_field(
        support.u_y_bounds,
        jnp.sin(phase_x + phase_y),
        0.02,
    )
    temperature = _interior_field(
        support.temperature_bounds,
        jnp.cos(phase_x - phase_y),
        0.02,
    )
    material = runtime.method.material
    pressure = density * material.gas_constant * temperature
    specific_internal_energy = material.specific_internal_energy(density, pressure)
    total_energy = density * specific_internal_energy + 0.5 * density * (
        velocity_x**2 + velocity_y**2
    )
    return jnp.stack(
        (
            density,
            density * velocity_x,
            density * velocity_y,
            total_energy,
        ),
        axis=-1,
    )


def _rejected_state(
    accepted: SmoothCompressibleKineticState,
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    artifact: LearnedEnergyEquilibriumArtifact,
    /,
) -> SmoothCompressibleKineticState:
    density = runtime.method.moments(accepted).conserved[..., 0]
    lower, upper = artifact.binding.plan.support.rho_bounds
    rejected_density = upper + max(0.05 * (upper - lower), 0.01 * abs(upper), 1.0e-6)
    scale = jnp.asarray(rejected_density, dtype=density.dtype) / density
    return SmoothCompressibleKineticState(
        accepted.particle_populations * scale[..., None],
        accepted.total_energy_populations * scale[..., None],
    )


def _native_outputs(
    mode: str,
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    artifact: LearnedEnergyEquilibriumArtifact,
    inputs: tuple[jax.Array, ...],
    /,
) -> tuple[jax.Array, ...]:
    binding = artifact.binding
    if mode == "frozen-equilibrium":
        return _frozen_equilibrium_outputs(runtime, binding, inputs[0])
    state = SmoothCompressibleKineticState(inputs[0], inputs[1])
    if mode == "one-step":
        return _one_step_outputs(runtime, binding, state, None)
    if mode == "fixed-horizon":
        return _fixed_horizon_outputs(
            runtime,
            binding,
            state,
            None,
            FIXED_HORIZON_STEPS,
        )
    raise ValueError("Unknown learned-energy export mode.")


def _contract_record(contract: DiscreteVelocityIREEContract, /) -> dict[str, Any]:
    return {
        "contract_id": contract.contract_id,
        "execution_mode": contract.execution_mode,
        "host_id": contract.host_id,
        "target_backend": contract.target_backend,
        "runtime_driver": contract.runtime_driver,
        "runtime_id": contract.runtime_id,
        "topology_id": contract.topology_id,
        "method_id": contract.method_id,
        "support_id": contract.support_id,
        "frozen_artifact_id": contract.frozen_artifact_id,
        "numeric_revision_id": contract.numeric_revision_id,
        "input_names": list(contract.input_names),
        "input_shapes": [list(shape) for shape in contract.input_shapes],
        "input_dtypes": list(contract.input_dtypes),
        "output_names": list(contract.output_names),
        "output_shapes": [list(shape) for shape in contract.output_shapes],
        "output_dtypes": list(contract.output_dtypes),
        "step_count": contract.step_count,
        "step_size": contract.step_size,
        "supports_reverse_mode": contract.supports_reverse_mode,
        "supports_training_export": contract.supports_training_export,
    }


def _prepare_contracts(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    artifact: LearnedEnergyEquilibriumArtifact,
    accepted_conserved: jax.Array,
    accepted_state: SmoothCompressibleKineticState,
    host_id: str,
    policy: IREEExportPolicy,
    /,
) -> dict[str, DiscreteVelocityIREEContract]:
    binding = artifact.binding
    contracts = {
        "frozen-equilibrium": prepare_discrete_velocity_iree_contract(
            runtime,
            binding,
            accepted_conserved,
            host_id=host_id,
            mode="frozen-equilibrium",
            policy=policy,
        ),
        "one-step": prepare_discrete_velocity_iree_contract(
            runtime,
            binding,
            accepted_state,
            host_id=host_id,
            mode="one-step",
            policy=policy,
        ),
        "fixed-horizon": prepare_discrete_velocity_iree_contract(
            runtime,
            binding,
            accepted_state,
            host_id=host_id,
            mode="fixed-horizon",
            step_count=FIXED_HORIZON_STEPS,
            policy=policy,
        ),
    }
    contracts["frozen-equilibrium"].pack_inputs(accepted_conserved)
    contracts["one-step"].pack_inputs(
        accepted_state.particle_populations,
        accepted_state.total_energy_populations,
    )
    contracts["fixed-horizon"].pack_inputs(
        accepted_state.particle_populations,
        accepted_state.total_energy_populations,
    )
    return contracts


def _state_diagnostic_arrays(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    outputs: Sequence[Any],
    /,
) -> dict[str, np.ndarray]:
    state = SmoothCompressibleKineticState(
        jnp.asarray(outputs[0]),
        jnp.asarray(outputs[1]),
    )
    moments = runtime.method.moments(state)
    velocities = runtime.method.quadrature.velocities
    return {
        "conserved": np.asarray(moments.conserved),
        "particle_stress": np.asarray(
            jnp.einsum(
                "...q,qi,qj->...ij",
                state.particle_populations,
                velocities,
                velocities,
            )
        ),
        "total_energy_density": np.asarray(
            jnp.sum(state.total_energy_populations, axis=-1)
        ),
        "total_energy_flux": np.asarray(
            jnp.einsum(
                "...q,qd->...d",
                state.total_energy_populations,
                velocities,
            )
        ),
    }


def _maximum_absolute(values: np.ndarray, /) -> float:
    return float(np.max(np.abs(values), initial=0.0))


def _case_summary(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    outputs: Sequence[Any],
    /,
    *,
    input_state: SmoothCompressibleKineticState | None,
    input_conserved: jax.Array,
) -> dict[str, Any]:
    arrays = tuple(np.asarray(value) for value in outputs)
    diagnostics = _state_diagnostic_arrays(runtime, arrays)
    conserved = diagnostics["conserved"]
    energy = diagnostics["total_energy_density"]
    stress = diagnostics["particle_stress"]
    flux = diagnostics["total_energy_flux"]
    local_conserved_residual = _maximum_absolute(conserved - np.asarray(input_conserved))
    if input_state is None:
        global_conservation_residual = local_conserved_residual
    else:
        initial = np.asarray(runtime.method.moments(input_state).conserved)
        axes = tuple(range(initial.ndim - 1))
        cell_volume = runtime.transport.cell_volume
        global_conservation_residual = _maximum_absolute(
            (np.sum(conserved, axis=axes) - np.sum(initial, axis=axes)) * cell_volume
        )
    if input_state is None:
        accepted_f_matches_input = None
        accepted_g_matches_input = None
    else:
        accepted_f_matches_input = bool(
            np.array_equal(arrays[0], np.asarray(input_state.particle_populations))
        )
        accepted_g_matches_input = bool(
            np.array_equal(arrays[1], np.asarray(input_state.total_energy_populations))
        )
    return {
        "successful": bool(arrays[2]),
        "rollback_applied": bool(arrays[3]),
        "status": int(arrays[4]),
        "first_failure_step": int(arrays[5]),
        "maximum_conservation_residual": float(arrays[6]),
        "maximum_energy_flux_error": float(arrays[7]),
        "minimum_f": float(arrays[8]),
        "minimum_g": float(arrays[9]),
        "minimum_support_margin": float(arrays[10]),
        "output_shapes": [list(value.shape) for value in arrays],
        "output_dtypes": [value.dtype.str for value in arrays],
        "accepted_f_matches_input_exactly": accepted_f_matches_input,
        "accepted_g_matches_input_exactly": accepted_g_matches_input,
        "all_outputs_finite": all(bool(np.all(np.isfinite(value))) for value in arrays),
        "minimum_f_matches_population": bool(
            np.array_equal(arrays[8], np.asarray(np.min(arrays[0])))
        ),
        "minimum_g_matches_population": bool(
            np.array_equal(arrays[9], np.asarray(np.min(arrays[1])))
        ),
        "local_conserved_maximum_absolute_residual": local_conserved_residual,
        "global_conservation_maximum_absolute_residual": global_conservation_residual,
        "energy_sum_maximum_absolute_residual": _maximum_absolute(
            energy - conserved[..., -1]
        ),
        "particle_stress_frobenius_norm": float(np.linalg.norm(stress)),
        "total_energy_density_l2_norm": float(np.linalg.norm(energy)),
        "total_energy_flux_l2_norm": float(np.linalg.norm(flux)),
        "stress_and_energy_diagnostics_finite": bool(
            np.all(np.isfinite(stress))
            and np.all(np.isfinite(energy))
            and np.all(np.isfinite(flux))
        ),
    }


def _array_comparison(
    name: str,
    native: Any,
    deployed: Any,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    /,
) -> dict[str, Any]:
    native_array = np.asarray(native)
    deployed_array = np.asarray(deployed)
    metadata_match = bool(
        tuple(native_array.shape) == expected_shape
        and tuple(deployed_array.shape) == expected_shape
        and native_array.dtype.str == expected_dtype
        and deployed_array.dtype.str == expected_dtype
    )
    finite = bool(
        np.all(np.isfinite(native_array)) and np.all(np.isfinite(deployed_array))
    )
    if np.issubdtype(native_array.dtype, np.inexact):
        absolute = np.abs(native_array - deployed_array)
        scale = np.maximum(
            np.abs(native_array),
            np.finfo(native_array.real.dtype).tiny,
        )
        within_tolerance = bool(
            np.allclose(
                native_array,
                deployed_array,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
        )
    else:
        native_numeric = native_array.astype(np.float64)
        deployed_numeric = deployed_array.astype(np.float64)
        absolute = np.abs(native_numeric - deployed_numeric)
        scale = np.maximum(np.abs(native_numeric), np.finfo(np.float64).tiny)
        within_tolerance = bool(np.array_equal(native_array, deployed_array))
    return {
        "name": name,
        "expected_shape": list(expected_shape),
        "expected_dtype": expected_dtype,
        "native_shape": list(native_array.shape),
        "native_dtype": native_array.dtype.str,
        "deployed_shape": list(deployed_array.shape),
        "deployed_dtype": deployed_array.dtype.str,
        "metadata_match": metadata_match,
        "finite": finite,
        "exactly_equal": bool(np.array_equal(native_array, deployed_array)),
        "within_tolerance": within_tolerance,
        "maximum_absolute_error": float(np.max(absolute, initial=0.0)),
        "maximum_relative_error": float(np.max(absolute / scale, initial=0.0)),
    }


def _diagnostic_comparison(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    native: Sequence[Any],
    deployed: Sequence[Any],
    /,
) -> dict[str, Any]:
    native_diagnostics = _state_diagnostic_arrays(runtime, native)
    deployed_diagnostics = _state_diagnostic_arrays(runtime, deployed)
    records = {}
    for name in (
        "conserved",
        "particle_stress",
        "total_energy_density",
        "total_energy_flux",
    ):
        expected = native_diagnostics[name]
        observed = deployed_diagnostics[name]
        difference = np.abs(expected - observed)
        scale = np.maximum(np.abs(expected), np.finfo(expected.dtype).tiny)
        records[name] = {
            "finite": bool(
                np.all(np.isfinite(expected)) and np.all(np.isfinite(observed))
            ),
            "within_tolerance": bool(
                np.allclose(
                    expected,
                    observed,
                    rtol=PARITY_RTOL,
                    atol=PARITY_ATOL,
                )
            ),
            "maximum_absolute_error": float(np.max(difference, initial=0.0)),
            "maximum_relative_error": float(np.max(difference / scale, initial=0.0)),
        }
    return {
        "fields": records,
        "all_finite": all(record["finite"] for record in records.values()),
        "all_within_tolerance": all(
            record["within_tolerance"] for record in records.values()
        ),
    }


def _comparison_record(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    contract: DiscreteVelocityIREEContract,
    native: tuple[jax.Array, ...],
    deployed: Any,
    /,
    *,
    input_state: SmoothCompressibleKineticState | None,
    input_conserved: jax.Array,
) -> dict[str, Any]:
    deployed_outputs = deployed if isinstance(deployed, tuple) else (deployed,)
    if len(native) != len(OUTPUT_NAMES) or len(deployed_outputs) != len(OUTPUT_NAMES):
        raise RuntimeError("D2V native or deployed output arity changed.")
    fields = [
        _array_comparison(name, native_value, deployed_value, shape, dtype)
        for name, native_value, deployed_value, shape, dtype in zip(
            contract.output_names,
            native,
            deployed_outputs,
            contract.output_shapes,
            contract.output_dtypes,
            strict=True,
        )
    ]
    return {
        "fields": fields,
        "all_fields_metadata_match": all(field["metadata_match"] for field in fields),
        "all_fields_finite": all(field["finite"] for field in fields),
        "all_fields_within_tolerance": all(field["within_tolerance"] for field in fields),
        "all_discrete_fields_exact": all(
            fields[index]["exactly_equal"] for index in (2, 3, 4, 5)
        ),
        "native": _case_summary(
            runtime,
            native,
            input_state=input_state,
            input_conserved=input_conserved,
        ),
        "deployed": _case_summary(
            runtime,
            deployed_outputs,
            input_state=input_state,
            input_conserved=input_conserved,
        ),
        "derived_diagnostics": _diagnostic_comparison(
            runtime,
            native,
            deployed_outputs,
        ),
    }


def _availability_record(availability: BackendAvailability, /) -> dict[str, Any]:
    capabilities = availability.capabilities
    return {
        "backend": availability.backend,
        "available": availability.available,
        "requirement": availability.requirement,
        "reason": availability.reason,
        "versions": {name: version for name, version in sorted(availability.versions)},
        "capability": "compiled-inference",
        "capability_declared": capabilities.supports("compiled-inference"),
        "execution": capabilities.execution,
        "host_only": capabilities.host_only,
    }


def _matched_iree_available(availability: BackendAvailability, /) -> bool:
    versions = dict(availability.versions)
    return bool(
        availability.backend == "iree"
        and availability.available
        and availability.capabilities.supports("compiled-inference")
        and "iree-base-compiler" in versions
        and "iree-base-runtime" in versions
        and versions["iree-base-compiler"] == versions["iree-base-runtime"]
    )


def _manifest_record(executable: IREEExecutable, /) -> dict[str, Any]:
    manifest = executable.manifest
    return {
        "artifact_id": manifest.artifact_id,
        "module_sha256": manifest.module_sha256,
        "compiler_version": manifest.compiler_version,
        "runtime_version": manifest.runtime_version,
        "target_backend": manifest.target_backend,
        "runtime_driver": manifest.runtime_driver,
        "function_name": manifest.function_name,
        "entry_point": manifest.entry_point,
        "calling_convention_version": manifest.calling_convention_version,
        "input_names": list(manifest.input_names),
        "input_shapes": [list(shape) for shape in manifest.input_shapes],
        "input_dtypes": list(manifest.input_dtypes),
        "output_names": list(manifest.output_names),
        "output_shapes": [list(shape) for shape in manifest.output_shapes],
        "output_dtypes": list(manifest.output_dtypes),
        "vectorized": manifest.vectorized,
        "has_preprocess": manifest.has_preprocess,
        "has_postprocess": manifest.has_postprocess,
        "validation_ok": manifest.validation_ok,
        "maximum_absolute_errors": list(manifest.maximum_absolute_errors or ()),
        "maximum_relative_errors": list(manifest.maximum_relative_errors or ()),
    }


def _execute_exports(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    artifact: LearnedEnergyEquilibriumArtifact,
    contracts: Mapping[str, DiscreteVelocityIREEContract],
    cases: Mapping[str, Mapping[str, tuple[jax.Array, ...]]],
    accepted_state: SmoothCompressibleKineticState,
    rejected_state: SmoothCompressibleKineticState,
    policy: IREEExportPolicy,
    /,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="phydrax-learned-energy-export-"
    ) as directory:
        root = Path(directory)
        for mode in ("frozen-equilibrium", "one-step", "fixed-horizon"):
            contract = contracts[mode]
            example: jax.Array | SmoothCompressibleKineticState
            if mode == "frozen-equilibrium":
                example = cases[mode]["accepted"][0]
            else:
                example = accepted_state
            destination = root / f"{mode}.phxiree"
            bundle = save_discrete_velocity_iree(
                runtime,
                artifact.binding,
                destination,
                example=example,
                host_id=contract.host_id,
                mode=mode,
                step_count=contract.step_count,
                policy=policy,
                rtol=PARITY_RTOL,
                atol=PARITY_ATOL,
            )
            executable = load_iree(
                destination,
                trusted_module_sha256=bundle.forward.manifest.module_sha256,
            )
            manifest = executable.manifest
            case_records = {}
            for case_name in ("accepted", "rejected"):
                inputs = cases[mode][case_name]
                packed = contract.pack_inputs(*inputs)
                native = _native_outputs(mode, runtime, artifact, packed)
                deployed = executable(*(np.asarray(value) for value in packed))
                input_state = None
                if mode != "frozen-equilibrium":
                    input_state = (
                        accepted_state if case_name == "accepted" else rejected_state
                    )
                input_conserved = (
                    inputs[0]
                    if mode == "frozen-equilibrium"
                    else runtime.method.moments(input_state).conserved
                )
                case_records[case_name] = _comparison_record(
                    runtime,
                    contract,
                    native,
                    deployed,
                    input_state=input_state,
                    input_conserved=input_conserved,
                )
            modes[mode] = {
                "performed": True,
                "execution_engine": f"{type(executable).__module__}.{type(executable).__name__}",
                "explicit_native_reference_engine": "jax-native",
                "fallback_attempted": False,
                "contract_matches_prepared": bundle.contract.contract_id
                == contract.contract_id,
                "manifest_matches_contract": bool(
                    manifest.target_backend == contract.target_backend
                    and manifest.runtime_driver == contract.runtime_driver
                    and manifest.input_names == contract.input_names
                    and manifest.input_shapes == contract.input_shapes
                    and manifest.input_dtypes == contract.input_dtypes
                    and manifest.output_names == contract.output_names
                    and manifest.output_shapes == contract.output_shapes
                    and manifest.output_dtypes == contract.output_dtypes
                ),
                "manifest": _manifest_record(executable),
                "cases": case_records,
            }
    return {
        "performed": True,
        "reason": None,
        "export_compilation_attempted": True,
        "export_execution_attempted": True,
        "fallback_attempted": False,
        "modes": modes,
    }


def _unavailable_exports(
    availability: BackendAvailability,
    contracts: Mapping[str, DiscreteVelocityIREEContract],
    /,
) -> dict[str, Any]:
    unavailable_reason = (
        availability.reason
        if not availability.available
        else "IREE compiler and runtime versions are not matched."
    )
    return {
        "performed": False,
        "reason": unavailable_reason,
        "requirement": availability.requirement,
        "export_compilation_attempted": False,
        "export_execution_attempted": False,
        "fallback_attempted": False,
        "modes": {
            mode: {
                "performed": False,
                "reason": unavailable_reason,
                "contract_id": contract.contract_id,
                "cases": {
                    "accepted": {"performed": False, "reason": unavailable_reason},
                    "rejected": {"performed": False, "reason": unavailable_reason},
                },
            }
            for mode, contract in contracts.items()
        },
    }


def _native_case_record(
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    artifact: LearnedEnergyEquilibriumArtifact,
    mode: str,
    inputs: tuple[jax.Array, ...],
    state: SmoothCompressibleKineticState | None,
    /,
) -> dict[str, Any]:
    outputs = _native_outputs(mode, runtime, artifact, inputs)
    conserved = inputs[0] if state is None else runtime.method.moments(state).conserved
    summary = _case_summary(
        runtime,
        outputs,
        input_state=state,
        input_conserved=conserved,
    )
    return summary


def _contract_gates(
    contracts: Mapping[str, DiscreteVelocityIREEContract],
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    artifact: LearnedEnergyEquilibriumArtifact,
    /,
) -> dict[str, bool]:
    population_shape = (
        *runtime.transport.spatial_shape,
        runtime.method.quadrature.population_count,
    )
    population_dtype = np.dtype(runtime.method.quadrature.velocities.dtype).str
    expected_output_shapes = (population_shape, population_shape, *((),) * 9)
    expected_output_dtypes = (
        population_dtype,
        population_dtype,
        np.dtype(np.bool_).str,
        np.dtype(np.bool_).str,
        np.dtype(np.int32).str,
        np.dtype(np.int32).str,
        *(population_dtype,) * 5,
    )
    expected_inputs = {
        "frozen-equilibrium": ("conserved",),
        "one-step": ("f", "g"),
        "fixed-horizon": ("f", "g"),
    }
    expected_input_shapes = {
        "frozen-equilibrium": ((*runtime.transport.spatial_shape, 4),),
        "one-step": (population_shape, population_shape),
        "fixed-horizon": (population_shape, population_shape),
    }
    expected_input_dtypes = {
        "frozen-equilibrium": (population_dtype,),
        "one-step": (population_dtype, population_dtype),
        "fixed-horizon": (population_dtype, population_dtype),
    }
    return {
        "all_three_forward_contracts_prepared": bool(
            tuple(contracts)
            == (
                "frozen-equilibrium",
                "one-step",
                "fixed-horizon",
            )
            and len({contract.contract_id for contract in contracts.values()}) == 3
        ),
        "abi_names_shapes_and_dtypes": all(
            contract.input_names == expected_inputs[mode]
            and contract.input_shapes == expected_input_shapes[mode]
            and contract.input_dtypes == expected_input_dtypes[mode]
            and contract.output_names == OUTPUT_NAMES
            and contract.output_shapes == expected_output_shapes
            and contract.output_dtypes == expected_output_dtypes
            for mode, contract in contracts.items()
        ),
        "bool_int_abi_separation": all(
            contract.output_dtypes[2:4] == (np.dtype(np.bool_).str,) * 2
            and contract.output_dtypes[4:6] == (np.dtype(np.int32).str,) * 2
            and contract.output_dtypes[2] != contract.output_dtypes[4]
            for contract in contracts.values()
        ),
        "model_support_runtime_identities": all(
            contract.runtime_id == runtime.prepared_id
            and contract.topology_id == runtime.transport.plan_id
            and contract.method_id == runtime.method.method_id
            and contract.support_id == artifact.binding.plan.support.support_id
            and contract.frozen_artifact_id == artifact.binding.prepared_id
            and contract.numeric_revision_id
            == artifact.binding.numeric_revision.revision_id
            for contract in contracts.values()
        ),
        "fixed_step_and_horizon": bool(
            contracts["frozen-equilibrium"].step_count == 1
            and contracts["one-step"].step_count == 1
            and contracts["fixed-horizon"].step_count == FIXED_HORIZON_STEPS
            and all(
                contract.step_size == runtime.required_step_size
                for contract in contracts.values()
            )
        ),
        "forward_only_no_vjp_or_training_export": all(
            not contract.supports_reverse_mode and not contract.supports_training_export
            for contract in contracts.values()
        ),
    }


def _deployment_gates(
    availability: BackendAvailability,
    deployment: Mapping[str, Any],
    runtime: PreparedSmoothCompressibleD2V17SpatialDynamics,
    /,
) -> dict[str, bool]:
    matched_versions = _matched_iree_available(availability)
    if not matched_versions:
        return {
            "matched_iree_compiler_runtime_available": False,
            "compiled_artifact_native_parity": False,
            "accepted_outputs_field_by_field": False,
            "rejected_outputs_field_by_field": False,
            "export_exact_f_g_rollback": False,
            "export_conservation_diagnostics": False,
            "export_stress_and_energy_diagnostics": False,
            "unavailable_backend_evidence_and_fail_closed": bool(
                not deployment["performed"]
                and not deployment["export_compilation_attempted"]
                and not deployment["export_execution_attempted"]
                and not deployment["fallback_attempted"]
                and bool(deployment["reason"])
                and bool(deployment["requirement"])
            ),
            "no_hidden_jax_fallback": not deployment["fallback_attempted"],
        }

    modes = deployment["modes"]
    comparisons = [
        modes[mode]["cases"][case]
        for mode in ("frozen-equilibrium", "one-step", "fixed-horizon")
        for case in ("accepted", "rejected")
    ]
    accepted = [
        modes[mode]["cases"]["accepted"]
        for mode in ("frozen-equilibrium", "one-step", "fixed-horizon")
    ]
    rejected = [
        modes[mode]["cases"]["rejected"]
        for mode in ("frozen-equilibrium", "one-step", "fixed-horizon")
    ]
    rollback_cases = [
        modes[mode]["cases"]["rejected"] for mode in ("one-step", "fixed-horizon")
    ]
    return {
        "matched_iree_compiler_runtime_available": matched_versions,
        "compiled_artifact_native_parity": bool(
            deployment["performed"]
            and all(
                modes[mode]["performed"]
                and modes[mode]["contract_matches_prepared"]
                and modes[mode]["manifest_matches_contract"]
                and modes[mode]["manifest"]["validation_ok"] is True
                for mode in modes
            )
        ),
        "accepted_outputs_field_by_field": all(
            comparison["all_fields_metadata_match"]
            and comparison["all_fields_finite"]
            and comparison["all_fields_within_tolerance"]
            and comparison["all_discrete_fields_exact"]
            and comparison["native"]["successful"]
            and not comparison["native"]["rollback_applied"]
            and comparison["deployed"]["successful"]
            and not comparison["deployed"]["rollback_applied"]
            for comparison in accepted
        ),
        "rejected_outputs_field_by_field": all(
            comparison["all_fields_metadata_match"]
            and comparison["all_fields_finite"]
            and comparison["all_fields_within_tolerance"]
            and comparison["all_discrete_fields_exact"]
            and not comparison["native"]["successful"]
            and comparison["native"]["rollback_applied"]
            and not comparison["deployed"]["successful"]
            and comparison["deployed"]["rollback_applied"]
            for comparison in rejected
        ),
        "export_exact_f_g_rollback": all(
            comparison["native"]["accepted_f_matches_input_exactly"]
            and comparison["native"]["accepted_g_matches_input_exactly"]
            and comparison["deployed"]["accepted_f_matches_input_exactly"]
            and comparison["deployed"]["accepted_g_matches_input_exactly"]
            and comparison["native"]["first_failure_step"] == 0
            and comparison["deployed"]["first_failure_step"] == 0
            and comparison["native"]["global_conservation_maximum_absolute_residual"]
            == 0.0
            and comparison["deployed"]["global_conservation_maximum_absolute_residual"]
            == 0.0
            for comparison in rollback_cases
        ),
        "export_conservation_diagnostics": all(
            comparison["native"]["maximum_conservation_residual"]
            <= runtime.conservation_tolerance
            and comparison["deployed"]["maximum_conservation_residual"]
            <= runtime.conservation_tolerance + PARITY_ATOL
            and comparison["native"]["minimum_f_matches_population"]
            and comparison["native"]["minimum_g_matches_population"]
            and comparison["deployed"]["minimum_f_matches_population"]
            and comparison["deployed"]["minimum_g_matches_population"]
            and comparison["native"]["global_conservation_maximum_absolute_residual"]
            <= runtime.conservation_tolerance
            and comparison["deployed"]["global_conservation_maximum_absolute_residual"]
            <= runtime.conservation_tolerance + PARITY_ATOL
            and comparison["fields"][6]["within_tolerance"]
            for comparison in accepted
        ),
        "export_stress_and_energy_diagnostics": all(
            comparison["native"]["stress_and_energy_diagnostics_finite"]
            and comparison["deployed"]["stress_and_energy_diagnostics_finite"]
            and comparison["native"]["energy_sum_maximum_absolute_residual"]
            <= PARITY_ATOL
            and comparison["deployed"]["energy_sum_maximum_absolute_residual"]
            <= PARITY_ATOL
            and comparison["derived_diagnostics"]["all_finite"]
            and comparison["derived_diagnostics"]["all_within_tolerance"]
            and comparison["fields"][7]["within_tolerance"]
            for comparison in comparisons
        ),
        "unavailable_backend_evidence_and_fail_closed": bool(
            availability.available
            and deployment["performed"]
            and deployment["reason"] is None
        ),
        "no_hidden_jax_fallback": bool(
            not deployment["fallback_attempted"]
            and all(
                modes[mode]["execution_engine"] == "phydrax.export._iree.IREEExecutable"
                and modes[mode]["explicit_native_reference_engine"] == "jax-native"
                and not modes[mode]["fallback_attempted"]
                for mode in modes
            )
        ),
    }


def qualification_report(
    *,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
    parent_spatial_path: Path = DEFAULT_PARENT_SPATIAL,
) -> dict[str, Any]:
    """Build deterministic forward-export qualification evidence."""

    if model_artifact_path.suffix != ".phxml":
        raise ValueError("Learned-energy export qualification requires a .phxml model.")
    input_hashes_before = {
        "model_artifact": _sha256(model_artifact_path),
        "parent_spatial": _sha256(parent_spatial_path),
    }
    parent = _read_parent_spatial(parent_spatial_path)
    artifact = read_learned_energy_equilibrium_artifact(model_artifact_path)
    runtime = _build_runtime(artifact, parent)
    environment = capture_environment()
    policy = IREEExportPolicy(target_backend="vmvx", runtime_driver="local-task")

    accepted_conserved = _accepted_conserved(runtime, artifact)
    frozen_accepted = _native_outputs(
        "frozen-equilibrium", runtime, artifact, (accepted_conserved,)
    )
    accepted_state = SmoothCompressibleKineticState(
        frozen_accepted[0],
        frozen_accepted[1],
    )
    rejected_state = _rejected_state(accepted_state, runtime, artifact)
    rejected_conserved = runtime.method.moments(rejected_state).conserved

    contracts = _prepare_contracts(
        runtime,
        artifact,
        accepted_conserved,
        accepted_state,
        environment.fingerprint,
        policy,
    )
    cases = {
        "frozen-equilibrium": {
            "accepted": (accepted_conserved,),
            "rejected": (rejected_conserved,),
        },
        "one-step": {
            "accepted": (
                accepted_state.particle_populations,
                accepted_state.total_energy_populations,
            ),
            "rejected": (
                rejected_state.particle_populations,
                rejected_state.total_energy_populations,
            ),
        },
        "fixed-horizon": {
            "accepted": (
                accepted_state.particle_populations,
                accepted_state.total_energy_populations,
            ),
            "rejected": (
                rejected_state.particle_populations,
                rejected_state.total_energy_populations,
            ),
        },
    }
    native_cases = {
        mode: {
            "accepted": _native_case_record(
                runtime,
                artifact,
                mode,
                mode_cases["accepted"],
                None if mode == "frozen-equilibrium" else accepted_state,
            ),
            "rejected": _native_case_record(
                runtime,
                artifact,
                mode,
                mode_cases["rejected"],
                None if mode == "frozen-equilibrium" else rejected_state,
            ),
        }
        for mode, mode_cases in cases.items()
    }

    availability = discrete_velocity_iree_availability()
    deployment = (
        _execute_exports(
            runtime,
            artifact,
            contracts,
            cases,
            accepted_state,
            rejected_state,
            policy,
        )
        if _matched_iree_available(availability)
        else _unavailable_exports(availability, contracts)
    )
    input_hashes_after = {
        "model_artifact": _sha256(model_artifact_path),
        "parent_spatial": _sha256(parent_spatial_path),
    }

    parent_identities = parent["identities"]
    binding = artifact.binding
    exact_periodic = bool(
        runtime.transport.spatial_shape
        == tuple(parent["configuration"]["coarse_resolution"])
        and runtime.transport.cell_spacing
        == tuple(parent["configuration"]["coarse_cell_spacing"])
        and runtime.required_step_size == parent["configuration"]["coarse_time_step"]
        and not runtime.allows_step_reduction
        and runtime.transport.plan_id == parent_identities["spatial_plan"]
        and runtime.prepared_id == parent_identities["prepared_spatial"]
    )
    stage_one_artifact = parent_identities["artifact"]
    model_is_parent_or_descendant = (
        artifact.artifact_id == stage_one_artifact
        or binding.plan.parent_artifact_id == stage_one_artifact
    )
    identity_chain = bool(
        model_is_parent_or_descendant
        and parent_identities["support"] == binding.plan.support.support_id
        and parent_identities["quadrature"] == runtime.method.quadrature.quadrature_id
        and parent_identities["material"] == runtime.method.material.material_id
        and parent_identities["transport_closure"] == runtime.method.transport.closure_id
        and parent_identities["kinetic_method"] == runtime.method.method_id
        and parent_identities["energy_plan"] == runtime.energy_plan.plan_id
    )
    native_rejected = [
        native_cases[mode]["rejected"]
        for mode in ("frozen-equilibrium", "one-step", "fixed-horizon")
    ]
    native_accepted = [
        native_cases[mode]["accepted"]
        for mode in ("frozen-equilibrium", "one-step", "fixed-horizon")
    ]
    native_rollback = [
        native_cases[mode]["rejected"] for mode in ("one-step", "fixed-horizon")
    ]
    contract_gates = _contract_gates(contracts, runtime, artifact)
    deployment_gates = _deployment_gates(availability, deployment, runtime)
    gates = {
        "parent_spatial_report_passed_and_bound": bool(
            parent["passed"]
            and parent["qualification_id"]
            == canonical_fingerprint(
                {key: value for key, value in parent.items() if key != "qualification_id"}
            )
        ),
        "model_support_runtime_identity_chain": identity_chain,
        "exact_fixed_periodic_d2v17_runtime": exact_periodic,
        "input_artifacts_unchanged": input_hashes_before == input_hashes_after,
        **contract_gates,
        "native_deliberate_rejections": all(
            not case["successful"]
            and case["rollback_applied"]
            and case["status"]
            == int(SmoothCompressibleD2VStepStatus.ENERGY_EQUILIBRIUM_FAILED)
            and case["first_failure_step"] == 0
            and case["all_outputs_finite"]
            for case in native_rejected
        ),
        "native_accepted_contracts": all(
            case["successful"]
            and not case["rollback_applied"]
            and case["status"] == int(SmoothCompressibleD2VStepStatus.SUCCESS)
            and case["first_failure_step"] == -1
            and case["all_outputs_finite"]
            for case in native_accepted
        ),
        "native_rejection_and_exact_f_g_rollback": all(
            not case["successful"]
            and case["rollback_applied"]
            and case["status"]
            == int(SmoothCompressibleD2VStepStatus.ENERGY_EQUILIBRIUM_FAILED)
            and case["first_failure_step"] == 0
            and case["accepted_f_matches_input_exactly"]
            and case["accepted_g_matches_input_exactly"]
            and case["global_conservation_maximum_absolute_residual"] == 0.0
            for case in native_rollback
        ),
        "native_bool_int_separation": all(
            mode_case["output_dtypes"][2:4] == [np.dtype(np.bool_).str] * 2
            and mode_case["output_dtypes"][4:6] == [np.dtype(np.int32).str] * 2
            and mode_case["output_dtypes"][2] != mode_case["output_dtypes"][4]
            for mode_cases in native_cases.values()
            for mode_case in mode_cases.values()
        ),
        "native_conservation_diagnostics": all(
            case["maximum_conservation_residual"]
            <= parent["thresholds"]["maximum_conservation_residual"]
            and case["global_conservation_maximum_absolute_residual"]
            <= parent["thresholds"]["maximum_conservation_residual"]
            and case["minimum_f_matches_population"]
            and case["minimum_g_matches_population"]
            for case in native_accepted
        ),
        "native_stress_and_energy_diagnostics": all(
            case["stress_and_energy_diagnostics_finite"]
            and case["energy_sum_maximum_absolute_residual"] <= PARITY_ATOL
            for mode_cases in native_cases.values()
            for case in mode_cases.values()
        ),
        "backend_availability_evidence": bool(
            availability.backend == "iree"
            and availability.capabilities.supports("compiled-inference")
            and availability.requirement
            and availability.reason
        ),
        **deployment_gates,
    }

    report = {
        "tool": "learned_energy_export_qualification",
        "scope": {
            "stage": "deterministic learned-energy forward export qualification",
            "claim": ("frozen fixed-shape periodic D2V17 forward IREE deployment only"),
            "included": [
                "frozen learned-equilibrium forward export",
                "one-step accepted-state forward export",
                "fixed-horizon accepted-state forward export",
                "ordered heterogeneous output ABI",
                "accepted and deliberately rejected native/IREE parity",
                "transactional f/g rollback",
                "conservation, stress, and total-energy diagnostics",
            ],
            "excluded": [
                "VJP export",
                "reverse-mode differentiation",
                "training export",
                "dynamic shapes",
                "nonperiodic boundaries",
                "shock or TVD qualification",
                "production-readiness claims",
            ],
        },
        "contracts": {
            "execution_modes": [
                "frozen-equilibrium",
                "one-step",
                "fixed-horizon",
            ],
            "fixed_horizon_steps": FIXED_HORIZON_STEPS,
            "output_order": list(OUTPUT_NAMES),
            "failure_transaction": (
                "one-step and fixed-horizon failure return exact predecessor f/g"
            ),
            "native_reference": "explicit JAX execution used only for comparison",
            "deployed_execution": "IREE VM only; no JAX fallback",
            "differentiation": "forward only; no VJP or training export",
        },
        "configuration": {
            "spatial_shape": list(runtime.transport.spatial_shape),
            "cell_spacing": list(runtime.transport.cell_spacing),
            "time_step": runtime.required_step_size,
            "particle_relaxation_time": PARTICLE_RELAXATION_TIME,
            "total_energy_relaxation_time": TOTAL_ENERGY_RELAXATION_TIME,
            "fixed_horizon_steps": FIXED_HORIZON_STEPS,
            "dtype": np.dtype(runtime.method.quadrature.velocities.dtype).name,
            "parity_rtol": PARITY_RTOL,
            "parity_atol": PARITY_ATOL,
            "target_backend": policy.target_backend,
            "runtime_driver": policy.runtime_driver,
        },
        "parent_spatial": {
            "path": str(parent_spatial_path),
            "qualification_id": parent["qualification_id"],
            "passed": parent["passed"],
        },
        "input_integrity": {
            "model_artifact_path": str(model_artifact_path),
            "parent_spatial_path": str(parent_spatial_path),
            "sha256_before": input_hashes_before,
            "sha256_after": input_hashes_after,
            "unchanged": input_hashes_before == input_hashes_after,
        },
        "identities": {
            "parent_spatial_qualification": parent["qualification_id"],
            "model_artifact": artifact.artifact_id,
            "prepared_binding": binding.prepared_id,
            "model_numeric_revision": binding.numeric_revision.revision_id,
            "model_semantic": binding.plan.semantic_id,
            "normalizer": binding.plan.normalizer.normalizer_id,
            "support": binding.plan.support.support_id,
            "quadrature": runtime.method.quadrature.quadrature_id,
            "material": runtime.method.material.material_id,
            "transport_closure": runtime.method.transport.closure_id,
            "kinetic_method": runtime.method.method_id,
            "energy_plan": runtime.energy_plan.plan_id,
            "spatial_plan": runtime.transport.plan_id,
            "prepared_spatial": runtime.prepared_id,
            "host": environment.fingerprint,
        },
        "prepared_contracts": {
            mode: _contract_record(contract) for mode, contract in contracts.items()
        },
        "native_cases": native_cases,
        "backend_availability": _availability_record(availability),
        "deployment": deployment,
        "environment": environment.to_dict(),
        "gates": gates,
    }
    report["passed"] = all(gates.values())
    report["qualification_status"] = "qualified" if report["passed"] else "unqualified"
    report["qualification_id"] = canonical_fingerprint(report)
    canonical_json(report)
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify frozen learned-energy D2V17 forward IREE export without claiming VJP or training export."
        )
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=DEFAULT_MODEL_ARTIFACT,
        help="checksum-validated .phxml learned energy-equilibrium model",
    )
    parser.add_argument(
        "--parent-spatial",
        type=Path,
        default=DEFAULT_PARENT_SPATIAL,
        help="passing learned-energy spatial qualification JSON",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="canonical JSON written atomically only when every gate passes",
    )
    return parser.parse_args()


def _validate_output_path(arguments: argparse.Namespace, /) -> None:
    output = arguments.output.resolve()
    protected = {
        arguments.model_artifact.resolve(),
        arguments.parent_spatial.resolve(),
        DEFAULT_MODEL_ARTIFACT.resolve(),
        DEFAULT_PARENT_SPATIAL.resolve(),
        Path("benchmarks/learned_energy_equilibrium.json").resolve(),
        Path("benchmarks/learned_energy_boundary.json").resolve(),
        Path("benchmarks/learned_energy_forcing.json").resolve(),
        Path("benchmarks/learned_energy_stage_two.json").resolve(),
    }
    if output in protected:
        raise ValueError(
            "Export qualification output must not overwrite an earlier artifact."
        )
    if arguments.output.suffix != ".json":
        raise ValueError("Export qualification output must be a .json path.")


def main() -> int:
    arguments = _parse_arguments()
    _validate_output_path(arguments)
    with jax.enable_x64(True):
        report = qualification_report(
            model_artifact_path=arguments.model_artifact,
            parent_spatial_path=arguments.parent_spatial,
        )
    write_json_atomic(arguments.output, report)
    if not report["passed"]:
        print(json.dumps(report, allow_nan=False, ensure_ascii=True, sort_keys=True))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
