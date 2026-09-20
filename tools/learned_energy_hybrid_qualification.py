#!/usr/bin/env python3
#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax.ein as ein
from benchmarks._io import write_json_atomic
from benchmarks._runtime import capture_environment
from phydrax._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_json,
)
from phydrax.applications.compressible_flow._unsteady import CompressibleShockTrackPlan
from phydrax.closure_data._kinetic_equilibrium import (
    EnergyEquilibriumSupportEvidence,
    PreparedLearnedEnergyEquilibriumBinding,
)
from phydrax.closure_data._kinetic_equilibrium_artifact import (
    LearnedEnergyEquilibriumArtifact,
    read_learned_energy_equilibrium_artifact,
)
from phydrax.discretization._axis import TensorGridPlan, UniformCellAxisSpec
from phydrax.discretization.discrete_velocity._hybrid import (
    FixedConformingFVKineticInterfacePlan,
)
from phydrax.discretization.discrete_velocity._hybrid_runtime import (
    DynamicHybridCompositeState,
    DynamicHybridOwnershipPlan,
    DynamicHybridOwnershipState,
    FixedHybridStageEvidence,
    FixedPartitionHybridState,
    FixedPartitionHybridStatus,
    PreparedFixedPartitionHybridRuntime,
)
from phydrax.discretization.discrete_velocity._smooth_compressible import (
    SmoothCompressibleD2VKineticMethod,
    SmoothCompressibleKineticState,
    SmoothCompressibleLearnedEquilibriumEvidence,
    SmoothCompressibleRealizabilityEvidence,
)
from phydrax.discretization.discrete_velocity._spatial import (
    D2V17PeriodicTransportPlan,
    PreparedSmoothCompressibleD2V17SpatialDynamics,
)
from phydrax.discretization.finite_volume import (
    ExtrapolationBoundary,
    FiniteVolumeBoundaryPair,
    FiniteVolumeBoundarySet,
    FiniteVolumeMethodPlan,
    FiniteVolumePlan,
    FluxPositivityPlan,
    HLLCFluxPlan,
    MUSCLReconstruction,
)
from phydrax.equations._conservation import (
    compile_conservation_problem,
    ConservationProblemIR,
)
from phydrax.equations._hyperbolic_systems import EulerSystem
from phydrax.equations._transport_closures import ConstantTransport
from phydrax.solver._finite_volume_runtime import (
    FiniteVolumeStepPolicy,
    PreparedFiniteVolumeRuntime,
)


DEFAULT_PARENT_SPATIAL = Path("benchmarks/learned_energy_spatial.json")
DEFAULT_PARENT_STAGE_TWO = Path("benchmarks/learned_energy_stage_two.json")
DEFAULT_MODEL_ARTIFACT = Path("benchmarks/learned_energy_stage_two.phxml")
OUTPUT_PATH = Path("benchmarks/learned_energy_hybrid.json")

SPATIAL_SHAPE = (81, 5)
CELL_SPACING = (0.01, 0.01)
FIXED_STEP_SIZE = 0.01
FINITE_VOLUME_CFL = 4.0
RIEMANN_STEP_COUNT = 4
RIEMANN_AMPLITUDE_FRACTION = 0.025
SSPRK3_WEIGHTS = (1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0)
PARTICLE_RELAXATION_TIME = 0.03
TOTAL_ENERGY_RELAXATION_TIME = 0.04

THRESHOLDS = {
    "maximum_interface_residual": 5.0e-10,
    "maximum_global_conservation_residual": 5.0e-10,
    "maximum_migration_conservation_residual": 5.0e-10,
    "maximum_fixed_fv_region_relative_l2_error": 5.0e-6,
    "maximum_feature_location_error": CELL_SPACING[0],
    "minimum_population": 0.0,
    "minimum_support_margin": 0.0,
}


class _ExpectedRefusal:
    """Capture one specifically named refusal without accepting foreign failures."""

    def __init__(self, exception_type: type[BaseException], message_fragment: str, /):
        self.exception_type = exception_type
        self.message_fragment = str(message_fragment)
        self.observed = False
        self.actual_type: str | None = None
        self.message: str | None = None

    def __enter__(self) -> _ExpectedRefusal:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del traceback
        if exception_type is None or exception is None:
            return False
        if not issubclass(exception_type, self.exception_type):
            return False
        message = str(exception)
        if self.message_fragment not in message:
            return False
        self.observed = True
        self.actual_type = exception_type.__name__
        self.message = message
        return True

    def record(self) -> dict[str, Any]:
        return {
            "observed": self.observed,
            "expected_type": self.exception_type.__name__,
            "actual_type": self.actual_type,
            "required_message_fragment": self.message_fragment,
            "message": self.message,
        }


def _mapping(value: Any, owner: str, /) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{owner} must be a JSON object.")
    return value


def _identifier(value: Any, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty identifier.")
    return value


def _strict_bool(value: Any, owner: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{owner} must be a JSON boolean.")
    return value


def _read_json_object(path: Path, owner: str, /) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), owner)


def _parent_records(
    spatial_path: Path,
    stage_two_path: Path,
    artifact: LearnedEnergyEquilibriumArtifact,
    /,
) -> dict[str, Any]:
    spatial = _read_json_object(spatial_path, "Spatial parent")
    stage_two = _read_json_object(stage_two_path, "Stage-two parent")
    if spatial["tool"] != "learned_energy_spatial_qualification":
        raise ValueError(
            "Spatial parent was not produced by the spatial qualification tool."
        )
    if stage_two["kind"] != "learned-energy-stage-two-qualification":
        raise ValueError("Stage-two parent has the wrong qualification kind.")
    spatial_ids = _mapping(spatial["identities"], "Spatial parent identities")
    stage_two_ids = _mapping(stage_two["identities"], "Stage-two parent identities")
    binding = artifact.binding
    plan = binding.plan
    compatibility = {
        "spatial_parent_passed": _strict_bool(spatial["passed"], "Spatial parent passed"),
        "stage_two_parent_passed": _strict_bool(
            stage_two["passed"], "Stage-two parent passed"
        ),
        "spatial_stage_one_artifact_matches_stage_two_parent": (
            _identifier(spatial_ids["artifact"], "Spatial model artifact")
            == _identifier(
                stage_two_ids["stage_one_artifact_id"],
                "Stage-two stage-one artifact",
            )
        ),
        "spatial_stage_one_revision_matches_stage_two_parent": (
            _identifier(spatial_ids["model_numeric_revision"], "Spatial model revision")
            == _identifier(
                stage_two_ids["stage_one_model_revision_id"],
                "Stage-two stage-one model revision",
            )
        ),
        "loaded_artifact_matches_stage_two_parent": (
            artifact.artifact_id
            == _identifier(stage_two_ids["stage_two_artifact_id"], "Stage-two artifact")
        ),
        "loaded_binding_matches_stage_two_parent": (
            binding.prepared_id
            == _identifier(
                stage_two_ids["stage_two_prepared_binding_id"],
                "Stage-two prepared binding",
            )
        ),
        "loaded_revision_matches_stage_two_parent": (
            binding.numeric_revision.revision_id
            == _identifier(
                stage_two_ids["final_model_revision_id"],
                "Stage-two final model revision",
            )
        ),
        "loaded_binding_plan_matches_stage_two_parent": (
            plan.plan_id
            == _identifier(
                stage_two_ids["stage_two_binding_plan_id"],
                "Stage-two binding plan",
            )
        ),
        "loaded_semantic_matches_stage_two_parent": (
            plan.semantic_id
            == _identifier(
                stage_two_ids["stage_two_semantic_id"],
                "Stage-two semantic identity",
            )
        ),
        "loaded_parent_artifact_matches_stage_two_parent": (
            plan.parent_artifact_id
            == _identifier(
                stage_two_ids["stage_one_artifact_id"],
                "Stage-two stage-one artifact",
            )
        ),
        "quadrature_matches_spatial_parent": (
            plan.equilibrium_plan.quadrature.quadrature_id
            == _identifier(spatial_ids["quadrature"], "Spatial quadrature")
        ),
        "material_matches_spatial_parent": (
            plan.material.material_id
            == _identifier(spatial_ids["material"], "Spatial material")
        ),
        "energy_plan_matches_spatial_parent": (
            plan.equilibrium_plan.plan_id
            == _identifier(spatial_ids["energy_plan"], "Spatial energy plan")
        ),
        "support_matches_both_parents": (
            plan.support.support_id
            == _identifier(spatial_ids["support"], "Spatial support")
            == _identifier(stage_two_ids["support_id"], "Stage-two support")
        ),
        "normalizer_matches_stage_two_parent": (
            plan.normalizer.normalizer_id
            == _identifier(stage_two_ids["normalizer_id"], "Stage-two normalizer")
        ),
    }
    parent_record = {
        "spatial": {
            "path": str(spatial_path),
            "qualification_id": _identifier(
                spatial["qualification_id"], "Spatial qualification"
            ),
            "prepared_runtime_id": _identifier(
                spatial_ids["prepared_spatial"], "Spatial prepared runtime"
            ),
            "transport_plan_id": _identifier(
                spatial_ids["spatial_plan"], "Spatial transport plan"
            ),
            "model_artifact_id": _identifier(
                spatial_ids["artifact"], "Spatial model artifact"
            ),
            "model_revision_id": _identifier(
                spatial_ids["model_numeric_revision"], "Spatial model revision"
            ),
            "prepared_binding_id": _identifier(
                spatial_ids["prepared_binding"], "Spatial prepared binding"
            ),
        },
        "stage_two": {
            "path": str(stage_two_path),
            "qualification_artifact_id": _identifier(
                stage_two["artifact_id"], "Stage-two qualification artifact"
            ),
            "training_runtime_id": _identifier(
                stage_two_ids["runtime_id"], "Stage-two training runtime"
            ),
            "training_plan_id": _identifier(
                stage_two_ids["training_plan_id"], "Stage-two training plan"
            ),
            "rollout_preparation_id": _identifier(
                stage_two_ids["rollout_preparation_id"],
                "Stage-two rollout preparation",
            ),
            "model_artifact_id": _identifier(
                stage_two_ids["stage_two_artifact_id"], "Stage-two model artifact"
            ),
            "model_revision_id": _identifier(
                stage_two_ids["final_model_revision_id"],
                "Stage-two final model revision",
            ),
            "prepared_binding_id": _identifier(
                stage_two_ids["stage_two_prepared_binding_id"],
                "Stage-two prepared binding",
            ),
            "binding_plan_id": _identifier(
                stage_two_ids["stage_two_binding_plan_id"],
                "Stage-two binding plan",
            ),
            "semantic_id": _identifier(
                stage_two_ids["stage_two_semantic_id"],
                "Stage-two semantic identity",
            ),
        },
        "loaded_frozen_model": {
            "artifact_id": artifact.artifact_id,
            "prepared_binding_id": binding.prepared_id,
            "binding_plan_id": plan.plan_id,
            "numeric_revision_id": binding.numeric_revision.revision_id,
            "semantic_id": plan.semantic_id,
            "support_id": plan.support.support_id,
            "parent_artifact_id": plan.parent_artifact_id,
        },
        "compatibility": compatibility,
        "all_compatible": all(compatibility.values()),
    }
    return parent_record


def _tree_exact(left: Any, right: Any, /) -> bool:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves):
        return False
    for left_value, right_value in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_value)
        right_array = np.asarray(right_value)
        if left_array.shape != right_array.shape or left_array.dtype != right_array.dtype:
            return False
        equal = (
            np.array_equal(left_array, right_array, equal_nan=True)
            if np.issubdtype(left_array.dtype, np.inexact)
            else np.array_equal(left_array, right_array)
        )
        if not equal:
            return False
    return True


def _tree_maximum_difference(left: Any, right: Any, /) -> float:
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves):
        return math.inf
    maximum = 0.0
    for left_value, right_value in zip(left_leaves, right_leaves, strict=True):
        left_array = np.asarray(left_value)
        right_array = np.asarray(right_value)
        if left_array.shape != right_array.shape:
            return math.inf
        if left_array.dtype.kind == "b" or right_array.dtype.kind == "b":
            difference = float(np.max(left_array != right_array, initial=False))
        else:
            difference = float(np.max(np.abs(left_array - right_array), initial=0.0))
        maximum = max(maximum, difference)
    return maximum


def _maximum_absolute(value: Any, /) -> float:
    return float(np.max(np.abs(np.asarray(value)), initial=0.0))


def _minimum(value: Any, /) -> float:
    return float(np.min(np.asarray(value), initial=math.inf))


def _relative_l2(candidate: Any, reference: Any, /) -> float:
    candidate_array = np.asarray(candidate)
    reference_array = np.asarray(reference)
    numerator = float(np.linalg.norm((candidate_array - reference_array).reshape(-1)))
    denominator = max(
        float(np.linalg.norm(reference_array.reshape(-1))), np.finfo(np.float64).tiny
    )
    return numerator / denominator


def _support_summary(evidence: EnergyEquilibriumSupportEvidence, /) -> dict[str, Any]:
    margins = {
        "rho": _minimum(evidence.rho_margin),
        "u_x": _minimum(evidence.u_x_margin),
        "u_y": _minimum(evidence.u_y_margin),
        "temperature": _minimum(evidence.temperature_margin),
        "mach": _minimum(evidence.mach_margin),
        "hull": _minimum(evidence.hull_margin),
        "particle_equilibrium": _minimum(evidence.particle_equilibrium_margin),
    }
    return {
        "all_successful": bool(np.asarray(jnp.all(evidence.successful))),
        "all_finite": bool(np.asarray(jnp.all(evidence.finite))),
        "all_primitive_supported": bool(
            np.asarray(jnp.all(evidence.primitive_supported))
        ),
        "all_model_outputs_finite": bool(
            np.asarray(jnp.all(evidence.model_output_finite))
        ),
        "minimum_margins": margins,
        "minimum_margin": min(margins.values()),
        "maximum_mach": float(np.max(np.asarray(evidence.mach), initial=0.0)),
        "support_id": evidence.support_id,
    }


def _support_summary_many(
    evidence: Sequence[EnergyEquilibriumSupportEvidence], /
) -> dict[str, Any]:
    summaries = tuple(_support_summary(value) for value in evidence)
    return {
        "all_successful": all(value["all_successful"] for value in summaries),
        "minimum_margin": min(value["minimum_margin"] for value in summaries),
        "interfaces": list(summaries),
    }


def _realizability_summary(
    evidence: SmoothCompressibleRealizabilityEvidence, /
) -> dict[str, Any]:
    return {
        "realizable": bool(np.asarray(evidence.realizable)),
        "all_local_realizable": bool(np.asarray(jnp.all(evidence.local_realizable))),
        "all_finite": bool(np.asarray(jnp.all(evidence.finite))),
        "all_macroscopic_admissible": bool(
            np.asarray(jnp.all(evidence.macroscopic_admissible))
        ),
        "all_populations_nonnegative": bool(
            np.asarray(jnp.all(evidence.populations_nonnegative))
        ),
        "minimum_density": _minimum(evidence.density),
        "minimum_pressure": _minimum(evidence.pressure),
        "minimum_temperature": _minimum(evidence.temperature),
        "minimum_particle_population": _minimum(evidence.minimum_particle_population),
        "minimum_total_energy_population": _minimum(
            evidence.minimum_total_energy_population
        ),
    }


def _method(
    binding: PreparedLearnedEnergyEquilibriumBinding, /
) -> SmoothCompressibleD2VKineticMethod:
    return SmoothCompressibleD2VKineticMethod(
        binding.plan.equilibrium_plan.quadrature,
        binding.plan.material,
        ConstantTransport(PARTICLE_RELAXATION_TIME, TOTAL_ENERGY_RELAXATION_TIME),
    )


def _finite_volume_runtime(
    method: SmoothCompressibleD2VKineticMethod,
    /,
    *,
    maximum_retries: int,
) -> PreparedFiniteVolumeRuntime:
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(value) for value in SPATIAL_SHAPE),
        axis_names=("x", "y"),
    ).prepare(
        jnp.asarray(
            (
                (0.0, 0.0),
                (
                    SPATIAL_SHAPE[0] * CELL_SPACING[0],
                    SPATIAL_SHAPE[1] * CELL_SPACING[1],
                ),
            ),
            dtype=jnp.float64,
        )
    )
    system = EulerSystem(2, material=method.material)
    discretization = FiniteVolumePlan(
        grid,
        component_names=system.component_names,
    ).prepare()
    extrapolation = FiniteVolumeBoundaryPair(
        ExtrapolationBoundary(), ExtrapolationBoundary()
    )
    problem = ConservationProblemIR(
        "learned-energy-static-hybrid-low-amplitude-riemann",
        "state",
        system,
        FiniteVolumeBoundarySet(
            ("x", "y"),
            (extrapolation, extrapolation),
        ),
    )
    method_plan = FiniteVolumeMethodPlan(MUSCLReconstruction(), HLLCFluxPlan())
    dynamics = compile_conservation_problem(problem, discretization, method_plan).dynamics
    return PreparedFiniteVolumeRuntime(
        dynamics,
        FluxPositivityPlan(),
        FiniteVolumeStepPolicy(
            cfl=FINITE_VOLUME_CFL,
            maximum_retries=maximum_retries,
        ),
    )


def _interfaces(
    finite_volume: PreparedFiniteVolumeRuntime,
    method: SmoothCompressibleD2VKineticMethod,
    /,
) -> tuple[
    tuple[FixedConformingFVKineticInterfacePlan, ...],
    tuple[int, ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
    tuple[tuple[int, int], ...],
]:
    rows = SPATIAL_SHAPE[1]
    upper = tuple(
        FixedConformingFVKineticInterfacePlan(
            method,
            finite_volume.dynamics.system,
            jnp.asarray((1.0, 0.0), dtype=jnp.float64),
            f"fixed-fv-upper-x-{y_index}",
        )
        for y_index in range(rows)
    )
    lower = tuple(
        FixedConformingFVKineticInterfacePlan(
            method,
            finite_volume.dynamics.system,
            jnp.asarray((-1.0, 0.0), dtype=jnp.float64),
            f"fixed-fv-lower-x-{y_index}",
        )
        for y_index in range(rows)
    )
    plans = (*upper, *lower)
    axes = (0,) * len(plans)
    face_indices = (
        *((SPATIAL_SHAPE[0], y_index) for y_index in range(rows)),
        *((0, y_index) for y_index in range(rows)),
    )
    finite_volume_cells = (
        *((SPATIAL_SHAPE[0] - 1, y_index) for y_index in range(rows)),
        *((0, y_index) for y_index in range(rows)),
    )
    kinetic_cells = (
        *((0, y_index) for y_index in range(rows)),
        *((SPATIAL_SHAPE[0] - 1, y_index) for y_index in range(rows)),
    )
    return plans, axes, face_indices, finite_volume_cells, kinetic_cells


def _fixed_runtime(
    binding: PreparedLearnedEnergyEquilibriumBinding,
    /,
    *,
    maximum_retries: int,
) -> PreparedFixedPartitionHybridRuntime:
    method = _method(binding)
    transport = D2V17PeriodicTransportPlan(
        method.quadrature,
        SPATIAL_SHAPE,
        CELL_SPACING,
        FIXED_STEP_SIZE,
    )
    spatial = PreparedSmoothCompressibleD2V17SpatialDynamics(
        method,
        binding.plan.equilibrium_plan,
        transport,
        conservation_tolerance=THRESHOLDS["maximum_global_conservation_residual"],
    )
    finite_volume = _finite_volume_runtime(method, maximum_retries=maximum_retries)
    plans, axes, face_indices, finite_volume_cells, kinetic_cells = _interfaces(
        finite_volume, method
    )
    return PreparedFixedPartitionHybridRuntime(
        finite_volume,
        spatial,
        binding,
        plans,
        axes,
        face_indices,
        finite_volume_cells,
        kinetic_cells,
        conservation_tolerance=THRESHOLDS["maximum_global_conservation_residual"],
    )


def _primitive_to_conserved(
    system: EulerSystem,
    rho: Any,
    u_x: Any,
    u_y: Any,
    temperature: Any,
    /,
) -> jax.Array:
    rho_array, u_x_array, u_y_array, temperature_array = jnp.broadcast_arrays(
        jnp.asarray(rho, dtype=jnp.float64),
        jnp.asarray(u_x, dtype=jnp.float64),
        jnp.asarray(u_y, dtype=jnp.float64),
        jnp.asarray(temperature, dtype=jnp.float64),
    )
    pressure = rho_array * system.material.gas_constant * temperature_array
    primitive = jnp.stack((rho_array, u_x_array, u_y_array, pressure), axis=-1)
    return system.primitive_to_conserved(primitive)


def _riemann_states(
    runtime: PreparedFixedPartitionHybridRuntime, /
) -> tuple[jax.Array, jax.Array, dict[str, Any]]:
    support = runtime.learned_energy.plan.support
    fraction = RIEMANN_AMPLITUDE_FRACTION
    rho_midpoint = 0.5 * sum(support.rho_bounds)
    temperature_midpoint = 0.5 * sum(support.temperature_bounds)
    u_x_midpoint = 0.5 * sum(support.u_x_bounds)
    u_y_midpoint = 0.5 * sum(support.u_y_bounds)
    rho_delta = fraction * (support.rho_bounds[1] - support.rho_bounds[0])
    temperature_delta = fraction * (
        support.temperature_bounds[1] - support.temperature_bounds[0]
    )
    system = runtime.finite_volume.dynamics.system
    left = _primitive_to_conserved(
        system,
        rho_midpoint + rho_delta,
        u_x_midpoint,
        u_y_midpoint,
        temperature_midpoint + temperature_delta,
    )
    right = _primitive_to_conserved(
        system,
        rho_midpoint - rho_delta,
        u_x_midpoint,
        u_y_midpoint,
        temperature_midpoint - temperature_delta,
    )
    record = {
        "amplitude_regime": "low",
        "fraction_of_declared_rho_and_temperature_span": fraction,
        "left_primitive": np.asarray(system.conserved_to_primitive(left)).tolist(),
        "right_primitive": np.asarray(system.conserved_to_primitive(right)).tolist(),
    }
    return left, right, record


def _riemann_field(
    runtime: PreparedFixedPartitionHybridRuntime,
    left: jax.Array,
    right: jax.Array,
    /,
) -> jax.Array:
    centers = runtime.finite_volume.dynamics.discretization.cell_centers
    discontinuity = 0.5 * SPATIAL_SHAPE[0] * CELL_SPACING[0]
    return jnp.where((centers[..., 0] < discontinuity)[..., None], left, right)


def _learned_equilibrium(
    method: SmoothCompressibleD2VKineticMethod,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    conserved: Any,
    /,
) -> tuple[
    SmoothCompressibleKineticState,
    EnergyEquilibriumSupportEvidence,
    SmoothCompressibleLearnedEquilibriumEvidence,
]:
    dual, support = binding.predict_dual_with_evidence(conserved)
    equilibrium, evidence = method.equilibrium_from_energy_dual_with_evidence(
        conserved,
        dual,
        binding.plan.equilibrium_plan,
    )
    return equilibrium, support, evidence


def _fixed_initial_state(
    runtime: PreparedFixedPartitionHybridRuntime,
    finite_volume_average: jax.Array,
    kinetic_conserved: jax.Array,
    /,
) -> tuple[
    FixedPartitionHybridState,
    EnergyEquilibriumSupportEvidence,
    SmoothCompressibleLearnedEquilibriumEvidence,
]:
    finite_volume = runtime.finite_volume.initialize_state(
        finite_volume_average,
        0.0,
        FIXED_STEP_SIZE,
    )
    kinetic, support, evidence = _learned_equilibrium(
        runtime.spatial.method,
        runtime.learned_energy,
        kinetic_conserved,
    )
    return runtime.initialize_state(finite_volume, kinetic), support, evidence


def _population_moment_flux(
    method: SmoothCompressibleD2VKineticMethod,
    particle_flux: jax.Array,
    energy_flux: jax.Array,
    /,
) -> jax.Array:
    return jnp.concatenate(
        (
            jnp.sum(particle_flux, axis=-1)[..., None],
            ein.contract("...q,qd->...d", particle_flux, method.quadrature.velocities),
            jnp.sum(energy_flux, axis=-1)[..., None],
        ),
        axis=-1,
    )


def _stage_record(
    runtime: PreparedFixedPartitionHybridRuntime,
    stage: Any,
    stage_index: int,
    /,
) -> dict[str, Any]:
    evidence = stage.evidence
    if not isinstance(evidence, FixedHybridStageEvidence):
        raise TypeError("Static hybrid trace retained foreign stage evidence.")
    common = evidence.common_fluxes
    particle_flux = jnp.stack(
        tuple(value.particle_population_flux for value in common), axis=0
    )
    energy_flux = jnp.stack(
        tuple(value.total_energy_population_flux for value in common), axis=0
    )
    conservative_flux = jnp.stack(
        tuple(value.common_conservative_flux for value in common), axis=0
    )
    recovered = _population_moment_flux(
        runtime.spatial.method, particle_flux, energy_flux
    )
    replacement_count = sum(
        int(np.count_nonzero(np.asarray(mask))) for mask in stage.replacement_masks
    )
    return {
        "phase": f"ssprk3_stage_{stage_index}",
        "stage_index": stage_index,
        "weight": SSPRK3_WEIGHTS[stage_index],
        "interface_count": len(common),
        "active_replacement_face_count": replacement_count,
        "all_learned_lifts_present": all(
            value.learned_lift_evidence is not None for value in common
        ),
        "all_learned_lifts_successful": all(
            bool(np.asarray(value.learned_lift_evidence.successful))
            for value in common
            if value.learned_lift_evidence is not None
        ),
        "support": _support_summary_many(evidence.learned_support),
        "successful": bool(np.asarray(evidence.successful)),
        "maximum_particle_population_flux": _maximum_absolute(particle_flux),
        "maximum_total_energy_population_flux": _maximum_absolute(energy_flux),
        "maximum_common_conservative_flux": _maximum_absolute(conservative_flux),
        "maximum_population_to_conservative_flux_residual": _maximum_absolute(
            recovered - conservative_flux
        ),
        "maximum_trace_common_flux_residual": _maximum_absolute(
            stage.interface_conservative_flux - conservative_flux
        ),
        "maximum_equal_opposite_flux_residual": max(
            _maximum_absolute(value.flux_equality_residual) for value in common
        ),
        "maximum_moment_lift_residual": max(
            _maximum_absolute(value.moment_lift_residual) for value in common
        ),
    }


def _exchange_record(
    runtime: PreparedFixedPartitionHybridRuntime,
    result: Any,
    /,
) -> dict[str, float]:
    del runtime
    audit = result.evidence.audit
    return {
        "maximum_link_route_moment_residual": _maximum_absolute(
            audit.kinetic_moment_exchange_residual
        ),
        "maximum_finite_volume_interface_residual": _maximum_absolute(
            audit.finite_volume_interface_flux_residual
        ),
        "maximum_global_single_application_residual": _maximum_absolute(
            audit.global_conservation_residual
        ),
        "maximum_unowned_outer_boundary_exchange": _maximum_absolute(
            audit.kinetic_outer_boundary_exchange
        ),
    }


def _fixed_step_record(
    runtime: PreparedFixedPartitionHybridRuntime,
    result: Any,
    step_index: int,
    /,
) -> dict[str, Any]:
    stages = [
        _stage_record(runtime, stage, stage_index)
        for stage_index, stage in enumerate(result.stage_flux_trace.stages)
    ]
    finite_volume_average = result.candidate.finite_volume.cell_average()
    finite_volume_pressure = runtime.finite_volume.dynamics.system.pressure(
        finite_volume_average
    )
    _, accepted_support = runtime.learned_energy.predict_dual_with_evidence(
        runtime.spatial.method.moments(result.runtime_state.kinetic).conserved
    )
    return {
        "step_index": step_index,
        "accepted": bool(np.asarray(result.evidence.accepted)),
        "rollback_applied": bool(np.asarray(result.evidence.rollback_applied)),
        "status": int(np.asarray(result.evidence.status)),
        "status_name": FixedPartitionHybridStatus(
            int(np.asarray(result.evidence.status))
        ).name,
        "exact_step": bool(np.asarray(result.evidence.exact_step)),
        "accepted_step_size": float(
            np.asarray(result.finite_volume.attempted.accepted_step_size)
        ),
        "finite_volume_retries": int(np.asarray(result.finite_volume.attempted.retries)),
        "finite_volume_positive": bool(
            np.asarray(result.evidence.finite_volume_positive)
        ),
        "minimum_finite_volume_density": _minimum(finite_volume_average[..., 0]),
        "minimum_finite_volume_pressure": _minimum(finite_volume_pressure),
        "kinetic_collision_accepted": bool(
            np.asarray(result.evidence.kinetic_collision_accepted)
        ),
        "learned_lifts_accepted": bool(
            np.asarray(result.evidence.learned_lifts_accepted)
        ),
        "kinetic_realizability": _realizability_summary(
            result.evidence.kinetic_realizability
        ),
        "accepted_kinetic_support": _support_summary(accepted_support),
        "stages": stages,
        "exchange": _exchange_record(runtime, result),
        "audit": {
            "maximum_kinetic_moment_exchange_residual": _maximum_absolute(
                result.evidence.audit.kinetic_moment_exchange_residual
            ),
            "maximum_finite_volume_ledger_residual": _maximum_absolute(
                result.evidence.audit.finite_volume_interface_flux_residual
            ),
            "maximum_global_conservation_residual": _maximum_absolute(
                result.evidence.audit.global_conservation_residual
            ),
            "maximum_absolute_residual": float(
                np.asarray(result.evidence.audit.maximum_absolute_residual)
            ),
        },
        "shock_owner": result.evidence.shock_owner,
        "ownership_differentiability": result.evidence.ownership_differentiability,
    }


def _run_fixed_hybrid(
    runtime: PreparedFixedPartitionHybridRuntime,
    initial: FixedPartitionHybridState,
    step_count: int,
    /,
) -> tuple[FixedPartitionHybridState, list[dict[str, Any]], Any]:
    state = initial
    records: list[dict[str, Any]] = []
    last_result: Any = None
    for step_index in range(step_count):
        result = runtime.advance(state, jnp.asarray(FIXED_STEP_SIZE, dtype=jnp.float64))
        records.append(_fixed_step_record(runtime, result, step_index))
        last_result = result
        state = result.runtime_state
        if not bool(np.asarray(result.evidence.accepted)):
            break
    return state, records, last_result


def _run_pure_finite_volume(
    runtime: PreparedFiniteVolumeRuntime,
    initial_average: jax.Array,
    step_count: int,
    /,
) -> tuple[Any, list[dict[str, Any]]]:
    state = runtime.initialize_state(initial_average, 0.0, FIXED_STEP_SIZE)
    records = []
    for step_index in range(step_count):
        result = runtime.advance_prescribed(
            state, jnp.asarray(FIXED_STEP_SIZE, dtype=jnp.float64)
        )
        records.append(
            {
                "step_index": step_index,
                "accepted": bool(np.asarray(result.accepted)),
                "accepted_step_size": float(
                    np.asarray(result.attempted.accepted_step_size)
                ),
                "retries": int(np.asarray(result.attempted.retries)),
                "positivity_valid": bool(
                    np.asarray(result.attempted.positivity.limited_state_valid)
                ),
                "status": int(np.asarray(result.attempted.runtime_state.last_status)),
            }
        )
        state = result.runtime_state
        if not bool(np.asarray(result.accepted)):
            break
    return state, records


def _feature_record(
    coordinates: jax.Array,
    hybrid_quantity: jax.Array,
    pure_fv_quantity: jax.Array,
    interval: tuple[float, float],
    quantity_name: str,
    /,
) -> dict[str, Any]:
    coordinate_array = np.asarray(coordinates)
    reference_array = np.asarray(pure_fv_quantity)
    midpoints = 0.5 * (coordinate_array[:-1] + coordinate_array[1:])
    gradients = np.diff(reference_array) / np.diff(coordinate_array)
    selected = (midpoints >= interval[0]) & (midpoints <= interval[1])
    if not np.any(selected):
        raise ValueError("Feature tracking interval contains no finite-volume face.")
    selected_gradients = gradients[selected]
    reference_gradient = selected_gradients[int(np.argmax(np.abs(selected_gradients)))]
    compression_sign = 1 if reference_gradient >= 0.0 else -1
    tracker = CompressibleShockTrackPlan(
        interval,
        compression_sign=compression_sign,
        minimum_gradient=0.0,
    )
    hybrid = tracker.evaluate(coordinates, hybrid_quantity)
    pure_fv = tracker.evaluate(coordinates, pure_fv_quantity)
    hybrid_location = float(np.asarray(hybrid.location))
    pure_fv_location = float(np.asarray(pure_fv.location))
    return {
        "tracker": "CompressibleShockTrackPlan",
        "tracker_plan_id": tracker.plan_id,
        "quantity": quantity_name,
        "definition": (
            "strongest discrete gradient in the pure-FV reference direction within the declared interval"
        ),
        "compression_sign": compression_sign,
        "search_interval": list(interval),
        "hybrid": {
            "location": hybrid_location,
            "face_index": int(np.asarray(hybrid.face_index)),
            "gradient_score": float(np.asarray(hybrid.pressure_gradient)),
            "peak_margin": float(np.asarray(hybrid.peak_margin)),
            "successful": bool(np.asarray(hybrid.successful)),
        },
        "pure_finite_volume": {
            "location": pure_fv_location,
            "face_index": int(np.asarray(pure_fv.face_index)),
            "gradient_score": float(np.asarray(pure_fv.pressure_gradient)),
            "peak_margin": float(np.asarray(pure_fv.peak_margin)),
            "successful": bool(np.asarray(pure_fv.successful)),
        },
        "absolute_location_difference": abs(hybrid_location - pure_fv_location),
    }


def _riemann_comparison(
    runtime: PreparedFixedPartitionHybridRuntime,
    hybrid_state: FixedPartitionHybridState,
    pure_fv_state: Any,
    hybrid_steps: Sequence[Mapping[str, Any]],
    pure_fv_steps: Sequence[Mapping[str, Any]],
    initial_record: Mapping[str, Any],
    /,
) -> dict[str, Any]:
    system = runtime.finite_volume.dynamics.system
    hybrid_average = hybrid_state.finite_volume.cell_average().reshape(
        (*SPATIAL_SHAPE, 4)
    )
    pure_average = pure_fv_state.cell_average().reshape((*SPATIAL_SHAPE, 4))
    hybrid_primitive = system.conserved_to_primitive(hybrid_average)
    pure_primitive = system.conserved_to_primitive(pure_average)
    coordinates = runtime.finite_volume.dynamics.discretization.cell_centers[:, 0, 0]
    discontinuity = 0.5 * SPATIAL_SHAPE[0] * CELL_SPACING[0]
    rarefaction_interval = (discontinuity - 0.065, discontinuity - 0.012)
    contact_interval = (discontinuity - 0.012, discontinuity + 0.012)
    shock_interval = (discontinuity + 0.012, discontinuity + 0.065)
    hybrid_density = jnp.mean(hybrid_primitive[..., 0], axis=1)
    pure_density = jnp.mean(pure_primitive[..., 0], axis=1)
    hybrid_pressure = jnp.mean(hybrid_primitive[..., -1], axis=1)
    pure_pressure = jnp.mean(pure_primitive[..., -1], axis=1)
    features = {
        "rarefaction": _feature_record(
            coordinates,
            hybrid_pressure,
            pure_pressure,
            rarefaction_interval,
            "pressure",
        ),
        "contact": _feature_record(
            coordinates,
            hybrid_density,
            pure_density,
            contact_interval,
            "density",
        ),
        "shock": _feature_record(
            coordinates,
            hybrid_pressure,
            pure_pressure,
            shock_interval,
            "pressure",
        ),
    }
    comparison_window = (coordinates >= discontinuity - 0.08) & (
        coordinates <= discontinuity + 0.08
    )
    window_hybrid = hybrid_average[comparison_window]
    window_pure = pure_average[comparison_window]
    return {
        "case": "low-amplitude Riemann-style discontinuity",
        "initial_state": dict(initial_record),
        "fixed_fv_region": {
            "definition": "central FV cells separated from the FV/kinetic interface",
            "coordinate_interval": [discontinuity - 0.08, discontinuity + 0.08],
            "interface_coordinate": SPATIAL_SHAPE[0] * CELL_SPACING[0],
            "relative_l2_error": _relative_l2(window_hybrid, window_pure),
            "maximum_absolute_state_error": _maximum_absolute(
                window_hybrid - window_pure
            ),
        },
        "whole_fv_grid": {
            "relative_l2_error": _relative_l2(hybrid_average, pure_average),
            "maximum_absolute_state_error": _maximum_absolute(
                hybrid_average - pure_average
            ),
        },
        "hybrid_step_count": len(hybrid_steps),
        "hybrid_all_steps_accepted": all(
            bool(record["accepted"]) for record in hybrid_steps
        ),
        "pure_finite_volume_step_records": list(pure_fv_steps),
        "features": features,
        "claim": "FV-owned low-amplitude fixed-region comparison only",
        "strong_shock_production_claimed": False,
        "support_maximum_mach": runtime.learned_energy.plan.support.maximum_mach,
        "strong_shock_interpretation": (
            "not qualified; the recorded support envelope and low-amplitude case do "
            "not establish strong-shock production"
        ),
    }


def _wrong_step_and_retry_refusals(
    runtime: PreparedFixedPartitionHybridRuntime,
    initial: FixedPartitionHybridState,
    /,
) -> dict[str, Any]:
    wrong_step = _ExpectedRefusal(ValueError, "exact fixed lattice step")
    with wrong_step:
        runtime.advance(
            initial,
            jnp.asarray(0.5 * FIXED_STEP_SIZE, dtype=jnp.float64),
        )
    retry_policy = _ExpectedRefusal(ValueError, "forbids finite-volume retries")
    with retry_policy:
        _fixed_runtime(runtime.learned_energy, maximum_retries=1)
    return {
        "required_step_size": runtime.required_step_size,
        "allows_step_reduction": runtime.allows_step_reduction,
        "configured_finite_volume_retries": runtime.finite_volume.policy.maximum_retries,
        "wrong_step_refusal": wrong_step.record(),
        "retry_policy_refusal": retry_policy.record(),
    }


def _accepted_fixed_checkpoint(
    runtime: PreparedFixedPartitionHybridRuntime,
    state: FixedPartitionHybridState,
    /,
) -> dict[str, Any]:
    checkpoint = runtime.checkpoint(state, "hybrid-fixed-accepted")
    restored = runtime.restore(checkpoint)
    return {
        "state_kind": "jointly_accepted_only",
        "checkpoint_id": checkpoint.checkpoint_id,
        "payload_id": checkpoint.payload_id,
        "runtime_id": checkpoint.runtime_id,
        "learned_energy_artifact_id": checkpoint.learned_energy_artifact_id,
        "restored_exactly": _tree_exact(restored, state),
        "maximum_restore_error": _tree_maximum_difference(restored, state),
    }


def _unsupported_conserved(
    system: EulerSystem,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    /,
) -> jax.Array:
    support = binding.plan.support
    rho = 0.5 * sum(support.rho_bounds)
    temperature = 0.5 * sum(support.temperature_bounds)
    u_x = support.u_x_bounds[1] + max(
        0.01, 0.05 * (support.u_x_bounds[1] - support.u_x_bounds[0])
    )
    u_y = 0.5 * sum(support.u_y_bounds)
    return _primitive_to_conserved(system, rho, u_x, u_y, temperature)


def _fixed_rollback(
    runtime: PreparedFixedPartitionHybridRuntime,
    initial_average: jax.Array,
    kinetic_conserved: jax.Array,
    /,
) -> dict[str, Any]:
    unsupported = _unsupported_conserved(
        runtime.finite_volume.dynamics.system, runtime.learned_energy
    )
    unsupported_average = initial_average.at[-1, :, :].set(unsupported)
    state, _, _ = _fixed_initial_state(runtime, unsupported_average, kinetic_conserved)
    result = runtime.advance(state, jnp.asarray(FIXED_STEP_SIZE, dtype=jnp.float64))
    candidate_checkpoint = _ExpectedRefusal(ValueError, "Only jointly accepted")
    with candidate_checkpoint:
        runtime.checkpoint(result.candidate, "rejected-fixed-candidate")
    accepted_checkpoint = runtime.checkpoint(
        result.runtime_state, "accepted-before-fixed-refusal"
    )
    restored = runtime.restore(accepted_checkpoint)
    return {
        "accepted": bool(np.asarray(result.evidence.accepted)),
        "rollback_applied": bool(np.asarray(result.evidence.rollback_applied)),
        "status": int(np.asarray(result.evidence.status)),
        "status_name": FixedPartitionHybridStatus(
            int(np.asarray(result.evidence.status))
        ).name,
        "learned_lifts_accepted": bool(
            np.asarray(result.evidence.learned_lifts_accepted)
        ),
        "runtime_state_exactly_previous": _tree_exact(result.runtime_state, state),
        "finite_volume_exactly_previous": _tree_exact(
            result.runtime_state.finite_volume, state.finite_volume
        ),
        "kinetic_exactly_previous": _tree_exact(
            result.runtime_state.kinetic, state.kinetic
        ),
        "candidate_maximum_change": _tree_maximum_difference(result.candidate, state),
        "rejected_candidate_checkpoint_refusal": candidate_checkpoint.record(),
        "accepted_predecessor_checkpoint_restored_exactly": _tree_exact(restored, state),
    }


def _dynamic_base_fields(
    plan: DynamicHybridOwnershipPlan,
    /,
) -> tuple[jax.Array, SmoothCompressibleKineticState, dict[str, Any]]:
    support = plan.learned_energy.plan.support
    rho = 0.5 * sum(support.rho_bounds)
    temperature = 0.5 * sum(support.temperature_bounds)
    u_x = 0.5 * sum(support.u_x_bounds)
    u_y = 0.5 * sum(support.u_y_bounds)
    system = EulerSystem(2, material=plan.method.material)
    conserved = _primitive_to_conserved(system, rho, u_x, u_y, temperature)
    field = jnp.broadcast_to(conserved, plan.spatial_shape + (4,))
    kinetic, learned_support, learned_lift = _learned_equilibrium(
        plan.method, plan.learned_energy, field
    )
    return (
        field,
        kinetic,
        {
            "support": _support_summary(learned_support),
            "lift_successful": bool(np.asarray(jnp.all(learned_lift.successful))),
            "maximum_lift_conservation_residual": _maximum_absolute(
                learned_lift.conservation_residual
            ),
        },
    )


def _dynamic_decision_record(
    method: SmoothCompressibleD2VKineticMethod,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    /,
) -> dict[str, Any]:
    shape = (17, 5)
    plan = DynamicHybridOwnershipPlan(
        method,
        binding,
        shape,
        enter_threshold=0.8,
        exit_threshold=0.2,
        minimum_dwell_steps=2,
        finite_volume_stencil_radius=(1, 0),
        kinetic_reach=(1, 0),
    )
    entry_cell = (2, 2)
    hysteresis_cell = (7, 2)
    unowned_hysteresis_cell = (10, 2)
    exit_cell = (12, 2)
    shock_cell = (16, 2)
    owned = (
        jnp.zeros(shape, dtype="bool")
        .at[hysteresis_cell]
        .set(True)
        .at[exit_cell]
        .set(True)
    )
    state = plan.initialize(owned)
    score = (
        jnp.full(shape, 0.5, dtype=jnp.float64)
        .at[entry_cell]
        .set(1.0)
        .at[exit_cell]
        .set(0.0)
        .at[shock_cell]
        .set(0.0)
    )
    shock = jnp.zeros(shape, dtype="bool").at[shock_cell].set(True)
    first = plan.propose(state, score, shock)
    repeated = plan.propose(state, score, shock)

    dwell_owned = jnp.zeros(shape, dtype="bool").at[exit_cell].set(True)
    dwell_state = DynamicHybridOwnershipState(
        dwell_owned,
        jnp.zeros(shape, dtype=jnp.int32),
        jnp.full(shape, -1, dtype=jnp.int32),
        jnp.zeros(shape, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    dwell_decision = plan.propose(
        dwell_state,
        jnp.zeros(shape, dtype=jnp.float64),
        jnp.zeros(shape, dtype="bool"),
    )
    return {
        "plan_id": plan.plan_id,
        "enter_threshold": plan.enter_threshold,
        "exit_threshold": plan.exit_threshold,
        "minimum_dwell_steps": plan.minimum_dwell_steps,
        "finite_volume_stencil_radius": list(plan.finite_volume_stencil_radius),
        "kinetic_reach": list(plan.kinetic_reach),
        "dilation_radius": list(plan.dilation_radius),
        "shock_owner": plan.shock_owner,
        "ownership_differentiability": plan.ownership_differentiability,
        "deterministic_repeat_exact": _tree_exact(first, repeated),
        "entered_cell": bool(np.asarray(first.entered_finite_volume[entry_cell])),
        "exited_cell": bool(np.asarray(first.exited_finite_volume[exit_cell])),
        "owned_hysteresis_cell_retained": bool(
            np.asarray(first.finite_volume_owned[hysteresis_cell])
        ),
        "unowned_hysteresis_cell_remained_kinetic": not bool(
            np.asarray(first.finite_volume_owned[unowned_hysteresis_cell])
        ),
        "shock_forced_cell_owned": bool(
            np.asarray(first.finite_volume_owned[shock_cell])
        ),
        "all_shock_cells_finite_volume_owned": bool(
            np.asarray(jnp.all(first.finite_volume_owned | ~first.shock_mask))
        ),
        "dilation_added_count": int(np.count_nonzero(np.asarray(first.dilation_added))),
        "dwell_blocked_exit": bool(
            np.asarray(dwell_decision.finite_volume_owned[exit_cell])
        ),
    }


def _dynamic_migration_record(
    method: SmoothCompressibleD2VKineticMethod,
    binding: PreparedLearnedEnergyEquilibriumBinding,
    /,
) -> tuple[dict[str, Any], DynamicHybridOwnershipPlan]:
    shape = (9, 5)
    plan = DynamicHybridOwnershipPlan(
        method,
        binding,
        shape,
        enter_threshold=0.8,
        exit_threshold=0.2,
        minimum_dwell_steps=0,
        finite_volume_stencil_radius=(0, 0),
        kinetic_reach=(0, 0),
    )
    finite_volume, kinetic, initialization = _dynamic_base_fields(plan)
    exit_cell = (2, 2)
    enter_cell = (6, 2)
    support = binding.plan.support
    system = EulerSystem(2, material=method.material)
    perturbed = _primitive_to_conserved(
        system,
        0.5 * sum(support.rho_bounds)
        + 0.02 * (support.rho_bounds[1] - support.rho_bounds[0]),
        0.5 * sum(support.u_x_bounds)
        + 0.02 * (support.u_x_bounds[1] - support.u_x_bounds[0]),
        0.5 * sum(support.u_y_bounds),
        0.5 * sum(support.temperature_bounds),
    )
    perturbed_kinetic, perturbed_support, perturbed_lift = _learned_equilibrium(
        method, binding, perturbed
    )
    kinetic = SmoothCompressibleKineticState(
        kinetic.particle_populations.at[enter_cell].set(
            perturbed_kinetic.particle_populations
        ),
        kinetic.total_energy_populations.at[enter_cell].set(
            perturbed_kinetic.total_energy_populations
        ),
    )
    owned = jnp.zeros(shape, dtype="bool").at[exit_cell].set(True)
    state = DynamicHybridCompositeState(
        finite_volume,
        kinetic,
        plan.initialize(owned),
    )
    score = (
        jnp.full(shape, 0.5, dtype=jnp.float64)
        .at[exit_cell]
        .set(0.0)
        .at[enter_cell]
        .set(1.0)
    )
    decision = plan.propose(state.ownership, score, jnp.zeros(shape, dtype="bool"))
    deferred = plan.migrate(state, decision, jnp.asarray(False))
    migrated = plan.migrate(state, decision, jnp.asarray(True))
    pre_kinetic_moments = method.moments(state.kinetic).conserved
    post_kinetic_moments = method.moments(migrated.runtime_state.kinetic).conserved
    kinetic_to_fv_exact = np.array_equal(
        np.asarray(migrated.runtime_state.finite_volume_conserved[enter_cell]),
        np.asarray(pre_kinetic_moments[enter_cell]),
    )
    finite_volume_to_kinetic_residual = (
        post_kinetic_moments[exit_cell] - state.finite_volume_conserved[exit_cell]
    )
    checkpoint = plan.checkpoint(migrated.runtime_state, "dynamic-owned-accepted")
    restored = plan.restore(checkpoint)
    record = {
        "plan_id": plan.plan_id,
        "ownership_differentiability": plan.ownership_differentiability,
        "initialization": initialization,
        "perturbed_enter_support": _support_summary(perturbed_support),
        "perturbed_enter_lift_successful": bool(np.asarray(perturbed_lift.successful)),
        "accepted_boundary_only": {
            "deferred_accepted": bool(np.asarray(deferred.evidence.accepted)),
            "deferred_rollback_applied": bool(
                np.asarray(deferred.evidence.rollback_applied)
            ),
            "deferred_state_exactly_previous": _tree_exact(deferred.runtime_state, state),
            "deferred_history_exactly_previous": _tree_exact(
                deferred.runtime_state.ownership, state.ownership
            ),
            "accepted_boundary": bool(np.asarray(migrated.evidence.accepted_boundary)),
        },
        "bidirectional_migration": {
            "accepted": bool(np.asarray(migrated.evidence.accepted)),
            "rollback_applied": bool(np.asarray(migrated.evidence.rollback_applied)),
            "entered_finite_volume": bool(
                np.asarray(migrated.evidence.entered_finite_volume[enter_cell])
            ),
            "exited_finite_volume": bool(
                np.asarray(migrated.evidence.exited_finite_volume[exit_cell])
            ),
            "kinetic_to_finite_volume_bit_exact": kinetic_to_fv_exact,
            "maximum_kinetic_to_finite_volume_residual": _maximum_absolute(
                migrated.evidence.kinetic_to_finite_volume_residual
            ),
            "maximum_finite_volume_to_kinetic_conserved_residual": (
                _maximum_absolute(finite_volume_to_kinetic_residual)
            ),
            "learned_support_successful_at_exit": bool(
                np.asarray(migrated.evidence.learned_support.successful[exit_cell])
            ),
            "learned_lift_successful_at_exit": bool(
                np.asarray(migrated.evidence.learned_lift.successful)
            ),
            "kinetic_realizable": bool(
                np.asarray(migrated.evidence.kinetic_realizability.realizable)
            ),
            "exit_transition_count": int(
                np.asarray(migrated.runtime_state.ownership.transition_count[exit_cell])
            ),
            "enter_transition_count": int(
                np.asarray(migrated.runtime_state.ownership.transition_count[enter_cell])
            ),
        },
        "checkpoint_restore": {
            "checkpoint_id": checkpoint.checkpoint_id,
            "payload_id": checkpoint.payload_id,
            "plan_id": checkpoint.plan_id,
            "learned_energy_artifact_id": checkpoint.learned_energy_artifact_id,
            "state_exactly_restored": _tree_exact(restored, migrated.runtime_state),
            "ownership_history_exactly_restored": _tree_exact(
                restored.ownership, migrated.runtime_state.ownership
            ),
            "maximum_restore_error": _tree_maximum_difference(
                restored, migrated.runtime_state
            ),
        },
    }
    return record, plan


def _dynamic_failed_lift(
    plan: DynamicHybridOwnershipPlan,
    /,
) -> dict[str, Any]:
    finite_volume, kinetic, _ = _dynamic_base_fields(plan)
    exit_cell = (2, 2)
    unsupported = _unsupported_conserved(
        EulerSystem(2, material=plan.method.material), plan.learned_energy
    )
    finite_volume = finite_volume.at[exit_cell].set(unsupported)
    owned = jnp.zeros(plan.spatial_shape, dtype="bool").at[exit_cell].set(True)
    state = DynamicHybridCompositeState(
        finite_volume,
        kinetic,
        plan.initialize(owned),
    )
    score = jnp.full(plan.spatial_shape, 0.5, dtype=jnp.float64).at[exit_cell].set(0.0)
    decision = plan.propose(
        state.ownership,
        score,
        jnp.zeros(plan.spatial_shape, dtype="bool"),
    )
    result = plan.migrate(state, decision, jnp.asarray(True))
    candidate_checkpoint = _ExpectedRefusal(
        ValueError, "Only accepted dynamic ownership states"
    )
    with candidate_checkpoint:
        plan.checkpoint(result.candidate, "rejected-dynamic-candidate")
    return {
        "accepted": bool(np.asarray(result.evidence.accepted)),
        "rollback_applied": bool(np.asarray(result.evidence.rollback_applied)),
        "accepted_boundary": bool(np.asarray(result.evidence.accepted_boundary)),
        "attempted_exit": bool(
            np.asarray(result.evidence.exited_finite_volume[exit_cell])
        ),
        "learned_support_successful_at_exit": bool(
            np.asarray(result.evidence.learned_support.successful[exit_cell])
        ),
        "runtime_state_exactly_previous": _tree_exact(result.runtime_state, state),
        "finite_volume_exactly_previous": np.array_equal(
            np.asarray(result.runtime_state.finite_volume_conserved),
            np.asarray(state.finite_volume_conserved),
        ),
        "kinetic_exactly_previous": _tree_exact(
            result.runtime_state.kinetic, state.kinetic
        ),
        "ownership_history_exactly_previous": _tree_exact(
            result.runtime_state.ownership, state.ownership
        ),
        "rejected_candidate_checkpoint_refusal": candidate_checkpoint.record(),
    }


def _all_fixed_stages_valid(records: Sequence[Mapping[str, Any]], /) -> bool:
    return len(records) == RIEMANN_STEP_COUNT and all(
        record["accepted"]
        and len(record["stages"]) == 3
        and tuple(stage["stage_index"] for stage in record["stages"]) == (0, 1, 2)
        and tuple(stage["weight"] for stage in record["stages"]) == SSPRK3_WEIGHTS
        and all(
            stage["interface_count"] == 2 * SPATIAL_SHAPE[1]
            and stage["active_replacement_face_count"] == 2 * SPATIAL_SHAPE[1]
            and stage["all_learned_lifts_present"]
            and stage["all_learned_lifts_successful"]
            and stage["successful"]
            for stage in record["stages"]
        )
        for record in records
    )


def _all_stage_flux_residuals_valid(records: Sequence[Mapping[str, Any]], /) -> bool:
    tolerance = THRESHOLDS["maximum_interface_residual"]
    return all(
        all(
            max(
                stage["maximum_population_to_conservative_flux_residual"],
                stage["maximum_trace_common_flux_residual"],
                stage["maximum_equal_opposite_flux_residual"],
                stage["maximum_moment_lift_residual"],
            )
            <= tolerance
            for stage in record["stages"]
        )
        for record in records
    )


def _all_exchange_residuals_valid(records: Sequence[Mapping[str, Any]], /) -> bool:
    tolerance = THRESHOLDS["maximum_interface_residual"]
    return all(max(record["exchange"].values()) <= tolerance for record in records)


def qualification_report(
    *,
    parent_spatial_path: Path = DEFAULT_PARENT_SPATIAL,
    parent_stage_two_path: Path = DEFAULT_PARENT_STAGE_TWO,
    model_artifact_path: Path = DEFAULT_MODEL_ARTIFACT,
) -> dict[str, Any]:
    """Execute deterministic FV/D2V17 static and dynamic hybrid qualification."""

    artifact = read_learned_energy_equilibrium_artifact(model_artifact_path)
    parents = _parent_records(parent_spatial_path, parent_stage_two_path, artifact)
    runtime = _fixed_runtime(artifact.binding, maximum_retries=0)
    left, right, riemann_initial_record = _riemann_states(runtime)
    finite_volume_average = _riemann_field(runtime, left, right)
    right_field = jnp.broadcast_to(right, SPATIAL_SHAPE + (4,))
    initial, initial_support, initial_lift = _fixed_initial_state(
        runtime, finite_volume_average, right_field
    )
    final, fixed_steps, last_result = _run_fixed_hybrid(
        runtime, initial, RIEMANN_STEP_COUNT
    )
    replay_final, replay_steps, replay_last_result = _run_fixed_hybrid(
        runtime, initial, RIEMANN_STEP_COUNT
    )
    pure_fv_final, pure_fv_steps = _run_pure_finite_volume(
        runtime.finite_volume, finite_volume_average, RIEMANN_STEP_COUNT
    )
    riemann = _riemann_comparison(
        runtime,
        final,
        pure_fv_final,
        fixed_steps,
        pure_fv_steps,
        riemann_initial_record,
    )
    fixed_checkpoint = _accepted_fixed_checkpoint(runtime, final)
    fixed_refusals = _wrong_step_and_retry_refusals(runtime, initial)
    fixed_rollback = _fixed_rollback(runtime, finite_volume_average, right_field)

    decision = _dynamic_decision_record(runtime.spatial.method, artifact.binding)
    migration, migration_plan = _dynamic_migration_record(
        runtime.spatial.method, artifact.binding
    )
    failed_dynamic_lift = _dynamic_failed_lift(migration_plan)

    interface_tolerance = THRESHOLDS["maximum_interface_residual"]
    conservation_tolerance = THRESHOLDS["maximum_global_conservation_residual"]
    migration_tolerance = THRESHOLDS["maximum_migration_conservation_residual"]
    feature_values = tuple(riemann["features"].values())
    identities = {
        "quadrature": runtime.spatial.method.quadrature.quadrature_id,
        "material": runtime.spatial.method.material.material_id,
        "transport_closure": runtime.spatial.method.transport.closure_id,
        "kinetic_method": runtime.spatial.method.method_id,
        "energy_plan": runtime.spatial.energy_plan.plan_id,
        "model_artifact": artifact.artifact_id,
        "model_numeric_revision": artifact.binding.numeric_revision.revision_id,
        "prepared_binding": artifact.binding.prepared_id,
        "finite_volume_runtime": runtime.finite_volume.runtime_id,
        "kinetic_spatial_runtime": runtime.spatial.prepared_id,
        "fixed_hybrid_runtime": runtime.runtime_id,
        "dynamic_ownership_plan": migration_plan.plan_id,
    }
    gates = {
        "parent_chain_and_frozen_model_bound": parents["all_compatible"],
        "runtime_ids_bound_to_loaded_model": bool(
            runtime.learned_energy.prepared_id == artifact.binding.prepared_id
            and runtime.spatial.energy_plan.plan_id
            == artifact.binding.plan.equilibrium_plan.plan_id
            and all(isinstance(value, str) and value for value in identities.values())
        ),
        "deterministic_reexecution_exact": bool(
            _tree_exact(final, replay_final)
            and fixed_steps == replay_steps
            and _tree_exact(last_result, replay_last_result)
        ),
        "three_phase_stage_fluxes_and_weights": bool(
            _all_fixed_stages_valid(fixed_steps)
            and math.isclose(sum(SSPRK3_WEIGHTS), 1.0)
        ),
        "one_common_population_conservative_flux": _all_stage_flux_residuals_valid(
            fixed_steps
        ),
        "single_application_no_flux_double_count": _all_exchange_residuals_valid(
            fixed_steps
        ),
        "equal_opposite_interface_and_fv_ledger": all(
            record["audit"]["maximum_kinetic_moment_exchange_residual"]
            <= interface_tolerance
            and record["audit"]["maximum_finite_volume_ledger_residual"]
            <= interface_tolerance
            and record["exchange"]["maximum_link_route_moment_residual"]
            <= interface_tolerance
            and record["exchange"]["maximum_finite_volume_interface_residual"]
            <= interface_tolerance
            and record["exchange"]["maximum_unowned_outer_boundary_exchange"]
            <= interface_tolerance
            for record in fixed_steps
        ),
        "joint_global_conservation": all(
            record["audit"]["maximum_global_conservation_residual"]
            <= conservation_tolerance
            and record["audit"]["maximum_absolute_residual"] <= conservation_tolerance
            for record in fixed_steps
        ),
        "fixed_dt_and_no_retry_refusal": bool(
            not fixed_refusals["allows_step_reduction"]
            and fixed_refusals["configured_finite_volume_retries"] == 0
            and fixed_refusals["wrong_step_refusal"]["observed"]
            and fixed_refusals["retry_policy_refusal"]["observed"]
            and all(
                record["exact_step"]
                and record["accepted_step_size"] == FIXED_STEP_SIZE
                and record["finite_volume_retries"] == 0
                for record in fixed_steps
            )
        ),
        "finite_volume_positivity": all(
            record["finite_volume_positive"]
            and record["minimum_finite_volume_density"] > 0.0
            and record["minimum_finite_volume_pressure"] > 0.0
            for record in fixed_steps
        ),
        "kinetic_support_positivity_and_realizability": bool(
            _support_summary(initial_support)["all_successful"]
            and bool(np.asarray(jnp.all(initial_lift.successful)))
            and all(
                record["learned_lifts_accepted"]
                and record["kinetic_collision_accepted"]
                and record["accepted_kinetic_support"]["all_successful"]
                and record["accepted_kinetic_support"]["minimum_margin"]
                >= THRESHOLDS["minimum_support_margin"]
                and record["kinetic_realizability"]["realizable"]
                and record["kinetic_realizability"]["minimum_particle_population"]
                >= THRESHOLDS["minimum_population"]
                and record["kinetic_realizability"]["minimum_total_energy_population"]
                >= THRESHOLDS["minimum_population"]
                for record in fixed_steps
            )
        ),
        "full_static_rollback": bool(
            not fixed_rollback["accepted"]
            and fixed_rollback["rollback_applied"]
            and not fixed_rollback["learned_lifts_accepted"]
            and fixed_rollback["runtime_state_exactly_previous"]
            and fixed_rollback["finite_volume_exactly_previous"]
            and fixed_rollback["kinetic_exactly_previous"]
            and fixed_rollback["candidate_maximum_change"] > 0.0
            and fixed_rollback["rejected_candidate_checkpoint_refusal"]["observed"]
            and fixed_rollback["accepted_predecessor_checkpoint_restored_exactly"]
        ),
        "accepted_static_checkpoint_restore": bool(
            fixed_checkpoint["restored_exactly"]
            and fixed_checkpoint["maximum_restore_error"] == 0.0
            and fixed_checkpoint["runtime_id"] == runtime.runtime_id
            and fixed_checkpoint["learned_energy_artifact_id"]
            == artifact.binding.prepared_id
        ),
        "dynamic_enter_exit_hysteresis": bool(
            decision["entered_cell"]
            and decision["exited_cell"]
            and decision["owned_hysteresis_cell_retained"]
            and decision["unowned_hysteresis_cell_remained_kinetic"]
            and decision["deterministic_repeat_exact"]
        ),
        "shock_forced_finite_volume_ownership": bool(
            decision["shock_owner"] == "finite_volume"
            and decision["shock_forced_cell_owned"]
            and decision["all_shock_cells_finite_volume_owned"]
        ),
        "dynamic_dilation_and_dwell": bool(
            decision["dilation_radius"] == [2, 0]
            and decision["dilation_added_count"] > 0
            and decision["dwell_blocked_exit"]
        ),
        "dynamic_accepted_boundary_only": bool(
            not migration["accepted_boundary_only"]["deferred_accepted"]
            and migration["accepted_boundary_only"]["deferred_rollback_applied"]
            and migration["accepted_boundary_only"]["deferred_state_exactly_previous"]
            and migration["accepted_boundary_only"]["deferred_history_exactly_previous"]
            and migration["accepted_boundary_only"]["accepted_boundary"]
        ),
        "bidirectional_conservative_migration": bool(
            migration["bidirectional_migration"]["accepted"]
            and migration["bidirectional_migration"]["entered_finite_volume"]
            and migration["bidirectional_migration"]["exited_finite_volume"]
            and migration["bidirectional_migration"]["kinetic_to_finite_volume_bit_exact"]
            and migration["bidirectional_migration"][
                "maximum_kinetic_to_finite_volume_residual"
            ]
            == 0.0
            and migration["bidirectional_migration"][
                "maximum_finite_volume_to_kinetic_conserved_residual"
            ]
            <= migration_tolerance
            and migration["bidirectional_migration"]["learned_support_successful_at_exit"]
            and migration["bidirectional_migration"]["learned_lift_successful_at_exit"]
            and migration["bidirectional_migration"]["kinetic_realizable"]
        ),
        "failed_dynamic_lift_full_rollback": bool(
            not failed_dynamic_lift["accepted"]
            and failed_dynamic_lift["rollback_applied"]
            and failed_dynamic_lift["accepted_boundary"]
            and failed_dynamic_lift["attempted_exit"]
            and not failed_dynamic_lift["learned_support_successful_at_exit"]
            and failed_dynamic_lift["runtime_state_exactly_previous"]
            and failed_dynamic_lift["finite_volume_exactly_previous"]
            and failed_dynamic_lift["kinetic_exactly_previous"]
            and failed_dynamic_lift["ownership_history_exactly_previous"]
            and failed_dynamic_lift["rejected_candidate_checkpoint_refusal"]["observed"]
        ),
        "ownership_checkpoint_restore": bool(
            migration["checkpoint_restore"]["state_exactly_restored"]
            and migration["checkpoint_restore"]["ownership_history_exactly_restored"]
            and migration["checkpoint_restore"]["maximum_restore_error"] == 0.0
            and migration["checkpoint_restore"]["plan_id"] == migration_plan.plan_id
            and migration["checkpoint_restore"]["learned_energy_artifact_id"]
            == artifact.binding.prepared_id
        ),
        "matched_low_amplitude_fixed_fv_region": bool(
            riemann["hybrid_step_count"] == RIEMANN_STEP_COUNT
            and riemann["hybrid_all_steps_accepted"]
            and len(pure_fv_steps) == RIEMANN_STEP_COUNT
            and all(record["accepted"] for record in pure_fv_steps)
            and riemann["fixed_fv_region"]["relative_l2_error"]
            <= THRESHOLDS["maximum_fixed_fv_region_relative_l2_error"]
            and all(
                feature["hybrid"]["successful"]
                and feature["pure_finite_volume"]["successful"]
                and feature["absolute_location_difference"]
                <= THRESHOLDS["maximum_feature_location_error"]
                for feature in feature_values
            )
        ),
        "honest_fv_owned_claim": bool(
            runtime.shock_owner == "finite_volume"
            and runtime.ownership_differentiability == "none"
            and decision["ownership_differentiability"] == "none"
            and riemann["claim"] == "FV-owned low-amplitude fixed-region comparison only"
            and not riemann["strong_shock_production_claimed"]
        ),
    }
    report = {
        "tool": "learned_energy_hybrid_qualification",
        "scope": {
            "claim": "FV-owned low-amplitude static and dynamic FV/D2V17 hybrid only",
            "included": [
                "structured two-dimensional finite volume with D2V17",
                "stage-resolved SSPRK3 common population flux replacement",
                "equal-and-opposite accepted interface exchange",
                "fixed step with no FV retry",
                "transactional joint rollback and accepted checkpoint restore",
                "nondifferentiable accepted-boundary ownership migration",
                "hysteresis, shock ownership, dilation, and dwell",
                "low-amplitude Riemann-style fixed-FV-region comparison",
            ],
            "excluded": [
                "strong-shock production",
                "differentiable ownership migration",
                "adaptive time stepping",
                "FV-only retry",
                "post-hoc conservation correction",
                "nonperiodic kinetic interior transport",
            ],
        },
        "contracts": {
            "stage_weights": list(SSPRK3_WEIGHTS),
            "interface_flux": (
                "one population flux per interface and SSPRK stage; its moments "
                "replace the FV face flux and its weighted integral is applied once "
                "with equal-and-opposite ownership"
            ),
            "finite_volume_shock_owner": True,
            "ownership_differentiability": "none",
            "migration_boundary": "accepted step boundaries only",
            "checkpoint_state": "accepted composite state and ownership history only",
            "kinetic_transport": "collision followed by closed x-interface-owned D2V17 routing",
        },
        "configuration": {
            "deterministic": True,
            "model_artifact_path": str(model_artifact_path),
            "spatial_shape": list(SPATIAL_SHAPE),
            "cell_spacing": list(CELL_SPACING),
            "fixed_step_size": FIXED_STEP_SIZE,
            "riemann_step_count": RIEMANN_STEP_COUNT,
            "interface_count": 2 * SPATIAL_SHAPE[1],
            "interface_axis": "closed_x_pair",
            "finite_volume_maximum_retries": runtime.finite_volume.policy.maximum_retries,
            "finite_volume_cfl": FINITE_VOLUME_CFL,
            "particle_relaxation_time": PARTICLE_RELAXATION_TIME,
            "total_energy_relaxation_time": TOTAL_ENERGY_RELAXATION_TIME,
        },
        "parents": parents,
        "identities": identities,
        "thresholds": dict(THRESHOLDS),
        "initial_kinetic_state": {
            "support": _support_summary(initial_support),
            "learned_lift_successful": bool(np.asarray(jnp.all(initial_lift.successful))),
            "maximum_lift_conservation_residual": _maximum_absolute(
                initial_lift.conservation_residual
            ),
        },
        "static_hybrid": {
            "runtime_id": runtime.runtime_id,
            "shock_owner": runtime.shock_owner,
            "ownership_differentiability": runtime.ownership_differentiability,
            "stage_weights": list(SSPRK3_WEIGHTS),
            "steps": fixed_steps,
            "final_state_id": array_tree_fingerprint(final),
            "deterministic_reexecution": {
                "final_state_exact": _tree_exact(final, replay_final),
                "records_exact": fixed_steps == replay_steps,
                "last_result_exact": _tree_exact(last_result, replay_last_result),
                "replayed_final_state_id": array_tree_fingerprint(replay_final),
            },
            "fixed_dt_and_retry_refusals": fixed_refusals,
            "accepted_checkpoint_restore": fixed_checkpoint,
            "failed_learned_lift_rollback": fixed_rollback,
        },
        "dynamic_ownership": {
            "decision": decision,
            "migration": migration,
            "failed_learned_lift_rollback": failed_dynamic_lift,
        },
        "riemann_comparison": riemann,
        "environment": capture_environment().to_dict(),
        "gates": gates,
    }
    report["passed"] = all(gates.values())
    report["qualification_id"] = canonical_fingerprint(report)
    return report


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify deterministic fixed-step FV/D2V17 static and dynamic hybrid "
            "execution with a frozen learned-energy artifact."
        )
    )
    parser.add_argument(
        "--parent-spatial",
        type=Path,
        default=DEFAULT_PARENT_SPATIAL,
        help="passing learned-energy spatial qualification JSON",
    )
    parser.add_argument(
        "--parent-stage-two",
        type=Path,
        default=DEFAULT_PARENT_STAGE_TWO,
        help="passing learned-energy stage-two qualification JSON",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        default=DEFAULT_MODEL_ARTIFACT,
        help="frozen stage-two learned-energy artifact",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_arguments()
    forbidden_outputs = {
        arguments.parent_spatial.resolve(),
        arguments.parent_stage_two.resolve(),
        arguments.model_artifact.resolve(),
    }
    if OUTPUT_PATH.resolve() in forbidden_outputs:
        raise ValueError(
            "Hybrid qualification output must not overwrite a parent artifact."
        )
    with jax.enable_x64(True):
        report = qualification_report(
            parent_spatial_path=arguments.parent_spatial,
            parent_stage_two_path=arguments.parent_stage_two,
            model_artifact_path=arguments.model_artifact,
        )
    if not report["passed"]:
        failed = sorted(name for name, passed in report["gates"].items() if not passed)
        print("failed qualification gates: " + ", ".join(failed), file=sys.stderr)
        return 1
    write_json_atomic(OUTPUT_PATH, report)
    print(canonical_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
