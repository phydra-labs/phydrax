#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Route-specific scientific qualification for native incompressible flow."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, assert_never, get_args, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._validation import finite_real_scalar, positive_integer
from phydrax.discretization.spectral import (
    PreparedSpectralVanishingViscosity,
    SpectralVanishingViscosityPlan,
    SpectralViscosityKernel,
    TensorSpectralDiscretization,
)


CAPABILITY = "incompressible-flow"
ROUTES = (
    "periodic-spectral",
    "periodic-spectral-vanishing-viscosity",
    "periodic-statistical-solution",
    "spectral-channel",
    "mac",
)
REQUIRED_CASES = {
    "periodic-spectral": (
        "taylor-green-decay",
        "manufactured-forcing-refinement-restart",
    ),
    "periodic-spectral-vanishing-viscosity": (
        "shear-decay-2d",
        "shear-decay-3d",
        "energy-identity-2d",
        "energy-identity-3d",
        "fixed-low-mode-vanishing",
    ),
    "periodic-statistical-solution": ("coupled-campaign",),
    "spectral-channel": (
        "couette",
        "poiseuille",
        "manufactured-sbdf2-refinement-restart",
    ),
    "mac": (
        "periodic-taylor-green",
        "stretched-couette",
        "stretched-poiseuille",
        "full-hybrid-iterative-comparison",
    ),
}


def _json_ready(value: object, /) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: object, /) -> str:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def content_address(value: object, /) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _identified(kind: str, fields: Mapping[str, object], id_key: str, /) -> Any:
    core = _json_ready({"kind": kind, **dict(fields)})
    assert isinstance(core, dict)
    return {**core, id_key: content_address(core)}


def external_reference_input(
    *,
    path: str | Path | None = None,
    checksum: str | None = None,
    nondimensionalization: Mapping[str, object] | None = None,
    uncertainty: Mapping[str, object] | None = None,
) -> dict[str, object] | None:
    """Register all external metadata without reading or executing external data."""

    supplied = (path, checksum, nondimensionalization, uncertainty)
    if all(value is None for value in supplied):
        return None
    names = ("path", "checksum", "nondimensionalization", "uncertainty")
    missing = [name for name, value in zip(names, supplied, strict=True) if value is None]
    if missing:
        raise ValueError(
            "External reference metadata is all-or-none; missing " + ", ".join(missing)
        )
    if not str(path) or not str(checksum):
        raise ValueError("External reference path and checksum must be non-empty.")
    if not isinstance(nondimensionalization, Mapping) or not nondimensionalization:
        raise TypeError("nondimensionalization must be a non-empty mapping.")
    if not isinstance(uncertainty, Mapping) or not uncertainty:
        raise TypeError("uncertainty must be a non-empty mapping.")
    return _identified(
        "external-reference-input",
        {
            "path": str(path),
            "declared_checksum": str(checksum),
            "checksum_verified": False,
            "nondimensionalization": dict(nondimensionalization),
            "uncertainty": dict(uncertainty),
            "executed": False,
        },
        "reference_input_id",
    )


def gate_outcome(
    gate: str,
    outcome: str,
    reason: str,
    /,
    *,
    metric: str | None = None,
    observed: object = None,
    criterion: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if outcome not in ("passed", "failed", "inconclusive"):
        raise ValueError("Gate outcome must be passed, failed, or inconclusive.")
    if not gate or not reason:
        raise ValueError("Gate and reason must be non-empty.")
    return _identified(
        "qualification-gate-outcome",
        {
            "gate": gate,
            "outcome": outcome,
            "metric": metric,
            "observed": observed,
            "criterion": None if criterion is None else dict(criterion),
            "reason": reason,
        },
        "gate_outcome_id",
    )


def numeric_gate(
    gate: str,
    metric: str,
    observed: float | None,
    maximum: float,
    /,
) -> dict[str, object]:
    criterion = {"comparison": "less-than-or-equal", "maximum": float(maximum)}
    if observed is None or not math.isfinite(float(observed)):
        return gate_outcome(
            gate,
            "failed",
            f"Required metric {metric!r} is missing or non-finite.",
            metric=metric,
            observed=None,
            criterion=criterion,
        )
    passed = float(observed) <= float(maximum)
    return gate_outcome(
        gate,
        "passed" if passed else "failed",
        (
            f"{metric} satisfied the configured upper bound."
            if passed
            else f"{metric} exceeded the configured upper bound."
        ),
        metric=metric,
        observed=float(observed),
        criterion=criterion,
    )


def minimum_numeric_gate(
    gate: str,
    metric: str,
    observed: float | None,
    minimum: float,
    /,
) -> dict[str, object]:
    criterion = {"comparison": "greater-than-or-equal", "minimum": float(minimum)}
    if observed is None or not math.isfinite(float(observed)):
        return gate_outcome(
            gate,
            "failed",
            f"Required metric {metric!r} is missing or non-finite.",
            metric=metric,
            observed=None,
            criterion=criterion,
        )
    passed = float(observed) >= float(minimum)
    return gate_outcome(
        gate,
        "passed" if passed else "failed",
        (
            f"{metric} satisfied the configured lower bound."
            if passed
            else f"{metric} did not reach the configured lower bound."
        ),
        metric=metric,
        observed=float(observed),
        criterion=criterion,
    )


def boolean_gate(
    gate: str,
    metric: str,
    observed: bool | None,
    /,
) -> dict[str, object]:
    criterion = {"comparison": "is", "expected": True}
    if observed is None:
        return gate_outcome(
            gate,
            "inconclusive",
            f"Required boolean evidence {metric!r} is missing.",
            metric=metric,
            criterion=criterion,
        )
    passed = bool(observed)
    return gate_outcome(
        gate,
        "passed" if passed else "failed",
        f"{metric} was {'true' if passed else 'false'}.",
        metric=metric,
        observed=passed,
        criterion=criterion,
    )


def make_qualification_artifact(
    *,
    route: str,
    support_tuple: phx.qualification.SupportTuple,
    inputs: Mapping[str, object],
    reference: Mapping[str, object],
    configuration: Mapping[str, object],
    metrics: Mapping[str, object],
    gates: Sequence[Mapping[str, object]],
    external_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build deterministic fail-closed evidence; this can never release support."""

    if route not in ROUTES:
        raise ValueError(f"Unknown incompressible-flow route {route!r}.")
    if not isinstance(support_tuple, phx.qualification.SupportTuple):
        raise TypeError("support_tuple must be SupportTuple.")
    if (
        support_tuple.capability != CAPABILITY
        or dict(support_tuple.attributes).get("route") != route
    ):
        raise ValueError("SupportTuple must exactly identify the requested route.")
    outcomes = [dict(value) for value in gates]
    for value in outcomes:
        if value.get("outcome") not in ("passed", "failed", "inconclusive"):
            raise ValueError("Every gate requires a valid outcome.")
        if not value.get("reason"):
            raise ValueError("Every gate requires an explicit reason.")
    if not outcomes:
        outcomes.append(
            gate_outcome(
                "route-gates",
                "inconclusive",
                "No scientific gate outcomes were supplied for the route.",
            )
        )
    for case in REQUIRED_CASES[route]:
        case_metrics = metrics.get(case)
        if not isinstance(case_metrics, Mapping) or not case_metrics:
            outcomes.append(
                gate_outcome(
                    f"required-case:{case}",
                    "inconclusive",
                    f"Route-required case {case!r} has no metric evidence.",
                    metric=case,
                )
            )
    failed = sorted(
        str(value["reason"]) for value in outcomes if value["outcome"] == "failed"
    )
    inconclusive = sorted(
        str(value["reason"]) for value in outcomes if value["outcome"] == "inconclusive"
    )
    status = "failed" if failed else "inconclusive" if inconclusive else "passed"
    normalized_metrics = _json_ready(dict(metrics))
    normalized_external = _json_ready(
        None if external_reference is None else dict(external_reference)
    )
    core = {
        "kind": "incompressible-flow-qualification-artifact",
        "route": route,
        "support_tuple": support_tuple.to_record(),
        "input": _identified("qualification-input", inputs, "input_id"),
        "reference": _identified("qualification-reference", reference, "reference_id"),
        "configuration": _identified(
            "qualification-configuration", configuration, "configuration_id"
        ),
        "external_reference": normalized_external,
        "metrics": normalized_metrics,
        "gates": sorted(outcomes, key=lambda value: str(value["gate"])),
        "status": status,
        "failed_reasons": failed,
        "inconclusive_reasons": inconclusive,
        "release_ready": False,
    }
    return {**core, "artifact_id": content_address(core)}


def verify_qualification_artifact(record: Mapping[str, object], /) -> None:
    if record.get("kind") != "incompressible-flow-qualification-artifact":
        raise ValueError("Input is not an incompressible-flow qualification artifact.")
    if record.get("release_ready") is not False:
        raise ValueError("Qualification artifacts must remain unreleased.")
    identifier = record.get("artifact_id")
    core = {key: value for key, value in record.items() if key != "artifact_id"}
    if not isinstance(identifier, str) or content_address(core) != identifier:
        raise ValueError("Qualification artifact has an invalid content address.")
    support = record.get("support_tuple")
    if not isinstance(support, Mapping):
        raise ValueError("Qualification artifact has no SupportTuple.")
    phx.qualification.SupportTuple.from_record(support)


def assemble_candidate_profile(
    artifacts: Sequence[Mapping[str, object]],
    /,
    *,
    name: str = "incompressible-flow.candidate",
    provider: str = "phydrax",
) -> dict[str, object]:
    """Create an unsigned, unreleased CapabilityProfile candidate."""

    records = tuple(dict(value) for value in artifacts)
    if not records:
        raise ValueError("At least one qualification artifact is required.")
    supports = []
    artifact_ids = []
    routes = []
    for record in records:
        verify_qualification_artifact(record)
        if record.get("status") != "passed":
            raise ValueError(
                f"Qualification artifact {record['artifact_id']} is not passed."
            )
        support = record["support_tuple"]
        assert isinstance(support, Mapping)
        supports.append(phx.qualification.SupportTuple.from_record(support))
        artifact_ids.append(str(record["artifact_id"]))
        routes.append(str(record["route"]))
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ValueError("Candidate assembly received duplicate artifact IDs.")
    profile = phx.qualification.CapabilityProfile(
        name,
        provider,
        tuple(supports),
        required_gates=(),
        release_evidence=(),
        released=False,
    )
    core = {
        "kind": "incompressible-flow-capability-profile-candidate",
        "qualification_artifact_ids": sorted(artifact_ids),
        "qualified_routes": sorted(routes),
        "profile": profile.to_record(),
        "release_ready": False,
        "signed": False,
    }
    return {**core, "candidate_id": content_address(core)}


def _periodic_space(count: int, dimension: int = 2) -> TensorSpectralDiscretization:
    return phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(dimension)),
        axis_names=("x", "y", "z")[:dimension],
        field_name="velocity",
    ).prepare(
        tuple(
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * math.pi)
            for _ in range(dimension)
        )
    )


def _taylor_green(space: Any) -> Any:
    x, y = jnp.meshgrid(space.axes[0].nodes, space.axes[1].nodes, indexing="ij")
    return jnp.stack((jnp.sin(x) * jnp.cos(y), -jnp.cos(x) * jnp.sin(y)), axis=-1)


def _periodic_compile(
    space: Any, viscosity: Any, *, forcing: Any = None, forcing_id: Any = None
) -> Any:
    problem = phx.equations.IncompressibleFlowProblem(
        2, viscosity, forcing=forcing, forcing_id=forcing_id
    )
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    return (
        problem,
        method,
        phx.equations.compile_periodic_incompressible_flow(problem, space, method),
    )


def _etdrk_solve(dynamics: Any, initial: Any, step: Any, steps: Any) -> Any:
    method = phx.solver.ETDRKMethod(4)
    times = jnp.arange(steps + 1, dtype="float64") * step
    result = phx.solver.solve_etdrk(method, dynamics.semilinear_drift, initial, times)
    return method, result


def periodic_spectral_qualification(
    *,
    mode_count: int = 12,
    viscosity: float = 0.01,
    step_size: float = 0.025,
    steps: int = 8,
    solution_tolerance: float = 1.0e-8,
    invariant_tolerance: float = 1.0e-9,
    restart_tolerance: float = 1.0e-10,
    external_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    count, step_count = int(mode_count), int(steps)
    dt, nu = float(step_size), float(viscosity)
    if count < 8 or step_count < 4 or step_count % 2 or dt <= 0 or nu < 0:
        raise ValueError("Periodic count/steps/step size/viscosity are invalid.")
    space = _periodic_space(count)
    base = _taylor_green(space)
    decay_problem, spatial_method, decay_dynamics = _periodic_compile(space, nu)
    initial = decay_dynamics.project_state(base)
    temporal_method, decay = _etdrk_solve(decay_dynamics, initial, dt, step_count)
    final_time = dt * step_count
    exact_decay = jnp.exp(-2.0 * nu * final_time) * base
    decay_error = float(
        jnp.max(jnp.abs(decay_dynamics.reconstruct_state(decay.states[-1]) - exact_decay))
    )
    decay_diagnostics = decay_dynamics.diagnostics(final_time, decay.states[-1])
    base_modal = space.project(base)

    def forcing(time: Any, state: Any, args: Any) -> Any:
        del state, args
        amplitude = 1.0 + 0.1 * jnp.sin(time)
        return (0.1 * jnp.cos(time) + 2.0 * nu * amplitude) * base_modal

    forced_problem, _, forced = _periodic_compile(
        space,
        nu,
        forcing=forcing,
        forcing_id="manufactured-time-dependent-taylor-green",
    )
    forced_initial = forced.project_state(base)
    _, coarse = _etdrk_solve(forced, forced_initial, dt, step_count)
    _, fine = _etdrk_solve(forced, forced_initial, 0.5 * dt, 2 * step_count)
    exact_forced = (1.0 + 0.1 * jnp.sin(final_time)) * base
    coarse_error = float(
        jnp.max(jnp.abs(forced.reconstruct_state(coarse.states[-1]) - exact_forced))
    )
    fine_error = float(
        jnp.max(jnp.abs(forced.reconstruct_state(fine.states[-1]) - exact_forced))
    )
    half = step_count // 2
    first = phx.solver.solve_etdrk(
        temporal_method,
        forced.semilinear_drift,
        forced_initial,
        jnp.arange(half + 1, dtype="float64") * dt,
    )
    second = phx.solver.solve_etdrk(
        temporal_method,
        forced.semilinear_drift,
        first.states[-1],
        (jnp.arange(half + 1, dtype="float64") + half) * dt,
    )
    restart_error = float(jnp.max(jnp.abs(second.states[-1] - coarse.states[-1])))
    forced_diagnostics = forced.diagnostics(final_time, fine.states[-1])
    maximum_wave = float(jnp.sqrt(jnp.max(forced.projector.wavenumber_squared)))
    statistics_plan = (
        phx.applications.incompressible_flow.PeriodicModalTurbulenceStatisticsPlan(
            forced,
            jnp.linspace(0.0, maximum_wave + 1.0e-12, count + 1),
        )
    )
    statistics = statistics_plan.evaluate(final_time, fine.states[-1])
    order = (
        math.log2(coarse_error / fine_error)
        if coarse_error > 0.0 and fine_error > 0.0
        else None
    )
    metrics = {
        "taylor-green-decay": {
            "maximum_velocity_error": decay_error,
            "divergence_norm": float(decay_diagnostics.divergence_norm),
            "energy_balance_defect": float(decay_diagnostics.energy_balance_defect),
            "solver_successful": bool(decay.successful),
        },
        "manufactured-forcing-refinement-restart": {
            "coarse_maximum_velocity_error": coarse_error,
            "fine_maximum_velocity_error": fine_error,
            "observed_temporal_order": order,
            "restart_maximum_modal_error": restart_error,
            "divergence_norm": float(forced_diagnostics.divergence_norm),
            "statistics_successful": bool(statistics.successful),
            "statistics_plan_id": statistics_plan.plan_id,
            "forcing_power": float(statistics.forcing_power),
            "solver_successful": bool(coarse.successful & fine.successful),
        },
    }
    gates = (
        numeric_gate(
            "decay-solution",
            "taylor-green-decay.maximum_velocity_error",
            decay_error,
            solution_tolerance,
        ),
        numeric_gate(
            "decay-divergence",
            "taylor-green-decay.divergence_norm",
            float(decay_diagnostics.divergence_norm),
            invariant_tolerance,
        ),
        boolean_gate(
            "decay-solver", "taylor-green-decay.solver_successful", bool(decay.successful)
        ),
        numeric_gate(
            "manufactured-solution",
            "manufactured-forcing-refinement-restart.fine_maximum_velocity_error",
            fine_error,
            solution_tolerance,
        ),
        boolean_gate(
            "manufactured-refinement",
            "manufactured-forcing-refinement-restart.fine_error_not_greater",
            fine_error <= coarse_error,
        ),
        numeric_gate(
            "manufactured-restart",
            "manufactured-forcing-refinement-restart.restart_maximum_modal_error",
            restart_error,
            restart_tolerance,
        ),
        boolean_gate(
            "native-statistics",
            "manufactured-forcing-refinement-restart.statistics_successful",
            bool(statistics.successful),
        ),
        boolean_gate(
            "manufactured-solver",
            "manufactured-forcing-refinement-restart.solver_successful",
            bool(coarse.successful & fine.successful),
        ),
    )
    support = phx.qualification.SupportTuple(
        CAPABILITY,
        {
            "route": "periodic-spectral",
            "spatial_method": decay_dynamics.resolved_method,
            "temporal_method": "etdrk4",
            "forcing": "compiler-modal",
            "statistics": "periodic-modal",
        },
    )
    return make_qualification_artifact(
        route="periodic-spectral",
        support_tuple=support,
        inputs={
            "domain": [[0.0, 2.0 * math.pi], [0.0, 2.0 * math.pi]],
            "initial_condition": "u=sin(x)cos(y);v=-cos(x)sin(y)",
            "manufactured_amplitude": "1+0.1*sin(t)",
            "discretization_prepared_id": space.prepared_id,
            "spatial_method_id": spatial_method.method_id,
            "spatial_method_prepared_id": forced.spatial_method.prepared_id,
            "temporal_method_id": temporal_method.method_id,
            "decay_problem_id": decay_problem.problem_id,
            "forced_problem_id": forced_problem.problem_id,
            "decay_compilation_id": decay_dynamics.compilation_id,
            "forced_compilation_id": forced.compilation_id,
        },
        reference={
            "source": "native-analytic",
            "decay": "exp(-2*viscosity*t)",
            "manufactured_solution": "(1+0.1*sin(t))*taylor-green-mode",
        },
        configuration={
            "mode_count": count,
            "viscosity": nu,
            "step_size": dt,
            "steps": step_count,
            "solution_tolerance": float(solution_tolerance),
            "invariant_tolerance": float(invariant_tolerance),
            "restart_tolerance": float(restart_tolerance),
            "timing_gate": False,
        },
        metrics=metrics,
        gates=gates,
        external_reference=external_reference,
    )


def _svv_reference_rate(
    plan: SpectralVanishingViscosityPlan, resolution: int, mode: float, /
) -> float:
    # Rohner--Mishra (2024), section 2: epsilon_K and m_K. The test box
    # has length 2*pi, so physical and integer wavenumbers coincide.
    cutoff = plan.cutoff_scale * resolution**plan.cutoff_exponent
    match plan.kernel:
        case "tadmor_step":
            profile = float(mode > cutoff)
        case "exponential_highpass":
            exponent = plan.highpass_exponent
            if exponent is None:
                raise ValueError("Exponential high-pass requires its declared exponent.")
            profile = -math.expm1(-((mode / cutoff) ** exponent))
        case _:
            assert_never(plan.kernel)
    return (
        -plan.strength
        * resolution ** (1.0 - 2.0 * plan.order)
        * mode ** (2.0 * plan.order)
        * profile
    )


def _svv_support(prepared: PreparedSpectralVanishingViscosity, /) -> dict[str, object]:
    return {
        "operator_scaling_admissible": prepared.operator_scaling_admissible,
        "cited_convergence_supported": prepared.cited_convergence_supported,
        "support_reason": prepared.support_reason,
    }


def _svv_shear_evidence(
    plan: SpectralVanishingViscosityPlan,
    count: int,
    dimension: int,
    dt: float,
    steps: int,
    solution_tolerance: float,
    invariant_tolerance: float,
    /,
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]]]:
    space = _periodic_space(count, dimension)
    problem = phx.equations.IncompressibleFlowProblem(dimension, 0.0)
    spatial_method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        problem, space, spatial_method, spectral_viscosity=plan
    )
    prepared = dynamics.spectral_viscosity
    if prepared is None:
        raise ValueError("The qualification requires a compiled SVV operator.")
    waves = np.stack(
        tuple(
            np.broadcast_to(np.asarray(wave), space.modal_shape)
            for wave in dynamics.projector.wavenumbers
        ),
        axis=-1,
    )
    squared_norm = np.sum(waves**2, axis=-1)
    resolution = count // 2 - (count % 2 == 0)
    oracle_rates = np.asarray(
        [
            _svv_reference_rate(plan, resolution, float(norm))
            for norm in np.sqrt(squared_norm).flat
        ]
    ).reshape(squared_norm.shape)
    retained = np.all(np.abs(waves) <= resolution, axis=-1)
    damped = retained & (oracle_rates < 0.0)
    if not damped.any():
        raise ValueError("The oracle requires a damped retained shear mode.")
    index = np.unravel_index(
        np.argmin(np.where(damped, squared_norm, np.inf)), squared_norm.shape
    )
    mode = tuple(int(round(value)) for value in waves[index])
    first_axis = next(axis for axis, value in enumerate(mode) if value != 0)
    second_axis = (first_axis + 1) % dimension
    polarization = np.zeros((dimension,), dtype=np.float64)
    polarization[first_axis] = -mode[second_axis]
    polarization[second_axis] = mode[first_axis]
    polarization /= np.linalg.norm(polarization)
    nodes = jnp.meshgrid(*(axis.nodes for axis in space.axes), indexing="ij")
    phase = sum(value * node for value, node in zip(mode, nodes, strict=True))
    # A unit polarization perpendicular to k gives an exact steady Euler mode:
    # (a*cos(k.x)).grad(a*cos(k.x)) = 0 since a.k = 0.
    base = jnp.cos(phase)[..., None] * jnp.asarray(polarization)
    initial = dynamics.project_state(base)
    temporal_method, result = _etdrk_solve(dynamics, initial, dt, steps)
    final_time = dt * steps
    mode_norm = math.sqrt(sum(value**2 for value in mode))
    rate = _svv_reference_rate(plan, resolution, mode_norm)
    velocity_error = float(
        jnp.max(
            jnp.abs(
                dynamics.reconstruct_state(result.states[-1])
                - math.exp(rate * final_time) * base
            )
        )
    )
    diagnostics = (
        dynamics.diagnostics(0.0, initial),
        dynamics.diagnostics(final_time, result.states[-1]),
    )
    # Orthonormal Fourier convention: E(0)=volume/4 for this unit cosine mode.
    expected_energies = tuple(
        (2.0 * math.pi) ** dimension / 4.0 * math.exp(2.0 * rate * time)
        for time in (0.0, final_time)
    )
    support = _svv_support(prepared)
    decay_case, energy_case = f"shear-decay-{dimension}d", f"energy-identity-{dimension}d"
    decay: dict[str, Any] = {
        **support,
        "integer_mode": mode,
        "analytic_decay_rate": rate,
        "oracle_resolution": resolution,
        "prepared_resolution": prepared.resolution,
        "prepared_rate_error": abs(float(prepared.decay_rate[index]) - rate),
        "maximum_velocity_error": velocity_error,
        "divergence_norm": max(float(value.divergence_norm) for value in diagnostics),
        "solver_successful": bool(result.successful),
        "discretization_prepared_id": space.prepared_id,
        "spatial_method_id": spatial_method.method_id,
        "temporal_method_id": temporal_method.method_id,
        "compilation_id": dynamics.compilation_id,
        "spectral_viscosity_id": prepared.prepared_id,
    }
    energy: dict[str, Any] = {
        **support,
        "maximum_energy_error": max(
            abs(float(value.kinetic_energy) - expected)
            for value, expected in zip(diagnostics, expected_energies, strict=True)
        ),
        "maximum_dissipation_error": max(
            abs(float(value.spectral_viscosity_dissipation) + 2.0 * rate * expected)
            for value, expected in zip(diagnostics, expected_energies, strict=True)
        ),
        "maximum_energy_rate_error": max(
            abs(float(value.spectral_viscosity_energy_rate) - 2.0 * rate * expected)
            for value, expected in zip(diagnostics, expected_energies, strict=True)
        ),
        "maximum_energy_balance_defect": max(
            abs(float(value.energy_balance_defect)) for value in diagnostics
        ),
        "spectral_viscosity_dissipation": float(
            diagnostics[-1].spectral_viscosity_dissipation
        ),
        "spectral_viscosity_energy_rate": float(
            diagnostics[-1].spectral_viscosity_energy_rate
        ),
        "positive_dissipation": rate < 0.0
        and all(
            float(value.spectral_viscosity_dissipation) > 0.0 for value in diagnostics
        ),
        "diagnostics_finite": all(bool(value.finite) for value in diagnostics),
    }
    gates = [
        boolean_gate(
            f"{decay_case}:resolution-oracle",
            f"{decay_case}.prepared_resolution_matches_oracle",
            prepared.resolution == resolution,
        ),
        numeric_gate(
            f"{decay_case}:rate-oracle",
            f"{decay_case}.prepared_rate_error",
            decay["prepared_rate_error"],
            invariant_tolerance,
        ),
        numeric_gate(
            f"{decay_case}:solution",
            f"{decay_case}.maximum_velocity_error",
            velocity_error,
            solution_tolerance,
        ),
        numeric_gate(
            f"{decay_case}:divergence",
            f"{decay_case}.divergence_norm",
            decay["divergence_norm"],
            invariant_tolerance,
        ),
        boolean_gate(
            f"{decay_case}:solver",
            f"{decay_case}.solver_successful",
            bool(result.successful),
        ),
        boolean_gate(
            f"{energy_case}:dissipation",
            f"{energy_case}.positive_dissipation",
            energy["positive_dissipation"],
        ),
        boolean_gate(
            f"{energy_case}:finite",
            f"{energy_case}.diagnostics_finite",
            energy["diagnostics_finite"],
        ),
    ]
    for metric in (
        "maximum_energy_error",
        "maximum_dissipation_error",
        "maximum_energy_rate_error",
        "maximum_energy_balance_defect",
    ):
        gates.append(
            numeric_gate(
                f"{energy_case}:{metric}",
                f"{energy_case}.{metric}",
                energy[metric],
                invariant_tolerance,
            )
        )
    return {decay_case: decay, energy_case: energy}, gates


def _svv_low_mode_evidence(
    plan: SpectralVanishingViscosityPlan,
    counts: tuple[int, ...],
    mode: int,
    tolerance: float,
    /,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, Any]] = []
    for count in counts:
        space = _periodic_space(count)
        prepared = plan.prepare(space)
        resolution = count // 2 - (count % 2 == 0)
        indices = tuple(
            int(np.flatnonzero(np.asarray(axis.modes.mode_numbers) == number)[0])
            for axis, number in zip(space.axes, (0, mode), strict=True)
        )
        rate = float(prepared.decay_rate[indices])
        rows.append(
            {
                **_svv_support(prepared),
                "mode_count": count,
                "resolution": prepared.resolution,
                "oracle_resolution": resolution,
                "cutoff_mode": prepared.cutoff_mode,
                "decay_rate": rate,
                "analytic_decay_rate": _svv_reference_rate(plan, resolution, mode),
                "absolute_rate_upper_bound": (
                    plan.strength
                    * resolution ** (1.0 - 2.0 * plan.order)
                    * mode ** (2.0 * plan.order)
                ),
                "prepared_id": prepared.prepared_id,
            }
        )
    rates = [abs(row["decay_rate"]) for row in rows]
    evidence: dict[str, Any] = {
        "operator_scaling_admissible": rows[-1]["operator_scaling_admissible"],
        "cited_convergence_supported": rows[-1]["cited_convergence_supported"],
        "support_reason": rows[-1]["support_reason"],
        "integer_mode": [0, mode],
        "spatial_dimension": 2,
        "resolutions": rows,
        "maximum_rate_error": max(
            abs(row["decay_rate"] - row["analytic_decay_rate"]) for row in rows
        ),
        "maximum_upper_bound_excess": max(
            0.0,
            *(abs(row["decay_rate"]) - row["absolute_rate_upper_bound"] for row in rows),
        ),
        "nonincreasing_absolute_rate": all(
            right <= left + tolerance
            for left, right in zip(rates[:-1], rates[1:], strict=True)
        ),
        "vanishing_upper_bound_exponent": 1.0 - 2.0 * plan.order,
        "upper_bound_decreases": (
            rows[-1]["absolute_rate_upper_bound"] < rows[0]["absolute_rate_upper_bound"]
        ),
        "last_absolute_rate": rates[-1],
        "finite": all(math.isfinite(rate) for rate in rates),
        "interpretation": "Finite-resolution fixed-mode operator evidence, not full-method or rough-data convergence.",
    }
    case = "fixed-low-mode-vanishing"
    gates = [
        boolean_gate(
            f"{case}:resolution-oracle",
            f"{case}.prepared_resolution_matches_oracle",
            all(row["resolution"] == row["oracle_resolution"] for row in rows),
        ),
        numeric_gate(
            f"{case}:oracle",
            f"{case}.maximum_rate_error",
            evidence["maximum_rate_error"],
            tolerance,
        ),
        numeric_gate(
            f"{case}:bound",
            f"{case}.maximum_upper_bound_excess",
            evidence["maximum_upper_bound_excess"],
            tolerance,
        ),
    ]
    for metric in ("nonincreasing_absolute_rate", "upper_bound_decreases", "finite"):
        gates.append(
            boolean_gate(f"{case}:{metric}", f"{case}.{metric}", evidence[metric])
        )
    return evidence, gates


def periodic_spectral_vanishing_viscosity_qualification(
    *,
    strength: float,
    order: float,
    cutoff_scale: float,
    cutoff_exponent: float,
    kernel: SpectralViscosityKernel,
    highpass_exponent: int | None = None,
    mode_count: int = 12,
    low_mode_counts: tuple[int, ...] = (12, 24, 48),
    fixed_low_mode: int = 1,
    step_size: float = 0.025,
    steps: int = 4,
    solution_tolerance: float = 1.0e-8,
    invariant_tolerance: float = 1.0e-9,
    external_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Qualify diagonal SVV and ETDRK4 without certifying cited convergence.

    Scientific parameters are explicit: Rohner--Mishra (2024), section 2
    supplies the scaling, not a universal strength or cutoff. Resolution,
    horizon, and tolerance defaults are finite qualification-campaign policy.
    """
    if mode_count < 8 or steps < 1 or not math.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("SVV qualification count/steps/step size are invalid.")
    if (
        fixed_low_mode < 1
        or len(low_mode_counts) < 3
        or any(
            count < 8 or (count - 1) // 2 < fixed_low_mode for count in low_mode_counts
        )
        or any(
            right <= left
            for left, right in zip(low_mode_counts[:-1], low_mode_counts[1:], strict=True)
        )
    ):
        raise ValueError(
            "Low-mode counts must strictly increase and retain the fixed mode "
            "at three or more resolutions."
        )
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (solution_tolerance, invariant_tolerance)
    ):
        raise ValueError("SVV qualification tolerances must be finite and positive.")
    plan = SpectralVanishingViscosityPlan(
        strength=strength,
        order=order,
        cutoff_scale=cutoff_scale,
        cutoff_exponent=cutoff_exponent,
        kernel=kernel,
        highpass_exponent=highpass_exponent,
    )
    metrics = {}
    gates = []
    for dimension in (2, 3):
        cases, case_gates = _svv_shear_evidence(
            plan,
            mode_count,
            dimension,
            step_size,
            steps,
            solution_tolerance,
            invariant_tolerance,
        )
        metrics.update(cases)
        gates.extend(case_gates)
    low_mode, low_gates = _svv_low_mode_evidence(
        plan, low_mode_counts, fixed_low_mode, invariant_tolerance
    )
    metrics["fixed-low-mode-vanishing"] = low_mode
    gates.extend(low_gates)
    route = "periodic-spectral-vanishing-viscosity"
    return make_qualification_artifact(
        route=route,
        support_tuple=phx.qualification.SupportTuple(
            CAPABILITY,
            {
                "route": route,
                "spatial_method": "periodic-incompressible-rotational-svv-diagonal",
                "temporal_method": "etdrk4",
                "kernel": plan.kernel,
                "spatial_dimensions": "2,3",
                "cited_convergence_supported": "false",
            },
        ),
        inputs={
            "domain": "isotropic [0,2*pi)^d",
            "initial_condition": "u=a*cos(n.x); |a|=1; a.n=0; smallest damped retained |n|",
            "molecular_viscosity": 0.0,
            "spectral_viscosity_plan_id": plan.plan_id,
        },
        reference={
            "source": "native-analytic",
            "publication": "https://www.sam.math.ethz.ch/sam_reports/reports_final/reports2024/2024-04_rev1.pdf",
            "section": "2 (spectral hyperviscosity approximation)",
            "decay": "exp(lambda_n*t)",
            "energy": "(2*pi)^d/4 * exp(2*lambda_n*t)",
            "dissipation": "-2*lambda_n*energy",
            "fixed_mode_bound": "|lambda_n| <= strength*K^(1-2*order)*|n|^(2*order)",
            "nonclaim": "Operator scaling does not certify initialization or full-method rough-data convergence.",
        },
        configuration={
            "strength": plan.strength,
            "order": plan.order,
            "cutoff_scale": plan.cutoff_scale,
            "cutoff_exponent": plan.cutoff_exponent,
            "kernel": plan.kernel,
            "highpass_exponent": plan.highpass_exponent,
            "mode_count": mode_count,
            "low_mode_counts": low_mode_counts,
            "fixed_low_mode": fixed_low_mode,
            "step_size": step_size,
            "steps": steps,
            "solution_tolerance": solution_tolerance,
            "invariant_tolerance": invariant_tolerance,
            "timing_gate": False,
        },
        metrics=metrics,
        gates=gates,
        external_reference=external_reference,
    )


def _statistical_configuration(
    mode_counts: tuple[int, ...],
    members: int,
    steps: int,
    step_size: float,
    num_projections: int,
    refinement_z: float,
) -> dict[str, Any]:
    counts = tuple(positive_integer(n, "mode_counts") for n in mode_counts)
    members = positive_integer(members, "members")
    steps = positive_integer(steps, "steps")
    projections = positive_integer(num_projections, "num_projections")
    dt = finite_real_scalar(step_size, "step_size")
    z = finite_real_scalar(refinement_z, "refinement_z")
    if (
        len(counts) < 3
        or counts[0] <= 8
        or any(b <= a or b % a for a, b in zip(counts[:-1], counts[1:], strict=True))
        or members < 3
        or projections < 2
        or dt <= 0
        or z <= 0
    ):
        raise ValueError(
            "Campaign requires at least three nested grids with coarse_count > 8, "
            "members >= 3, projections >= 2, and positive dt and refinement_z."
        )
    return {
        "mode_counts": counts,
        "members": members,
        "steps": steps,
        "step_size": dt,
        "num_projections": projections,
        "refinement_z": z,
        "comparison_nodes": "Each pair's coarse injection nodes; distances integrate "
        "the component-mean W1 with the same physical-domain quadrature measure.",
        "uncertainty": "Delete the same coupled member at every resolution; paired "
        "jackknife SE for distances and their consecutive differences.",
        "maximum_workspace_bytes": 2**28,
        "time_policy": "Shared dt/save grid, fail-closed Courant guard at every resolution.",
        "domain_length": 2.0 * math.pi,
        "radial_averaging": "ball",
        "invariant_tolerance": 1e-9,
        "timing_gate": False,
    }


def _statistical_initial_states(
    spaces: tuple[TensorSpectralDiscretization, ...],
    config: Mapping[str, Any],
) -> tuple[Any, tuple[Any, ...]]:
    flow = phx.applications.incompressible_flow
    field = phx.stochastic.PeriodicPowerLawFieldPlan(
        phx.stochastic.PowerLawSpectralAmplitude(
            config["power_law_exponent"], scale=config["velocity_scale"]
        ),
        component_shape=(2,),
        solenoidal=True,
    )
    measure = flow.PeriodicFlowInitialMeasure(flow.PowerLawVelocityFamily(field)).prepare(
        spaces[-1], member_chunk_size=min(config["members"], 4)
    )
    finest = measure.realize(
        jax.random.key(config["seed"]),
        jnp.arange(config["members"], dtype=jnp.uint32),
    )
    initials = tuple(
        flow.restrict_periodic_flow_initial_state(
            finest, space, maximum_bytes=config["maximum_workspace_bytes"]
        )
        for space in spaces[:-1]
    ) + (finest,)
    return measure, initials


def _statistical_invariant_evidence(
    dynamics: Any,
    result: Any,
    row: dict[str, Any],
    tolerance: float,
) -> list[dict[str, object]]:
    count = row["mode_count"]
    states = result.states.reshape((-1, *dynamics.state_shape))
    times = jnp.tile(result.times, result.states.shape[0])
    diagnostics = jax.vmap(lambda t, u: dynamics.diagnostics(t, u))(times, states)
    row.update(
        {
            "maximum_divergence": float(
                jnp.max(jnp.abs(jax.vmap(dynamics.projector.divergence)(states)))
            ),
            "maximum_energy_balance_defect": float(
                jnp.max(jnp.abs(diagnostics.energy_balance_defect))
            ),
            "minimum_svv_dissipation": float(
                jnp.min(diagnostics.spectral_viscosity_dissipation)
            ),
        }
    )
    return [
        numeric_gate(
            f"divergence:{count}",
            "maximum_divergence",
            row["maximum_divergence"],
            tolerance,
        ),
        numeric_gate(
            f"energy-budget:{count}",
            "maximum_energy_balance_defect",
            row["maximum_energy_balance_defect"],
            tolerance,
        ),
        boolean_gate(
            f"svv-dissipation:{count}",
            "positive_svv_dissipation",
            bool(jnp.all(diagnostics.spectral_viscosity_dissipation > 0)),
        ),
        boolean_gate(
            f"diagnostics-finite:{count}",
            "diagnostics_finite",
            bool(jnp.all(diagnostics.finite)),
        ),
    ]


def _statistical_execute_resolution(
    space: TensorSpectralDiscretization,
    initial: Any,
    viscosity: SpectralVanishingViscosityPlan,
    config: Mapping[str, Any],
) -> tuple[Any, dict[str, Any], dict[str, object], list[dict[str, object]]]:
    flow = phx.applications.incompressible_flow
    spatial = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    dynamics = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.0),
        space,
        spatial,
        spectral_viscosity=viscosity,
    )
    method = phx.solver.CourantGuardedETDRKMethod(
        phx.solver.ETDRKMethod(4),
        maximum_courant_number=config["maximum_courant_number"],
    )
    prepared = flow.prepare_periodic_flow_ensemble(
        dynamics,
        method,
        initial,
        start_time=0.0,
        end_time=config["steps"] * config["step_size"],
        step_size=config["step_size"],
        member_chunk_size=min(config["members"], 4),
    )
    wrapper = prepared.execute()
    result = wrapper.result

    def restrictions(values: jax.Array) -> Any:
        return jax.vmap(
            lambda u: dynamics.advective_courant_restriction(
                u, maximum_courant_number=config["maximum_courant_number"]
            )
        )(values)

    initial_restriction = restrictions(initial.velocity)
    terminal = restrictions(result.states[:, -1])
    native = result.solutions.evidence
    successful = bool(
        jnp.all(result.status == int(phx.solver.DeterministicEnsembleStatus.SUCCESS))
    ) and bool(jnp.all(result.valid))
    finite = all(
        np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result.states)
    )
    row: dict[str, Any] = {
        "mode_count": space.modal_shape[0],
        "resolution": space.modal_shape,
        "solver_successful": successful,
        "trajectory_finite": finite,
        "member_status": np.asarray(result.status).tolist(),
        "initial_maximum_step": np.asarray(initial_restriction.maximum_step).tolist(),
        "refused_step": None
        if native is None
        else np.asarray(native.refused_step).tolist(),
        "guard_evidence": {
            "held_terminal_state_maximum_step": np.asarray(
                terminal.maximum_step
            ).tolist(),
            "held_terminal_state_courant_frequency": np.asarray(
                terminal.courant_frequency
            ).tolist(),
            "restriction_finite": np.asarray(terminal.finite).tolist(),
            "step_residuals": np.asarray(result.solutions.residuals).tolist(),
            "reference_model": terminal.reference_model,
            "nonlinear_stability_certified": terminal.nonlinear_stability_certified,
            "native_refused_evidence": []
            if native is None
            else [np.asarray(leaf).tolist() for leaf in jax.tree.leaves(native.refused)],
        },
    }
    count = row["mode_count"]
    gates = [
        gate_outcome(
            f"solver-success:{count}",
            "passed" if successful else "failed",
            "Every member completed the native guarded trajectory."
            if successful
            else "Native solver/guard refusal stopped a member; failed paths are not compared.",
            metric="solver_successful",
            observed={"successful": successful, "guard_evidence": row["guard_evidence"]},
            criterion={"comparison": "is", "expected": True},
        ),
        boolean_gate(f"trajectory-finite:{count}", "trajectory_finite", finite),
    ]
    if successful and finite:
        gates.extend(
            _statistical_invariant_evidence(
                dynamics, result, row, config["invariant_tolerance"]
            )
        )
    svv = dynamics.spectral_viscosity
    if svv is None:
        raise ValueError("Campaign requires a compiled SVV operator.")
    identities = {
        "mode_count": count,
        "compilation_id": dynamics.compilation_id,
        "method_id": prepared.ensemble.plan.problem.method.method_id,
        "spatial_method_id": spatial.method_id,
        "ensemble_prepared_id": prepared.ensemble.prepared_id,
        "ensemble_result_id": result.result_id,
        "discretization_prepared_id": space.prepared_id,
        "spectral_viscosity_plan_id": viscosity.plan_id,
        "spectral_viscosity_prepared_id": svv.prepared_id,
        "measure_id": initial.measure_id,
        "coupling_id": initial.coupling_id,
    }
    return wrapper, row, identities, gates


def _statistical_jackknife(pair: Any) -> np.ndarray:
    """Exact equal-weight empirical W1 on the public metric's injection nodes."""
    c, f = pair.coarse, pair.fine
    coarse, fine = c.initial_state.discretization, f.initial_state.discretization
    cv = np.asarray(jax.vmap(coarse.reconstruct)(c.result.states[:, -1]))
    fv = np.asarray(jax.vmap(fine.reconstruct)(f.result.states[:, -1]))
    injection = (
        (slice(None),)
        + tuple(
            slice(None, None, fa.nodes.size // ca.nodes.size)
            for ca, fa in zip(coarse.axes, fine.axes, strict=True)
        )
        + (slice(None),)
    )
    fv = fv[injection]
    members = cv.shape[0]
    cv, fv = cv.reshape((members, -1, 2)), fv.reshape((members, -1, 2))
    weights = np.asarray(coarse.quadrature_weights).reshape(-1)
    estimates = np.empty((members,), dtype=np.float64)
    for removed in range(members):
        keep = np.arange(members) != removed
        differences = np.abs(np.sort(cv[keep], axis=0) - np.sort(fv[keep], axis=0))
        estimates[removed] = np.sum(weights * differences.mean(axis=(0, 2)))
    return estimates


def _jackknife_standard_error(estimates: np.ndarray) -> np.ndarray:
    members = estimates.shape[0]
    return np.sqrt(
        (members - 1)
        / members
        * np.sum((estimates - estimates.mean(axis=0)) ** 2, axis=0)
    )


def _statistical_comparison(
    wrappers: Sequence[Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    flow = phx.applications.incompressible_flow
    comparisons: list[dict[str, Any]] = []
    deleted: list[np.ndarray] = []
    nonclaims: list[str] = []
    for j, (coarse, fine) in enumerate(zip(wrappers[:-1], wrappers[1:], strict=True)):
        pair = flow.align_coupled_ensembles(coarse, fine)
        policy = {
            "member_chunk_size": min(config["members"], 4),
            "maximum_workspace_bytes": config["maximum_workspace_bytes"],
        }
        distance = flow.componentwise_empirical_wasserstein(pair, p=1.0, **policy)
        upper = flow.paired_coupling_upper_bound(pair, p=1.0, **policy)
        count = coarse.initial_state.discretization.modal_shape[0]
        sliced = flow.sliced_joint_marginal_distance(
            pair,
            ((0, count // 2), (count, count + count // 2)),
            key=jax.random.key(config["seed"] + 1 + j),
            num_projections=config["num_projections"],
            p=1.0,
            **policy,
        )
        estimates = _statistical_jackknife(pair)
        deleted.append(estimates)
        comparisons.append(
            {
                "mode_counts": config["mode_counts"][j : j + 2],
                "componentwise_empirical_wasserstein": float(distance.value),
                "jackknife_standard_error": float(_jackknife_standard_error(estimates)),
                "paired_coupling_upper_bound": float(upper.value),
                "sliced_joint_marginal_distance": np.asarray(sliced.values).tolist(),
                "projection_standard_error": np.asarray(
                    sliced.projection_standard_error
                ).tolist(),
            }
        )
        nonclaims.extend((distance.nonclaim, upper.nonclaim, sliced.nonclaim))
    jackknife = np.stack(deleted, axis=1)
    distances = np.asarray(
        [row["componentwise_empirical_wasserstein"] for row in comparisons]
    )
    differences = distances[:-1] - distances[1:]
    errors = _jackknife_standard_error(jackknife[:, :-1] - jackknife[:, 1:])
    return {
        "comparisons": comparisons,
        "refinement_differences": differences.tolist(),
        "paired_difference_standard_errors": errors.tolist(),
        "delete_one_member_distances": jackknife.tolist(),
        "member_count": config["members"],
        "refinement_z": config["refinement_z"],
        "nonclaims": list(dict.fromkeys(nonclaims))
        + [
            "Paired jackknife is finite-campaign uncertainty, not a population "
            "confidence bound or statistical-solution convergence certification."
        ],
    }


def _statistical_scaling(
    wrappers: Sequence[Any],
    spaces: tuple[TensorSpectralDiscretization, ...],
    radii: jax.Array,
) -> dict[str, Any]:
    moments = tuple(
        phx.discretization.PeriodicRadialIncrementMomentPlan(
            space, radii, averaging="ball", component_shape=(2,)
        )(wrapper.result.states[:, -1])
        for space, wrapper in zip(spaces, wrappers, strict=True)
    )
    scaling = phx.applications.incompressible_flow.StructureFunctionScalingPlan(
        (float(radii[0]), float(radii[-1]))
    ).evaluate(moments, spaces)
    return {
        "fits": [
            {
                "exponent": f.exponent,
                "standard_error": f.standard_error,
                "constant": f.constant,
                "residual": f.residual,
                "condition_number": f.condition_number,
                "status": f.status,
                "reason": f.reason,
            }
            for f in scaling.fits
        ],
        "common_exponent": scaling.common_exponent,
        "envelope_constant": scaling.envelope_constant,
        "consistent_within_campaign": scaling.consistent_within_campaign,
        "nonclaim": scaling.nonclaim,
    }


def _statistical_metric_gates(metrics: Mapping[str, Any]) -> list[dict[str, object]]:
    differences = np.asarray(metrics["refinement_differences"])
    errors = np.asarray(metrics["paired_difference_standard_errors"])
    z = metrics["refinement_z"]
    finite = bool(
        np.isfinite(differences).all()
        and np.isfinite(errors).all()
        and np.isfinite(metrics["delete_one_member_distances"]).all()
    )
    for row in metrics["comparisons"]:
        finite = finite and all(
            np.isfinite(row[key]).all()
            for key in (
                "componentwise_empirical_wasserstein",
                "jackknife_standard_error",
                "paired_coupling_upper_bound",
                "sliced_joint_marginal_distance",
                "projection_standard_error",
            )
        )
    outcome = (
        "failed"
        if finite and np.any(differences < -z * errors)
        else "passed"
        if finite and np.all(differences > z * errors)
        else "inconclusive"
    )
    scaling = metrics["scaling"]
    reasons = [
        f["reason"] or "Scaling fit abstained."
        for f in scaling["fits"]
        if f["status"] == "abstained"
    ]
    if not scaling["consistent_within_campaign"]:
        reasons.append("Scaling exponents are inconsistent within the coupled campaign.")
    return [
        boolean_gate("metric-evidence-finite", "finite_metric_evidence", finite),
        gate_outcome(
            "finite-campaign-refinement",
            outcome,
            {
                "passed": "Every consecutive W1 distance significantly decreased.",
                "failed": "A consecutive W1 distance significantly increased.",
                "inconclusive": "Refinement is noise-limited or metric evidence is nonfinite.",
            }[outcome],
            observed={
                "differences": differences.tolist(),
                "paired_standard_errors": errors.tolist(),
            },
            criterion={"z": z, "decrease": "d_j - d_(j+1) > z * paired_SE"},
        ),
        gate_outcome(
            "finite-campaign-scaling",
            "inconclusive" if reasons else "passed",
            "; ".join(reasons)
            if reasons
            else "All scaling fits agreed within the finite campaign.",
            observed=scaling["consistent_within_campaign"],
        ),
    ]


def periodic_statistical_solution_qualification(
    *,
    strength: float,
    order: float,
    cutoff_scale: float,
    cutoff_exponent: float,
    kernel: SpectralViscosityKernel,
    power_law_exponent: float,
    velocity_scale: float,
    maximum_courant_number: float,
    highpass_exponent: int | None = None,
    mode_counts: tuple[int, ...] = (16, 32, 64),
    members: int = 6,
    seed: int = 0,
    step_size: float = 1e-4,
    steps: int = 2,
    num_projections: int = 32,
    refinement_z: float = 2.0,
    external_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Finite coupled refinement campaign, never a convergence proof.

    The exponent is declared campaign policy; the power-law family is motivated
    by Herde et al. (2024), Appendix B.2.2, Eq. (42). Lanthaler--Mishra--
    Parés-Pulido (arXiv:1909.06615) motivates metrics and structure functions,
    not the exponent. Paired delete-one-member jackknife uncertainty belongs
    to the same coupled W1 statistic and its consecutive refinement differences.
    """
    config = _statistical_configuration(
        mode_counts, members, steps, step_size, num_projections, refinement_z
    )
    viscosity = SpectralVanishingViscosityPlan(
        strength=strength,
        order=order,
        cutoff_scale=cutoff_scale,
        cutoff_exponent=cutoff_exponent,
        kernel=kernel,
        highpass_exponent=highpass_exponent,
    )
    config.update(
        {
            "seed": seed,
            "power_law_exponent": power_law_exponent,
            "velocity_scale": velocity_scale,
            "maximum_courant_number": maximum_courant_number,
            "strength": viscosity.strength,
            "order": viscosity.order,
            "cutoff_scale": viscosity.cutoff_scale,
            "cutoff_exponent": viscosity.cutoff_exponent,
            "kernel": viscosity.kernel,
            "highpass_exponent": viscosity.highpass_exponent,
        }
    )
    spaces = tuple(_periodic_space(n) for n in config["mode_counts"])
    measure, initials = _statistical_initial_states(spaces, config)
    radii = jnp.linspace(4 * jnp.pi / config["mode_counts"][0], jnp.pi / 2, 6)
    config["radius_window"] = (float(radii[0]), float(radii[-1]))
    wrappers, runtime, identities, gates = [], [], [], []
    for space, initial in zip(spaces, initials, strict=True):
        wrapper, row, ids, resolution_gates = _statistical_execute_resolution(
            space, initial, viscosity, config
        )
        wrappers.append(wrapper)
        runtime.append(row)
        identities.append(ids)
        gates.extend(resolution_gates)
    metrics: dict[str, Any] = {"runtime": runtime}
    if any(g["outcome"] != "passed" for g in gates):
        metrics["nonclaim"] = (
            "Failed runtime campaign; statistical metrics were not evaluated."
        )
    else:
        metrics.update(_statistical_comparison(wrappers, config))
        metrics["scaling"] = _statistical_scaling(wrappers, spaces, radii)
        gates.extend(_statistical_metric_gates(metrics))
    route = "periodic-statistical-solution"
    return make_qualification_artifact(
        route=route,
        support_tuple=phx.qualification.SupportTuple(
            CAPABILITY,
            {
                "route": route,
                "kernel": viscosity.kernel,
                "spatial_dimensions": "2",
                "spatial_method": "fourier-pseudospectral",
                "temporal_method": "courant-guarded-etdrk4",
                "convergence_certified": "false",
            },
        ),
        inputs={
            "measure_id": initials[-1].measure_id,
            "coupling_id": initials[-1].coupling_id,
            "measure_prepared_id": measure.prepared_id,
            "spectral_viscosity_plan_id": viscosity.plan_id,
            "resolutions": identities,
        },
        reference={
            "source": "finite-native-campaign",
            "publication": "https://arxiv.org/abs/1909.06615",
            "metric_reference": "Lanthaler--Mishra--Parés-Pulido: metrics and structure functions",
            "initial_family_reference": "Herde et al. 2024, Appendix B.2.2, Eq. (42)",
            "nonclaim": "Finite campaign only; compactness and continuum convergence are not certified.",
        },
        configuration=config,
        metrics={"coupled-campaign": metrics},
        gates=gates,
        external_reference=external_reference,
    )


def _channel_space(shape: Any) -> Any:
    nx, ny, nz = shape
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(nx),
            phx.discretization.ChebyshevBasisPlan(ny),
            phx.discretization.FourierBasisPlan(nz),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )


def _channel_restart(dynamics: Any, initial: Any, dt: Any, steps: Any) -> Any:
    method = phx.solver.ChannelSBDF2Method()
    prepared = method.prepare(dynamics, dt)
    state = prepared.initialize(initial, 0.0, None)
    midpoint = None
    for step in range(steps):
        # ty: ignore[invalid-argument-type]
        state = prepared.step(step, step * dt, state, dt, None).accepted_state
        if step + 1 == steps // 2:
            midpoint = state
    assert midpoint is not None
    restarted_prepared = method.prepare(dynamics, dt)
    restarted = midpoint
    for step in range(steps // 2, steps):
        restarted = restarted_prepared.step(
            # ty: ignore[invalid-argument-type]
            step,
            step * dt,
            restarted,
            dt,
            None,
        ).accepted_state
    return method, prepared, state, restarted


def spectral_channel_qualification(
    *,
    shape: tuple[int, int, int] = (4, 10, 4),
    viscosity: float = 0.1,
    step_size: float = 0.01,
    steps: int = 4,
    solution_tolerance: float = 1.0e-8,
    invariant_tolerance: float = 1.0e-9,
    restart_tolerance: float = 1.0e-10,
    minimum_temporal_order: float = 1.8,
    external_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    shape_ = tuple(shape)
    nu, dt, step_count = float(viscosity), float(step_size), int(steps)
    minimum_order = float(minimum_temporal_order)
    if (
        min(shape_) < 4
        or nu <= 0
        or dt <= 0
        or step_count < 4
        or step_count % 2
        or not math.isfinite(minimum_order)
        or minimum_order <= 0.0
    ):
        raise ValueError("Channel shape/viscosity/step size/steps/order are invalid.")
    space = _channel_space(shape_)
    y = space.axes[1].nodes
    couette = jnp.zeros(space.physical_shape + (3,)).at[..., 0].set(y[None, :, None])
    couette_plan = phx.discretization.ChannelStokesPlan(
        space,
        nu,
        lower_wall_velocity=(-1.0, 0.0, 0.0),
        upper_wall_velocity=(1.0, 0.0, 0.0),
    )
    couette_prepared = couette_plan.prepare(1.0)
    couette_result = couette_prepared.solve(space.project(couette))
    couette_error = float(
        jnp.max(jnp.abs(space.reconstruct(couette_result.velocity) - couette))
    )
    poiseuille = (
        jnp.zeros(space.physical_shape + (3,)).at[..., 0].set(1.0 - y[None, :, None] ** 2)
    )
    poiseuille_plan = phx.discretization.ChannelStokesPlan(
        space,
        nu,
        mean_constraint=phx.discretization.ChannelMeanConstraint(
            "pressure_gradient", (2.0 * nu, 0.0)
        ),
    )
    poiseuille_prepared = poiseuille_plan.prepare(1.0)
    poiseuille_result = poiseuille_prepared.solve(space.project(poiseuille))
    poiseuille_error = float(
        jnp.max(jnp.abs(space.reconstruct(poiseuille_result.velocity) - poiseuille))
    )
    base = poiseuille
    base_modal = space.project(base)
    constant = jnp.zeros_like(base).at[..., 0].set(1.0)
    constant_modal = space.project(constant)

    def forcing(time: Any, state: Any, args: Any) -> Any:
        del state, args
        amplitude = 1.0 + 0.1 * jnp.sin(time)
        return 0.1 * jnp.cos(time) * base_modal + 2.0 * nu * amplitude * constant_modal

    manufactured_plan = phx.discretization.ChannelStokesPlan(space, nu)
    spatial_method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    problem = phx.equations.IncompressibleFlowProblem(
        3,
        nu,
        forcing=forcing,
        forcing_id="manufactured-time-dependent-channel-profile",
    )
    dynamics = phx.equations.compile_channel_flow(
        problem, manufactured_plan, spatial_method
    )
    initial = dynamics.project_state(base)
    temporal_method = phx.solver.ChannelSBDF2Method()
    coarse = phx.solver.solve_channel_sbdf2(
        dynamics,
        initial,
        jnp.arange(step_count + 1, dtype="float64") * dt,
        method=temporal_method,
    )
    fine = phx.solver.solve_channel_sbdf2(
        dynamics,
        initial,
        jnp.arange(2 * step_count + 1, dtype="float64") * (0.5 * dt),
        method=temporal_method,
    )
    exact = (1.0 + 0.1 * jnp.sin(step_count * dt)) * base
    coarse_error = float(
        jnp.max(jnp.abs(dynamics.reconstruct_state(coarse.velocity[-1]) - exact))
    )
    fine_error = float(
        jnp.max(jnp.abs(dynamics.reconstruct_state(fine.velocity[-1]) - exact))
    )
    _, temporal_prepared, uninterrupted, restarted = _channel_restart(
        dynamics, initial, dt, step_count
    )
    restart_error = float(
        jnp.max(jnp.abs(uninterrupted.current_velocity - restarted.current_velocity))
    )
    maximum_divergence = float(jnp.nanmax(fine.diagnostics.divergence_norm))
    stats_plan = phx.applications.incompressible_flow.SpectralChannelStatisticsPlan(
        space, density=1.0, kinematic_viscosity=nu
    )
    couette_stats = stats_plan.evaluate(space.project(couette))
    poiseuille_stats = stats_plan.evaluate(space.project(poiseuille))
    order = (
        math.log2(coarse_error / fine_error)
        if coarse_error > 0 and fine_error > 0
        else None
    )
    metrics = {
        "couette": {
            "maximum_velocity_error": couette_error,
            "divergence_norm": float(couette_result.diagnostics.divergence_norm),
            "wall_residual": float(couette_result.diagnostics.wall_residual),
            "lower_wall_shear": float(couette_stats.lower_wall_shear),
            "upper_wall_shear": float(couette_stats.upper_wall_shear),
            "solver_successful": bool(couette_result.successful),
        },
        "poiseuille": {
            "maximum_velocity_error": poiseuille_error,
            "divergence_norm": float(poiseuille_result.diagnostics.divergence_norm),
            "wall_residual": float(poiseuille_result.diagnostics.wall_residual),
            "bulk_velocity": float(poiseuille_stats.bulk_velocity),
            "solver_successful": bool(poiseuille_result.successful),
        },
        "manufactured-sbdf2-refinement-restart": {
            "coarse_maximum_velocity_error": coarse_error,
            "fine_maximum_velocity_error": fine_error,
            "observed_temporal_order": order,
            "restart_maximum_modal_error": restart_error,
            "maximum_divergence_norm": maximum_divergence,
            "solver_successful": bool(coarse.successful & fine.successful),
        },
    }
    gates = (
        numeric_gate(
            "couette-solution",
            "couette.maximum_velocity_error",
            couette_error,
            solution_tolerance,
        ),
        numeric_gate(
            "poiseuille-solution",
            "poiseuille.maximum_velocity_error",
            poiseuille_error,
            solution_tolerance,
        ),
        boolean_gate(
            "couette-solver", "couette.solver_successful", bool(couette_result.successful)
        ),
        boolean_gate(
            "poiseuille-solver",
            "poiseuille.solver_successful",
            bool(poiseuille_result.successful),
        ),
        minimum_numeric_gate(
            "manufactured-order",
            "manufactured-sbdf2-refinement-restart.observed_temporal_order",
            order,
            minimum_order,
        ),
        boolean_gate(
            "manufactured-refinement",
            "manufactured-sbdf2-refinement-restart.fine_error_not_greater",
            fine_error <= coarse_error,
        ),
        numeric_gate(
            "manufactured-restart",
            "manufactured-sbdf2-refinement-restart.restart_maximum_modal_error",
            restart_error,
            restart_tolerance,
        ),
        numeric_gate(
            "manufactured-divergence",
            "manufactured-sbdf2-refinement-restart.maximum_divergence_norm",
            maximum_divergence,
            invariant_tolerance,
        ),
        boolean_gate(
            "manufactured-solver",
            "manufactured-sbdf2-refinement-restart.solver_successful",
            bool(coarse.successful & fine.successful),
        ),
    )
    support = phx.qualification.SupportTuple(
        CAPABILITY,
        {
            "route": "spectral-channel",
            "spatial_method": "fourier-chebyshev-fourier",
            "stokes_route": couette_plan.route,
            "temporal_method": "sbdf2",
            "forcing": "compiler-modal",
            "statistics": "spectral-channel",
        },
    )
    return make_qualification_artifact(
        route="spectral-channel",
        support_tuple=support,
        inputs={
            "domain": [[0.0, 2.0 * math.pi], [-1.0, 1.0], [0.0, 2.0 * math.pi]],
            "shape": list(shape_),
            "couette_profile": "u=y",
            "poiseuille_profile": "u=1-y^2",
            "manufactured_amplitude": "1+0.1*sin(t)",
            "discretization_prepared_id": space.prepared_id,
            "spatial_method_id": spatial_method.method_id,
            "spatial_method_prepared_id": dynamics.spatial_method.prepared_id,
            "temporal_method_id": temporal_method.method_id,
            "temporal_prepared_id": temporal_prepared.method_id,
            "couette_plan_id": couette_plan.plan_id,
            "couette_prepared_id": couette_prepared.prepared_id,
            "poiseuille_plan_id": poiseuille_plan.plan_id,
            "poiseuille_prepared_id": poiseuille_prepared.prepared_id,
            "manufactured_compilation_id": dynamics.compilation_id,
            "statistics_plan_id": stats_plan.plan_id,
        },
        reference={
            "source": "native-analytic",
            "couette": "u=y",
            "poiseuille": "u=1-y^2; pressure-gradient=2*viscosity",
            "manufactured": "(1+0.1*sin(t))*(1-y^2)",
        },
        configuration={
            "shape": list(shape_),
            "viscosity": nu,
            "step_size": dt,
            "steps": step_count,
            "solution_tolerance": float(solution_tolerance),
            "invariant_tolerance": float(invariant_tolerance),
            "restart_tolerance": float(restart_tolerance),
            "minimum_temporal_order": minimum_order,
            "timing_gate": False,
        },
        metrics=metrics,
        gates=gates,
        external_reference=external_reference,
    )


def _maximum_abs(values: Any) -> Any:
    return max(float(jnp.max(jnp.abs(value))) for value in values)


def _mac_route(plan: Any, result: Any) -> Any:
    linear, transform, hybrid = result.linear, result.transform, result.hybrid
    return {
        "requested_solve_method": plan.solve_method,
        "selected_solve_method": result.solve_method,
        "constant_route": plan.constant_route,
        "plan_id": plan.plan_id,
        "projection_id": result.projection_id,
        "operator_id": plan.operator_id,
        "pressure_problem_id": plan.pressure_problem_id,
        "maximum_resource_bytes": int(result.maximum_resource_bytes),
        "hybrid_action_defect": (
            None if hybrid is None else float(result.hybrid_action_defect)
        ),
        "pressure_residual_norm": _maximum_abs((result.pressure_residual,)),
        "divergence_before_norm": _maximum_abs((result.divergence_before,)),
        "divergence_after_norm": _maximum_abs((result.divergence_after,)),
        "gauge_defect": float(result.gauge_defect),
        "iterations": None if linear is None else int(linear.diagnostics.iterations),
        "matvec_count": None if linear is None else int(linear.diagnostics.matvec_count),
        "route_residual_norm": (
            float(transform.residual_norm)
            if transform is not None
            else float(hybrid.residual_norm)
            if hybrid is not None
            else float(linear.diagnostics.residual_norm)
        ),
        "hybrid_resources": (
            None
            if hybrid is None
            else {
                "line_count": int(hybrid.resources.line_count),
                "line_size": int(hybrid.resources.line_size),
                "factor_count": int(hybrid.resources.factor_count),
                "factor_bytes": int(hybrid.resources.factor_bytes),
                "workspace_bytes": int(hybrid.resources.workspace_bytes),
                "total_bytes": int(hybrid.resources.total_bytes),
            }
        ),
        "finite": bool(result.finite),
        "converged": bool(result.converged),
    }


def _mac_poiseuille_refinement(count: int, /) -> tuple[float, float, str, str, str]:
    edges = jnp.linspace(0.0, 1.0, count + 1) ** 1.5
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
            phx.discretization.NonuniformCellAxisSpec(edges),
            phx.discretization.UniformCellAxisSpec(count, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [2.0 * jnp.pi, 1.0, 2.0 * jnp.pi]]))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(discretization).prepare()
    zero = phx.discretization.MACBoundaryProvider(jnp.zeros((3,)))
    walls = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("y", "lower", "no-slip", provider=zero),
            phx.discretization.MACBoundarySide("y", "upper", "no-slip", provider=zero),
        ),
    ).prepare()
    momentum = phx.discretization.MACMomentumPlan(operators, boundaries=walls).prepare()
    x_faces, y_faces, z_faces = discretization.face_centers
    velocity = (
        x_faces[..., 1] * (1.0 - x_faces[..., 1]),
        jnp.zeros(y_faces.shape[:-1], dtype=y_faces.dtype),
        jnp.zeros(z_faces.shape[:-1], dtype=z_faces.dtype),
    )
    laplacian = momentum.laplacian(velocity)
    residual = laplacian[0] + 2.0
    maximum_error = _maximum_abs((residual, laplacian[1], laplacian[2]))
    dual_measure = operators.face_dual_measures[0]
    weighted_l1_error = float(
        jnp.sum(dual_measure * jnp.abs(residual)) / jnp.sum(dual_measure)
    )
    return (
        maximum_error,
        weighted_l1_error,
        grid.prepared_id,
        operators.prepared_id,
        momentum.prepared_id,
    )


def mac_qualification(
    *,
    periodic_count: int = 8,
    stretched_count: int = 6,
    route_tolerance: float = 1.0e-7,
    invariant_tolerance: float = 1.0e-8,
    minimum_spatial_order: float = 0.8,
    external_reference: Mapping[str, object] | None = None,
) -> dict[str, object]:
    pn, sn = int(periodic_count), int(stretched_count)
    minimum_order = float(minimum_spatial_order)
    if min(pn, sn) < 4 or not math.isfinite(minimum_order) or minimum_order <= 0.0:
        raise ValueError(
            "MAC qualification counts and minimum_spatial_order are invalid."
        )
    periodic_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(pn, periodic=True),
            phx.discretization.UniformCellAxisSpec(pn, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [2.0 * jnp.pi, 2.0 * jnp.pi]]))
    periodic_fv = phx.discretization.FiniteVolumePlan(periodic_grid).prepare()
    periodic_operators = phx.discretization.MACOperatorPlan(periodic_fv).prepare()
    x_faces, y_faces = periodic_fv.face_centers
    taylor_green = (
        jnp.sin(x_faces[..., 0]) * jnp.cos(x_faces[..., 1]),
        -jnp.cos(y_faces[..., 0]) * jnp.sin(y_faces[..., 1]),
    )
    full_plan = phx.solver.MACPressureProjectionPlan(
        periodic_operators, solve_method="transform", tolerance=1.0e-10
    )
    periodic_iterative_plan = phx.solver.MACPressureProjectionPlan(
        periodic_operators, solve_method="iterative", tolerance=1.0e-10
    )
    full = full_plan.project(taylor_green, 1.0)
    periodic_iterative = periodic_iterative_plan.project(taylor_green, 1.0)
    periodic_difference = _maximum_abs(
        tuple(
            a - b for a, b in zip(full.velocity, periodic_iterative.velocity, strict=True)
        )
    )
    edges = jnp.linspace(0.0, 1.0, sn + 1) ** 1.5
    stretched_grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(sn, periodic=True),
            phx.discretization.NonuniformCellAxisSpec(edges),
            phx.discretization.UniformCellAxisSpec(sn, periodic=True),
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [2.0 * jnp.pi, 1.0, 2.0 * jnp.pi]]))
    stretched_fv = phx.discretization.FiniteVolumePlan(stretched_grid).prepare()
    operators = phx.discretization.MACOperatorPlan(stretched_fv).prepare()
    zero = phx.discretization.MACBoundaryProvider(jnp.zeros((3,)))
    upper = phx.discretization.MACBoundaryProvider(jnp.asarray([1.0, 0.0, 0.0]))
    zero_walls = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("y", "lower", "no-slip", provider=zero),
            phx.discretization.MACBoundarySide("y", "upper", "no-slip", provider=zero),
        ),
    ).prepare()
    couette_walls = phx.discretization.MACBoundaryPlan(
        operators,
        (
            phx.discretization.MACBoundarySide("y", "lower", "no-slip", provider=zero),
            phx.discretization.MACBoundarySide("y", "upper", "no-slip", provider=upper),
        ),
    ).prepare()
    couette_momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=couette_walls
    ).prepare()
    poiseuille_momentum = phx.discretization.MACMomentumPlan(
        operators, boundaries=zero_walls
    ).prepare()
    xf, yf, zf = stretched_fv.face_centers
    zeros_y = jnp.zeros(yf.shape[:-1], dtype=yf.dtype)
    zeros_z = jnp.zeros(zf.shape[:-1], dtype=zf.dtype)
    couette = (xf[..., 1], zeros_y, zeros_z)
    poiseuille = (xf[..., 1] * (1.0 - xf[..., 1]), zeros_y, zeros_z)
    couette_error = _maximum_abs(couette_momentum.laplacian(couette))
    poiseuille_laplacian = poiseuille_momentum.laplacian(poiseuille)
    poiseuille_error = _maximum_abs(
        (
            poiseuille_laplacian[0] + 2.0,
            poiseuille_laplacian[1],
            poiseuille_laplacian[2],
        )
    )
    poiseuille_dual_measure = operators.face_dual_measures[0]
    poiseuille_weighted_l1_error = float(
        jnp.sum(poiseuille_dual_measure * jnp.abs(poiseuille_laplacian[0] + 2.0))
        / jnp.sum(poiseuille_dual_measure)
    )
    (
        fine_poiseuille_error,
        fine_poiseuille_weighted_l1_error,
        fine_grid_id,
        fine_operators_id,
        fine_momentum_id,
    ) = _mac_poiseuille_refinement(2 * sn)
    poiseuille_order = (
        math.log2(poiseuille_weighted_l1_error / fine_poiseuille_weighted_l1_error)
        if poiseuille_weighted_l1_error > 0.0 and fine_poiseuille_weighted_l1_error > 0.0
        else None
    )
    couette_stats_plan = phx.applications.incompressible_flow.MACPlaneWallStatisticsPlan(
        operators,
        density=1.0,
        kinematic_viscosity=1.0,
        upper_wall_velocity=jnp.asarray((1.0, 0.0, 0.0)),
    )
    poiseuille_stats_plan = (
        phx.applications.incompressible_flow.MACPlaneWallStatisticsPlan(
            operators, density=1.0, kinematic_viscosity=1.0
        )
    )
    couette_stats = couette_stats_plan.evaluate(couette)
    poiseuille_stats = poiseuille_stats_plan.evaluate(poiseuille)
    route_velocity = (
        jnp.sin(xf[..., 0]) * jnp.cos(jnp.pi * xf[..., 1]) * jnp.cos(xf[..., 2]),
        -2.0 * jnp.cos(yf[..., 0]) * jnp.sin(jnp.pi * yf[..., 1]) * jnp.cos(yf[..., 2]),
        zeros_z,
    )
    hybrid_plan = phx.solver.MACPressureProjectionPlan(
        operators,
        boundaries=zero_walls,
        solve_method="hybrid",
        hybrid_line_axis=1,
        tolerance=1.0e-10,
    )
    stretched_iterative_plan = phx.solver.MACPressureProjectionPlan(
        operators, boundaries=zero_walls, solve_method="iterative", tolerance=1.0e-10
    )
    hybrid = hybrid_plan.project(route_velocity, 1.0)
    stretched_iterative = stretched_iterative_plan.project(route_velocity, 1.0)
    stretched_difference = _maximum_abs(
        tuple(
            a - b
            for a, b in zip(hybrid.velocity, stretched_iterative.velocity, strict=True)
        )
    )
    metrics = {
        "periodic-taylor-green": {
            "full": _mac_route(full_plan, full),
            "iterative": _mac_route(periodic_iterative_plan, periodic_iterative),
            "route_velocity_maximum_difference": periodic_difference,
        },
        "stretched-couette": {
            "laplacian_maximum_error": couette_error,
            "lower_wall_shear": float(
                couette_stats.lower_wall_shear[couette_stats.streamwise_axis]
            ),
            "upper_wall_shear": float(
                couette_stats.upper_wall_shear[couette_stats.streamwise_axis]
            ),
            "statistics_successful": bool(couette_stats.successful),
        },
        "stretched-poiseuille": {
            "coarse_laplacian_maximum_error": poiseuille_error,
            "fine_laplacian_maximum_error": fine_poiseuille_error,
            "coarse_weighted_l1_laplacian_error": poiseuille_weighted_l1_error,
            "fine_weighted_l1_laplacian_error": (fine_poiseuille_weighted_l1_error),
            "observed_weighted_l1_spatial_order": poiseuille_order,
            "fine_error_not_greater": (
                fine_poiseuille_weighted_l1_error <= poiseuille_weighted_l1_error
            ),
            "bulk_velocity": float(
                poiseuille_stats.bulk_velocity[poiseuille_stats.streamwise_axis]
            ),
            "statistics_successful": bool(poiseuille_stats.successful),
        },
        "full-hybrid-iterative-comparison": {
            "full_geometry": "uniform-periodic",
            "hybrid_geometry": "stretched-wall-normal",
            "hybrid": _mac_route(hybrid_plan, hybrid),
            "stretched_iterative": _mac_route(
                stretched_iterative_plan, stretched_iterative
            ),
            "hybrid_iterative_velocity_maximum_difference": stretched_difference,
            "full_unavailable_on_stretched_reason": "stretched line is not full-transform eligible",
            "hybrid_unavailable_on_periodic_reason": "hybrid requires a nonperiodic line axis",
        },
    }
    gates = (
        numeric_gate(
            "periodic-route-agreement",
            "periodic-taylor-green.route_velocity_maximum_difference",
            periodic_difference,
            route_tolerance,
        ),
        numeric_gate(
            "periodic-divergence",
            "periodic-taylor-green.full.divergence_after_norm",
            _maximum_abs((full.divergence_after,)),
            invariant_tolerance,
        ),
        numeric_gate(
            "stretched-couette",
            "stretched-couette.laplacian_maximum_error",
            couette_error,
            route_tolerance,
        ),
        boolean_gate(
            "stretched-poiseuille-refinement",
            "stretched-poiseuille.fine_error_not_greater",
            fine_poiseuille_weighted_l1_error <= poiseuille_weighted_l1_error,
        ),
        minimum_numeric_gate(
            "stretched-poiseuille-order",
            "stretched-poiseuille.observed_weighted_l1_spatial_order",
            poiseuille_order,
            minimum_order,
        ),
        numeric_gate(
            "stretched-route-agreement",
            "full-hybrid-iterative-comparison.hybrid_iterative_velocity_maximum_difference",
            stretched_difference,
            route_tolerance,
        ),
        boolean_gate(
            "route-convergence",
            "full-hybrid-iterative-comparison.all_available_routes_converged",
            bool(
                full.converged
                & periodic_iterative.converged
                & hybrid.converged
                & stretched_iterative.converged
            ),
        ),
    )
    support = phx.qualification.SupportTuple(
        CAPABILITY,
        {
            "route": "mac",
            "geometry": "uniform-periodic+stretched-wall-normal",
            "pressure_routes": "full+hybrid+iterative",
            "momentum": "symmetry-preserving",
            "statistics": "mac-plane-wall",
        },
    )
    return make_qualification_artifact(
        route="mac",
        support_tuple=support,
        inputs={
            "periodic_count": pn,
            "stretched_count": sn,
            "periodic_taylor_green": "u=sin(x)cos(y);v=-cos(x)sin(y)",
            "stretched_couette": "u=y",
            "stretched_poiseuille": "u=y*(1-y)",
            "periodic_grid_prepared_id": periodic_grid.prepared_id,
            "periodic_operator_prepared_id": periodic_operators.prepared_id,
            "stretched_grid_prepared_id": stretched_grid.prepared_id,
            "stretched_operator_prepared_id": operators.prepared_id,
            "couette_momentum_prepared_id": couette_momentum.prepared_id,
            "poiseuille_momentum_prepared_id": poiseuille_momentum.prepared_id,
            "fine_stretched_grid_prepared_id": fine_grid_id,
            "fine_stretched_operator_prepared_id": fine_operators_id,
            "fine_poiseuille_momentum_prepared_id": fine_momentum_id,
            "full_plan_id": full_plan.plan_id,
            "periodic_iterative_plan_id": periodic_iterative_plan.plan_id,
            "hybrid_plan_id": hybrid_plan.plan_id,
            "stretched_iterative_plan_id": stretched_iterative_plan.plan_id,
            "couette_statistics_plan_id": couette_stats_plan.plan_id,
            "poiseuille_statistics_plan_id": poiseuille_stats_plan.plan_id,
        },
        reference={
            "source": "native-analytic-and-route-comparison",
            "periodic_taylor_green": "divergence-free staggered face samples",
            "stretched_couette_laplacian": "zero",
            "stretched_poiseuille_laplacian": "-2 streamwise",
        },
        configuration={
            "periodic_count": pn,
            "stretched_count": sn,
            "route_tolerance": float(route_tolerance),
            "invariant_tolerance": float(invariant_tolerance),
            "minimum_spatial_order": minimum_order,
            "pressure_solver_tolerance": 1.0e-10,
            "timing_gate": False,
        },
        metrics=metrics,
        gates=gates,
        external_reference=external_reference,
    )


def _json_object(text: str | None, name: str) -> Any:
    if text is None:
        return None
    value = json.loads(text)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{name} must be a non-empty JSON object.")
    return value


def _external(arguments: Any) -> Any:
    return external_reference_input(
        path=arguments.external_reference_path,
        checksum=arguments.external_reference_checksum,
        nondimensionalization=_json_object(
            arguments.external_nondimensionalization, "nondimensionalization"
        ),
        uncertainty=_json_object(arguments.external_uncertainty, "uncertainty"),
    )


def _add_external(parser: Any) -> None:
    parser.add_argument("--external-reference-path")
    parser.add_argument("--external-reference-checksum")
    parser.add_argument("--external-nondimensionalization")
    parser.add_argument("--external-uncertainty")


def _write(payload: Any, output: Any) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, output)
    print(text, end="")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create unreleased route-specific incompressible-flow qualification evidence."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    periodic = commands.add_parser("periodic-spectral")
    periodic.add_argument("--output", type=Path, required=True)
    periodic.add_argument("--mode-count", type=int, default=12)
    periodic.add_argument("--viscosity", type=float, default=0.01)
    periodic.add_argument("--step-size", type=float, default=0.025)
    periodic.add_argument("--steps", type=int, default=8)
    periodic.add_argument("--solution-tolerance", type=float, default=1.0e-8)
    periodic.add_argument("--invariant-tolerance", type=float, default=1.0e-9)
    periodic.add_argument("--restart-tolerance", type=float, default=1.0e-10)
    _add_external(periodic)
    svv = commands.add_parser("periodic-spectral-vanishing-viscosity")
    svv.add_argument("--output", type=Path, required=True)
    svv.add_argument("--strength", type=float, required=True)
    svv.add_argument("--order", type=float, required=True)
    svv.add_argument("--cutoff-scale", type=float, required=True)
    svv.add_argument("--cutoff-exponent", type=float, required=True)
    svv.add_argument("--kernel", choices=get_args(SpectralViscosityKernel), required=True)
    svv.add_argument("--highpass-exponent", type=int)
    svv.add_argument("--mode-count", type=int, default=12)
    svv.add_argument("--low-mode-counts", type=int, nargs="+", default=(12, 24, 48))
    svv.add_argument("--fixed-low-mode", type=int, default=1)
    svv.add_argument("--step-size", type=float, default=0.025)
    svv.add_argument("--steps", type=int, default=4)
    svv.add_argument("--solution-tolerance", type=float, default=1.0e-8)
    svv.add_argument("--invariant-tolerance", type=float, default=1.0e-9)
    _add_external(svv)
    statistical = commands.add_parser("periodic-statistical-solution")
    statistical.add_argument("--output", type=Path, required=True)
    statistical.add_argument("--strength", type=float, required=True)
    statistical.add_argument("--order", type=float, required=True)
    statistical.add_argument("--cutoff-scale", type=float, required=True)
    statistical.add_argument("--cutoff-exponent", type=float, required=True)
    statistical.add_argument(
        "--kernel", choices=get_args(SpectralViscosityKernel), required=True
    )
    statistical.add_argument("--highpass-exponent", type=int)
    statistical.add_argument("--power-law-exponent", type=float, required=True)
    statistical.add_argument("--velocity-scale", type=float, required=True)
    statistical.add_argument("--maximum-courant-number", type=float, required=True)
    statistical.add_argument("--mode-counts", type=int, nargs="+", default=(16, 32, 64))
    statistical.add_argument("--members", type=int, default=6)
    statistical.add_argument("--seed", type=int, default=0)
    statistical.add_argument("--step-size", type=float, default=1e-4)
    statistical.add_argument("--steps", type=int, default=2)
    statistical.add_argument("--num-projections", type=int, default=32)
    statistical.add_argument("--refinement-z", type=float, default=2.0)
    _add_external(statistical)
    channel = commands.add_parser("spectral-channel")
    channel.add_argument("--output", type=Path, required=True)
    channel.add_argument("--shape", type=int, nargs=3, default=(4, 10, 4))
    channel.add_argument("--viscosity", type=float, default=0.1)
    channel.add_argument("--step-size", type=float, default=0.01)
    channel.add_argument("--steps", type=int, default=4)
    channel.add_argument("--solution-tolerance", type=float, default=1.0e-8)
    channel.add_argument("--invariant-tolerance", type=float, default=1.0e-9)
    channel.add_argument("--restart-tolerance", type=float, default=1.0e-10)
    channel.add_argument("--minimum-temporal-order", type=float, default=1.8)
    _add_external(channel)
    mac = commands.add_parser("mac")
    mac.add_argument("--output", type=Path, required=True)
    mac.add_argument("--periodic-count", type=int, default=8)
    mac.add_argument("--stretched-count", type=int, default=6)
    mac.add_argument("--route-tolerance", type=float, default=1.0e-7)
    mac.add_argument("--invariant-tolerance", type=float, default=1.0e-8)
    mac.add_argument("--minimum-spatial-order", type=float, default=0.8)
    _add_external(mac)
    assemble = commands.add_parser("assemble")
    assemble.add_argument("artifacts", type=Path, nargs="+")
    assemble.add_argument("--output", type=Path, required=True)
    assemble.add_argument("--name", default="incompressible-flow.candidate")
    assemble.add_argument("--provider", default="phydrax")
    arguments = parser.parse_args()
    if arguments.command == "periodic-spectral":
        payload = periodic_spectral_qualification(
            mode_count=arguments.mode_count,
            viscosity=arguments.viscosity,
            step_size=arguments.step_size,
            steps=arguments.steps,
            solution_tolerance=arguments.solution_tolerance,
            invariant_tolerance=arguments.invariant_tolerance,
            restart_tolerance=arguments.restart_tolerance,
            external_reference=_external(arguments),
        )
    elif arguments.command == "periodic-spectral-vanishing-viscosity":
        payload = periodic_spectral_vanishing_viscosity_qualification(
            strength=arguments.strength,
            order=arguments.order,
            cutoff_scale=arguments.cutoff_scale,
            cutoff_exponent=arguments.cutoff_exponent,
            kernel=arguments.kernel,
            highpass_exponent=arguments.highpass_exponent,
            mode_count=arguments.mode_count,
            low_mode_counts=tuple(arguments.low_mode_counts),
            fixed_low_mode=arguments.fixed_low_mode,
            step_size=arguments.step_size,
            steps=arguments.steps,
            solution_tolerance=arguments.solution_tolerance,
            invariant_tolerance=arguments.invariant_tolerance,
            external_reference=_external(arguments),
        )
    elif arguments.command == "periodic-statistical-solution":
        payload = periodic_statistical_solution_qualification(
            strength=arguments.strength,
            order=arguments.order,
            cutoff_scale=arguments.cutoff_scale,
            cutoff_exponent=arguments.cutoff_exponent,
            kernel=arguments.kernel,
            highpass_exponent=arguments.highpass_exponent,
            power_law_exponent=arguments.power_law_exponent,
            velocity_scale=arguments.velocity_scale,
            maximum_courant_number=arguments.maximum_courant_number,
            mode_counts=tuple(arguments.mode_counts),
            members=arguments.members,
            seed=arguments.seed,
            step_size=arguments.step_size,
            steps=arguments.steps,
            num_projections=arguments.num_projections,
            refinement_z=arguments.refinement_z,
            external_reference=_external(arguments),
        )
    elif arguments.command == "spectral-channel":
        payload = spectral_channel_qualification(
            shape=tuple(arguments.shape),
            viscosity=arguments.viscosity,
            step_size=arguments.step_size,
            steps=arguments.steps,
            solution_tolerance=arguments.solution_tolerance,
            invariant_tolerance=arguments.invariant_tolerance,
            restart_tolerance=arguments.restart_tolerance,
            minimum_temporal_order=arguments.minimum_temporal_order,
            external_reference=_external(arguments),
        )
    elif arguments.command == "mac":
        payload = mac_qualification(
            periodic_count=arguments.periodic_count,
            stretched_count=arguments.stretched_count,
            minimum_spatial_order=arguments.minimum_spatial_order,
            route_tolerance=arguments.route_tolerance,
            invariant_tolerance=arguments.invariant_tolerance,
            external_reference=_external(arguments),
        )
    else:
        payload = assemble_candidate_profile(
            tuple(json.loads(path.read_text()) for path in arguments.artifacts),
            name=arguments.name,
            provider=arguments.provider,
        )
    _write(payload, arguments.output)


if __name__ == "__main__":
    main()
