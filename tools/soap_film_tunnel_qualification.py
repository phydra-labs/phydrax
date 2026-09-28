#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification campaigns for the planar soap-film tunnel.

Campaigns (references in ``docs/guides_soap_film_tunnel.md``):

- terminal uniform flow: an empty channel with free-slip wires fed at
  ``u_T = rho h g / C`` must stay uniform; records the maximum relative
  velocity and thickness defects and the ledger residuals;
- cylinder wake: shedding Strouhal number with cycle-to-cycle standard error,
  mean drag and lift amplitude coefficients under rim/mesh refinement at fixed
  declared Reynolds number. The comparison reports both the Williamson (1988)
  parallel-shedding fit ``St = -3.3265 / Re + 0.1816 + 1.6e-4 Re``
  (47 < Re < 180) and the Allen--Vincenti wall-interference velocity correction,
  together with cell Reynolds number, ledgers, film Mach number and statuses;
- runtime: host preparation, lowering, executable compilation, first execution,
  repeated warm execution, and compiler/logical memory are separate evidence.

Requires the optional phydrax-meshcore library for the channel
triangulation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.applications.soap_film_tunnel import (
    PreparedSoapFilmTunnel,
    SoapFilmInflow,
    SoapFilmTunnelEvidence,
    SoapFilmTunnelGeometry,
    SoapFilmTunnelPlan,
    SoapFilmTunnelResult,
    SoapFilmTunnelState,
    SoapFilmWireKind,
)
from phydrax.applications.soap_film_tunnel._tunnel import _run_soap_film_tunnel


_DIAMETER = 1.0e-3
_DENSITY, _THICKNESS, _CONCENTRATION = 1000.0, 1.5e-6, 1.6e-6
_SPEED, _GRAVITY, _BULK_VISCOSITY = 1.0, 9.81, 1.0e-3


def _plan(
    geometry: SoapFilmTunnelGeometry, reynolds: float, wire_kind: SoapFilmWireKind
) -> SoapFilmTunnelPlan:
    viscosity = _SPEED * geometry.reference_length_m / reynolds
    surface_viscosity = 0.5 * (
        _DENSITY * _THICKNESS * viscosity - _BULK_VISCOSITY * _THICKNESS
    )
    return SoapFilmTunnelPlan(
        geometry,
        phx.interfacial_transport.LangmuirSurfactantLaw(0.072, 298.15, 4.0e-6),
        SoapFilmInflow(
            velocity_m_s=_SPEED,
            thickness_m=_THICKNESS,
            surface_concentration_mol_m2=_CONCENTRATION,
        ),
        density_kg_m3=_DENSITY,
        air_drag_coefficient_kg_m2_s=_DENSITY * _THICKNESS * _GRAVITY / _SPEED,
        gravity_m_s2=_GRAVITY,
        viscosity_pa_s=_BULK_VISCOSITY,
        surface_shear_viscosity_n_s_m=surface_viscosity,
        surface_dilatational_viscosity_n_s_m=surface_viscosity,
        surface_diffusivity_m2_s=1.0e-9,
        wire_kind=wire_kind,
    )


def _timed_prepare(plan: SoapFilmTunnelPlan) -> tuple[PreparedSoapFilmTunnel, float]:
    start = perf_counter()
    prepared = plan.prepare()
    return prepared, perf_counter() - start


def _execute(
    prepared: PreparedSoapFilmTunnel,
    state: SoapFilmTunnelState,
    step_size: float,
    steps: int,
    /,
) -> tuple[SoapFilmTunnelResult, dict[str, Any]]:
    """Compile and execute one physical trajectory plus a short warm benchmark."""
    dynamic_step_size = jnp.asarray(step_size, dtype=jnp.float64)
    compiled_run: Any = _run_soap_film_tunnel
    arguments = (prepared, state, dynamic_step_size, steps)
    compiled, compilation = measure_lower_and_compile(
        lambda: compiled_run.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, physical_execution = measure_synchronized(lambda: compiled(*arguments))
    warm_steps = min(steps, 2)
    warm_arguments = (prepared, state, dynamic_step_size, warm_steps)
    warm_compiled, warm_compilation = measure_lower_and_compile(
        lambda: compiled_run.lower(*warm_arguments),
        lambda lowered: lowered.compile(),
    )
    warm_first_result, warm_first_execution = measure_synchronized(
        lambda: warm_compiled(*warm_arguments)
    )
    warm_result, warm = measure_repeated(
        lambda: warm_compiled(*warm_arguments), warmup=0, repeats=1
    )
    expected_status = result.evidence.status[:warm_steps]
    if not bool(
        jnp.array_equal(expected_status, warm_first_result.evidence.status)
        & jnp.array_equal(expected_status, warm_result.evidence.status)
    ):
        raise RuntimeError("Warm tunnel chunk changed the deterministic status record.")
    cost = compiled.compiled.cost_analysis()
    memory = compiled.compiled.memory_analysis()
    unavailable = (
        "The selected backend did not report compiler analysis."
        if not cost and memory is None
        else None
    )
    compiler = compiler_evidence(
        cost,
        memory,
        source="jax-compiled-executable",
        unavailable_reason=unavailable,
    )
    return result, {
        "lowering_s": compilation.lowering_seconds,
        "compilation_s": compilation.compilation_seconds,
        "physical_execution_s": physical_execution,
        "representative_warm_chunk": {
            "steps": warm_steps,
            "lowering_s": warm_compilation.lowering_seconds,
            "compilation_s": warm_compilation.compilation_seconds,
            "first_execution_s": warm_first_execution,
            "warm_execution": warm.to_seconds_dict(),
        },
        "compiler": asdict(compiler),
        "compiler_estimated_device_memory_bytes": (
            compiler.estimated_device_memory_bytes
        ),
        "logical_retained_input_bytes": logical_array_bytes(arguments),
        "logical_result_bytes": logical_array_bytes(result),
        "recurrence": "one jax.lax.scan with unroll=1",
    }


def _ledgers(evidence: SoapFilmTunnelEvidence) -> dict[str, Any]:
    return {
        "maximum_volume_residual_m3": float(
            jnp.max(jnp.abs(evidence.volume_residual_m3))
        ),
        "maximum_surfactant_residual_mol": float(
            jnp.max(jnp.abs(evidence.surfactant_residual_mol))
        ),
        "maximum_momentum_residual_n_s": float(
            jnp.max(jnp.abs(evidence.momentum_residual_n_s))
        ),
        "completed_steps": evidence.status.size,
        "accepted_fraction": float(jnp.mean(evidence.accepted)),
        "all_accepted": bool(jnp.all(evidence.accepted)),
        "all_converged": bool(jnp.all(evidence.converged)),
        "all_nonlinear_successful": bool(jnp.all(evidence.nonlinear_status == 0)),
        "maximum_film_mach_number": float(jnp.max(evidence.film_mach_number)),
        "maximum_courant_number": float(jnp.max(evidence.courant_number)),
        "minimum_thickness_m": float(jnp.min(evidence.minimum_thickness_m)),
        "mean_inflow_volume_rate_m3_s": float(jnp.mean(evidence.inflow_volume_rate_m3_s)),
        "mean_outflow_volume_rate_m3_s": float(
            jnp.mean(evidence.outflow_volume_rate_m3_s)
        ),
    }


def terminal_uniform_flow(steps: int) -> dict[str, Any]:
    geometry = SoapFilmTunnelGeometry(
        8 * _DIAMETER, 4 * _DIAMETER, mesh_size_m=0.5 * _DIAMETER
    )
    prepared, prepare_time = _timed_prepare(_plan(geometry, 150.0, "free-slip"))
    step_size = 0.1 * 0.5 * _DIAMETER / _SPEED
    result, performance = _execute(prepared, prepared.initial_state(), step_size, steps)
    velocity = np.asarray(prepared.velocity(result.state))
    thickness = np.asarray(prepared.thickness(result.state))
    return {
        "vertices": prepared.flow.plan.surface.topology.num_vertices,
        "steps": steps,
        "prepare_s": prepare_time,
        "performance": performance,
        "terminal_velocity_m_s": float(prepared.scales.terminal_velocity_m_s),
        "maximum_relative_velocity_defect": float(
            np.max(np.abs(velocity - np.asarray((_SPEED, 0.0, 0.0)))) / _SPEED
        ),
        "maximum_relative_thickness_defect": float(
            np.max(np.abs(thickness / _THICKNESS - 1.0))
        ),
        **_ledgers(result.evidence),
    }


def cylinder_wake(
    resolutions: tuple[tuple[int, float], ...],
    *,
    reynolds: float,
    periods: float,
) -> list[dict[str, Any]]:
    rows = []
    for rim_segments, mesh_size in resolutions:
        geometry = SoapFilmTunnelGeometry(
            12 * _DIAMETER,
            6 * _DIAMETER,
            mesh_size_m=mesh_size * _DIAMETER,
            obstacle_diameter_m=_DIAMETER,
            obstacle_center_m=(3 * _DIAMETER, 3 * _DIAMETER),
            rim_segments=rim_segments,
        )
        prepared, prepare_time = _timed_prepare(_plan(geometry, reynolds, "no-slip"))
        points = np.asarray(prepared.flow.plan.surface.coordinates)
        velocity = np.zeros_like(points)
        velocity[:, 0] = _SPEED
        velocity[:, 1] = (
            0.1
            * _SPEED
            * np.exp(
                -(((points[:, 0] - 4 * _DIAMETER) / _DIAMETER) ** 2)
                - ((points[:, 1] - 3 * _DIAMETER) / _DIAMETER) ** 2
            )
        )
        step_size = 0.1 * (np.pi * _DIAMETER / rim_segments) / _SPEED
        # Twice the requested record: the first half is the transient.
        steps = int(np.ceil(2.0 * periods * _DIAMETER / (0.2 * _SPEED) / step_size))
        result, performance = _execute(
            prepared, prepared.initial_state(velocity), step_size, steps
        )
        estimate = prepared.strouhal(result, transient_s=0.5 * float(result.state.time_s))
        reference = prepared.cylinder_wake_reference(estimate.mean_drag_coefficient)
        final_velocity = np.asarray(prepared.velocity(result.state))
        wake = (points[:, 0] > 3.5 * _DIAMETER) & (points[:, 0] < 8.0 * _DIAMETER)
        wake_cross_stream_rms = float(np.sqrt(np.mean(final_velocity[wake, 1] ** 2)))
        rows.append(
            {
                "rim_segments": rim_segments,
                "mesh_size_over_diameter": mesh_size,
                "vertices": prepared.flow.plan.surface.topology.num_vertices,
                "triangulation_status": prepared.channel.triangulation.status,
                "minimum_angle_degrees": prepared.channel.triangulation.minimum_angle_degrees,
                "reynolds_number": float(prepared.scales.reynolds_number),
                "cell_reynolds_number": float(prepared.scales.cell_reynolds_number),
                "inflow_film_mach_number": float(prepared.scales.film_mach_number),
                "blockage_ratio": float(prepared.scales.blockage_ratio),
                "steps": steps,
                "step_size_s": step_size,
                "prepare_s": prepare_time,
                "performance": performance,
                "shedding_status": estimate.status.name,
                "strouhal_number": float(estimate.strouhal_number),
                "shedding_periods": int(estimate.periods),
                "strouhal_standard_error": float(estimate.strouhal_standard_error),
                "lift_coefficient_amplitude": float(estimate.lift_coefficient_amplitude),
                "mean_drag_coefficient": float(estimate.mean_drag_coefficient),
                "unconfined_reference_strouhal_number": float(
                    reference.unconfined_strouhal_number
                ),
                "blockage_corrected_reference_strouhal_number": float(
                    reference.blockage_corrected_strouhal_number
                ),
                "blockage_velocity_correction": float(reference.velocity_correction),
                "corrected_reynolds_number": float(reference.corrected_reynolds_number),
                "difference_from_blockage_reference": float(
                    estimate.strouhal_number
                    - reference.blockage_corrected_strouhal_number
                ),
                "wake_cross_stream_rms_m_s": wake_cross_stream_rms,
                **_ledgers(result.evidence),
            }
        )
    return rows


def run(
    *, smoke: bool = False, middle_row: dict[str, Any] | None = None
) -> dict[str, Any]:
    if smoke and middle_row is not None:
        raise ValueError("A reused middle wake row is only valid for the full campaign.")
    resolutions = (
        ((24, 0.5),)
        if smoke
        else ((24, 0.5), (48, 0.3))
        if middle_row is not None
        else ((24, 0.5), (32, 0.4), (48, 0.3))
    )
    terminal = terminal_uniform_flow(4 if smoke else 40)
    wake = cylinder_wake(resolutions, reynolds=150.0, periods=2.0 if smoke else 6.0)
    if middle_row is not None:
        expected_steps = int(
            np.ceil(
                2.0
                * 6.0
                * _DIAMETER
                / (0.2 * _SPEED)
                / (0.1 * np.pi * _DIAMETER / (32 * _SPEED))
            )
        )
        if (
            middle_row.get("rim_segments") != 32
            or middle_row.get("mesh_size_over_diameter") != 0.4
            or middle_row.get("steps") != expected_steps
            or middle_row.get("completed_steps") != expected_steps
            or middle_row.get("all_accepted") is not True
            or middle_row.get("all_converged") is not True
            or middle_row.get("all_nonlinear_successful") is not True
            or middle_row.get("reynolds_number") != 150.0
        ):
            raise ValueError(
                "The reused middle row must be the Re=150, rim-32, mesh-0.4d "
                f"{expected_steps}-step qualification trajectory."
            )
        wake.insert(1, middle_row)
    finest = wake[-1]
    uncertainty = finest["strouhal_standard_error"]
    reference_lower = finest["unconfined_reference_strouhal_number"]
    reference_upper = finest["blockage_corrected_reference_strouhal_number"]
    estimate = finest["strouhal_number"]
    reference_consistent = bool(
        np.isfinite(estimate)
        and np.isfinite(uncertainty)
        and reference_lower - 3.0 * uncertainty
        <= estimate
        <= reference_upper + 3.0 * uncertainty
    )
    terminal_successful = (
        terminal["completed_steps"] == terminal["steps"]
        and terminal["all_accepted"]
        and terminal["all_converged"]
        and terminal["all_nonlinear_successful"]
    )
    wake_successful = all(
        row["completed_steps"] == row["steps"]
        and row["all_accepted"]
        and row["all_converged"]
        and row["all_nonlinear_successful"]
        for row in wake
    )
    gates = {
        "terminal_uniform_flow": (
            terminal_successful
            and terminal["maximum_relative_velocity_defect"] < 1.0e-8
            and terminal["maximum_relative_thickness_defect"] < 1.0e-8
        ),
        "all_steps_accepted": wake_successful,
        "finest_wake_shedding": finest["shedding_status"] == "SHEDDING",
        "finest_record_has_multiple_periods": finest["shedding_periods"] >= 2,
        "finest_strouhal_in_blockage_bracket": reference_consistent,
    }
    return {
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "devices": sorted(device.device_kind for device in jax.devices()),
            "load_average_at_report": list(os.getloadavg()),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "x64": bool(jax.config.read("jax_enable_x64")),
        },
        "reference": (
            "Williamson 1988 parallel-shedding St(Re) fit plus Allen--Vincenti "
            "NACA TR 782 wall-interference velocity correction; the correction "
            "is comparison evidence, not a film-model calibration"
        ),
        "gates": gates,
        "successful": all(gates.values()),
        "terminal_uniform_flow": terminal,
        "cylinder_wake": wake,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--middle-row",
        type=Path,
        help="Reuse a preserved rim-32/mesh-0.4d full-horizon wake row.",
    )
    arguments = parser.parse_args()
    middle_row = (
        None
        if arguments.middle_row is None
        else json.loads(arguments.middle_row.read_text())
    )
    report = run(smoke=arguments.smoke, middle_row=middle_row)
    text = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n")
    print(text)
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
