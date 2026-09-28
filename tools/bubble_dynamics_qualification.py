#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Qualification campaigns for radial bubble dynamics.

Each campaign runs a tolerance, resolution or parameter sequence against an
exact or independently computed reference and records the metrics, the
resulting gate decision and the runtime identity. Gates whose primary-source
reference values could not be verified are reported as failed rather than
approximated.

Usage, from the repository root:
`PYTHONPATH=. python tools/bubble_dynamics_qualification.py [--campaign NAME] [--output PATH]`.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags

import phydrax as phx
import phydrax.bubble_dynamics as bd
from phydrax.equations import TaitBarotropicMaterial
from phydrax.qualification import QualificationRuntimeIdentity


AMBIENT = 101325.0
TEMPERATURE = 293.15
DENSITY = 998.0
SOUND_SPEED = 1481.0
Record = dict[str, Any]


def _digest(payload: Record, /) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _runtime_identity() -> Record:
    identity = QualificationRuntimeIdentity(
        _digest(
            {
                "kind": "bubble-dynamics-qualification-build",
                "phydrax": importlib.metadata.version("phydrax"),
                "path": str(Path(phx.__file__).resolve().parent),
            }
        ),
        _digest(
            {
                "kind": "bubble-dynamics-qualification-environment",
                "python": platform.python_version(),
                "platform": platform.platform(),
                "jax": jax.__version__,
                "numpy": np.__version__,
            }
        ),
        jax.default_backend(),
        f"processes-{jax.process_count()}-devices-{jax.device_count()}",
        str(jnp.asarray(0.0).dtype),
    )
    return dict(identity.to_record())


def _model(
    equation: bd.RadialBubbleEquation,
    *,
    ambient: float = AMBIENT,
    gas: bd.AbstractBubbleGasLaw | None = None,
    interface: bd.AbstractBubbleInterfaceLaw | None = None,
    viscosity: float = 0.0,
    sound_speed: float = SOUND_SPEED,
) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        equation,
        bd.PolytropicBubbleGasLaw(1.4) if gas is None else gas,
        bd.NewtonianBubbleLiquidLaw(viscosity),
        bd.CleanBubbleInterfaceLaw(0.0) if interface is None else interface,
        bd.BubbleEnvironment(ambient, TEMPERATURE),
        liquid_density=DENSITY,
        liquid_sound_speed=sound_speed,
    )


def rayleigh_collapse() -> Record:
    radius, pressure, ratio = 1.0e-3, 1.0e5, 1.0e-2
    reference = float(bd.rayleigh_collapse_time(radius, pressure, DENSITY, final_radius_ratio=ratio))
    errors = []
    steps = []
    successful = []
    for tolerance in (1.0e-6, 1.0e-8, 1.0e-10):
        plan = bd.SingleBubblePlan(
            _model("rayleigh_plesset", ambient=0.0),
            bd.ConstantPressureDrive(pressure),
            np.linspace(0.0, 1.2 * reference, 5),
            events=bd.BubbleEventPolicy(minimum_radius_ratio=ratio, mach_limit=None),
            relative_tolerance=tolerance,
            absolute_tolerance=1.0e-2 * tolerance,
        )
        result = bd.solve_single_bubble(plan.prepare(radius))
        errors.append(abs(float(result.terminal_time) / reference - 1.0))
        steps.append(int(result.evidence.accepted_steps))
        successful.append(bool(result.successful))
    constant = float(bd.rayleigh_collapse_time(1.0, 1.0, 1.0))
    native_successful = all(successful)
    return {
        "reference": "Rayleigh (1917) empty-cavity collapse, exact incomplete-beta integral",
        "tolerances": [1.0e-6, 1.0e-8, 1.0e-10],
        "relative_event_time_error": errors,
        "accepted_steps": steps,
        "solve_successful": successful,
        "collapse_constant": constant,
        "successful": native_successful,
        "passed": native_successful
        and errors[-1] < 1.0e-8
        and errors[-1] <= errors[0]
        and abs(constant - 0.914681) < 5.0e-7,
    }


def rayleigh_plesset_energy() -> Record:
    radius, tension, kappa = 1.0e-4, 0.072, 1.4
    drifts = []
    successful = []
    for tolerance in (1.0e-7, 1.0e-9, 1.0e-11):
        plan = bd.SingleBubblePlan(
            _model(
                "rayleigh_plesset",
                gas=bd.PolytropicBubbleGasLaw(kappa),
                interface=bd.CleanBubbleInterfaceLaw(tension),
            ),
            bd.ConstantPressureDrive(0.0),
            np.linspace(0.0, 1.2e-4, 241),
            relative_tolerance=tolerance,
            absolute_tolerance=1.0e-2 * tolerance,
        )
        prepared = plan.prepare(radius, initial_radius=1.5 * radius)
        result = bd.solve_single_bubble(prepared)
        successful.append(bool(result.successful))
        r = np.asarray(result.trajectory.radius)
        u = np.asarray(result.trajectory.wall_velocity)
        volume = 4.0 * np.pi * r**3 / 3.0
        reference_volume = 4.0 * np.pi * radius**3 / 3.0
        kinetic = 2.0 * np.pi * DENSITY * r**3 * u**2
        energy = (
            kinetic
            + AMBIENT * volume
            + 4.0 * np.pi * tension * r**2
            + float(prepared.equilibrium.gas_pressure)
            * reference_volume**kappa
            * volume ** (1.0 - kappa)
            / (kappa - 1.0)
        )
        drifts.append(float(np.max(np.abs(energy - energy[0])) / np.max(kinetic)))
    native_successful = all(successful)
    return {
        "reference": "Hamiltonian of undamped Rayleigh-Plesset with polytropic gas and surface tension",
        "tolerances": [1.0e-7, 1.0e-9, 1.0e-11],
        "relative_energy_drift": drifts,
        "solve_successful": successful,
        "successful": native_successful,
        "passed": native_successful
        and drifts[-1] < 1.0e-7
        and drifts[-1] < drifts[0],
    }


def _driven_result(
    model: bd.RadialBubbleModel, amplitude: float
) -> bd.SingleBubbleResult:
    plan = bd.SingleBubblePlan(
        model,
        bd.HarmonicPressureDrive(amplitude, 2.0 * np.pi * 2.0e5),
        np.linspace(0.0, 2.0e-5, 201),
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
    )
    return bd.solve_single_bubble(plan.prepare(1.0e-5))


def compressible_limits() -> Record:
    reference_result = _driven_result(
        _model("rayleigh_plesset", viscosity=1.0e-3), 3.0e4
    )
    reference = np.asarray(reference_result.trajectory.radius)
    successful = [bool(reference_result.successful)]
    factors = (1.0, 10.0, 100.0, 1000.0)
    keller = []
    for factor in factors:
        result = _driven_result(
            _model(
                "keller_miksis",
                viscosity=1.0e-3,
                sound_speed=factor * SOUND_SPEED,
            ),
            3.0e4,
        )
        successful.append(bool(result.successful))
        keller.append(
            float(np.max(np.abs(np.asarray(result.trajectory.radius) - reference)))
        )
    orders = [float(np.log10(keller[index] / keller[index + 1])) for index in range(len(keller) - 1)]
    material = TaitBarotropicMaterial(DENSITY, SOUND_SPEED, exponent=7.15, background_pressure=AMBIENT)
    gilmore_differences = []
    amplitudes = (3.0e3, 1.0e4, 3.0e4, 1.0e5)
    for amplitude in amplitudes:
        gilmore = bd.RadialBubbleModel(
            "gilmore",
            bd.PolytropicBubbleGasLaw(1.4),
            bd.NewtonianBubbleLiquidLaw(1.0e-3),
            bd.CleanBubbleInterfaceLaw(0.0),
            bd.BubbleEnvironment(AMBIENT, TEMPERATURE),
            liquid_material=material,
        )
        keller_result = _driven_result(
            _model("keller_miksis", viscosity=1.0e-3), amplitude
        )
        gilmore_result = _driven_result(gilmore, amplitude)
        successful.extend(
            (bool(keller_result.successful), bool(gilmore_result.successful))
        )
        keller_radius = np.asarray(keller_result.trajectory.radius)
        difference = np.max(
            np.abs(np.asarray(gilmore_result.trajectory.radius) - keller_radius)
        )
        gilmore_differences.append(float(difference / np.ptp(keller_radius)))
    native_successful = all(successful)
    return {
        "reference": "Keller-Miksis to Rayleigh-Plesset as c grows; Gilmore(Tait) to Keller-Miksis at weak drive",
        "sound_speed_factors": list(factors),
        "keller_miksis_max_difference_m": keller,
        "observed_order_in_inverse_sound_speed": orders,
        "gilmore_amplitudes_pa": list(amplitudes),
        "gilmore_relative_difference": gilmore_differences,
        "solve_successful": successful,
        "successful": native_successful,
        "passed": native_successful
        and min(orders) > 0.9
        and gilmore_differences[0] < 1.0e-4,
    }


def minnaert_and_coated_resonance() -> Record:
    errors = []
    successful = []
    for tension in (0.0, 0.072):
        model = _model("rayleigh_plesset", interface=bd.CleanBubbleInterfaceLaw(tension))
        response = bd.linear_bubble_response(model, 1.0e-3, 2.0 * np.pi * 3.0e3)
        exact = float(bd.minnaert_angular_frequency(1.0e-3, AMBIENT, DENSITY, 1.4, surface_tension=tension))
        errors.append(abs(float(response.resonance_frequency) / exact - 1.0))
        successful.append(bool(response.successful))
    radius, elasticity, initial = 2.0e-6, 0.55, 0.02
    shell = bd.MarmottantShell(elasticity, initial, 0.072, 1.5e-8)
    coated = bd.linear_bubble_response(
        _model("keller_miksis", gas=bd.PolytropicBubbleGasLaw(1.07), interface=shell, viscosity=1.0e-3),
        radius,
        2.0 * np.pi * 3.0e6,
    )
    successful.append(bool(coated.successful))
    buckling = radius / np.sqrt(1.0 + initial / elasticity)
    stiffness = (
        3.0 * 1.07 * (AMBIENT + 2.0 * initial / radius)
        - 2.0 * initial / radius
        + 4.0 * elasticity * radius / buckling**2
    )
    exact_coated = np.sqrt(stiffness / (DENSITY * radius**2))
    coated_error = abs(float(coated.natural_frequency[0]) / exact_coated - 1.0)
    native_successful = all(successful)
    return {
        "reference": "Minnaert (1933) with capillarity; Marmottant et al. (2005) elastic-branch resonance",
        "minnaert_relative_error": errors,
        "minnaert_1mm_hz": float(bd.minnaert_angular_frequency(1.0e-3, AMBIENT, DENSITY, 1.4)) / (2.0 * np.pi),
        "coated_relative_error": coated_error,
        "response_successful": successful,
        "successful": native_successful,
        "passed": native_successful
        and max(errors) < 1.0e-10
        and coated_error < 1.0e-10,
    }


def _marmottant_transitions(tolerance: float) -> tuple[np.ndarray, bool]:
    model = _model(
        "keller_miksis",
        gas=bd.PolytropicBubbleGasLaw(1.07),
        interface=bd.MarmottantShell(0.55, 0.02, 0.072, 1.5e-8),
        viscosity=1.0e-3,
    )
    plan = bd.SingleBubblePlan(
        model,
        bd.HarmonicPressureDrive(5.0e4, 2.0 * np.pi * 2.9e6),
        np.linspace(0.0, 4.0 / 2.9e6, 41),
        relative_tolerance=tolerance,
        absolute_tolerance=1.0e-2 * tolerance,
    )
    result = bd.solve_single_bubble(plan.prepare(2.0e-6))
    tape = result.evidence.regime_tape
    return np.asarray(tape.times)[: int(tape.count)], bool(
        result.successful & ~tape.capacity_exceeded
    )


def marmottant_events() -> Record:
    tolerances = (1.0e-6, 1.0e-7, 1.0e-8, 1.0e-9, 1.0e-11)
    outcomes = [_marmottant_transitions(tolerance) for tolerance in tolerances]
    sequences = [outcome[0] for outcome in outcomes]
    successful = [outcome[1] for outcome in outcomes]
    count = min(sequence.shape[0] for sequence in sequences)
    reference = sequences[-1][:count]
    errors = [float(np.max(np.abs(sequence[:count] - reference)) * 2.9e6) for sequence in sequences[:-1]]
    native_successful = all(successful)
    return {
        "reference": "event times at rtol 1e-11",
        "tolerances": list(tolerances[:-1]),
        "transition_count": int(count),
        "max_event_time_error_in_periods": errors,
        "solve_successful": successful,
        "successful": native_successful,
        "passed": native_successful
        and count >= 6
        and errors[-1] < 1.0e-6
        and errors[-1] < errors[0],
    }


MARMOTTANT_2005 = {
    "reference": (
        "Marmottant, van der Meer, Emmer, Versluis, de Jong, Hilgenfeldt & Lohse, "
        "J. Acoust. Soc. Am. 118(6), 3499-3505 (2005), Eqs. (3)-(4) and Fig. 5(b)"
    ),
    "doi": "10.1121/1.2109427",
    "url": (
        "http://liphy-annuaire.univ-grenoble-alpes.fr/pages_personnelles/"
        "philippe_marmottant/Publications_files/Marmottant2005_JASACoatedBubble.pdf"
    ),
    "pdf_sha256": "efbe0e1bcf85b02b82fcecb1cd0e4c591d04acae0cc5adffc25d3c75dcf60a45",
    "figure_raster": "page 5 image object 17, 1500x565 px at 300 ppi (poppler pdfimages)",
    "figure_raster_sha256": "306798f96eab6bbff96a7637cc218c834b40e1fd18e3a28e353bcf3004bc6245",
}
# Fig. 5(b) caption: R_buckling = R0 = 0.975 um, chi = 1 N/m, kappa_s = 15e-9 (printed
# "N"; surface dilatational viscosity, kg/s), sigma_break-up > 1 N/m (resistant shell),
# rho_l = 1e3 kg/m^3, mu = 1e-3 Pa s, c = 1480 m/s, kappa = 1.095, 2.9 MHz, 130 kPa.
FIG5B_RADIUS = 0.975e-6
FIG5B_FREQUENCY = 2.9e6
FIG5B_AMPLITUDE = 130.0e3
FIG5B_ELASTICITY = 1.0
FIG5B_SHELL_VISCOSITY = 15.0e-9
FIG5B_BREAKUP_TENSION = 1.0
FIG5B_DENSITY = 1.0e3
FIG5B_VISCOSITY = 1.0e-3
FIG5B_SOUND_SPEED = 1480.0
FIG5B_POLYTROPIC_INDEX = 1.095
# Sec. I: pure water, 73 mN/m. Enters only after break-up, which this case never reaches.
FIG5B_WATER_TENSION = 0.073
# Not stated in the paper: the ambient pressure (standard atmosphere assumed) and the
# simulated burst. Fig. 5(b) shows expansion first and five compressions, so the burst
# here is an unwindowed rarefaction-first sine of five cycles starting at the figure's
# t = 0. The figure's reduced first expansion and its fast recovery after 1.75 us
# imply tapered burst edges that the paper does not specify. The gate therefore
# covers the burst interior: the five minima, maxima 2-5, the first nine rest-level
# crossings and the four complete buckled intervals. The edges are reported only.
FIG5B_CYCLES = 5
FIG5B_INTERIOR_CROSSINGS = 9
# Digitized Fig. 5(b): line-centroid extrema (parabolic fit over +-6 px) and the
# centroids of the line where it crosses the digitized rest level R0 = 0.97492 um.
# Calibration from the axis ticks: 0.47995 nm/px and 3.8286 ns/px, residuals
# 0.13 nm and 1.2 ns. Line width 2 px. Standard uncertainty 0.5 nm and 4 ns.
FIG5B_DIGITIZED_MAXIMA_UM = (
    (0.1334, 0.99547),
    (0.4742, 1.00886),
    (0.8197, 1.00854),
    (1.1644, 1.00863),
    (1.5091, 1.00803),
    (1.8447, 0.97831),
)
FIG5B_DIGITIZED_MINIMA_UM = (
    (0.3284, 0.84579),
    (0.6733, 0.84537),
    (1.0178, 0.84529),
    (1.3625, 0.84525),
    (1.7075, 0.84525),
)
FIG5B_DIGITIZED_CROSSINGS_US = (
    ("buckle", 0.1895),
    ("unbuckle", 0.4406),
    ("buckle", 0.5298),
    ("unbuckle", 0.7841),
    ("buckle", 0.8747),
    ("unbuckle", 1.1285),
    ("buckle", 1.2198),
    ("unbuckle", 1.4742),
    ("buckle", 1.5646),
    ("unbuckle", 1.8413),
    ("buckle", 1.9224),
)
FIG5B_DIGITIZED_FINAL_LEVEL_UM = 0.97338
FIG5B_RADIUS_UNCERTAINTY_UM = 0.5e-3
FIG5B_TIME_UNCERTAINTY_US = 4.0e-3


def _fig5b_model(equation: bd.RadialBubbleEquation, ambient: float) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        equation,
        bd.PolytropicBubbleGasLaw(FIG5B_POLYTROPIC_INDEX),
        bd.NewtonianBubbleLiquidLaw(FIG5B_VISCOSITY),
        bd.MarmottantShell(
            FIG5B_ELASTICITY,
            0.0,
            FIG5B_WATER_TENSION,
            FIG5B_SHELL_VISCOSITY,
            rupture_surface_tension=FIG5B_BREAKUP_TENSION,
        ),
        bd.BubbleEnvironment(ambient, TEMPERATURE),
        liquid_density=FIG5B_DENSITY,
        liquid_sound_speed=FIG5B_SOUND_SPEED,
    )


def _fig5b_solve(equation: bd.RadialBubbleEquation, ambient: float, frequency: float) -> Record:
    """Rarefaction-first unwindowed burst from rest at `R0 = R_buckling`."""
    end = (FIG5B_CYCLES + 1.4) / frequency
    samples = np.linspace(0.0, 1.05 * end, 16001)
    burst = samples <= FIG5B_CYCLES / frequency
    pressure = np.where(burst, -FIG5B_AMPLITUDE * np.sin(2.0 * np.pi * frequency * samples), 0.0)
    plan = bd.SingleBubblePlan(
        _fig5b_model(equation, ambient),
        bd.SampledPressureDrive(samples, pressure),
        np.linspace(0.0, end, 4401),
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
        events=bd.BubbleEventPolicy(regime_capacity=24),
    )
    result = bd.solve_single_bubble(plan.prepare(FIG5B_RADIUS))
    tape = result.evidence.regime_tape
    count = int(tape.count)
    return {
        "status": int(result.status),
        "successful": bool(result.successful),
        "accepted_steps": int(result.evidence.accepted_steps),
        "times": np.asarray(result.trajectory.times),
        "radius": np.asarray(result.trajectory.radius),
        "velocity": np.asarray(result.trajectory.wall_velocity),
        "gas_pressure": np.asarray(result.trajectory.gas_pressure),
        "transition_times": np.asarray(tape.times)[:count],
        "to_buckled": np.remainder(np.asarray(tape.to_regime)[:count], 3) == 0,
        "tape_overflow": bool(tape.capacity_exceeded),
        "valid": np.asarray(result.trajectory.valid),
    }


def _extrema(times: np.ndarray, radius: np.ndarray, velocity: np.ndarray) -> list[tuple[str, float, float]]:
    """Wall-velocity sign changes refined by a parabola through three samples."""
    step = times[1] - times[0]
    rows = []
    for index in np.where(np.sign(velocity[:-1]) * np.sign(velocity[1:]) < 0.0)[0]:
        center = int(np.clip(index + (abs(velocity[index + 1]) < abs(velocity[index])), 1, times.size - 2))
        before, middle, after = radius[center - 1 : center + 2]
        shift = 0.5 * (before - after) / (before - 2.0 * middle + after)
        value = middle - 0.25 * (before - after) * shift
        rows.append(("max" if velocity[index] > 0.0 else "min", times[center] + shift * step, value))
    return rows


def _asymmetry(radius: np.ndarray, rest: float) -> float:
    """Marmottant's `ΔR+/ΔR−` with `ΔR+ = max R − R0`, `ΔR− = R0 − min R`."""
    return float((np.max(radius) - rest) / (rest - np.min(radius)))


def _fig5b_crossings(solution: Record) -> Record:
    """Rest-level crossings localized by the regime tape against the digitized figure."""
    simulated = [
        ("buckle" if buckled else "unbuckle", 1.0e6 * time)
        for buckled, time in zip(solution["to_buckled"], solution["transition_times"], strict=True)
    ]
    interior = simulated[:FIG5B_INTERIOR_CROSSINGS]
    reference = FIG5B_DIGITIZED_CROSSINGS_US[:FIG5B_INTERIOR_CROSSINGS]
    complete = [kind for kind, _ in interior] == [kind for kind, _ in reference]
    residual = [mine - theirs for (_, mine), (_, theirs) in zip(interior, reference)]
    starts = range(0, FIG5B_INTERIOR_CROSSINGS - 1, 2)
    durations = [interior[i + 1][1] - interior[i][1] for i in starts if i + 1 < len(interior)]
    digitized = [reference[i + 1][1] - reference[i][1] for i in starts]
    duration_error = [mine - theirs for mine, theirs in zip(durations, digitized)]
    return {
        "interior_kinds_match": complete,
        "interior_residual_us": residual,
        "buckled_duration_simulated_us": durations,
        "buckled_duration_digitized_us": digitized,
        "buckled_duration_error_us": duration_error,
        "post_burst_simulated_us": simulated[FIG5B_INTERIOR_CROSSINGS:],
        "post_burst_digitized_us": FIG5B_DIGITIZED_CROSSINGS_US[FIG5B_INTERIOR_CROSSINGS:],
        "crossing_gate": complete and max(map(abs, residual)) <= 3.0 * FIG5B_TIME_UNCERTAINTY_US,
        "duration_gate": complete
        and max(map(abs, duration_error)) <= 3.0 * np.sqrt(2.0) * FIG5B_TIME_UNCERTAINTY_US,
    }


def _fig5b_comparison(solution: Record) -> Record:
    """Burst-interior signatures gated against the digitized Fig. 5(b); edges reported."""
    valid = solution["valid"]
    times = 1.0e6 * solution["times"][valid]
    radius = 1.0e6 * solution["radius"][valid]
    extrema = _extrema(times, radius, solution["velocity"][valid])
    maxima = [(time, value) for kind, time, value in extrema if kind == "max"]
    minima = [(time, value) for kind, time, value in extrema if kind == "min"]
    interior_error = [
        mine[1] - theirs[1] for mine, theirs in zip(maxima[1:5], FIG5B_DIGITIZED_MAXIMA_UM[1:5])
    ] + [mine[1] - theirs[1] for mine, theirs in zip(minima[:5], FIG5B_DIGITIZED_MINIMA_UM)]
    rest = 1.0e6 * FIG5B_RADIUS
    figure_plus = max(value for _, value in FIG5B_DIGITIZED_MAXIMA_UM) - rest
    figure_minus = rest - min(value for _, value in FIG5B_DIGITIZED_MINIMA_UM)
    figure_ratio = figure_plus / figure_minus
    ratio_uncertainty = (
        figure_ratio * FIG5B_RADIUS_UNCERTAINTY_UM * np.hypot(1.0 / figure_plus, 1.0 / figure_minus)
    )
    simulated_ratio = _asymmetry(radius, rest)
    crossings = _fig5b_crossings(solution)
    bump_time, bump_radius = FIG5B_DIGITIZED_MAXIMA_UM[-1]
    return {
        "simulated_maxima_us_um": maxima[:6],
        "simulated_minima_us_um": minima[:5],
        "interior_extremum_error_um": interior_error,
        "asymmetry_simulated": simulated_ratio,
        "asymmetry_digitized": figure_ratio,
        "asymmetry_digitized_uncertainty": ratio_uncertainty,
        "crossings": crossings,
        "burst_edges_not_gated": {
            "first_maximum_digitized_us_um": FIG5B_DIGITIZED_MAXIMA_UM[0],
            "first_maximum_simulated_us_um": maxima[0] if maxima else None,
            "post_burst_bump_digitized_us_um": (bump_time, bump_radius),
            "radius_simulated_at_bump_time_um": float(np.interp(bump_time, times, radius)),
            "final_level_digitized_um": FIG5B_DIGITIZED_FINAL_LEVEL_UM,
            "radius_simulated_at_1.95us_um": float(np.interp(1.95, times, radius)),
        },
        "gates": {
            "compression_only": simulated_ratio < 1.0,
            "asymmetry": abs(simulated_ratio - figure_ratio) <= 3.0 * ratio_uncertainty,
            "interior_extrema": len(maxima) >= 5
            and len(minima) >= 5
            and max(map(abs, interior_error)) <= 3.0 * FIG5B_RADIUS_UNCERTAINTY_UM,
            "interior_crossings": crossings["crossing_gate"],
            "buckled_durations": crossings["duration_gate"],
        },
    }


def marmottant_fig5b() -> Record:
    primary = _fig5b_solve("rayleigh_plesset_gas_radiation", AMBIENT, FIG5B_FREQUENCY)
    comparison = _fig5b_comparison(primary)
    period = 1.0 / FIG5B_FREQUENCY
    # Paper Eq. (7): the period average of p_g R^4 over that of R^4, "1.2 P0" for Fig. 5(b).
    averaged_by_period = []
    for cycle in (1, 2, 3):
        window = (primary["times"] >= cycle * period) & (primary["times"] < (cycle + 1) * period)
        fourth = primary["radius"][window] ** 4
        averaged_by_period.append(
            float(np.sum(primary["gas_pressure"][window] * fourth) / np.sum(fourth) / AMBIENT)
        )
    averaged = float(np.mean(averaged_by_period))
    rest = FIG5B_RADIUS
    ambient_sensitivity = _fig5b_solve(
        "rayleigh_plesset_gas_radiation", 1.0e5, FIG5B_FREQUENCY
    )
    radiation_sensitivity = _fig5b_solve(
        "rayleigh_plesset_radiation", AMBIENT, FIG5B_FREQUENCY
    )
    sensitivity = {
        "ambient_100kpa": _fig5b_comparison(ambient_sensitivity),
        "full_wall_radiation": _fig5b_comparison(radiation_sensitivity),
    }
    frequency_ratios = {}
    sensitivity_successful = [
        bool(ambient_sensitivity["successful"]),
        bool(radiation_sensitivity["successful"]),
    ]
    for frequency in (1.0e6, 4.0e6):
        solution = _fig5b_solve("rayleigh_plesset_gas_radiation", AMBIENT, frequency)
        sensitivity_successful.append(bool(solution["successful"]))
        frequency_ratios[f"{frequency / 1.0e6:g}MHz"] = _asymmetry(
            solution["radius"][solution["valid"]], rest
        )
    frequency_ratios["2.9MHz"] = comparison["asymmetry_simulated"]
    spread = (max(frequency_ratios.values()) - min(frequency_ratios.values())) / comparison[
        "asymmetry_simulated"
    ]
    statements = {
        "compression_only_ratio_below_one": comparison["gates"]["compression_only"],
        "averaged_gas_pressure_over_p0_by_period_2_to_4": averaged_by_period,
        "averaged_gas_pressure_over_p0": averaged,
        "averaged_gas_pressure_rounds_to_1.2": round(averaged, 1) == 1.2,
        "frequency_1_to_4_mhz_asymmetry_ratios": frequency_ratios,
        "frequency_1_to_4_mhz_relative_spread": spread,
    }
    gates = dict(comparison["gates"])
    gates["averaged_gas_pressure"] = statements["averaged_gas_pressure_rounds_to_1.2"]
    gates["solver"] = primary["successful"] and not primary["tape_overflow"]
    native_successful = bool(gates["solver"] and all(sensitivity_successful))
    return {
        "source": MARMOTTANT_2005,
        "equation": "rayleigh_plesset_gas_radiation (paper Eq. 3)",
        "parameters": {
            "caption": {
                "buckling_radius_m": FIG5B_RADIUS,
                "shell_elasticity_n_per_m": FIG5B_ELASTICITY,
                "shell_viscosity_kg_per_s": FIG5B_SHELL_VISCOSITY,
                "breakup_tension_n_per_m": FIG5B_BREAKUP_TENSION,
                "liquid_density_kg_per_m3": FIG5B_DENSITY,
                "liquid_viscosity_pa_s": FIG5B_VISCOSITY,
                "sound_speed_m_per_s": FIG5B_SOUND_SPEED,
                "polytropic_index": FIG5B_POLYTROPIC_INDEX,
                "frequency_hz": FIG5B_FREQUENCY,
                "amplitude_pa": FIG5B_AMPLITUDE,
            },
            "section_1_water_tension_n_per_m": FIG5B_WATER_TENSION,
            "assumed_ambient_pressure_pa": AMBIENT,
            "inferred_burst": (
                f"rarefaction-first unwindowed sine, {FIG5B_CYCLES} cycles, onset at the "
                "figure's t = 0"
            ),
        },
        "gated_scope": (
            "burst interior: five minima, maxima 2-5, the first nine rest-level crossings "
            "and four complete buckled intervals; the first expansion and the post-burst "
            "recovery depend on the unstated burst edges and are reported only"
        ),
        "digitization": {
            "maxima_us_um": FIG5B_DIGITIZED_MAXIMA_UM,
            "minima_us_um": FIG5B_DIGITIZED_MINIMA_UM,
            "rest_level_crossings_us": FIG5B_DIGITIZED_CROSSINGS_US,
            "final_level_um": FIG5B_DIGITIZED_FINAL_LEVEL_UM,
            "radius_uncertainty_um": FIG5B_RADIUS_UNCERTAINTY_UM,
            "time_uncertainty_us": FIG5B_TIME_UNCERTAINTY_US,
        },
        "status": primary["status"],
        "accepted_steps": primary["accepted_steps"],
        "comparison": comparison,
        "text_statements": statements,
        "sensitivity": sensitivity,
        "gates": gates,
        "successful": native_successful,
        "passed": native_successful and all(gates.values()),
    }


def prosperetti_thermal() -> Record:
    radius, gamma, conductivity = 1.0e-5, 1.4, 0.0262
    amount = AMBIENT * float(bd.sphere_volume(radius)) / (bd.MOLAR_GAS_CONSTANT * TEMPERATURE)
    diffusivity = conductivity * (gamma - 1.0) * float(bd.sphere_volume(radius)) / (
        gamma * bd.MOLAR_GAS_CONSTANT * amount
    )
    peclets = np.asarray([0.5, 5.0, 50.0, 500.0])
    frequencies = peclets * diffusivity / radius**2
    exact = np.asarray(bd.prosperetti_polytropic_index(gamma, peclets))
    exact_damping = 3.0 * AMBIENT * exact.imag / (2.0 * frequencies * 1000.0 * radius**2)
    rows = []
    successful = []
    for nodes in (4, 8, 12, 16, 24):
        model = bd.RadialBubbleModel(
            "rayleigh_plesset",
            bd.SpectralThermalBubbleGasLaw(gamma, conductivity, node_count=nodes),
            bd.NewtonianBubbleLiquidLaw(0.0),
            bd.CleanBubbleInterfaceLaw(0.0),
            bd.BubbleEnvironment(AMBIENT, TEMPERATURE),
            liquid_density=1000.0,
            liquid_sound_speed=1500.0,
        )
        response = bd.linear_bubble_response(model, radius, frequencies)
        successful.append(bool(response.successful))
        rows.append(
            {
                "nodes": nodes,
                "polytropic_relative_error": float(
                    np.max(np.abs(np.asarray(response.effective_polytropic_index) / exact.real - 1.0))
                ),
                "thermal_damping_relative_error": float(
                    np.max(np.abs(np.asarray(response.thermal_damping) / exact_damping - 1.0))
                ),
                "successful": bool(response.successful),
            }
        )
    native_successful = all(successful)
    return {
        "reference": "Prosperetti (1977, 1991) exact linear homobaric theory, isothermal wall",
        "peclet_numbers": peclets.tolist(),
        "refinement": rows,
        "successful": native_successful,
        "passed": native_successful
        and rows[-1]["thermal_damping_relative_error"] < 1.0e-6,
    }


class _DissolutionFloor:
    """Terminal `solve_ivp` event when the radius reaches `floor`."""

    terminal = True

    def __init__(self, floor: float, /) -> None:
        self.floor = floor

    def __call__(self, time: float, state: np.ndarray) -> float:
        del time
        return float(state[0] - self.floor)


def _history_reference(properties: bd.GasSolutionProperties, radius: float, end: float) -> float:
    """Independent method of lines: EP interface balance with a resolved half-line heat equation.

    The Duhamel history term `√D D_t^{1/2} Δc` equals `−D ∂_x u(0, t)` for
    `u_t = D u_xx`, `u(0, t) = Δc(t)`, `u(x, 0) = 0`; the half-line is resolved on
    a geometric grid and integrated with implicit BDF.
    """
    diffusivity = float(properties.diffusivity)
    saturation = float(properties.saturation_concentration)
    ratio = float(properties.saturation_ratio)
    tension = float(properties.surface_tension)
    temperature = float(properties.temperature)
    points = np.concatenate(([0.0], np.geomspace(1.0e-10, 20.0 * np.sqrt(diffusivity * end), 700)))
    spacing = np.diff(points)
    interior = points[1:-1]
    left, right = spacing[:-1], spacing[1:]
    lower = 2.0 * diffusivity / (left * (left + right))
    upper = 2.0 * diffusivity / (right * (left + right))
    operator = diags([lower[1:], -(lower + upper), upper[:-1]], [-1, 0, 1]).tocsc()

    def excess(r: float) -> float:
        return saturation * (1.0 + 2.0 * tension / (r * AMBIENT)) - ratio * saturation

    def amount_slope(r: float) -> float:
        return (3.0 * AMBIENT * r**2 + 4.0 * tension * r) * 4.0 * np.pi / (
            3.0 * bd.MOLAR_GAS_CONSTANT * temperature
        )

    def field(_: float, state: np.ndarray) -> np.ndarray:
        r, profile = state[0], state[1:]
        boundary = excess(r)
        h0, h1 = spacing[0], spacing[0] + spacing[1]
        gradient = (
            -boundary * (h0 + h1) / (h0 * h1) + profile[0] * h1 / (h0 * (h1 - h0)) - profile[1] * h0 / (h1 * (h1 - h0))
        )
        flux = diffusivity * boundary / r - diffusivity * gradient
        rate = -4.0 * np.pi * r**2 * flux / amount_slope(r)
        diffusion = operator @ profile
        diffusion[0] += lower[0] * boundary
        return np.concatenate(([rate], diffusion))

    size = interior.shape[0] + 1
    # Radius couples to the first two profile nodes; the first node sees the radius
    # through the Dirichlet value; the profile itself is tridiagonal.
    pattern = diags([np.ones(size - 1), np.ones(size), np.ones(size - 1)], [-1, 0, 1]).tolil()
    pattern[0, 2] = 1.0
    solution = solve_ivp(
        field,
        (0.0, end),
        np.concatenate(([radius], np.zeros(interior.shape[0]))),
        method="BDF",
        rtol=1.0e-9,
        atol=np.concatenate(([1.0e-15], np.full(interior.shape[0], 1.0e-12))),
        events=_DissolutionFloor(1.0e-3 * radius),
        jac_sparsity=pattern.tocsc(),
    )
    if solution.t_events[0].size == 0:
        raise RuntimeError("Independent Epstein-Plesset reference did not dissolve.")
    return float(solution.t_events[0][0])


def epstein_plesset() -> Record:
    radius = 1.0e-5
    closed = bd.GasSolutionProperties(2.0e-9, 0.6, 0.2, 0.0, AMBIENT, TEMPERATURE)
    lifetime = float(bd.quasi_static_dissolution_time(radius, closed))
    quasi = bd.solve_epstein_plesset(
        bd.EpsteinPlessetPlan(
            closed, np.linspace(0.0, 1.2 * lifetime, 5), route="quasi_static", initial_radius=radius
        ).prepare(radius)
    )
    closed_error = abs(float(quasi.evidence.lifetime) / (lifetime * (1.0 - 1.0e-6)) - 1.0)
    laplace = bd.GasSolutionProperties(2.0e-9, 0.6, 0.8, 0.072, AMBIENT, TEMPERATURE)
    horizon = 20.0
    # Kvaerno5 needs about 15.5k accepted and 15.6k rejected steps to reach R/R0 = 1e-3
    # here. The step count is dominated by the stiff history modes and the final
    # Laplace-driven collapse, so the plan's default budget of 16384 ends in MAX_STEPS
    # at R/R0 = 0.01.
    history_plan = bd.EpsteinPlessetPlan(
        laplace,
        np.linspace(0.0, horizon, 5),
        route="full_history",
        initial_radius=radius,
        maximum_steps=65536,
    )
    history = bd.solve_epstein_plesset(history_plan.prepare(radius))
    reference = _history_reference(laplace, radius, horizon)
    history_error = abs(float(history.evidence.lifetime) / reference - 1.0)
    native_successful = bool(quasi.successful and history.successful)
    return {
        "reference": (
            "closed-form quasi-static lifetime; independent method-of-lines half-line diffusion "
            "for the full Duhamel history with Laplace pressure"
        ),
        "closed_form_lifetime_s": lifetime,
        "closed_form_relative_error": closed_error,
        "quasi_static_status": bd.BubbleDynamicsStatus(int(quasi.status)).name,
        "full_history_lifetime_s": float(history.evidence.lifetime),
        "full_history_status": bd.BubbleDynamicsStatus(int(history.status)).name,
        "full_history_accepted_steps": int(history.evidence.accepted_steps),
        "full_history_rejected_steps": int(history.evidence.rejected_steps),
        "independent_lifetime_s": reference,
        "full_history_relative_error": history_error,
        "history_kernel_error": history_plan.history_kernel_error,
        "amount_residual": float(history.evidence.amount_residual),
        "successful": native_successful,
        "passed": native_successful
        and closed_error < 1.0e-8
        and int(history.status) == int(bd.BubbleDynamicsStatus.DISSOLVED)
        and history_error < 1.0e-3,
    }


def lohse_zhang() -> Record:
    properties = bd.GasSolutionProperties(2.0e-9, 0.6, 2.0, 0.072, AMBIENT, TEMPERATURE)
    plan = bd.PinnedSurfaceBubblePlan(properties, 1.0, np.linspace(0.0, 2.0e-2, 11), contact="pinned")
    equilibrium = plan.equilibrium(1.0e-6)
    trajectories = []
    successful = []
    for start in (40.0, 8.0):
        result = bd.solve_surface_bubble(plan.prepare(1.0e-6, np.pi - np.radians(start)))
        trajectories.append(float(np.degrees(np.asarray(result.gas_contact_angle)[-1])))
        successful.append(bool(result.successful))
    angle = float(np.degrees(equilibrium.gas_contact_angle))
    native_successful = all(successful)
    return {
        "reference": "Lohse & Zhang (2015): sin(theta_gas) = zeta L / L_c, L_c = 4 sigma / p0; theta ~ 20.6 deg at L = 1 um, zeta = 1",
        "equilibrium_gas_angle_deg": angle,
        "equilibrium_liquid_angle_deg": float(np.degrees(equilibrium.liquid_contact_angle)),
        "stability_derivative_per_s": float(equilibrium.stability_derivative),
        "final_angles_from_40_and_8_deg": trajectories,
        "solve_successful": successful,
        "successful": native_successful,
        "passed": native_successful
        and abs(angle - 20.6) < 0.05
        and bool(equilibrium.stable)
        and all(abs(value - angle) < 0.5 for value in trajectories),
    }


def _cloud_pair(
    model: bd.RadialBubbleModel,
    radii: tuple[float, float],
    distance: float,
    initial_radii: tuple[float, float] | None = None,
) -> bd.BubbleSpeciesGroup:
    return bd.BubbleSpeciesGroup(
        model,
        np.asarray(radii),
        np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        bubble_ids=(0, 1),
        initial_radii=None if initial_radii is None else np.asarray(initial_radii),
    )


def _crossing_frequency(times: np.ndarray, signal: np.ndarray) -> float:
    rising = np.nonzero((signal[:-1] < 0.0) & (signal[1:] >= 0.0))[0]
    crossings = times[rising] - signal[rising] * (times[rising + 1] - times[rising]) / (
        signal[rising + 1] - signal[rising]
    )
    return 2.0 * np.pi / float(np.mean(np.diff(crossings)))


def cloud_two_bubble_modes() -> Record:
    radius = 10.0e-6
    omega0 = float(bd.minnaert_angular_frequency(radius, AMBIENT, DENSITY, 1.4, surface_tension=0.0))
    times = np.linspace(0.0, 8.0 * np.pi / omega0, 1601)[1:]
    rows: list[Record] = []
    for ratio in (3.0, 4.0, 8.0):
        for sign in (1.0, -1.0):
            displaced = (radius * (1.0 + 1.0e-4), radius * (1.0 + sign * 1.0e-4))
            group = _cloud_pair(_model("rayleigh_plesset"), (radius, radius), ratio * radius, displaced)
            plan = bd.BubbleCloudPlan(
                (group,),
                bd.ConstantPressureDrive(0.0),
                times,
                relative_tolerance=1.0e-10,
                absolute_tolerance=1.0e-12,
            )
            result = bd.solve_bubble_cloud(plan.prepare())
            signal = np.asarray(result.trajectory.radius[:, 0]) - radius
            measured = _crossing_frequency(np.asarray(result.trajectory.times), signal)
            expected = omega0 / np.sqrt(1.0 + sign / ratio)
            kinetic = 2.0 * np.pi * DENSITY * radius**3 * float(
                np.nanmax(np.asarray(result.trajectory.wall_velocity)) ** 2
            )
            rows.append(
                {
                    "distance_over_radius": ratio,
                    "mode": "in-phase" if sign > 0.0 else "anti-phase",
                    "status": int(result.status),
                    "successful": bool(result.successful),
                    "measured_angular_frequency": measured,
                    "expected_angular_frequency": expected,
                    "relative_error": measured / expected - 1.0,
                    "work_residual_over_kinetic": abs(float(result.evidence.work_residual)) / kinetic,
                }
            )
    native_successful = all(row["successful"] for row in rows)
    return {
        "reference": "two identical coupled Rayleigh-Plesset bubbles: omega^2 = omega0^2/(1 +- R0/d)",
        "rows": rows,
        "successful": native_successful,
        "passed": native_successful
        and all(
            abs(row["relative_error"]) < 1.0e-4
            and row["work_residual_over_kinetic"] < 1.0e-6
            for row in rows
        ),
    }


def _linear_bjerknes_toward(
    model: bd.RadialBubbleModel,
    radii: tuple[float, float],
    distance: float,
    amplitude: float,
    omega: float,
) -> tuple[float, bool]:
    """Return linear mean force toward the neighbor and native response status."""
    volumes = []
    successful = []
    for radius in radii:
        response = bd.linear_bubble_response(model, radius, np.array([omega]))
        volumes.append(4.0 * np.pi * radius**2 * complex(response.radius_response[0]) * amplitude)
        successful.append(bool(response.successful))
    force = DENSITY * omega**2 * (volumes[0] * np.conj(volumes[1])).real / (
        8.0 * np.pi * distance**2
    )
    return force, all(successful)


def cloud_bjerknes_sign() -> Record:
    amplitude, omega, distance = 1.0e4, 2.0 * np.pi * 4.0e5, 200.0e-6
    model = _model(
        "keller_miksis", viscosity=2.0e-2, interface=bd.CleanBubbleInterfaceLaw(0.072)
    )
    rows: dict[str, Record] = {}
    for label, radii in (("both-below", (2.0e-6, 3.0e-6)), ("straddling", (2.0e-6, 30.0e-6))):
        plan = bd.BubbleCloudPlan(
            (_cloud_pair(model, radii, distance),),
            bd.HarmonicPressureDrive(amplitude, omega),
            np.linspace(0.0, 40.0e-6, 1601)[1:],
        )
        result = bd.solve_bubble_cloud(plan.prepare())
        forces = bd.mean_bjerknes_forces(result, 20.0e-6, 40.0e-6)
        simulated = float(np.asarray(forces.secondary)[0, 0])
        linear, linear_successful = _linear_bjerknes_toward(
            model, radii, distance, amplitude, omega
        )
        rows[label] = {
            "status": int(result.status),
            "successful": bool(result.successful) and linear_successful,
            "simulated_force_toward_neighbor_n": simulated,
            "linear_isolated_estimate_n": linear,
            "ratio": simulated / linear,
        }
    native_successful = all(row["successful"] for row in rows.values())
    return {
        "reference": "Bjerknes sign rule; linear isolated-bubble secondary force rho w^2 Re(V0 V1*)/(8 pi d^2)",
        "rows": rows,
        "successful": native_successful,
        "passed": native_successful
        and rows["both-below"]["simulated_force_toward_neighbor_n"] > 0.0
        and rows["straddling"]["simulated_force_toward_neighbor_n"] < 0.0
        and all(0.7 < row["ratio"] < 1.3 for row in rows.values()),
    }


def _cloud_lattice(count_per_side: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(count_per_side)
    axis = np.arange(count_per_side) * 60.0e-6
    lattice = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    points = lattice + rng.uniform(-10.0e-6, 10.0e-6, size=lattice.shape)
    radii = rng.uniform(6.0e-6, 12.0e-6, size=lattice.shape[0])
    return points, radii, rng.uniform(-2.0, 2.0, size=lattice.shape[0])


def cloud_fmm_accuracy() -> Record:
    rows: list[Record] = []
    # N=125 is direct P2P for this geometry; N=343 exercises far-field order.
    for side in (7,):
        points, radii, velocity = _cloud_lattice(side)
        count = radii.shape[0]
        corrections = {}
        for label, route, order in (
            ("dense", "dense", 5),
            ("fmm-3", "fmm", 3),
            ("fmm-5", "fmm", 5),
            ("fmm-6", "fmm", 6),
        ):
            group = bd.BubbleSpeciesGroup(
                _model("rayleigh_plesset", interface=bd.CleanBubbleInterfaceLaw(0.072)),
                radii,
                points,
                bubble_ids=tuple(range(count)),
                initial_radii=1.05 * radii,
                initial_wall_velocities=velocity,
            )
            plan = bd.BubbleCloudPlan(
                (group,),
                bd.HarmonicPressureDrive(2.0e4, 2.0 * np.pi * 1.0e5),
                np.array([1.0e-6]),
                route="dense" if route == "dense" else "fmm",
                resources=bd.BubbleCloudResourcePolicy(
                    maximum_dense_entries=count * count,
                    fmm_order=order,
                    coupling_tolerance=1.0e-13,
                ),
            )
            prepared = plan.prepare()
            rates = prepared.rates(prepared.initial_state, 2.5e-7)
            corrections[label] = (
                np.asarray(rates.acceleration - rates.uncoupled_acceleration),
                int(rates.coupling_iterations),
                bool(rates.coupling_successful),
                float(rates.coupling_residual),
            )
        dense = corrections["dense"][0]
        scale = float(np.max(np.abs(dense)))
        rows.append(
            {
                "bubble_count": count,
                "errors": {
                    label: float(np.max(np.abs(value[0] - dense)) / scale)
                    for label, value in corrections.items()
                    if label != "dense"
                },
                "iterations": {label: value[1] for label, value in corrections.items()},
                "successful": all(value[2] for value in corrections.values()),
                "residuals": {label: value[3] for label, value in corrections.items()},
            }
        )
    native_successful = all(row["successful"] for row in rows)
    return {
        "reference": "dense Cholesky coupled accelerations (same cloud, same state)",
        "qualified_order": 6,
        "declared_tolerance": 1.0e-5,
        "rows": rows,
        "successful": native_successful,
        "passed": native_successful
        and all(
            row["errors"]["fmm-6"] < 1.0e-5
            and row["errors"]["fmm-6"] < row["errors"]["fmm-5"]
            and row["errors"]["fmm-5"] < row["errors"]["fmm-3"]
            for row in rows
        ),
    }


def cloud_retarded_limit() -> Record:
    times = np.linspace(0.0, 6.0e-6, 61)[1:]
    drive = bd.HarmonicPressureDrive(3.0e4, 2.0 * np.pi * 2.0e5)
    rows: list[Record] = []
    for sound_speed in (1481.0, 5924.0, 23696.0):
        model = _model(
            "rayleigh_plesset",
            interface=bd.CleanBubbleInterfaceLaw(0.072),
            sound_speed=sound_speed,
        )
        group = _cloud_pair(model, (10.0e-6, 8.0e-6), 80.0e-6)
        incompressible = bd.solve_bubble_cloud(bd.BubbleCloudPlan((group,), drive, times).prepare())
        retarded = bd.solve_bubble_cloud(
            bd.BubbleCloudPlan((group,), drive, times, coupling="retarded").prepare()
        )
        rows.append(
            {
                "sound_speed": sound_speed,
                "status": (int(incompressible.status), int(retarded.status)),
                "successful": bool(incompressible.successful & retarded.successful),
                "max_radius_difference_m": float(
                    np.max(
                        np.abs(np.asarray(retarded.trajectory.radius - incompressible.trajectory.radius))
                    )
                ),
                "retardation_ratio": float(retarded.evidence.retardation_ratio),
                "history_occupancy": int(retarded.evidence.history_occupancy),
                "accepted_steps": int(retarded.evidence.accepted_steps),
            }
        )
    differences: list[float] = [row["max_radius_difference_m"] for row in rows]
    native_successful = all(row["successful"] for row in rows)
    return {
        "reference": "retarded neighbour coupling -> incompressible coupling as c -> infinity (first order in d/c)",
        "rows": rows,
        "successful": native_successful,
        "passed": native_successful
        and differences[1] < 0.4 * differences[0]
        and differences[2] < 0.4 * differences[1],
    }


def cloud_emission_linear() -> Record:
    radius, amplitude, frequency = 5.0e-6, 50.0, 3.0e5
    omega = 2.0 * np.pi * frequency
    period = 1.0 / frequency
    model = _model(
        "keller_miksis",
        viscosity=1.0e-3,
        interface=bd.CleanBubbleInterfaceLaw(0.072),
    )
    response = bd.linear_bubble_response(model, radius, np.array([omega]))
    rows: list[Record] = []
    for distance in (2.0e-3, 1.0e-2):
        delay = distance / SOUND_SPEED
        window = 10.0 * period
        end = 20.0 * period + delay
        emission_times = np.linspace(end - window, end, 801)
        plan = bd.BubbleCloudPlan(
            (bd.BubbleSpeciesGroup(model, np.array([radius]), np.zeros((1, 3)), bubble_ids=(0,)),),
            bd.HarmonicPressureDrive(amplitude, omega),
            np.linspace(0.0, end, 401)[1:],
            emission=bd.FarFieldEmissionPlan(np.array([[distance, 0.0, 0.0]]), emission_times),
        )
        result = bd.solve_bubble_cloud(plan.prepare())
        if result.emission is None:
            raise RuntimeError("The plan requested far-field emission.")
        pressure = np.asarray(result.emission.pressure)[:, 0]
        weights = np.full(emission_times.shape, emission_times[1] - emission_times[0])
        weights[[0, -1]] *= 0.5
        measured = 2.0 * abs(np.sum(weights * pressure * np.exp(-1j * omega * emission_times))) / window
        expected = DENSITY * omega**2 * radius**2 * abs(complex(response.radius_response[0])) * amplitude / distance
        rows.append(
            {
                "distance_m": distance,
                "measured_amplitude_pa": measured,
                "linear_monopole_amplitude_pa": expected,
                "relative_error": measured / expected - 1.0,
                "successful": bool(result.successful),
                "covered": bool(np.all(np.asarray(result.emission.evidence.covered))),
            }
        )
    native_successful = bool(response.successful) and all(
        row["successful"] for row in rows
    )
    return {
        "reference": "linear monopole far field rho w^2 R0^2 |R_hat| p_a / r with linear_bubble_response",
        "rows": rows,
        "successful": native_successful,
        "passed": native_successful
        and all(row["covered"] and abs(row["relative_error"]) < 1.0e-2 for row in rows),
    }


CAMPAIGNS: dict[str, Callable[[], Record]] = {
    "rayleigh-collapse": rayleigh_collapse,
    "rayleigh-plesset-energy": rayleigh_plesset_energy,
    "compressible-limits": compressible_limits,
    "minnaert-and-coated-resonance": minnaert_and_coated_resonance,
    "marmottant-event-convergence": marmottant_events,
    "compression-only-reference": marmottant_fig5b,
    "prosperetti-thermal": prosperetti_thermal,
    "epstein-plesset": epstein_plesset,
    "lohse-zhang": lohse_zhang,
    "cloud-two-bubble-modes": cloud_two_bubble_modes,
    "cloud-bjerknes-sign": cloud_bjerknes_sign,
    "cloud-fmm-accuracy": cloud_fmm_accuracy,
    "cloud-retarded-limit": cloud_retarded_limit,
    "cloud-emission-linear": cloud_emission_linear,
}


def run_qualification(names: tuple[str, ...], /) -> Record:
    campaigns = {}
    for name in names:
        started = time.perf_counter()
        record = CAMPAIGNS[name]()
        record["seconds"] = time.perf_counter() - started
        campaigns[name] = record
    return {
        "identity": _runtime_identity(),
        "campaigns": campaigns,
        "successful": bool(campaigns)
        and all(
            record["successful"] is True and record["passed"] is True
            for record in campaigns.values()
        ),
    }


def _json_ready(value: object, /) -> object:
    """Record non-finite metrics explicitly instead of failing JSON encoding."""
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--campaign",
        action="append",
        choices=tuple(CAMPAIGNS),
        help="Run only the named campaign(s); by default every campaign runs.",
    )
    arguments = parser.parse_args()
    report = run_qualification(tuple(arguments.campaign or CAMPAIGNS))
    encoded = json.dumps(_json_ready(report), indent=2, sort_keys=True, allow_nan=False)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
