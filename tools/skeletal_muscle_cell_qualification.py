#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Produce source, convergence, stiffness, pulse, rollback, and fatigue evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.skeletal_muscle.cellular import (
    ShortenCellState,
    ShortenFastTwitchModel,
    ShortenIntegrationPlan,
    ShortenPulseProtocol,
)


_REFERENCE = {
    "vS": np.array(
        [
            -79.974,
            -70.56984893234592,
            -65.70913559330221,
            -60.78146659943978,
            -50.85026953450061,
            -16.708012321459826,
            30.95461429513052,
            34.243156102644186,
            29.068672923369597,
            23.57911324817365,
            18.850029745397745,
        ]
    ),
    "vT": np.array(
        [
            -80.2,
            -79.61131460404869,
            -78.31889340369032,
            -76.6957615003432,
            -74.50525447267584,
            -70.24867907985126,
            -60.522923881167905,
            -48.54234923766906,
            -34.540547388150806,
            -14.372310683780894,
            6.195523220276491,
        ]
    ),
    "Ca_1": np.array(
        [
            0.1,
            0.2429379353792191,
            0.2955065541906228,
            0.3539320121406062,
            0.42105582260196767,
            0.5025792472471213,
            0.6177056990291135,
            0.8183843821635061,
            1.2168809621401484,
            2.1980420224751724,
            5.254755884007505,
        ]
    ),
    "Ca_2": np.array(
        [
            0.1,
            0.15634130294752863,
            0.1899304617885617,
            0.21645863304654478,
            0.23781586122958662,
            0.25558566826599305,
            0.2713194202402873,
            0.28723076564259564,
            0.3072660627068752,
            0.341045508490308,
            0.4218805896020381,
        ]
    ),
    "A_2": np.array(
        [
            0.23,
            0.23027145724221798,
            0.23051068011035883,
            0.23072998972707892,
            0.23093147657653515,
            0.2311153703402217,
            0.23128160565639028,
            0.2314301159009796,
            0.23156089727517648,
            0.23167404936823247,
            0.23176986440800512,
        ]
    ),
}


def _trajectory_error(model, states) -> dict[str, float]:
    errors = {}
    for name, reference in _REFERENCE.items():
        observed = np.asarray(states[:, model.state_layout.index(name)])
        scale = np.maximum(np.abs(reference), 1.0e-3)
        errors[name] = float(np.max(np.abs(observed - reference) / scale))
    return errors


def qualify() -> dict[str, object]:
    model = ShortenFastTwitchModel()
    state = model.initialize(dtype=jnp.float64)
    initial = model.evaluate(0.0, state)
    expected_rhs = {
        "vS": 142.22126573539276,
        "vT": -1.1303020276710498,
        "A_1": -0.030599999999999995,
        "A_2": 0.0029999999999999953,
        "P": 2.4005899999999997e-5,
    }
    rhs_error = max(
        abs(float(initial.state_rate_per_ms[model.state_layout.index(name)]) - value)
        / max(abs(value), 1.0e-8)
        for name, value in expected_rhs.items()
    )
    expected_currents = {
        "I_T": 1.5066666666666606,
        "I_HH": 150.0,
        "I_Cl": 4.1054172489320715,
        "I_IR": 1.1083423178755314,
        "I_DR": 3.9299144639919512e-7,
        "I_Na": -1.5742434721521985,
        "I_NaK": 2.6325511102937238,
        "I_ionic_s": -143.72793240205942,
        "I_Cl_t": 0.3608557572244595,
        "I_IR_t": 0.9789025374472027,
        "I_DR_t": 1.551916436039707e-7,
        "I_Na_t": -0.15781307233069097,
        "I_NaK_t": 0.26224553902732267,
        "I_ionic_t": 1.4441909165599374,
    }
    current_error = max(
        abs(float(initial.algebraic_value(name)) - value)
        / max(abs(value), 1.0e-8)
        for name, value in expected_currents.items()
    )

    reference_grid = np.linspace(0.0, 1.0, 11)
    reference_trajectory = ShortenIntegrationPlan(
        model,
        reference_grid,
        relative_tolerance=2.0e-8,
        absolute_tolerance=2.0e-10,
    ).prepare().integrate()
    source_errors = _trajectory_error(model, reference_trajectory.states)

    coarse = ShortenIntegrationPlan(
        model,
        [0.0, 0.5, 1.0],
        relative_tolerance=2.0e-5,
        absolute_tolerance=2.0e-7,
    ).prepare().integrate()
    refined = ShortenIntegrationPlan(
        model,
        [0.0, 0.5, 1.0],
        relative_tolerance=2.0e-7,
        absolute_tolerance=2.0e-9,
    ).prepare().integrate()
    refinement_error = float(
        jnp.linalg.norm(coarse.states[-1] - refined.states[-1])
        / jnp.maximum(jnp.linalg.norm(refined.states[-1]), 1.0)
    )

    pulse_edges = ShortenPulseProtocol().event_times_ms()
    fatigue_grid = np.unique(
        np.concatenate((np.arange(0.0, 451.0, 5.0), np.asarray(pulse_edges)))
    )
    fatigue = ShortenIntegrationPlan(
        model,
        fatigue_grid,
        relative_tolerance=3.0e-6,
        absolute_tolerance=3.0e-8,
    ).prepare().integrate()
    a2 = np.asarray(fatigue.states[:, model.state_layout.index("A_2")])
    ca2 = np.asarray(fatigue.states[:, model.state_layout.index("Ca_2")])
    fatigue_times = np.asarray(fatigue.times_ms)
    a2_baseline = float(a2[0])
    twitch_peaks = []
    calcium_peaks = []
    for onset in np.arange(0.0, 401.0, 50.0):
        window = (fatigue_times >= onset) & (fatigue_times < onset + 50.0)
        twitch_peaks.append(float(np.max(a2[window]) - a2_baseline))
        calcium_peaks.append(float(np.max(ca2[window])))
    fatigue_ratio = twitch_peaks[-1] / twitch_peaks[0]

    kinetics = model.evaluate(0.75, state, stimulus_current_uA_per_cm2=0.0)
    fastest_source_time_ms = 1.0 / model.parameters[
        model.parameter_layout.index("k_Lm")
    ]
    stiffness_ratio = float(
        jnp.max(kinetics.gate_time_constant_ms) / fastest_source_time_ms
    )
    gate_full = model.exact_gate_update(0.75, state, 0.02)
    gate_half = model.exact_gate_update(0.75, state, 0.01)
    gate_refined = model.exact_gate_update(0.75, gate_half, 0.01)
    exact_gate_semigroup_error = float(
        jnp.max(jnp.abs(gate_full[8:18] - gate_refined[8:18]))
    )

    prepared = ShortenIntegrationPlan(model, [0.0, 0.5]).prepare()
    misaligned = ShortenCellState(0.1, state)
    rejected = prepared.candidate(misaligned, 0)
    rolled_back = rejected.commit()
    rollback_exact = bool(
        (rolled_back.time_ms == misaligned.time_ms)
        & jnp.all(rolled_back.values == misaligned.values)
        & ~rejected.successful
    )

    compiled_rhs = eqx.filter_jit(
        lambda configured, value: configured.rhs(0.75, value)
    )(model, state)
    batch = jnp.stack((state, state.at[0].add(0.01)))
    vectorized_rhs = jax.vmap(lambda value: model.rhs(0.75, value))(batch)
    _, rhs_tangent = jax.jvp(
        lambda value: model.rhs(0.75, value),
        (state,),
        (jnp.full_like(state, 1.0e-6),),
    )
    transformation_finite = bool(
        jnp.all(jnp.isfinite(compiled_rhs))
        & jnp.all(jnp.isfinite(vectorized_rhs))
        & jnp.all(jnp.isfinite(rhs_tangent))
    )

    checks = {
        "layout": (
            model.state_layout.count == 56
            and model.algebraic_layout.count == 71
            and model.constant_layout.count == 105
        ),
        "source_rhs": rhs_error < 2.0e-8,
        "source_currents": current_error < 2.0e-8,
        "source_trajectory": max(source_errors.values()) < 2.0e-4,
        "pulse_alignment": bool(
            jnp.all(
                ShortenPulseProtocol().current(jnp.asarray([0.0, 0.5, 50.0, 50.5]))
                == jnp.asarray([150.0, 0.0, 150.0, 0.0])
            )
        ),
        "temporal_refinement": refinement_error < 5.0e-5,
        "stiffness_evidence": stiffness_ratio > 1.0e6,
        "exact_gate": exact_gate_semigroup_error < 2.0e-6,
        "rollback": rollback_exact,
        "transformations": transformation_finite,
        "twitch": all(peak > 0.0 for peak in twitch_peaks),
        "fatigue": np.isfinite(fatigue_ratio) and fatigue_ratio > 0.0,
        "all_steps": bool(
            jnp.all(reference_trajectory.successful)
            & jnp.all(coarse.successful)
            & jnp.all(refined.successful)
            & jnp.all(fatigue.successful)
        ),
    }
    return {
        "qualified": all(checks.values()),
        "checks": checks,
        "source": {
            "revision": model.source_revision,
            "sha256": model.source_sha256,
            "license": model.source_license,
            "doi": "10.1007/s10974-007-9125-6",
            "reference": (
                "Shorten, O'Callaghan, Davidson & Soboleva, J Muscle Res "
                "Cell Motil 28 (2007) 293-313"
            ),
            "opencor_run": "626177e490c6a1959557976e",
            "opencor_simulator": "OpenCOR 2021-10-05",
        },
        "state_count_resolution": {
            "authoritative_fast_twitch_differential_equations": 56,
            "last_state": "razumova/P_C_SR",
            "last_state_index": 55,
            "opendihu_actual_original_adapter": "CellmlAdapter<56,71>",
            "opendihu_57_state_model": "new_slow_TK_2014_12_08.cellml",
        },
        "maximum_relative_rhs_error": rhs_error,
        "maximum_relative_current_error": current_error,
        "opencor_maximum_relative_trace_errors": source_errors,
        "temporal_refinement_relative_error": refinement_error,
        "gate_time_scale_ratio": stiffness_ratio,
        "exact_gate_semigroup_error": exact_gate_semigroup_error,
        "twitch_peaks_A2_uM_above_initial": twitch_peaks,
        "calcium_peaks_Ca2_uM": calcium_peaks,
        "ninth_to_first_twitch_ratio": fatigue_ratio,
        "force_owner": "Shorten razumova/A_2 biochemical tension driver (uM)",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = qualify()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not result["qualified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
