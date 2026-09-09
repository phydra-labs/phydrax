#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Circuit parity, terminal localization and fixed-grid derivative observations."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax import nonlinear, solver
from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications import battery
from phydrax.applications.battery._release_contracts import (
    CampaignCase,
    CIRCUIT_ECM_ADVANCED_METRICS,
)
from tools._battery_circuit_ecm_campaign import (
    _OUTPUTS,
    _PARAMETERS,
    _parameters,
    _SCALES,
)
from tools.battery_campaign_registry import (
    CampaignEntry,
    metric_key,
    PreparedCampaign,
    trajectory_observation,
)


_KIND = "circuit-ecm-lifecycle"
_INPUTS = {
    **_PARAMETERS,
    "fixed_step_s": 0.01,
    "event_fine_step_s": 0.005,
    "voltage_stop_v": 3.85,
    "difference_step_a": 1e-4,
}


def _experiment(times, *, native, terminal=False):
    protocol = battery.BatteryProtocolPlan(
        (
            battery.CurrentStepPlan(
                1.0, stop_guards=(battery.VoltageStopGuard("above"),) if terminal else ()
            ),
            battery.RestStepPlan(1.0),
        ),
        node_side="right",
    )
    if native:
        adapter = battery.CircuitConnectedEcmAdapter(battery.CircuitConnectedEcmPlan(1))
        guards = tuple(
            guard.guard_id for guard in adapter.native_guards(adapter.prepare())
        )
        numerical = battery.BatteryDAESolvePlan(
            solver.DAESolvePolicy(
                method=solver.BDFMethod(2),
                nonlinear_termination=nonlinear.NonlinearTermination(
                    absolute_step=0.0, relative_step=0.0
                ),
            ),
            guard_ids=guards,
        )
        profile, support = battery.CIRCUIT_ECM_CANDIDATE, battery.CIRCUIT_ECM_SUPPORT
        initial = battery.CircuitConnectedEcmInitialCondition(400.0, 300.0, relaxed=True)
    else:
        adapter = battery.ThermalEquivalentCircuitAdapter(
            battery.ThermalEquivalentCircuitPlan(1)
        )
        numerical = battery.BatteryDiffraxSolvePlan(
            relative_tolerance=1e-10, absolute_tolerance=1e-12
        )
        profile, support = battery.THERMAL_ECM_CANDIDATE, battery.THERMAL_ECM_SUPPORT
        initial = battery.ThermalEquivalentCircuitInitialCondition(
            400.0, 300.0, relaxed=True
        )
    prepared = battery.BatteryExperimentPlan(
        adapter,
        protocol,
        battery.BatteryOutputPlan(_OUTPUTS),
        numerical,
        jnp.asarray(times),
        profile,
        support,
    ).prepare()
    parameters = _parameters()

    def run(amplitude):
        values = battery.BatteryProtocolValues(
            protocol,
            jnp.reshape(amplitude, (1,)),
            jnp.asarray((3.85,)) if terminal else (),
        )
        return prepared.run(parameters, initial, values)

    return prepared, run


def prepare_campaign(sample_times_s):
    times = np.asarray(tuple(sample_times_s), dtype=float)
    expected = np.linspace(0.0, 2.0, 201)
    if times.shape != expected.shape or not np.allclose(
        times, expected, rtol=0.0, atol=1e-14
    ):
        raise ValueError(
            "Circuit lifecycle requires the declared 0.01-second [0,2] grid."
        )
    preparations, runs = zip(
        _experiment(times, native=True),
        _experiment(times, native=False),
        _experiment(times, native=True, terminal=True),
        _experiment(np.linspace(0.0, 2.0, 401), native=True, terminal=True),
        strict=True,
    )
    native, standalone, coarse, fine = runs
    amplitude = jnp.asarray(2.0)
    step = jnp.asarray(1e-4)
    weights = jnp.asarray((0.2, -0.3, 0.1, 0.4, -0.2))

    def selected(current):
        return native(current).outputs.values[-1] / jnp.asarray(_SCALES)

    def execute():
        base, minus, plus = (
            native(amplitude),
            native(amplitude - step),
            native(amplitude + step),
        )
        _, tangent = jax.jvp(selected, (amplitude,), (jnp.ones_like(amplitude),))
        _, pullback = jax.vjp(selected, amplitude)
        reverse = pullback(weights)[0]
        difference = (plus.outputs.values[-1] - minus.outputs.values[-1]) / (
            2.0 * step * jnp.asarray(_SCALES)
        )
        return (
            base,
            standalone(amplitude),
            coarse(amplitude),
            fine(amplitude),
            minus,
            plus,
            tangent,
            reverse,
            difference,
        )

    return PreparedCampaign(
        _KIND,
        canonical_fingerprint([item.preparation_id for item in preparations]),
        canonical_fingerprint(_INPUTS),
        canonical_fingerprint({"model": battery.CIRCUIT_ECM_SUPPORT.support_tuple_id}),
        execute,
        execute,
    )


def _same_topology(left, right):
    if not np.array_equal(
        left.native_solution.replay.segment_active,
        right.native_solution.replay.segment_active,
    ):
        return False
    for a, b in zip(
        left.native_solution.segments, right.native_solution.segments, strict=True
    ):
        if not all(
            np.array_equal(x, y)
            for x, y in (
                (a.step_history.valid, b.step_history.valid),
                (a.step_history.orders, b.step_history.orders),
                (
                    np.asarray(a.step_history.accepted_times)[
                        np.asarray(a.step_history.valid)
                    ],
                    np.asarray(b.step_history.accepted_times)[
                        np.asarray(b.step_history.valid)
                    ],
                ),
                (a.events.valid, b.events.valid),
                (a.events.event_indices, b.events.event_indices),
            )
        ):
            return False
    return True


def _event_failures(result):
    termination = result.termination
    time = float(termination.time_s)
    failures = int(not bool(result.successful)) + int(not bool(termination.terminated))
    failures += int(int(termination.reason_index) != 0) + int(not 0.28 < time < 0.30)
    failures += int(not bool(result.native_solution.replay.successful))
    count = 0
    for active, segment in zip(
        result.native_solution.replay.segment_active,
        result.native_solution.segments,
        strict=True,
    ):
        if not bool(active):
            continue
        event = segment.events
        count += int(event.event_count)
        mask = np.asarray(event.valid)
        failures += int(not bool(event.successful))
        failures += int(np.any(np.abs(np.asarray(event.guard_residuals)[mask]) > 1e-8))
        failures += int(
            not np.array_equal(
                np.asarray(event.states_before)[mask],
                np.asarray(event.states_after)[mask],
            )
        )
        event_times = np.asarray(event.event_times)[mask]
        failures += int(
            np.any(event_times < np.asarray(event.replay.bracket_left_times)[mask])
        )
        failures += int(
            np.any(event_times > np.asarray(event.replay.bracket_right_times)[mask])
        )
    return failures + int(count != 1)


def raw_output(spec, executed, campaign_directory):
    del spec, campaign_directory
    jax.block_until_ready(executed)
    base, standalone, coarse, fine, minus, plus, tangent, reverse, difference = executed
    weights = np.asarray((0.2, -0.3, 0.1, 0.4, -0.2))
    tangent, reverse, difference = (
        np.asarray(tangent),
        np.asarray(reverse),
        np.asarray(difference),
    )
    finite = bool(
        np.all(np.isfinite(tangent))
        and np.all(np.isfinite(reverse))
        and np.all(np.isfinite(difference))
    )
    derivative_failures = int(not finite)
    for result in (base, minus, plus):
        derivative_failures += int(not bool(result.successful))
        derivative_failures += int(
            not bool(result.native_solution.replay.derivative_valid)
        )
        derivative_failures += int(not _same_topology(base, result))
    paired = bool(base.successful) and bool(standalone.successful)
    paired = (
        paired
        and bool(np.all(base.outputs.valid))
        and bool(np.all(standalone.outputs.valid))
    )
    parity = (
        float(
            np.max(
                np.abs(
                    np.asarray(base.outputs.values)
                    - np.asarray(standalone.outputs.values)
                )
                / _SCALES
            )
        )
        if paired
        else None
    )
    projection = float(weights @ tangent) if finite else None
    values = {
        "circuit:standalone-parity": {"maximum-normalized-parity-error": parity},
        "circuit:terminal-event": {
            "event-time-refinement-error": abs(
                float(fine.termination.time_s) - float(coarse.termination.time_s)
            )
            if bool(fine.termination.terminated) and bool(coarse.termination.terminated)
            else None,
            "event-contract-failures": _event_failures(coarse) + _event_failures(fine),
        },
        "circuit:fixed-path-jvp-vjp": {
            "duality-error": abs(projection - float(reverse)) / max(1.0, abs(projection))
            if finite
            else None,
            "finite-difference-error": float(np.max(np.abs(tangent - difference)))
            if finite
            else None,
            "derivative-contract-failures": derivative_failures,
        },
    }
    metrics = {}
    for case, measured in values.items():
        for name, value in measured.items():
            available = value is not None and np.isfinite(value)
            metrics[metric_key(case, name)] = {
                "value": value if available else None,
                "unavailable_reason": None
                if available
                else "native-lifecycle-observation-unavailable",
            }
    observations = []
    for case, result in zip(
        CIRCUIT_ECM_ADVANCED_METRICS, (base, fine, base), strict=True
    ):
        observations.append(
            trajectory_observation(
                case, result.require_evidence_identity(), result.outputs
            )
        )
    return {"metrics": metrics, "observations": observations}


def campaign_entry():
    cases = tuple(
        CampaignCase(case, tuple(sorted(_INPUTS.items())), metrics)
        for case, metrics in CIRCUIT_ECM_ADVANCED_METRICS.items()
    )
    return CampaignEntry(
        _KIND,
        battery.CIRCUIT_ECM_CANDIDATE,
        battery.CIRCUIT_ECM_SUPPORT,
        cases,
        0,
        (),
        prepare_campaign,
        raw_output,
        ("metrics", "observations"),
    )
