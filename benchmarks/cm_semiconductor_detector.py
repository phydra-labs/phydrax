#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Separated detector field-solve and prescribed-response benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    logical_array_bytes,
    measure_host,
    measure_repeated,
)

from phydrax.applications.semiconductor._detector import (
    DetectorBiasElectrostaticPlan,
    DetectorElectrode,
    DetectorResourcePolicy,
    DetectorWeightingFieldPlan,
    SemiconductorDetectorPlan,
)
from phydrax.applications.semiconductor._detector_response import (
    DetectorTrajectoryRoute,
    PrescribedShockleyRamoPlan,
)
from phydrax.applications.semiconductor._quantities import SQUARE_METER
from phydrax.discretization import (
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.dynamics import StateLayout, TrajectoryData


jax.config.update("jax_enable_x64", True)
_EPSILON = 11.7 * 8.8541878128e-12


def _case(node_intervals: int, trajectory_samples: int):
    grid = TensorGridPlan(
        (UniformCellAxisSpec(node_intervals, periodic=False),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0e-3]]))
    bridge = StructuredCochainBridge(grid)
    x = np.asarray(bridge.cochain.coordinates[0])[:, 0]
    electrodes = (
        DetectorElectrode("anode", np.isclose(x, x[0])),
        DetectorElectrode("cathode", np.isclose(x, x[-1])),
    )
    policy = DetectorResourcePolicy(
        maximum_nodes=node_intervals + 1,
        maximum_edges=node_intervals,
        maximum_electrodes=2,
        maximum_linear_iterations=max(64, 4 * node_intervals),
        maximum_trajectory_cases=1,
        maximum_trajectory_samples=trajectory_samples,
        maximum_interpolation_routes=2 * trajectory_samples,
    )
    detector = SemiconductorDetectorPlan(
        bridge,
        electrodes,
        policy,
        permittivity=_EPSILON,
        node_transverse_measure=1.0e-4,
        edge_transverse_measure=1.0e-4,
        transverse_unit=SQUARE_METER,
    )
    bias = DetectorBiasElectrostaticPlan(detector, jnp.asarray([100.0, 0.0]))
    weighting = DetectorWeightingFieldPlan(detector)
    layout = StateLayout((1,), component_names=("x",))
    trajectory = TrajectoryData(
        jnp.linspace(0.0, 1.0e-6, trajectory_samples),
        jnp.linspace(1.0e-6, 1.0e-3 - 1.0e-6, trajectory_samples)[:, None],
        state_layout=layout,
        source_id=f"detector-benchmark-{node_intervals}-{trajectory_samples}",
    )
    return detector, bias, weighting, trajectory, layout


def benchmark_case(
    node_intervals: int,
    trajectory_samples: int,
    repeats: int,
) -> dict[str, object]:
    prepared, preparation_seconds = measure_host(
        lambda: _case(node_intervals, trajectory_samples)
    )
    detector, bias_plan, weighting_plan, trajectory, layout = prepared
    zero = jnp.zeros((detector.node_count,))
    bias, bias_timing = measure_repeated(
        lambda: bias_plan.solve(zero), warmup=1, repeats=repeats
    )
    weighting, weighting_timing = measure_repeated(
        weighting_plan.solve, warmup=1, repeats=repeats
    )
    response_plan = PrescribedShockleyRamoPlan(
        weighting_plan,
        weighting,
        DetectorTrajectoryRoute(layout, (0,)),
    )
    response, response_timing = measure_repeated(
        lambda: response_plan.evaluate(trajectory, -1.602176634e-19),
        warmup=1,
        repeats=repeats,
    )
    return {
        "node_intervals": node_intervals,
        "nodes": detector.node_count,
        "edges": detector.edge_count,
        "electrodes": detector.electrode_count,
        "trajectory_samples": trajectory_samples,
        "interpolation_routes": response.resources.interpolation_route_count,
        "preparation_seconds": preparation_seconds,
        "bias_solve": bias_timing.to_milliseconds_dict(),
        "weighting_solve": weighting_timing.to_milliseconds_dict(),
        "response_evaluation": response_timing.to_milliseconds_dict(),
        "linear_iterations": {
            "bias": int(bias.electrostatic.linear.diagnostics.iterations),
            "weighting": [
                int(value.linear.diagnostics.iterations) for value in weighting.solves
            ],
        },
        "matvec_count": {
            "bias": int(bias.electrostatic.linear.diagnostics.matvec_count),
            "weighting": [
                int(value.linear.diagnostics.matvec_count) for value in weighting.solves
            ],
        },
        "scientific_residuals": {
            "bias_poisson": float(bias.electrostatic.residual_norm),
            "bias_charge_closure": float(bias.charge_closure_defect),
            "weighting_partition": float(weighting.evidence.partition_of_unity_defect),
            "capacitance_reciprocity": float(
                weighting.evidence.capacitance_reciprocity_defect
            ),
            "response_endpoint_closure": float(
                jnp.max(response.current_integral_closure_defect)
            ),
        },
        "successful": {
            "bias": bool(bias.successful),
            "weighting": bool(weighting.evidence.certified),
            "response": bool(response.successful),
        },
        "logical_array_bytes": {
            "detector_plan": logical_array_bytes(detector),
            "bias_result": logical_array_bytes(bias),
            "weighting_result": logical_array_bytes(weighting),
            "trajectory": logical_array_bytes(trajectory),
            "response_result": logical_array_bytes(response),
        },
        "measured_peak_host_bytes": None,
        "measured_peak_device_bytes": None,
        "peak_measurement_note": (
            "This harness reports logical payload bytes separately; no allocator peak "
            "API is used, so host and device peaks are explicitly unavailable."
        ),
        "plan_ids": {
            "detector": detector.plan_id,
            "resource_policy": detector.resources.policy_id,
            "bias": bias_plan.plan_id,
            "weighting": weighting_plan.plan_id,
            "response": response_plan.plan_id,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.repeats < 1:
        raise ValueError("repeats must be positive.")
    payload = {
        "environment": capture_environment().to_dict(),
        "records": [
            benchmark_case(nodes, samples, arguments.repeats)
            for nodes, samples in ((16, 16), (64, 128), (256, 512))
        ],
        "qualification_claim": False,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
