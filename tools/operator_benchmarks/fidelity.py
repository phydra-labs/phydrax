#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import time
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

import phydrax as phx


class AnalyticScalingOperator(phx.nn.operator.AbstractOperatorModel):
    """Small deterministic operator used to benchmark fidelity composition overhead."""

    operator_architecture = "analytic-scaling-benchmark"

    scale: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale: float, /):
        self.scale = jnp.asarray(scale)
        self.in_size = 1
        self.out_size = 1

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        values = batch.input("state").values
        if values is None:
            raise ValueError("AnalyticScalingOperator requires sampled input values.")
        return self.scale * values

    def __call__(self, batch, /, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


@dataclass(frozen=True)
class FidelityCorrectionBenchmarkResult:
    target_rmse: float
    cold_seconds: float
    execution_seconds: float
    case_count: int
    resolution: int

    @property
    def passed(self) -> bool:
        return self.target_rmse <= 1e-12


def run_fidelity_correction_benchmark(
    *,
    case_count: int = 32,
    resolution: int = 64,
) -> FidelityCorrectionBenchmarkResult:
    """Benchmark low-model plus correction composition on a field-valued batch."""

    cases = int(case_count)
    points = int(resolution)
    if cases <= 0 or points <= 1:
        raise ValueError("case_count must be positive and resolution must exceed one.")
    axis = phx.nn.operator.OperatorAxis("x", jnp.linspace(0.0, 1.0, points))
    offsets = jnp.linspace(-0.2, 0.2, cases)[:, None]
    values = jnp.sin(2.0 * jnp.pi * axis.nodes)[None, :] + offsets
    dataset = phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"output": values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )
    low = phx.fidelity.FidelityLevelSpec(
        "coarse",
        problem_id="periodic-field",
        observable_id="state",
        model_id="coarse-operator",
        approximation_id="coarse",
        observable_contract_id="target-grid-field",
    )
    high = phx.fidelity.FidelityLevelSpec(
        "fine",
        problem_id="periodic-field",
        observable_id="state",
        model_id="fine-operator",
        approximation_id="fine",
        observable_contract_id="target-grid-field",
    )
    path = phx.fidelity.FidelityHierarchy(
        (low, high),
        (phx.fidelity.FidelityRelation("coarse", "fine"),),
        target_level_id="fine",
    ).linear_path()
    model = phx.nn.operator.architectures.FidelityCorrectionOperator(
        AnalyticScalingOperator(0.8),
        AnalyticScalingOperator(0.2),
        path,
    )
    evaluate = eqx.filter_jit(
        lambda candidate, batch: candidate.__call_operator_batch__(batch)
    )
    started = time.perf_counter()
    prediction = evaluate(model, dataset.batch)
    jax.block_until_ready(prediction)
    cold = time.perf_counter() - started
    started = time.perf_counter()
    prediction = evaluate(model, dataset.batch)
    jax.block_until_ready(prediction)
    execution = time.perf_counter() - started
    rmse = float(jnp.sqrt(jnp.mean((prediction - values) ** 2)))
    return FidelityCorrectionBenchmarkResult(
        target_rmse=rmse,
        cold_seconds=cold,
        execution_seconds=execution,
        case_count=cases,
        resolution=points,
    )


__all__ = [
    "AnalyticScalingOperator",
    "FidelityCorrectionBenchmarkResult",
    "run_fidelity_correction_benchmark",
]
