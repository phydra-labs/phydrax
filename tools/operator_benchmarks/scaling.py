from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .runner import evaluate_operator, parameter_count
from .scenarios import OperatorBenchmarkEvaluation


@dataclass(frozen=True)
class OperatorScalingPoint:
    sample_shape: tuple[int, ...]
    sample_count: int
    compile_seconds: float
    inference_seconds: float
    peak_memory_bytes: int | None


@dataclass(frozen=True)
class OperatorScalingProfile:
    architecture: str
    parameter_count: int
    points: tuple[OperatorScalingPoint, ...]
    inference_exponent: float | None
    memory_exponent: float | None

    def to_dict(self):
        return asdict(self)


def _scaling_exponent(x, y) -> float | None:
    x_array = np.asarray(x, dtype=float)
    y_array = np.asarray(y, dtype=float)
    valid = (x_array > 0.0) & (y_array > 0.0)
    if int(np.sum(valid)) < 2:
        return None
    return float(np.polyfit(np.log(x_array[valid]), np.log(y_array[valid]), 1)[0])


def profile_resolution_scaling(
    model,
    evaluations: tuple[OperatorBenchmarkEvaluation, ...],
    /,
    *,
    architecture: str,
    repeats: int = 10,
) -> OperatorScalingProfile:
    """Profile one parameterization over increasing query resolutions."""
    if len(evaluations) < 2:
        raise ValueError("Scaling profiles require at least two resolutions.")
    points = []
    for evaluation in evaluations:
        result = evaluate_operator(model, evaluation, repeats=repeats)
        sample_shape = evaluation.batch.require_single_query().sample_shape
        points.append(
            OperatorScalingPoint(
                sample_shape=sample_shape,
                sample_count=int(np.prod(sample_shape)),
                compile_seconds=result.compile_seconds,
                inference_seconds=result.inference_seconds,
                peak_memory_bytes=result.peak_memory_bytes,
            )
        )
    memory_counts = [
        point.sample_count for point in points if point.peak_memory_bytes is not None
    ]
    memory_values = [
        point.peak_memory_bytes for point in points if point.peak_memory_bytes is not None
    ]
    return OperatorScalingProfile(
        architecture=str(architecture),
        parameter_count=parameter_count(model),
        points=tuple(points),
        inference_exponent=_scaling_exponent(
            [point.sample_count for point in points],
            [point.inference_seconds for point in points],
        ),
        memory_exponent=_scaling_exponent(memory_counts, memory_values),
    )


def assert_resolution_independent_parameters(
    models: tuple[object, ...],
    /,
) -> None:
    """Require identical parameter counts across resolution-specific constructions."""
    counts = tuple(parameter_count(model) for model in models)
    if len(set(counts)) != 1:
        raise AssertionError(f"Parameter counts vary with resolution: {counts}.")


__all__ = [
    "OperatorScalingPoint",
    "OperatorScalingProfile",
    "assert_resolution_independent_parameters",
    "profile_resolution_scaling",
]
