#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..integration import WeightedSampleTarget
from ._multiple_testing import (
    adjust_p_values,
    MultipleTestingMethod,
    MultipleTestingResult,
)
from ._particle import normalize_log_weights
from ._posterior_reweighting import _flatten_target


_CALIBRATION_RANK_ADDRESS = SampleAddress(
    "uq",
    "simulation-calibration-rank",
    target="posterior-rank",
    role="diagnostic",
)
TiePolicy = Literal["interval", "randomized"]


def _identifier(value: str, role: str, /) -> str:
    identifier = str(value).strip()
    if not identifier:
        raise ValueError(f"{role} must be non-empty.")
    return identifier


def _path_leaves(tree: PyTree[Any], /) -> dict[str, Array]:
    return {
        jax.tree_util.keystr(path) or "<root>": jnp.asarray(value)
        for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]
    }


class SimulationCalibrationCase(StrictModule, NonTrainableState):
    truth: PyTree[Array]
    posterior: WeightedSampleTarget
    valid: Array
    case_id: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    status: str = eqx.field(static=True)

    def __init__(
        self,
        truth: PyTree[Any],
        posterior: WeightedSampleTarget,
        /,
        *,
        case_id: str,
        analysis_id: str,
        valid: bool = True,
        status: str = "success",
    ):
        if not isinstance(posterior, WeightedSampleTarget):
            raise TypeError("posterior must be WeightedSampleTarget.")
        truth_ = jax.tree_util.tree_map(jnp.asarray, truth)
        if not jax.tree_util.tree_leaves(truth_) or any(
            bool(jnp.any(~jnp.isfinite(value)))
            for value in jax.tree_util.tree_leaves(truth_)
        ):
            raise ValueError("Simulation-calibration truth must contain finite arrays.")
        self.truth = truth_
        self.posterior = posterior
        self.valid = jnp.asarray(valid, dtype=bool)
        self.case_id = _identifier(case_id, "calibration case ID")
        self.analysis_id = _identifier(analysis_id, "calibration analysis ID")
        self.status = _identifier(status, "calibration case status")


class SimulationCalibrationPlan(StrictModule):
    parameter_paths: tuple[str, ...] = eqx.field(static=True)
    num_bins: int = eqx.field(static=True)
    tie_policy: TiePolicy = eqx.field(static=True)
    minimum_valid_cases: int = eqx.field(static=True)
    multiple_testing_method: MultipleTestingMethod = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_paths: Sequence[str],
        /,
        *,
        num_bins: int = 10,
        tie_policy: TiePolicy = "interval",
        minimum_valid_cases: int = 20,
        multiple_testing_method: MultipleTestingMethod = "holm",
        alpha: float = 0.05,
    ):
        paths = tuple(str(value).strip() for value in parameter_paths)
        bins = int(num_bins)
        minimum = int(minimum_valid_cases)
        level = float(alpha)
        if (
            not paths
            or any(not value for value in paths)
            or len(set(paths)) != len(paths)
        ):
            raise ValueError(
                "Calibration parameter paths must be distinct and non-empty."
            )
        if bins < 2 or minimum < bins or tie_policy not in ("interval", "randomized"):
            raise ValueError(
                "Calibration bins, valid-case count, or tie policy is invalid."
            )
        if multiple_testing_method not in ("bonferroni", "holm", "benjamini-hochberg"):
            raise ValueError("Unknown calibration multiple-testing method.")
        if not 0.0 < level < 1.0:
            raise ValueError("Calibration alpha must lie strictly between zero and one.")
        self.parameter_paths = paths
        self.num_bins = bins
        self.tie_policy = tie_policy
        self.minimum_valid_cases = minimum
        self.multiple_testing_method = multiple_testing_method
        self.alpha = level
        self.plan_id = canonical_fingerprint(
            {
                "kind": "simulation-calibration-plan",
                "paths": list(paths),
                "bins": bins,
                "tie_policy": tie_policy,
                "minimum_valid_cases": minimum,
                "multiple_testing": multiple_testing_method,
                "alpha": level,
            }
        )


class SimulationCalibrationResult(StrictModule):
    ranks: Array
    lower_ranks: Array
    upper_ranks: Array
    case_valid: Array
    histogram_counts: Array
    raw_p_values: Array
    multiple_testing: MultipleTestingResult
    valid_case_count: Array
    passed: Array
    component_labels: tuple[str, ...] = eqx.field(static=True)
    case_ids: tuple[str, ...] = eqx.field(static=True)
    failed_case_ids: tuple[str, ...] = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def simulation_calibration(
    cases: Sequence[SimulationCalibrationCase],
    plan: SimulationCalibrationPlan,
    /,
    *,
    key: Array | None = None,
) -> SimulationCalibrationResult:
    items = tuple(cases)
    if not items or any(
        not isinstance(item, SimulationCalibrationCase) for item in items
    ):
        raise TypeError("cases must contain SimulationCalibrationCase values.")
    case_ids = tuple(item.case_id for item in items)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Simulation-calibration case IDs must be unique.")
    analysis_id = items[0].analysis_id
    if any(item.analysis_id != analysis_id for item in items[1:]):
        raise ValueError("Simulation-calibration cases must share one analysis ID.")
    if not isinstance(plan, SimulationCalibrationPlan):
        raise TypeError("plan must be SimulationCalibrationPlan.")
    if plan.tie_policy == "randomized" and key is None:
        raise ValueError("Randomized calibration ranks require an explicit key.")
    first_truth = _path_leaves(items[0].truth)
    if any(path not in first_truth for path in plan.parameter_paths):
        raise ValueError("Calibration path is absent from the truth PyTree.")
    labels = []
    for path in plan.parameter_paths:
        shape = tuple(first_truth[path].shape)
        if not shape:
            labels.append(path)
        else:
            labels.extend(f"{path}{index}" for index in np.ndindex(shape))
    component_labels = tuple(labels)
    rank_rows = []
    lower_rows = []
    upper_rows = []
    validity = []
    for case_index, case in enumerate(items):
        truth = _path_leaves(case.truth)
        samples, log_weights, active, _ = _flatten_target(case.posterior)
        sample_paths = _path_leaves(samples)
        structure_valid = all(
            path in truth
            and path in sample_paths
            and truth[path].shape == first_truth[path].shape
            and sample_paths[path].shape[1:] == truth[path].shape
            for path in plan.parameter_paths
        )
        sample_values_valid = structure_valid and all(
            bool(
                jnp.all(
                    ~active[:, None]
                    | jnp.isfinite(sample_paths[path].reshape((active.size, -1)))
                )
            )
            for path in plan.parameter_paths
        )
        case_valid = bool(case.valid) and sample_values_valid
        if not structure_valid:
            lower = jnp.zeros((len(component_labels),))
            upper = jnp.zeros_like(lower)
        else:
            normalized, _, weights_valid = normalize_log_weights(
                jnp.where(active, log_weights, -jnp.inf)
            )
            case_valid = case_valid and bool(weights_valid)
            weights = jnp.exp(normalized)
            lower_values = []
            upper_values = []
            for path in plan.parameter_paths:
                sample_components = sample_paths[path].reshape((weights.size, -1))
                truth_components = truth[path].reshape((-1,))
                lower_values.extend(
                    jnp.sum(
                        weights * (sample_components[:, index] < truth_components[index])
                    )
                    for index in range(int(truth_components.size))
                )
                upper_values.extend(
                    jnp.sum(
                        weights * (sample_components[:, index] <= truth_components[index])
                    )
                    for index in range(int(truth_components.size))
                )
            lower = jnp.stack(tuple(lower_values))
            upper = jnp.stack(tuple(upper_values))
        if plan.tie_policy == "randomized" and key is not None:
            random = jr.uniform(
                derive_key(key, _CALIBRATION_RANK_ADDRESS, case_index), lower.shape
            )
            rank = lower + random * (upper - lower)
        else:
            rank = 0.5 * (lower + upper)
        rank_rows.append(jnp.where(case_valid, rank, 0.0))
        lower_rows.append(jnp.where(case_valid, lower, 0.0))
        upper_rows.append(jnp.where(case_valid, upper, 0.0))
        validity.append(case_valid)
    ranks = jnp.stack(tuple(rank_rows))
    lower_ranks = jnp.stack(tuple(lower_rows))
    upper_ranks = jnp.stack(tuple(upper_rows))
    case_valid = jnp.asarray(validity, dtype=bool)
    valid_count = jnp.sum(case_valid)
    edges = jnp.linspace(0.0, 1.0, plan.num_bins + 1)
    histograms = []
    p_values = []
    for component in range(ranks.shape[1]):
        bins = jnp.clip(
            jnp.searchsorted(edges, ranks[:, component], side="right") - 1,
            0,
            plan.num_bins - 1,
        )
        counts = (
            jnp.zeros((plan.num_bins,), dtype=jnp.int32)
            .at[bins]
            .add(case_valid.astype(jnp.int32))
        )
        expected = valid_count / plan.num_bins
        statistic = jnp.sum(
            (counts - expected) ** 2 / jnp.where(expected > 0.0, expected, 1.0)
        )
        p_value = jsp.special.gammaincc(0.5 * (plan.num_bins - 1), 0.5 * statistic)
        histograms.append(counts)
        p_values.append(p_value)
    histogram_counts = jnp.stack(tuple(histograms))
    raw_p_values = jnp.stack(tuple(p_values))
    testing = adjust_p_values(
        raw_p_values,
        method=plan.multiple_testing_method,
        alpha=plan.alpha,
    )
    enough = valid_count >= plan.minimum_valid_cases
    passed = enough & jnp.all(~testing.rejected)
    failed_ids = tuple(
        case.case_id for case, valid in zip(items, validity, strict=True) if not valid
    )
    return SimulationCalibrationResult(
        ranks,
        lower_ranks,
        upper_ranks,
        case_valid,
        histogram_counts,
        raw_p_values,
        testing,
        valid_count,
        passed,
        component_labels,
        case_ids,
        failed_ids,
        analysis_id,
        plan.plan_id,
    )


__all__ = [
    "SimulationCalibrationCase",
    "SimulationCalibrationPlan",
    "SimulationCalibrationResult",
    "TiePolicy",
    "simulation_calibration",
]
