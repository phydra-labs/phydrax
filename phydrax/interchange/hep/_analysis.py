#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from ...applications.collider_analysis import (
        BinnedLikelihoodPlan,
        HistogramPlan,
        WeightedHistogram,
    )


def binned_likelihood_to_pyhf_workspace(
    plan: BinnedLikelihoodPlan,
    observations,
    /,
    *,
    channel_name: str = "channel",
    sample_name: str = "prediction",
    measurement_name: str = "measurement",
    parameter_of_interest: str | None = None,
) -> dict[str, object]:
    """Lower the admitted linear-nuisance subset to an external pyhf workspace."""
    from ...applications.collider_analysis import BinnedLikelihoodPlan

    if not isinstance(plan, BinnedLikelihoodPlan):
        raise TypeError("plan must be BinnedLikelihoodPlan.")
    observed = np.asarray(observations, dtype=float)
    nominal = np.asarray(plan.nominal_expectation)
    if (
        observed.shape != nominal.shape
        or np.any(~np.isfinite(observed))
        or np.any(observed < 0.0)
    ):
        raise ValueError(
            "observations must be finite nonnegative values aligned with bins."
        )
    channel = str(channel_name).strip()
    sample = str(sample_name).strip()
    measurement = str(measurement_name).strip()
    if not channel or not sample or not measurement:
        raise ValueError("pyhf channel, sample, and measurement names are required.")
    modifiers = []
    for index, name in enumerate(plan.nuisance_names):
        effect = np.asarray(plan.nuisance_effects[index]) * float(
            plan.constraint_standard_deviations[index]
        )
        high = nominal + effect
        low = nominal - effect
        if np.any(high < 0.0) or np.any(low < 0.0):
            raise ValueError("The pyhf histosys lowering would create a negative bin.")
        modifiers.append(
            {
                "name": name,
                "type": "histosys",
                "data": {"hi_data": high.tolist(), "lo_data": low.tolist()},
            }
        )
    parameters = []
    poi = "" if parameter_of_interest is None else str(parameter_of_interest).strip()
    if poi and poi not in plan.nuisance_names:
        raise ValueError("parameter_of_interest must name one admitted nuisance.")
    return {
        "channels": [
            {
                "name": channel,
                "samples": [
                    {"name": sample, "data": nominal.tolist(), "modifiers": modifiers}
                ],
            }
        ],
        "observations": [{"name": channel, "data": observed.tolist()}],
        "measurements": [
            {
                "name": measurement,
                "config": {"poi": poi, "parameters": parameters},
            }
        ],
        "version": "1.0.0",
    }


def pyhf_workspace_json(workspace: Mapping[str, object], /) -> str:
    """Serialize an already constructed external pyhf workspace deterministically."""
    if not isinstance(workspace, Mapping):
        raise TypeError("workspace must be a mapping.")
    return json.dumps(
        dict(workspace), allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def weighted_histogram_to_hepdata(
    plan: HistogramPlan,
    histogram: WeightedHistogram,
    /,
    *,
    dependent_name: str,
    qualifiers: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Map one weighted histogram to HEPData-style independent/dependent variables."""
    from ...applications.collider_analysis import HistogramPlan, WeightedHistogram

    if not isinstance(plan, HistogramPlan) or not isinstance(
        histogram, WeightedHistogram
    ):
        raise TypeError("plan and histogram must use collider-analysis types.")
    if plan.plan_id != histogram.plan_id or not bool(histogram.finite):
        raise ValueError("Only a finite histogram from the exact plan can be exported.")
    name = str(dependent_name).strip()
    if not name:
        raise ValueError("dependent_name must be non-empty.")
    qualifier_values = []
    for key, value in sorted((qualifiers or {}).items()):
        key_ = str(key).strip()
        value_ = str(value).strip()
        if not key_ or not value_:
            raise ValueError("HEPData qualifiers must have non-empty names and values.")
        qualifier_values.append({"name": key_, "value": value_})
    edges = np.asarray(plan.edges)
    sum_weights = np.asarray(histogram.sum_weights)
    uncertainties = np.sqrt(np.asarray(histogram.sum_squared_weights))
    return {
        "independent_variables": [
            {
                "header": {"name": plan.observable_id, "units": plan.unit_id},
                "values": [
                    {"low": float(edges[index]), "high": float(edges[index + 1])}
                    for index in range(plan.bin_count)
                ],
            }
        ],
        "dependent_variables": [
            {
                "header": {"name": name},
                "qualifiers": qualifier_values,
                "values": [
                    {
                        "value": float(sum_weights[index]),
                        "errors": [
                            {"label": "MC stat", "symerror": float(uncertainties[index])}
                        ],
                    }
                    for index in range(plan.bin_count)
                ],
            }
        ],
        "underflow": float(histogram.underflow_sum_weights),
        "overflow": float(histogram.overflow_sum_weights),
    }


__all__ = [
    "binned_likelihood_to_pyhf_workspace",
    "pyhf_workspace_json",
    "weighted_histogram_to_hepdata",
]
