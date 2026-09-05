# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ....dynamics import TrajectoryData
from ....dynamics.analysis import validate_markov_models
from ....dynamics.identification import (
    fit_markov_state_model,
    fit_tica,
    fit_vac,
    fit_vamp,
)
from ....stochastic.path_sampling import (
    FirstPassagePathEnsemble,
    fit_committor,
    StateRegionPlan,
)
from ....units import conversion_factor, SECOND, UnitDefinition
from .._construct import _identifier


@dataclass(frozen=True, slots=True)
class ProteinBasinDefinitions:
    """Independent native state-region definitions, not clusters fitted to a trace."""

    names: tuple[str, ...]
    regions: tuple[StateRegionPlan, ...]
    source_id: str

    def __post_init__(self):
        _identifier(self.source_id, "independent basin source")
        if (
            len(self.names) < 2
            or len(self.names) != len(self.regions)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError(
                "At least two uniquely named, independently defined basins are required."
            )
        if any(not isinstance(region, StateRegionPlan) for region in self.regions):
            raise TypeError("Basin predicates must use native StateRegionPlan.")

    def assign(self, features):
        memberships = jnp.stack(
            tuple(region.contains(features) for region in self.regions), axis=-1
        )
        counts = jnp.sum(memberships, axis=-1)
        if bool(jnp.any(counts > 1)):
            raise ValueError("Basin predicates overlap on observed support.")
        return jnp.where(counts == 1, jnp.argmax(memberships, axis=-1), -1)

    def first_passage_ensemble(self, source: int, target: int):
        if (
            source == target
            or not 0 <= source < len(self.regions)
            or not 0 <= target < len(self.regions)
        ):
            raise ValueError("Distinct in-range source/target basins are required.")
        return FirstPassagePathEnsemble(self.regions[source], self.regions[target])


@dataclass(frozen=True, slots=True)
class ProteinKineticWorkflow:
    """Native kinetic estimators with explicit physical-time and bias admission.

    ``source_kind`` refuses optimizer traces, static predictions and MC sweeps.
    SI-convertible time does not by itself calibrate a coarse dynamics model:
    ``time_calibration_id`` must identify that additional physical evidence.
    Biased configurations cannot recover unbiased kinetics, even with weights.
    """

    data: TrajectoryData
    time_unit: UnitDefinition
    conditions_id: str
    time_calibration_id: str
    source_kind: str = "physical-dynamics"
    configuration_bias_id: str | None = None

    def __post_init__(self):
        if not isinstance(self.data, TrajectoryData):
            raise TypeError(
                "Kinetic consumers require native reset-aware TrajectoryData."
            )
        if self.source_kind != "physical-dynamics":
            raise ValueError(
                "Only declared physical dynamics can support physical kinetic analysis; "
                "optimizer traces are not trajectories."
            )
        _identifier(self.conditions_id, "physical conditions")
        _identifier(self.time_calibration_id, "physical time calibration")
        conversion_factor(self.time_unit, SECOND)
        if (
            self.data.coordinate_id != self.time_unit.unit_id
            or self.data.coordinate_kind != "continuous"
        ):
            raise ValueError(
                "Trajectory coordinate identity must match its physical time unit."
            )
        if self.configuration_bias_id is not None:
            raise ValueError(
                "Configuration bias/weights do not define unbiased path kinetics; "
                "a separately derived native path measure is required."
            )

    def require_uniform_lag(self, lag=1, *, relative_tolerance=1e-8):
        transitions = self.data.transitions(lag)
        valid = np.asarray(transitions.valid)
        deltas = np.asarray(
            transitions.target_coordinates - transitions.source_coordinates
        )[valid]
        if deltas.size == 0 or not np.all(np.isfinite(deltas)) or np.any(deltas <= 0):
            raise ValueError(
                "No positive valid physical lag pairs remain after reset/missing-sample exclusion."
            )
        if not np.isfinite(relative_tolerance) or relative_tolerance < 0:
            raise ValueError("Lag tolerance must be finite and nonnegative.")
        if np.max(deltas) - np.min(deltas) > relative_tolerance * float(np.mean(deltas)):
            raise ValueError(
                "Irregular physical lags cannot support a single physical timescale claim."
            )
        return float(np.mean(deltas))

    def vamp(self, library, *, lag=1, **options):
        self.require_uniform_lag(lag)
        return fit_vamp(self.data, library, lag=lag, **options)

    def vac(self, library, *, lag=1, **options):
        self.require_uniform_lag(lag)
        return fit_vac(self.data, library, lag=lag, **options)

    def tica(self, *, lag=1, **options):
        self.require_uniform_lag(lag)
        return fit_tica(self.data, lag=lag, **options)

    def markov(
        self, basin_features, basins: ProteinBasinDefinitions, *, lag=1, **options
    ):
        self.require_uniform_lag(lag)
        assignments = basins.assign(basin_features)
        return fit_markov_state_model(
            self.data, assignments, state_count=len(basins.names), lag=lag, **options
        )

    def chapman_kolmogorov(
        self, basin_features, basins, *, lag=1, multiplier=2, **options
    ):
        short = self.markov(basin_features, basins, lag=lag, **options)
        long = self.markov(basin_features, basins, lag=lag * multiplier, **options)
        return validate_markov_models(short, long, multiplier)

    def committor(
        self, plan, initial_features, outcomes, *, shooting_source_id, weights=None
    ):
        """Fit native q from independent physical shooting outcomes.

        Outcomes must be observed A/B first hits, not inferred structure scores;
        censored paths require omission with separately retained censoring evidence.
        The outcome-generating path ensemble is retained as ``shooting_source_id``.
        """
        _identifier(shooting_source_id, "physical shooting source")
        return fit_committor(plan, initial_features, outcomes, weights=weights)


__all__ = ["ProteinBasinDefinitions", "ProteinKineticWorkflow"]
