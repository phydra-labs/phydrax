# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Two-column temperature-anomaly twin with native ETKF, forecast and restart.

Run from the repository: PYTHONPATH=. python examples/geophysical_assimilation.py
The physical model is two equal-area heat reservoirs with Newtonian radiative
relaxation and conservative inter-column exchange. Its exact linear propagator
is used; the ensemble adds independent unresolved temperature forcing. This is
an explicitly labelled linear twin, not a global atmospheric validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.applications.geophysics._ensembles import (
    geophysical_analysis_increments,
    GeophysicalAnalysisInventory,
    GeophysicalEnsembleAxis,
    GeophysicalEnsembleLineage,
    prepare_geophysical_assimilation,
)
from phydrax.applications.geophysics._observations import (
    prepare_geophysical_observations,
    prepare_tensor_observation_operator,
)
from phydrax.applications.geophysics._quantities import GeophysicalQuantity
from phydrax.applications.geophysics._time import GeophysicalTimeSpec
from phydrax.discretization import DiscreteFieldSpace, TensorDofLayout
from phydrax.dynamics import StateLayout
from phydrax.linalg import ArraySpace
from phydrax.stochastic import GaussianStatePrior, StateSpaceStepContext
from phydrax.units import JOULE, KELVIN
from phydrax.uq import (
    ensemble_filter_step,
    ensemble_kalman_smoother,
    ensemble_transform_kalman_filter,
    initialize_ensemble_filter,
    read_ensemble_filter_checkpoint,
    write_ensemble_filter_checkpoint,
)


def temperature_propagator(state, duration):
    """Exact reservoir solution; radiative rate 0.03/day, exchange 0.12/day."""
    common = 0.5 * (state[0] + state[1]) * jnp.exp(-0.03 * duration)
    contrast = 0.5 * (state[0] - state[1]) * jnp.exp(-0.27 * duration)
    return jnp.stack((common + contrast, common - contrast))


def temperature_transition(key, state, t0, t1, context):
    del context
    duration = t1 - t0
    # Unresolved independent heat forcing, temperature scale 0.025 K/sqrt(day).
    noise = (
        0.025
        * jnp.sqrt(duration)
        * jax.random.normal(key, state.shape, dtype=state.dtype)
    )
    return temperature_propagator(state, duration) + noise


def make_twin_problem(*, steps=8):
    quantity = GeophysicalQuantity(
        "temperature_anomaly",
        "temperature_anomaly",
        KELVIN,
        reference_configuration="fixed-climatology",
    )
    clock = GeophysicalTimeSpec(unit="d")
    space = DiscreteFieldSpace(
        "temperature_anomaly",
        "two-equal-area-columns",
        TensorDofLayout(("column",), (2,)),
        ArraySpace((2,), dtype=jnp.float64),
        representation="point_value",
    )
    layout = StateLayout(
        (2,),
        axes=("column",),
        component_names=("west_temperature_anomaly", "east_temperature_anomaly"),
    )
    operator = prepare_tensor_observation_operator(
        space,
        {"column": [0.0, 1.0]},
        {"column": [0.0, 1.0]},
        quantity,
        source_support_id=space.support_id,
        time=clock,
        kind="grid",
    )
    times = jnp.arange(steps, dtype=jnp.float64)
    initial_truth = jnp.asarray([2.0, 0.5])
    truth = jax.vmap(lambda time: temperature_propagator(initial_truth, time))(times)
    noise = 0.05 * jax.random.normal(jax.random.key(81), truth.shape, dtype=truth.dtype)
    available = np.ones(truth.shape, dtype=bool)
    available[1::2, 1] = False
    observed = np.asarray(truth + noise).copy()
    observed[~available] = np.nan
    observations = prepare_geophysical_observations(
        operator,
        times,
        observed,
        0.05,
        quantity=quantity,
        time=clock,
        target_support_id=operator.transfer.target.support_id,
        availability=available,
        representativeness_std=0.03,
    )
    coordinates = {
        "initial_condition": "biased-climatological-prior",
        "scenario": "unforced-relaxation",
        "parameter": "radiation-003-exchange-012-per-day",
        "structural": "two-equal-columns",
        "internal_stochastic": "unresolved-heat-0025-K-sqrt-day",
    }
    lineage = GeophysicalEnsembleLineage(
        tuple(
            GeophysicalEnsembleAxis(kind, (label,)) for kind, label in coordinates.items()
        ),
        coordinates,
    )
    problem = prepare_geophysical_assimilation(
        layout,
        observations,
        source_field=space,
        prior=GaussianStatePrior(
            jnp.asarray([-1.0, -1.0]),
            jnp.eye(2) * 1.5,
            state_shape=(2,),
            prior_id="cold-biased-reservoir-prior",
        ),
        transition=temperature_transition,
        model_id="radiative-exchange-two-column",
        approximation_id="exact-linear-flow-plus-independent-heat-forcing",
        model_time=clock,
        initial_time=0.0,
        lineage=lineage,
    )
    # Each 1e12 m2 column has 4e8 J/(m2 K) heat capacity.
    inventory = GeophysicalAnalysisInventory(
        "heat_anomaly",
        GeophysicalQuantity(
            "heat_anomaly", "energy", JOULE, reference_configuration="fixed-climatology"
        ),
        jnp.full((2,), 4.0e20),
        layout,
    )
    return problem, truth, inventory, lineage


def run_twin(*, ensemble_size=24):
    problem, truth, inventory, lineage = make_twin_problem()
    root = jax.random.key(17)
    result = ensemble_transform_kalman_filter(
        root, problem, ensemble_size=ensemble_size, raise_on_failure=True
    )
    smoother = ensemble_kalman_smoother(result)
    budget = geophysical_analysis_increments(result, (inventory,))[inventory.name]
    initialized = initialize_ensemble_filter(root, problem, ensemble_size=ensemble_size)
    open_mean = jnp.mean(initialized.ensemble, axis=0)
    open_loop = jax.vmap(lambda time: temperature_propagator(open_mean, time))(
        problem.observations.times
    )
    analysis_mean = jnp.mean(result.analysis_ensembles, axis=1)
    # A genuine post-analysis two-day numerical forecast, not re-assimilated truth.
    end = problem.observations.times[-1]
    future_ensemble = jnp.stack(
        [
            temperature_transition(
                lineage.key(root, "internal_stochastic", step=8, member=member),
                state,
                end,
                end + 2.0,
                StateSpaceStepContext.empty(),
            )
            for member, state in enumerate(result.final_state.ensemble)
        ]
    )
    future_truth = temperature_propagator(truth[-1], 2.0)
    streamed = initialized
    for _ in range(3):
        streamed, _ = ensemble_filter_step(problem, streamed)
    with TemporaryDirectory() as directory:
        path = Path(directory) / "assimilation-restart.npz"
        write_ensemble_filter_checkpoint(path, problem, streamed)
        restarted = read_ensemble_filter_checkpoint(
            path, problem, ensemble_size=ensemble_size
        )
        for _ in range(problem.observations.num_steps - restarted.step_index):
            restarted, _ = ensemble_filter_step(problem, restarted)
    mean_increment = np.asarray(budget.mean_increment)
    expected_increment = (
        np.sum(
            np.asarray(result.analysis_ensembles - result.forecast_ensembles), axis=-1
        ).mean(axis=-1)
        * 4.0e20
    )
    return {
        "analysis_rmse_K": float(jnp.sqrt(jnp.mean((analysis_mean - truth) ** 2))),
        "unassimilated_rmse_K": float(jnp.sqrt(jnp.mean((open_loop - truth) ** 2))),
        "smoothed_rmse_K": float(
            jnp.sqrt(jnp.mean((jnp.mean(smoother.ensembles, axis=1) - truth) ** 2))
        ),
        "two_day_forecast_rmse_K": float(
            jnp.sqrt(jnp.mean((jnp.mean(future_ensemble, axis=0) - future_truth) ** 2))
        ),
        "initial_forecast_spread_K2": float(
            jnp.sum(jnp.var(result.forecast_ensembles[0], axis=0, ddof=1))
        ),
        "initial_analysis_spread_K2": float(
            jnp.sum(jnp.var(result.analysis_ensembles[0], axis=0, ddof=1))
        ),
        "observed_counts": np.asarray(result.observed_counts).tolist(),
        "analysis_heat_impulses_J": mean_increment.tolist(),
        "budget_relative_residual": float(
            np.max(np.abs(mean_increment - expected_increment))
            / max(1.0, np.max(np.abs(expected_increment)))
        ),
        "restart_bitwise_equal": bool(
            np.array_equal(
                np.asarray(restarted.ensemble), np.asarray(result.final_state.ensemble)
            )
        ),
        "restart_key_equal": bool(
            np.array_equal(
                np.asarray(jax.random.key_data(restarted.root_key)),
                np.asarray(jax.random.key_data(root)),
            )
        ),
        "lineage_id": lineage.lineage_id,
        "smoother_valid": bool(jnp.all(smoother.valid)),
    }


if __name__ == "__main__":
    with jax.enable_x64(True):
        print(json.dumps(run_twin(), indent=2))
