#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Normalized stochastic thermodynamics on canonical discrete paths."""

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._analysis import autocorrelation_uncertainty, CorrelatedUncertainty
from ._core import path_trajectory_id, PathBuffer


class DiscretePathThermodynamicsPlan(StrictModule, NonTrainableState):
    """Sign, normalization, UQ, and bounded-resource contract for path results.

    Heat is positive into the system and the first law is ΔE = W + Q. The
    medium entropy is therefore -βQ, in dimensionless entropy units.
    """

    inverse_temperature: float = eqx.field(static=True)
    maximum_paths: int = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    autocorrelation_lag: int = eqx.field(static=True)
    first_law_tolerance: float = eqx.field(static=True)
    detailed_balance_tolerance: float = eqx.field(static=True)
    reversal_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        inverse_temperature: float,
        /,
        *,
        maximum_paths: int,
        maximum_steps: int,
        autocorrelation_lag: int,
        first_law_tolerance: float = 1.0e-10,
        detailed_balance_tolerance: float = 1.0e-10,
        reversal_tolerance: float = 1.0e-12,
    ):
        beta = float(inverse_temperature)
        path_capacity = int(maximum_paths)
        step_capacity = int(maximum_steps)
        lag = int(autocorrelation_lag)
        tolerances = (
            float(first_law_tolerance),
            float(detailed_balance_tolerance),
            float(reversal_tolerance),
        )
        if (
            not math.isfinite(beta)
            or beta <= 0.0
            or path_capacity < 2
            or step_capacity <= 0
            or lag <= 0
            or lag >= path_capacity
            or not all(math.isfinite(value) and value >= 0.0 for value in tolerances)
        ):
            raise ValueError("Discrete path thermodynamic controls are invalid.")
        self.inverse_temperature = beta
        self.maximum_paths = path_capacity
        self.maximum_steps = step_capacity
        self.autocorrelation_lag = lag
        self.first_law_tolerance = tolerances[0]
        self.detailed_balance_tolerance = tolerances[1]
        self.reversal_tolerance = tolerances[2]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "normalized-discrete-path-thermodynamics",
                "inverse_temperature": beta,
                "maximum_paths": path_capacity,
                "maximum_steps": step_capacity,
                "autocorrelation_lag": lag,
                "first_law_tolerance": tolerances[0],
                "detailed_balance_tolerance": tolerances[1],
                "reversal_tolerance": tolerances[2],
            }
        )


class DiscretePathThermodynamicsResult(StrictModule, NonTrainableState):
    path_ids: tuple[str, ...] = eqx.field(static=True)
    normalized_forward_probability: Array
    normalized_reverse_probability: Array
    path_entropy_production: Array
    thermodynamic_entropy_production: Array
    medium_entropy_change: Array
    system_entropy_change: Array
    energy_change: Array
    work: Array
    heat_into_system: Array
    first_law_residual: Array
    detailed_balance_residual: Array
    integral_fluctuation_average: Array
    reversal_residual: Array
    forward_effective_sample_size: Array
    reverse_effective_sample_size: Array
    mean_entropy_production: Array
    entropy_uncertainty: CorrelatedUncertainty
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def normalized_discrete_path_thermodynamics(
    plan: DiscretePathThermodynamicsPlan,
    paths: tuple[PathBuffer, ...],
    forward_log_weight: ArrayLike,
    reverse_log_weight: ArrayLike,
    energy_change: ArrayLike,
    work: ArrayLike,
    heat_into_system: ArrayLike,
    system_entropy_change: ArrayLike,
    /,
) -> DiscretePathThermodynamicsResult:
    """Normalize path weights and test first-law, reversal, and fluctuation identities."""

    if not isinstance(plan, DiscretePathThermodynamicsPlan):
        raise TypeError("plan must be DiscretePathThermodynamicsPlan.")
    if not isinstance(paths, tuple) or any(
        not isinstance(path, PathBuffer) for path in paths
    ):
        raise TypeError("paths must be a tuple of PathBuffer values.")
    count = len(paths)
    if count < 2 or count > plan.maximum_paths:
        raise ValueError("Path count lies outside the declared resource bound.")
    if any(path.capacity > plan.maximum_steps + 1 for path in paths):
        raise ValueError("A path exceeds maximum_steps before thermodynamic reduction.")
    arrays = tuple(
        jnp.asarray(value, dtype=jnp.float64)
        for value in (
            forward_log_weight,
            reverse_log_weight,
            energy_change,
            work,
            heat_into_system,
            system_entropy_change,
        )
    )
    if any(value.shape != (count,) for value in arrays):
        raise ValueError("Every thermodynamic vector must contain one value per path.")
    if not all(np.all(np.isfinite(np.asarray(value))) for value in arrays):
        raise ValueError("Discrete path thermodynamics requires finite retained samples.")
    forward_log, reverse_log, delta_energy, work_, heat, delta_system_entropy = arrays
    log_forward_normalizer = jsp.special.logsumexp(forward_log)
    log_reverse_normalizer = jsp.special.logsumexp(reverse_log)
    log_forward_probability = forward_log - log_forward_normalizer
    log_reverse_probability = reverse_log - log_reverse_normalizer
    forward_probability = jnp.exp(log_forward_probability)
    reverse_probability = jnp.exp(log_reverse_probability)
    path_entropy = log_forward_probability - log_reverse_probability
    medium_entropy = -plan.inverse_temperature * heat
    thermodynamic_entropy = delta_system_entropy + medium_entropy
    first_law = delta_energy - work_ - heat
    detailed_balance = path_entropy - thermodynamic_entropy
    fluctuation_average = jnp.sum(forward_probability * jnp.exp(-path_entropy))
    reversal_residual = jnp.abs(fluctuation_average - 1.0)
    forward_ess = 1.0 / jnp.sum(forward_probability * forward_probability)
    reverse_ess = 1.0 / jnp.sum(reverse_probability * reverse_probability)
    mean_entropy = jnp.sum(forward_probability * thermodynamic_entropy)
    uncertainty = autocorrelation_uncertainty(
        thermodynamic_entropy,
        maximum_lag=plan.autocorrelation_lag,
    )
    path_valid = jnp.all(
        jnp.stack(tuple(path.valid() & path.time_reversed().valid() for path in paths))
    )
    first_law_scale = jnp.maximum(
        1.0,
        jnp.max(
            jnp.maximum(
                jnp.abs(delta_energy),
                jnp.maximum(jnp.abs(work_), jnp.abs(heat)),
            )
        ),
    )
    entropy_scale = jnp.maximum(
        1.0,
        jnp.max(jnp.maximum(jnp.abs(path_entropy), jnp.abs(thermodynamic_entropy))),
    )
    successful = (
        path_valid
        & uncertainty.valid
        & jnp.all(jnp.isfinite(forward_probability))
        & jnp.all(jnp.isfinite(reverse_probability))
        & (jnp.max(jnp.abs(first_law)) <= plan.first_law_tolerance * first_law_scale)
        & (
            jnp.max(jnp.abs(detailed_balance))
            <= plan.detailed_balance_tolerance * entropy_scale
        )
        & (reversal_residual <= plan.reversal_tolerance)
    )
    identifiers = tuple(path_trajectory_id(path) for path in paths)
    result_id = canonical_fingerprint(
        {
            "kind": "discrete-path-thermodynamics-result",
            "plan": plan.plan_id,
            "paths": identifiers,
        }
    )
    return DiscretePathThermodynamicsResult(
        identifiers,
        forward_probability,
        reverse_probability,
        path_entropy,
        thermodynamic_entropy,
        medium_entropy,
        delta_system_entropy,
        delta_energy,
        work_,
        heat,
        first_law,
        detailed_balance,
        fluctuation_average,
        reversal_residual,
        forward_ess,
        reverse_ess,
        mean_entropy,
        uncertainty,
        successful,
        plan.plan_id,
        result_id,
    )


__all__ = [
    "DiscretePathThermodynamicsPlan",
    "DiscretePathThermodynamicsResult",
    "normalized_discrete_path_thermodynamics",
]
