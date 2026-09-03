#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._model import AbstractArrayModel
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import regularized_heaviside_values


ParticlePhase = Literal["outside", "inside"]


class RelaxedStoppingResult(StrictModule):
    stopping_weights: Array
    survival_probability: Array
    cumulative_stopping_probability: Array


class ProbabilisticStefanBatch(StrictModule, NonTrainableState):
    """Quadrature, paths, and randomized Gaussian moments for Stefan growth."""

    times: Array
    domain_points: Array
    domain_weights: Array
    initial_solid_fraction: Array
    liquid_paths: Array
    solid_paths: Array
    test_centers: Array
    test_inverse_widths: Array

    def __init__(
        self,
        *,
        times: ArrayLike,
        domain_points: ArrayLike,
        domain_weights: ArrayLike,
        initial_solid_fraction: ArrayLike,
        liquid_paths: ArrayLike,
        solid_paths: ArrayLike,
        test_centers: ArrayLike,
        test_inverse_widths: ArrayLike,
    ):
        times_ = jnp.asarray(times, dtype=float)
        points = jnp.asarray(domain_points, dtype=float)
        weights = jnp.asarray(domain_weights, dtype=float)
        initial = jnp.asarray(initial_solid_fraction, dtype=float)
        liquid = jnp.asarray(liquid_paths, dtype=float)
        solid = jnp.asarray(solid_paths, dtype=float)
        centers = jnp.asarray(test_centers, dtype=float)
        inverse_widths = jnp.asarray(test_inverse_widths, dtype=float)
        if times_.ndim != 1 or times_.size < 2 or bool(jnp.any(jnp.diff(times_) <= 0.0)):
            raise ValueError("times must be a strictly increasing non-empty vector.")
        if points.ndim != 2 or min(points.shape) <= 0:
            raise ValueError("domain_points must have shape (point, coordinate).")
        count, dimension = points.shape
        if weights.shape != (count,) or initial.shape != (count,):
            raise ValueError("Domain weights and initial fraction must match points.")
        expected_path_tail = (times_.size, dimension)
        if liquid.ndim != 3 or liquid.shape[1:] != expected_path_tail:
            raise ValueError("liquid_paths must have shape (path, time, coordinate).")
        if solid.ndim != 3 or solid.shape[1:] != expected_path_tail:
            raise ValueError("solid_paths must have shape (path, time, coordinate).")
        if liquid.shape[0] <= 0 or solid.shape[0] <= 0:
            raise ValueError(
                "Probabilistic Stefan batches require both path populations."
            )
        if centers.ndim != 2 or centers.shape[1] != dimension or centers.shape[0] <= 0:
            raise ValueError("test_centers must have shape (test, coordinate).")
        if inverse_widths.shape != (centers.shape[0],) or bool(
            jnp.any(inverse_widths <= 0.0)
        ):
            raise ValueError(
                "test_inverse_widths must be positive with one value per test."
            )
        if bool(jnp.any(weights < 0.0)) or bool(
            jnp.any((initial < 0.0) | (initial > 1.0))
        ):
            raise ValueError("Domain weights/fractions violate their physical ranges.")
        arrays = (
            times_,
            points,
            weights,
            initial,
            liquid,
            solid,
            centers,
            inverse_widths,
        )
        if any(not bool(jnp.all(jnp.isfinite(value))) for value in arrays):
            raise ValueError("Probabilistic Stefan batch arrays must be finite.")
        self.times = times_
        self.domain_points = points
        self.domain_weights = weights
        self.initial_solid_fraction = initial
        self.liquid_paths = liquid
        self.solid_paths = solid
        self.test_centers = centers
        self.test_inverse_widths = inverse_widths


class ProbabilisticStefanParameters(StrictModule, NonTrainableState):
    latent_heat: Array
    liquid_mass: Array
    solid_mass: Array
    liquid_sign: int
    interface_width: float
    maximum_phase_change: float
    jump_penalty: float

    def __init__(
        self,
        *,
        latent_heat: ArrayLike,
        liquid_mass: ArrayLike = 1.0,
        solid_mass: ArrayLike = 1.0,
        liquid_sign: Literal[-1, 1] = 1,
        interface_width: float,
        maximum_phase_change: float = 1.0,
        jump_penalty: float = 0.0,
    ):
        self.latent_heat = _positive_scalar(latent_heat, "latent_heat")
        self.liquid_mass = _positive_scalar(liquid_mass, "liquid_mass")
        self.solid_mass = _positive_scalar(solid_mass, "solid_mass")
        if liquid_sign not in (-1, 1):
            raise ValueError("liquid_sign must be -1 or 1.")
        self.liquid_sign = int(liquid_sign)
        self.interface_width = _positive_float(interface_width, "interface_width")
        change = float(maximum_phase_change)
        penalty = float(jump_penalty)
        if not math.isfinite(change) or change < 0.0:
            raise ValueError("maximum_phase_change must be finite and nonnegative.")
        if not math.isfinite(penalty) or penalty < 0.0:
            raise ValueError("jump_penalty must be finite and nonnegative.")
        self.maximum_phase_change = change
        self.jump_penalty = penalty


class ProbabilisticStefanLoss(StrictModule):
    total: Array
    moment: Array
    jump: Array
    moment_residual: Array
    current_phase_moments: Array
    liquid_absorbed_moments: Array
    solid_absorbed_moments: Array


class ProbabilisticLevelSetStefan(StrictModule):
    level_set: Callable[[Array], Array]

    def __init__(self, level_set: Callable[[Array], Array], /):
        if not callable(level_set):
            raise TypeError("level_set must be callable.")
        self.level_set = level_set


class ProbabilisticStefanFitResult(StrictModule):
    model: ProbabilisticLevelSetStefan
    loss_history: Array
    final_loss: ProbabilisticStefanLoss


def relaxed_first_passage_weights(
    signed_distances: ArrayLike,
    /,
    *,
    width: float,
    particle_phase: ParticlePhase,
) -> RelaxedStoppingResult:
    """Convert a signed-distance history into differentiable first-stop weights.

    Histories end in the time axis. Outside particles stop upon entering the
    negative phase; inside particles stop upon entering the positive phase.
    """

    values = jnp.asarray(signed_distances)
    if values.ndim < 1 or jnp.iscomplexobj(values):
        raise ValueError("signed_distances must be a real array with a time axis.")
    if particle_phase == "outside":
        hazard = regularized_heaviside_values(-values, width=width)
    elif particle_phase == "inside":
        hazard = regularized_heaviside_values(values, width=width)
    else:
        raise ValueError("particle_phase must be 'outside' or 'inside'.")
    hazard = jnp.clip(hazard, 0.0, 1.0)
    survival_before = jnp.concatenate(
        (
            jnp.ones_like(hazard[..., :1]),
            jnp.cumprod(1.0 - hazard[..., :-1], axis=-1),
        ),
        axis=-1,
    )
    stopping = hazard * survival_before
    survival = jnp.cumprod(1.0 - hazard, axis=-1)
    return RelaxedStoppingResult(
        stopping_weights=stopping,
        survival_probability=survival,
        cumulative_stopping_probability=jnp.cumsum(stopping, axis=-1),
    )


def probabilistic_stefan_moment_loss(
    model: ProbabilisticLevelSetStefan,
    batch: ProbabilisticStefanBatch,
    parameters: ProbabilisticStefanParameters,
    /,
) -> ProbabilisticStefanLoss:
    """Evaluate the weak probabilistic Stefan growth identity."""

    if not isinstance(model, ProbabilisticLevelSetStefan):
        raise TypeError("model must be ProbabilisticLevelSetStefan.")
    if not isinstance(batch, ProbabilisticStefanBatch):
        raise TypeError("batch must be ProbabilisticStefanBatch.")
    if not isinstance(parameters, ProbabilisticStefanParameters):
        raise TypeError("parameters must be ProbabilisticStefanParameters.")

    test_on_domain = _gaussian_tests(
        batch.domain_points,
        batch.test_centers,
        batch.test_inverse_widths,
    )
    initial_moments = ein.contract(
        "m,m,mk->k",
        batch.domain_weights,
        batch.initial_solid_fraction,
        test_on_domain,
    )

    def phase_at_time(time):
        points = jnp.concatenate(
            (
                batch.domain_points,
                jnp.full((batch.domain_points.shape[0], 1), time),
            ),
            axis=-1,
        )
        values = jax.vmap(lambda point: _level_set_call(model.level_set, point))(points)
        return regularized_heaviside_values(
            -values,
            width=parameters.interface_width,
        )

    phase = jax.vmap(phase_at_time)(batch.times)
    current_moments = ein.contract(
        "m,tm,mk->tk",
        batch.domain_weights,
        phase,
        test_on_domain,
    )

    liquid_distance = _path_level_sets(model.level_set, batch.liquid_paths, batch.times)
    solid_distance = _path_level_sets(model.level_set, batch.solid_paths, batch.times)
    liquid_stopping = relaxed_first_passage_weights(
        liquid_distance,
        width=parameters.interface_width,
        particle_phase="outside",
    )
    solid_stopping = relaxed_first_passage_weights(
        solid_distance,
        width=parameters.interface_width,
        particle_phase="inside",
    )
    liquid_tests = _gaussian_tests(
        batch.liquid_paths,
        batch.test_centers,
        batch.test_inverse_widths,
    )
    solid_tests = _gaussian_tests(
        batch.solid_paths,
        batch.test_centers,
        batch.test_inverse_widths,
    )
    liquid_increment = jnp.mean(
        liquid_stopping.stopping_weights[..., None] * liquid_tests,
        axis=0,
    )
    solid_increment = jnp.mean(
        solid_stopping.stopping_weights[..., None] * solid_tests,
        axis=0,
    )
    liquid_absorbed = parameters.liquid_mass * jnp.cumsum(liquid_increment, axis=0)
    solid_absorbed = parameters.solid_mass * jnp.cumsum(solid_increment, axis=0)
    residual = (
        initial_moments[None, :]
        - current_moments
        - (parameters.liquid_sign * liquid_absorbed - solid_absorbed)
        / parameters.latent_heat
    )
    moment_loss = jnp.mean(residual**2)

    phase_change = jnp.sum(
        batch.domain_weights[None, :] * jnp.abs(phase[1:] - phase[:-1]),
        axis=-1,
    )
    jump_excess = jax.nn.relu(phase_change - parameters.maximum_phase_change)
    jump_loss = parameters.jump_penalty * jnp.mean(jump_excess**2)
    return ProbabilisticStefanLoss(
        total=moment_loss + jump_loss,
        moment=moment_loss,
        jump=jump_loss,
        moment_residual=residual,
        current_phase_moments=current_moments,
        liquid_absorbed_moments=liquid_absorbed,
        solid_absorbed_moments=solid_absorbed,
    )


def fit_probabilistic_level_set_stefan(
    model: ProbabilisticLevelSetStefan,
    batch: ProbabilisticStefanBatch,
    parameters: ProbabilisticStefanParameters,
    /,
    *,
    steps: int,
    optimizer: optax.GradientTransformation | None = None,
    jit: bool = True,
) -> ProbabilisticStefanFitResult:
    """Fit a deep level set against relaxed probabilistic Stefan moments."""

    count = int(steps)
    if count < 0:
        raise ValueError("steps must be nonnegative.")
    transformation = optax.adam(1.0e-3) if optimizer is None else optimizer
    trainable, fixed = eqx.partition(model, eqx.is_inexact_array)
    state = transformation.init(trainable)

    def step(current, optimizer_state):
        objective = lambda value: (
            probabilistic_stefan_moment_loss(
                eqx.combine(value, fixed),
                batch,
                parameters,
            ).total
        )
        loss, gradient = eqx.filter_value_and_grad(objective)(current)
        updates, next_state = transformation.update(gradient, optimizer_state, current)
        return optax.apply_updates(current, updates), next_state, loss

    run_step = eqx.filter_jit(step) if jit else step
    history = []
    for _ in range(count):
        trainable, state, value = run_step(trainable, state)
        history.append(value)
    fitted = eqx.combine(trainable, fixed)
    final = probabilistic_stefan_moment_loss(fitted, batch, parameters)
    return ProbabilisticStefanFitResult(
        model=fitted,
        loss_history=jnp.stack(history) if history else jnp.zeros((0,)),
        final_loss=final,
    )


def _path_level_sets(level_set, paths: Array, times: Array, /) -> Array:
    path_count, time_count, dimension = paths.shape
    time_values = jnp.broadcast_to(times[None, :, None], (path_count, time_count, 1))
    spacetime = jnp.concatenate((paths, time_values), axis=-1)
    flat = spacetime.reshape((path_count * time_count, dimension + 1))
    return jax.vmap(lambda point: _level_set_call(level_set, point))(flat).reshape(
        (path_count, time_count)
    )


def _gaussian_tests(points: Array, centers: Array, inverse_widths: Array, /) -> Array:
    displacement = points[..., :, None, :] - centers
    squared = jnp.sum(displacement * displacement, axis=-1)
    return jnp.exp(-squared * inverse_widths)


def _level_set_call(level_set: Callable[[Array], Array], point: Array, /) -> Array:
    value = (
        level_set(point, key=None)
        if isinstance(level_set, AbstractArrayModel)
        else level_set(point)
    )
    scalar = jnp.asarray(value)
    if scalar.shape != () or jnp.iscomplexobj(scalar):
        raise ValueError("Probabilistic Stefan level_set must return one real scalar.")
    return scalar


def _positive_float(value: float, name: str, /) -> float:
    scalar = float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return scalar


def _positive_scalar(value: ArrayLike, name: str, /) -> Array:
    scalar = jnp.asarray(value, dtype=float)
    if scalar.shape != () or not bool(jnp.isfinite(scalar)) or float(scalar) <= 0.0:
        raise ValueError(f"{name} must be one finite positive scalar.")
    return scalar


__all__ = [
    "ParticlePhase",
    "ProbabilisticLevelSetStefan",
    "ProbabilisticStefanBatch",
    "ProbabilisticStefanFitResult",
    "ProbabilisticStefanLoss",
    "ProbabilisticStefanParameters",
    "RelaxedStoppingResult",
    "fit_probabilistic_level_set_stefan",
    "probabilistic_stefan_moment_loss",
    "relaxed_first_passage_weights",
]
