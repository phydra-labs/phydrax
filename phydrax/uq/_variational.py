#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
from jaxtyping import Array, PyTree

from .._frozendict import frozendict
from .._strict import StrictModule
from ._checkpoint import (
    checkpoint_compatibility,
    CheckpointCorruptionError,
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)
from ._posterior import PosteriorProblem
from ._posterior_predictive import (
    predict_from_position_samples,
    sample_observations_from_position_samples,
)
from ._predictive import PredictiveField


_TRAINING_TAG = 0
_FINAL_SAMPLING_TAG = 1


def _tree_nbytes(tree: Any, /) -> int:
    return sum(int(leaf.nbytes) for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf))


def _tree_all_finite(tree: Any, /) -> Array:
    leaves = [leaf for leaf in jax.tree.leaves(tree) if eqx.is_array(leaf)]
    if not leaves:
        return jnp.asarray(False)
    return jnp.all(jnp.stack([jnp.all(jnp.isfinite(leaf)) for leaf in leaves]))


class AbstractVariationalFamily(StrictModule):
    """Normalized distribution over unconstrained posterior coordinates."""

    @property
    @abstractmethod
    def family_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def sample_and_log_prob(
        self,
        key: Array,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[PyTree[Array], Array]:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, value: PyTree[Any], /) -> Array:
        raise NotImplementedError


class MeanFieldGaussianFamily(AbstractVariationalFamily):
    """Reparameterized diagonal Gaussian over an arbitrary array PyTree."""

    location: PyTree[Array]
    raw_scale: PyTree[Array]
    scale_floor: float = eqx.field(static=True)

    def __init__(
        self,
        location: PyTree[Any],
        raw_scale: PyTree[Any],
        /,
        *,
        scale_floor: float = 1e-6,
    ):
        location_tree = jax.tree.map(jnp.asarray, location)
        raw_scale_tree = jax.tree.map(jnp.asarray, raw_scale)
        if jax.tree.structure(location_tree) != jax.tree.structure(raw_scale_tree):
            raise ValueError("location and raw_scale must share one PyTree structure.")
        locations = jax.tree.leaves(location_tree)
        raw_scales = jax.tree.leaves(raw_scale_tree)
        if not locations:
            raise ValueError("A variational family requires at least one array leaf.")
        for location_leaf, raw_scale_leaf in zip(
            locations,
            raw_scales,
            strict=True,
        ):
            if location_leaf.shape != raw_scale_leaf.shape:
                raise ValueError("location and raw_scale leaf shapes must agree.")
            if not jnp.issubdtype(location_leaf.dtype, jnp.floating):
                raise TypeError("Variational locations must be real floating arrays.")
            if not jnp.issubdtype(raw_scale_leaf.dtype, jnp.floating):
                raise TypeError("Variational scales must be real floating arrays.")
        floor = float(scale_floor)
        if not isfinite(floor) or floor <= 0.0:
            raise ValueError("scale_floor must be positive and finite.")
        self.location = location_tree
        self.raw_scale = raw_scale_tree
        self.scale_floor = floor

    @classmethod
    def from_position(
        cls,
        position: PyTree[Any],
        /,
        *,
        initial_scale: float = 0.1,
        scale_floor: float = 1e-6,
    ) -> "MeanFieldGaussianFamily":
        scale = float(initial_scale)
        floor = float(scale_floor)
        if not isfinite(scale) or scale <= floor:
            raise ValueError("initial_scale must be finite and exceed scale_floor.")
        location = jax.tree.map(jnp.asarray, position)
        raw_value = jnp.log(jnp.expm1(jnp.asarray(scale - floor)))
        raw_scale = jax.tree.map(
            lambda leaf: jnp.full_like(leaf, raw_value),
            location,
        )
        return cls(location, raw_scale, scale_floor=floor)

    @property
    def family_id(self) -> str:
        return "mean-field-gaussian"

    @property
    def scale(self) -> PyTree[Array]:
        return jax.tree.map(
            lambda value: jax.nn.softplus(value) + self.scale_floor,
            self.raw_scale,
        )

    def sample_and_log_prob(
        self,
        key: Array,
        /,
        *,
        sample_shape: tuple[int, ...] = (),
    ) -> tuple[PyTree[Array], Array]:
        shape = tuple(int(size) for size in sample_shape)
        if any(size <= 0 for size in shape):
            raise ValueError("sample_shape dimensions must be positive.")
        locations, treedef = jax.tree.flatten(self.location)
        scales = jax.tree.leaves(self.scale)
        keys = jr.split(key, len(locations))
        samples = treedef.unflatten(
            [
                location
                + scale
                * jr.normal(
                    sample_key,
                    shape + location.shape,
                    dtype=location.dtype,
                )
                for sample_key, location, scale in zip(
                    keys,
                    locations,
                    scales,
                    strict=True,
                )
            ]
        )
        return samples, self.log_prob(samples)

    def log_prob(self, value: PyTree[Any], /) -> Array:
        if jax.tree.structure(value) != jax.tree.structure(self.location):
            raise ValueError("value has an incompatible variational PyTree structure.")
        terms = []
        log_two_pi = jnp.log(jnp.asarray(2.0 * jnp.pi))
        for sample, location, scale in zip(
            jax.tree.leaves(value),
            jax.tree.leaves(self.location),
            jax.tree.leaves(self.scale),
            strict=True,
        ):
            sample_array = jnp.asarray(sample)
            if location.shape and (
                sample_array.ndim < location.ndim
                or sample_array.shape[-location.ndim :] != location.shape
            ):
                raise ValueError(
                    "A variational value leaf has an invalid trailing shape."
                )
            standardized = (sample_array - location) / scale
            element = -0.5 * (
                jnp.square(standardized) + log_two_pi + 2.0 * jnp.log(scale)
            )
            axes = tuple(range(element.ndim - location.ndim, element.ndim))
            terms.append(jnp.sum(element, axis=axes) if axes else element)
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total


class VariationalConfig(StrictModule):
    """Static reverse-KL optimization and recording controls."""

    num_steps: int = eqx.field(static=True)
    samples_per_step: int = eqx.field(static=True)
    learning_rate: float = eqx.field(static=True)
    gradient_clip: float = eqx.field(static=True)
    record_every: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        num_steps: int = 1000,
        samples_per_step: int = 16,
        learning_rate: float = 1e-3,
        gradient_clip: float = 100.0,
        record_every: int = 10,
    ):
        steps = int(num_steps)
        samples = int(samples_per_step)
        interval = int(record_every)
        rate = float(learning_rate)
        clipping = float(gradient_clip)
        if steps < 1 or samples < 1 or interval < 1:
            raise ValueError(
                "Variational step, sample, and record counts must be positive."
            )
        if not isfinite(rate) or rate <= 0.0:
            raise ValueError("learning_rate must be positive and finite.")
        if not isfinite(clipping) or clipping <= 0.0:
            raise ValueError("gradient_clip must be positive and finite.")
        self.num_steps = steps
        self.samples_per_step = samples
        self.learning_rate = rate
        self.gradient_clip = clipping
        self.record_every = interval

    def as_dict(self) -> dict[str, int | float]:
        return {
            "num_steps": self.num_steps,
            "samples_per_step": self.samples_per_step,
            "learning_rate": self.learning_rate,
            "gradient_clip": self.gradient_clip,
            "record_every": self.record_every,
        }


class VariationalDiagnostics(StrictModule):
    """Recorded ELBO, gradient norms, and finite optimization state."""

    steps: Array
    elbo: Array
    gradient_norm: Array
    finite: Array
    completed_steps: int = eqx.field(static=True)


class VariationalResult(StrictModule):
    """Fitted normalized posterior approximation and unconstrained draws."""

    problem: PosteriorProblem
    family: AbstractVariationalFamily
    samples: PyTree[Array]
    unconstrained_samples: PyTree[Array]
    log_target: Array
    log_variational: Array
    diagnostics: VariationalDiagnostics
    root_key: Array
    config: VariationalConfig
    duration_seconds: float = eqx.field(static=True)
    optimization_duration_seconds: float = eqx.field(static=True)
    sampling_duration_seconds: float = eqx.field(static=True)
    sample_memory_bytes: int = eqx.field(static=True)
    family_memory_bytes: int = eqx.field(static=True)
    approximation_id: str = eqx.field(static=True)

    @property
    def num_draws(self) -> int:
        return int(self.log_target.shape[0])

    def predict(
        self,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        draw_dim: str = "__phydra_uq_draw",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        return predict_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            *args,
            sample_dims=(draw_dim,),
            sample_sources=("epistemic",),
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )

    def sample_observations(
        self,
        key: Array,
        *args: Any,
        batch_size: int | None = None,
        valid_policy: Literal["record", "raise"] = "record",
        draw_dim: str = "__phydra_uq_draw",
        observation_dim: str = "__phydra_uq_observation",
        **kwargs: Any,
    ) -> PredictiveField | frozendict[str, PredictiveField]:
        return sample_observations_from_position_samples(
            self.problem,
            self.unconstrained_samples,
            key,
            *args,
            sample_dims=(draw_dim,),
            sample_sources=("epistemic",),
            observation_dim=observation_dim,
            batch_size=batch_size,
            valid_policy=valid_policy,
            **kwargs,
        )


def _write_variational_checkpoint(
    destination: Path,
    *,
    compatibility,
    completed: int,
    family,
    optimizer_state,
    recorded_steps,
    elbo_history,
    gradient_history,
    finite_history,
    duration_seconds: float,
) -> None:
    arrays = {
        "recorded_steps": jnp.asarray(recorded_steps, dtype=jnp.int32),
        "elbo_history": jnp.asarray(elbo_history),
        "gradient_history": jnp.asarray(gradient_history),
        "finite_history": jnp.asarray(finite_history, dtype=bool),
    }
    state = {
        "completed_steps": int(completed),
        "duration_seconds": float(duration_seconds),
        "family_tree": pack_array_tree("family", family, arrays),
        "optimizer_tree": pack_array_tree("optimizer", optimizer_state, arrays),
    }
    write_checkpoint_archive(
        destination,
        kind="variational",
        compatibility=compatibility,
        state=state,
        arrays=arrays,
    )


def _read_variational_checkpoint(
    source: Path,
    *,
    compatibility,
    family_template,
    optimizer_template,
):
    state, arrays = read_checkpoint_archive(
        source,
        kind="variational",
        compatibility=compatibility,
    )
    completed = int(state.get("completed_steps", -1))
    if completed < 0:
        raise CheckpointCorruptionError("Variational completed step count is invalid.")
    family = unpack_array_tree(state["family_tree"], arrays, family_template)
    optimizer_state = unpack_array_tree(
        state["optimizer_tree"], arrays, optimizer_template
    )
    required = (
        "recorded_steps",
        "elbo_history",
        "gradient_history",
        "finite_history",
    )
    if any(name not in arrays for name in required):
        raise CheckpointCorruptionError("Variational history arrays are incomplete.")
    recorded_steps = jnp.asarray(arrays["recorded_steps"], dtype=jnp.int32)
    elbo_history = jnp.asarray(arrays["elbo_history"])
    gradient_history = jnp.asarray(arrays["gradient_history"])
    finite_history = jnp.asarray(arrays["finite_history"], dtype=bool)
    if not (
        recorded_steps.ndim == 1
        and elbo_history.shape == recorded_steps.shape
        and gradient_history.shape == recorded_steps.shape
        and finite_history.shape == recorded_steps.shape
    ):
        raise CheckpointCorruptionError("Variational history shapes are incompatible.")
    return (
        completed,
        family,
        optimizer_state,
        list(recorded_steps),
        list(elbo_history),
        list(gradient_history),
        list(finite_history),
        float(state.get("duration_seconds", 0.0)),
    )


def fit_variational(
    problem: PosteriorProblem,
    /,
    *,
    key: Array,
    family: AbstractVariationalFamily | None = None,
    config: VariationalConfig | None = None,
    num_samples: int = 1000,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int | None = None,
    checkpoint_id: str | None = None,
    resume_from: str | Path | None = None,
) -> VariationalResult:
    """Fit a normalized reverse-KL posterior in unconstrained coordinates."""

    if not isinstance(problem, PosteriorProblem):
        raise TypeError("problem must be a PosteriorProblem.")
    family_ = (
        MeanFieldGaussianFamily.from_position(problem.initial_position)
        if family is None
        else family
    )
    if not isinstance(family_, AbstractVariationalFamily):
        raise TypeError("family must implement AbstractVariationalFamily or be None.")
    config_ = VariationalConfig() if config is None else config
    if not isinstance(config_, VariationalConfig):
        raise TypeError("config must be VariationalConfig or None.")
    draws = int(num_samples)
    if draws < 1:
        raise ValueError("num_samples must be positive.")
    destination = (
        Path(checkpoint_path)
        if checkpoint_path is not None
        else (Path(resume_from) if resume_from is not None else None)
    )
    if checkpoint_every is not None and destination is None:
        raise ValueError("checkpoint_every requires checkpoint_path or resume_from.")
    checkpoint_interval = (
        min(100, config_.num_steps) if checkpoint_every is None else int(checkpoint_every)
    )
    if checkpoint_interval < 1:
        raise ValueError("checkpoint_every must be positive.")
    if destination is not None and (checkpoint_id is None or not str(checkpoint_id)):
        raise ValueError("checkpoint_id is required for variational checkpointing.")

    settings = {
        "algorithm": "reverse-kl-variational",
        "family_id": family_.family_id,
        "config": {
            name: value
            for name, value in config_.as_dict().items()
            if name != "num_steps"
        },
        "num_samples": draws,
        "root_key": [int(value) for value in jr.key_data(key).reshape(-1)],
    }
    compatibility = (
        checkpoint_compatibility(
            problem,
            checkpoint_id=str(checkpoint_id),
            settings=settings,
        )
        if destination is not None
        else None
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config_.gradient_clip),
        optax.adam(config_.learning_rate),
    )
    dynamic_family, static_family = eqx.partition(family_, eqx.is_inexact_array)
    optimizer_state = optimizer.init(dynamic_family)
    started = perf_counter()

    if resume_from is None:
        completed = 0
        recorded_steps: list[Any] = []
        elbo_history: list[Any] = []
        gradient_history: list[Any] = []
        finite_history: list[Any] = []
        previous_duration = 0.0
    else:
        if compatibility is None:
            raise RuntimeError("Variational resume compatibility was not initialized.")
        (
            completed,
            restored_family,
            optimizer_state,
            recorded_steps,
            elbo_history,
            gradient_history,
            finite_history,
            previous_duration,
        ) = _read_variational_checkpoint(
            Path(resume_from),
            compatibility=compatibility,
            family_template=family_,
            optimizer_template=optimizer_state,
        )
        if completed > config_.num_steps:
            raise ValueError("Checkpoint exceeds the configured variational steps.")
        dynamic_family, static_family = eqx.partition(
            restored_family, eqx.is_inexact_array
        )

    def loss_function(current_dynamic, step_key):
        current_family = eqx.combine(current_dynamic, static_family)
        positions, log_variational = current_family.sample_and_log_prob(
            step_key,
            sample_shape=(config_.samples_per_step,),
        )
        log_target = jax.vmap(problem.log_density)(positions)
        loss = jnp.mean(log_variational - log_target)
        finite = (
            jnp.isfinite(loss)
            & jnp.all(jnp.isfinite(log_variational))
            & jnp.all(jnp.isfinite(log_target))
            & _tree_all_finite(positions)
        )
        return loss, finite

    @eqx.filter_jit
    def update(current_dynamic, current_optimizer_state, step_key):
        (loss, finite), gradient = eqx.filter_value_and_grad(
            loss_function,
            has_aux=True,
        )(current_dynamic, step_key)
        gradient_norm = optax.tree.norm(gradient)
        updates, next_optimizer_state = optimizer.update(
            gradient,
            current_optimizer_state,
            current_dynamic,
        )
        next_dynamic = eqx.apply_updates(current_dynamic, updates)
        finite = finite & jnp.isfinite(gradient_norm) & _tree_all_finite(next_dynamic)
        return next_dynamic, next_optimizer_state, loss, gradient_norm, finite

    optimization_started = perf_counter()
    while completed < config_.num_steps:
        step_key = jr.fold_in(
            jr.fold_in(key, _TRAINING_TAG),
            jnp.asarray(completed, dtype=jnp.uint32),
        )
        (
            dynamic_family,
            optimizer_state,
            loss,
            gradient_norm,
            finite,
        ) = update(dynamic_family, optimizer_state, step_key)
        jax.block_until_ready(loss)
        completed += 1
        if not bool(finite):
            raise FloatingPointError(
                f"Variational optimization became nonfinite at step {completed}."
            )
        if completed % config_.record_every == 0 or completed == config_.num_steps:
            recorded_steps.append(jnp.asarray(completed, dtype=jnp.int32))
            elbo_history.append(-loss)
            gradient_history.append(gradient_norm)
            finite_history.append(finite)
        if (
            destination is not None
            and compatibility is not None
            and (completed % checkpoint_interval == 0 or completed == config_.num_steps)
        ):
            _write_variational_checkpoint(
                destination,
                compatibility=compatibility,
                completed=completed,
                family=eqx.combine(dynamic_family, static_family),
                optimizer_state=optimizer_state,
                recorded_steps=recorded_steps,
                elbo_history=elbo_history,
                gradient_history=gradient_history,
                finite_history=finite_history,
                duration_seconds=(previous_duration + perf_counter() - started),
            )
    optimization_duration = perf_counter() - optimization_started
    fitted_family = eqx.combine(dynamic_family, static_family)
    sampling_started = perf_counter()
    unconstrained_samples, log_variational = fitted_family.sample_and_log_prob(
        jr.fold_in(key, _FINAL_SAMPLING_TAG),
        sample_shape=(draws,),
    )
    log_target = jax.vmap(problem.log_density)(unconstrained_samples)
    samples = problem.parameter_space.constrain(unconstrained_samples)
    jax.block_until_ready(log_target)
    sampling_duration = perf_counter() - sampling_started
    diagnostics = VariationalDiagnostics(
        steps=jnp.asarray(recorded_steps, dtype=jnp.int32),
        elbo=jnp.asarray(elbo_history),
        gradient_norm=jnp.asarray(gradient_history),
        finite=jnp.asarray(finite_history, dtype=bool),
        completed_steps=completed,
    )
    duration = previous_duration + perf_counter() - started
    return VariationalResult(
        problem=problem,
        family=fitted_family,
        samples=samples,
        unconstrained_samples=unconstrained_samples,
        log_target=log_target,
        log_variational=log_variational,
        diagnostics=diagnostics,
        root_key=jnp.asarray(key),
        config=config_,
        duration_seconds=duration,
        optimization_duration_seconds=optimization_duration,
        sampling_duration_seconds=sampling_duration,
        sample_memory_bytes=(
            _tree_nbytes(samples)
            + _tree_nbytes(unconstrained_samples)
            + int(log_target.nbytes)
            + int(log_variational.nbytes)
        ),
        family_memory_bytes=_tree_nbytes(fitted_family),
        approximation_id=f"reverse-kl/{fitted_family.family_id}",
    )


__all__ = [
    "AbstractVariationalFamily",
    "fit_variational",
    "MeanFieldGaussianFamily",
    "VariationalConfig",
    "VariationalDiagnostics",
    "VariationalResult",
]
