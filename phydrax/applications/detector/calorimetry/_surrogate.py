#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

from phydrax._strict import StrictModule

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ....domain import HyperRectangle, TimeInterval
from ....dynamics import ContinuousSystem, StateLayout
from ....nn.models import MLP
from ....solver import DiffraxEvolution, FunctionalSolver
from ....terms import FlowMatchingTerm
from ....transport import EndpointCouplingSample, LinearEndpointInterpolant
from ._corpus import CalorimeterCorpus
from ._geometry import CalorimeterGeometry


class ConditionalCalorimeterVelocity(StrictModule):
    """Sparse shared-cell velocity over fixed heterogeneous calorimeter adjacency."""

    network: MLP
    geometry: CalorimeterGeometry
    condition_names: tuple[str, ...] = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)

    def __init__(
        self,
        geometry: CalorimeterGeometry,
        condition_names,
        /,
        *,
        width: int,
        depth: int,
        key,
    ):
        if not isinstance(geometry, CalorimeterGeometry):
            raise TypeError("geometry must be CalorimeterGeometry.")
        names = tuple(str(value).strip() for value in condition_names)
        if (
            not names
            or any(not value for value in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("condition_names must be distinct non-empty strings.")
        input_size = 2 + geometry.geometry_features.shape[1] + len(names) + 1
        self.network = MLP(
            in_size=input_size,
            out_size=1,
            width_size=int(width),
            depth=int(depth),
            key=key,
        )
        self.geometry = geometry
        self.condition_names = names
        self.geometry_id = geometry.geometry_id
        self.cell_count = geometry.cell_count

    def __call__(self, state, time, condition):
        state_ = jnp.asarray(state)
        condition_ = jnp.asarray(condition)
        if state_.shape != (self.cell_count,) or condition_.shape != (
            len(self.condition_names),
        ):
            raise ValueError("Calorimeter velocity state or condition shape is invalid.")
        neighbor_sum = (
            jnp.zeros_like(state_)
            .at[self.geometry.receivers]
            .add(state_[self.geometry.senders])
        )
        degree = jnp.zeros_like(state_).at[self.geometry.receivers].add(1.0)
        neighbor_mean = neighbor_sum / jnp.maximum(degree, 1.0)
        features = jnp.concatenate(
            (
                state_[:, None],
                neighbor_mean[:, None],
                self.geometry.geometry_features,
                jnp.broadcast_to(condition_, (self.cell_count, condition_.shape[0])),
                jnp.full((self.cell_count, 1), time, dtype=state_.dtype),
            ),
            axis=1,
        )
        velocity = jax.vmap(self.network)(features)[:, 0]
        return jnp.where(self.geometry.active & ~self.geometry.dead, velocity, 0.0)


def _velocity_function(model):
    state_domain = HyperRectangle(
        jnp.full((model.cell_count,), -1.0e12),
        jnp.full((model.cell_count,), 1.0e12),
        label="x",
    )
    domain = state_domain @ TimeInterval(0.0, 1.0)
    domain = domain @ HyperRectangle(
        jnp.full((len(model.condition_names),), -1.0e12),
        jnp.full((len(model.condition_names),), 1.0e12),
        label="condition",
    )
    return domain.Function("x", "t", "condition")(model)


def _endpoints(corpus, indices, key, count, energy_scale):
    noise_key, index_key = jr.split(key)
    index = jnp.asarray(indices)[jr.randint(index_key, (count,), 0, len(indices))]
    target = jnp.sqrt(corpus.cell_energies[index] / energy_scale)
    noise = jr.normal(noise_key, target.shape, dtype=target.dtype)
    return EndpointCouplingSample(
        source=noise,
        target=target,
        source_indices=jnp.arange(count),
        target_indices=index,
        valid=jnp.ones((count,), dtype=jnp.bool_),
        log_weights=jnp.zeros((count,)),
        context={"condition": corpus.conditions[index]},
        coupling_id=corpus.dataset_id,
        provenance="standard-Gaussian-to-square-root-cell-energy",
    )


@dataclass(frozen=True)
class CalorimeterFitResult:
    model: ConditionalCalorimeterVelocity
    corpus: CalorimeterCorpus
    energy_scale: float
    condition_minimum: object
    condition_maximum: object
    initial_training_loss: float
    final_training_loss: float
    validation_loss: float
    training_steps: int
    fit_id: str
    scientific_claim: str = (
        "trained conditional fast-simulation proposal; qualification required"
    )


def fit_calorimeter_flow(
    corpus: CalorimeterCorpus,
    /,
    *,
    key,
    steps: int = 200,
    pairs_per_step: int = 32,
    width: int = 64,
    depth: int = 2,
    learning_rate: float = 1.0e-3,
    energy_scale: float = 1.0,
    commercial_use: bool = False,
    maximum_training_steps: int = 100_000,
    maximum_pairs: int = 65_536,
    maximum_width: int = 1024,
    maximum_depth: int = 16,
) -> CalorimeterFitResult:
    if not isinstance(corpus, CalorimeterCorpus):
        raise TypeError("corpus must be CalorimeterCorpus.")
    for manifest in corpus.rights:
        manifest.require_rights(training_use=True, commercial_use=commercial_use)
    for value, maximum in (
        (steps, maximum_training_steps),
        (pairs_per_step, maximum_pairs),
        (width, maximum_width),
        (depth, maximum_depth),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 1 <= value <= maximum
        ):
            raise ValueError(
                "Calorimeter training configuration exceeds its finite resource policy."
            )
    scale = float(energy_scale)
    rate = float(learning_rate)
    if not np.isfinite(scale) or scale <= 0.0 or not np.isfinite(rate) or rate <= 0.0:
        raise ValueError("energy_scale and learning_rate must be finite and positive.")
    model_key, train_key, evaluation_key, validation_key = jr.split(key, 4)
    model = ConditionalCalorimeterVelocity(
        corpus.geometry,
        corpus.condition_names,
        width=width,
        depth=depth,
        key=model_key,
    )
    interpolant = LinearEndpointInterpolant((corpus.geometry.cell_count,))
    provider = lambda sample_key: _endpoints(
        corpus,
        corpus.train_indices,
        jr.fold_in(sample_key, jr.key_data(train_key)[0]),
        pairs_per_step,
        scale,
    )
    training = FlowMatchingTerm(
        "velocity",
        provider,
        interpolant,
        sampling_mode="resample",
    )
    evaluation = FlowMatchingTerm(
        "velocity",
        _endpoints(corpus, corpus.train_indices, evaluation_key, pairs_per_step, scale),
        interpolant,
    )
    validation = FlowMatchingTerm(
        "velocity",
        _endpoints(
            corpus, corpus.validation_indices, validation_key, pairs_per_step, scale
        ),
        interpolant,
    )
    function = _velocity_function(model)
    initial = float(evaluation.loss({"velocity": function}, key=evaluation_key))
    solved = FunctionalSolver(functions={"velocity": function}, terms=(training,)).solve(
        num_iter=steps,
        optim=optax.adam(rate),
        jit=True,
        keep_best=False,
        log_every=0,
    )
    learned_function = solved.functions["velocity"]
    learned = learned_function.func.function
    final = float(evaluation.loss({"velocity": learned_function}, key=evaluation_key))
    heldout = float(validation.loss({"velocity": learned_function}, key=validation_key))
    if not np.all(np.isfinite((initial, final, heldout))):
        raise FloatingPointError("Calorimeter flow training produced nonfinite loss.")
    training_conditions = np.asarray(corpus.conditions)[list(corpus.train_indices)]
    minimum = jnp.asarray(np.min(training_conditions, axis=0))
    maximum = jnp.asarray(np.max(training_conditions, axis=0))
    fit_id = canonical_fingerprint(
        {
            "kind": "conditional-calorimeter-flow-fit",
            "dataset": corpus.dataset_id,
            "geometry": corpus.geometry.geometry_id,
            "weights": array_tree_fingerprint(learned),
            "steps": steps,
            "pairs": pairs_per_step,
            "width": width,
            "depth": depth,
            "learning_rate": rate,
            "energy_scale": scale,
            "key": np.asarray(jr.key_data(key)).tolist(),
        }
    )
    return CalorimeterFitResult(
        learned,
        corpus,
        scale,
        minimum,
        maximum,
        initial,
        final,
        heldout,
        steps,
        fit_id,
    )


class _CalorimeterField(StrictModule):
    model: ConditionalCalorimeterVelocity

    def __call__(self, time, state, condition):
        return self.model(state, time, condition)


class PreparedCalorimeterSampler(StrictModule):
    evolution: DiffraxEvolution
    model: ConditionalCalorimeterVelocity
    energy_scale: float = eqx.field(static=True)
    condition_minimum: object
    condition_maximum: object
    fit_id: str = eqx.field(static=True)
    maximum_samples: int = eqx.field(static=True)


@dataclass(frozen=True)
class CalorimeterFastSimulation:
    encoded_showers: object
    cell_energies: object
    conditions: object
    solver_valid: object
    solver_status: object
    in_domain: object
    valid: object
    fit_id: str


def prepare_calorimeter_sampler(
    fit: CalorimeterFitResult,
    /,
    *,
    rtol: float = 1.0e-5,
    atol: float = 1.0e-7,
    maximum_steps: int = 1024,
    maximum_samples: int = 65_536,
    commercial_use: bool = False,
    export: bool = False,
) -> PreparedCalorimeterSampler:
    if not isinstance(fit, CalorimeterFitResult):
        raise TypeError("fit must be CalorimeterFitResult.")
    for manifest in fit.corpus.rights:
        manifest.require_rights(
            commercial_use=commercial_use,
            export=export,
        )
    if maximum_steps < 1 or maximum_samples < 1:
        raise ValueError("Sampling resource bounds must be positive.")
    system = ContinuousSystem(
        _CalorimeterField(fit.model),
        state_layout=StateLayout((fit.model.cell_count,)),
        system_id=fit.fit_id,
    )
    return PreparedCalorimeterSampler(
        DiffraxEvolution(system, rtol=rtol, atol=atol, max_steps=maximum_steps),
        fit.model,
        fit.energy_scale,
        fit.condition_minimum,
        fit.condition_maximum,
        fit.fit_id,
        int(maximum_samples),
    )


def sample_calorimeter_showers(
    sampler: PreparedCalorimeterSampler,
    key,
    conditions,
    /,
) -> CalorimeterFastSimulation:
    if not isinstance(sampler, PreparedCalorimeterSampler):
        raise TypeError("sampler must be PreparedCalorimeterSampler.")
    context = jnp.asarray(conditions)
    if context.ndim != 2 or context.shape[1] != len(sampler.model.condition_names):
        raise ValueError("conditions must have shape (sample, condition_feature).")
    count = context.shape[0]
    if not 1 <= count <= sampler.maximum_samples:
        raise ValueError("Sample count exceeds maximum_samples.")
    in_domain = jnp.all(
        (context >= sampler.condition_minimum) & (context <= sampler.condition_maximum),
        axis=1,
    )
    noise = jr.normal(key, (count, sampler.model.cell_count), dtype=context.dtype)

    def one(state, condition):
        result = sampler.evolution.advance(state, 0.0, 1.0, condition)
        return result.final_state, result.valid, result.status

    encoded, solver_valid, solver_status = jax.vmap(one)(noise, context)
    decoded = encoded * encoded * sampler.energy_scale
    active = sampler.model.geometry.active & ~sampler.model.geometry.dead
    decoded = jnp.where(active[None, :], decoded, 0.0)
    physical = jnp.all(jnp.isfinite(decoded) & (decoded >= 0.0), axis=1)
    valid = solver_valid & in_domain & physical
    return CalorimeterFastSimulation(
        encoded,
        jnp.where(valid[:, None], decoded, jnp.nan),
        context,
        solver_valid,
        solver_status,
        in_domain,
        valid,
        sampler.fit_id,
    )


__all__ = [
    "CalorimeterFastSimulation",
    "CalorimeterFitResult",
    "ConditionalCalorimeterVelocity",
    "PreparedCalorimeterSampler",
    "fit_calorimeter_flow",
    "prepare_calorimeter_sampler",
    "sample_calorimeter_showers",
]
