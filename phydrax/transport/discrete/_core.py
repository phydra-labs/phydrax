#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._strict import AbstractAttribute, StrictModule
from ...pgm import (
    contrastive_divergence_loss,
    DiscreteFactorGraph,
    factor_graph_log_score,
    GibbsSchedule,
    GibbsState,
    PreparedChromaticGibbs,
    sample_gibbs,
)


class AbstractDiscreteNoisingKernel(StrictModule):
    kernel_id: AbstractAttribute[str]

    @abstractmethod
    def sample(self, key: Key[Array, ""], state: Array, cardinalities: Array, /) -> Array:
        raise NotImplementedError

    @abstractmethod
    def log_prob(self, next_state: Array, state: Array, cardinalities: Array, /) -> Array:
        raise NotImplementedError


class CategoricalNoisingKernel(AbstractDiscreteNoisingKernel):
    """Independent retain-or-uniform finite-state noising transition."""

    retention: float = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)

    def __init__(self, retention: float, /):
        value = float(retention)
        if not isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("retention must lie in [0, 1].")
        self.retention = value
        self.kernel_id = f"categorical-noise:{value}"

    def sample(self, key, state, cardinalities, /):
        retain_key, noise_key = jr.split(key)
        retain = jr.bernoulli(retain_key, self.retention, state.shape)
        uniform = jr.uniform(noise_key, state.shape)
        noise = jnp.floor(uniform * cardinalities).astype(jnp.int32)
        return jnp.where(retain, state, noise)

    def log_prob(self, next_state, state, cardinalities, /):
        same = next_state == state
        probability = (1.0 - self.retention) / cardinalities
        probability = probability + same * self.retention
        valid = (next_state >= 0) & (next_state < cardinalities)
        return jnp.sum(jnp.where(valid, jnp.log(probability), -jnp.inf), axis=-1)


class DiscreteForwardProcess(StrictModule):
    kernels: tuple[AbstractDiscreteNoisingKernel, ...]
    process_id: str = eqx.field(static=True)

    def __init__(self, kernels: Sequence[AbstractDiscreteNoisingKernel], /):
        values = tuple(kernels)
        if not values or any(
            not isinstance(value, AbstractDiscreteNoisingKernel) for value in values
        ):
            raise ValueError("kernels must be a nonempty sequence of noising kernels.")
        self.kernels = values
        self.process_id = canonical_fingerprint(
            {
                "kind": "discrete-forward-process",
                "kernels": [value.kernel_id for value in values],
            }
        )

    def sample_path(
        self, key: Key[Array, ""], initial: Array, cardinalities: Array, /
    ) -> Array:
        initial_state = jnp.asarray(initial, dtype=jnp.int32)
        cards = jnp.asarray(cardinalities, dtype=jnp.int32)
        if cards.shape != initial_state.shape[-1:] or bool(jnp.any(cards < 1)):
            raise ValueError(
                "cardinalities must be positive and match the state event axis."
            )
        states = [initial_state]
        keys = jr.split(key, len(self.kernels))
        for kernel, subkey in zip(self.kernels, keys):
            states.append(kernel.sample(subkey, states[-1], cards))
        return jnp.stack(states)


class FactorGraphReverseKernel(StrictModule):
    """One reverse denoising conditional implemented by clamped factor-graph Gibbs."""

    graph: DiscreteFactorGraph
    prepared: PreparedChromaticGibbs
    input_variables: Array
    output_variables: Array
    schedule: GibbsSchedule
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        graph: DiscreteFactorGraph,
        prepared: PreparedChromaticGibbs,
        input_variables: ArrayLike,
        output_variables: ArrayLike,
        schedule: GibbsSchedule,
        /,
    ):
        if not isinstance(graph, DiscreteFactorGraph):
            raise TypeError("graph must be DiscreteFactorGraph.")
        if not isinstance(prepared, PreparedChromaticGibbs):
            raise TypeError("prepared must be PreparedChromaticGibbs.")
        inputs = jnp.asarray(input_variables, dtype=jnp.int32).reshape((-1,))
        outputs = jnp.asarray(output_variables, dtype=jnp.int32).reshape((-1,))
        if prepared.graph.structure_id != graph.structure_id:
            raise ValueError("prepared Gibbs plan does not match graph.")
        if inputs.size == 0 or outputs.size == 0:
            raise ValueError("Reverse kernels require input and output variables.")
        if set(map(int, inputs.tolist())) & set(map(int, outputs.tolist())):
            raise ValueError("Reverse input and output variables must be disjoint.")
        input_host = tuple(int(value) for value in inputs.tolist())
        output_host = tuple(int(value) for value in outputs.tolist())
        if (
            len(set(input_host)) != len(input_host)
            or len(set(output_host)) != len(output_host)
            or min(input_host + output_host) < 0
            or max(input_host + output_host) >= graph.num_variables
        ):
            raise ValueError("Reverse variable indices must be unique and in bounds.")
        if not isinstance(schedule, GibbsSchedule):
            raise TypeError("schedule must be GibbsSchedule.")
        self.graph = graph
        self.prepared = prepared
        self.input_variables = inputs
        self.output_variables = outputs
        self.schedule = schedule
        self.kernel_id = canonical_fingerprint(
            {
                "kind": "factor-graph-reverse-kernel",
                "graph": graph.structure_id,
                "inputs": inputs.tolist(),
                "outputs": outputs.tolist(),
                "schedule": [
                    schedule.warmup_sweeps,
                    schedule.num_draws,
                    schedule.sweeps_per_draw,
                ],
            }
        )

    def sample(self, key: Key[Array, ""], noisy: Array, initial: GibbsState, /) -> Array:
        if initial.positions.shape[1:] != (self.graph.num_variables,):
            raise ValueError("initial Gibbs state does not match the reverse graph.")
        values = jnp.asarray(noisy, dtype=jnp.int32)
        expected = (initial.num_chains, int(self.input_variables.shape[0]))
        if values.shape != expected:
            raise ValueError(f"noisy must have shape {expected}; got {values.shape}.")
        input_cardinalities = self.graph.cardinalities[self.input_variables]
        if bool(jnp.any((values < 0) | (values >= input_cardinalities[jnp.newaxis, :]))):
            raise ValueError("noisy contains a state outside reverse-input support.")
        positions = initial.positions.at[:, self.input_variables].set(values)
        state = GibbsState(
            positions,
            self.prepared.precision.accumulation(
                factor_graph_log_score(self.graph, positions)
            ),
        )
        clamped = (
            jnp.zeros((self.graph.num_variables,), dtype=bool)
            .at[self.input_variables]
            .set(True)
        )
        result = sample_gibbs(
            self.prepared,
            state,
            key=key,
            schedule=self.schedule,
            clamped=clamped,
        )
        return result.samples[:, -1, self.output_variables]


class DiscreteDenoisingProcess(StrictModule):
    forward: DiscreteForwardProcess
    reverse: tuple[FactorGraphReverseKernel, ...]
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        forward: DiscreteForwardProcess,
        reverse: Sequence[FactorGraphReverseKernel],
        /,
    ):
        if not isinstance(forward, DiscreteForwardProcess):
            raise TypeError("forward must be DiscreteForwardProcess.")
        reverse_values = tuple(reverse)
        if any(
            not isinstance(value, FactorGraphReverseKernel) for value in reverse_values
        ):
            raise TypeError("reverse must contain FactorGraphReverseKernel values.")
        if len(reverse_values) != len(forward.kernels):
            raise ValueError("Reverse kernel count must match forward noising steps.")
        self.forward = forward
        self.reverse = reverse_values
        self.process_id = canonical_fingerprint(
            {
                "kind": "discrete-denoising-process",
                "forward": forward.process_id,
                "reverse": [value.kernel_id for value in reverse_values],
            }
        )

    def sample_reverse(
        self,
        key: Key[Array, ""],
        noisy: ArrayLike,
        initial_states: Sequence[GibbsState],
        /,
    ) -> Array:
        """Apply reverse kernels from the noisiest layer to the data layer."""
        states = tuple(initial_states)
        if len(states) != len(self.reverse):
            raise ValueError(
                "initial_states must provide one Gibbs state per reverse layer."
            )
        value = jnp.asarray(noisy, dtype=jnp.int32)
        keys = jr.split(key, len(self.reverse))
        for kernel, state, subkey in zip(
            reversed(self.reverse),
            reversed(states),
            keys,
        ):
            value = kernel.sample(subkey, value, state)
        return value


class RecoveryLikelihoodObjective(StrictModule):
    """Layerwise positive/negative phase objective for factor-graph reverse kernels."""

    def __call__(
        self,
        kernel: FactorGraphReverseKernel,
        positive_assignments: Array,
        negative_assignments: Array,
        /,
    ):
        return contrastive_divergence_loss(
            kernel.graph,
            positive_assignments,
            negative_assignments,
        )


class AdaptiveMixingState(StrictModule):
    penalties: Array
    previous_correlation: Array
    epoch: Array


class AdaptiveMixingPenalty(StrictModule):
    """Closed-loop layerwise penalty control from lagged chain correlation."""

    target: float = eqx.field(static=True)
    update_fraction: float = eqx.field(static=True)
    minimum: float = eqx.field(static=True)
    maximum: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        target: float = 0.03,
        update_fraction: float = 0.2,
        minimum: float = 1e-4,
        maximum: float = 1.0,
    ):
        values = tuple(
            float(value) for value in (target, update_fraction, minimum, maximum)
        )
        if (
            not 0 < values[0] < 1
            or not 0 < values[1] < 1
            or not 0 < values[2] <= values[3]
        ):
            raise ValueError("Invalid adaptive mixing penalty configuration.")
        self.target, self.update_fraction, self.minimum, self.maximum = values

    def initialize(self, layers: int, *, initial: float = 0.0) -> AdaptiveMixingState:
        if layers < 1 or (initial != 0.0 and not self.minimum <= initial <= self.maximum):
            raise ValueError("Invalid layer count or initial penalty.")
        return AdaptiveMixingState(
            penalties=jnp.full((layers,), initial),
            previous_correlation=jnp.full((layers,), jnp.inf),
            epoch=jnp.asarray(0, dtype=jnp.int32),
        )

    def update(
        self, state: AdaptiveMixingState, correlation: ArrayLike, /
    ) -> AdaptiveMixingState:
        values = jnp.asarray(correlation, dtype=float)
        if values.shape != state.penalties.shape:
            raise ValueError("correlation must match layer penalties.")
        base = jnp.maximum(state.penalties, self.minimum)
        decreased = (1.0 - self.update_fraction) * base
        increased = (1.0 + self.update_fraction) * base
        proposed = jnp.where(
            values < self.target,
            decreased,
            jnp.where(values > state.previous_correlation, increased, base),
        )
        proposed = jnp.where(proposed < self.minimum, 0.0, proposed)
        proposed = jnp.minimum(proposed, self.maximum)
        return AdaptiveMixingState(
            penalties=proposed,
            previous_correlation=values,
            epoch=state.epoch + 1,
        )


class HybridDiscreteEmbedding(StrictModule):
    """Explicit deterministic encoder/decoder around one discrete latent process."""

    encoder: Callable = eqx.field(static=True)
    decoder: Callable = eqx.field(static=True)
    process: DiscreteDenoisingProcess
    embedding_id: str = eqx.field(static=True)

    def __init__(
        self, encoder: Callable, decoder: Callable, process: DiscreteDenoisingProcess, /
    ):
        if not isinstance(process, DiscreteDenoisingProcess):
            raise TypeError("process must be DiscreteDenoisingProcess.")
        if not callable(encoder) or not callable(decoder):
            raise TypeError("encoder and decoder must be callable.")
        self.encoder = encoder
        self.decoder = decoder
        self.process = process
        self.embedding_id = canonical_fingerprint(
            {"kind": "hybrid-discrete-embedding", "process": process.process_id}
        )

    def encode(self, value: Any, /) -> Array:
        return jnp.asarray(self.encoder(value), dtype=jnp.int32)

    def decode(self, value: Array, /) -> Any:
        return self.decoder(value)


__all__ = [
    "AbstractDiscreteNoisingKernel",
    "AdaptiveMixingPenalty",
    "AdaptiveMixingState",
    "CategoricalNoisingKernel",
    "DiscreteDenoisingProcess",
    "DiscreteForwardProcess",
    "FactorGraphReverseKernel",
    "HybridDiscreteEmbedding",
    "RecoveryLikelihoodObjective",
]
