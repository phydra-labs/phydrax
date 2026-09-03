#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Named finite VMC amplitudes with explicit local caches and no registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from phydrax.ein import contract

from ..._sampling import derive_key, SampleAddress
from ..._sampling._targets import IncrementalMarkovTarget
from ..._strict import StrictModule
from ...linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ...operators.quantum._amplitude import LogAmplitude
from ...tensor_network import MatrixProductState


_DIRECT_SAMPLE_ADDRESS = SampleAddress(
    "quantum", "autoregressive-spin", target="spin", role="direct-sample"
)


def _spin_configuration(configuration: ArrayLike, count: int, /) -> Array:
    spins = jnp.asarray(configuration)
    if spins.shape != (count,):
        raise ValueError(f"spin configuration must have shape ({count},).")
    return spins


def _from_complex_log(value: Array, valid: Array) -> LogAmplitude:
    return LogAmplitude(
        jnp.real(value),
        jnp.exp(1j * jnp.imag(value)),
        valid=valid & jnp.isfinite(value),
    )


class JastrowSpinCache(StrictModule):
    spins: Array
    local_fields: Array
    complex_log_amplitude: Array
    valid: Array


class JastrowSpinAmplitude(StrictModule):
    """Finite complex symmetric spin Jastrow amplitude."""

    fields: Array
    couplings: Array
    site_count: int = eqx.field(static=True)

    def __init__(self, fields: ArrayLike, couplings: ArrayLike, /):
        h, matrix = jnp.asarray(fields), jnp.asarray(couplings)
        if h.ndim != 1 or matrix.shape != (h.shape[0], h.shape[0]) or h.size < 1:
            raise ValueError(
                "fields/couplings require shapes (sites,) and (sites,sites)."
            )
        self.fields = h.astype(jnp.result_type(h.dtype, matrix.dtype, 1j))
        self.couplings = 0.5 * (matrix + matrix.T).astype(self.fields.dtype)
        self.site_count = int(h.shape[0])

    def initialize_cache(self, configuration: ArrayLike, /) -> JastrowSpinCache:
        spins = _spin_configuration(configuration, self.site_count)
        valid = jnp.all(jnp.isfinite(spins)) & jnp.all(jnp.abs(spins) == 1)
        local = self.fields + self.couplings @ spins
        value = contract("i,i->", self.fields, spins) + 0.5 * contract(
            "i,ij,j->", spins, self.couplings, spins
        )
        return JastrowSpinCache(
            spins=spins, local_fields=local, complex_log_amplitude=value, valid=valid
        )

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        cache = self.initialize_cache(configuration)
        return _from_complex_log(cache.complex_log_amplitude, cache.valid)

    def propose_flips(
        self,
        cache: JastrowSpinCache,
        indices: ArrayLike,
        active: ArrayLike,
        /,
    ) -> tuple[Array, JastrowSpinCache, Array]:
        sites, mask = (
            jnp.asarray(indices, dtype=jnp.int32),
            jnp.asarray(active, dtype=bool),
        )
        if sites.ndim != 1 or mask.shape != sites.shape:
            raise ValueError("indices/active must be matching fixed-capacity vectors.")
        toggles = (
            jnp.zeros((self.site_count,), dtype=jnp.int32)
            .at[sites]
            .add(mask.astype(jnp.int32))
        )
        proposed_spins = cache.spins * jnp.where(toggles % 2 == 1, -1, 1)
        proposed = self.initialize_cache(proposed_spins)
        ratio = proposed.complex_log_amplitude - cache.complex_log_amplitude
        return ratio, proposed, cache.valid & proposed.valid


class RestrictedBoltzmannCache(StrictModule):
    spins: Array
    hidden_preactivation: Array
    complex_log_amplitude: Array
    valid: Array


class RestrictedBoltzmannAmplitude(StrictModule):
    """Finite complex RBM amplitude with exact hidden-preactivation flip cache."""

    visible_bias: Array
    hidden_bias: Array
    weights: Array
    site_count: int = eqx.field(static=True)
    hidden_count: int = eqx.field(static=True)

    def __init__(
        self, visible_bias: ArrayLike, hidden_bias: ArrayLike, weights: ArrayLike, /
    ):
        visible, hidden, matrix = map(jnp.asarray, (visible_bias, hidden_bias, weights))
        if (
            visible.ndim != 1
            or hidden.ndim != 1
            or matrix.shape != (hidden.shape[0], visible.shape[0])
        ):
            raise ValueError(
                "RBM shapes must be visible=(sites,), hidden=(hidden,), weights=(hidden,sites)."
            )
        dtype = jnp.result_type(visible.dtype, hidden.dtype, matrix.dtype, 1j)
        self.visible_bias = visible.astype(dtype)
        self.hidden_bias = hidden.astype(dtype)
        self.weights = matrix.astype(dtype)
        self.site_count = int(visible.shape[0])
        self.hidden_count = int(hidden.shape[0])

    def initialize_cache(self, configuration: ArrayLike, /) -> RestrictedBoltzmannCache:
        spins = _spin_configuration(configuration, self.site_count)
        hidden = self.hidden_bias + self.weights @ spins
        value = contract("i,i->", self.visible_bias, spins) + jnp.sum(
            jnp.log(2.0 * jnp.cosh(hidden))
        )
        valid = (
            jnp.all(jnp.isfinite(spins))
            & jnp.all(jnp.abs(spins) == 1)
            & jnp.isfinite(value)
        )
        return RestrictedBoltzmannCache(
            spins=spins,
            hidden_preactivation=hidden,
            complex_log_amplitude=value,
            valid=valid,
        )

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        cache = self.initialize_cache(configuration)
        return _from_complex_log(cache.complex_log_amplitude, cache.valid)

    def propose_flips(
        self, cache: RestrictedBoltzmannCache, indices: ArrayLike, active: ArrayLike, /
    ):
        sites, mask = (
            jnp.asarray(indices, dtype=jnp.int32),
            jnp.asarray(active, dtype=bool),
        )
        if sites.ndim != 1 or mask.shape != sites.shape:
            raise ValueError("indices/active must be matching fixed-capacity vectors.")
        toggles = (
            jnp.zeros((self.site_count,), dtype=jnp.int32)
            .at[sites]
            .add(mask.astype(jnp.int32))
        )
        proposed_spins = cache.spins * jnp.where(toggles % 2 == 1, -1, 1)
        proposed = self.initialize_cache(proposed_spins)
        return (
            proposed.complex_log_amplitude - cache.complex_log_amplitude,
            proposed,
            cache.valid & proposed.valid,
        )


def _spin_cache_incremental_target(model, target_id: str):
    def initialize(position):
        cache = model.initialize_cache(position)
        return 2.0 * jnp.real(cache.complex_log_amplitude), cache

    def propose(current, cache, proposed_position, payload):
        indices, active = payload
        ratio, proposed_cache, valid = model.propose_flips(
            cache,
            indices,
            active,
        )
        residual = jnp.max(jnp.abs(proposed_cache.spins - proposed_position))
        return (
            2.0 * jnp.real(ratio),
            proposed_cache,
            valid & (residual == 0.0),
        )

    def select(current, proposed, accepted):
        return jax.tree_util.tree_map(
            lambda proposed_leaf, current_leaf: jnp.where(
                accepted, proposed_leaf, current_leaf
            ),
            proposed,
            current,
        )

    return IncrementalMarkovTarget(
        initialize=initialize,
        propose=propose,
        select=select,
        refresh=initialize,
        target_id=target_id,
        refresh_cadence=32,
    )


def jastrow_incremental_target(
    model: JastrowSpinAmplitude,
    /,
    *,
    target_id: str = "jastrow-spin",
) -> IncrementalMarkovTarget:
    if not isinstance(model, JastrowSpinAmplitude):
        raise TypeError("model must be JastrowSpinAmplitude.")
    return _spin_cache_incremental_target(model, target_id)


def rbm_incremental_target(
    model: RestrictedBoltzmannAmplitude,
    /,
    *,
    target_id: str = "restricted-boltzmann",
) -> IncrementalMarkovTarget:
    if not isinstance(model, RestrictedBoltzmannAmplitude):
        raise TypeError("model must be RestrictedBoltzmannAmplitude.")
    return _spin_cache_incremental_target(model, target_id)


class AutoregressiveSpinAmplitude(StrictModule):
    """Exactly normalized finite ordered binary-spin amplitude."""

    conditional_bias: Array
    conditional_weights: Array
    phase_bias: Array
    phase_weights: Array
    site_count: int = eqx.field(static=True)

    def __init__(
        self,
        conditional_bias,
        conditional_weights,
        /,
        *,
        phase_bias=None,
        phase_weights=None,
    ):
        bias, weights = jnp.asarray(conditional_bias), jnp.asarray(conditional_weights)
        if bias.ndim != 1 or weights.shape != (bias.shape[0], bias.shape[0]):
            raise ValueError(
                "Autoregressive bias/weights require (sites,) and (sites,sites)."
            )
        count = int(bias.shape[0])
        phases = jnp.zeros_like(bias) if phase_bias is None else jnp.asarray(phase_bias)
        phase_matrix = (
            jnp.zeros_like(weights)
            if phase_weights is None
            else jnp.asarray(phase_weights)
        )
        if phases.shape != bias.shape or phase_matrix.shape != weights.shape:
            raise ValueError("Autoregressive phase arrays must match probability arrays.")
        lower = jnp.tril(weights, k=-1)
        phase_lower = jnp.tril(phase_matrix, k=-1)
        self.conditional_bias = bias
        self.conditional_weights = lower
        self.phase_bias = phases
        self.phase_weights = phase_lower
        self.site_count = count

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        spins = _spin_configuration(configuration, self.site_count)
        logits = self.conditional_bias + self.conditional_weights @ spins
        log_probability = jnp.sum(jax.nn.log_sigmoid(spins * logits))
        phase_angle = jnp.sum(spins * (self.phase_bias + self.phase_weights @ spins))
        valid = (
            jnp.all(jnp.abs(spins) == 1)
            & jnp.isfinite(log_probability)
            & jnp.isfinite(phase_angle)
        )
        return LogAmplitude(0.5 * log_probability, jnp.exp(1j * phase_angle), valid=valid)

    def sample(self, key: Key[Array, ""], /, *, sample_index: int = 0) -> Array:
        spins = jnp.zeros((self.site_count,), dtype=self.conditional_bias.dtype)
        for site in range(self.site_count):
            site_key = derive_key(key, _DIRECT_SAMPLE_ADDRESS, sample_index, site)
            logit = self.conditional_bias[site] + contract(
                "i,i->", self.conditional_weights[site], spins
            )
            positive = jr.bernoulli(site_key, jax.nn.sigmoid(logit))
            spins = spins.at[site].set(jnp.where(positive, 1.0, -1.0))
        return spins


class SlaterJastrowAmplitude(StrictModule):
    """Finite Slater determinant plus caller-supplied cusp-declared Jastrow."""

    orbital_evaluator: Callable[[Array], Array] = eqx.field(static=True)
    jastrow: Callable[[Array], Array] = eqx.field(static=True)
    electron_count: int = eqx.field(static=True)
    cusp_id: str = eqx.field(static=True)

    def __init__(
        self,
        orbital_evaluator: Callable,
        jastrow: Callable,
        /,
        *,
        electron_count: int,
        cusp_id: str,
    ):
        if not callable(orbital_evaluator) or not callable(jastrow):
            raise TypeError("orbital_evaluator and jastrow must be callable.")
        count = int(electron_count)
        if count <= 0 or not isinstance(cusp_id, str) or not cusp_id:
            raise ValueError("electron_count and cusp_id must be declared.")
        self.orbital_evaluator = orbital_evaluator
        self.jastrow = jastrow
        self.electron_count = count
        self.cusp_id = cusp_id

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        coordinates = jnp.asarray(configuration)
        if coordinates.ndim != 2 or coordinates.shape[0] != self.electron_count:
            raise ValueError("configuration must have shape (electron_count, dimension).")
        matrix = jnp.asarray(self.orbital_evaluator(coordinates))
        if matrix.shape != (self.electron_count, self.electron_count):
            raise ValueError("orbital_evaluator must return a square Slater matrix.")
        prepared = factorize(DenseLinearOperator(matrix), FactorizationPolicy("lu"))
        log_abs = prepared.log_abs_determinant()
        phase = prepared.determinant_sign().astype(jnp.result_type(matrix.dtype, 1j))
        correlation = jnp.asarray(self.jastrow(coordinates))
        if correlation.shape != ():
            raise ValueError("jastrow must return one scalar complex log factor.")
        value = log_abs + jnp.real(correlation)
        total_phase = phase * jnp.exp(1j * jnp.imag(correlation))
        valid = (
            jnp.all(jnp.isfinite(matrix)) & jnp.isfinite(value) & (jnp.abs(phase) > 0.0)
        )
        return LogAmplitude(value, total_phase, valid=valid)


class CircuitAmplitude(StrictModule):
    """Finite basis-amplitude adapter over the canonical dense QuantumProgram."""

    prepared: Any
    initial_state: Array
    maximum_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        prepared,
        initial_state: ArrayLike,
        /,
        *,
        maximum_dimension: int,
    ):
        from ...solver._quantum_program import PreparedDenseQuantumProgram

        if not isinstance(prepared, PreparedDenseQuantumProgram):
            raise TypeError("prepared must be PreparedDenseQuantumProgram.")
        if prepared.plan.state_kind != "state-vector":
            raise ValueError("CircuitAmplitude requires a state-vector program.")
        maximum = int(maximum_dimension)
        if prepared.plan.cost.total_dimension > maximum or maximum <= 0:
            raise ValueError("CircuitAmplitude finite Hilbert capacity exceeded.")
        state = jnp.asarray(initial_state)
        if state.shape != (prepared.plan.cost.total_dimension,):
            raise ValueError("initial_state dimension does not match the program.")
        self.prepared = prepared
        self.initial_state = state
        self.maximum_dimension = maximum

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        from ...solver._quantum_program import execute_dense_quantum_program

        digits = jnp.asarray(configuration, dtype=jnp.int32)
        layout = self.prepared.plan.layout
        if digits.shape != (layout.wire_count,):
            raise ValueError("configuration must select one basis value per wire.")
        index = jnp.asarray(0, dtype=jnp.int32)
        valid = jnp.asarray(True)
        for digit, dimension in zip(digits, layout.local_dimensions, strict=True):
            valid = valid & (digit >= 0) & (digit < dimension)
            index = index * dimension + digit
        result = execute_dense_quantum_program(
            self.prepared,
            self.initial_state,
        )
        amplitude = result.final_state[index]
        magnitude = jnp.abs(amplitude)
        phase = jnp.where(
            magnitude > 0.0,
            amplitude / magnitude,
            1.0 + 0.0j,
        )
        return LogAmplitude(
            jnp.where(magnitude > 0.0, jnp.log(magnitude), -jnp.inf),
            phase,
            valid=valid & result.diagnostics.successful & jnp.isfinite(amplitude),
        )


class TensorNetworkAmplitude(StrictModule):
    """Direct finite-configuration contraction of an existing MPS."""

    state: MatrixProductState

    def __call__(self, configuration: ArrayLike, /) -> LogAmplitude:
        indices = jnp.asarray(configuration, dtype=jnp.int32)
        if indices.shape != (self.state.site_count,):
            raise ValueError("configuration must select one physical index per MPS site.")
        environment = self.state.tensors[0][0, indices[0], :]
        valid = (indices[0] >= 0) & (indices[0] < self.state.physical_dimensions[0])
        for site, tensor in enumerate(self.state.tensors[1:], start=1):
            valid = (
                valid
                & (indices[site] >= 0)
                & (indices[site] < self.state.physical_dimensions[site])
            )
            environment = environment @ tensor[:, indices[site], :]
        amplitude = environment[0]
        magnitude = jnp.abs(amplitude)
        phase = jnp.where(magnitude > 0.0, amplitude / magnitude, 1.0 + 0.0j)
        return LogAmplitude(
            jnp.where(magnitude > 0.0, jnp.log(magnitude), -jnp.inf),
            phase,
            valid=valid & jnp.isfinite(amplitude),
        )


__all__ = [
    "AutoregressiveSpinAmplitude",
    "CircuitAmplitude",
    "JastrowSpinAmplitude",
    "JastrowSpinCache",
    "RestrictedBoltzmannAmplitude",
    "jastrow_incremental_target",
    "rbm_incremental_target",
    "RestrictedBoltzmannCache",
    "SlaterJastrowAmplitude",
    "TensorNetworkAmplitude",
]
