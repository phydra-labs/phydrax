#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from .._fingerprint import canonical_fingerprint
from .._iteration import (
    bind_iteration_scope,
    finalize_iteration,
    initialize_iteration,
    IterationCapabilities,
    IterationCoordinates,
    IterationEvidence,
    IterationPhase,
    IterationPlan,
    IterationRecord,
    update_iteration,
)
from .._sampling._adaptation import (
    adapt_proposal_scale,
    initialize_proposal_adaptation,
    RobbinsMonroScalePolicy,
)
from .._sampling._addressing import derive_key, SampleAddress
from .._sampling._chain import AbstractChainSampleResult
from .._sampling._hamiltonian import HamiltonianAdaptationPlan
from .._strict import StrictModule
from ..metrix import (
    AbstractStateGeometry,
    FlatTorusStateGeometry,
    LieAlgebraCoordinateMetric,
    LieGroupStateGeometry,
    PointwiseStateGeometry,
    SpecialUnitaryGroup,
    UnitaryGroup,
)


_MOMENTUM_ADDRESS = SampleAddress(
    "markov",
    "compact-group-hamiltonian",
    target="momentum",
    role="transition",
)
_ACCEPT_ADDRESS = SampleAddress(
    "markov",
    "compact-group-hamiltonian",
    target="acceptance",
    role="transition",
)


class CompactGeometricTarget(StrictModule):
    """One real target density relative to flat-torus or product-Haar measure."""

    evaluate: Callable[[Array], Array]
    geometry: AbstractStateGeometry
    configuration_shape: tuple[int, ...] = eqx.field(static=True)
    local_coordinate_shape: tuple[int, ...] = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)
    target_id: str = eqx.field(static=True)

    def __init__(
        self,
        evaluate: Callable[[Array], Array],
        geometry: AbstractStateGeometry,
        /,
        *,
        configuration_shape: tuple[int, ...],
        local_coordinate_shape: tuple[int, ...],
        reference_measure: str,
        target_id: str,
    ):
        if not callable(evaluate):
            raise TypeError("evaluate must be callable.")
        if not isinstance(geometry, AbstractStateGeometry):
            raise TypeError("geometry must implement AbstractStateGeometry.")
        point_shape = tuple(int(size) for size in configuration_shape)
        local_shape = tuple(int(size) for size in local_coordinate_shape)
        if not point_shape or any(size <= 0 for size in point_shape):
            raise ValueError("configuration_shape must contain positive dimensions.")
        if not local_shape or any(size <= 0 for size in local_shape):
            raise ValueError("local_coordinate_shape must contain positive dimensions.")
        if reference_measure not in ("flat-torus", "product-haar"):
            raise ValueError("Unsupported compact geometric reference measure.")
        identifier = str(target_id)
        if not identifier:
            raise ValueError("target_id must be non-empty.")
        self.evaluate = evaluate
        self.geometry = geometry
        self.configuration_shape = point_shape
        self.local_coordinate_shape = local_shape
        self.reference_measure = reference_measure
        self.target_id = identifier

    def __call__(self, position: ArrayLike, /) -> Array:
        value = jnp.asarray(self.evaluate(jnp.asarray(position)))
        if value.shape != () or jnp.iscomplexobj(value):
            raise ValueError("A compact geometric target must return one real scalar.")
        return value


class PreparedCompactGroupHamiltonianKernel(StrictModule):
    """Frozen fixed-step HMC over a flat torus or product compact Lie group."""

    target: CompactGeometricTarget
    coordinate_metric: LieAlgebraCoordinateMetric | None
    step_size: Array
    valid: Array
    leapfrog_steps: int = eqx.field(static=True)
    divergence_threshold: float = eqx.field(static=True)
    target_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    geometry_kind: str = eqx.field(static=True)


class CompactGroupHamiltonianChainState(StrictModule):
    position: Array
    log_target: Array
    gradient: Array
    step_index: Array
    valid: Array


class CompactGroupHamiltonianIterationMetrics(StrictModule):
    accepted: Array
    acceptance_probability: Array
    energy_error: Array
    divergent: Array
    nonfinite: Array
    membership_failure: Array
    leapfrog_steps: Array
    log_target: Array


class CompactGroupHamiltonianSampleResult(AbstractChainSampleResult):
    samples: Array
    log_target: Array
    accepted: Array
    acceptance_probability: Array
    energy_error: Array
    divergent: Array
    nonfinite: Array
    membership_failure: Array
    leapfrog_steps: Array
    final_state: CompactGroupHamiltonianChainState
    root_key: Array
    frozen_step_size: Array
    iteration_evidence: IterationEvidence | None
    target_id: str = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    @property
    def num_chains(self) -> int:
        return int(self.log_target.shape[0])

    @property
    def num_draws(self) -> int:
        return int(self.log_target.shape[1])

    @property
    def chain_provenance(self) -> str:
        return f"compact-group-hamiltonian:{self.kernel_id}:{self.target_id}"

    @property
    def acceptance_rate(self) -> Array:
        return jnp.mean(self.accepted.astype(float), axis=1)


class CompactGroupHamiltonianAdaptationResult(StrictModule):
    kernel: PreparedCompactGroupHamiltonianKernel
    final_state: CompactGroupHamiltonianChainState
    step_size_history: Array
    acceptance_history: Array
    valid: Array
    frozen: bool = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def _geometry_kind(target: CompactGeometricTarget, /) -> str:
    geometry = target.geometry
    if isinstance(geometry, FlatTorusStateGeometry):
        if target.configuration_shape != target.local_coordinate_shape:
            raise ValueError("Flat-torus point and local coordinate shapes must agree.")
        return "flat-torus"
    if isinstance(geometry, PointwiseStateGeometry) and isinstance(
        geometry.geometry, LieGroupStateGeometry
    ):
        return "pointwise-left-lie-group"
    raise TypeError(
        "Compact-group HMC supports FlatTorusStateGeometry or pointwise "
        "LieGroupStateGeometry."
    )


def prepare_compact_group_hamiltonian_kernel(
    target: CompactGeometricTarget,
    /,
    *,
    step_size: float,
    leapfrog_steps: int = 8,
    divergence_threshold: float = 1000.0,
) -> PreparedCompactGroupHamiltonianKernel:
    """Prepare one fixed-capacity product-Haar/flat-torus HMC kernel."""
    if not isinstance(target, CompactGeometricTarget):
        raise TypeError("target must be CompactGeometricTarget.")
    size = float(step_size)
    steps = int(leapfrog_steps)
    threshold = float(divergence_threshold)
    if not np.isfinite(size) or size <= 0.0:
        raise ValueError("step_size must be finite and positive.")
    if steps <= 0:
        raise ValueError("leapfrog_steps must be positive.")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("divergence_threshold must be finite and positive.")
    geometry_kind = _geometry_kind(target)
    if geometry_kind == "flat-torus" and target.reference_measure != "flat-torus":
        raise ValueError("Flat-torus HMC requires the flat-torus reference measure.")
    if (
        geometry_kind == "pointwise-left-lie-group"
        and target.reference_measure != "product-haar"
    ):
        raise ValueError("Matrix compact-group HMC requires product-Haar measure.")
    metric = None
    if geometry_kind == "pointwise-left-lie-group":
        geometry = target.geometry
        assert isinstance(geometry, PointwiseStateGeometry)
        group_geometry = geometry.geometry
        assert isinstance(group_geometry, LieGroupStateGeometry)
        if not isinstance(group_geometry.group, (UnitaryGroup, SpecialUnitaryGroup)):
            raise TypeError("Matrix compact-group HMC currently supports U(N) and SU(N).")
        metric = LieAlgebraCoordinateMetric(group_geometry.group)
        if target.local_coordinate_shape[-1] != metric.dimension:
            raise ValueError("Target local coordinates do not match the Lie algebra.")
    kernel_id = canonical_fingerprint(
        {
            "kind": "compact-group-hamiltonian-kernel",
            "target": target.target_id,
            "geometry": target.geometry.geometry_id,
            "metric": None if metric is None else metric.metric_id,
            "step_size": size,
            "leapfrog_steps": steps,
            "divergence_threshold": threshold,
        }
    )
    valid = jnp.asarray(True) if metric is None else metric.valid
    return PreparedCompactGroupHamiltonianKernel(
        target=target,
        coordinate_metric=metric,
        step_size=jnp.asarray(size),
        valid=valid,
        leapfrog_steps=steps,
        divergence_threshold=threshold,
        target_id=target.target_id,
        kernel_id=kernel_id,
        geometry_kind=geometry_kind,
    )


def _value_and_gradient(
    kernel: PreparedCompactGroupHamiltonianKernel,
    position: Array,
    /,
) -> tuple[Array, Array]:
    dtype = jnp.real(position).dtype
    zero = jnp.zeros(kernel.target.local_coordinate_shape, dtype=dtype)

    def local_log_target(local):
        point = kernel.target.geometry.retract(position, local)
        return kernel.target(point)

    value = kernel.target(position)
    gradient = jax.grad(local_log_target)(zero)
    return value, gradient


def initialize_compact_group_hamiltonian_state(
    kernel: PreparedCompactGroupHamiltonianKernel,
    initial_positions: ArrayLike,
    /,
) -> CompactGroupHamiltonianChainState:
    """Initialize explicit chain states from a leading chain axis."""
    if not isinstance(kernel, PreparedCompactGroupHamiltonianKernel):
        raise TypeError("kernel must be PreparedCompactGroupHamiltonianKernel.")
    positions = jnp.asarray(initial_positions)
    expected_rank = len(kernel.target.configuration_shape) + 1
    if (
        positions.ndim != expected_rank
        or positions.shape[1:] != kernel.target.configuration_shape
        or positions.shape[0] < 1
    ):
        raise ValueError(
            "initial_positions must have leading chain axis followed by the "
            f"configuration shape {kernel.target.configuration_shape}."
        )
    values, gradients = jax.vmap(lambda point: _value_and_gradient(kernel, point))(
        positions
    )
    membership = jax.vmap(kernel.target.geometry.contains)(positions)
    finite_positions = jnp.all(
        jnp.isfinite(positions).reshape((positions.shape[0], -1)), axis=1
    )
    finite_gradients = jnp.all(
        jnp.isfinite(gradients).reshape((gradients.shape[0], -1)), axis=1
    )
    valid = (
        kernel.valid
        & membership
        & finite_positions
        & jnp.isfinite(values)
        & finite_gradients
    )
    values = eqx.error_if(
        values,
        ~jnp.all(valid),
        "Initial compact-group HMC states must be finite group members.",
    )
    return CompactGroupHamiltonianChainState(
        position=positions,
        log_target=values,
        gradient=gradients,
        step_index=jnp.asarray(0, dtype=jnp.uint32),
        valid=valid,
    )


def _sample_momentum(
    kernel: PreparedCompactGroupHamiltonianKernel,
    key: Key[Array, ""],
    /,
) -> Array:
    dtype = kernel.step_size.dtype
    if kernel.coordinate_metric is None:
        return jr.normal(key, kernel.target.local_coordinate_shape, dtype=dtype)
    leading = kernel.target.local_coordinate_shape[:-1]
    return kernel.coordinate_metric.sample_momentum(key, leading, dtype=dtype)


def _velocity(
    kernel: PreparedCompactGroupHamiltonianKernel,
    momentum: Array,
    /,
) -> Array:
    if kernel.coordinate_metric is None:
        return momentum
    return kernel.coordinate_metric.solve(momentum)


def _kinetic(
    kernel: PreparedCompactGroupHamiltonianKernel,
    momentum: Array,
    /,
) -> Array:
    if kernel.coordinate_metric is None:
        return 0.5 * jnp.sum(momentum**2)
    return kernel.coordinate_metric.kinetic_energy(momentum)


def _trajectory(
    kernel: PreparedCompactGroupHamiltonianKernel,
    position: Array,
    gradient: Array,
    momentum: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
    def step(carry, _):
        q, p, g, active, used, nonfinite, membership_failure = carry
        half = p + 0.5 * kernel.step_size * g
        proposed_q = kernel.target.geometry.retract(
            q,
            kernel.step_size * _velocity(kernel, half),
        )
        proposed_value, proposed_gradient = _value_and_gradient(kernel, proposed_q)
        proposed_p = half + 0.5 * kernel.step_size * proposed_gradient
        member = kernel.target.geometry.contains(proposed_q)
        finite = (
            jnp.all(jnp.isfinite(proposed_q))
            & jnp.isfinite(proposed_value)
            & jnp.all(jnp.isfinite(proposed_gradient))
            & jnp.all(jnp.isfinite(proposed_p))
        )
        commit = active & member & finite
        return (
            jnp.where(commit, proposed_q, q),
            jnp.where(commit, proposed_p, p),
            jnp.where(commit, proposed_gradient, g),
            commit,
            used + active.astype(jnp.int32),
            nonfinite | (active & ~finite),
            membership_failure | (active & ~member),
        ), None

    initial = (
        position,
        momentum,
        gradient,
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(False),
        jnp.asarray(False),
    )
    (q, p, g, _, used, nonfinite, membership_failure), _ = jax.lax.scan(
        step,
        initial,
        None,
        length=kernel.leapfrog_steps,
    )
    return q, -p, g, kernel.target(q), used, nonfinite, membership_failure


def _one_transition(
    kernel: PreparedCompactGroupHamiltonianKernel,
    position: Array,
    log_target: Array,
    gradient: Array,
    state_valid: Array,
    key: Key[Array, ""],
    chain: Array,
    step_index: Array,
    /,
):
    momentum_key = derive_key(key, _MOMENTUM_ADDRESS, chain, step_index)
    accept_key = derive_key(key, _ACCEPT_ADDRESS, chain, step_index)
    momentum = _sample_momentum(kernel, momentum_key)
    proposed_q, proposed_p, proposed_g, proposed_value, used, nonfinite, membership = (
        _trajectory(kernel, position, gradient, momentum)
    )
    energy_error = (-proposed_value + _kinetic(kernel, proposed_p)) - (
        -log_target + _kinetic(kernel, momentum)
    )
    divergent = (
        nonfinite
        | membership
        | ~jnp.isfinite(energy_error)
        | (jnp.abs(energy_error) > kernel.divergence_threshold)
    )
    log_acceptance = jnp.minimum(-energy_error, 0.0)
    accepted = (
        state_valid & ~divergent & (jnp.log(jr.uniform(accept_key)) < log_acceptance)
    )
    return (
        jnp.where(accepted, proposed_q, position),
        jnp.where(accepted, proposed_value, log_target),
        jnp.where(accepted, proposed_g, gradient),
        accepted,
        jnp.where(
            state_valid & ~divergent & jnp.isfinite(log_acceptance),
            jnp.exp(log_acceptance),
            0.0,
        ),
        energy_error,
        divergent,
        nonfinite,
        membership,
        used,
    )


def _iteration_metrics(outputs, draw_index: int, /):
    return CompactGroupHamiltonianIterationMetrics(
        accepted=outputs[2][:, draw_index],
        acceptance_probability=outputs[3][:, draw_index],
        energy_error=outputs[4][:, draw_index],
        divergent=outputs[5][:, draw_index],
        nonfinite=outputs[6][:, draw_index],
        membership_failure=outputs[7][:, draw_index],
        leapfrog_steps=outputs[8][:, draw_index],
        log_target=outputs[1][:, draw_index],
    )


def sample_compact_group_hamiltonian(
    kernel: PreparedCompactGroupHamiltonianKernel,
    state: CompactGroupHamiltonianChainState,
    /,
    *,
    key: Key[Array, ""],
    num_draws: int,
    iteration: IterationPlan | None = None,
) -> CompactGroupHamiltonianSampleResult:
    """Advance persistent compact-group chains with a frozen HMC kernel."""
    if not isinstance(kernel, PreparedCompactGroupHamiltonianKernel):
        raise TypeError("kernel must be PreparedCompactGroupHamiltonianKernel.")
    if not isinstance(state, CompactGroupHamiltonianChainState):
        raise TypeError("state must be CompactGroupHamiltonianChainState.")
    if iteration is not None and not isinstance(iteration, IterationPlan):
        raise TypeError("iteration must be IterationPlan or None.")
    draws = int(num_draws)
    if draws <= 0:
        raise ValueError("num_draws must be positive.")
    chain_indices = jnp.arange(state.position.shape[0], dtype=jnp.uint32)

    def draw(carry, _):
        positions, values, gradients, valid, index = carry
        result = jax.vmap(
            lambda q, value, gradient, state_valid, chain: _one_transition(
                kernel,
                q,
                value,
                gradient,
                state_valid,
                key,
                chain,
                index,
            )
        )(positions, values, gradients, valid, chain_indices)
        next_positions, next_values, next_gradients = result[:3]
        return (
            next_positions,
            next_values,
            next_gradients,
            valid,
            index + jnp.asarray(1, dtype=jnp.uint32),
        ), (next_positions, next_values) + result[3:]

    (positions, values, gradients, valid, index), raw_outputs = jax.lax.scan(
        draw,
        (
            state.position,
            state.log_target,
            state.gradient,
            state.valid,
            state.step_index,
        ),
        None,
        length=draws,
    )
    outputs = tuple(jnp.swapaxes(value, 0, 1) for value in raw_outputs)
    samples, log_values = outputs[:2]
    final_membership = jax.vmap(kernel.target.geometry.contains)(positions)
    final = CompactGroupHamiltonianChainState(
        position=positions,
        log_target=values,
        gradient=gradients,
        step_index=index,
        valid=valid
        & kernel.valid
        & final_membership
        & jnp.isfinite(values)
        & jnp.all(jnp.isfinite(positions).reshape((positions.shape[0], -1)), axis=1)
        & jnp.all(jnp.isfinite(gradients).reshape((gradients.shape[0], -1)), axis=1),
    )

    iteration_evidence = None
    if iteration is not None:
        capabilities = IterationCapabilities(
            ("terminal", "step", "output"),
            mapped_records=True,
        )
        scope = bind_iteration_scope(
            iteration,
            capabilities,
            f"compact-group-hamiltonian:{kernel.kernel_id}",
        )
        zero_bool = jnp.zeros_like(state.valid)
        zero_float = jnp.zeros_like(state.log_target)
        zero_int = jnp.zeros_like(state.log_target, dtype=jnp.int32)
        initial = IterationRecord(
            IterationCoordinates(
                IterationPhase.START,
                state.step_index.astype(jnp.int32),
                active=state.valid,
                committed=jnp.zeros_like(state.valid),
            ),
            jnp.where(state.valid, 0, 1).astype(jnp.int32),
            CompactGroupHamiltonianIterationMetrics(
                zero_bool,
                zero_float,
                zero_float,
                zero_bool,
                zero_bool,
                zero_bool,
                zero_int,
                state.log_target,
            ),
        )
        observed = initialize_iteration(iteration, initial)
        if iteration.granularity in ("step", "output"):
            for draw_index in range(draws):
                active = state.valid & kernel.valid
                record = IterationRecord(
                    IterationCoordinates(
                        IterationPhase.COMMIT,
                        state.step_index.astype(jnp.int32) + draw_index + 1,
                        invocation=draw_index,
                        attempt=state.step_index.astype(jnp.int32) + draw_index + 1,
                        accepted=state.step_index.astype(jnp.int32) + draw_index + 1,
                        active=active,
                        committed=active,
                    ),
                    jnp.where(active, 0, 1).astype(jnp.int32),
                    _iteration_metrics(outputs, draw_index),
                )
                observed = update_iteration(iteration, observed, record, allow_stop=False)
        terminal = IterationRecord(
            IterationCoordinates(
                IterationPhase.TERMINAL,
                final.step_index.astype(jnp.int32),
                attempt=final.step_index.astype(jnp.int32),
                accepted=final.step_index.astype(jnp.int32),
                active=final.valid,
                committed=final.valid,
                terminal=True,
            ),
            jnp.where(final.valid, 0, 1).astype(jnp.int32),
            _iteration_metrics(outputs, draws - 1),
        )
        iteration_evidence = finalize_iteration(
            iteration,
            scope,
            capabilities,
            observed,
            terminal,
        )

    return CompactGroupHamiltonianSampleResult(
        samples=samples,
        log_target=log_values,
        accepted=outputs[2],
        acceptance_probability=outputs[3],
        energy_error=outputs[4],
        divergent=outputs[5],
        nonfinite=outputs[6],
        membership_failure=outputs[7],
        leapfrog_steps=outputs[8],
        final_state=final,
        root_key=jnp.asarray(key),
        frozen_step_size=kernel.step_size,
        iteration_evidence=iteration_evidence,
        target_id=kernel.target_id,
        kernel_id=kernel.kernel_id,
        claim="finite-fixed-step-product-haar-hamiltonian-chain",
    )


def _scale_policy(plan: HamiltonianAdaptationPlan, /) -> RobbinsMonroScalePolicy:
    return RobbinsMonroScalePolicy(
        target_acceptance=plan.target_acceptance,
        learning_rate=plan.adaptation_rate,
        decay_power=0.5,
        minimum_scale=plan.minimum_step_size,
        maximum_scale=plan.maximum_step_size,
        warmup_chunks=plan.warmup_steps,
    )


def adapt_compact_group_hamiltonian(
    kernel: PreparedCompactGroupHamiltonianKernel,
    state: CompactGroupHamiltonianChainState,
    plan: HamiltonianAdaptationPlan,
    /,
    *,
    key: Key[Array, ""],
) -> CompactGroupHamiltonianAdaptationResult:
    """Run finite step-size warmup and return one frozen group-HMC kernel."""
    if not isinstance(kernel, PreparedCompactGroupHamiltonianKernel):
        raise TypeError("kernel must be PreparedCompactGroupHamiltonianKernel.")
    if not isinstance(state, CompactGroupHamiltonianChainState):
        raise TypeError("state must be CompactGroupHamiltonianChainState.")
    if not isinstance(plan, HamiltonianAdaptationPlan):
        raise TypeError("plan must be HamiltonianAdaptationPlan.")
    scale_policy = _scale_policy(plan)
    adaptive = initialize_proposal_adaptation(scale_policy, kernel.step_size)
    current = state
    adapted = kernel
    sizes = []
    acceptances = []
    for _ in range(plan.warmup_steps):
        adapted = eqx.tree_at(
            lambda value: value.step_size,
            adapted,
            adaptive.scale,
        )
        draw = sample_compact_group_hamiltonian(
            adapted,
            current,
            key=key,
            num_draws=1,
        )
        acceptance = jnp.mean(draw.acceptance_probability)
        sizes.append(adaptive.scale)
        acceptances.append(acceptance)
        adaptive = adapt_proposal_scale(scale_policy, adaptive, acceptance)
        current = draw.final_state
    adapted = eqx.tree_at(
        lambda value: value.step_size,
        adapted,
        adaptive.scale,
    )
    size_history = (
        jnp.stack(sizes) if sizes else jnp.empty((0,), dtype=kernel.step_size.dtype)
    )
    acceptance_history = (
        jnp.stack(acceptances)
        if acceptances
        else jnp.empty((0,), dtype=kernel.step_size.dtype)
    )
    return CompactGroupHamiltonianAdaptationResult(
        kernel=adapted,
        final_state=current,
        step_size_history=size_history,
        acceptance_history=acceptance_history,
        valid=(
            adapted.valid
            & jnp.all(current.valid)
            & adaptive.valid
            & jnp.all(jnp.isfinite(size_history))
            & jnp.all(jnp.isfinite(acceptance_history))
        ),
        frozen=True,
        claim="finite-warmup-only-production-frozen",
    )


__all__ = [
    "CompactGeometricTarget",
    "CompactGroupHamiltonianAdaptationResult",
    "CompactGroupHamiltonianChainState",
    "CompactGroupHamiltonianIterationMetrics",
    "CompactGroupHamiltonianSampleResult",
    "PreparedCompactGroupHamiltonianKernel",
    "adapt_compact_group_hamiltonian",
    "initialize_compact_group_hamiltonian_state",
    "prepare_compact_group_hamiltonian_kernel",
    "sample_compact_group_hamiltonian",
]
