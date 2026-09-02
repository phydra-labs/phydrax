#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._trainable import combine_trainable
from ..domain import BatchEvaluator, DomainFunction, GridBatch, PointBatch
from ..integration import (
    AdaptiveIntegration,
    FixedIntegration,
    PerStepIntegration,
    PointIntegrationBatch,
)
from ..linalg import ArraySpace, stochastic_trace
from ..nn.neural_tangent import (
    analyze_ntk,
    NTKDiagnosticsPolicy,
    prepare_empirical_ntk,
)
from ..terms import ResidualBlockRef, ResidualPenalty
from ._functional_objective import (
    _ObjectiveValues,
    _PreparedObjective,
    evaluate_prepared_objective,
    evaluate_prepared_scalar_remainder,
)
from ._functional_residual import (
    evaluate_prepared_residual_term,
    prepare_functional_residual,
    PreparedFunctionalResidual,
    ResidualRootBlock,
)
from ._functional_training import FunctionalTrainingPlan, PseudoTransientPolicy


class _BlockScaledEvaluator(StrictModule, BatchEvaluator):
    source: DomainFunction
    weights: Array
    blocks: Any

    def __init__(self, source: DomainFunction, weights: Any, blocks: Any, /):
        self.source = source
        self.weights = jnp.asarray(weights)
        self.blocks = blocks

    def _scale(self, data: Array, dims: tuple[str | None, ...], /) -> Array:
        if self.weights.ndim == 0:
            return data * self.weights
        if self.blocks is None:
            raise ValueError(
                "Vector pseudo-time inverse steps require a ResidualBlockLayout."
            )
        if self.weights.shape != (self.blocks.block_count,):
            raise ValueError(
                "Pseudo-time inverse steps must align with residual block names."
            )
        event_positions = tuple(
            index for index, dimension in enumerate(dims) if dimension is None
        )
        if self.blocks.event_axis >= len(event_positions):
            raise ValueError("Pseudo-time block event axis is unavailable.")
        axis = event_positions[self.blocks.event_axis]
        expanded = jnp.repeat(self.weights, jnp.asarray(self.blocks.sizes))
        shape = [1] * data.ndim
        shape[axis] = self.blocks.event_size
        return data * expanded.reshape(tuple(shape))

    def __call_batch__(
        self,
        batch: PointBatch | GridBatch,
        /,
        *,
        key=None,
        **kwargs: Any,
    ) -> cx.Field:
        value = self.source(batch, key=key, **kwargs)
        if not isinstance(value, cx.Field):
            raise TypeError("Pseudo-time source evaluation must return a coordax.Field.")
        return cx.Field(self._scale(jnp.asarray(value.data), value.dims), dims=value.dims)

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        value = jnp.asarray(self.source.func(*args, key=key, **kwargs))
        dims = (None,) * value.ndim
        return self._scale(value, dims)


def _scale_domain_function(
    source: DomainFunction,
    weights: Any,
    blocks: Any,
    /,
) -> DomainFunction:
    return DomainFunction(
        domain=source.domain,
        deps=source.deps,
        func=_BlockScaledEvaluator(source, weights, blocks),
        metadata=source.metadata,
    )


class PseudoTransientResidualTransform(StrictModule):
    previous_functions: Mapping[str, DomainFunction]
    policies: tuple[PseudoTransientPolicy, ...]
    inverse_steps: tuple[Array, ...]
    enforcement: Any

    def __init__(
        self,
        previous_functions: Mapping[str, DomainFunction],
        policies: Sequence[PseudoTransientPolicy],
        inverse_steps: Sequence[Any],
        enforcement: Any,
        /,
    ):
        policies_ = tuple(policies)
        steps = tuple(jnp.asarray(value) for value in inverse_steps)
        if len(policies_) != len(steps):
            raise ValueError("Pseudo-time policies and inverse steps must align.")
        self.previous_functions = previous_functions
        self.policies = policies_
        self.inverse_steps = steps
        self.enforcement = enforcement

    def _policy(self, term_index: int, /):
        selected = tuple(
            (policy, inverse_step)
            for policy, inverse_step in zip(
                self.policies, self.inverse_steps, strict=True
            )
            if policy.term_index == int(term_index)
        )
        if len(selected) > 1:
            raise RuntimeError("Multiple pseudo-time policies target one residual term.")
        return None if not selected else selected[0]

    def residual_override(self, params, residual, term):
        selected = self._policy(term.index)
        current_functions = combine_trainable(params, residual.non_trainable)
        current = (
            current_functions
            if self.enforcement is None
            else self.enforcement.apply(current_functions)
        )
        physical = term.term.condition.residual(current)
        if selected is None:
            return physical
        policy, inverse_step = selected
        previous = (
            self.previous_functions
            if self.enforcement is None
            else self.enforcement.apply(self.previous_functions)
        )
        current_state = policy.relaxation.field(current)
        previous_state = policy.relaxation.field(previous)
        state_change = current_state - previous_state
        return physical + _scale_domain_function(
            state_change,
            inverse_step,
            policy.relaxation.blocks,
        )

    def term_blocks(self, params, residual, term):
        from ._functional_residual import evaluate_prepared_residual_term

        return evaluate_prepared_residual_term(
            params,
            residual.non_trainable,
            residual.enforcement,
            term,
            iteration=residual.iteration,
            residual_override=self.residual_override(params, residual, term),
        )

class _CausalScaledEvaluator(StrictModule, BatchEvaluator):
    source: DomainFunction
    gates: tuple[cx.Field, ...]
    blocks: Any

    def __init__(
        self,
        source: DomainFunction,
        gates: Sequence[cx.Field],
        blocks: Any,
        /,
    ):
        gates_ = tuple(gates)
        if not gates_ or any(not isinstance(gate, cx.Field) for gate in gates_):
            raise TypeError("Causal gates must be coordax.Field values.")
        if blocks is None and len(gates_) != 1:
            raise ValueError("An unnamed residual accepts one shared causal gate.")
        if blocks is not None and len(gates_) not in (1, blocks.block_count):
            raise ValueError("Causal gates must be shared or align with residual blocks.")
        self.source = source
        self.gates = gates_
        self.blocks = blocks

    def __call_batch__(self, batch, /, *, key=None, **kwargs: Any) -> cx.Field:
        value = self.source(batch, key=key, **kwargs)
        if not isinstance(value, cx.Field):
            raise TypeError("Causal residual evaluation must return a coordax.Field.")
        if self.blocks is None:
            return value * cx.Field(
                jnp.sqrt(jnp.asarray(self.gates[0].data)),
                dims=self.gates[0].dims,
            )
        event_positions = tuple(
            index for index, dimension in enumerate(value.dims) if dimension is None
        )
        axis = event_positions[self.blocks.event_axis]
        pieces = self.blocks.split(value)
        scaled = tuple(
            block
            * cx.Field(jnp.sqrt(jnp.asarray(gate.data)), dims=gate.dims)
            for block, gate in zip(
                pieces,
                self.gates if len(self.gates) > 1 else self.gates * len(pieces),
                strict=True,
            )
        )
        return cx.Field(
            jnp.concatenate(tuple(block.data for block in scaled), axis=axis),
            dims=value.dims,
        )

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        raise TypeError("Causal residual transforms require a prepared batch.")


def _causal_domain_function(
    source: DomainFunction,
    gates: Sequence[cx.Field],
    blocks: Any,
    /,
) -> DomainFunction:
    return DomainFunction(
        domain=source.domain,
        deps=source.deps,
        func=_CausalScaledEvaluator(source, gates, blocks),
        metadata=source.metadata,
    )


class CausalResidualTransform(StrictModule):
    inner: Any
    policies: tuple[Any, ...]
    gates: tuple[tuple[int, tuple[cx.Field, ...]], ...]

    def __init__(self, inner: Any, policies: Sequence[Any], gates, /):
        self.inner = inner
        self.policies = tuple(policies)
        self.gates = tuple(gates)

    def _policy(self, term_index: int, /):
        selected = tuple(
            policy for policy in self.policies if policy.term_index == int(term_index)
        )
        return None if not selected else selected[0]

    def _gates(self, term_index: int, /):
        selected = tuple(gates for index, gates in self.gates if index == int(term_index))
        return None if not selected else selected[0]

    def residual_override(self, params, residual, term):
        if self.inner is None:
            current_functions = combine_trainable(params, residual.non_trainable)
            current = (
                current_functions
                if residual.enforcement is None
                else residual.enforcement.apply(current_functions)
            )
            base = term.term.condition.residual(current)
        else:
            base = self.inner.residual_override(params, residual, term)
        policy = self._policy(term.index)
        if policy is None:
            return base
        gates = self._gates(term.index)
        if gates is None:
            raise RuntimeError("Causal residual gates were not prepared.")
        return _causal_domain_function(base, gates, term.term.blocks)

    def term_blocks(self, params, residual, term):
        from ._functional_residual import evaluate_prepared_residual_term

        return evaluate_prepared_residual_term(
            params,
            residual.non_trainable,
            residual.enforcement,
            term,
            iteration=residual.iteration,
            residual_override=self.residual_override(params, residual, term),
        )


def _squared_residual_field(value: cx.Field, /) -> cx.Field:
    data = jnp.real(jnp.conj(value.data) * value.data)
    dims = value.dims
    for axis in reversed(
        tuple(index for index, dimension in enumerate(dims) if dimension is None)
    ):
        data = jnp.sum(data, axis=axis)
        dims = dims[:axis] + dims[axis + 1 :]
    return cx.Field(data, dims=dims)


def _causal_gate_fields(score, coefficient, time, schedule, /):
    times = jnp.asarray(time.data)
    masks = tuple(
        cx.Field(
            (
                (times >= schedule.bounds(index)[0])
                & (
                    times <= schedule.bounds(index)[1]
                    if index == schedule.slab_count - 1
                    else times < schedule.bounds(index)[1]
                )
            ).astype(float),
            dims=time.dims,
        )
        for index in range(schedule.slab_count)
    )
    coverage = sum(
        (jnp.asarray(mask.data) for mask in masks),
        start=jnp.zeros_like(jnp.asarray(masks[0].data)),
    )
    if not bool(jnp.all(coverage == 1.0)):
        raise ValueError("Causal slabs must partition every prepared collocation point.")
    losses = []
    for mask in masks:
        weighted = coefficient * mask
        support = jnp.sum(jnp.asarray(weighted.data))
        if not bool(jnp.isfinite(support)) or float(support) <= 0.0:
            raise ValueError("Every causal slab requires positive finite support.")
        numerator = jnp.sum(jnp.asarray((weighted * score).data))
        losses.append(numerator / support)
    gates = schedule.causal_weights(jnp.stack(tuple(losses)))
    multiplier = sum(
        (
            mask * jnp.asarray(gates[index])
            for index, mask in enumerate(masks)
        ),
        start=cx.Field(jnp.zeros_like(masks[0].data), dims=masks[0].dims),
    )
    return multiplier


def _prepare_causal_gates(residual, params, policies, inner, /):
    current_functions = combine_trainable(params, residual.non_trainable)
    current = (
        current_functions
        if residual.enforcement is None
        else residual.enforcement.apply(current_functions)
    )
    prepared_gates = []
    for policy in policies:
        matching = tuple(
            term for term in residual.terms if term.index == policy.term_index
        )
        if len(matching) != 1:
            raise ValueError("Causal policy must select one prepared residual term.")
        term = matching[0]
        batch = term.realization.batch
        if not isinstance(batch, PointIntegrationBatch):
            raise TypeError("Causal residual training initially requires point integration.")
        time = batch.points[policy.time_label]
        if not isinstance(time, cx.Field) or any(
            dimension is None for dimension in time.dims
        ):
            raise TypeError("Causal time coordinates must be scalar named-axis fields.")
        if policy.gate_signal == "surrogate" and inner is not None:
            override = inner.residual_override(params, residual, term)
        else:
            override = term.term.condition.residual(current)
        data = term.term._quadratic_residual_data(
            current,
            override,
            realization=term.realization,
            iter_=residual.iteration,
        )
        if len(data.residuals) != 1:
            raise TypeError("Causal residual training does not accept component unions.")
        residual_value = data.residuals[0]
        coefficient = data.coefficients[0]
        blocks = (
            (residual_value,)
            if term.term.blocks is None
            else term.term.blocks.split(residual_value)
        )
        if policy.per_block:
            gates = tuple(
                _causal_gate_fields(
                    _squared_residual_field(block),
                    coefficient,
                    time,
                    policy.schedule,
                )
                for block in blocks
            )
        else:
            gates = (
                _causal_gate_fields(
                    _squared_residual_field(residual_value),
                    coefficient,
                    time,
                    policy.schedule,
                ),
            )
        prepared_gates.append((term.index, gates))
    return tuple(prepared_gates)

class BalancedResidualTransform(StrictModule):
    inner: Any
    references: tuple[Any, ...]
    multipliers: Array

    def __init__(self, inner: Any, references: Sequence[Any], multipliers: Any, /):
        references_ = tuple(references)
        values = jnp.asarray(multipliers, dtype=float).reshape((-1,))
        if len(references_) != int(values.size):
            raise ValueError("Balance references and multipliers must align.")
        self.inner = inner
        self.references = references_
        self.multipliers = values

    def term_blocks(self, params, residual, term):
        if self.inner is None:
            blocks = evaluate_prepared_residual_term(
                params,
                residual.non_trainable,
                residual.enforcement,
                term,
                iteration=residual.iteration,
            )
        else:
            blocks = self.inner.term_blocks(params, residual, term)
        scaled = []
        for block in blocks:
            multiplier = jnp.asarray(1.0, dtype=block.values.dtype)
            for reference, value in zip(
                self.references, self.multipliers, strict=True
            ):
                matches = reference.term_index == block.term_index and (
                    reference.block_name is None
                    or reference.block_name == block.block_name
                )
                multiplier = jnp.where(matches, value, multiplier)
            scaled.append(
                ResidualRootBlock(
                    jnp.sqrt(multiplier) * block.values,
                    term_index=block.term_index,
                    term_label=block.term_label,
                    block_name=block.block_name,
                    source_index=block.source_index,
                    coordinate_kind=block.coordinate_kind,
                    event_shape=block.event_shape,
                )
            )
        return tuple(scaled)


def _tree_norm(tree, /) -> Array:
    leaves = tuple(
        leaf for leaf in jax.tree.leaves(tree) if eqx.is_inexact_array(leaf)
    )
    if not leaves:
        return jnp.asarray(0.0)
    return jnp.sqrt(
        sum(
            (jnp.real(jnp.vdot(leaf, leaf)) for leaf in leaves),
            start=jnp.asarray(0.0),
        )
    )

def _tree_inner(left, right, /) -> Array:
    left_leaves = tuple(
        leaf for leaf in jax.tree.leaves(left) if eqx.is_inexact_array(leaf)
    )
    right_leaves = tuple(
        leaf for leaf in jax.tree.leaves(right) if eqx.is_inexact_array(leaf)
    )
    if len(left_leaves) != len(right_leaves):
        raise ValueError("Gradient trees have incompatible trainable leaves.")
    return sum(
        (
            jnp.real(jnp.vdot(left_leaf, right_leaf))
            for left_leaf, right_leaf in zip(
                left_leaves, right_leaves, strict=True
            )
        ),
        start=jnp.asarray(0.0),
    )


def _gradient_alignment(gradients, /) -> Array:
    active = []
    for gradient in gradients:
        norm = _tree_norm(gradient)
        if bool(jnp.isfinite(norm)) and float(norm) > 0.0:
            active.append((gradient, norm))
    if len(active) < 2:
        return jnp.asarray(jnp.nan)
    units = tuple(
        jax.tree.map(lambda leaf, _norm=norm: leaf / _norm, gradient)
        for gradient, norm in active
    )
    count = len(units)
    mean = jax.tree.map(
        lambda *leaves: sum(leaves) / count,
        *units,
    )
    return (count * _tree_inner(mean, mean) - 1.0) / (count - 1)


def _interstep_alignment(previous, current, /) -> Array:
    if previous is None:
        return jnp.asarray(jnp.nan)
    left_norm = _tree_norm(previous)
    right_norm = _tree_norm(current)
    valid = (
        jnp.isfinite(left_norm)
        & jnp.isfinite(right_norm)
        & (left_norm > 0.0)
        & (right_norm > 0.0)
    )
    value = _tree_inner(previous, current) / jnp.where(
        valid, left_norm * right_norm, 1.0
    )
    return jnp.where(valid, value, jnp.nan)


def _residual_reference_indices(layout, reference, /) -> Array:
    if reference.block_name is not None:
        return layout.logical_indices(reference.term_index, reference.block_name)
    pieces = tuple(
        jnp.arange(entry.start, entry.stop, dtype=jnp.int32)
        for entry in layout.entries
        if entry.term_index == reference.term_index
    )
    if not pieces:
        raise KeyError(f"Unknown residual term {reference.term_index}.")
    return pieces[0] if len(pieces) == 1 else jnp.concatenate(pieces)


def _gradient_balance_statistics(residual, params, references, /):
    gradients = []
    norms = []
    for reference in references:
        indices = _residual_reference_indices(residual.layout, reference)

        def block_loss(candidate, _indices=indices):
            roots = residual.roots(candidate)[_indices]
            return jnp.real(jnp.vdot(roots, roots))

        gradient = eqx.filter_grad(block_loss)(params)
        gradients.append(gradient)
        norms.append(_tree_norm(gradient))
    return tuple(gradients), jnp.stack(tuple(norms))


def _residual_reference_available(layout, reference, /) -> bool:
    return any(
        entry.term_index == reference.term_index
        and (
            reference.block_name is None
            or entry.block_name == reference.block_name
        )
        for entry in layout.entries
    )


def _updated_balance_multipliers(
    residual,
    params,
    policy,
    old,
    key,
    /,
):
    if policy.method == "gradient_norm":
        gradients, statistics = _gradient_balance_statistics(
            residual, params, policy.blocks
        )
        acceptable = jnp.isfinite(statistics) & (statistics > 0.0)
    else:
        traces = []
        errors = []
        for index, reference in enumerate(policy.blocks):
            indices = _residual_reference_indices(residual.layout, reference)

            def block_roots(candidate, _indices=indices):
                return residual.roots(candidate)[_indices]

            output = block_roots(params)
            ntk = prepare_empirical_ntk(
                block_roots,
                params,
                output_space=ArraySpace(output.shape, dtype=output.dtype),
                ntk_id=(
                    f"balance:term={reference.term_index}:"
                    f"block={reference.block_name or '*'}"
                ),
            )
            estimate = stochastic_trace(
                ntk.kernel,
                key=jr.fold_in(key, index),
                num_probes=policy.ntk_probes,
                max_dimension=1,
            )
            traces.append(jnp.real(estimate.estimate))
            errors.append(jnp.real(estimate.standard_error))
        gradients = ()
        statistics = jnp.stack(tuple(traces))
        standard_errors = jnp.stack(tuple(errors))
        relative_error = standard_errors / jnp.maximum(
            jnp.abs(statistics), jnp.finfo(statistics.dtype).tiny
        )
        acceptable = (
            jnp.isfinite(statistics)
            & (statistics > 0.0)
            & jnp.isfinite(relative_error)
            & (relative_error <= policy.maximum_relative_standard_error)
        )
    valid_statistics = jnp.where(acceptable, statistics, jnp.nan)
    mean = jnp.nanmean(valid_statistics)
    candidate = mean / jnp.where(acceptable, statistics, 1.0)
    smoothed = policy.momentum * old + (1.0 - policy.momentum) * candidate
    bounded = jnp.clip(smoothed, policy.minimum, policy.maximum)
    updated = jnp.where(acceptable & jnp.isfinite(mean), bounded, old)
    updated = updated / jnp.mean(updated)
    return jax.lax.stop_gradient(updated), gradients, statistics

def _functional_ntk_diagnostics(residual, params, policy, key, /):
    roots = residual.roots(params)
    ntk = prepare_empirical_ntk(
        residual.roots,
        params,
        output_space=ArraySpace(roots.shape, dtype=roots.dtype),
        ntk_id=f"functional-training:step={residual.iteration}",
    )
    return analyze_ntk(
        ntk,
        policy=NTKDiagnosticsPolicy(
            num_probes=policy.ntk_probes,
            eigenvalue_count=policy.ntk_eigenvalues,
        ),
        key=key,
    )


def _functional_ntk_diagnostic_values(diagnostics, /) -> dict[str, Array]:
    if diagnostics is None:
        return {}
    return {
        "ntk/trace": diagnostics.trace,
        "ntk/trace_standard_error": diagnostics.trace_standard_error,
        "ntk/trace_square": diagnostics.trace_square,
        "ntk/largest_eigenvalue": diagnostics.largest_eigenvalue,
        "ntk/stable_rank": diagnostics.stable_rank,
        "ntk/effective_rank": diagnostics.effective_rank,
        "ntk/numerical_rank": diagnostics.numerical_rank,
        "ntk/nullity": diagnostics.nullity,
        "ntk/active_condition_number": diagnostics.active_condition_number,
        "ntk/finite": diagnostics.finite,
        "ntk/converged": diagnostics.converged,
    }

class PreparedFunctionalUpdate(StrictModule):
    """Physical objective and immutable same-update optimizer surrogate."""

    physical: _PreparedObjective
    residual: PreparedFunctionalResidual | None
    pseudo_inverse_steps: tuple[Array, ...]
    term_multipliers: Array
    block_gradients: tuple[Any, ...]
    balance_statistics: Array
    intra_gradient_alignment: Array
    inter_gradient_alignment: Array

    diagnostic_gradient: Any

    def __init__(
        self,
        physical: _PreparedObjective,
        residual: PreparedFunctionalResidual | None,
        pseudo_inverse_steps: Sequence[Any] = (),
        term_multipliers: Any = (),
        block_gradients: Sequence[Any] = (),
        balance_statistics: Any = (),
        intra_gradient_alignment: Any = jnp.nan,
        inter_gradient_alignment: Any = jnp.nan,
        diagnostic_gradient: Any = None,
        /,
    ):
        if not isinstance(physical, _PreparedObjective):
            raise TypeError("physical must be a _PreparedObjective.")
        if residual is not None and not isinstance(residual, PreparedFunctionalResidual):
            raise TypeError("residual must be PreparedFunctionalResidual or None.")
        self.physical = physical
        self.residual = residual
        self.pseudo_inverse_steps = tuple(
            jnp.asarray(value) for value in pseudo_inverse_steps
        )
        self.term_multipliers = jnp.asarray(
            term_multipliers, dtype=float
        ).reshape((-1,))
        self.block_gradients = tuple(block_gradients)
        self.balance_statistics = jnp.asarray(
            balance_statistics, dtype=float
        ).reshape((-1,))
        self.intra_gradient_alignment = jnp.asarray(intra_gradient_alignment)
        self.inter_gradient_alignment = jnp.asarray(inter_gradient_alignment)
        self.diagnostic_gradient = diagnostic_gradient

    @property
    def iteration(self) -> Any:
        return self.physical.iteration

    @property
    def terms(self) -> tuple[Any, ...]:
        return self.physical.terms

    def physical_values(self, functions: PyTree[Any], /) -> _ObjectiveValues:
        return evaluate_prepared_objective(self.physical, functions)

    def surrogate_loss(
        self,
        params: PyTree[Any],
        non_trainable: PyTree[Any],
        /,
    ) -> Array:
        functions = combine_trainable(params, non_trainable)
        residual_value = (
            jnp.asarray(0.0, dtype=float)
            if self.residual is None
            else self.residual.loss(params)
        )
        remainder = evaluate_prepared_scalar_remainder(self.physical, functions)
        return residual_value + remainder

    def surrogate_values(
        self,
        params: PyTree[Any],
        non_trainable: PyTree[Any],
        /,
    ) -> _ObjectiveValues:
        """Return the physical breakdown plus the optimizer-surrogate total."""
        functions = combine_trainable(params, non_trainable)
        physical = evaluate_prepared_objective(self.physical, functions)
        surrogate = self.surrogate_loss(params, non_trainable)
        return _ObjectiveValues(
            surrogate,
            physical.term_values,
            physical.model_loss_values,
        )



def _validate_pseudo_source(policy, term, /) -> None:
    source = term.term.source
    if isinstance(source, FixedIntegration):
        if policy.freshness != "experimental_fixed":
            raise ValueError(
                "Pseudo-transient training requires fresh collocation support; "
                "use freshness='experimental_fixed' to opt into fixed support."
            )
        return
    if isinstance(source, PerStepIntegration):
        return
    if isinstance(source, AdaptiveIntegration):
        refresh_every = int(source.policy.refresh_every)
        if policy.freshness == "every_update" and refresh_every != 1:
            raise ValueError(
                "Pseudo-time freshness='every_update' requires refresh_every=1."
            )
        return
    raise TypeError("Pseudo-transient residual terms require a resampled integration source.")


def _block_squared_norms(blocks, names, /) -> Array:
    values = []
    for name in names:
        selected = tuple(block for block in blocks if block.block_name == name)
        if not selected:
            raise ValueError(f"Pseudo-time residual block {name!r} is unavailable.")
        values.append(
            sum(
                (jnp.real(jnp.vdot(block.values, block.values)) for block in selected),
                start=jnp.asarray(0.0),
            )
        )
    return jnp.stack(tuple(values))


def _adapt_pseudo_inverse_steps(
    residual,
    params,
    previous_functions,
    policies,
    inverse_steps,
    /,
):
    if not policies:
        return tuple(inverse_steps)
    current_functions = combine_trainable(params, residual.non_trainable)
    current = (
        current_functions
        if residual.enforcement is None
        else residual.enforcement.apply(current_functions)
    )
    previous = (
        previous_functions
        if residual.enforcement is None
        else residual.enforcement.apply(previous_functions)
    )
    step = int(jax.device_get(jnp.asarray(residual.iteration)).reshape(()))
    updated = []
    for policy, old in zip(policies, inverse_steps, strict=True):
        matching = tuple(
            term for term in residual.terms if term.index == policy.term_index
        )
        old_ = jnp.asarray(old)
        if not matching:
            updated.append(old_)
            continue
        if len(matching) != 1:
            raise ValueError("Pseudo-time policy must select one residual term.")
        term = matching[0]
        _validate_pseudo_source(policy, term)
        current_residual = term.term.condition.residual(current)
        current_state = policy.relaxation.field(current)
        previous_state = policy.relaxation.field(previous)
        if not current_state.domain.same_support(
            current_residual.domain
        ) or not previous_state.domain.same_support(current_residual.domain):
            raise ValueError(
                "Pseudo-time relaxation state and residual domains are incompatible."
            )
        residual_layout = term.term.blocks
        relaxation_blocks = policy.relaxation.blocks
        if relaxation_blocks is not None and (
            residual_layout is None
            or relaxation_blocks.names != residual_layout.names
            or relaxation_blocks.sizes != residual_layout.sizes
            or relaxation_blocks.event_axis != residual_layout.event_axis
        ):
            raise ValueError(
                "Pseudo-time relaxation and residual block layouts must match."
            )
        if old_.ndim == 1 and (
            relaxation_blocks is None
            or int(old_.size) != relaxation_blocks.block_count
        ):
            raise ValueError(
                "Vector pseudo-time inverse steps require one value per "
                "relaxation block."
            )
        adaptation = policy.adaptation
        if adaptation is None or not adaptation.due(step):
            updated.append(old_)
            continue
        previous_residual = term.term.condition.residual(previous)
        residual_change = current_residual - previous_residual
        state_change = current_state - previous_state
        residual_blocks = evaluate_prepared_residual_term(
            params,
            residual.non_trainable,
            residual.enforcement,
            term,
            iteration=residual.iteration,
            residual_override=residual_change,
        )
        state_blocks = evaluate_prepared_residual_term(
            params,
            residual.non_trainable,
            residual.enforcement,
            term,
            iteration=residual.iteration,
            residual_override=state_change,
        )
        names = (
            (term.label,)
            if policy.relaxation.blocks is None
            else policy.relaxation.blocks.names
        )
        residual_norm = jnp.sqrt(_block_squared_norms(residual_blocks, names))
        state_norm = jnp.sqrt(_block_squared_norms(state_blocks, names))
        if old_.ndim == 0:
            residual_norm = jnp.sqrt(jnp.sum(residual_norm**2))
            state_norm = jnp.sqrt(jnp.sum(state_norm**2))
        candidate = residual_norm / jnp.where(state_norm > 0.0, state_norm, 1.0)
        valid = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.all(state_norm >= adaptation.minimum_state_displacement)
            & jnp.all(residual_norm >= adaptation.minimum_residual_displacement)
        )
        candidate = jnp.clip(
            candidate,
            adaptation.minimum_inverse_step,
            adaptation.maximum_inverse_step,
        )
        smoothed = adaptation.momentum * old_ + (1.0 - adaptation.momentum) * candidate
        updated.append(jax.lax.stop_gradient(jnp.where(valid, smoothed, old_)))
    return tuple(updated)

def prepare_functional_update(
    physical: _PreparedObjective,
    params: PyTree[Any],
    non_trainable: PyTree[Any],
    enforcement: Any,
    /,
    *,
    training: FunctionalTrainingPlan | None = None,
    previous_functions: Mapping[str, DomainFunction] | None = None,
    pseudo_inverse_steps: Sequence[Any] = (),
    term_multipliers: Any = (),
    previous_gradient: Any = None,
) -> PreparedFunctionalUpdate:
    """Compile one physical objective into an immutable optimizer surrogate."""
    residual = None
    if any(
        isinstance(prepared_term.term, ResidualPenalty)
        for prepared_term in physical.terms
    ):
        residual = prepare_functional_residual(
            physical,
            params,
            non_trainable,
            enforcement,
        )
    steps = tuple(jnp.asarray(value) for value in pseudo_inverse_steps)
    transform = None
    if training is not None and training.pseudo_transient:
        if residual is None:
            raise ValueError("Pseudo-transient training requires residual terms.")
        if previous_functions is None:
            raise ValueError("Pseudo-transient training requires previous functions.")
        if not steps:
            steps = tuple(
                policy.initial_inverse_step for policy in training.pseudo_transient
            )
        steps = _adapt_pseudo_inverse_steps(
            residual,
            params,
            previous_functions,
            training.pseudo_transient,
            steps,
        )
        transform = PseudoTransientResidualTransform(
            previous_functions,
            training.pseudo_transient,
            steps,
            enforcement,
        )
    if training is not None and training.causal:
        if residual is None:
            raise ValueError("Causal training requires residual terms.")
        active_causal = tuple(
            policy
            for policy in training.causal
            if any(term.index == policy.term_index for term in residual.terms)
        )
        if active_causal:
            gates = _prepare_causal_gates(
                residual,
                params,
                active_causal,
                transform,
            )
            transform = CausalResidualTransform(
                transform,
                active_causal,
                gates,
            )
    if transform is not None:
        residual = PreparedFunctionalResidual(
            residual.terms,
            residual.non_trainable,
            residual.enforcement,
            residual.layout,
            residual.iteration,
            transform,
        )
    multipliers = jnp.asarray(term_multipliers, dtype=float).reshape((-1,))
    block_gradients: tuple[Any, ...] = ()
    balance_statistics = jnp.zeros((0,), dtype=float)
    if training is not None and training.term_balance is not None:
        if residual is None:
            raise ValueError("Functional term balancing requires residual terms.")
        if multipliers.size == 0:
            multipliers = jnp.ones(
                (len(training.term_balance.blocks),), dtype=float
            )
        step = int(jax.device_get(jnp.asarray(physical.iteration)).reshape(()))
        if training.term_balance.due(step):
            all_available = all(
                _residual_reference_available(residual.layout, reference)
                for reference in training.term_balance.blocks
            )
            if all_available:
                multipliers, block_gradients, balance_statistics = (
                    _updated_balance_multipliers(
                        residual,
                        params,
                        training.term_balance,
                        multipliers,
                        physical.model_loss_key,
                    )
                )
            else:
                balance_statistics = jnp.full(
                    (len(training.term_balance.blocks),),
                    jnp.nan,
                )
        transform = BalancedResidualTransform(
            residual.transform,
            training.term_balance.blocks,
            multipliers,
        )
        residual = PreparedFunctionalResidual(
            residual.terms,
            residual.non_trainable,
            residual.enforcement,
            residual.layout,
            residual.iteration,
            transform,
        )
    intra_alignment = jnp.asarray(jnp.nan)
    inter_alignment = jnp.asarray(jnp.nan)
    diagnostic_gradient = None
    if (
        training is not None
        and training.diagnostics is not None
        and training.diagnostics.gradient_alignment
    ):
        diagnostic_step = int(
            jax.device_get(jnp.asarray(physical.iteration)).reshape(())
        )
        if training.diagnostics.due(diagnostic_step):
            if residual is None:
                raise ValueError("Gradient alignment requires residual terms.")
            references = (
                tuple(
                    reference
                    for reference in training.term_balance.blocks
                    if _residual_reference_available(residual.layout, reference)
                )
                if training.term_balance is not None
                else tuple(
                    ResidualBlockRef(term_index, block_name)
                    for term_index, block_name in residual.layout.logical_blocks
                )
            )
            gradients = block_gradients
            if not gradients:
                gradients, _ = _gradient_balance_statistics(
                    residual, params, references
                )
            intra_alignment = _gradient_alignment(gradients)
            diagnostic_gradient = eqx.filter_grad(residual.loss)(params)
            inter_alignment = _interstep_alignment(
                previous_gradient, diagnostic_gradient
            )
    return PreparedFunctionalUpdate(
        physical,
        residual,
        steps,
        multipliers,
        block_gradients,
        balance_statistics,
        intra_alignment,
        inter_alignment,
        diagnostic_gradient,
    )


__all__ = ["PreparedFunctionalUpdate", "prepare_functional_update"]
