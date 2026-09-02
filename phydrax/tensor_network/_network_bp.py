#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence

import equinox as eqx
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._precision import precision_itemsize
from .._strict import StrictModule


class FactorTensor(StrictModule):
    factor_id: str = eqx.field(static=True)
    variables: tuple[str, ...] = eqx.field(static=True)
    values: Array

    def __init__(self, factor_id: str, variables: Sequence[str], values: ArrayLike, /):
        identifier = str(factor_id)
        variables_ = tuple(str(name) for name in variables)
        values_ = jnp.asarray(values)
        if (
            not identifier
            or len(set(variables_)) != len(variables_)
            or values_.ndim != len(variables_)
        ):
            raise ValueError(
                "Factor tensors require an ID and one distinct variable per axis."
            )
        if not jnp.issubdtype(values_.dtype, jnp.inexact):
            raise TypeError("Factor tensors must use real or complex inexact values.")
        self.factor_id = identifier
        self.variables = variables_
        self.values = values_


class FactorGraphNetwork(StrictModule):
    variable_names: tuple[str, ...] = eqx.field(static=True)
    cardinalities: tuple[int, ...] = eqx.field(static=True)
    factors: tuple[FactorTensor, ...]
    is_forest: bool = eqx.field(static=True)
    network_id: str = eqx.field(static=True)

    def __init__(
        self,
        variable_cardinalities: Mapping[str, int],
        factors: Sequence[FactorTensor],
        /,
    ):
        names = tuple(sorted(str(name) for name in variable_cardinalities))
        cardinalities = tuple(int(variable_cardinalities[name]) for name in names)
        factors_ = tuple(factors)
        if (
            not names
            or any(not name for name in names)
            or any(value < 1 for value in cardinalities)
        ):
            raise ValueError(
                "Factor-graph variables require names and positive cardinalities."
            )
        if not factors_ or any(
            not isinstance(factor, FactorTensor) for factor in factors_
        ):
            raise TypeError("factors must be a nonempty sequence of FactorTensor values.")
        ids = tuple(factor.factor_id for factor in factors_)
        if len(set(ids)) != len(ids):
            raise ValueError("Factor IDs must be unique.")
        card = dict(zip(names, cardinalities, strict=True))
        for factor in factors_:
            if any(name not in card for name in factor.variables):
                raise ValueError("A factor references an undeclared variable.")
            if factor.values.shape != tuple(card[name] for name in factor.variables):
                raise ValueError(
                    "A factor tensor shape differs from variable cardinalities."
                )
        dtype = str(factors_[0].values.dtype)
        if any(str(factor.values.dtype) != dtype for factor in factors_):
            raise TypeError("All factor tensors must use one dtype.")
        forest = _is_factor_forest(names, factors_)
        self.variable_names = names
        self.cardinalities = cardinalities
        self.factors = factors_
        self.is_forest = forest
        self.network_id = canonical_fingerprint(
            {
                "kind": "factor-graph-network",
                "variables": tuple(zip(names, cardinalities, strict=True)),
                "factors": tuple(
                    (factor.factor_id, factor.variables, factor.values.shape)
                    for factor in factors_
                ),
                "dtype": dtype,
            }
        )


class NetworkBPPolicy(StrictModule):
    maximum_iterations: int = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    maximum_message_elements: int = eqx.field(static=True)
    maximum_factor_elements: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_iterations: int,
        /,
        *,
        tolerance: float = 1e-8,
        damping: float = 0.0,
        maximum_message_elements: int = 10_000_000,
        maximum_factor_elements: int = 100_000_000,
        maximum_workspace_bytes: int = 2**31,
    ):
        iterations = int(maximum_iterations)
        tolerance_ = float(tolerance)
        damping_ = float(damping)
        message_limit = int(maximum_message_elements)
        factor_limit = int(maximum_factor_elements)
        workspace = int(maximum_workspace_bytes)
        if iterations < 1 or tolerance_ <= 0.0 or not 0.0 <= damping_ < 1.0:
            raise ValueError("BP iteration, tolerance, and damping values are invalid.")
        if message_limit < 1 or factor_limit < 1 or workspace < 1:
            raise ValueError("BP resource limits must be positive.")
        self.maximum_iterations = iterations
        self.tolerance = tolerance_
        self.damping = damping_
        self.maximum_message_elements = message_limit
        self.maximum_factor_elements = factor_limit
        self.maximum_workspace_bytes = workspace
        self.policy_id = canonical_fingerprint(
            {
                "kind": "network-bp-policy",
                "maximum_iterations": iterations,
                "tolerance": tolerance_,
                "damping": damping_,
                "maximum_message_elements": message_limit,
                "maximum_factor_elements": factor_limit,
                "maximum_workspace_bytes": workspace,
            }
        )


class NetworkBPEvidence(StrictModule):
    network_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    replay_id: str = eqx.field(static=True)
    residual_history: Array
    active_mask: Array
    convergence_mask: Array
    converged: Array
    nonnegative: Array
    finite: Array
    accepted: Array
    exact: Array
    claim: str = eqx.field(static=True)
    global_error_bound_claimed: bool = eqx.field(static=True)
    admitted_message_elements: int = eqx.field(static=True)
    admitted_workspace_bytes: int = eqx.field(static=True)


class NetworkBPResult(StrictModule):
    variable_beliefs: tuple[Array, ...]
    factor_to_variable_messages: tuple[Array, ...]
    variable_to_factor_messages: tuple[Array, ...]
    log_partition: Array
    evidence: NetworkBPEvidence


def _is_factor_forest(
    names: tuple[str, ...], factors: tuple[FactorTensor, ...], /
) -> bool:
    adjacency: dict[str, list[str]] = {f"v:{name}": [] for name in names}
    for factor in factors:
        factor_node = f"f:{factor.factor_id}"
        adjacency[factor_node] = []
        for name in factor.variables:
            variable_node = f"v:{name}"
            adjacency[factor_node].append(variable_node)
            adjacency[variable_node].append(factor_node)
    visited: set[str] = set()
    for root in adjacency:
        if root in visited:
            continue
        stack = [(root, "")]
        while stack:
            node, parent = stack.pop()
            if node in visited:
                return False
            visited.add(node)
            for neighbor in adjacency[node]:
                if neighbor != parent:
                    stack.append((neighbor, node))
    return True


def _normalize(message: Array, /) -> Array:
    total = jnp.sum(message)
    return message / jnp.where(jnp.abs(total) > 0.0, total, 1.0)


def run_network_belief_propagation(
    network: FactorGraphNetwork,
    policy: NetworkBPPolicy,
    /,
) -> NetworkBPResult:
    """Run bounded synchronous sum-product BP on trees or loopy factor graphs."""

    if not isinstance(network, FactorGraphNetwork) or not isinstance(
        policy, NetworkBPPolicy
    ):
        raise TypeError("network and policy have invalid types.")
    cardinality = dict(zip(network.variable_names, network.cardinalities, strict=True))
    incidences = tuple(
        (factor_index, variable_index, variable)
        for factor_index, factor in enumerate(network.factors)
        for variable_index, variable in enumerate(factor.variables)
    )
    message_elements = sum(cardinality[variable] for _, _, variable in incidences) * 2
    factor_elements = sum(factor.values.size for factor in network.factors)
    itemsize = precision_itemsize(str(network.factors[0].values.dtype))
    workspace_bytes = (message_elements * 3 + factor_elements) * itemsize
    if message_elements > policy.maximum_message_elements:
        raise MemoryError(
            "BP messages exceed maximum_message_elements before allocation."
        )
    if factor_elements > policy.maximum_factor_elements:
        raise MemoryError("BP factors exceed maximum_factor_elements before allocation.")
    if workspace_bytes > policy.maximum_workspace_bytes:
        raise MemoryError("BP exceeds maximum_workspace_bytes before allocation.")
    if not incidences:
        dtype = network.factors[0].values.dtype
        beliefs = tuple(
            jnp.ones((cardinality[name],), dtype=dtype) / cardinality[name]
            for name in network.variable_names
        )
        log_partition = sum(
            jnp.log(jnp.abs(factor.values)) for factor in network.factors
        ) + sum(jnp.log(cardinality[name]) for name in network.variable_names)
        residuals = jnp.zeros(
            (policy.maximum_iterations,), dtype=jnp.real(network.factors[0].values).dtype
        )
        convergence = jnp.arange(policy.maximum_iterations) == 0
        finite = jnp.isfinite(log_partition) & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in beliefs))
        )
        nonnegative = jnp.all(
            jnp.stack(
                tuple(
                    (jnp.real(factor.values) >= 0.0) & (jnp.imag(factor.values) == 0.0)
                    for factor in network.factors
                )
            )
        )
        accepted = finite & nonnegative
        replay_id = canonical_fingerprint(
            {
                "kind": "network-bp-replay",
                "network": network.network_id,
                "policy": policy.policy_id,
            }
        )
        evidence = NetworkBPEvidence(
            network.network_id,
            policy.policy_id,
            replay_id,
            residuals,
            convergence,
            convergence,
            jnp.asarray(True),
            nonnegative,
            finite,
            accepted,
            accepted,
            "exact disconnected scalar-factor evaluation",
            False,
            message_elements,
            workspace_bytes,
        )
        return NetworkBPResult(beliefs, (), (), log_partition, evidence)

    factor_to_variable = tuple(
        jnp.ones((cardinality[variable],), dtype=network.factors[0].values.dtype)
        / cardinality[variable]
        for _, _, variable in incidences
    )
    variable_to_factor = factor_to_variable
    residual_history = []
    active_history = []
    convergence_history = []
    active = jnp.asarray(True)

    for _ in range(policy.maximum_iterations):
        active_history.append(active)
        new_variable = []
        for target, (_, _, variable) in enumerate(incidences):
            message = jnp.ones_like(factor_to_variable[target])
            for source, (source_factor, _, source_variable) in enumerate(incidences):
                if source_variable == variable and source_factor != incidences[target][0]:
                    message = message * factor_to_variable[source]
            new_variable.append(_normalize(message))
        new_factor = []
        for target, (factor_index, variable_index, _) in enumerate(incidences):
            factor = network.factors[factor_index]
            symbols = tuple(oe.get_symbol(axis) for axis in range(len(factor.variables)))
            inputs = ["".join(symbols)]
            operands = [factor.values]
            for source, (source_factor, source_axis, _) in enumerate(incidences):
                if source_factor == factor_index and source_axis != variable_index:
                    inputs.append(symbols[source_axis])
                    operands.append(new_variable[source])
            equation = ",".join(inputs) + "->" + symbols[variable_index]
            proposed = _normalize(oe.contract(equation, *operands, optimize="greedy"))
            damped = _normalize(
                (1.0 - policy.damping) * proposed
                + policy.damping * factor_to_variable[target]
            )
            new_factor.append(damped)
        residual = jnp.max(
            jnp.stack(
                tuple(
                    jnp.max(jnp.abs(new - old))
                    for new, old in zip(new_factor, factor_to_variable, strict=True)
                )
            )
        )
        converged_now = active & (residual <= policy.tolerance)
        variable_to_factor = tuple(
            jnp.where(active, new, old)
            for new, old in zip(new_variable, variable_to_factor, strict=True)
        )
        factor_to_variable = tuple(
            jnp.where(active, new, old)
            for new, old in zip(new_factor, factor_to_variable, strict=True)
        )
        residual_history.append(jnp.where(active, residual, 0.0))
        convergence_history.append(converged_now)
        active = active & ~converged_now

    beliefs = []
    variable_normalizers = []
    for variable in network.variable_names:
        belief = jnp.ones((cardinality[variable],), dtype=network.factors[0].values.dtype)
        for message, (_, _, source_variable) in zip(
            factor_to_variable, incidences, strict=True
        ):
            if source_variable == variable:
                belief = belief * message
        variable_normalizers.append(jnp.sum(belief))
        beliefs.append(_normalize(belief))
    factor_normalizers = []
    for factor_index, factor in enumerate(network.factors):
        weighted = factor.values
        for source, (source_factor, source_axis, _) in enumerate(incidences):
            if source_factor == factor_index:
                shape = [1] * factor.values.ndim
                shape[source_axis] = variable_to_factor[source].shape[0]
                weighted = weighted * variable_to_factor[source].reshape(tuple(shape))
        factor_normalizers.append(jnp.sum(weighted))
    edge_normalizers = tuple(
        jnp.sum(factor_message * variable_message)
        for factor_message, variable_message in zip(
            factor_to_variable, variable_to_factor, strict=True
        )
    )
    log_partition = sum(jnp.log(jnp.abs(value)) for value in factor_normalizers)
    log_partition = log_partition + sum(
        jnp.log(jnp.abs(value)) for value in variable_normalizers
    )
    log_partition = log_partition - sum(
        jnp.log(jnp.abs(value)) for value in edge_normalizers
    )
    residuals = jnp.stack(tuple(residual_history))
    active_mask = jnp.stack(tuple(active_history))
    convergence_mask = jnp.stack(tuple(convergence_history))
    converged = jnp.any(convergence_mask)
    all_values = (
        tuple(beliefs) + factor_to_variable + variable_to_factor + (log_partition,)
    )
    finite = jnp.all(
        jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in all_values))
    )
    nonnegative = jnp.all(
        jnp.stack(
            tuple(
                jnp.all(jnp.real(factor.values) >= 0.0)
                & jnp.all(jnp.imag(factor.values) == 0.0)
                for factor in network.factors
            )
        )
    )
    accepted = finite & nonnegative & converged
    exact = accepted & jnp.asarray(network.is_forest)
    replay_id = canonical_fingerprint(
        {
            "kind": "network-bp-replay",
            "network": network.network_id,
            "policy": policy.policy_id,
        }
    )
    evidence = NetworkBPEvidence(
        network.network_id,
        policy.policy_id,
        replay_id,
        residuals,
        active_mask,
        convergence_mask,
        converged,
        nonnegative,
        finite,
        accepted,
        exact,
        "exact sum-product on a converged factor forest; otherwise loopy Bethe "
        "approximation with no global error bound",
        False,
        message_elements,
        workspace_bytes,
    )
    return NetworkBPResult(
        tuple(beliefs),
        factor_to_variable,
        variable_to_factor,
        log_partition,
        evidence,
    )


__all__ = [
    "FactorGraphNetwork",
    "FactorTensor",
    "NetworkBPEvidence",
    "NetworkBPPolicy",
    "NetworkBPResult",
    "run_network_belief_propagation",
]
