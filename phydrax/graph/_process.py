from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu

from ._ir import GraphIR


GraphFeatureName = Literal["nodes", "edges", "globals"]
GraphRolloutReduction = Literal["mean", "sum"]


def _expand_mask(mask: jnp.ndarray, template: jnp.ndarray, /) -> jnp.ndarray:
    return jnp.reshape(mask, mask.shape + (1,) * (template.ndim - mask.ndim))


def _mask_for_feature(graph: GraphIR, feature: GraphFeatureName, /) -> jnp.ndarray | None:
    if feature == "nodes":
        return graph.node_mask
    if feature == "edges":
        return graph.edge_mask
    return graph.graph_mask


def _tree_axpy(
    base: Any,
    tangent: Any,
    scale: jnp.ndarray,
    mask: jnp.ndarray | None,
    /,
) -> Any:
    if tangent is None:
        return base
    if base is None:
        out = jtu.tree_map(lambda d: scale * d, tangent)
        if mask is None:
            return out
        return jtu.tree_map(lambda y: _expand_mask(mask, y).astype(y.dtype) * y, out)

    def _leaf(x, d):
        y = x + scale * d
        if mask is None:
            return y
        m = _expand_mask(mask, y).astype(bool)
        return jnp.where(m, y, x)

    return jtu.tree_map(_leaf, base, tangent)


def _add_scaled_graph(base: GraphIR, tangent: GraphIR, scale: jnp.ndarray, /) -> GraphIR:
    return base.replace(
        nodes=_tree_axpy(
            base.nodes,
            tangent.nodes,
            scale,
            _mask_for_feature(base, "nodes"),
        ),
        edges=_tree_axpy(
            base.edges,
            tangent.edges,
            scale,
            _mask_for_feature(base, "edges"),
        ),
        globals=_tree_axpy(
            base.globals,
            tangent.globals,
            scale,
            _mask_for_feature(base, "globals"),
        ),
        validate=False,
    )


def _add_weighted_graph(
    base: GraphIR,
    terms: Sequence[tuple[GraphIR, jnp.ndarray]],
    /,
) -> GraphIR:
    out = base
    for tangent, scale in terms:
        out = _add_scaled_graph(out, tangent, scale)
    return out


def _require_graph_rate(value: Any, /) -> GraphIR:
    if not isinstance(value, GraphIR):
        raise TypeError("Graph process models must return a phydrax.graph.GraphIR.")
    return value


def euler_step(
    graph: GraphIR,
    vector_field: Callable[[GraphIR], GraphIR],
    /,
    *,
    dt: float | jnp.ndarray,
) -> GraphIR:
    """Advance a graph state by one explicit Euler step."""
    rate = _require_graph_rate(vector_field(graph))
    return _add_scaled_graph(graph, rate, jnp.asarray(dt, dtype=float))


def rk4_step(
    graph: GraphIR,
    vector_field: Callable[[GraphIR], GraphIR],
    /,
    *,
    dt: float | jnp.ndarray,
) -> GraphIR:
    """Advance a graph state by one classical RK4 step."""
    dt_arr = jnp.asarray(dt, dtype=float)
    half = jnp.asarray(0.5, dtype=float)
    sixth = jnp.asarray(1.0 / 6.0, dtype=float)

    k1 = _require_graph_rate(vector_field(graph))
    k2 = _require_graph_rate(vector_field(_add_scaled_graph(graph, k1, half * dt_arr)))
    k3 = _require_graph_rate(vector_field(_add_scaled_graph(graph, k2, half * dt_arr)))
    k4 = _require_graph_rate(vector_field(_add_scaled_graph(graph, k3, dt_arr)))
    return _add_weighted_graph(
        graph,
        (
            (k1, sixth * dt_arr),
            (k2, 2.0 * sixth * dt_arr),
            (k3, 2.0 * sixth * dt_arr),
            (k4, sixth * dt_arr),
        ),
    )


class EulerGraphStepper(eqx.Module):
    """`GraphIR -> GraphIR` explicit Euler process wrapper."""

    vector_field: Callable[[GraphIR], GraphIR]
    dt: float = eqx.field(static=True)

    def __init__(self, vector_field: Callable[[GraphIR], GraphIR], /, *, dt: float):
        self.vector_field = vector_field
        self.dt = float(dt)

    def __call__(self, graph: GraphIR) -> GraphIR:
        return euler_step(graph, self.vector_field, dt=self.dt)


class RK4GraphStepper(eqx.Module):
    """`GraphIR -> GraphIR` classical RK4 process wrapper."""

    vector_field: Callable[[GraphIR], GraphIR]
    dt: float = eqx.field(static=True)

    def __init__(self, vector_field: Callable[[GraphIR], GraphIR], /, *, dt: float):
        self.vector_field = vector_field
        self.dt = float(dt)

    def __call__(self, graph: GraphIR) -> GraphIR:
        return rk4_step(graph, self.vector_field, dt=self.dt)


def rollout(
    stepper: Callable[[GraphIR], GraphIR],
    graph: GraphIR,
    /,
    *,
    steps: int,
    include_initial: bool = True,
) -> tuple[GraphIR, ...]:
    """Return an autoregressive graph rollout as a tuple of graph states."""
    n = int(steps)
    if n < 0:
        raise ValueError("steps must be non-negative.")
    states: list[GraphIR] = []
    current = graph
    if include_initial:
        states.append(current)
    for _ in range(n):
        current = _require_graph_rate(stepper(current))
        states.append(current)
    return tuple(states)


def rollout_features(
    stepper: Callable[[GraphIR], GraphIR],
    graph: GraphIR,
    /,
    *,
    steps: int,
    feature: GraphFeatureName = "nodes",
    include_initial: bool = True,
) -> Any:
    """Stack one feature payload from an autoregressive graph rollout."""
    if feature not in ("nodes", "edges", "globals"):
        raise ValueError("feature must be 'nodes', 'edges', or 'globals'.")
    states = rollout(stepper, graph, steps=steps, include_initial=include_initial)
    payloads = [getattr(state, feature) for state in states]
    if any(payload is None for payload in payloads):
        raise ValueError(f"Cannot stack missing graph feature {feature!r}.")
    return jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *payloads)


def _tree_squared_error(prediction: Any, target: Any, /) -> Any:
    return jtu.tree_map(
        lambda pred, tgt: jnp.square(
            jnp.asarray(pred, dtype=float) - jnp.asarray(tgt, dtype=float)
        ),
        prediction,
        target,
    )


def _tree_reduce_squared_error(
    squared: Any,
    /,
    *,
    reduction: GraphRolloutReduction,
) -> jnp.ndarray:
    leaves = jtu.tree_leaves(squared)
    if not leaves:
        raise ValueError("rollout target must contain at least one array leaf.")
    total = jnp.asarray(0.0, dtype=float)
    count = 0
    for leaf in leaves:
        arr = jnp.asarray(leaf, dtype=float)
        total = total + jnp.sum(arr)
        count += int(arr.size)
    if reduction == "sum":
        return total.reshape(())
    if reduction == "mean":
        return (total / float(max(count, 1))).reshape(())
    raise ValueError("reduction must be 'mean' or 'sum'.")


def rollout_feature_loss(
    stepper: Callable[[GraphIR], GraphIR],
    graph: GraphIR,
    target: Any,
    /,
    *,
    steps: int | None = None,
    feature: GraphFeatureName = "nodes",
    include_initial: bool = True,
    reduction: GraphRolloutReduction = "mean",
) -> jnp.ndarray:
    """Return supervised squared loss for an autoregressive graph rollout feature."""
    if steps is None:
        leaves = jtu.tree_leaves(target)
        if not leaves:
            raise ValueError("target must contain at least one array leaf.")
        length = int(jnp.asarray(leaves[0]).shape[0])
        steps = length - 1 if include_initial else length
    prediction = rollout_features(
        stepper,
        graph,
        steps=int(steps),
        feature=feature,
        include_initial=include_initial,
    )
    squared = _tree_squared_error(prediction, target)
    return _tree_reduce_squared_error(squared, reduction=reduction)


class AutoregressiveGraphRollout(eqx.Module):
    """Callable wrapper around autoregressive graph rollouts."""

    stepper: Callable[[GraphIR], GraphIR]
    steps: int = eqx.field(static=True)
    include_initial: bool = eqx.field(static=True)

    def __init__(
        self,
        stepper: Callable[[GraphIR], GraphIR],
        /,
        *,
        steps: int,
        include_initial: bool = True,
    ):
        self.stepper = stepper
        self.steps = int(steps)
        self.include_initial = bool(include_initial)
        if self.steps < 0:
            raise ValueError("steps must be non-negative.")

    def __call__(self, graph: GraphIR) -> tuple[GraphIR, ...]:
        return rollout(
            self.stepper,
            graph,
            steps=self.steps,
            include_initial=self.include_initial,
        )

    def features(self, graph: GraphIR, /, *, feature: GraphFeatureName = "nodes") -> Any:
        return rollout_features(
            self.stepper,
            graph,
            steps=self.steps,
            feature=feature,
            include_initial=self.include_initial,
        )


__all__ = [
    "AutoregressiveGraphRollout",
    "EulerGraphStepper",
    "GraphFeatureName",
    "GraphRolloutReduction",
    "RK4GraphStepper",
    "euler_step",
    "rk4_step",
    "rollout",
    "rollout_feature_loss",
    "rollout_features",
]
