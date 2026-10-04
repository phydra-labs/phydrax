"""Payload declarations shared by graph consumers of the streamed relation runner."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array

from .._strict import StrictModule
from ..sparse._streamed import StreamedPayloadSpec


@final
class RouteLocal(StrictModule):
    """Explicit declaration that a graph callback depends on one route only.

    Streamed graph operators evaluate the wrapped function once per route on
    unbatched rows, without an edge axis. Graph-wide callbacks (for example a
    normalization over all edges) are not route-local and are refused rather
    than reinterpreted per route. Array leaves of a wrapped module remain
    trainable dynamic leaves.
    """

    function: Callable[..., Any]

    def __init__(self, function: Callable[..., Any], /) -> None:
        if not callable(function):
            raise TypeError("RouteLocal requires a callable.")
        self.function = function

    def __call__(self, *rows: Any) -> Any:
        return self.function(*rows)


def require_route_local(
    function: Callable[..., Any] | None, name: str, /
) -> RouteLocal | None:
    """Admit only explicitly route-local callbacks into streamed graph operators."""
    if function is not None and not isinstance(function, RouteLocal):
        raise TypeError(
            f"{name} must be declared route-local with phydrax.graph.RouteLocal; "
            "graph-wide callbacks are not evaluated per route."
        )
    return function


def row_spec(tree: Any, /) -> Any:
    """Abstract one-row structure of a payload whose leaves lead with rows."""
    return jtu.tree_map(
        lambda leaf: jax.ShapeDtypeStruct(jnp.shape(leaf)[1:], jnp.result_type(leaf)),
        tree,
    )


def declared_payload(
    edge_function: Callable[..., Any],
    epilogue: Callable[..., Any],
    parameters: Any,
    source_data: Any,
    receiver_data: Any,
    edge_data: Any,
    /,
    *,
    edge_output: bool,
) -> StreamedPayloadSpec:
    """Declare the exact per-event and per-receiver payload by abstract evaluation.

    Abstract evaluation traces the consumer's own row callbacks once at
    declaration; it never executes numerical work or probes the runner.
    """
    source, receiver, edge = (
        row_spec(source_data),
        row_spec(receiver_data),
        row_spec(edge_data),
    )
    produced = eqx.filter_eval_shape(edge_function, parameters, source, receiver, edge)
    message, routed = produced if edge_output else (produced, None)
    output = eqx.filter_eval_shape(epilogue, parameters, receiver, message)
    return StreamedPayloadSpec(message=message, output=output, edge_output=routed)


def entity_graph_ids(counts: Any, total_length: int, /) -> Array:
    """Owning graph of each padded entity; padding names the zero sentinel row."""
    counts_ = jnp.asarray(counts)
    graph_count = counts_.shape[0]
    real_length = int(counts_.sum())
    if real_length > total_length:
        raise ValueError("Graph entity counts exceed the padded entity capacity.")
    ids = jnp.repeat(
        jnp.arange(graph_count, dtype=jnp.int32),
        counts_,
        total_repeat_length=real_length,
    )
    return jnp.concatenate(
        (ids, jnp.full((total_length - real_length,), graph_count, dtype=jnp.int32))
    )


def graph_feature_table(globals_: Any, /) -> Any:
    """Per-graph features with one trailing zero row addressed by padded entities."""
    return jtu.tree_map(
        lambda leaf: jnp.concatenate(
            (
                jnp.asarray(leaf),
                jnp.zeros((1,) + jnp.shape(leaf)[1:], dtype=jnp.result_type(leaf)),
            ),
            axis=0,
        ),
        globals_,
    )


__all__ = [
    "declared_payload",
    "entity_graph_ids",
    "graph_feature_table",
    "require_route_local",
    "RouteLocal",
    "row_spec",
]
