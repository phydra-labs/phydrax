"""Discover a Markov-equivalence class without promoting it to an SCM."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def main() -> None:
    rng = np.random.default_rng(23)
    samples = 1000
    x = rng.normal(size=samples)
    y = x + rng.normal(scale=0.7, size=samples)
    z = y + rng.normal(scale=0.7, size=samples)
    schema = phx.causal.CausalSchema(
        tuple(phx.causal.CausalVariable(name=name) for name in ("x", "y", "z"))
    )
    dataset = phx.causal.CausalDataset(
        schema=schema,
        values=(jnp.asarray(x), jnp.asarray(y), jnp.asarray(z)),
    )
    result = phx.causal.discover_pc_stable(
        dataset,
        phx.causal.PCStablePlan(
            ci_test=phx.causal.FisherZTest(alpha=0.001),
            knowledge=phx.causal.DiscoveryBackgroundKnowledge(schema=schema),
            resources=phx.causal.DiscoveryResourcePolicy(
                maximum_ci_tests=100,
                maximum_conditioning_depth=1,
            ),
        ),
    )
    if isinstance(result.graph, (phx.causal.CausalCPDAG, phx.causal.CausalPDAG)):
        directed_edges = result.graph.directed_edges
        undirected_edges = result.graph.undirected_edges
    else:
        directed_edges = ()
        undirected_edges = ()
    print(
        {
            "status": result.status.value,
            "graph_kind": type(result.graph).__name__,
            "directed_edges": directed_edges,
            "undirected_edges": undirected_edges,
            "conditional_on_discovery": True,
        }
    )


if __name__ == "__main__":
    main()
