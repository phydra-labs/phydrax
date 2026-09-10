"""Simulate an intervention and an exact structural counterfactual."""

from __future__ import annotations

import jax
import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    schema = phx.causal.CausalSchema(
        (phx.causal.CausalVariable(name="x"), phx.causal.CausalVariable(name="y"))
    )
    graph = phx.causal.CausalDAG(schema=schema, directed_edges=(("x", "y"),))
    root = phx.causal.InvertibleNoiseMechanism(
        output="x",
        parents=(),
        noise_sampler=lambda key, n: jax.random.normal(key, (n,)),
        forward=lambda parents, noise: noise,
        inverse=lambda parents, value: value,
        semantic_ids=("root-noise", "root-forward", "root-inverse"),
        numeric_ids=("root-noise-r0", "root-forward-r0", "root-inverse-r0"),
    )
    child = phx.causal.InvertibleNoiseMechanism(
        output="y",
        parents=("x",),
        noise_sampler=lambda key, n: jax.random.normal(key, (n,)),
        forward=lambda parents, noise: 2.0 * parents[0] + noise,
        inverse=lambda parents, value: value - 2.0 * parents[0],
        semantic_ids=("child-noise", "child-forward", "child-inverse"),
        numeric_ids=("child-noise-r0", "child-forward-r0", "child-inverse-r0"),
    )
    scm = phx.causal.SCMPlan(
        graph=graph, mechanisms=(root, child), exogenous_independence=True
    )
    intervention = phx.causal.build_intervention_regime(
        scm,
        (phx.causal.PerfectIntervention(variable="x", value=3.0),),
    )
    interventional = phx.causal.sample_scm(intervention, jax.random.key(1), 1024)

    factual = phx.causal.FactualObservation(
        schema=schema,
        values=(jnp.asarray([1.0]), jnp.asarray([2.5])),
    )
    abduction = phx.causal.abduct_factual(scm, factual)
    counterfactual = phx.causal.evaluate_counterfactual(
        intervention,
        factual,
        abduction,
    )
    print(
        {
            "interventional_y_mean": float(jnp.mean(interventional.value("y"))),
            "counterfactual_y": float(counterfactual.value("y")[0]),
            "counterfactual_status": counterfactual.status.name,
        }
    )


if __name__ == "__main__":
    main()
