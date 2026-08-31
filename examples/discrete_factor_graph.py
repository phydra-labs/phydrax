"""Exact inference, belief propagation, and Gibbs sampling on one Ising chain."""

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def main():
    graph = phx.pgm.ising_factor_graph(
        jnp.asarray([0.2, -0.1, 0.05, 0.0]),
        jnp.asarray([[0, 1], [1, 2], [2, 3]]),
        jnp.asarray([0.4, -0.25, 0.3]),
        name="spin",
    )

    exact = phx.pgm.enumerate_factor_graph(graph)

    sum_plan = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.SumProductBeliefPropagation(),
    )
    sum_result = phx.pgm.run_belief_propagation(
        sum_plan,
        phx.pgm.initialize_belief_propagation(sum_plan),
    )

    max_plan = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.MaxProductBeliefPropagation(),
    )
    max_result = phx.pgm.run_belief_propagation(
        max_plan,
        phx.pgm.initialize_belief_propagation(max_plan),
    )

    gibbs_plan = phx.pgm.prepare_chromatic_gibbs(graph)
    gibbs_state = phx.pgm.initialize_gibbs(
        gibbs_plan,
        jnp.asarray(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
            ]
        ),
    )
    gibbs = phx.pgm.sample_gibbs(
        gibbs_plan,
        gibbs_state,
        key=jr.key(17),
        schedule=phx.pgm.GibbsSchedule(
            warmup_sweeps=50,
            num_draws=200,
            sweeps_per_draw=2,
        ),
    )

    exact_means = []
    sampled_means = []
    offsets = graph.variable_state_offsets
    for variable in range(graph.num_variables):
        probability_one = exact.variable_probabilities.values[offsets[variable] + 1]
        exact_means.append(2.0 * probability_one - 1.0)
        sampled_means.append(jnp.mean(2.0 * gibbs.samples[..., variable] - 1.0))

    print("exact log normalizer:", float(exact.log_normalizer))
    print("sum-product log normalizer:", float(sum_result.log_normalizer))
    print("exact MAP:", exact.map_assignment)
    print("max-product MAP:", max_result.map_assignment)
    print("exact spin means:", jnp.asarray(exact_means))
    print("sampled spin means:", jnp.asarray(sampled_means))
    print("maximum R-hat:", float(gibbs.diagnostics.max_rhat))


if __name__ == "__main__":
    main()
