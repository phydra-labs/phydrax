"""Bounded exact inference, scheduled BP, and advanced sampling on one Ising chain."""

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
    elimination_plan = phx.pgm.plan_variable_elimination(
        graph,
        resources=phx.pgm.FactorGraphResourcePolicy(
            maximum_treewidth=4,
            maximum_elimination_elements=4096,
        ),
    )
    eliminated = phx.pgm.variable_elimination(elimination_plan)
    normalized_law = phx.pgm.NormalizedFactorGraphLaw(
        elimination_plan,
        eliminated,
    )

    sum_plan = phx.pgm.prepare_belief_propagation(
        graph,
        phx.pgm.SumProductBeliefPropagation(),
    )
    sum_state = phx.pgm.initialize_belief_propagation(sum_plan)
    sum_result = phx.pgm.run_belief_propagation(
        sum_plan,
        sum_state,
        schedule=phx.pgm.BeliefPropagationSchedulePolicy("forest"),
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
    online = phx.pgm.reduce_gibbs_chain(
        gibbs_plan,
        gibbs.final_state,
        phx.pgm.MomentReducer(),
        key=jr.key(23),
        num_sweeps=200,
        policy=phx.pgm.GibbsScanPolicy("random-scan"),
    )

    exact_means = []
    sampled_means = []
    offsets = graph.variable_state_offsets
    for variable in range(graph.num_variables):
        probability_one = exact.variable_probabilities.values[offsets[variable] + 1]
        exact_means.append(2.0 * probability_one - 1.0)
        sampled_means.append(jnp.mean(2.0 * gibbs.samples[..., variable] - 1.0))

    print("exact log normalizer:", float(exact.log_normalizer))
    print("elimination log normalizer:", float(eliminated.log_normalizer))
    print("sum-product log normalizer:", float(sum_result.log_normalizer))
    print("exact MAP:", exact.map_assignment)
    print("max-product MAP:", max_result.map_assignment)
    print("exact spin means:", jnp.asarray(exact_means))
    print("sampled spin means:", jnp.asarray(sampled_means))
    print("normalized-law samples:", normalized_law.sample(jr.key(29), (3,)))
    print("online state means:", online.reduction["mean"])
    print("maximum R-hat:", float(gibbs.diagnostics.max_rhat))


if __name__ == "__main__":
    main()
