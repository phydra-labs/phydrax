#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


sm = phx.applications.solid_mechanics
mn = sm.member_network
operations = (
    phx.optim.PrecedenceOperation("foundation"),
    phx.optim.PrecedenceOperation("frame", predecessors=("foundation",)),
    phx.optim.PrecedenceOperation("cable", predecessors=("frame",)),
)
space = phx.optim.PrecedenceSpace(operations)
search = mn.ConstructionSequenceSearchProblem(
    space,
    lambda node: (True, None),
    lambda node: float(len(node.completed)),
    lambda node: float(sum((index + 1) ** 2 for index, _ in enumerate(node.completed))),
)
sequence = mn.search_construction_sequences(search)
combination = sm.standards.LoadCombination(
    "ultimate",
    {"dead": 1.2, "live": 1.6},
    category="ultimate",
    clause_id="LC-1",
)
standard = sm.standards.GenericLimitStateStandard(
    (combination,), resistance_factor=0.9, edition="declared-2026"
)
demand = combination.combine({"dead": 10.0, "live": 5.0})
clause = standard.member_resistance(
    demand, 30.0, clause_id="M-1", governing_case="ultimate"
)
random_model = mn.StructuralRandomModel(
    jnp.asarray((0.0,)), jnp.asarray(((1.0,),)), ("load",)
)
limit_state = mn.StructuralLimitState(
    lambda parameter: 1.0 - parameter[0], limit_state_id="normal-threshold"
)
reliability = mn.monte_carlo_reliability(
    random_model, limit_state, jax.random.PRNGKey(0), 20_000
)
print("sequence", sequence.incumbent.completed)
print("sequence objective", sequence.objective)
print("clause utilization", clause.utilization)
print("failure probability", reliability.failure_probability)
