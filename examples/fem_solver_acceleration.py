#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


space = phx.linalg.ArraySpace((2,))
operator = phx.linalg.DenseLinearOperator(
    jnp.asarray([[2.0, 0.0], [0.0, 3.0]]), source=space, target=space
)
history = phx.linalg.HistoryInitialGuess(operator, "example-shifted-family", capacity=3)
history = history.update(operator, jnp.asarray([1.0, 0.0]), time=0.0)
history = history.update(operator, jnp.asarray([0.0, 2.0]), time=1.0)
history_solve = phx.linalg.solve(
    phx.linalg.LinearSystem(operator),
    operator.mv(jnp.asarray([2.0, 6.0])),
    policy=phx.linalg.LinearSolvePolicy(phx.linalg.FGMRES(restart=2)),
    initial_guess=history,
)
history_evidence = history_solve.initial_guess

derivative = jnp.zeros((2, 2))
metric = jnp.ones((1, 2, 2, 3))
mass = jnp.ones((1, 2, 2))
gathers = jnp.asarray([[0, 1, 2, 3]], dtype=jnp.int32)
tensor = phx.equations.fem.CollocatedTensorProductOperator(
    derivative, metric, mass, gathers, 4
)
tensor_value = jnp.arange(1.0, 5.0)
tensor_defect = jnp.linalg.norm(tensor.mv(tensor_value) - tensor_value)

flow = phx.applications.incompressible_flow
flow_operators = flow.IncompressibleFlowOperators(
    lambda velocity, time, args: jnp.zeros_like(velocity),
    lambda rhs, gamma, time, args: rhs / gamma,
    lambda velocity, time, args: velocity,
    lambda rhs, time, args: -rhs,
    lambda pressure, time, args: pressure,
)
flow_state, flow_diagnostics = flow.pressure_correction_step(
    flow.IncompressibleFlowState(jnp.asarray([1.0, 2.0]), jnp.zeros((2,))),
    1.0,
    flow_operators,
    flow.IncompressibleFlowPolicy(pressure_increment=False),
    0.0,
)

if (
    not bool(history_evidence.accepted)
    or history_evidence.proposal_residual_norm > 1.0e-12
    or not jnp.allclose(history_solve.value, jnp.asarray([2.0, 6.0]))
    or tensor_defect > 1.0e-12
    or flow_diagnostics.divergence_after > 1.0e-12
):
    raise RuntimeError("FEM solver-acceleration smoke failed.")

print(
    {
        "history_guess_accepted": bool(history_evidence.accepted),
        "history_proposal_residual": float(history_evidence.proposal_residual_norm),
        "history_solution": history_solve.value.tolist(),
        "collocated_tensor_defect": float(tensor_defect),
        "corrected_velocity": flow_state.velocity.tolist(),
        "corrected_divergence": float(flow_diagnostics.divergence_after),
    }
)
