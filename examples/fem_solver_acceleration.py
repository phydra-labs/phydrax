#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


space = phx.linalg.ArraySpace((2,))
operator = phx.linalg.DenseLinearOperator(
    jnp.asarray([[2.0, 0.0], [0.0, 3.0]]), source=space, target=space
)
history = phx.linalg.LinearSolveHistory.empty(
    operator,
    phx.linalg.LinearSolveHistoryPolicy("projection", capacity=3),
    "example-shifted-family",
)
history = history.update(operator, jnp.asarray([1.0, 0.0]), time=0.0)
history = history.update(operator, jnp.asarray([0.0, 2.0]), time=1.0)
guess, history_diagnostics = history.initial_guess(operator.mv(jnp.asarray([2.0, 6.0])))

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
    not jnp.allclose(guess, jnp.asarray([2.0, 6.0]))
    or history_diagnostics.projection_residual_norm > 1.0e-12
    or tensor_defect > 1.0e-12
    or flow_diagnostics.divergence_after > 1.0e-12
):
    raise RuntimeError("FEM solver-acceleration smoke failed.")

print(
    {
        "history_guess": guess.tolist(),
        "history_projection_residual": float(
            history_diagnostics.projection_residual_norm
        ),
        "collocated_tensor_defect": float(tensor_defect),
        "corrected_velocity": flow_state.velocity.tolist(),
        "corrected_divergence": float(flow_diagnostics.divergence_after),
    }
)
