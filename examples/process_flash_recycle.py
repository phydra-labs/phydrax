import json

import jax.numpy as jnp

import phydrax as phx


def residual(value):
    product, recycle = value
    return jnp.asarray((product + recycle - 10.0, recycle - 0.25 * product))


flowsheet = phx.process_systems.EquationOrientedFlowsheet.create(
    residual,
    jnp.asarray((10.0, 2.0)),
    jnp.asarray((10.0, 2.0)),
    lower_bounds=jnp.zeros(2),
)
result = flowsheet.solve(jnp.asarray((5.0, 1.0)))
print(
    json.dumps(
        {
            "successful": bool(result.successful),
            "product_and_recycle": result.variables.tolist(),
            "scaled_residual_norm": float(result.scaled_residual_norm),
        }
    )
)
