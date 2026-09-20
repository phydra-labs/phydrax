import json

import jax.numpy as jnp

import phydrax as phx


state = phx.materials.MaterialState(
    jnp.asarray((300.0, 400.0)),
    jnp.asarray((101325.0, 101325.0)),
    jnp.asarray(((1.0, 0.0), (0.5, 0.5))),
)
field = phx.materials.SpatialMaterialField.create(
    jnp.asarray(((0.0,), (1.0,))), jnp.asarray((1.0, 2.0)), state
)
model = phx.materials.SpatialICMEModel.create(
    jnp.asarray((100.0, 200.0)), jnp.asarray((1.0, 1.0)), homogenization="hill"
)
result = model.advance(
    field,
    jnp.asarray((350.0, 450.0)),
    jnp.asarray(((0.0, 1.0), (0.0, 1.0))),
    1.0,
)
print(
    json.dumps(
        {
            "effective_properties": result.effective_properties.tolist(),
            "phase_balance_residual": result.phase_balance_residual.tolist(),
        }
    )
)
