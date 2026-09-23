import json

import jax.numpy as jnp

import phydrax as phx


event = phx.manufacturing.ToolpathEvent(
    "track",
    "deposit",
    0.0,
    1.0,
    "machine",
    (0.0,),
    (1.0,),
    power_w=100.0,
    mass_rate_kg_s=1.0,
)
runtime = phx.manufacturing.ManufacturingRuntime.create(
    phx.manufacturing.ProcessSchedule.create((event,)),
    jnp.asarray(((0.0,), (1.0,))),
    jnp.ones(2),
    1.0,
)
workflow = phx.applications.additive_manufacturing.SpatialDEDWorkflow.create(
    runtime,
    jnp.asarray((10.0, 10.0)),
    jnp.asarray(((1.0, -1.0), (-1.0, 1.0))),
    jnp.zeros(2),
    phx.materials.SpatialICMEModel.create(
        jnp.asarray((1.0, 2.0)), jnp.asarray((1.0, 1.0))
    ),
    ambient_temperature_k=300.0,
    thermal_expansion_k_inv=1e-5,
    elastic_modulus_pa=1e9,
    reference_temperature_k=300.0,
)
state = phx.applications.additive_manufacturing.SpatialDEDState(
    phx.manufacturing.ManufacturingRuntimeState.initialize(2),
    jnp.full(2, 300.0),
    jnp.asarray(((1.0, 0.0), (1.0, 0.0))),
)
result = workflow.advance(state, 1.0, jnp.asarray(((0.0, 1.0), (0.0, 1.0))))
if not bool(result.successful):
    raise RuntimeError("Single-track DED workflow failed")
print(
    json.dumps(
        {
            "successful": True,
            "maximum_temperature_k": float(jnp.max(result.state.temperature_k)),
            "deposited_mass_kg": float(jnp.sum(result.state.runtime.deposited_mass_kg)),
            "energy_balance_residual_j": float(result.energy_balance_residual_j),
        }
    )
)
