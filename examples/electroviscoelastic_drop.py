import json

import jax.numpy as jnp

import phydrax as phx


ehd = phx.electrohydrodynamics.CoupledElectrohydrodynamicSolver.create(
    jnp.ones(2),
    jnp.eye(2),
    jnp.eye(2),
    jnp.zeros((2, 2)),
    jnp.eye(2),
    jnp.ones(2) * 2.0,
    jnp.ones(2) * 0.5,
    1,
)
law = phx.rheology.ViscoelasticLaw("oldroyd-b", 1.0, 2.0)
rheology = phx.rheology.SpatialConformationSolver.create(
    jnp.ones(2), jnp.zeros((2, 2)), law
)
workflow = phx.applications.electroviscoelastic.SpatialElectroViscoelasticWorkflow.create(
    ehd,
    rheology,
    rheology,
    jnp.asarray((1.0, 0.0)),
    jnp.zeros((2, 2)),
    jnp.zeros((2, 2)),
    jnp.asarray((0,)),
    jnp.asarray((1,)),
    jnp.asarray(((1.0,),)),
    jnp.ones(1),
    jnp.zeros((1, 1)),
)
state = phx.applications.electroviscoelastic.SpatialElectroViscoelasticState(
    phx.electrohydrodynamics.ElectrohydrodynamicState(
        jnp.asarray((1.0, -1.0)), jnp.zeros(2), jnp.zeros((2, 1))
    ),
    jnp.ones((2, 1, 1)),
    jnp.ones((2, 1, 1)),
    jnp.zeros(1),
)
result = workflow.advance(state, jnp.zeros(2), 0.1)
print(
    json.dumps(
        {
            "successful": bool(result.successful),
            "traction": result.interface_traction_pa.tolist(),
            "surface_charge_balance_residual_c": float(
                result.surface_charge_balance_residual_c
            ),
            "polymer_energy_j": float(result.polymer_free_energy_j),
        }
    )
)
