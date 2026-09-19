"""Run representative native omniphysics foundation controls."""

import json

import jax.numpy as jnp

import phydrax as phx


medium = phx.acoustics.AcousticMedium(1.2, 343.0)
pressure = phx.acoustics.monopole_pressure(1.0e-6, 1.0, 1000.0, medium)
current = phx.electrochemistry.butler_volmer_current_density(1.0, 0.01, 298.15)
wear = phx.tribology.archard_wear_depth(1.0e6, 10.0, 1.0e-5, 2.0e9)
flash = phx.process_systems.isothermal_flash(
    jnp.asarray((0.5, 0.5)), jnp.asarray((2.0, 0.5))
)

print(
    json.dumps(
        {
            "monopole_pressure_magnitude_pa": float(jnp.abs(pressure)),
            "butler_volmer_current_a_m2": float(current),
            "wear_depth_m": float(wear),
            "flash_vapor_fraction": float(flash.vapor_fraction),
            "flash_material_balance_error": float(flash.material_balance_error),
        },
        indent=2,
        sort_keys=True,
    )
)
