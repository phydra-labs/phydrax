# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Apply conservative grey and polarized radiation feedback to GRMHD matter."""

from __future__ import annotations

from jax import config


config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp

import phydrax as phx


def main() -> None:
    scale = phx.RelativityScaleContract.geometric(phx.units.KILOGRAM)
    convention = phx.metrix.RelativityConvention.canonical()
    shape = (4,)
    identity = jnp.broadcast_to(jnp.eye(3), shape + (3, 3))
    geometry = phx.metrix.ADMGridGeometry(
        jnp.ones(shape),
        jnp.zeros(shape + (3,)),
        identity,
        identity,
        jnp.ones(shape),
        jnp.zeros(shape + (3, 3)),
        jnp.ones(shape, dtype=bool),
        jnp.ones(shape, dtype=bool),
        snapshot_token=jnp.asarray(0, dtype=jnp.int32),
        chart_id="cartesian",
        convention_id=convention.convention_id,
        scale_id=scale.scale_id,
        topology_id="feedback-line",
        geometry_lineage_id="minkowski-line",
    )
    eos = phx.equations.GammaLawEOS(scale, 4.0 / 3.0)
    material = phx.equations.IdealValenciaGRMHDSystem(
        eos,
        scale,
        convention=convention,
        maximum_magnetization=1.0e6,
        recovery_iterations=32,
        enthalpy_iterations=32,
    )
    radiation = phx.equations.GRGreyM1RadiationSystem(scale, convention)
    interaction = phx.equations.GRGreyRadiationInteractionPlan(
        radiation,
        phx.equations.ConstantGRGreyOpacityPlan(
            planck_absorption=0.5,
            planck_emission=0.0,
            scattering=0.1,
        ),
    )
    source = phx.solver.GRRMHDImplicitSourcePlan(
        material, interaction, caloric_temperature_scale=1.0
    )

    primitive = jnp.zeros(shape + (8,))
    primitive = primitive.at[..., 0].set(1.0)
    primitive = primitive.at[..., 4].set(0.2)
    primitive = primitive.at[..., 5].set(0.1)
    material_state = material.primitive_to_conserved(primitive, geometry)
    radiation_state = jnp.broadcast_to(jnp.asarray((2.0, 0.05, 0.0, 0.0)), shape + (4,))
    coupled = jax.jit(
        lambda material_values, radiation_values: source.advance(
            material_values, radiation_values, 0.05, geometry
        )
    )(material_state, radiation_state)
    if not bool(coupled.accepted):
        raise RuntimeError("The conservative grey source solve was not accepted.")

    polarized = phx.solver.GRPolarizedRadiationFeedbackPlan(scale, convention)
    stokes = jnp.broadcast_to(jnp.asarray(((1.0, 0.2, 0.0, 0.0),)), shape + (1, 4))
    polarized_state = polarized.initialize(
        stokes,
        coupled.material_state[..., 4],
        coupled.material_state[..., 1:4],
        geometry,
    )
    absorption = jnp.broadcast_to(jnp.asarray((0.25, 0.0, 0.0, 0.0)), shape + (1, 4))
    faraday = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 0.5)), shape + (1, 3))
    propagation = phx.solver.polarized_propagation_matrix(absorption, faraday)
    direction = jnp.broadcast_to(jnp.asarray((1.0, 0.0, 0.0)), shape + (1, 3))
    feedback = polarized.advance(
        polarized_state,
        0.1,
        jnp.zeros_like(stokes),
        propagation,
        direction,
        geometry,
    )
    if not bool(feedback.accepted):
        raise RuntimeError("The polarized feedback step was not accepted.")

    print(
        "grey_source_energy_defect", float(jnp.max(jnp.abs(coupled.ledger.energy_defect)))
    )
    print(
        "grey_source_momentum_defect",
        float(jnp.max(jnp.abs(coupled.ledger.momentum_defect))),
    )
    print(
        "polarized_feedback_energy_defect",
        float(jnp.max(jnp.abs(feedback.ledger.energy_balance_residual))),
    )
    print(
        "polarized_feedback_momentum_defect",
        float(jnp.max(jnp.abs(feedback.ledger.momentum_balance_residual))),
    )
    print("minimum_stokes_i", float(jnp.min(feedback.state.stokes[..., 0])))


if __name__ == "__main__":
    main()
