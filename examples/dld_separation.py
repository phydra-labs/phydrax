#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json

import jax
import jax.numpy as jnp

import phydrax as phx


_FLOW_ID = "synthetic-dld-demonstration-flow"


def _velocity(time, position, args):
    del time, args
    return jnp.broadcast_to(jnp.asarray((1.0, 0.0)), position.shape)


def main() -> None:
    mf = phx.applications.microfluidics
    topology = mf.DLDTopology(
        row_count=4,
        column_count=1,
        period_rows=4,
        outlet_count=2,
        particle_capacity=2,
    )
    design = mf.DLDDesign(
        post_radius=0.1,
        axial_pitch=1.0,
        lateral_pitch=1.0,
        row_shift=0.25,
        channel_lower=0.0,
        channel_upper=3.0,
        first_row_x=1.0,
        outlet_x=5.0,
        depth=0.1,
        length_unit_id="mm",
    )
    geometry = mf.DLDGeometryPlan(topology, design)
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray((0, 1)), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    population = phx.discretization.ParticlePopulationPlan(particles)
    properties = phx.discretization.FiniteParticleProperties(
        jnp.asarray((0.05, 0.05)),
        jnp.ones((2,)),
        jnp.ones((2,)),
        jnp.zeros((2, 2, 2)),
    )
    units = phx.discretization.FiniteParticleTransportUnits(
        length_unit_id="mm",
        time_unit_id="s",
        mass_unit_id="kg",
        temperature_unit_id="K",
        frame="device",
    )
    field = phx.discretization.FiniteParticleVelocityFieldPlan(
        _velocity,
        provider_id=_FLOW_ID,
        velocity_unit_id="mm/s",
        frame="device",
    )
    transport = phx.solver.FiniteParticleTransportPlan(
        population,
        properties,
        units,
        field,
        geometry,
        motion=phx.discretization.FiniteParticleMotionKind.OVERDAMPED_STOKES,
    )
    workflow = mf.DLDWorkflowPlan(
        geometry,
        transport,
        mf.DLDOutletPlan(5.0, jnp.asarray((0.0, 1.5, 3.0))),
        mf.DLDMetricPlan(2, 2, jnp.asarray((0, 1))),
        jnp.asarray((0, 1)),
        step_count=6,
        step_size=1.0,
        flow_model_id=_FLOW_ID,
        screening=mf.DLDEmpiricalScreenPlan(
            coefficient=1.4,
            exponent=0.48,
            minimum_shift_fraction=0.1,
            maximum_shift_fraction=0.4,
            source_id="demonstration-coefficient-set",
        ),
    )
    state = transport.initialize(
        population.initialize(),
        jnp.asarray(((0.0, 1.2), (0.0, 2.0))),
        jnp.zeros((2, 2)),
        jax.random.key(0),
    )
    result = workflow.run(
        state,
        phx.AdmissibilityHeader(
            jnp.asarray(1.0),
            jnp.asarray(0, dtype=jnp.uint32),
            _FLOW_ID,
            "synthetic-demonstration-evidence",
        ),
        volume_flow=jnp.asarray(2.0),
        pressure_drop=jnp.asarray(4.0),
    )
    if not bool(result.successful):
        raise RuntimeError("DLD separation workflow failed")
    print(
        json.dumps(
            {
                "scientific_status": "synthetic-invariant-demonstration",
                "successful": True,
                "terminal_codes": [
                    int(value) for value in result.final_state.terminal_code
                ],
                "transfer_matrix": result.metrics.transfer_matrix.tolist(),
                "purity": result.metrics.purity.tolist(),
                "recovery": result.metrics.recovery.tolist(),
                "throughput_proxy": float(result.metrics.throughput_proxy),
                "hydraulic_resistance": float(result.metrics.hydraulic_resistance),
                "critical_diameter": float(result.screening.critical_diameter),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
