"""One-dimensional dielectric grating with order-resolved power."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


jax.config.update("jax_enable_x64", True)


def main() -> None:
    harmonic_plan = LatticeHarmonicPlan.parallelogramic((3,), (65,))
    harmonics = harmonic_plan.prepare(jnp.asarray(((1.0, 0.0),)))
    density = (harmonics.fractional_coordinates[..., 0] < 0.5).astype(jnp.float64)
    patterned = fm.FrequencyMaxwellMaterial(
        1.0 + 3.0 * density,
        material_id="dielectric-grating",
        passive=True,
        reciprocal=True,
    )
    vacuum = fm.FrequencyMaxwellMaterial(
        1.0,
        material_id="vacuum",
        passive=True,
        reciprocal=True,
    )
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="superstrate"),
        (
            fm.FourierModalLayer(
                patterned,
                0.2,
                fm.VectorFourierFactorizationPlan(
                    fm.AnalyticInterfaceFramePlan(
                        jnp.broadcast_to(
                            jnp.asarray((0.0, 1.0)),
                            harmonics.sample_shape + (2,),
                        ),
                        frame_id="lamellar-tangent",
                    )
                ),
                layer_id="grating",
            ),
        ),
        fm.HomogeneousMaxwellPort(vacuum, port_id="substrate"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(
        problem,
        fm.FourierModalSolvePolicy(
            boundary=fm.BoundaryCascadePolicy(
                doublings=10,
                initializer_order=6,
                paired_error=True,
                relative_tolerance=1e-7,
            )
        ),
    )
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[harmonics.plan.layout.zero_index],
        "te",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    print(
        {
            "left_incoming_power": float(result.weighted_left_incoming_power),
            "right_incoming_power": float(result.weighted_right_incoming_power),
            "left_outgoing_power": float(result.weighted_left_outgoing_power),
            "right_outgoing_power": float(result.weighted_right_outgoing_power),
            "net_port_power_into_stack": float(result.weighted_net_port_power_into_stack),
            "power_audit_residual": float(result.power_audit_residual),
            "status": int(result.status),
            "paired_error": float(result.diagnostics.maximum_boundary_paired_error),
        }
    )


if __name__ == "__main__":
    main()
