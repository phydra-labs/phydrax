"""Localized current sheet reconstructed through Brillouin-zone integration."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.discretization.spectral import BrillouinZonePlan, LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


jax.config.update("jax_enable_x64", True)


def main() -> None:
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    rule = BrillouinZonePlan((3,)).prepare(harmonics)
    dielectric = fm.FrequencyMaxwellMaterial(
        4.0,
        material_id="emitter-host",
        passive=True,
        reciprocal=True,
    )
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    left_half = fm.FourierModalLayer(
        dielectric,
        0.1,
        fm.DirectFourierFactorizationPlan(),
        layer_id="host-left",
    )
    right_half = fm.FourierModalLayer(
        dielectric,
        0.1,
        fm.DirectFourierFactorizationPlan(),
        layer_id="host-right",
    )
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="superstrate"),
        (left_half, fm.FourierModalSourcePlane("dipole-plane"), right_half),
        fm.HomogeneousMaxwellPort(vacuum, port_id="substrate"),
    )
    policy = fm.FourierModalSolvePolicy(
        boundary=fm.BoundaryCascadePolicy(
            doublings=9,
            paired_error=False,
            relative_tolerance=1e-7,
        )
    )
    prepared = fm.prepare_brillouin_zone_maxwell(problem, rule, policy=policy)
    excitations = []
    for case in prepared.cases:
        coefficient = fm.point_source_coefficients(
            harmonics,
            case.problem.bloch_wavevector,
            jnp.asarray((0.25, 0.0)),
        )
        current = jnp.zeros((3, harmonics.harmonic_count, 1), dtype=jnp.complex128)
        current = current.at[1, :, 0].set(coefficient)
        zero_port = jnp.zeros((2 * harmonics.harmonic_count, 1), dtype=jnp.complex128)
        excitations.append(
            fm.FourierModalExcitation(
                zero_port,
                zero_port,
                source_ids=("dipole-plane",),
                electric_currents=(current,),
                magnetic_currents=(jnp.zeros_like(current),),
            )
        )
    result = fm.solve_fourier_modal_case_batch(prepared, tuple(excitations))
    emitted_grid = result.reflected_power + result.transmitted_power
    total = fm.integrate_brillouin_power(emitted_grid, rule)
    print(
        {
            "emitted_power": float(total[0]),
            "statuses": [int(value) for value in result.status.reshape((-1,))],
            "brillouin_points": int(np.prod(rule.plan.grid_shape)),
        }
    )


if __name__ == "__main__":
    main()
