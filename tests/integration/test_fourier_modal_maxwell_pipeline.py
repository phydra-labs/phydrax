from __future__ import annotations

import jax.numpy as jnp

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


def test_periodic_layer_pipeline_produces_scattering_fields_and_farfield() -> None:
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    film = fm.FrequencyMaxwellMaterial(2.25, material_id="film")
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (
            fm.FourierModalLayer(
                film,
                0.1,
                fm.DirectFourierFactorizationPlan(),
                layer_id="film",
            ),
        ),
        fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
    )
    prepared = fm.prepare_fourier_modal_maxwell(
        problem,
        fm.FourierModalSolvePolicy(
            boundary=fm.BoundaryCascadePolicy(
                doublings=6,
                initializer_order=7,
                paired_error=False,
                relative_tolerance=1e-7,
            )
        ),
    )
    excitation = fm.plane_wave_excitation(
        prepared.scattering,
        harmonics.plan.layout.mode_ids[0],
        "te",
    )
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    field = fm.fields_in_layer(prepared, result, 0, 0.05)
    farfield = fm.diffraction_order_far_field(prepared, result)
    assert int(result.status) == int(fm.FourierModalSolveStatus.SUCCESS)
    assert field.electric_field.shape == (3, 3, 1)
    assert farfield.power.shape == (1, 2, 1)
