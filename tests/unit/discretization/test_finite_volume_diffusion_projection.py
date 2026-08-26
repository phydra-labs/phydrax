#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _grid(count=24, *, periodic=True):
    return phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=periodic),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))


def test_harmonic_face_interpolation_resolves_discontinuous_material_interface():
    grid = _grid(4, periodic=False)
    values = jnp.asarray([1.0, 1.0, 9.0, 9.0])
    interpolation = phx.discretization.FaceCoefficientPlan(grid, kind="harmonic")
    faces = interpolation.interpolate(values, "x")

    np.testing.assert_allclose(faces[2], 1.8, rtol=1e-12)
    np.testing.assert_allclose(faces[jnp.asarray([0, -1])], [1.0, 9.0])


def test_compressible_viscous_flux_vanishes_for_constant_primitive_state():
    grid = _grid(12, periodic=True)
    system = phx.equations.CompressibleNavierStokesSystem(
        phx.equations.ConstantTransport(0.1, 0.2)
    )
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    primitive = jnp.tile(jnp.asarray([1.0, 0.3, 1.2]), (12, 1))
    state = system.primitive_to_conserved(primitive)
    viscous = phx.discretization.ViscousFluxPlan()
    halo = phx.discretization.FiniteVolumeHaloPlan(
        discretization,
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    ).prepare()

    fluxes = viscous.face_fluxes(
        system, 0.0, state, discretization, halo
    )
    residual = viscous.residual(
        system, 0.0, state, discretization, halo
    )

    np.testing.assert_allclose(fluxes[0], 0.0, atol=1e-13)
    np.testing.assert_allclose(residual, 0.0, atol=1e-13)






def test_compiled_finite_volume_linearization_matches_direct_jvp():
    grid = _grid(20, periodic=True)
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="linearization-transport",
    )
    problem = phx.equations.ConservationProblemIR(
        "linearization",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(("x",)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.PiecewiseConstantReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(
        problem, discretization, method
    )
    x = grid.structured_axes[0].interval_centers
    state = jnp.sin(2.0 * jnp.pi * x)[..., None]
    tangent = jnp.cos(4.0 * jnp.pi * x)[..., None]
    _, linearized, _ = compiled.linearize(0.0, state)
    _, expected = jax.jvp(lambda value: compiled(0.0, value), (state,), (tangent,))

    np.testing.assert_allclose(linearized(tangent), expected, rtol=1e-12, atol=1e-12)
