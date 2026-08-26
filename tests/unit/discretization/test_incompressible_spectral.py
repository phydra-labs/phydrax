import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _periodic_space(count=8):
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(count),
            phx.discretization.FourierBasisPlan(count),
        ),
        axis_names=("x", "y"),
        field_name="velocity",
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))


def _channel_space():
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.ChebyshevBasisPlan(8),
            phx.discretization.FourierBasisPlan(4),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(jnp.asarray([[0.0, -1.0, 0.0], [2.0 * jnp.pi, 1.0, 2.0 * jnp.pi]]))


def test_periodic_incompressible_dynamics_preserves_constraints_and_gradients():
    space = _periodic_space()
    method = phx.discretization.PseudospectralMethodPlan(
        dealiasing=phx.discretization.PaddingDealiasingPlan(2)
    )
    compiled = phx.equations.compile_periodic_incompressible_flow(
        phx.equations.IncompressibleFlowProblem(2, 0.01),
        space,
        method,
    )
    x, y = jnp.meshgrid(
        space.axes[0].nodes,
        space.axes[1].nodes,
        indexing="ij",
    )
    velocity = jnp.stack((jnp.sin(2.0 * jnp.pi * y), jnp.sin(2.0 * jnp.pi * x)), axis=-1)
    state = compiled.project_state(velocity)
    rate = jax.jit(lambda value: compiled(jnp.asarray(0.0), value, None))(state)
    diagnostics = compiled.diagnostics(state)
    derivative = jax.grad(
        lambda amplitude: (
            compiled.diagnostics(
                compiled.project_state(amplitude * velocity)
            ).kinetic_energy
        )
    )(jnp.asarray(1.0))

    assert compiled.spatial_method.dealiasing.report.exact
    assert compiled.projector.divergence_norm(state) < 1e-12
    assert compiled.projector.divergence_norm(rate) < 1e-11
    assert jnp.all(jnp.isfinite(compiled.pressure_coefficients(0.0, state)))
    assert diagnostics.imaginary_leakage < 1e-12
    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(np.asarray(derivative), 1.0, atol=1e-10)


def test_hermitian_coordinates_and_spectral_symmetry_preserve_real_field_norm():
    space = _periodic_space()
    x, y = jnp.meshgrid(
        space.axes[0].nodes,
        space.axes[1].nodes,
        indexing="ij",
    )
    velocity = jnp.stack((jnp.sin(2.0 * jnp.pi * x), jnp.cos(2.0 * jnp.pi * y)), axis=-1)
    state = space.project(velocity)
    coordinates = phx.discretization.HermitianSpectralCoordinates(
        space, component_shape=(2,)
    )
    real = coordinates.to_real_coordinates(state)
    restored = coordinates.from_real_coordinates(real)
    first = phx.discretization.TensorSpectralSymmetry(
        space,
        translations=(0.25, 0.0),
        component_count=2,
    )
    second = phx.discretization.TensorSpectralSymmetry(
        space,
        translations=(0.125, 0.0),
        component_count=2,
    )
    reflected = phx.discretization.TensorSpectralSymmetry(
        space,
        axis_signs=(-1, 1),
        translations=(0.2, 0.1),
        component_count=2,
    )
    identity = first.compose(first.inverse())

    np.testing.assert_allclose(np.asarray(restored), np.asarray(state), atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(real)),
        np.asarray(jnp.linalg.norm(state)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(first.compose(second).apply(state)),
        np.asarray(first.apply(second.apply(state))),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(reflected.compose(first).apply(state)),
        np.asarray(reflected.apply(first.apply(state))),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(reflected.compose(reflected.inverse()).apply(state)),
        np.asarray(state),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(identity.apply(state)), np.asarray(state), atol=1e-12
    )
    assert coordinates.reality_defect(first.apply(state)) < 1e-12


def test_channel_stokes_enforces_couette_walls_and_bulk_flux():
    space = _channel_space()
    y = space.axes[1].nodes
    couette = jnp.zeros(space.physical_shape + (3,)).at[..., 0].set(y[None, :, None])
    couette_modal = space.project(couette)
    prescribed = phx.discretization.ChannelStokesPlan(
        space,
        0.1,
        lower_wall_velocity=(-1.0, 0.0, 0.0),
        upper_wall_velocity=(1.0, 0.0, 0.0),
    ).prepare(1.0)
    prescribed_result = prescribed.solve(couette_modal)
    reconstructed = space.reconstruct(prescribed_result.velocity)
    fixed_flux = phx.discretization.ChannelStokesPlan(
        space,
        0.1,
        mean_constraint=phx.discretization.ChannelMeanConstraint("bulk_flux", (0.4, 0.0)),
    ).prepare(1.0)
    flux_result = fixed_flux.solve(jnp.zeros_like(couette_modal))

    assert bool(prescribed_result.successful)
    np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(couette), atol=1e-11)
    assert prescribed_result.diagnostics.divergence_norm < 1e-11
    assert prescribed_result.diagnostics.wall_residual < 1e-11
    assert prescribed_result.diagnostics.pressure_gauge_residual < 1e-11
    assert bool(flux_result.successful)
    np.testing.assert_allclose(
        np.asarray(flux_result.diagnostics.bulk_velocity),
        np.asarray([0.4, 0.0]),
        atol=1e-11,
    )
    assert jnp.abs(flux_result.pressure_gradient[0]) > 0.0
    assert flux_result.diagnostics.divergence_norm < 1e-11
    assert flux_result.diagnostics.wall_residual < 1e-11
