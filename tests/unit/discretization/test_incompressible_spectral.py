import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

import phydrax as phx


def _periodic_space(count=8):
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(count),
            phx.discretization.FourierBasisPlan(count),
        ),
        axis_names=("x", "y"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
        )
    )


def _channel_space(wall_count=8):
    return phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(4),
            phx.discretization.ChebyshevBasisPlan(wall_count),
            phx.discretization.FourierBasisPlan(4),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
            phx.discretization.AxisDomain.interval(-1.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 2.0 * jnp.pi),
        )
    )


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
    diagnostics = compiled.diagnostics(0.0, state)
    derivative = jax.grad(
        lambda amplitude: (
            compiled.diagnostics(
                0.0,
                compiled.project_state(amplitude * velocity),
            ).kinetic_energy
        )
    )(jnp.asarray(1.0))

    assert compiled.spatial_method.dealiasing.report.exact
    assert compiled.projector.divergence_norm(state) < 1e-12
    assert compiled.projector.divergence_norm(rate) < 1e-11
    assert jnp.all(jnp.isfinite(compiled.pressure_coefficients(0.0, state)))
    assert diagnostics.imaginary_leakage < 1e-12
    assert jnp.abs(diagnostics.advective_energy_rate) < 1e-10
    assert (
        jnp.abs(diagnostics.molecular_energy_rate + diagnostics.molecular_dissipation)
        < 1e-10
    )
    assert jnp.abs(diagnostics.energy_balance_defect) < 1e-10
    assert diagnostics.pressure_gauge_residual < 1e-12
    assert jnp.isfinite(derivative)
    np.testing.assert_allclose(np.asarray(derivative), 1.0, atol=1e-10)


def test_periodic_leray_removes_gradient_rhs_and_is_idempotent_in_three_dimensions():
    space = phx.discretization.TensorSpectralPlan(
        (
            phx.discretization.FourierBasisPlan(6),
            phx.discretization.FourierBasisPlan(6),
            phx.discretization.FourierBasisPlan(6),
        ),
        axis_names=("x", "y", "z"),
        field_name="velocity",
    ).prepare(
        (
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
            phx.discretization.AxisDomain.periodic(0.0, 1.0),
        )
    )
    projector = phx.discretization.PeriodicLerayProjector(space)
    potential = jnp.ones(space.modal_shape, dtype=complex)
    gradient = jnp.stack(
        tuple(1j * wave * potential for wave in projector.wavenumbers), axis=-1
    )
    projected_gradient = projector.project(gradient)
    candidate = (
        jnp.arange(projector.state_size, dtype=float)
        .reshape(projector.state_shape)
        .astype(complex)
    )
    projected = projector.project(candidate)

    np.testing.assert_allclose(projected_gradient, 0.0, atol=2e-12)
    np.testing.assert_allclose(projector.project(projected), projected, atol=2e-12)
    assert projector.divergence_norm(projected) < 2e-11
    pressure = projector.pressure_from_unconstrained_rhs(gradient)
    assert (
        jnp.max(jnp.abs(jnp.where(projector.wavenumber_squared == 0.0, pressure, 0.0)))
        < 1e-12
    )


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

    dense = (
        phx.discretization.ChannelStokesPlan(
            space,
            0.1,
            lower_wall_velocity=(-1.0, 0.0, 0.0),
            upper_wall_velocity=(1.0, 0.0, 0.0),
            route="dense_reference",
        )
        .prepare(1.0)
        .solve(couette_modal)
    )
    assert bool(prescribed_result.successful)
    np.testing.assert_allclose(np.asarray(reconstructed), np.asarray(couette), atol=1e-11)
    assert prescribed_result.diagnostics.divergence_norm < 1e-11
    assert prescribed_result.diagnostics.wall_residual < 1e-11
    assert prescribed_result.diagnostics.pressure_gauge_residual < 1e-11
    assert prescribed.report.upper_bandwidth == 8
    assert prescribed.report.lower_bandwidth == 0
    assert prescribed.report.correction_rank == 4
    assert prescribed.report.persistent_bytes == (
        prescribed.report.shared_basis_bytes
        + prescribed.report.operator_bytes
        + prescribed.report.factor_bytes
    )
    assert prescribed.report.preparation_bytes >= prescribed.report.persistent_bytes
    assert prescribed.report.required_unsharded_axes == ("y",)
    assert prescribed.blocks is None
    np.testing.assert_allclose(
        prescribed_result.velocity, dense.velocity, atol=2e-11, rtol=2e-11
    )
    assert bool(flux_result.successful)
    np.testing.assert_allclose(
        np.asarray(flux_result.diagnostics.bulk_velocity),
        np.asarray([0.4, 0.0]),
        atol=1e-11,
    )
    assert jnp.abs(flux_result.pressure_gradient[0]) > 0.0
    assert flux_result.diagnostics.divergence_norm < 1e-11
    assert flux_result.diagnostics.wall_residual < 1e-11


def _manufactured_nonzero_channel_mode(space, prepared, ix, iz):
    analysis = space.axes[1].modal_transform.analysis
    derivative = space.axes[1].derivative_matrix
    y = space.axes[1].nodes
    kx = prepared.streamwise_wavenumbers[ix, iz]
    kz = prepared.spanwise_wavenumbers[ix, iz]
    wave_square = kx**2 + kz**2
    wall_normal = analysis @ (1.0 - y**2) ** 2
    vorticity = analysis @ ((0.3 + 0.2j) * (1.0 - y**2))
    derivative_v = derivative @ wall_normal
    velocity_u = (1j * kx * derivative_v - 1j * kz * vorticity) / wave_square
    velocity_w = (1j * kz * derivative_v + 1j * kx * vorticity) / wave_square
    pressure = analysis @ ((0.2 - 0.1j) * y)
    velocity = jnp.zeros(space.modal_shape + (3,), dtype=complex)
    velocity = velocity.at[ix, :, iz].set(
        jnp.stack((velocity_u, wall_normal, velocity_w), axis=-1)
    )
    pressure_modes = jnp.zeros(space.modal_shape, dtype=complex)
    pressure_modes = pressure_modes.at[ix, :, iz].set(pressure)
    second_derivative = oe.contract(
        "ij,xjzc->xizc", derivative @ derivative, velocity, backend="jax"
    )
    helmholtz = (prepared.shift + 0.1 * wave_square) * velocity - 0.1 * second_derivative
    pressure_derivative = derivative @ pressure
    gradient = jnp.zeros_like(velocity)
    gradient = gradient.at[ix, :, iz, 0].set(1j * kx * pressure)
    gradient = gradient.at[ix, :, iz, 1].set(pressure_derivative)
    gradient = gradient.at[ix, :, iz, 2].set(1j * kz * pressure)
    return velocity, pressure_modes, helmholtz + gradient


def test_channel_pressure_elimination_matches_manufactured_primitive_oracle_and_ad():
    space = _channel_space()
    banded = phx.discretization.ChannelStokesPlan(space, 0.1).prepare(1.0)
    dense = phx.discretization.ChannelStokesPlan(
        space, 0.1, route="dense_reference"
    ).prepare(1.0)
    manufactured_cases = (
        _manufactured_nonzero_channel_mode(space, banded, 1, 0),
        _manufactured_nonzero_channel_mode(space, banded, 0, 1),
        _manufactured_nonzero_channel_mode(space, banded, 1, 1),
    )
    for expected_velocity, expected_pressure, rhs in manufactured_cases:
        banded_result = banded.solve(rhs)
        dense_result = dense.solve(rhs)
        assert bool(banded_result.successful)
        assert bool(dense_result.successful)
        np.testing.assert_allclose(
            banded_result.velocity, expected_velocity, atol=3e-10, rtol=3e-10
        )
        np.testing.assert_allclose(
            banded_result.pressure, expected_pressure, atol=3e-10, rtol=3e-10
        )
        np.testing.assert_allclose(
            banded_result.velocity, dense_result.velocity, atol=3e-10, rtol=3e-10
        )
        np.testing.assert_allclose(
            banded_result.pressure, dense_result.pressure, atol=3e-10, rtol=3e-10
        )
        assert banded_result.diagnostics.momentum_constraint_residual < 1e-10
        assert banded_result.diagnostics.divergence_norm < 1e-10
        assert banded_result.diagnostics.wall_residual < 1e-10

    rhs = manufactured_cases[-1][2]
    coordinates = jnp.stack((jnp.real(rhs), jnp.imag(rhs)), axis=-1)
    direction = coordinates

    def solve_coordinates(values):
        modal_rhs = values[..., 0] + 1j * values[..., 1]
        result = banded.solve(modal_rhs)
        return jnp.concatenate(
            (
                jnp.real(result.velocity).reshape((-1,)),
                jnp.imag(result.velocity).reshape((-1,)),
                jnp.real(result.pressure).reshape((-1,)),
                jnp.imag(result.pressure).reshape((-1,)),
            )
        )

    output, tangent = jax.jvp(solve_coordinates, (coordinates,), (direction,))
    cotangent = jnp.linspace(-0.5, 0.5, output.size, dtype=output.dtype)
    _, pullback = jax.vjp(solve_coordinates, coordinates)
    input_cotangent = pullback(cotangent)[0]
    np.testing.assert_allclose(
        jnp.vdot(tangent, cotangent),
        jnp.vdot(direction, input_cotangent),
        atol=2e-9,
        rtol=2e-9,
    )
    assert jnp.all(jnp.isfinite(tangent))
    assert jnp.all(jnp.isfinite(input_cotangent))


def test_channel_zero_mode_recovers_pressure_and_preserves_all_wall_traces():
    space = _channel_space()
    lower = (-0.4, 0.15, 0.2)
    upper = (0.6, 0.15, -0.3)
    imposed_gradient = (0.12, -0.08)
    constraint = phx.discretization.ChannelMeanConstraint(
        "pressure_gradient", imposed_gradient
    )
    banded = phx.discretization.ChannelStokesPlan(
        space,
        0.1,
        lower_wall_velocity=lower,
        upper_wall_velocity=upper,
        mean_constraint=constraint,
    ).prepare(1.0)
    dense = phx.discretization.ChannelStokesPlan(
        space,
        0.1,
        lower_wall_velocity=lower,
        upper_wall_velocity=upper,
        mean_constraint=constraint,
        route="dense_reference",
    ).prepare(1.0)
    analysis = space.axes[1].modal_transform.analysis
    derivative = space.axes[1].derivative_matrix
    y = space.axes[1].nodes
    scale = banded.horizontal_constant_scale
    physical_velocity = jnp.stack(
        (
            lower[0] + 0.5 * (upper[0] - lower[0]) * (y + 1.0),
            jnp.full_like(y, lower[1]),
            lower[2] + 0.5 * (upper[2] - lower[2]) * (y + 1.0),
        ),
        axis=-1,
    )
    zero_velocity = scale * (analysis @ physical_velocity)
    zero_pressure = scale * (analysis @ (0.35 * y))
    second_derivative = derivative @ derivative
    zero_rhs = zero_velocity - 0.1 * (second_derivative @ zero_velocity)
    zero_rhs = zero_rhs.at[0, 0].add(-scale * imposed_gradient[0])
    zero_rhs = zero_rhs.at[0, 2].add(-scale * imposed_gradient[1])
    zero_rhs = zero_rhs.at[:, 1].add(derivative @ zero_pressure)
    rhs = jnp.zeros(space.modal_shape + (3,), dtype=complex)
    rhs = rhs.at[0, :, 0].set(zero_rhs)
    expected_velocity = jnp.zeros_like(rhs).at[0, :, 0].set(zero_velocity)
    expected_pressure = (
        jnp.zeros(space.modal_shape, dtype=complex).at[0, :, 0].set(zero_pressure)
    )
    banded_result = banded.solve(rhs)
    dense_result = dense.solve(rhs)
    assert bool(banded_result.successful)
    assert bool(dense_result.successful)
    np.testing.assert_allclose(
        banded_result.velocity, expected_velocity, atol=3e-10, rtol=3e-10
    )
    np.testing.assert_allclose(
        banded_result.pressure, expected_pressure, atol=3e-10, rtol=3e-10
    )
    np.testing.assert_allclose(
        banded_result.velocity, dense_result.velocity, atol=3e-10, rtol=3e-10
    )
    np.testing.assert_allclose(
        banded_result.pressure, dense_result.pressure, atol=3e-10, rtol=3e-10
    )
    np.testing.assert_allclose(
        banded_result.pressure_gradient, imposed_gradient, atol=1e-12
    )
    assert banded_result.diagnostics.momentum_constraint_residual < 1e-10
    assert banded_result.diagnostics.divergence_norm < 1e-10
    assert banded_result.diagnostics.wall_residual < 1e-10
    assert banded_result.diagnostics.pressure_gauge_residual < 1e-10


def test_channel_tau_rank_is_fixed_and_factor_storage_is_linear_in_wall_count():
    small = phx.discretization.ChannelStokesPlan(_channel_space(6), 0.1).prepare(1.0)
    large = phx.discretization.ChannelStokesPlan(_channel_space(10), 0.1).prepare(1.0)
    assert small.report.correction_rank == large.report.correction_rank == 4
    assert small.ultraspherical.helmholtz.rank == 2
    assert small.ultraspherical.biharmonic.rank == 4
    assert small.ultraspherical.pressure_recovery.rank == 1
    assert large.report.factor_bytes > small.report.factor_bytes
    assert large.report.factor_bytes / small.report.factor_bytes < 2.0
