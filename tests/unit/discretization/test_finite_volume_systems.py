#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest

import phydrax as phx


def _cell_grid(shape, *, periodic=None):
    periodic = (False,) * len(shape) if periodic is None else periodic
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xyz"[: len(shape)]),
    ).prepare(jnp.stack((jnp.zeros(len(shape)), jnp.ones(len(shape)))))


def _sine_cell_averages(edges):
    widths = edges[1:] - edges[:-1]
    return (jnp.cos(2.0 * jnp.pi * edges[:-1]) - jnp.cos(2.0 * jnp.pi * edges[1:])) / (
        2.0 * jnp.pi * widths
    )


_ALE_SOLVER_TYPES = (
    phx.discretization.RusanovFluxPlan,
    phx.discretization.HLLFluxPlan,
    phx.discretization.HLLCFluxPlan,
    phx.discretization.EinfeldtHLLFluxPlan,
)


def _compile_two_dimensional_system(
    geometry_kind,
    system,
    interface_solver,
    *,
    entropy_pair=None,
):
    vertices = np.asarray(
        (
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        )
    )
    if geometry_kind in ("structured", "mapped"):
        grid = _cell_grid((2, 2))
        discretization = phx.discretization.FiniteVolumePlan(
            grid,
            component_names=system.component_names,
        ).prepare()
        if geometry_kind == "mapped":
            discretization = phx.discretization.MappedFiniteVolumePlan(
                discretization,
                lambda point: point,
                mapping_id=f"{geometry_kind}-compile-identity",
            ).prepare()
        pair = phx.discretization.FiniteVolumeBoundaryPair(
            phx.discretization.ExtrapolationBoundary(),
            phx.discretization.ExtrapolationBoundary(),
        )
        boundaries = phx.discretization.FiniteVolumeBoundarySet(
            ("x", "y"),
            (pair, pair),
        )
        method = phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            interface_solver,
        )
    elif geometry_kind == "triangle":
        discretization = phx.discretization.TriangleFiniteVolumePlan(
            vertices,
            np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
            component_names=system.component_names,
        ).prepare()
        boundaries = phx.discretization.TriangleFiniteVolumeBoundarySet(
            discretization.boundary_patch_names,
            {
                name: phx.discretization.ExtrapolationBoundary()
                for name in discretization.boundary_patch_names
            },
        )
        method = phx.discretization.TriangleFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            interface_solver,
        )
    elif geometry_kind == "unstructured":
        discretization = phx.discretization.UnstructuredFiniteVolumePlan(
            vertices,
            quadrilaterals=np.asarray(((0, 1, 2, 3),), dtype=np.int32),
            component_names=system.component_names,
        ).prepare()
        boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
            discretization.boundary_patch_names,
            {
                name: phx.discretization.ExtrapolationBoundary()
                for name in discretization.boundary_patch_names
            },
        )
        method = phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            interface_solver,
        )
    else:
        raise AssertionError(f"Unknown finite-volume geometry kind: {geometry_kind}")
    problem = phx.equations.ConservationProblemIR(
        f"{geometry_kind}-{type(interface_solver).__name__}-compile",
        "state",
        system,
        boundaries,
    )
    return phx.equations.compile_conservation_problem(
        problem,
        discretization,
        method,
        entropy_pair=entropy_pair,
    )


@pytest.mark.parametrize("geometry_kind", ("structured", "mapped"))
def test_entropy_pair_compiles_for_structured_geometry(geometry_kind):
    system = phx.equations.EulerSystem(2)
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    without_pair = _compile_two_dimensional_system(
        geometry_kind,
        system,
        phx.discretization.RusanovFluxPlan(),
    )
    with_pair = _compile_two_dimensional_system(
        geometry_kind,
        system,
        phx.discretization.RusanovFluxPlan(),
        entropy_pair=pair,
    )

    assert with_pair.dynamics.entropy_pair is pair
    assert with_pair.dynamics.dynamics_id != without_pair.dynamics.dynamics_id
    assert with_pair.compilation_id != without_pair.compilation_id


@pytest.mark.parametrize("geometry_kind", ("triangle", "unstructured"))
def test_entropy_pair_rejects_unsupported_finite_volume_geometry(geometry_kind):
    system = phx.equations.EulerSystem(2)
    pair = phx.equations.ideal_gas_euler_entropy_pair(system)
    with pytest.raises(ValueError, match="structured and mapped"):
        _compile_two_dimensional_system(
            geometry_kind,
            system,
            phx.discretization.RusanovFluxPlan(),
            entropy_pair=pair,
        )


def _two_dimensional_euler_faces():
    system = phx.equations.EulerSystem(2)
    left = system.primitive_to_conserved(
        jnp.asarray(
            (
                (1.0, 0.7, -0.25, 1.1),
                (0.8, -0.35, 0.45, 0.75),
                (1.2, 0.1, 0.6, 1.35),
            )
        )
    )
    right = system.primitive_to_conserved(
        jnp.asarray(
            (
                (0.9, -0.2, 0.3, 0.85),
                (1.05, 0.4, -0.15, 1.2),
                (0.7, -0.5, 0.2, 0.65),
            )
        )
    )
    inverse_sqrt_two = 1.0 / jnp.sqrt(2.0)
    normals = jnp.asarray(
        (
            (1.0, 0.0),
            (0.0, -1.0),
            (inverse_sqrt_two, inverse_sqrt_two),
        )
    )
    return system, left, right, normals


def _galilean_transform(values, velocity_shift):
    density = values[..., 0]
    momentum = values[..., 1:-1]
    energy = values[..., -1]
    shifted_momentum = momentum - density[..., None] * velocity_shift
    shifted_energy = (
        energy
        - jnp.sum(momentum * velocity_shift, axis=-1)
        + 0.5 * density * jnp.sum(velocity_shift**2)
    )
    return jnp.concatenate(
        (
            density[..., None],
            shifted_momentum,
            shifted_energy[..., None],
        ),
        axis=-1,
    )


def _rotate_two_dimensional_euler_values_to_global(values, normal):
    normal_ = jnp.asarray(normal)
    tangent = jnp.stack((-normal_[1], normal_[0]))
    vector = values[..., 1:2] * normal_ + values[..., 2:3] * tangent
    return jnp.concatenate((values[..., :1], vector, values[..., -1:]), axis=-1)


def _rotated_strong_rarefaction(normal, direction=1):
    system = phx.equations.EulerSystem(2)
    if direction == 1:
        left_primitive = jnp.asarray((1.0, -2.0, 0.7, 0.01))
        right_primitive = jnp.asarray((0.1, 10.0, -0.4, 10.0))
    else:
        left_primitive = jnp.asarray((0.1, -10.0, -0.4, 10.0))
        right_primitive = jnp.asarray((1.0, 2.0, 0.7, 0.01))
    local_left = system.primitive_to_conserved(left_primitive)
    local_right = system.primitive_to_conserved(right_primitive)
    normal_ = jnp.asarray(normal)
    return (
        system,
        local_left,
        local_right,
        _rotate_two_dimensional_euler_values_to_global(local_left, normal_),
        _rotate_two_dimensional_euler_values_to_global(local_right, normal_),
        normal_,
    )


@pytest.mark.parametrize(
    ("normal", "direction"),
    (
        ((0.6, 0.8), 1),
        ((-0.8, 0.6), -1),
    ),
)
def test_einfeldt_normal_bounds_include_rotated_roe_rarefaction_extrema_and_jit(
    normal,
    direction,
):
    system, local_left, local_right, left, right, normal_ = _rotated_strong_rarefaction(
        normal, direction
    )
    endpoint_lower, endpoint_upper = system.signal_bounds(local_left, local_right, 0)
    _, _, roe_speeds = system.eigensystem(local_left, local_right, 0)
    if direction == 1:
        assert jnp.min(roe_speeds) < endpoint_lower
    else:
        assert jnp.max(roe_speeds) > endpoint_upper

    solver = phx.discretization.EinfeldtHLLFluxPlan()
    axis_result = solver.face_flux(system, local_left, local_right, 0)
    normal_result = solver.normal_face_flux(system, left, right, normal_)
    expected_normal_flux = _rotate_two_dimensional_euler_values_to_global(
        axis_result.normal_flux, normal_
    )
    np.testing.assert_allclose(
        normal_result.normal_flux, expected_normal_flux, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(normal_result.max_speed, axis_result.max_speed, rtol=2e-12)

    def compiled_normal_flux(left_state, right_state):
        result = solver.normal_face_flux(system, left_state, right_state, normal_)
        return result.normal_flux, result.max_speed

    compiled_flux, compiled_speed = jax.jit(compiled_normal_flux)(left, right)
    np.testing.assert_allclose(compiled_flux, normal_result.normal_flux, rtol=2e-12)
    np.testing.assert_allclose(compiled_speed, normal_result.max_speed, rtol=2e-12)


@pytest.mark.parametrize(
    ("normal", "direction", "grid_velocity"),
    (
        ((0.6, 0.8), 1, 15.0),
        ((-0.8, 0.6), -1, -15.0),
    ),
)
def test_einfeldt_ale_normal_bounds_shift_union_of_endpoint_and_roe_extrema(
    normal,
    direction,
    grid_velocity,
):
    system, local_left, local_right, left, right, normal_ = _rotated_strong_rarefaction(
        normal, direction
    )
    endpoint_lower, endpoint_upper = system.signal_bounds(local_left, local_right, 0)
    _, _, roe_speeds = system.eigensystem(local_left, local_right, 0)
    lower = jnp.minimum(endpoint_lower, jnp.min(roe_speeds)) - grid_velocity
    upper = jnp.maximum(endpoint_upper, jnp.max(roe_speeds)) - grid_velocity
    lower = jnp.minimum(lower, 0.0)
    upper = jnp.maximum(upper, 0.0)
    left_flux = system.physical_flux(local_left, 0) - grid_velocity * local_left
    right_flux = system.physical_flux(local_right, 0) - grid_velocity * local_right
    expected_local_flux = (
        upper * left_flux
        - lower * right_flux
        + lower * upper * (local_right - local_left)
    ) / (upper - lower)
    expected_flux = _rotate_two_dimensional_euler_values_to_global(
        expected_local_flux, normal_
    )

    solver = phx.discretization.EinfeldtHLLFluxPlan()
    result = solver.normal_ale_face_flux(
        system,
        left[None, :],
        right[None, :],
        normal_[None, :],
        jnp.asarray((grid_velocity,)),
    )
    np.testing.assert_allclose(
        result.normal_flux[0], expected_flux, rtol=2e-12, atol=2e-12
    )
    np.testing.assert_allclose(
        result.max_speed[0],
        jnp.maximum(jnp.abs(lower), jnp.abs(upper)),
        rtol=2e-12,
    )

    stationary = solver.normal_face_flux(
        system, left[None, :], right[None, :], normal_[None, :]
    )
    zero_grid = solver.normal_ale_face_flux(
        system,
        left[None, :],
        right[None, :],
        normal_[None, :],
        jnp.zeros((1,)),
    )
    np.testing.assert_array_equal(zero_grid.normal_flux, stationary.normal_flux)
    np.testing.assert_array_equal(zero_grid.max_speed, stationary.max_speed)


def test_einfeldt_normal_fallback_update_is_admissible_for_strong_rarefaction():
    system, _, _, left, right, normal = _rotated_strong_rarefaction((0.6, 0.8))
    fallback = phx.discretization.FluxPositivityPlan().fallback_flux
    result = fallback.normal_face_flux(system, left, right, normal)
    timestep = 0.9 / result.max_speed
    left_physical_flux = system.physical_normal_flux(left, normal)
    right_physical_flux = system.physical_normal_flux(right, normal)
    updated = jnp.stack(
        (
            left - timestep * (result.normal_flux - left_physical_flux),
            right - timestep * (right_physical_flux - result.normal_flux),
        )
    )

    assert jnp.all(system.admissible(updated))


def test_einfeldt_normal_flux_rejects_scalar_systems():
    system = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="scalar-einfeldt-normal-rejection",
    )
    left = jnp.ones((1, 1))
    right = 2.0 * left
    normal = jnp.asarray(((1.0, 0.0),))
    solver = phx.discretization.EinfeldtHLLFluxPlan()

    with pytest.raises(TypeError, match="Euler-compatible"):
        solver.normal_face_flux(system, left, right, normal)
    with pytest.raises(TypeError, match="Euler-compatible"):
        solver.normal_ale_face_flux(system, left, right, normal, jnp.zeros((1,)))


@pytest.mark.parametrize(
    "geometry_kind",
    ("structured", "mapped", "triangle", "unstructured"),
)
def test_einfeldt_compile_rejects_scalar_and_accepts_euler(geometry_kind):
    velocity = jnp.asarray((0.4, -0.15))
    scalar = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: velocity[axis] * state,
        lambda left, right, axis, args: jnp.full(
            left.shape[:-1],
            jnp.abs(velocity[axis]),
            dtype=left.dtype,
        ),
        system_id=f"{geometry_kind}-scalar-einfeldt-compile-rejection",
    )

    with pytest.raises(ValueError, match="Euler-compatible"):
        _compile_two_dimensional_system(
            geometry_kind,
            scalar,
            phx.discretization.EinfeldtHLLFluxPlan(),
        )

    euler = phx.equations.EulerSystem(2)
    compiled = _compile_two_dimensional_system(
        geometry_kind,
        euler,
        phx.discretization.EinfeldtHLLFluxPlan(),
    )
    assert compiled.problem.system is euler
    assert isinstance(
        compiled.method.interface_solver,
        phx.discretization.EinfeldtHLLFluxPlan,
    )


@pytest.mark.parametrize("geometry_kind", ("structured", "mapped", "triangle"))
def test_two_material_vof_rejects_non_unstructured_discretizations(geometry_kind):
    eos = phx.equations.TwoMaterialEOSClosure(
        phx.equations.IdealGasMaterial(1.4),
        phx.equations.StiffenedGasMaterial(4.4, 2.0, 1.0),
    )
    system = phx.equations.TwoMaterialVOFSystem(2, eos=eos)

    with pytest.raises(ValueError, match="requires prepared unstructured VOF coupling"):
        _compile_two_dimensional_system(
            geometry_kind,
            system,
            phx.discretization.RusanovFluxPlan(),
        )


@pytest.mark.parametrize("solver_type", _ALE_SOLVER_TYPES)
def test_normal_ale_flux_has_exact_zero_grid_velocity_parity(solver_type):
    system, left, right, normals = _two_dimensional_euler_faces()
    solver = solver_type()

    stationary = solver.normal_face_flux(system, left, right, normals)
    ale = solver.normal_ale_face_flux(
        system, left, right, normals, jnp.zeros(left.shape[:-1])
    )

    np.testing.assert_array_equal(ale.normal_flux, stationary.normal_flux)
    np.testing.assert_array_equal(ale.max_speed, stationary.max_speed)


def test_smoothed_rusanov_ale_flux_has_exact_zero_grid_velocity_parity():
    system, left, right, normals = _two_dimensional_euler_faces()
    solver = phx.discretization.RusanovFluxPlan(smooth_epsilon=0.15)

    stationary = solver.normal_face_flux(system, left, right, normals)
    ale = solver.normal_ale_face_flux(
        system, left, right, normals, jnp.zeros(left.shape[:-1])
    )

    np.testing.assert_array_equal(ale.normal_flux, stationary.normal_flux)
    np.testing.assert_array_equal(ale.max_speed, stationary.max_speed)


@pytest.mark.parametrize("solver_type", _ALE_SOLVER_TYPES)
def test_normal_ale_constant_state_flux_and_relative_signal_bound(solver_type):
    system, state, _, normals = _two_dimensional_euler_faces()
    grid_velocity = jnp.asarray((0.25, -0.4, 0.1))
    solver = solver_type()

    result = solver.normal_ale_face_flux(system, state, state, normals, grid_velocity)
    expected_flux = (
        system.physical_normal_flux(state, normals) - grid_velocity[..., None] * state
    )
    lower, upper = system.normal_signal_bounds(state, state, normals)
    expected_speed = jnp.maximum(
        jnp.abs(lower - grid_velocity), jnp.abs(upper - grid_velocity)
    )

    np.testing.assert_allclose(result.normal_flux, expected_flux, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(result.max_speed, expected_speed, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("solver_type", _ALE_SOLVER_TYPES)
def test_normal_ale_flux_is_galilean_covariant_with_moving_grid(solver_type):
    system, left, right, normals = _two_dimensional_euler_faces()
    solver = solver_type()
    grid_velocity = jnp.asarray((0.3, -0.15, 0.5))
    frame_velocity = jnp.asarray((0.45, -0.2))
    shifted_grid_velocity = grid_velocity - jnp.sum(normals * frame_velocity, axis=-1)

    original = solver.normal_ale_face_flux(system, left, right, normals, grid_velocity)
    shifted = solver.normal_ale_face_flux(
        system,
        _galilean_transform(left, frame_velocity),
        _galilean_transform(right, frame_velocity),
        normals,
        shifted_grid_velocity,
    )

    np.testing.assert_allclose(
        shifted.normal_flux,
        _galilean_transform(original.normal_flux, frame_velocity),
        rtol=3e-11,
        atol=3e-11,
    )
    np.testing.assert_allclose(shifted.max_speed, original.max_speed, rtol=2e-12)


def test_hll_hllc_and_einfeldt_ale_fluxes_share_consistent_wave_regions():
    system, left, right, normals = _two_dimensional_euler_faces()
    grid_velocity = jnp.asarray((0.3, -0.15, 0.5))
    hll = phx.discretization.HLLFluxPlan().normal_ale_face_flux(
        system, left, right, normals, grid_velocity
    )
    einfeldt = phx.discretization.EinfeldtHLLFluxPlan().normal_ale_face_flux(
        system, left, right, normals, grid_velocity
    )
    np.testing.assert_array_equal(einfeldt.normal_flux, hll.normal_flux)
    np.testing.assert_array_equal(einfeldt.max_speed, hll.max_speed)

    fully_right_running_grid_velocity = jnp.full(left.shape[:-1], -10.0)
    expected = (
        system.physical_normal_flux(left, normals)
        - fully_right_running_grid_velocity[..., None] * left
    )
    for solver in (
        phx.discretization.HLLFluxPlan(),
        phx.discretization.HLLCFluxPlan(),
        phx.discretization.EinfeldtHLLFluxPlan(),
    ):
        result = solver.normal_ale_face_flux(
            system,
            left,
            right,
            normals,
            fully_right_running_grid_velocity,
        )
        np.testing.assert_allclose(result.normal_flux, expected, rtol=1e-12, atol=1e-12)


def test_hllc_ale_flux_exactly_resolves_a_contact_moving_with_the_grid():
    system = phx.equations.EulerSystem(2)
    contact_velocity = 0.6
    pressure = 1.25
    left = system.primitive_to_conserved(
        jnp.asarray(((1.0, contact_velocity, 0.2, pressure),))
    )
    right = system.primitive_to_conserved(
        jnp.asarray(((0.25, contact_velocity, 0.2, pressure),))
    )
    result = phx.discretization.HLLCFluxPlan().normal_ale_face_flux(
        system,
        left,
        right,
        jnp.asarray(((1.0, 0.0),)),
        jnp.asarray((contact_velocity,)),
    )

    np.testing.assert_allclose(
        result.normal_flux,
        jnp.asarray(((0.0, pressure, 0.0, pressure * contact_velocity),)),
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("solver_type", _ALE_SOLVER_TYPES)
def test_normal_ale_flux_is_jittable_and_differentiable_in_grid_velocity(
    solver_type,
):
    system, left, right, normals = _two_dimensional_euler_faces()
    solver = solver_type()

    def total_flux(grid_velocity):
        return jnp.sum(
            solver.normal_ale_face_flux(
                system, left, right, normals, grid_velocity
            ).normal_flux
        )

    grid_velocity = jnp.asarray((0.2, -0.1, 0.35))
    compiled_value = jax.jit(total_flux)(grid_velocity)
    gradient = jax.grad(total_flux)(grid_velocity)

    assert jnp.isfinite(compiled_value)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0.0)


@pytest.mark.parametrize("solver_type", _ALE_SOLVER_TYPES)
def test_normal_ale_flux_rejects_invalid_grid_velocity_inputs(solver_type):
    system, left, right, normals = _two_dimensional_euler_faces()
    solver = solver_type()

    with pytest.raises(ValueError, match="face batch shape"):
        solver.normal_ale_face_flux(system, left, right, normals, jnp.zeros((3, 1)))
    with pytest.raises(ValueError, match="scalar broadcasting"):
        solver.normal_ale_face_flux(system, left, right, normals, 0.0)
    with pytest.raises(TypeError, match="floating dtype"):
        solver.normal_ale_face_flux(
            system, left, right, normals, jnp.zeros((3,), dtype=jnp.int32)
        )
    with pytest.raises(TypeError, match="floating dtype"):
        solver.normal_ale_face_flux(
            system, left, right, normals, jnp.zeros((3,), dtype=jnp.complex64)
        )
    with pytest.raises((ValueError, eqx.EquinoxRuntimeError), match="must be finite"):
        solver.normal_ale_face_flux(
            system, left, right, normals, jnp.asarray((0.0, jnp.nan, 0.0))
        )


@pytest.mark.parametrize("solver_type", _ALE_SOLVER_TYPES)
def test_normal_ale_flux_accepts_scalar_velocity_only_for_unbatched_face(
    solver_type,
):
    system, left, right, normals = _two_dimensional_euler_faces()
    result = solver_type().normal_ale_face_flux(
        system, left[0], right[0], normals[0], jnp.asarray(0.2)
    )

    assert result.normal_flux.shape == left.shape[-1:]
    assert result.max_speed.shape == ()


@pytest.mark.parametrize("method", ["weno_z", "teno", "mp5"])
def test_high_resolution_reconstruction_retains_fifth_order_smooth_accuracy(method):
    errors = []
    for cells in (32, 64, 128):
        edges = jnp.linspace(0.0, 1.0, cells + 1)
        values = _sine_cell_averages(edges)
        reconstruction = phx.discretization.HighResolutionReconstructionPlan(method)
        depth = reconstruction.radius
        ghosted = jnp.concatenate((values[-depth:], values, values[:depth]))
        left_ghosted, _ = reconstruction.reconstruct(ghosted)
        left = left_ghosted[depth : depth + cells]
        exact = jnp.sin(2.0 * jnp.pi * edges[1:])
        errors.append(float(jnp.sqrt(jnp.mean((left - exact) ** 2))))
    rate = np.log2(errors[-2] / errors[-1])
    assert rate > 4.5


def test_multidimensional_euler_roundtrip_and_directional_flux_shapes():
    system = phx.equations.EulerSystem(2)
    primitive = jnp.asarray([[1.0, 0.3, -0.1, 1.2], [0.7, -0.2, 0.4, 0.8]])
    state = system.primitive_to_conserved(primitive)

    np.testing.assert_allclose(
        system.conserved_to_primitive(state), primitive, rtol=1e-12
    )
    assert system.physical_flux(state, 0).shape == state.shape
    assert system.physical_flux(state, 1).shape == state.shape
    reflected = system.reflect_state(state, 1)
    np.testing.assert_allclose(reflected[..., 2], -state[..., 2])
    normal = jnp.asarray((1.0, 1.0)) / jnp.sqrt(2.0)
    reflected_normal = system.reflect_normal_state(state, normal)
    momentum = state[..., 1:3]
    reflected_momentum = reflected_normal[..., 1:3]
    np.testing.assert_allclose(
        jnp.sum(reflected_momentum * normal, axis=-1),
        -jnp.sum(momentum * normal, axis=-1),
        rtol=1e-12,
        atol=1e-12,
    )
    tangent = jnp.asarray((-normal[1], normal[0]))
    np.testing.assert_allclose(
        jnp.sum(reflected_momentum * tangent, axis=-1),
        jnp.sum(momentum * tangent, axis=-1),
        rtol=1e-12,
        atol=1e-12,
    )


def test_euler_roe_eigensystem_roundtrips_state_jump_in_two_dimensions():
    system = phx.equations.EulerSystem(2)
    left = system.primitive_to_conserved(jnp.asarray([[1.0, 0.3, 0.1, 1.0]]))
    right = system.primitive_to_conserved(jnp.asarray([[0.8, -0.1, 0.2, 0.7]]))
    left_matrix, right_matrix, eigenvalues = system.eigensystem(left, right, 0)
    jump = right - left
    recovered = oe.contract(
        "...ij,...j->...i",
        right_matrix,
        oe.contract("...ij,...j->...i", left_matrix, jump),
    )

    np.testing.assert_allclose(recovered, jump, rtol=2e-11, atol=2e-11)
    assert eigenvalues.shape == jump.shape


def test_euler_normal_eigensystem_roundtrips_oblique_state_jump():
    system = phx.equations.EulerSystem(2)
    left = system.primitive_to_conserved(jnp.asarray([[1.0, 0.3, 0.1, 1.0]]))
    right = system.primitive_to_conserved(jnp.asarray([[0.8, -0.1, 0.2, 0.7]]))
    normal = jnp.asarray(((0.6, 0.8),))
    left_matrix, right_matrix, eigenvalues = system.normal_eigensystem(
        left, right, normal
    )
    jump = right - left
    recovered = oe.contract(
        "...ij,...j->...i",
        right_matrix,
        oe.contract("...ij,...j->...i", left_matrix, jump),
    )
    np.testing.assert_allclose(recovered, jump, rtol=3e-11, atol=3e-11)
    assert eigenvalues.shape == jump.shape


def test_entropy_flux_is_consistent_and_dissipative_variant_has_nonpositive_pairing():
    system = phx.equations.EulerSystem()
    left = system.primitive_to_conserved(jnp.asarray([[1.0, 0.4, 1.0], [0.7, -0.2, 0.8]]))
    right = system.primitive_to_conserved(jnp.asarray([[0.9, 0.1, 0.9], [1.1, 0.3, 1.2]]))
    central = phx.discretization.EntropyConservativeEulerFluxPlan()
    stable = phx.discretization.EntropyStableEulerFluxPlan()
    entropy_pair = phx.equations.ideal_gas_euler_entropy_pair(system)

    equal = central.face_flux(system, left, left, 0)
    np.testing.assert_allclose(
        equal.normal_flux, system.physical_flux(left, 0), rtol=1e-12
    )
    assert jnp.all(stable.entropy_dissipation(system, left, right) <= 2e-13)
    central_flux = central.face_flux(system, left, right, 0).normal_flux
    stable_flux = stable.face_flux(system, left, right, 0).normal_flux
    np.testing.assert_allclose(
        entropy_pair.interface_entropy_residual(
            left,
            right,
            central_flux,
            0,
        ),
        0.0,
        atol=1e-10,
    )
    assert jnp.all(
        entropy_pair.interface_entropy_residual(
            left,
            right,
            stable_flux,
            0,
        )
        <= 2e-13
    )


def test_characteristic_weno_euler_step_preserves_positive_sod_state_and_mass():
    cells = 120
    grid = _cell_grid((cells,))
    system = phx.equations.EulerSystem()
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    reconstruction = phx.discretization.HighResolutionReconstructionPlan("weno_z")
    characteristic = phx.discretization.CharacteristicReconstructionPlan(
        reconstruction,
        phx.discretization.CharacteristicSystem(
            lambda left, right, args: system.eigensystem(left, right, 0, args),
            system_id=system.system_id,
        ),
    )
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,))
    method = phx.discretization.FiniteVolumeMethodPlan(
        characteristic,
        phx.discretization.HLLCFluxPlan(),
        positivity=phx.discretization.ConvexStateLimiterPlan(),
    )
    problem = phx.equations.ConservationProblemIR("sod", "state", system, boundaries)
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    x = grid.structured_axes[0].interval_centers
    primitive = jnp.stack(
        (
            jnp.where(x < 0.5, 1.0, 0.125),
            jnp.zeros_like(x),
            jnp.where(x < 0.5, 1.0, 0.1),
        ),
        axis=-1,
    )
    state = system.primitive_to_conserved(primitive)
    initial_mass = jnp.sum(discretization.cell_volumes * state[:, 0])
    stepper = phx.solver.UnsplitFiniteVolumeSSPRK3Plan(compiled.dynamics)
    time = jnp.asarray(0.0)
    for _ in range(10):
        dt = compiled.stable_step(state, cfl=0.25)
        result = stepper.advance(time, state, dt)
        state, time = result.state, result.time

    assert jnp.all(system.admissible(state))
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes * state[:, 0]),
        initial_mass,
        atol=2e-10,
    )


def test_multispecies_and_mhd_fluxes_preserve_declared_components():
    multispecies = phx.equations.MultispeciesEulerSystem((1.4, 1.67))
    primitive = jnp.asarray([[0.6, 0.4, 0.2, 1.0]])
    multispecies_state = multispecies.primitive_to_conserved(primitive)
    assert (
        multispecies.physical_flux(multispecies_state, 0).shape
        == multispecies_state.shape
    )
    assert jnp.all(multispecies.admissible(multispecies_state))

    mhd = phx.equations.IdealMHDSystem()
    mhd_primitive = jnp.asarray([[1.0, 0.1, 0.0, 0.0, 1.0, 0.75, 0.1, 0.0]])
    mhd_state = mhd.primitive_to_conserved(mhd_primitive)
    flux = mhd.physical_flux(mhd_state, 0)
    np.testing.assert_allclose(flux[..., 5], 0.0, atol=0.0)
    assert jnp.all(mhd.admissible(mhd_state))


def test_unsplit_two_dimensional_scalar_residual_preserves_periodic_mass():
    grid = _cell_grid((18, 14), periodic=(True, True))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    velocity = (0.7, -0.2)
    system = phx.equations.ScalarConservationSystem(
        2,
        lambda state, axis, args: velocity[axis] * state,
        lambda left, right, axis, args: jnp.full(left.shape[:-1], abs(velocity[axis])),
        system_id="two-dimensional-transport",
    )
    boundaries = phx.discretization.FiniteVolumeBoundarySet.periodic(("x", "y"))
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.MUSCLReconstruction(),
        phx.discretization.RusanovFluxPlan(),
    )
    problem = phx.equations.ConservationProblemIR(
        "two-dimensional-transport", "state", system, boundaries
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    x = grid.structured_axes[0].interval_centers
    y = grid.structured_axes[1].interval_centers
    state = (
        jnp.sin(2.0 * jnp.pi * x)[:, None] + 0.3 * jnp.cos(2.0 * jnp.pi * y)[None, :]
    )[..., None]

    residual = compiled(jnp.asarray(0.0), state)
    np.testing.assert_allclose(
        jnp.sum(discretization.cell_volumes[..., None] * residual), 0.0, atol=2e-11
    )


def test_nonuniform_weno_prepares_ghost_geometry_for_bounded_faces():
    edges = jnp.asarray([0.0, 0.08, 0.2, 0.38, 0.62, 0.82, 1.0])
    widths = edges[1:] - edges[:-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    axis = phx.discretization.AxisDiscretization(
        nodes=centers,
        quad_weights=widths,
        basis="uniform",
        domain=phx.discretization.AxisDomain.interval(0.0, 1.0),
        primary_entity="interval",
        lower_endpoint_included=False,
        upper_endpoint_included=False,
    )
    grid = phx.discretization.PreparedTensorGrid((axis,), axis_names=("x",))
    discretization = phx.discretization.FiniteVolumePlan(grid).prepare()
    system = phx.equations.ScalarConservationSystem(
        1,
        lambda state, axis, args: state,
        lambda left, right, axis, args: jnp.ones(left.shape[:-1]),
        system_id="bounded-nonuniform-advection",
    )
    pair = phx.discretization.FiniteVolumeBoundaryPair(
        phx.discretization.ExtrapolationBoundary(),
        phx.discretization.ExtrapolationBoundary(),
    )
    problem = phx.equations.ConservationProblemIR(
        "bounded-nonuniform-advection",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet(("x",), (pair,)),
    )
    method = phx.discretization.FiniteVolumeMethodPlan(
        phx.discretization.NonuniformWENOReconstructionPlan(edges),
        phx.discretization.RusanovFluxPlan(),
    )
    compiled = phx.equations.compile_conservation_problem(problem, discretization, method)
    state = jnp.ones(discretization.state_shape)
    fluxes, _ = compiled.face_fluxes(jnp.asarray(0.0), state)

    assert fluxes[0].shape == (7, 1)
    np.testing.assert_allclose(compiled(jnp.asarray(0.0), state), 0.0, atol=1e-12)
