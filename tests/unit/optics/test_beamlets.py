#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization import TensorGridPlan, UniformAxisSpec
from phydrax.geometry import RigidFrame
from phydrax.optics.beamlets._core import (
    beamlet_curvature,
    beamlet_lagrange_invariant,
    BeamletFrame,
    BeamletStatus,
    deterministic_beamlet_frame,
    gaussian_beamlets_at_waist,
    GaussianBeamletState,
    GaussianWaistSpecification,
    transport_beamlet_frame,
    transport_gaussian_beamlets,
)
from phydrax.optics.beamlets._qualification import (
    NineRayTraceSamples,
    qualify_nine_ray_differential_map,
)
from phydrax.optics.beamlets._reconstruction import (
    BeamletReconstructionPlan,
    reconstruct_gaussian_beamlets,
)
from phydrax.optics.geometric._interface import OpticalRayState
from phydrax.optics.geometric._paraxial import DifferentialRayMap
from phydrax.optics.wave._fields import PlaneFieldSpace


def _ray(*, z=0.0, optical_path=0.0):
    return OpticalRayState(
        jnp.asarray((0.0, 0.0, z)),
        jnp.asarray((0.0, 0.0, 1.0)),
        1.0,
        geometric_path_lengths=z,
        optical_path_lengths=optical_path,
    )


def _frame(*, z=0.0):
    return BeamletFrame(RigidFrame(jnp.eye(3), jnp.asarray((0.0, 0.0, z))))


def _space(count=41, extent=2.0, *, z=0.0):
    grid = TensorGridPlan(
        (UniformAxisSpec(count), UniformAxisSpec(count)),
        axis_names=("u", "v"),
    ).prepare(jnp.asarray(((-extent, -extent), (extent, extent))))
    frame = RigidFrame(jnp.eye(3), jnp.asarray((0.0, 0.0, z)))
    return PlaneFieldSpace(grid, frame, "finite-window")


def _map(jacobian, input_frame, output_frame, *, source="system"):
    return DifferentialRayMap(
        jnp.zeros((4,)),
        jnp.zeros((4,)),
        jnp.asarray(jacobian),
        jnp.asarray(1.0),
        jnp.asarray(True),
        jnp.asarray(True),
        jnp.asarray(0, dtype=jnp.int32),
        input_frame_id=input_frame.frame_id,
        output_frame_id=output_frame.frame_id,
        source_prepared_id=source,
        coordinate_convention="(u,v,nθu,nθv)",
    )


def test_moving_beamlet_frames_are_deterministic_and_right_handed():
    first = deterministic_beamlet_frame((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    repeated = deterministic_beamlet_frame((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    transported = transport_beamlet_frame(
        first,
        (0.0, 0.0, 1.0),
        (0.1, -0.2, 1.0),
    )

    np.testing.assert_array_equal(first.frame.rotation, repeated.frame.rotation)
    assert first.frame_id == repeated.frame_id
    np.testing.assert_allclose(
        transported.frame.rotation.T @ transported.frame.rotation,
        jnp.eye(3),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        np.linalg.det(transported.frame.rotation),
        1.0,
        atol=2e-6,
    )


def test_fundamental_beamlet_reconstructs_analytic_gaussian():
    state = gaussian_beamlets_at_waist(
        _ray(),
        GaussianWaistSpecification((0.7, 0.7)),
        _frame(),
        2.0 * jnp.pi,
        3.0,
        topology_id="branch",
        source_prepared_id="system",
    )
    space = _space()
    result = reconstruct_gaussian_beamlets(
        BeamletReconstructionPlan(space, 0.0, tile_size=127).prepare(),
        state,
    )
    coordinates = space.transverse_coordinates
    expected = jnp.exp(-jnp.sum(coordinates * coordinates, axis=-1) / 0.7**2)

    np.testing.assert_allclose(result.field.values, expected, rtol=2e-5, atol=2e-6)
    assert int(result.evidence.tile_count) > 1
    assert bool(result.successful)


def test_astigmatic_waist_rotation_rotates_complex_curvature():
    angle = jnp.pi / 4.0
    wavenumber = 5.0
    state = gaussian_beamlets_at_waist(
        _ray(),
        GaussianWaistSpecification((0.5, 1.0), angle),
        _frame(),
        wavenumber,
        2.0,
        topology_id="branch",
        source_prepared_id="system",
    )
    curvature = beamlet_curvature(state)
    rotation = np.asarray(
        (
            (np.cos(np.pi / 4.0), -np.sin(np.pi / 4.0)),
            (np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)),
        )
    )
    expected = (2j / wavenumber) * rotation @ np.diag((4.0, 1.0)) @ rotation.T

    np.testing.assert_allclose(curvature.curvature, expected, rtol=2e-6, atol=2e-7)
    assert bool(curvature.successful)


def test_symplectic_transport_preserves_lagrange_invariant():
    input_frame = _frame()
    distance = 0.75
    output_frame = _frame(z=distance)
    state = gaussian_beamlets_at_waist(
        _ray(),
        GaussianWaistSpecification((0.6, 0.9), 0.2),
        input_frame,
        7.0,
        2.0,
        topology_id="branch",
        source_prepared_id="system",
    )
    jacobian = jnp.asarray(
        (
            (1.0, 0.0, distance, 0.0),
            (0.0, 1.0, 0.0, distance),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    transported = transport_gaussian_beamlets(
        state,
        _map(jacobian, input_frame, output_frame),
        _ray(z=distance, optical_path=distance),
        output_frame,
    )

    np.testing.assert_allclose(
        beamlet_lagrange_invariant(transported.state.lagrangian_state),
        state.reference_invariant,
        rtol=2e-6,
        atol=2e-7,
    )
    assert float(transported.evidence.symplectic_error) < 1e-6
    assert bool(transported.successful)


def test_free_space_transport_reconstructs_complex_gaussian_field():
    distance = 0.8
    waist_radius = 0.65
    medium_wavenumber = 6.0
    input_frame = _frame()
    output_frame = _frame(z=distance)
    waist = gaussian_beamlets_at_waist(
        _ray(),
        GaussianWaistSpecification((waist_radius, waist_radius)),
        input_frame,
        medium_wavenumber,
        3.0,
        topology_id="branch",
        source_prepared_id="system",
    )
    free_space_map = jnp.asarray(
        (
            (1.0, 0.0, distance, 0.0),
            (0.0, 1.0, 0.0, distance),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    transported = transport_gaussian_beamlets(
        waist,
        _map(free_space_map, input_frame, output_frame),
        _ray(z=distance, optical_path=distance),
        output_frame,
    )
    space = _space(count=61, extent=1.8, z=distance)
    result = reconstruct_gaussian_beamlets(
        BeamletReconstructionPlan(space, distance, tile_size=256).prepare(),
        transported.state,
    )

    rayleigh_range = 0.5 * medium_wavenumber * waist_radius**2
    reduced_distance = distance / rayleigh_range
    radius_squared = jnp.sum(space.transverse_coordinates**2, axis=-1)
    expected = (
        jnp.exp(1j * medium_wavenumber * distance)
        / (1.0 + 1j * reduced_distance)
        * jnp.exp(
            1j
            * (reduced_distance + 1j)
            * radius_squared
            / (waist_radius**2 * (1.0 + reduced_distance**2))
        )
    )

    np.testing.assert_allclose(result.field.values, expected, rtol=2e-5, atol=2e-6)
    assert bool(transported.successful)
    assert bool(result.successful)


def test_transport_exposes_topology_and_caustic_failures():
    frame = _frame()
    state = gaussian_beamlets_at_waist(
        _ray(),
        GaussianWaistSpecification((1.0, 1.0)),
        frame,
        4.0,
        2.0,
        topology_id="branch",
        source_prepared_id="system",
    )
    identity_map = _map(jnp.eye(4), frame, frame)
    mismatch = transport_gaussian_beamlets(
        state,
        identity_map,
        _ray(),
        frame,
        output_topology_id="other-branch",
    )
    singular_lagrangian = jnp.concatenate(
        (jnp.zeros((2, 2), dtype=complex), jnp.eye(2, dtype=complex)), axis=0
    )
    caustic_state = GaussianBeamletState(
        _ray(),
        frame,
        singular_lagrangian,
        1.0,
        4.0,
        2.0,
        topology_id="branch",
        source_prepared_id="system",
    )
    caustic = transport_gaussian_beamlets(
        caustic_state,
        identity_map,
        _ray(),
        frame,
    )

    assert int(mismatch.evidence.status) == int(BeamletStatus.TOPOLOGY_MISMATCH)
    assert int(caustic.evidence.status) == int(BeamletStatus.CAUSTIC)
    assert not bool(mismatch.successful)
    assert not bool(caustic.successful)


def _nine_ray_samples(step):
    inputs = jnp.zeros((9, 4))
    for axis in range(4):
        inputs = inputs.at[1 + 2 * axis, axis].set(step)
        inputs = inputs.at[2 + 2 * axis, axis].set(-step)
    outputs = inputs + 0.4 * inputs**3
    return NineRayTraceSamples(inputs, outputs, jnp.full((4,), step))


def test_nine_ray_qualification_reports_centered_second_order_convergence():
    qualification = qualify_nine_ray_differential_map(
        _map(jnp.eye(4), _frame(), _frame()),
        _nine_ray_samples(0.1),
        refined_samples=_nine_ray_samples(0.05),
    )

    np.testing.assert_allclose(qualification.observed_order, 2.0, rtol=2e-4)
    assert float(qualification.relative_jacobian_error) > 0.0
    assert bool(qualification.valid)
