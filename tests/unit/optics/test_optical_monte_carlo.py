#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

from phydrax._physical import RelativityScaleContract
from phydrax._strict import StrictModule
from phydrax.optics.geometric._nonsequential import (
    NonSequentialSurfaceKind,
    NonSequentialSurfaceTable,
)
from phydrax.optics.transport._optical_monte_carlo import (
    ExplicitPhotonSource,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalScatteringSample,
    OpticalTransportStatus,
    OpticalVarianceReduction,
    prepare_optical_monte_carlo,
    PreparedOpticalMonteCarlo,
    rotate_jones_to_scattering_frame,
    simulate_optical_photons,
    TissueOpticalMedium,
)


_SI = RelativityScaleContract.si()
_WAVELENGTH = 500e-9


def _plane(z: float, negative_medium: int, positive_medium: int, indices: Any) -> Any:
    vertices = jnp.asarray(
        [[-100.0, -100.0, z], [100.0, -100.0, z], [100.0, 100.0, z], [-100.0, 100.0, z]]
    )
    triangles = jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32)
    return NonSequentialSurfaceTable(
        vertices,
        triangles,
        jnp.asarray([negative_medium, negative_medium]),
        jnp.asarray([positive_medium, positive_medium]),
        jnp.asarray(indices),
        surface_ids=jnp.asarray([0, 0]),
    )


def _pencil(count: int, z: float, direction: Any = (0.0, 0.0, 1.0)) -> tuple[Any, Any]:
    return (
        np.broadcast_to(np.asarray([0.0, 0.0, z]), (count, 3)),
        np.broadcast_to(np.asarray(direction, dtype=np.float64), (count, 3)),
    )


def _prepared(
    surfaces: Any, medium: Any, maximum_interactions: int, **kwargs: Any
) -> PreparedOpticalMonteCarlo:
    return prepare_optical_monte_carlo(
        OpticalMonteCarloPlan(
            surfaces,
            medium,
            relativity=_SI,
            maximum_interactions=maximum_interactions,
            **kwargs,
        )
    )


def test_photon_identities_follow_launch_order_with_carry_and_reserved_refusal() -> None:
    origins, directions = _pencil(4, 0.0)
    state = launch_optical_photons(
        origins, directions, 0, wavelengths=_WAVELENGTH, first_identity=(0, 2**32 - 2)
    )
    np.testing.assert_array_equal(state.id_hi, np.asarray([0, 0, 1, 1], dtype=np.uint32))
    np.testing.assert_array_equal(
        state.id_lo, np.asarray([2**32 - 2, 2**32 - 1, 0, 1], dtype=np.uint32)
    )
    assert state.photon_count == 4
    with pytest.raises(ValueError, match="reserved"):
        launch_optical_photons(
            origins,
            directions,
            0,
            wavelengths=_WAVELENGTH,
            first_identity=(2**32 - 1, 2**32 - 4),
        )
    with pytest.raises(ValueError, match="perpendicular"):
        launch_optical_photons(
            origins,
            directions,
            0,
            wavelengths=_WAVELENGTH,
            transverse_axes=np.broadcast_to(np.asarray([0.0, 0.6, 0.8]), (4, 3)),
        )


def test_results_are_invariant_to_photon_order_batch_size_and_photon_count() -> None:
    medium = TissueOpticalMedium(
        jnp.asarray([0.2, 0.0]),
        jnp.asarray([0.8, 0.0]),
        jnp.asarray([0.6, 0.0]),
        jnp.asarray([1.4, 1.0]),
    )
    surfaces = _plane(0.5, 0, 1, (1.4, 1.0))
    key = jr.key(2024)
    count = 64
    origins, directions = _pencil(count, 0.0)
    state = launch_optical_photons(origins, directions, 0, wavelengths=_WAVELENGTH)
    whole = simulate_optical_photons(
        _prepared(surfaces, medium, 6, photon_batch_size=64),
        ExplicitPhotonSource(state),
        key,
    )
    rebatched = simulate_optical_photons(
        _prepared(surfaces, medium, 6, photon_batch_size=7),
        ExplicitPhotonSource(state),
        key,
    )
    permutation = np.random.default_rng(3).permutation(count)
    permuted = simulate_optical_photons(
        _prepared(surfaces, medium, 6, photon_batch_size=16),
        ExplicitPhotonSource(
            jax.tree_util.tree_map(lambda value: value[permutation], state)
        ),
        key,
    )
    subset = simulate_optical_photons(
        _prepared(surfaces, medium, 6, photon_batch_size=16),
        ExplicitPhotonSource(
            launch_optical_photons(
                origins[5:21],
                directions[5:21],
                0,
                wavelengths=_WAVELENGTH,
                first_identity=(0, 5),
            )
        ),
        key,
    )
    assert bool(jnp.any(whole.interaction_counts > 2))

    def per_photon(result: Any, rows: Any) -> tuple[Any, ...]:
        return tuple(
            np.asarray(values)[rows]
            for values in (
                result.terminal_state.positions,
                result.terminal_state.directions,
                result.terminal_state.jones_vectors,
                result.terminal_state.weights,
                result.terminal_state.times,
                result.per_photon_tallies.absorption,
                result.per_photon_tallies.escape,
                result.interaction_counts,
                result.status,
            )
        )

    everything = np.arange(count)
    for reference, candidate in (
        (per_photon(whole, everything), per_photon(rebatched, everything)),
        (per_photon(whole, everything), per_photon(permuted, np.argsort(permutation))),
        (per_photon(whole, np.arange(5, 21)), per_photon(subset, np.arange(16))),
    ):
        # Identity-keyed draws make histories identical; float fields may differ
        # only by the rounding of differently vectorized arithmetic.
        for expected, actual in zip(reference, candidate, strict=True):
            if np.issubdtype(expected.dtype, np.integer):
                np.testing.assert_array_equal(expected, actual)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(whole.tallies.escape, rebatched.tallies.escape, rtol=1e-12)


def test_jones_vector_stays_unit_norm_and_transverse_through_scattering() -> None:
    count = 512
    medium = TissueOpticalMedium(
        jnp.asarray([0.0]), jnp.asarray([2.0]), jnp.asarray([0.8]), jnp.asarray([1.0])
    )
    surfaces = _plane(-50.0, 0, 0, (1.0,))
    rng = np.random.default_rng(11)
    directions = rng.normal(size=(count, 3))
    directions /= np.linalg.norm(directions, axis=-1)[:, None]
    origins = np.zeros((count, 3))
    circular = np.broadcast_to(np.asarray([1.0, 1.0j]) / np.sqrt(2.0), (count, 2)).astype(
        np.complex128
    )
    state = launch_optical_photons(
        origins, directions, 0, wavelengths=_WAVELENGTH, jones_vectors=circular
    )
    single = simulate_optical_photons(
        _prepared(surfaces, medium, 1), ExplicitPhotonSource(state), jr.key(5)
    )
    multiple = simulate_optical_photons(
        _prepared(surfaces, medium, 7), ExplicitPhotonSource(state), jr.key(5)
    )
    for result in (single, multiple):
        assert bool(jnp.all(result.terminal_live))
        assert bool(
            jnp.all(
                result.status
                == int(OpticalTransportStatus.INTERACTION_CAPACITY_EXHAUSTED)
            )
        )
        terminal = result.terminal_state
        axes = np.asarray(terminal.transverse_axes[:, 0])
        exits = np.asarray(terminal.directions[:, 0])
        jones = np.asarray(terminal.jones_vectors[:, 0])
        np.testing.assert_allclose(np.sum(np.abs(jones) ** 2, axis=-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(axes, axis=-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(np.sum(axes * exits, axis=-1), 0.0, atol=1e-12)
        assert float(result.maximum_polarization_defect) < 1e-12
    # After one scattering the carried axis is the scattering-plane normal s,
    # perpendicular to both the launch and the scattered direction, and the
    # scalar medium keeps the field's components on (s, p): they equal the
    # projections of the launch field vector on s and on direction × s.
    axes = np.asarray(single.terminal_state.transverse_axes[:, 0])
    np.testing.assert_allclose(np.sum(axes * directions, axis=-1), 0.0, atol=1e-12)
    launch_axes = np.asarray(state.transverse_axes)
    field = circular[:, :1] * launch_axes + circular[:, 1:] * np.cross(
        directions, launch_axes
    )
    incident_p = np.cross(directions, axes)
    np.testing.assert_allclose(
        np.asarray(single.terminal_state.jones_vectors[:, 0]),
        np.stack(
            (np.sum(field * axes, axis=-1), np.sum(field * incident_p, axis=-1)), axis=-1
        ),
        atol=1e-12,
    )


def test_roulette_preserves_expected_weight_and_closes_the_ledger() -> None:
    count = 16_384
    medium = TissueOpticalMedium(
        jnp.asarray([0.9]), jnp.asarray([0.1]), jnp.asarray([0.0]), jnp.asarray([1.0])
    )
    origins, directions = _pencil(count, 0.0)
    state = launch_optical_photons(origins, directions, 0, wavelengths=_WAVELENGTH)
    result = simulate_optical_photons(
        _prepared(
            _plane(-10.0, 0, 0, (1.0,)),
            medium,
            1,
            variance_reduction=OpticalVarianceReduction(
                roulette_threshold=0.2, roulette_survival_probability=0.25
            ),
        ),
        ExplicitPhotonSource(state),
        jr.key(77),
    )
    survivors = np.asarray(result.terminal_live[:, 0])
    survival = survivors.mean()
    assert abs(survival - 0.25) < 4.0 * np.sqrt(0.25 * 0.75 / count)
    np.testing.assert_allclose(
        np.asarray(result.terminal_state.weights[:, 0])[survivors], 0.4, atol=1e-12
    )
    assert abs(float(result.tallies.roulette)) < 4.0 * float(
        result.standard_errors.roulette
    )
    np.testing.assert_allclose(result.tallies.absorption, [0.9], atol=1e-12)
    assert float(result.maximum_absolute_ledger_residual) < 1e-12


def test_time_of_flight_and_wavelength_are_carried_across_a_dielectric() -> None:
    thickness = 0.75
    index = 1.33
    medium = TissueOpticalMedium(
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([0.0, 0.0]),
        jnp.asarray([index, 1.0]),
    )
    origins, directions = _pencil(3, 0.0)
    state = launch_optical_photons(
        origins,
        directions,
        0,
        wavelengths=np.asarray([400e-9, 500e-9, 600e-9]),
        times=np.asarray([0.0, 1.0, 2.0]),
    )
    result = simulate_optical_photons(
        _prepared(
            _plane(thickness, 0, 1, (index, 1.0)),
            medium,
            1,
            branch_capacity=2,
            variance_reduction=OpticalVarianceReduction(
                interface_branching="expected-split"
            ),
        ),
        ExplicitPhotonSource(state),
        jr.key(3),
    )
    terminal = result.terminal_state
    assert bool(jnp.all(result.terminal_live))
    expected_time = index * thickness / float(_SI.speed_of_light)
    np.testing.assert_allclose(
        terminal.times,
        np.broadcast_to(np.asarray([0.0, 1.0, 2.0])[:, None] + expected_time, (3, 2)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        terminal.wavelengths,
        np.broadcast_to(np.asarray([400e-9, 500e-9, 600e-9])[:, None], (3, 2)),
    )
    fresnel = ((index - 1.0) / (index + 1.0)) ** 2
    np.testing.assert_allclose(
        terminal.weights, np.asarray([[fresnel, 1.0 - fresnel]] * 3), atol=1e-12
    )
    np.testing.assert_array_equal(terminal.medium_indices, [[0, 1]] * 3)
    np.testing.assert_array_equal(terminal.id_lo, [[0, 0], [1, 1], [2, 2]])
    assert float(result.maximum_polarization_defect) < 1e-12


class _SpectralAbsorber(StrictModule):
    """Protocol medium whose absorption scales with the squared wavelength."""

    reference_wavelength: float = eqx.field(static=True)
    reference_absorption: float = eqx.field(static=True)
    medium_count: int = eqx.field(static=True)
    medium_id: str = eqx.field(static=True)

    def __init__(self, reference_wavelength: float, reference_absorption: float) -> None:
        self.reference_wavelength = reference_wavelength
        self.reference_absorption = reference_absorption
        self.medium_count = 2
        self.medium_id = "spectral-absorber"

    def refractive_index(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        del medium_indices
        return jnp.ones(wavelengths.shape, dtype=jnp.float64)

    def extinction(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        scale = (wavelengths / self.reference_wavelength) ** 2
        return jnp.where(medium_indices == 0, self.reference_absorption * scale, 0.0)

    def albedo(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        del medium_indices
        return jnp.zeros(wavelengths.shape, dtype=jnp.float64)

    def spectral_support(self, medium_indices: Array, wavelengths: Array, /) -> Array:
        del medium_indices
        return jnp.ones(wavelengths.shape, dtype=jnp.bool_)

    def scatter(
        self,
        medium_indices: Array,
        wavelengths: Array,
        jones_vectors: Array,
        keys: Array,
        /,
    ) -> OpticalScatteringSample:
        del medium_indices
        uniforms = jax.vmap(lambda key: jr.uniform(key, (2,), dtype=jnp.float64))(keys)
        azimuths = 2.0 * jnp.pi * uniforms[:, 1]
        return OpticalScatteringSample(
            2.0 * uniforms[:, 0] - 1.0,
            azimuths,
            rotate_jones_to_scattering_frame(jones_vectors, azimuths),
            wavelengths,
            jnp.zeros(wavelengths.shape, dtype=jnp.float64),
        )


class _QuarterResponse(StrictModule):
    detector_response_id: str = eqx.field(static=True)

    def __init__(self) -> None:
        self.detector_response_id = "quarter"

    def respond(
        self,
        detector_indices: Array,
        wavelengths: Array,
        times: Array,
        jones_vectors: Array,
        incidence_cosines: Array,
        /,
    ) -> Array:
        del detector_indices, times, jones_vectors, incidence_cosines
        return jnp.full(wavelengths.shape, 0.25, dtype=jnp.float64)


def test_medium_and_detector_protocols_receive_wavelength_and_close_the_ledger() -> None:
    count = 8192
    thickness = 0.5
    absorption = 0.8
    medium = _SpectralAbsorber(_WAVELENGTH, absorption)
    prepared = _prepared(_plane(thickness, 0, 1, (1.0, 1.0)), medium, 2)
    origins, directions = _pencil(count, 0.0)
    for factor in (1.0, 2.0):
        state = launch_optical_photons(
            origins, directions, 0, wavelengths=factor * _WAVELENGTH
        )
        result = simulate_optical_photons(
            prepared, ExplicitPhotonSource(state), jr.key(9)
        )
        expected = np.exp(-absorption * factor**2 * thickness)
        assert abs(float(result.tallies.escape) - expected) < max(
            4.0 * float(result.standard_errors.escape), 0.01
        )
        assert float(result.maximum_absolute_ledger_residual) < 1e-12
    vertices = jnp.asarray(
        [[-2.0, -2.0, 1.0], [2.0, -2.0, 1.0], [2.0, 2.0, 1.0], [-2.0, 2.0, 1.0]]
    )
    detector_surface = NonSequentialSurfaceTable(
        vertices,
        jnp.asarray([[0, 1, 2], [0, 2, 3]], dtype=jnp.int32),
        jnp.asarray([0, 0]),
        jnp.asarray([0, 0]),
        jnp.asarray([1.0]),
        surface_ids=jnp.asarray([0, 0]),
        surface_kinds=jnp.asarray([int(NonSequentialSurfaceKind.DETECTOR)] * 2),
        detector_indices=jnp.asarray([0, 0]),
        detector_acceptance_cosines=jnp.asarray([0.5, 0.5]),
    )
    clear = TissueOpticalMedium(
        jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)), jnp.ones((1,))
    )
    origins, directions = _pencil(2, 0.0)
    state = launch_optical_photons(origins, directions, 0, wavelengths=_WAVELENGTH)
    detected = simulate_optical_photons(
        _prepared(detector_surface, clear, 1, detector_response=_QuarterResponse()),
        ExplicitPhotonSource(state),
        jr.key(1),
    )
    np.testing.assert_allclose(detected.per_photon_tallies.detector, [[0.25], [0.25]])
    np.testing.assert_allclose(detected.per_photon_tallies.absorption, [[0.75], [0.75]])
    np.testing.assert_allclose(
        detected.per_photon_tallies.ledger_residual, 0.0, atol=1e-12
    )
    assert bool(detected.all_successful)
