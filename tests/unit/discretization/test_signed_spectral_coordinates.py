from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.discretization.spectral._signed_coordinates import (
    SignedHermitianSpectralCoordinates,
)


def _signed_coordinates(
    *, coefficient_dtype: object = jnp.complex128
) -> SignedHermitianSpectralCoordinates:
    return SignedHermitianSpectralCoordinates(
        (6,),
        np.asarray((0, 2, 1, 3, 5, 4)),
        np.asarray((1, -1, -1, -1, 1, 1)),
        valid_mask=np.asarray((True, True, True, True, False, False)),
        coefficient_dtype=coefficient_dtype,
        layout_id="signed-coordinate-test-layout",
    )


def test_signed_projection_round_trip_and_isometry_respect_the_involution() -> None:
    coordinates = _signed_coordinates()
    state = jnp.asarray(
        (1.0 + 2.0j, 2.0 + 3.0j, -4.0 + 5.0j, 6.0 + 7.0j, 8.0 - 9.0j, -1.0j),
        dtype=jnp.complex128,
    )

    projected = coordinates.project(state)
    partners = np.asarray(coordinates.conjugate_indices)
    signs = np.asarray(coordinates.conjugate_signs)
    valid = np.asarray(coordinates.valid_mask)
    projected_host = np.asarray(projected)
    real = coordinates.to_real_coordinates(projected)
    restored = coordinates.from_real_coordinates(real)

    np.testing.assert_array_equal(partners[partners], np.arange(partners.size))
    np.testing.assert_array_equal(signs * signs[partners], np.ones(signs.shape))
    np.testing.assert_allclose(
        projected_host[valid],
        (signs * np.conj(projected_host[partners]))[valid],
        atol=1e-12,
    )
    np.testing.assert_array_equal(projected_host[~valid], 0.0)
    np.testing.assert_allclose(np.asarray(coordinates.project(projected)), projected_host)
    np.testing.assert_allclose(np.asarray(restored), projected_host, atol=1e-12)
    np.testing.assert_allclose(
        np.vdot(projected_host, projected_host).real,
        np.dot(np.asarray(real), np.asarray(real)),
        atol=1e-12,
    )
    assert projected_host[0].imag == pytest.approx(0.0)
    assert projected_host[3].real == pytest.approx(0.0)
    assert np.asarray(real)[0] == pytest.approx(projected_host[0].real)
    assert np.asarray(real)[1] == pytest.approx(projected_host[3].imag)
    assert coordinates.evidence.norm_relation == "isometry"
    assert float(coordinates.defect(projected)) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("coefficient_dtype", "coordinate_dtype"),
    ((jnp.complex64, jnp.float32), (jnp.complex128, jnp.float64)),
)
def test_signed_coordinates_preserve_real_and_complex_precision(
    coefficient_dtype: object, coordinate_dtype: object
) -> None:
    coordinates = SignedHermitianSpectralCoordinates(
        (3,),
        np.asarray((0, 2, 1)),
        np.asarray((1, -1, -1)),
        coefficient_dtype=coefficient_dtype,
        layout_id=f"precision-{jnp.dtype(coefficient_dtype).name}",
    )
    real = jnp.asarray((1.25, -2.0, 0.75), dtype=coordinate_dtype)

    state = coordinates.from_real_coordinates(real)

    assert state.dtype == jnp.dtype(coefficient_dtype)
    assert coordinates.coordinate_space.dtype == np.dtype(coordinate_dtype)
    assert coordinates.to_real_coordinates(state).dtype == jnp.dtype(coordinate_dtype)
    np.testing.assert_allclose(
        np.asarray(coordinates.to_real_coordinates(state)), np.asarray(real)
    )
    np.testing.assert_allclose(
        np.asarray(state[2]), -np.conj(np.asarray(state[1])), atol=1e-6
    )


@pytest.mark.parametrize(
    ("partners", "signs", "valid_mask", "message"),
    (
        ((0, 1, 3), (1, 1, 1), None, "out-of-range"),
        ((0, 0, 2), (1, 1, 1), None, "involution"),
        ((1, 0, 2), (1, 1, 1), (True, False, True), "invariant"),
        ((0, 2, 1), (1, 1, -1), None, "compose"),
        ((0, 2, 1), (1, 0, 0), None, "only real"),
    ),
)
def test_signed_coordinates_reject_invalid_involution_maps(
    partners: tuple[int, ...],
    signs: tuple[int, ...],
    valid_mask: tuple[bool, ...] | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        SignedHermitianSpectralCoordinates(
            (3,),
            np.asarray(partners),
            np.asarray(signs),
            valid_mask=valid_mask,
            coefficient_dtype=jnp.complex128,
            layout_id="invalid-map",
        )


def test_signed_coordinates_reject_real_coefficient_storage() -> None:
    with pytest.raises(TypeError, match="complex dtype"):
        SignedHermitianSpectralCoordinates(
            (1,),
            np.asarray((0,)),
            np.asarray((1,)),
            coefficient_dtype=jnp.float64,
            layout_id="real-coefficients",
        )


def test_tensor_spectral_real_coordinates_follow_prepared_precision() -> None:
    for physical_dtype, coefficient_dtype, coordinate_dtype in (
        (jnp.float32, jnp.complex64, jnp.float32),
        (jnp.float64, jnp.complex128, jnp.float64),
    ):
        precision = phx.discretization.SpectralPrecisionPolicy(physical_dtype)
        space = phx.discretization.TensorSpectralPlan(
            (phx.discretization.FourierBasisPlan(6),),
            axis_names=("x",),
            precision=precision,
        ).prepare((phx.discretization.AxisDomain.periodic(0.0, 1.0),))
        coordinates = space.real_coordinates(component_shape=(2,))
        real = jnp.arange(1, coordinates.coordinate_size + 1, dtype=coordinate_dtype)

        state = coordinates.from_real_coordinates(real)

        assert state.shape == space.modal_shape + (2,)
        assert state.dtype == jnp.dtype(coefficient_dtype)
        assert coordinates.coordinate_space.dtype == np.dtype(coordinate_dtype)
        np.testing.assert_allclose(
            np.asarray(coordinates.to_real_coordinates(state)), np.asarray(real)
        )


def test_spherical_real_coordinates_apply_signed_phases_and_mask_padding() -> None:
    space = phx.discretization.SphericalSpectralPlan(4).prepare()
    coordinates = space.real_coordinates(component_shape=(2,))
    raw = jnp.arange(np.prod(coordinates.state_shape), dtype=jnp.float64).reshape(
        coordinates.state_shape
    )
    raw = raw.astype(jnp.complex128) + 1j * (raw[::-1] + 0.5)

    projected = coordinates.project(raw)
    projected_host = np.asarray(projected)
    valid = np.broadcast_to(
        np.asarray(space.layout.valid_mask)[..., None], projected_host.shape
    )
    center = space.plan.bandlimit - 1

    np.testing.assert_array_equal(projected_host[~valid], 0.0)
    np.testing.assert_allclose(projected_host[:, center, :].imag, 0.0, atol=1e-12)
    for order in range(1, space.plan.bandlimit):
        phase = -1.0 if order % 2 else 1.0
        np.testing.assert_allclose(
            projected_host[order:, center - order, :],
            phase * np.conj(projected_host[order:, center + order, :]),
            atol=1e-12,
        )

    real = coordinates.to_real_coordinates(projected)
    np.testing.assert_allclose(
        np.asarray(coordinates.from_real_coordinates(real)), projected_host, atol=1e-12
    )
    np.testing.assert_allclose(
        np.dot(np.asarray(real), np.asarray(real)),
        np.vdot(projected_host, projected_host).real,
        atol=1e-12,
    )


def test_complex_spherical_space_rejects_real_field_coordinates() -> None:
    precision = phx.discretization.SpectralPrecisionPolicy(jnp.complex128)
    space = phx.discretization.SphericalSpectralPlan(
        3, reality=False, precision=precision
    ).prepare()

    with pytest.raises(ValueError, match="do not have a real-field involution"):
        space.real_coordinates()


def test_lattice_real_coordinates_pair_each_harmonic_with_its_conjugate() -> None:
    lattice = LatticeHarmonicPlan.parallelogramic((5,), (9,)).prepare(
        jnp.asarray(((2.0, 0.0),), dtype=jnp.float64)
    )
    coordinates = lattice.real_coordinates(component_shape=(2,))
    real = jnp.linspace(
        -1.5,
        2.5,
        coordinates.coordinate_size,
        dtype=coordinates.coordinate_space.dtype,
    )

    state = coordinates.from_real_coordinates(real)
    partners = np.asarray(lattice.plan.layout.conjugate_indices)

    np.testing.assert_allclose(
        np.asarray(state[partners]), np.conj(np.asarray(state)), atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(state[lattice.plan.layout.zero_index]).imag, 0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(coordinates.to_real_coordinates(state)), np.asarray(real), atol=1e-12
    )
    assert coordinates.coordinate_size == lattice.harmonic_count * 2
