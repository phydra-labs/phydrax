import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy import integrate

from phydrax._trainable import partition_parameters
from phydrax.nn.atomistic._radial import (
    AgnesiTransform,
    BesselRadialBasis,
    normalized_silu_scale,
    PolynomialCutoff,
    RadialEmbedding,
    RadialMLP,
    RadialPostprocess,
)


pytestmark = pytest.mark.strict_jax

CUTOFF = 5.0
RADII = np.asarray([0.3, 1.1, 2.5, 3.7, 4.9], dtype=np.float64)
COVALENT_RADII = np.asarray([0.31, 0.76, 0.66], dtype=np.float64)
ATOMIC_NUMBERS = (1, 6, 8)


def _envelope(r: np.ndarray, p: int) -> np.ndarray:
    u = r / CUTOFF
    value = (
        1.0
        - (p + 1.0) * (p + 2.0) / 2.0 * u**p
        + p * (p + 2.0) * u ** (p + 1)
        - p * (p + 1.0) / 2.0 * u ** (p + 2)
    )
    return np.where(r < CUTOFF, value, 0.0)


def _agnesi(r: np.ndarray, r0: np.ndarray, a: float, q: float, p: float) -> np.ndarray:
    ratio = r / r0
    return 1.0 / (1.0 + a * ratio**q / (1.0 + ratio ** (q - p)))


def _transform() -> AgnesiTransform:
    return AgnesiTransform(
        COVALENT_RADII, atomic_numbers=ATOMIC_NUMBERS, a=1.0805, q=0.9183, p=4.5791
    )


def test_native_bessel_basis_matches_closed_form() -> None:
    basis = BesselRadialBasis.native(CUTOFF, 6)
    n = np.arange(1, 7, dtype=np.float64)
    expected = (
        math.sqrt(2.0 / CUTOFF)
        * np.sin(n * math.pi * RADII[:, None] / CUTOFF)
        / RADII[:, None]
    )

    np.testing.assert_allclose(basis(jnp.asarray(RADII)), expected, rtol=1e-13)


def test_source_bessel_values_are_bound_exactly() -> None:
    frequencies = np.asarray([0.61, 1.27, 1.93], dtype=np.float64)
    basis = BesselRadialBasis(frequencies, 0.7, CUTOFF)

    np.testing.assert_array_equal(np.asarray(basis.frequencies), frequencies)
    assert basis.basis_id != BesselRadialBasis.native(CUTOFF, 3).basis_id


@pytest.mark.parametrize("power", [5, 6], ids=["p5", "p6"])
def test_polynomial_cutoff_values_and_c2_join(power: int) -> None:
    cutoff = PolynomialCutoff(CUTOFF, power)

    def envelope(r: jax.Array) -> jax.Array:
        return cutoff(r)

    scaled = RADII / CUTOFF
    term_magnitude = (
        1.0
        + (power + 1.0) * (power + 2.0) / 2.0 * scaled**power
        + power * (power + 2.0) * scaled ** (power + 1)
        + power * (power + 1.0) / 2.0 * scaled ** (power + 2)
    )
    # Near the cutoff, cancellation makes output-relative error ill-conditioned.
    # Bound rounding against the unreduced terms and conservative operation count.
    rounding = (3 * power + 10) * np.finfo(np.float64).eps
    forward_bound = rounding / (1.0 - rounding) * term_magnitude
    np.testing.assert_array_less(
        np.abs(np.asarray(cutoff(jnp.asarray(RADII))) - _envelope(RADII, power)),
        forward_bound,
    )
    below = jnp.asarray(CUTOFF * (1.0 - 1.0e-7))
    first = jax.grad(envelope)
    second = jax.grad(first)
    assert abs(float(envelope(below))) < 1e-12
    assert abs(float(first(below))) < 1e-10
    assert abs(float(second(below))) < 1e-5
    for radius in (CUTOFF, 1.5 * CUTOFF):
        assert float(envelope(jnp.asarray(radius))) == 0.0
        assert float(first(jnp.asarray(radius))) == 0.0
    assert cutoff.regularity_order == 2


def test_agnesi_uses_explicit_species_radii_in_model_order() -> None:
    transform = _transform()
    sender = jnp.asarray([0, 1, 2, 2, 1], dtype=jnp.int32)
    receiver = jnp.asarray([1, 1, 0, 2, 2], dtype=jnp.int32)
    r0 = 0.5 * (COVALENT_RADII[np.asarray(sender)] + COVALENT_RADII[np.asarray(receiver)])

    np.testing.assert_allclose(
        transform(jnp.asarray(RADII), sender, receiver),
        _agnesi(RADII, r0, 1.0805, 0.9183, 4.5791),
        rtol=1e-13,
    )


def test_embedding_applies_cutoff_to_untransformed_distance() -> None:
    basis = BesselRadialBasis.native(CUTOFF, 4)
    embedding = RadialEmbedding(
        basis, PolynomialCutoff(CUTOFF, 5), transform=_transform()
    )
    sender = jnp.asarray([0, 1, 2, 0, 1], dtype=jnp.int32)
    receiver = jnp.asarray([2, 0, 1, 0, 1], dtype=jnp.int32)
    r0 = 0.5 * (COVALENT_RADII[np.asarray(sender)] + COVALENT_RADII[np.asarray(receiver)])
    transformed = _agnesi(RADII, r0, 1.0805, 0.9183, 4.5791)
    n = np.arange(1, 5, dtype=np.float64)
    expected = (
        math.sqrt(2.0 / CUTOFF)
        * np.sin(n * math.pi * transformed[:, None] / CUTOFF)
        / transformed[:, None]
        * _envelope(RADII, 5)[:, None]
    )

    np.testing.assert_allclose(embedding(RADII, sender, receiver), expected, rtol=1e-12)
    assert embedding.pair_dependent


def test_padded_lanes_are_exact_zero_with_finite_gradients() -> None:
    embedding = RadialEmbedding(
        BesselRadialBasis.native(CUTOFF, 4), PolynomialCutoff(CUTOFF, 5)
    )
    radius = jnp.asarray([0.0, 2.0], dtype=jnp.float64)
    species = jnp.zeros((2,), dtype=jnp.int32)
    valid = jnp.asarray([False, True])

    def total(r: jax.Array) -> jax.Array:
        return jnp.sum(embedding(r, species, species, valid=valid))

    features = embedding(radius, species, species, valid=valid)
    gradient = jax.grad(total)(radius)

    np.testing.assert_array_equal(np.asarray(features[0]), np.zeros(4))
    assert np.all(np.isfinite(np.asarray(gradient)))
    assert float(gradient[0]) == 0.0


def _reference_mlp(
    weights: list[np.ndarray], scale: float, postprocess: str, x: np.ndarray
) -> np.ndarray:
    value = x
    for index, weight in enumerate(weights):
        value = value @ (weight / math.sqrt(weight.shape[0]))
        if index < len(weights) - 1:
            value = scale * value / (1.0 + np.exp(-value))
    return np.tanh(value**2) if postprocess == "tanh-square" else value


@pytest.mark.parametrize("postprocess", ["none", "tanh-square"])
def test_radial_mlp_matches_unit_variance_reference(
    postprocess: RadialPostprocess,
) -> None:
    mlp = RadialMLP.initialize(
        (4, 7, 5, 3), key=jr.key(2), activation_scale=1.679, postprocess=postprocess
    )
    features = np.random.default_rng(3).normal(size=(6, 4))
    expected = _reference_mlp(
        [np.asarray(weight) for weight in mlp.weights], 1.679, postprocess, features
    )

    np.testing.assert_allclose(mlp(jnp.asarray(features)), expected, rtol=1e-12)
    np.testing.assert_allclose(
        mlp.project(mlp.penultimate(jnp.asarray(features))), expected, rtol=1e-12
    )


def test_bias_free_mlp_vanishes_at_zero_features() -> None:
    mlp = RadialMLP.initialize((4, 8, 2), key=jr.key(1), postprocess="tanh-square")

    np.testing.assert_array_equal(np.asarray(mlp(jnp.zeros((4,)))), np.zeros(2))


def test_native_activation_scale_is_the_standard_normal_second_moment() -> None:
    def integrand(z: float) -> float:
        silu = z / (1.0 + math.exp(-z))
        return silu * silu * math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

    moment, _ = integrate.quad(integrand, -40.0, 40.0, epsabs=1e-14, epsrel=1e-13)

    assert normalized_silu_scale() == pytest.approx(1.0 / math.sqrt(moment), rel=1e-11)


def test_radial_parameters_are_mlp_weights_only() -> None:
    mlp = RadialMLP.initialize((4, 6, 2), key=jr.key(0))
    embedding = RadialEmbedding(
        BesselRadialBasis.native(CUTOFF, 4),
        PolynomialCutoff(CUTOFF, 5),
        transform=_transform(),
    )

    assert len(jax.tree_util.tree_leaves(partition_parameters(mlp)[0])) == 2
    assert jax.tree_util.tree_leaves(partition_parameters(embedding)[0]) == []
    gradient = eqx.filter_grad(lambda m: jnp.sum(m(jnp.ones((3, 4))) ** 2))(mlp)
    assert all(bool(jnp.any(weight != 0.0)) for weight in gradient.weights)


def test_radial_refusals() -> None:
    with pytest.raises(ValueError, match="share one cutoff"):
        RadialEmbedding(BesselRadialBasis.native(CUTOFF, 4), PolynomialCutoff(4.0, 5))
    with pytest.raises(ValueError, match="inconsistent widths"):
        RadialMLP((jnp.ones((4, 3)), jnp.ones((2, 1))), activation_scale=1.0)
    with pytest.raises(ValueError):
        RadialMLP((jnp.ones((4, 3)),), activation_scale=1.0, postprocess="softplus")  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="finite and positive"):
        AgnesiTransform(
            np.asarray([0.3, -0.1]), atomic_numbers=(1, 6), a=1.0, q=1.0, p=4.0
        )
    with pytest.raises(ValueError, match="distinct"):
        AgnesiTransform(
            np.asarray([0.3, 0.4]), atomic_numbers=(6, 6), a=1.0, q=1.0, p=4.0
        )
