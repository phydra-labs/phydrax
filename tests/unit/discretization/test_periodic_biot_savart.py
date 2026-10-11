from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization import AxisDomain, FourierBasisPlan, TensorSpectralPlan
from phydrax.discretization.spectral import PeriodicLerayProjector
from phydrax.discretization.spectral._space import TensorSpectralDiscretization


def _space(dimension: int) -> TensorSpectralDiscretization:
    return TensorSpectralPlan(
        tuple(FourierBasisPlan(7) for _ in range(dimension))
    ).prepare(tuple(AxisDomain.periodic(0.3, 2.3) for _ in range(dimension)))


def _curl(space: TensorSpectralDiscretization, velocity: jax.Array) -> jax.Array:
    if len(space.axes) == 2:
        return space.modal_derivative(velocity[..., 1], axis=0) - space.modal_derivative(
            velocity[..., 0], axis=1
        )
    return jnp.stack(
        tuple(
            space.modal_derivative(velocity[..., (a + 2) % 3], axis=(a + 1) % 3)
            - space.modal_derivative(velocity[..., (a + 1) % 3], axis=(a + 2) % 3)
            for a in range(3)
        ),
        axis=-1,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_random_bandlimited_curl_divergence_and_jit(dimension: int) -> None:
    space = _space(dimension)
    projector = PeriodicLerayProjector(space)
    rng = np.random.default_rng(472)
    random = jnp.asarray(
        rng.normal(size=(3,) + projector.state_shape)
        + 1j * rng.normal(size=(3,) + projector.state_shape),
        dtype=jnp.complex128,
    )
    original = jax.vmap(projector.project)(random)
    omega = jax.vmap(lambda u: _curl(space, u))(original)
    recovered = eqx.filter_jit(
        lambda w: projector.velocity_from_vorticity(w, tolerance=1e-12)
    )(omega)
    np.testing.assert_allclose(
        jax.vmap(lambda u: _curl(space, u))(recovered), omega, atol=4e-14
    )
    np.testing.assert_allclose(jax.vmap(projector.divergence)(recovered), 0.0, atol=1e-14)
    np.testing.assert_allclose(
        recovered,
        original.at[(slice(None),) + (0,) * dimension + (slice(None),)].set(0),
        atol=2e-15,
    )


@pytest.mark.parametrize("dimension", [2, 3])
def test_mean_refused_eager_and_traced(dimension: int) -> None:
    projector = PeriodicLerayProjector(_space(dimension))
    shape = projector.discretization.modal_shape + (() if dimension == 2 else (3,))
    value = jnp.ones(shape, dtype=jnp.complex128)
    with pytest.raises(Exception, match="mean-zero"):
        projector.velocity_from_vorticity(value, tolerance=1e-12)
    compiled = eqx.filter_jit(
        lambda w: projector.velocity_from_vorticity(w, tolerance=1e-12)
    )
    with pytest.raises(Exception, match="mean-zero"):
        compiled(value).block_until_ready()


def test_non_solenoidal_refused() -> None:
    projector = PeriodicLerayProjector(_space(3))
    omega = jnp.stack(projector.wavenumbers, axis=-1).astype(jnp.complex128)
    with pytest.raises(Exception, match="solenoidal"):
        eqx.filter_jit(lambda w: projector.velocity_from_vorticity(w, tolerance=1e-12))(
            omega
        ).block_until_ready()


@pytest.mark.parametrize("tolerance", [-1.0, 1.0, np.nan, np.inf])
def test_tolerance_refusal(tolerance: float) -> None:
    projector = PeriodicLerayProjector(_space(2))
    with pytest.raises(ValueError, match="tolerance"):
        projector.velocity_from_vorticity(
            jnp.zeros((7, 7), dtype=jnp.complex128), tolerance=tolerance
        )
