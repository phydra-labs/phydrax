"""Consumer-visible agreement of optional native Qlat gauge kernels."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.backends.lattice import (
    create_lattice_provider,
    LatticeKernelCapabilityError,
    LatticeKernelProvider,
    NativeJaxLatticeProvider,
    qlat_availability,
)


@pytest.fixture
def provider() -> LatticeKernelProvider:
    availability = qlat_availability()
    if not availability.available:
        pytest.skip(availability.reason)
    return create_lattice_provider("qlat")


def _links() -> jax.Array:
    rng = np.random.default_rng(901)
    matrices = rng.normal(size=(256, 4, 3, 3)) + 1j * rng.normal(size=(256, 4, 3, 3))
    unitary, triangular = np.linalg.qr(matrices)
    phases = np.diagonal(triangular, axis1=-2, axis2=-1)
    unitary *= (phases / np.abs(phases))[..., None, :]
    unitary[..., :, 0] /= np.linalg.det(unitary)[..., None]
    return jnp.asarray(unitary, dtype=jnp.complex128)


def test_qlat_gauge_action_matches_native(provider: LatticeKernelProvider) -> None:
    links = _links()
    mask = jnp.ones((256, 6), dtype=jnp.bool_)
    expected = NativeJaxLatticeProvider().gauge_action(
        links, 5.7, mask, lattice_shape=(4,) * 4, periodic=(True,) * 4
    )
    actual = provider.gauge_action(
        links, 5.7, mask, lattice_shape=(4,) * 4, periodic=(True,) * 4
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-13, atol=2e-11)


@pytest.mark.parametrize("masked", [False, True])
def test_qlat_gauge_force_matches_native(
    provider: LatticeKernelProvider, masked: bool
) -> None:
    links = _links()
    mask = jnp.ones((256, 4), dtype=jnp.bool_)
    if masked:
        mask = mask.at[::3, 1].set(False)
    expected = NativeJaxLatticeProvider().gauge_force(
        links, 5.7, mask, lattice_shape=(4,) * 4, periodic=(True,) * 4
    )
    actual = provider.gauge_force(
        links, 5.7, mask, lattice_shape=(4,) * 4, periodic=(True,) * 4
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_qlat_refuses_partial_action_ownership(provider: LatticeKernelProvider) -> None:
    mask = jnp.ones((256, 6), dtype=jnp.bool_).at[0, 0].set(False)
    with pytest.raises(LatticeKernelCapabilityError, match="fully owned"):
        provider.gauge_action(
            _links(), 5.7, mask, lattice_shape=(4,) * 4, periodic=(True,) * 4
        )


def test_qlat_refuses_bounded_geometry(provider: LatticeKernelProvider) -> None:
    with pytest.raises(LatticeKernelCapabilityError, match="periodic 4D"):
        provider.gauge_force(
            _links(),
            5.7,
            jnp.ones((256, 4), dtype=jnp.bool_),
            lattice_shape=(4,) * 4,
            periodic=(True, True, True, False),
        )


def test_qlat_refuses_unadvertised_jit(provider: LatticeKernelProvider) -> None:
    def action(links: jax.Array) -> jax.Array:
        return provider.gauge_force(
            links,
            5.7,
            jnp.ones((256, 4), dtype=jnp.bool_),
            lattice_shape=(4,) * 4,
            periodic=(True,) * 4,
        )

    with pytest.raises(LatticeKernelCapabilityError, match="eager host"):
        jax.jit(action)(_links())


def test_qlat_refuses_single_precision(provider: LatticeKernelProvider) -> None:
    with pytest.raises(LatticeKernelCapabilityError, match="precision envelope"):
        provider.gauge_force(
            _links().astype(jnp.complex64),
            5.7,
            jnp.ones((256, 4), dtype=jnp.bool_),
            lattice_shape=(4,) * 4,
            periodic=(True,) * 4,
        )
