"""Native canonical provider geometry and owned halo-output semantics."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.backends.lattice import NativeJaxLatticeProvider


def test_bounded_halo_block_action_force_and_transformations() -> None:
    shape = (4, 4)
    sites = 16
    rng = np.random.default_rng(343)
    raw = rng.normal(size=(sites, 2, 3, 3)) + 1j * rng.normal(size=(sites, 2, 3, 3))
    links, _ = np.linalg.qr(raw)
    links[..., :, 0] /= np.linalg.det(links)[..., None]
    raw_tangent = rng.normal(size=links.shape) + 1j * rng.normal(size=links.shape)
    tangent = raw_tangent - raw_tangent.conj().swapaxes(-1, -2)
    tangent -= np.trace(tangent, axis1=-2, axis2=-1)[..., None, None] * np.eye(3) / 3
    provider = NativeJaxLatticeProvider()
    with jax.enable_x64():
        values = jnp.asarray(links)
        direction = jnp.asarray(tangent @ links)
        action_mask = jnp.ones((sites, 1), dtype=jnp.bool_)
        force_mask = jnp.ones((sites, 2), dtype=jnp.bool_)
        periodic = (False, False)
        action = lambda u: provider.gauge_action(
            u, 5.7, action_mask, lattice_shape=shape, periodic=periodic
        )
        lattice = links.reshape(shape + (2, 3, 3))
        total = 0.0
        for x in range(3):
            for y in range(3):
                loop = (
                    lattice[x, y, 0]
                    @ lattice[x + 1, y, 1]
                    @ lattice[x, y + 1, 0].conj().T
                    @ lattice[x, y, 1].conj().T
                )
                total += 5.7 * (1 - np.trace(loop).real / 3)
        np.testing.assert_allclose(action(values), total, atol=2e-12)
        _, derivative = jax.jvp(action, (values,), (direction,))
        force = provider.gauge_force(
            values, 5.7, force_mask, lattice_shape=shape, periodic=periodic
        )
        np.testing.assert_allclose(
            derivative,
            jnp.real(jnp.trace(jnp.asarray(tangent) @ force, axis1=-2, axis2=-1)).sum(),
            atol=3e-11,
        )
        owned = force_mask.at[0:4].set(False)
        masked = provider.gauge_force(
            values, 5.7, owned, lattice_shape=shape, periodic=periodic
        )
        np.testing.assert_allclose(
            masked, jnp.where(owned[..., None, None], force, 0), atol=1e-13
        )
        np.testing.assert_allclose(jax.jit(action)(values), action(values), atol=2e-12)
        np.testing.assert_allclose(
            jax.vmap(action)(jnp.stack((values, values))),
            jnp.stack((action(values), action(values))),
            atol=2e-12,
        )
        _, pullback = jax.vjp(action, values)
        np.testing.assert_allclose(
            jnp.real(jnp.sum(pullback(jnp.asarray(1.0))[0] * direction)),
            derivative,
            atol=3e-11,
        )
        np.testing.assert_allclose(
            jax.checkpoint(action)(values), action(values), atol=2e-12
        )
