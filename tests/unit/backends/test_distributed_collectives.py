"""Native named-axis reduce-scatter and its all-gather adjoint.

Lane cases run the owner regions as named vmap lanes; device cases bind one
owner per visible device through ``shard_map``. Forced CPU devices prove the
functional collective contract only, not accelerator qualification.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, PartitionSpec

from phydrax.backends.distributed import (
    JAX_NATIVE_DISTRIBUTION,
    JaxCollectiveProvider,
    MPI4JAX_CAPABILITIES,
    Mpi4JaxCollectiveProvider,
)


AXIS = "owner"


def _lanes[**P](function: Callable[P, jax.Array], owners: int) -> Callable[P, jax.Array]:
    return jax.vmap(function, axis_name=AXIS, axis_size=owners)


def test_reduce_scatter_is_declared_only_where_implemented() -> None:
    assert "reduce_scatter" in JAX_NATIVE_DISTRIBUTION.collectives
    assert JaxCollectiveProvider.capabilities is JAX_NATIVE_DISTRIBUTION
    # mpi4jax exposes no reduce-scatter primitive: the rank-local MPI route
    # declares no such capability and offers no emulation.
    assert "reduce_scatter" not in MPI4JAX_CAPABILITIES.distribution.collectives
    assert not hasattr(Mpi4JaxCollectiveProvider, "reduce_scatter")


@pytest.mark.parametrize("owners", [1, 2, 3, 5])
def test_lane_reduce_scatter_sums_each_owner_block_once(owners: int) -> None:
    block = 3
    rng = np.random.default_rng(owners)
    values = rng.normal(size=(owners, owners * block, 2))
    provider = JaxCollectiveProvider(AXIS)
    observed = _lanes(provider.reduce_scatter, owners)(jnp.asarray(values))
    expected = np.sum(values, axis=0).reshape((owners, block, 2))
    assert observed.shape == (owners, block, 2)
    np.testing.assert_allclose(np.asarray(observed), expected, rtol=1e-14, atol=1e-14)


def test_lane_reduce_scatter_along_a_trailing_axis() -> None:
    owners, block = 4, 2
    rng = np.random.default_rng(3)
    values = rng.normal(size=(owners, 5, owners * block))
    provider = JaxCollectiveProvider(AXIS)
    observed = _lanes(
        lambda value: provider.reduce_scatter(value, scatter_axis=1), owners
    )(jnp.asarray(values))
    total = np.sum(values, axis=0)
    expected = np.stack(
        [total[:, owner * block : (owner + 1) * block] for owner in range(owners)]
    )
    np.testing.assert_allclose(np.asarray(observed), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("owners", [1, 2, 4])
def test_reduce_scatter_and_tiled_all_gather_are_mutual_adjoints(owners: int) -> None:
    """``<G x, y> == <x, G^T y>`` summed over owners, and each VJP is the other."""
    block = 3
    rng = np.random.default_rng(10 + owners)
    owned = jnp.asarray(rng.normal(size=(owners, block, 2)))
    columns = jnp.asarray(rng.normal(size=(owners, owners * block, 2)))
    provider = JaxCollectiveProvider(AXIS)

    def gather(value: jax.Array) -> jax.Array:
        return provider.all_gather(value, axis=0, tiled=True)

    gathered = _lanes(gather, owners)(owned)
    scattered = _lanes(provider.reduce_scatter, owners)(columns)
    np.testing.assert_allclose(
        float(jnp.vdot(gathered, columns)),
        float(jnp.vdot(owned, scattered)),
        rtol=1e-13,
    )

    def gather_vjp(value: jax.Array, cotangent: jax.Array) -> jax.Array:
        _, pullback = jax.vjp(gather, value)
        return pullback(cotangent)[0]

    def scatter_vjp(value: jax.Array, cotangent: jax.Array) -> jax.Array:
        _, pullback = jax.vjp(provider.reduce_scatter, value)
        return pullback(cotangent)[0]

    np.testing.assert_allclose(
        np.asarray(_lanes(gather_vjp, owners)(owned, columns)),
        np.asarray(scattered),
        rtol=1e-14,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(_lanes(scatter_vjp, owners)(columns, owned)),
        np.asarray(gathered),
        rtol=1e-14,
        atol=1e-14,
    )


@pytest.mark.parametrize("owners", [2, 3, 4])
def test_device_reduce_scatter_matches_lane_reference(owners: int) -> None:
    devices = jax.devices()
    if len(devices) < owners:
        pytest.skip(f"Requires at least {owners} visible devices.")
    mesh = Mesh(np.asarray(devices[:owners]), (AXIS,))
    block = 2
    rng = np.random.default_rng(20 + owners)
    values = rng.normal(size=(owners * owners * block, 3))
    provider = JaxCollectiveProvider(AXIS)
    observed = jax.shard_map(
        provider.reduce_scatter,
        mesh=mesh,
        in_specs=PartitionSpec(AXIS),
        out_specs=PartitionSpec(AXIS),
        check_vma=False,
    )(jnp.asarray(values))
    lanes = _lanes(provider.reduce_scatter, owners)(
        jnp.asarray(values.reshape((owners, owners * block, 3)))
    )
    np.testing.assert_allclose(
        np.asarray(observed),
        np.asarray(lanes).reshape((owners * block, 3)),
        rtol=1e-14,
        atol=1e-14,
    )
