"""Canonical Wilson loops on full or halo-bounded tensor lattice blocks."""

from math import prod

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def canonical_links(
    links: ArrayLike, lattice_shape: tuple[int, ...], periodic: tuple[bool, ...]
) -> Array:
    values = jnp.asarray(links)
    dimension = len(lattice_shape)
    if (
        dimension < 2
        or len(periodic) != dimension
        or any(size < 1 for size in lattice_shape)
    ):
        raise ValueError(
            "Lattice shape and boundary flags must define at least two axes."
        )
    if (
        values.ndim != 4
        or values.shape[:2] != (prod(lattice_shape), dimension)
        or values.shape[-1] != values.shape[-2]
    ):
        raise ValueError("Links must have shape (site, direction, color, color).")
    return values


def wilson_geometry(
    links: ArrayLike, lattice_shape: tuple[int, ...], periodic: tuple[bool, ...]
) -> tuple[Array, Array, Array]:
    values = canonical_links(links, lattice_shape, periodic)
    dimension = len(lattice_shape)
    colors = values.shape[-1]
    field = values.reshape(lattice_shape + (dimension, colors, colors))
    coordinates = jnp.indices(lattice_shape)
    plaquettes = []
    valid_faces = []
    staples = jnp.zeros_like(field)
    for mu in range(dimension):
        for nu in range(mu + 1, dimension):
            valid = jnp.ones(lattice_shape, dtype=jnp.bool_)
            if not periodic[mu]:
                valid = valid & (coordinates[mu] < lattice_shape[mu] - 1)
            if not periodic[nu]:
                valid = valid & (coordinates[nu] < lattice_shape[nu] - 1)
            u = field[..., mu, :, :]
            v = field[..., nu, :, :]
            u_nu = jnp.roll(u, -1, axis=nu)
            v_mu = jnp.roll(v, -1, axis=mu)
            v_dag = jnp.swapaxes(jnp.conj(v), -1, -2)
            u_nu_dag = jnp.swapaxes(jnp.conj(u_nu), -1, -2)
            v_mu_dag = jnp.swapaxes(jnp.conj(v_mu), -1, -2)
            plaquettes.append(u @ v_mu @ u_nu_dag @ v_dag)
            valid_faces.append(valid)
            # The same oriented face contributes to each of its four edges.
            forward_mu = v @ u_nu @ v_mu_dag
            forward_nu = u @ v_mu @ u_nu_dag
            forward_mu = jnp.where(valid[..., None, None], forward_mu, 0)
            forward_nu = jnp.where(valid[..., None, None], forward_nu, 0)
            backward_mu = jnp.roll(v_dag @ u @ v_mu, 1, axis=nu)
            backward_nu = jnp.roll(
                jnp.swapaxes(jnp.conj(u), -1, -2) @ v @ u_nu, 1, axis=mu
            )
            backward_mu = jnp.where(
                jnp.roll(valid, 1, axis=nu)[..., None, None], backward_mu, 0
            )
            backward_nu = jnp.where(
                jnp.roll(valid, 1, axis=mu)[..., None, None], backward_nu, 0
            )
            staples = staples.at[..., mu, :, :].add(forward_mu + backward_mu)
            staples = staples.at[..., nu, :, :].add(forward_nu + backward_nu)
    planes = dimension * (dimension - 1) // 2
    return (
        jnp.stack(plaquettes, axis=dimension).reshape(
            (prod(lattice_shape), planes, colors, colors)
        ),
        jnp.stack(valid_faces, axis=dimension).reshape((prod(lattice_shape), planes)),
        staples.reshape(values.shape),
    )
