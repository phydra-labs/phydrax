#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import ShapeDtypeStruct
from jaxtyping import Array, Float, Key
from scipy.stats.qmc import Halton, LatinHypercube, Sobol


_SUPPORTED_SAMPLERS: tuple[str, ...] = (
    "uniform",
    "latin_hypercube",
    "halton",
    "halton_scrambled",
    "sobol",
    "sobol_scrambled",
)


def _normalize_sampler_name(sampler: str) -> str:
    sampler_ = sampler.lower()
    if sampler_ not in _SUPPORTED_SAMPLERS:
        raise ValueError(f"`sampler` must be one of {list(_SUPPORTED_SAMPLERS)}.")
    return sampler_


def seed_from_key(key: object) -> int:
    """Derive a deterministic host seed from a JAX PRNG key value."""
    key_words = np.asarray(key, dtype=np.uint32).reshape(-1)
    h = 1469598103934665603  # FNV-1a 64-bit offset basis
    prime = 1099511628211  # FNV-1a 64-bit prime
    mask = (1 << 64) - 1
    for w in key_words:
        h ^= int(w)
        h = (h * prime) & mask
    return int(h)


def get_sampler_host(
    sampler: str,
    *,
    dim: int,
    seed: int | np.random.Generator,
) -> Callable[[int], np.ndarray]:
    """Host sampler for points in [0,1]^dim (for use inside host callbacks)."""
    sampler_ = _normalize_sampler_name(sampler)
    dim_ = int(dim)

    if sampler_ == "uniform":
        rng = (
            seed
            if isinstance(seed, np.random.Generator)
            else np.random.default_rng(int(seed))
        )

        def sample(n: int) -> np.ndarray:
            return np.asarray(rng.random((int(n), dim_)), dtype=float)

        return sample

    if sampler_ == "latin_hypercube":
        engine = LatinHypercube(dim_, seed=seed)
    elif sampler_ == "halton":
        engine = Halton(dim_, seed=seed)
    elif sampler_ == "halton_scrambled":
        engine = Halton(dim_, scramble=True, seed=seed)
    elif sampler_ == "sobol":
        engine = Sobol(dim_, seed=seed)
    else:  # "sobol_scrambled"
        engine = Sobol(dim_, scramble=True, seed=seed)

    def sample(n: int) -> np.ndarray:
        n_ = int(n)
        return np.asarray(engine.random(n_), dtype=float).reshape((n_, dim_))

    return sample


def get_sampler(
    sampler: str,
) -> Callable[[int, int, Key[Array, ""]], Float[Array, "num_points dim"]]:
    sampler_ = _normalize_sampler_name(sampler)

    if sampler_ == "uniform":

        def uniform(n: int, dim: int, key: Key[Array, ""]):
            return jr.uniform(key, shape=(int(n), int(dim)), dtype=float)

        return uniform

    def wrapped_sampler_fn(n: int, dim: int, key: Key[Array, ""]):
        def _sampler_fn_host(n, dim, key):
            rng = np.random.default_rng(seed_from_key(key))
            host_sampler = get_sampler_host(sampler_, dim=int(dim), seed=rng)
            return np.asarray(host_sampler(int(n)), dtype=np.dtype(dtype))

        zeros = jnp.zeros((n, dim), dtype=float)
        dtype = zeros.dtype
        shape_dtype = ShapeDtypeStruct(zeros.shape, zeros.dtype)
        return jax.pure_callback(_sampler_fn_host, shape_dtype, n, dim, key)

    return wrapped_sampler_fn


__all__ = [
    "get_sampler",
    "get_sampler_host",
    "seed_from_key",
]
