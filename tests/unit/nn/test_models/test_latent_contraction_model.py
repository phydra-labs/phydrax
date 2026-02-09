#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.nn.models import LatentContractionModel, LatentExecutionPolicy
from phydrax.nn.models.core._base import _AbstractBaseModel


FlatTopology = Literal["flat", "best_effort_flat", "strict_flat"]


def _as_scalar(x):
    arr = jnp.asarray(x)
    if arr.ndim == 0:
        return arr
    if arr.size != 1:
        raise ValueError("Expected scalar input for scalar factor model.")
    return arr.reshape(())


class XYLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = 2
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        x = jnp.asarray(x)
        return jnp.stack([x[0] + x[1], x[0] - x[1]], axis=-1)


class ScalarLatentModel(_AbstractBaseModel):
    in_size: int | str
    out_size: int | str

    def __init__(self) -> None:
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x, /, *, key=jr.key(0)):
        x = _as_scalar(x)
        return jnp.stack([x, 2.0 * x], axis=-1)


def test_latent_contraction_mixed_inputs_shape_and_values():
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
    )
    xs = jnp.array([0.0, 1.0])
    ys = jnp.array([2.0, 3.0])
    p = (jnp.array([1.0, 2.0]),)
    out = model({"x": (xs, ys), "p": p})

    X, Y = jnp.meshgrid(xs, ys, indexing="ij")
    p_axis = p[0]
    expected = (3.0 * X - Y)[..., None] * p_axis[None, None, :]
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected)


def test_latent_contraction_aligned_points():
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
    )
    points = jnp.array(
        [
            [0.0, 2.0, 1.0],
            [1.0, 3.0, 2.0],
        ],
        dtype=float,
    )
    out = jnp.stack([model(p) for p in points], axis=0)
    expected = (3.0 * points[:, 0] - points[:, 1]) * points[:, 2]
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected)


def test_latent_contraction_dense_factor_batches():
    model = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
    )
    x_batch = jnp.array(
        [
            [0.0, 2.0],
            [1.0, 3.0],
            [2.0, 1.5],
        ],
        dtype=float,
    )
    p_batch = jnp.array([1.0, 2.0, 3.0], dtype=float)
    out = model({"x": x_batch, "p": p_batch})
    expected = (3.0 * x_batch[:, 0] - x_batch[:, 1])[:, None] * p_batch[None, :]
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected)


@pytest.mark.parametrize("topology", ("flat", "best_effort_flat", "strict_flat"))
def test_latent_contraction_flat_topologies_match_grouped(topology: FlatTopology):
    grouped = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
    )
    planned = LatentContractionModel(
        latent_size=2,
        out_size="scalar",
        x=XYLatentModel(),
        p=ScalarLatentModel(),
        execution_policy=LatentExecutionPolicy(topology=topology, fallback="warn"),
    )
    x = (jnp.array([0.0, 0.5, 1.0]), jnp.array([1.0, 2.0]))
    p = (jnp.array([1.0, 2.0]),)
    expected = grouped({"x": x, "p": p})
    out = planned({"x": x, "p": p})
    assert out.shape == expected.shape
    assert jnp.allclose(out, expected)
