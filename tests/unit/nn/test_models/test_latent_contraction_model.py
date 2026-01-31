#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr

from phydrax.nn.models import LatentContractionModel
from phydrax.nn.models.core._base import _AbstractBaseModel


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
