#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ..._utils import _get_size
from ..core._base import _AbstractBaseModel, _AbstractStructuredInputModel


class MagnitudeDirectionModel(_AbstractStructuredInputModel):
    r"""Combine a magnitude model and a direction model.

    Given a scalar magnitude $m(x)$ and a direction field $d(x)$, returns

    $$
    y(x)=m(x)\,\frac{d(x)}{\|d(x)\|}.
    $$

    If `direction_model.out_size == "scalar"`, no normalization is applied.
    """

    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]
    magnitude_model: _AbstractBaseModel
    direction_model: _AbstractBaseModel

    def __init__(
        self,
        magnitude_model: _AbstractBaseModel,
        direction_model: _AbstractBaseModel,
        /,
    ):
        if magnitude_model.in_size != direction_model.in_size:
            raise ValueError(
                "Magnitude and direction models must share the same in_size."
            )
        if _get_size(magnitude_model.out_size) != 1:
            raise ValueError("Magnitude model must have scalar out_size.")

        self.magnitude_model = magnitude_model
        self.direction_model = direction_model
        self.in_size = magnitude_model.in_size
        self.out_size = direction_model.out_size

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Evaluate $y(x)=m(x)\,\frac{d(x)}{\|d(x)\|}$.

        Uses a safe normalization $\|d\|_\text{safe}=\mathop{\text{max}}(\|d\|,1)$ to avoid
        division by zero. If `direction_model.out_size == "scalar"`, the
        normalization is skipped.
        """
        key_mag, key_dir = jr.split(key, 2)
        magnitude_model = self.magnitude_model
        direction_model = self.direction_model
        if isinstance(x, tuple):
            if not isinstance(magnitude_model, _AbstractStructuredInputModel):
                raise TypeError(
                    "MagnitudeDirectionModel received tuple input, but "
                    "`magnitude_model` does not support structured inputs."
                )
            if not isinstance(direction_model, _AbstractStructuredInputModel):
                raise TypeError(
                    "MagnitudeDirectionModel received tuple input, but "
                    "`direction_model` does not support structured inputs."
                )
            mag = magnitude_model(x, key=key_mag)
            direction = direction_model(x, key=key_dir)
        else:
            x_arr = jnp.asarray(x)
            mag = magnitude_model(x_arr, key=key_mag)
            direction = direction_model(x_arr, key=key_dir)

        dir_size = _get_size(self.out_size)
        if dir_size == 1:
            unit_dir = direction
        else:
            y = jnp.asarray(direction)
            if y.ndim == 0:
                raise ValueError(
                    "Direction model out_size > 1 but returned scalar output."
                )
            norm = jnp.linalg.norm(y, axis=-1, keepdims=True)
            norm_safe = jnp.where(norm == 0, 1.0, norm)
            unit_dir = y / norm_safe

        if mag.ndim == unit_dir.ndim - 1:
            mag = jnp.expand_dims(mag, axis=-1)

        return mag * unit_dir
