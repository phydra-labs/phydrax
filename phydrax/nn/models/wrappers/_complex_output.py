#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Literal, overload

import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ...._strict import StrictModule
from ..._utils import _get_size
from ..core._base import _AbstractBaseModel, _AbstractStructuredInputModel


class ComplexOutputModel(StrictModule):
    r"""Wrap model(s) and return complex outputs.

    Two modes:
    - Single model with real output size $2k$: split the last axis into
      $(\Re z,\Im z)$ and return $z=\Re z + i\,\Im z$.
    - Pair `(real_model, imag_model)` with real output size $k$: return
      $z=u_{\text{re}}+i\,u_{\text{im}}$.

    If $k=1$ / `"scalar"`, returns complex scalars (squeezes a trailing unit feature axis).
    """

    # Internal storage
    _single_model: _AbstractBaseModel | None = None
    _real_model: _AbstractBaseModel | None = None
    _imag_model: _AbstractBaseModel | None = None

    # Cached metadata
    _in_size: int | tuple[int, ...] | Literal["scalar"] = 1
    _base_out_size: int = 1

    @overload
    def __init__(self, model: _AbstractBaseModel, /): ...

    @overload
    def __init__(self, models: tuple[_AbstractBaseModel, _AbstractBaseModel], /): ...

    def __init__(
        self,
        model_or_models: _AbstractBaseModel
        | tuple[_AbstractBaseModel, _AbstractBaseModel],
        /,
    ):
        if isinstance(model_or_models, tuple):
            if len(model_or_models) != 2:
                raise ValueError("ComplexOutputModel expects a (real, imag) model pair.")
            real_model, imag_model = model_or_models
            if not isinstance(real_model, _AbstractBaseModel) or not isinstance(
                imag_model, _AbstractBaseModel
            ):
                raise TypeError(
                    "ComplexOutputModel expects models to be _AbstractBaseModel."
                )
            # Validate in_size compatibility
            if real_model.in_size != imag_model.in_size:
                raise ValueError(
                    f"Mismatched in_size: {real_model.in_size} != {imag_model.in_size}."
                )

            # Validate out_size compatibility and determine base k
            k_r = _get_size(real_model.out_size)
            k_i = _get_size(imag_model.out_size)
            if k_r != k_i:
                raise ValueError(
                    f"Mismatched out_size: {real_model.out_size} != {imag_model.out_size}."
                )

            self._single_model = None
            self._real_model = real_model
            self._imag_model = imag_model
            self._in_size = real_model.in_size
            self._base_out_size = k_r

        else:
            model = model_or_models
            # Validate single-model out_size is even (2k)
            out_sz = _get_size(model.out_size)
            if out_sz % 2 != 0:
                raise ValueError(
                    f"Single model must have even out_size (2k). Got {model.out_size}."
                )

            self._single_model = model
            self._real_model = None
            self._imag_model = None
            self._in_size = model.in_size
            self._base_out_size = out_sz // 2

    @property
    def in_size(self) -> int | tuple[int, ...] | Literal["scalar"]:
        return self._in_size

    @property
    def out_size(self) -> str:
        k = self._base_out_size
        return "complex_scalar" if k == 1 else f"complex_{k}"

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        r"""Evaluate the wrapped model(s) and return complex outputs.

        - **Single-model mode**: if the wrapped model returns $[u_{\text{re}},u_{\text{im}}]$
          (feature axis size $2k$), this returns $u_{\text{re}}+i\,u_{\text{im}}$.
        - **Two-model mode**: returns $u_{\text{re}}(x)+i\,u_{\text{im}}(x)$.
        """
        if self._single_model is not None:
            model = self._single_model
            if isinstance(x, tuple):
                if not isinstance(model, _AbstractStructuredInputModel):
                    raise TypeError(
                        "ComplexOutputModel received tuple input, but the wrapped model "
                        "does not support structured inputs."
                    )
                y = model(x, key=key)
            else:
                x_arr = jnp.asarray(x)
                y = model(x_arr, key=key)
            # Expect final axis to be size 2k. Split into real/imag halves.
            if y.ndim == 0:
                raise ValueError(
                    "Single-model complex wrapper received scalar output; expected feature axis of size 2."
                )
            k = self._base_out_size
            # Split last axis: (..., 2k) -> (..., k), (..., k)
            real, imag = jnp.split(y, 2, axis=-1)
            z = real + 1j * imag
            # If base k corresponds to scalar, squeeze trailing axis
            if k == 1:
                z = jnp.squeeze(z, axis=-1)
            return z

        assert self._real_model is not None and self._imag_model is not None
        real_model = self._real_model
        imag_model = self._imag_model
        key_r, key_i = jr.split(key, 2)
        if isinstance(x, tuple):
            if not isinstance(real_model, _AbstractStructuredInputModel):
                raise TypeError(
                    "ComplexOutputModel received tuple input, but the real-part "
                    "model does not support structured inputs."
                )
            if not isinstance(imag_model, _AbstractStructuredInputModel):
                raise TypeError(
                    "ComplexOutputModel received tuple input, but the imag-part "
                    "model does not support structured inputs."
                )
            real = real_model(x, key=key_r)
            imag = imag_model(x, key=key_i)
        else:
            x_arr = jnp.asarray(x)
            real = real_model(x_arr, key=key_r)
            imag = imag_model(x_arr, key=key_i)

        # Align shapes conservatively for scalar vs vector outputs
        if real.shape != imag.shape:
            # Attempt minimal broadcasting for a trailing unit feature axis
            if (
                real.ndim + 1 == imag.ndim
                and imag.shape[-1] == 1
                and real.shape == imag.shape[:-1]
            ):
                real = jnp.expand_dims(real, axis=-1)
            elif (
                imag.ndim + 1 == real.ndim
                and real.shape[-1] == 1
                and imag.shape == real.shape[:-1]
            ):
                imag = jnp.expand_dims(imag, axis=-1)
            else:
                raise ValueError(
                    f"Mismatched output shapes for real/imag models: {real.shape} vs {imag.shape}."
                )

        z = real + 1j * imag
        return z
