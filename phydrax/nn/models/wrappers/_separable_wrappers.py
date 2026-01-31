#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import functools as ft
import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key
from opt_einsum import contract

from ...._doc import DOC_KEY0
from ..._utils import _get_size, _identity
from .._utils import _contract_str, _stack_separable
from ..core._base import _AbstractBaseModel, _AbstractStructuredInputModel


class LatentContractionModel(_AbstractStructuredInputModel):
    r"""Latent contraction wrapper for product-domain factor models.

    This implements a low-rank (CP-style) factorization over a product input
    space. For factors $x=(x^{(1)},\dots,x^{(d)})$ and latent size $L$, each
    factor model returns features that can be reshaped to
    $g_i(x^{(i)})\in\mathbb{R}^{L\times m}$ (with $m=\texttt{out\_size}$). The
    contraction returns

    $$
    u_o(x)=\sum_{\ell=1}^{L}\prod_{i=1}^{d} g_{i,\ell,o}(x^{(i)}).
    $$

    Each factor model may return either:
    - $L\cdot m$ features (interpreted as $(L,m)$), or
    - $L$ features (broadcast across the $m$ outputs).
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    latent_size: int
    factor_names: tuple[str, ...]
    factor_models: tuple[_AbstractBaseModel, ...]
    output_activation: Callable
    keep_outputs_complex: bool

    _factor_sizes: tuple[int, ...]
    _total_in_size: int

    def __init__(
        self,
        *,
        latent_size: int,
        out_size: int | Literal["scalar"],
        factors: Mapping[str, _AbstractBaseModel] | None = None,
        output_activation: Callable | None = None,
        keep_outputs_complex: bool = False,
        key: Key[Array, ""] = DOC_KEY0,
        **factor_models: _AbstractBaseModel,
    ):
        r"""Create a latent contraction model.

        **Keyword arguments:**

        - `latent_size`: Rank $L$ of the factorization.
        - `out_size`: Output size $m$ (or `"scalar"`).
        - `factors` / `**factor_models`: Factor models $g_i$ mapping factor inputs
          to latent features.
        - `output_activation`: Optional activation applied after contraction (wrap it
          yourself if you want adaptive behavior).
        - `keep_outputs_complex`: If `True`, keeps complex outputs when the
          factors are complex-valued; otherwise returns the real part.

        Each factor model should return either $L$ features or $L\cdot m$ features
        so the wrapper can reshape to $(L,m)$.
        """
        del key
        if factors is None:
            factors = factor_models
        elif factor_models:
            raise ValueError("Provide either factors=... or **factor_models, not both.")
        if not factors:
            raise ValueError("LatentContractionModel requires at least one factor model.")

        self.factor_names = tuple(factors.keys())
        self.factor_models = tuple(factors.values())
        self.latent_size = int(latent_size)
        self.out_size = out_size
        self._factor_sizes = tuple(_get_size(m.in_size) for m in self.factor_models)
        self._total_in_size = int(sum(self._factor_sizes))
        self.in_size = self._total_in_size

        self.output_activation = (
            _identity if output_activation is None else output_activation
        )
        self.keep_outputs_complex = bool(keep_outputs_complex)

    def __call__(
        self,
        x: Array | tuple[Array, ...] | Mapping[str, Any],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        if isinstance(x, Mapping):
            factor_inputs = [x[name] for name in self.factor_names]  # ty: ignore
            return self._call_factorwise(factor_inputs, key=key)
        if isinstance(x, tuple):
            if len(self.factor_models) == 1:
                factor_inputs = [x]
            else:
                if len(x) != len(self.factor_models):
                    raise ValueError(
                        "Tuple input must match the number of factor models."
                    )
                factor_inputs = list(x)
            return self._call_factorwise(factor_inputs, key=key)
        return self._call_aligned(jnp.asarray(x), key=key)

    def _call_factorwise(
        self, factor_inputs: Sequence[Any], /, *, key: Key[Array, ""] = DOC_KEY0
    ) -> Array:
        if len(factor_inputs) != len(self.factor_models):
            raise ValueError("Factor input count does not match the model factors.")
        keys = self._split_key(key)
        latents = []
        batch_shapes: list[tuple[int, ...]] = []
        for name, model, pts, k in zip(
            self.factor_names, self.factor_models, factor_inputs, keys, strict=True
        ):
            lats, shape = self._eval_factor(model, pts, name=name, key=k)
            latents.append(lats)
            batch_shapes.append(shape)
        equation = self._contraction_equation(batch_shapes)
        out = contract(equation, *latents)
        return self._finalize(out)

    def _call_aligned(self, x: Array, /, *, key: Key[Array, ""] = DOC_KEY0) -> Array:
        factor_inputs = self._split_aligned_input(x)
        keys = self._split_key(key)
        latents = []
        batch_size = None
        for name, model, pts, k in zip(
            self.factor_names, self.factor_models, factor_inputs, keys, strict=True
        ):
            lats, shape = self._eval_factor_array(model, pts, name=name, key=k)
            latents.append(lats)
            if shape:
                if len(shape) != 1:
                    raise ValueError(
                        "Aligned inputs must have a single batch axis per factor."
                    )
                if batch_size is None:
                    batch_size = shape[0]
                elif batch_size != shape[0]:
                    raise ValueError(
                        "Aligned inputs require matching batch sizes across factors."
                    )
        prod = ft.reduce(jnp.multiply, latents, jnp.array(1.0))
        out = jnp.sum(prod, axis=-2)
        return self._finalize(out)

    def _eval_factor(
        self,
        model: _AbstractBaseModel,
        pts: Any,
        /,
        *,
        name: str,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[Array, tuple[int, ...]]:
        if isinstance(pts, tuple):
            coords = tuple(jnp.asarray(c) for c in pts)
            if not coords:
                raise ValueError(f"Factor {name!r} received an empty coord tuple.")
            if not all(c.ndim == 1 for c in coords):
                raise ValueError(
                    f"Factor {name!r} coord-separable input must be 1D arrays."
                )
            in_dim = _get_size(model.in_size)
            if len(coords) != in_dim:
                raise ValueError(
                    f"Factor {name!r} expected {in_dim} coord axes, got {len(coords)}."
                )
            points = _stack_separable(coords)
            out = jax.vmap(ft.partial(model, key=key))(points)
            batch_shape = tuple(int(c.shape[0]) for c in coords)
            out = jnp.asarray(out)
            if out.ndim == 1:
                out = out[:, None]
            out = out.reshape(*batch_shape, *out.shape[1:])
            return self._reshape_latents(out, name=name), batch_shape
        return self._eval_factor_array(model, jnp.asarray(pts), name=name, key=key)

    def _eval_factor_array(
        self,
        model: _AbstractBaseModel,
        pts: Array,
        /,
        *,
        name: str,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[Array, tuple[int, ...]]:
        arr = jnp.asarray(pts)
        in_dim = _get_size(model.in_size)
        batch_shape: tuple[int, ...]
        if in_dim == 1:
            if arr.ndim == 0:
                out = model(arr, key=key)
                batch_shape = ()
            elif arr.ndim == 1 and arr.shape[0] == 1:
                out = model(arr.reshape(()), key=key)
                batch_shape = ()
            elif arr.ndim == 2 and arr.shape == (1, 1):
                out = model(arr.reshape(()), key=key)
                batch_shape = ()
            else:
                raise ValueError(
                    f"Factor {name!r} expected scalar input with shape () or (1,), got {arr.shape}. "
                    "Pass a tuple of 1D coordinate arrays for coord-separable axes or vmap for batches."
                )
        else:
            if arr.ndim == 1:
                if arr.shape[0] != in_dim:
                    raise ValueError(
                        f"Factor {name!r} expected shape ({in_dim},), got {arr.shape}."
                    )
                out = model(arr, key=key)
                batch_shape = ()
            else:
                raise ValueError(
                    f"Factor {name!r} expected shape ({in_dim},), got {arr.shape}. "
                    "Pass a tuple of 1D coordinate arrays for coord-separable axes or vmap for batches."
                )
        out = jnp.asarray(out)
        if out.ndim == len(batch_shape):
            out = out[..., None]
        if batch_shape:
            out = out.reshape(*batch_shape, out.shape[-1])
        return self._reshape_latents(out, name=name), batch_shape

    def _reshape_latents(self, out: Array, /, *, name: str) -> Array:
        out_dim = int(out.shape[-1])
        out_size = _get_size(self.out_size)
        latent_size = int(self.latent_size)
        if out_dim == latent_size * out_size:
            return out.reshape(*out.shape[:-1], latent_size, out_size)
        if out_dim == latent_size:
            latents = out.reshape(*out.shape[:-1], latent_size, 1)
            if out_size != 1:
                latents = jnp.broadcast_to(latents, latents.shape[:-1] + (out_size,))
            return latents
        if latent_size == 1 and out_dim == out_size:
            return out.reshape(*out.shape[:-1], 1, out_size)
        raise ValueError(
            f"Factor {name!r} returned {out_dim} features; expected "
            f"{latent_size} or {latent_size * out_size}."
        )

    def _split_aligned_input(self, x: Array, /) -> list[Array]:
        x_arr = jnp.asarray(x)
        if x_arr.ndim == 0:
            if self._total_in_size != 1:
                raise ValueError(
                    f"Aligned input has size 1 but model expects {self._total_in_size}."
                )
            return [x_arr]
        if x_arr.ndim == 1:
            if x_arr.shape[0] != self._total_in_size:
                raise ValueError(
                    f"Aligned input expected shape ({self._total_in_size},), got {x_arr.shape}."
                )
            splits = []
            start = 0
            for size in self._factor_sizes:
                seg = x_arr[start : start + size]
                if size == 1:
                    seg = seg.reshape(())
                splits.append(seg)
                start += size
            return splits
        raise ValueError(
            f"Aligned input expected shape ({self._total_in_size},), got {x_arr.shape}. "
            "Use vmap for batched inputs."
        )

    def _contraction_equation(self, batch_shapes: Sequence[tuple[int, ...]]) -> str:
        letters = string.ascii_lowercase.replace("l", "").replace("o", "")
        needed = sum(len(shape) for shape in batch_shapes)
        if needed > len(letters):
            raise ValueError("Too many batch axes to build contraction equation.")
        idx = 0
        terms = []
        out_terms = []
        for shape in batch_shapes:
            idxs = "".join(letters[idx : idx + len(shape)])
            idx += len(shape)
            terms.append(f"{idxs}lo")
            out_terms.append(idxs)
        return ",".join(terms) + "->" + "".join(out_terms) + "o"

    def _finalize(self, out: Array) -> Array:
        if jnp.iscomplexobj(out) and not self.keep_outputs_complex:
            out = jnp.real(out)
        out = self.output_activation(out)
        if self.out_size == "scalar":
            out = jnp.squeeze(out, axis=-1)
        return out

    def _split_key(self, key: Key[Array, ""] | None, /) -> Key[Array, " n_models"]:
        if key is None:
            key = DOC_KEY0
        return jr.split(key, len(self.factor_models))


class Separable(_AbstractStructuredInputModel):
    r"""Separable wrapper using pre-initialized scalar models per coordinate.

    Each coordinate model maps a scalar $x_i$ to `latent_size * out_size` features
    (reshaped to $(L,m)$). The wrapper multiplies per-coordinate features
    elementwise and sums over the latent axis:

    $$
    u_o(x)=\sum_{\ell=1}^{L}\prod_{i=1}^{d} g_{i,\ell,o}(x_i).
    $$

    Supports regular array inputs and separable tuple inputs (a tuple of 1D coordinate arrays).
    """

    in_size: int | Literal["scalar"]
    out_size: int | Literal["scalar"]

    latent_size: int
    models: tuple[_AbstractBaseModel, ...]
    output_activation: Callable
    keep_outputs_complex: bool
    _replicated_scalar_input: bool
    _clones: int
    _base_in_dim: int

    def __init__(
        self,
        *,
        in_size: int | Literal["scalar"],
        out_size: int | Literal["scalar"],
        latent_size: int,
        models: Sequence[_AbstractBaseModel],
        output_activation: Callable | None = None,
        keep_outputs_complex: bool = False,
        split_input: int | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ):
        r"""Create a separable wrapper.

        **Keyword arguments:**

        - `in_size`: Input dimension $d$ (or `"scalar"`).
        - `out_size`: Output size $m$ (or `"scalar"`).
        - `latent_size`: Rank $L$ in the separable expansion.
        - `models`: Sequence of scalar-input models, one per coordinate (and per
          `split_input` clone), each returning `latent_size * out_size` features.
        - `output_activation`: Optional activation applied after the contraction (wrap
          it yourself if you want adaptive behavior).
        - `split_input`: If provided and `in_size="scalar"`, replicates the scalar
          input across `split_input` coordinate models.
        """
        del key
        in_dim = _get_size(in_size)
        clones = (
            int(split_input) if (split_input is not None and int(split_input) > 1) else 1
        )
        expected_models = in_dim * clones
        if len(models) != expected_models:
            raise ValueError(
                "Number of coordinate models must equal in_size * split_input. "
                f"Got {len(models)} models but expected {expected_models}."
            )
        self.in_size = in_size
        self.out_size = out_size
        self.latent_size = latent_size
        self.models = tuple(models)
        self._replicated_scalar_input = in_dim == 1 and clones > 1
        self._clones = clones
        self._base_in_dim = in_dim
        self.output_activation = (
            _identity if output_activation is None else output_activation
        )
        self.keep_outputs_complex = bool(keep_outputs_complex)

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        r"""Evaluate the separable model.

        For vector inputs $x=(x_1,\dots,x_d)$ this computes

        $$
        u_o(x)=\sum_{\ell=1}^{L}\prod_{i=1}^{d} g_{i,\ell,o}(x_i),
        $$

        where $g_i$ are the per-coordinate scalar models (including any
        `split_input` replication).

        **Inputs:**

        - `x`: either a single point of shape `(d,)` (or scalar `()` in the
          replicated scalar-input case), or a separable tuple `(x_1,...,x_d)` of
          1D coordinate arrays.
        """
        # If the input is a tuple of arrays, dispatch to the dedicated paths.
        # - separable: a tuple of 1D coordinate arrays
        if isinstance(x, tuple):
            x_tuple: tuple[Array, ...] = tuple(jnp.asarray(xi) for xi in x)
            if x_tuple[0].ndim == 1:  # Separable
                if not all(xi.ndim == 1 for xi in x_tuple):
                    raise ValueError("Separable tuple input must contain only 1D arrays.")
                return self._call_separable(x_tuple, key=key, **kwargs)
            raise ValueError(
                f"Invalid input ndim for the first array: {x_tuple[0].ndim}. Expected 1."
            )

        keys = self._split_key(key)
        x_arr = jnp.asarray(x)
        clones = self._clones
        in_dim = self._base_in_dim

        # Scalar-input case: treat (N,) and (N,1) as a batch of scalars.
        if in_dim == 1:
            if x_arr.ndim == 0:
                out_list = []
                for idx in range(clones):
                    out_list.append(self.models[idx](x_arr, key=keys[idx]))
                outputs = ft.reduce(jnp.multiply, out_list, jnp.array(1.0))
                out = contract("lo->o", self._reshape_latents(outputs))
            else:
                if x_arr.ndim == 2 and x_arr.shape == (1, 1):
                    x_arr = jnp.squeeze(x_arr, axis=(0, 1))
                if x_arr.ndim == 1 and x_arr.shape[0] == 1:
                    x_arr = jnp.squeeze(x_arr, axis=0)
                if x_arr.ndim != 0:
                    raise ValueError(
                        f"Invalid input shape {x_arr.shape}. Expected scalar input. "
                        "Use vmap for batched inputs."
                    )
                out_list = []
                for idx in range(clones):
                    out_list.append(self.models[idx](x_arr, key=keys[idx]))
                outputs = ft.reduce(jnp.multiply, out_list, jnp.array(1.0))
                out = contract("lo->o", self._reshape_latents(outputs))

        # Standard (vector) input case: x is a single point (d,) or a batch (N,d).
        elif x_arr.ndim == 1:
            if x_arr.shape[0] != in_dim:
                raise ValueError(f"Expected shape ({in_dim},) got {x_arr.shape}.")
            out_list = []
            idx = 0
            for i in range(in_dim):
                xi = x_arr[i]
                for _ in range(clones):
                    out_list.append(self.models[idx](xi, key=keys[idx]))
                    idx += 1
            outputs = ft.reduce(jnp.multiply, out_list, jnp.array(1.0))
            out = contract("lo->o", self._reshape_latents(outputs))
        else:
            raise ValueError(
                f"Invalid input shape {x_arr.shape}. Expected ({in_dim},). "
                "Use vmap for batched inputs."
            )

        if jnp.iscomplexobj(out) and not self.keep_outputs_complex:
            out = jnp.real(out)
        out = self.output_activation(out)
        if self.out_size == "scalar":
            out = jnp.squeeze(out, axis=-1)
        return out

    def _reshape_latents(self, latents: Array, /) -> Array:
        if latents.ndim == 1:
            return latents.reshape(self.latent_size, _get_size(self.out_size))
        return latents.reshape(-1, self.latent_size, _get_size(self.out_size))

    def _call_separable(
        self,
        x: tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
        **kwargs: Any,
    ) -> Array:
        del kwargs
        if len(x) != self._base_in_dim:
            raise ValueError(
                "If passing a tuple of arrays to `Separable`, its length must equal "
                f"the base input dimension. Got {len(x)} and base in_size {self._base_in_dim}."
            )

        keys = self._split_key(key)
        clones = self._clones
        outputs_dim = []
        idx = 0
        for xi in x:
            clones_lat = []
            for _ in range(clones):
                out = jax.vmap(ft.partial(self.models[idx], key=keys[idx]))(xi)
                clones_lat.append(self._reshape_latents(out))
                idx += 1
            outputs_dim.append(ft.reduce(jnp.multiply, clones_lat))

        if len(outputs_dim) == 1:
            out = contract("nlo->no", outputs_dim[0])
        else:
            out = contract(_contract_str(len(outputs_dim)), *outputs_dim)

        if jnp.iscomplexobj(out) and not self.keep_outputs_complex:
            out = jnp.real(out)
        out = self.output_activation(out)
        if self.out_size == "scalar":
            out = jnp.squeeze(out, axis=-1)
        return out

    def _split_key(self, key: Key[Array, ""] | None, /) -> Key[Array, " n_models"]:
        if key is None:
            key = DOC_KEY0
        return jr.split(key, len(self.models))
