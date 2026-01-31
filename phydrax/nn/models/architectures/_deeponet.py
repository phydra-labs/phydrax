#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import opt_einsum as oe
from jaxtyping import Array, Key

from ...._doc import DOC_KEY0
from ....domain._grid import broadcasted_grid
from ..._utils import _get_size
from ..core._base import _AbstractBaseModel, _AbstractStructuredInputModel


class DeepONet(_AbstractStructuredInputModel):
    r"""Minimal Deep Operator Network (DeepONet) for structured inputs.

    DeepONet represents an operator \(G\) by a low-rank expansion

    $$
    (Gf)(x)\;\approx\;\sum_{k=1}^p b_k(f)\,t_k(x),
    $$

    where the **branch** network maps the input function/data \(f\) to coefficients
    \(b(f)\in\mathbb{R}^p\) and the **trunk** network maps coordinates \(x\) to basis
    values \(t(x)\in\mathbb{R}^p\) (optionally replicated for vector outputs).

    Expects a tuple input `(branch_input, coords...)`. For point inputs, `coords`
    may be a single array with trailing dimension `coord_dim`. For coord-separable
    grid inputs, `coords` should be `coord_dim` separate 1D axis arrays.
    """

    branch: _AbstractBaseModel
    trunk: _AbstractBaseModel
    latent_size: int
    coord_dim: int
    out_size: int | Literal["scalar"]
    in_size: int | Literal["scalar"]

    def __init__(
        self,
        *,
        branch: _AbstractBaseModel,
        trunk: _AbstractBaseModel,
        coord_dim: int,
        latent_size: int,
        out_size: int | Literal["scalar"] = "scalar",
        in_size: int | Literal["scalar"] = "scalar",
    ):
        self.branch = branch
        self.trunk = trunk
        self.coord_dim = int(coord_dim)
        self.latent_size = int(latent_size)
        self.out_size = out_size
        self.in_size = in_size

        if self.coord_dim <= 0:
            raise ValueError("coord_dim must be positive.")
        if self.latent_size <= 0:
            raise ValueError("latent_size must be positive.")

        if _get_size(self.branch.out_size) != self.latent_size:
            raise ValueError(
                "branch.out_size must match latent_size; got "
                f"{self.branch.out_size!r} and {self.latent_size}."
            )

        expected_trunk_out = self.latent_size * _get_size(self.out_size)
        if _get_size(self.trunk.out_size) != expected_trunk_out:
            raise ValueError(
                "trunk.out_size must be latent_size*out_size; got "
                f"{self.trunk.out_size!r} but expected {expected_trunk_out}."
            )

        if self.coord_dim == 1:
            if _get_size(self.trunk.in_size) != 1:
                raise ValueError(
                    "trunk.in_size must be scalar/1 for coord_dim=1; got "
                    f"{self.trunk.in_size!r}."
                )
        else:
            if _get_size(self.trunk.in_size) != self.coord_dim:
                raise ValueError(
                    "trunk.in_size must match coord_dim; got "
                    f"{self.trunk.in_size!r} and {self.coord_dim}."
                )

    def __call__(
        self,
        x: Array | tuple[Array, ...],
        /,
        *,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        if not isinstance(x, tuple):
            raise ValueError("DeepONet requires a tuple input (branch_input, coords...).")
        if len(x) < 2:
            raise ValueError("DeepONet requires at least (branch_input, coords...).")

        k_branch, k_trunk = jr.split(key, 2)

        branch_in = jnp.asarray(x[0])
        branch_vec = branch_in.reshape((-1,))
        b = jnp.asarray(self.branch(branch_vec, key=k_branch)).reshape(
            (self.latent_size,)
        )

        coords = tuple(jnp.asarray(c) for c in x[1:])
        if len(coords) == self.coord_dim:
            grid = broadcasted_grid(coords)
            pts = grid.reshape((-1, self.coord_dim))
            leading_shape = grid.shape[:-1]
        elif len(coords) == 1:
            pts0 = jnp.asarray(coords[0])
            if pts0.shape[-1] != self.coord_dim:
                raise ValueError(
                    "DeepONet point-coordinate input must have trailing dimension "
                    f"{self.coord_dim}, got shape {pts0.shape}."
                )
            pts = pts0.reshape((-1, self.coord_dim))
            leading_shape = pts0.shape[:-1]
        else:
            raise ValueError(
                "DeepONet coordinate inputs must be either coord_dim separate 1D axes "
                "or a single array with trailing coord_dim."
            )

        def _trunk_eval(z: Array) -> Array:
            return self.trunk(z, key=k_trunk)

        t_flat = jax.vmap(_trunk_eval)(pts)
        t = jnp.asarray(t_flat).reshape(
            (pts.shape[0], _get_size(self.out_size), self.latent_size)
        )
        y_flat = oe.contract("k,nok->no", b, t)
        y = y_flat.reshape(leading_shape + (_get_size(self.out_size),))
        if self.out_size == "scalar":
            return y[..., 0]
        return y


__all__ = ["DeepONet"]
