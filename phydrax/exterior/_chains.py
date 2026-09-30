#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from abc import abstractmethod
from typing import final, Literal

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from ..typing import Bool, Dim, Identifier, Inexact, Integer, Size, VariadicDim


type SegmentWeight = Literal["uniform", "phase"]


class ChainPointDim(Dim):
    """Number of independently queried chains."""


class ChainSlotDim(Dim):
    """Fixed route capacity of each chain."""


class ChainDofDim(Dim):
    """Coefficient extent of the admitted form degree."""


class ChainValueDims(VariadicDim):
    """Exterior components returned by a point reconstruction."""


@final
class PreparedChainQuery(StrictModule):
    """Fixed-slot chain routes and their Euclidean gather/deposit dual.

    Coefficients have shape ``(queries, slots, *value_shape)``. Invalid slots
    contribute exactly zero, including when their sentinel index is negative.
    Phase-weighted deposits use the conjugate transpose, not the transpose.
    Failure and overflow remain explicit: consumers must admit their queries.
    """

    __strict_contract__ = True

    indices: Integer[ChainPointDim, ChainSlotDim]
    coefficients: Inexact[ChainPointDim, ChainSlotDim, ChainValueDims]
    valid: Bool[ChainPointDim, ChainSlotDim]
    successful: Bool[ChainPointDim]
    overflow: Bool[ChainPointDim]
    dof_count: Size[ChainDofDim] = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    kernel_id: Identifier = eqx.field(static=True)

    def __init__(
        self,
        indices: ArrayLike,
        coefficients: ArrayLike,
        valid: ArrayLike,
        successful: ArrayLike,
        overflow: ArrayLike,
        /,
        *,
        dof_count: int,
        degree: int,
        kernel_id: str,
    ) -> None:
        routes = jnp.asarray(indices)
        weights = jnp.asarray(coefficients)
        active = jnp.asarray(valid)
        success = jnp.asarray(successful)
        exceeded = jnp.asarray(overflow)
        if routes.ndim != 2 or not jnp.issubdtype(routes.dtype, jnp.integer):
            raise ValueError("Chain indices must be a rank-two integer array.")
        if weights.shape[:2] != routes.shape or weights.ndim < 2:
            raise ValueError("Chain coefficients must preserve the query/slot axes.")
        if active.shape != routes.shape or active.dtype != jnp.bool_:
            raise ValueError("Chain valid slots must be a matching boolean array.")
        if success.shape != (routes.shape[0],) or success.dtype != jnp.bool_:
            raise ValueError("Chain successful must be boolean per query.")
        if exceeded.shape != success.shape or exceeded.dtype != jnp.bool_:
            raise ValueError("Chain overflow must be boolean per query.")
        if dof_count < 1 or degree < 0 or not kernel_id:
            raise ValueError(
                "Chain metadata requires positive size and explicit identity."
            )
        self.indices = routes
        self.coefficients = weights
        self.valid = active
        coefficient_axes = tuple(range(2, weights.ndim))
        finite = jnp.isfinite(weights)
        if coefficient_axes:
            finite = jnp.all(finite, axis=coefficient_axes)
        admitted = jnp.all(
            ~active | ((routes >= 0) & (routes < dof_count) & finite), axis=1
        )
        self.successful = success & admitted & ~exceeded
        self.overflow = exceeded
        self.dof_count = dof_count
        self.degree = degree
        self.kernel_id = kernel_id

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.coefficients.shape[2:]

    def gather(self, cochain: ArrayLike, /) -> Array:
        """Evaluate a scalar-coefficient cochain on the prepared chains."""
        values = jnp.asarray(cochain)
        if values.shape != (self.dof_count,):
            raise ValueError("Cochain shape does not match the chain degree.")
        active = self.valid & (self.indices >= 0) & (self.indices < self.dof_count)
        indices = jnp.clip(self.indices, 0, self.dof_count - 1)
        expand = (1,) * len(self.value_shape)
        coefficients = jnp.where(
            active.reshape((*active.shape, *expand)), self.coefficients, 0
        )
        selected = jnp.where(active, values[indices], 0).reshape(
            (*indices.shape, *expand)
        )
        return jnp.sum(coefficients * selected, axis=1)

    def deposit(self, values: ArrayLike, /) -> Array:
        """Apply the exact conjugate adjoint of :meth:`gather`."""
        weights = jnp.asarray(values)
        if weights.shape != (self.indices.shape[0], *self.value_shape):
            raise ValueError("Deposit values do not match the chain query value shape.")
        active = self.valid & (self.indices >= 0) & (self.indices < self.dof_count)
        expand = (1,) * len(self.value_shape)
        coefficients = jnp.where(
            active.reshape((*active.shape, *expand)), self.coefficients, 0
        )
        products = jnp.conj(coefficients) * weights[:, None]
        axes = tuple(range(2, products.ndim))
        scalar_products = jnp.sum(products, axis=axes) if axes else products
        contributions = jnp.where(active, scalar_products, 0)
        indices = jnp.clip(self.indices, 0, self.dof_count - 1)
        return (
            jnp.zeros((self.dof_count,), dtype=contributions.dtype)
            .at[indices.reshape((-1,))]
            .add(contributions.reshape((-1,)))
        )


class AbstractChainIntegrationKernel(StrictModule):
    """A prepared reconstruction whose chain integration commutes with d.

    ``phase_rate`` is dimensionless: a segment parameterized by
    ``start + t * (end - start)`` is weighted by ``exp(+1j * phase_rate * t)``.
    Physical Fourier phases supply the initial phase separately.
    """

    dof_counts: eqx.AbstractVar[tuple[int, ...]]
    dof_offsets: eqx.AbstractVar[tuple[tuple[int, ...], ...]]
    kernel_id: eqx.AbstractVar[str]

    @abstractmethod
    def integrate_points(self, points: ArrayLike, /) -> PreparedChainQuery:
        raise NotImplementedError

    @abstractmethod
    def integrate_segments(
        self,
        start: ArrayLike,
        end: ArrayLike,
        /,
        *,
        weight: SegmentWeight = "uniform",
        phase_rate: ArrayLike | None = None,
        maximum_segments: int | None = None,
    ) -> PreparedChainQuery:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, points: ArrayLike, degree: int, /) -> PreparedChainQuery:
        raise NotImplementedError


__all__ = ["AbstractChainIntegrationKernel", "PreparedChainQuery", "SegmentWeight"]
