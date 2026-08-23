#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ._projective import ComplexProjectiveAtlas


class ProjectiveHypersurface(StrictModule):
    """One homogeneous polynomial hypersurface in complex projective space."""

    polynomial: Callable[[Array], Array]
    atlas: ComplexProjectiveAtlas
    projective_dimension: int = eqx.field(static=True)
    degree: int = eqx.field(static=True)
    hypersurface_id: str = eqx.field(static=True)

    def __init__(
        self,
        polynomial: Callable[[Array], Array],
        projective_dimension: int,
        degree: int,
        /,
        *,
        hypersurface_id: str,
    ):
        if not callable(polynomial):
            raise TypeError("polynomial must be callable.")
        dimension = int(projective_dimension)
        degree_ = int(degree)
        if dimension < 1 or degree_ < 1:
            raise ValueError(
                "Projective dimension and polynomial degree must be positive."
            )
        identifier = str(hypersurface_id)
        if not identifier:
            raise ValueError("hypersurface_id must be non-empty.")
        self.polynomial = polynomial
        self.atlas = ComplexProjectiveAtlas(dimension)
        self.projective_dimension = dimension
        self.degree = degree_
        self.hypersurface_id = identifier

    @property
    def complex_dimension(self) -> int:
        return self.projective_dimension - 1

    @property
    def calabi_yau_degree(self) -> bool:
        return self.degree == self.projective_dimension + 1

    def homogeneous_coordinates(
        self,
        chart_index: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        local = self.atlas.conventions[int(chart_index)].to_complex(coordinates)
        homogeneous = jnp.ones(
            local.shape[:-1] + (self.projective_dimension + 1,), dtype=local.dtype
        )
        axes = tuple(
            index
            for index in range(self.projective_dimension + 1)
            if index != int(chart_index)
        )
        return homogeneous.at[..., jnp.asarray(axes)].set(local)

    def local_polynomial(
        self,
        chart_index: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        homogeneous = self.homogeneous_coordinates(chart_index, coordinates)
        if homogeneous.ndim == 1:
            value = jnp.asarray(self.polynomial(homogeneous))
        else:
            value = jax.vmap(self.polynomial)(homogeneous)
        if value.shape != homogeneous.shape[:-1]:
            raise ValueError("Projective polynomial must be scalar-valued.")
        return value

    def local_smoothness_margin(
        self,
        chart_index: int,
        coordinates: ArrayLike,
        /,
    ) -> Array:
        points = jnp.asarray(coordinates)
        convention = self.atlas.conventions[int(chart_index)]

        def local(point: Array) -> Array:
            return self.local_polynomial(chart_index, point)

        if points.ndim == 1:
            jacobian = jax.jacfwd(local)(points)
            return jnp.linalg.norm(jacobian)
        flat = points.reshape((-1, convention.chart.dimension))
        margins = jax.vmap(lambda point: jnp.linalg.norm(jax.jacfwd(local)(point)))(flat)
        return margins.reshape(points.shape[:-1])

    def residual(self, chart_index: int, coordinates: ArrayLike, /) -> Array:
        return jnp.abs(self.local_polynomial(chart_index, coordinates))


def fermat_hypersurface(projective_dimension: int, /) -> ProjectiveHypersurface:
    """Return ``sum Z_i^(N+1)=0`` in CP^N."""
    dimension = int(projective_dimension)
    degree = dimension + 1
    return ProjectiveHypersurface(
        lambda homogeneous: jnp.sum(homogeneous**degree),
        dimension,
        degree,
        hypersurface_id=f"fermat:{degree}:CP{dimension}",
    )


__all__ = ["ProjectiveHypersurface", "fermat_hypersurface"]
