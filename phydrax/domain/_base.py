#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import functools as ft
from abc import abstractmethod
from collections.abc import Callable, Sequence
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Key

from .._doc import DOC_KEY0
from .._strict import AbstractAttribute
from ._domain import _AbstractUnaryDomain


class _AbstractGeometry(_AbstractUnaryDomain):
    """Abstract (spatial) geometry."""

    adf: AbstractAttribute[Callable[[Array], Array]]

    @property
    @abstractmethod
    def volume(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def spatial_dim(self) -> int:
        raise NotImplementedError

    @property
    def dim(self) -> int:
        return self.spatial_dim

    @property
    def label(self) -> str:
        return "x"

    @property
    def var_dim(self) -> int:
        return int(self.spatial_dim)

    @property
    @abstractmethod
    def bounds(self) -> Float[Array, "2 spatial_dim"]:
        raise NotImplementedError

    @property
    def time(self) -> Literal[False]:
        return False

    @ft.cached_property
    def mesh_bounds(self) -> Float[Array, "2 spatial_dim"]:
        """Axis-aligned bounding box as `[[mins...], [maxs...]]` (raw values)."""
        bounds = jnp.asarray(self.bounds, dtype=float)
        sd = int(self.spatial_dim)
        if bounds.shape != (2, sd):
            raise ValueError(
                f"{type(self).__name__}.bounds must have shape (2, {sd}), got {bounds.shape}."
            )
        return bounds

    @ft.cached_property
    def volume_proportion(self) -> Float[Array, ""]:
        """Fraction of the AABB volume occupied by the geometry (defaults to 1.0)."""
        return jnp.array(1.0, dtype=float)

    @property
    def boundary_measure_value(self) -> Array:
        """Total boundary measure value (boundary length / surface area).

        Concrete geometries should override this where applicable. For 1D geometries,
        this defaults to counting measure on the two endpoints (value = 2).
        """
        if self.spatial_dim == 1:
            return jnp.array(2.0, dtype=float)
        raise NotImplementedError(
            f"{type(self).__name__} must implement `boundary_measure_value`."
        )

    @abstractmethod
    def estimate_boundary_subset_measure(
        self,
        where: Callable[[Array], Bool[Array, ""]],
        *,
        num_samples: int = 4096,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        """Estimate boundary subset measure of {x: where(x)=True}."""
        raise NotImplementedError

    def _check_points_on_boundary(self, points: Array) -> Array:
        return eqx.error_if(
            points,
            pred=~self._on_boundary(points),
            msg="All points must be on the boundary of the domain.",
        )

    @abstractmethod
    def sample_interior(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def sample_boundary(
        self,
        num_points: int,
        *,
        where: Callable | None = None,
        sampler: str = "latin_hypercube",
        key: Key[Array, ""] = DOC_KEY0,
    ) -> Array:
        raise NotImplementedError

    @abstractmethod
    def _sample_interior_separable(
        self,
        num_points: int | Sequence[int],
        *,
        sampler: str = "latin_hypercube",
        where: Callable | None = None,
        key: Key[Array, ""] = DOC_KEY0,
    ) -> tuple[tuple[Array, ...], Bool[Array, "..."]]:
        """Internal helper for separable interior sampling."""
        raise NotImplementedError

    @abstractmethod
    def _contains(self, points: Array) -> Bool[Array, " num_points"]:
        raise NotImplementedError

    @abstractmethod
    def _on_boundary(self, points: Array) -> Bool[Array, " num_points"]:
        raise NotImplementedError

    @abstractmethod
    def _boundary_normals(self, points: Array) -> Float[Array, "num_points spatial_dim"]:
        raise NotImplementedError
