#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ._cochain import CochainBoundaryKind, CochainBoundaryPolicy, CochainComplexIR


class CochainField(StrictModule):
    """One degree-safe field over a canonical cochain complex."""

    complex: CochainComplexIR
    values: Array
    degree: int = eqx.field(static=True)
    boundary_policy: CochainBoundaryKind = eqx.field(static=True)
    field_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex: CochainComplexIR,
        values: ArrayLike,
        degree: int,
        /,
        *,
        boundary_policy: CochainBoundaryKind = "absolute",
        field_id: str,
    ):
        if not isinstance(complex, CochainComplexIR):
            raise TypeError("complex must be a CochainComplexIR.")
        degree_ = int(degree)
        if degree_ < 0 or degree_ > complex.max_degree:
            raise ValueError("Cochain field degree is outside the complex.")
        CochainBoundaryPolicy(boundary_policy)
        values_ = jnp.asarray(values)
        node_count = int(complex.graph.num_nodes)
        if values_.ndim == 0 or values_.shape[0] != node_count:
            raise ValueError(
                f"Cochain field values require leading graph-node size {node_count}."
            )
        identifier = str(field_id)
        if not identifier:
            raise ValueError("field_id must be non-empty.")
        self.complex = complex
        self.values = values_
        self.degree = degree_
        self.boundary_policy = boundary_policy
        self.field_id = identifier

    @property
    def active_values(self) -> Array:
        return self.values[self.complex.cell_entities(self.degree)]

    def exterior_derivative(self, /, *, field_id: str | None = None) -> CochainField:
        if self.degree >= self.complex.max_degree:
            raise ValueError("Exterior derivative of a top-degree cochain is zero.")
        active = self.complex.discretization.exterior_derivative(
            self.degree,
            self.active_values,
            boundary_policy=self.boundary_policy,
        )
        values = (
            jnp.zeros_like(self.values)
            .at[self.complex.cell_entities(self.degree + 1)]
            .set(active)
        )
        return CochainField(
            self.complex,
            values,
            self.degree + 1,
            boundary_policy=self.boundary_policy,
            field_id=field_id or f"d({self.field_id})",
        )

    def codifferential(self, /, *, field_id: str | None = None) -> CochainField:
        if self.degree <= 0:
            raise ValueError("Codifferential requires positive cochain degree.")
        active = self.complex.discretization.codifferential(
            self.degree,
            self.active_values,
            boundary_policy=self.boundary_policy,
        )
        values = (
            jnp.zeros_like(self.values)
            .at[self.complex.cell_entities(self.degree - 1)]
            .set(active)
        )
        return CochainField(
            self.complex,
            values,
            self.degree - 1,
            boundary_policy=self.boundary_policy,
            field_id=field_id or f"delta({self.field_id})",
        )

    def inner(self, other: CochainField, /) -> Array:
        self._require_compatible(other)
        star = self.complex.hodge_stars[self.degree]
        left = self.active_values
        right = other.active_values
        shape = (star.shape[0],) + (1,) * (left.ndim - 1)
        return jnp.real(jnp.sum(jnp.conj(left) * star.reshape(shape) * right))

    def norm_squared(self) -> Array:
        return self.inner(self)

    def scaled(
        self, scalar: ArrayLike, /, *, field_id: str | None = None
    ) -> CochainField:
        return CochainField(
            self.complex,
            jnp.asarray(scalar) * self.values,
            self.degree,
            boundary_policy=self.boundary_policy,
            field_id=field_id or f"scaled({self.field_id})",
        )

    def add(self, other: CochainField, /, *, field_id: str | None = None) -> CochainField:
        self._require_compatible(other)
        return CochainField(
            self.complex,
            self.values + other.values,
            self.degree,
            boundary_policy=self.boundary_policy,
            field_id=field_id or f"{self.field_id}+{other.field_id}",
        )

    def _require_compatible(self, other: CochainField, /) -> None:
        if not isinstance(other, CochainField):
            raise TypeError("other must be a CochainField.")
        if self.complex.fingerprint != other.complex.fingerprint:
            raise ValueError("Cochain fields belong to different metric complexes.")
        if self.degree != other.degree or self.boundary_policy != other.boundary_policy:
            raise ValueError("Cochain field degrees and boundary policies must match.")
        if self.values.shape != other.values.shape:
            raise ValueError("Cochain field value shapes must match.")


__all__ = ["CochainField"]
