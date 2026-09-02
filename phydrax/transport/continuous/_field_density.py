#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from ..._fingerprint import canonical_fingerprint
from ..._probability import AbstractProbabilityLaw
from ..._strict import StrictModule
from ...stochastic._path_diffusion import TrajectoryEventLayout


class HybridFlowSample(StrictModule):
    mode: Array
    value: Array


class HybridFlowLaw(StrictModule):
    """Finite counting mass times normalized conditional continuous laws."""

    mode_probabilities: Array
    conditional_laws: tuple[AbstractProbabilityLaw, ...]
    mode_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)

    def __init__(
        self,
        mode_probabilities: ArrayLike,
        conditional_laws: tuple[AbstractProbabilityLaw, ...],
        /,
        *,
        mode_id: str,
    ):
        probabilities = jnp.asarray(mode_probabilities, dtype=float)
        laws = tuple(conditional_laws)
        if probabilities.ndim != 1 or probabilities.size == 0:
            raise ValueError("mode_probabilities must be nonempty and rank one.")
        if len(laws) != probabilities.size or any(
            not isinstance(item, AbstractProbabilityLaw) for item in laws
        ):
            raise TypeError("conditional_laws must align mode probabilities.")
        if len({item.event_shape for item in laws}) != 1:
            raise ValueError("all conditional laws must share one event shape.")
        if not bool(
            jnp.all(jnp.isfinite(probabilities) & (probabilities > 0.0))
        ) or not bool(jnp.isclose(jnp.sum(probabilities), 1.0)):
            raise ValueError(
                "mode probabilities must be finite, positive, and normalized."
            )
        if not mode_id:
            raise ValueError("mode_id must be non-empty.")
        self.mode_probabilities = probabilities
        self.conditional_laws = laws
        self.mode_id = mode_id
        self.reference_measure = "counting-product"

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.conditional_laws[0].event_shape

    def sample(
        self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()
    ) -> HybridFlowSample:
        shape = tuple(int(size) for size in sample_shape)
        mode_key, value_key = jr.split(key)
        modes = jr.categorical(mode_key, jnp.log(self.mode_probabilities), shape=shape)
        keys = jr.split(value_key, max(prod(shape), 1))
        flat_modes = modes.reshape((-1,))

        def one(mode, sample_key):
            branches = tuple(
                lambda law=law: law.sample(sample_key) for law in self.conditional_laws
            )
            return jax.lax.switch(mode, branches)

        values = jax.vmap(one)(flat_modes, keys).reshape(shape + self.event_shape)
        return HybridFlowSample(mode=modes, value=values)

    def log_prob(self, mode: ArrayLike, value: ArrayLike, /) -> Array:
        modes = jnp.asarray(mode, dtype=jnp.int32)
        values = jnp.asarray(value)
        leading = values.shape[: -len(self.event_shape)]
        if modes.shape != leading:
            raise ValueError("mode must align value leading sample axes.")
        flat_values = values.reshape((-1,) + self.event_shape)
        flat_modes = modes.reshape((-1,))

        def one(selected_mode, item):
            branches = tuple(
                lambda law=law: law.log_prob(item) for law in self.conditional_laws
            )
            return jnp.log(self.mode_probabilities[selected_mode]) + jax.lax.switch(
                selected_mode, branches
            )

        return jax.vmap(one)(flat_modes, flat_values).reshape(leading)


class TrajectoryFlowLaw(AbstractProbabilityLaw):
    """Density on finite trajectory coefficients, not path-space RN density."""

    coefficient_law: AbstractProbabilityLaw
    layout: TrajectoryEventLayout
    support_tolerance: Array
    law_id: str = eqx.field(static=True)

    def __init__(
        self,
        coefficient_law: AbstractProbabilityLaw,
        layout: TrajectoryEventLayout,
        /,
        *,
        support_tolerance: float = 1.0e-8,
        law_id: str | None = None,
    ):
        if not isinstance(coefficient_law, AbstractProbabilityLaw) or not isinstance(
            layout, TrajectoryEventLayout
        ):
            raise TypeError("coefficient_law and layout have incompatible types.")
        if coefficient_law.event_shape != (layout.coefficient_layout.rank,):
            raise ValueError("coefficient law event shape must equal layout rank.")
        self.coefficient_law = coefficient_law
        self.layout = layout
        self.support_tolerance = jnp.asarray(support_tolerance)
        self.law_id = law_id or canonical_fingerprint(
            {
                "kind": "finite-trajectory-flow-law-v1",
                "layout": layout.layout_id,
                "coefficient_shape": coefficient_law.event_shape,
            }
        )

    @property
    def event_shape(self) -> tuple[int, ...]:
        return self.layout.coefficient_layout.event_shape

    @property
    def batch_shape(self) -> tuple[int, ...]:
        return self.coefficient_law.batch_shape

    @property
    def density_measure_kind(self) -> str:
        return "trajectory"

    def sample(self, key: Key[Array, ""], sample_shape: tuple[int, ...] = ()) -> Array:
        return self.layout.synthesize(self.coefficient_law.sample(key, sample_shape))

    def log_prob(self, value: ArrayLike, /) -> Array:
        coefficients, residual = self.layout.coefficients(value)
        density = (
            self.coefficient_law.log_prob(coefficients)
            - self.layout.coefficient_layout.log_volume
        )
        return jnp.where(residual <= self.support_tolerance, density, -jnp.inf)

    def contains(self, value: ArrayLike, /) -> Array:
        coefficients, residual = self.layout.coefficients(value)
        return (residual <= self.support_tolerance) & self.coefficient_law.contains(
            coefficients
        )


class PreparedFieldQuery(StrictModule):
    query_points: Array
    mask: Array
    capacity: int = eqx.field(static=True)
    query_id: str = eqx.field(static=True)


class FiniteFieldSample(StrictModule):
    coefficients: Array
    values: Array
    query_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)


class FiniteFieldFlowLaw(StrictModule):
    """Normalized coefficient law with query-independent probability semantics."""

    coefficient_law: AbstractProbabilityLaw
    decoder: Any
    query_evidence: Any
    field_space_id: str = eqx.field(static=True)
    law_id: str = eqx.field(static=True)
    reference_measure: str = eqx.field(static=True)

    def __init__(
        self,
        coefficient_law: AbstractProbabilityLaw,
        decoder: Any,
        /,
        *,
        field_space_id: str,
        query_evidence: Any = None,
        law_id: str | None = None,
    ):
        if not isinstance(coefficient_law, AbstractProbabilityLaw):
            raise TypeError("coefficient_law must be an AbstractProbabilityLaw.")
        if not callable(decoder):
            raise TypeError(
                "decoder must be callable as decoder(coefficients, query_points)."
            )
        if query_evidence is not None and not callable(query_evidence):
            raise TypeError("query_evidence must be callable or None.")
        if not field_space_id:
            raise ValueError("field_space_id must be non-empty.")
        self.coefficient_law = coefficient_law
        self.decoder = decoder
        self.query_evidence = query_evidence
        self.field_space_id = field_space_id
        self.law_id = law_id or canonical_fingerprint(
            {
                "kind": "finite-field-flow-law-v1",
                "field_space": field_space_id,
                "coefficient_shape": coefficient_law.event_shape,
            }
        )
        self.reference_measure = "coefficient-space"

    def coefficient_log_prob(self, coefficients: ArrayLike, /) -> Array:
        return self.coefficient_law.log_prob(coefficients)

    def sample_field(
        self,
        key: Key[Array, ""],
        query: PreparedFieldQuery,
        sample_shape: tuple[int, ...] = (),
    ) -> FiniteFieldSample:
        if not isinstance(query, PreparedFieldQuery):
            raise TypeError("query must be a PreparedFieldQuery.")
        coefficients = self.coefficient_law.sample(key, sample_shape)
        values = self.decoder(coefficients, query.query_points)
        query_mask = query.mask.reshape(
            (1,) * len(sample_shape)
            + query.mask.shape
            + (1,) * (values.ndim - len(sample_shape) - 1)
        )
        values = jnp.where(query_mask, values, 0.0)
        return FiniteFieldSample(
            coefficients=coefficients,
            values=values,
            query_id=query.query_id,
            law_id=self.law_id,
        )

    def decoded_log_prob(
        self,
        coefficients: ArrayLike,
        query: PreparedFieldQuery,
        /,
    ) -> Array:
        """Return injective-query Hausdorff density only with explicit evidence."""
        if self.query_evidence is None:
            raise ValueError("Query density requires explicit injective map evidence.")
        evidence = self.query_evidence(query.query_points, query.mask)
        if not bool(evidence.valid) or not bool(evidence.rank_margin > 0.0):
            raise ValueError("Query decoder map evidence is invalid or rank deficient.")
        return self.coefficient_law.log_prob(coefficients) - jnp.asarray(
            evidence.log_volume
        )


class ConditionalFiniteFieldFlowLaw(StrictModule):
    """Fixed-context conditional coefficient law and arbitrary-query decoder."""

    source_encoder: Any
    conditional_coefficient_law: Any
    decoder: Any
    field_space_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_encoder: Any,
        conditional_coefficient_law: Any,
        decoder: Any,
        /,
        *,
        field_space_id: str,
    ):
        if (
            not callable(source_encoder)
            or not callable(conditional_coefficient_law)
            or not callable(decoder)
        ):
            raise TypeError("conditional field components must be callable.")
        self.source_encoder = source_encoder
        self.conditional_coefficient_law = conditional_coefficient_law
        self.decoder = decoder
        self.field_space_id = field_space_id

    def condition(self, source: Any, /) -> FiniteFieldFlowLaw:
        context = self.source_encoder(source)
        law = self.conditional_coefficient_law(context)
        return FiniteFieldFlowLaw(
            law,
            self.decoder,
            field_space_id=self.field_space_id,
            law_id=f"conditional-field:{self.field_space_id}",
        )


def prepare_field_query(
    law: FiniteFieldFlowLaw,
    query_points: ArrayLike,
    /,
    *,
    capacity: int,
    mask: ArrayLike | None = None,
) -> PreparedFieldQuery:
    """Freeze decoding topology without changing coefficient normalization."""
    if not isinstance(law, FiniteFieldFlowLaw):
        raise TypeError("law must be a FiniteFieldFlowLaw.")
    points = jnp.asarray(query_points)
    maximum = int(capacity)
    if points.ndim != 2 or points.shape[0] != maximum or maximum <= 0:
        raise ValueError("query_points must have shape (capacity, coordinate_dimension).")
    active = (
        jnp.ones((maximum,), dtype=bool)
        if mask is None
        else jnp.asarray(mask, dtype=bool)
    )
    if active.shape != (maximum,) or not bool(jnp.any(active)):
        raise ValueError("query mask must align capacity and contain an active point.")
    query_id = canonical_fingerprint(
        {
            "kind": "prepared-field-query-v1",
            "law": law.law_id,
            "capacity": maximum,
            "coordinate_dimension": int(points.shape[1]),
            "active": int(jnp.sum(active)),
        }
    )
    return PreparedFieldQuery(
        query_points=points,
        mask=active,
        capacity=maximum,
        query_id=query_id,
    )


__all__ = [
    "ConditionalFiniteFieldFlowLaw",
    "FiniteFieldFlowLaw",
    "FiniteFieldSample",
    "HybridFlowLaw",
    "HybridFlowSample",
    "PreparedFieldQuery",
    "TrajectoryFlowLaw",
    "prepare_field_query",
]
