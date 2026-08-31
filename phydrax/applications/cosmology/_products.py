#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._scales import CosmologyScaleContract


CosmologyProductSource = Literal["native", "external"]
CosmologyDifferentiability = Literal["native-parameter", "coordinate-only", "constant"]


class CosmologyProductProvenance(StrictModule, NonTrainableState):
    """Static producer, model, numerical-policy, and scale identity."""

    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    numerical_policy_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    source_kind: CosmologyProductSource = eqx.field(static=True)
    differentiability: CosmologyDifferentiability = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        producer: str,
        producer_version: str,
        model_id: str,
        numerical_policy_id: str,
        scale_id: str,
        source_kind: CosmologyProductSource,
        differentiability: CosmologyDifferentiability,
    ):
        values = tuple(
            str(value).strip()
            for value in (
                producer,
                producer_version,
                model_id,
                numerical_policy_id,
                scale_id,
            )
        )
        if any(not value for value in values):
            raise ValueError("Cosmology product provenance fields must be non-empty.")
        if source_kind not in ("native", "external"):
            raise ValueError("source_kind must be 'native' or 'external'.")
        if differentiability not in (
            "native-parameter",
            "coordinate-only",
            "constant",
        ):
            raise ValueError("Unknown cosmology product differentiability contract.")
        (
            self.producer,
            self.producer_version,
            self.model_id,
            self.numerical_policy_id,
            self.scale_id,
        ) = values
        self.source_kind = source_kind
        self.differentiability = differentiability
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "cosmology-product-provenance",
                "producer": values[0],
                "producer_version": values[1],
                "model_id": values[2],
                "numerical_policy_id": values[3],
                "scale_id": values[4],
                "source_kind": source_kind,
                "differentiability": differentiability,
            }
        )


def _validated_nodes(values: ArrayLike, name: str, /) -> Array:
    nodes = jnp.asarray(values).reshape((-1,))
    if nodes.size < 2:
        raise ValueError(f"{name} requires at least two nodes.")
    return eqx.error_if(
        nodes,
        jnp.any(~jnp.isfinite(nodes)) | jnp.any(jnp.diff(nodes) <= 0.0),
        f"{name} nodes must be finite and strictly increasing.",
    )


def _validated_query(values: ArrayLike, nodes: Array, name: str, /) -> Array:
    query = jnp.asarray(values, dtype=nodes.dtype)
    return eqx.error_if(
        query,
        jnp.any(~jnp.isfinite(query))
        | jnp.any(query < nodes[0])
        | jnp.any(query > nodes[-1]),
        f"{name} query is outside the tabulated domain.",
    )


def _validate_common(
    scale: CosmologyScaleContract,
    provenance: CosmologyProductProvenance,
    /,
) -> None:
    if not isinstance(scale, CosmologyScaleContract):
        raise TypeError("scale must be a CosmologyScaleContract.")
    if not isinstance(provenance, CosmologyProductProvenance):
        raise TypeError("provenance must be CosmologyProductProvenance.")
    if scale.scale_id != provenance.scale_id:
        raise ValueError("Cosmology product scale and provenance disagree.")


class ExpansionHistory(StrictModule):
    """Tabulated Hubble expansion with explicit coordinates and provenance."""

    scale_factors: Array
    hubble_values: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance

    def __init__(
        self,
        scale_factors: ArrayLike,
        hubble_values: ArrayLike,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        /,
    ):
        _validate_common(scale, provenance)
        nodes = _validated_nodes(scale_factors, "ExpansionHistory")
        hubble = jnp.asarray(hubble_values, dtype=nodes.dtype)
        if hubble.shape != nodes.shape:
            raise ValueError("ExpansionHistory values must match scale-factor nodes.")
        hubble = eqx.error_if(
            hubble,
            jnp.any(~jnp.isfinite(hubble)) | jnp.any(hubble <= 0.0),
            "ExpansionHistory Hubble values must be finite and positive.",
        )
        self.scale_factors = nodes
        self.hubble_values = hubble
        self.scale = scale
        self.provenance = provenance

    def hubble(self, scale_factor: ArrayLike, /) -> Array:
        query = _validated_query(scale_factor, self.scale_factors, "ExpansionHistory")
        return jnp.interp(query, self.scale_factors, self.hubble_values)


class LagrangianGrowthHistory(StrictModule):
    """First- and second-order Lagrangian growth and logarithmic rates."""

    scale_factors: Array
    first_order_growth: Array
    first_order_rate: Array
    second_order_growth: Array
    second_order_rate: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance

    def __init__(
        self,
        scale_factors: ArrayLike,
        first_order_growth: ArrayLike,
        first_order_rate: ArrayLike,
        second_order_growth: ArrayLike,
        second_order_rate: ArrayLike,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        /,
    ):
        _validate_common(scale, provenance)
        nodes = _validated_nodes(scale_factors, "LagrangianGrowthHistory")
        values = tuple(
            jnp.asarray(value, dtype=nodes.dtype)
            for value in (
                first_order_growth,
                first_order_rate,
                second_order_growth,
                second_order_rate,
            )
        )
        if any(value.shape != nodes.shape for value in values):
            raise ValueError("Lagrangian growth arrays must match scale-factor nodes.")
        stacked = jnp.stack(values)
        stacked = eqx.error_if(
            stacked,
            jnp.any(~jnp.isfinite(stacked))
            | jnp.any(values[0] <= 0.0)
            | jnp.any(values[2] <= 0.0),
            "Lagrangian growth values must be finite with positive D1 and D2.",
        )
        (
            self.first_order_growth,
            self.first_order_rate,
            self.second_order_growth,
            self.second_order_rate,
        ) = tuple(stacked[index] for index in range(4))
        self.scale_factors = nodes
        self.scale = scale
        self.provenance = provenance

    def evaluate(self, scale_factor: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        query = _validated_query(
            scale_factor, self.scale_factors, "LagrangianGrowthHistory"
        )
        return tuple(
            jnp.interp(query, self.scale_factors, values)
            for values in (
                self.first_order_growth,
                self.first_order_rate,
                self.second_order_growth,
                self.second_order_rate,
            )
        )


class MatterPowerTable(StrictModule):
    """Tabulated linear density power P_delta_delta(k, a)."""

    scale_factors: Array
    wavenumbers: Array
    power_values: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    spatial_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        scale_factors: ArrayLike,
        wavenumbers: ArrayLike,
        power_values: ArrayLike,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        /,
        *,
        spatial_dimension: int = 3,
    ):
        _validate_common(scale, provenance)
        scales = _validated_nodes(scale_factors, "MatterPowerTable scale factor")
        wavenumber = _validated_nodes(wavenumbers, "MatterPowerTable wavenumber")
        wavenumber = eqx.error_if(
            wavenumber,
            jnp.any(wavenumber <= 0.0),
            "MatterPowerTable wavenumbers must be positive.",
        )
        power = jnp.asarray(power_values, dtype=scales.dtype)
        expected = (scales.size, wavenumber.size)
        if power.shape != expected:
            raise ValueError(f"MatterPowerTable power_values must have shape {expected}.")
        power = eqx.error_if(
            power,
            jnp.any(~jnp.isfinite(power)) | jnp.any(power < 0.0),
            "MatterPowerTable values must be finite and non-negative.",
        )
        dimension = int(spatial_dimension)
        if dimension not in (1, 2, 3):
            raise ValueError("MatterPowerTable spatial_dimension must be 1, 2, or 3.")
        self.scale_factors = scales
        self.wavenumbers = wavenumber
        self.power_values = power
        self.scale = scale
        self.provenance = provenance
        self.spatial_dimension = dimension

    def evaluate(self, wavenumber: ArrayLike, scale_factor: ArrayLike, /) -> Array:
        query_k = _validated_query(
            wavenumber, self.wavenumbers, "MatterPowerTable wavenumber"
        )
        query_a = jnp.asarray(scale_factor, dtype=self.scale_factors.dtype)
        if query_a.shape != ():
            raise ValueError("MatterPowerTable scale-factor query must be scalar.")
        query_a = _validated_query(
            query_a, self.scale_factors, "MatterPowerTable scale factor"
        )
        flat_k = query_k.reshape((-1,))
        at_each_scale = jax.vmap(lambda row: jnp.interp(flat_k, self.wavenumbers, row))(
            self.power_values
        )
        values = jax.vmap(
            lambda column: jnp.interp(query_a, self.scale_factors, column),
            in_axes=1,
            out_axes=0,
        )(at_each_scale)
        return values.reshape(query_k.shape)


__all__ = [
    "CosmologyDifferentiability",
    "CosmologyProductProvenance",
    "CosmologyProductSource",
    "ExpansionHistory",
    "LagrangianGrowthHistory",
    "MatterPowerTable",
]
