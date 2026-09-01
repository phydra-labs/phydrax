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
MatterField = Literal["cold_baryon", "total_matter", "massive_neutrino_total"]
MatterPowerStage = Literal["linear", "nonlinear"]
TransferGauge = Literal["synchronous", "newtonian", "gauge-invariant"]
ShotNoiseConvention = Literal["none", "included", "subtracted"]


_DIFFERENTIABILITY_ORDER = {
    "constant": 0,
    "coordinate-only": 1,
    "native-parameter": 2,
}
_MATTER_FIELDS = ("cold_baryon", "total_matter", "massive_neutrino_total")
_GAUGES = ("synchronous", "newtonian", "gauge-invariant")


class CosmologyRealizationSignature(StrictModule):
    """Dynamic physical realization paired with a static model-form contract."""

    parameter_values: Array
    parameter_names: tuple[str, ...] = eqx.field(static=True)
    model_form_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_values: ArrayLike,
        parameter_names: tuple[str, ...],
        model_form_id: str,
        scale_id: str,
        /,
    ):
        names = tuple(str(name).strip() for name in parameter_names)
        values = jnp.asarray(parameter_values).reshape((-1,))
        if not names or len(names) != values.size or any(not name for name in names):
            raise ValueError("Cosmology realization names must match parameter values.")
        if not str(model_form_id).strip() or not str(scale_id).strip():
            raise ValueError("Cosmology realization identities must be non-empty.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Cosmology realization parameters must be finite.",
        )
        self.parameter_values = values
        self.parameter_names = names
        self.model_form_id = str(model_form_id)
        self.scale_id = str(scale_id)

    def require_compatible(
        self, other: CosmologyRealizationSignature, token: ArrayLike, /
    ) -> Array:
        if not isinstance(other, CosmologyRealizationSignature):
            raise TypeError("other must be CosmologyRealizationSignature.")
        if (
            self.parameter_names != other.parameter_names
            or self.model_form_id != other.model_form_id
            or self.scale_id != other.scale_id
        ):
            raise ValueError("Cosmology realization contracts disagree.")
        value = jnp.asarray(token)
        return eqx.error_if(
            value,
            jnp.any(self.parameter_values != other.parameter_values),
            "Cosmology products come from different physical realizations.",
        )


class CosmologyProductProvenance(StrictModule, NonTrainableState):
    """Static producer, model-form, request, policy, scale, and parent identity."""

    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    model_form_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    numerical_policy_id: str = eqx.field(static=True)
    physics_policy_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    parent_product_ids: tuple[str, ...] = eqx.field(static=True)
    source_kind: CosmologyProductSource = eqx.field(static=True)
    differentiability: CosmologyDifferentiability = eqx.field(static=True)
    provenance_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        producer: str,
        producer_version: str,
        model_form_id: str,
        request_id: str,
        numerical_policy_id: str,
        physics_policy_id: str,
        scale_id: str,
        source_kind: CosmologyProductSource,
        differentiability: CosmologyDifferentiability,
        parent_product_ids: tuple[str, ...] = (),
    ):
        values = tuple(
            str(value).strip()
            for value in (
                producer,
                producer_version,
                model_form_id,
                request_id,
                numerical_policy_id,
                physics_policy_id,
                scale_id,
            )
        )
        parents = tuple(str(value).strip() for value in parent_product_ids)
        if any(not value for value in values) or any(not value for value in parents):
            raise ValueError("Cosmology product provenance fields must be non-empty.")
        if source_kind not in ("native", "external"):
            raise ValueError("source_kind must be 'native' or 'external'.")
        if differentiability not in _DIFFERENTIABILITY_ORDER:
            raise ValueError("Unknown cosmology product differentiability contract.")
        (
            self.producer,
            self.producer_version,
            self.model_form_id,
            self.request_id,
            self.numerical_policy_id,
            self.physics_policy_id,
            self.scale_id,
        ) = values
        self.parent_product_ids = parents
        self.source_kind = source_kind
        self.differentiability = differentiability
        self.provenance_id = canonical_fingerprint(
            {
                "kind": "cosmology-product-provenance",
                "producer": values[0],
                "producer_version": values[1],
                "model_form_id": values[2],
                "request_id": values[3],
                "numerical_policy_id": values[4],
                "physics_policy_id": values[5],
                "scale_id": values[6],
                "parent_product_ids": list(parents),
                "source_kind": source_kind,
                "differentiability": differentiability,
            }
        )


def combine_differentiability(
    *policies: CosmologyDifferentiability,
) -> CosmologyDifferentiability:
    if not policies:
        raise ValueError("At least one differentiability policy is required.")
    if any(policy not in _DIFFERENTIABILITY_ORDER for policy in policies):
        raise ValueError("Unknown cosmology differentiability policy.")
    return min(policies, key=_DIFFERENTIABILITY_ORDER.__getitem__)


def _stored(values: Array, policy: CosmologyDifferentiability, /) -> Array:
    if policy == "native-parameter":
        return values
    return jax.lax.stop_gradient(values)


def _evaluated(values: Array, policy: CosmologyDifferentiability, /) -> Array:
    if policy == "constant":
        return jax.lax.stop_gradient(values)
    return values


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
    realization: CosmologyRealizationSignature,
    /,
) -> None:
    if not isinstance(scale, CosmologyScaleContract):
        raise TypeError("scale must be a CosmologyScaleContract.")
    if not isinstance(provenance, CosmologyProductProvenance):
        raise TypeError("provenance must be CosmologyProductProvenance.")
    if not isinstance(realization, CosmologyRealizationSignature):
        raise TypeError("realization must be CosmologyRealizationSignature.")
    if scale.scale_id != provenance.scale_id or scale.scale_id != realization.scale_id:
        raise ValueError("Cosmology product scale identities disagree.")
    if provenance.model_form_id != realization.model_form_id:
        raise ValueError("Cosmology product model-form identities disagree.")


class ExpansionHistory(StrictModule):
    """Tabulated Hubble expansion with explicit realization and provenance."""

    scale_factors: Array
    hubble_values: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature

    def __init__(
        self,
        scale_factors: ArrayLike,
        hubble_values: ArrayLike,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        /,
    ):
        _validate_common(scale, provenance, realization)
        nodes = _validated_nodes(scale_factors, "ExpansionHistory")
        hubble = jnp.asarray(hubble_values, dtype=nodes.dtype)
        if hubble.shape != nodes.shape:
            raise ValueError("ExpansionHistory values must match scale-factor nodes.")
        hubble = eqx.error_if(
            hubble,
            jnp.any(~jnp.isfinite(hubble)) | jnp.any(hubble <= 0.0),
            "ExpansionHistory Hubble values must be finite and positive.",
        )
        policy = provenance.differentiability
        self.scale_factors = _stored(nodes, policy)
        self.hubble_values = _stored(hubble, policy)
        self.scale = scale
        self.provenance = provenance
        self.realization = realization

    def hubble(self, scale_factor: ArrayLike, /) -> Array:
        query = _validated_query(scale_factor, self.scale_factors, "ExpansionHistory")
        values = jnp.interp(query, self.scale_factors, self.hubble_values)
        return _evaluated(values, self.provenance.differentiability)


class LagrangianGrowthHistory(StrictModule):
    """First- and second-order Lagrangian growth and logarithmic rates."""

    scale_factors: Array
    first_order_growth: Array
    first_order_rate: Array
    second_order_growth: Array
    second_order_rate: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature

    def __init__(
        self,
        scale_factors: ArrayLike,
        first_order_growth: ArrayLike,
        first_order_rate: ArrayLike,
        second_order_growth: ArrayLike,
        second_order_rate: ArrayLike,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        /,
    ):
        _validate_common(scale, provenance, realization)
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
        policy = provenance.differentiability
        stacked = _stored(stacked, policy)
        self.scale_factors = _stored(nodes, policy)
        (
            self.first_order_growth,
            self.first_order_rate,
            self.second_order_growth,
            self.second_order_rate,
        ) = tuple(stacked[index] for index in range(4))
        self.scale = scale
        self.provenance = provenance
        self.realization = realization

    def evaluate(self, scale_factor: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        query = _validated_query(
            scale_factor, self.scale_factors, "LagrangianGrowthHistory"
        )
        values = tuple(
            jnp.interp(query, self.scale_factors, values)
            for values in (
                self.first_order_growth,
                self.first_order_rate,
                self.second_order_growth,
                self.second_order_rate,
            )
        )
        return tuple(
            _evaluated(value, self.provenance.differentiability) for value in values
        )


class MatterPowerDescriptor(StrictModule, NonTrainableState):
    """Static physical field, gauge, stage, noise, and dimension semantics."""

    left_field: MatterField = eqx.field(static=True)
    right_field: MatterField = eqx.field(static=True)
    gauge: TransferGauge = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    stage: MatterPowerStage = eqx.field(static=True)
    shot_noise: ShotNoiseConvention = eqx.field(static=True)
    spatial_dimension: int = eqx.field(static=True)
    descriptor_id: str = eqx.field(static=True)

    def __init__(
        self,
        left_field: MatterField,
        right_field: MatterField,
        /,
        *,
        gauge: TransferGauge = "synchronous",
        normalization: str = "dimensionless-density-contrast",
        stage: MatterPowerStage = "linear",
        shot_noise: ShotNoiseConvention = "none",
        spatial_dimension: int = 3,
    ):
        if left_field not in _MATTER_FIELDS or right_field not in _MATTER_FIELDS:
            raise ValueError("Unknown matter field identity.")
        if gauge not in _GAUGES:
            raise ValueError("Unknown transfer gauge.")
        if stage not in ("linear", "nonlinear"):
            raise ValueError("Matter power stage must be linear or nonlinear.")
        if shot_noise not in ("none", "included", "subtracted"):
            raise ValueError("Unknown shot-noise convention.")
        dimension = int(spatial_dimension)
        if dimension not in (1, 2, 3):
            raise ValueError("Matter power spatial dimension must be 1, 2, or 3.")
        normalization_ = str(normalization).strip()
        if not normalization_:
            raise ValueError("Matter power normalization must be non-empty.")
        self.left_field = left_field
        self.right_field = right_field
        self.gauge = gauge
        self.normalization = normalization_
        self.stage = stage
        self.shot_noise = shot_noise
        self.spatial_dimension = dimension
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "matter-power-descriptor",
                "left_field": left_field,
                "right_field": right_field,
                "gauge": gauge,
                "normalization": normalization_,
                "stage": stage,
                "shot_noise": shot_noise,
                "spatial_dimension": dimension,
            }
        )

    @property
    def is_auto(self) -> bool:
        return self.left_field == self.right_field

    @property
    def is_linear_cold_baryon_auto(self) -> bool:
        return (
            self.left_field == "cold_baryon"
            and self.right_field == "cold_baryon"
            and self.stage == "linear"
            and self.shot_noise == "none"
        )


class MatterPowerTable(StrictModule):
    """Tabulated, semantically named density-contrast power P_xy(k, a)."""

    scale_factors: Array
    wavenumbers: Array
    power_values: Array
    descriptor: MatterPowerDescriptor
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature

    def __init__(
        self,
        scale_factors: ArrayLike,
        wavenumbers: ArrayLike,
        power_values: ArrayLike,
        descriptor: MatterPowerDescriptor,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        /,
    ):
        _validate_common(scale, provenance, realization)
        if not isinstance(descriptor, MatterPowerDescriptor):
            raise TypeError("descriptor must be MatterPowerDescriptor.")
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
        invalid = jnp.any(~jnp.isfinite(power))
        if descriptor.is_auto:
            invalid = invalid | jnp.any(power < 0.0)
        power = eqx.error_if(
            power,
            invalid,
            "MatterPowerTable values violate auto/cross-power constraints.",
        )
        policy = provenance.differentiability
        self.scale_factors = _stored(scales, policy)
        self.wavenumbers = _stored(wavenumber, policy)
        self.power_values = _stored(power, policy)
        self.descriptor = descriptor
        self.scale = scale
        self.provenance = provenance
        self.realization = realization

    @property
    def power_unit(self) -> str:
        return f"{self.scale.length_unit}^{self.descriptor.spatial_dimension}"

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
        values = values.reshape(query_k.shape)
        return _evaluated(values, self.provenance.differentiability)


class LinearTransferDescriptor(StrictModule, NonTrainableState):
    """Static field ordering, gauge, normalization, and k/q convention."""

    fields: tuple[str, ...] = eqx.field(static=True)
    gauge: TransferGauge = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    wavenumber_coordinate: Literal["k", "q"] = eqx.field(static=True)
    descriptor_id: str = eqx.field(static=True)

    def __init__(
        self,
        fields: tuple[str, ...],
        /,
        *,
        gauge: TransferGauge,
        normalization: str,
        wavenumber_coordinate: Literal["k", "q"] = "k",
    ):
        fields_ = tuple(str(field).strip() for field in fields)
        if (
            not fields_
            or any(not field for field in fields_)
            or len(set(fields_)) != len(fields_)
        ):
            raise ValueError("Linear transfer fields must be non-empty and unique.")
        if gauge not in _GAUGES:
            raise ValueError("Unknown transfer gauge.")
        normalization_ = str(normalization).strip()
        if not normalization_:
            raise ValueError("Transfer normalization must be non-empty.")
        if wavenumber_coordinate not in ("k", "q"):
            raise ValueError("Transfer wavenumber coordinate must be k or q.")
        self.fields = fields_
        self.gauge = gauge
        self.normalization = normalization_
        self.wavenumber_coordinate = wavenumber_coordinate
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "linear-transfer-descriptor",
                "fields": list(fields_),
                "gauge": gauge,
                "normalization": normalization_,
                "wavenumber_coordinate": wavenumber_coordinate,
            }
        )


class LinearTransferTable(StrictModule):
    """Signed linear transfer fields on fixed (field, a, k-or-q) coordinates."""

    scale_factors: Array
    wavenumbers: Array
    transfer_values: Array
    descriptor: LinearTransferDescriptor
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature

    def __init__(
        self,
        scale_factors: ArrayLike,
        wavenumbers: ArrayLike,
        transfer_values: ArrayLike,
        descriptor: LinearTransferDescriptor,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        /,
    ):
        _validate_common(scale, provenance, realization)
        if not isinstance(descriptor, LinearTransferDescriptor):
            raise TypeError("descriptor must be LinearTransferDescriptor.")
        scales = _validated_nodes(scale_factors, "LinearTransferTable scale factor")
        wavenumber = _validated_nodes(wavenumbers, "LinearTransferTable wavenumber")
        values = jnp.asarray(transfer_values, dtype=scales.dtype)
        expected = (len(descriptor.fields), scales.size, wavenumber.size)
        if values.shape != expected:
            raise ValueError(f"LinearTransferTable values must have shape {expected}.")
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Linear transfer values must be finite.",
        )
        policy = provenance.differentiability
        self.scale_factors = _stored(scales, policy)
        self.wavenumbers = _stored(wavenumber, policy)
        self.transfer_values = _stored(values, policy)
        self.descriptor = descriptor
        self.scale = scale
        self.provenance = provenance
        self.realization = realization

    def evaluate(
        self, field: str, wavenumber: ArrayLike, scale_factor: ArrayLike, /
    ) -> Array:
        field_ = str(field).strip()
        if field_ not in self.descriptor.fields:
            raise ValueError(f"Unknown transfer field {field_!r}.")
        index = self.descriptor.fields.index(field_)
        query_k = _validated_query(
            wavenumber, self.wavenumbers, "LinearTransferTable wavenumber"
        )
        query_a = jnp.asarray(scale_factor, dtype=self.scale_factors.dtype)
        if query_a.shape != ():
            raise ValueError("LinearTransferTable scale-factor query must be scalar.")
        query_a = _validated_query(
            query_a, self.scale_factors, "LinearTransferTable scale factor"
        )
        flat_k = query_k.reshape((-1,))
        at_each_scale = jax.vmap(lambda row: jnp.interp(flat_k, self.wavenumbers, row))(
            self.transfer_values[index]
        )
        values = jax.vmap(
            lambda column: jnp.interp(query_a, self.scale_factors, column),
            in_axes=1,
            out_axes=0,
        )(at_each_scale)
        values = values.reshape(query_k.shape)
        return _evaluated(values, self.provenance.differentiability)


class ThermodynamicsHistory(StrictModule):
    """External or native thermodynamic history on fixed scale-factor nodes."""

    scale_factors: Array
    ionization_fraction: Array
    baryon_temperature: Array
    opacity_derivative: Array
    visibility: Array
    scale: CosmologyScaleContract
    provenance: CosmologyProductProvenance
    realization: CosmologyRealizationSignature

    def __init__(
        self,
        scale_factors: ArrayLike,
        ionization_fraction: ArrayLike,
        baryon_temperature: ArrayLike,
        opacity_derivative: ArrayLike,
        visibility: ArrayLike,
        scale: CosmologyScaleContract,
        provenance: CosmologyProductProvenance,
        realization: CosmologyRealizationSignature,
        /,
    ):
        _validate_common(scale, provenance, realization)
        nodes = _validated_nodes(scale_factors, "ThermodynamicsHistory")
        values = tuple(
            jnp.asarray(value, dtype=nodes.dtype)
            for value in (
                ionization_fraction,
                baryon_temperature,
                opacity_derivative,
                visibility,
            )
        )
        if any(value.shape != nodes.shape for value in values):
            raise ValueError("Thermodynamics arrays must match scale-factor nodes.")
        stacked = eqx.error_if(
            jnp.stack(values),
            jnp.any(~jnp.isfinite(jnp.stack(values)))
            | jnp.any(values[0] < 0.0)
            | jnp.any(values[1] < 0.0)
            | jnp.any(values[3] < 0.0),
            "Thermodynamics values violate finite/non-negative constraints.",
        )
        policy = provenance.differentiability
        stacked = _stored(stacked, policy)
        self.scale_factors = _stored(nodes, policy)
        (
            self.ionization_fraction,
            self.baryon_temperature,
            self.opacity_derivative,
            self.visibility,
        ) = tuple(stacked[index] for index in range(4))
        self.scale = scale
        self.provenance = provenance
        self.realization = realization


def reconstruct_total_matter_power(
    cold_baryon: MatterPowerTable,
    neutrino: MatterPowerTable,
    cross: MatterPowerTable,
    cold_baryon_fraction: ArrayLike,
    neutrino_fraction: ArrayLike,
    /,
) -> MatterPowerTable:
    expected = (
        cold_baryon.descriptor.left_field,
        cold_baryon.descriptor.right_field,
        neutrino.descriptor.left_field,
        neutrino.descriptor.right_field,
        cross.descriptor.left_field,
        cross.descriptor.right_field,
    )
    if expected != (
        "cold_baryon",
        "cold_baryon",
        "massive_neutrino_total",
        "massive_neutrino_total",
        "cold_baryon",
        "massive_neutrino_total",
    ):
        raise ValueError(
            "Total-matter reconstruction requires cb, neutrino, and cb-neutrino spectra."
        )
    if (
        cold_baryon.descriptor.stage != neutrino.descriptor.stage
        or cold_baryon.descriptor.stage != cross.descriptor.stage
        or cold_baryon.descriptor.gauge != neutrino.descriptor.gauge
        or cold_baryon.descriptor.gauge != cross.descriptor.gauge
        or cold_baryon.descriptor.spatial_dimension
        != neutrino.descriptor.spatial_dimension
        or cold_baryon.descriptor.spatial_dimension != cross.descriptor.spatial_dimension
    ):
        raise ValueError("Total-matter spectrum descriptors disagree.")
    if (
        cold_baryon.scale_factors.shape != neutrino.scale_factors.shape
        or cold_baryon.scale_factors.shape != cross.scale_factors.shape
        or cold_baryon.wavenumbers.shape != neutrino.wavenumbers.shape
        or cold_baryon.wavenumbers.shape != cross.wavenumbers.shape
    ):
        raise ValueError("Total-matter spectrum grids disagree.")
    token = cold_baryon.realization.require_compatible(
        neutrino.realization, cold_baryon.power_values
    )
    token = cold_baryon.realization.require_compatible(cross.realization, token)
    fcb = jnp.asarray(cold_baryon_fraction, dtype=token.dtype)
    fnu = jnp.asarray(neutrino_fraction, dtype=token.dtype)
    if fcb.shape != () or fnu.shape != ():
        raise ValueError("Matter fractions must be scalar.")
    fractions = eqx.error_if(
        jnp.stack((fcb, fnu)),
        (fcb < 0.0) | (fnu < 0.0) | (jnp.abs(fcb + fnu - 1.0) > 1.0e-10),
        "Cold-baryon and neutrino fractions must be non-negative and sum to one.",
    )
    values = (
        fractions[0] ** 2 * cold_baryon.power_values
        + 2.0 * fractions[0] * fractions[1] * cross.power_values
        + fractions[1] ** 2 * neutrino.power_values
    )
    policy = combine_differentiability(
        cold_baryon.provenance.differentiability,
        neutrino.provenance.differentiability,
        cross.provenance.differentiability,
    )
    provenance = CosmologyProductProvenance(
        producer="phydrax.applications.cosmology.reconstruct_total_matter_power",
        producer_version="native",
        model_form_id=cold_baryon.provenance.model_form_id,
        request_id=cold_baryon.provenance.request_id,
        numerical_policy_id="algebraic-total-matter-reconstruction",
        physics_policy_id="cb-plus-massive-neutrino-cross-power",
        scale_id=cold_baryon.scale.scale_id,
        source_kind="native" if policy == "native-parameter" else "external",
        differentiability=policy,
        parent_product_ids=(
            cold_baryon.provenance.provenance_id,
            neutrino.provenance.provenance_id,
            cross.provenance.provenance_id,
        ),
    )
    descriptor = MatterPowerDescriptor(
        "total_matter",
        "total_matter",
        gauge=cold_baryon.descriptor.gauge,
        normalization=cold_baryon.descriptor.normalization,
        stage=cold_baryon.descriptor.stage,
        shot_noise="none",
        spatial_dimension=cold_baryon.descriptor.spatial_dimension,
    )
    return MatterPowerTable(
        cold_baryon.scale_factors,
        cold_baryon.wavenumbers,
        values,
        descriptor,
        cold_baryon.scale,
        provenance,
        cold_baryon.realization,
    )


__all__ = [
    "CosmologyDifferentiability",
    "CosmologyProductProvenance",
    "CosmologyProductSource",
    "CosmologyRealizationSignature",
    "ExpansionHistory",
    "LagrangianGrowthHistory",
    "LinearTransferDescriptor",
    "LinearTransferTable",
    "MatterField",
    "MatterPowerDescriptor",
    "MatterPowerStage",
    "MatterPowerTable",
    "ShotNoiseConvention",
    "ThermodynamicsHistory",
    "TransferGauge",
    "combine_differentiability",
    "reconstruct_total_matter_power",
]
