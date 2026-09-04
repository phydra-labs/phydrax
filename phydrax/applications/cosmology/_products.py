#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...series import SampledSeries, SampledSeriesReconstruction, SeriesSupport
from ...units import derived_unit, UnitDefinition
from ._closure import CosmologyRealizationSignature, DifferentiationContract
from ._scales import CosmologyScaleContract


CosmologyProductSource = Literal["native", "external"]
MatterField = Literal["cold_baryon", "total_matter", "massive_neutrino_total"]
MatterPowerStage = Literal["linear", "nonlinear"]
TransferGauge = Literal["synchronous", "newtonian", "gauge-invariant"]
ShotNoiseConvention = Literal["none", "included", "subtracted"]


_MATTER_FIELDS = ("cold_baryon", "total_matter", "massive_neutrino_total")
_GAUGES = ("synchronous", "newtonian", "gauge-invariant")


class CosmologyProductProvenance(StrictModule, NonTrainableState):
    """Static producer, request, policy, scale, lineage, and derivative identity."""

    producer: str = eqx.field(static=True)
    producer_version: str = eqx.field(static=True)
    model_form_id: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)
    numerical_policy_id: str = eqx.field(static=True)
    physics_policy_id: str = eqx.field(static=True)
    scale_id: str = eqx.field(static=True)
    parent_product_ids: tuple[str, ...] = eqx.field(static=True)
    source_kind: CosmologyProductSource = eqx.field(static=True)
    differentiation: DifferentiationContract
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
        differentiation: DifferentiationContract | str,
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
        differentiation_ = (
            DifferentiationContract.from_label(differentiation)
            if isinstance(differentiation, str)
            else differentiation
        )
        if not isinstance(differentiation_, DifferentiationContract):
            raise TypeError("differentiation must be DifferentiationContract.")
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
        self.differentiation = differentiation_
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
                "differentiation": differentiation_.contract_id,
            }
        )


def combine_differentiation(
    *contracts: DifferentiationContract,
) -> DifferentiationContract:
    if not contracts:
        raise ValueError("At least one differentiation contract is required.")
    if any(not isinstance(value, DifferentiationContract) for value in contracts):
        raise TypeError("All values must be DifferentiationContract.")
    return contracts[0].meet(*contracts[1:])


def _stored(values: Array, contract_: DifferentiationContract, /) -> Array:
    if contract_.stored_values:
        return values
    return jax.lax.stop_gradient(values)


def _evaluated(values: Array, contract_: DifferentiationContract, /) -> Array:
    if not contract_.query_coordinates:
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


class ExpansionHistory(StrictModule):
    """Tabulated Hubble expansion with explicit realization and provenance."""

    reconstruction: SampledSeriesReconstruction
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
        policy = provenance.differentiation
        support = SeriesSupport(
            _stored(nodes, policy),
            coordinate_name="scale_factor",
            coordinate_id=f"scale-factor:{provenance.provenance_id}",
        )
        series = SampledSeries(
            support,
            _stored(hubble, policy),
            series_id=f"expansion-history:{provenance.provenance_id}",
        )
        self.reconstruction = SampledSeriesReconstruction(
            series,
            interpolation="linear",
            bounds="error",
        )
        self.scale = scale
        self.provenance = provenance
        self.realization = realization

    @property
    def scale_factors(self) -> Array:
        return self.reconstruction.series.support.coordinates

    @property
    def hubble_values(self) -> Array:
        return self.reconstruction.series.values

    def hubble(self, scale_factor: ArrayLike, /) -> Array:
        query = _validated_query(scale_factor, self.scale_factors, "ExpansionHistory")
        values = self.reconstruction.evaluate(query).values
        return _evaluated(values, self.provenance.differentiation)


class LagrangianGrowthHistory(StrictModule):
    """First- and second-order Lagrangian growth and logarithmic rates."""

    reconstruction: SampledSeriesReconstruction
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
        policy = provenance.differentiation
        stored = _stored(stacked, policy)
        support = SeriesSupport(
            _stored(nodes, policy),
            coordinate_name="scale_factor",
            coordinate_id=f"scale-factor:{provenance.provenance_id}",
        )
        series = SampledSeries(
            support,
            tuple(stored[index] for index in range(4)),
            series_id=f"lagrangian-growth:{provenance.provenance_id}",
        )
        self.reconstruction = SampledSeriesReconstruction(
            series,
            interpolation="linear",
            bounds="error",
        )
        self.scale = scale
        self.provenance = provenance
        self.realization = realization

    @property
    def scale_factors(self) -> Array:
        return self.reconstruction.series.support.coordinates

    @property
    def first_order_growth(self) -> Array:
        return self.reconstruction.series.values[0]

    @property
    def first_order_rate(self) -> Array:
        return self.reconstruction.series.values[1]

    @property
    def second_order_growth(self) -> Array:
        return self.reconstruction.series.values[2]

    @property
    def second_order_rate(self) -> Array:
        return self.reconstruction.series.values[3]

    def evaluate(self, scale_factor: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        query = _validated_query(
            scale_factor, self.scale_factors, "LagrangianGrowthHistory"
        )
        values = self.reconstruction.evaluate(query).values
        return tuple(
            _evaluated(value, self.provenance.differentiation) for value in values
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
    power_unit: UnitDefinition = eqx.field(static=True)
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
        policy = provenance.differentiation
        self.scale_factors = _stored(scales, policy)
        self.wavenumbers = _stored(wavenumber, policy)
        self.power_values = _stored(power, policy)
        self.descriptor = descriptor
        self.scale = scale
        self.power_unit = derived_unit(
            f"{scale.length_unit.symbol}^{descriptor.spatial_dimension}",
            ((scale.length_unit, descriptor.spatial_dimension),),
        )
        self.provenance = provenance
        self.realization = realization

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
        return _evaluated(values, self.provenance.differentiation)


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
        policy = provenance.differentiation
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
        return _evaluated(values, self.provenance.differentiation)


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
        policy = provenance.differentiation
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
    differentiation = combine_differentiation(
        cold_baryon.provenance.differentiation,
        neutrino.provenance.differentiation,
        cross.provenance.differentiation,
    )
    provenance = CosmologyProductProvenance(
        producer="phydrax.applications.cosmology.reconstruct_total_matter_power",
        producer_version="native",
        model_form_id=cold_baryon.provenance.model_form_id,
        request_id=cold_baryon.provenance.request_id,
        numerical_policy_id="algebraic-total-matter-reconstruction",
        physics_policy_id="cb-plus-massive-neutrino-cross-power",
        scale_id=cold_baryon.scale.scale_id,
        source_kind="native",
        differentiation=differentiation,
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


def cosmology_product_content_id(product, /) -> str:
    """Return a host-side content identity for an immutable cosmology product."""
    if isinstance(product, ExpansionHistory):
        descriptor = "expansion-history"
        coordinates = product.scale_factors
        payload = product.hubble_values
    elif isinstance(product, LagrangianGrowthHistory):
        descriptor = "lagrangian-growth-history"
        coordinates = product.scale_factors
        payload = (
            product.first_order_growth,
            product.first_order_rate,
            product.second_order_growth,
            product.second_order_rate,
        )
    elif isinstance(product, MatterPowerTable):
        descriptor = product.descriptor.descriptor_id
        coordinates = (product.scale_factors, product.wavenumbers)
        payload = product.power_values
    elif isinstance(product, LinearTransferTable):
        descriptor = product.descriptor.descriptor_id
        coordinates = (product.scale_factors, product.wavenumbers)
        payload = product.transfer_values
    elif isinstance(product, ThermodynamicsHistory):
        descriptor = "thermodynamics-history"
        coordinates = product.scale_factors
        payload = (
            product.ionization_fraction,
            product.baryon_temperature,
            product.opacity_derivative,
            product.visibility,
        )
    else:
        raise TypeError("Unsupported cosmology product type.")
    return canonical_fingerprint(
        {
            "kind": "cosmology-product-content",
            "descriptor": descriptor,
            "realization": product.realization.content_id(),
            "scale": product.scale.scale_id,
            "coordinates": array_tree_fingerprint(coordinates),
            "payload": array_tree_fingerprint(payload),
            "parents": list(product.provenance.parent_product_ids),
            "differentiation": product.provenance.differentiation.contract_id,
        }
    )


__all__ = [
    "CosmologyProductProvenance",
    "CosmologyProductSource",
    "CosmologyRealizationSignature",
    "DifferentiationContract",
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
    "combine_differentiation",
    "cosmology_product_content_id",
    "reconstruct_total_matter_power",
]
