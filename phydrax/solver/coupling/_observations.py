#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Observation bindings of spatial coupled problems.

An observation binding names one component field and the measurement it
predicts: a ``QuantitySpec`` (physical meaning, unit, compatibility), a scalar
``ValueLayout``, a ``SampleSupport``, and ``SamplingSemantics`` (spatial kind,
sample times or intervals). Preparation lowers it once through the owner's
published capabilities -- prepared field queries for point values and
derivatives, exact side traces for boundary traces, averages, and integrals,
and residual-reaction fluxes for flux content -- and every accepted coupled
solution then evaluates it into a ``PreparedQuantityField`` whose identities
equal those of data prepared from the same records.

The declared sampling must be the operation the binding performs, and the
quantity unit must have the dimension that operation produces: a point value
is never an average or an integral, and a flux content (a residual reaction
integrated over facets) is never a flux density or a field value.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from fractions import Fraction
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..._validation import canonical_identifier
from ...discretization import (
    FacetTraceRule,
    FieldQueryCoverage,
    IntegrationDomain,
    PreparedFieldQuery,
    PreparedFluxAction,
    PreparedTraceAction,
)
from ...measurement import (
    PointSampleSupport,
    PreparedQuantityField,
    QuantitySpec,
    SampleSupport,
    SamplingSemantics,
    SpatialSamplingKind,
    ValueLayout,
)
from ...typing import checked, parse
from ...units import conversion_factor, derived_unit, UnitDefinition
from ._components import (
    AbstractReconstructionComponent,
    AbstractSpatialComponent,
    AbstractTraceComponent,
)


BoundaryStatistic: TypeAlias = Literal["trace", "average", "integral"]

type FieldValues = Mapping[tuple[str, str], Array]


@final
class MeasurementIdentity(StrictModule, NonTrainableState):
    """Static identities of one predicted measurement.

    Built from the same host records as the observed data, so a prediction and
    its data compare only when quantity, layout, support, sampling, and unit
    identities all agree. Coupled field observations are real scalars.
    """

    quantity_id: str = eqx.field(static=True)
    compatibility_id: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    sampling_id: str = eqx.field(static=True)
    quantity_name: str = eqx.field(static=True)
    spatial_kind: SpatialSamplingKind = eqx.field(static=True)
    sample_shape: tuple[int, ...] = eqx.field(static=True)
    unit: UnitDefinition

    @checked
    def __init__(
        self,
        quantity: QuantitySpec,
        layout: ValueLayout,
        support: SampleSupport,
        sampling: SamplingSemantics,
        /,
    ) -> None:
        if not isinstance(support, SampleSupport):
            raise TypeError("support must be a SampleSupport.")
        if layout.component_shape != ():
            raise ValueError("Coupled field observations are scalar per sample.")
        self.quantity_id = quantity.quantity_id
        self.compatibility_id = quantity.compatibility_id
        self.layout_id = layout.layout_id
        self.support_id = support.support_id
        self.sampling_id = sampling.sampling_id
        self.quantity_name = quantity.name
        self.spatial_kind = sampling.spatial_kind
        self.sample_shape = tuple(support.sample_shape)
        self.unit = quantity.unit

    def require_kind(self, kind: SpatialSamplingKind, operation: str, /) -> None:
        if self.spatial_kind is not kind:
            raise ValueError(
                f"The observation takes {operation}, sampled as {kind.value!r}; the "
                f"declared sampling is {self.spatial_kind.value!r}. A point value, an "
                "average, and an integral are distinct measurements."
            )

    def scale(self, source: UnitDefinition, operation: str, /) -> float:
        """Exact factor from the operation's unit to the quantity unit."""
        if source.dimension != self.unit.dimension:
            raise ValueError(
                f"The observation takes {operation} with dimension "
                f"{source.dimension.dimension_id}, but quantity {self.quantity_name!r} "
                f"has dimension {self.unit.dimension.dimension_id}; a density, a "
                "content, and a field value are never identified."
            )
        return float(conversion_factor(source, self.unit))

    def field(
        self, binding_id: str, scale: float, values: Array, valid: Array, /
    ) -> PreparedQuantityField:
        scaled = values if scale == 1.0 else values * jnp.asarray(scale, values.dtype)
        return PreparedQuantityField(
            scaled,
            valid,
            standard_uncertainty=None,
            quantity_id=self.quantity_id,
            compatibility_id=self.compatibility_id,
            layout_id=self.layout_id,
            support_id=self.support_id,
            sampling_id=self.sampling_id,
            unit_id=self.unit.unit_id,
            field_id=binding_id,
        )


def _power(unit: UnitDefinition, length: UnitDefinition, order: int, /) -> UnitDefinition:
    if order == 0:
        return unit
    return derived_unit(
        f"{unit.symbol}*{length.symbol}^{order}", ((unit, 1), (length, Fraction(order)))
    )


def _coordinate_length_unit(
    support: PointSampleSupport, declared: UnitDefinition | None, /
) -> UnitDefinition:
    """Length unit of the sample coordinates, which the support's contract owns."""
    unit = support.coordinate_contract.length_unit
    if declared is None:
        return unit
    if not isinstance(declared, UnitDefinition):
        raise TypeError("length_unit must be a UnitDefinition.")
    if declared.unit_id != unit.unit_id:
        raise ValueError(
            f"length_unit {declared.symbol!r} contradicts the support's coordinate "
            f"contract, whose coordinates are in {unit.symbol!r}; one set of sample "
            "coordinates has one length unit."
        )
    return unit


class AbstractPreparedObservation(StrictModule):
    """One observation lowered through its owner's prepared actions.

    ``approximation`` labels what is observed: the field itself
    (``"exact"``), a labeled polynomial projection (``"h1-projection"``), or
    a residual reaction (``"variational-reaction"``). ``complete`` is true
    when every sample is valid for every field state; sample validity is fixed
    at preparation.
    """

    binding_id: eqx.AbstractVar[str]
    component: eqx.AbstractVar[str]
    field: eqx.AbstractVar[str]
    approximation: eqx.AbstractVar[str]
    identity: eqx.AbstractVar[MeasurementIdentity]

    @abc.abstractmethod
    def evaluate(
        self, fields: FieldValues, arguments: Mapping[str, object], /
    ) -> PreparedQuantityField:
        """Predicted measurement from full field coefficients and runtime arguments."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def complete(self) -> bool:
        """Whether every sample of every prediction is valid."""
        raise NotImplementedError


class AbstractObservationBinding(StrictModule):
    """Declared observation of named component fields.

    Concrete bindings name components and fields, never owner objects, so
    substituting a component's method under the same name keeps the binding.
    """

    binding_id: eqx.AbstractVar[str]
    components: eqx.AbstractVar[tuple[str, ...]]

    @abc.abstractmethod
    def prepare(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> AbstractPreparedObservation:
        raise NotImplementedError


def _component[T](
    components: Mapping[str, AbstractSpatialComponent],
    name: str,
    field: str,
    kind: type[T],
    capability: str,
    /,
) -> T:
    component = components[name]
    component.field(field)
    if not isinstance(component, kind):
        raise TypeError(f"Component {name!r} publishes no {capability}.")
    return component


# --- Point values and derivatives -------------------------------------------------------


@final
class PreparedPointObservation(AbstractPreparedObservation):
    binding_id: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    identity: MeasurementIdentity
    query: PreparedFieldQuery
    valid_samples: Array
    scale: float = eqx.field(static=True)

    @property
    def complete(self) -> bool:
        return self.valid_samples.shape[0] == self.identity.sample_shape[0]

    def evaluate(
        self, fields: FieldValues, arguments: Mapping[str, object], /
    ) -> PreparedQuantityField:
        del arguments
        admitted = self.query.apply(fields[(self.component, self.field)])
        count = self.identity.sample_shape[0]
        values = (
            jnp.zeros((count,), dtype=admitted.dtype).at[self.valid_samples].set(admitted)
        )
        valid = jnp.zeros((count,), dtype=jnp.bool_).at[self.valid_samples].set(True)
        return self.identity.field(self.binding_id, self.scale, values, valid)


@final
class FieldPointObservation(AbstractObservationBinding, NonTrainableState):
    """Point values (or one partial derivative) of a component field.

    The points are the active points of the ``PointSampleSupport``; the
    sampling must be ``POINT``. ``derivative`` is a coordinate multi-index; the
    quantity unit must then have the dimension of ``field_unit /
    length_unit^order``. The sample coordinates are in the length unit of the
    support's coordinate contract, so ``length_unit`` defaults to it and a
    differing declaration is refused. ``coverage="complete"`` predicts every
    sample and refuses points outside the component and inactive support
    samples; ``coverage="masked"`` reports both as invalid samples. Point
    location is the component's owner numerics
    (``VariationalComponent.location_policy``), so substituting the
    component's method keeps this binding unchanged.
    """

    binding_id: str = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)
    field: str = eqx.field(static=True)
    identity: MeasurementIdentity
    points: Array
    samples: Array
    field_unit: UnitDefinition
    length_unit: UnitDefinition
    derivative: tuple[int, ...] | None = eqx.field(static=True)
    coverage: FieldQueryCoverage = eqx.field(static=True)

    @checked
    def __init__(
        self,
        binding_id: str,
        component: str,
        field: str,
        /,
        *,
        quantity: QuantitySpec,
        support: PointSampleSupport,
        sampling: SamplingSemantics,
        field_unit: UnitDefinition,
        layout: ValueLayout | None = None,
        derivative: tuple[int, ...] | None = None,
        length_unit: UnitDefinition | None = None,
        coverage: FieldQueryCoverage = "complete",
    ) -> None:
        identity = MeasurementIdentity(
            quantity,
            ValueLayout.scalar() if layout is None else layout,
            support,
            sampling,
        )
        identity.require_kind(SpatialSamplingKind.POINT, "point values")
        coverage_ = parse(coverage, FieldQueryCoverage, "coverage")
        active = support.active_mask
        if active is None:
            raise RuntimeError("PointSampleSupport did not materialize its active_mask.")
        samples = np.flatnonzero(active).astype(np.int32)
        if samples.size == 0:
            raise ValueError("A point observation needs at least one active sample.")
        if coverage_ == "complete" and samples.size != active.size:
            raise ValueError(
                "A complete point observation predicts every sample, but the support "
                "has inactive samples; declare coverage='masked' to report them invalid."
            )
        length = _coordinate_length_unit(support, length_unit)
        order = 0 if derivative is None else sum(derivative)
        identity.scale(_power(field_unit, length, -order), "point values")
        self.binding_id = canonical_identifier(binding_id, "binding_id")
        self.components = (canonical_identifier(component, "component"),)
        self.field = canonical_identifier(field, "field")
        self.identity = identity
        self.points = jnp.asarray(support.points[samples])
        self.samples = jnp.asarray(samples)
        self.field_unit = field_unit
        self.length_unit = length
        self.derivative = None if derivative is None else tuple(derivative)
        self.coverage = coverage_

    def prepare(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> PreparedPointObservation:
        component = _component(
            components,
            self.components[0],
            self.field,
            AbstractReconstructionComponent,
            "pointwise field reconstruction",
        )
        reconstruction = component.prepare_field_reconstruction(self.field)
        points = np.asarray(self.points)
        dimension = reconstruction.physical_dimension
        if np.any(points[:, dimension:] != 0.0):
            raise ValueError(
                f"Sample points have nonzero coordinates beyond the component's "
                f"{dimension} physical dimensions."
            )
        query = reconstruction.prepare_query(
            points[:, :dimension], derivative=self.derivative, coverage=self.coverage
        )
        order = 0 if self.derivative is None else sum(self.derivative)
        return PreparedPointObservation(
            binding_id=self.binding_id,
            component=self.components[0],
            field=self.field,
            approximation=query.approximation,
            identity=self.identity,
            query=query,
            valid_samples=jnp.asarray(
                np.asarray(self.samples)[np.asarray(query.admitted)]
            ),
            scale=self.identity.scale(
                _power(self.field_unit, self.length_unit, -order), "point values"
            ),
        )


# --- Boundary traces, averages, and integrals -------------------------------------------


@final
class PreparedBoundaryObservation(AbstractPreparedObservation):
    binding_id: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    identity: MeasurementIdentity
    trace: PreparedTraceAction
    statistic: BoundaryStatistic = eqx.field(static=True)
    scale: float = eqx.field(static=True)

    @property
    def complete(self) -> bool:
        return True

    def evaluate(
        self, fields: FieldValues, arguments: Mapping[str, object], /
    ) -> PreparedQuantityField:
        del arguments
        values = self.trace.apply(fields[(self.component, self.field)])
        weights = self.trace.weights
        match self.statistic:
            case "trace":
                samples = values.reshape((-1,))
            case "integral":
                samples = jnp.sum(weights * values).reshape((1,))
            case "average":
                samples = (jnp.sum(weights * values) / jnp.sum(weights)).reshape((1,))
            case _:
                raise ValueError(f"Unknown boundary statistic {self.statistic!r}.")
        valid = jnp.ones(samples.shape, dtype=jnp.bool_)
        return self.identity.field(self.binding_id, self.scale, samples, valid)


_STATISTIC_KINDS: dict[BoundaryStatistic, SpatialSamplingKind] = {
    "trace": SpatialSamplingKind.POINT,
    "average": SpatialSamplingKind.SURFACE_AVERAGE,
    "integral": SpatialSamplingKind.PATH_INTEGRAL,
}


@final
class FieldBoundaryObservation(AbstractObservationBinding, NonTrainableState):
    """Exact side trace of a component field on selected exterior facets.

    ``"trace"`` samples the trace at the facet quadrature sites of ``rule``
    (``POINT`` sampling on a ``PointSampleSupport`` at exactly those sites),
    ``"average"`` is the facet-measure mean (``SURFACE_AVERAGE``, the unit of
    the field), and ``"integral"`` the facet-measure integral of a
    one-dimensional boundary (``PATH_INTEGRAL``, unit ``field_unit *
    length_unit``). Traces are the owner's exact traces (virtual elements
    included), integrated by the rule's exact-degree quadrature. On a
    ``PointSampleSupport`` the length unit is that of its coordinate contract
    (``length_unit`` defaults to it and a differing declaration is refused); an
    integral over another support declares it.
    """

    binding_id: str = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)
    field: str = eqx.field(static=True)
    domain: IntegrationDomain
    statistic: BoundaryStatistic = eqx.field(static=True)
    rule: FacetTraceRule
    identity: MeasurementIdentity
    support_points: Array | None
    field_unit: UnitDefinition
    length_unit: UnitDefinition | None

    @checked
    def __init__(
        self,
        binding_id: str,
        component: str,
        field: str,
        domain: IntegrationDomain,
        /,
        *,
        statistic: BoundaryStatistic,
        rule: FacetTraceRule,
        quantity: QuantitySpec,
        support: SampleSupport,
        sampling: SamplingSemantics,
        field_unit: UnitDefinition,
        length_unit: UnitDefinition | None = None,
        layout: ValueLayout | None = None,
    ) -> None:
        statistic_ = parse(statistic, BoundaryStatistic, "statistic")
        if not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet":
            raise ValueError("A boundary observation acts on an exterior-facet domain.")
        identity = MeasurementIdentity(
            quantity,
            ValueLayout.scalar() if layout is None else layout,
            support,
            sampling,
        )
        identity.require_kind(_STATISTIC_KINDS[statistic_], f"a boundary {statistic_}")
        if statistic_ == "trace" and not isinstance(support, PointSampleSupport):
            raise TypeError("A trace observation samples a PointSampleSupport.")
        if statistic_ != "trace" and identity.sample_shape != (1,):
            raise ValueError(f"A boundary {statistic_} is one sample.")
        if isinstance(support, PointSampleSupport):
            length_unit = _coordinate_length_unit(support, length_unit)
        if statistic_ == "integral" and length_unit is None:
            raise ValueError("A boundary integral declares the length_unit.")
        self.binding_id = canonical_identifier(binding_id, "binding_id")
        self.components = (canonical_identifier(component, "component"),)
        self.field = canonical_identifier(field, "field")
        self.domain = domain
        self.statistic = statistic_
        self.rule = rule
        self.identity = identity
        self.support_points = (
            jnp.asarray(support.points)
            if isinstance(support, PointSampleSupport)
            else None
        )
        self.field_unit = field_unit
        self.length_unit = length_unit

    def _source_unit(self, facet_dimension: int, /) -> UnitDefinition:
        if self.statistic != "integral":
            return self.field_unit
        if facet_dimension != 1 or self.length_unit is None:
            raise ValueError(
                "Boundary integrals are path integrals over one-dimensional facets; "
                "no surface-integral sampling is declared."
            )
        return _power(self.field_unit, self.length_unit, facet_dimension)

    def prepare(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> PreparedBoundaryObservation:
        component = _component(
            components,
            self.components[0],
            self.field,
            AbstractTraceComponent,
            "side traces",
        )
        trace = component.prepare_side_trace(self.field, self.domain, rule=self.rule)
        sites = np.asarray(trace.sites)
        dimension = sites.shape[-1]
        if self.statistic == "trace":
            expected = sites.reshape((-1, dimension))
            points = np.asarray(self.support_points)
            if points.shape[0] != expected.shape[0] or not np.allclose(
                points[:, :dimension], expected, rtol=0.0, atol=1.0e-12
            ):
                raise ValueError(
                    "The support points are not the trace's facet quadrature sites."
                )
            if np.any(points[:, dimension:] != 0.0):
                raise ValueError("Support points leave the component's dimensions.")
        return PreparedBoundaryObservation(
            binding_id=self.binding_id,
            component=self.components[0],
            field=self.field,
            approximation=trace.descriptor.approximation,
            identity=self.identity,
            trace=trace,
            statistic=self.statistic,
            scale=self.identity.scale(
                self._source_unit(dimension - 1), f"a boundary {self.statistic}"
            ),
        )


# --- Flux content -------------------------------------------------------------------------


@final
class PreparedFluxObservation(AbstractPreparedObservation):
    binding_id: str = eqx.field(static=True)
    component: str = eqx.field(static=True)
    field: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    identity: MeasurementIdentity
    flux: PreparedFluxAction
    scale: float = eqx.field(static=True)

    @property
    def complete(self) -> bool:
        return True

    def evaluate(
        self, fields: FieldValues, arguments: Mapping[str, object], /
    ) -> PreparedQuantityField:
        reaction = self.flux.evaluate(
            fields[(self.component, self.field)], arguments[self.component]
        )
        samples = jnp.sum(reaction).reshape((1,))
        valid = jnp.ones((1,), dtype=jnp.bool_)
        return self.identity.field(self.binding_id, self.scale, samples, valid)


@final
class FieldFluxObservation(AbstractObservationBinding, NonTrainableState):
    """Outward conormal flux content of a component through selected facets.

    The owner's residual reaction on the facets' trace rows is paired with the
    discrete indicator ``w = sum_i phi_i`` of those rows, whose trace is one on
    the selected facets: ``<R(u), w>`` is the outward flux content through the
    facets plus the reaction-weighted flux on the one-element fringe where
    ``w`` decays. It equals the total outward flux exactly when the facets
    form a complete boundary part whose fringe carries no flux reaction (for
    example the whole boundary). The sampling is ``PATH_INTEGRAL`` and the
    quantity unit has the dimension of ``reaction_unit`` (flux content per
    residual row); it is never a flux density. The reaction is that of the
    steady residual, so it observes accepted steady solutions only; a
    transient reaction also carries the capacity action ``C u'``, and a
    coupled-transition observation port refuses this binding.
    """

    binding_id: str = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)
    field: str = eqx.field(static=True)
    domain: IntegrationDomain
    rule: FacetTraceRule
    identity: MeasurementIdentity
    reaction_unit: UnitDefinition

    @checked
    def __init__(
        self,
        binding_id: str,
        component: str,
        field: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: QuantitySpec,
        support: SampleSupport,
        sampling: SamplingSemantics,
        reaction_unit: UnitDefinition,
        layout: ValueLayout | None = None,
    ) -> None:
        if not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet":
            raise ValueError("A flux observation acts on an exterior-facet domain.")
        identity = MeasurementIdentity(
            quantity,
            ValueLayout.scalar() if layout is None else layout,
            support,
            sampling,
        )
        identity.require_kind(SpatialSamplingKind.PATH_INTEGRAL, "a flux content")
        if identity.sample_shape != (1,):
            raise ValueError("A flux content is one sample.")
        identity.scale(reaction_unit, "a flux content")
        self.binding_id = canonical_identifier(binding_id, "binding_id")
        self.components = (canonical_identifier(component, "component"),)
        self.field = canonical_identifier(field, "field")
        self.domain = domain
        self.rule = rule
        self.identity = identity
        self.reaction_unit = reaction_unit

    def prepare(
        self, components: Mapping[str, AbstractSpatialComponent], /
    ) -> PreparedFluxObservation:
        component = _component(
            components,
            self.components[0],
            self.field,
            AbstractTraceComponent,
            "conormal fluxes",
        )
        trace = component.prepare_side_trace(self.field, self.domain, rule=self.rule)
        if np.asarray(trace.sites).shape[-1] != 2:
            raise ValueError(
                "Flux content is a path integral over the one-dimensional boundary of "
                "a planar component."
            )
        flux = component.prepare_conormal_flux(trace)
        if flux.descriptor.representation != "residual-reaction":
            raise ValueError("Flux content pairs a residual-reaction flux.")
        return PreparedFluxObservation(
            binding_id=self.binding_id,
            component=self.components[0],
            field=self.field,
            approximation=flux.descriptor.approximation,
            identity=self.identity,
            flux=flux,
            scale=self.identity.scale(self.reaction_unit, "a flux content"),
        )


def prepare_observations(
    bindings: tuple[AbstractObservationBinding, ...],
    components: Mapping[str, AbstractSpatialComponent],
    /,
) -> tuple[AbstractPreparedObservation, ...]:
    """Lower every observation binding once through its owner's capabilities."""
    prepared = tuple(binding.prepare(components) for binding in bindings)
    for binding, observation in zip(bindings, prepared, strict=True):
        if not isinstance(observation, AbstractPreparedObservation):
            raise TypeError(
                f"Observation {binding.binding_id!r} did not prepare an "
                "AbstractPreparedObservation."
            )
    return prepared


__all__ = [
    "AbstractObservationBinding",
    "AbstractPreparedObservation",
    "BoundaryStatistic",
    "FieldBoundaryObservation",
    "FieldFluxObservation",
    "FieldPointObservation",
    "MeasurementIdentity",
    "PreparedBoundaryObservation",
    "PreparedFluxObservation",
    "PreparedPointObservation",
    "prepare_observations",
]
