#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ._realization import BasisRealization


MultiIndex = tuple[int, ...]
JetFunction = Callable[
    [BasisRealization, ArrayLike, "FixedParametricRoute", tuple[MultiIndex, ...]],
    ArrayLike,
]
TraceFunction = Callable[
    [
        BasisRealization,
        ArrayLike,
        "FixedParametricRoute",
        "SideEvidence",
        tuple[MultiIndex, ...],
    ],
    ArrayLike,
]


def _identifier(name: str, value: str) -> str:
    result = str(value)
    if not result:
        raise ValueError(f"{name} must be non-empty.")
    return result


@dataclass(frozen=True, slots=True)
class FixedParametricRoute:
    """Exact patch/cell route for a parameter point; no locator is consulted."""

    patch_id: str
    cell_id: str
    coordinates: tuple[float, ...]
    route_id: str

    def __init__(
        self,
        patch_id: str,
        cell_id: str,
        coordinates: Sequence[float],
        /,
        *,
        route_id: str | None = None,
    ):
        patch = _identifier("patch_id", patch_id)
        cell = _identifier("cell_id", cell_id)
        coordinate_values = tuple(float(value) for value in coordinates)
        if not coordinate_values or any(
            not jnp.isfinite(value) for value in coordinate_values
        ):
            raise ValueError("Parametric route coordinates must be finite and non-empty.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "fixed-parametric-route",
                    "patch": patch,
                    "cell": cell,
                    "coordinates": list(coordinate_values),
                }
            )
            if route_id is None
            else _identifier("route_id", route_id)
        )
        object.__setattr__(self, "patch_id", patch)
        object.__setattr__(self, "cell_id", cell)
        object.__setattr__(self, "coordinates", coordinate_values)
        object.__setattr__(self, "route_id", identifier)

    @property
    def parametric_dimension(self) -> int:
        return len(self.coordinates)


@dataclass(frozen=True, slots=True)
class InterfaceContinuity:
    """Certified regularity of one basis across one exact interface."""

    axis: int
    continuity_order: int
    negative_cell_id: str
    positive_cell_id: str
    certificate_id: str

    def __post_init__(self) -> None:
        axis = int(self.axis)
        order = int(self.continuity_order)
        if axis < 0 or order < -1:
            raise ValueError("Interface axis/order must be non-negative and at least -1.")
        negative = _identifier("negative_cell_id", self.negative_cell_id)
        positive = _identifier("positive_cell_id", self.positive_cell_id)
        if negative == positive:
            raise ValueError("An interface must separate two distinct cells.")
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "continuity_order", order)
        object.__setattr__(self, "negative_cell_id", negative)
        object.__setattr__(self, "positive_cell_id", positive)
        object.__setattr__(
            self, "certificate_id", _identifier("certificate_id", self.certificate_id)
        )


@dataclass(frozen=True, slots=True)
class ContinuityEvidence:
    """Complete interface regularity evidence at one fixed route."""

    realization_id: str
    route_id: str
    interfaces: tuple[InterfaceContinuity, ...]
    exhaustive: bool
    certificate_id: str

    def __post_init__(self) -> None:
        realization = _identifier("realization_id", self.realization_id)
        route = _identifier("route_id", self.route_id)
        interfaces = tuple(self.interfaces)
        if any(not isinstance(value, InterfaceContinuity) for value in interfaces):
            raise TypeError("interfaces must contain InterfaceContinuity values.")
        axes = tuple(value.axis for value in interfaces)
        if len(set(axes)) != len(axes):
            raise ValueError(
                "Continuity evidence may contain at most one interface per axis."
            )
        object.__setattr__(self, "realization_id", realization)
        object.__setattr__(self, "route_id", route)
        object.__setattr__(self, "interfaces", interfaces)
        object.__setattr__(self, "exhaustive", bool(self.exhaustive))
        object.__setattr__(
            self, "certificate_id", _identifier("certificate_id", self.certificate_id)
        )

    @classmethod
    def interior(
        cls, realization: BasisRealization, route: FixedParametricRoute, /
    ) -> ContinuityEvidence:
        if not isinstance(realization, BasisRealization):
            raise TypeError("realization must be a BasisRealization.")
        if not isinstance(route, FixedParametricRoute):
            raise TypeError("route must be a FixedParametricRoute.")
        certificate = canonical_fingerprint(
            {
                "kind": "interior-parametric-route",
                "realization": realization.realization_id,
                "route": route.route_id,
            }
        )
        return cls(realization.realization_id, route.route_id, (), True, certificate)


@dataclass(frozen=True, slots=True)
class SideEvidence:
    """Explicit one-sided route or outward boundary side used by a trace."""

    realization_id: str
    route_id: str
    cell_id: str
    axis: int
    side: int
    interface_certificate_id: str
    certificate_id: str
    boundary_occurrence_id: str | None = None
    outward_normal: tuple[float, ...] | None = None
    frame_id: str | None = None

    def __post_init__(self) -> None:
        realization = _identifier("realization_id", self.realization_id)
        route = _identifier("route_id", self.route_id)
        cell = _identifier("cell_id", self.cell_id)
        axis = int(self.axis)
        side = int(self.side)
        if axis < 0 or side not in (-1, 1):
            raise ValueError("Side evidence requires a non-negative axis and side ±1.")
        boundary = (
            None
            if self.boundary_occurrence_id is None
            else _identifier("boundary_occurrence_id", self.boundary_occurrence_id)
        )
        normal = (
            None
            if self.outward_normal is None
            else tuple(float(value) for value in self.outward_normal)
        )
        frame = None if self.frame_id is None else _identifier("frame_id", self.frame_id)
        if (boundary is None) != (normal is None) or (normal is None) != (frame is None):
            raise ValueError(
                "Boundary side evidence requires occurrence, outward normal, and frame."
            )
        if normal is not None:
            norm = float(jnp.linalg.norm(jnp.asarray(normal)))
            if not normal or not jnp.isfinite(norm) or norm <= 0.0:
                raise ValueError("Outward normal must be a finite nonzero vector.")
        object.__setattr__(self, "realization_id", realization)
        object.__setattr__(self, "route_id", route)
        object.__setattr__(self, "cell_id", cell)
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "side", side)
        object.__setattr__(
            self,
            "interface_certificate_id",
            _identifier("interface_certificate_id", self.interface_certificate_id),
        )
        object.__setattr__(
            self, "certificate_id", _identifier("certificate_id", self.certificate_id)
        )
        object.__setattr__(self, "boundary_occurrence_id", boundary)
        object.__setattr__(self, "outward_normal", normal)
        object.__setattr__(self, "frame_id", frame)


@dataclass(frozen=True, slots=True)
class BasisQueryProvider:
    """Basis-family adapter for fixed-route jet and conformity-aware trace actions."""

    provider_id: str
    jet_function: JetFunction
    trace_function: TraceFunction

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "provider_id", _identifier("provider_id", self.provider_id)
        )
        if not callable(self.jet_function) or not callable(self.trace_function):
            raise TypeError("Basis query provider functions must be callable.")


@dataclass(frozen=True, slots=True)
class ParametricJetResult:
    """Value/jet samples with the route and regularity proof that make them meaningful."""

    values: Array
    multi_indices: tuple[MultiIndex, ...]
    route: FixedParametricRoute
    continuity: ContinuityEvidence
    sides: tuple[SideEvidence, ...]
    realization_id: str
    provider_id: str
    query_id: str

    @property
    def value(self) -> Array:
        if self.multi_indices != ((0,) * self.route.parametric_dimension,):
            raise ValueError("This jet result contains more than the value component.")
        return self.values[0]


@dataclass(frozen=True, slots=True)
class ParametricTraceResult:
    """One conformity-aware boundary trace with explicit outward-side evidence."""

    values: Array
    multi_indices: tuple[MultiIndex, ...]
    route: FixedParametricRoute
    side: SideEvidence
    continuity: ContinuityEvidence
    realization_id: str
    provider_id: str
    query_id: str


@dataclass(frozen=True, slots=True)
class ParametricQueryPlan:
    """Arbitrary fixed-route value, jet, and trace queries for one BasisRealization."""

    realization: BasisRealization
    provider: BasisQueryProvider
    plan_id: str

    def __init__(self, realization: BasisRealization, provider: BasisQueryProvider, /):
        if not isinstance(realization, BasisRealization):
            raise TypeError("realization must be a BasisRealization.")
        if not isinstance(provider, BasisQueryProvider):
            raise TypeError("provider must be a BasisQueryProvider.")
        identifier = canonical_fingerprint(
            {
                "kind": "parametric-query-plan",
                "realization": realization.realization_id,
                "provider": provider.provider_id,
            }
        )
        object.__setattr__(self, "realization", realization)
        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "plan_id", identifier)

    def _indices(
        self,
        route: FixedParametricRoute,
        multi_indices: Sequence[Sequence[int]],
        /,
    ) -> tuple[MultiIndex, ...]:
        if not isinstance(route, FixedParametricRoute):
            raise TypeError("route must be a FixedParametricRoute.")
        indices = tuple(tuple(int(value) for value in item) for item in multi_indices)
        if not indices:
            raise ValueError("A parametric jet requires at least one multi-index.")
        if any(
            len(item) != route.parametric_dimension or any(value < 0 for value in item)
            for item in indices
        ):
            raise ValueError(
                "Jet multi-indices must be non-negative and match route dimension."
            )
        if len(set(indices)) != len(indices):
            raise ValueError("Jet multi-indices must be unique.")
        return indices

    def _evidence(
        self,
        route: FixedParametricRoute,
        indices: tuple[MultiIndex, ...],
        continuity: ContinuityEvidence,
        sides: Sequence[SideEvidence],
        /,
    ) -> tuple[SideEvidence, ...]:
        if not isinstance(continuity, ContinuityEvidence):
            raise TypeError("continuity must be ContinuityEvidence.")
        if (
            continuity.realization_id != self.realization.realization_id
            or continuity.route_id != route.route_id
        ):
            raise ValueError("Continuity evidence does not bind this realization/route.")
        if not continuity.exhaustive:
            raise ValueError("Continuity evidence must exhaust every interface at route.")
        side_values = tuple(sides)
        if any(not isinstance(value, SideEvidence) for value in side_values):
            raise TypeError("sides must contain SideEvidence values.")
        side_by_axis = {value.axis: value for value in side_values}
        if len(side_by_axis) != len(side_values):
            raise ValueError("At most one side may be selected per interface axis.")
        maximum_total_order = max(sum(item) for item in indices)
        for interface in continuity.interfaces:
            selected = side_by_axis.get(interface.axis)
            if maximum_total_order <= interface.continuity_order:
                if selected is not None:
                    self._validate_side(route, interface, selected)
                continue
            if selected is None:
                raise ValueError(
                    "Requested jet is not single-valued across an interface; select a side."
                )
            self._validate_side(route, interface, selected)
        for side in side_values:
            if side.axis not in {value.axis for value in continuity.interfaces}:
                raise ValueError("Side evidence names an interface absent from evidence.")
        return side_values

    def _validate_side(
        self,
        route: FixedParametricRoute,
        interface: InterfaceContinuity,
        side: SideEvidence,
        /,
    ) -> None:
        expected_cell = (
            interface.negative_cell_id if side.side < 0 else interface.positive_cell_id
        )
        if (
            side.realization_id != self.realization.realization_id
            or side.route_id != route.route_id
            or side.axis != interface.axis
            or side.cell_id != expected_cell
            or side.cell_id != route.cell_id
            or side.interface_certificate_id != interface.certificate_id
        ):
            raise ValueError("Side evidence does not certify the fixed route/interface.")

    def value(
        self,
        coefficients: ArrayLike,
        route: FixedParametricRoute,
        /,
        *,
        continuity: ContinuityEvidence,
        sides: Sequence[SideEvidence] = (),
    ) -> ParametricJetResult:
        zero = (0,) * route.parametric_dimension
        return self.jet(
            coefficients,
            route,
            (zero,),
            continuity=continuity,
            sides=sides,
        )

    def jet(
        self,
        coefficients: ArrayLike,
        route: FixedParametricRoute,
        multi_indices: Sequence[Sequence[int]],
        /,
        *,
        continuity: ContinuityEvidence,
        sides: Sequence[SideEvidence] = (),
    ) -> ParametricJetResult:
        indices = self._indices(route, multi_indices)
        side_values = self._evidence(route, indices, continuity, sides)
        values = jnp.asarray(
            self.provider.jet_function(self.realization, coefficients, route, indices)
        )
        if values.ndim == 0 or values.shape[0] != len(indices):
            raise ValueError(
                "Basis jet provider must return one leading entry per multi-index."
            )
        query_id = canonical_fingerprint(
            {
                "kind": "parametric-jet-query",
                "plan": self.plan_id,
                "route": route.route_id,
                "multi_indices": [list(value) for value in indices],
                "continuity": continuity.certificate_id,
                "sides": [value.certificate_id for value in side_values],
            }
        )
        return ParametricJetResult(
            values,
            indices,
            route,
            continuity,
            side_values,
            self.realization.realization_id,
            self.provider.provider_id,
            query_id,
        )

    def trace(
        self,
        coefficients: ArrayLike,
        route: FixedParametricRoute,
        side: SideEvidence,
        /,
        *,
        continuity: ContinuityEvidence,
        multi_indices: Sequence[Sequence[int]] | None = None,
    ) -> ParametricTraceResult:
        if not isinstance(side, SideEvidence):
            raise TypeError("side must be SideEvidence.")
        if (
            side.realization_id != self.realization.realization_id
            or side.route_id != route.route_id
            or side.cell_id != route.cell_id
        ):
            raise ValueError("Trace side evidence does not bind the fixed route.")
        if side.boundary_occurrence_id is None:
            raise ValueError(
                "Trace queries require outward-certified boundary side evidence."
            )
        indices = self._indices(
            route,
            ((0,) * route.parametric_dimension,)
            if multi_indices is None
            else multi_indices,
        )
        if (
            continuity.realization_id != self.realization.realization_id
            or continuity.route_id != route.route_id
            or not continuity.exhaustive
        ):
            raise ValueError("Trace continuity evidence does not bind this fixed route.")
        values = jnp.asarray(
            self.provider.trace_function(
                self.realization, coefficients, route, side, indices
            )
        )
        if values.ndim == 0 or values.shape[0] != len(indices):
            raise ValueError(
                "Basis trace provider must return one leading entry per multi-index."
            )
        query_id = canonical_fingerprint(
            {
                "kind": "parametric-trace-query",
                "plan": self.plan_id,
                "route": route.route_id,
                "multi_indices": [list(value) for value in indices],
                "continuity": continuity.certificate_id,
                "side": side.certificate_id,
                "boundary": side.boundary_occurrence_id,
            }
        )
        return ParametricTraceResult(
            values,
            indices,
            route,
            side,
            continuity,
            self.realization.realization_id,
            self.provider.provider_id,
            query_id,
        )


__all__ = [
    "BasisQueryProvider",
    "ContinuityEvidence",
    "FixedParametricRoute",
    "InterfaceContinuity",
    "MultiIndex",
    "ParametricJetResult",
    "ParametricQueryPlan",
    "ParametricTraceResult",
    "SideEvidence",
]
