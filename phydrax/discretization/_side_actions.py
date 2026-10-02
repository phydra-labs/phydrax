#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared side actions: traces, conormal fluxes, and imposition provenance.

A side action binds one owner's original coefficient space to a supported
trace or observation space on selected facets. It records the trace quantity,
representation, orientation, geometry revision, and exactness evidence, and it
keeps the primal action, the coordinate dual pullback (the exact transpose that
injects loads into the owner's residual rows), and the pairing-aware Hilbert
adjoint as distinct operations. Geometric value, normal, and tangential traces
are published by discretizations; conormal fluxes are published by compiled
physics owners from their physical operator and are never inferred from a
value trace. Providers implement small structural protocols; there is no
mandatory discretization superclass.
"""

from __future__ import annotations

import abc
from math import prod
from typing import assert_never, final, Literal, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, positive_integer
from ..exterior._form_type import FormType, FormValueSpec
from ..linalg import (
    AbstractLinearOperator,
    AbstractPairing,
    AbstractVectorSpace,
    AdjointLinearOperator,
    ArraySpace,
    DiagonalPairing,
    DualSpace,
    DualTransposeLinearOperator,
    FunctionLinearOperator,
    prepare_linearization,
    PreparedLinearization,
)
from ..typing import checked, ConvertibleToArray, parse
from ._integration_domain import IntegrationDomain
from ._reference_cell import (
    _NamedFacetShape,
    FacetShape,
    reference_cell_topology,
    ReferenceCellTopology,
)
from ._views import FieldTraceSide


SideTraceQuantity: TypeAlias = Literal["value", "normal", "tangential", "conormal-flux"]
SideRepresentation: TypeAlias = Literal[
    "quadrature-values", "cell-average", "face-state", "residual-reaction"
]
SideOrientation: TypeAlias = Literal["outward", "canonical", "unoriented"]
SideApproximation: TypeAlias = Literal[
    "exact", "h1-projection", "l2-projection", "variational-reaction"
]
FacetRuleFamily: TypeAlias = Literal["gauss-legendre", "gauss-lobatto-legendre"]
ImpositionKind: TypeAlias = Literal["strong", "natural", "robin", "weak"]
SideRouteMode: TypeAlias = Literal["componentwise", "contracted"]

_VALUE_AXES = "uvw"
_COMPONENT_AXES = "xyz"


def _sorted_rows(values: ConvertibleToArray, name: str, /) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim != 1 or not np.issubdtype(rows.dtype, np.integer):
        raise ValueError(f"{name} must be a rank-1 integer array.")
    rows = rows.astype(np.int32)
    if np.any(rows < 0) or np.any(np.diff(rows) <= 0):
        raise ValueError(f"{name} must be sorted, unique, and non-negative.")
    return rows


@final
class FacetTraceRule(StrictModule, NonTrainableState):
    """Reference quadrature on one facet shape for trace and load routes.

    Parameters live on the unit reference facet: a point has no parameter, an
    edge uses `t` in `[0, 1]`, a triangle uses `(s, t)` with `s, t >= 0` and
    `s + t <= 1`, and a quadrilateral uses `(u, v)` in `[0, 1]^2`. `points` is
    the number of points per parametric axis. Gauss--Lobatto--Legendre rules
    include the facet end points (the collocated nodes of nodal spectral
    elements) and are tensor rules, so triangular facets require
    Gauss--Legendre.
    """

    family: FacetRuleFamily = eqx.field(static=True)
    points: int = eqx.field(static=True)
    rule_id: str = eqx.field(static=True)

    def __init__(
        self, family: FacetRuleFamily = "gauss-legendre", /, *, points: int
    ) -> None:
        family = parse(family, FacetRuleFamily, "family")
        count = positive_integer(points, "points")
        if family == "gauss-lobatto-legendre" and count < 2:
            raise ValueError("Gauss-Lobatto-Legendre facet rules need two points.")
        self.family = family
        self.points = count
        self.rule_id = canonical_fingerprint(
            {"kind": "facet-trace-rule", "family": family, "points": count}
        )

    def _axis(self) -> tuple[np.ndarray, np.ndarray]:
        from ..integration import (
            GaussLegendreRule,
            GaussLobattoLegendreRule,
            interval_rule_data,
        )

        match self.family:
            case "gauss-legendre":
                data = interval_rule_data(GaussLegendreRule(self.points))
            case "gauss-lobatto-legendre":
                data = interval_rule_data(GaussLobattoLegendreRule(self.points))
            case _:
                assert_never(self.family)
        nodes = np.asarray(data.nodes, dtype=np.float64)
        weights = np.asarray(data.weights, dtype=np.float64)
        return 0.5 * (nodes + 1.0), 0.5 * weights

    def _descriptor_reference(
        self, shape: ReferenceCellTopology, /
    ) -> tuple[np.ndarray, np.ndarray]:
        from .._polynomial._cubature import simplex_rule_data, tensor_product_rule_data

        if shape != reference_cell_topology(shape.name):
            raise ValueError("Facet reference topology must be canonical.")
        dimension = shape.dimension
        if dimension == 0:
            return np.zeros((1, 0), dtype=np.float64), np.ones((1,), dtype=np.float64)
        if shape.name.startswith("simplex:") or shape.name in (
            "interval",
            "triangle",
            "tetrahedron",
        ):
            if self.family != "gauss-legendre":
                raise ValueError("Simplex facets use collapsed Gauss-Legendre rules.")
            degree = 2 * self.points - dimension
            if degree < 0:
                raise ValueError(
                    "Facet points are insufficient to integrate the simplex measure."
                )
            rule = simplex_rule_data(dimension, degree)
        elif shape.name.startswith("tensor:") or shape.name in (
            "quadrilateral",
            "hexahedron",
        ):
            match self.family:
                case "gauss-legendre":
                    rule = tensor_product_rule_data(
                        dimension, self.points, family="gauss"
                    )
                case "gauss-lobatto-legendre":
                    rule = tensor_product_rule_data(
                        dimension, self.points, family="lobatto"
                    )
                case _:
                    assert_never(self.family)
        else:
            raise ValueError(
                "Facet quadrature requires a simplex or tensor reference topology."
            )
        return np.asarray(rule.points, dtype=np.float64), np.asarray(
            rule.weights, dtype=np.float64
        )

    def reference(self, shape: FacetShape, /) -> tuple[np.ndarray, np.ndarray]:
        """Return host `(parameters, weights)` on the unit reference facet."""
        if isinstance(shape, ReferenceCellTopology):
            return self._descriptor_reference(shape)
        shape = parse(shape, _NamedFacetShape, "shape")
        match shape:
            case "point":
                return np.zeros((1, 0), dtype=np.float64), np.ones((1,), np.float64)
            case "edge":
                axis, weights = self._axis()
                return axis[:, None], weights
            case "quadrilateral":
                axis, weights = self._axis()
                first, second = np.meshgrid(axis, axis, indexing="ij")
                combined = weights[:, None] * weights[None, :]
                parameters = np.stack((first, second), axis=-1).reshape((-1, 2))
                return parameters, combined.reshape((-1,))
            case "triangle":
                if self.family != "gauss-legendre":
                    raise ValueError(
                        "Triangular facets use collapsed Gauss-Legendre rules; "
                        "Gauss-Lobatto-Legendre facet rules are tensor rules."
                    )
                axis, weights = self._axis()
                first, second = np.meshgrid(axis, axis, indexing="ij")
                parameters = np.stack((first, (1.0 - first) * second), axis=-1)
                combined = weights[:, None] * weights[None, :] * (1.0 - first)
                return parameters.reshape((-1, 2)), combined.reshape((-1,))
            case _:
                assert_never(shape)

    def exact_degree(self, shape: FacetShape, /) -> int | None:
        """Total polynomial degree integrated exactly (`None` for point facets)."""
        if isinstance(shape, ReferenceCellTopology):
            if shape != reference_cell_topology(shape.name):
                raise ValueError("Facet reference topology must be canonical.")
            if shape.dimension == 0:
                return None
            if shape.name.startswith("simplex:") or shape.name in (
                "interval",
                "triangle",
                "tetrahedron",
            ):
                if self.family != "gauss-legendre":
                    raise ValueError("Simplex facets use collapsed Gauss-Legendre rules.")
                degree = 2 * self.points - shape.dimension
                if degree < 0:
                    raise ValueError(
                        "Facet points are insufficient to integrate the simplex measure."
                    )
                return degree
            if shape.name.startswith("tensor:") or shape.name in (
                "quadrilateral",
                "hexahedron",
            ):
                match self.family:
                    case "gauss-legendre":
                        return 2 * self.points - 1
                    case "gauss-lobatto-legendre":
                        return 2 * self.points - 3
                    case _:
                        assert_never(self.family)
            raise ValueError(
                "Facet quadrature requires a simplex or tensor reference topology."
            )
        shape = parse(shape, _NamedFacetShape, "shape")
        match shape:
            case "point":
                return None
            case "edge" | "quadrilateral":
                match self.family:
                    case "gauss-legendre":
                        return 2 * self.points - 1
                    case "gauss-lobatto-legendre":
                        return 2 * self.points - 3
                    case _:
                        assert_never(self.family)
            case "triangle":
                return 2 * self.points - 2
            case _:
                assert_never(shape)


@final
class SideActionDescriptor(StrictModule, NonTrainableState):
    """Identity, semantics, and exactness evidence of one prepared side action.

    `owner_id` names the publishing owner (a prepared discretization or a
    compiled physics problem) and `field_space_id` its original coefficient
    space. `facets` are the selected facet entities of `entity_set_id` in the
    action's canonical order. `representation` states what the data at the
    sites mean: `"quadrature-values"` is the field's own trace, `"cell-average"`
    is the side cell's average repeated at every site of the facet (a
    first-order finite-volume face state), `"face-state"` is a reconstructed
    finite-volume face state at the sites, and `"residual-reaction"` a
    covector on owner residual rows. `revision_id` identifies the geometry
    realization the sites, measures, and normals were prepared on.
    `approximation` is `"exact"` only when the action evaluates the discrete
    field's own trace; `trace_degree` is the polynomial degree of that trace
    along each facet (`None` when not polynomial) and `quadrature_exact_degree`
    the exactness of the facet rule used for loads and pairings.
    """

    owner_id: str = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    quantity: SideTraceQuantity = eqx.field(static=True)
    representation: SideRepresentation = eqx.field(static=True)
    orientation: SideOrientation = eqx.field(static=True)
    approximation: SideApproximation = eqx.field(static=True)
    side: FieldTraceSide = eqx.field(static=True)
    domain_kind: str = eqx.field(static=True)
    entity_set_id: str = eqx.field(static=True)
    facets: Array
    revision_id: str = eqx.field(static=True)
    rule_id: str | None = eqx.field(static=True)
    trace_degree: int | None = eqx.field(static=True)
    quadrature_exact_degree: int | None = eqx.field(static=True)
    descriptor_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        owner_id: str,
        field_space_id: str,
        quantity: SideTraceQuantity,
        representation: SideRepresentation,
        orientation: SideOrientation,
        approximation: SideApproximation,
        side: FieldTraceSide,
        domain: IntegrationDomain,
        revision_id: str,
        rule: FacetTraceRule | None,
        trace_degree: int | None,
        quadrature_exact_degree: int | None,
    ) -> None:
        owner = canonical_identifier(owner_id, "owner_id")
        space = canonical_identifier(field_space_id, "field_space_id")
        revision = canonical_identifier(revision_id, "revision_id")
        quantity = parse(quantity, SideTraceQuantity, "quantity")
        representation = parse(representation, SideRepresentation, "representation")
        orientation = parse(orientation, SideOrientation, "orientation")
        approximation = parse(approximation, SideApproximation, "approximation")
        side = parse(side, FieldTraceSide, "side")
        match domain.kind:
            case "exterior_facet" | "interior_facet":
                pass
            case _:
                raise ValueError("Side actions act on exterior or interior facets.")
        facets = np.asarray(domain.entity_indices, dtype=np.int32)
        if facets.size == 0:
            raise ValueError("A side action requires at least one selected facet.")
        if rule is not None and not isinstance(rule, FacetTraceRule):
            raise TypeError("rule must be a FacetTraceRule or None.")
        for name, degree in (
            ("trace_degree", trace_degree),
            ("quadrature_exact_degree", quadrature_exact_degree),
        ):
            if degree is not None and (
                isinstance(degree, bool) or not isinstance(degree, int) or degree < 0
            ):
                raise ValueError(f"{name} must be a non-negative int or None.")
        if quantity == "conormal-flux" and orientation == "unoriented":
            raise ValueError("A conormal flux must declare its normal orientation.")
        self.owner_id = owner
        self.field_space_id = space
        self.quantity = quantity
        self.representation = representation
        self.orientation = orientation
        self.approximation = approximation
        self.side = side
        self.domain_kind = domain.kind
        self.entity_set_id = domain.entity_set_id
        self.facets = jnp.asarray(facets)
        self.revision_id = revision
        self.rule_id = None if rule is None else rule.rule_id
        self.trace_degree = trace_degree
        self.quadrature_exact_degree = quadrature_exact_degree
        self.descriptor_id = canonical_fingerprint(
            {
                "kind": "side-action-descriptor",
                "owner": owner,
                "field_space": space,
                "quantity": quantity,
                "representation": representation,
                "orientation": orientation,
                "approximation": approximation,
                "side": side,
                "domain_kind": domain.kind,
                "entity_set": domain.entity_set_id,
                "facets": array_tree_fingerprint(facets),
                "revision": revision,
                "rule": self.rule_id,
                "trace_degree": trace_degree,
                "quadrature_exact_degree": quadrature_exact_degree,
            }
        )


class AbstractSideRoute(StrictModule):
    """Owner-prepared primal gather and exact transpose scatter of one side."""

    @property
    @abc.abstractmethod
    def coefficient_shape(self) -> tuple[int, ...]:
        raise NotImplementedError

    @property
    def row_shape(self) -> tuple[int, ...]:
        """Leading coefficient axes whose C-order flat index is a coefficient row.

        Support rows and residual reactions index the coefficient array with
        these axes flattened; the remaining axes are components. A one-axis
        row layout is the default.
        """
        return self.coefficient_shape[:1]

    @property
    @abc.abstractmethod
    def output_shape(self) -> tuple[int, ...]:
        """`(facets, sites_per_facet, *value_shape)` of the trace data."""
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, coefficients: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def transpose(self, values: Array, /) -> Array:
        raise NotImplementedError


@final
class SideGatherRoute(AbstractSideRoute, NonTrainableState):
    """Per-facet gather, local contraction, and scatter-add transpose.

    `dofs` has shape `(facets, local)` and indexes the coefficient rows: the
    C-order flattening of the leading `row_shape` axes of the coefficient
    array (by default its leading axis alone), an exact layout map for owners
    whose coefficients carry a tensor row layout. Padded local slots carry zero
    weight. In `"componentwise"` mode the weights have shape
    `(facets, sites, local)` and act on every coefficient component alike
    (value traces). In `"contracted"` mode the weights have shape
    `(facets, sites, local, *value_shape, *component_shape)` and contract the
    coefficient components into the trace value (normal or tangential traces
    of vector fields). No global coefficient-by-site matrix is formed.
    """

    dofs: Array
    weights: Array
    mode: SideRouteMode = eqx.field(static=True)
    global_shape: tuple[int, ...] = eqx.field(static=True)
    trace_value_shape: tuple[int, ...] = eqx.field(static=True)
    gather_row_shape: tuple[int, ...] = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        dofs: ConvertibleToArray,
        weights: ConvertibleToArray,
        /,
        *,
        coefficient_shape: tuple[int, ...],
        mode: SideRouteMode = "componentwise",
        value_shape: tuple[int, ...] = (),
        row_shape: tuple[int, ...] | None = None,
    ) -> None:
        mode = parse(mode, SideRouteMode, "mode")
        routes = np.asarray(dofs)
        values = np.asarray(weights)
        shape = tuple(
            positive_integer(size, "coefficient_shape") for size in coefficient_shape
        )
        value_shape_ = tuple(
            positive_integer(size, "value_shape") for size in value_shape
        )
        if not shape:
            raise ValueError("coefficient_shape must have a leading row axis.")
        rows = (
            shape[:1]
            if row_shape is None
            else tuple(positive_integer(size, "row_shape") for size in row_shape)
        )
        if not rows or shape[: len(rows)] != rows:
            raise ValueError("row_shape must be the leading axes of coefficient_shape.")
        components = shape[len(rows) :]
        if routes.ndim != 2 or not np.issubdtype(routes.dtype, np.integer):
            raise ValueError("Side route dofs must be a (facets, local) integer array.")
        if np.any(routes < 0) or np.any(routes >= prod(rows)):
            raise ValueError("Side route dofs lie outside the coefficient rows.")
        if not np.issubdtype(values.dtype, np.floating) or not np.all(
            np.isfinite(values)
        ):
            raise ValueError("Side route weights must be finite real floating point.")
        match mode:
            case "componentwise":
                if value_shape_ != components:
                    raise ValueError(
                        "Componentwise routes keep the coefficient components as values."
                    )
                expected_rank = 3
            case "contracted":
                expected_rank = 3 + len(value_shape_) + len(components)
            case _:
                assert_never(mode)
        if (
            values.ndim != expected_rank
            or values.shape[0] != routes.shape[0]
            or values.shape[2] != routes.shape[1]
            or (mode == "contracted" and values.shape[3:] != value_shape_ + components)
        ):
            raise ValueError("Side route weights do not match dofs and value axes.")
        self.dofs = jnp.asarray(routes.astype(np.int32))
        self.weights = jnp.asarray(values)
        self.mode = mode
        self.global_shape = shape
        self.trace_value_shape = value_shape_
        self.gather_row_shape = rows
        self.route_id = canonical_fingerprint(
            {
                "kind": "side-gather-route",
                "mode": mode,
                "dofs": array_tree_fingerprint(routes.astype(np.int32)),
                "weights": array_tree_fingerprint(values),
                "coefficient_shape": list(shape),
                "row_shape": list(rows),
                "value_shape": list(value_shape_),
            }
        )

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return self.global_shape

    @property
    def row_shape(self) -> tuple[int, ...]:
        return self.gather_row_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (*self.weights.shape[:2], *self.trace_value_shape)

    def _subscripts(self) -> tuple[str, str]:
        values = _VALUE_AXES[: len(self.trace_value_shape)]
        components = _COMPONENT_AXES[
            : len(self.global_shape) - len(self.gather_row_shape)
        ]
        return values, components

    def _row_layout(self) -> tuple[int, ...]:
        """Coefficient shape with the row axes flattened into one leading axis."""
        return (
            prod(self.gather_row_shape),
            *self.global_shape[len(self.gather_row_shape) :],
        )

    def apply(self, coefficients: Array, /) -> Array:
        gathered = coefficients.reshape(self._row_layout())[self.dofs]
        match self.mode:
            case "componentwise":
                return ein.contract("fql,fl...->fq...", self.weights, gathered)
            case "contracted":
                values, components = self._subscripts()
                return ein.contract(
                    f"fql{values}{components},fl{components}->fq{values}",
                    self.weights,
                    gathered,
                )
            case _:
                assert_never(self.mode)

    def transpose(self, values: Array, /) -> Array:
        match self.mode:
            case "componentwise":
                payload = ein.contract("fql,fq...->fl...", self.weights, values)
            case "contracted":
                value_axes, components = self._subscripts()
                payload = ein.contract(
                    f"fql{value_axes}{components},fq{value_axes}->fl{components}",
                    self.weights,
                    values,
                )
            case _:
                assert_never(self.mode)
        zeros = jnp.zeros(self._row_layout(), dtype=payload.dtype)
        return zeros.at[self.dofs].add(payload).reshape(self.global_shape)


@final
class PreparedTraceAction(StrictModule, NonTrainableState):
    """Discretization-owned linear trace on selected facets.

    `apply` maps the owner's full coefficient array to trace data of shape
    `(facets, sites_per_facet, *value_shape)` at the physical `sites`.
    `dual_pullback` is the exact coordinate transpose from trace covectors to
    covectors on the owner's residual rows; `inject_load` pulls back a load
    density through the facet measure (`sum_q w_q g_q v(x_q)`), which is the
    work pairing of a boundary load with the trace. `hilbert_adjoint` requires
    an explicitly declared coefficient Riesz pairing; the trace space pairing
    is always the facet measure. `support_rows` are the coefficient rows on
    which the trace depends (the rows a reaction flux on this side lives on):
    C-order flat indices of the route's `row_shape` axes, which
    `flatten_rows`/`unflatten_rows` map exactly to and from the coefficient
    layout.
    """

    descriptor: SideActionDescriptor
    route: AbstractSideRoute
    coefficient_space: ArraySpace
    sites: Array
    weights: Array
    normals: Array
    support_rows: Array
    form: FormValueSpec | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        descriptor: SideActionDescriptor,
        route: AbstractSideRoute,
        coefficient_space: ArraySpace,
        /,
        *,
        sites: ConvertibleToArray,
        weights: ConvertibleToArray,
        normals: ConvertibleToArray,
        support_rows: ConvertibleToArray,
        form: FormValueSpec | None = None,
    ) -> None:
        if descriptor.quantity == "conormal-flux":
            raise ValueError(
                "Conormal fluxes are published by compiled physics owners as "
                "PreparedFluxAction values, not as geometric traces."
            )
        match descriptor.representation:
            case "quadrature-values" | "cell-average" | "face-state":
                pass
            case "residual-reaction":
                raise ValueError(
                    "Geometric traces are site values; residual reactions are "
                    "published as PreparedFluxAction values."
                )
            case _:
                assert_never(descriptor.representation)
        if route.coefficient_shape != coefficient_space.shape:
            raise ValueError("The side route acts on another coefficient shape.")
        sites_ = np.asarray(sites)
        weights_ = np.asarray(weights)
        normals_ = np.asarray(normals)
        output = route.output_shape
        if form is not None:
            if not isinstance(form, FormValueSpec):
                raise TypeError("form must be FormValueSpec or None.")
            if form.value_shape != output[2:]:
                raise ValueError("The trace form proxy must match the route value shape.")
        if sites_.ndim != 3 or sites_.shape[:2] != output[:2]:
            raise ValueError(
                "Sites must have shape (facets, sites_per_facet, dimension)."
            )
        if weights_.shape != output[:2] or normals_.shape != sites_.shape:
            raise ValueError("Measure weights and normals must match the sites.")
        if output[0] != descriptor.facets.shape[0]:
            raise ValueError("The route must produce one trace row per selected facet.")
        if not (
            np.all(np.isfinite(sites_))
            and np.all(np.isfinite(normals_))
            and np.all(np.isfinite(weights_))
            and np.all(weights_ > 0.0)
        ):
            raise ValueError("Side sites and normals must be finite; weights positive.")
        rows = _sorted_rows(support_rows, "support_rows")
        if rows.size == 0 or rows[-1] >= prod(route.row_shape):
            raise ValueError("support_rows must be nonempty coefficient rows.")
        self.descriptor = descriptor
        self.route = route
        self.coefficient_space = coefficient_space
        self.sites = jnp.asarray(sites_)
        self.weights = jnp.asarray(weights_)
        self.normals = jnp.asarray(normals_)
        self.support_rows = jnp.asarray(rows)
        self.form = form

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.route.output_shape[2:]

    @property
    def form_type(self) -> FormType | None:
        """Declared intrinsic form semantics, absent for non-form traces."""
        return None if self.form is None else self.form.form_type

    @property
    def row_shape(self) -> tuple[int, ...]:
        """Coefficient axes that `support_rows` index after C-order flattening."""
        return self.route.row_shape

    def flatten_rows(self, values: Array, /) -> Array:
        """View coefficient-shaped values with the row axes as one leading axis."""
        rows = self.row_shape
        return values.reshape((prod(rows), *self.coefficient_space.shape[len(rows) :]))

    def unflatten_rows(self, values: Array, /) -> Array:
        """Exact inverse of `flatten_rows` onto the coefficient layout."""
        return values.reshape(self.coefficient_space.shape)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self.route.output_shape

    @property
    def action_id(self) -> str:
        if self.form is None:
            return self.descriptor.descriptor_id
        return canonical_fingerprint(
            {
                "kind": "form-side-trace-action",
                "descriptor": self.descriptor.descriptor_id,
                "form": self.form.value_spec_id,
            }
        )

    def require_revision(self, revision_id: str, /) -> None:
        """Refuse a geometry realization other than the prepared one."""
        if revision_id != self.descriptor.revision_id:
            raise ValueError(
                "The side action was prepared on another geometry revision; "
                "prepare it again on the refreshed owner."
            )

    def _measure(self) -> Array:
        return self.weights.reshape(self.weights.shape + (1,) * len(self.value_shape))

    def apply(self, coefficients: ArrayLike, /) -> Array:
        """Evaluate the trace data of the owner coefficients."""
        return self.route.apply(self.coefficient_space.validate(coefficients))

    def dual_pullback(self, covector: ArrayLike, /) -> Array:
        """Exact coordinate transpose onto the owner's residual rows."""
        values = jnp.asarray(covector)
        if values.shape != self.output_shape:
            raise ValueError(f"Trace covectors must have shape {self.output_shape}.")
        return self.route.transpose(values)

    def inject_load(self, density: ArrayLike, /) -> Array:
        """Pull a load density back through the facet measure into residual rows."""
        values = jnp.asarray(density)
        if values.shape != self.output_shape:
            raise ValueError(f"Load densities must have shape {self.output_shape}.")
        return self.route.transpose(self._measure() * values)

    def trace_space(self) -> ArraySpace:
        """Trace-data space paired by the facet measure."""
        output = jax.eval_shape(self.route.apply, self.coefficient_space.structure())
        measure = jnp.broadcast_to(self._measure(), output.shape).astype(output.dtype)
        return ArraySpace(
            output.shape,
            dtype=output.dtype,
            pairing=DiagonalPairing(
                measure,
                pairing_id=canonical_fingerprint(
                    {"kind": "side-measure-pairing", "action": self.action_id}
                ),
            ),
        )

    def as_linear_operator(
        self, /, *, coefficient_pairing: AbstractPairing | None = None
    ) -> FunctionLinearOperator:
        """Matrix-free trace operator from the coefficient to the trace space.

        `transpose_mv` is the coordinate `dual_pullback`; `adjoint_mv` is the
        Hilbert adjoint relative to `coefficient_pairing` (Euclidean when
        omitted) and the facet measure.
        """
        source = ArraySpace(
            self.coefficient_space.shape,
            dtype=self.coefficient_space.dtype,
            pairing=coefficient_pairing,
        )
        target = self.trace_space()
        return FunctionLinearOperator(
            self.route.apply,
            source=source,
            target=target,
            transpose_action=self.route.transpose,
            operator_id=canonical_fingerprint(
                {
                    "kind": "prepared-trace-operator",
                    "action": self.action_id,
                    "source": source.space_id,
                    "target": target.space_id,
                }
            ),
        )

    def dual_pullback_operator(self) -> AbstractLinearOperator:
        """Coordinate transpose from trace covectors to owner residual rows."""
        return DualTransposeLinearOperator(self.as_linear_operator())

    @checked
    def hilbert_adjoint(
        self, coefficient_pairing: AbstractPairing, /
    ) -> AdjointLinearOperator:
        """Hilbert adjoint relative to a declared coefficient Riesz pairing."""
        return AdjointLinearOperator(
            self.as_linear_operator(coefficient_pairing=coefficient_pairing)
        )


class AbstractSideFluxEvaluator(StrictModule):
    """Physics-owned conormal flux of one side as a function of the full state."""

    @property
    @abc.abstractmethod
    def state_space(self) -> AbstractVectorSpace:
        """The owner's full (unconstrained) state space."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate(self, state: PyTree[Array], args: object, /) -> Array:
        raise NotImplementedError


@final
class PreparedFluxAction(StrictModule, NonTrainableState):
    """Physics-owned conormal flux on the facets of one prepared trace.

    `"residual-reaction"` fluxes are covectors on `trace.support_rows`: the
    owner's full weak residual restricted to the rows of the side, which equals
    the outward conormal flux tested against those basis functions at a state
    that satisfies the owner's remaining rows. `"quadrature-values"` fluxes
    are densities at `trace.sites`. The descriptor's approximation labels a
    projected flux explicitly; a flux is never derived from a value trace.
    """

    descriptor: SideActionDescriptor
    trace: PreparedTraceAction
    evaluator: AbstractSideFluxEvaluator

    @checked
    def __init__(
        self,
        descriptor: SideActionDescriptor,
        trace: PreparedTraceAction,
        evaluator: AbstractSideFluxEvaluator,
        /,
    ) -> None:
        if descriptor.quantity != "conormal-flux":
            raise ValueError("A flux action must declare the conormal-flux quantity.")
        if descriptor.field_space_id != trace.descriptor.field_space_id or not bool(
            np.array_equal(
                np.asarray(descriptor.facets), np.asarray(trace.descriptor.facets)
            )
        ):
            raise ValueError("The flux and its trace must share field space and facets.")
        if descriptor.revision_id != trace.descriptor.revision_id:
            raise ValueError("The flux and its trace must share one geometry revision.")
        self._output_shape(descriptor, trace)
        self.descriptor = descriptor
        self.trace = trace
        self.evaluator = evaluator

    @staticmethod
    def _output_shape(
        descriptor: SideActionDescriptor, trace: PreparedTraceAction, /
    ) -> tuple[int, ...]:
        match descriptor.representation:
            case "residual-reaction":
                return (
                    trace.support_rows.shape[0],
                    *trace.coefficient_space.shape[len(trace.row_shape) :],
                )
            case "quadrature-values":
                return trace.output_shape
            case "cell-average" | "face-state":
                raise ValueError(
                    "Flux actions are residual reactions or quadrature flux "
                    "densities; cell-average and face-state data are traces."
                )
            case _:
                assert_never(descriptor.representation)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._output_shape(self.descriptor, self.trace)

    @property
    def action_id(self) -> str:
        return self.descriptor.descriptor_id

    def evaluate(self, state: PyTree[Array], args: object = None, /) -> Array:
        """Evaluate the flux of one full owner state.

        The evaluator's output shape is checked against the representation when
        the flux is traced with its actual runtime arguments; owners whose
        coefficients read runtime arguments have no argument-free evaluation.
        """
        value = self.evaluator.evaluate(self.evaluator.state_space.validate(state), args)
        if value.shape != self.output_shape:
            raise ValueError(
                "The flux evaluator output does not match its representation."
            )
        return value

    def linearize(
        self, state: PyTree[Array], args: object = None, /
    ) -> PreparedLinearization:
        """Prepare the flux linearization at one full owner state."""
        return prepare_linearization(
            lambda value: self.evaluate(value, args),
            state,
            source=self.evaluator.state_space,
            target=ArraySpace(self.output_shape, dtype=self.trace.weights.dtype),
            linearization_id=canonical_fingerprint(
                {"kind": "prepared-flux-linearization", "action": self.action_id}
            ),
        )

    def embed(self, values: ArrayLike, /) -> Array:
        """Place a residual-reaction covector on the owner's full residual rows."""
        if self.descriptor.representation != "residual-reaction":
            raise ValueError(
                "Only residual-reaction fluxes live on owner rows; pair quadrature "
                "flux densities with a trace action's inject_load."
            )
        covector = jnp.asarray(values)
        if covector.shape != self.output_shape:
            raise ValueError(f"Reaction covectors must have shape {self.output_shape}.")
        zeros = jnp.zeros(self.trace.coefficient_space.shape, dtype=covector.dtype)
        rows = self.trace.flatten_rows(zeros).at[self.trace.support_rows].set(covector)
        return self.trace.unflatten_rows(rows)

    def residual_space(self) -> DualSpace:
        return DualSpace(self.trace.coefficient_space)


@final
class BoundaryImposition(StrictModule, NonTrainableState):
    """Provenance of one boundary law imposed by a compiled owner.

    `"strong"` impositions eliminate coefficient `rows` through a constraint
    map and lift; `"natural"` loads, `"robin"` terms, and `"weak"` (Nitsche or
    interior-penalty) terms act on the selected `facets` of `entity_set_id`.
    Either location may be absent when the owner only knows the other one.
    """

    kind: ImpositionKind = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    entity_set_id: str | None = eqx.field(static=True)
    facets: Array | None
    rows: Array | None
    imposition_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: ImpositionKind,
        /,
        *,
        field_space_id: str,
        source_id: str,
        entity_set_id: str | None = None,
        facets: ConvertibleToArray | None = None,
        rows: ConvertibleToArray | None = None,
    ) -> None:
        kind = parse(kind, ImpositionKind, "kind")
        space = canonical_identifier(field_space_id, "field_space_id")
        source = canonical_identifier(source_id, "source_id")
        if (facets is None) != (entity_set_id is None):
            raise ValueError("Imposed facets require their entity_set_id and vice versa.")
        entity_set = (
            None
            if entity_set_id is None
            else canonical_identifier(entity_set_id, "entity_set_id")
        )
        facet_values = (
            None if facets is None else np.unique(np.asarray(facets, dtype=np.int32))
        )
        row_values = None if rows is None else np.unique(np.asarray(rows, dtype=np.int32))
        if facet_values is None and row_values is None:
            raise ValueError("An imposition must locate its facets or its rows.")
        if kind == "strong" and row_values is None:
            raise ValueError("A strong imposition must name its constrained rows.")
        self.kind = kind
        self.field_space_id = space
        self.source_id = source
        self.entity_set_id = entity_set
        self.facets = None if facet_values is None else jnp.asarray(facet_values)
        self.rows = None if row_values is None else jnp.asarray(row_values)
        self.imposition_id = canonical_fingerprint(
            {
                "kind": "boundary-imposition",
                "imposition": kind,
                "field_space": space,
                "source": source,
                "entity_set": entity_set,
                "facets": None
                if facet_values is None
                else array_tree_fingerprint(facet_values),
                "rows": None
                if row_values is None
                else array_tree_fingerprint(row_values),
            }
        )

    @checked
    def overlaps(self, action: PreparedTraceAction, /) -> bool:
        """Whether this imposition already acts on the facets or rows of `action`."""
        if action.descriptor.field_space_id != self.field_space_id:
            return False
        if (
            self.facets is not None
            and self.entity_set_id == action.descriptor.entity_set_id
            and np.intersect1d(
                np.asarray(self.facets), np.asarray(action.descriptor.facets)
            ).size
        ):
            return True
        return bool(
            self.rows is not None
            and np.intersect1d(
                np.asarray(self.rows), np.asarray(action.support_rows)
            ).size
        )


class SideTraceProvider(Protocol):
    """Discretization that publishes geometric traces on selected facets."""

    def prepare_side_trace(
        self,
        field_name: str,
        domain: IntegrationDomain,
        /,
        *,
        rule: FacetTraceRule,
        quantity: SideTraceQuantity = "value",
        side: FieldTraceSide = "owner",
    ) -> PreparedTraceAction: ...


class ConormalFluxProvider(Protocol):
    """Compiled physics owner that publishes the conormal flux of its operator."""

    def prepare_conormal_flux(
        self, trace: PreparedTraceAction, /
    ) -> PreparedFluxAction: ...


class BoundaryImpositionProvider(Protocol):
    """Compiled owner that reports the provenance of its boundary impositions."""

    def boundary_impositions(self) -> tuple[BoundaryImposition, ...]: ...


__all__ = [
    "AbstractSideFluxEvaluator",
    "AbstractSideRoute",
    "BoundaryImposition",
    "BoundaryImpositionProvider",
    "ConormalFluxProvider",
    "FacetRuleFamily",
    "FacetTraceRule",
    "ImpositionKind",
    "PreparedFluxAction",
    "PreparedTraceAction",
    "SideActionDescriptor",
    "SideApproximation",
    "SideGatherRoute",
    "SideOrientation",
    "SideRepresentation",
    "SideRouteMode",
    "SideTraceProvider",
    "SideTraceQuantity",
]
