#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared discrete-field reconstructions viewed as domain functions.

A `PreparedFieldReconstruction` owns the exact coordinate evaluation of one
discrete field representation (finite-element tabulation, spectral synthesis,
interpolation, finite-volume reconstruction). A `DiscreteFieldFunctionView`
binds coefficients to an explicit, equivalent `GeometryDomain` and exposes the
reconstruction as a `DomainFunction` whose coordinate derivatives are the exact
reconstruction derivatives and fail closed beyond the evidenced order.
"""

from __future__ import annotations

import abc
import operator
from collections.abc import Callable
from enum import IntEnum
from functools import partial
from typing import Any, final, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jax.custom_derivatives import SymbolicZero
from jaxtyping import Array, ArrayLike

from .._differentiation import (
    _regularity_payload,
    DerivativeRegularity,
    GradientLevel,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model._ports import intrinsic_model_ports, ValuePort
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier


if TYPE_CHECKING:
    from ..domain import DomainFunction, GeometryDomain
    from ..domain._derivative import DerivativeRule
    from ..domain._domain import Domain
    from ..geometry import CompiledGeometry


FieldTraceSide: TypeAlias = Literal["owner", "neighbor", "average"]
FieldSupportCoverage: TypeAlias = Literal["complete", "partial"]
_TRACE_SIDES: tuple[FieldTraceSide, ...] = ("owner", "neighbor", "average")
_INVALID_QUERY_MESSAGE = (
    "Discrete field view query is invalid at one or more points (outside the "
    "reconstruction support, on a non-smooth locus without a bound trace side, "
    "unresolved trace side, ill-conditioned, or failed location); inspect "
    "DiscreteFieldFunctionView.query(...) evidence."
)


class FieldQueryStatus(IntEnum):
    """Pointwise status of one reconstruction query."""

    VALID = 0
    OUTSIDE_SUPPORT = 1
    SIDE_REQUIRED = 2
    SIDE_UNRESOLVED = 3
    ILL_CONDITIONED = 4
    NONFINITE = 5
    LOCATION_FAILED = 6


class InterpolationTransposeEvidence(StrictModule):
    """Duality evidence `<R c, w> = <c, R^T w>` for one linear reconstruction route."""

    primal_pairing: Array
    transpose_pairing: Array
    residual: Array
    scale: Array
    finite: Array
    valid: Array


def transpose_duality_evidence(
    values: Array,
    dual: Array,
    coefficients: Array,
    scattered: Array,
    /,
    *,
    tolerance: float,
) -> InterpolationTransposeEvidence:
    """Compare the primal and transpose pairings of one linear route.

    A real-valued route of complex coefficients is only R-linear; its JAX
    transpose pairs through the real part, `<R c, w> = Re(sum(c * R^T w))`.
    """
    primal = jnp.sum(values * dual)
    transpose = jnp.sum(coefficients * scattered)
    if jnp.iscomplexobj(coefficients) and not jnp.iscomplexobj(values):
        transpose = jnp.real(transpose)
    residual = primal - transpose
    scale = jnp.maximum(
        jnp.asarray(1.0, dtype=residual.real.dtype),
        jnp.maximum(jnp.abs(primal), jnp.abs(transpose)),
    )
    finite = jnp.all(jnp.isfinite(jnp.stack((primal, transpose, residual, scale))))
    valid = finite & (jnp.abs(residual) <= tolerance * scale)
    return InterpolationTransposeEvidence(
        primal, transpose, residual, scale, finite, valid
    )


@final
class FieldTracePolicy(StrictModule, NonTrainableState):
    """How one reconstruction resolves one-sided traces.

    `"single-valued"` reconstructions agree from every side up to their maximum
    derivative order, so a trace is the evaluation itself. `"cell-sided"`
    reconstructions have a non-smooth locus between cells; a trace side selects
    the owner cells, the neighbor cells, or the arithmetic average of every
    containing cell's limit.
    """

    kind: Literal["single-valued", "cell-sided"] = eqx.field(static=True)
    sides: tuple[FieldTraceSide, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: Literal["single-valued", "cell-sided"],
        /,
        *,
        sides: tuple[FieldTraceSide, ...] = _TRACE_SIDES,
    ):
        match kind:
            case "single-valued" | "cell-sided":
                pass
            case _:
                raise ValueError(
                    "Field trace policy kind must be 'single-valued' or 'cell-sided'."
                )
        sides_ = tuple(sides)
        if not sides_ or any(side not in _TRACE_SIDES for side in sides_):
            raise ValueError(f"Trace sides must be a nonempty subset of {_TRACE_SIDES}.")
        if len(set(sides_)) != len(sides_):
            raise ValueError("Trace sides must be unique.")
        ordered = tuple(side for side in _TRACE_SIDES if side in sides_)
        self.kind = kind
        self.sides = ordered
        self.policy_id = canonical_fingerprint(
            {"kind": "field-trace-policy", "policy": kind, "sides": list(ordered)}
        )


@final
class FieldSideBinding(StrictModule, NonTrainableState):
    """Validated one-sided trace selection for fixed trace sites.

    `cell_mask` restricts evaluation to the cells of the bound side (`None` for
    averages and single-valued reconstructions); `site_cells` records the side
    cell resolved at each site (`-1` when no single cell is selected).
    """

    side: FieldTraceSide = eqx.field(static=True)
    cell_mask: Array | None
    sites: Array
    site_cells: Array
    reconstruction_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        side: FieldTraceSide,
        cell_mask: ArrayLike | None,
        sites: ArrayLike,
        site_cells: ArrayLike,
        /,
        *,
        reconstruction_id: str,
    ):
        if side not in _TRACE_SIDES:
            raise ValueError(f"side must be one of {_TRACE_SIDES}.")
        sites_ = np.asarray(sites)
        cells = np.asarray(site_cells, dtype=np.int32)
        if sites_.ndim != 2 or sites_.shape[0] == 0:
            raise ValueError("Trace sites must be a nonempty (points, dimension) array.")
        if cells.shape != sites_.shape[:1]:
            raise ValueError("site_cells must provide one cell per trace site.")
        mask = None if cell_mask is None else np.asarray(cell_mask, dtype=np.bool_)
        if mask is not None and (mask.ndim != 1 or not np.any(mask)):
            raise ValueError("A trace cell mask must be a nonempty rank-1 selection.")
        identifier = canonical_identifier(reconstruction_id, "reconstruction_id")
        self.side = side
        self.cell_mask = None if mask is None else jnp.asarray(mask)
        self.sites = jnp.asarray(sites_)
        self.site_cells = jnp.asarray(cells)
        self.reconstruction_id = identifier
        self.binding_id = canonical_fingerprint(
            {
                "kind": "field-side-binding",
                "reconstruction": identifier,
                "side": side,
                "cell_mask": None if mask is None else array_tree_fingerprint(mask),
                "sites": array_tree_fingerprint(sites_),
                "site_cells": array_tree_fingerprint(cells),
            }
        )


class FieldQueryEvidence(StrictModule):
    """Pointwise status, conditioning, and support multiplicity of one query."""

    status: Array
    conditioning: Array
    support_count: Array
    kernel_id: str = eqx.field(static=True)

    def __init__(
        self,
        status: ArrayLike,
        conditioning: ArrayLike,
        support_count: ArrayLike,
        /,
        *,
        kernel_id: str,
    ):
        status_ = jnp.asarray(status, dtype=jnp.int32)
        conditioning_ = jnp.asarray(conditioning)
        count = jnp.asarray(support_count, dtype=jnp.int32)
        if status_.ndim != 1 or conditioning_.shape != status_.shape:
            raise ValueError("Query conditioning must have one value per point.")
        if count.shape != status_.shape:
            raise ValueError("Query support counts must have one value per point.")
        if not jnp.issubdtype(conditioning_.dtype, jnp.floating):
            raise TypeError("Query conditioning must be real floating point.")
        self.status = status_
        self.conditioning = conditioning_
        self.support_count = count
        self.kernel_id = canonical_identifier(kernel_id, "kernel_id")

    @property
    def valid(self) -> Array:
        return self.status == int(FieldQueryStatus.VALID)


class FieldQueryResult(StrictModule):
    """Reconstructed values with their pointwise query evidence."""

    values: Array
    evidence: FieldQueryEvidence

    @property
    def valid(self) -> Array:
        return self.evidence.valid


class AbstractFieldReconstructionKernel(StrictModule):
    """Family-owned exact coordinate evaluation of one discrete field.

    `locate(points, derivative, side)` prepares a query route for points of
    shape `(n, d)` and the coordinate multi-index `derivative` (one count per
    axis) together with pointwise `FieldQueryEvidence`; `apply(route,
    coefficients)` evaluates the exact reconstruction derivative with shape
    `(n, *value_shape)`; `transpose(route, cotangent)` is its exact algebraic
    transpose for coefficient-linear reconstructions.

    Status contract: with `side=None`, a point on a non-smooth locus (inside
    more than one cell) is valid only when the derivative order does not exceed
    the reconstruction's continuity, otherwise `SIDE_REQUIRED`. A bound
    `"owner"`/`"neighbor"` side restricts the containing cells to its cell mask
    and reports `SIDE_UNRESOLVED` unless one cell (or a set agreeing at the
    requested order) remains; `"average"` averages every containing cell's
    limit. Points outside the support are `OUTSIDE_SUPPORT`; the kernel never
    substitutes a nearest cell or source.
    """

    @property
    @abc.abstractmethod
    def kernel_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def cell_count(self) -> int:
        """Number of side-bindable cells; zero for single-valued reconstructions."""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def support_coverage(self) -> FieldSupportCoverage:
        """Whether every point of the support geometry has a defined evaluation."""
        raise NotImplementedError

    @abc.abstractmethod
    def locate(
        self,
        points: Array,
        derivative: tuple[int, ...],
        side: FieldSideBinding | None,
        /,
    ) -> tuple[Any, FieldQueryEvidence]:
        raise NotImplementedError

    @abc.abstractmethod
    def apply(self, route: Any, coefficients: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def transpose(self, route: Any, cotangent: Array, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def bind_side(
        self,
        sites: np.ndarray,
        side: FieldTraceSide,
        cell_ids: np.ndarray | None,
        /,
    ) -> tuple[np.ndarray | None, np.ndarray]:
        """Resolve `(cell_mask, site_cells)` for fixed trace sites on the host.

        Raise `ValueError` when a site is outside the support, when `cell_ids`
        do not contain their sites, or when an owner/neighbor side cannot be
        derived without explicit `cell_ids`.
        """
        raise NotImplementedError


@final
class PreparedFieldReconstruction(StrictModule, NonTrainableState):
    """Exact, evidenced coordinate reconstruction of one discrete field.

    `regularity` is the declared value regularity of the reconstruction (for
    example `C^0` piecewise degree-`k` for Lagrange `P_k` finite elements,
    smooth for spectral synthesis, `C^{-1}` degree zero for finite-volume cell
    averages). Coordinate derivatives are evaluated exactly up to
    `maximum_derivative_order`; higher orders raise `ValueError`.
    `support_geometry` is the explicit region the reconstruction covers; views
    bind only to an equivalent `GeometryDomain`.
    """

    kernel: AbstractFieldReconstructionKernel
    support_geometry: CompiledGeometry
    value_port: ValuePort
    regularity: DerivativeRegularity
    trace_policy: FieldTracePolicy
    value_shape: tuple[int, ...] = eqx.field(static=True)
    coefficient_shape: tuple[int, ...] = eqx.field(static=True)
    physical_dimension: int = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    coefficient_linear: bool = eqx.field(static=True)
    field_space_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    reconstruction_id: str = eqx.field(static=True)

    def __init__(
        self,
        kernel: AbstractFieldReconstructionKernel,
        /,
        *,
        support_geometry: CompiledGeometry,
        value_port: ValuePort,
        regularity: DerivativeRegularity,
        trace_policy: FieldTracePolicy,
        coefficient_shape: tuple[int, ...],
        physical_dimension: int,
        maximum_derivative_order: int,
        field_space_id: str,
        support_id: str,
        coefficient_linear: bool = True,
    ):
        from ..geometry import CompiledGeometry, GeometryKind

        if not isinstance(kernel, AbstractFieldReconstructionKernel):
            raise TypeError("kernel must be an AbstractFieldReconstructionKernel.")
        if not isinstance(support_geometry, CompiledGeometry):
            raise TypeError("support_geometry must be a CompiledGeometry.")
        if not isinstance(value_port, ValuePort):
            raise TypeError("value_port must be a ValuePort.")
        if not isinstance(regularity, DerivativeRegularity):
            raise TypeError("regularity must be a DerivativeRegularity.")
        if not isinstance(trace_policy, FieldTracePolicy):
            raise TypeError("trace_policy must be a FieldTracePolicy.")
        if not isinstance(coefficient_linear, bool):
            raise TypeError("coefficient_linear must be a bool.")
        dimension = _positive_int(physical_dimension, "physical_dimension")
        maximum = _nonnegative_int(maximum_derivative_order, "maximum_derivative_order")
        shape = tuple(
            _nonnegative_int(size, "coefficient_shape") for size in coefficient_shape
        )
        if not shape:
            raise ValueError("coefficient_shape must have at least one axis.")
        if support_geometry.kind is not GeometryKind.REGION:
            raise ValueError("support_geometry must be a region geometry.")
        if support_geometry.ambient_dimension != dimension:
            raise ValueError(
                "support_geometry ambient dimension must equal physical_dimension."
            )
        for order in range(1, maximum + 1):
            level, _ = regularity.admits_order(order)
            if level is GradientLevel.NONE:
                raise ValueError(
                    f"maximum_derivative_order={maximum} exceeds the orders admitted "
                    f"by the declared regularity (order {order} is degenerate)."
                )
        single_valued = regularity.continuity == "smooth" or (
            regularity.continuity >= maximum
        )
        match trace_policy.kind:
            case "single-valued":
                if not single_valued:
                    raise ValueError(
                        "A reconstruction whose continuity is below its maximum "
                        "derivative order requires a cell-sided trace policy."
                    )
                if kernel.cell_count != 0:
                    raise ValueError(
                        "Single-valued reconstructions must not declare trace cells."
                    )
            case "cell-sided":
                if kernel.cell_count <= 0:
                    raise ValueError("Cell-sided reconstructions must declare cells.")
            case _:
                raise ValueError(f"Unknown trace policy kind {trace_policy.kind!r}.")
        match kernel.support_coverage:
            case "complete" | "partial":
                pass
            case _:
                raise ValueError(
                    "Kernel support_coverage must be 'complete' or 'partial'."
                )
        space = canonical_identifier(field_space_id, "field_space_id")
        support = canonical_identifier(support_id, "support_id")
        self.kernel = kernel
        self.support_geometry = support_geometry
        self.value_port = value_port
        self.regularity = regularity
        self.trace_policy = trace_policy
        self.value_shape = value_port.event_shape
        self.coefficient_shape = shape
        self.physical_dimension = dimension
        self.maximum_derivative_order = maximum
        self.coefficient_linear = coefficient_linear
        self.field_space_id = space
        self.support_id = support
        self.reconstruction_id = canonical_fingerprint(
            {
                "kind": "prepared-field-reconstruction",
                "kernel": kernel.kernel_id,
                "support": support,
                "field_space": space,
                "value_port": value_port.port_id,
                "regularity": _regularity_payload(regularity),
                "trace_policy": trace_policy.policy_id,
                "coefficient_shape": list(shape),
                "physical_dimension": dimension,
                "maximum_derivative_order": maximum,
                "coefficient_linear": coefficient_linear,
            }
        )

    def derivative_index(self, derivative: tuple[int, ...] | None, /) -> tuple[int, ...]:
        """Return a validated coordinate multi-index (zeros for `None`)."""
        if derivative is None:
            return (0,) * self.physical_dimension
        index = tuple(derivative)
        if len(index) != self.physical_dimension:
            raise ValueError(
                f"derivative must give one order per coordinate axis "
                f"({self.physical_dimension})."
            )
        orders = tuple(_nonnegative_int(order, "derivative") for order in index)
        total = sum(orders)
        if total > self.maximum_derivative_order:
            raise ValueError(
                f"Coordinate derivative of order {total} exceeds the "
                f"maximum_derivative_order={self.maximum_derivative_order} of "
                f"reconstruction {self.kernel.kernel_id!r}."
            )
        return orders

    def validate_coefficients(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != self.coefficient_shape:
            raise ValueError(
                f"Coefficients must have shape {self.coefficient_shape}, "
                f"got {values.shape}."
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            raise TypeError("Coefficients must be an inexact array.")
        return values

    def _points(self, points: ArrayLike, /) -> Array:
        values = jnp.asarray(points)
        if values.ndim == 1 and self.physical_dimension == values.shape[0]:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.physical_dimension:
            raise ValueError(
                f"Query points must have shape (points, {self.physical_dimension})."
            )
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError("Query points must be real floating point.")
        return values

    def _side(self, side: FieldSideBinding | None, /) -> FieldSideBinding | None:
        if side is None:
            return None
        if not isinstance(side, FieldSideBinding):
            raise TypeError("side must be a FieldSideBinding or None.")
        if side.reconstruction_id != self.reconstruction_id:
            raise ValueError("The trace binding belongs to a different reconstruction.")
        return side

    def validity(
        self,
        points: ArrayLike,
        /,
        *,
        derivative: tuple[int, ...] | None = None,
        side: FieldSideBinding | None = None,
    ) -> FieldQueryEvidence:
        """Return pointwise query evidence without evaluating coefficients."""
        _, evidence = self.kernel.locate(
            self._points(points), self.derivative_index(derivative), self._side(side)
        )
        return evidence

    def evaluate(
        self,
        coefficients: ArrayLike,
        points: ArrayLike,
        /,
        *,
        side: FieldSideBinding | None = None,
    ) -> FieldQueryResult:
        return self.derivative(coefficients, points, None, side=side)

    def derivative(
        self,
        coefficients: ArrayLike,
        points: ArrayLike,
        derivative: tuple[int, ...] | None,
        /,
        *,
        side: FieldSideBinding | None = None,
    ) -> FieldQueryResult:
        """Evaluate the exact coordinate derivative `derivative` with evidence."""
        values = self.validate_coefficients(coefficients)
        route, evidence = self.kernel.locate(
            self._points(points), self.derivative_index(derivative), self._side(side)
        )
        return FieldQueryResult(self.kernel.apply(route, values), evidence)

    def bind_trace(
        self,
        points: ArrayLike,
        /,
        *,
        side: FieldTraceSide,
        cell_ids: ArrayLike | None = None,
    ) -> FieldSideBinding:
        """Validate one-sided trace sites and return their side binding."""
        if side not in self.trace_policy.sides:
            raise ValueError(
                f"Trace side {side!r} is not supported; expected one of "
                f"{self.trace_policy.sides}."
            )
        sites = np.asarray(points)
        if sites.ndim == 1 and sites.shape[0] == self.physical_dimension:
            sites = sites[None]
        if sites.ndim != 2 or sites.shape[1] != self.physical_dimension:
            raise ValueError(
                f"Trace sites must have shape (points, {self.physical_dimension})."
            )
        if not np.all(np.isfinite(sites)):
            raise ValueError("Trace sites must be finite.")
        cells = None if cell_ids is None else np.asarray(cell_ids, dtype=np.int32)
        if cells is not None and cells.shape != sites.shape[:1]:
            raise ValueError("cell_ids must provide one cell per trace site.")
        match self.trace_policy.kind:
            case "single-valued":
                if cells is not None:
                    raise ValueError(
                        "Single-valued reconstructions have no trace cells to select."
                    )
                mask, site_cells = None, np.full(sites.shape[0], -1, dtype=np.int32)
            case "cell-sided":
                mask, site_cells = self.kernel.bind_side(sites, side, cells)
            case _:
                raise ValueError(f"Unknown trace policy kind {self.trace_policy.kind!r}.")
        return FieldSideBinding(
            side, mask, sites, site_cells, reconstruction_id=self.reconstruction_id
        )

    def trace(
        self,
        coefficients: ArrayLike,
        points: ArrayLike,
        /,
        *,
        side: FieldTraceSide,
        cell_ids: ArrayLike | None = None,
        derivative: tuple[int, ...] | None = None,
    ) -> FieldQueryResult:
        """Evaluate a one-sided trace (or trace derivative) at fixed sites."""
        binding = self.bind_trace(points, side=side, cell_ids=cell_ids)
        return self.derivative(coefficients, binding.sites, derivative, side=binding)

    def transpose(
        self,
        points: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        derivative: tuple[int, ...] | None = None,
        side: FieldSideBinding | None = None,
    ) -> Array:
        """Apply the exact algebraic transpose (scatter) of one query route.

        Every query point must be valid: an invalid route (outside the support,
        unresolved side, ill-conditioned, ...) has no transpose and raises
        `ValueError` naming the invalid points and their status (traced queries
        fail at runtime).
        """
        route, dual = self._transpose_route(points, cotangent, derivative, side)
        return self.kernel.transpose(route, dual)

    def duality_evidence(
        self,
        coefficients: ArrayLike,
        points: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        derivative: tuple[int, ...] | None = None,
        side: FieldSideBinding | None = None,
        tolerance: float = 1.0e-10,
    ) -> InterpolationTransposeEvidence:
        """Check `<R c, w> = <c, R^T w>` on one shared, fully valid query route.

        Invalid query points raise `ValueError` as in `transpose`.
        """
        values = self.validate_coefficients(coefficients)
        route, dual = self._transpose_route(points, cotangent, derivative, side)
        limit = float(tolerance)
        if not np.isfinite(limit) or limit < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        return transpose_duality_evidence(
            self.kernel.apply(route, values),
            dual,
            values,
            self.kernel.transpose(route, dual),
            tolerance=limit,
        )

    def _transpose_route(
        self,
        points: ArrayLike,
        cotangent: ArrayLike,
        derivative: tuple[int, ...] | None,
        side: FieldSideBinding | None,
        /,
    ) -> tuple[Any, Array]:
        if not self.coefficient_linear:
            raise ValueError(
                "The reconstruction is nonlinear in its coefficients and has no "
                "algebraic transpose; differentiate it under its branch policy."
            )
        query = self._points(points)
        route, evidence = self.kernel.locate(
            query, self.derivative_index(derivative), self._side(side)
        )
        dual = jnp.asarray(cotangent)
        if dual.shape != (query.shape[0], *self.value_shape):
            raise ValueError(
                "Cotangent must have shape (points, *value_shape) = "
                f"{(query.shape[0], *self.value_shape)}."
            )
        return route, _checked_queries(dual, evidence)


def _positive_int(value: Any, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _nonnegative_int(value: Any, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must contain ints.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return int(value)


def _checked_queries(value: Array, evidence: FieldQueryEvidence, /) -> Array:
    """Return `value` only when every query point of `evidence` is valid.

    Concrete evidence raises `ValueError` naming the invalid points and their
    status; traced evidence fails through a runtime check.
    """
    if isinstance(evidence.status, jax_core.Tracer):
        return eqx.error_if(value, ~evidence.valid, _INVALID_QUERY_MESSAGE)
    valid = np.asarray(evidence.valid)
    if not bool(np.all(valid)):
        invalid = np.flatnonzero(~valid)
        statuses = sorted(
            {
                FieldQueryStatus(int(status)).name
                for status in np.asarray(evidence.status)[invalid]
            }
        )
        raise ValueError(
            f"{_INVALID_QUERY_MESSAGE} Invalid points {invalid.tolist()[:8]} "
            f"with status {', '.join(statuses)}."
        )
    return value


@partial(jax.custom_jvp, nondiff_argnums=(0, 1))
def _field_values(
    static: Any,
    derivative: tuple[int, ...],
    dynamic: Any,
    coefficients: Array,
    points: Array,
    /,
) -> Array:
    reconstruction, side = eqx.combine(dynamic, static)
    route, evidence = reconstruction.kernel.locate(points, derivative, side)
    values = reconstruction.kernel.apply(route, coefficients)
    return eqx.error_if(values, ~evidence.valid, _INVALID_QUERY_MESSAGE)


def _field_values_jvp(
    static: Any,
    derivative: tuple[int, ...],
    primals: tuple[Any, Array, Array],
    tangents: tuple[Any, Any, Any],
) -> tuple[Array, Array]:
    dynamic, coefficients, points = primals
    dynamic_tangent, coefficient_tangent, point_tangent = tangents
    if any(
        not isinstance(leaf, SymbolicZero)
        for leaf in jax.tree.leaves(
            dynamic_tangent, is_leaf=lambda item: isinstance(item, SymbolicZero)
        )
    ):
        raise ValueError(
            "Discrete field reconstruction data is FIXED; differentiate the "
            "coefficients or the query coordinates only."
        )
    reconstruction, side = eqx.combine(dynamic, static)
    route, evidence = reconstruction.kernel.locate(points, derivative, side)
    values = eqx.error_if(
        reconstruction.kernel.apply(route, coefficients),
        ~evidence.valid,
        _INVALID_QUERY_MESSAGE,
    )
    tangent = jnp.zeros_like(values)
    if not isinstance(coefficient_tangent, SymbolicZero):
        tangent = (
            tangent
            + jax.jvp(
                lambda current: reconstruction.kernel.apply(route, current),
                (coefficients,),
                (coefficient_tangent,),
            )[1]
        )
    if not isinstance(point_tangent, SymbolicZero):
        order = sum(derivative) + 1
        if order > reconstruction.maximum_derivative_order:
            raise ValueError(
                f"Coordinate derivative of order {order} exceeds the "
                f"maximum_derivative_order={reconstruction.maximum_derivative_order} "
                f"of reconstruction {reconstruction.kernel.kernel_id!r}."
            )
        value_axes = (None,) * len(reconstruction.value_shape)
        for axis in range(reconstruction.physical_dimension):
            raised = tuple(
                count + (index == axis) for index, count in enumerate(derivative)
            )
            partial_values = _field_values(static, raised, dynamic, coefficients, points)
            tangent = (
                tangent + partial_values * point_tangent[(slice(None), axis, *value_axes)]
            )
    return values, tangent


_field_values.defjvp(_field_values_jvp, symbolic_zeros=True)


@final
class DiscreteFieldEvaluator(StrictModule, NonTrainableState):
    """Coordinate evaluator of one discrete field view and one derivative.

    Generic automatic differentiation of this evaluator with respect to its
    coordinates evaluates the exact reconstruction derivative and raises
    `ValueError` beyond `maximum_derivative_order`; reconstruction data is FIXED.
    Invalid queries fail at runtime instead of returning a plausible value.
    """

    reconstruction: PreparedFieldReconstruction
    coefficients: Array
    side: FieldSideBinding | None
    derivative: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedFieldReconstruction,
        coefficients: ArrayLike,
        /,
        *,
        side: FieldSideBinding | None = None,
        derivative: tuple[int, ...] | None = None,
    ):
        if not isinstance(reconstruction, PreparedFieldReconstruction):
            raise TypeError("reconstruction must be a PreparedFieldReconstruction.")
        self.reconstruction = reconstruction
        self.coefficients = reconstruction.validate_coefficients(coefficients)
        self.side = reconstruction._side(side)
        self.derivative = reconstruction.derivative_index(derivative)

    @property
    def order(self) -> int:
        return sum(self.derivative)

    @property
    def regularity(self) -> DerivativeRegularity:
        """Declared value regularity of the evaluated derivative."""
        if self.order == 0:
            return self.reconstruction.regularity
        return self.reconstruction.regularity.differentiate(self.order)

    @property
    def value_port(self) -> ValuePort | None:
        """Value identity of an undifferentiated view; derivatives are undeclared."""
        return self.reconstruction.value_port if self.order == 0 else None

    def raised(self, axis: int, order: int, /) -> DiscreteFieldEvaluator:
        derivative = tuple(
            count + (order if index == axis else 0)
            for index, count in enumerate(self.derivative)
        )
        return DiscreteFieldEvaluator(
            self.reconstruction,
            self.coefficients,
            side=self.side,
            derivative=derivative,
        )

    def derivative_rule_for(self, function: DomainFunction, /) -> DerivativeRule | None:
        # A structural `DerivativeRuleProvider`: the exact view rule is rebuilt
        # from this evaluator, so the field holds its reconstruction data once.
        from ..domain import CallbackDerivativeRule

        if len(function.deps) != 1:
            return None
        return CallbackDerivativeRule(
            _FieldViewDerivative(self, function.domain, function.deps[0])
        )

    def __call__(self, x: Any, /, *, key: Any = None, **kwargs: Any) -> Array:
        del key, kwargs
        dimension = self.reconstruction.physical_dimension
        if isinstance(x, tuple):
            if len(x) != dimension:
                raise ValueError(
                    f"Coordinate-separable view queries need {dimension} axes."
                )
            axes = tuple(jnp.asarray(coordinate).reshape((-1,)) for coordinate in x)
            points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        else:
            points = jnp.asarray(x)
            if points.ndim == 0:
                if dimension != 1:
                    raise ValueError("Scalar view queries require a 1D reconstruction.")
                points = points.reshape((1,))
        if points.shape[-1] != dimension:
            raise ValueError(f"View query points must end in dimension {dimension}.")
        leading = points.shape[:-1]
        flat = points.reshape((-1, dimension))
        if not isinstance(flat, jax_core.Tracer):
            # Eager queries are concrete: report invalid queries as a ValueError.
            # Traced queries fail through the runtime check in `_field_values`.
            _checked_queries(
                flat,
                self.reconstruction.validity(
                    flat, derivative=self.derivative, side=self.side
                ),
            )
        dynamic, static = eqx.partition((self.reconstruction, self.side), eqx.is_array)
        values = _field_values(static, self.derivative, dynamic, self.coefficients, flat)
        return values.reshape((*leading, *self.reconstruction.value_shape))


@final
class _FieldViewDerivative(StrictModule):
    """Exact derivative rule of one view.

    Derivatives along the bound variable are always the reconstruction's own
    derivative (or a `ValueError`); the rule never defers them to generic
    lowering. Only a variable the view does not depend on returns `None`, whose
    generic lowering is the identically zero derivative by independence.
    """

    evaluator: DiscreteFieldEvaluator
    domain: Domain
    variable: str = eqx.field(static=True)

    def __call__(
        self,
        *,
        var: str,
        axis: int | None,
        order: int,
        mode: str,
        backend: str,
        basis: str,
        periodic: bool,
    ) -> DomainFunction | None:
        del mode, basis, periodic

        match backend:
            case "ad" | "jet":
                pass
            case _:
                raise ValueError(
                    f"Discrete field views differentiate exactly; backend {backend!r} "
                    "is not a reconstruction derivative route."
                )
        if var != self.variable:
            return None
        dimension = self.evaluator.reconstruction.physical_dimension
        if axis is None:
            if dimension != 1:
                raise ValueError("A vector view variable requires an explicit axis.")
            axis_ = 0
        else:
            axis_ = int(axis)
            if not 0 <= axis_ < dimension:
                raise ValueError(f"axis must be in [0, {dimension}).")
        return _view_function(
            self.evaluator.raised(axis_, int(order)), self.domain, self.variable
        )


def _view_function(
    evaluator: DiscreteFieldEvaluator, domain: Domain, variable: str, /
) -> DomainFunction:
    from ..domain import DomainFunction

    return DomainFunction(domain=domain, deps=(variable,), func=evaluator)


@final
class DiscreteFieldFunctionView(StrictModule, NonTrainableState):
    """Discrete field coefficients bound to an equivalent geometry domain.

    The view requires an explicit `GeometryDomain` whose compiled geometry is
    equivalent to the reconstruction's `support_geometry`; discrete mesh or grid
    identity alone never establishes a domain.
    """

    reconstruction: PreparedFieldReconstruction
    coefficients: Array
    domain: GeometryDomain
    variable: str = eqx.field(static=True)
    field_name: str | None = eqx.field(static=True)

    def __init__(
        self,
        reconstruction: PreparedFieldReconstruction,
        coefficients: ArrayLike,
        domain: GeometryDomain,
        /,
        *,
        variable: str,
        field_name: str | None = None,
    ):
        from ..domain import GeometryDomain

        if not isinstance(reconstruction, PreparedFieldReconstruction):
            raise TypeError("reconstruction must be a PreparedFieldReconstruction.")
        if not isinstance(domain, GeometryDomain):
            raise TypeError("Discrete field views require an explicit GeometryDomain.")
        variable_ = canonical_identifier(variable, "variable")
        if variable_ not in domain.labels:
            raise ValueError(
                f"variable {variable_!r} is not a label of the domain {domain.labels}."
            )
        if domain.spatial_dim != reconstruction.physical_dimension:
            raise ValueError(
                "The domain dimension differs from the reconstruction dimension."
            )
        if not domain.geometry.equivalent(reconstruction.support_geometry):
            raise ValueError(
                "The GeometryDomain is not equivalent to the reconstruction's "
                "support_geometry; bind the view to the geometry the "
                "reconstruction was prepared on."
            )
        self.reconstruction = reconstruction
        self.coefficients = reconstruction.validate_coefficients(coefficients)
        self.domain = domain
        self.variable = variable_
        self.field_name = (
            None if field_name is None else canonical_identifier(field_name, "field_name")
        )

    def query(
        self,
        points: ArrayLike,
        /,
        *,
        derivative: tuple[int, ...] | None = None,
        side: FieldSideBinding | None = None,
    ) -> FieldQueryResult:
        """Evaluate the reconstruction with explicit pointwise evidence."""
        return self.reconstruction.derivative(
            self.coefficients, points, derivative, side=side
        )

    def as_domain_function(self) -> DomainFunction:
        """Return the view as a `DomainFunction` with an exact derivative rule."""
        if self.reconstruction.kernel.support_coverage != "complete":
            raise ValueError(
                "The reconstruction does not define an evaluation at every point of "
                "its support geometry; query it with explicit evidence instead."
            )
        return _view_function(
            DiscreteFieldEvaluator(self.reconstruction, self.coefficients),
            self.domain,
            self.variable,
        )

    def trace(
        self,
        points: ArrayLike,
        /,
        *,
        side: FieldTraceSide,
        cell_ids: ArrayLike | None = None,
    ) -> DomainFunction:
        """Return a side-bound `DomainFunction` validated at fixed trace sites.

        `points` are the trace sites (for example interface facet quadrature
        points). `"owner"`/`"neighbor"` traces evaluate the limits from the cells
        of that side (`cell_ids` names the side cell of each site when the
        reconstruction cannot derive it); `"average"` averages every containing
        cell's limit. Derivatives of the returned field use the bound side.
        """
        binding = self.reconstruction.bind_trace(points, side=side, cell_ids=cell_ids)
        return _view_function(
            DiscreteFieldEvaluator(self.reconstruction, self.coefficients, side=binding),
            self.domain,
            self.variable,
        )


_CONSTANT_PORT = object()


def _declared_value_port(field: DomainFunction, /) -> ValuePort | object | None:
    from ..domain._function import _ConstCallable, _TrainableConstCallable
    from ..domain._model_function import ConcatenatedModelEvaluator

    function = field.func
    if not field.deps or isinstance(function, (_ConstCallable, _TrainableConstCallable)):
        return _CONSTANT_PORT
    if isinstance(function, DiscreteFieldEvaluator):
        return function.value_port
    if isinstance(function, ConcatenatedModelEvaluator):
        ports = intrinsic_model_ports(function.raw_model)
        if ports is not None and len(ports.outputs) == 1:
            return ports.outputs[0]
    return None


def _port_mismatches(left: ValuePort, right: ValuePort, /) -> tuple[str, ...]:
    mismatches = []
    if left.dimensions is None or right.dimensions is None:
        mismatches.append("units (undeclared)")
    elif tuple(item.dimension_id for item in left.dimensions) != tuple(
        item.dimension_id for item in right.dimensions
    ):
        mismatches.append("units")
    identities = (
        ("event_shape", left.event_shape, right.event_shape),
        ("frame_id", left.frame_id, right.frame_id),
        ("axis_keys", left.axis_keys, right.axis_keys),
        ("normalization_id", left.normalization_id, right.normalization_id),
        ("variance", left.variance, right.variance),
    )
    mismatches.extend(name for name, first, second in identities if first != second)
    return tuple(mismatches)


def require_compatible_field_composition(
    left: DomainFunction,
    right: DomainFunction,
    op: Callable[[Any, Any], Any],
    /,
) -> None:
    """Refuse pointwise algebra that mixes a view with an incompatible field.

    Domain joining already refuses operands whose supports collide; an operand
    depending on the view variable must additionally denote the view's exact
    geometry support (traced joins only compare schemas). Sums and
    differences of an undifferentiated view additionally require the other
    operand to declare a value port with equal units, event shape, frame, axis
    identity, normalization, and variance; constants are read in the view's
    units. Derivative validity of the result is owned by each operand's exact
    derivative contract.
    """
    for view, other in ((left, right), (right, left)):
        evaluator = view.func
        if not isinstance(evaluator, DiscreteFieldEvaluator):
            continue
        (variable,) = view.deps
        if variable in other.deps and not other.domain.factor(variable).same_support(
            view.domain.factor(variable)
        ):
            raise ValueError(
                f"The operand's support for {variable!r} is not the view's geometry "
                "support; the operands would not share query points."
            )
        if op not in (operator.add, operator.sub) or evaluator.order != 0:
            continue
        other_port = _declared_value_port(other)
        if other_port is _CONSTANT_PORT:
            continue
        if other_port is None:
            raise ValueError(
                "Adding a discrete field view requires the other field to declare a "
                "value port (units, frame, axis identity); bind a port-declaring "
                "model or another view."
            )
        mismatches = _port_mismatches(evaluator.value_port, other_port)
        if mismatches:
            raise ValueError(
                "Discrete field view and operand value ports are incompatible: "
                f"{', '.join(mismatches)}."
            )


__all__ = [
    "AbstractFieldReconstructionKernel",
    "DiscreteFieldEvaluator",
    "DiscreteFieldFunctionView",
    "FieldQueryEvidence",
    "FieldQueryResult",
    "FieldQueryStatus",
    "FieldSideBinding",
    "FieldSupportCoverage",
    "FieldTracePolicy",
    "FieldTraceSide",
    "InterpolationTransposeEvidence",
    "PreparedFieldReconstruction",
    "require_compatible_field_composition",
    "transpose_duality_evidence",
]
