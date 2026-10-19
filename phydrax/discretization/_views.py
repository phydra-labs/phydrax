#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared discrete-field reconstructions viewed as domain functions.

A `PreparedFieldReconstruction` owns the coordinate evaluation of one
discrete field representation (finite-element tabulation, spectral synthesis,
interpolation, finite-volume reconstruction). A `DiscreteFieldFunctionView`
binds coefficients to an explicit, equivalent `GeometryDomain` and exposes the
reconstruction as a `DomainFunction` whose coordinate derivatives are the
reconstruction derivatives and fail closed beyond the evidenced order.
"""

from __future__ import annotations

import abc
import operator
from collections.abc import Callable
from enum import Enum, IntEnum
from functools import partial
from typing import Any, assert_never, final, get_args, Literal, TYPE_CHECKING, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.custom_derivatives import SymbolicZero
from jax.typing import ArrayLike, DTypeLike

from .. import domain as imported_domain, geometry as imported_geometry
from .._differentiation import (
    DerivativeRegularity,
    GradientLevel,
    regularity_payload,
)
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._model._ports import intrinsic_model_ports, ValuePort
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import canonical_identifier, nonnegative_integer, positive_integer
from ..domain import (
    _function as imported_domain__function,
    _model_function as imported_domain__model_function,
)
from ..typing import checked, parse


if TYPE_CHECKING:
    from ..domain import DomainFunction
    from ..domain._derivative import DerivativeRule
    from ..domain._domain import Domain
    from ._field_query import FieldQueryCoverage, PreparedFieldQuery


FieldTraceSide: TypeAlias = Literal["owner", "neighbor", "average"]
FieldSupportCoverage: TypeAlias = Literal["complete", "partial"]
FieldApproximation: TypeAlias = Literal["exact", "h1-projection", "l2-projection"]
type FieldTraceKind = Literal["single-valued", "cell-sided"]
_INVALID_QUERY_MESSAGE = (
    "DiscreteFieldFunctionView.query must select valid reconstruction points; "
    "got invalid support, trace, conditioning, or location evidence"
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

    kind: FieldTraceKind = eqx.field(static=True)
    sides: tuple[FieldTraceSide, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: FieldTraceKind,
        /,
        *,
        sides: tuple[FieldTraceSide, ...] = get_args(FieldTraceSide),
    ) -> None:
        kind = parse(kind, FieldTraceKind, "FieldTracePolicy.kind")
        sides_ = tuple(sides)
        if not sides_:
            raise ValueError(
                f"FieldTracePolicy.sides must be a nonempty subset of supported trace sides; got {sides!r}"
            )
        sides_ = tuple(parse(side, FieldTraceSide, "sides") for side in sides_)
        if len(set(sides_)) != len(sides_):
            raise ValueError(f"FieldTracePolicy.sides must be unique; got {sides_}")
        ordered = tuple(side for side in get_args(FieldTraceSide) if side in sides_)
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
    ) -> None:
        side = parse(side, FieldTraceSide, "side")
        sites_ = np.asarray(sites)
        cells = np.asarray(site_cells, dtype=np.int32)
        if sites_.ndim != 2 or sites_.shape[0] == 0:
            raise ValueError(
                "FieldSideBinding.sites must be a nonempty (points, dimension) array; got incompatible site shape"
            )
        if cells.shape != sites_.shape[:1]:
            raise ValueError(
                "FieldSideBinding.site_cells must provide one cell per trace site; got incompatible cell shape"
            )
        mask = None if cell_mask is None else np.asarray(cell_mask, dtype=np.bool_)
        if mask is not None and (mask.ndim != 1 or not np.any(mask)):
            raise ValueError(
                "FieldSideBinding.cell_mask must be a nonempty rank-1 selection; got invalid mask shape"
            )
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
    ) -> None:
        status_ = jnp.asarray(status, dtype=jnp.int32)
        conditioning_ = jnp.asarray(conditioning)
        count = jnp.asarray(support_count, dtype=jnp.int32)
        if status_.ndim != 1 or conditioning_.shape != status_.shape:
            raise ValueError(
                "FieldQueryEvidence.conditioning must have one value per point; got incompatible conditioning shape"
            )
        if count.shape != status_.shape:
            raise ValueError(
                "FieldQueryEvidence.support_count must have one value per point; got incompatible support-count shape"
            )
        if not jnp.issubdtype(conditioning_.dtype, jnp.floating):
            raise TypeError(
                "FieldQueryEvidence.conditioning must be real floating point; got another dtype"
            )
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
    """Family-owned coordinate evaluation of one discrete field.

    `locate(points, derivative, side)` prepares a query route for points of
    shape `(n, d)` and the coordinate multi-index `derivative` (one count per
    axis) together with pointwise `FieldQueryEvidence`; `apply(route,
    coefficients)` evaluates the reconstruction derivative with shape
    `(n, *value_shape)`; `transpose(route, cotangent)` is its algebraic
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
    """Reconstruct discrete-field coordinates with derivative and support evidence.

    `regularity` is the declared value regularity of the reconstruction (for
    example `C^0` piecewise degree-`k` for Lagrange `P_k` finite elements,
    smooth for spectral synthesis, `C^{-1}` degree zero for finite-volume cell
    averages). Coordinate derivatives are evaluated exactly up to
    `maximum_derivative_order`; higher orders raise `ValueError`.
    `support_geometry` is the explicit region or embedded manifold the
    reconstruction covers; views bind only to an equivalent `GeometryDomain`.
    `approximation` distinguishes evaluation of the discrete field itself from a
    labeled polynomial projection of it (for example the virtual-element
    `"h1-projection"` and `"l2-projection"` interior channels).
    `coefficient_dtype` is the native coefficient storage dtype (for example the
    complex modal storage of spectral fields); `None` declares real
    coefficients in the precision of the query points.
    """

    kernel: AbstractFieldReconstructionKernel
    support_geometry: imported_geometry.CompiledGeometry
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
    approximation: FieldApproximation = eqx.field(static=True)
    coefficient_dtype: np.dtype | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        kernel: AbstractFieldReconstructionKernel,
        /,
        *,
        support_geometry: imported_geometry.CompiledGeometry,
        value_port: ValuePort,
        regularity: DerivativeRegularity,
        trace_policy: FieldTracePolicy,
        coefficient_shape: tuple[int, ...],
        physical_dimension: int,
        maximum_derivative_order: int,
        field_space_id: str,
        support_id: str,
        coefficient_linear: bool = True,
        approximation: FieldApproximation = "exact",
        coefficient_dtype: DTypeLike | None = None,
    ) -> None:

        if not isinstance(coefficient_linear, bool):
            raise TypeError(
                f"PreparedFieldReconstruction.coefficient_linear must be a bool; got {coefficient_linear!r}"
            )
        approximation = parse(approximation, FieldApproximation, "approximation")
        native = None if coefficient_dtype is None else np.dtype(coefficient_dtype)
        if native is not None and not np.issubdtype(native, np.inexact):
            raise TypeError(
                f"PreparedFieldReconstruction.coefficient_dtype must be floating or complex; got {native}"
            )
        dimension = positive_integer(physical_dimension, "physical_dimension")
        maximum = nonnegative_integer(
            maximum_derivative_order, "maximum_derivative_order"
        )
        shape = tuple(
            nonnegative_integer(size, "coefficient_shape") for size in coefficient_shape
        )
        if not shape:
            raise ValueError(
                "PreparedFieldReconstruction.coefficient_shape must have at least one axis; got an empty shape"
            )
        if support_geometry.kind not in (
            imported_geometry.GeometryKind.REGION,
            imported_geometry.GeometryKind.MANIFOLD,
        ):
            raise ValueError(
                "PreparedFieldReconstruction.support_geometry must be a region or manifold; got another geometry kind"
            )
        if support_geometry.ambient_dimension != dimension:
            raise ValueError(
                "PreparedFieldReconstruction.support_geometry.ambient_dimension must equal physical_dimension; got a different ambient dimension"
            )
        for order in range(1, maximum + 1):
            level, _ = regularity.admits_order(order)
            if level is GradientLevel.NONE:
                raise ValueError(
                    f"PreparedFieldReconstruction.maximum_derivative_order must be admitted by declared regularity; got {maximum=} with degenerate order {order}"
                )
        single_valued = regularity.continuity == "smooth" or (
            regularity.continuity >= maximum
        )
        match trace_policy.kind:
            case "single-valued":
                if not single_valued:
                    raise ValueError(
                        "PreparedFieldReconstruction.trace_policy must be cell-sided below maximum derivative continuity; got a single-valued trace policy"
                    )
                if kernel.cell_count != 0:
                    raise ValueError(
                        "PreparedFieldReconstruction.kernel.cell_count must be zero for single-valued traces; got declared trace cells"
                    )
            case "cell-sided":
                if kernel.cell_count <= 0:
                    raise ValueError(
                        "PreparedFieldReconstruction.kernel.cell_count must be positive for cell-sided traces; got no declared cells"
                    )
            case invalid:
                assert_never(invalid)
        parse(
            kernel.support_coverage,
            FieldSupportCoverage,
            "PreparedFieldReconstruction.kernel.support_coverage",
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
        self.approximation = approximation
        self.coefficient_dtype = native
        self.field_space_id = space
        self.support_id = support
        self.reconstruction_id = canonical_fingerprint(
            {
                "kind": "prepared-field-reconstruction",
                "kernel": kernel.kernel_id,
                "support": support,
                "field_space": space,
                "value_port": value_port.port_id,
                "regularity": regularity_payload(regularity),
                "trace_policy": trace_policy.policy_id,
                "coefficient_shape": list(shape),
                "physical_dimension": dimension,
                "maximum_derivative_order": maximum,
                "coefficient_linear": coefficient_linear,
                "approximation": approximation,
                "coefficient_dtype": None if native is None else native.name,
            }
        )

    def derivative_index(self, derivative: tuple[int, ...] | None, /) -> tuple[int, ...]:
        """Return a validated coordinate multi-index (zeros for `None`)."""
        if derivative is None:
            return (0,) * self.physical_dimension
        index = tuple(derivative)
        if len(index) != self.physical_dimension:
            raise ValueError(
                f"PreparedFieldReconstruction.derivative must give one order per coordinate axis; got a different rank for dimension {self.physical_dimension}"
            )
        orders = tuple(nonnegative_integer(order, "derivative") for order in index)
        total = sum(orders)
        if total > self.maximum_derivative_order:
            raise ValueError(
                f"PreparedFieldReconstruction.maximum_derivative_order must admit the coordinate derivative; got order {total} above {self.maximum_derivative_order}"
            )
        return orders

    def validate_coefficients(self, coefficients: ArrayLike, /) -> Array:
        values = jnp.asarray(coefficients)
        if values.shape != self.coefficient_shape:
            raise ValueError(
                f"PreparedFieldReconstruction.coefficients must have shape {self.coefficient_shape}; got {values.shape}"
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            raise TypeError(
                "PreparedFieldReconstruction.coefficients must be an inexact array; got another coefficient dtype"
            )
        return values

    def _points(self, points: ArrayLike, /) -> Array:
        values = jnp.asarray(points)
        if values.ndim == 1 and self.physical_dimension == values.shape[0]:
            values = values[None]
        if values.ndim != 2 or values.shape[1] != self.physical_dimension:
            raise ValueError(
                f"PreparedFieldReconstruction.points must have shape (points, {self.physical_dimension}); got {values.shape}"
            )
        if not jnp.issubdtype(values.dtype, jnp.floating):
            raise TypeError(
                "PreparedFieldReconstruction.points must be real floating point; got another point dtype"
            )
        return values

    def _side(self, side: FieldSideBinding | None, /) -> FieldSideBinding | None:
        if side is None:
            return None
        if side.reconstruction_id != self.reconstruction_id:
            raise ValueError(
                "PreparedFieldReconstruction.side.reconstruction_id must match the reconstruction; got a different trace binding"
            )
        return side

    @checked
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

    @checked
    def evaluate(
        self,
        coefficients: ArrayLike,
        points: ArrayLike,
        /,
        *,
        side: FieldSideBinding | None = None,
    ) -> FieldQueryResult:
        return self.derivative(coefficients, points, None, side=side)

    @checked
    def derivative(
        self,
        coefficients: ArrayLike,
        points: ArrayLike,
        derivative: tuple[int, ...] | None,
        /,
        *,
        side: FieldSideBinding | None = None,
    ) -> FieldQueryResult:
        """Evaluate the coordinate derivative `derivative` with evidence."""
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
                f"PreparedFieldReconstruction.side must be supported by the trace policy; got {side!r}"
            )
        sites = np.asarray(points)
        if sites.ndim == 1 and sites.shape[0] == self.physical_dimension:
            sites = sites[None]
        if sites.ndim != 2 or sites.shape[1] != self.physical_dimension:
            raise ValueError(
                f"PreparedFieldReconstruction.trace_sites must have shape (points, {self.physical_dimension}); got {sites.shape}"
            )
        if not np.all(np.isfinite(sites)):
            raise ValueError(
                "PreparedFieldReconstruction.trace_sites must be finite; got nonfinite trace coordinates"
            )
        cells = None if cell_ids is None else np.asarray(cell_ids, dtype=np.int32)
        if cells is not None and cells.shape != sites.shape[:1]:
            raise ValueError(
                "PreparedFieldReconstruction.cell_ids must provide one cell per trace site; got incompatible cell shape"
            )
        match self.trace_policy.kind:
            case "single-valued":
                if cells is not None:
                    raise ValueError(
                        "PreparedFieldReconstruction.cell_ids must be absent for single-valued traces; got selected trace cells"
                    )
                mask, site_cells = None, np.full(sites.shape[0], -1, dtype=np.int32)
            case "cell-sided":
                mask, site_cells = self.kernel.bind_side(sites, side, cells)
            case invalid:
                assert_never(invalid)
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

    def prepare_query(
        self,
        points: ArrayLike,
        /,
        *,
        derivative: tuple[int, ...] | None = None,
        side: FieldTraceSide | None = None,
        cell_ids: ArrayLike | None = None,
        coverage: FieldQueryCoverage = "complete",
    ) -> PreparedFieldQuery:
        """Locate fixed points once and return their reusable prepared query.

        The query binds the points, derivative multi-index, trace side, and
        coverage policy to one located owner route; repeated `apply`/`transpose`
        calls with changing coefficients reuse that route without relocating.
        Invalid-point refusal or masking follows the coverage policy; masking admits the
        valid points only and retains the evidence of every requested point.
        """
        # Field views → prepared field queries → field-view validation.
        from ._field_query import PreparedFieldQuery

        if side is None and cell_ids is not None:
            raise ValueError(
                "PreparedFieldReconstruction.cell_ids must have a side selection; got None"
            )
        binding = (
            None
            if side is None
            else self.bind_trace(points, side=side, cell_ids=cell_ids)
        )
        return PreparedFieldQuery(
            self, points, derivative=derivative, side=binding, coverage=coverage
        )

    @checked
    def transpose(
        self,
        points: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        derivative: tuple[int, ...] | None = None,
        side: FieldSideBinding | None = None,
    ) -> Array:
        """Apply the algebraic transpose (scatter) of one query route.

        Every query point must be valid: an invalid route (outside the support,
        unresolved side, ill-conditioned, ...) has no transpose and raises
        `ValueError` naming the invalid points and their status (traced queries
        fail at runtime).
        """
        route, dual = self._transpose_route(points, cotangent, derivative, side)
        return self.kernel.transpose(route, dual)

    @checked
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
            raise ValueError(
                f"PreparedFieldReconstruction.tolerance must be finite and nonnegative; got {tolerance!r}"
            )
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
                "PreparedFieldReconstruction.coefficient_linear must be true for an algebraic transpose; got nonlinear reconstruction"
            )
        query = self._points(points)
        route, evidence = self.kernel.locate(
            query, self.derivative_index(derivative), self._side(side)
        )
        dual = jnp.asarray(cotangent)
        if dual.shape != (query.shape[0], *self.value_shape):
            raise ValueError(
                f"PreparedFieldReconstruction.cotangent must have shape (points, *value_shape); got an incompatible shape for {(query.shape[0], *self.value_shape)}"
            )
        return route, checked_queries(dual, evidence)


def checked_queries(value: Array, evidence: FieldQueryEvidence, /) -> Array:
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
            f"DiscreteFieldFunctionView.query must contain valid reconstruction points; got invalid indices {invalid.tolist()[:8]} with status {', '.join(statuses)}"
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
            "DiscreteFieldEvaluator.reconstruction must remain FIXED under differentiation; got a nonzero reconstruction-data tangent"
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
                f"DiscreteFieldEvaluator.maximum_derivative_order must admit the coordinate derivative; got order {order} above {reconstruction.maximum_derivative_order}"
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
    coordinates evaluates the reconstruction derivative and raises
    `ValueError` beyond `maximum_derivative_order`; reconstruction data is FIXED.
    Invalid queries fail at runtime instead of returning a plausible value.
    """

    reconstruction: PreparedFieldReconstruction
    coefficients: Array
    side: FieldSideBinding | None
    derivative: tuple[int, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reconstruction: PreparedFieldReconstruction,
        coefficients: ArrayLike,
        /,
        *,
        side: FieldSideBinding | None = None,
        derivative: tuple[int, ...] | None = None,
    ) -> None:
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
        # A structural `DerivativeRuleProvider`: the view rule is rebuilt
        # from this evaluator, so the field holds its reconstruction data once.

        if len(function.deps) != 1:
            return None
        return imported_domain.CallbackDerivativeRule(
            _FieldViewDerivative(self, function.domain, function.deps[0])
        )

    def __call__(self, x: Any, /, *, key: Any = None, **kwargs: Any) -> Array:
        del key, kwargs
        dimension = self.reconstruction.physical_dimension
        if isinstance(x, tuple):
            if len(x) != dimension:
                raise ValueError(
                    f"DiscreteFieldEvaluator.coordinates must provide one axis per physical dimension; got a different axis count for dimension {dimension}"
                )
            axes = tuple(jnp.asarray(coordinate).reshape((-1,)) for coordinate in x)
            points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
        else:
            points = jnp.asarray(x)
            if points.ndim == 0:
                if dimension != 1:
                    raise ValueError(
                        "DiscreteFieldEvaluator.coordinates must use a 1-D reconstruction for scalar queries; got another physical dimension"
                    )
                points = points.reshape((1,))
        if points.shape[-1] != dimension:
            raise ValueError(
                f"DiscreteFieldEvaluator.points must end in dimension {dimension}; got another trailing dimension"
            )
        leading = points.shape[:-1]
        flat = points.reshape((-1, dimension))
        if not isinstance(flat, jax_core.Tracer):
            # Eager queries are concrete: report invalid queries as a ValueError.
            # Traced queries fail through the runtime check in `_field_values`.
            checked_queries(
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
    """Derivative rule of one view.

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
                    f"FieldViewDerivative.backend must be a reconstruction derivative route; got {backend!r}"
                )
        if var != self.variable:
            return None
        dimension = self.evaluator.reconstruction.physical_dimension
        if axis is None:
            if dimension != 1:
                raise ValueError(
                    "FieldViewDerivative.axis must be explicit for a vector view variable; got None"
                )
            axis_ = 0
        else:
            axis_ = int(axis)
            if not 0 <= axis_ < dimension:
                raise ValueError(
                    f"FieldViewDerivative.axis must lie in [0, {dimension}); got {axis!r}"
                )
        return _view_function(
            self.evaluator.raised(axis_, int(order)), self.domain, self.variable
        )


def _view_function(
    evaluator: DiscreteFieldEvaluator, domain: Domain, variable: str, /
) -> imported_domain.DomainFunction:

    return imported_domain.DomainFunction(domain=domain, deps=(variable,), func=evaluator)


@final
class DiscreteFieldFunctionView(StrictModule, NonTrainableState):
    """Discrete field coefficients bound to an equivalent geometry domain.

    The view requires an explicit `GeometryDomain` whose compiled geometry is
    equivalent to the reconstruction's `support_geometry`; discrete mesh or grid
    identity alone never establishes a domain.
    """

    reconstruction: PreparedFieldReconstruction
    coefficients: Array
    domain: imported_domain.GeometryDomain
    variable: str = eqx.field(static=True)
    field_name: str | None = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reconstruction: PreparedFieldReconstruction,
        coefficients: ArrayLike,
        domain: imported_domain.GeometryDomain,
        /,
        *,
        variable: str,
        field_name: str | None = None,
    ) -> None:

        variable_ = canonical_identifier(variable, "variable")
        if variable_ not in domain.labels:
            raise ValueError(
                f"DiscreteFieldFunctionView.variable must be a label of the domain; got {variable_!r} outside {domain.labels}"
            )
        if domain.spatial_dim != reconstruction.physical_dimension:
            raise ValueError(
                "DiscreteFieldFunctionView.domain.dimension must equal the reconstruction dimension; got a different dimension"
            )
        if not domain.geometry.equivalent(reconstruction.support_geometry):
            raise ValueError(
                "DiscreteFieldFunctionView.domain must be equivalent to support_geometry; got a different geometry support"
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
        """Return the view as a `DomainFunction` with a derivative rule."""
        if self.reconstruction.kernel.support_coverage != "complete":
            raise ValueError(
                "DiscreteFieldFunctionView.reconstruction.kernel.support_coverage must define evaluation at every support point; got partial support coverage"
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


class _ConstantPort(Enum):
    CONSTANT = "constant"


def _declared_value_port(field: DomainFunction, /) -> ValuePort | _ConstantPort | None:
    if not field.deps:
        return _ConstantPort.CONSTANT
    match field.func:
        case (
            imported_domain__function.ConstCallable()
            | imported_domain__function.TrainableConstCallable()
        ):
            return _ConstantPort.CONSTANT
        case DiscreteFieldEvaluator() as function:
            return function.value_port
        case imported_domain__model_function.ConcatenatedModelEvaluator() as function:
            ports = intrinsic_model_ports(function.raw_model)
            if ports is not None and len(ports.outputs) == 1:
                return ports.outputs[0]
            return None
        case _:
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


@checked
def require_compatible_field_composition(
    left: imported_domain.DomainFunction,
    right: imported_domain.DomainFunction,
    op: Callable[[Array, Array], Array],
    /,
) -> None:
    """Refuse pointwise algebra that mixes a view with an incompatible field.

    Domain joining already refuses operands whose supports collide; an operand
    depending on the view variable must additionally denote the view's
    geometry support (traced joins only compare schemas). Sums and
    differences of an undifferentiated view additionally require the other
    operand to declare a value port with equal units, event shape, frame, axis
    identity, normalization, and variance; constants are read in the view's
    units. Derivative validity of the result is owned by each operand's
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
                f"field_composition.support must share query points for {variable!r}; got different geometry support"
            )
        if op not in (operator.add, operator.sub) or evaluator.order != 0:
            continue
        other_port = _declared_value_port(other)
        match other_port:
            case _ConstantPort.CONSTANT:
                continue
            case ValuePort():
                pass
            case None:
                raise ValueError(
                    "field_composition.operand.value_port must declare units, frame, and axis identity; got an undeclared value port"
                )
            case invalid:
                assert_never(invalid)
        view_port = evaluator.reconstruction.value_port
        mismatches = _port_mismatches(view_port, other_port)
        if mismatches:
            raise ValueError(
                f"field_composition.operand.value_port must be compatible with the view; got mismatches {', '.join(mismatches)}"
            )


__all__ = [
    "AbstractFieldReconstructionKernel",
    "DiscreteFieldEvaluator",
    "DiscreteFieldFunctionView",
    "FieldApproximation",
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
