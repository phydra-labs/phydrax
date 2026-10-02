#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared point queries of one discrete-field reconstruction.

A `PreparedFieldQuery` locates fixed points once against a
`PreparedFieldReconstruction` and retains the owner route, the pointwise
evidence of every requested point, and the admitted subset. Repeated
observations with changing coefficients reuse the route: `apply` gathers and
contracts the prepared route, `transpose` is the owner's exact scatter, and no
dense coefficient-by-point matrix is formed. Coefficient-linear queries expose
an `AbstractLinearOperator` whose declared pairings define its Hilbert adjoint;
nonlinear reconstructions expose an explicit linearization at a supplied
coefficient state instead of a fabricated global transpose.
"""

from __future__ import annotations

from typing import Any, assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike, DTypeLike

from .._differentiation import DerivativeRegularity
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._validation import finite_real_scalar
from ..linalg import (
    AbstractPairing,
    ArraySpace,
    FunctionLinearOperator,
    prepare_linearization,
    PreparedLinearization,
)
from ..typing import checked, parse
from ._views import (
    _checked_queries,
    FieldApproximation,
    FieldQueryEvidence,
    FieldSideBinding,
    InterpolationTransposeEvidence,
    PreparedFieldReconstruction,
    transpose_duality_evidence,
)


FieldQueryCoverage: TypeAlias = Literal["complete", "masked"]


def _host_points(points: ArrayLike, dimension: int, /) -> np.ndarray:
    values = np.asarray(points)
    if values.ndim == 1 and values.shape[0] == dimension:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != dimension or values.shape[0] == 0:
        raise ValueError(
            f"Prepared query points must have shape (points, {dimension}) with at "
            "least one point."
        )
    if not np.issubdtype(values.dtype, np.floating):
        raise TypeError("Prepared query points must be real floating point.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Prepared query points must be finite.")
    return values


@final
class PreparedFieldQuery(StrictModule, NonTrainableState):
    """Fixed points, derivative, side, and coverage bound to one located route.

    `evidence` holds the pointwise status of every requested point; `admitted`
    indexes the points whose route is retained. `apply` returns the values at
    the admitted points only, with shape `(admitted_count, *value_shape)`, so a
    masked query never reports a value at a rejected point. `complete` is true
    when every requested point is admitted; a consumer whose law requires full
    support must refuse an incomplete query.
    """

    reconstruction: PreparedFieldReconstruction
    route: Any
    evidence: FieldQueryEvidence
    points: Array
    admitted: Array
    side: FieldSideBinding | None
    derivative: tuple[int, ...] = eqx.field(static=True)
    coverage: FieldQueryCoverage = eqx.field(static=True)
    admitted_count: int = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    query_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        reconstruction: PreparedFieldReconstruction,
        points: ArrayLike,
        /,
        *,
        derivative: tuple[int, ...] | None = None,
        side: FieldSideBinding | None = None,
        coverage: FieldQueryCoverage = "complete",
    ) -> None:
        coverage_ = parse(coverage, FieldQueryCoverage, "coverage")
        index = reconstruction.derivative_index(derivative)
        binding = reconstruction._side(side)
        host = _host_points(points, reconstruction.physical_dimension)
        query = reconstruction._points(jnp.asarray(host))
        route, evidence = reconstruction.kernel.locate(query, index, binding)
        valid = np.asarray(evidence.valid)
        admitted = np.flatnonzero(valid).astype(np.int32)
        match coverage_:
            case "complete":
                _checked_queries(query, evidence)
            case "masked":
                if admitted.size == 0:
                    raise ValueError(
                        "No prepared query point is valid; inspect the location "
                        "evidence of the reconstruction before querying."
                    )
                if admitted.size != valid.size:
                    route, subset = reconstruction.kernel.locate(
                        query[jnp.asarray(admitted)], index, binding
                    )
                    if not bool(np.all(np.asarray(subset.valid))):
                        raise RuntimeError(
                            "Relocating the admitted query points changed their "
                            "validity; the reconstruction kernel is not deterministic."
                        )
            case _:
                assert_never(coverage_)
        self.reconstruction = reconstruction
        self.route = route
        self.evidence = evidence
        self.points = query
        self.admitted = jnp.asarray(admitted)
        self.side = binding
        self.derivative = index
        self.coverage = coverage_
        self.admitted_count = admitted.size
        self.complete = admitted.size == valid.size
        self.query_id = canonical_fingerprint(
            {
                "kind": "prepared-field-query",
                "reconstruction": reconstruction.reconstruction_id,
                "points": array_tree_fingerprint(host),
                "derivative": list(index),
                "side": None if binding is None else binding.binding_id,
                "coverage": coverage_,
                "admitted": array_tree_fingerprint(admitted),
            }
        )

    @property
    def approximation(self) -> FieldApproximation:
        """Whether the query evaluates the field itself or a labeled projection."""
        return self.reconstruction.approximation

    @property
    def coefficient_linear(self) -> bool:
        """Whether the prepared route is linear in the coefficients."""
        return self.reconstruction.coefficient_linear

    @property
    def value_shape(self) -> tuple[int, ...]:
        return self.reconstruction.value_shape

    @property
    def coefficient_shape(self) -> tuple[int, ...]:
        return self.reconstruction.coefficient_shape

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (self.admitted_count, *self.reconstruction.value_shape)

    @property
    def admitted_points(self) -> Array:
        return self.points[self.admitted]

    @property
    def order(self) -> int:
        return sum(self.derivative)

    @property
    def regularity(self) -> DerivativeRegularity:
        """Declared value regularity of the queried derivative."""
        if self.order == 0:
            return self.reconstruction.regularity
        return self.reconstruction.regularity.differentiate(self.order)

    @checked
    def require_reconstruction(
        self, reconstruction: PreparedFieldReconstruction, /
    ) -> None:
        """Refuse a reconstruction other than the one this route was located on.

        Coefficient refresh reuses the prepared route; a geometry, support, or
        field-space change produces another reconstruction and requires a new
        query.
        """
        if reconstruction.reconstruction_id != self.reconstruction.reconstruction_id:
            raise ValueError(
                "The prepared query was located on another reconstruction revision; "
                "prepare a new query on the refreshed reconstruction."
            )

    def apply(self, coefficients: ArrayLike, /) -> Array:
        """Evaluate the admitted points on the prepared route."""
        values = self.reconstruction.validate_coefficients(coefficients)
        return self.reconstruction.kernel.apply(self.route, values)

    def transpose(self, cotangent: ArrayLike, /) -> Array:
        """Exact algebraic transpose (scatter) of a coefficient-linear route."""
        self._require_linear()
        dual = jnp.asarray(cotangent)
        if dual.shape != self.output_shape:
            raise ValueError(
                f"Cotangent must have shape (admitted, *value_shape) = "
                f"{self.output_shape}; got {dual.shape}."
            )
        return self.reconstruction.kernel.transpose(self.route, dual)

    def duality_evidence(
        self,
        coefficients: ArrayLike,
        cotangent: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
    ) -> InterpolationTransposeEvidence:
        """Check `<R c, w> = <c, R^T w>` on the prepared route."""
        limit = finite_real_scalar(tolerance, "tolerance")
        if limit < 0.0:
            raise ValueError("tolerance must be non-negative.")
        values = self.reconstruction.validate_coefficients(coefficients)
        dual = jnp.asarray(cotangent)
        scattered = self.transpose(dual)
        return transpose_duality_evidence(
            self.apply(values), dual, values, scattered, tolerance=limit
        )

    def as_linear_operator(
        self,
        /,
        *,
        dtype: DTypeLike | None = None,
        coefficient_pairing: AbstractPairing | None = None,
        value_pairing: AbstractPairing | None = None,
    ) -> FunctionLinearOperator:
        """Return the matrix-free query operator on declared vector spaces.

        The coefficient dtype defaults to the reconstruction's native
        `coefficient_dtype` (real coefficients in the point precision when it
        declares none); an explicit `dtype` must agree with a declared native
        dtype. `coefficient_pairing` and `value_pairing` declare the Riesz maps of
        the coefficient and value spaces (Euclidean when omitted); the operator's
        `transpose_mv` is the coordinate transpose and its `adjoint_mv` is the
        Hilbert adjoint relative to those pairings.

        A real-valued query of complex coefficients (real-output spectral
        synthesis) is only real-linear. Its operator acts on the realified
        coefficient space: shape `(*coefficient_shape, 2)` of the real dtype,
        whose trailing axis holds the (real, imaginary) parts, so the
        coordinate transpose and every declared real pairing are exact.
        """
        self._require_linear()
        native = self.reconstruction.coefficient_dtype
        if dtype is None:
            coefficient_dtype = self.points.dtype if native is None else native
        else:
            coefficient_dtype = jnp.dtype(dtype)
            if native is not None and coefficient_dtype != native:
                raise ValueError(
                    f"dtype {coefficient_dtype.name} differs from the native "
                    f"coefficient dtype {native.name} of the reconstruction."
                )
        coefficients = ArraySpace(self.coefficient_shape, dtype=coefficient_dtype)
        output = jax.eval_shape(self.apply, coefficients.structure())
        realified = jnp.issubdtype(
            coefficient_dtype, jnp.complexfloating
        ) and not jnp.issubdtype(output.dtype, jnp.complexfloating)
        if realified:
            source = ArraySpace(
                (*self.coefficient_shape, 2),
                dtype=jnp.finfo(coefficient_dtype).dtype,
                pairing=coefficient_pairing,
            )
            function, transpose = self._realified_apply, self._realified_transpose
        else:
            source = ArraySpace(
                self.coefficient_shape,
                dtype=coefficient_dtype,
                pairing=coefficient_pairing,
            )
            function, transpose = self.apply, self.transpose
        target = ArraySpace(output.shape, dtype=output.dtype, pairing=value_pairing)
        return FunctionLinearOperator(
            function,
            source=source,
            target=target,
            transpose_action=transpose,
            operator_id=canonical_fingerprint(
                {
                    "kind": "prepared-field-query-operator",
                    "query": self.query_id,
                    "realified": realified,
                    "source": source.space_id,
                    "target": target.space_id,
                }
            ),
        )

    def _realified_apply(self, parts: Array, /) -> Array:
        """Evaluate `Re(R (a + i b))` at realified coefficients `(a, b)`."""
        return self.apply(jax.lax.complex(parts[..., 0], parts[..., 1]))

    def _realified_transpose(self, cotangent: Array, /) -> Array:
        """Real transpose `(Re(R^T w), -Im(R^T w))` of the realified query."""
        scattered = self.transpose(cotangent)
        return jnp.stack((jnp.real(scattered), -jnp.imag(scattered)), axis=-1)

    def linearize(self, coefficients: ArrayLike, /) -> PreparedLinearization:
        """Prepare the exact route linearization at one coefficient state."""
        values = self.reconstruction.validate_coefficients(coefficients)
        source = ArraySpace(self.coefficient_shape, dtype=values.dtype)
        output = jax.eval_shape(self.apply, source.structure())
        return prepare_linearization(
            self.apply,
            values,
            source=source,
            target=ArraySpace(output.shape, dtype=output.dtype),
            linearization_id=canonical_fingerprint(
                {"kind": "prepared-field-query-linearization", "query": self.query_id}
            ),
        )

    def _require_linear(self) -> None:
        if not self.reconstruction.coefficient_linear:
            raise ValueError(
                "The reconstruction is nonlinear in its coefficients and has no "
                "algebraic transpose; use linearize(coefficients) at a supplied "
                "coefficient state."
            )


__all__ = ["FieldQueryCoverage", "PreparedFieldQuery"]
