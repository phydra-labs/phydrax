#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._bounds import Bounds
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ._cones import AbstractConvexCone, NonnegativeCone, ProductCone, ZeroCone


def _matrix_and_rhs(
    matrix: ArrayLike | None,
    rhs: ArrayLike | None,
    /,
    *,
    variables: int,
    dtype: jnp.dtype,
    name: str,
) -> tuple[Array, Array]:
    if matrix is None:
        if rhs is not None:
            raise ValueError(f"{name} and its right-hand side must be supplied together.")
        return jnp.empty((0, variables), dtype=dtype), jnp.empty((0,), dtype=dtype)
    if rhs is None:
        raise ValueError(f"{name} and its right-hand side must be supplied together.")
    matrix_ = jnp.asarray(matrix)
    rhs_ = jnp.asarray(rhs)
    if matrix_.ndim < 2 or int(matrix_.shape[-1]) != variables:
        raise ValueError(
            f"{name} must end in shape (constraints, {variables}); got {matrix_.shape}."
        )
    rows = int(matrix_.shape[-2])
    if rhs_.ndim < 1 or int(rhs_.shape[-1]) != rows:
        raise ValueError(f"The right-hand side for {name} must end in shape ({rows},).")
    if jnp.issubdtype(matrix_.dtype, jnp.complexfloating) or jnp.issubdtype(
        rhs_.dtype, jnp.complexfloating
    ):
        raise TypeError("Canonical program data must be real-valued.")
    return matrix_.astype(dtype), rhs_.astype(dtype)


def _broadcast_shape(shapes: Sequence[tuple[int, ...]], /) -> tuple[int, ...]:
    return tuple(int(size) for size in np.broadcast_shapes(*shapes))


def _conic_bound_indices(
    bounds: Bounds,
    batch_shape: tuple[int, ...],
    variables: int,
    dtype: jnp.dtype,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected = batch_shape + (variables,)
    lower = np.broadcast_to(
        np.asarray(bounds.lower, dtype=np.dtype(dtype)),
        expected,
    ).reshape((-1, variables))
    upper = np.broadcast_to(
        np.asarray(bounds.upper, dtype=np.dtype(dtype)),
        expected,
    ).reshape((-1, variables))
    lower_finite = np.isfinite(lower)
    upper_finite = np.isfinite(upper)
    fixed = lower_finite & upper_finite & (lower == upper)
    roles = np.stack((lower_finite, upper_finite, fixed), axis=-1)
    if not np.all(roles == roles[:1]):
        raise ValueError(
            "ConicProgram bounds require a shared finite/fixed role pattern across a batch."
        )
    return (
        np.flatnonzero(fixed[0]),
        np.flatnonzero(lower_finite[0] & ~fixed[0]),
        np.flatnonzero(upper_finite[0] & ~fixed[0]),
    )


class ConicProgram(StrictModule):
    """Quadratic-conic program ``min 1/2 xᵀPx + qᵀx`` with ``Ax+s=b, s in K``."""

    quadratic: Array | None
    linear: Array
    constraint_matrix: Array
    constraint_rhs: Array
    lower_bounds: Array
    upper_bounds: Array
    bounds: Bounds
    cone: AbstractConvexCone
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_constraints: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)
    convexity_evidence: str = eqx.field(static=True)

    def __init__(
        self,
        quadratic: ArrayLike | None,
        linear: ArrayLike,
        constraint_matrix: ArrayLike,
        constraint_rhs: ArrayLike,
        cone: AbstractConvexCone,
        /,
        *,
        bounds: Bounds | None = None,
        problem_id: str = "canonical-conic-program",
        convexity_evidence: str = "asserted",
    ):
        linear_ = jnp.asarray(linear)
        matrix = jnp.asarray(constraint_matrix)
        rhs = jnp.asarray(constraint_rhs)
        if linear_.ndim < 1:
            raise ValueError("linear must have at least one dimension.")
        variables = int(linear_.shape[-1])
        if variables < 1:
            raise ValueError("ConicProgram requires at least one decision variable.")
        if matrix.ndim < 2 or int(matrix.shape[-1]) != variables:
            raise ValueError(
                "constraint_matrix must end in shape (constraints, variables)."
            )
        constraints = int(matrix.shape[-2])
        if rhs.ndim < 1 or int(rhs.shape[-1]) != constraints:
            raise ValueError(
                f"constraint_rhs must end in shape ({constraints},); got {rhs.shape}."
            )
        if not isinstance(cone, AbstractConvexCone):
            raise TypeError("cone must be an AbstractConvexCone.")
        if cone.dimension != constraints:
            raise ValueError(
                f"cone dimension {cone.dimension} does not match {constraints} constraints."
            )
        quadratic_ = None if quadratic is None else jnp.asarray(quadratic)
        if quadratic_ is not None and (
            quadratic_.ndim < 2 or tuple(quadratic_.shape[-2:]) != (variables, variables)
        ):
            raise ValueError(
                f"quadratic must end in shape ({variables}, {variables}) or be None."
            )
        arrays = (linear_, matrix, rhs) + (() if quadratic_ is None else (quadratic_,))
        if any(jnp.issubdtype(value.dtype, jnp.complexfloating) for value in arrays):
            raise TypeError("ConicProgram data must be real-valued.")
        dtype = jnp.result_type(*(value.dtype for value in arrays), jnp.float32)
        linear_ = linear_.astype(dtype)
        matrix = matrix.astype(dtype)
        rhs = rhs.astype(dtype)
        if quadratic_ is not None:
            quadratic_ = quadratic_.astype(dtype)
            quadratic_ = 0.5 * quadratic_ + 0.5 * jnp.swapaxes(quadratic_, -1, -2)
        batch = _broadcast_shape(
            (
                linear_.shape[:-1],
                matrix.shape[:-2],
                rhs.shape[:-1],
                () if quadratic_ is None else quadratic_.shape[:-2],
            )
        )
        linear_ = jnp.broadcast_to(linear_, batch + (variables,))
        matrix = jnp.broadcast_to(matrix, batch + (constraints, variables))
        rhs = jnp.broadcast_to(rhs, batch + (constraints,))
        if quadratic_ is not None:
            quadratic_ = jnp.broadcast_to(quadratic_, batch + (variables, variables))
        bounds_ = Bounds() if bounds is None else bounds
        if not isinstance(bounds_, Bounds):
            raise TypeError("bounds must be a Bounds or None.")
        lower, upper = bounds_.materialize(linear_)
        lower = jnp.asarray(lower, dtype=dtype)
        upper = jnp.asarray(upper, dtype=dtype)
        fixed_indices, lower_indices, upper_indices = _conic_bound_indices(
            bounds_,
            batch,
            variables,
            dtype,
        )
        identifier = str(problem_id)
        evidence = str(convexity_evidence)
        if not identifier or not evidence:
            raise ValueError("problem_id and convexity_evidence must be non-empty.")
        self.quadratic = quadratic_
        self.linear = linear_
        self.constraint_matrix = matrix
        self.constraint_rhs = rhs
        self.lower_bounds = lower
        self.upper_bounds = upper
        self.bounds = bounds_
        self.cone = cone
        self.batch_shape = batch
        self.num_variables = variables
        self.num_constraints = constraints
        self.problem_id = identifier
        self.convexity_evidence = evidence
        self.structure_id = canonical_fingerprint(
            {
                "kind": "conic-program",
                "problem_id": identifier,
                "batch_shape": list(batch),
                "variables": variables,
                "constraints": constraints,
                "quadratic": quadratic_ is not None,
                "bound_roles": {
                    "fixed": fixed_indices.tolist(),
                    "lower": lower_indices.tolist(),
                    "upper": upper_indices.tolist(),
                },
                "cone": cone.cone_id,
                "dtype": str(dtype),
            }
        )

    @property
    def is_linear(self) -> bool:
        return self.quadratic is None


class LinearProgram(StrictModule):
    """Linear program with equalities, inequalities, and native variable bounds."""

    linear: Array
    equality_matrix: Array
    equality_rhs: Array
    inequality_matrix: Array
    inequality_rhs: Array
    lower_bounds: Array
    upper_bounds: Array
    bounds: Bounds
    canonical: ConicProgram
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    num_variables: int = eqx.field(static=True)
    num_equalities: int = eqx.field(static=True)
    num_inequalities: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)

    def __init__(
        self,
        linear: ArrayLike,
        /,
        *,
        equality_matrix: ArrayLike | None = None,
        equality_rhs: ArrayLike | None = None,
        inequality_matrix: ArrayLike | None = None,
        inequality_rhs: ArrayLike | None = None,
        bounds: Bounds | None = None,
        problem_id: str = "canonical-linear-program",
    ):
        linear_ = jnp.asarray(linear)
        if linear_.ndim < 1:
            raise ValueError("linear must have at least one dimension.")
        variables = int(linear_.shape[-1])
        if variables < 1:
            raise ValueError("LinearProgram requires at least one decision variable.")
        if jnp.issubdtype(linear_.dtype, jnp.complexfloating):
            raise TypeError("LinearProgram data must be real-valued.")
        dtype = jnp.result_type(linear_.dtype, jnp.float32)
        linear_ = linear_.astype(dtype)
        equality, equality_rhs_ = _matrix_and_rhs(
            equality_matrix,
            equality_rhs,
            variables=variables,
            dtype=dtype,
            name="equality_matrix",
        )
        inequality, inequality_rhs_ = _matrix_and_rhs(
            inequality_matrix,
            inequality_rhs,
            variables=variables,
            dtype=dtype,
            name="inequality_matrix",
        )
        equalities = int(equality.shape[-2])
        inequalities = int(inequality.shape[-2])
        batch = _broadcast_shape(
            (
                linear_.shape[:-1],
                equality.shape[:-2],
                equality_rhs_.shape[:-1],
                inequality.shape[:-2],
                inequality_rhs_.shape[:-1],
            )
        )
        linear_ = jnp.broadcast_to(linear_, batch + (variables,))
        equality = jnp.broadcast_to(equality, batch + (equalities, variables))
        equality_rhs_ = jnp.broadcast_to(equality_rhs_, batch + (equalities,))
        inequality = jnp.broadcast_to(inequality, batch + (inequalities, variables))
        inequality_rhs_ = jnp.broadcast_to(inequality_rhs_, batch + (inequalities,))
        matrix = jnp.concatenate((equality, inequality), axis=-2)
        rhs = jnp.concatenate((equality_rhs_, inequality_rhs_), axis=-1)
        canonical = ConicProgram(
            None,
            linear_,
            matrix,
            rhs,
            ProductCone((ZeroCone(equalities), NonnegativeCone(inequalities))),
            bounds=bounds,
            problem_id=problem_id,
            convexity_evidence="construction",
        )
        self.linear = canonical.linear
        self.equality_matrix = equality
        self.equality_rhs = equality_rhs_
        self.inequality_matrix = inequality
        self.inequality_rhs = inequality_rhs_
        self.lower_bounds = canonical.lower_bounds
        self.upper_bounds = canonical.upper_bounds
        self.bounds = canonical.bounds
        self.canonical = canonical
        self.batch_shape = batch
        self.num_variables = variables
        self.num_equalities = equalities
        self.num_inequalities = inequalities
        self.problem_id = canonical.problem_id
        self.structure_id = canonical.structure_id

    def as_quadratic_program(self):
        """Lower to the dense QP execution path only when that path is selected."""

        from ._quadratic import QuadraticProgram

        return QuadraticProgram(
            jnp.zeros(
                self.batch_shape + (self.num_variables, self.num_variables),
                dtype=self.linear.dtype,
            ),
            self.linear,
            equality_matrix=self.equality_matrix,
            equality_rhs=self.equality_rhs,
            inequality_matrix=self.inequality_matrix,
            inequality_rhs=self.inequality_rhs,
            bounds=self.bounds,
            problem_id=self.problem_id,
            convexity_evidence="construction",
        )


__all__ = ["ConicProgram", "LinearProgram"]
