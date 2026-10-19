#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike
from jaxtyping import PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._operators import (
    AbstractLinearOperator,
    FunctionLinearOperator,
    IdentityLinearOperator,
)
from ._orthogonalization import _scaled_vector_normalization
from ._pairings import EuclideanPairing
from ._policies import RankPolicy
from ._rank import numerical_rank_data
from ._spaces import AbstractVectorSpace, ArraySpace, DualSpace


class ConstraintMap(StrictModule, NonTrainableState):
    """Affine parameterization ``u = P z + g`` over declared vector spaces."""

    full_space: AbstractVectorSpace
    reduced_space: AbstractVectorSpace
    prolongation: AbstractLinearOperator
    dual_pullback: AbstractLinearOperator
    constraint_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        full_space: AbstractVectorSpace,
        reduced_space: AbstractVectorSpace,
        prolongation: AbstractLinearOperator,
        /,
        *,
        constraint_id: str | None = None,
    ) -> None:
        if not isinstance(full_space, AbstractVectorSpace) or not isinstance(
            reduced_space, AbstractVectorSpace
        ):
            raise TypeError("Constraint spaces must be AbstractVectorSpace values.")
        if not prolongation.source.compatible(
            reduced_space
        ) or not prolongation.target.compatible(full_space):
            raise ValueError("Constraint prolongation must map reduced to full space.")
        if prolongation.batch_shape:
            raise ValueError("Constraint prolongation must be unbatched.")
        full_dual = DualSpace(full_space)
        reduced_dual = DualSpace(reduced_space)
        pullback = FunctionLinearOperator(
            lambda value: prolongation.transpose_mv(value),
            source=full_dual,
            target=reduced_dual,
            transpose_action=lambda value: prolongation.mv(value),
            operator_id=canonical_fingerprint(
                {
                    "kind": "constraint-dual-pullback",
                    "prolongation": prolongation.operator_id,
                }
            ),
        )
        if constraint_id is not None and not isinstance(constraint_id, str):
            raise TypeError("constraint_id must be a string or None.")
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "constraint-map",
                    "full_space": full_space.space_id,
                    "reduced_space": reduced_space.space_id,
                    "prolongation": prolongation.operator_id,
                }
            )
            if constraint_id is None
            else constraint_id
        )
        if not resolved_id:
            raise ValueError("constraint_id must be non-empty.")
        self.full_space = full_space
        self.reduced_space = reduced_space
        self.prolongation = prolongation
        self.dual_pullback = pullback
        self.constraint_id = resolved_id

    def expand(self, reduced: object, lift: object, /) -> PyTree[Array]:
        reduced_ = self.reduced_space.validate(reduced)
        lift_ = self.full_space.validate(lift)
        return self.full_space.validate(
            jax.tree.map(
                lambda correction, offset: correction + offset,
                self.prolongation.mv(reduced_),
                lift_,
            )
        )

    def homogeneous_correction(self, reduced: object, /) -> PyTree[Array]:
        return self.full_space.validate(
            self.prolongation.mv(self.reduced_space.validate(reduced))
        )

    def pullback_dual(self, residual: PyTree[Any], /) -> PyTree[Array]:
        return self.dual_pullback.mv(residual)

    def reduce_vector(self, residual: PyTree[Any], /) -> PyTree[Array]:
        return self.prolongation.adjoint_mv(self.full_space.validate(residual))


class OrthogonalHyperplaneConstraint(StrictModule, NonTrainableState):
    """Matrix-free orthonormal chart of a real covector's kernel.

    The Householder vector and covector remain numerical leaves. ``valid``
    certifies the nonzero finite row; callers retain responsibility for refusing
    an invalid chart. Storage and each chart application are linear in dimension.
    """

    constraint: ConstraintMap
    covector: Array
    unit_covector: Array
    normalization_scale: Array
    normalized_row_norm: Array
    row_norm: Array
    valid: Array

    @checked
    def __init__(
        self, space: ArraySpace, covector: ArrayLike, /, *, constraint_id: str
    ) -> None:
        if not isinstance(space.pairing, EuclideanPairing):
            raise ValueError(
                "OrthogonalHyperplaneConstraint.space.pairing must be Euclidean; "
                f"got {type(space.pairing).__name__}"
            )
        if len(space.shape) != 1 or space.size < 1:
            raise ValueError(
                "OrthogonalHyperplaneConstraint.space must be a nonempty one-dimensional array space; "
                f"got shape={space.shape}"
            )
        covector = space.validate(covector)
        if not jnp.issubdtype(covector.dtype, jnp.floating):
            raise TypeError(
                "OrthogonalHyperplaneConstraint.covector must be real floating; "
                f"got dtype={covector.dtype}"
            )
        unit, norm, scale, scaled_norm, valid = _scaled_vector_normalization(covector)
        sign = jnp.where(unit[-1] >= 0.0, 1.0, -1.0)
        reflection = unit.at[-1].add(sign)
        denominator = jnp.vdot(reflection, reflection)

        def reflect(value: Array) -> Array:
            return value - reflection * (2.0 * jnp.vdot(reflection, value) / denominator)

        def embed(value: Array) -> Array:
            return reflect(jnp.concatenate((value, jnp.zeros((1,), dtype=value.dtype))))

        def restrict(value: Array) -> Array:
            return reflect(value)[:-1]

        full = space
        reduced = ArraySpace((covector.size - 1,), dtype=covector.dtype)
        prolongation = FunctionLinearOperator(
            embed,
            source=reduced,
            target=full,
            transpose_action=restrict,
            operator_id=f"{constraint_id}:householder",
        )
        self.constraint = ConstraintMap(
            full, reduced, prolongation, constraint_id=constraint_id
        )
        self.covector = covector
        self.unit_covector = unit
        self.normalization_scale = scale
        self.normalized_row_norm = scaled_norm
        self.row_norm = norm
        self.valid = valid

    def lift(self, right: Array, /) -> Array:
        """Return the minimum-norm lift of ``covector @ x = right``."""
        return self.unit_covector * (
            (right / self.normalization_scale) / self.normalized_row_norm
        )

    def transpose_lift(self, value: Array, /) -> Array:
        """Apply the coordinate transpose of the minimum-norm lift."""
        return (
            jnp.vdot(self.unit_covector, value) / self.normalization_scale
        ) / self.normalized_row_norm


class _OrthogonalConstraintFrame(StrictModule, NonTrainableState):
    """Compose orthonormal kernel charts for independent real covectors.

    Rows and inverse-response directions remain numerical leaves. The finite
    affine correction sequence returns the minimum-norm lift without storing a
    dense complement or inverse. Invalid banks retain rank evidence and return
    undefined lifts; callers must refuse them before solving.
    """

    rows: Array
    directions: Array
    constraint: ConstraintMap
    pivot_norms: Array
    singular_values: Array
    rank: Array
    rank_cutoff: Array
    condition_estimate: Array
    valid: Array
    _rhs_space: ArraySpace

    @checked
    def __init__(
        self,
        space: ArraySpace,
        rows: Array,
        rank_policy: RankPolicy,
        *,
        constraint_id: str,
    ) -> None:
        if not isinstance(space.pairing, EuclideanPairing):
            raise ValueError(
                "_OrthogonalConstraintFrame.space.pairing must be Euclidean; "
                f"got {type(space.pairing).__name__}"
            )
        if len(space.shape) != 1 or space.size < 1:
            raise ValueError(
                "_OrthogonalConstraintFrame.space must be a nonempty one-dimensional "
                f"array space; got shape={space.shape}"
            )
        if rows.ndim != 2:
            raise ValueError(
                "_OrthogonalConstraintFrame.rows must have shape (n, k); "
                f"got shape={rows.shape}"
            )
        count = rows.shape[1]
        if not 1 <= count <= space.size:
            raise ValueError(
                "_OrthogonalConstraintFrame.rows must have 1 <= k <= n; "
                f"got shape={rows.shape}, n={space.size}"
            )
        rows = ArraySpace((space.size, count), dtype=space.dtype).validate(rows)
        if not jnp.issubdtype(rows.dtype, jnp.floating):
            raise TypeError(
                "_OrthogonalConstraintFrame.rows must be real floating; "
                f"got dtype={rows.dtype}"
            )

        scale = jax.lax.stop_gradient(jnp.max(jnp.abs(rows)))
        finite = jnp.all(jnp.isfinite(rows))
        safe_scale = jnp.where(finite & (scale > 0.0), scale, 1.0)
        scaled_rows = jax.lax.optimization_barrier(rows / safe_scale)

        def spectrum(value: Array) -> Array:
            return jnp.linalg.svd(value, full_matrices=False, compute_uv=False)

        def undefined_spectrum(value: Array) -> Array:
            return jnp.full((count,), jnp.nan, dtype=value.dtype)

        singular_values = (
            jax.lax.cond(finite, spectrum, undefined_spectrum, scaled_rows) * scale
        )
        rank_data = numerical_rank_data(singular_values, space.size, count, rank_policy)
        prolongation: AbstractLinearOperator = IdentityLinearOperator(
            space, operator_id=f"{constraint_id}:identity"
        )
        directions: list[Array] = []
        pivot_norms: list[Array] = []
        valid = finite & rank_data.finite & rank_data.full_rank
        # Each structural step removes one coordinate; no solver iteration or
        # full-dimensional complement is involved, including the final n = k step.
        for column in range(count):
            match prolongation.source:
                case ArraySpace() as reduced_space:
                    chart = OrthogonalHyperplaneConstraint(
                        reduced_space,
                        jnp.asarray(prolongation.transpose_mv(rows[:, column])),
                        constraint_id=f"{constraint_id}:row-{column}",
                    )
                case _:
                    raise TypeError(
                        "_OrthogonalConstraintFrame.prolongation.source must be "
                        f"ArraySpace; got {type(prolongation.source).__name__}"
                    )
            direction = jnp.asarray(
                prolongation.mv(chart.lift(jnp.ones((), dtype=rows.dtype)))
            )
            directions.append(direction)
            pivot_norms.append(chart.row_norm)
            valid = valid & chart.valid & jnp.all(jnp.isfinite(direction))
            prolongation = prolongation @ chart.constraint.prolongation

        self.rows = rows
        self.directions = jnp.stack(directions, axis=1)
        self.constraint = ConstraintMap(
            space, prolongation.source, prolongation, constraint_id=constraint_id
        )
        self.pivot_norms = jnp.stack(pivot_norms)
        self.singular_values = singular_values
        self.rank = rank_data.rank
        self.rank_cutoff = rank_data.cutoff
        self.condition_estimate = rank_data.condition_estimate
        self.valid = valid
        self._rhs_space = ArraySpace((count,), dtype=rows.dtype)

    def _lift_coordinates(self, right: Array, /) -> Array:
        lifted = jnp.zeros((self.rows.shape[0],), dtype=self.rows.dtype)
        for column in range(self.rows.shape[1]):
            lifted = lifted + self.directions[:, column] * (
                right[column] - self.rows[:, column] @ lifted
            )
        return lifted

    def lift(self, rhs: Array, /) -> Array:
        """Return the minimum-norm row lift, or NaNs for an invalid bank."""
        lifted = self._lift_coordinates(self._rhs_space.validate(rhs))
        return jnp.where(self.valid, lifted, jnp.full_like(lifted, jnp.nan))

    def transpose_lift(self, value: Array, /) -> Array:
        """Apply the coordinate transpose, retaining invalid-bank evidence."""
        value = jnp.asarray(self.constraint.full_space.validate(value))
        transposed = jax.linear_transpose(
            self._lift_coordinates, self._rhs_space.structure()
        )(value)[0]
        return jnp.where(self.valid, transposed, jnp.full_like(transposed, jnp.nan))


def compose_constraint_maps(
    outer: ConstraintMap,
    inner: ConstraintMap,
    /,
    *,
    constraint_id: str | None = None,
) -> ConstraintMap:
    """Compose nested affine charts' homogeneous maps.

    If ``outer`` expands ``y`` into ``x`` and ``inner`` expands ``z`` into
    ``y``, the returned map expands ``z`` into ``x``. Their dynamic lifts
    compose as ``outer.expand(inner_lift, outer_lift)``.
    """
    if not isinstance(outer, ConstraintMap) or not isinstance(inner, ConstraintMap):
        raise TypeError("outer and inner must be ConstraintMap values.")
    if not inner.full_space.compatible(outer.reduced_space):
        raise ValueError(
            "The inner full space must match the outer reduced space exactly."
        )
    if constraint_id is not None and not isinstance(constraint_id, str):
        raise TypeError("constraint_id must be a string or None.")
    identifier = (
        canonical_fingerprint(
            {
                "kind": "composed-constraint-map",
                "outer": outer.constraint_id,
                "inner": inner.constraint_id,
            }
        )
        if constraint_id is None
        else constraint_id
    )
    return ConstraintMap(
        outer.full_space,
        inner.reduced_space,
        outer.prolongation @ inner.prolongation,
        constraint_id=identifier,
    )


__all__ = [
    "ConstraintMap",
    "OrthogonalHyperplaneConstraint",
    "compose_constraint_maps",
]
