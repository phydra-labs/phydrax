#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from .._fingerprint import canonical_fingerprint
from ._assembly import assemble_diagonal
from ._costs import PreconditionerCostEstimate
from ._materialization import MaterializationPolicy
from ._operators import (
    _generic_adjoint,
    AbstractLinearOperator,
    DenseLinearOperator,
    DiagonalLinearOperator,
)
from ._preconditioner_properties import (
    _preconditioner_properties_payload,
    PreconditionerProperties,
)
from ._preconditioners import (
    _prepared_action_cost,
    _validated,
    AbstractPreconditioner,
)
from ._preconditioning import (
    _validate_setup_operator,
    AbstractPreconditionerBuilder,
    PreconditionerRefreshPolicy,
)
from ._properties import (
    LinearCapabilityError,
    OperatorCapabilities,
    OperatorProperties,
)
from ._spaces import _coordinate_dtype, _has_diagonal_pairing
from ._sparse_contract import AbstractSparseLinearOperator
from ._spectral import estimate_spectral_bounds


ChebyshevScaling: TypeAlias = Literal["none", "symmetric-jacobi"]
ChebyshevBoundsSource: TypeAlias = Literal["explicit", "estimated"]


def _scale_vector(space, diagonal: Array, vector: PyTree[Any], /) -> PyTree[Array]:
    coordinates = space.flatten(vector)
    return space.unflatten(diagonal * coordinates)


class _SymmetricJacobiLinearOperator(AbstractLinearOperator):
    operator: AbstractLinearOperator
    inverse_sqrt_diagonal: Array

    def __init__(self, operator: AbstractLinearOperator, inverse_sqrt_diagonal: Array, /):
        _validate_setup_operator(operator)
        diagonal = jnp.asarray(inverse_sqrt_diagonal)
        if diagonal.shape != (operator.source.size,):
            raise ValueError("inverse_sqrt_diagonal must match the operator dimension.")
        self.operator = operator
        self.inverse_sqrt_diagonal = diagonal
        self.source = operator.source
        self.target = operator.target

        pairing_preserves_structure = _has_diagonal_pairing(operator.source)
        self_adjoint = pairing_preserves_structure and operator.properties.certifies(
            "self_adjoint"
        )
        positive_definite = pairing_preserves_structure and operator.properties.certifies(
            "positive_definite"
        )
        positive_semidefinite = (
            pairing_preserves_structure
            and operator.properties.certifies("positive_semidefinite")
        )
        claims = {
            "self_adjoint": self_adjoint,
            "positive_definite": positive_definite,
            "positive_semidefinite": positive_semidefinite,
        }
        self.properties = OperatorProperties(
            **claims,
            evidence={name: "transformed" for name, claimed in claims.items() if claimed},
        )
        self.capabilities = OperatorCapabilities(
            transpose=operator.capabilities.transpose,
            adjoint=operator.capabilities.transpose,
            materialize=False,
        )
        self.batch_shape = ()
        self.operator_id = canonical_fingerprint(
            {
                "kind": "symmetric-jacobi-scaled",
                "operator": operator.operator_id,
                "space": operator.source.space_id,
            }
        )

    def scale(self, vector: PyTree[Any], /) -> PyTree[Array]:
        return _scale_vector(self.source, self.inverse_sqrt_diagonal, vector)

    def mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        scaled = self.scale(self.source.validate(vector))
        return self.scale(self.operator.mv(scaled))

    def transpose_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if not self.capabilities.transpose:
            raise LinearCapabilityError(
                "The setup operator does not provide an algebraic transpose action."
            )
        scaled = self.scale(self.target.validate(vector))
        return self.scale(self.operator.transpose_mv(scaled))

    def adjoint_mv(self, vector: PyTree[Any], /) -> PyTree[Array]:
        if not self.capabilities.adjoint:
            raise LinearCapabilityError(
                "The setup operator does not provide the transpose action needed for "
                "an adjoint."
            )
        return _generic_adjoint(self, vector)

    def _materialize(self, /) -> Array:
        raise LinearCapabilityError(
            "A symmetric-Jacobi scaled operator is intentionally matrix-free."
        )


class ChebyshevPreconditioner(AbstractPreconditioner):
    """Fixed-degree Chebyshev semi-iteration prepared as a linear action."""

    effective_operator: AbstractLinearOperator
    alpha: Array
    beta: Array
    lower_bound: Array
    upper_bound: Array
    degree: int = eqx.field(static=True)
    scaling: ChebyshevScaling = eqx.field(static=True)
    bounds_source: ChebyshevBoundsSource = eqx.field(static=True)
    builder_id: str = eqx.field(static=True)
    setup_operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        effective_operator: AbstractLinearOperator,
        alpha: ArrayLike,
        beta: ArrayLike,
        /,
        *,
        lower_bound: ArrayLike,
        upper_bound: ArrayLike,
        scaling: ChebyshevScaling,
        bounds_source: ChebyshevBoundsSource,
        properties: PreconditionerProperties,
        builder_id: str,
        setup_operator_id: str,
        preconditioner_id: str,
    ):
        _validate_setup_operator(effective_operator)
        alpha_ = jnp.asarray(alpha)
        beta_ = jnp.asarray(beta)
        if alpha_.ndim != 1 or beta_.shape != alpha_.shape or alpha_.size < 1:
            raise ValueError(
                "Chebyshev recurrence coefficients must be equal nonempty vectors."
            )
        alpha_ = _validated(
            alpha_,
            jnp.any(~jnp.isfinite(alpha_)) | jnp.any(~jnp.isfinite(beta_)),
            "Chebyshev recurrence coefficients must be finite.",
        )
        alpha_ = jax.lax.stop_gradient(alpha_)
        beta_ = jax.lax.stop_gradient(beta_)
        lower_ = jnp.asarray(lower_bound)
        upper_ = jnp.asarray(upper_bound)
        if lower_.shape != () or upper_.shape != ():
            raise ValueError("Chebyshev bounds must be scalar.")
        lower_ = _validated(
            lower_,
            ~jnp.isfinite(lower_)
            | ~jnp.isfinite(upper_)
            | (lower_ <= 0.0)
            | (lower_ > upper_),
            "Chebyshev bounds must be finite and satisfy 0 < lower <= upper.",
        )
        lower_ = jax.lax.stop_gradient(lower_)
        upper_ = jax.lax.stop_gradient(upper_)
        if scaling not in ("none", "symmetric-jacobi"):
            raise ValueError("Unknown Chebyshev scaling.")
        scaled_effective = isinstance(effective_operator, _SymmetricJacobiLinearOperator)
        if (scaling == "symmetric-jacobi") != scaled_effective:
            raise ValueError("scaling must match the effective operator representation.")
        if bounds_source not in ("explicit", "estimated"):
            raise ValueError("Unknown Chebyshev bounds source.")
        if not isinstance(properties, PreconditionerProperties):
            raise TypeError("properties must be PreconditionerProperties.")
        builder_id_ = str(builder_id)
        setup_operator_id_ = str(setup_operator_id)
        preconditioner_id_ = str(preconditioner_id)
        if not builder_id_ or not setup_operator_id_ or not preconditioner_id_:
            raise ValueError("Chebyshev provenance IDs must be nonempty.")
        original_operator = (
            effective_operator.operator
            if isinstance(effective_operator, _SymmetricJacobiLinearOperator)
            else effective_operator
        )
        if setup_operator_id_ != original_operator.operator_id:
            raise ValueError("setup_operator_id must match the prepared setup operator.")
        self.effective_operator = effective_operator
        self.alpha = alpha_
        self.beta = beta_
        self.lower_bound = lower_
        self.upper_bound = upper_
        self.degree = int(alpha_.size)
        self.scaling = scaling
        self.bounds_source = bounds_source
        self.builder_id = builder_id_
        self.setup_operator_id = setup_operator_id_
        self.space = effective_operator.source
        self.properties = properties
        self.preconditioner_id = preconditioner_id_

    @property
    def interval(self) -> tuple[Array, Array]:
        return self.lower_bound, self.upper_bound

    @property
    def setup_operator(self) -> AbstractLinearOperator:
        if isinstance(self.effective_operator, _SymmetricJacobiLinearOperator):
            return self.effective_operator.operator
        return self.effective_operator

    @property
    def inverse_sqrt_diagonal(self) -> Array | None:
        if isinstance(self.effective_operator, _SymmetricJacobiLinearOperator):
            return self.effective_operator.inverse_sqrt_diagonal
        return None

    def apply(
        self,
        residual: PyTree[Any],
        /,
        *,
        iteration: ArrayLike | None = None,
    ) -> PyTree[Array]:
        del iteration
        value = self.space.validate(residual)
        if isinstance(self.effective_operator, _SymmetricJacobiLinearOperator):
            value = self.effective_operator.scale(value)
        zeros = jax.tree.map(jnp.zeros_like, value)

        def step(index, state):
            approximation, direction, current_residual = state
            alpha = self.alpha[index]
            beta = self.beta[index]
            next_direction = jax.tree.map(
                lambda residual_leaf, direction_leaf: (
                    alpha * residual_leaf + beta * direction_leaf
                ),
                current_residual,
                direction,
            )
            image = self.effective_operator.mv(next_direction)
            next_approximation = jax.tree.map(
                lambda approximation_leaf, direction_leaf: (
                    approximation_leaf + direction_leaf
                ),
                approximation,
                next_direction,
            )
            next_residual = jax.tree.map(
                lambda residual_leaf, image_leaf: residual_leaf - image_leaf,
                current_residual,
                image,
            )
            return next_approximation, next_direction, next_residual

        approximation, _, final_residual = jax.lax.fori_loop(
            0,
            self.degree,
            step,
            (zeros, zeros, value),
        )
        if isinstance(self.effective_operator, _SymmetricJacobiLinearOperator):
            approximation = self.effective_operator.scale(approximation)
        coordinates = self.space.flatten(approximation)
        residual_coordinates = self.space.flatten(final_residual)
        coordinates = _validated(
            coordinates,
            jnp.any(~jnp.isfinite(coordinates))
            | jnp.any(~jnp.isfinite(residual_coordinates)),
            "Chebyshev preconditioning produced non-finite values.",
        )
        return self.space.unflatten(coordinates)

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        del materialization
        multiplier = 7 if self.scaling == "symmetric-jacobi" else 5
        return _prepared_action_cost(
            self,
            setup_operator,
            apply_workspace_multiplier=multiplier,
            reason="supplied fixed-degree Chebyshev action storage and workspace",
        )


class ChebyshevPreconditionerBuilder(AbstractPreconditionerBuilder):
    """Prepare a fixed-degree Chebyshev approximate inverse."""

    degree: int = eqx.field(static=True)
    interval: tuple[float, float] | None = eqx.field(static=True)
    estimation_steps: int = eqx.field(static=True)
    margin: float = eqx.field(static=True)
    scaling: ChebyshevScaling = eqx.field(static=True)
    properties: PreconditionerProperties | None
    _builder_id: str = eqx.field(static=True)

    def __init__(
        self,
        degree: int,
        /,
        *,
        interval: Sequence[float] | None = None,
        estimation_steps: int = 16,
        margin: float = 0.05,
        scaling: ChebyshevScaling = "none",
        properties: PreconditionerProperties | None = None,
    ):
        degree_ = int(degree)
        estimation_steps_ = int(estimation_steps)
        margin_ = float(margin)
        if degree_ < 1:
            raise ValueError("degree must be at least one.")
        if estimation_steps_ < 1:
            raise ValueError("estimation_steps must be at least one.")
        if not np.isfinite(margin_) or margin_ < 0.0 or margin_ >= 1.0:
            raise ValueError("margin must be finite and satisfy 0 <= margin < 1.")
        if scaling not in ("none", "symmetric-jacobi"):
            raise ValueError("scaling must be either 'none' or 'symmetric-jacobi'.")
        if properties is not None and not isinstance(
            properties, PreconditionerProperties
        ):
            raise TypeError("properties must be PreconditionerProperties or None.")
        if interval is None:
            interval_ = None
        else:
            values = tuple(interval)
            if len(values) != 2:
                raise ValueError("interval must contain exactly two bounds.")
            lower, upper = (float(value) for value in values)
            if (
                not np.isfinite(lower)
                or not np.isfinite(upper)
                or lower <= 0.0
                or lower > upper
            ):
                raise ValueError(
                    "interval must be finite and satisfy 0 < lower <= upper."
                )
            interval_ = (lower, upper)

        self.degree = degree_
        self.interval = interval_
        self.estimation_steps = estimation_steps_
        self.margin = margin_
        self.scaling = scaling
        self.properties = properties
        self._builder_id = canonical_fingerprint(
            {
                "kind": "chebyshev-preconditioner-builder",
                "degree": degree_,
                "interval": interval_,
                "estimation_steps": estimation_steps_,
                "margin": margin_,
                "scaling": scaling,
                "properties": (
                    None
                    if properties is None
                    else _preconditioner_properties_payload(properties)
                ),
            }
        )

    @property
    def builder_id(self) -> str:
        return self._builder_id

    @property
    def default_refresh(self) -> PreconditionerRefreshPolicy:
        return "numeric"

    def properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
    ) -> PreconditionerProperties:
        return self._properties_for(
            setup_operator,
            bounds_certified=self.interval is not None,
        )

    def _properties_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        bounds_certified: bool,
    ) -> PreconditionerProperties:
        _validate_setup_operator(setup_operator)
        structural_transform = self.scaling == "none" or _has_diagonal_pairing(
            setup_operator.source
        )
        structural_self_adjoint = (
            structural_transform and setup_operator.properties.certifies("self_adjoint")
        )
        structural_positive = (
            bounds_certified
            and structural_transform
            and setup_operator.properties.certifies("positive_definite")
        )
        supplied_self_adjoint = self.properties is not None and self.properties.certifies(
            "self_adjoint"
        )
        supplied_positive = self.properties is not None and self.properties.certifies(
            "positive_definite"
        )
        self_adjoint = structural_self_adjoint or supplied_self_adjoint
        positive_definite = structural_positive or supplied_positive
        claims = {
            "linear": True,
            "stationary": True,
            "self_adjoint": self_adjoint,
            "positive_definite": positive_definite,
        }
        evidence = {"linear": "construction", "stationary": "construction"}
        if self_adjoint:
            evidence["self_adjoint"] = (
                self.properties.evidence_for("self_adjoint")
                if supplied_self_adjoint
                else "transformed"
            )
        if positive_definite:
            evidence["positive_definite"] = (
                self.properties.evidence_for("positive_definite")
                if supplied_positive
                else "asserted"
                if self.interval is not None
                else "verified"
            )
        return PreconditionerProperties(**claims, evidence=evidence)

    def cost_for(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy | None = None,
    ) -> PreconditionerCostEstimate:
        _validate_setup_operator(setup_operator)
        dimension = setup_operator.source.size
        itemsize = _coordinate_dtype(setup_operator.source).itemsize
        real_itemsize = np.empty(
            (), dtype=_coordinate_dtype(setup_operator.source)
        ).real.dtype.itemsize
        diagonal_operator = isinstance(setup_operator, DiagonalLinearOperator)
        sparse_operator = isinstance(setup_operator, AbstractSparseLinearOperator)
        direct_diagonal = diagonal_operator or sparse_operator
        if materialization is not None and not isinstance(
            materialization, MaterializationPolicy
        ):
            raise TypeError("materialization must be MaterializationPolicy or None.")
        needs_dense_diagonal = self.scaling == "symmetric-jacobi" and not direct_diagonal
        dense_entries = dimension * dimension
        dense_bytes = dense_entries * itemsize
        policy_allows_dense = materialization is None or (
            dense_entries <= materialization.max_entries
            and dense_bytes <= materialization.max_bytes
        )
        diagonal_feasible = not needs_dense_diagonal or (
            setup_operator.capabilities.materialize and policy_allows_dense
        )
        diagonal_matvecs = (
            0
            if self.scaling == "none"
            or direct_diagonal
            or isinstance(setup_operator, DenseLinearOperator)
            else dimension
        )
        if self.scaling == "symmetric-jacobi" and sparse_operator:
            storage = setup_operator.sparse_storage()
            index_itemsize = storage.indices.dtype.itemsize
            sparse_workspace = storage.values.size * (
                index_itemsize + jnp.dtype(bool).itemsize
            ) + dimension * (2 * index_itemsize + itemsize)
        else:
            sparse_workspace = 0
        estimate_dimension = min(self.estimation_steps, dimension)
        estimate_matvecs = 0 if self.interval is not None else estimate_dimension
        estimator_workspace = (
            0
            if self.interval is not None
            else (
                (estimate_dimension + 1) * dimension
                + estimate_dimension * estimate_dimension
            )
            * itemsize
        )
        diagonal_workspace = (
            0
            if self.scaling == "none"
            else dimension * itemsize
            if diagonal_operator or isinstance(setup_operator, DenseLinearOperator)
            else sparse_workspace
            if sparse_operator
            else dimension * dimension * itemsize
        )
        estimable = self.interval is not None or (
            self.estimation_steps >= dimension
            and setup_operator.properties.certifies("self_adjoint")
            and (self.scaling == "none" or _has_diagonal_pairing(setup_operator.source))
        )
        if not estimable:
            reason = (
                "estimated bounds require a certified full-space self-adjoint estimate"
            )
        elif not diagonal_feasible:
            reason = (
                "symmetric-Jacobi diagonal extraction exceeds materialization "
                "capabilities or policy limits"
            )
        else:
            reason = "fixed Chebyshev recurrence and optional symmetric-Jacobi scaling"
        return PreconditionerCostEstimate(
            component=self.builder_id,
            storage_bytes=(
                (2 * self.degree + 2) * real_itemsize
                + (dimension * real_itemsize if self.scaling == "symmetric-jacobi" else 0)
            ),
            preparation_workspace_bytes=diagonal_workspace + estimator_workspace,
            apply_workspace_bytes_per_rhs=(
                (5 + (2 if self.scaling == "symmetric-jacobi" else 0))
                * dimension
                * itemsize
            ),
            setup_matvec_count=diagonal_matvecs + estimate_matvecs,
            accepted=estimable and diagonal_feasible,
            reason=reason,
        )

    def prepare(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        _validate_setup_operator(setup_operator)
        effective_operator = self._effective_operator(
            setup_operator, materialization=materialization
        )
        lower, upper, bounds_source = self._bounds(effective_operator)
        alpha, beta = _chebyshev_recurrence(lower, upper, self.degree)
        properties = self._properties_for(setup_operator, bounds_certified=True)
        return ChebyshevPreconditioner(
            effective_operator,
            alpha,
            beta,
            lower_bound=lower,
            upper_bound=upper,
            scaling=self.scaling,
            bounds_source=bounds_source,
            properties=properties,
            builder_id=self.builder_id,
            setup_operator_id=setup_operator.operator_id,
            preconditioner_id=canonical_fingerprint(
                {
                    "kind": "prepared-chebyshev",
                    "builder": self.builder_id,
                    "setup_operator": setup_operator.operator_id,
                    "bounds_source": bounds_source,
                    "scaling": self.scaling,
                }
            ),
        )

    def refresh(
        self,
        preconditioner: AbstractPreconditioner,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractPreconditioner:
        if not isinstance(preconditioner, ChebyshevPreconditioner):
            raise TypeError("Chebyshev refresh requires a ChebyshevPreconditioner.")
        return self.prepare(setup_operator, materialization=materialization)

    def _effective_operator(
        self,
        setup_operator: AbstractLinearOperator,
        /,
        *,
        materialization: MaterializationPolicy,
    ) -> AbstractLinearOperator:
        if self.scaling == "none":
            return setup_operator
        diagonal = jax.lax.stop_gradient(
            assemble_diagonal(setup_operator, materialization=materialization)
        )
        real_diagonal = jnp.real(diagonal)
        real_diagonal = _validated(
            real_diagonal,
            jnp.any(~jnp.isfinite(diagonal))
            | jnp.any(jnp.imag(diagonal) != 0.0)
            | jnp.any(real_diagonal <= 0.0),
            "symmetric-Jacobi scaling requires a finite positive real diagonal.",
        )
        inverse_sqrt_diagonal = jax.lax.stop_gradient(jax.lax.rsqrt(real_diagonal))
        return _SymmetricJacobiLinearOperator(setup_operator, inverse_sqrt_diagonal)

    def _bounds(
        self,
        effective_operator: AbstractLinearOperator,
        /,
    ) -> tuple[Array, Array, ChebyshevBoundsSource]:
        coordinate_dtype = _coordinate_dtype(effective_operator.source)
        real_dtype = np.empty((), dtype=coordinate_dtype).real.dtype
        if self.interval is not None:
            lower = jnp.asarray(self.interval[0], dtype=real_dtype)
            upper = jnp.asarray(self.interval[1], dtype=real_dtype)
            return (
                jax.lax.stop_gradient(lower),
                jax.lax.stop_gradient(upper),
                "explicit",
            )
        if not effective_operator.properties.certifies("self_adjoint"):
            raise ValueError(
                "Estimated Chebyshev bounds require certified self-adjoint effective "
                "operator structure."
            )
        estimate = estimate_spectral_bounds(
            effective_operator,
            max_dimension=self.estimation_steps,
        )
        lower_widening = jnp.asarray(1.0 - self.margin, dtype=real_dtype)
        upper_widening = jnp.asarray(1.0 + self.margin, dtype=real_dtype)
        lower = jnp.asarray(estimate.lower, dtype=real_dtype) * lower_widening
        upper = jnp.asarray(estimate.upper, dtype=real_dtype) * upper_widening
        lower = jax.lax.stop_gradient(lower)
        upper = jax.lax.stop_gradient(upper)
        lower = _validated(
            lower,
            ~estimate.converged
            | ~jnp.isfinite(lower)
            | ~jnp.isfinite(upper)
            | (lower <= 0.0)
            | (lower > upper),
            "Estimated Chebyshev bounds must certify the full positive spectrum.",
        )
        return lower, upper, "estimated"


def _chebyshev_recurrence(
    lower: Array,
    upper: Array,
    degree: int,
    /,
) -> tuple[Array, Array]:
    center = 0.5 * (lower + upper)
    radius = 0.5 * (upper - lower)
    first_alpha = jnp.reciprocal(center)
    first_beta = jnp.zeros_like(center)
    safe_radius = jnp.where(radius > 0.0, radius, jnp.ones_like(radius))
    sigma = center / safe_radius
    initial_rho = safe_radius / center

    def step(rho, _):
        next_rho = jnp.reciprocal(2.0 * sigma - rho)
        alpha = 2.0 * next_rho / safe_radius
        beta = next_rho * rho
        return next_rho, (alpha, beta)

    _, (remaining_alpha, remaining_beta) = jax.lax.scan(
        step,
        initial_rho,
        xs=None,
        length=degree - 1,
    )
    alpha = jnp.concatenate((first_alpha[None], remaining_alpha))
    beta = jnp.concatenate((first_beta[None], remaining_beta))
    scalar_interval = radius == 0.0
    alpha = jnp.where(scalar_interval, jnp.full_like(alpha, first_alpha), alpha)
    beta = jnp.where(scalar_interval, jnp.zeros_like(beta), beta)
    alpha = jax.lax.stop_gradient(alpha)
    beta = jax.lax.stop_gradient(beta)
    alpha = _validated(
        alpha,
        jnp.any(~jnp.isfinite(alpha)) | jnp.any(~jnp.isfinite(beta)),
        "Chebyshev recurrence coefficients must be finite.",
    )
    return alpha, beta


__all__ = [
    "ChebyshevBoundsSource",
    "ChebyshevPreconditioner",
    "ChebyshevPreconditionerBuilder",
    "ChebyshevScaling",
]
