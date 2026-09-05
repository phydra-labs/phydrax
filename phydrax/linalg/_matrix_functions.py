#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
from jaxtyping import Array, ArrayLike, PyTree

from phydrax.ein import contract

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._polynomial._orthogonal import legendre_rule_data
from .._strict import StrictModule
from ._certificates import _operator_numeric_fingerprint
from ._linear_transform import AbstractLinearTransform, DenseLinearTransform
from ._operators import (
    AbstractLinearOperator,
    DenseLinearOperator,
    FunctionLinearOperator,
)
from ._policies import DifferentiationPolicy
from ._spaces import PyTreeSpace, RHSLayout
from .krylov import (
    arnoldi,
    KrylovBreakdownStatus,
    KrylovDecomposition,
    lanczos,
    PreparedKrylovProjection,
)
from .krylov._decompositions import _norm_from_squared


MatrixFunctionKind: TypeAlias = Literal[
    "exp",
    "phi1",
    "phi2",
    "phi3",
    "sin",
    "cos",
    "log",
    "sqrt",
    "inverse-sqrt",
    "fractional",
    "resolvent",
]
MatrixFunctionMethod: TypeAlias = Literal[
    "auto",
    "spectral",
    "chebyshev",
    "lanczos",
    "arnoldi",
]


def _operator_numerical_fingerprint(operator: AbstractLinearOperator, /) -> str:
    """Bind an operator identifier to all array-valued numerical state."""
    return canonical_fingerprint(
        {
            "operator": operator.operator_id,
            "state": array_tree_fingerprint(operator),
        }
    )


class MatrixFunctionPolicy(StrictModule):
    method: MatrixFunctionMethod = eqx.field(static=True)
    max_dimension: int = eqx.field(static=True)
    orthogonalization: Literal["modified", "double", "selective", "full"] = eqx.field(
        static=True
    )
    error_tolerance: float = eqx.field(static=True)
    differentiation: DifferentiationPolicy

    def __init__(
        self,
        method: MatrixFunctionMethod = "auto",
        /,
        *,
        max_dimension: int = 32,
        orthogonalization: Literal[
            "modified", "double", "selective", "full"
        ] = "selective",
        error_tolerance: float = 1e-8,
        differentiation: DifferentiationPolicy | None = None,
    ):
        if method not in (
            "auto",
            "spectral",
            "chebyshev",
            "lanczos",
            "arnoldi",
        ):
            raise ValueError("Unknown matrix-function method.")
        dimension = int(max_dimension)
        tolerance = float(error_tolerance)
        if dimension < 1 or not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Matrix-function dimension and tolerance must be valid.")
        if orthogonalization not in ("modified", "double", "selective", "full"):
            raise ValueError("Unknown matrix-function orthogonalization policy.")
        differentiation_ = (
            DifferentiationPolicy("algorithmic")
            if differentiation is None
            else differentiation
        )
        if not isinstance(differentiation_, DifferentiationPolicy):
            raise TypeError("differentiation must be a DifferentiationPolicy or None.")
        self.method = method
        self.max_dimension = dimension
        self.orthogonalization = orthogonalization
        self.error_tolerance = tolerance
        self.differentiation = differentiation_


class TransformDiagonalRepresentation(StrictModule):
    """Exact or truncated operator representation through one linear transform."""

    operator: AbstractLinearOperator
    modal_values: Array
    transform: AbstractLinearTransform
    representation_id: str = eqx.field(static=True)
    operator_fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        operator: AbstractLinearOperator,
        modal_values: ArrayLike,
        analysis_or_transform: ArrayLike | AbstractLinearTransform,
        synthesis: ArrayLike | None = None,
        /,
        *,
        representation_id: str | None = None,
    ):
        if isinstance(analysis_or_transform, AbstractLinearTransform):
            if synthesis is not None:
                raise ValueError(
                    "synthesis must be omitted when a linear transform is supplied."
                )
            transform = analysis_or_transform
        else:
            if synthesis is None:
                raise ValueError(
                    "Dense transform construction requires synthesis coordinates."
                )
            transform = DenseLinearTransform(analysis_or_transform, synthesis)
        self._initialize(
            operator,
            modal_values,
            transform,
            representation_id=representation_id,
        )

    @classmethod
    def from_transform(
        cls,
        operator: AbstractLinearOperator,
        modal_values: ArrayLike,
        transform: AbstractLinearTransform,
        /,
        *,
        representation_id: str | None = None,
    ) -> "TransformDiagonalRepresentation":
        return cls(
            operator,
            modal_values,
            transform,
            representation_id=representation_id,
        )

    def _initialize(
        self,
        operator: AbstractLinearOperator,
        modal_values: ArrayLike,
        transform: AbstractLinearTransform,
        /,
        *,
        representation_id: str | None,
    ) -> None:
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError("operator must be an AbstractLinearOperator.")
        if not operator.source.compatible(operator.target) or operator.batch_shape:
            raise ValueError(
                "Transform-diagonal representations require an unbatched endomorphism."
            )
        if not isinstance(transform, AbstractLinearTransform):
            raise TypeError("transform must be an AbstractLinearTransform.")
        if (
            transform.physical_space.size != operator.source.size
            or transform.modal_space.size < 1
        ):
            raise ValueError("Transform spaces do not match the represented operator.")
        values = jnp.asarray(modal_values)
        if values.shape != transform.modal_space.shape:
            raise ValueError("modal_values must match transform.modal_space shape.")
        if not bool(np.all(np.isfinite(np.asarray(values)))):
            raise ValueError("Modal values must be finite.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "transform-diagonal-representation",
                    "operator": operator.operator_id,
                    "transform": transform.transform_id,
                    "modal_shape": list(values.shape),
                }
            )
            if representation_id is None
            else str(representation_id)
        )
        if not identifier:
            raise ValueError("representation_id must be non-empty.")
        self.operator = operator
        self.modal_values = values
        self.transform = transform
        self.representation_id = identifier
        self.operator_fingerprint = _operator_numerical_fingerprint(operator)

    @property
    def rank(self) -> int:
        return self.transform.modal_space.size

    def analyze_coordinates(self, coordinates: ArrayLike, /) -> Array:
        value = jnp.asarray(coordinates).reshape(self.transform.physical_space.shape)
        return self.transform.analyze(value).reshape((-1,))

    def synthesize_coordinates(self, coefficients: ArrayLike, /) -> Array:
        value = jnp.asarray(coefficients).reshape(self.transform.modal_space.shape)
        return self.transform.synthesize(value).reshape((-1,))


class MatrixFunctionResult(StrictModule):
    value: PyTree[Array]
    error_estimate: Array
    residual_estimate: Array
    converged: Array
    effective_dimension: Array
    matvec_count: Array
    breakdown_status: Array
    method: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    provenance: str = eqx.field(static=True)


def _coerce_matrix_operator(
    operator: AbstractLinearOperator | Callable[[PyTree[Any]], PyTree[Array]],
    vector: PyTree[Any],
    /,
) -> AbstractLinearOperator:
    if isinstance(operator, AbstractLinearOperator):
        return operator
    if not callable(operator):
        raise TypeError(
            "operator must be an AbstractLinearOperator or a linear callable."
        )
    space = PyTreeSpace(vector)
    return FunctionLinearOperator(operator, source=space, target=space)


def _validate_reusable_projection(
    projection: PreparedKrylovProjection,
    operator: AbstractLinearOperator,
    coordinates: Array,
    /,
) -> Array:
    if not projection.operator.source.compatible(operator.source):
        raise ValueError("Reusable Krylov projection source space does not match.")
    if projection.operator is not operator:
        leaves = jax.tree.leaves(operator)
        if any(isinstance(leaf, jax_core.Tracer) for leaf in leaves):
            raise ValueError(
                "A traced reusable projection must be evaluated with projection.operator."
            )
        if (
            projection.operator.operator_id != operator.operator_id
            or projection.operator_fingerprint != _operator_numeric_fingerprint(operator)
        ):
            raise ValueError(
                "Reusable Krylov projection numerical operator state does not match."
            )
    same_start = jnp.array_equal(
        coordinates, projection.initial_coordinates, equal_nan=True
    )
    if isinstance(same_start, jax_core.Tracer):
        coordinates = eqx.error_if(
            coordinates,
            ~same_start,
            "Reusable Krylov projection starting vector does not match.",
        )
    elif not bool(same_start):
        raise ValueError("Reusable Krylov projection starting vector does not match.")
    return coordinates


def _unflatten_promoted(template: PyTree[Any], coordinates: Array, /) -> PyTree[Array]:
    value = jnp.asarray(coordinates)
    leaves, treedef = jax.tree.flatten(template)
    rebuilt = []
    offset = 0
    for leaf in leaves:
        array = jnp.asarray(leaf)
        size = int(array.size)
        rebuilt.append(value[offset : offset + size].reshape(array.shape))
        offset += size
    if value.shape != (offset,):
        raise ValueError(
            f"Matrix-function coordinates must have shape {(offset,)}; got {value.shape}."
        )
    return jax.tree.unflatten(treedef, rebuilt)


def _validate_real_branch_domain(
    scalar: Array,
    coordinates: Array,
    kind: MatrixFunctionKind,
    /,
    *,
    power: float | None,
    spectral: TransformDiagonalRepresentation | None,
    spectral_bounds: tuple[float, float] | None,
    positive_definite: bool,
) -> Array:
    if jnp.issubdtype(coordinates.dtype, jnp.complexfloating) or jnp.issubdtype(
        scalar.dtype, jnp.complexfloating
    ):
        return scalar
    fractional_integer = (
        kind == "fractional" and power is not None and float(power).is_integer()
    )
    requires_strict_positive = kind in ("log", "inverse-sqrt") or (
        kind == "fractional"
        and power is not None
        and power < 0.0
        and not fractional_integer
    )
    requires_nonnegative = kind == "sqrt" or (
        kind == "fractional" and not fractional_integer
    )
    requires_nonzero = fractional_integer and power is not None and power < 0.0
    if not (requires_strict_positive or requires_nonnegative or requires_nonzero):
        return scalar
    if spectral is not None and not jnp.issubdtype(
        spectral.modal_values.dtype, jnp.complexfloating
    ):
        arguments = scalar * spectral.modal_values
    elif spectral_bounds is not None:
        arguments = scalar * jnp.asarray(spectral_bounds, dtype=scalar.dtype)
    elif positive_definite:
        arguments = scalar[None]
    else:
        return scalar
    if requires_strict_positive:
        invalid = jnp.any(arguments <= 0)
    elif requires_nonnegative:
        invalid = jnp.any(arguments < 0)
    else:
        invalid = jnp.any(arguments == 0)
    return eqx.error_if(
        scalar,
        invalid,
        (
            f"Real-dtype {kind} action has a scaled spectrum outside its real "
            "domain; the requested branch is singular or complex. Use an "
            "explicitly complex input to request complex branch semantics."
        ),
    )


def _concrete_zero_scalar(value: Array, /) -> bool:
    try:
        return bool(np.asarray(value == 0))
    except (TypeError, ValueError):
        return False


def _zero_scale_action(
    coordinates: Array,
    kind: MatrixFunctionKind,
    /,
    *,
    power: float | None,
    shift: complex | float | None,
) -> Array:
    if kind in ("exp", "phi1", "cos"):
        return coordinates
    if kind == "phi2":
        return jnp.asarray(0.5, dtype=coordinates.dtype) * coordinates
    if kind == "phi3":
        return jnp.asarray(1.0 / 6.0, dtype=coordinates.dtype) * coordinates
    if kind == "sin" or kind == "sqrt":
        return jnp.zeros_like(coordinates)
    if kind == "resolvent":
        if shift is None:
            raise ValueError("A resolvent matrix function requires shift.")
        shift_ = jnp.asarray(shift)
        shift_ = eqx.error_if(
            shift_,
            shift_ == 0,
            "The zero-scale resolvent is singular when shift is zero.",
        )
        dtype = jnp.result_type(coordinates.dtype, shift_.dtype)
        return coordinates.astype(dtype) / shift_.astype(dtype)
    if kind == "fractional":
        if power is None:
            raise ValueError("A fractional matrix function requires power.")
        if power > 0.0:
            return jnp.zeros_like(coordinates)
        if power == 0.0:
            return coordinates
    raise ValueError(f"The zero-scale {kind} action is singular.")


def _batched_dense_matrix_function_action(
    operator: DenseLinearOperator,
    vector: PyTree[Any],
    scale: ArrayLike,
    /,
    *,
    kind: MatrixFunctionKind,
    power: float | None,
    shift: complex | float | None,
    policy: MatrixFunctionPolicy,
    rhs_layout: RHSLayout | None,
) -> MatrixFunctionResult:
    from ._runtime import _pack_rhs, _unpack_value

    canonical, layout = _pack_rhs(
        operator.source,
        operator.batch_shape,
        vector,
        rhs_layout,
    )
    scale_ = jnp.asarray(scale)
    if scale_.shape == ():
        scale_ = jnp.broadcast_to(scale_, operator.batch_shape)
    elif scale_.shape != operator.batch_shape:
        raise ValueError("scale must be scalar or have the exact operator batch shape.")
    matrix = operator.matrix
    if policy.differentiation.mode in ("rhs-only", "none"):
        matrix = jax.lax.stop_gradient(matrix)
    if policy.differentiation.mode == "none":
        canonical = jax.lax.stop_gradient(canonical)
        scale_ = jax.lax.stop_gradient(scale_)
    batch_count = int(np.prod(operator.batch_shape))
    size = operator.source.size
    matrices = matrix.reshape((batch_count, size, size))
    scales = scale_.reshape((batch_count,))
    right_hand_sides = canonical.reshape((batch_count, size, canonical.shape[-1]))

    def apply_one(matrix_, scale_value, right_hand_side):
        function_matrix = _small_matrix_function(
            matrix_,
            scale_value,
            kind,
            power=power,
            shift=shift,
            self_adjoint=operator.properties.certifies("self_adjoint"),
        )
        return contract(
            "ij,jk->ik",
            function_matrix,
            right_hand_side,
            backend="jax",
        )

    result = jax.vmap(apply_one)(matrices, scales, right_hand_sides)
    result = result.reshape(operator.batch_shape + (size, canonical.shape[-1]))
    finite = jnp.all(jnp.isfinite(result), axis=(-2, -1))
    zero = jnp.zeros(operator.batch_shape, dtype=result.real.dtype)
    dimension = jnp.full(operator.batch_shape, size, dtype=jnp.int32)
    return MatrixFunctionResult(
        value=_unpack_value(operator.source, result, layout),
        error_estimate=zero,
        residual_estimate=zero,
        converged=finite,
        effective_dimension=dimension,
        matvec_count=jnp.zeros(operator.batch_shape, dtype=jnp.int32),
        breakdown_status=jnp.full(
            operator.batch_shape,
            int(KrylovBreakdownStatus.NONE),
            dtype=jnp.int32,
        ),
        method="batched-dense-exact",
        kind=kind,
        provenance=(
            "exact independent dense matrix functions over a static leading batch"
        ),
    )


def matrix_function_action(
    operator: AbstractLinearOperator | Callable[[PyTree[Any]], PyTree[Array]],
    vector: PyTree[Any],
    scale: ArrayLike = 1.0,
    /,
    *,
    kind: MatrixFunctionKind,
    power: float | None = None,
    shift: complex | float | None = None,
    policy: MatrixFunctionPolicy | None = None,
    spectral: TransformDiagonalRepresentation | None = None,
    spectral_bounds: tuple[float, float] | None = None,
    decomposition: KrylovDecomposition | PreparedKrylovProjection | None = None,
    rhs_layout: RHSLayout | None = None,
) -> MatrixFunctionResult:
    """Apply a matrix function with explicit convergence and provenance."""
    operator = _coerce_matrix_operator(operator, vector)
    if not operator.source.compatible(operator.target):
        raise ValueError("Matrix functions require an endomorphism.")
    self_adjoint = operator.properties.certifies("self_adjoint")
    positive_definite = operator.properties.certifies("positive_definite")
    if kind not in (
        "exp",
        "phi1",
        "phi2",
        "phi3",
        "sin",
        "cos",
        "log",
        "sqrt",
        "inverse-sqrt",
        "fractional",
        "resolvent",
    ):
        raise ValueError("Unknown matrix-function kind.")
    if kind == "fractional" and power is None:
        raise ValueError("fractional actions require power.")
    if kind == "resolvent" and shift is None:
        raise ValueError("resolvent actions require shift.")
    if kind in ("log", "sqrt", "inverse-sqrt", "fractional") and (
        spectral is None and not positive_definite and spectral_bounds is None
    ):
        raise ValueError(
            f"{kind} requires positive-definite evidence, spectral bounds, or an "
            "explicit spectral representation."
        )
    selected = MatrixFunctionPolicy() if policy is None else policy
    if not isinstance(selected, MatrixFunctionPolicy):
        raise TypeError("policy must be a MatrixFunctionPolicy or None.")
    if operator.batch_shape:
        if not isinstance(operator, DenseLinearOperator):
            raise ValueError(
                "Batched matrix-function actions currently require an explicit "
                "DenseLinearOperator; no hidden materialization is performed."
            )
        if spectral is not None or decomposition is not None:
            raise ValueError(
                "Batched dense actions do not accept unbatched spectral/projection "
                "artifacts."
            )
        return _batched_dense_matrix_function_action(
            operator,
            vector,
            scale,
            kind=kind,
            power=power,
            shift=shift,
            policy=selected,
            rhs_layout=rhs_layout,
        )
    validated_vector = operator.source.validate(vector)
    coordinates = operator.source.flatten(validated_vector)
    scale_ = jnp.asarray(scale)
    if scale_.shape != ():
        raise ValueError("scale must be scalar.")
    scalar = scale_.astype(jnp.result_type(coordinates.dtype, scale_.dtype))
    scalar = _validate_real_branch_domain(
        scalar,
        coordinates,
        kind,
        power=power,
        spectral=spectral,
        spectral_bounds=spectral_bounds,
        positive_definite=positive_definite,
    )
    method = selected.method
    if method == "auto":
        if spectral is not None:
            method = "spectral"
        elif self_adjoint and spectral_bounds is not None:
            method = "chebyshev"
        elif self_adjoint:
            method = "lanczos"
        else:
            method = "arnoldi"

    zero_scale = _concrete_zero_scalar(scalar)
    if zero_scale:
        exact = _zero_scale_action(
            coordinates,
            kind,
            power=power,
            shift=shift,
        )
        zero = jnp.asarray(0.0, dtype=coordinates.real.dtype)
        finite = jnp.all(jnp.isfinite(exact))
        return MatrixFunctionResult(
            value=_unflatten_promoted(validated_vector, exact),
            error_estimate=zero,
            residual_estimate=zero,
            converged=finite,
            effective_dimension=jnp.asarray(0, dtype=jnp.int32),
            matvec_count=jnp.asarray(0, dtype=jnp.int32),
            breakdown_status=jnp.asarray(
                int(KrylovBreakdownStatus.NONE), dtype=jnp.int32
            ),
            method=method,
            kind=kind,
            provenance="exact zero-scale matrix-function action",
        )

    if method == "spectral":
        if spectral is None:
            raise ValueError("Spectral execution requires a representation.")
        same_operator = spectral.operator is operator
        if not same_operator and (
            spectral.operator.operator_id != operator.operator_id
            or spectral.operator_fingerprint != _operator_numerical_fingerprint(operator)
        ):
            raise ValueError(
                "Spectral representation numerical operator state does not match."
            )
        multipliers = _scalar_function(
            scalar * spectral.modal_values,
            kind,
            power=power,
            shift=shift,
        )
        value = spectral.synthesize_coordinates(
            multipliers * spectral.analyze_coordinates(coordinates)
        )
        complete = spectral.rank == operator.source.size
        error = jnp.asarray(
            0.0 if complete else jnp.nan,
            dtype=coordinates.real.dtype,
        )
        return MatrixFunctionResult(
            value=_unflatten_promoted(validated_vector, value),
            error_estimate=error,
            residual_estimate=error,
            converged=jnp.all(jnp.isfinite(value)) & jnp.asarray(complete),
            effective_dimension=jnp.asarray(spectral.rank, dtype=jnp.int32),
            matvec_count=jnp.asarray(0, dtype=jnp.int32),
            breakdown_status=jnp.asarray(
                int(KrylovBreakdownStatus.NONE), dtype=jnp.int32
            ),
            method="spectral",
            kind=kind,
            provenance=(
                "explicit spectral representation"
                if complete
                else "truncated explicit spectral representation"
            ),
        )

    def action(value):
        return operator.target.flatten(operator.mv(operator.source.unflatten(value)))

    if method == "chebyshev":
        if not self_adjoint:
            raise ValueError(
                "Chebyshev execution requires certified self-adjoint structure."
            )
        if spectral_bounds is None:
            raise ValueError("Chebyshev execution requires spectral_bounds.")
        result, error_estimate = _chebyshev_action(
            action,
            coordinates,
            scalar,
            kind,
            spectral_bounds,
            degree=selected.max_dimension,
            power=power,
            shift=shift,
        )
        # A finite sampled Chebyshev tail is not a certified bound on the
        # omitted infinite tail.  Preserve the approximation but never claim
        # convergence without such a bound.
        error_estimate = jnp.asarray(jnp.nan, dtype=coordinates.real.dtype)
        finite = jnp.all(jnp.isfinite(result))
        return MatrixFunctionResult(
            value=_unflatten_promoted(validated_vector, result),
            error_estimate=error_estimate,
            residual_estimate=error_estimate,
            converged=finite & (error_estimate <= selected.error_tolerance),
            effective_dimension=jnp.asarray(selected.max_dimension, dtype=jnp.int32),
            matvec_count=jnp.asarray(max(selected.max_dimension - 1, 0), dtype=jnp.int32),
            breakdown_status=jnp.asarray(
                int(KrylovBreakdownStatus.NONE), dtype=jnp.int32
            ),
            method="chebyshev",
            kind=kind,
            provenance="Chebyshev interval approximation without a certified tail bound",
        )

    reused_projection = isinstance(decomposition, PreparedKrylovProjection)
    if reused_projection:
        if selected.method not in ("auto", decomposition.method):
            raise ValueError(
                "Matrix-function policy method conflicts with the reusable projection."
            )
        method = decomposition.method
        coordinates = _validate_reusable_projection(decomposition, operator, coordinates)
        decomposition = decomposition.decomposition
        matvec_count = jnp.asarray(0, dtype=jnp.int32)
    elif decomposition is not None:
        if not isinstance(decomposition, KrylovDecomposition):
            raise TypeError(
                "decomposition must be a PreparedKrylovProjection, "
                "KrylovDecomposition, or None."
            )
        raise ValueError(
            "Unbound Krylov decompositions do not carry numerical operator and "
            "normalized-start bindings; use prepare_krylov_projection."
        )
    else:
        dimension = min(selected.max_dimension, operator.source.size)
        if method == "lanczos":
            if not self_adjoint:
                raise ValueError("Lanczos requires certified self-adjoint structure.")
            decomposition = lanczos(
                action,
                coordinates,
                max_dimension=dimension,
                inner=lambda left, right: operator.source.inner(
                    operator.source.unflatten(left),
                    operator.source.unflatten(right),
                ),
                orthogonalization=selected.orthogonalization,
            )
        elif method == "arnoldi":
            decomposition = arnoldi(
                action,
                coordinates,
                max_dimension=dimension,
                inner=lambda left, right: operator.source.inner(
                    operator.source.unflatten(left),
                    operator.source.unflatten(right),
                ),
                orthogonalization=selected.orthogonalization,
            )
        else:
            raise ValueError(f"Unsupported matrix-function method {method!r}.")
        matvec_count = decomposition.matvec_count
    projected = decomposition.projected[:-1]
    function = _small_matrix_function(
        projected,
        scalar,
        kind,
        power=power,
        shift=shift,
        self_adjoint=(method == "lanczos"),
        active_dimension=decomposition.effective_dimension,
    )
    norm = _norm_from_squared(operator.source.inner(validated_vector, validated_vector))
    dimension = decomposition.effective_dimension
    active = jnp.arange(projected.shape[0]) < dimension
    coefficients = jnp.where(active, function[:, 0], 0)
    basis_coordinates = jnp.swapaxes(decomposition.basis[:-1], -1, -2)
    result_coordinates = norm * (basis_coordinates @ coefficients)

    capacity = projected.shape[0]
    index = jnp.maximum(dimension - 1, 0)
    augmented = jnp.zeros(
        (capacity + 1, capacity + 1),
        dtype=projected.dtype,
    )
    augmented = augmented.at[:capacity, :capacity].set(projected)
    augmented = augmented.at[dimension, index].set(
        decomposition.projected[dimension, index]
    )
    augmented = augmented.at[dimension, dimension].set(projected[index, index])
    omitted_function = _small_matrix_function(
        augmented,
        scalar,
        kind,
        power=power,
        shift=shift,
        self_adjoint=False,
        active_dimension=dimension + 1,
    )
    residual_estimate = jnp.where(
        dimension > 0,
        norm * jnp.abs(omitted_function[dimension, 0]),
        0.0,
    )
    finite = jnp.all(jnp.isfinite(result_coordinates)) & jnp.isfinite(residual_estimate)
    acceptable_status = (
        (decomposition.breakdown_status == int(KrylovBreakdownStatus.NONE))
        | (decomposition.breakdown_status == int(KrylovBreakdownStatus.HAPPY))
        | (decomposition.breakdown_status == int(KrylovBreakdownStatus.NEAR_BREAKDOWN))
        | (
            decomposition.breakdown_status
            == int(KrylovBreakdownStatus.RANK_DEFICIENT_START)
        )
    )
    converged = (
        finite & acceptable_status & (residual_estimate <= selected.error_tolerance)
    )
    return MatrixFunctionResult(
        value=_unflatten_promoted(validated_vector, result_coordinates),
        error_estimate=residual_estimate,
        residual_estimate=residual_estimate,
        converged=converged,
        effective_dimension=dimension,
        matvec_count=matvec_count,
        breakdown_status=decomposition.breakdown_status,
        method=method,
        kind=kind,
        provenance=(
            f"reused bound {method} projection with omitted-mode estimate"
            if reused_projection
            else f"phydrax-native {method} projection with omitted-mode estimate"
        ),
    )


def matrix_exponential_action(
    operator, vector, scale: ArrayLike = 1.0, /, **kwargs
) -> MatrixFunctionResult:
    return matrix_function_action(operator, vector, scale, kind="exp", **kwargs)


def matrix_phi1_action(
    operator, vector, scale: ArrayLike = 1.0, /, **kwargs
) -> MatrixFunctionResult:
    return matrix_function_action(operator, vector, scale, kind="phi1", **kwargs)


def matrix_phi2_action(
    operator, vector, scale: ArrayLike = 1.0, /, **kwargs
) -> MatrixFunctionResult:
    return matrix_function_action(operator, vector, scale, kind="phi2", **kwargs)


def matrix_phi3_action(
    operator, vector, scale: ArrayLike = 1.0, /, **kwargs
) -> MatrixFunctionResult:
    return matrix_function_action(operator, vector, scale, kind="phi3", **kwargs)


def _phi_function_value(value: Array, order: int, /) -> Array:
    """Evaluate first through third scalar phi functions with stable zero limits."""
    order_ = int(order)
    if order_ not in (1, 2, 3):
        raise ValueError("Phi-function order must be one, two, or three.")
    epsilon = jnp.finfo(value.real.dtype).eps
    threshold = epsilon ** (1.0 / 3.0) if order_ == 3 else jnp.sqrt(epsilon)
    safe = jnp.where(jnp.abs(value) > threshold, value, 1)
    if order_ == 1:
        series = (
            1
            + value / 2
            + value**2 / 6
            + value**3 / 24
            + value**4 / 120
            + value**5 / 720
            + value**6 / 5040
        )
        quotient = jnp.expm1(value) / safe
    elif order_ == 2:
        series = (
            0.5
            + value / 6
            + value**2 / 24
            + value**3 / 120
            + value**4 / 720
            + value**5 / 5040
            + value**6 / 40320
        )
        quotient = (jnp.expm1(value) - value) / safe**2
    else:
        series = (
            1.0 / 6.0
            + value / 24
            + value**2 / 120
            + value**3 / 720
            + value**4 / 5040
            + value**5 / 40320
            + value**6 / 362880
        )
        quotient = (jnp.expm1(value) - value - 0.5 * value**2) / safe**3
    return jnp.where(jnp.abs(value) > threshold, quotient, series)


def _scalar_function(
    value: Array,
    kind: MatrixFunctionKind,
    /,
    *,
    power: float | None,
    shift: complex | float | None,
) -> Array:
    if kind == "exp":
        return jnp.exp(value)
    if kind == "sin":
        return jnp.sin(value)
    if kind == "cos":
        return jnp.cos(value)
    if kind == "log":
        return jnp.log(value)
    if kind == "sqrt":
        return jnp.sqrt(value)
    if kind == "inverse-sqrt":
        return 1.0 / jnp.sqrt(value)
    if kind == "fractional":
        if power is None:
            raise ValueError("A fractional matrix function requires power.")
        return value**power
    if kind == "resolvent":
        if shift is None:
            raise ValueError("A resolvent matrix function requires shift.")
        shift_ = jnp.asarray(shift)
        dtype = jnp.result_type(value.dtype, shift_.dtype)
        return 1.0 / (shift_.astype(dtype) - value.astype(dtype))
    if kind == "phi1":
        return _phi_function_value(value, 1)
    if kind == "phi2":
        return _phi_function_value(value, 2)
    if kind == "phi3":
        return _phi_function_value(value, 3)
    raise ValueError("Unknown matrix-function kind.")


def _small_matrix_function(
    matrix: Array,
    scale: Array,
    kind: MatrixFunctionKind,
    /,
    *,
    power: float | None,
    shift: complex | float | None,
    self_adjoint: bool,
    active_dimension: Array | None = None,
) -> Array:
    scaled = scale * matrix
    if active_dimension is not None:
        active = jnp.arange(matrix.shape[0]) < active_dimension
        active_block = active[:, None] & active[None, :]
        scaled = jnp.where(active_block, scaled, 0)
        scaled = scaled + jnp.diag((~active).astype(scaled.dtype))
    if kind == "exp":
        return jsp.linalg.expm(scaled)
    if kind in ("phi1", "phi2", "phi3"):
        size = matrix.shape[0]
        blocks = {"phi1": 2, "phi2": 3, "phi3": 4}[kind]
        augmented = jnp.zeros((blocks * size, blocks * size), dtype=scaled.dtype)
        augmented = augmented.at[:size, :size].set(scaled)
        identity = jnp.eye(size, dtype=scaled.dtype)
        for index in range(blocks - 1):
            augmented = augmented.at[
                index * size : (index + 1) * size,
                (index + 1) * size : (index + 2) * size,
            ].set(identity)
        exponential = jsp.linalg.expm(augmented)
        return exponential[:size, (blocks - 1) * size : blocks * size]
    if kind in ("sin", "cos"):
        complex_dtype = jnp.result_type(scaled.dtype, jnp.complex64)
        complex_scaled = scaled.astype(complex_dtype)
        positive = jsp.linalg.expm(1j * complex_scaled)
        negative = jsp.linalg.expm(-1j * complex_scaled)
        result = (
            (positive - negative) / (2j) if kind == "sin" else (positive + negative) / 2
        )
        return (
            result
            if jnp.issubdtype(scaled.dtype, jnp.complexfloating)
            else jnp.real(result)
        )
    if kind == "resolvent":
        if shift is None:
            raise ValueError("A resolvent matrix function requires shift.")
        shift_ = jnp.asarray(shift)
        dtype = jnp.result_type(scaled.dtype, shift_.dtype)
        matrix_ = scaled.astype(dtype)
        identity = jnp.eye(matrix.shape[0], dtype=dtype)
        return jnp.linalg.solve(
            shift_.astype(dtype) * identity - matrix_,
            identity,
        )
    if self_adjoint and not jnp.issubdtype(scale.dtype, jnp.complexfloating):
        eigenvalues, eigenvectors = jnp.linalg.eigh(scaled)
        multiplier_arguments = (
            eigenvalues.astype(scaled.dtype)
            if jnp.issubdtype(scaled.dtype, jnp.complexfloating)
            else eigenvalues
        )
        multipliers = _scalar_function(
            multiplier_arguments,
            kind,
            power=power,
            shift=shift,
        )
        return (eigenvectors * multipliers) @ jnp.conj(eigenvectors.T)
    return _general_primary_matrix_function(
        scaled,
        kind,
        power=power,
    )


def _general_primary_matrix_function(
    matrix: Array,
    kind: MatrixFunctionKind,
    /,
    *,
    power: float | None,
) -> Array:
    """Evaluate primary functions without assuming a diagonalizable matrix."""
    if kind == "fractional" and power is not None and float(power).is_integer():
        return jnp.linalg.matrix_power(matrix, int(power))
    logarithm = _general_matrix_logarithm(matrix)
    if kind == "log":
        return logarithm
    if kind == "sqrt":
        exponent = 0.5
    elif kind == "inverse-sqrt":
        exponent = -0.5
    elif kind == "fractional":
        if power is None:
            raise ValueError("A fractional matrix function requires power.")
        exponent = power
    else:
        raise ValueError(f"No general primary-matrix route exists for {kind!r}.")
    return jsp.linalg.expm(jnp.asarray(exponent, dtype=matrix.real.dtype) * logarithm)


def _general_matrix_logarithm(matrix: Array, /) -> Array:
    """Principal logarithm by a matrix-resolvent Gauss--Legendre formula."""
    size = matrix.shape[0]
    identity = jnp.eye(size, dtype=matrix.dtype)
    difference = matrix - identity
    rule = legendre_rule_data(32, "gauss", dtype=matrix.real.dtype)
    nodes = jnp.asarray(0.5 * (rule.nodes + 1.0), dtype=matrix.real.dtype)
    weights = jnp.asarray(0.5 * rule.weights, dtype=matrix.real.dtype)
    resolvents = identity + nodes[:, None, None] * difference
    right = jnp.broadcast_to(difference, resolvents.shape)
    integrand = jnp.linalg.solve(resolvents, right)
    return jnp.sum(weights[:, None, None] * integrand, axis=0)


def _chebyshev_action(
    action,
    vector: Array,
    scale: Array,
    kind: MatrixFunctionKind,
    bounds: tuple[float, float],
    /,
    *,
    degree: int,
    power: float | None,
    shift: complex | float | None,
) -> tuple[Array, Array]:
    lower, upper = (float(value) for value in bounds)
    if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
        raise ValueError("spectral_bounds must be finite and strictly increasing.")
    indices = jnp.arange(degree, dtype=vector.real.dtype)
    theta = jnp.pi * (indices + 0.5) / degree
    nodes = jnp.cos(theta)
    center = 0.5 * (upper + lower)
    radius = 0.5 * (upper - lower)
    samples = _scalar_function(
        scale * (center + radius * nodes),
        kind,
        power=power,
        shift=shift,
    )
    coefficients = (2.0 / degree) * (jnp.cos(indices[:, None] * theta[None, :]) @ samples)

    def normalized(value):
        return (action(value) - center * value) / radius

    previous = vector
    last_term = 0.5 * coefficients[0] * previous
    result = last_term
    if degree == 1:
        return result, jnp.linalg.norm(last_term)
    current = normalized(vector)
    last_term = coefficients[1] * current
    result = result + last_term
    for index in range(2, degree):
        following = 2 * normalized(current) - previous
        last_term = coefficients[index] * following
        result = result + last_term
        previous, current = current, following
    return result, jnp.linalg.norm(last_term)


__all__ = [
    "MatrixFunctionKind",
    "MatrixFunctionMethod",
    "MatrixFunctionPolicy",
    "MatrixFunctionResult",
    "TransformDiagonalRepresentation",
    "matrix_exponential_action",
    "matrix_function_action",
    "matrix_phi1_action",
    "matrix_phi2_action",
    "matrix_phi3_action",
]
