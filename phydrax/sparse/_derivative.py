#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, Key, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import (
    AbstractVectorSpace,
    LinearizationPolicy,
    OperatorProperties,
    PreparedLinearization,
)
from ..linalg._operators import _validate_properties
from ..linalg._spaces import _has_euclidean_pairing
from ._coloring import (
    native_coloring,
    SparseColoring,
    SparseDerivativeCompiler,
    SparseDerivativeKind,
    SparseDerivativeMode,
    SparseHessianMode,
    SparseJacobianMode,
)
from ._linear import SparseCoordinateOperator
from ._pattern import SparsePattern
from ._relation import EdgeRelation


_USE_COMPILED_ARGUMENTS = object()


class SparseDerivativePlan(StrictModule):
    """Reusable sparse derivative structure with native compressed JAX execution."""

    function: Callable[[Array, Any], Array]
    arguments: Any
    source: AbstractVectorSpace
    target: AbstractVectorSpace
    coloring: SparseColoring
    properties: OperatorProperties
    argument_structure: Any = eqx.field(static=True)
    argument_specs: tuple[tuple[tuple[int, ...], str], ...] = eqx.field(static=True)
    derivative_kind: SparseDerivativeKind = eqx.field(static=True)
    chunk_size: int | None = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array, Any], Array],
        arguments: Any,
        source: AbstractVectorSpace,
        target: AbstractVectorSpace,
        coloring: SparseColoring,
        properties: OperatorProperties,
        /,
        *,
        derivative_kind: SparseDerivativeKind,
        chunk_size: int | None,
        plan_id: str,
    ):
        if not callable(function):
            raise TypeError("function must be callable.")
        if not isinstance(source, AbstractVectorSpace) or not isinstance(
            target, AbstractVectorSpace
        ):
            raise TypeError("source and target must be AbstractVectorSpace values.")
        if not isinstance(coloring, SparseColoring):
            raise TypeError("coloring must be a SparseColoring.")
        if not isinstance(properties, OperatorProperties):
            raise TypeError("properties must be OperatorProperties.")
        if derivative_kind not in ("jacobian", "hessian"):
            raise ValueError(f"Unknown sparse derivative kind {derivative_kind!r}.")
        chunk = None if chunk_size is None else int(chunk_size)
        if chunk is not None and chunk < 1:
            raise ValueError("chunk_size must be positive or None.")
        identifier = str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        arguments_ = _canonicalize_arguments(arguments)
        argument_structure, argument_specs = _argument_signature(arguments_)
        self.function = function
        self.arguments = arguments_
        self.source = source
        self.target = target
        self.coloring = coloring
        self.properties = properties
        self.argument_structure = argument_structure
        self.argument_specs = argument_specs
        self.derivative_kind = derivative_kind
        self.chunk_size = chunk
        self.plan_id = identifier

    @property
    def pattern(self) -> SparsePattern:
        return self.coloring.pattern

    @property
    def num_colors(self) -> int:
        return self.coloring.num_colors

    @property
    def nnz(self) -> int:
        return self.pattern.nnz

    @property
    def mode(self) -> SparseDerivativeMode:
        return self.coloring.mode

    def coefficients(
        self,
        point: PyTree[Any],
        args: Any = _USE_COMPILED_ARGUMENTS,
        /,
    ) -> Array:
        """Evaluate derivative entries in the plan's canonical sparse route order."""

        coordinates = self.source.flatten(self.source.validate(point))
        arguments = (
            self.arguments
            if args is _USE_COMPILED_ARGUMENTS
            else _canonicalize_arguments(args)
        )
        _validate_argument_signature(
            arguments,
            self.argument_structure,
            self.argument_specs,
        )
        return _evaluate_compressed(
            self.function,
            coordinates,
            arguments,
            self.coloring,
            derivative_kind=self.derivative_kind,
            chunk_size=self.chunk_size,
        )

    def operator(
        self,
        point: PyTree[Any],
        args: Any = _USE_COMPILED_ARGUMENTS,
        /,
    ) -> SparseCoordinateOperator:
        """Evaluate this derivative as a structured sparse linear operator."""

        return SparseCoordinateOperator(
            self.pattern.relation,
            self.coefficients(point, args),
            source=self.source,
            target=self.target,
            properties=self.properties,
            operator_id=f"{self.plan_id}:operator",
        )


class PreparedSparseDerivative(StrictModule):
    """One sparse derivative value paired with reusable linearized actions."""

    plan: SparseDerivativePlan
    operator: SparseCoordinateOperator
    linearization: PreparedLinearization
    prepared_id: str = eqx.field(static=True)

    def jvp(self, tangent: PyTree[Any], /) -> PyTree[Array]:
        return self.linearization.jvp(tangent)

    def vjp(self, cotangent: PyTree[Any], /) -> PyTree[Array]:
        return self.linearization.vjp(cotangent)


def prepare_sparse_linearization(
    plan: SparseDerivativePlan,
    point: PyTree[Any],
    args: Any = _USE_COMPILED_ARGUMENTS,
    /,
) -> PreparedSparseDerivative:
    """Evaluate sparse coefficients once and retain reusable JVP/VJP actions."""
    if not isinstance(plan, SparseDerivativePlan):
        raise TypeError("plan must be a SparseDerivativePlan.")
    point_ = plan.source.validate(point)
    coordinates = plan.source.flatten(point_)
    arguments = (
        plan.arguments
        if args is _USE_COMPILED_ARGUMENTS
        else _canonicalize_arguments(args)
    )
    _validate_argument_signature(
        arguments,
        plan.argument_structure,
        plan.argument_specs,
    )
    if plan.derivative_kind == "hessian":
        primal_coordinates = jax.grad(plan.function, argnums=0)(
            coordinates,
            arguments,
        )
    else:
        primal_coordinates = jnp.asarray(plan.function(coordinates, arguments))
    primal = plan.target.unflatten(primal_coordinates)
    operator = plan.operator(point_, arguments)
    identifier = canonical_fingerprint(
        {
            "kind": "prepared-sparse-linearization",
            "plan": plan.plan_id,
        }
    )
    linearization = PreparedLinearization(
        source=plan.source,
        target=plan.target,
        point=point_,
        primal=primal,
        pushforward=operator.mv,
        pullback=operator.transpose_mv,
        policy=LinearizationPolicy(),
        linearization_id=identifier,
    )
    return PreparedSparseDerivative(
        plan=plan,
        operator=operator,
        linearization=linearization,
        prepared_id=identifier,
    )


class SparseDerivativeVerification(StrictModule):
    """Sample-point matrix-free verification diagnostics for one derivative plan."""

    passed: Array
    maximum_absolute_error: Array
    maximum_relative_error: Array
    reference_scale: Array
    num_probes: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    scope: str = eqx.field(static=True)

    def __init__(
        self,
        passed: Array,
        maximum_absolute_error: Array,
        maximum_relative_error: Array,
        reference_scale: Array,
        /,
        *,
        num_probes: int,
        plan_id: str,
    ):
        self.passed = jnp.asarray(passed, dtype=bool)
        self.maximum_absolute_error = jnp.asarray(maximum_absolute_error)
        self.maximum_relative_error = jnp.asarray(maximum_relative_error)
        self.reference_scale = jnp.asarray(reference_scale)
        self.num_probes = int(num_probes)
        self.plan_id = str(plan_id)
        self.scope = "sample-point"


def compile_sparse_jacobian(
    function: Callable[[PyTree[Array], Any], PyTree[Array]],
    point: PyTree[Any],
    /,
    *,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    sample_args: Any = None,
    structure: EdgeRelation | SparsePattern | SparseColoring | None = None,
    compiler: SparseDerivativeCompiler = "auto",
    mode: SparseJacobianMode | None = None,
    symmetric: bool = False,
    chunk_size: int | None = None,
    properties: OperatorProperties | None = None,
    plan_id: str | None = None,
) -> SparseDerivativePlan:
    """Compile a sparse Jacobian into a provider-neutral native execution plan."""

    if not callable(function):
        raise TypeError("function must be callable.")
    _validate_spaces(source, target)
    point_ = source.validate(point)

    def coordinate_function(coordinates: Array, arguments: Any) -> Array:
        value = function(source.unflatten(coordinates), arguments)
        return target.flatten(value)

    return _compile_sparse_derivative(
        coordinate_function,
        source.flatten(point_),
        sample_args,
        source=source,
        target=target,
        structure=structure,
        compiler=compiler,
        derivative_kind="jacobian",
        mode=mode,
        symmetric=bool(symmetric),
        chunk_size=chunk_size,
        properties=properties,
        plan_id=plan_id,
    )


def compile_sparse_hessian(
    function: Callable[[PyTree[Array], Any], Array],
    point: PyTree[Any],
    /,
    *,
    space: AbstractVectorSpace,
    sample_args: Any = None,
    structure: EdgeRelation | SparsePattern | SparseColoring | None = None,
    compiler: SparseDerivativeCompiler = "auto",
    mode: SparseHessianMode | None = None,
    chunk_size: int | None = None,
    properties: OperatorProperties | None = None,
    plan_id: str | None = None,
) -> SparseDerivativePlan:
    """Compile a real Euclidean Hessian into a native sparse execution plan."""

    if not callable(function):
        raise TypeError("function must be callable.")
    _validate_spaces(space, space)
    if not _has_euclidean_pairing(space):
        raise ValueError(
            "Sparse Hessians currently require a Euclidean pairing; "
            "non-Euclidean Hessians require an explicit primal-to-dual contract."
        )
    point_ = space.validate(point)

    def coordinate_function(coordinates: Array, arguments: Any) -> Array:
        value = jnp.asarray(function(space.unflatten(coordinates), arguments))
        if value.shape != ():
            raise ValueError("Sparse Hessian functions must return a scalar array.")
        return value

    return _compile_sparse_derivative(
        coordinate_function,
        space.flatten(point_),
        sample_args,
        source=space,
        target=space,
        structure=structure,
        compiler=compiler,
        derivative_kind="hessian",
        mode=mode,
        symmetric=True,
        chunk_size=chunk_size,
        properties=properties,
        plan_id=plan_id,
    )


def verify_sparse_derivative(
    plan: SparseDerivativePlan,
    point: PyTree[Any],
    /,
    *,
    key: Key[Array, ""],
    args: Any = _USE_COMPILED_ARGUMENTS,
    num_probes: int = 3,
    relative_tolerance: float | None = None,
    absolute_tolerance: float | None = None,
) -> SparseDerivativeVerification:
    """Compare sparse actions with direct JVPs at one point without densification."""

    if not isinstance(plan, SparseDerivativePlan):
        raise TypeError("plan must be a SparseDerivativePlan.")
    probes = int(num_probes)
    if probes < 1:
        raise ValueError("num_probes must be positive.")
    coordinates = plan.source.flatten(plan.source.validate(point))
    arguments = (
        plan.arguments
        if args is _USE_COMPILED_ARGUMENTS
        else _canonicalize_arguments(args)
    )
    _validate_argument_signature(
        arguments,
        plan.argument_structure,
        plan.argument_specs,
    )
    directions = jr.normal(key, (probes, plan.source.size), dtype=coordinates.dtype)
    operator = plan.operator(point, arguments)

    def sparse_action(direction: Array) -> Array:
        image = operator.mv(plan.source.unflatten(direction))
        return plan.target.flatten(image)

    if plan.derivative_kind == "jacobian":

        def reference_action(direction: Array) -> Array:
            return jax.jvp(
                lambda value: plan.function(value, arguments),
                (coordinates,),
                (direction,),
            )[1]

    else:
        gradient = jax.grad(lambda value: plan.function(value, arguments))

        def reference_action(direction: Array) -> Array:
            return jax.jvp(gradient, (coordinates,), (direction,))[1]

    sparse_values = jax.vmap(sparse_action)(directions)
    reference_values = jax.vmap(reference_action)(directions)
    error = jnp.abs(sparse_values - reference_values).reshape((-1,))
    reference_magnitude = jnp.abs(reference_values).reshape((-1,))
    maximum_absolute_error = jnp.max(
        jnp.concatenate((jnp.zeros((1,), dtype=error.dtype), error))
    )
    reference_scale = jnp.max(
        jnp.concatenate(
            (jnp.zeros((1,), dtype=reference_magnitude.dtype), reference_magnitude)
        )
    )
    epsilon = jnp.finfo(coordinates.dtype).eps
    relative = (
        100.0 * epsilon if relative_tolerance is None else float(relative_tolerance)
    )
    absolute = (
        100.0 * epsilon if absolute_tolerance is None else float(absolute_tolerance)
    )
    if relative < 0.0 or absolute < 0.0:
        raise ValueError("Verification tolerances must be non-negative.")
    denominator = jnp.maximum(reference_scale, jnp.finfo(coordinates.dtype).tiny)
    maximum_relative_error = maximum_absolute_error / denominator
    passed = maximum_absolute_error <= absolute + relative * reference_scale
    return SparseDerivativeVerification(
        passed,
        maximum_absolute_error,
        maximum_relative_error,
        reference_scale,
        num_probes=probes,
        plan_id=plan.plan_id,
    )


def _compile_sparse_derivative(
    function: Callable[[Array, Any], Array],
    coordinates: Array,
    sample_args: Any,
    /,
    *,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    structure: EdgeRelation | SparsePattern | SparseColoring | None,
    compiler: SparseDerivativeCompiler,
    derivative_kind: SparseDerivativeKind,
    mode: SparseDerivativeMode | None,
    symmetric: bool,
    chunk_size: int | None,
    properties: OperatorProperties | None,
    plan_id: str | None,
) -> SparseDerivativePlan:
    if compiler not in ("auto", "native", "asdex"):
        raise ValueError(f"Unknown sparse derivative compiler {compiler!r}.")
    chunk = None if chunk_size is None else int(chunk_size)
    if chunk is not None and chunk < 1:
        raise ValueError("chunk_size must be positive or None.")
    sample_args = _canonicalize_arguments(sample_args)
    _, argument_specs = _argument_signature(sample_args)
    if not jnp.issubdtype(coordinates.dtype, jnp.floating):
        raise TypeError(
            "Sparse derivative compilation currently requires real floating-point "
            f"coordinates; got {coordinates.dtype}."
        )
    converted = eqx.filter_closure_convert(function, coordinates, sample_args)
    output = jax.eval_shape(converted, coordinates, sample_args)
    if derivative_kind == "jacobian":
        if output.shape != (target.size,):
            raise ValueError(
                f"Sparse Jacobian output must have shape {(target.size,)}; "
                f"got {output.shape}."
            )
    elif output.shape != ():
        raise ValueError("Sparse Hessian output must be scalar.")
    if not jnp.issubdtype(output.dtype, jnp.floating):
        raise TypeError(
            "Sparse derivative compilation currently requires real floating-point "
            f"outputs; got {output.dtype}."
        )
    if derivative_kind == "jacobian" and jnp.dtype(output.dtype) != jnp.dtype(
        coordinates.dtype
    ):
        raise TypeError(
            "Sparse derivative source and target coordinates must use the same "
            f"dtype; got {coordinates.dtype} and {output.dtype}."
        )

    coloring = _resolve_coloring(
        converted,
        coordinates,
        sample_args,
        source=source,
        target=target,
        structure=structure,
        compiler=compiler,
        derivative_kind=derivative_kind,
        mode=mode,
        symmetric=symmetric,
    )
    _validate_coloring_shape(coloring, source, target, derivative_kind)
    properties_ = _derivative_properties(
        source,
        target,
        derivative_kind=derivative_kind,
        symmetric=symmetric or coloring.symmetric,
        properties=properties,
    )
    payload = {
        "kind": f"sparse-{derivative_kind}-plan",
        "source": source.space_id,
        "target": target.space_id,
        "coloring": coloring.coloring_id,
        "chunk_size": chunk,
        "argument_specs": [
            {"shape": list(shape), "dtype": dtype} for shape, dtype in argument_specs
        ],
        "properties": {
            "diagonal": properties_.diagonal,
            "triangular": properties_.triangular,
            "self_adjoint": properties_.self_adjoint,
            "positive_definite": properties_.positive_definite,
            "positive_semidefinite": properties_.positive_semidefinite,
            "block_diagonal": properties_.block_diagonal,
            "rank": properties_.rank,
            "evidence": properties_.evidence,
        },
    }
    identifier = canonical_fingerprint(payload) if plan_id is None else str(plan_id)
    if not identifier:
        raise ValueError("plan_id must be non-empty.")
    return SparseDerivativePlan(
        converted,
        sample_args,
        source,
        target,
        coloring,
        properties_,
        derivative_kind=derivative_kind,
        chunk_size=chunk,
        plan_id=identifier,
    )


def _resolve_coloring(
    function: Callable[[Array, Any], Array],
    coordinates: Array,
    sample_args: Any,
    /,
    *,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    structure: EdgeRelation | SparsePattern | SparseColoring | None,
    compiler: SparseDerivativeCompiler,
    derivative_kind: SparseDerivativeKind,
    mode: SparseDerivativeMode | None,
    symmetric: bool,
) -> SparseColoring:
    if isinstance(structure, SparseColoring):
        if compiler == "asdex":
            raise ValueError("compiler must be 'auto' or 'native' for a reused coloring.")
        _validate_reused_coloring(structure, derivative_kind, mode, symmetric)
        return structure
    if isinstance(structure, EdgeRelation):
        pattern = SparsePattern(
            structure,
            symmetric=symmetric,
            origin="declared",
        )
    elif isinstance(structure, SparsePattern):
        pattern = structure
        if symmetric and not pattern.symmetric:
            raise ValueError("The supplied sparse pattern must be symmetric.")
    elif structure is None:
        pattern = None
    else:
        raise TypeError(
            "structure must be an EdgeRelation, SparsePattern, SparseColoring, or None."
        )

    selected_compiler = compiler
    if selected_compiler == "auto":
        selected_compiler = "native" if pattern is not None else "asdex"
    if selected_compiler == "native":
        if pattern is None:
            raise ValueError("Native sparse compilation requires a declared pattern.")
        return native_coloring(
            pattern,
            derivative_kind=derivative_kind,
            mode=mode,
        )

    from ._asdex import compile_asdex_coloring

    return compile_asdex_coloring(
        function,
        coordinates,
        sample_args,
        source=source,
        target=target,
        pattern=pattern,
        derivative_kind=derivative_kind,
        mode=mode,
        symmetric=symmetric,
    )


def _evaluate_compressed(
    function: Callable[[Array, Any], Array],
    coordinates: Array,
    arguments: Any,
    coloring: SparseColoring,
    /,
    *,
    derivative_kind: SparseDerivativeKind,
    chunk_size: int | None,
) -> Array:
    if coloring.num_colors == 0:
        return jnp.empty((0,), dtype=coordinates.dtype)

    mode = coloring.mode
    if derivative_kind == "jacobian":
        if mode == "fwd":
            _, action = jax.linearize(
                lambda value: function(value, arguments), coordinates
            )
        elif mode == "rev":
            _, pullback = jax.vjp(lambda value: function(value, arguments), coordinates)
            action = lambda seed: pullback(seed)[0]
        else:
            raise ValueError(f"Invalid Jacobian mode {mode!r}.")
    else:
        scalar_function = lambda value: function(value, arguments)
        if mode == "fwd_over_rev":
            _, action = jax.linearize(jax.grad(scalar_function), coordinates)
        elif mode == "rev_over_rev":
            _, pullback = jax.vjp(jax.grad(scalar_function), coordinates)
            action = lambda seed: pullback(seed)[0]
        elif mode == "rev_over_fwd":
            action = lambda seed: jax.grad(
                lambda value: jax.jvp(
                    scalar_function,
                    (value,),
                    (seed,),
                )[1]
            )(coordinates)
        else:
            raise ValueError(f"Invalid Hessian mode {mode!r}.")

    compressed = _apply_color_chunks(
        action,
        coloring.colors,
        coloring.num_colors,
        coordinates.dtype,
        chunk_size,
    )
    return compressed[coloring.gather_colors, coloring.gather_elements]


def _apply_color_chunks(
    action: Callable[[Array], Array],
    colors: Array,
    num_colors: int,
    dtype: jnp.dtype,
    chunk_size: int | None,
    /,
) -> Array:
    chunk = num_colors if chunk_size is None else min(chunk_size, num_colors)
    blocks = []
    for start in range(0, num_colors, chunk):
        stop = min(start + chunk, num_colors)
        color_ids = jnp.arange(start, stop, dtype=jnp.int32)
        seeds = (color_ids[:, None] == colors[None, :]).astype(dtype)
        blocks.append(jax.vmap(action)(seeds))
    return blocks[0] if len(blocks) == 1 else jnp.concatenate(blocks, axis=0)


def _validate_reused_coloring(
    coloring: SparseColoring,
    derivative_kind: SparseDerivativeKind,
    mode: SparseDerivativeMode | None,
    symmetric: bool,
    /,
) -> None:
    if derivative_kind == "jacobian" and coloring.mode not in ("fwd", "rev"):
        raise ValueError("A Jacobian plan requires a Jacobian coloring mode.")
    if derivative_kind == "hessian" and coloring.mode not in (
        "fwd_over_rev",
        "rev_over_fwd",
        "rev_over_rev",
    ):
        raise ValueError("A Hessian plan requires a Hessian coloring mode.")
    if mode is not None and coloring.mode != mode:
        raise ValueError("Reused coloring mode does not match the requested mode.")
    if symmetric and not coloring.pattern.symmetric:
        raise ValueError("Reused coloring pattern must be symmetric.")


def _validate_coloring_shape(
    coloring: SparseColoring,
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    derivative_kind: SparseDerivativeKind,
    /,
) -> None:
    if coloring.pattern.shape != (target.size, source.size):
        raise ValueError(
            "Sparse coloring shape does not match the declared source and target spaces."
        )
    if derivative_kind == "hessian" and not coloring.pattern.symmetric:
        raise ValueError("Sparse Hessian coloring must have a symmetric pattern.")


def _derivative_properties(
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    /,
    *,
    derivative_kind: SparseDerivativeKind,
    symmetric: bool,
    properties: OperatorProperties | None,
) -> OperatorProperties:
    if properties is None:
        self_adjoint = (
            (derivative_kind == "hessian" or symmetric)
            and source.compatible(target)
            and _has_euclidean_pairing(source)
        )
        resolved = OperatorProperties(
            self_adjoint=self_adjoint,
            evidence={"self_adjoint": "construction"} if self_adjoint else None,
        )
    else:
        resolved = properties
    if not isinstance(resolved, OperatorProperties):
        raise TypeError("properties must be an OperatorProperties value or None.")
    _validate_properties(resolved, source, target)
    return resolved


def _canonicalize_arguments(arguments: Any, /) -> Any:
    def as_array(leaf: Any, /) -> Array:
        if isinstance(leaf, (str, bytes)):
            raise TypeError(
                "Sparse derivative arguments must be array-like PyTree leaves."
            )
        return jnp.asarray(leaf)

    return jax.tree.map(as_array, arguments)


def _argument_signature(
    arguments: Any, /
) -> tuple[Any, tuple[tuple[tuple[int, ...], str], ...]]:
    leaves, structure = jax.tree_util.tree_flatten(arguments)
    specs: list[tuple[tuple[int, ...], str]] = []
    for leaf in leaves:
        if isinstance(leaf, (str, bytes)):
            raise TypeError(
                "Sparse derivative arguments must be array-like PyTree leaves."
            )
        array = jnp.asarray(leaf)
        specs.append(
            (tuple(int(size) for size in array.shape), jnp.dtype(array.dtype).str)
        )
    return structure, tuple(specs)


def _validate_argument_signature(
    arguments: Any,
    expected_structure: Any,
    expected_specs: tuple[tuple[tuple[int, ...], str], ...],
    /,
) -> None:
    structure, specs = _argument_signature(arguments)
    if structure != expected_structure:
        raise ValueError(
            "Runtime derivative arguments have a different PyTree structure."
        )
    if specs != expected_specs:
        raise ValueError(
            "Runtime derivative arguments must preserve every sample leaf shape and dtype."
        )


def _validate_spaces(
    source: AbstractVectorSpace,
    target: AbstractVectorSpace,
    /,
) -> None:
    if not isinstance(source, AbstractVectorSpace) or not isinstance(
        target, AbstractVectorSpace
    ):
        raise TypeError("source and target must be AbstractVectorSpace values.")


__all__ = [
    "PreparedSparseDerivative",
    "SparseDerivativePlan",
    "SparseDerivativeVerification",
    "compile_sparse_hessian",
    "compile_sparse_jacobian",
    "prepare_sparse_linearization",
    "verify_sparse_derivative",
]
