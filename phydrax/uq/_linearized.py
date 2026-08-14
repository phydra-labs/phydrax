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
from jax.flatten_util import ravel_pytree
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._uncertainty import UncertaintySource, validate_uncertainty_source
from ._covariance import (
    _apply_covariance,
    _dense_factor_directions,
    _matrix_tolerance,
    _validate_covariance_template,
    _validate_tree_shapes,
    AbstractCovariance,
    covariance_representation,
    CovarianceOperator,
    DenseCovariance,
    DiagonalCovariance,
    FactorCovariance,
)


class LinearizedVarianceEstimate(StrictModule):
    """Hutchinson estimate of a first-order output-covariance diagonal."""

    variance: PyTree[Array]
    standard_error: PyTree[Array]
    num_probes: int = eqx.field(static=True)
    probe_distribution: str = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    approximation: str = eqx.field(static=True)

    def __init__(
        self,
        variance: PyTree[Array],
        standard_error: PyTree[Array],
        /,
        *,
        num_probes: int,
        probe_distribution: str,
    ):
        self.variance = variance
        self.standard_error = standard_error
        self.num_probes = int(num_probes)
        self.probe_distribution = str(probe_distribution)
        self.exact = False
        self.approximation = "first_order_hutchinson"


class LinearizedDenseCovariance(StrictModule):
    """Guarded dense output covariance with a deterministic PyTree layout."""

    matrix: Array
    hermitian_defect: Array
    leaf_paths: tuple[str, ...] = eqx.field(static=True)
    leaf_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    unravel: Any = eqx.field(static=True)

    def __init__(
        self,
        matrix: Array,
        hermitian_defect: Array,
        /,
        *,
        output_template: PyTree[Array],
        unravel: Any,
    ):
        path_leaves = jax.tree_util.tree_flatten_with_path(output_template)[0]
        self.matrix = jnp.asarray(matrix)
        self.hermitian_defect = jnp.asarray(hermitian_defect)
        self.leaf_paths = tuple(
            jax.tree_util.keystr(path) or "<root>" for path, _ in path_leaves
        )
        self.leaf_shapes = tuple(
            tuple(int(size) for size in leaf.shape) for _, leaf in path_leaves
        )
        self.dimension = int(self.matrix.shape[0])
        self.unravel = unravel

    def covariance_vector_product(
        self,
        vector: PyTree[Array],
        /,
    ) -> PyTree[Array]:
        """Apply the materialized covariance and restore the output PyTree."""
        flat_vector, _ = ravel_pytree(vector)
        if flat_vector.shape != (self.dimension,):
            raise ValueError(
                "Dense covariance vector has incompatible flattened dimension; "
                f"expected {self.dimension}, got {flat_vector.size}."
            )
        return self.unravel(self.matrix @ flat_vector)


class LinearizedPropagationResult(StrictModule):
    """Matrix-free first-order moments around one nominal scientific input."""

    mean: PyTree[Array]
    input_template: PyTree[Array]
    covariance: AbstractCovariance
    pushforward_fn: Callable[[PyTree[Array]], PyTree[Array]]
    pullback_fn: Callable[[PyTree[Array]], PyTree[Array]]
    input_unravel: Any = eqx.field(static=True)
    output_unravel: Any = eqx.field(static=True)
    source: UncertaintySource = eqx.field(static=True)
    input_covariance_representation: str = eqx.field(static=True)
    approximation: str = eqx.field(static=True)
    input_dimension: int = eqx.field(static=True)
    output_dimension: int = eqx.field(static=True)
    coordinate_covariance: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        mean: PyTree[Array],
        input_template: PyTree[Array],
        covariance: AbstractCovariance,
        pushforward: Callable[[PyTree[Array]], PyTree[Array]],
        pullback: Callable[[PyTree[Array]], PyTree[Array]],
        source: UncertaintySource,
        coordinate_covariance: bool = True,
    ):
        _validate_array_tree(input_template, owner="Linearized input", finite=True)
        _validate_array_tree(mean, owner="Linearized output", finite=True)
        input_dimension, input_unravel = _validate_covariance_template(
            covariance,
            input_template,
        )
        flat_output, output_unravel = ravel_pytree(mean)
        output_dimension = int(flat_output.size)
        if output_dimension <= 0:
            raise ValueError("Linearized outputs must contain at least one scalar.")

        input_structure = jax.tree_util.tree_structure(input_template)
        input_shapes = tuple(
            tuple(int(size) for size in leaf.shape)
            for leaf in jax.tree_util.tree_leaves(input_template)
        )
        output_structure = jax.tree_util.tree_structure(mean)
        output_shapes = tuple(
            tuple(int(size) for size in leaf.shape)
            for leaf in jax.tree_util.tree_leaves(mean)
        )
        input_probe = jax.tree_util.tree_map(jnp.ones_like, input_template)
        output_probe = jax.tree_util.tree_map(jnp.ones_like, mean)
        _validate_tree_shapes(
            pushforward(input_probe),
            output_structure,
            output_shapes,
            owner="Linearized pushforward output",
        )
        _validate_tree_shapes(
            pullback(output_probe),
            input_structure,
            input_shapes,
            owner="Linearized pullback output",
        )

        self.mean = mean
        self.input_template = input_template
        self.covariance = covariance
        self.pushforward_fn = pushforward
        self.pullback_fn = pullback
        self.input_unravel = input_unravel
        self.output_unravel = output_unravel
        self.source = validate_uncertainty_source(
            source,
            owner="LinearizedPropagationResult.source",
        )
        self.input_covariance_representation = covariance_representation(covariance)
        self.approximation = "first_order"
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.coordinate_covariance = bool(coordinate_covariance)

    def pushforward(self, tangent: PyTree[Array], /) -> PyTree[Array]:
        """Apply the local derivative to an input tangent."""
        _validate_like(tangent, self.input_template, owner="Input tangent")
        return self.pushforward_fn(tangent)

    def pullback(self, cotangent: PyTree[Array], /) -> PyTree[Array]:
        """Apply the local Euclidean or Hermitian pullback."""
        _validate_like(cotangent, self.mean, owner="Output cotangent")
        return self.pullback_fn(cotangent)

    def covariance_vector_product(
        self,
        vector: PyTree[Array],
        /,
    ) -> PyTree[Array]:
        """Apply ``J Cₓ Jᴴ`` without materializing either Jacobian or covariance."""
        input_covector = self.pullback(vector)
        covariance_vector = _apply_covariance(
            self.covariance,
            input_covector,
            unravel=self.input_unravel,
        )
        return self.pushforward(covariance_vector)

    def exact_variance(
        self,
        *,
        batch_size: int | None = None,
    ) -> PyTree[Array]:
        """Return the exact output diagonal under the first-order approximation."""
        if not self.coordinate_covariance:
            raise ValueError(
                "Pointwise variance requires a coordinate covariance; "
                "Hilbert-operator covariance supports only vector products."
            )
        if isinstance(self.covariance, CovarianceOperator):
            raise ValueError(
                "Exact variance requires diagonal, dense, or factor input covariance; "
                "use estimate_variance for a covariance operator."
            )
        chunk = _batch_size(batch_size, self.input_dimension)
        accumulator = jax.tree_util.tree_map(
            lambda value: jnp.zeros_like(jnp.real(value)),
            self.mean,
        )
        if isinstance(self.covariance, DiagonalCovariance):
            directions = None
            rank = self.input_dimension
        else:
            directions, rank = self._factor_directions()
        for start in range(0, rank, chunk):
            stop = min(start + chunk, rank)
            if directions is None:
                selected = self._diagonal_directions(start, stop)
            else:
                selected = jax.tree_util.tree_map(
                    lambda value: value[start:stop],
                    directions,
                )
            propagated = jax.vmap(self.pushforward_fn)(selected)
            contribution = jax.tree_util.tree_map(
                lambda value: jnp.sum(jnp.abs(value) ** 2, axis=0),
                propagated,
            )
            accumulator = jax.tree_util.tree_map(
                lambda total, value: total + value,
                accumulator,
                contribution,
            )
        return accumulator

    def estimate_variance(
        self,
        key: Array,
        /,
        *,
        num_probes: int,
        batch_size: int | None = None,
    ) -> LinearizedVarianceEstimate:
        """Estimate the output diagonal with keyed Hutchinson probes."""
        if not self.coordinate_covariance:
            raise ValueError(
                "Variance estimation requires a coordinate covariance; "
                "Hilbert-operator covariance supports only vector products."
            )
        count = int(num_probes)
        if count < 2:
            raise ValueError("num_probes must be at least two to estimate uncertainty.")
        chunk = _batch_size(batch_size, count)
        flat_mean, _ = ravel_pytree(self.mean)
        real_dtype = jnp.asarray(jnp.real(flat_mean)).dtype
        sample_sum = jnp.zeros((self.output_dimension,), dtype=real_dtype)
        sample_square_sum = jnp.zeros_like(sample_sum)
        complex_output = jnp.issubdtype(flat_mean.dtype, jnp.complexfloating)
        distribution = "complex_phase" if complex_output else "rademacher"

        for start in range(0, count, chunk):
            size = min(chunk, count - start)
            probes = _flat_probes(
                key,
                start,
                size,
                self.output_dimension,
                dtype=flat_mean.dtype,
                complex_output=complex_output,
            )
            probe_trees = jax.vmap(self.output_unravel)(probes)
            applied_trees = jax.vmap(self.covariance_vector_product)(probe_trees)
            applied = jax.vmap(lambda value: ravel_pytree(value)[0])(applied_trees)
            samples = jnp.real(jnp.conj(probes) * applied)
            sample_sum = sample_sum + jnp.sum(samples, axis=0)
            sample_square_sum = sample_square_sum + jnp.sum(samples**2, axis=0)

        estimate = sample_sum / count
        sample_variance = (sample_square_sum - sample_sum**2 / count) / (count - 1)
        standard_error = jnp.sqrt(jnp.maximum(sample_variance, 0.0) / count)
        return LinearizedVarianceEstimate(
            self.output_unravel(estimate),
            self.output_unravel(standard_error),
            num_probes=count,
            probe_distribution=distribution,
        )

    def materialize_covariance(
        self,
        *,
        max_dimension: int = 256,
        batch_size: int | None = None,
    ) -> LinearizedDenseCovariance:
        """Materialize a guarded dense output covariance in flattened output order."""
        if not self.coordinate_covariance:
            raise ValueError(
                "Dense materialization requires a coordinate covariance; "
                "Hilbert-operator covariance supports only vector products."
            )
        maximum = int(max_dimension)
        if maximum <= 0:
            raise ValueError("max_dimension must be positive.")
        if self.output_dimension > maximum:
            raise ValueError(
                "Dense output covariance exceeds max_dimension; "
                f"got {self.output_dimension} > {maximum}."
            )
        chunk = _batch_size(batch_size, self.output_dimension)
        flat_mean, _ = ravel_pytree(self.mean)
        columns: list[Array] = []
        for start in range(0, self.output_dimension, chunk):
            stop = min(start + chunk, self.output_dimension)
            probes = jnp.eye(
                self.output_dimension,
                dtype=flat_mean.dtype,
            )[start:stop]
            probe_trees = jax.vmap(self.output_unravel)(probes)
            applied_trees = jax.vmap(self.covariance_vector_product)(probe_trees)
            applied = jax.vmap(lambda value: ravel_pytree(value)[0])(applied_trees)
            columns.append(jnp.swapaxes(applied, 0, 1))
        matrix = jnp.concatenate(tuple(columns), axis=1)
        adjoint = jnp.conj(matrix.T)
        scale = jnp.maximum(
            jnp.linalg.norm(matrix), jnp.finfo(jnp.real(matrix).dtype).eps
        )
        defect = jnp.linalg.norm(matrix - adjoint) / scale
        tolerance = _matrix_tolerance(matrix) / jnp.maximum(
            jnp.max(jnp.abs(matrix)),
            jnp.ones((), dtype=jnp.real(matrix).dtype),
        )
        if bool(defect > tolerance):
            raise ValueError(
                "Materialized output covariance is not Hermitian within tolerance; "
                f"relative defect={float(defect):.3e}."
            )
        hermitian = 0.5 * (matrix + adjoint)
        return LinearizedDenseCovariance(
            hermitian,
            defect,
            output_template=self.mean,
            unravel=self.output_unravel,
        )

    def _diagonal_directions(
        self,
        start: int,
        stop: int,
        /,
    ) -> PyTree[Array]:
        covariance = self.covariance
        if not isinstance(covariance, DiagonalCovariance):
            raise TypeError("Diagonal directions require diagonal input covariance.")
        flat_variance, _ = ravel_pytree(covariance.variance)
        flat_template, _ = ravel_pytree(self.input_template)
        size = stop - start
        indices = jnp.arange(start, stop)
        flat_directions = jnp.zeros(
            (size, self.input_dimension),
            dtype=jnp.result_type(flat_template.dtype, flat_variance.dtype),
        )
        flat_directions = flat_directions.at[jnp.arange(size), indices].set(
            jnp.sqrt(flat_variance[indices])
        )
        directions = jax.vmap(self.input_unravel)(flat_directions)
        return jax.tree_util.tree_map(
            lambda value, template: value.astype(template.dtype),
            directions,
            self.input_template,
        )

    def _factor_directions(self) -> tuple[PyTree[Array], int]:
        def input_dtype(directions):
            return jax.tree_util.tree_map(
                lambda value, template: value.astype(template.dtype),
                directions,
                self.input_template,
            )

        if isinstance(self.covariance, FactorCovariance):
            return input_dtype(self.covariance.factors), self.covariance.rank
        if isinstance(self.covariance, DenseCovariance):
            directions = _dense_factor_directions(
                self.covariance,
                unravel=self.input_unravel,
            )
            return input_dtype(directions), self.input_dimension
        raise TypeError("Covariance operators do not expose factor directions.")


def propagate_linearized(
    forward: Callable[[PyTree[Array]], PyTree[Array]],
    center: PyTree[Array],
    covariance: AbstractCovariance,
    /,
    *,
    source: UncertaintySource = "input",
    complex_linear: bool = False,
) -> LinearizedPropagationResult:
    """Propagate covariance through a local JAX linearization without a Jacobian."""
    if not callable(forward):
        raise TypeError("forward must be callable.")
    _validate_array_tree(center, owner="Linearization center", finite=True)
    if not isinstance(covariance, AbstractCovariance):
        raise TypeError("covariance must implement AbstractCovariance.")
    mean, pushforward = jax.linearize(forward, center)
    _validate_array_tree(mean, owner="Linearized forward output", finite=True)
    complex_values = _tree_is_complex(center) or _tree_is_complex(mean)
    if complex_values and not complex_linear:
        raise ValueError(
            "Complex linearized propagation requires complex_linear=True; represent "
            "non-holomorphic models with explicit real and imaginary coordinates."
        )
    transpose = jax.linear_transpose(pushforward, center)

    pullback = jax.tree_util.Partial(
        _complex_pullback if complex_values else _real_pullback,
        transpose,
    )

    return LinearizedPropagationResult(
        mean=mean,
        input_template=center,
        covariance=covariance,
        pushforward=pushforward,
        pullback=pullback,
        source=source,
    )


def propagate_linearized_map(
    mean: PyTree[Array],
    input_template: PyTree[Array],
    covariance: AbstractCovariance,
    /,
    *,
    pushforward: Callable[[PyTree[Array]], PyTree[Array]],
    pullback: Callable[[PyTree[Array]], PyTree[Array]],
    source: UncertaintySource = "input",
    coordinate_covariance: bool = True,
) -> LinearizedPropagationResult:
    """Propagate covariance through caller-supplied tangent and adjoint actions."""
    if not callable(pushforward) or not callable(pullback):
        raise TypeError("pushforward and pullback must be callable.")
    if not isinstance(covariance, AbstractCovariance):
        raise TypeError("covariance must implement AbstractCovariance.")
    return LinearizedPropagationResult(
        mean=mean,
        input_template=input_template,
        covariance=covariance,
        pushforward=pushforward,
        pullback=pullback,
        source=source,
        coordinate_covariance=coordinate_covariance,
    )


def _validate_array_tree(
    value: PyTree[Any],
    /,
    *,
    owner: str,
    finite: bool,
) -> None:
    leaves = jax.tree_util.tree_leaves(value)
    if not leaves or any(not eqx.is_inexact_array(leaf) for leaf in leaves):
        raise TypeError(f"{owner} must be a non-empty PyTree of inexact arrays.")
    if finite and any(bool(jnp.any(~jnp.isfinite(leaf))) for leaf in leaves):
        raise ValueError(f"{owner} must be finite.")


def _validate_like(
    value: PyTree[Array],
    template: PyTree[Array],
    /,
    *,
    owner: str,
) -> None:
    _validate_tree_shapes(
        value,
        jax.tree_util.tree_structure(template),
        tuple(
            tuple(int(size) for size in leaf.shape)
            for leaf in jax.tree_util.tree_leaves(template)
        ),
        owner=owner,
    )


def _real_pullback(transpose: Callable[..., Any], cotangent: PyTree[Array], /):
    return transpose(cotangent)[0]


def _complex_pullback(transpose: Callable[..., Any], cotangent: PyTree[Array], /):
    return _conjugate_tree(transpose(_conjugate_tree(cotangent))[0])


def _tree_is_complex(value: PyTree[Array], /) -> bool:
    return any(
        jnp.issubdtype(leaf.dtype, jnp.complexfloating)
        for leaf in jax.tree_util.tree_leaves(value)
    )


def _conjugate_tree(value: PyTree[Array], /) -> PyTree[Array]:
    return jax.tree_util.tree_map(jnp.conj, value)


def _flat_probes(
    key: Array,
    start: int,
    count: int,
    dimension: int,
    /,
    *,
    dtype: Any,
    complex_output: bool,
) -> Array:
    indices = jnp.arange(start, start + count, dtype=jnp.uint32)
    keys = jax.vmap(lambda index: jr.fold_in(key, index))(indices)
    if complex_output:
        phase_indices = jax.vmap(
            lambda probe_key: jr.randint(
                probe_key,
                (dimension,),
                0,
                4,
            )
        )(keys)
        phases = jnp.asarray((1.0, 1.0j, -1.0, -1.0j), dtype=dtype)
        return phases[phase_indices]
    return jax.vmap(
        lambda probe_key: jr.rademacher(
            probe_key,
            (dimension,),
            dtype=dtype,
        )
    )(keys)


def _batch_size(value: int | None, total: int, /) -> int:
    if value is None:
        return total
    size = int(value)
    if size <= 0:
        raise ValueError("batch_size must be positive or None.")
    return min(size, total)


__all__ = [
    "LinearizedDenseCovariance",
    "LinearizedPropagationResult",
    "LinearizedVarianceEstimate",
    "propagate_linearized",
    "propagate_linearized_map",
]
