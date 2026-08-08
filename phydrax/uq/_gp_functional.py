#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from phydrax.kernels import AbstractPositiveDefiniteKernel

from .._strict import StrictModule
from ._gp_backend import (
    exact_gp_conditioner_from_covariances,
    exact_gp_log_probability,
    fitc_factors_from_covariances,
    sparse_gp_conditioner_from_covariances,
    sparse_gp_log_probability_from_factors,
)
from ._gp_condition import _sample_gaussian_psd


class LinearDifferentialFunctional(StrictModule):
    """Finite linear combination of coordinate partial derivatives."""

    coefficients: Array
    derivative_orders: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    functional_id: str = eqx.field(static=True)

    def __init__(
        self,
        derivative_orders: Sequence[Sequence[int]],
        coefficients: ArrayLike,
        /,
        *,
        functional_id: str = "linear_differential",
    ):
        orders = tuple(tuple(int(order) for order in term) for term in derivative_orders)
        if not orders or not orders[0]:
            raise ValueError(
                "A differential functional needs at least one term and coordinate."
            )
        coordinate_size = len(orders[0])
        if any(len(term) != coordinate_size for term in orders):
            raise ValueError("All derivative multi-indices need equal coordinate size.")
        if any(order < 0 for term in orders for order in term):
            raise ValueError("Derivative orders must be nonnegative.")
        coefficient_array = jnp.asarray(coefficients, dtype=float)
        if coefficient_array.ndim not in (1, 2) or coefficient_array.shape[-1] != len(
            orders
        ):
            raise ValueError("coefficients must have shape (term,) or (point, term).")
        if coefficient_array.ndim == 2 and coefficient_array.shape[0] <= 0:
            raise ValueError("Point-varying coefficients cannot be empty.")
        if not isinstance(functional_id, str) or not functional_id:
            raise ValueError("functional_id must be a nonempty string.")
        self.coefficients = eqx.error_if(
            coefficient_array,
            jnp.any(~jnp.isfinite(coefficient_array)),
            "Differential-functional coefficients must be finite.",
        )
        self.derivative_orders = orders
        self.functional_id = functional_id

    @property
    def coordinate_size(self) -> int:
        return len(self.derivative_orders[0])

    @property
    def num_terms(self) -> int:
        return len(self.derivative_orders)

    @property
    def required_derivative_order(self) -> int:
        return max(sum(term) for term in self.derivative_orders)

    @property
    def is_point_varying(self) -> bool:
        return self.coefficients.ndim == 2

    def coefficient_matrix(self, point_count: int, /) -> Array:
        if self.coefficients.ndim == 1:
            return jnp.broadcast_to(
                self.coefficients,
                (int(point_count), self.num_terms),
            )
        if self.coefficients.shape[0] != int(point_count):
            raise ValueError(
                "Point-varying functional coefficients must align with block points."
            )
        return self.coefficients

    def __add__(self, other: Any) -> LinearDifferentialFunctional:
        if not isinstance(other, LinearDifferentialFunctional):
            return NotImplemented
        if self.coordinate_size != other.coordinate_size:
            raise ValueError("Differential functionals need equal coordinate size.")
        left, right = _align_coefficients(self.coefficients, other.coefficients)
        return LinearDifferentialFunctional(
            self.derivative_orders + other.derivative_orders,
            jnp.concatenate((left, right), axis=-1),
            functional_id=f"SumFunctional[{self.functional_id},{other.functional_id}]",
        )

    def __sub__(self, other: Any) -> LinearDifferentialFunctional:
        if not isinstance(other, LinearDifferentialFunctional):
            return NotImplemented
        return self + (-other)

    def __neg__(self) -> LinearDifferentialFunctional:
        return LinearDifferentialFunctional(
            self.derivative_orders,
            -self.coefficients,
            functional_id=f"NegFunctional[{self.functional_id}]",
        )

    def __mul__(self, scale: Any) -> LinearDifferentialFunctional:
        scale_array = jnp.asarray(scale, dtype=float)
        if scale_array.ndim == 0:
            coefficients = self.coefficients * scale_array
        elif scale_array.ndim == 1:
            if scale_array.shape[0] <= 0:
                raise ValueError("Point-varying functional scale cannot be empty.")
            if self.coefficients.ndim == 1:
                coefficients = scale_array[:, None] * self.coefficients
            elif self.coefficients.shape[0] == scale_array.shape[0]:
                coefficients = self.coefficients * scale_array[:, None]
            else:
                raise ValueError(
                    "Point-varying scale must align with functional coefficients."
                )
        else:
            raise ValueError("Functional scale must be scalar or pointwise.")
        return LinearDifferentialFunctional(
            self.derivative_orders,
            coefficients,
            functional_id=f"ScaledFunctional[{self.functional_id}]",
        )

    def __rmul__(self, scale: Any) -> LinearDifferentialFunctional:
        return self * scale


def value_functional(coordinate_size: int, /) -> LinearDifferentialFunctional:
    """Evaluate a latent field value."""
    size = _coordinate_size(coordinate_size)
    return LinearDifferentialFunctional(
        ((0,) * size,),
        jnp.ones((1,)),
        functional_id="ValueFunctional",
    )


def partial_derivative_functional(
    coordinate_size: int,
    coordinate: int,
    /,
    *,
    order: int = 1,
) -> LinearDifferentialFunctional:
    """Evaluate one coordinate partial derivative."""
    size = _coordinate_size(coordinate_size)
    axis = int(coordinate)
    derivative_order = int(order)
    if axis < 0 or axis >= size:
        raise ValueError("Derivative coordinate is outside the point dimension.")
    if derivative_order <= 0:
        raise ValueError("Derivative order must be positive.")
    orders = [0] * size
    orders[axis] = derivative_order
    return LinearDifferentialFunctional(
        (tuple(orders),),
        jnp.ones((1,)),
        functional_id=f"PartialDerivativeFunctional[{axis},{derivative_order}]",
    )


def directional_derivative_functional(
    direction: ArrayLike,
    /,
) -> LinearDifferentialFunctional:
    """Evaluate a first derivative along a supplied coordinate direction."""
    vector = jnp.asarray(direction, dtype=float)
    if vector.ndim != 1 or vector.shape[0] <= 0:
        raise ValueError("direction must be a nonempty coordinate vector.")
    vector = eqx.error_if(
        vector,
        jnp.any(~jnp.isfinite(vector)) | (jnp.linalg.norm(vector) <= 0.0),
        "direction must be finite and nonzero.",
    )
    size = int(vector.shape[0])
    orders = tuple(
        tuple(1 if coordinate == axis else 0 for coordinate in range(size))
        for axis in range(size)
    )
    return LinearDifferentialFunctional(
        orders,
        vector,
        functional_id="DirectionalDerivativeFunctional",
    )


def laplacian_functional(coordinate_size: int, /) -> LinearDifferentialFunctional:
    """Evaluate the coordinate Laplacian of a latent field."""
    size = _coordinate_size(coordinate_size)
    orders = tuple(
        tuple(2 if coordinate == axis else 0 for coordinate in range(size))
        for axis in range(size)
    )
    return LinearDifferentialFunctional(
        orders,
        jnp.ones((size,)),
        functional_id="LaplacianFunctional",
    )


class FunctionalObservationBlock(StrictModule):
    """Named points sharing one differential-operator structure."""

    points: Array
    functional: LinearDifferentialFunctional
    name: str = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        functional: LinearDifferentialFunctional,
        /,
        *,
        name: str,
    ):
        point_array = _as_points(points)
        if point_array.shape[0] <= 0:
            raise ValueError("Functional observation blocks cannot be empty.")
        if not isinstance(functional, LinearDifferentialFunctional):
            raise TypeError("functional must be a LinearDifferentialFunctional.")
        if point_array.shape[1] != functional.coordinate_size:
            raise ValueError("Functional and point coordinate sizes must agree.")
        functional.coefficient_matrix(int(point_array.shape[0]))
        if not isinstance(name, str) or not name:
            raise ValueError("Functional block name must be a nonempty string.")
        self.points = eqx.error_if(
            point_array,
            jnp.any(~jnp.isfinite(point_array)),
            "Functional observation points must be finite.",
        )
        self.functional = functional
        self.name = name

    @property
    def num_observations(self) -> int:
        return int(self.points.shape[0])

    @property
    def coordinate_size(self) -> int:
        return int(self.points.shape[1])


class FunctionalDesign(StrictModule):
    """Ordered collection of heterogeneous linear-functional blocks."""

    blocks: tuple[FunctionalObservationBlock, ...]
    block_index: Array
    block_names: tuple[str, ...] = eqx.field(static=True)
    block_offsets: tuple[int, ...] = eqx.field(static=True)
    coordinate_size: int = eqx.field(static=True)

    def __init__(self, blocks: Sequence[FunctionalObservationBlock], /):
        block_tuple = tuple(blocks)
        if not block_tuple:
            raise ValueError("A functional design needs at least one block.")
        if any(
            not isinstance(block, FunctionalObservationBlock) for block in block_tuple
        ):
            raise TypeError(
                "Functional designs contain FunctionalObservationBlock values."
            )
        coordinate_size = block_tuple[0].coordinate_size
        if any(block.coordinate_size != coordinate_size for block in block_tuple):
            raise ValueError("Functional blocks need equal point coordinate size.")
        names = tuple(block.name for block in block_tuple)
        if len(set(names)) != len(names):
            raise ValueError("Functional block names must be distinct.")
        counts = tuple(block.num_observations for block in block_tuple)
        offsets = [0]
        for count in counts:
            offsets.append(offsets[-1] + count)
        self.blocks = block_tuple
        self.block_index = jnp.concatenate(
            tuple(
                jnp.full((count,), index, dtype=jnp.int32)
                for index, count in enumerate(counts)
            )
        )
        self.block_names = names
        self.block_offsets = tuple(offsets)
        self.coordinate_size = coordinate_size

    @classmethod
    def from_points(
        cls,
        points: ArrayLike,
        functional: LinearDifferentialFunctional,
        /,
        *,
        name: str = "functional",
    ) -> FunctionalDesign:
        return cls((FunctionalObservationBlock(points, functional, name=name),))

    @property
    def num_observations(self) -> int:
        return int(self.block_index.shape[0])

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def flatten(
        self,
        values: ArrayLike | tuple[ArrayLike, ...],
        /,
        *,
        name: str = "functional values",
    ) -> Array:
        """Align flat values or one vector per named block."""
        if isinstance(values, tuple):
            if len(values) != self.num_blocks:
                raise ValueError(f"{name} must contain one array per functional block.")
            arrays = tuple(jnp.asarray(value, dtype=float) for value in values)
            for array, block in zip(arrays, self.blocks, strict=True):
                if array.shape != (block.num_observations,):
                    raise ValueError(f"{name} block arrays must align with block points.")
            return jnp.concatenate(arrays)
        array = jnp.asarray(values, dtype=float)
        if array.shape != (self.num_observations,):
            raise ValueError(f"{name} must align with flattened functional observations.")
        return array

    def split(self, values: ArrayLike, /) -> tuple[Array, ...]:
        """Split a flat vector in named-block order."""
        flat = self.flatten(values)
        return tuple(
            flat[start:stop]
            for start, stop in zip(
                self.block_offsets[:-1],
                self.block_offsets[1:],
                strict=True,
            )
        )


class FunctionalGaussianProcessLikelihoodState(StrictModule):
    """Kernel, observation noise, jitter, and optional interdomain inducing design."""

    kernel: AbstractPositiveDefiniteKernel
    noise_scale: Array
    jitter: Array
    inducing_design: FunctionalDesign | None
    noise_layout: Literal["block", "observation"] = eqx.field(static=True)

    def __init__(
        self,
        *,
        kernel: AbstractPositiveDefiniteKernel,
        noise_scale: ArrayLike,
        noise_layout: Literal["block", "observation"] = "block",
        jitter: ArrayLike = 1e-8,
        inducing_design: FunctionalDesign | None = None,
    ):
        if not isinstance(kernel, AbstractPositiveDefiniteKernel):
            raise TypeError("kernel must be a positive-definite kernel.")
        noise = jnp.asarray(noise_scale, dtype=float)
        if noise.ndim > 1 or (noise.ndim == 1 and noise.shape[0] <= 0):
            raise ValueError("noise_scale must be scalar or a nonempty vector.")
        if noise_layout not in ("block", "observation"):
            raise ValueError("noise_layout must be 'block' or 'observation'.")
        jitter_array = jnp.asarray(jitter, dtype=float)
        if jitter_array.ndim != 0:
            raise ValueError("jitter must be scalar.")
        if inducing_design is not None and not isinstance(
            inducing_design, FunctionalDesign
        ):
            raise TypeError("inducing_design must be a FunctionalDesign or None.")
        self.kernel = kernel
        self.noise_scale = eqx.error_if(
            noise,
            jnp.any(~jnp.isfinite(noise)) | jnp.any(noise < 0.0),
            "noise_scale must be finite and nonnegative.",
        )
        self.jitter = eqx.error_if(
            jitter_array,
            ~jnp.isfinite(jitter_array) | (jitter_array <= 0.0),
            "jitter must be finite and strictly positive.",
        )
        self.inducing_design = inducing_design
        self.noise_layout = noise_layout

    @property
    def is_sparse(self) -> bool:
        return self.inducing_design is not None

    def observation_noise(self, design: FunctionalDesign, /) -> Array:
        if self.noise_scale.ndim == 0:
            return jnp.broadcast_to(self.noise_scale, (design.num_observations,))
        expected = (
            design.num_blocks if self.noise_layout == "block" else design.num_observations
        )
        if self.noise_scale.shape != (expected,):
            raise ValueError(
                f"{self.noise_layout.title()} noise must align with the functional design."
            )
        if self.noise_layout == "block":
            return self.noise_scale[design.block_index]
        return self.noise_scale


class FunctionalGaussianProcessCondition(StrictModule):
    """Conditioned latent discrepancy over heterogeneous functional queries."""

    design: FunctionalDesign
    mean: Array
    covariance: Array
    variance: Array

    def __init__(
        self,
        *,
        design: FunctionalDesign,
        mean: ArrayLike,
        covariance: ArrayLike,
        variance: ArrayLike,
    ):
        if not isinstance(design, FunctionalDesign):
            raise TypeError("design must be a FunctionalDesign.")
        mean_array = design.flatten(mean, name="conditioned functional mean")
        variance_array = design.flatten(
            variance,
            name="conditioned functional variance",
        )
        covariance_array = jnp.asarray(covariance, dtype=float)
        count = design.num_observations
        if covariance_array.shape != (count, count):
            raise ValueError("Conditioned functional covariance has invalid shape.")
        self.design = design
        self.mean = mean_array
        self.covariance = covariance_array
        self.variance = variance_array

    def split_mean(self) -> tuple[Array, ...]:
        return self.design.split(self.mean)

    def split_variance(self) -> tuple[Array, ...]:
        return self.design.split(self.variance)

    def sample(self, key: Array, /, *, num_samples: int) -> Array:
        count = int(num_samples)
        if count <= 0:
            raise ValueError("num_samples must be positive.")
        return _sample_gaussian_psd(
            self.mean,
            self.covariance,
            key,
            num_samples=count,
        )


class FunctionalGaussianProcessDiscrepancy(StrictModule):
    """Exact or FITC GP discrepancy conditioned by linear functionals."""

    design: FunctionalDesign
    observations: Array

    def __init__(
        self,
        design: FunctionalDesign | Sequence[FunctionalObservationBlock],
        observations: ArrayLike | tuple[ArrayLike, ...],
        /,
    ):
        resolved_design = (
            design if isinstance(design, FunctionalDesign) else FunctionalDesign(design)
        )
        values = resolved_design.flatten(
            observations,
            name="functional GP observations",
        )
        self.design = resolved_design
        self.observations = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Functional GP observations must be finite.",
        )

    def residual(
        self,
        physical_mean: ArrayLike | tuple[ArrayLike, ...],
        /,
    ) -> Array:
        mean = self.design.flatten(physical_mean, name="functional physical mean")
        return self.observations - mean

    def log_marginal_likelihood(
        self,
        physical_mean: ArrayLike | tuple[ArrayLike, ...],
        /,
        *,
        state: FunctionalGaussianProcessLikelihoodState,
    ) -> Array:
        """Marginalize a latent field under exact or interdomain FITC inference."""
        _validate_functional_state(state, self.design)
        noise = state.observation_noise(self.design)
        residual = self.residual(physical_mean)
        if state.inducing_design is None:
            covariance = functional_kernel_matrix(
                state.kernel,
                self.design,
                self.design,
            ) + jnp.diag(noise * noise + state.jitter)
            return exact_gp_log_probability(
                residual,
                jnp.linalg.cholesky(covariance),
            )
        features, diagonal, correction_cholesky, _ = _functional_fitc_factors(
            state,
            self.design,
            noise,
        )
        return sparse_gp_log_probability_from_factors(
            residual,
            features,
            diagonal,
            correction_cholesky,
        )

    def condition(
        self,
        physical_mean: ArrayLike | tuple[ArrayLike, ...],
        query_design: FunctionalDesign | Sequence[FunctionalObservationBlock],
        /,
        *,
        state: FunctionalGaussianProcessLikelihoodState,
    ) -> FunctionalGaussianProcessCondition:
        """Condition arbitrary value or differential-functional queries."""
        resolved_query = (
            query_design
            if isinstance(query_design, FunctionalDesign)
            else FunctionalDesign(query_design)
        )
        _validate_functional_state(state, self.design)
        _validate_design_pair(self.design, resolved_query)
        residual = self.residual(physical_mean)
        noise = state.observation_noise(self.design)
        if state.inducing_design is None:
            observation_covariance = functional_kernel_matrix(
                state.kernel,
                self.design,
                self.design,
            ) + jnp.diag(noise * noise + state.jitter)
            projection, covariance, variance = exact_gp_conditioner_from_covariances(
                jnp.linalg.cholesky(observation_covariance),
                functional_kernel_matrix(
                    state.kernel,
                    resolved_query,
                    self.design,
                ),
                functional_kernel_matrix(
                    state.kernel,
                    resolved_query,
                    resolved_query,
                ),
            )
        else:
            features, diagonal, correction_cholesky, inducing_cholesky = (
                _functional_fitc_factors(state, self.design, noise)
            )
            projection, covariance, variance = sparse_gp_conditioner_from_covariances(
                functional_kernel_matrix(
                    state.kernel,
                    resolved_query,
                    state.inducing_design,
                ),
                functional_kernel_diagonal(state.kernel, resolved_query),
                features=features,
                diagonal=diagonal,
                correction_cholesky=correction_cholesky,
                inducing_cholesky=inducing_cholesky,
            )
        return FunctionalGaussianProcessCondition(
            design=resolved_query,
            mean=projection @ residual,
            covariance=covariance,
            variance=variance,
        )


def functional_kernel_matrix(
    kernel: AbstractPositiveDefiniteKernel,
    left: FunctionalDesign,
    right: FunctionalDesign,
    /,
) -> Array:
    """Assemble covariance between two ordered functional designs."""
    if not isinstance(kernel, AbstractPositiveDefiniteKernel):
        raise TypeError("kernel must be a positive-definite kernel.")
    _validate_design_pair(left, right)
    rows = tuple(
        jnp.concatenate(
            tuple(
                _functional_block_matrix(kernel, left_block, right_block)
                for right_block in right.blocks
            ),
            axis=1,
        )
        for left_block in left.blocks
    )
    return jnp.concatenate(rows, axis=0)


def functional_kernel_diagonal(
    kernel: AbstractPositiveDefiniteKernel,
    design: FunctionalDesign,
    /,
) -> Array:
    """Assemble functional prior variances without a dense covariance matrix."""
    if not isinstance(kernel, AbstractPositiveDefiniteKernel):
        raise TypeError("kernel must be a positive-definite kernel.")
    if not isinstance(design, FunctionalDesign):
        raise TypeError("design must be a FunctionalDesign.")
    values = []
    for block in design.blocks:
        _validate_regularity(kernel, block.functional)
        coefficients = block.functional.coefficient_matrix(block.num_observations)
        values.append(
            jax.vmap(
                lambda point, coefficient: _functional_pairwise(
                    kernel,
                    point,
                    block.functional.derivative_orders,
                    coefficient,
                    point,
                    block.functional.derivative_orders,
                    coefficient,
                )
            )(block.points, coefficients)
        )
    return jnp.concatenate(tuple(values))


def _functional_block_matrix(
    kernel: AbstractPositiveDefiniteKernel,
    left: FunctionalObservationBlock,
    right: FunctionalObservationBlock,
) -> Array:
    _validate_regularity(kernel, left.functional)
    _validate_regularity(kernel, right.functional)
    left_coefficients = left.functional.coefficient_matrix(left.num_observations)
    right_coefficients = right.functional.coefficient_matrix(right.num_observations)
    return jax.vmap(
        lambda left_point, left_coefficient: jax.vmap(
            lambda right_point, right_coefficient: _functional_pairwise(
                kernel,
                left_point,
                left.functional.derivative_orders,
                left_coefficient,
                right_point,
                right.functional.derivative_orders,
                right_coefficient,
            )
        )(right.points, right_coefficients)
    )(left.points, left_coefficients)


def _functional_pairwise(
    kernel: AbstractPositiveDefiniteKernel,
    left_point: Array,
    left_orders: tuple[tuple[int, ...], ...],
    left_coefficients: Array,
    right_point: Array,
    right_orders: tuple[tuple[int, ...], ...],
    right_coefficients: Array,
) -> Array:
    rows = []
    for left_order in left_orders:
        row = []
        for right_order in right_orders:
            function: Callable[[Array, Array], Array] = kernel.pairwise
            for coordinate, count in enumerate(left_order):
                for _ in range(count):
                    function = _coordinate_derivative(function, 0, coordinate)
            for coordinate, count in enumerate(right_order):
                for _ in range(count):
                    function = _coordinate_derivative(function, 1, coordinate)
            row.append(function(left_point, right_point))
        rows.append(jnp.stack(tuple(row)))
    partials = jnp.stack(tuple(rows))
    return left_coefficients @ partials @ right_coefficients


def _coordinate_derivative(
    function: Callable[[Array, Array], Array],
    argument: int,
    coordinate: int,
) -> Callable[[Array, Array], Array]:
    def derivative(left: Array, right: Array) -> Array:
        return jax.grad(function, argnums=argument)(left, right)[coordinate]

    return derivative


def _functional_fitc_factors(
    state: FunctionalGaussianProcessLikelihoodState,
    design: FunctionalDesign,
    noise: Array,
) -> tuple[Array, Array, Array, Array]:
    inducing = state.inducing_design
    if inducing is None:
        raise ValueError("Functional FITC factors require an inducing design.")
    _validate_design_pair(design, inducing)
    return fitc_factors_from_covariances(
        functional_kernel_matrix(state.kernel, design, inducing),
        functional_kernel_diagonal(state.kernel, design),
        functional_kernel_matrix(state.kernel, inducing, inducing),
        noise_scale=noise,
        jitter=state.jitter,
    )


def _validate_functional_state(
    state: FunctionalGaussianProcessLikelihoodState,
    design: FunctionalDesign,
) -> None:
    if not isinstance(state, FunctionalGaussianProcessLikelihoodState):
        raise TypeError("state must be a FunctionalGaussianProcessLikelihoodState.")
    if state.inducing_design is not None:
        _validate_design_pair(design, state.inducing_design)


def _validate_design_pair(left: FunctionalDesign, right: FunctionalDesign) -> None:
    if not isinstance(left, FunctionalDesign) or not isinstance(right, FunctionalDesign):
        raise TypeError("Functional covariance requires FunctionalDesign values.")
    if left.coordinate_size != right.coordinate_size:
        raise ValueError("Functional designs need equal point coordinate size.")


def _validate_regularity(
    kernel: AbstractPositiveDefiniteKernel,
    functional: LinearDifferentialFunctional,
) -> None:
    supported = kernel.max_derivative_order
    required = functional.required_derivative_order
    if supported is not None and required > supported:
        raise ValueError(
            f"{kernel.kernel_id} certifies derivative order {supported}, "
            f"but {functional.functional_id} requires order {required}."
        )


def _align_coefficients(left: Array, right: Array) -> tuple[Array, Array]:
    if left.ndim == right.ndim:
        if left.ndim == 2 and left.shape[0] != right.shape[0]:
            raise ValueError("Point-varying functional coefficients must align.")
        return left, right
    if left.ndim == 1:
        return jnp.broadcast_to(left, (right.shape[0], left.shape[0])), right
    return left, jnp.broadcast_to(right, (left.shape[0], right.shape[0]))


def _coordinate_size(value: int, /) -> int:
    size = int(value)
    if size <= 0:
        raise ValueError("coordinate_size must be positive.")
    return size


def _as_points(value: ArrayLike, /) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("Functional points must have shape (point, coordinate).")
    return array


__all__ = [
    "FunctionalDesign",
    "FunctionalGaussianProcessCondition",
    "FunctionalGaussianProcessDiscrepancy",
    "FunctionalGaussianProcessLikelihoodState",
    "FunctionalObservationBlock",
    "LinearDifferentialFunctional",
    "directional_derivative_functional",
    "functional_kernel_diagonal",
    "functional_kernel_matrix",
    "laplacian_functional",
    "partial_derivative_functional",
    "value_functional",
]
