#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..conditions._ir import ArrayCodomain, FieldCodomain, ProductFieldSpec
from ._base import AbstractPositiveDefiniteKernel
from ._finite_feature import kernel_feature_rank, kernel_features
from ._operator_valued import (
    AbstractOperatorValuedKernel,
    operator_kernel_feature_rank,
    operator_kernel_features,
)


KernelMetricMode = Literal["independent", "coupled"]
KernelFunctionalExactness = Literal[
    "analytic", "finite-feature", "fixed-realization", "selected-section"
]


class _IdentityInput(StrictModule):
    def __call__(self, value: Array, /) -> Array:
        return value


class KernelInputAdapter(StrictModule):
    """Explicit conversion from a physical field input to a kernel input."""

    function: Callable[[Array], Array]
    adapter_id: str = eqx.field(static=True)
    output_ndim: int = eqx.field(static=True)

    def __init__(
        self,
        function: Callable[[Array], Array] | None = None,
        /,
        *,
        adapter_id: str = "identity",
        output_ndim: int = 1,
    ):
        function_ = _IdentityInput() if function is None else function
        if not callable(function_):
            raise TypeError("Kernel input adapters must be callable.")
        identifier = str(adapter_id)
        ndim = int(output_ndim)
        if not identifier or ndim <= 0:
            raise ValueError("Kernel input adapters need an ID and positive output_ndim.")
        self.function = function_
        self.adapter_id = identifier
        self.output_ndim = ndim

    def __call__(self, value: Array, /) -> Array:
        result = jnp.asarray(self.function(value))
        if result.ndim != self.output_ndim:
            raise ValueError("Kernel input adapter output rank changed.")
        return result


class KernelFunctionalTerm(StrictModule):
    """One field contribution to finite rows of a bounded RKHS functional."""

    points: Array
    coefficients: Array
    field_name: str = eqx.field(static=True)
    derivative_orders: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        points: ArrayLike,
        derivative_orders: Sequence[Sequence[int]],
        coefficients: ArrayLike,
        /,
    ):
        field = str(field_name)
        point_array = jnp.asarray(points)
        coefficient_array = jnp.asarray(coefficients)
        orders = tuple(
            tuple(int(value) for value in order) for order in derivative_orders
        )
        if not field:
            raise ValueError("Kernel functional field names must be nonempty.")
        if (
            point_array.ndim < 2
            or int(point_array.shape[0]) <= 0
            or any(int(size) <= 0 for size in point_array.shape[1:])
        ):
            raise ValueError("Functional points need nonempty point and input axes.")
        if not jnp.issubdtype(point_array.dtype, jnp.inexact):
            point_array = point_array.astype(float)
        if not jnp.issubdtype(coefficient_array.dtype, jnp.inexact):
            coefficient_array = coefficient_array.astype(float)
        if not orders or any(any(value < 0 for value in order) for order in orders):
            raise ValueError("Derivative multi-indices must be nonnegative and nonempty.")
        if coefficient_array.ndim != 4:
            raise ValueError(
                "Functional coefficients must have shape (row, point, term, fiber)."
            )
        if coefficient_array.shape[1:3] != (point_array.shape[0], len(orders)):
            raise ValueError(
                "Functional coefficients do not align with points and terms."
            )
        if int(coefficient_array.shape[0]) <= 0 or int(coefficient_array.shape[3]) <= 0:
            raise ValueError("Functional row and fiber axes must be nonempty.")
        self.points = eqx.error_if(
            point_array,
            jnp.any(~jnp.isfinite(point_array)),
            "Kernel functional points must be finite.",
        )
        self.coefficients = eqx.error_if(
            coefficient_array,
            jnp.any(~jnp.isfinite(coefficient_array)),
            "Kernel functional coefficients must be finite.",
        )
        self.field_name = field
        self.derivative_orders = orders

    @property
    def row_count(self) -> int:
        return int(self.coefficients.shape[0])

    @property
    def fiber_dimension(self) -> int:
        return int(self.coefficients.shape[-1])


class KernelFunctional(StrictModule):
    """Ordered finite rows assembled from point, jet, or realized integral actions."""

    terms: tuple[KernelFunctionalTerm, ...]
    functional_id: str = eqx.field(static=True)
    exactness: KernelFunctionalExactness = eqx.field(static=True)
    realization_id: str | None = eqx.field(static=True)

    def __init__(
        self,
        terms: Sequence[KernelFunctionalTerm],
        /,
        *,
        functional_id: str,
        exactness: KernelFunctionalExactness = "analytic",
        realization_id: str | None = None,
    ):
        terms_ = tuple(terms)
        identifier = str(functional_id)
        if not terms_ or any(
            not isinstance(term, KernelFunctionalTerm) for term in terms_
        ):
            raise TypeError("KernelFunctional requires typed functional terms.")
        if len({term.row_count for term in terms_}) != 1:
            raise ValueError("Every functional term must have the same row count.")
        if not identifier:
            raise ValueError("functional_id must be nonempty.")
        if exactness not in (
            "analytic",
            "finite-feature",
            "fixed-realization",
            "selected-section",
        ):
            raise ValueError("Unknown kernel functional exactness.")
        realization = None if realization_id is None else str(realization_id)
        if realization_id is not None and not realization:
            raise ValueError("realization_id must be nonempty when supplied.")
        if exactness == "fixed-realization" and realization is None:
            raise ValueError("Fixed-realization functionals require realization_id.")
        self.terms = terms_
        self.functional_id = identifier
        self.exactness = exactness
        self.realization_id = realization

    @property
    def row_count(self) -> int:
        return self.terms[0].row_count


class KernelGramEvidence(StrictModule):
    """Structural and numerical facts for one functional covariance assembly."""

    hermitian_residual: Array
    minimum_diagonal: Array
    finite: Array
    positive_semidefinite: Array
    left_id: str = eqx.field(static=True)
    right_id: str = eqx.field(static=True)
    metric_id: str = eqx.field(static=True)
    representation: str = eqx.field(static=True)
    exactness: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        left_id: str,
        right_id: str,
        metric_id: str,
        representation: str,
        exactness: str,
        hermitian_residual: Any,
        minimum_diagonal: Any,
        finite: Any,
        positive_semidefinite: Any,
    ):
        self.left_id = str(left_id)
        self.right_id = str(right_id)
        self.metric_id = str(metric_id)
        self.representation = str(representation)
        self.exactness = str(exactness)
        self.hermitian_residual = jnp.asarray(hermitian_residual)
        self.minimum_diagonal = jnp.asarray(minimum_diagonal)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.positive_semidefinite = jnp.asarray(positive_semidefinite, dtype=bool)


class KernelGram(StrictModule):
    matrix: Array
    evidence: KernelGramEvidence

    def __init__(self, matrix: ArrayLike, evidence: KernelGramEvidence, /):
        if not isinstance(evidence, KernelGramEvidence):
            raise TypeError("evidence must be KernelGramEvidence.")
        matrix_ = jnp.asarray(matrix)
        if matrix_.ndim != 2:
            raise ValueError("Kernel Gram values must be matrices.")
        self.matrix = matrix_
        self.evidence = evidence


class ProductFieldKernelMetric(StrictModule):
    """Positive product-RKHS metric over ordered, possibly coupled fields."""

    field_spec: ProductFieldSpec
    kernels: tuple[AbstractPositiveDefiniteKernel | AbstractOperatorValuedKernel, ...]
    adapters: tuple[KernelInputAdapter, ...]
    channel_indices: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    field_shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    mode: KernelMetricMode = eqx.field(static=True)
    kernel_structure_id: str = eqx.field(static=True)
    geometry_revision_id: str = eqx.field(static=True)
    numeric_version: Array

    def __init__(
        self,
        field_spec: ProductFieldSpec,
        kernels: Sequence[AbstractPositiveDefiniteKernel | AbstractOperatorValuedKernel],
        adapters: Sequence[KernelInputAdapter],
        channel_indices: Sequence[Sequence[int]],
        /,
        *,
        mode: KernelMetricMode,
        geometry_revision_id: str = "fixed",
        numeric_version: Any = 0,
    ):
        if not isinstance(field_spec, ProductFieldSpec):
            raise TypeError("field_spec must be a ProductFieldSpec.")
        kernels_ = tuple(kernels)
        adapters_ = tuple(adapters)
        channels_ = tuple(tuple(int(index) for index in item) for item in channel_indices)
        if mode not in ("independent", "coupled"):
            raise ValueError("Kernel metric mode must be independent or coupled.")
        expected_kernels = len(field_spec.fields) if mode == "independent" else 1
        if len(kernels_) != expected_kernels:
            raise ValueError("Kernel count does not match the product metric mode.")
        if any(
            not isinstance(
                kernel, (AbstractPositiveDefiniteKernel, AbstractOperatorValuedKernel)
            )
            for kernel in kernels_
        ):
            raise TypeError("Product metrics require positive-definite kernels.")
        if len(adapters_) != len(field_spec.fields) or any(
            not isinstance(adapter, KernelInputAdapter) for adapter in adapters_
        ):
            raise TypeError("Product metrics require one input adapter per field.")
        shapes = tuple(_field_shape(field.codomain) for field in field_spec.fields)
        dimensions = tuple(max(1, prod(shape)) for shape in shapes)
        if len(channels_) != len(dimensions) or any(
            len(channel) != dimension
            for channel, dimension in zip(channels_, dimensions, strict=True)
        ):
            raise ValueError("Channel layout must match every field fiber dimension.")
        if mode == "independent":
            for kernel, dimension, channel in zip(
                kernels_, dimensions, channels_, strict=True
            ):
                if channel != tuple(range(dimension)):
                    raise ValueError(
                        "Independent field channels must be locally contiguous."
                    )
                if isinstance(kernel, AbstractOperatorValuedKernel) and (
                    kernel.output_dimension != dimension
                ):
                    raise ValueError(
                        "Operator-valued kernel output dimension is incompatible."
                    )
        else:
            coupled = kernels_[0]
            if not isinstance(coupled, AbstractOperatorValuedKernel):
                raise TypeError(
                    "Coupled product metrics require an operator-valued kernel."
                )
            flattened = tuple(index for channel in channels_ for index in channel)
            if sorted(flattened) != list(range(coupled.output_dimension)):
                raise ValueError(
                    "Coupled channel layout must partition the kernel fiber."
                )
        revision = str(geometry_revision_id)
        if not revision:
            raise ValueError("geometry_revision_id must be nonempty.")
        version = jnp.asarray(numeric_version, dtype=jnp.int32)
        if version.ndim != 0:
            raise ValueError("numeric_version must be scalar.")
        self.field_spec = field_spec
        self.kernels = kernels_
        self.adapters = adapters_
        self.channel_indices = channels_
        self.field_shapes = shapes
        self.mode = mode
        self.geometry_revision_id = revision
        self.numeric_version = eqx.error_if(
            version, version < 0, "numeric_version must be nonnegative."
        )
        self.kernel_structure_id = canonical_fingerprint(
            {
                "kind": "product-field-kernel-metric",
                "field_spec": field_spec.field_spec_id,
                "mode": mode,
                "kernels": tuple(kernel.kernel_id for kernel in kernels_),
                "adapters": tuple(adapter.adapter_id for adapter in adapters_),
                "channels": channels_,
                "geometry": revision,
            }
        )

    @classmethod
    def independent(
        cls,
        field_spec: ProductFieldSpec,
        kernels: Mapping[
            str, AbstractPositiveDefiniteKernel | AbstractOperatorValuedKernel
        ],
        /,
        *,
        input_adapters: Mapping[str, KernelInputAdapter] | None = None,
        geometry_revision_id: str = "fixed",
        numeric_version: Any = 0,
    ) -> ProductFieldKernelMetric:
        missing = tuple(name for name in field_spec.names if name not in kernels)
        extra = tuple(name for name in kernels if name not in field_spec.names)
        if missing or extra:
            raise ValueError(f"Kernel field mismatch; missing={missing}, extra={extra}.")
        adapters = _ordered_adapters(field_spec, input_adapters)
        shapes = tuple(_field_shape(field.codomain) for field in field_spec.fields)
        channels = tuple(tuple(range(max(1, prod(shape)))) for shape in shapes)
        return cls(
            field_spec,
            tuple(kernels[name] for name in field_spec.names),
            adapters,
            channels,
            mode="independent",
            geometry_revision_id=geometry_revision_id,
            numeric_version=numeric_version,
        )

    @classmethod
    def coupled(
        cls,
        field_spec: ProductFieldSpec,
        kernel: AbstractOperatorValuedKernel,
        /,
        *,
        channel_layout: Mapping[str, Sequence[int]] | None = None,
        input_adapter: KernelInputAdapter | None = None,
        geometry_revision_id: str = "fixed",
        numeric_version: Any = 0,
    ) -> ProductFieldKernelMetric:
        if not isinstance(kernel, AbstractOperatorValuedKernel):
            raise TypeError("kernel must be an AbstractOperatorValuedKernel.")
        adapter = KernelInputAdapter() if input_adapter is None else input_adapter
        if not isinstance(adapter, KernelInputAdapter):
            raise TypeError("input_adapter must be a KernelInputAdapter.")
        dimensions = tuple(
            max(1, prod(_field_shape(field.codomain))) for field in field_spec.fields
        )
        if channel_layout is None:
            offset = 0
            channels_list = []
            for dimension in dimensions:
                channels_list.append(tuple(range(offset, offset + dimension)))
                offset += dimension
            channels = tuple(channels_list)
        else:
            missing = tuple(
                name for name in field_spec.names if name not in channel_layout
            )
            extra = tuple(name for name in channel_layout if name not in field_spec.names)
            if missing or extra:
                raise ValueError(
                    f"Channel field mismatch; missing={missing}, extra={extra}."
                )
            channels = tuple(tuple(channel_layout[name]) for name in field_spec.names)
        return cls(
            field_spec,
            (kernel,),
            tuple(adapter for _ in field_spec.fields),
            channels,
            mode="coupled",
            geometry_revision_id=geometry_revision_id,
            numeric_version=numeric_version,
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return self.field_spec.names

    @property
    def metric_id(self) -> str:
        return self.kernel_structure_id

    def field_dimension(self, field_name: str, /) -> int:
        return max(1, prod(self.field_shapes[self._field_index(field_name)]))

    def point_functional(
        self,
        field_name: str,
        points: ArrayLike,
        covectors: ArrayLike | None = None,
        /,
        *,
        functional_id: str = "point",
    ) -> KernelFunctional:
        point_array = jnp.asarray(points)
        point_count = int(point_array.shape[0])
        dimension = self.field_dimension(field_name)
        if covectors is None:
            coefficients = jnp.eye(point_count * dimension).reshape(
                (point_count * dimension, point_count, 1, dimension)
            )
        else:
            covectors_ = jnp.asarray(covectors)
            if covectors_.ndim != 3 or covectors_.shape[1:] != (
                point_count,
                dimension,
            ):
                raise ValueError("Point covectors must have shape (row, point, fiber).")
            coefficients = covectors_[:, :, None, :]
        coordinate_count = prod(point_array.shape[1:])
        term = KernelFunctionalTerm(
            field_name,
            point_array,
            ((0,) * coordinate_count,),
            coefficients,
        )
        self._validate_term(term)
        return KernelFunctional((term,), functional_id=functional_id)

    def jet_functional(
        self,
        field_name: str,
        points: ArrayLike,
        derivative_orders: Sequence[Sequence[int]],
        coefficients: ArrayLike,
        /,
        *,
        functional_id: str = "jet",
    ) -> KernelFunctional:
        term = KernelFunctionalTerm(
            field_name,
            points,
            derivative_orders,
            coefficients,
        )
        self._validate_term(term)
        return KernelFunctional((term,), functional_id=functional_id)

    def integral_functional(
        self,
        field_name: str,
        points: ArrayLike,
        weights: ArrayLike,
        covectors: ArrayLike | None = None,
        /,
        *,
        functional_id: str = "integral",
        realization_id: str,
    ) -> KernelFunctional:
        point_array = jnp.asarray(points)
        weights_ = jnp.asarray(weights)
        if weights_.shape != (point_array.shape[0],):
            raise ValueError("Integral weights must align with the support points.")
        dimension = self.field_dimension(field_name)
        if covectors is None:
            identity = jnp.eye(dimension, dtype=weights_.dtype)
            coefficients = oe.contract("p,ia->ipa", weights_, identity)[:, :, None, :]
        else:
            covectors_ = jnp.asarray(covectors)
            if covectors_.ndim == 1:
                if covectors_.shape != (dimension,):
                    raise ValueError("Integral covector has the wrong fiber dimension.")
                covectors_ = covectors_[None, None, :]
            elif (
                covectors_.ndim != 3
                or covectors_.shape[-1] != dimension
                or covectors_.shape[1] not in (1, point_array.shape[0])
            ):
                raise ValueError(
                    "Integral covectors must have shape (row, point-or-one, fiber)."
                )
            if covectors_.shape[1] == 1:
                covectors_ = jnp.broadcast_to(
                    covectors_,
                    (covectors_.shape[0], point_array.shape[0], dimension),
                )
            coefficients = (covectors_ * weights_[None, :, None])[:, :, None, :]
        coordinate_count = prod(point_array.shape[1:])
        term = KernelFunctionalTerm(
            field_name,
            point_array,
            ((0,) * coordinate_count,),
            coefficients,
        )
        self._validate_term(term)
        return KernelFunctional(
            (term,),
            functional_id=functional_id,
            exactness="fixed-realization",
            realization_id=realization_id,
        )

    def block(
        self,
        left_field: str,
        left: ArrayLike,
        right_field: str,
        right: ArrayLike,
        /,
    ) -> Array:
        left_index = self._field_index(left_field)
        right_index = self._field_index(right_field)
        left_point = self.adapters[left_index](jnp.asarray(left))
        right_point = self.adapters[right_index](jnp.asarray(right))
        left_dimension = self.field_dimension(left_field)
        right_dimension = self.field_dimension(right_field)
        if self.mode == "independent":
            if left_index != right_index:
                dtype = jnp.result_type(left_point, right_point)
                return jnp.zeros((left_dimension, right_dimension), dtype=dtype)
            kernel = self.kernels[left_index]
            if isinstance(kernel, AbstractPositiveDefiniteKernel):
                return kernel.pairwise(left_point, right_point) * jnp.eye(
                    left_dimension,
                    dtype=jnp.result_type(left_point, right_point),
                )
            return kernel.block(left_point, right_point)
        kernel = self.kernels[0]
        if not isinstance(kernel, AbstractOperatorValuedKernel):
            raise RuntimeError("Coupled metric lost its operator-valued kernel.")
        full = kernel.block(left_point, right_point)
        left_channels = jnp.asarray(self.channel_indices[left_index], dtype=jnp.int32)
        right_channels = jnp.asarray(self.channel_indices[right_index], dtype=jnp.int32)
        return full[left_channels[:, None], right_channels[None, :]]

    def functional_gram(
        self,
        left: KernelFunctional,
        right: KernelFunctional | None = None,
        /,
    ) -> KernelGram:
        return kernel_functional_gram(self, left, right)

    def representer(
        self,
        functional: KernelFunctional,
        field_name: str,
        /,
    ) -> KernelSection:
        return KernelSection(self, functional, field_name)

    def feature_rank(self) -> int | None:
        if self.mode == "coupled":
            kernel = self.kernels[0]
            if not isinstance(kernel, AbstractOperatorValuedKernel):
                return None
            return operator_kernel_feature_rank(kernel)
        total = 0
        for name, kernel in zip(self.field_names, self.kernels, strict=True):
            dimension = self.field_dimension(name)
            if isinstance(kernel, AbstractPositiveDefiniteKernel):
                rank = kernel_feature_rank(kernel)
                if rank is None:
                    return None
                total += int(rank) * dimension
            else:
                rank = operator_kernel_feature_rank(kernel)
                if rank is None:
                    return None
                total += int(rank)
        return total

    def field_features(self, field_name: str, points: ArrayLike, /) -> Array:
        """Evaluate one field's exact features in the product metric coordinates."""
        rank = self.feature_rank()
        if rank is None:
            raise TypeError("Product metric has no exact finite-feature representation.")
        field_index = self._field_index(field_name)
        point_array = jnp.asarray(points)
        if point_array.ndim < 2 or int(point_array.shape[0]) <= 0:
            raise ValueError("Field feature queries need a nonempty leading point axis.")
        local = jax.vmap(lambda point: self._point_features(field_index, point))(
            point_array
        )
        if self.mode == "coupled":
            return local
        before = sum(
            self._field_feature_rank(name) for name in self.field_names[:field_index]
        )
        after = int(rank) - before - int(local.shape[-1])
        return jnp.pad(local, ((0, 0), (0, 0), (before, after)))

    def functional_features(self, functional: KernelFunctional, /) -> Array:
        rank = self.feature_rank()
        if rank is None:
            raise TypeError("Product metric has no exact finite-feature representation.")
        if self.mode == "coupled":
            term_blocks = tuple(self._term_features(term) for term in functional.terms)
            return sum(term_blocks[1:], term_blocks[0])
        blocks = []
        for field_name in self.field_names:
            field_blocks = [
                self._term_features(term)
                for term in functional.terms
                if term.field_name == field_name
            ]
            if field_blocks:
                blocks.append(sum(field_blocks[1:], field_blocks[0]))
            else:
                blocks.append(
                    jnp.zeros(
                        (functional.row_count, self._field_feature_rank(field_name)),
                        dtype=functional.terms[0].coefficients.dtype,
                    )
                )
        return jnp.concatenate(tuple(blocks), axis=-1)

    def _term_features(self, term: KernelFunctionalTerm, /) -> Array:
        self._validate_term(term)
        field_index = self._field_index(term.field_name)
        point_features = []
        for order in term.derivative_orders:
            derivative = _differentiate_feature(
                lambda point: self._point_features(field_index, point),
                order,
            )
            point_features.append(jax.vmap(derivative)(term.points))
        values = jnp.stack(tuple(point_features), axis=1)
        return oe.contract(
            "ipta,ptar->ir",
            jnp.conj(term.coefficients),
            values,
        )

    def _point_features(self, field_index: int, point: Array, /) -> Array:
        name = self.field_names[field_index]
        adapted = self.adapters[field_index](point)
        if self.mode == "coupled":
            kernel = self.kernels[0]
            if not isinstance(kernel, AbstractOperatorValuedKernel):
                raise RuntimeError("Coupled metric lost its kernel.")
            features = operator_kernel_features(kernel, adapted[None, ...])[0]
            channels = jnp.asarray(self.channel_indices[field_index], dtype=jnp.int32)
            return features[channels]
        kernel = self.kernels[field_index]
        if isinstance(kernel, AbstractPositiveDefiniteKernel):
            features = kernel_features(kernel, adapted[None, ...])[0]
            dimension = self.field_dimension(name)
            identity = jnp.eye(dimension, dtype=features.dtype)
            return oe.contract("ab,r->abr", identity, features).reshape(
                (dimension, dimension * features.shape[0])
            )
        return operator_kernel_features(kernel, adapted[None, ...])[0]

    def _field_feature_rank(self, field_name: str, /) -> int:
        field_index = self._field_index(field_name)
        if self.mode == "coupled":
            kernel = self.kernels[0]
            if not isinstance(kernel, AbstractOperatorValuedKernel):
                raise RuntimeError("Coupled metric lost its kernel.")
            rank = operator_kernel_feature_rank(kernel)
        else:
            kernel = self.kernels[field_index]
            if isinstance(kernel, AbstractPositiveDefiniteKernel):
                scalar_rank = kernel_feature_rank(kernel)
                rank = (
                    None
                    if scalar_rank is None
                    else scalar_rank * self.field_dimension(field_name)
                )
            else:
                rank = operator_kernel_feature_rank(kernel)
        if rank is None:
            raise TypeError("Field kernel has no finite-feature representation.")
        return int(rank)

    def _field_index(self, field_name: str, /) -> int:
        name = str(field_name)
        if name not in self.field_names:
            raise ValueError(f"Unknown metric field {name!r}.")
        return self.field_names.index(name)

    def _validate_term(self, term: KernelFunctionalTerm, /) -> None:
        field_index = self._field_index(term.field_name)
        if term.fiber_dimension != self.field_dimension(term.field_name):
            raise ValueError("Functional fiber dimension does not match its field.")
        coordinate_count = prod(term.points.shape[1:])
        if any(len(order) != coordinate_count for order in term.derivative_orders):
            raise ValueError("Derivative multi-index does not match the point shape.")
        adapter = self.adapters[field_index]
        kernel = (
            self.kernels[field_index] if self.mode == "independent" else self.kernels[0]
        )
        if adapter.output_ndim != kernel.input_ndim:
            raise ValueError(
                "Kernel input adapter rank does not match kernel.input_ndim."
            )
        supported = kernel.max_derivative_order
        if supported is not None and any(
            sum(order) > supported for order in term.derivative_orders
        ):
            raise ValueError(
                "Functional derivative order exceeds the kernel certificate."
            )


class KernelSection(StrictModule):
    """Vector-valued Riesz representers for all rows of one functional."""

    metric: ProductFieldKernelMetric
    functional: KernelFunctional
    field_name: str = eqx.field(static=True)

    def __init__(
        self,
        metric: ProductFieldKernelMetric,
        functional: KernelFunctional,
        field_name: str,
        /,
    ):
        if not isinstance(metric, ProductFieldKernelMetric):
            raise TypeError("metric must be a ProductFieldKernelMetric.")
        if not isinstance(functional, KernelFunctional):
            raise TypeError("functional must be a KernelFunctional.")
        metric._field_index(field_name)
        self.metric = metric
        self.functional = functional
        self.field_name = str(field_name)

    def __call__(self, points: ArrayLike, /) -> Array:
        return kernel_functional_representer(
            self.metric,
            self.functional,
            self.field_name,
            points,
        )


def kernel_functional_gram(
    metric: ProductFieldKernelMetric,
    left: KernelFunctional,
    right: KernelFunctional | None = None,
    /,
) -> KernelGram:
    """Assemble action-action covariance with Hermitian complex conventions."""
    if not isinstance(metric, ProductFieldKernelMetric):
        raise TypeError("metric must be a ProductFieldKernelMetric.")
    if not isinstance(left, KernelFunctional):
        raise TypeError("left must be a KernelFunctional.")
    right_ = left if right is None else right
    if not isinstance(right_, KernelFunctional):
        raise TypeError("right must be a KernelFunctional or None.")
    dtype = jnp.result_type(
        *(term.coefficients.dtype for term in (*left.terms, *right_.terms))
    )
    matrix = jnp.zeros((left.row_count, right_.row_count), dtype=dtype)
    for left_term in left.terms:
        metric._validate_term(left_term)
        for right_term in right_.terms:
            metric._validate_term(right_term)
            term_matrix = _term_gram(metric, left_term, right_term)
            matrix = matrix.astype(jnp.result_type(matrix, term_matrix)) + term_matrix
    self_gram = right is None or right_ is left
    if self_gram:
        skew = matrix - jnp.conj(matrix.T)
        hermitian_residual = jnp.max(jnp.abs(skew), initial=0.0)
        diagonal = jnp.real(jnp.diag(matrix))
        minimum_diagonal = jnp.min(diagonal, initial=jnp.inf)
        hermitian = 0.5 * (matrix + jnp.conj(matrix.T))
        eigenvalues = jnp.linalg.eigvalsh(hermitian)
        minimum_eigenvalue = jnp.min(eigenvalues, initial=jnp.inf)
        scale = jnp.maximum(jnp.max(jnp.abs(matrix), initial=0.0), 1.0)
        tolerance = 64.0 * jnp.finfo(diagonal.dtype).eps * scale
        psd = minimum_eigenvalue >= -tolerance
    else:
        hermitian_residual = jnp.asarray(jnp.nan)
        minimum_diagonal = jnp.asarray(jnp.nan)
        psd = jnp.asarray(False)
    evidence = KernelGramEvidence(
        left_id=left.functional_id,
        right_id=right_.functional_id,
        metric_id=metric.metric_id,
        representation="canonical",
        exactness=_joint_exactness(left, right_),
        hermitian_residual=hermitian_residual,
        minimum_diagonal=minimum_diagonal,
        finite=jnp.all(jnp.isfinite(matrix)),
        positive_semidefinite=psd,
    )
    return KernelGram(matrix, evidence)


def kernel_functional_diagonal(
    metric: ProductFieldKernelMetric,
    functional: KernelFunctional,
    /,
) -> Array:
    """Return the functional covariance diagonal in declared row order."""
    return jnp.real(jnp.diag(kernel_functional_gram(metric, functional).matrix))


def kernel_functional_representer(
    metric: ProductFieldKernelMetric,
    functional: KernelFunctional,
    field_name: str,
    points: ArrayLike,
    /,
) -> Array:
    """Evaluate all Riesz representers as ``(point, fiber, row)``."""
    query = jnp.asarray(points)
    if query.ndim < 2:
        raise ValueError("Representer queries need a leading point axis.")
    dimension = metric.field_dimension(field_name)
    dtype = jnp.result_type(query, *(term.coefficients for term in functional.terms))
    result = jnp.zeros((query.shape[0], dimension, functional.row_count), dtype=dtype)
    for term in functional.terms:
        metric._validate_term(term)
        term_values = []
        for order in term.derivative_orders:
            derivative = _differentiate_second_block(
                lambda left, right: metric.block(
                    field_name,
                    left,
                    term.field_name,
                    right,
                ),
                order,
            )
            term_values.append(
                jax.vmap(
                    lambda left: jax.vmap(lambda right: derivative(left, right))(
                        term.points
                    )
                )(query)
            )
        values = jnp.stack(tuple(term_values), axis=2)
        result = result + oe.contract(
            "qptba,ipta->qbi",
            values,
            term.coefficients,
        )
    return result


def _term_gram(
    metric: ProductFieldKernelMetric,
    left: KernelFunctionalTerm,
    right: KernelFunctionalTerm,
    /,
) -> Array:
    values_by_terms = []
    for left_order in left.derivative_orders:
        right_values = []
        for right_order in right.derivative_orders:
            derivative = _differentiate_block(
                lambda first, second: metric.block(
                    left.field_name,
                    first,
                    right.field_name,
                    second,
                ),
                left_order,
                right_order,
            )
            right_values.append(
                jax.vmap(
                    lambda first: jax.vmap(lambda second: derivative(first, second))(
                        right.points
                    )
                )(left.points)
            )
        values_by_terms.append(jnp.stack(tuple(right_values), axis=2))
    values = jnp.stack(tuple(values_by_terms), axis=1)
    return oe.contract(
        "ipta,ptqsab,jqsb->ij",
        jnp.conj(left.coefficients),
        values,
        right.coefficients,
    )


def _differentiate_block(
    function: Callable[[Array, Array], Array],
    left_order: Sequence[int],
    right_order: Sequence[int],
    /,
) -> Callable[[Array, Array], Array]:
    differentiated = function
    for coordinate, count in enumerate(left_order):
        for _ in range(int(count)):
            previous = differentiated
            differentiated = (
                lambda left, right, previous=previous, coordinate=coordinate: jax.jacfwd(
                    lambda flat: previous(flat.reshape(left.shape), right)
                )(left.reshape((-1,)))[..., coordinate]
            )
    for coordinate, count in enumerate(right_order):
        for _ in range(int(count)):
            previous = differentiated
            differentiated = (
                lambda left, right, previous=previous, coordinate=coordinate: jax.jacfwd(
                    lambda flat: previous(left, flat.reshape(right.shape))
                )(right.reshape((-1,)))[..., coordinate]
            )
    return differentiated


def _differentiate_second_block(
    function: Callable[[Array, Array], Array],
    order: Sequence[int],
    /,
) -> Callable[[Array, Array], Array]:
    return _differentiate_block(function, (0,) * len(order), order)


def _differentiate_feature(
    function: Callable[[Array], Array],
    order: Sequence[int],
    /,
) -> Callable[[Array], Array]:
    differentiated = function
    for coordinate, count in enumerate(order):
        for _ in range(int(count)):
            previous = differentiated
            differentiated = lambda point, previous=previous, coordinate=coordinate: (
                jax.jacfwd(lambda flat: previous(flat.reshape(point.shape)))(
                    point.reshape((-1,))
                )[..., coordinate]
            )
    return differentiated


def _field_shape(codomain: Any, /) -> tuple[int, ...]:
    if isinstance(codomain, ArrayCodomain):
        return codomain.shape
    if isinstance(codomain, FieldCodomain):
        return codomain.value.shape
    raise TypeError("Kernel field metrics require array or field-valued field codomains.")


def _ordered_adapters(
    field_spec: ProductFieldSpec,
    adapters: Mapping[str, KernelInputAdapter] | None,
    /,
) -> tuple[KernelInputAdapter, ...]:
    if adapters is None:
        return tuple(KernelInputAdapter() for _ in field_spec.fields)
    missing = tuple(name for name in field_spec.names if name not in adapters)
    extra = tuple(name for name in adapters if name not in field_spec.names)
    if missing or extra:
        raise ValueError(f"Adapter field mismatch; missing={missing}, extra={extra}.")
    return tuple(adapters[name] for name in field_spec.names)


def _joint_exactness(left: KernelFunctional, right: KernelFunctional, /) -> str:
    values = {left.exactness, right.exactness}
    if "selected-section" in values:
        return "selected-section"
    if "fixed-realization" in values:
        return "fixed-realization"
    if "finite-feature" in values:
        return "finite-feature"
    return "analytic"


__all__ = [
    "KernelFunctional",
    "KernelFunctionalExactness",
    "KernelFunctionalTerm",
    "KernelGram",
    "KernelGramEvidence",
    "KernelInputAdapter",
    "KernelMetricMode",
    "KernelSection",
    "ProductFieldKernelMetric",
    "kernel_functional_diagonal",
    "kernel_functional_gram",
    "kernel_functional_representer",
]
