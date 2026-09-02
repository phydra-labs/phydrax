#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from math import prod
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ...linalg import ArraySpace, DiagonalPairing
from .._axis import AxisDiscretization
from .._axis_domain import AxisDomain
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import (
    AbstractDiscretizationPlan,
    validate_prepared_metadata,
)
from .._measure import DiscreteMeasure
from .._periodic_cell import PeriodicCell
from .._spaces import DiscreteFieldSpace, TensorDofLayout
from .._tensor import (
    _axis_eigenvalues,
    _axis_modes,
    _smallest_tensor_indices,
    AbstractStrongFormDiscretization,
)
from .._tensor_support import PreparedTensorGrid
from ._basis import (
    AbstractSpectralBasisPlan,
    CosineBasisPlan,
    FourierBasisPlan,
    PreparedSpectralAxis,
    SineBasisPlan,
)
from ._precision import SpectralPrecisionPolicy


def _names(values: Sequence[str], count: int, /) -> tuple[str, ...]:
    names = tuple(str(value) for value in values)
    if (
        len(names) != count
        or any(not value for value in names)
        or len(set(names)) != count
    ):
        raise ValueError("axis_names must contain one unique non-empty name per basis.")
    return names


def _apply_axis_transform(
    value: Array,
    axis: int,
    function,
    /,
) -> Array:
    moved = jnp.moveaxis(value, axis, -1)
    leading_shape = moved.shape[:-1]
    flattened = moved.reshape((-1, moved.shape[-1]))
    transformed = jax.vmap(function)(flattened)
    restored = transformed.reshape(leading_shape + (transformed.shape[-1],))
    return jnp.moveaxis(restored, -1, axis)


class TensorSpectralPlan(AbstractDiscretizationPlan):
    """Global tensor-product spectral field-space plan."""

    bases: tuple[AbstractSpectralBasisPlan, ...]
    axis_names: tuple[str, ...] = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    precision: SpectralPrecisionPolicy
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        bases: Sequence[AbstractSpectralBasisPlan],
        /,
        *,
        axis_names: Sequence[str] | None = None,
        field_name: str = "state",
        precision: SpectralPrecisionPolicy | None = None,
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
    ):
        bases_ = tuple(bases)
        if not bases_ or not all(
            isinstance(value, AbstractSpectralBasisPlan) for value in bases_
        ):
            raise TypeError("bases must contain AbstractSpectralBasisPlan values.")
        names = _names(
            tuple(f"axis{index}" for index in range(len(bases_)))
            if axis_names is None
            else axis_names,
            len(bases_),
        )
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        precision_ = SpectralPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, SpectralPrecisionPolicy):
            raise TypeError("precision must be a SpectralPrecisionPolicy or None.")
        key_ = (
            DiscretizationKey(
                "tensor_spectral",
                DiscretizationRole.PHYSICAL,
                domain_labels=names,
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = (
            DiscretizationCapability.PROJECTION,
            DiscretizationCapability.RECONSTRUCTION,
            DiscretizationCapability.STRONG_DERIVATIVE,
            DiscretizationCapability.SPECTRAL_TRANSFORM,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.FIELD_TRANSFER,
            DiscretizationCapability.DIFFERENTIABLE_GEOMETRY,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "tensor-spectral-plan",
                    "bases": [basis.plan_id for basis in bases_],
                    "axis_names": list(names),
                    "field": field,
                    "precision": precision_.policy_id,
                    "key": key_.key_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.bases = bases_
        self.axis_names = names
        self.field_name = field
        self.precision = precision_
        self.key = key_
        self.capabilities = capabilities
        self.plan_id = identifier

    def prepare(
        self,
        domains: Sequence[AxisDomain],
        /,
        *,
        numeric_version: str = "0",
    ) -> "TensorSpectralDiscretization":
        domains_ = tuple(domains)
        if len(domains_) != len(self.bases) or not all(
            isinstance(domain, AxisDomain) for domain in domains_
        ):
            raise TypeError("domains must contain one AxisDomain per spectral basis.")
        axes = tuple(
            basis.prepare(domain, precision=self.precision)
            for basis, domain in zip(self.bases, domains_, strict=True)
        )
        return TensorSpectralDiscretization(
            self,
            axes,
            numeric_version=numeric_version,
        )


class TensorSpectralDiscretization(AbstractStrongFormDiscretization):
    """Prepared global tensor spectral space with modal primary state."""

    plan: TensorSpectralPlan
    axes: tuple[PreparedSpectralAxis, ...]
    grid: PreparedTensorGrid
    periodic_cell: PeriodicCell | None
    modal_space: DiscreteFieldSpace
    physical_space: DiscreteFieldSpace
    key: DiscretizationKey
    support: Any
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport
    modal_shape: tuple[int, ...] = eqx.field(static=True)
    physical_shape: tuple[int, ...] = eqx.field(static=True)
    boundary_conditions: tuple[str, ...] = eqx.field(static=True)
    _quadrature_weights: Array
    _points: Array

    def __init__(
        self,
        plan: TensorSpectralPlan,
        axes: Sequence[PreparedSpectralAxis],
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(plan, TensorSpectralPlan):
            raise TypeError("plan must be a TensorSpectralPlan.")
        axes_ = tuple(axes)
        if len(axes_) != len(plan.bases) or not all(
            isinstance(axis, PreparedSpectralAxis) for axis in axes_
        ):
            raise TypeError("axes must contain one PreparedSpectralAxis per basis.")
        if any(
            axis.plan.plan_id != basis.plan_id
            for axis, basis in zip(axes_, plan.bases, strict=True)
        ):
            raise ValueError("Prepared axes must originate from the tensor basis plans.")
        periodic_indices = tuple(
            axis_index for axis_index, axis in enumerate(axes_) if axis.periodic
        )
        if periodic_indices:
            ambient_dimension = len(axes_)
            vectors = np.zeros((len(periodic_indices), ambient_dimension), dtype=float)
            for row, axis_index in enumerate(periodic_indices):
                vectors[row, axis_index] = float(axes_[axis_index].length)
            origin = np.asarray(
                [0.0 if axis.bounds is None else float(axis.bounds[0]) for axis in axes_]
            )
            periodic_cell = PeriodicCell(
                vectors,
                origin=origin,
                periodic_axes=(True,) * len(periodic_indices),
            )
        else:
            periodic_cell = None
        grid = PreparedTensorGrid(
            tuple(axis.axis_discretization() for axis in axes_),
            axis_names=plan.axis_names,
        )
        modal_shape = tuple(axis.mode_count for axis in axes_)
        physical_shape = tuple(axis.physical_count for axis in axes_)
        modal_layout = TensorDofLayout(plan.axis_names, modal_shape)
        modal_vector_space = ArraySpace(
            modal_shape,
            dtype=jnp.dtype(plan.precision.coefficient_dtype),
        )
        modal_space = DiscreteFieldSpace(
            plan.field_name,
            grid.support.support_id,
            modal_layout,
            modal_vector_space,
            representation="modal_coefficient",
            conformity="unrestricted",
            projection_id=canonical_fingerprint(
                {
                    "kind": "tensor-spectral-projection",
                    "axes": [axis.axis_id for axis in axes_],
                }
            ),
            reconstruction_id=canonical_fingerprint(
                {
                    "kind": "tensor-spectral-reconstruction",
                    "axes": [axis.axis_id for axis in axes_],
                }
            ),
        )
        weights = grid.quadrature_weights.astype(plan.precision.physical_dtype)
        physical_layout = TensorDofLayout(
            plan.axis_names,
            physical_shape,
            location_id=grid.primary_entity_layout.location_id,
        )
        physical_space = DiscreteFieldSpace(
            f"{plan.field_name}_values",
            grid.support.support_id,
            physical_layout,
            ArraySpace(
                physical_shape,
                dtype=jnp.dtype(plan.precision.physical_dtype),
                pairing=DiagonalPairing(weights),
            ),
            representation="point_value",
            conformity="unrestricted",
            projection_id=modal_space.projection_id,
            reconstruction_id=modal_space.reconstruction_id,
        )
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            diagnostics=tuple(
                f"{name}:{axis.family}:{axis.mode_count}"
                for name, axis in zip(plan.axis_names, axes_, strict=True)
            ),
            resource_counts={
                "axes": len(axes_),
                "modal_points": int(prod(modal_shape)),
                "physical_points": int(prod(physical_shape)),
                "coefficient_bytes": int(prod(modal_shape))
                * np.dtype(plan.precision.coefficient_dtype).itemsize,
                "physical_bytes": int(prod(physical_shape))
                * np.dtype(plan.precision.physical_dtype).itemsize,
            },
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=grid.support,
            field_spaces=(modal_space, physical_space),
            measures=(grid.measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "tensor-spectral-discretization",
                "plan": plan.plan_id,
                "axes": [axis.axis_id for axis in axes_],
                "modal_space": modal_space.field_space_id,
                "physical_space": physical_space.field_space_id,
                "periodic_cell": (
                    None if periodic_cell is None else periodic_cell.cell_id
                ),
                "version": version,
            }
        )
        self.plan = plan
        self.axes = axes_
        self.periodic_cell = periodic_cell
        self.grid = grid
        self.modal_space = modal_space
        self.physical_space = physical_space
        self.key = plan.key
        self.support = grid.support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = prepared_id
        self.numeric_version = version
        self.preparation = preparation
        self.modal_shape = modal_shape
        self.physical_shape = physical_shape
        self.boundary_conditions = tuple(axis.boundary for axis in axes_)
        self._quadrature_weights = weights
        self._points = grid.points

    @classmethod
    def from_axes(
        cls,
        axes: Sequence[AxisDiscretization],
        /,
        *,
        axis_names: Sequence[str] | None = None,
        field_names: Sequence[str] = ("state",),
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
        numeric_version: str = "0",
        dtype: Any = float,
    ) -> "TensorSpectralDiscretization":
        """Prepare a modal tensor space from already materialized numerical axes."""
        axes_ = tuple(axes)
        if not axes_ or not all(isinstance(axis, AxisDiscretization) for axis in axes_):
            raise TypeError("axes must contain AxisDiscretization values.")
        fields = tuple(str(value) for value in field_names)
        if len(fields) != 1 or not fields[0]:
            raise ValueError(
                "Tensor spectral spaces declare one base field; PDE compilation "
                "derives additional field spaces."
            )
        plans = []
        domains = []
        for axis in axes_:
            count = int(axis.nodes.size)
            if axis.basis == "fourier":
                plans.append(FourierBasisPlan(count))
            elif axis.basis == "sine":
                plans.append(SineBasisPlan(count))
            elif axis.basis == "cosine":
                plans.append(CosineBasisPlan(count))
            elif axis.basis == "uniform":
                raise ValueError(
                    "Uniform axes require FiniteDifferencePlan; use FourierAxisSpec "
                    "for a global periodic spectral basis."
                )
            else:
                raise ValueError(
                    "Axis migration supports Fourier, sine, and cosine bases; "
                    "polynomial and rational families require an explicit spectral "
                    "basis plan so their node rule and mapping are preserved."
                )
            domains.append(axis.domain)
        precision = SpectralPrecisionPolicy(dtype)
        plan = TensorSpectralPlan(
            tuple(plans),
            axis_names=axis_names,
            field_name=fields[0],
            precision=precision,
            key=key,
            plan_id=plan_id,
        )
        return plan.prepare(
            tuple(domains),
            numeric_version=numeric_version,
        )

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.modal_shape

    @property
    def discretization_id(self) -> str:
        return self.prepared_id

    @property
    def num_points(self) -> int:
        return int(prod(self.physical_shape))

    @property
    def num_modes(self) -> int:
        return int(prod(self.modal_shape))

    @property
    def quadrature_weights(self) -> Array:
        return self._quadrature_weights

    @property
    def points(self) -> Array:
        return self._points

    @property
    def precision_evidence(self):
        return self.plan.precision.evidence()

    @property
    def resource_evidence_id(self) -> str:
        return self.preparation.report_id

    def _validate_leading(
        self,
        value: ArrayLike,
        shape: tuple[int, ...],
        name: str,
        /,
    ) -> Array:
        array = jnp.asarray(value)
        if array.ndim < len(shape) or tuple(array.shape[: len(shape)]) != shape:
            raise ValueError(f"{name} must begin with shape {shape}; got {array.shape}.")
        return array

    def project(self, values: ArrayLike, /) -> Array:
        result = self._validate_leading(values, self.physical_shape, "Physical values")
        for axis, prepared in enumerate(self.axes):
            result = _apply_axis_transform(result, axis, prepared.analyze)
        return result.astype(jnp.dtype(self.plan.precision.coefficient_dtype))

    def reconstruct(
        self,
        coefficients: ArrayLike,
        /,
        *,
        real_output: bool | None = None,
    ) -> Array:
        result = self._validate_leading(
            coefficients,
            self.modal_shape,
            "Modal coefficients",
        ).astype(jnp.dtype(self.plan.precision.coefficient_dtype))
        for axis in reversed(range(len(self.axes))):
            result = _apply_axis_transform(result, axis, self.axes[axis].synthesize)
        real = (
            not self.plan.precision.physical_dtype.startswith("complex")
            if real_output is None
            else bool(real_output)
        )
        if real:
            return self.plan.precision.output(jnp.real(result))
        return result

    def imaginary_leakage(self, coefficients: ArrayLike, /) -> Array:
        result = self.reconstruct(coefficients, real_output=False)
        return jnp.max(jnp.abs(jnp.imag(result)), initial=0.0)

    def modal_derivative(
        self,
        coefficients: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        result = self._validate_leading(
            coefficients,
            self.modal_shape,
            "Modal coefficients",
        )
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= len(self.axes):
            raise ValueError(f"axis must lie in [0, {len(self.axes)}).")
        prepared = self.axes[axis_]
        derivative_order = int(order)
        if derivative_order < 0:
            raise ValueError("Spectral derivative order must be non-negative.")
        if derivative_order == 0:
            return result
        if prepared.derivative_matrix is not None:
            output = result
            for _ in range(derivative_order):
                output = _apply_axis_transform(
                    output,
                    axis_,
                    lambda vector: prepared.derivative_matrix @ vector,
                )
            return output
        multiplier = prepared.derivative_multiplier(derivative_order)
        shape = [1] * result.ndim
        shape[axis_] = multiplier.size
        return result * multiplier.reshape(tuple(shape))

    def derivative_values(
        self,
        coefficients: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= len(self.axes):
            raise ValueError(f"axis must lie in [0, {len(self.axes)}).")
        derivative_order = int(order)
        if derivative_order < 0:
            raise ValueError("Spectral derivative order must be non-negative.")
        if derivative_order == 0:
            return self.reconstruct(coefficients)
        prepared = self.axes[axis_]
        if (
            prepared.derivative_matrix is not None
            or prepared.family == "fourier"
            or (prepared.family in ("sine", "cosine") and derivative_order % 2 == 0)
        ):
            return self.reconstruct(
                self.modal_derivative(
                    coefficients,
                    axis=axis_,
                    order=derivative_order,
                )
            )
        if prepared.family in (
            "rational_chebyshev_line",
            "rational_chebyshev_half_line",
        ):
            raise ValueError(
                "This constrained rational basis lacks a prepared derivative action."
            )
        if prepared.family not in ("sine", "cosine", "chebyshev", "legendre"):
            raise ValueError("This basis does not expose physical derivative values.")
        from ...operators.differential._array_ops import _basis_nth_derivative

        values = self.reconstruct(coefficients)
        basis = prepared.family if prepared.family in ("sine", "cosine") else "poly"
        return _basis_nth_derivative(
            values,
            prepared.nodes,
            axis=axis_,
            order=derivative_order,
            basis=basis,
        )

    def partial_derivative_values(
        self,
        values: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        return self.derivative_values(self.project(values), axis=axis, order=order)

    def partial_derivative(
        self,
        values: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        """Differentiate physical values and return physical values."""
        return self.partial_derivative_values(values, axis=axis, order=order)

    def _selected_axes(
        self,
        axes: int | Sequence[int] | None,
        /,
    ) -> tuple[int, ...]:
        selected = (
            tuple(range(len(self.axes)))
            if axes is None
            else (int(axes),)
            if isinstance(axes, int)
            else tuple(int(axis) for axis in axes)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(axis < 0 or axis >= len(self.axes) for axis in selected)
        ):
            raise ValueError("Spatial axes must be unique valid spectral axes.")
        return selected

    def gradient(
        self,
        values: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        value = self._validate_leading(values, self.physical_shape, "Physical values")
        selected = self._selected_axes(axes)
        return jnp.stack(
            tuple(self.partial_derivative(value, axis=axis) for axis in selected),
            axis=-1,
        )

    def divergence(
        self,
        values: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
        dual: bool = False,
    ) -> Array:
        value = self._validate_leading(values, self.physical_shape, "Physical vector")
        selected = self._selected_axes(axes)
        if value.ndim <= len(self.physical_shape) or value.shape[-1] != len(selected):
            raise ValueError(
                "Divergence requires a trailing component axis matching selected axes."
            )
        result = jnp.zeros_like(value[..., 0])
        for component, axis in enumerate(selected):
            basis = self.axes[axis].family
            if dual and basis in ("sine", "cosine"):
                from .._tensor import _dual_basis_first_derivative

                derivative = _dual_basis_first_derivative(
                    value[..., component],
                    self.axes[axis].nodes,
                    axis=axis,
                    basis=basis,
                )
            else:
                derivative = self.partial_derivative(
                    value[..., component],
                    axis=axis,
                )
            result = result + derivative
        return result

    def laplacian(
        self,
        values: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        value = self._validate_leading(values, self.physical_shape, "Physical values")
        selected = self._selected_axes(axes)
        diagonal_families = ("fourier", "sine", "cosine")
        if any(
            self.axes[axis].derivative_matrix is None
            and self.axes[axis].family not in diagonal_families
            for axis in selected
        ):
            output = jnp.zeros_like(value)
            for axis in selected:
                output = output + self.partial_derivative(value, axis=axis, order=2)
            return output
        return self.reconstruct(self.modal_laplacian(self.project(value), axes=selected))

    def integral(
        self,
        values: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        result = self._validate_leading(values, self.physical_shape, "Physical values")
        selected = self._selected_axes(axes)
        for axis in sorted(selected, reverse=True):
            result = jnp.tensordot(
                self.axes[axis].quadrature_weights,
                result,
                axes=((0,), (axis,)),
            )
        return result

    def laplacian_matrix(self) -> Array:
        identity = jnp.eye(
            self.num_points, dtype=jnp.dtype(self.plan.precision.physical_dtype)
        )
        columns = jax.vmap(
            lambda vector: self.laplacian(vector.reshape(self.physical_shape)).reshape(
                (-1,)
            )
        )(identity)
        return columns.T

    def modal_laplacian(
        self,
        coefficients: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        result = self._validate_leading(
            coefficients,
            self.modal_shape,
            "Modal coefficients",
        )
        selected = (
            tuple(range(len(self.axes)))
            if axes is None
            else (int(axes),)
            if isinstance(axes, int)
            else tuple(int(axis) for axis in axes)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(axis < 0 or axis >= len(self.axes) for axis in selected)
        ):
            raise ValueError("Laplacian axes must be unique valid spectral axes.")
        output = jnp.zeros_like(result)
        for axis in selected:
            output = output + self.modal_derivative(result, axis=axis, order=2)
        return output

    def laplacian_values(
        self,
        coefficients: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        return self.reconstruct(self.modal_laplacian(coefficients, axes=axes))

    def integral_modal(
        self,
        coefficients: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        result = self.reconstruct(coefficients)
        selected = (
            tuple(range(len(self.axes)))
            if axes is None
            else (int(axes),)
            if isinstance(axes, int)
            else tuple(int(axis) for axis in axes)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or any(axis < 0 or axis >= len(self.axes) for axis in selected)
        ):
            raise ValueError("Integral axes must be unique valid spectral axes.")
        for axis in sorted(selected, reverse=True):
            result = jnp.tensordot(
                self.axes[axis].quadrature_weights,
                result,
                axes=((0,), (axis,)),
            )
        return result

    def flatten(self, coefficients: ArrayLike, /) -> Array:
        value = self._validate_leading(
            coefficients,
            self.modal_shape,
            "Modal coefficients",
        )
        return value.reshape((self.num_modes,) + value.shape[len(self.modal_shape) :])

    def unflatten(self, coefficients: ArrayLike, /) -> Array:
        value = jnp.asarray(coefficients)
        if value.ndim < 1 or int(value.shape[0]) != self.num_modes:
            raise ValueError(
                f"Flattened modal state must begin with ({self.num_modes},); "
                f"got {value.shape}."
            )
        return value.reshape(self.modal_shape + value.shape[1:])

    def laplacian_eigenvalues(self) -> Array:
        values = jnp.zeros(
            self.modal_shape, dtype=jnp.dtype(self.plan.precision.physical_dtype)
        )
        for axis, prepared in enumerate(self.axes):
            eigenvalues = prepared.laplacian_eigenvalues()
            shape = [1] * len(self.modal_shape)
            shape[axis] = eigenvalues.size
            values = values + eigenvalues.reshape(tuple(shape))
        return values

    def eigenpairs(self, *, rank: int | None = None) -> tuple[Array, Array]:
        count = self.num_modes
        retained = count if rank is None else int(rank)
        if retained <= 0 or retained > count:
            raise ValueError(f"rank must lie in [1, {count}].")
        if any(axis.family not in ("fourier", "sine", "cosine") for axis in self.axes):
            raise ValueError(
                "Exact Laplacian eigenpairs require Fourier, sine, or cosine axes."
            )
        axis_discretizations = tuple(axis.axis_discretization() for axis in self.axes)
        axis_values = tuple(
            _axis_eigenvalues(axis, prepared.family)
            for axis, prepared in zip(axis_discretizations, self.axes, strict=True)
        )
        selected = _smallest_tensor_indices(axis_values, retained)
        values = jnp.asarray(
            [
                sum(axis_values[i][mode] for i, mode in enumerate(index))
                for index in selected
            ],
            dtype=jnp.dtype(self.plan.precision.physical_dtype),
        )
        modes = jnp.ones(
            self.physical_shape + (retained,),
            dtype=jnp.dtype(self.plan.precision.physical_dtype),
        )
        for axis_index, (axis, prepared) in enumerate(
            zip(axis_discretizations, self.axes, strict=True)
        ):
            requested = np.asarray([index[axis_index] for index in selected], dtype=int)
            axis_modes = jnp.asarray(
                _axis_modes(axis, prepared.family, requested),
                dtype=modes.dtype,
            )
            shape = [1] * (len(self.physical_shape) + 1)
            shape[axis_index] = self.physical_shape[axis_index]
            shape[-1] = retained
            modes = modes * axis_modes.reshape(tuple(shape))
        return values, modes


__all__ = [
    "TensorSpectralDiscretization",
    "TensorSpectralPlan",
]
