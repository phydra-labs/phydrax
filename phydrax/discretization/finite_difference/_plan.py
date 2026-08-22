#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from .._core import (
    DiscretizationCapability,
    DiscretizationKey,
    DiscretizationRole,
    PreparationReport,
)
from .._lifecycle import AbstractDiscretizationPlan, validate_prepared_metadata
from .._measure import DiscreteMeasure
from .._spaces import DiscreteFieldSpace
from .._support import DiscreteSupport
from .._tensor import AbstractStrongFormDiscretization
from .._tensor_support import GridLocation, PreparedTensorGrid
from ._boundary import (
    AxisBoundaryPair,
    BoundaryRealizationPlan,
    HaloPlan,
)
from ._operators import prepare_linear_stencil, PreparedStencilOperator
from ._request import DerivativeRequest
from ._stencil import BoundaryStencilSet, StencilFootprint


def _spatial_axes(
    axes: int | Sequence[int] | None,
    rank: int,
    /,
) -> tuple[int, ...]:
    selected = (
        tuple(range(rank))
        if axes is None
        else (
            (int(axes),) if isinstance(axes, int) else tuple(int(axis) for axis in axes)
        )
    )
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Spatial axes must be non-empty and distinct.")
    if any(axis < 0 or axis >= rank for axis in selected):
        raise ValueError(f"Spatial axes must lie in [0, {rank}).")
    return selected


class FiniteDifferencePlan(AbstractDiscretizationPlan):
    """Strong local-stencil calculus over one prepared tensor support."""

    grid: PreparedTensorGrid
    field_name: str = eqx.field(static=True)
    requests: tuple[DerivativeRequest, ...]
    key: DiscretizationKey
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        grid: PreparedTensorGrid,
        requests: Sequence[DerivativeRequest],
        /,
        *,
        field_name: str = "state",
        key: DiscretizationKey | None = None,
        plan_id: str | None = None,
    ):
        if not isinstance(grid, PreparedTensorGrid):
            raise TypeError("grid must be a PreparedTensorGrid.")
        requests_ = tuple(requests)
        if not requests_ or not all(
            isinstance(request, DerivativeRequest) for request in requests_
        ):
            raise TypeError("requests must contain one or more DerivativeRequest values.")
        names = tuple(request.name for request in requests_)
        if len(set(names)) != len(names):
            raise ValueError("Derivative request names must be unique.")
        field = str(field_name)
        if not field:
            raise ValueError("field_name must be non-empty.")
        key_ = (
            DiscretizationKey(
                "finite_difference",
                DiscretizationRole.PHYSICAL,
                domain_labels=grid.axis_names,
            )
            if key is None
            else key
        )
        if not isinstance(key_, DiscretizationKey):
            raise TypeError("key must be a DiscretizationKey.")
        capabilities = (
            DiscretizationCapability.STRONG_DERIVATIVE,
            DiscretizationCapability.MATRIX_FREE,
            DiscretizationCapability.RECONSTRUCTION,
        )
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "finite-difference-plan",
                    "grid": grid.prepared_id,
                    "field": field,
                    "requests": [request.request_id for request in requests_],
                    "key": key_.key_id,
                }
            )
            if plan_id is None
            else str(plan_id)
        )
        if not identifier:
            raise ValueError("plan_id must be non-empty.")
        self.grid = grid
        self.field_name = field
        self.requests = requests_
        self.key = key_
        self.capabilities = capabilities
        self.plan_id = identifier

    def prepare(
        self,
        /,
        *,
        numeric_version: str = "0",
    ) -> "PreparedFiniteDifferenceDiscretization":
        return PreparedFiniteDifferenceDiscretization(
            self,
            numeric_version=numeric_version,
        )


class PreparedFiniteDifferenceDiscretization(AbstractStrongFormDiscretization):
    """Prepared local finite-difference operators and exact location spaces."""

    grid: PreparedTensorGrid
    stencils: tuple[BoundaryStencilSet, ...]
    operators: tuple[PreparedStencilOperator, ...]
    operator_names: tuple[str, ...] = eqx.field(static=True)
    locations: tuple[GridLocation, ...]
    aggregate_footprint: StencilFootprint
    key: DiscretizationKey
    support: DiscreteSupport
    field_spaces: tuple[DiscreteFieldSpace, ...]
    measures: tuple[DiscreteMeasure, ...]
    halo_plan: HaloPlan
    capabilities: tuple[DiscretizationCapability, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    numeric_version: str = eqx.field(static=True)
    preparation: PreparationReport

    def __init__(
        self,
        plan: FiniteDifferencePlan,
        /,
        *,
        numeric_version: str = "0",
    ):
        if not isinstance(plan, FiniteDifferencePlan):
            raise TypeError("plan must be a FiniteDifferencePlan.")
        location_by_id: dict[str, GridLocation] = {}
        for request in plan.requests:
            location_by_id[request.source_location.location_id] = request.source_location
            location_by_id[request.target_location.location_id] = request.target_location
        locations = tuple(location_by_id.values())
        space_by_location = {
            location.location_id: plan.grid.field_space(
                plan.field_name
                if location.location_id == plan.grid.centered_location.location_id
                else f"{plan.field_name}@{location.location_id[:12]}",
                location=location,
            )
            for location in locations
        }
        stencils = tuple(
            prepare_linear_stencil(plan.grid, request) for request in plan.requests
        )
        operators = tuple(
            PreparedStencilOperator(
                stencil,
                space_by_location[stencil.stencil.source_location.location_id],
                space_by_location[stencil.stencil.target_location.location_id],
            )
            for stencil in stencils
        )
        footprint = stencils[0].stencil.footprint
        for stencil in stencils[1:]:
            footprint = footprint.union(stencil.stencil.footprint)
        preparation = PreparationReport(
            capabilities=plan.capabilities,
            resource_counts={
                "axes": len(plan.grid.axis_names),
                "points": plan.grid.size,
                "operators": len(operators),
                "maximum_stencil_width": max(
                    int(stencil.stencil.indices.shape[1]) for stencil in stencils
                ),
            },
        )
        boundary_plans = []
        request_by_axis = {request.axis: request for request in plan.requests}
        for axis_name in plan.grid.axis_names:
            request = request_by_axis.get(axis_name)
            if request is None:
                continue
            axis_index = plan.grid.axis_names.index(axis_name)
            if request.boundary == "periodic":
                boundary_plans.append(
                    BoundaryRealizationPlan(
                        AxisBoundaryPair(axis_name, "periodic", "periodic"),
                        "periodic",
                    )
                )
            else:
                boundary_plans.append(
                    BoundaryRealizationPlan(
                        AxisBoundaryPair(axis_name, "one_sided", "one_sided"),
                        "closure",
                        lower_width=footprint.lower[axis_index],
                        upper_width=footprint.upper[axis_index],
                    )
                )
        halo_plan = HaloPlan(
            footprint,
            physical_boundaries=boundary_plans,
        )
        spaces, measures, capabilities = validate_prepared_metadata(
            key=plan.key,
            support=plan.grid.support,
            field_spaces=tuple(space_by_location.values()),
            measures=(plan.grid.measure,),
            capabilities=plan.capabilities,
            preparation=preparation,
        )
        version = str(numeric_version)
        if not version:
            raise ValueError("numeric_version must be non-empty.")
        prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-finite-difference",
                "plan": plan.plan_id,
                "stencils": [stencil.stencil.stencil_id for stencil in stencils],
                "numeric_version": version,
            }
        )
        self.grid = plan.grid
        self.stencils = stencils
        self.operators = operators
        self.operator_names = tuple(request.name for request in plan.requests)
        self.locations = locations
        self.aggregate_footprint = footprint
        self.halo_plan = halo_plan
        self.key = plan.key
        self.support = plan.grid.support
        self.field_spaces = spaces
        self.measures = measures
        self.capabilities = capabilities
        self.plan_id = plan.plan_id
        self.prepared_id = prepared_id
        self.numeric_version = version
        self.preparation = preparation

    @property
    def state_shape(self) -> tuple[int, ...]:
        return self.grid.shape

    @property
    def points(self) -> Array:
        return self.grid.points

    @property
    def quadrature_weights(self) -> Array:
        return self.grid.quadrature_weights

    @property
    def discretization_id(self) -> str:
        return self.prepared_id

    @property
    def num_points(self) -> int:
        return self.grid.size

    def _validate_state(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        rank = len(self.state_shape)
        if value.shape[:rank] != self.state_shape:
            raise ValueError(
                f"State must begin with shape {self.state_shape}; got {value.shape}."
            )
        return value

    def partial_derivative(
        self,
        state: ArrayLike,
        /,
        *,
        axis: int,
        order: int = 1,
    ) -> Array:
        axis_ = int(axis)
        if axis_ < 0 or axis_ >= len(self.grid.axis_names):
            raise ValueError("axis is outside the finite-difference grid rank.")
        value = self._validate_state(state)
        operator = self.operator(f"d_{self.grid.axis_names[axis_]}_{int(order)}")
        if value.shape == self.state_shape:
            return operator.mv(value)
        trailing_shape = value.shape[len(self.state_shape) :]
        channels = value.reshape(self.state_shape + (-1,))
        differentiated = jax.vmap(operator.mv, in_axes=-1, out_axes=-1)(channels)
        return differentiated.reshape(self.state_shape + trailing_shape)

    def gradient(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        selected = _spatial_axes(axes, len(self.state_shape))
        value = self._validate_state(state)
        return jnp.stack(
            [self.partial_derivative(value, axis=axis, order=1) for axis in selected],
            axis=-1,
        )

    def divergence(
        self,
        vector: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
        dual: bool = False,
    ) -> Array:
        del dual
        selected = _spatial_axes(axes, len(self.state_shape))
        value = self._validate_state(vector)
        if value.ndim <= len(self.state_shape) or value.shape[-1] != len(selected):
            raise ValueError(
                "Divergence requires one trailing component per selected grid axis."
            )
        result = jnp.zeros_like(value[..., 0])
        for component, axis in enumerate(selected):
            result = result + self.partial_derivative(
                value[..., component], axis=axis, order=1
            )
        return result

    def laplacian(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        value = self._validate_state(state)
        selected = _spatial_axes(axes, len(self.state_shape))
        result = jnp.zeros_like(value)
        for axis in selected:
            result = result + self.partial_derivative(value, axis=axis, order=2)
        return result

    def integral(
        self,
        state: ArrayLike,
        /,
        *,
        axes: int | Sequence[int] | None = None,
    ) -> Array:
        result = self._validate_state(state)
        selected = _spatial_axes(axes, len(self.state_shape))
        entities = self.grid.primary_entity_layout.axis_entities
        for axis in sorted(selected, reverse=True):
            weights = self.grid.structured_axes[axis].measure(entities[axis])
            result = jnp.tensordot(weights, result, axes=((0,), (axis,)))
        return result

    def flatten(self, state: ArrayLike, /) -> Array:
        value = self._validate_state(state)
        return value.reshape((self.num_points,) + value.shape[len(self.state_shape) :])

    def unflatten(self, state: ArrayLike, /) -> Array:
        value = jnp.asarray(state)
        if value.ndim < 1 or value.shape[0] != self.num_points:
            raise ValueError(
                f"Flattened state must begin with ({self.num_points},); got {value.shape}."
            )
        return value.reshape(self.state_shape + value.shape[1:])

    def laplacian_matrix(self, /) -> Array:
        matrices = tuple(
            self.operator(f"d_{axis}_2")._materialize() for axis in self.grid.axis_names
        )
        result = matrices[0]
        for matrix in matrices[1:]:
            result = result + matrix
        return result

    def eigenpairs(self, *, rank: int | None = None) -> tuple[Array, Array]:
        if len(self.grid.axis_names) != 1:
            raise ValueError("Finite-difference eigenpairs support one periodic axis.")
        representation = self.transform_diagonalization(f"d_{self.grid.axis_names[0]}_2")
        spectrum = np.asarray(representation.modal_values).reshape((-1,))
        scale = max(1.0, float(np.max(np.abs(spectrum))))
        if np.max(np.abs(np.imag(spectrum))) > 1e-10 * scale:
            raise ValueError("Real FD eigenpairs require a symmetric periodic stencil.")
        point_count = spectrum.size
        point_index = np.arange(point_count, dtype=float)
        eigenvalues = [float(-np.real(spectrum[0]))]
        columns = [np.ones((point_count,), dtype=float)]
        for frequency in range(1, (point_count + 1) // 2):
            angle = 2.0 * np.pi * frequency * point_index / point_count
            eigenvalue = float(-np.real(spectrum[frequency]))
            eigenvalues.extend((eigenvalue, eigenvalue))
            columns.extend((np.cos(angle), np.sin(angle)))
        if point_count % 2 == 0:
            frequency = point_count // 2
            eigenvalues.append(float(-np.real(spectrum[frequency])))
            columns.append(np.cos(np.pi * point_index))
        values = np.asarray(eigenvalues)
        modes = np.stack(columns, axis=-1)
        order = np.argsort(values, kind="stable")
        retained = point_count if rank is None else int(rank)
        if retained <= 0 or retained > point_count:
            raise ValueError("rank must lie within the finite-difference state size.")
        selected = order[:retained]
        weights = np.asarray(self.grid.quadrature_weights).reshape((-1,))
        selected_modes = modes[:, selected]
        norms = np.sqrt(np.sum(weights[:, None] * selected_modes**2, axis=0))
        selected_modes = selected_modes / norms[None, :]
        return jnp.asarray(values[selected]), jnp.asarray(selected_modes)

    def transform_diagonalization(self, name: str, /):
        """Return a certified FFT representation of a periodic 1D stencil."""
        from ...linalg import FFTLinearTransform, TransformDiagonalRepresentation

        if len(self.grid.axis_names) != 1:
            raise ValueError("Transform diagonalization supports one periodic axis.")
        stencil_set = self.stencil(name)
        if stencil_set.kind != "periodic":
            raise ValueError("Transform diagonalization requires periodic closure.")
        operator = self.operator(name)
        if not operator.source.compatible(operator.target):
            raise ValueError("Transform diagonalization requires one endomorphism space.")
        count = self.grid.shape[0]
        weights = np.asarray(self.grid.quadrature_weights).reshape((-1,))
        if not np.allclose(weights, weights[0], rtol=1e-10, atol=1e-12):
            raise ValueError("Fourier diagonalization requires a uniform measure.")
        row_valid = np.asarray(stencil_set.stencil.valid[0], dtype=bool)
        indices = np.asarray(stencil_set.stencil.indices[0], dtype=np.int32)[row_valid]
        coefficients = np.asarray(stencil_set.stencil.weights[0])[row_valid]
        relative = indices % count
        modes = np.arange(count)
        modal_values = np.sum(
            coefficients[None, :]
            * np.exp(2j * np.pi * modes[:, None] * relative[None, :] / float(count)),
            axis=1,
        )
        transform = FFTLinearTransform(
            count,
            dtype=jnp.result_type(operator.source.dtype, jnp.complex64),
        )
        return TransformDiagonalRepresentation.from_transform(
            operator,
            modal_values,
            transform,
            representation_id=canonical_fingerprint(
                {
                    "kind": "finite-difference-fourier-representation",
                    "operator": operator.operator_id,
                    "stencil": stencil_set.stencil.stencil_id,
                    "transform": transform.transform_id,
                }
            ),
        )

    def operator(self, name: str, /) -> PreparedStencilOperator:
        value = str(name)
        for operator_name, operator in zip(
            self.operator_names,
            self.operators,
            strict=True,
        ):
            if operator_name == value:
                return operator
        raise KeyError(f"Unknown finite-difference operator {value!r}.")

    def stencil(self, name: str, /) -> BoundaryStencilSet:
        value = str(name)
        for operator_name, stencil in zip(
            self.operator_names,
            self.stencils,
            strict=True,
        ):
            if operator_name == value:
                return stencil
        raise KeyError(f"Unknown finite-difference stencil {value!r}.")


def periodic_finite_difference(
    grid: PreparedTensorGrid,
    /,
    *,
    accuracy_order: int = 2,
) -> PreparedFiniteDifferenceDiscretization:
    """Prepare first/second derivatives on every periodic tensor axis."""
    if not isinstance(grid, PreparedTensorGrid) or not all(
        axis.periodic for axis in grid.structured_axes
    ):
        raise ValueError("periodic_finite_difference requires an all-periodic grid.")
    requests = tuple(
        request
        for axis in grid.axis_names
        for request in (
            DerivativeRequest(
                f"d_{axis}_1",
                grid,
                axis,
                derivative_order=1,
                accuracy_order=accuracy_order,
                boundary="periodic",
            ),
            DerivativeRequest(
                f"d_{axis}_2",
                grid,
                axis,
                derivative_order=2,
                accuracy_order=accuracy_order,
                boundary="periodic",
            ),
        )
    )
    return FiniteDifferencePlan(
        grid,
        requests,
        field_name="periodic_fd_state",
    ).prepare()


__all__ = [
    "FiniteDifferencePlan",
    "PreparedFiniteDifferenceDiscretization",
    "periodic_finite_difference",
]
