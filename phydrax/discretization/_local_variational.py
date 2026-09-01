#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Sequence
from math import prod

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._integration_domain import IntegrationDomain
from ._lifecycle import AbstractPreparedDiscretization
from ._spaces import DiscreteFieldSpace
from ._topology import EntitySelection


class LocalFieldBinding(StrictModule, NonTrainableState):
    """Declared relation between a public field value and a local kernel value."""

    name: str = eqx.field(static=True)
    field_space: DiscreteFieldSpace
    representation: str = eqx.field(static=True)
    conformity: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    public_shape: tuple[int, ...] = eqx.field(static=True)
    execution_shape: tuple[int, ...] = eqx.field(static=True)
    local_width: int = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    binding_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        field_space: DiscreteFieldSpace,
        /,
        *,
        component_shape: Sequence[int],
        public_shape: Sequence[int],
        execution_shape: Sequence[int],
        local_width: int,
        layout_id: str,
    ):
        name_ = str(name)
        if not isinstance(field_space, DiscreteFieldSpace):
            raise TypeError("field_space must be a DiscreteFieldSpace.")
        components = tuple(int(value) for value in component_shape)
        public = tuple(int(value) for value in public_shape)
        execution = tuple(int(value) for value in execution_shape)
        width = int(local_width)
        layout = str(layout_id)
        if (
            not name_
            or name_ != field_space.name
            or any(value <= 0 for value in components + public + execution)
            or width <= 0
            or not layout
            or prod(public) != prod(execution)
        ):
            raise ValueError("Local field binding metadata is inconsistent.")
        self.name = name_
        self.field_space = field_space
        self.representation = field_space.representation
        self.conformity = field_space.conformity
        self.component_shape = components
        self.public_shape = public
        self.execution_shape = execution
        self.local_width = width
        self.layout_id = layout
        self.binding_id = canonical_fingerprint(
            {
                "kind": "local-field-binding",
                "name": name_,
                "field_space": field_space.field_space_id,
                "representation": field_space.representation,
                "conformity": field_space.conformity,
                "component_shape": components,
                "public_shape": public,
                "execution_shape": execution,
                "local_width": width,
                "layout": layout,
            }
        )

    def flatten(self, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        suffix = len(self.public_shape)
        if suffix and values_.shape[-suffix:] != self.public_shape:
            raise ValueError("Local public value does not match its declared shape.")
        prefix = values_.shape[:-suffix] if suffix else values_.shape
        return values_.reshape(prefix + self.execution_shape)

    def unflatten(self, values: ArrayLike, /) -> Array:
        values_ = jnp.asarray(values)
        suffix = len(self.execution_shape)
        if suffix and values_.shape[-suffix:] != self.execution_shape:
            raise ValueError("Local execution value does not match its declared shape.")
        prefix = values_.shape[:-suffix] if suffix else values_.shape
        return values_.reshape(prefix + self.public_shape)


class LocalReferenceActions(StrictModule, NonTrainableState):
    """Method-neutral reference interpolation and transpose actions."""

    __strict_abstract__ = True

    action_id: AbstractAttribute[str]
    local_width: AbstractAttribute[int]
    point_count: AbstractAttribute[int]
    maximum_derivative_order: AbstractAttribute[int]
    kernel_modes: AbstractAttribute[tuple[str, ...]]

    @abc.abstractmethod
    def realize_reference_actions(self, runtime: object, /) -> LocalReferenceActions:
        """Realize numeric reference data without changing its structural identity."""
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def interpolate_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def reference_gradient(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def reference_gradient_transpose(
        self, runtime: object, gradients: ArrayLike, /
    ) -> Array:
        raise NotImplementedError

    def reference_hessian(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        raise NotImplementedError("Reference Hessian actions were not prepared.")

    def reference_hessian_transpose(
        self, runtime: object, hessians: ArrayLike, /
    ) -> Array:
        raise NotImplementedError(
            "Reference Hessian transpose actions were not prepared."
        )

    @abc.abstractmethod
    def trace(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def trace_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        raise NotImplementedError


class LocalMetricResult(StrictModule, NonTrainableState):
    """Runtime geometry at the points used by one prepared local region."""

    points: Array
    physical_weights: Array
    jacobian: Array
    inverse_jacobian: Array
    normals: Array
    valid: Array

    def __init__(
        self,
        points: ArrayLike,
        physical_weights: ArrayLike,
        jacobian: ArrayLike,
        inverse_jacobian: ArrayLike,
        /,
        *,
        normals: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ):
        points_ = jnp.asarray(points)
        weights = jnp.asarray(physical_weights)
        jacobian_ = jnp.asarray(jacobian)
        inverse = jnp.asarray(inverse_jacobian)
        if (
            points_.ndim != 3
            or weights.shape != points_.shape[:2]
            or jacobian_.ndim != 4
            or jacobian_.shape[:2] != points_.shape[:2]
            or jacobian_.shape[2] != points_.shape[-1]
            or inverse.shape
            != jacobian_.shape[:2] + (jacobian_.shape[3], jacobian_.shape[2])
        ):
            raise ValueError("Local metric point, weight, and Jacobian shapes disagree.")
        normals_ = (
            jnp.empty((0,), dtype=points_.dtype)
            if normals is None
            else jnp.asarray(normals)
        )
        if normals_.size and normals_.shape != points_.shape:
            raise ValueError("Local metric normals must match physical point shape.")
        valid_ = (
            jnp.ones((points_.shape[0],), dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        if valid_.shape != (points_.shape[0],):
            raise ValueError("Local metric validity must have one value per entity.")
        self.points = points_
        self.physical_weights = weights
        self.jacobian = jacobian_
        self.inverse_jacobian = inverse
        self.normals = normals_
        self.valid = valid_

    def physical_gradient(self, reference_gradients: ArrayLike, /) -> Array:
        gradients = jnp.asarray(reference_gradients)
        if gradients.shape[:2] != self.points.shape[:2]:
            raise ValueError("Reference gradients do not match local metric points.")
        return oe.contract("cq...r,cqrd->cq...d", gradients, self.inverse_jacobian)

    def reference_gradient_transpose(self, physical_gradients: ArrayLike, /) -> Array:
        gradients = jnp.asarray(physical_gradients)
        if gradients.shape[:2] != self.points.shape[:2]:
            raise ValueError("Physical gradients do not match local metric points.")
        return oe.contract("cq...d,cqrd->cq...r", gradients, self.inverse_jacobian)


class LocalGeometryActions(StrictModule, NonTrainableState):
    """Prepared geometry realization for one fixed local region."""

    __strict_abstract__ = True

    action_id: AbstractAttribute[str]
    runtime_layout_id: AbstractAttribute[str]
    entity_count: AbstractAttribute[int]
    domain_kind: AbstractAttribute[str]

    @abc.abstractmethod
    def realize(self, runtime: object, /) -> LocalMetricResult:
        raise NotImplementedError


class PreparedLocalRegion(StrictModule, NonTrainableState):
    """Method-neutral local gathers, actions, and fixed entity routes."""

    domain: IntegrationDomain
    field_names: tuple[str, ...] = eqx.field(static=True)
    block_name: str = eqx.field(static=True)
    cell_kind: str = eqx.field(static=True)
    field_gathers: tuple[Array, ...]
    neighbour_gathers: tuple[Array, ...]
    reference_actions: tuple[LocalReferenceActions, ...]
    geometry_actions: LocalGeometryActions
    entity_indices: Array
    owner_cells: Array
    neighbour_cells: Array
    owner_local_entities: Array
    neighbour_local_entities: Array
    trace_permutations: Array
    valid: Array
    region_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: IntegrationDomain,
        field_names: Sequence[str],
        field_gathers: Sequence[ArrayLike],
        reference_actions: Sequence[LocalReferenceActions],
        geometry_actions: LocalGeometryActions,
        /,
        *,
        block_name: str = "local",
        cell_kind: str = "parametric",
        neighbour_gathers: Sequence[ArrayLike] = (),
        entity_indices: ArrayLike | None = None,
        owner_cells: ArrayLike | None = None,
        neighbour_cells: ArrayLike | None = None,
        owner_local_entities: ArrayLike | None = None,
        neighbour_local_entities: ArrayLike | None = None,
        trace_permutations: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ):
        if not isinstance(domain, IntegrationDomain):
            raise TypeError("domain must be an IntegrationDomain.")
        block = str(block_name)
        cell = str(cell_kind)
        if not block or not cell:
            raise ValueError("Prepared local block metadata must be non-empty.")
        names = tuple(str(name) for name in field_names)
        gathers = tuple(np.asarray(value, dtype=np.int32) for value in field_gathers)
        references = tuple(reference_actions)
        if (
            not names
            or any(not name for name in names)
            or len(set(names)) != len(names)
            or len(gathers) != len(names)
            or len(references) != len(names)
            or not all(isinstance(value, LocalReferenceActions) for value in references)
            or not isinstance(geometry_actions, LocalGeometryActions)
        ):
            raise ValueError("Prepared local region field actions are incomplete.")
        entities = np.asarray(
            domain.entity_indices if entity_indices is None else entity_indices,
            dtype=np.int32,
        )
        count = entities.size
        if entities.ndim != 1 or any(
            gather.shape != (count, reference.local_width)
            for gather, reference in zip(gathers, references, strict=True)
        ):
            raise ValueError("Prepared local gathers do not match reference widths.")
        if (
            geometry_actions.entity_count != count
            or geometry_actions.domain_kind != domain.kind
        ):
            raise ValueError("Prepared local geometry does not match its domain.")

        def route(value: ArrayLike | None, fallback: ArrayLike, /) -> np.ndarray:
            result = np.asarray(fallback if value is None else value, dtype=np.int32)
            if result.shape != (count,):
                raise ValueError("Prepared local routes must have one value per entity.")
            return result

        owners = route(owner_cells, domain.owner_cells)
        neighbours = route(neighbour_cells, domain.neighbour_cells)
        owner_local = route(owner_local_entities, domain.owner_local_entities)
        neighbour_local = route(neighbour_local_entities, domain.neighbour_local_entities)
        neighbour = tuple(
            np.asarray(value, dtype=np.int32) for value in neighbour_gathers
        )
        if neighbour and (
            len(neighbour) != len(names)
            or any(value.shape[0] != count for value in neighbour)
        ):
            raise ValueError("Prepared neighbour gathers do not match region fields.")
        traces = (
            np.empty((count, 0), dtype=np.int32)
            if trace_permutations is None
            else np.asarray(trace_permutations, dtype=np.int32)
        )
        if traces.ndim != 2 or traces.shape[0] != count:
            raise ValueError("Prepared trace permutations must be rank-2 by entity.")
        valid_ = (
            np.ones((count,), dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != (count,):
            raise ValueError("Prepared local validity must have one value per entity.")
        self.domain = domain
        self.block_name = block
        self.cell_kind = cell
        self.field_names = names
        self.field_gathers = tuple(jnp.asarray(value) for value in gathers)
        self.neighbour_gathers = tuple(jnp.asarray(value) for value in neighbour)
        self.reference_actions = references
        self.geometry_actions = geometry_actions
        self.entity_indices = jnp.asarray(entities)
        self.owner_cells = jnp.asarray(owners)
        self.neighbour_cells = jnp.asarray(neighbours)
        self.owner_local_entities = jnp.asarray(owner_local)
        self.neighbour_local_entities = jnp.asarray(neighbour_local)
        self.trace_permutations = jnp.asarray(traces)
        self.valid = jnp.asarray(valid_)
        self.region_id = canonical_fingerprint(
            {
                "kind": "prepared-local-region",
                "domain": domain.domain_id,
                "block": block,
                "cell_kind": cell,
                "fields": names,
                "gathers": [array_tree_fingerprint(value) for value in gathers],
                "neighbour_gathers": [
                    array_tree_fingerprint(value) for value in neighbour
                ],
                "references": [value.action_id for value in references],
                "geometry": geometry_actions.action_id,
                "entities": array_tree_fingerprint(entities),
                "owners": array_tree_fingerprint(owners),
                "neighbours": array_tree_fingerprint(neighbours),
                "owner_local": array_tree_fingerprint(owner_local),
                "neighbour_local": array_tree_fingerprint(neighbour_local),
                "trace_permutations": array_tree_fingerprint(traces),
                "valid": array_tree_fingerprint(valid_),
            }
        )


class AbstractPreparedLocalDiscretization(AbstractPreparedDiscretization):
    """Prepared discretization capable of method-neutral local variational work."""

    block_space: AbstractAttribute[object]
    precision_policy: AbstractAttribute[object]

    default_runtime: AbstractAttribute[object]

    @abc.abstractmethod
    def _field_index(self, name: str, /) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def integration_domain(
        self, kind: str, selection: EntitySelection | None = None, /
    ) -> IntegrationDomain:
        raise NotImplementedError

    @abc.abstractmethod
    def local_field_binding(self, name: str, /) -> LocalFieldBinding:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_local_regions(
        self,
        domain: IntegrationDomain,
        /,
        *,
        field_names: tuple[str, ...],
        maximum_derivative_order: int,
        kernel_mode: str,
    ) -> tuple[PreparedLocalRegion, ...]:
        raise NotImplementedError

    @abc.abstractmethod
    def validate_local_runtime(self, runtime: object, /) -> None:
        raise NotImplementedError


__all__ = [
    "AbstractPreparedLocalDiscretization",
    "LocalFieldBinding",
    "LocalGeometryActions",
    "LocalMetricResult",
    "LocalReferenceActions",
    "PreparedLocalRegion",
]
