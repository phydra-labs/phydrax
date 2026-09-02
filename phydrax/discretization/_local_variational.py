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
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from ._integration_domain import IntegrationDomain
from ._lifecycle import AbstractPreparedDiscretization
from ._spaces import DiscreteFieldSpace
from ._topology import EntitySelection


class LocalVariationalRequest(StrictModule, NonTrainableState):
    """Structural local action requirements presented to a prepared provider."""

    action_kind: str = eqx.field(static=True)
    region_kind: str = eqx.field(static=True)
    differential_operators: tuple[str, ...] = eqx.field(static=True)
    jet_kinds: tuple[str, ...] = eqx.field(static=True)
    requested_kernel_mode: str = eqx.field(static=True)
    requested_operator_realization: str = eqx.field(static=True)
    requested_reference_realization_id: str | None = eqx.field(static=True)
    action_semantics: tuple[str, ...] = eqx.field(static=True)
    constraint_mode: str = eqx.field(static=True)
    material_mode: str = eqx.field(static=True)
    history_mode: str = eqx.field(static=True)
    explicit_rules: bool = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        action_kind: str,
        region_kind: str,
        differential_operators: Sequence[str],
        jet_kinds: Sequence[str],
        /,
        *,
        requested_kernel_mode: str,
        requested_operator_realization: str,
        requested_reference_realization_id: str | None = None,
        action_semantics: Sequence[str] = (),
        constraint_mode: str = "external_map",
        material_mode: str = "none",
        history_mode: str = "none",
        explicit_rules: bool = False,
    ):
        action = str(action_kind)
        region = str(region_kind)
        operators = tuple(dict.fromkeys(str(value) for value in differential_operators))
        jets = tuple(dict.fromkeys(str(value) for value in jet_kinds))
        kernel = str(requested_kernel_mode)
        realization = str(requested_operator_realization)
        reference = (
            None
            if requested_reference_realization_id is None
            else str(requested_reference_realization_id)
        )
        semantics = tuple(dict.fromkeys(str(value) for value in action_semantics))
        constraint = str(constraint_mode)
        material = str(material_mode)
        history = str(history_mode)
        if (
            not action
            or not region
            or not operators
            or not jets
            or not kernel
            or not realization
            or reference == ""
            or any(not value for value in operators + jets + semantics)
            or not constraint
            or not material
            or not history
        ):
            raise ValueError("Local variational request capabilities must be complete.")
        self.action_kind = action
        self.region_kind = region
        self.differential_operators = operators
        self.jet_kinds = jets
        self.requested_kernel_mode = kernel
        self.requested_operator_realization = realization
        self.requested_reference_realization_id = reference
        self.action_semantics = semantics
        self.constraint_mode = constraint
        self.material_mode = material
        self.history_mode = history
        self.explicit_rules = bool(explicit_rules)
        self.request_id = canonical_fingerprint(
            {
                "kind": "local-variational-request",
                "action": action,
                "region": region,
                "operators": operators,
                "jets": jets,
                "kernel": kernel,
                "operator_realization": realization,
                "reference_realization": reference,
                "semantics": semantics,
                "constraint": constraint,
                "material": material,
                "history": history,
                "explicit_rules": bool(explicit_rules),
            }
        )


class LocalVariationalSelection(StrictModule, NonTrainableState):
    """One provider-selected execution and realization offer."""

    execution_kind: str = eqx.field(static=True)
    kernel_mode: str = eqx.field(static=True)
    operator_realization: str = eqx.field(static=True)
    reference_realization_id: str = eqx.field(static=True)
    offer_id: str = eqx.field(static=True)
    selection_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_kind: str,
        kernel_mode: str,
        operator_realization: str,
        reference_realization_id: str,
        offer_id: str,
        /,
    ):
        execution = str(execution_kind)
        kernel = str(kernel_mode)
        realization = str(operator_realization)
        reference = str(reference_realization_id)
        offer = str(offer_id)
        if execution not in ("prepared-local", "native") or any(
            not value for value in (kernel, realization, reference, offer)
        ):
            raise ValueError("Local variational selection is incomplete.")
        self.execution_kind = execution
        self.kernel_mode = kernel
        self.operator_realization = realization
        self.reference_realization_id = reference
        self.offer_id = offer
        self.selection_id = canonical_fingerprint(
            {
                "kind": "local-variational-selection",
                "execution": execution,
                "kernel": kernel,
                "operator_realization": realization,
                "reference_realization": reference,
                "offer": offer,
            }
        )


class LocalVariationalOffer(StrictModule, NonTrainableState):
    """A provider's explicit structural offer for local variational execution."""

    execution_kind: str = eqx.field(static=True)
    region_kinds: tuple[str, ...] = eqx.field(static=True)
    action_kinds: tuple[str, ...] = eqx.field(static=True)
    differential_operators: tuple[str, ...] = eqx.field(static=True)
    jet_kinds: tuple[str, ...] = eqx.field(static=True)
    kernel_modes: tuple[str, ...] = eqx.field(static=True)
    operator_realizations: tuple[str, ...] = eqx.field(static=True)
    reference_realization_ids: tuple[str, ...] = eqx.field(static=True)
    automatic_kernel_mode: str = eqx.field(static=True)
    automatic_operator_realization: str = eqx.field(static=True)
    automatic_reference_realization_id: str = eqx.field(static=True)
    action_semantics: tuple[str, ...] = eqx.field(static=True)
    constraint_modes: tuple[str, ...] = eqx.field(static=True)
    material_modes: tuple[str, ...] = eqx.field(static=True)
    history_modes: tuple[str, ...] = eqx.field(static=True)
    explicit_rules: bool = eqx.field(static=True)
    offer_id: str = eqx.field(static=True)

    def __init__(
        self,
        execution_kind: str,
        region_kinds: Sequence[str],
        action_kinds: Sequence[str],
        differential_operators: Sequence[str],
        jet_kinds: Sequence[str],
        kernel_modes: Sequence[str],
        operator_realizations: Sequence[str],
        reference_realization_ids: Sequence[str],
        /,
        *,
        automatic_kernel_mode: str,
        automatic_operator_realization: str,
        automatic_reference_realization_id: str,
        action_semantics: Sequence[str] = (),
        constraint_modes: Sequence[str] = ("external_map",),
        material_modes: Sequence[str] = ("none",),
        history_modes: Sequence[str] = ("none",),
        explicit_rules: bool = False,
    ):
        execution = str(execution_kind)

        def identities(values: Sequence[str], /) -> tuple[str, ...]:
            return tuple(dict.fromkeys(str(value) for value in values))

        regions = identities(region_kinds)
        actions = identities(action_kinds)
        operators = identities(differential_operators)
        jets = identities(jet_kinds)
        kernels = identities(kernel_modes)
        realizations = identities(operator_realizations)
        references = identities(reference_realization_ids)
        automatic_kernel = str(automatic_kernel_mode)
        automatic_realization = str(automatic_operator_realization)
        automatic_reference = str(automatic_reference_realization_id)
        semantics = identities(action_semantics)
        constraints = identities(constraint_modes)
        materials = identities(material_modes)
        histories = identities(history_modes)
        groups = (
            regions,
            actions,
            operators,
            jets,
            kernels,
            realizations,
            references,
            constraints,
            materials,
            histories,
        )
        if (
            execution not in ("prepared-local", "native")
            or any(not group or any(not value for value in group) for group in groups)
            or automatic_kernel not in kernels
            or automatic_realization not in realizations
            or automatic_reference not in references
            or any(not value for value in semantics)
        ):
            raise ValueError("Local variational offer capabilities are inconsistent.")
        self.execution_kind = execution
        self.region_kinds = regions
        self.action_kinds = actions
        self.differential_operators = operators
        self.jet_kinds = jets
        self.kernel_modes = kernels
        self.operator_realizations = realizations
        self.reference_realization_ids = references
        self.automatic_kernel_mode = automatic_kernel
        self.automatic_operator_realization = automatic_realization
        self.automatic_reference_realization_id = automatic_reference
        self.action_semantics = semantics
        self.constraint_modes = constraints
        self.material_modes = materials
        self.history_modes = histories
        self.explicit_rules = bool(explicit_rules)
        self.offer_id = canonical_fingerprint(
            {
                "kind": "local-variational-offer",
                "execution": execution,
                "regions": regions,
                "actions": actions,
                "operators": operators,
                "jets": jets,
                "kernels": kernels,
                "operator_realizations": realizations,
                "reference_realizations": references,
                "automatic_kernel": automatic_kernel,
                "automatic_operator_realization": automatic_realization,
                "automatic_reference_realization": automatic_reference,
                "semantics": semantics,
                "constraints": constraints,
                "materials": materials,
                "histories": histories,
                "explicit_rules": bool(explicit_rules),
            }
        )

    def select(
        self, request: LocalVariationalRequest, /
    ) -> LocalVariationalSelection | None:
        if not isinstance(request, LocalVariationalRequest):
            raise TypeError("request must be LocalVariationalRequest.")
        kernel = (
            self.automatic_kernel_mode
            if request.requested_kernel_mode == "auto"
            else request.requested_kernel_mode
        )
        realization = (
            self.automatic_operator_realization
            if request.requested_operator_realization == "auto"
            else request.requested_operator_realization
        )
        reference = (
            self.automatic_reference_realization_id
            if request.requested_reference_realization_id is None
            else request.requested_reference_realization_id
        )
        supported = (
            request.action_kind in self.action_kinds
            and request.region_kind in self.region_kinds
            and set(request.differential_operators).issubset(self.differential_operators)
            and set(request.jet_kinds).issubset(self.jet_kinds)
            and set(request.action_semantics).issubset(self.action_semantics)
            and request.constraint_mode in self.constraint_modes
            and request.material_mode in self.material_modes
            and request.history_mode in self.history_modes
            and (not request.explicit_rules or self.explicit_rules)
            and kernel in self.kernel_modes
            and realization in self.operator_realizations
            and reference in self.reference_realization_ids
        )
        if not supported:
            return None
        return LocalVariationalSelection(
            self.execution_kind,
            kernel,
            realization,
            reference,
            self.offer_id,
        )


class LocalVariationalCapabilities(StrictModule, NonTrainableState):
    """All structural local execution offers made by one prepared provider."""

    provider_id: str = eqx.field(static=True)
    offers: tuple[LocalVariationalOffer, ...]
    capabilities_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider_id: str,
        offers: Sequence[LocalVariationalOffer],
        /,
    ):
        provider = str(provider_id)
        offers_ = tuple(offers)
        if (
            not provider
            or not offers_
            or any(not isinstance(offer, LocalVariationalOffer) for offer in offers_)
            or len({offer.offer_id for offer in offers_}) != len(offers_)
        ):
            raise ValueError("Local variational capabilities require unique offers.")
        self.provider_id = provider
        self.offers = offers_
        self.capabilities_id = canonical_fingerprint(
            {
                "kind": "local-variational-capabilities",
                "provider": provider,
                "offers": [offer.offer_id for offer in offers_],
            }
        )

    def select(self, request: LocalVariationalRequest, /) -> LocalVariationalSelection:
        if not isinstance(request, LocalVariationalRequest):
            raise TypeError("request must be LocalVariationalRequest.")
        for offer in self.offers:
            selection = offer.select(request)
            if selection is not None:
                return selection
        raise ValueError(
            "Prepared provider does not offer the requested local variational "
            f"capabilities ({request.action_kind!r}, {request.region_kind!r}, "
            f"operators={request.differential_operators!r}, "
            f"jets={request.jet_kinds!r})."
        )


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
    realization_id: AbstractAttribute[str]
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

    @abc.abstractmethod
    def reference_hessian(
        self, runtime: object, local_coefficients: ArrayLike, /
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def reference_hessian_transpose(
        self, runtime: object, hessians: ArrayLike, /
    ) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def trace(self, runtime: object, local_coefficients: ArrayLike, /) -> Array:
        raise NotImplementedError

    @abc.abstractmethod
    def trace_transpose(self, runtime: object, values: ArrayLike, /) -> Array:
        raise NotImplementedError


class LocalMetricResult(StrictModule, NonTrainableState):
    """Runtime full-dimensional or manifold geometry for one local region."""

    points: Array
    physical_weights: Array
    jacobian: Array
    inverse_jacobian: Array
    metric_tensor: Array
    inverse_metric: Array
    inverse_hessian: Array
    normals: Array
    valid: Array
    metric_kind: str = eqx.field(static=True)
    reference_dimension: int = eqx.field(static=True)
    physical_dimension: int = eqx.field(static=True)

    def __init__(
        self,
        points: ArrayLike,
        physical_weights: ArrayLike,
        jacobian: ArrayLike,
        inverse_jacobian: ArrayLike,
        /,
        *,
        inverse_hessian: ArrayLike | None = None,
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
        physical_dimension = int(jacobian_.shape[2])
        reference_dimension = int(jacobian_.shape[3])
        if reference_dimension > physical_dimension:
            raise ValueError(
                "Local metrics require reference dimension no larger than physical "
                "dimension."
            )
        metric = ein.contract("cqdr,cqds->cqrs", jacobian_, jacobian_)
        inverse_metric = ein.contract("cqrd,cqds->cqrs", inverse, inverse)
        inverse_hessian_ = (
            jnp.empty((0,), dtype=points_.dtype)
            if inverse_hessian is None
            else jnp.asarray(inverse_hessian)
        )
        expected_inverse_hessian = points_.shape[:2] + (
            reference_dimension,
            physical_dimension,
            physical_dimension,
        )
        if inverse_hessian_.size and inverse_hessian_.shape != expected_inverse_hessian:
            raise ValueError(
                "Local inverse-map Hessians must have axes "
                "(entity, point, reference, physical, physical)."
            )
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
        self.metric_tensor = metric
        self.inverse_metric = inverse_metric
        self.inverse_hessian = inverse_hessian_
        self.normals = normals_
        self.valid = valid_
        self.metric_kind = (
            "full-dimensional"
            if reference_dimension == physical_dimension
            else "manifold"
        )
        self.reference_dimension = reference_dimension
        self.physical_dimension = physical_dimension

    def physical_gradient(self, reference_gradients: ArrayLike, /) -> Array:
        gradients = jnp.asarray(reference_gradients)
        if (
            gradients.shape[:2] != self.points.shape[:2]
            or gradients.shape[-1] != self.reference_dimension
        ):
            raise ValueError("Reference gradients do not match local metric points.")
        return ein.contract("cq...r,cqrd->cq...d", gradients, self.inverse_jacobian)

    def reference_gradient_transpose(self, physical_gradients: ArrayLike, /) -> Array:
        gradients = jnp.asarray(physical_gradients)
        if (
            gradients.shape[:2] != self.points.shape[:2]
            or gradients.shape[-1] != self.physical_dimension
        ):
            raise ValueError("Physical gradients do not match local metric points.")
        return ein.contract("cq...d,cqrd->cq...r", gradients, self.inverse_jacobian)

    def physical_hessian(
        self,
        reference_gradients: ArrayLike,
        reference_hessians: ArrayLike,
        /,
    ) -> Array:
        gradients = jnp.asarray(reference_gradients)
        hessians = jnp.asarray(reference_hessians)
        if (
            gradients.shape[:2] != self.points.shape[:2]
            or gradients.shape[-1] != self.reference_dimension
            or hessians.shape[:2] != self.points.shape[:2]
            or hessians.shape[-2:] != (self.reference_dimension, self.reference_dimension)
        ):
            raise ValueError("Reference Hessian jets do not match local metric points.")
        if not self.inverse_hessian.size:
            raise ValueError(
                "Physical Hessians require prepared inverse-map second derivatives."
            )
        transformed = ein.contract(
            "cq...rs,cqrd,cqse->cq...de",
            hessians,
            self.inverse_jacobian,
            self.inverse_jacobian,
        )
        correction = ein.contract(
            "cq...r,cqrde->cq...de",
            gradients,
            self.inverse_hessian,
        )
        return transformed + correction


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
    def local_variational_capabilities(self, /) -> LocalVariationalCapabilities:
        raise NotImplementedError

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
    "LocalVariationalCapabilities",
    "LocalVariationalOffer",
    "LocalVariationalRequest",
    "LocalVariationalSelection",
    "PreparedLocalRegion",
]
