#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import IntegrationDomain
from ..linalg import OperatorProperties


IntegrationRule: TypeAlias = Any


def _rule_data(rule: IntegrationRule, /):
    from ..integration import reference_rule_data

    return reference_rule_data(rule)


def _rule_id(rule: IntegrationRule, /) -> str:
    data = _rule_data(rule)
    return canonical_fingerprint(
        {
            "kind": "integration-rule",
            "rule_type": type(rule).__name__,
            "reference_domain": data.cell,
            "points": array_tree_fingerprint(np.asarray(data.points)),
            "weights": array_tree_fingerprint(np.asarray(data.weights)),
        }
    )


def _normalize_rules(
    rules: Mapping[str, IntegrationRule] | Sequence[tuple[str, IntegrationRule]],
    /,
) -> tuple[tuple[str, IntegrationRule], ...]:
    items = tuple(rules.items()) if isinstance(rules, Mapping) else tuple(rules)
    normalized = tuple(sorted(((str(name), rule) for name, rule in items)))
    names = tuple(name for name, _ in normalized)
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("Integration-rule block names must be unique and non-empty.")
    for _, rule in normalized:
        _rule_data(rule)
    return normalized


class VariationalActionDescriptor(StrictModule, NonTrainableState):
    """Static lowering contract carried by every variational action."""

    action_kind: str = eqx.field(static=True)
    default_domain_kind: str = eqx.field(static=True)
    output_fields: tuple[str, ...] = eqx.field(static=True)
    input_fields: tuple[str, ...] = eqx.field(static=True)
    operators: tuple[tuple[str, str], ...] = eqx.field(static=True)
    coefficient_values: tuple["VariationalCoefficient", ...]
    provider_offers: tuple[str, ...] = eqx.field(static=True)
    evaluator: Callable | None

    def __init__(
        self,
        action_kind: str,
        default_domain_kind: str,
        output_fields: Sequence[str],
        input_fields: Sequence[str],
        operators: Sequence[tuple[str, str]],
        /,
        *,
        coefficient_values: Sequence["VariationalCoefficient"] = (),
        provider_offers: Sequence[str] = ("native",),
        evaluator: Callable | None = None,
    ):
        kind = str(action_kind)
        domain = str(default_domain_kind)
        outputs = tuple(str(value) for value in output_fields)
        inputs = tuple(str(value) for value in input_fields)
        operations = tuple((str(field), str(operator)) for field, operator in operators)
        coefficients = tuple(coefficient_values)
        offers = tuple(sorted(set(str(value) for value in provider_offers)))
        if (
            kind
            not in (
                "residual",
                "energy",
                "functional",
                "bilinear",
                "linear",
                "pairwise-volume-flux",
            )
            or domain not in ("cell", "exterior_facet", "interior_facet")
            or not outputs
            or not inputs
            or any(not value for value in (*outputs, *inputs))
            or not all(
                isinstance(value, VariationalCoefficient) for value in coefficients
            )
            or not offers
            or any(not value for value in offers)
            or (evaluator is not None and not callable(evaluator))
        ):
            raise ValueError("Variational action descriptor is invalid.")
        self.action_kind = kind
        self.default_domain_kind = domain
        self.output_fields = outputs
        self.input_fields = inputs
        self.operators = operations
        self.coefficient_values = coefficients
        self.provider_offers = offers
        self.evaluator = evaluator


class VariationalCoefficient(StrictModule, NonTrainableState):
    """Coefficient data bound to an explicit variational evaluation layout."""

    value: Array
    evaluator: Callable[[Array, object], ArrayLike] | None
    location: str = eqx.field(static=True)
    support_id: str | None = eqx.field(static=True)
    entity_set_id: str | None = eqx.field(static=True)
    field_space_id: str | None = eqx.field(static=True)
    rule_id: str | None = eqx.field(static=True)
    side: str = eqx.field(static=True)
    layout_axes: tuple[str, ...] = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    coefficient_id: str = eqx.field(static=True)

    def __init__(
        self,
        value: ArrayLike | Callable[[Array, object], ArrayLike],
        /,
        *,
        coefficient_id: str | None = None,
        location: str = "point",
        support_id: str | None = None,
        entity_set_id: str | None = None,
        field_space_id: str | None = None,
        rule_id: str | None = None,
        side: str = "none",
        layout_axes: Sequence[str] | None = None,
    ):
        location_ = str(location)
        if location_ not in ("point", "cell", "facet", "quadrature", "dof"):
            raise ValueError(
                "Coefficient location must be point, cell, facet, quadrature, or dof."
            )
        side_ = str(side)
        if side_ not in ("none", "plus", "minus"):
            raise ValueError("Coefficient side must be none, plus, or minus.")
        if side_ != "none" and location_ != "facet":
            raise ValueError("Plus/minus coefficient sides require facet location.")
        support = None if support_id is None else str(support_id)
        entity_set = None if entity_set_id is None else str(entity_set_id)
        field_space = None if field_space_id is None else str(field_space_id)
        rule = None if rule_id is None else str(rule_id)
        if any(item == "" for item in (support, entity_set, field_space, rule)):
            raise ValueError("Coefficient identities must be non-empty or None.")
        if location_ in ("cell", "facet", "quadrature") and (
            support is None or entity_set is None
        ):
            raise ValueError(
                "Entity and quadrature coefficients require support_id and entity_set_id."
            )
        if location_ == "quadrature" and rule is None:
            raise ValueError("Quadrature coefficients require rule_id.")
        if location_ == "dof" and (support is None or field_space is None):
            raise ValueError("DOF coefficients require support_id and field_space_id.")
        if callable(value) and location_ != "point":
            raise ValueError("Callable coefficients require point location.")
        default_axes = {
            "point": (),
            "cell": ("entity",),
            "facet": ("entity",),
            "quadrature": ("entity", "quadrature"),
            "dof": ("dof",),
        }[location_]
        axes = default_axes if layout_axes is None else tuple(str(v) for v in layout_axes)
        if axes != default_axes:
            raise ValueError(
                f"{location_} coefficients require canonical layout axes "
                f"{default_axes!r}; got {axes!r}."
            )
        if callable(value):
            if coefficient_id is None or not str(coefficient_id):
                raise ValueError(
                    "Callable coefficients require an explicit coefficient_id."
                )
            array = jnp.asarray(0.0)
            evaluator = cast(Callable[[Array, object], ArrayLike], value)
            identifier = str(coefficient_id)
            data_fingerprint = None
        else:
            array = jnp.asarray(value)
            if not jnp.issubdtype(array.dtype, jnp.inexact):
                array = array.astype(float)
            evaluator = None
            data_fingerprint = array_tree_fingerprint(np.asarray(array))
            identifier = (
                canonical_fingerprint(
                    {
                        "kind": "variational-coefficient",
                        "data": data_fingerprint,
                        "location": location_,
                        "support": support,
                        "entity_set": entity_set,
                        "field_space": field_space,
                        "rule": rule,
                        "side": side_,
                        "layout_axes": list(axes),
                    }
                )
                if coefficient_id is None
                else str(coefficient_id)
            )
            if not identifier:
                raise ValueError("coefficient_id must be non-empty.")
        self.value = array
        self.evaluator = evaluator
        self.location = location_
        self.support_id = support
        self.entity_set_id = entity_set
        self.field_space_id = field_space
        self.rule_id = rule
        self.side = side_
        self.layout_axes = axes
        self.layout_id = canonical_fingerprint(
            {
                "kind": "variational-coefficient-layout",
                "callable": evaluator is not None,
                "shape": None if evaluator is not None else list(array.shape),
                "dtype": None if evaluator is not None else str(array.dtype),
                "location": location_,
                "support": support,
                "entity_set": entity_set,
                "field_space": field_space,
                "rule": rule,
                "side": side_,
                "layout_axes": list(axes),
            }
        )
        self.coefficient_id = canonical_fingerprint(
            {
                "kind": "variational-coefficient-binding",
                "identity": identifier,
                "layout": self.layout_id,
                "data": data_fingerprint,
            }
        )

    @property
    def constant(self) -> bool:
        return self.evaluator is None and self.location == "point"

    def evaluate(
        self,
        points: Array,
        args: object = None,
        /,
        *,
        entity_indices: ArrayLike | None = None,
        dof_indices: ArrayLike | None = None,
        dof_orientations: ArrayLike | None = None,
        basis_values: ArrayLike | None = None,
        support_id: str | None = None,
        entity_set_id: str | None = None,
        field_space_id: str | None = None,
        rule_id: str | None = None,
        side: str | None = None,
    ) -> Array:
        for name, declared, observed in (
            ("support", self.support_id, support_id),
            ("entity set", self.entity_set_id, entity_set_id),
            ("field space", self.field_space_id, field_space_id),
            ("quadrature rule", self.rule_id, rule_id),
        ):
            if declared is not None and declared != (
                None if observed is None else str(observed)
            ):
                raise ValueError(
                    f"Coefficient {name} identity does not match evaluation."
                )
        if side is not None and self.side != str(side):
            raise ValueError("Coefficient facet side does not match evaluation.")
        if self.evaluator is not None:
            return jnp.asarray(self.evaluator(points, args))
        if self.location == "point":
            return jnp.broadcast_to(self.value, points.shape[:-1] + self.value.shape)
        if self.location == "dof":
            if dof_indices is None or basis_values is None:
                raise ValueError("DOF coefficient evaluation requires routes and basis.")
            routes = jnp.asarray(dof_indices, dtype=jnp.int32)
            basis = jnp.asarray(basis_values)
            local = self.value[routes]
            if dof_orientations is not None:
                orientations = jnp.asarray(dof_orientations)
                if orientations.shape != routes.shape:
                    raise ValueError(
                        "Coefficient orientations must match gathered DOF routes."
                    )
                local = local * orientations.reshape(
                    orientations.shape + (1,) * (local.ndim - orientations.ndim)
                )
            if basis.ndim == 2:
                return ein.contract("qi,ci...->cq...", basis, local)
            if basis.ndim == 3:
                return ein.contract("cqi,ci...->cq...", basis, local)
            raise ValueError("DOF coefficient basis must have rank two or three.")
        if entity_indices is None:
            raise ValueError("Entity/quadrature coefficients require entity indices.")
        indices = jnp.asarray(entity_indices, dtype=jnp.int32)
        selected = self.value[indices]
        if self.location in ("cell", "facet"):
            shape = (selected.shape[0],) + (1,) * (points.ndim - 2) + selected.shape[1:]
            return jnp.broadcast_to(
                selected.reshape(shape),
                points.shape[:-1] + selected.shape[1:],
            )
        if selected.shape[: points.ndim - 1] != points.shape[:-1]:
            raise ValueError(
                "Quadrature coefficient leading shape must match selected points."
            )
        return selected


def coefficient(
    value: ArrayLike | Callable[[Array, object], ArrayLike],
    /,
    *,
    coefficient_id: str | None = None,
    location: str = "point",
    support_id: str | None = None,
    entity_set_id: str | None = None,
    field_space_id: str | None = None,
    rule_id: str | None = None,
    side: str = "none",
    layout_axes: Sequence[str] | None = None,
) -> VariationalCoefficient:
    return VariationalCoefficient(
        value,
        coefficient_id=coefficient_id,
        location=location,
        support_id=support_id,
        entity_set_id=entity_set_id,
        field_space_id=field_space_id,
        rule_id=rule_id,
        side=side,
        layout_axes=layout_axes,
    )


class DiffusionAction(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    diffusivity: VariationalCoefficient
    action_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, IntegrationRule], ...]

    def __init__(
        self,
        field_name: str,
        diffusivity=1.0,
        /,
        *,
        action_id="diffusion",
        domain=None,
        rules=(),
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not identifier:
            raise ValueError("Diffusion field and action IDs must be non-empty.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("DiffusionAction requires a cell integration domain.")
        self.field_name = field
        self.diffusivity = (
            diffusivity
            if isinstance(diffusivity, VariationalCoefficient)
            else coefficient(diffusivity)
        )
        self.action_id = identifier
        self.domain = domain
        self.rules = _normalize_rules(rules)

    @property
    def descriptor(self) -> VariationalActionDescriptor:
        return VariationalActionDescriptor(
            "residual",
            "cell",
            (self.field_name,),
            (self.field_name,),
            ((self.field_name, "grad"),),
            coefficient_values=(self.diffusivity,),
            provider_offers=("prepared-local",),
        )


class TensorDiffusionAction(StrictModule, NonTrainableState):
    """Physical tensor diffusion ``q = D grad(u)`` for one scalar field."""

    field_name: str = eqx.field(static=True)
    diffusivity: VariationalCoefficient
    tensor_axes: tuple[str, str] = eqx.field(static=True)
    properties: OperatorProperties
    action_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, IntegrationRule], ...]

    def __init__(
        self,
        field_name: str,
        diffusivity=1.0,
        /,
        *,
        tensor_axes: Sequence[str] = ("flux", "gradient"),
        properties: OperatorProperties | None = None,
        action_id="tensor-diffusion",
        domain=None,
        rules=(),
    ):
        field = str(field_name)
        identifier = str(action_id)
        axes = tuple(str(value) for value in tensor_axes)
        properties_ = OperatorProperties() if properties is None else properties
        if not field or not identifier:
            raise ValueError("Tensor-diffusion field and action IDs must be non-empty.")
        if axes not in (("flux", "gradient"), ("gradient", "flux")):
            raise ValueError(
                "Tensor diffusion axes must explicitly order 'flux' and 'gradient'."
            )
        if not isinstance(properties_, OperatorProperties):
            raise TypeError("properties must be OperatorProperties or None.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("TensorDiffusionAction requires a cell integration domain.")
        self.field_name = field
        self.diffusivity = (
            diffusivity
            if isinstance(diffusivity, VariationalCoefficient)
            else coefficient(diffusivity)
        )
        self.tensor_axes = cast(tuple[str, str], axes)
        self.properties = properties_
        self.action_id = identifier
        self.domain = domain
        self.rules = _normalize_rules(rules)

    def physical_tensor(
        self,
        values: ArrayLike,
        dimension: int,
        /,
        *,
        leading_shape: Sequence[int] | None = None,
    ) -> Array:
        """Canonicalize coefficient values to ``(..., flux, gradient)`` axes."""
        array = jnp.asarray(values)
        dimension_ = int(dimension)
        if dimension_ < 1:
            raise ValueError("Physical tensor dimension must be positive.")
        if leading_shape is None:
            leading = (
                array.shape[:-2]
                if array.shape[-2:] == (dimension_, dimension_)
                else array.shape
            )
        else:
            leading = tuple(int(value) for value in leading_shape)
        if array.shape == leading:
            return array[..., None, None] * jnp.eye(dimension_, dtype=array.dtype)
        if array.shape != leading + (dimension_, dimension_):
            raise ValueError(
                "Tensor diffusivity must have either scalar coefficient shape or "
                "two trailing physical tensor axes."
            )
        return (
            array
            if self.tensor_axes == ("flux", "gradient")
            else jnp.swapaxes(array, -2, -1)
        )

    @property
    def descriptor(self) -> VariationalActionDescriptor:
        return VariationalActionDescriptor(
            "residual",
            "cell",
            (self.field_name,),
            (self.field_name,),
            ((self.field_name, "grad"),),
            coefficient_values=(self.diffusivity,),
            provider_offers=("prepared-local",),
        )


class MassAction(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    coefficient: VariationalCoefficient
    action_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, IntegrationRule], ...]

    def __init__(
        self, field_name: str, value=1.0, /, *, action_id="mass", domain=None, rules=()
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not identifier:
            raise ValueError("Mass field and action IDs must be non-empty.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("MassAction requires a cell integration domain.")
        self.field_name = field
        self.coefficient = (
            value if isinstance(value, VariationalCoefficient) else coefficient(value)
        )
        self.action_id = identifier
        self.domain = domain
        self.rules = _normalize_rules(rules)

    @property
    def descriptor(self) -> VariationalActionDescriptor:
        return VariationalActionDescriptor(
            "residual",
            "cell",
            (self.field_name,),
            (self.field_name,),
            ((self.field_name, "value"),),
            coefficient_values=(self.coefficient,),
            provider_offers=("prepared-local",),
        )


class SourceAction(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    source: VariationalCoefficient
    action_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, IntegrationRule], ...]

    def __init__(
        self, field_name: str, source, /, *, action_id="source", domain=None, rules=()
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not identifier:
            raise ValueError("Source field and action IDs must be non-empty.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "cell"
        ):
            raise ValueError("SourceAction requires a cell integration domain.")
        self.field_name = field
        self.source = (
            source if isinstance(source, VariationalCoefficient) else coefficient(source)
        )
        self.action_id = identifier
        self.domain = domain
        self.rules = _normalize_rules(rules)

    @property
    def descriptor(self) -> VariationalActionDescriptor:
        return VariationalActionDescriptor(
            "linear",
            "cell",
            (self.field_name,),
            (self.field_name,),
            ((self.field_name, "value"),),
            coefficient_values=(self.source,),
            provider_offers=("prepared-local",),
        )


class BoundaryLoadAction(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    load: VariationalCoefficient
    action_id: str = eqx.field(static=True)
    domain: IntegrationDomain | None
    rules: tuple[tuple[str, IntegrationRule], ...]

    def __init__(
        self,
        field_name: str,
        load,
        /,
        *,
        action_id="boundary-load",
        domain=None,
        rules=(),
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not identifier:
            raise ValueError("Boundary-load field and action IDs must be non-empty.")
        if domain is not None and (
            not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet"
        ):
            raise ValueError("BoundaryLoadAction requires an exterior-facet domain.")
        self.field_name = field
        self.load = (
            load if isinstance(load, VariationalCoefficient) else coefficient(load)
        )
        self.action_id = identifier
        self.domain = domain
        self.rules = _normalize_rules(rules)

    @property
    def descriptor(self) -> VariationalActionDescriptor:
        return VariationalActionDescriptor(
            "linear",
            "exterior_facet",
            (self.field_name,),
            (self.field_name,),
            ((self.field_name, "value"),),
            coefficient_values=(self.load,),
            provider_offers=("prepared-local",),
        )


__all__ = [
    "BoundaryLoadAction",
    "DiffusionAction",
    "TensorDiffusionAction",
    "IntegrationRule",
    "MassAction",
    "SourceAction",
    "VariationalActionDescriptor",
    "VariationalCoefficient",
    "coefficient",
]
