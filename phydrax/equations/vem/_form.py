#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization import IntegrationDomain
from ...discretization.vem import (
    VirtualElementRuntimeData,
    VirtualElementStabilizationPolicy,
)
from ...linalg import OperatorProperties
from .._variational import (
    BoundaryLoadAction,
    coefficient,
    DiffusionAction,
    MassAction,
    SourceAction,
    VariationalCoefficient,
)


class VirtualElementRobinAction(StrictModule, NonTrainableState):
    field_name: str = eqx.field(static=True)
    coefficient: VariationalCoefficient
    value: VariationalCoefficient
    domain: IntegrationDomain
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_name: str,
        coefficient_value,
        value,
        domain: IntegrationDomain,
        /,
        *,
        action_id: str = "robin",
    ):
        field = str(field_name)
        identifier = str(action_id)
        if not field or not identifier:
            raise ValueError("Robin field and action IDs must be non-empty.")
        if not isinstance(domain, IntegrationDomain) or domain.kind != "exterior_facet":
            raise ValueError(
                "VirtualElementRobinAction requires an exterior-facet domain."
            )
        self.field_name = field
        self.coefficient = (
            coefficient_value
            if isinstance(coefficient_value, VariationalCoefficient)
            else coefficient(coefficient_value)
        )
        self.value = (
            value if isinstance(value, VariationalCoefficient) else coefficient(value)
        )
        self.domain = domain
        self.action_id = identifier


VirtualElementAction = (
    DiffusionAction
    | MassAction
    | SourceAction
    | BoundaryLoadAction
    | VirtualElementRobinAction
)


class VirtualElementForm(StrictModule, NonTrainableState):
    form_id: str = eqx.field(static=True)
    field_name: str = eqx.field(static=True)
    actions: tuple[VirtualElementAction, ...]
    declared_properties: OperatorProperties

    def __init__(
        self,
        form_id: str,
        field_name: str,
        actions: Sequence[VirtualElementAction],
        /,
        *,
        properties: OperatorProperties | None = None,
    ):
        identifier = str(form_id)
        field = str(field_name)
        actions_ = tuple(actions)
        if not identifier or not field or not actions_:
            raise ValueError("VEM form ID, field and actions must be non-empty.")
        if not all(
            isinstance(
                action,
                (
                    DiffusionAction,
                    MassAction,
                    SourceAction,
                    BoundaryLoadAction,
                    VirtualElementRobinAction,
                ),
            )
            for action in actions_
        ):
            raise TypeError("VirtualElementForm contains an unsupported action.")
        if any(action.field_name != field for action in actions_):
            raise ValueError("Every VEM action must target the form field.")
        action_ids = tuple(action.action_id for action in actions_)
        if len(set(action_ids)) != len(action_ids):
            raise ValueError("VEM action IDs must be unique.")
        properties_ = OperatorProperties() if properties is None else properties
        self.form_id = canonical_fingerprint(
            {
                "kind": "virtual-element-form",
                "declared_id": identifier,
                "field": field,
                "actions": [
                    {
                        "type": type(action).__name__,
                        "id": action.action_id,
                        "domain": None
                        if action.domain is None
                        else action.domain.domain_id,
                    }
                    for action in actions_
                ],
            }
        )
        self.field_name = field
        self.actions = actions_
        self.declared_properties = properties_


class VirtualElementExecutionPolicy(StrictModule, NonTrainableState):
    realization: str = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    quadrature_degree_offset: int = eqx.field(static=True)
    stiffness_stabilization: VirtualElementStabilizationPolicy
    mass_stabilization: VirtualElementStabilizationPolicy
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        realization: str = "matrix_free",
        accumulation: str = "fast",
        quadrature_degree_offset: int = 2,
        stiffness_stabilization: VirtualElementStabilizationPolicy | None = None,
        mass_stabilization: VirtualElementStabilizationPolicy | None = None,
    ):
        realization_ = str(realization)
        accumulation_ = str(accumulation)
        offset = int(quadrature_degree_offset)
        if realization_ not in ("matrix_free", "sparse"):
            raise ValueError("VEM realization must be matrix_free or sparse.")
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown VEM accumulation policy.")
        if offset < 0:
            raise ValueError("quadrature_degree_offset must be nonnegative.")
        stiffness = (
            VirtualElementStabilizationPolicy()
            if stiffness_stabilization is None
            else stiffness_stabilization
        )
        mass = (
            VirtualElementStabilizationPolicy()
            if mass_stabilization is None
            else mass_stabilization
        )
        self.realization = realization_
        self.accumulation = accumulation_
        self.quadrature_degree_offset = offset
        self.stiffness_stabilization = stiffness
        self.mass_stabilization = mass
        self.policy_id = canonical_fingerprint(
            {
                "kind": "virtual-element-execution-policy",
                "realization": realization_,
                "accumulation": accumulation_,
                "quadrature_degree_offset": offset,
                "stiffness_stabilization": stiffness.policy_id,
                "mass_stabilization": mass.policy_id,
            }
        )


class VirtualElementExecutionContext(StrictModule):
    runtime: VirtualElementRuntimeData
    time: object
    lift: object
    lift_rate: object
    user_args: object

    def __init__(
        self,
        runtime: VirtualElementRuntimeData,
        /,
        *,
        time: ArrayLike = 0.0,
        lift: object = None,
        lift_rate: object = None,
        user_args: object = None,
    ):
        if not isinstance(runtime, VirtualElementRuntimeData):
            raise TypeError("runtime must be VirtualElementRuntimeData.")
        self.runtime = runtime
        self.time = jnp.asarray(time)
        self.lift = lift
        self.lift_rate = lift_rate
        self.user_args = user_args


__all__ = [
    "VirtualElementAction",
    "VirtualElementExecutionContext",
    "VirtualElementExecutionPolicy",
    "VirtualElementForm",
    "VirtualElementRobinAction",
]
