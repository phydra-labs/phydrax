#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, Literal

import equinox as eqx
from jaxtyping import ArrayLike

from phydrax.conditions import Absorbing, Dirichlet, Initial, Neumann, Robin
from phydrax.conditions._base import AbstractCondition
from phydrax.domain import (
    Boundary,
    ComponentSum,
    DomainComponent,
    DomainFunction,
    Fixed,
    FixedEnd,
    FixedStart,
)

from .._frozendict import frozendict
from .._strict import StrictModule
from ._ansatz import (
    enforce_dirichlet,
    enforce_neumann,
    enforce_robin,
    enforce_sommerfeld,
)


EnforcementStage = Literal["boundary", "initial", "interior"]
EnforcementKind = Literal[
    "dirichlet",
    "neumann",
    "robin",
    "absorbing",
    "initial",
    "custom",
]
EnforcementTransform = Callable[
    [DomainFunction, Callable[[str], DomainFunction]], DomainFunction
]


class DerivativeRequirement(StrictModule):
    """One derivative capability required by a hard transform."""

    field: str = eqx.field(static=True)
    variable: str = eqx.field(static=True)
    order: int = eqx.field(static=True)

    def __init__(self, field: str, variable: str, order: int, /):
        resolved_order = int(order)
        if resolved_order < 0:
            raise ValueError("Derivative requirement order must be nonnegative.")
        self.field = str(field)
        self.variable = str(variable)
        self.order = resolved_order


def _stage(component: DomainComponent, evolution_var: str, /) -> EnforcementStage:
    selections = tuple(
        component.spec.selection_for(label) for label in component.domain.labels
    )
    if any(isinstance(selection, Boundary) for selection in selections):
        return "boundary"
    if evolution_var in component.domain.labels and isinstance(
        component.spec.selection_for(evolution_var), (FixedStart, FixedEnd, Fixed)
    ):
        return "initial"
    return "interior"


def _kind(condition: AbstractCondition, /) -> EnforcementKind:
    if isinstance(condition, Dirichlet):
        return "dirichlet"
    if isinstance(condition, Neumann):
        return "neumann"
    if isinstance(condition, Robin):
        return "robin"
    if isinstance(condition, Absorbing):
        return "absorbing"
    if isinstance(condition, Initial):
        return "initial"
    return "custom"


def _default_requirements(
    condition: AbstractCondition,
    field: str,
    /,
) -> tuple[DerivativeRequirement, ...]:
    if isinstance(condition, (Neumann, Robin)):
        return (DerivativeRequirement(field, condition.var, 1),)
    if isinstance(condition, Absorbing):
        return (
            DerivativeRequirement(field, condition.var, 1),
            DerivativeRequirement(field, condition.time_var, 1),
        )
    if isinstance(condition, Initial):
        return (DerivativeRequirement(field, condition.evolution_var, condition.order),)
    return ()


class EnforcementSpec(StrictModule):
    """Typed declaration of one hard condition transform."""

    condition: AbstractCondition
    field: str = eqx.field(static=True)
    dependencies: tuple[str, ...] = eqx.field(static=True)
    stage: EnforcementStage = eqx.field(static=True)
    kind: EnforcementKind = eqx.field(static=True)
    derivative_requirements: tuple[DerivativeRequirement, ...]
    initial_derivative_order: int = eqx.field(static=True)
    evolution_var: str = eqx.field(static=True)
    transform: EnforcementTransform | None = eqx.field(static=True)
    options: frozendict[str, Any]

    def __init__(
        self,
        condition: AbstractCondition,
        /,
        *,
        field: str | None = None,
        dependencies: Sequence[str] | None = None,
        stage: EnforcementStage | None = None,
        kind: EnforcementKind | None = None,
        derivative_requirements: Sequence[DerivativeRequirement] | None = None,
        initial_derivative_order: int | None = None,
        evolution_var: str = "t",
        transform: EnforcementTransform | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        if not isinstance(condition, AbstractCondition):
            raise TypeError("EnforcementSpec condition must be an AbstractCondition.")
        if isinstance(condition.on, ComponentSum):
            raise TypeError("Hard enforcement requires one DomainComponent support.")
        if not isinstance(condition.on, DomainComponent):
            raise TypeError("Hard enforcement requires one DomainComponent support.")
        target_field = condition.fields[0] if field is None else str(field)
        if target_field not in condition.fields:
            raise ValueError(
                f"Enforcement target field {target_field!r} is not in condition fields "
                f"{condition.fields!r}."
            )
        resolved_dependencies = (
            tuple(name for name in condition.fields if name != target_field)
            if dependencies is None
            else tuple(str(name) for name in dependencies)
        )
        if target_field in resolved_dependencies:
            raise ValueError("Enforcement dependencies must exclude the target field.")
        if len(set(resolved_dependencies)) != len(resolved_dependencies):
            raise ValueError("Enforcement dependencies must be unique.")
        resolved_stage = _stage(condition.on, str(evolution_var)) if stage is None else stage
        if resolved_stage not in ("boundary", "initial", "interior"):
            raise ValueError("stage must be 'boundary', 'initial', or 'interior'.")
        resolved_kind = _kind(condition) if kind is None else kind
        if resolved_kind not in (
            "dirichlet",
            "neumann",
            "robin",
            "absorbing",
            "initial",
            "custom",
        ):
            raise ValueError("Unsupported enforcement transform kind.")
        if resolved_kind == "custom" and transform is None:
            raise ValueError("Custom enforcement requires transform=.")
        if resolved_kind != "custom" and transform is not None:
            raise ValueError("transform= is only valid for kind='custom'.")
        if derivative_requirements is None:
            requirements = _default_requirements(condition, target_field)
        else:
            requirements = tuple(derivative_requirements)
        if any(not isinstance(value, DerivativeRequirement) for value in requirements):
            raise TypeError(
                "derivative_requirements must contain DerivativeRequirement values."
            )
        if initial_derivative_order is None:
            initial_order = condition.order if isinstance(condition, Initial) else 0
        else:
            initial_order = int(initial_derivative_order)
        if initial_order < 0:
            raise ValueError("initial_derivative_order must be nonnegative.")

        self.condition = condition
        self.field = target_field
        self.dependencies = resolved_dependencies
        self.stage = resolved_stage
        self.kind = resolved_kind
        self.derivative_requirements = requirements
        self.initial_derivative_order = initial_order
        self.evolution_var = str(evolution_var)
        self.transform = transform
        self.options = frozendict({} if options is None else options)

    @property
    def component(self) -> DomainComponent:
        return cast(DomainComponent, self.condition.on)

    @property
    def co_vars(self) -> tuple[str, ...]:
        return self.dependencies

    @property
    def max_derivative_order(self) -> int:
        return max((requirement.order for requirement in self.derivative_requirements), default=0)

    @property
    def time_derivative_order(self) -> int:
        return self.initial_derivative_order

    @property
    def initial_target(self) -> DomainFunction | ArrayLike | None:
        if self.stage != "initial":
            return None
        if isinstance(self.condition, (Dirichlet, Initial)):
            return self.condition.target
        return None

    def apply(
        self,
        value: DomainFunction,
        get_field: Callable[[str], DomainFunction],
        /,
    ) -> DomainFunction:
        if self.kind == "custom":
            if self.transform is None:
                raise RuntimeError("Custom enforcement has no transform.")
            return self.transform(value, get_field)
        condition = self.condition
        component = self.component
        if isinstance(condition, Dirichlet):
            var = str(self.options.get("var", "x"))
            return enforce_dirichlet(value, component, var=var, target=condition.target)
        if isinstance(condition, Neumann):
            return enforce_neumann(
                value,
                component,
                var=condition.var,
                target=condition.target,
                mode=condition.mode,
            )
        if isinstance(condition, Robin):
            return enforce_robin(
                value,
                component,
                var=condition.var,
                dirichlet_coeff=condition.dirichlet_coefficient,
                neumann_coeff=condition.neumann_coefficient,
                target=condition.target,
                mode=condition.mode,
            )
        if isinstance(condition, Absorbing):
            return enforce_sommerfeld(
                value,
                component,
                var=condition.var,
                time_var=condition.time_var,
                wavespeed=condition.wavespeed,
                target=condition.target,
                mode=condition.mode,
            )
        if isinstance(condition, Initial):
            raise RuntimeError("Initial targets are compiled as a joint overlay.")
        raise TypeError(f"Unsupported hard condition {type(condition).__name__}.")


__all__ = [
    "DerivativeRequirement",
    "EnforcementKind",
    "EnforcementSpec",
    "EnforcementStage",
]
