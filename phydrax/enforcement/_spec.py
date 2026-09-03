#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast, Literal

import equinox as eqx
from jaxtyping import ArrayLike

from phydrax.conditions._base import AbstractCondition
from phydrax.conditions._ir import Condition
from phydrax.conditions._trace import (
    equal,
    field_jet,
    FieldJet,
    LinearTraceEquation,
    LinearTraceExpression,
)
from phydrax.conditions.boundary import Absorbing, Dirichlet, Neumann, Robin
from phydrax.conditions.initial import Initial
from phydrax.domain import (
    Boundary,
    ComponentSum,
    DomainComponent,
    DomainFunction,
    Fixed,
    FixedEnd,
    FixedStart,
    GraphDomain,
)

from .._frozendict import frozendict
from .._strict import StrictModule
from ._ansatz import (
    enforce_dirichlet,
    enforce_neumann,
    enforce_robin,
    enforce_sommerfeld,
)
from ._realization import AbstractFieldRealization


EnforcementStage = Literal["boundary", "initial", "interior", "global"]
EnforcementKind = Literal[
    "dirichlet",
    "neumann",
    "robin",
    "absorbing",
    "graph",
    "traction",
    "initial",
    "functional",
]


class EnforcementProofObligations(StrictModule):
    derivative_requirements: tuple["DerivativeRequirement", ...]
    pivot_identity: str = eqx.field(static=True)
    preservation_identities: tuple[str, ...] = eqx.field(static=True)
    output_shape: tuple[int, ...] | None = eqx.field(static=True)
    output_dtype: str | None = eqx.field(static=True)
    support_identity: str = eqx.field(static=True)
    provider_certified: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        derivative_requirements: Sequence["DerivativeRequirement"] = (),
        pivot_identity: str,
        preservation_identities: Sequence[str] = (),
        output_shape: tuple[int, ...] | None = None,
        output_dtype: str | None = None,
        support_identity: str,
        provider_certified: bool = False,
    ):
        if not pivot_identity or not support_identity:
            raise ValueError("Enforcement proof identities must be nonempty.")
        self.derivative_requirements = tuple(derivative_requirements)
        self.pivot_identity = pivot_identity
        self.preservation_identities = tuple(preservation_identities)
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.support_identity = support_identity
        self.provider_certified = bool(provider_certified)


class TraceLifting(StrictModule):
    kind: EnforcementKind = eqx.field(static=True)
    variable: str = eqx.field(static=True)
    mode: str = eqx.field(static=True)
    dirichlet_coefficient: Any
    neumann_coefficient: Any
    time_variable: str = eqx.field(static=True)
    wavespeed: Any
    lame_lambda: Any
    shear_modulus: Any

    def __init__(
        self,
        kind: EnforcementKind,
        variable: str,
        *,
        mode: str = "coordinate",
        dirichlet_coefficient: Any = 1.0,
        neumann_coefficient: Any = 1.0,
        time_variable: str = "t",
        wavespeed: Any = 1.0,
        lame_lambda: Any = 1.0,
        shear_modulus: Any = 1.0,
    ):
        if kind not in (
            "dirichlet",
            "neumann",
            "robin",
            "absorbing",
            "initial",
            "graph",
            "traction",
        ):
            raise ValueError("Unsupported typed trace lifting kind.")
        self.kind = kind
        self.variable = str(variable)
        self.mode = str(mode)
        self.dirichlet_coefficient = dirichlet_coefficient
        self.neumann_coefficient = neumann_coefficient
        self.time_variable = str(time_variable)
        self.wavespeed = wavespeed
        self.lame_lambda = lame_lambda
        self.shear_modulus = shear_modulus


class AffineEnforcementTransform(StrictModule):
    equation: LinearTraceEquation
    lifting: TraceLifting
    proof: EnforcementProofObligations

    def apply(
        self,
        value: DomainFunction,
        get_field: Callable[[str], DomainFunction],
        component: DomainComponent,
        pivot: str,
        /,
    ) -> DomainFunction:
        pivot_terms = tuple(
            (coefficient, jet)
            for coefficient, jet in self.equation.lhs.terms
            if jet.field == pivot
        )
        if len(pivot_terms) != 1:
            raise ValueError("Affine enforcement requires one declared pivot-field term.")
        pivot_coefficient, pivot_jet = pivot_terms[0]
        target = self.equation.rhs
        for coefficient, jet in self.equation.lhs.terms:
            if jet.field == pivot:
                continue
            if jet.order != 0 or jet.normal:
                raise ValueError(
                    "Cross-field derivative traces require a certified lifting provider."
                )
            target = target - coefficient * get_field(jet.field)
        target = target / pivot_coefficient
        if self.lifting.kind == "dirichlet":
            return enforce_dirichlet(
                value, component, var=self.lifting.variable, target=target
            )
        if self.lifting.kind == "neumann":
            return enforce_neumann(
                value,
                component,
                var=self.lifting.variable,
                target=target,
                mode=self.lifting.mode,
            )
        if self.lifting.kind == "robin":
            return enforce_robin(
                value,
                component,
                var=self.lifting.variable,
                dirichlet_coeff=self.lifting.dirichlet_coefficient,
                neumann_coeff=self.lifting.neumann_coefficient,
                target=target,
                mode=self.lifting.mode,
            )
        if self.lifting.kind == "absorbing":
            return enforce_sommerfeld(
                value,
                component,
                var=self.lifting.variable,
                time_var=self.lifting.time_variable,
                wavespeed=self.lifting.wavespeed,
                target=target,
                mode=self.lifting.mode,
            )
        if self.lifting.kind == "graph":
            from ._graph import enforce_graph_values

            return enforce_graph_values(value, component, target=target)
        if self.lifting.kind == "traction":
            from ._ansatz import enforce_traction

            return enforce_traction(
                value,
                component,
                var=self.lifting.variable,
                lambda_=self.lifting.lame_lambda,
                mu=self.lifting.shear_modulus,
                target=target,
            )
        raise RuntimeError("Initial targets are compiled as a joint overlay.")


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
    raise TypeError(
        "Arbitrary hard conditions require a typed AffineEnforcementTransform."
    )


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


def _built_in_transform(
    condition: AbstractCondition,
    field: str,
    requirements: tuple[DerivativeRequirement, ...],
    options: Mapping[str, Any],
    /,
) -> AffineEnforcementTransform:
    if isinstance(condition, Dirichlet):
        variable = str(options.get("var", condition.on.domain.labels[0]))
        jet = field_jet(field, variable)
        target = condition.target
        graph_support = isinstance(condition.on.domain, GraphDomain) or any(
            isinstance(condition.on.domain.factor(label), GraphDomain)
            for label in condition.on.domain.labels
        )
        lifting = TraceLifting(
            "graph" if graph_support else "dirichlet",
            variable,
        )
    elif isinstance(condition, Neumann):
        variable = condition.var
        jet = field_jet(field, variable, 1, normal=condition.mode == "normal")
        target = condition.target
        lifting = TraceLifting("neumann", variable, mode=condition.mode)
    elif isinstance(condition, Robin):
        variable = condition.var
        jet = field_jet(field, variable)
        target = condition.target
        lifting = TraceLifting(
            "robin",
            variable,
            mode=condition.mode,
            dirichlet_coefficient=condition.dirichlet_coefficient,
            neumann_coefficient=condition.neumann_coefficient,
        )
    elif isinstance(condition, Absorbing):
        variable = condition.var
        jet = field_jet(field, variable)
        target = condition.target
        lifting = TraceLifting(
            "absorbing",
            variable,
            mode=condition.mode,
            time_variable=condition.time_var,
            wavespeed=condition.wavespeed,
        )
    elif isinstance(condition, Initial):
        variable = condition.evolution_var
        jet = field_jet(field, variable, condition.order)
        target = condition.target
        lifting = TraceLifting("initial", variable)
    else:
        raise TypeError(
            "Arbitrary hard conditions require a typed AffineEnforcementTransform."
        )
    proof = EnforcementProofObligations(
        derivative_requirements=requirements,
        pivot_identity=f"{field}:{variable}:{jet.order}",
        support_identity=repr(condition.on.spec),
        provider_certified=True,
    )
    return AffineEnforcementTransform(equal(jet, target), lifting, proof)


class EnforcementSpec(StrictModule):
    """Declaration of one local ansatz or one prepared field realization."""

    condition: AbstractCondition | Condition
    field: str = eqx.field(static=True)
    dependencies: tuple[str, ...] = eqx.field(static=True)
    stage: EnforcementStage = eqx.field(static=True)
    kind: EnforcementKind = eqx.field(static=True)
    derivative_requirements: tuple[DerivativeRequirement, ...]
    initial_derivative_order: int = eqx.field(static=True)
    evolution_var: str = eqx.field(static=True)
    transform: AffineEnforcementTransform | None
    realization: AbstractFieldRealization | None
    options: frozendict[str, Any]

    def __init__(
        self,
        condition: AbstractCondition | Condition,
        /,
        *,
        field: str | None = None,
        dependencies: Sequence[str] | None = None,
        stage: EnforcementStage | None = None,
        kind: EnforcementKind | None = None,
        derivative_requirements: Sequence[DerivativeRequirement] | None = None,
        initial_derivative_order: int | None = None,
        evolution_var: str = "t",
        transform: AffineEnforcementTransform | None = None,
        realization: AbstractFieldRealization | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        resolved_options = {} if options is None else dict(options)
        if isinstance(condition, Condition):
            if not isinstance(realization, AbstractFieldRealization):
                raise TypeError(
                    "Typed Condition enforcement requires an AbstractFieldRealization."
                )
            if any(
                value is not None
                for value in (
                    field,
                    dependencies,
                    stage,
                    kind,
                    derivative_requirements,
                    initial_derivative_order,
                    transform,
                )
            ):
                raise ValueError(
                    "Typed Condition enforcement derives fields and staging from its "
                    "prepared realization."
                )
            sources = condition.fields.sources
            if not sources:
                raise ValueError("Typed enforcement requires at least one source field.")
            self.condition = condition
            self.field = sources[0]
            self.dependencies = tuple(sources[1:])
            self.stage = "global"
            self.kind = "functional"
            self.derivative_requirements = ()
            self.initial_derivative_order = 0
            self.evolution_var = str(evolution_var)
            self.transform = None
            self.realization = realization
            self.options = frozendict(resolved_options)
            return

        if not isinstance(condition, AbstractCondition):
            raise TypeError(
                "EnforcementSpec condition must be an AbstractCondition or Condition."
            )
        if realization is not None:
            raise TypeError(
                "Prepared realizations consume typed Condition declarations; call "
                "condition.as_condition() before constructing this specification."
            )
        if transform is not None and not isinstance(
            transform, AffineEnforcementTransform
        ):
            raise TypeError(
                "transform must be an AffineEnforcementTransform; untyped callables "
                "are not accepted."
            )
        if isinstance(condition.on, ComponentSum):
            raise TypeError(
                "Local ansatz enforcement requires one DomainComponent support; "
                "ComponentSum conditions require a typed joint realization."
            )
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
        resolved_stage = (
            _stage(condition.on, str(evolution_var)) if stage is None else stage
        )
        if resolved_stage not in ("boundary", "initial", "interior"):
            raise ValueError(
                "Local ansatz stage must be 'boundary', 'initial', or 'interior'."
            )
        resolved_kind = (
            transform.lifting.kind
            if transform is not None
            else _kind(condition)
            if kind is None
            else kind
        )
        if resolved_kind not in (
            "dirichlet",
            "neumann",
            "robin",
            "absorbing",
            "initial",
            "graph",
            "traction",
        ):
            raise ValueError("Unsupported local enforcement kind.")
        requirements = (
            _default_requirements(condition, target_field)
            if derivative_requirements is None
            else tuple(derivative_requirements)
        )
        if any(not isinstance(value, DerivativeRequirement) for value in requirements):
            raise TypeError(
                "derivative_requirements must contain DerivativeRequirement values."
            )
        initial_order = (
            condition.order
            if initial_derivative_order is None and isinstance(condition, Initial)
            else 0
            if initial_derivative_order is None
            else int(initial_derivative_order)
        )
        if initial_order < 0:
            raise ValueError("initial_derivative_order must be nonnegative.")
        resolved_transform = (
            _built_in_transform(
                condition,
                target_field,
                requirements,
                resolved_options,
            )
            if transform is None
            else transform
        )
        equation_fields = frozenset(
            jet.field for _, jet in resolved_transform.equation.lhs.terms
        )
        if target_field not in equation_fields:
            raise ValueError("Typed enforcement equation does not contain its field.")
        if not equation_fields.issubset(frozenset(condition.fields)):
            raise ValueError("Typed enforcement equation references undeclared fields.")
        self.condition = condition
        self.field = target_field
        self.dependencies = resolved_dependencies
        self.stage = resolved_stage
        self.kind = resolved_transform.lifting.kind
        self.derivative_requirements = requirements
        self.initial_derivative_order = initial_order
        self.evolution_var = str(evolution_var)
        self.transform = resolved_transform
        self.realization = None
        self.options = frozendict(resolved_options)

    @property
    def component(self) -> DomainComponent:
        if not isinstance(self.condition, AbstractCondition):
            raise TypeError("A typed joint realization has no single local component.")
        return cast(DomainComponent, self.condition.on)

    @property
    def co_vars(self) -> tuple[str, ...]:
        return self.dependencies

    @property
    def max_derivative_order(self) -> int:
        return max(
            (requirement.order for requirement in self.derivative_requirements), default=0
        )

    @property
    def time_derivative_order(self) -> int:
        return self.initial_derivative_order

    @property
    def initial_target(self) -> DomainFunction | ArrayLike | None:
        if self.stage != "initial" or not isinstance(self.condition, AbstractCondition):
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
        if self.transform is None:
            raise TypeError(
                "Prepared joint realizations are applied by EnforcementProgram."
            )
        return self.transform.apply(value, get_field, self.component, self.field)


__all__ = [
    "AffineEnforcementTransform",
    "DerivativeRequirement",
    "EnforcementKind",
    "EnforcementProofObligations",
    "EnforcementSpec",
    "EnforcementStage",
    "FieldJet",
    "LinearTraceEquation",
    "LinearTraceExpression",
    "TraceLifting",
    "equal",
    "field_jet",
]
