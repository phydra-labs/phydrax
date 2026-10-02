#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import prod
from typing import Any, final, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from ..typing import PRNGKey
from ._ir import PDEExpression, PDEProblemIR


if TYPE_CHECKING:
    from phydrax.domain import DerivativeBackend, DomainFunction

    from .._differentiation import DerivativeRegularity, RegularityPolicy
    from ..operators.differential._requests import DerivativeStep
    from ..operators.differential._taylor_contracts import TaylorContractionPolicy
    from ..operators.differential._taylor_planning import TaylorContractionPlan


@final
@dataclass(frozen=True, slots=True)
class DifferentialDirection:
    coordinate: str
    axis: int | None
    order: int
    size: int
    component: bool = False


@final
@dataclass(frozen=True, slots=True)
class LinearDifferentialFamily:
    """Compact mixed-radix population of contractions of one exact operand."""

    operand: PDEExpression
    directions: tuple[DifferentialDirection, ...]
    component: int | None = None

    @property
    def population_size(self) -> int:
        return prod(item.size for item in self.directions if item.axis is None)

    @property
    def order(self) -> int:
        return sum(item.order for item in self.directions)

    @property
    def has_sum(self) -> bool:
        return any(item.axis is None for item in self.directions)

    @property
    def identity(self) -> tuple[object, ...]:
        return (self.operand, self.directions, self.component)


def normalize_linear_expression(expression: PDEExpression, /) -> PDEExpression:
    """Distribute only linear derivative chains, never move a coefficient."""
    arguments = tuple(normalize_linear_expression(item) for item in expression.args)
    node = PDEExpression(
        expression.op,
        arguments,
        symbol=expression.symbol,
        value=expression.value,
        coordinate=expression.coordinate,
        axis=expression.axis,
        order=expression.order,
        region=expression.region,
        dimension=expression.dimension,
        product=expression.product,
    )
    if node.op in ("derivative", "laplacian", "divergence", "gradient") and arguments:
        operand = arguments[0]
        if operand.op in ("add", "negate"):
            children = tuple(
                normalize_linear_expression(
                    PDEExpression(
                        node.op,
                        (item,),
                        coordinate=node.coordinate,
                        axis=node.axis,
                        order=node.order,
                    )
                )
                for item in operand.args
            )
            return PDEExpression(operand.op, children)
    return node


def _gradient_component_direction(
    operand: PDEExpression, selected: int | None, problem: PDEProblemIR, /
) -> DifferentialDirection:
    if operand.coordinate is None or selected is None:
        raise ValueError("A gradient component requires a coordinate and axis.")
    coordinate = next(
        item for item in problem.coordinates if item.name == operand.coordinate
    )
    return DifferentialDirection(coordinate.name, selected, 1, coordinate.size)


def _chain_direction(
    node: PDEExpression,
    problem: PDEProblemIR,
    directions: list[DifferentialDirection],
    component: int | None,
    /,
) -> DifferentialDirection:
    if node.coordinate is None:
        raise ValueError("A differential chain requires a declared coordinate.")
    coordinate = next(
        item for item in problem.coordinates if item.name == node.coordinate
    )
    match node.op:
        case "derivative":
            axis = 0 if coordinate.size == 1 and node.axis is None else node.axis
            if axis is None:
                raise ValueError(
                    "Multivariate partial derivatives require an explicit axis."
                )
            return DifferentialDirection(
                coordinate.name, axis, node.order, coordinate.size
            )
        case "laplacian":
            return DifferentialDirection(coordinate.name, None, 2, coordinate.size)
        case "divergence":
            if component is not None or any(item.component for item in directions):
                raise ValueError(
                    "A differential chain cannot select two divergence components."
                )
            return DifferentialDirection(coordinate.name, None, 1, coordinate.size, True)
        case _:
            raise ValueError("Unsupported native linear differential chain.")


def extract_linear_differential(
    expression: PDEExpression, problem: PDEProblemIR, /
) -> LinearDifferentialFamily | None:
    """Peel a maximal linear chain, keeping coefficients inside its exact root."""
    node = expression
    directions: list[DifferentialDirection] = []
    component: int | None = None
    while node.op in ("derivative", "laplacian", "divergence", "component"):
        if node.op == "component":
            selected = node.axis
            operand = node.args[0]
            if operand.op == "gradient":
                directions.append(
                    _gradient_component_direction(operand, selected, problem)
                )
                node = operand.args[0]
                continue
            if component is not None:
                break
            component = selected
            node = operand
            continue
        directions.append(_chain_direction(node, problem, directions, component))
        node = node.args[0]
    if not directions:
        return None
    return LinearDifferentialFamily(node, tuple(directions), component)


def _direction_ids(family: LinearDifferentialFamily, /) -> tuple[str, ...]:
    return tuple(
        f"sum_{number}:{item.coordinate}"
        if item.axis is None
        else f"partial:{item.coordinate}:{item.axis}"
        for number, item in enumerate(family.directions)
    )


def _contraction_signature(
    family: LinearDifferentialFamily, /
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}
    for name, item in zip(_direction_ids(family), family.directions, strict=True):
        counts[name] = counts.get(name, 0) + item.order
    return tuple(sorted(counts.items()))


def prepare_family(
    family: LinearDifferentialFamily,
    policy: TaylorContractionPolicy | None,
    /,
    *,
    gaussian: bool = False,
    regularity: DerivativeRegularity | None = None,
    regularity_policy: RegularityPolicy | None = None,
) -> TaylorContractionPlan:
    from ..operators.differential._taylor_contracts import TaylorContractionRequest
    from ..operators.differential._taylor_planning import plan_taylor_contractions

    if gaussian:
        directions = ("gaussian",)
        orders = (4,)
    else:
        signature = _contraction_signature(family)
        directions = tuple(name for name, _ in signature)
        orders = tuple(order for _, order in signature)
    return plan_taylor_contractions(
        (TaylorContractionRequest(directions, orders),),
        policy=policy,
        regularity=regularity,
        regularity_policy=regularity_policy,
    )


def _gaussian_contraction_directions(
    primals: tuple[Array, ...],
    labels: tuple[str, ...],
    family: LinearDifferentialFamily,
    probe: Array | None,
    /,
) -> dict[str, tuple[Array, ...]]:
    if probe is None:
        raise ValueError("Gaussian contraction requires a probe.")
    position = labels.index(family.directions[0].coordinate)
    return {
        "gaussian": tuple(
            probe if current == position else jnp.zeros_like(value)
            for current, value in enumerate(primals)
        )
    }


def _family_contraction_directions(
    primals: tuple[Array, ...],
    labels: tuple[str, ...],
    family: LinearDifferentialFamily,
    index: Array,
    probe: Array | None,
    gaussian: bool,
    /,
) -> tuple[dict[str, tuple[Array, ...]], int | Array | None]:
    """Decode an index without building a Cartesian pair table or Hessian."""
    selected_component: int | Array | None = family.component
    if gaussian:
        return (
            _gaussian_contraction_directions(primals, labels, family, probe),
            selected_component,
        )
    directions: dict[str, tuple[Array, ...]] = {}
    remainder = index
    for name, item in zip(_direction_ids(family), family.directions, strict=True):
        if name in directions:
            continue
        axis: int | Array | None = item.axis
        if axis is None:
            axis = remainder % item.size
            remainder = remainder // item.size
        if item.component and probe is None:
            selected_component = axis
        position = labels.index(item.coordinate)
        primal = primals[position]
        vector = (
            probe
            if probe is not None and item.axis is None
            else jax.nn.one_hot(axis, item.size, dtype=primal.dtype).reshape(primal.shape)
        )
        directions[name] = tuple(
            vector if current == position else jnp.zeros_like(value)
            for current, value in enumerate(primals)
        )
    return directions, selected_component


def _evaluate_family_contraction(
    function: Callable[..., Array],
    primals: tuple[Array, ...],
    directions: dict[str, tuple[Array, ...]],
    family: LinearDifferentialFamily,
    backend: DerivativeBackend,
    plan: TaylorContractionPlan | None,
    gaussian: bool,
    /,
) -> Array:
    if backend == "jet":
        from ..operators.differential._taylor_execution import (
            evaluate_taylor_contractions,
        )

        if plan is None:
            raise RuntimeError("A Taylor contraction requires its prepared static plan.")
        result = evaluate_taylor_contractions(function, primals, directions, plan)
        value = jnp.where(
            result.derivative_valid[0],
            result.values[0],
            jnp.full_like(result.values[0], jnp.nan),
        )
    elif backend == "ad":
        differentiated = function
        items = (("gaussian", 4),) if gaussian else _contraction_signature(family)
        for name, order in items:
            tangent = directions[name]
            for _ in range(order):
                previous = differentiated

                def directional(
                    *values: Array,
                    previous: Callable[..., Array] = previous,
                    tangent: tuple[Array, ...] = tangent,
                ) -> Array:
                    return jax.jvp(previous, values, tangent)[1]

                differentiated = directional
        value = differentiated(*primals)
    else:
        raise ValueError("Native contraction backend must be 'ad' or 'jet'.")
    return value


def family_contribution(
    function: Callable[..., Array],
    primals: tuple[Array, ...],
    labels: tuple[str, ...],
    family: LinearDifferentialFamily,
    index: Array,
    /,
    *,
    backend: DerivativeBackend,
    plan: TaylorContractionPlan | None,
    probe: Array | None = None,
    gaussian: bool = False,
) -> Array:
    """Decode an index without building a Cartesian pair table or Hessian."""
    directions, selected_component = _family_contraction_directions(
        primals, labels, family, index, probe, gaussian
    )
    value = _evaluate_family_contraction(
        function, primals, directions, family, backend, plan, gaussian
    )
    if selected_component is not None:
        value = value[..., selected_component]
    if probe is not None and any(item.component for item in family.directions):
        size = next(item.size for item in family.directions if item.component)
        if value.shape != (size,) or probe.size != size:
            raise ValueError(
                "Divergence requires the declared vector event and coordinate direction extents."
            )
        value = jnp.sum(value.reshape((size,)) * probe.reshape((size,)))
    return value / 3.0 if gaussian else value


@final
class _ExactFamilyCallable(StrictModule):
    operand: DomainFunction
    family: LinearDifferentialFamily = eqx.field(static=True)
    plan: TaylorContractionPlan = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    components: int = eqx.field(static=True)

    def __call__(
        self, *args: Any, key: PRNGKey | None = None, iter: object = None
    ) -> Array:
        del iter
        primals = tuple(jnp.asarray(value) for value in args)

        def function(*values: Array) -> Array:
            positions = {label: index for index, label in enumerate(self.labels)}
            value = jnp.asarray(
                self.operand.func(
                    *(values[positions[label]] for label in self.operand.deps), key=key
                )
            )
            if value.size != self.components:
                raise ValueError(
                    "The differential operand event shape conflicts with its PDE IR component type."
                )
            return value

        def contribution(index: Array) -> Array:
            return family_contribution(
                function,
                primals,
                self.labels,
                self.family,
                index,
                backend="jet",
                plan=self.plan,
            )

        first = contribution(jnp.asarray(0, dtype=jnp.int32))

        def add(index: Array, value: Array) -> Array:
            return value + contribution(index)

        return jax.lax.fori_loop(1, self.family.population_size, add, first)


@final
@dataclass(frozen=True, slots=True)
class NativeContributionPath:
    steps: tuple[DerivativeStep, ...]
    component: int


def prepare_native_contribution_paths(
    family: LinearDifferentialFamily,
    policy: TaylorContractionPolicy | None,
    /,
    *,
    backend: DerivativeBackend = "jet",
) -> tuple[NativeContributionPath, ...] | None:
    from ..operators.differential._requests import DerivativeStep
    from ..operators.differential._taylor_contracts import TaylorContractionResources

    summed = tuple(item for item in family.directions if item.component)
    if not summed:
        return ()
    if len(summed) != 1 or family.component is not None:
        raise ValueError(
            "Native divergence requires one unambiguous vector component projection."
        )
    resources = TaylorContractionResources() if policy is None else policy.resources
    if summed[0].size > resources.max_linear_terms:
        return None
    result: list[NativeContributionPath] = []
    for component in range(summed[0].size):
        steps = tuple(
            DerivativeStep(
                "laplacian" if item.axis is None and not item.component else "partial",
                item.coordinate,
                axis=component if item.component else item.axis,
                order=item.order,
                backend=backend,
            )
            for item in reversed(family.directions)
        )
        result.append(NativeContributionPath(steps, component))
    return tuple(result)


@final
class NativeFamilyCallable(StrictModule):
    functions: tuple[DomainFunction, ...]
    argument_positions: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    component_indices: tuple[int, ...] = eqx.field(static=True)
    event_components: int = eqx.field(static=True)

    @property
    def num_contributions(self) -> int:
        return len(self.functions)

    def contribution(
        self, index: Array, arguments: tuple[Array, ...], key: PRNGKey | None, /
    ) -> Array:
        branches: list[Callable[[tuple[Array, ...]], Array]] = []
        for function, positions, component in zip(
            self.functions, self.argument_positions, self.component_indices, strict=True
        ):

            def branch(
                values: tuple[Array, ...],
                function: DomainFunction = function,
                positions: tuple[int, ...] = positions,
                component: int = component,
            ) -> Array:
                value = jnp.asarray(
                    function.func(*(values[position] for position in positions), key=key)
                )
                if value.shape != (self.event_components,):
                    raise ValueError(
                        "Native divergence paths require the declared vector event shape before component projection."
                    )
                return value[component]

            branches.append(branch)
        return jax.lax.switch(index, tuple(branches), arguments)

    def __call__(
        self, *args: Any, key: PRNGKey | None = None, iter: object = None
    ) -> Array:
        del iter
        arguments = tuple(jnp.asarray(value) for value in args)

        def contribution(index: Array) -> Array:
            return self.contribution(index, arguments, key)

        values = jax.lax.map(
            contribution, jnp.arange(self.num_contributions, dtype=jnp.int32)
        )
        return jnp.sum(values, axis=0)


def try_owned_family(
    family: LinearDifferentialFamily,
    operand: DomainFunction,
    /,
    *,
    backend: DerivativeBackend = "jet",
    native_paths: tuple[NativeContributionPath, ...] | None = None,
) -> DomainFunction | None:
    from phydrax.domain import DomainFunction

    from ..domain._evaluation import FunctionBinding, PointwiseEvaluator
    from ..operators.differential._domain_ops import (
        _structured_derivative_provider,
        _try_derivative_path,
    )

    if any(item.component for item in family.directions):
        if (
            operand.derivative_rule is None
            and _structured_derivative_provider(operand) is None
        ):
            return None
        if native_paths is None:
            raise ValueError(
                "Native divergence contribution paths exceed the bounded Taylor linear-term resources."
            )
        functions: list[DomainFunction] = []
        components: list[int] = []
        declined = 0
        for path in native_paths:
            native = _try_derivative_path(operand, path.steps)
            if native is None:
                declined += 1
            else:
                functions.append(native.promote(operand.domain))
                components.append(path.component)
        if not functions:
            return None
        if declined:
            raise ValueError(
                "Native divergence ownership is incomplete across actual coordinate contributions; generic differentiation cannot replace accepted scientific paths."
            )
        labels = operand.domain.labels
        return DomainFunction(
            domain=operand.domain,
            deps=labels,
            func=PointwiseEvaluator(
                NativeFamilyCallable(
                    functions=tuple(functions),
                    argument_positions=tuple(
                        tuple(labels.index(label) for label in function.deps)
                        for function in functions
                    ),
                    component_indices=tuple(components),
                    event_components=next(
                        item.size for item in family.directions if item.component
                    ),
                ),
                binding=FunctionBinding(pass_key=True),
            ),
            metadata=operand.metadata,
        )
    from ..operators.differential._requests import DerivativeStep

    steps = tuple(
        DerivativeStep(
            "laplacian" if item.axis is None else "partial",
            item.coordinate,
            axis=item.axis,
            order=item.order,
            backend=backend,
        )
        for item in reversed(family.directions)
    )
    owned = _try_derivative_path(operand, steps)
    if owned is not None and family.component is not None:
        from ._compile import _map_value

        owned = _map_value(owned, lambda value: value[..., family.component])
    return owned


def compile_exact_family(
    family: LinearDifferentialFamily,
    operand: DomainFunction,
    policy: TaylorContractionPolicy | None,
    /,
    *,
    components: int,
    plan: TaylorContractionPlan | None = None,
    native_paths: tuple[NativeContributionPath, ...] | None = None,
) -> DomainFunction:
    from phydrax.domain import DomainFunction

    from ..domain._evaluation import FunctionBinding, PointwiseEvaluator
    from ..operators.differential._requests import (
        admit_direct_derivative,
        field_regularity,
    )

    if native_paths is None:
        native_paths = prepare_native_contribution_paths(family, policy)
    owned = try_owned_family(family, operand, native_paths=native_paths)
    if owned is not None:
        return owned
    if plan is None:
        plan = prepare_family(family, policy, regularity=field_regularity(operand))
    for coordinate in dict.fromkeys(item.coordinate for item in family.directions):
        admit_direct_derivative(operand, coordinate, plan.required_regularity_order)
    labels = operand.domain.labels
    return DomainFunction(
        domain=operand.domain,
        deps=labels,
        func=PointwiseEvaluator(
            _ExactFamilyCallable(
                operand=operand,
                family=family,
                plan=plan,
                labels=labels,
                components=components,
            ),
            binding=FunctionBinding(pass_key=True),
        ),
        metadata=operand.metadata,
    )
