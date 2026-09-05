#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from math import prod
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._assignment_core import hungarian_assignment_one
from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._differential_algebraic import DAEStructure, DifferentialAlgebraicSystem


DAETearingPolicy = Literal["none", "automatic", "declared"]


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{owner} must be a non-empty string.")
    return value.strip()


def _shape(value: Sequence[int], owner: str, /) -> tuple[int, ...]:
    result = tuple(int(size) for size in value)
    if any(size <= 0 for size in result):
        raise ValueError(f"{owner} dimensions must be positive.")
    return result


class DAEDerivativeIncidence(StrictModule, NonTrainableState):
    variable_name: str = eqx.field(static=True)
    derivative_order: int = eqx.field(static=True)

    def __init__(self, variable_name: str, derivative_order: int = 0, /):
        if not isinstance(derivative_order, int) or isinstance(derivative_order, bool):
            raise TypeError("derivative_order must be an integer.")
        if derivative_order < 0:
            raise ValueError("derivative_order must be nonnegative.")
        self.variable_name = _identifier(variable_name, "variable_name")
        self.derivative_order = derivative_order


class DAEVariableBlock(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)
    scale: Array

    def __init__(
        self,
        name: str,
        shape: Sequence[int] = (),
        maximum_derivative_order: int = 1,
        scale: ArrayLike = 1.0,
        /,
    ):
        if not isinstance(maximum_derivative_order, int) or isinstance(
            maximum_derivative_order, bool
        ):
            raise TypeError("maximum_derivative_order must be an integer.")
        if maximum_derivative_order < 0:
            raise ValueError("maximum_derivative_order must be nonnegative.")
        shape_ = _shape(shape, "DAEVariableBlock shape") if shape else ()
        scale_ = jnp.broadcast_to(jnp.asarray(scale), shape_ or ())
        if not jnp.issubdtype(scale_.dtype, jnp.inexact):
            scale_ = scale_.astype(float)
        scale_ = eqx.error_if(
            scale_,
            jnp.any(~jnp.isfinite(scale_)) | jnp.any(scale_ <= 0),
            "DAE variable scale must be positive and finite.",
        )
        self.name = _identifier(name, "DAEVariableBlock name")
        self.shape = shape_
        self.maximum_derivative_order = maximum_derivative_order
        self.scale = scale_

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1


class _DAEExecutionVariable(StrictModule, NonTrainableState):
    """Hashable structural variable metadata retained by compiled callables."""

    name: str = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    maximum_derivative_order: int = eqx.field(static=True)

    @property
    def size(self) -> int:
        return prod(self.shape) if self.shape else 1


class DAEJet(StrictModule):
    """Finite named variable jet supplied to declared residual blocks."""

    derivatives: tuple[tuple[Array, ...], ...]
    variable_names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self, variable_names: Sequence[str], derivatives: Sequence[Sequence[ArrayLike]], /
    ):
        names = tuple(variable_names)
        values = tuple(tuple(jnp.asarray(value) for value in jet) for jet in derivatives)
        if len(names) != len(values) or len(set(names)) != len(names):
            raise ValueError("DAEJet names and derivative blocks must be unique/aligned.")
        if any(not jet for jet in values):
            raise ValueError("Every DAEJet variable requires a zeroth derivative.")
        self.variable_names = names
        self.derivatives = values

    def value(self, variable_name: str, derivative_order: int = 0, /) -> Array:
        try:
            index = self.variable_names.index(variable_name)
        except ValueError as error:
            raise KeyError(f"Unknown DAE jet variable {variable_name!r}.") from error
        if derivative_order < 0 or derivative_order >= len(self.derivatives[index]):
            raise KeyError(
                f"Derivative order {derivative_order} is unavailable for {variable_name!r}."
            )
        return self.derivatives[index][derivative_order]

    def restrict(self, variable_names: Sequence[str], /) -> DAEJet:
        names = tuple(variable_names)
        return DAEJet(
            names,
            tuple(self.derivatives[self.variable_names.index(name)] for name in names),
        )


class DAEEquationBlock(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    residual: Callable[[Array, DAEJet, Any], Array]
    incidence: tuple[DAEDerivativeIncidence, ...]

    def __init__(
        self,
        name: str,
        residual: Callable[[Array, DAEJet, Any], Array],
        incidence: Sequence[DAEDerivativeIncidence],
        /,
    ):
        edges = tuple(incidence)
        if not callable(residual):
            raise TypeError("DAEEquationBlock residual must be callable.")
        if not edges or any(
            not isinstance(edge, DAEDerivativeIncidence) for edge in edges
        ):
            raise ValueError("DAEEquationBlock requires declared derivative incidence.")
        identities = tuple((edge.variable_name, edge.derivative_order) for edge in edges)
        if len(set(identities)) != len(identities):
            raise ValueError("DAEEquationBlock incidence edges must be unique.")
        self.name = _identifier(name, "DAEEquationBlock name")
        self.residual = residual
        self.incidence = edges


class DAEPort(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    potentials: tuple[str, ...] = eqx.field(static=True)
    flows: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, name: str, potentials: Sequence[str], flows: Sequence[str], /):
        potentials_ = tuple(_identifier(value, "potential") for value in potentials)
        flows_ = tuple(_identifier(value, "flow") for value in flows)
        if not potentials_ and not flows_:
            raise ValueError(
                "A DAEPort requires at least one potential or flow variable."
            )
        if len(set(potentials_ + flows_)) != len(potentials_ + flows_):
            raise ValueError("DAEPort variables must be unique.")
        self.name = _identifier(name, "DAEPort name")
        self.potentials = potentials_
        self.flows = flows_


class DAEConnection(StrictModule, NonTrainableState):
    port_ids: tuple[str, ...] = eqx.field(static=True)
    orientations: tuple[int, ...] = eqx.field(static=True)

    def __init__(self, port_ids: Sequence[str], orientations: Sequence[int], /):
        ports = tuple(_identifier(value, "port_id") for value in port_ids)
        signs = tuple(int(value) for value in orientations)
        if len(ports) < 2 or len(ports) != len(signs) or len(set(ports)) != len(ports):
            raise ValueError("A connection requires at least two unique oriented ports.")
        if any(sign not in (-1, 1) for sign in signs):
            raise ValueError("Connection orientations must be -1 or +1.")
        self.port_ids = ports
        self.orientations = signs


class DAEComponent(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    variables: tuple[DAEVariableBlock, ...]
    equations: tuple[DAEEquationBlock, ...]
    ports: tuple[DAEPort, ...]

    def __init__(
        self,
        name: str,
        variables: Sequence[DAEVariableBlock],
        equations: Sequence[DAEEquationBlock],
        ports: Sequence[DAEPort] = (),
        /,
    ):
        variables_ = tuple(variables)
        equations_ = tuple(equations)
        ports_ = tuple(ports)
        if not variables_ or any(
            not isinstance(value, DAEVariableBlock) for value in variables_
        ):
            raise ValueError("A DAEComponent requires variable blocks.")
        if any(not isinstance(value, DAEEquationBlock) for value in equations_):
            raise TypeError("component equations must be DAEEquationBlock values.")
        if any(not isinstance(value, DAEPort) for value in ports_):
            raise TypeError("component ports must be DAEPort values.")
        variable_names = tuple(value.name for value in variables_)
        if len(set(variable_names)) != len(variable_names):
            raise ValueError("Component variable names must be unique.")
        if len({value.name for value in equations_}) != len(equations_):
            raise ValueError("Component equation names must be unique.")
        if len({value.name for value in ports_}) != len(ports_):
            raise ValueError("Component port names must be unique.")
        variables_set = set(variable_names)
        for equation in equations_:
            unknown = {edge.variable_name for edge in equation.incidence} - variables_set
            if unknown:
                raise ValueError(
                    f"Equation {equation.name!r} references unknown variables {sorted(unknown)}."
                )
        for port in ports_:
            unknown = set(port.potentials + port.flows) - variables_set
            if unknown:
                raise ValueError(
                    f"Port {port.name!r} references unknown variables {sorted(unknown)}."
                )
        self.name = _identifier(name, "DAEComponent name")
        self.variables = variables_
        self.equations = equations_
        self.ports = ports_


class AcausalDAESource(StrictModule, NonTrainableState):
    components: tuple[DAEComponent, ...]
    connections: tuple[DAEConnection, ...]
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        components: Sequence[DAEComponent],
        connections: Sequence[DAEConnection] = (),
        /,
    ):
        components_ = tuple(components)
        connections_ = tuple(connections)
        if not components_ or any(
            not isinstance(value, DAEComponent) for value in components_
        ):
            raise ValueError("AcausalDAESource requires DAE components.")
        if any(not isinstance(value, DAEConnection) for value in connections_):
            raise TypeError("connections must be DAEConnection values.")
        names = tuple(value.name for value in components_)
        if len(set(names)) != len(names):
            raise ValueError("DAE component names must be unique.")
        port_ids = {
            f"{component.name}.{port.name}"
            for component in components_
            for port in component.ports
        }
        for connection in connections_:
            unknown = set(connection.port_ids) - port_ids
            if unknown:
                raise ValueError(
                    f"Connection references unknown ports {sorted(unknown)}."
                )
        self.components = components_
        self.connections = connections_
        self.source_id = canonical_fingerprint(
            {
                "kind": "acausal-dae-source",
                "components": sorted(names),
                "connections": sorted(
                    tuple(sorted(value.port_ids)) for value in connections_
                ),
            }
        )


class DAEStructuralPolicy(StrictModule, NonTrainableState):
    maximum_differentiations: int = eqx.field(static=True)
    maximum_tears: int = eqx.field(static=True)
    tearing: DAETearingPolicy = eqx.field(static=True)
    declared_tears: tuple[str, ...] = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_differentiations: int,
        maximum_tears: int,
        /,
        *,
        tearing: DAETearingPolicy = "automatic",
        declared_tears: Sequence[str] = (),
    ):
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in (maximum_differentiations, maximum_tears)
        ):
            raise TypeError("DAE structural capacities must be integers.")
        if maximum_differentiations < 0 or maximum_tears < 0:
            raise ValueError("DAE structural capacities must be nonnegative.")
        if tearing not in ("none", "automatic", "declared"):
            raise ValueError("Unknown DAE tearing policy.")
        tears = tuple(_identifier(value, "declared tear") for value in declared_tears)
        if tearing != "declared" and tears:
            raise ValueError("declared_tears require tearing='declared'.")
        self.maximum_differentiations = maximum_differentiations
        self.maximum_tears = maximum_tears
        self.tearing = tearing
        self.declared_tears = tears
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dae-structural-policy",
                "maximum_differentiations": maximum_differentiations,
                "maximum_tears": maximum_tears,
                "tearing": tearing,
                "declared_tears": list(tears),
            }
        )


class DAEStructuralAnalysis(StrictModule, NonTrainableState):
    variable_names: tuple[str, ...] = eqx.field(static=True)
    equation_names: tuple[str, ...] = eqx.field(static=True)
    original_incidence: tuple[tuple[tuple[str, int], ...], ...] = eqx.field(static=True)
    augmented_incidence: tuple[tuple[tuple[str, int], ...], ...] = eqx.field(static=True)
    matching: tuple[tuple[str, str], ...] = eqx.field(static=True)
    differentiation_counts: tuple[int, ...] = eqx.field(static=True)
    unmatched_equations: tuple[str, ...] = eqx.field(static=True)
    unmatched_variables: tuple[str, ...] = eqx.field(static=True)
    block_triangular_order: tuple[str, ...] = eqx.field(static=True)
    selected_tears: tuple[str, ...] = eqx.field(static=True)
    structural_index: int = eqx.field(static=True)
    hypothesis_scope: str = eqx.field(static=True)
    status: str = eqx.field(static=True)
    analysis_id: str = eqx.field(static=True)

    @property
    def successful(self) -> bool:
        return self.status == "success"


class _AssembledEquation(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    residual: Callable[[Array, DAEJet, Any], Array]
    incidence: tuple[DAEDerivativeIncidence, ...]


class _Assembly(StrictModule, NonTrainableState):
    variables: tuple[DAEVariableBlock, ...]
    equations: tuple[_AssembledEquation, ...]
    source_id: str = eqx.field(static=True)


class _ComponentResidual(StrictModule):
    residual: Callable[[Array, DAEJet, Any], Array]
    local_names: tuple[str, ...] = eqx.field(static=True)
    global_names: tuple[str, ...] = eqx.field(static=True)

    def __call__(self, time: Array, jet: DAEJet, args: Any, /) -> Array:
        local = DAEJet(
            self.local_names,
            tuple(
                jet.derivatives[jet.variable_names.index(name)]
                for name in self.global_names
            ),
        )
        return jnp.asarray(self.residual(time, local, args))


class _PotentialResidual(StrictModule):
    left: str = eqx.field(static=True)
    right: str = eqx.field(static=True)

    def __call__(self, time: Array, jet: DAEJet, args: Any, /) -> Array:
        del time, args
        return jet.value(self.left) - jet.value(self.right)


class _FlowResidual(StrictModule):
    variables: tuple[str, ...] = eqx.field(static=True)
    orientations: tuple[int, ...] = eqx.field(static=True)

    def __call__(self, time: Array, jet: DAEJet, args: Any, /) -> Array:
        del time, args
        values = tuple(
            sign * jet.value(name)
            for name, sign in zip(self.variables, self.orientations, strict=True)
        )
        result = values[0]
        for value in values[1:]:
            result = result + value
        return result


def _assemble(source: AcausalDAESource, /) -> _Assembly:
    variables = []
    equations = []
    ports: dict[str, tuple[DAEComponent, DAEPort]] = {}
    for component in source.components:
        local_names = tuple(variable.name for variable in component.variables)
        global_names = tuple(f"{component.name}.{name}" for name in local_names)
        for variable, global_name in zip(component.variables, global_names, strict=True):
            variables.append(
                DAEVariableBlock(
                    global_name,
                    variable.shape,
                    variable.maximum_derivative_order,
                    variable.scale,
                )
            )
        name_map = dict(zip(local_names, global_names, strict=True))
        for equation in component.equations:
            equations.append(
                _AssembledEquation(
                    f"{component.name}.{equation.name}",
                    _ComponentResidual(equation.residual, local_names, global_names),
                    tuple(
                        DAEDerivativeIncidence(
                            name_map[edge.variable_name], edge.derivative_order
                        )
                        for edge in equation.incidence
                    ),
                )
            )
        for port in component.ports:
            ports[f"{component.name}.{port.name}"] = (component, port)
    variables_by_name = {value.name: value for value in variables}
    for connection_index, connection in enumerate(source.connections):
        bound = tuple(ports[port_id] for port_id in connection.port_ids)
        potential_counts = {len(port.potentials) for _, port in bound}
        flow_counts = {len(port.flows) for _, port in bound}
        if len(potential_counts) != 1 or len(flow_counts) != 1:
            raise ValueError(
                "Connected DAE ports must expose identical potential/flow arity."
            )
        reference_component, reference_port = bound[0]
        for port_position, (component, port) in enumerate(bound[1:], start=1):
            for coordinate, (left_local, right_local) in enumerate(
                zip(reference_port.potentials, port.potentials, strict=True)
            ):
                left = f"{reference_component.name}.{left_local}"
                right = f"{component.name}.{right_local}"
                if variables_by_name[left].shape != variables_by_name[right].shape:
                    raise ValueError(
                        "Connected potential variable shapes must match exactly."
                    )
                equations.append(
                    _AssembledEquation(
                        f"connection[{connection_index}].potential[{port_position},{coordinate}]",
                        _PotentialResidual(left, right),
                        (
                            DAEDerivativeIncidence(left, 0),
                            DAEDerivativeIncidence(right, 0),
                        ),
                    )
                )
        for coordinate in range(next(iter(flow_counts))):
            flow_variables = tuple(
                f"{component.name}.{port.flows[coordinate]}" for component, port in bound
            )
            if len({variables_by_name[name].shape for name in flow_variables}) != 1:
                raise ValueError("Connected flow variable shapes must match exactly.")
            equations.append(
                _AssembledEquation(
                    f"connection[{connection_index}].flow[{coordinate}]",
                    _FlowResidual(flow_variables, connection.orientations),
                    tuple(DAEDerivativeIncidence(name, 0) for name in flow_variables),
                )
            )
    return _Assembly(
        tuple(sorted(variables, key=lambda value: value.name)),
        tuple(sorted(equations, key=lambda value: value.name)),
        source.source_id,
    )


def _maximum_matching(
    equations: tuple[_AssembledEquation, ...], variable_names: tuple[str, ...], /
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    adjacency = {
        equation.name: tuple(sorted({edge.variable_name for edge in equation.incidence}))
        for equation in equations
    }
    pair_left: dict[str, str] = {}
    pair_right: dict[str, str] = {}
    distance: dict[str, int] = {}

    def breadth_first() -> bool:
        queue: deque[str] = deque()
        for equation in sorted(adjacency):
            if equation not in pair_left:
                distance[equation] = 0
                queue.append(equation)
            else:
                distance[equation] = -1
        found = False
        while queue:
            equation = queue.popleft()
            for variable in adjacency[equation]:
                paired = pair_right.get(variable)
                if paired is None:
                    found = True
                elif distance[paired] < 0:
                    distance[paired] = distance[equation] + 1
                    queue.append(paired)
        return found

    def depth_first(equation: str) -> bool:
        for variable in adjacency[equation]:
            paired = pair_right.get(variable)
            if paired is None or (
                distance.get(paired, -1) == distance[equation] + 1 and depth_first(paired)
            ):
                pair_left[equation] = variable
                pair_right[variable] = equation
                return True
        distance[equation] = -1
        return False

    while breadth_first():
        for equation in sorted(adjacency):
            if equation not in pair_left:
                depth_first(equation)
    return (
        pair_left,
        tuple(sorted(set(adjacency) - set(pair_left))),
        tuple(sorted(set(variable_names) - set(pair_right))),
    )


_structural_assignment = jax.jit(hungarian_assignment_one)


def _minimum_differentiation_matching(
    equations: tuple[_AssembledEquation, ...],
    variables: tuple[DAEVariableBlock, ...],
    /,
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    """Match highest declared derivatives before differentiating constraints.

    Cardinality alone can match an energy equation to its algebraic flow and a
    temperature equality to its differential state. Differentiating that
    equality then removes a physical constraint even for an index-one system.
    First seek a zero-differentiation perfect matching using the sparse graph.
    Only genuinely differentiated, square systems need weighted assignment.
    """
    names = tuple(variable.name for variable in variables)
    orders = {variable.name: variable.maximum_derivative_order for variable in variables}
    if not any(orders.values()):
        return _maximum_matching(equations, names)
    highest = tuple(
        _AssembledEquation(
            equation.name,
            equation.residual,
            tuple(
                edge
                for edge in equation.incidence
                if edge.derivative_order == orders[edge.variable_name]
            ),
        )
        for equation in equations
    )
    matching = _maximum_matching(highest, names)
    if not matching[1] and not matching[2]:
        return matching
    cardinality = _maximum_matching(equations, names)
    if cardinality[1] or cardinality[2]:
        return cardinality
    indices = {name: index for index, name in enumerate(names)}
    costs = np.zeros((len(equations), len(names)), dtype=float)
    valid = np.zeros(costs.shape, dtype=bool)
    for row, equation in enumerate(equations):
        for edge in equation.incidence:
            column = indices[edge.variable_name]
            cost = max(orders[edge.variable_name] - edge.derivative_order, 0)
            if not valid[row, column] or cost < costs[row, column]:
                costs[row, column] = cost
            valid[row, column] = True
    columns, _, _, feasible, _ = _structural_assignment(
        jnp.asarray(costs), jnp.asarray(valid)
    )
    if not bool(feasible):
        raise RuntimeError(
            "Native weighted assignment rejected a fully matched DAE graph."
        )
    return (
        {
            equation.name: names[column]
            for equation, column in zip(equations, np.asarray(columns), strict=True)
        },
        (),
        (),
    )


def _strong_components(
    graph: Mapping[str, tuple[str, ...]], /
) -> tuple[tuple[str, ...], ...]:
    index = 0
    stack: list[str] = []
    indices: dict[str, int] = {}
    lows: dict[str, int] = {}
    on_stack: set[str] = set()
    output: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lows[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in graph.get(node, ()):
            if target not in indices:
                visit(target)
                lows[node] = min(lows[node], lows[target])
            elif target in on_stack:
                lows[node] = min(lows[node], indices[target])
        if lows[node] == indices[node]:
            component = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == node:
                    break
            output.append(tuple(sorted(component)))

    for node in sorted(graph):
        if node not in indices:
            visit(node)
    return tuple(sorted(output, key=lambda values: values[0]))


def analyze_dae_structure(
    source: AcausalDAESource, policy: DAEStructuralPolicy, /
) -> DAEStructuralAnalysis:
    """Analyze a finite declared derivative graph; never probe an opaque residual."""
    if not isinstance(source, AcausalDAESource):
        raise TypeError("source must be an AcausalDAESource.")
    if not isinstance(policy, DAEStructuralPolicy):
        raise TypeError("policy must be a DAEStructuralPolicy.")
    assembly = _assemble(source)
    variable_names = tuple(value.name for value in assembly.variables)
    variable_by_name = {value.name: value for value in assembly.variables}
    matching, unmatched_equations, unmatched_variables = (
        _minimum_differentiation_matching(assembly.equations, assembly.variables)
    )
    equation_by_name = {value.name: value for value in assembly.equations}
    differentiations = []
    augmented = []
    for equation in assembly.equations:
        matched = matching.get(equation.name)
        edge_order = max(
            (
                edge.derivative_order
                for edge in equation.incidence
                if edge.variable_name == matched
            ),
            default=0,
        )
        count = (
            max(variable_by_name[matched].maximum_derivative_order - edge_order, 0)
            if matched is not None
            else 0
        )
        differentiations.append(count)
        augmented.append(
            tuple(
                (edge.variable_name, edge.derivative_order + count)
                for edge in equation.incidence
                if edge.derivative_order + count
                <= variable_by_name[edge.variable_name].maximum_derivative_order
            )
        )
    graph: dict[str, list[str]] = {name: [] for name in variable_names}
    for equation_name, matched in matching.items():
        graph[matched].extend(
            edge.variable_name
            for edge in equation_by_name[equation_name].incidence
            if edge.variable_name != matched
        )
    components = _strong_components(
        {name: tuple(sorted(set(targets))) for name, targets in graph.items()}
    )
    block_order = tuple(name for component in components for name in component)
    if policy.tearing == "declared":
        unknown = set(policy.declared_tears) - set(variable_names)
        if unknown:
            raise ValueError(
                f"Declared tears reference unknown variables {sorted(unknown)}."
            )
        selected_tears = policy.declared_tears
    elif policy.tearing == "automatic":
        selected_tears = tuple(
            component[0] for component in components if len(component) > 1
        )
    else:
        selected_tears = ()
    maximum = max(differentiations, default=0)
    if len(assembly.equations) != len(assembly.variables):
        status = "nonsquare"
    elif unmatched_equations or unmatched_variables:
        status = "structurally-singular"
    elif maximum > policy.maximum_differentiations:
        status = "differentiation-capacity-exceeded"
    elif len(selected_tears) > policy.maximum_tears:
        status = "tear-capacity-exceeded"
    else:
        status = "success"
    structural_index = maximum + 1 if status == "success" else -1
    original = tuple(
        tuple((edge.variable_name, edge.derivative_order) for edge in equation.incidence)
        for equation in assembly.equations
    )
    analysis_id = canonical_fingerprint(
        {
            "kind": "dae-structural-analysis",
            "source": source.source_id,
            "policy": policy.policy_id,
            "variables": list(variable_names),
            "equations": [value.name for value in assembly.equations],
            "matching": sorted(matching.items()),
            "differentiations": differentiations,
            "tears": list(selected_tears),
            "status": status,
        }
    )
    return DAEStructuralAnalysis(
        variable_names,
        tuple(value.name for value in assembly.equations),
        original,
        tuple(augmented),
        tuple(sorted(matching.items())),
        tuple(differentiations),
        unmatched_equations,
        unmatched_variables,
        block_order,
        selected_tears,
        structural_index,
        "declared finite derivative incidence under quasi-regularity",
        status,
        analysis_id,
    )


def _execution_layout(
    variables: tuple[_DAEExecutionVariable, ...], /
) -> tuple[tuple[tuple[tuple[int, int], ...], ...], int]:
    offsets = []
    cursor = 0
    for variable in variables:
        block = []
        for _ in range(max(variable.maximum_derivative_order, 1)):
            block.append((cursor, cursor + variable.size))
            cursor += variable.size
        offsets.append(tuple(block))
    return tuple(offsets), cursor


def _jet_from_execution(
    variables: tuple[_DAEExecutionVariable, ...],
    offsets: tuple[tuple[tuple[int, int], ...], ...],
    state: Array,
    rate: Array,
    /,
) -> DAEJet:
    values = []
    for variable, blocks in zip(variables, offsets, strict=True):
        derivatives = [
            state[lower:upper].reshape(variable.shape) for lower, upper in blocks
        ]
        if variable.maximum_derivative_order > 0:
            lower, upper = blocks[-1]
            derivatives.append(rate[lower:upper].reshape(variable.shape))
        values.append(tuple(derivatives))
    return DAEJet(tuple(value.name for value in variables), tuple(values))


def _shifted_jet(jet: DAEJet, /) -> DAEJet:
    return DAEJet(
        jet.variable_names,
        tuple(
            tuple(
                derivatives[index + 1]
                if index + 1 < len(derivatives)
                else jnp.zeros_like(value)
                for index, value in enumerate(derivatives)
            )
            for derivatives in jet.derivatives
        ),
    )


def _total_derivative(
    residual: Callable[[Array, DAEJet, Any], Array],
    count: int,
    time: Array,
    jet: DAEJet,
    args: Any,
    /,
) -> Array:
    if count == 0:
        return jnp.asarray(residual(time, jet, args))

    def previous(t: Array, values: tuple[tuple[Array, ...], ...]) -> Array:
        return _total_derivative(
            residual, count - 1, t, DAEJet(jet.variable_names, values), args
        )

    shifted = _shifted_jet(jet)
    return jax.jvp(
        previous, (time, jet.derivatives), (jnp.ones_like(time), shifted.derivatives)
    )[1]


class _ReducedResidual(StrictModule):
    variables: tuple[_DAEExecutionVariable, ...]
    equations: tuple[_AssembledEquation, ...]
    offsets: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)
    differentiations: tuple[int, ...] = eqx.field(static=True)

    def __call__(
        self, time: Array, state: Array, state_rate: Array, args: Any, /
    ) -> Array:
        jet = _jet_from_execution(self.variables, self.offsets, state, state_rate)
        rows = [
            _total_derivative(equation.residual, count, time, jet, args).reshape((-1,))
            for equation, count in zip(self.equations, self.differentiations, strict=True)
        ]
        for variable, blocks in zip(self.variables, self.offsets, strict=True):
            for order in range(max(variable.maximum_derivative_order - 1, 0)):
                lower, upper = blocks[order]
                next_lower, next_upper = blocks[order + 1]
                rows.append(state_rate[lower:upper] - state[next_lower:next_upper])
        return jnp.concatenate(tuple(rows))


class _ResidualAudit(StrictModule):
    variables: tuple[_DAEExecutionVariable, ...]
    equations: tuple[_AssembledEquation, ...]
    offsets: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)

    def __call__(
        self, time: Array, state: Array, state_rate: Array, args: Any = None, /
    ) -> Array:
        jet = _jet_from_execution(self.variables, self.offsets, state, state_rate)
        return jnp.concatenate(
            tuple(
                equation.residual(time, jet, args).reshape((-1,))
                for equation in self.equations
            )
        )


class _Reconstruction(StrictModule):
    variables: tuple[_DAEExecutionVariable, ...]
    offsets: tuple[tuple[tuple[int, int], ...], ...] = eqx.field(static=True)

    def __call__(self, state: ArrayLike, rate: ArrayLike, /) -> DAEJet:
        return _jet_from_execution(
            self.variables, self.offsets, jnp.asarray(state), jnp.asarray(rate)
        )


class ReducedDAECompilation(StrictModule, NonTrainableState):
    system: DifferentialAlgebraicSystem
    reconstruction: _Reconstruction
    residual_audit: _ResidualAudit
    structure: DAEStructure
    fixed_state_mask: Array
    fixed_rate_mask: Array
    analysis: DAEStructuralAnalysis
    compilation_id: str = eqx.field(static=True)


def _verify_declared_incidence(assembly: _Assembly, args: Any, /) -> None:
    names = tuple(value.name for value in assembly.variables)
    ones = tuple(
        tuple(
            jnp.ones(variable.shape) for _ in range(variable.maximum_derivative_order + 1)
        )
        for variable in assembly.variables
    )
    jet = DAEJet(names, ones)
    for equation in assembly.equations:
        declared = {
            (edge.variable_name, edge.derivative_order) for edge in equation.incidence
        }
        for variable_index, variable in enumerate(assembly.variables):
            for order in range(variable.maximum_derivative_order + 1):
                tangent_values = tuple(
                    tuple(
                        jnp.ones_like(value)
                        if other_index == variable_index and derivative_index == order
                        else jnp.zeros_like(value)
                        for derivative_index, value in enumerate(derivatives)
                    )
                    for other_index, derivatives in enumerate(jet.derivatives)
                )
                action = jax.jvp(
                    lambda values: equation.residual(
                        jnp.asarray(1.0), DAEJet(names, values), args
                    ),
                    (jet.derivatives,),
                    (tangent_values,),
                )[1]
                if (
                    np.any(np.asarray(jax.device_get(jnp.abs(action) > 1.0e-12)))
                    and (variable.name, order) not in declared
                ):
                    raise ValueError(
                        f"Equation {equation.name!r} has an undeclared JVP "
                        f"incidence on {variable.name!r} derivative {order}."
                    )


def compile_acausal_dae(
    source: AcausalDAESource, policy: DAEStructuralPolicy, /, *, args: Any = None
) -> ReducedDAECompilation:
    """Lower successful declared structural analysis into the canonical DAE runtime."""
    analysis = analyze_dae_structure(source, policy)
    if not analysis.successful:
        raise ValueError(
            "DAE structural analysis failed closed with status "
            f"{analysis.status!r}; unmatched equations="
            f"{analysis.unmatched_equations}, unmatched variables="
            f"{analysis.unmatched_variables}."
        )
    assembly = _assemble(source)
    execution_variables = tuple(
        _DAEExecutionVariable(
            variable.name,
            variable.shape,
            variable.maximum_derivative_order,
        )
        for variable in assembly.variables
    )
    _verify_declared_incidence(assembly, args)
    offsets, state_size = _execution_layout(execution_variables)
    sample_jet = DAEJet(
        tuple(value.name for value in assembly.variables),
        tuple(
            tuple(
                jnp.ones(value.shape) for _ in range(value.maximum_derivative_order + 1)
            )
            for value in assembly.variables
        ),
    )
    equation_sizes = tuple(
        int(jnp.asarray(equation.residual(jnp.asarray(1.0), sample_jet, args)).size)
        for equation in assembly.equations
    )
    matching = dict(analysis.matching)
    variable_by_name = {value.name: value for value in assembly.variables}
    for equation, size in zip(assembly.equations, equation_sizes, strict=True):
        matched = matching[equation.name]
        if size != variable_by_name[matched].size:
            raise ValueError(
                f"Equation {equation.name!r} residual size {size} does not "
                f"match its structurally matched variable {matched!r} size "
                f"{variable_by_name[matched].size}."
            )
    if sum(equation_sizes) != sum(value.size for value in assembly.variables):
        raise ValueError("Original DAE residual and variable scalar counts must match.")
    residual = _ReducedResidual(
        execution_variables,
        assembly.equations,
        offsets,
        analysis.differentiation_counts,
    )
    variable_roles = []
    for variable in assembly.variables:
        role = "differential" if variable.maximum_derivative_order > 0 else "algebraic"
        variable_roles.extend(
            [role] * variable.size * max(variable.maximum_derivative_order, 1)
        )
    equation_roles = []
    for equation, size in zip(assembly.equations, equation_sizes, strict=True):
        matched = variable_by_name[matching[equation.name]]
        role = "differential" if matched.maximum_derivative_order > 0 else "algebraic"
        equation_roles.extend([role] * size)
    for variable in assembly.variables:
        equation_roles.extend(
            ["differential"]
            * variable.size
            * max(variable.maximum_derivative_order - 1, 0)
        )
    structure = DAEStructure(
        tuple(variable_roles), equation_roles=tuple(equation_roles), component_axis=-1
    )
    state_scale = jnp.concatenate(
        tuple(
            jnp.broadcast_to(variable.scale, variable.shape).reshape((-1,))
            for variable in assembly.variables
            for _ in range(max(variable.maximum_derivative_order, 1))
        )
    )
    system_id = f"reduced-dae:{analysis.analysis_id}"
    system = DifferentialAlgebraicSystem(
        residual,
        state_shape=(state_size,),
        structure=structure,
        state_scale=state_scale,
        state_rate_scale=state_scale,
        residual_scale=jnp.ones((state_size,), dtype=state_scale.dtype),
        system_id=system_id,
    )
    differential = structure.differential_variable_mask((state_size,))
    compilation_id = canonical_fingerprint(
        {
            "kind": "reduced-dae-compilation",
            "source": source.source_id,
            "analysis": analysis.analysis_id,
            "system": system_id,
        }
    )
    return ReducedDAECompilation(
        system,
        _Reconstruction(execution_variables, offsets),
        _ResidualAudit(execution_variables, assembly.equations, offsets),
        structure,
        differential,
        ~differential,
        analysis,
        compilation_id,
    )


__all__ = [
    "AcausalDAESource",
    "DAEComponent",
    "DAEConnection",
    "DAEDerivativeIncidence",
    "DAEEquationBlock",
    "DAEJet",
    "DAEPort",
    "DAEStructuralAnalysis",
    "DAEStructuralPolicy",
    "DAEVariableBlock",
    "ReducedDAECompilation",
    "analyze_dae_structure",
    "compile_acausal_dae",
]
