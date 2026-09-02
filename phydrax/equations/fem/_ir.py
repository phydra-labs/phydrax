#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import equinox as eqx
import jax

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FieldSlotRole = Literal[
    "unknown",
    "cell-local",
    "trace",
    "coefficient",
    "quadrature-state",
    "test",
    "trial",
    "control",
]
RegionKind = Literal[
    "cell",
    "exterior-facet",
    "interior-facet",
    "interface",
    "smoothing-patch",
    "embedded-volume",
    "embedded-interface",
    "contact-pair",
]
DifferentialOperator = Literal[
    "value",
    "grad",
    "sym-grad",
    "div",
    "curl",
    "normal-trace",
    "tangential-trace",
    "jump",
    "average",
    "smoothed-grad",
    "shape-average",
    "primitive-moment",
]
ActionKind = Literal[
    "residual",
    "energy",
    "bilinear",
    "linear",
    "functional",
    "material",
    "pairwise-volume-flux",
]


class FieldSlot(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    role: FieldSlotRole = eqx.field(static=True)
    space_id: str = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    slot_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: FieldSlotRole,
        space_id: str,
        /,
        *,
        value_shape: Sequence[int] = (),
    ):
        name_ = str(name)
        space = str(space_id)
        shape = tuple(int(size) for size in value_shape)
        if not name_ or not space or any(size <= 0 for size in shape):
            raise ValueError("Field slot name, space, or value shape is invalid.")
        if role not in (
            "unknown",
            "cell-local",
            "trace",
            "coefficient",
            "quadrature-state",
            "test",
            "trial",
            "control",
        ):
            raise ValueError("Unknown field slot role.")
        self.name = name_
        self.role = role
        self.space_id = space
        self.value_shape = shape
        self.slot_id = canonical_fingerprint(
            {
                "kind": "finite-element-field-slot",
                "name": name_,
                "role": role,
                "space": space,
                "value_shape": list(shape),
            }
        )


class RegionIR(StrictModule, NonTrainableState):
    region_kind: RegionKind = eqx.field(static=True)
    domain_id: str = eqx.field(static=True)
    rule_ids: tuple[tuple[str, str], ...] = eqx.field(static=True)
    region_id: str = eqx.field(static=True)

    def __init__(
        self,
        region_kind: RegionKind,
        domain_id: str,
        rule_ids: Sequence[tuple[str, str]],
        /,
    ):
        domain = str(domain_id)
        rules = tuple(sorted((str(block), str(rule)) for block, rule in rule_ids))
        if not domain or any(not block or not rule for block, rule in rules):
            raise ValueError("Region domain/rule identities must be non-empty.")
        if region_kind not in (
            "cell",
            "exterior-facet",
            "interior-facet",
            "interface",
            "smoothing-patch",
            "embedded-volume",
            "embedded-interface",
            "contact-pair",
        ):
            raise ValueError("Unknown integration region kind.")
        self.region_kind = region_kind
        self.domain_id = domain
        self.rule_ids = rules
        self.region_id = canonical_fingerprint(
            {
                "kind": "finite-element-region-ir",
                "region_kind": region_kind,
                "domain": domain,
                "rules": [list(item) for item in rules],
            }
        )


class FiniteElementActionIR(StrictModule, NonTrainableState):
    action_kind: ActionKind = eqx.field(static=True)
    output_slots: tuple[str, ...] = eqx.field(static=True)
    input_slots: tuple[str, ...] = eqx.field(static=True)
    operators: tuple[tuple[str, DifferentialOperator], ...] = eqx.field(static=True)
    region: RegionIR
    kernel_id: str = eqx.field(static=True)
    action_id: str = eqx.field(static=True)

    def __init__(
        self,
        action_kind: ActionKind,
        output_slots: Sequence[str],
        input_slots: Sequence[str],
        operators: Sequence[tuple[str, DifferentialOperator]],
        region: RegionIR,
        kernel_id: str,
        /,
    ):
        outputs = tuple(str(name) for name in output_slots)
        inputs = tuple(str(name) for name in input_slots)
        operations = tuple((str(name), operation) for name, operation in operators)
        kernel = str(kernel_id)
        if (
            not outputs
            or any(not name for name in outputs)
            or not inputs
            or any(not name for name in inputs)
            or not kernel
        ):
            raise ValueError("Local action field/kernel identifiers must be non-empty.")
        if len(set(outputs)) != len(outputs):
            raise ValueError("Local action output slots must be unique.")
        if len(set(inputs)) != len(inputs):
            raise ValueError("Local action input slots must be unique.")
        if not isinstance(region, RegionIR):
            raise TypeError("region must be RegionIR.")
        if action_kind not in (
            "residual",
            "energy",
            "bilinear",
            "linear",
            "functional",
            "material",
            "pairwise-volume-flux",
        ):
            raise ValueError("Unknown finite-element action kind.")
        self.action_kind = action_kind
        self.output_slots = outputs
        self.input_slots = inputs
        self.operators = operations
        self.region = region
        self.kernel_id = kernel
        self.action_id = canonical_fingerprint(
            {
                "kind": "finite-element-action",
                "action_kind": action_kind,
                "outputs": list(outputs),
                "inputs": list(inputs),
                "operators": [list(item) for item in operations],
                "region": region.region_id,
                "kernel": kernel,
            }
        )


class LocalActionIR(StrictModule, NonTrainableState):
    slots: tuple[FieldSlot, ...]
    actions: tuple[FiniteElementActionIR, ...]
    ir_id: str = eqx.field(static=True)

    def __init__(
        self,
        slots: Sequence[FieldSlot],
        actions: Sequence[FiniteElementActionIR],
        /,
    ):
        slots_ = tuple(slots)
        actions_ = tuple(actions)
        if not slots_ or not actions_:
            raise ValueError("LocalActionIR requires slots and actions.")
        names = tuple(slot.name for slot in slots_)
        if len(set(names)) != len(names):
            raise ValueError("LocalActionIR slot names must be unique.")
        declared = set(names)
        for action in actions_:
            if any(name not in declared for name in action.output_slots) or any(
                name not in declared for name in action.input_slots
            ):
                raise ValueError("Finite-element action references an undeclared slot.")
        self.slots = slots_
        self.actions = actions_
        self.ir_id = canonical_fingerprint(
            {
                "kind": "finite-element-local-action-ir",
                "slots": [slot.slot_id for slot in slots_],
                "actions": [action.action_id for action in actions_],
            }
        )


OperatorValueRole = Literal[
    "state",
    "coefficient",
    "geometry",
    "trace",
    "test",
    "trial",
    "dual",
    "status",
]
OperatorOpcode = Literal[
    "gather",
    "orient",
    "interpolate",
    "differentiate",
    "metric-transform",
    "physical-flux",
    "numerical-flux",
    "source",
    "mortar-project",
    "lift",
    "scatter",
    "mass-solve",
    "kernel",
    "reduction",
]
OperatorADPolicy = Literal["analytic", "autodiff", "custom", "unsupported"]


class OperatorValue(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    role: OperatorValueRole = eqx.field(static=True)
    value_shape: tuple[int, ...] = eqx.field(static=True)
    dtype_name: str = eqx.field(static=True)
    layout_id: str = eqx.field(static=True)
    value_id: str = eqx.field(static=True)

    def __init__(
        self,
        name: str,
        role: OperatorValueRole,
        /,
        *,
        value_shape: Sequence[int] = (),
        dtype_name: str = "dynamic",
        layout_id: str,
    ):
        name_ = str(name)
        shape = tuple(int(value) for value in value_shape)
        dtype = str(dtype_name)
        layout = str(layout_id)
        if (
            not name_
            or role
            not in (
                "state",
                "coefficient",
                "geometry",
                "trace",
                "test",
                "trial",
                "dual",
                "status",
            )
            or any(value <= 0 for value in shape)
            or not dtype
            or not layout
        ):
            raise ValueError("Operator value metadata is incomplete.")
        self.name = name_
        self.role = role
        self.value_shape = shape
        self.dtype_name = dtype
        self.layout_id = layout
        self.value_id = canonical_fingerprint(
            {
                "kind": "finite-element-operator-value",
                "name": name_,
                "role": role,
                "value_shape": list(shape),
                "dtype": dtype,
                "layout": layout,
            }
        )


class OperatorNode(StrictModule, NonTrainableState):
    opcode: OperatorOpcode = eqx.field(static=True)
    input_names: tuple[str, ...] = eqx.field(static=True)
    output_names: tuple[str, ...] = eqx.field(static=True)
    kernel_id: str = eqx.field(static=True)
    ad_policy: OperatorADPolicy = eqx.field(static=True)
    recompute: bool = eqx.field(static=True)
    node_id: str = eqx.field(static=True)

    def __init__(
        self,
        opcode: OperatorOpcode,
        input_names: Sequence[str],
        output_names: Sequence[str],
        kernel_id: str,
        /,
        *,
        ad_policy: OperatorADPolicy = "autodiff",
        recompute: bool = False,
    ):
        inputs = tuple(str(value) for value in input_names)
        outputs = tuple(str(value) for value in output_names)
        kernel = str(kernel_id)
        if (
            opcode
            not in (
                "gather",
                "orient",
                "interpolate",
                "differentiate",
                "metric-transform",
                "physical-flux",
                "numerical-flux",
                "source",
                "mortar-project",
                "lift",
                "scatter",
                "mass-solve",
                "kernel",
                "reduction",
            )
            or not inputs
            or not outputs
            or any(not value for value in (*inputs, *outputs))
            or not kernel
            or ad_policy not in ("analytic", "autodiff", "custom", "unsupported")
        ):
            raise ValueError("Operator node metadata is incomplete.")
        self.opcode = opcode
        self.input_names = inputs
        self.output_names = outputs
        self.kernel_id = kernel
        self.ad_policy = ad_policy
        self.recompute = bool(recompute)
        self.node_id = canonical_fingerprint(
            {
                "kind": "finite-element-operator-node",
                "opcode": opcode,
                "inputs": inputs,
                "outputs": outputs,
                "kernel": kernel,
                "ad_policy": ad_policy,
                "recompute": bool(recompute),
            }
        )


class OperatorProgram(StrictModule, NonTrainableState):
    values: tuple[OperatorValue, ...]
    nodes: tuple[OperatorNode, ...]
    output_names: tuple[str, ...] = eqx.field(static=True)
    bucket_id: str = eqx.field(static=True)
    program_id: str = eqx.field(static=True)

    def __init__(
        self,
        values: Sequence[OperatorValue],
        nodes: Sequence[OperatorNode],
        output_names: Sequence[str],
        /,
        *,
        bucket_id: str,
    ):
        values_ = tuple(values)
        nodes_ = tuple(nodes)
        outputs = tuple(str(value) for value in output_names)
        bucket = str(bucket_id)
        names = tuple(value.name for value in values_)
        if (
            not values_
            or not nodes_
            or len(set(names)) != len(names)
            or not outputs
            or any(not value for value in outputs)
            or not bucket
        ):
            raise ValueError(
                "OperatorProgram values, nodes, outputs, or bucket are invalid."
            )
        declared = set(names)
        produced = set()
        for node in nodes_:
            if any(
                name not in declared and name not in produced for name in node.input_names
            ):
                raise ValueError("Operator node consumes an undeclared value.")
            produced.update(node.output_names)
        if any(name not in declared and name not in produced for name in outputs):
            raise ValueError("OperatorProgram output is undeclared.")
        self.values = values_
        self.nodes = nodes_
        self.output_names = outputs
        self.bucket_id = bucket
        self.program_id = canonical_fingerprint(
            {
                "kind": "finite-element-operator-program",
                "values": [value.value_id for value in values_],
                "nodes": [node.node_id for node in nodes_],
                "outputs": outputs,
                "bucket": bucket,
            }
        )

    def execute(
        self,
        inputs: Mapping[str, Any],
        kernels: Mapping[str, Callable],
        /,
    ) -> tuple[Any, ...]:
        values = dict(inputs)
        for expected in self.values:
            if expected.name not in values:
                raise ValueError(f"Missing OperatorProgram input {expected.name!r}.")
        for node in self.nodes:
            if node.kernel_id not in kernels:
                raise ValueError(f"Missing OperatorProgram kernel {node.kernel_id!r}.")
            output = kernels[node.kernel_id](*(values[name] for name in node.input_names))
            output_values = output if isinstance(output, tuple) else (output,)
            if len(output_values) != len(node.output_names):
                raise ValueError(
                    "OperatorProgram kernel returned the wrong output count."
                )
            values.update(zip(node.output_names, output_values, strict=True))
        return tuple(values[name] for name in self.output_names)


def operator_program_from_local_ir(
    ir: LocalActionIR,
    /,
    *,
    bucket_id: str,
) -> OperatorProgram:
    if not isinstance(ir, LocalActionIR):
        raise TypeError("ir must be LocalActionIR.")
    role_map = {
        "unknown": "state",
        "cell-local": "state",
        "trace": "trace",
        "coefficient": "coefficient",
        "quadrature-state": "state",
        "test": "test",
        "trial": "trial",
        "control": "coefficient",
    }
    values = tuple(
        OperatorValue(
            slot.name,
            role_map[slot.role],
            value_shape=slot.value_shape,
            layout_id=slot.space_id,
        )
        for slot in ir.slots
    )
    nodes = []
    outputs = []
    for index, action in enumerate(ir.actions):
        opcode: OperatorOpcode = (
            "physical-flux"
            if action.action_kind == "pairwise-volume-flux"
            else "numerical-flux"
            if action.region.region_kind in ("interior-facet", "interface")
            else "source"
            if action.action_kind == "linear"
            else "kernel"
        )
        names = tuple(f"{name}@action-{index}" for name in action.output_slots)
        nodes.append(
            OperatorNode(
                opcode,
                action.input_slots,
                names,
                action.kernel_id,
            )
        )
        outputs.extend(names)
    return OperatorProgram(values, tuple(nodes), tuple(outputs), bucket_id=bucket_id)


class OperatorFusionPlan(StrictModule, NonTrainableState):
    groups: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, program: OperatorProgram, /):
        if not isinstance(program, OperatorProgram):
            raise TypeError("program must be OperatorProgram.")
        consumers: dict[str, int] = {}
        for node in program.nodes:
            for name in node.input_names:
                consumers[name] = consumers.get(name, 0) + 1
        groups = []
        current = []
        for index, node in enumerate(program.nodes):
            current.append(index)
            fusible = (
                len(node.output_names) == 1
                and consumers.get(node.output_names[0], 0) <= 1
                and node.opcode not in ("reduction", "mass-solve", "scatter")
            )
            if not fusible:
                groups.append(tuple(current))
                current = []
        if current:
            groups.append(tuple(current))
        self.groups = tuple(groups)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "operator-fusion-plan",
                "program": program.program_id,
                "groups": self.groups,
            }
        )


class LoweredOperatorProgram(StrictModule, NonTrainableState):
    program: OperatorProgram
    fusion: OperatorFusionPlan
    kernels: tuple[tuple[str, Callable], ...] = eqx.field(static=True)
    lowered_id: str = eqx.field(static=True)

    def __init__(
        self,
        program: OperatorProgram,
        kernels: Mapping[str, Callable],
        /,
    ):
        if not isinstance(program, OperatorProgram):
            raise TypeError("program must be OperatorProgram.")
        required = tuple(dict.fromkeys(node.kernel_id for node in program.nodes))
        if set(required) != set(kernels) or any(
            not callable(kernels[name]) for name in required
        ):
            raise ValueError("Lowered operator kernels must exactly match the program.")
        fusion = OperatorFusionPlan(program)
        self.program = program
        self.fusion = fusion
        self.kernels = tuple((name, kernels[name]) for name in required)
        self.lowered_id = canonical_fingerprint(
            {
                "kind": "lowered-operator-program",
                "program": program.program_id,
                "fusion": fusion.plan_id,
                "kernels": required,
            }
        )

    def __call__(self, inputs: Mapping[str, Any], /) -> tuple[Any, ...]:
        return self.program.execute(inputs, dict(self.kernels))

    def linearize(self, inputs: Mapping[str, Any], /):
        names = tuple(value.name for value in self.program.values)
        values = tuple(inputs[name] for name in names)

        def execute(*arguments):
            result = self(
                {name: value for name, value in zip(names, arguments, strict=True)}
            )
            return result[0] if len(result) == 1 else result

        output, pushforward = jax.linearize(execute, *values)
        _, pullback = jax.vjp(execute, *values)
        return output, pushforward, pullback


def lower_operator_program(
    program: OperatorProgram,
    kernels: Mapping[str, Callable],
    /,
) -> LoweredOperatorProgram:
    return LoweredOperatorProgram(program, kernels)


__all__ = [
    "ActionKind",
    "DifferentialOperator",
    "FieldSlot",
    "FieldSlotRole",
    "FiniteElementActionIR",
    "LocalActionIR",
    "OperatorADPolicy",
    "LoweredOperatorProgram",
    "lower_operator_program",
    "OperatorFusionPlan",
    "OperatorNode",
    "OperatorOpcode",
    "OperatorProgram",
    "OperatorValue",
    "OperatorValueRole",
    "RegionIR",
    "RegionKind",
    "operator_program_from_local_ir",
]
