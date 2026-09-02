#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import core as jax_core

from ._fingerprint import canonical_fingerprint
from ._precision import precision_dtype_name, ScalarPrecisionDType


RewritePrimitive: TypeAlias = Literal[
    "dot_general",
    "conv_general_dilated",
    "reduce_sum",
    "reduce_prod",
    "elementwise",
]

_SUPPORTED_PURE = frozenset(
    {
        "abs",
        "add",
        "broadcast_in_dim",
        "clamp",
        "concatenate",
        "cond",
        "convert_element_type",
        "conv_general_dilated",
        "cos",
        "div",
        "dot_general",
        "dynamic_slice",
        "eq",
        "exp",
        "expm1",
        "ge",
        "gt",
        "integer_pow",
        "le",
        "log",
        "log1p",
        "lt",
        "max",
        "min",
        "mul",
        "neg",
        "ne",
        "pow",
        "reduce_and",
        "reduce_max",
        "reduce_min",
        "reduce_or",
        "reduce_prod",
        "reduce_sum",
        "reshape",
        "scan",
        "select_n",
        "sin",
        "slice",
        "sqrt",
        "squeeze",
        "stop_gradient",
        "sub",
        "tanh",
        "transpose",
        "while",
    }
)

_JAX_PRECISION = {
    "default": jax.lax.Precision.DEFAULT,
    "high": jax.lax.Precision.HIGH,
    "highest": jax.lax.Precision.HIGHEST,
}


@dataclass(frozen=True, slots=True)
class PrecisionRewriteRule:
    """One typed rewrite for a finite public JAX primitive family."""

    primitive: RewritePrimitive
    accumulator_dtype: ScalarPrecisionDType
    output_dtype: ScalarPrecisionDType
    precision: Literal["default", "high", "highest"] = "high"

    def __init__(
        self,
        primitive: RewritePrimitive,
        accumulator_dtype: Any,
        output_dtype: Any,
        precision: Literal["default", "high", "highest"] = "high",
    ):
        if primitive not in (
            "dot_general",
            "conv_general_dilated",
            "reduce_sum",
            "reduce_prod",
            "elementwise",
        ):
            raise ValueError(f"Unsupported rewrite primitive family {primitive!r}.")
        if precision not in ("default", "high", "highest"):
            raise ValueError("precision must be default, high, or highest.")
        object.__setattr__(self, "primitive", primitive)
        object.__setattr__(
            self, "accumulator_dtype", precision_dtype_name(accumulator_dtype)
        )
        object.__setattr__(self, "output_dtype", precision_dtype_name(output_dtype))
        object.__setattr__(self, "precision", precision)


@dataclass(frozen=True, slots=True)
class PrecisionRewritePolicy:
    rules: tuple[PrecisionRewriteRule, ...]
    unsupported: Literal["error", "recorded-pass-through"] = "error"

    def __init__(
        self,
        rules: Sequence[PrecisionRewriteRule],
        unsupported: Literal["error", "recorded-pass-through"] = "error",
    ):
        rules_ = tuple(rules)
        if not rules_ or not all(
            isinstance(rule, PrecisionRewriteRule) for rule in rules_
        ):
            raise TypeError("rules must contain PrecisionRewriteRule values.")
        names = tuple(rule.primitive for rule in rules_)
        if len(set(names)) != len(names):
            raise ValueError("Precision rewrite primitive families must be unique.")
        if unsupported not in ("error", "recorded-pass-through"):
            raise ValueError("Unknown unsupported-equation policy.")
        object.__setattr__(self, "rules", rules_)
        object.__setattr__(self, "unsupported", unsupported)


@dataclass(frozen=True, slots=True)
class PrecisionRewritePlan:
    closed_jaxpr: Any
    rules: tuple[PrecisionRewriteRule, ...]
    input_treedef: jax.tree_util.PyTreeDef
    output_treedef: jax.tree_util.PyTreeDef
    input_signature: tuple[tuple[tuple[int, ...], str], ...]
    output_signature: tuple[tuple[tuple[int, ...], str], ...]
    original_fingerprint: str
    rewritten_fingerprint: str
    equation_records: tuple[tuple[str, str], ...]
    provider: str
    device_identity: str
    plan_id: str


@dataclass(frozen=True, slots=True)
class PrecisionSelectionCandidate:
    name: str
    policy: PrecisionRewritePolicy

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Precision selection candidate name must be non-empty.")
        if not isinstance(self.policy, PrecisionRewritePolicy):
            raise TypeError("candidate policy must be a PrecisionRewritePolicy.")


@dataclass(frozen=True, slots=True)
class PrecisionSelectionPolicy:
    mode: Literal["compatible", "calibrated"]
    candidates: tuple[PrecisionSelectionCandidate, ...]
    relative_tolerance: float
    absolute_tolerance: float
    warmups: int
    repeats: int

    def __init__(
        self,
        candidates: Sequence[PrecisionSelectionCandidate],
        *,
        mode: Literal["compatible", "calibrated"] = "compatible",
        relative_tolerance: float = 1e-5,
        absolute_tolerance: float = 1e-6,
        warmups: int = 1,
        repeats: int = 3,
    ):
        candidates_ = tuple(candidates)
        if mode not in ("compatible", "calibrated"):
            raise ValueError("Selection mode must be compatible or calibrated.")
        if not candidates_ or not all(
            isinstance(item, PrecisionSelectionCandidate) for item in candidates_
        ):
            raise TypeError("candidates must contain selection candidates.")
        names = tuple(item.name for item in candidates_)
        if len(set(names)) != len(names):
            raise ValueError("Candidate names must be unique.")
        relative_tolerance_ = float(relative_tolerance)
        absolute_tolerance_ = float(absolute_tolerance)
        if (
            not np.isfinite(relative_tolerance_)
            or not np.isfinite(absolute_tolerance_)
            or relative_tolerance_ < 0
            or absolute_tolerance_ < 0
        ):
            raise ValueError("Selection tolerances must be finite and non-negative.")
        if int(warmups) < 0 or int(repeats) < 1:
            raise ValueError("Selection warmups/repeats are invalid.")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "candidates", candidates_)
        object.__setattr__(self, "relative_tolerance", relative_tolerance_)
        object.__setattr__(self, "absolute_tolerance", absolute_tolerance_)
        object.__setattr__(self, "warmups", int(warmups))
        object.__setattr__(self, "repeats", int(repeats))


@dataclass(frozen=True, slots=True)
class PrecisionSelectionEvidence:
    selected: str
    candidates: tuple[tuple[str, str, float, float], ...]
    device_identity: str
    workload_fingerprint: str
    claim: str
    evidence_id: str


@dataclass(frozen=True, slots=True)
class PreparedPrecisionSelection:
    plan: PrecisionRewritePlan
    evidence: PrecisionSelectionEvidence


def _signature(tree: Any, /) -> tuple[tuple[tuple[int, ...], str], ...]:
    return tuple(
        (tuple(int(size) for size in leaf.shape), jnp.dtype(leaf.dtype).name)
        for leaf in jax.tree.leaves(tree)
    )


def _device_identity(device: Any | None, /) -> str:
    resolved = jax.devices()[0] if device is None else device
    return f"{resolved.platform}:{resolved.device_kind}:{resolved.id}"


def _rule_map(policy: PrecisionRewritePolicy, /) -> dict[str, PrecisionRewriteRule]:
    return {rule.primitive: rule for rule in policy.rules}


def _rewrite_nested(
    value: Any,
    policy: PrecisionRewritePolicy,
    path: str,
    /,
) -> tuple[Any, tuple[tuple[str, str], ...], bool]:
    if isinstance(value, jax_core.ClosedJaxpr):
        rewritten, records = _rewrite_jaxpr(
            value.jaxpr, policy, path, allow_boundary_dtype_change=False
        )
        return value.replace(jaxpr=rewritten), records, True
    if isinstance(value, jax_core.Jaxpr):
        rewritten, records = _rewrite_jaxpr(
            value, policy, path, allow_boundary_dtype_change=False
        )
        return rewritten, records, True
    if isinstance(value, tuple):
        values = []
        records = []
        changed = False
        for index, item in enumerate(value):
            rewritten, nested, item_changed = _rewrite_nested(
                item,
                policy,
                f"{path}/{index}",
            )
            values.append(rewritten)
            records.extend(nested)
            changed = changed or item_changed
        return tuple(values), tuple(records), changed
    return value, (), False


def _rewrite_jaxpr(
    jaxpr: Any,
    policy: PrecisionRewritePolicy,
    path: str,
    /,
    *,
    allow_boundary_dtype_change: bool,
):
    rules = _rule_map(policy)
    equations = []
    records: list[tuple[str, str]] = []
    boundary_variables = frozenset(jaxpr.outvars)
    consumed_variables = frozenset(
        variable for equation in jaxpr.eqns for variable in equation.invars
    )
    for index, equation in enumerate(jaxpr.eqns):
        equation_path = f"{path}/{index}:{equation.primitive.name}"
        if equation.effects:
            raise ValueError(f"Effectful precision equation at {equation_path}.")
        primitive = equation.primitive.name
        if primitive not in _SUPPORTED_PURE:
            if policy.unsupported == "error":
                raise ValueError(f"Unsupported precision equation at {equation_path}.")
            equations.append(equation)
            records.append((equation_path, "recorded-pass-through"))
            continue
        params = dict(equation.params)
        nested_changed = False
        for parameter_name, parameter_value in tuple(params.items()):
            rewritten_parameter, nested_records, changed = _rewrite_nested(
                parameter_value,
                policy,
                f"{equation_path}/{parameter_name}",
            )
            if changed:
                params[parameter_name] = rewritten_parameter
                nested_changed = True
            records.extend(nested_records)
        if nested_changed:
            equation = equation.replace(params=params)
        family = primitive if primitive in rules else None
        if primitive not in (
            "dot_general",
            "conv_general_dilated",
            "reduce_sum",
            "reduce_prod",
        ):
            family = "elementwise" if "elementwise" in rules else None
        if family in ("dot_general", "conv_general_dilated"):
            rule = rules[family]
            output_dtype = jnp.dtype(equation.outvars[0].aval.dtype).name
            changes_output_dtype = rule.output_dtype != output_dtype
            terminal_boundary = all(
                variable in boundary_variables and variable not in consumed_variables
                for variable in equation.outvars
            )
            if changes_output_dtype and (
                not allow_boundary_dtype_change or not terminal_boundary
            ):
                raise ValueError(
                    f"Rewrite at {equation_path} would change an internal aval dtype; "
                    "only terminal boundary outputs may declare a new output dtype."
                )
            if not allow_boundary_dtype_change and rule.accumulator_dtype != output_dtype:
                raise ValueError(
                    f"Nested rewrite at {equation_path} would change its accumulator "
                    "dtype across a typed sub-jaxpr boundary."
                )
            params["precision"] = _JAX_PRECISION[rule.precision]
            params["preferred_element_type"] = jnp.dtype(rule.accumulator_dtype)
            equation = equation.replace(params=params)
            records.append((equation_path, f"rewritten:{family}"))
        elif family in ("reduce_sum", "reduce_prod"):
            rule = rules[family]
            output_dtype = jnp.dtype(equation.outvars[0].aval.dtype).name
            changes_output_dtype = rule.output_dtype != output_dtype
            terminal_boundary = all(
                variable in boundary_variables and variable not in consumed_variables
                for variable in equation.outvars
            )
            if changes_output_dtype and (
                not allow_boundary_dtype_change or not terminal_boundary
            ):
                raise ValueError(
                    f"Reduction rewrite at {equation_path} would change an internal "
                    "aval dtype; only terminal boundary outputs may declare a new "
                    "output dtype."
                )
            if not allow_boundary_dtype_change and rule.accumulator_dtype != output_dtype:
                raise ValueError(
                    f"Nested reduction rewrite at {equation_path} would change its "
                    "accumulator dtype across a typed sub-jaxpr boundary."
                )
            records.append((equation_path, f"rewritten:{family}"))
        elif family == "elementwise":
            rule = rules[family]
            for variable in equation.outvars:
                if hasattr(variable, "aval") and hasattr(variable.aval, "dtype"):
                    if jnp.dtype(variable.aval.dtype).name != rule.output_dtype:
                        raise ValueError(
                            f"Elementwise rewrite at {equation_path} would change avals."
                        )
            records.append((equation_path, "validated:elementwise"))
        else:
            records.append((equation_path, "unchanged-supported"))
        equations.append(equation)
    return jaxpr.replace(eqns=equations), tuple(records)


def prepare_precision_rewrite(
    function: Callable[..., Any],
    example_args: Sequence[Any],
    policy: PrecisionRewritePolicy,
    *,
    device: Any | None = None,
) -> PrecisionRewritePlan:
    """Prepare a finite pure-JAX precision rewrite bound to exact avals/device."""
    if not callable(function):
        raise TypeError("function must be callable.")
    if not isinstance(policy, PrecisionRewritePolicy):
        raise TypeError("policy must be a PrecisionRewritePolicy.")
    arguments = tuple(example_args)
    input_treedef = jax.tree.structure(arguments)
    output_shape = jax.eval_shape(function, *arguments)
    output_treedef = jax.tree.structure(output_shape)
    closed = jax.make_jaxpr(function)(*arguments)
    rewritten, records = _rewrite_jaxpr(
        closed.jaxpr, policy, "root", allow_boundary_dtype_change=True
    )
    rewritten_closed = closed.replace(jaxpr=rewritten)
    original_fingerprint = canonical_fingerprint(
        {"kind": "original-jaxpr", "jaxpr": str(closed)}
    )
    rewritten_fingerprint = canonical_fingerprint(
        {
            "kind": "rewritten-jaxpr",
            "jaxpr": str(rewritten_closed),
            "records": records,
        }
    )
    device_identity = _device_identity(device)
    input_signature = _signature(arguments)
    rule_map = _rule_map(policy)
    rewritten_output_dtypes = {
        variable: rule.output_dtype
        for equation in rewritten.eqns
        if (rule := rule_map.get(equation.primitive.name)) is not None
        and equation.primitive.name
        in ("dot_general", "conv_general_dilated", "reduce_sum", "reduce_prod")
        for variable in equation.outvars
    }
    output_signature = tuple(
        (
            tuple(int(size) for size in variable.aval.shape),
            rewritten_output_dtypes.get(variable, jnp.dtype(variable.aval.dtype).name),
        )
        for variable in rewritten.outvars
    )
    plan_id = canonical_fingerprint(
        {
            "kind": "precision-rewrite-plan",
            "inputs": input_signature,
            "outputs": output_signature,
            "original": original_fingerprint,
            "rewritten": rewritten_fingerprint,
            "records": records,
            "device": device_identity,
        }
    )
    return PrecisionRewritePlan(
        closed_jaxpr=rewritten_closed,
        rules=policy.rules,
        input_treedef=input_treedef,
        output_treedef=output_treedef,
        input_signature=input_signature,
        output_signature=output_signature,
        original_fingerprint=original_fingerprint,
        rewritten_fingerprint=rewritten_fingerprint,
        equation_records=records,
        provider="jax-public-jaxpr",
        device_identity=device_identity,
        plan_id=plan_id,
    )


def _execute_jaxpr(
    closed_jaxpr: Any,
    rules: tuple[PrecisionRewriteRule, ...],
    arguments: Sequence[Any],
    /,
) -> list[Any]:
    environment: dict[Any, Any] = {}
    rule_map = {rule.primitive: rule for rule in rules}

    def read(variable):
        if isinstance(variable, jax_core.Literal):
            return variable.val
        if type(variable) is not jax_core.Var:
            raise TypeError("Precision JAXPR inputs must be Literal or Var atoms.")
        return environment[variable]

    def write(variable, value):
        if type(variable) is jax_core.Var:
            environment[variable] = value
        elif not isinstance(variable, jax_core.Var):
            raise TypeError("Precision JAXPR outputs must be Var atoms.")

    for variable, value in zip(
        closed_jaxpr.jaxpr.constvars,
        closed_jaxpr.consts,
        strict=True,
    ):
        write(variable, value)
    for variable, value in zip(
        closed_jaxpr.jaxpr.invars,
        arguments,
        strict=True,
    ):
        write(variable, value)
    for equation in closed_jaxpr.jaxpr.eqns:
        inputs = [read(variable) for variable in equation.invars]
        rule = rule_map.get(equation.primitive.name)
        if rule is not None and equation.primitive.name in (
            "dot_general",
            "conv_general_dilated",
            "reduce_sum",
            "reduce_prod",
        ):
            inputs = [
                jnp.asarray(value, dtype=rule.accumulator_dtype) for value in inputs
            ]
        outputs = equation.primitive.bind(*inputs, **equation.params)
        if not equation.primitive.multiple_results:
            outputs = [outputs]
        if rule is not None and equation.primitive.name in (
            "dot_general",
            "conv_general_dilated",
            "reduce_sum",
            "reduce_prod",
        ):
            outputs = [jnp.asarray(value, dtype=rule.output_dtype) for value in outputs]
        for variable, value in zip(equation.outvars, outputs, strict=True):
            write(variable, value)
    return [read(variable) for variable in closed_jaxpr.jaxpr.outvars]


def execute_precision_rewrite(plan: PrecisionRewritePlan, *args: Any):
    """Execute one prepared rewrite after exact input/device validation."""
    if not isinstance(plan, PrecisionRewritePlan):
        raise TypeError("plan must be a PrecisionRewritePlan.")
    arguments = tuple(args)
    if jax.tree.structure(arguments) != plan.input_treedef:
        raise ValueError("Precision rewrite input PyTree changed.")
    if _signature(arguments) != plan.input_signature:
        raise ValueError("Precision rewrite input aval signature changed.")
    if _device_identity(None) != plan.device_identity:
        raise ValueError("Precision rewrite device identity changed.")
    flat_args = jax.tree.leaves(arguments)
    flat_output = _execute_jaxpr(plan.closed_jaxpr, plan.rules, flat_args)
    output = jax.tree.unflatten(plan.output_treedef, flat_output)
    if _signature(output) != plan.output_signature:
        raise RuntimeError("Precision rewrite output aval signature changed.")
    return output


def _block_until_ready(tree: Any, /) -> None:
    for leaf in jax.tree.leaves(tree):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _maximum_error(left: Any, right: Any, /) -> tuple[float, float]:
    absolute = 0.0
    relative = 0.0
    left_leaves = jax.tree.leaves(left)
    right_leaves = jax.tree.leaves(right)
    if len(left_leaves) != len(right_leaves):
        return float("inf"), float("inf")
    for lhs, rhs in zip(left_leaves, right_leaves, strict=True):
        lhs_ = np.asarray(lhs)
        rhs_ = np.asarray(rhs)
        if lhs_.shape != rhs_.shape:
            return float("inf"), float("inf")
        if not np.all(np.isfinite(lhs_)) or not np.all(np.isfinite(rhs_)):
            return float("inf"), float("inf")
        error = np.abs(lhs_ - rhs_)
        scale = np.maximum(np.abs(rhs_), np.finfo(np.result_type(rhs_, float)).tiny)
        relative_error = error / scale
        if not np.all(np.isfinite(error)) or not np.all(np.isfinite(relative_error)):
            return float("inf"), float("inf")
        absolute = max(absolute, float(np.max(error, initial=0.0)))
        relative = max(relative, float(np.max(relative_error, initial=0.0)))
    return absolute, relative


def prepare_precision_selection(
    function: Callable[..., Any],
    example_args: Sequence[Any],
    policy: PrecisionSelectionPolicy,
    *,
    reference: Callable[..., Any] | None = None,
    device: Any | None = None,
) -> PreparedPrecisionSelection:
    """Select only among a finite ordered candidate set for one workload."""
    if not isinstance(policy, PrecisionSelectionPolicy):
        raise TypeError("policy must be a PrecisionSelectionPolicy.")
    arguments = tuple(example_args)
    reference_function = function if reference is None else reference
    reference_value = reference_function(*arguments)
    accepted: list[
        tuple[PrecisionSelectionCandidate, PrecisionRewritePlan, float, float, float]
    ] = []
    records: list[tuple[str, str, float, float]] = []
    for candidate in policy.candidates:
        try:
            plan = prepare_precision_rewrite(
                function,
                arguments,
                candidate.policy,
                device=device,
            )
            value = execute_precision_rewrite(plan, *arguments)
            _block_until_ready(value)
            absolute, relative = _maximum_error(value, reference_value)
            if (
                absolute > policy.absolute_tolerance
                and relative > policy.relative_tolerance
            ):
                records.append(
                    (candidate.name, "numerical-rejection", absolute, relative)
                )
                continue
            compiled = jax.jit(lambda *items: execute_precision_rewrite(plan, *items))
            for _ in range(policy.warmups):
                _block_until_ready(compiled(*arguments))
            timings = []
            repeat_count = policy.repeats if policy.mode == "calibrated" else 1
            for _ in range(repeat_count):
                start = perf_counter()
                _block_until_ready(compiled(*arguments))
                timings.append(perf_counter() - start)
            latency = float(np.median(np.asarray(timings)))
            accepted.append((candidate, plan, absolute, relative, latency))
            records.append((candidate.name, "accepted", absolute, relative))
        except (TypeError, ValueError, RuntimeError):
            records.append(
                (candidate.name, "execution-rejection", float("inf"), float("inf"))
            )
    if not accepted:
        raise ValueError(
            "No precision selection candidate satisfied the workload contract."
        )
    selected = (
        min(accepted, key=lambda item: (item[4], policy.candidates.index(item[0])))
        if policy.mode == "calibrated"
        else accepted[0]
    )
    workload = canonical_fingerprint(
        {"kind": "precision-selection-workload", "signature": _signature(arguments)}
    )
    fingerprint_records = tuple(
        (
            name,
            disposition,
            absolute if np.isfinite(absolute) else "inf",
            relative if np.isfinite(relative) else "inf",
        )
        for name, disposition, absolute, relative in records
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "precision-selection-evidence",
            "selected": selected[0].name,
            "records": fingerprint_records,
            "device": selected[1].device_identity,
            "workload": workload,
        }
    )
    evidence = PrecisionSelectionEvidence(
        selected[0].name,
        tuple(records),
        selected[1].device_identity,
        workload,
        "selected only among the declared candidates for this traced workload",
        evidence_id,
    )
    return PreparedPrecisionSelection(selected[1], evidence)


__all__ = [
    "PreparedPrecisionSelection",
    "PrecisionRewritePlan",
    "PrecisionRewritePolicy",
    "PrecisionRewriteRule",
    "PrecisionSelectionCandidate",
    "PrecisionSelectionEvidence",
    "PrecisionSelectionPolicy",
    "execute_precision_rewrite",
    "prepare_precision_rewrite",
    "prepare_precision_selection",
]
