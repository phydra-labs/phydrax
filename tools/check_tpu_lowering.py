# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hardware-free TPU lowering preflight for bounded float32 scientific kernels.

Each selected scenario builds its kernels with the native Phydrax owners and
runs two stages over the same jitted functions. ``cpu-reference`` executes them
on CPU and checks independent references and known results. ``tpu-export``
applies ``jax.export.export(jax.jit(f), platforms=("tpu",))`` to abstract
inputs; JVP, VJP, second, and mixed derivatives are transformed before export.

The process selects CPU only. It never requests a TPU client, compiles for or
executes on TPU, or disables an export safety check. An exported StableHLO
module is not XLA:TPU compilation or execution evidence, and every report keeps
``hardware_qualified`` false. Hardware qualification belongs to
``tools/qualify_accelerator_wheels.py`` on an actual TPU worker.

Exit status: 0 when every CPU check passes and every declared target matches
its expected outcome (``successful``); a declared export refusal still leaves
``fully_exported`` false. 1 for a failed check, execution error, unconstructed
target, unexpected export outcome, or a strongly typed float64 value in an
exported float32 route; 2 when the CPU-only selection cannot be established or
arguments are invalid.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import export as jax_export
from jax.core import ShapedArray
from jax.extend.core import subjaxprs

import phydrax as phx
from phydrax._publication import publish_bytes
from phydrax.backends import frontend_index_patch_identity
from tools._accelerator_process import JSON_LIMIT, TEXT_LIMIT
from tools.check_accelerator_extras import sha256


KIND = "phydrax-tpu-lowering-preflight"
TARGET_PLATFORMS = ("tpu",)
ExportOutcome: TypeAlias = Literal[
    "exported",
    "host-callback",
    "unstable-custom-call",
    "lowering-unsupported",
    "lowering-error",
]
CALLBACK_PRIMITIVES = frozenset(
    {"buffer_callback", "debug_callback", "io_callback", "pure_callback"}
)
UNSTABLE_CUSTOM_CALL = "custom calls whose targets have no compatibility guarantees"
WIDE_FLOATS = frozenset({np.dtype(np.float64), np.dtype(np.complex128)})
MODULE_FLOAT64_TYPE = re.compile(r"(?:<|x)f64>")
CUSTOM_CALL_TARGET = re.compile(
    r"custom_call\s*@([\w.$]+)|call_target_name\s*=\s*\"([^\"]+)\""
)
Record: TypeAlias = dict[str, Any]


def _float(value: Any) -> float | str:
    number = float(value)
    return number if math.isfinite(number) else repr(number)


def _error(error: BaseException) -> Record:
    return {
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
        "message": str(error)[:TEXT_LIMIT],
    }


def _flat(arrays: Sequence[Any]) -> np.ndarray:
    return np.concatenate([np.ravel(np.asarray(array, np.float64)) for array in arrays])


def _record(name: str, passed: bool, **evidence: Any) -> Record:
    return {"name": name, "passed": bool(passed), **evidence}


def close_check(
    name: str, actual: Any, expected: Any, *, rtol: float, atol: float
) -> Record:
    """Elementwise ``|actual - expected| <= atol + rtol |expected|`` with finite values."""
    actual_ = np.asarray(actual, np.float64)
    expected_ = np.asarray(expected, np.float64)
    if actual_.shape != expected_.shape:
        return _record(
            name, False, reason=f"shape {actual_.shape} differs from {expected_.shape}"
        )
    error = np.abs(actual_ - expected_)
    passed = np.all(np.isfinite(actual_)) and np.all(
        error <= atol + rtol * np.abs(expected_)
    )
    return _record(
        name,
        bool(passed),
        max_abs_error=_float(np.max(error, initial=0.0)),
        rtol=rtol,
        atol=atol,
    )


def relative_check(name: str, actual: Any, expected: Any, *, tolerance: float) -> Record:
    """Euclidean ``||actual - expected|| / ||expected|| <= tolerance``."""
    actual_ = np.asarray(actual, np.float64)
    expected_ = np.asarray(expected, np.float64)
    if actual_.shape != expected_.shape:
        return _record(
            name, False, reason=f"shape {actual_.shape} differs from {expected_.shape}"
        )
    scale = max(float(np.linalg.norm(expected_)), np.finfo(np.float64).tiny)
    error = float(np.linalg.norm(actual_ - expected_)) / scale
    return _record(
        name,
        math.isfinite(error) and error <= tolerance,
        relative_error=_float(error),
        tolerance=tolerance,
    )


def _values(array: np.ndarray) -> Any:
    if np.issubdtype(array.dtype, np.floating):
        return [_float(value) for value in array.ravel()]
    return array.tolist()


def equal_check(name: str, actual: Any, expected: Any) -> Record:
    actual_ = np.asarray(actual)
    expected_ = np.asarray(expected)
    return _record(
        name,
        actual_.shape == expected_.shape and bool(np.array_equal(actual_, expected_)),
        shape=list(actual_.shape),
        actual=_values(actual_),
        expected=_values(expected_),
    )


def _leaves(prefix: str, tree: Any) -> dict[str, Any]:
    return {
        f"{prefix}/{index:03d}": leaf for index, leaf in enumerate(jax.tree.leaves(tree))
    }


def _collect(outputs: Mapping[str, np.ndarray], prefix: str) -> list[np.ndarray]:
    return [outputs[key] for key in sorted(outputs) if key.startswith(f"{prefix}/")]


def _program_facts(jaxpr: Any) -> tuple[set[str], set[str]]:
    """Primitive names, and primitives producing strongly typed 64-bit floating values.

    Weakly typed Python scalars (for example ``jnp.where(mask, x, 0.0)`` under
    x64) are normalized to the strong operand dtype and are not float64 work.
    """
    names: set[str] = set()
    wide: set[str] = set()
    pending = [jaxpr]
    while pending:
        current = pending.pop()
        for equation in current.eqns:
            names.add(equation.primitive.name)
            if any(
                isinstance(variable.aval, ShapedArray)
                and variable.aval.dtype in WIDE_FLOATS
                and not variable.aval.weak_type
                for variable in equation.outvars
            ):
                wide.add(equation.primitive.name)
        pending.extend(subjaxprs(current))
    return names, wide


def _refusal(error: BaseException, callbacks: Sequence[str]) -> ExportOutcome:
    """Classify an export refusal; none of these is a TPU runtime observation."""
    if callbacks:
        return "host-callback"
    if isinstance(error, ValueError) and UNSTABLE_CUSTOM_CALL in str(error):
        return "unstable-custom-call"
    if isinstance(error, NotImplementedError):
        return "lowering-unsupported"
    return "lowering-error"


class _Target:
    """One transformed function over array leaves, executed and exported unchanged."""

    def __init__(
        self, name: str, function: Callable[..., Mapping[str, Any]], arguments: tuple
    ) -> None:
        dynamic, static = eqx.partition(arguments, eqx.is_array)
        leaves, treedef = jax.tree.flatten(dynamic)

        def flat(*values: Any) -> dict[str, jax.Array]:
            rebuilt = eqx.combine(jax.tree.unflatten(treedef, values), static)
            return {key: jnp.asarray(value) for key, value in function(*rebuilt).items()}

        flat.__name__ = flat.__qualname__ = name.replace("-", "_")
        self.treedef = treedef
        self.static = static
        self.jitted = jax.jit(flat)
        self.specs = tuple(
            jax.ShapeDtypeStruct(aval.shape, aval.dtype, weak_type=aval.weak_type)
            for aval in map(jax.typeof, leaves)
        )
        self.outputs: dict[str, np.ndarray] | None = None

    def execute(self, arguments: tuple) -> dict[str, np.ndarray]:
        dynamic, static = eqx.partition(arguments, eqx.is_array)
        leaves, treedef = jax.tree.flatten(dynamic)
        if treedef != self.treedef or eqx.tree_equal(static, self.static) is not True:
            raise ValueError("Re-execution must keep the registered argument structure.")
        if tuple(jax.typeof(leaf).dtype for leaf in leaves) != tuple(
            spec.dtype for spec in self.specs
        ):
            raise ValueError("Re-execution must keep the registered argument dtypes.")
        outputs = jax.device_get(self.jitted(*leaves))
        jax.effects_barrier()
        return {key: np.asarray(value) for key, value in outputs.items()}

    def export(self, expectation: ExportOutcome) -> Record:
        record: Record = {
            "expectation": expectation,
            "abstract_inputs": [f"{spec.dtype}{list(spec.shape)}" for spec in self.specs],
        }
        try:
            jaxpr = self.jitted.trace(*self.specs).jaxpr
        except Exception as error:  # Recorded as this target's refusal evidence.
            return {
                **record,
                "outcome": "lowering-error",
                "phase": "trace",
                "error": _error(error),
                "matched": expectation == "lowering-error",
            }
        primitives, wide = _program_facts(jaxpr)
        callbacks = sorted(primitives & CALLBACK_PRIMITIVES)
        record["callback_primitives"] = callbacks
        record["strong_float64_primitives"] = sorted(wide)
        try:
            exported = jax_export.export(self.jitted, platforms=TARGET_PLATFORMS)(
                *self.specs
            )
        except Exception as error:  # Recorded as this target's refusal evidence.
            outcome = _refusal(error, callbacks)
            return {
                **record,
                "outcome": outcome,
                "phase": "export",
                "error": _error(error),
                "matched": expectation == outcome,
            }
        evidence = _exported_evidence(exported)
        return {
            **record,
            **evidence,
            "matched": expectation == "exported"
            and not wide
            and not evidence["disabled_safety_checks"],
        }


def _exported_evidence(exported: Any) -> Record:
    module = exported.mlir_module()
    artifact = bytes(exported.mlir_module_serialized)
    return {
        "outcome": "exported",
        "platforms": list(exported.platforms),
        "in_avals": [str(aval) for aval in exported.in_avals],
        "out_avals": [str(aval) for aval in exported.out_avals],
        "nr_devices": exported.nr_devices,
        "calling_convention_version": exported.calling_convention_version,
        "ordered_effects": [str(effect) for effect in exported.ordered_effects],
        "unordered_effects": [str(effect) for effect in exported.unordered_effects],
        "disabled_safety_checks": [
            str(check) for check in exported.disabled_safety_checks
        ],
        "custom_call_targets": sorted(
            {first or second for first, second in CUSTOM_CALL_TARGET.findall(module)}
        ),
        # Raw text evidence; includes weak scalar literals converted to float32.
        "module_f64_type_occurrences": len(MODULE_FLOAT64_TYPE.findall(module)),
        "stablehlo_bytes": len(artifact),
        "stablehlo_sha256": hashlib.sha256(artifact).hexdigest(),
    }


class Scenario:
    """Registers declared targets while their CPU reference stage executes them."""

    def __init__(self, spec: ScenarioSpec) -> None:
        self.spec = spec
        self.targets: dict[str, _Target] = {}
        self.checks: list[Record] = []

    def run(
        self, name: str, function: Callable[..., Mapping[str, Any]], *arguments: Any
    ) -> dict[str, np.ndarray]:
        if name not in self.spec.exports:
            raise ValueError(f"{name!r} is not a declared target of {self.spec.name!r}.")
        if name in self.targets:
            raise ValueError(f"Target {name!r} is already registered.")
        target = _Target(name, function, arguments)
        self.targets[name] = target
        target.outputs = target.execute(arguments)
        return target.outputs

    def rerun(self, name: str, *arguments: Any) -> dict[str, np.ndarray]:
        return self.targets[name].execute(arguments)

    def check(self, record: Record) -> None:
        self.checks.append(record)


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    name: str
    owner: str
    exports: Mapping[str, ExportOutcome]
    build: Callable[[Scenario], None]


def _output_checks(targets: Mapping[str, _Target]) -> list[Record]:
    floating = [
        (f"{name}/{key}", value)
        for name, target in targets.items()
        if target.outputs is not None
        for key, value in target.outputs.items()
        if np.issubdtype(value.dtype, np.floating)
    ]
    wrong = sorted(
        f"{label}:{value.dtype}" for label, value in floating if value.dtype != np.float32
    )
    nonfinite = sorted(
        label for label, value in floating if not np.all(np.isfinite(value))
    )
    return [
        _record(
            "float32-floating-outputs",
            not wrong,
            floating_outputs=len(floating),
            offending=wrong,
        ),
        _record(
            "finite-floating-outputs",
            not nonfinite,
            floating_outputs=len(floating),
            offending=nonfinite,
        ),
    ]


def _export_counts(targets: Mapping[str, Record]) -> dict[str, int]:
    values = tuple(targets.values())
    return {
        "declared": len(values),
        "required_exports": sum(item["expectation"] == "exported" for item in values),
        "expected_refusals": sum(item["expectation"] != "exported" for item in values),
        "exported": sum(item["outcome"] == "exported" for item in values),
        "refused": sum(
            item["outcome"] not in ("exported", "not-attempted") for item in values
        ),
        "matched_exports": sum(
            item["matched"] and item["outcome"] == "exported" for item in values
        ),
        "matched_refusals": sum(
            item["matched"] and item["outcome"] != "exported" for item in values
        ),
        "not_attempted": sum(item["outcome"] == "not-attempted" for item in values),
        "mismatched": sum(not item["matched"] for item in values),
    }


def run_scenario(spec: ScenarioSpec) -> Record:
    """Run the CPU reference stage, then export every declared target for TPU."""
    scenario = Scenario(spec)
    error = None
    try:
        spec.build(scenario)
    except Exception as caught:  # The stage records the failure; export still runs.
        error = _error(caught)
    checks = [*scenario.checks, *_output_checks(scenario.targets)]
    cpu_passed = error is None and all(check["passed"] for check in checks)
    targets = {
        name: (
            scenario.targets[name].export(expectation)
            if name in scenario.targets
            else {
                "expectation": expectation,
                "outcome": "not-attempted",
                "matched": False,
                "reason": "The CPU reference stage ended before this target was built.",
            }
        )
        for name, expectation in spec.exports.items()
    }
    counts = _export_counts(targets)
    return {
        "owner": spec.owner,
        "cpu_reference": {"passed": cpu_passed, "checks": checks, "error": error},
        "tpu_export": {"targets": targets, "counts": counts},
        # Every declared target lowered for TPU; a declared refusal keeps this false.
        "fully_exported": counts["exported"] == counts["declared"],
        "successful": cpu_passed and counts["mismatched"] == 0,
    }


def _distribution(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> Record:
    snapshot = phx.config.snapshot()
    return {
        "python": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
        },
        "host": {"system": platform.system(), "machine": platform.machine()},
        "distributions": {
            name: _distribution(name) for name in ("jax", "jaxlib", "equinox", "phydrax")
        },
        "phydrax_source": str(Path(phx.__file__).resolve().parent),
        "tool_sha256": sha256(Path(__file__).resolve()),
        "configuration": {
            "platforms": None if snapshot.platforms is None else list(snapshot.platforms),
            "enable_x64": snapshot.enable_x64,
            "matmul_precision": snapshot.matmul_precision,
            "dtype_promotion": snapshot.dtype_promotion,
        },
        "observed_device_platforms": sorted(
            {device.platform for device in jax.devices()}
        ),
        "frontend_index_patch": frontend_index_patch_identity(),
    }


def _report(environment: Record, refusal: str | None) -> Record:
    return {
        "kind": KIND,
        "target_platforms": list(TARGET_PLATFORMS),
        "tpu_compiled": False,
        "tpu_executed": False,
        "hardware_qualified": False,
        "environment": environment,
        "refusal": refusal,
        "scenarios": {},
        "counts": {},
        "fully_exported": False,
        "successful": False,
    }


def preflight(specs: Sequence[ScenarioSpec]) -> Record:
    """Run the selected scenarios in the current process, which must expose only CPU."""
    if not specs:
        raise ValueError("Select at least one scenario.")
    environment = _environment()
    observed = environment["observed_device_platforms"]
    if observed != ["cpu"]:
        return _report(
            environment, f"CPU references require only CPU devices; observed {observed}."
        )
    report = _report(environment, None)
    scenarios = {spec.name: run_scenario(spec) for spec in specs}
    totals = [scenario["tpu_export"]["counts"] for scenario in scenarios.values()]
    report["scenarios"] = scenarios
    report["counts"] = {
        "scenarios": len(scenarios),
        "successful_scenarios": sum(item["successful"] for item in scenarios.values()),
        "fully_exported_scenarios": sum(
            item["fully_exported"] for item in scenarios.values()
        ),
        **{key: sum(count[key] for count in totals) for key in totals[0]},
    }
    report["fully_exported"] = all(item["fully_exported"] for item in scenarios.values())
    report["successful"] = all(item["successful"] for item in scenarios.values())
    return report


# Scenarios. Each declares every export target and its expected outcome up front;
# a known export limitation is a declared outcome, never a substitute for a
# required export.


def _pinn(scenario: Scenario) -> None:
    from phydrax.axes import AxisArray
    from phydrax.operators.differential import laplacian
    from phydrax.optim import adam

    box = phx.domain.HyperRectangle(
        np.zeros(2, np.float32), np.ones(2, np.float32), dtype=jnp.float32
    )
    model = phx.nn.models.MLP(
        in_size=2,
        out_size="scalar",
        width_size=8,
        depth=2,
        activation=jnp.tanh,
        dtype=jnp.float32,
        key=jax.random.key(7),
    )
    points = np.asarray([[0.2, 0.3], [0.4, 0.7], [0.7, 0.5], [0.8, 0.9]], np.float32)
    forcing = (
        2 * np.pi**2 * np.sin(np.pi * points[:, 0]) * np.sin(np.pi * points[:, 1])
    ).astype(np.float32)
    parameters, structure = eqx.partition(model, eqx.is_inexact_array)
    treedef = jax.tree.structure(parameters)
    optimizer = adam(1e-3)

    def network(candidate: Any) -> Any:
        return eqx.combine(candidate, structure)

    def curvature(candidate: Any) -> jax.Array:
        field = box.Model("x")(network(candidate))
        coordinates = AxisArray(jnp.asarray(points), dims=("point", None))
        return laplacian(field, var="x")({"x": coordinates}).data

    def objective(candidate: Any) -> jax.Array:
        return jnp.mean((curvature(candidate) + forcing) ** 2)

    def laplacian_reference(candidate: Any) -> dict[str, Any]:
        def trace(point: jax.Array) -> jax.Array:
            return jnp.trace(jax.hessian(network(candidate))(point))

        return {
            "laplacian": curvature(candidate),
            "hessian_trace": jax.vmap(trace)(jnp.asarray(points)),
        }

    def value_and_gradient(candidate: Any) -> dict[str, Any]:
        value, gradient = jax.value_and_grad(objective)(candidate)
        return {"loss": value, **_leaves("gradient", gradient)}

    def directional(candidate: Any, direction: Any) -> dict[str, Any]:
        value, tangent = jax.jvp(objective, (candidate,), (direction,))
        return {"loss": value, "directional_derivative": tangent}

    def curvature_product(candidate: Any, direction: Any) -> dict[str, Any]:
        _, product = jax.jvp(jax.grad(objective), (candidate,), (direction,))
        return _leaves("hvp", product)

    def adam_step(candidate: Any, state: Any) -> dict[str, Any]:
        value, gradient = jax.value_and_grad(objective)(candidate)
        updates, next_state = optimizer.update(gradient, state, candidate)
        return {
            "loss": value,
            **_leaves("parameters", eqx.apply_updates(candidate, updates)),
            **_leaves("state", next_state),
        }

    def shifted(direction: Any, step: float) -> Any:
        return jax.tree.map(
            lambda value, delta: value + np.float32(step) * delta, parameters, direction
        )

    reference = scenario.run("laplacian-reference", laplacian_reference, parameters)
    scenario.check(
        close_check(
            "phydrax-laplacian-equals-hessian-trace",
            reference["laplacian"],
            reference["hessian_trace"],
            rtol=1e-4,
            atol=1e-4,
        )
    )
    initial = scenario.run("objective-value-and-gradient", value_and_gradient, parameters)
    gradient = _collect(initial, "gradient")
    gradient_norm = float(np.linalg.norm(_flat(gradient)))
    unit = jax.tree.unflatten(
        treedef, [(leaf / gradient_norm).astype(np.float32) for leaf in gradient]
    )
    forward = scenario.run("objective-jvp", directional, parameters, unit)
    scenario.check(
        close_check(
            "jvp-along-gradient-equals-gradient-norm",
            forward["directional_derivative"],
            gradient_norm,
            rtol=1e-4,
            atol=1e-6,
        )
    )
    plus = scenario.rerun("objective-jvp", shifted(unit, 0.01), unit)["loss"]
    minus = scenario.rerun("objective-jvp", shifted(unit, -0.01), unit)["loss"]
    scenario.check(
        close_check(
            "jvp-matches-central-difference",
            forward["directional_derivative"],
            (np.float64(plus) - np.float64(minus)) / 0.02,
            rtol=0.03,
            atol=0.02,
        )
    )
    _pinn_curvature_checks(
        scenario, parameters, treedef, unit, curvature_product, shifted
    )
    _pinn_training_checks(scenario, parameters, optimizer, adam_step, initial)


def _pinn_curvature_checks(
    scenario: Scenario,
    parameters: Any,
    treedef: Any,
    unit: Any,
    curvature_product: Callable[..., dict[str, Any]],
    shifted: Callable[[Any, float], Any],
) -> None:
    rng = np.random.default_rng(17)
    other = [rng.standard_normal(np.shape(leaf)) for leaf in jax.tree.leaves(parameters)]
    other_norm = np.linalg.norm(_flat(other))
    other_tree = jax.tree.unflatten(
        treedef, [(leaf / other_norm).astype(np.float32) for leaf in other]
    )
    along_unit = _flat(
        _collect(
            scenario.run("objective-hvp", curvature_product, parameters, unit), "hvp"
        )
    )
    along_other = _flat(
        _collect(scenario.rerun("objective-hvp", parameters, other_tree), "hvp")
    )
    scenario.check(
        close_check(
            "hessian-vector-products-are-symmetric",
            np.dot(_flat(jax.tree.leaves(other_tree)), along_unit),
            np.dot(_flat(jax.tree.leaves(unit)), along_other),
            rtol=2e-3,
            atol=1e-4,
        )
    )
    plus = _collect(
        scenario.rerun("objective-value-and-gradient", shifted(unit, 0.01)), "gradient"
    )
    minus = _collect(
        scenario.rerun("objective-value-and-gradient", shifted(unit, -0.01)), "gradient"
    )
    scenario.check(
        relative_check(
            "hvp-matches-gradient-central-difference",
            along_unit,
            (_flat(plus) - _flat(minus)) / 0.02,
            tolerance=0.05,
        )
    )


def _pinn_training_checks(
    scenario: Scenario,
    parameters: Any,
    optimizer: Any,
    adam_step: Callable[..., dict[str, Any]],
    initial: Mapping[str, np.ndarray],
) -> None:
    parameter_tree = jax.tree.structure(parameters)
    state = optimizer.init(parameters)
    state_tree = jax.tree.structure(state)
    step = scenario.run("adam-step", adam_step, parameters, state)
    gradient = _flat(_collect(initial, "gradient"))
    # Adam's bias-corrected first step is -lr * g / (|g| + eps) elementwise.
    scenario.check(
        close_check(
            "first-adam-step-matches-closed-form",
            _flat(_collect(step, "parameters")) - _flat(jax.tree.leaves(parameters)),
            -1e-3 * gradient / (np.abs(gradient) + 1e-8),
            rtol=1e-3,
            atol=5e-7,
        )
    )
    losses = [float(step["loss"])]
    for _ in range(4):
        current = jax.tree.unflatten(parameter_tree, _collect(step, "parameters"))
        state = jax.tree.unflatten(state_tree, _collect(step, "state"))
        step = scenario.rerun("adam-step", current, state)
        losses.append(float(step["loss"]))
    final = scenario.rerun(
        "objective-value-and-gradient",
        jax.tree.unflatten(parameter_tree, _collect(step, "parameters")),
    )
    losses.append(float(final["loss"]))
    scenario.check(
        _record(
            "adam-steps-decrease-objective",
            losses[-1] < losses[0],
            losses=[_float(value) for value in losses],
        )
    )


def _attention(scenario: Scenario) -> None:
    from phydrax.nn.layers import AttentionExecution, MeasureAwareAttention

    def attention(
        execution: AttentionExecution, block_size: int | None
    ) -> MeasureAwareAttention:
        return MeasureAwareAttention(
            source_channels=3,
            query_channels=2,
            out_channels=2,
            num_heads=2,
            head_dim=4,
            execution=execution,
            block_size=block_size,
            dtype=jnp.float32,
            key=jax.random.key(11),
        )

    # float32 "auto" without a block size is the fused XLA route.
    fused = attention("auto", None)
    blockwise = attention("blockwise", 2)
    dense = attention("dense", None)
    rng = np.random.default_rng(3)
    source = rng.standard_normal((2, 5, 3)).astype(np.float32)
    query = rng.standard_normal((2, 4, 2)).astype(np.float32)
    weights = np.asarray(
        [[0.5, 0.0, 1.25, 0.75, 2.0], [1.0, 0.25, 0.5, 1.5, 0.8]], np.float32
    )
    direction = rng.standard_normal(query.shape)
    direction = (direction / np.linalg.norm(direction)).astype(np.float32)
    parameter_dtypes = sorted(
        {
            str(leaf.dtype)
            for module in (fused, blockwise, dense)
            for leaf in jax.tree.leaves(eqx.filter(module, eqx.is_inexact_array))
        }
    )
    scenario.check(
        _record(
            "float32-parameters", parameter_dtypes == ["float32"], dtypes=parameter_dtypes
        )
    )
    expected = _attention_reference(fused, source, query, weights)
    for name, module in (
        ("fused-primal", fused),
        ("blockwise-primal", blockwise),
        ("dense-primal", dense),
    ):
        output = scenario.run(name, _attend, module, source, query, weights)["output"]
        scenario.check(
            close_check(
                f"{name}-matches-float64-reference",
                output,
                expected,
                rtol=1e-4,
                atol=1e-5,
            )
        )
        # Source 1 of batch 0 has zero measure, so its payload has no influence.
        moved = source.copy()
        moved[0, 1] += 3.0
        scenario.check(
            equal_check(
                f"{name}-ignores-zero-measure-source",
                scenario.rerun(name, module, moved, query, weights)["output"],
                output,
            )
        )
    _attention_derivative_checks(
        scenario, (fused, blockwise), source, query, weights, direction
    )


def _attention_reference(
    module: Any, source: np.ndarray, query: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Measure-weighted softmax core in float64, independent of the kernel routes."""
    projected = np.asarray(module.project_query(jnp.asarray(query)), np.float64)
    keys, values = (
        np.asarray(array, np.float64)
        for array in module.project_source(jnp.asarray(source))
    )
    measure = weights.astype(np.float64)
    positive = measure > 0.0
    logits = np.einsum("bqhd,bshd->bhqs", projected, keys) / math.sqrt(keys.shape[-1])
    logits = np.where(
        positive[:, None, None, :],
        logits + np.log(np.where(positive, measure, 1.0))[:, None, None, :],
        -np.inf,
    )
    logits -= np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    attended = np.einsum("bhqs,bshd->bqhd", probabilities, values)
    flattened = attended.reshape(attended.shape[:2] + (-1,))
    return np.asarray(module.output(jnp.asarray(flattened, jnp.float32)), np.float64)


def _attend(module: Any, source: Any, query: Any, weights: Any) -> dict[str, Any]:
    return {"output": module(source, query, weights)}


def _attention_loss(module: Any, source: Any, query: Any, weights: Any) -> jax.Array:
    return 0.5 * jnp.sum(module(source, query, weights) ** 2)


def _attention_gradients(
    module: Any, source: Any, query: Any, weights: Any
) -> dict[str, Any]:
    parameters, rest = eqx.partition(module, eqx.is_inexact_array)

    def loss(candidate: Any, source_: Any, query_: Any) -> jax.Array:
        return _attention_loss(eqx.combine(candidate, rest), source_, query_, weights)

    value, (parameter_gradient, source_gradient, query_gradient) = jax.value_and_grad(
        loss, argnums=(0, 1, 2)
    )(parameters, source, query)
    return {
        "loss": value,
        "source_gradient": source_gradient,
        "query_gradient": query_gradient,
        **_leaves("parameter-gradient", parameter_gradient),
    }


def _attention_query_tangent(
    module: Any, source: Any, query: Any, weights: Any, direction: Any
) -> dict[str, Any]:
    output, tangent = jax.jvp(
        lambda value: module(source, value, weights), (query,), (direction,)
    )
    return {"output": output, "tangent": tangent}


def _attention_query_hvp(
    module: Any, source: Any, query: Any, weights: Any, direction: Any
) -> dict[str, Any]:
    gradient = jax.grad(lambda value: _attention_loss(module, source, value, weights))
    return {"hvp": jax.jvp(gradient, (query,), (direction,))[1]}


def _attention_mixed(
    module: Any, source: Any, query: Any, weights: Any, direction: Any
) -> dict[str, Any]:
    parameters, rest = eqx.partition(module, eqx.is_inexact_array)

    def directional(candidate: Any) -> jax.Array:
        return jax.jvp(
            lambda value: _attention_loss(
                eqx.combine(candidate, rest), source, value, weights
            ),
            (query,),
            (direction,),
        )[1]

    return _leaves("mixed", jax.grad(directional)(parameters))


def _attention_derivative_checks(
    scenario: Scenario,
    modules: tuple[Any, Any],
    source: np.ndarray,
    query: np.ndarray,
    weights: np.ndarray,
    direction: np.ndarray,
) -> None:
    fused, blockwise = modules
    step = np.float32(0.01)
    plus, minus = query + step * direction, query - step * direction
    arguments = (source, query, weights)
    tangent = scenario.run(
        "fused-query-jvp", _attention_query_tangent, fused, *arguments, direction
    )["tangent"]
    scenario.check(
        relative_check(
            "query-jvp-matches-central-difference",
            tangent,
            (
                scenario.rerun("fused-primal", fused, source, plus, weights)["output"]
                - scenario.rerun("fused-primal", fused, source, minus, weights)["output"]
            )
            / (2 * step),
            tolerance=1e-2,
        )
    )
    gradients = scenario.run("fused-gradients", _attention_gradients, fused, *arguments)
    gradients_plus = scenario.rerun("fused-gradients", fused, source, plus, weights)
    gradients_minus = scenario.rerun("fused-gradients", fused, source, minus, weights)
    hvp = scenario.run(
        "fused-query-hvp", _attention_query_hvp, fused, *arguments, direction
    )["hvp"]
    scenario.check(
        relative_check(
            "query-hvp-matches-gradient-central-difference",
            hvp,
            (gradients_plus["query_gradient"] - gradients_minus["query_gradient"])
            / (2 * step),
            tolerance=2e-2,
        )
    )
    mixed = scenario.run(
        "fused-parameter-query-mixed", _attention_mixed, fused, *arguments, direction
    )
    scenario.check(
        relative_check(
            "parameter-query-mixed-matches-central-difference",
            _flat(_collect(mixed, "mixed")),
            (
                _flat(_collect(gradients_plus, "parameter-gradient"))
                - _flat(_collect(gradients_minus, "parameter-gradient"))
            )
            / (2 * step),
            tolerance=2e-2,
        )
    )
    blockwise_gradients = scenario.run(
        "blockwise-gradients", _attention_gradients, blockwise, *arguments
    )
    scenario.check(
        close_check(
            "blockwise-gradients-match-fused",
            _flat([blockwise_gradients[key] for key in sorted(blockwise_gradients)]),
            _flat([gradients[key] for key in sorted(gradients)]),
            rtol=1e-4,
            atol=1e-5,
        )
    )
    blockwise_hvp = scenario.run(
        "blockwise-query-hvp", _attention_query_hvp, blockwise, *arguments, direction
    )["hvp"]
    scenario.check(
        close_check(
            "blockwise-hvp-matches-fused", blockwise_hvp, hvp, rtol=1e-3, atol=1e-5
        )
    )


def _contraction(scenario: Scenario) -> None:
    tn = phx.tensor_network
    leg = tn.ContractionLeg
    structure = tn.ContractionStructure(
        (
            tn.ContractionOperand("a", (leg("i", 3), leg("j", 4))),
            tn.ContractionOperand("b", (leg("j", 4), leg("k", 5))),
            tn.ContractionOperand("c", (leg("k", 5), leg("l", 2))),
        ),
        ("i", "l"),
    )
    plan = tn.plan_contraction(structure, dtype="float32", optimizer="optimal")
    scenario.check(equal_check("plan-storage-dtype", plan.dtype, "float32"))
    rng = np.random.default_rng(5)
    operands = tuple(
        rng.standard_normal(shape).astype(np.float32)
        for shape in ((3, 4), (4, 5), (5, 2))
    )
    tangents = tuple(
        rng.standard_normal(item.shape).astype(np.float32) for item in operands
    )
    cotangent = rng.standard_normal((3, 2)).astype(np.float32)
    prepared = tn.prepare_contraction(plan, operands)

    def with_operands(item: Any, values: Any) -> Any:
        return eqx.tree_at(lambda current: current.operands, item, values)

    def execute(item: Any) -> dict[str, Any]:
        result = tn.execute_contraction(item)
        return {"value": result.value, "finite": result.evidence.finite}

    def reverse(item: Any, seed: Any) -> dict[str, Any]:
        result = tn.execute_contraction_reverse(item, cotangent=seed)
        return {"value": result.value, **_leaves("gradient", result.operand_cotangents)}

    def forward(item: Any, directions: Any) -> dict[str, Any]:
        value, tangent = jax.jvp(
            lambda values: tn.execute_contraction(with_operands(item, values)).value,
            (item.operands,),
            (directions,),
        )
        return {"value": value, "tangent": tangent}

    def reverse_forward(item: Any, seed: Any, directions: Any) -> dict[str, Any]:
        _, tangent = jax.jvp(
            lambda values: (
                tn.execute_contraction_reverse(
                    with_operands(item, values), cotangent=seed
                ).operand_cotangents
            ),
            (item.operands,),
            (directions,),
        )
        return _leaves("gradient-tangent", tangent)

    a, b, c = (item.astype(np.float64) for item in operands)
    ta, tb, tc = (item.astype(np.float64) for item in tangents)
    g = cotangent.astype(np.float64)
    executed = scenario.run("execute", execute, prepared)
    scenario.check(
        close_check(
            "value-matches-float64-product",
            executed["value"],
            a @ b @ c,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    scenario.check(equal_check("evidence-finite", executed["finite"], True))
    reversed_ = scenario.run("reverse", reverse, prepared, cotangent)
    scenario.check(
        close_check(
            "reverse-value-matches-forward",
            reversed_["value"],
            executed["value"],
            rtol=1e-6,
            atol=1e-6,
        )
    )
    scenario.check(
        close_check(
            "reverse-gradients-match-closed-form",
            _flat(_collect(reversed_, "gradient")),
            _flat((g @ (b @ c).T, a.T @ g @ c.T, (a @ b).T @ g)),
            rtol=1e-5,
            atol=1e-5,
        )
    )
    tangent = scenario.run("jvp", forward, prepared, tangents)["tangent"]
    scenario.check(
        close_check(
            "jvp-matches-product-rule",
            tangent,
            ta @ b @ c + a @ tb @ c + a @ b @ tc,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    second = scenario.run("reverse-jvp", reverse_forward, prepared, cotangent, tangents)
    scenario.check(
        close_check(
            "reverse-jvp-matches-closed-form",
            _flat(_collect(second, "gradient-tangent")),
            _flat(
                (
                    g @ (tb @ c + b @ tc).T,
                    ta.T @ g @ c.T + a.T @ g @ tc.T,
                    (ta @ b + a @ tb).T @ g,
                )
            ),
            rtol=1e-5,
            atol=1e-5,
        )
    )


def _solve(scenario: Scenario) -> None:
    la = phx.linalg
    space = la.ArraySpace((8,), dtype=jnp.float32)

    def action(value: jax.Array) -> jax.Array:
        zero = jnp.zeros((1,), value.dtype)
        return (
            3 * value
            - jnp.concatenate((zero, value[:-1]))
            - jnp.concatenate((value[1:], zero))
        )

    properties = la.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "construction",
            "positive_definite": "construction",
            "positive_semidefinite": "construction",
        },
    )
    problem = la.LinearSystem(
        la.FunctionLinearOperator(
            action,
            source=space,
            target=space,
            transpose_action=action,
            properties=properties,
        )
    )

    def policy(relative: float, steps: int) -> Any:
        return la.LinearSolvePolicy(
            la.PCG(),
            tolerance=la.TolerancePolicy(
                relative=relative, absolute=0.0, max_steps=steps
            ),
            differentiation=la.DifferentiationPolicy("mathematical"),
            failure=la.FailurePolicy("status"),
        )

    bounded = policy(1e-4, 32)
    exhausted = policy(1e-8, 1)

    def solve(target: Any) -> dict[str, Any]:
        result = la.solve(problem, target, policy=bounded)
        return {
            "value": result.value,
            "status": result.status,
            "residual_norm": result.diagnostics.residual_norm,
            "iterations": result.diagnostics.iterations,
        }

    def exhaust(target: Any) -> dict[str, Any]:
        result = la.solve(problem, target, policy=exhausted)
        return {
            "value": result.value,
            "status": result.status,
            "iterations": result.diagnostics.iterations,
        }

    def solution(target: Any) -> Any:
        return la.solve(problem, target, policy=bounded).value

    def forward(target: Any, direction: Any) -> dict[str, Any]:
        value, tangent = jax.jvp(solution, (target,), (direction,))
        return {"value": value, "tangent": tangent}

    def reverse(target: Any, seed: Any) -> dict[str, Any]:
        value, pullback = jax.vjp(solution, target)
        return {"value": value, "rhs_cotangent": pullback(seed)[0]}

    matrix = 3 * np.eye(8) - np.eye(8, k=-1) - np.eye(8, k=1)
    rhs = np.arange(1, 9, dtype=np.float32)
    rng = np.random.default_rng(13)
    direction = rng.standard_normal(8).astype(np.float32)
    seed = rng.standard_normal(8).astype(np.float32)
    solved = scenario.run("solve", solve, rhs)
    scenario.check(
        equal_check(
            "solve-status-success",
            int(solved["status"]),
            int(la.LinearSolveStatus.SUCCESS),
        )
    )
    residual = float(np.linalg.norm(matrix @ solved["value"].astype(np.float64) - rhs))
    scenario.check(
        _record(
            "true-residual-within-tolerance",
            residual <= 1.1e-4 * float(np.linalg.norm(rhs)),
            residual=_float(residual),
            bound=_float(1.1e-4 * float(np.linalg.norm(rhs))),
            iterations=int(solved["iterations"]),
        )
    )
    scenario.check(
        close_check(
            "solution-matches-dense-solve",
            solved["value"],
            np.linalg.solve(matrix, rhs),
            rtol=2e-4,
            atol=2e-5,
        )
    )
    failed = scenario.run("max-steps-exhaustion", exhaust, rhs)
    scenario.check(
        equal_check(
            "exhaustion-reports-maximum-steps",
            int(failed["status"]),
            int(la.LinearSolveStatus.MAXIMUM_STEPS_REACHED),
        )
    )
    tangent = scenario.run("rhs-jvp", forward, rhs, direction)["tangent"]
    scenario.check(
        relative_check(
            "rhs-jvp-matches-dense-solve",
            tangent,
            np.linalg.solve(matrix, direction),
            tolerance=1e-3,
        )
    )
    cotangent = scenario.run("rhs-vjp", reverse, rhs, seed)["rhs_cotangent"]
    scenario.check(
        relative_check(
            "rhs-vjp-matches-dense-transpose-solve",
            cotangent,
            np.linalg.solve(matrix.T, seed),
            tolerance=1e-3,
        )
    )


def _sparse(scenario: Scenario) -> None:
    from phydrax.sparse import EdgeRelation, gather_routes, linear_apply, route_reduce

    sources = np.asarray([0, 1, 1, 3, 2, 0], np.int32)
    targets = np.asarray([0, 0, 2, 1, 2, 1], np.int32)
    valid = np.asarray([True, True, True, True, False, True])
    relation = EdgeRelation(sources, targets, source_size=4, target_size=3, valid=valid)
    rng = np.random.default_rng(9)
    coefficients = rng.standard_normal(6).astype(np.float32)
    values = rng.standard_normal((4, 2)).astype(np.float32)
    seed = rng.standard_normal((3, 2)).astype(np.float32)
    coefficient_tangent = rng.standard_normal(6).astype(np.float32)
    value_tangent = rng.standard_normal((4, 2)).astype(np.float32)

    def apply(rel: Any, weights: Any, payload: Any) -> dict[str, Any]:
        return {"value": linear_apply(rel, weights, payload)}

    def apply_deterministic(rel: Any, weights: Any, payload: Any) -> dict[str, Any]:
        return {
            "value": linear_apply(rel, weights, payload, accumulation="deterministic")
        }

    def maximum(rel: Any, payload: Any) -> dict[str, Any]:
        return {"value": route_reduce(rel, gather_routes(rel, payload), reduction="max")}

    def reverse(rel: Any, weights: Any, payload: Any, cotangent: Any) -> dict[str, Any]:
        value, pullback = jax.vjp(lambda c, v: linear_apply(rel, c, v), weights, payload)
        weight_cotangent, payload_cotangent = pullback(cotangent)
        return {
            "value": value,
            "coefficient_cotangent": weight_cotangent,
            "value_cotangent": payload_cotangent,
        }

    def forward(rel: Any, weights: Any, payload: Any, dw: Any, dv: Any) -> dict[str, Any]:
        value, tangent = jax.jvp(
            lambda c, v: linear_apply(rel, c, v), (weights, payload), (dw, dv)
        )
        return {"value": value, "tangent": tangent}

    def mixed(
        rel: Any, weights: Any, payload: Any, dv: Any, cotangent: Any
    ) -> dict[str, Any]:
        def coefficient_gradient(v: Any) -> Any:
            return jax.grad(lambda c: jnp.sum(cotangent * linear_apply(rel, c, v)))(
                weights
            )

        return {"mixed": jax.jvp(coefficient_gradient, (payload,), (dv,))[1]}

    active = np.flatnonzero(valid)

    def routed(weights: np.ndarray, payload: np.ndarray) -> np.ndarray:
        result = np.zeros((3, 2))
        for route in active:
            result[targets[route]] += weights[route] * payload[sources[route]]
        return result

    def route_inner(cotangent: np.ndarray, payload: np.ndarray) -> np.ndarray:
        result = np.zeros(6)
        for route in active:
            result[route] = cotangent[targets[route]] @ payload[sources[route]]
        return result

    c, x, g = (item.astype(np.float64) for item in (coefficients, values, seed))
    expected = routed(c, x)
    fast = scenario.run("linear-apply-fast", apply, relation, coefficients, values)[
        "value"
    ]
    scenario.check(
        close_check("fast-matches-route-sum", fast, expected, rtol=1e-6, atol=1e-6)
    )
    ordered = scenario.run(
        "linear-apply-deterministic", apply_deterministic, relation, coefficients, values
    )["value"]
    scenario.check(
        close_check(
            "deterministic-matches-route-sum", ordered, expected, rtol=1e-6, atol=1e-6
        )
    )
    largest = np.stack(
        [
            values[sources[active[targets[active] == target]]].max(axis=0)
            for target in range(3)
        ]
    )
    scenario.check(
        equal_check(
            "route-max-matches-valid-routes",
            scenario.run("route-max", maximum, relation, values)["value"],
            largest,
        )
    )
    reversed_ = scenario.run(
        "linear-apply-vjp", reverse, relation, coefficients, values, seed
    )
    transpose = np.zeros((4, 2))
    for route in active:
        transpose[sources[route]] += c[route] * g[targets[route]]
    scenario.check(
        close_check(
            "coefficient-cotangent-matches-route-inner",
            reversed_["coefficient_cotangent"],
            route_inner(g, x),
            rtol=1e-5,
            atol=1e-6,
        )
    )
    scenario.check(
        equal_check(
            "invalid-route-coefficient-cotangent-is-zero",
            reversed_["coefficient_cotangent"][4],
            np.float32(0.0),
        )
    )
    scenario.check(
        close_check(
            "value-cotangent-matches-transpose",
            reversed_["value_cotangent"],
            transpose,
            rtol=1e-5,
            atol=1e-6,
        )
    )
    tangent = scenario.run(
        "linear-apply-jvp",
        forward,
        relation,
        coefficients,
        values,
        coefficient_tangent,
        value_tangent,
    )["tangent"]
    scenario.check(
        close_check(
            "jvp-matches-bilinear-product-rule",
            tangent,
            routed(coefficient_tangent.astype(np.float64), x)
            + routed(c, value_tangent.astype(np.float64)),
            rtol=1e-5,
            atol=1e-6,
        )
    )
    second = scenario.run(
        "coefficient-value-mixed",
        mixed,
        relation,
        coefficients,
        values,
        value_tangent,
        seed,
    )["mixed"]
    scenario.check(
        close_check(
            "mixed-derivative-matches-route-inner",
            second,
            route_inner(g, value_tangent.astype(np.float64)),
            rtol=1e-5,
            atol=1e-6,
        )
    )


def _callbacks(scenario: Scenario) -> None:
    events: list[int] = []

    def observe(index: Any) -> None:
        events.append(int(index))

    def ordered_loop(start: Any) -> dict[str, Any]:
        def body(index: jax.Array, value: jax.Array) -> jax.Array:
            jax.debug.callback(observe, index, ordered=True)
            return jax.lax.cond(index % 2 == 0, lambda x: x + 2, lambda x: x - 1, value)

        return {"value": jax.lax.fori_loop(0, 6, body, start)}

    def bounded_loops(start: Any) -> dict[str, Any]:
        counted = jax.lax.while_loop(
            lambda value: value < 6, lambda value: value + 1, start
        )
        summed, _ = jax.lax.scan(
            lambda value, index: (value + index, None),
            start,
            jnp.arange(6, dtype=start.dtype),
        )
        return {"counted": counted, "summed": summed}

    start = np.asarray(0, np.int32)
    result = scenario.run("ordered-callback-loop", ordered_loop, start)
    scenario.check(equal_check("ordered-loop-value", result["value"], 3))
    scenario.check(equal_check("ordered-callback-events", events, list(range(6))))
    loops = scenario.run("callback-free-loops", bounded_loops, start)
    scenario.check(equal_check("while-loop-count", loops["counted"], 6))
    scenario.check(equal_check("scan-sum", loops["summed"], 15))


SCENARIOS: dict[str, ScenarioSpec] = {
    spec.name: spec
    for spec in (
        ScenarioSpec(
            "pinn-poisson-mlp",
            "phydrax.nn.models.MLP, phydrax.operators.differential.laplacian, phydrax.optim.adam",
            {
                "laplacian-reference": "exported",
                "objective-value-and-gradient": "exported",
                "objective-jvp": "exported",
                "objective-hvp": "exported",
                "adam-step": "exported",
            },
            _pinn,
        ),
        ScenarioSpec(
            "measure-aware-attention",
            "phydrax.nn.layers.MeasureAwareAttention",
            {
                "fused-primal": "exported",
                "blockwise-primal": "exported",
                "dense-primal": "exported",
                "fused-query-jvp": "exported",
                "fused-gradients": "exported",
                "fused-query-hvp": "exported",
                "fused-parameter-query-mixed": "exported",
                "blockwise-gradients": "exported",
                "blockwise-query-hvp": "exported",
            },
            _attention,
        ),
        ScenarioSpec(
            "prepared-contraction",
            "phydrax.tensor_network.prepare_contraction",
            {
                "execute": "exported",
                "reverse": "exported",
                "jvp": "exported",
                "reverse-jvp": "exported",
            },
            _contraction,
        ),
        ScenarioSpec(
            "bounded-pcg-solve",
            "phydrax.linalg.solve with PCG",
            # The canonical solve keeps fail-closed `eqx.error_if` runtime guards
            # (Krylov step bound, operator numeric version), which lower to host
            # callbacks: TPU runtime callbacks, but a refused export. The CPU
            # reference still executes and checks every target.
            {
                "solve": "host-callback",
                "max-steps-exhaustion": "host-callback",
                "rhs-jvp": "host-callback",
                "rhs-vjp": "host-callback",
            },
            _solve,
        ),
        ScenarioSpec(
            "sparse-route-gather-reduce",
            "phydrax.sparse.EdgeRelation routes",
            {
                "linear-apply-fast": "exported",
                "linear-apply-deterministic": "exported",
                "route-max": "exported",
                "linear-apply-vjp": "exported",
                "linear-apply-jvp": "exported",
                "coefficient-value-mixed": "exported",
            },
            _sparse,
        ),
        ScenarioSpec(
            "ordered-host-callback",
            "jax.debug.callback in compiled loops",
            {"ordered-callback-loop": "host-callback", "callback-free-loops": "exported"},
            _callbacks,
        ),
    )
}


def _select_cpu() -> str | None:
    try:
        phx.config.configure(phx.config.RuntimeConfig(platforms=("cpu",)))
    except phx.config.ConfigurationApplicationError as error:
        return f"CPU-only selection was refused: {error}"
    return None


def write_report(path: Path, report: Record) -> None:
    payload = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    publish_bytes(
        path, payload.encode("utf-8"), maximum_bytes=JSON_LIMIT, mode="atomic_replace"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=tuple(SCENARIOS),
        help="Repeat to select scenarios; defaults to every scenario",
    )
    args = parser.parse_args(argv)
    names = (
        tuple(SCENARIOS) if args.scenario is None else tuple(dict.fromkeys(args.scenario))
    )
    refusal = _select_cpu()
    report = (
        preflight(tuple(SCENARIOS[name] for name in names))
        if refusal is None
        else _report(_environment(), refusal)
    )
    write_report(args.output, report)
    if report["refusal"] is not None:
        return 2
    return 0 if report["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
