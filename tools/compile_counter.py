# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import inspect
import logging
import re
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

import jax


@dataclass(frozen=True)
class CompileEvent:
    kernel_name: str
    compile_seconds: float
    stage: str
    signature: str | None
    first_party_stack: tuple[str, ...]
    lowering_equations: str | None = None
    lowering_layouts: str | None = None

    def to_record(self) -> dict[str, object]:
        record: dict[str, object] = {
            "kernel_name": self.kernel_name,
            "compile_seconds": self.compile_seconds,
            "stage": self.stage,
            "signature": self.signature,
            "first_party_stack": list(self.first_party_stack),
        }
        if self.lowering_equations is not None:
            record["lowering_equations"] = self.lowering_equations
            record["lowering_layouts"] = self.lowering_layouts
        return record


_COMPLETED = re.compile(r"Finished XLA compilation (?:of )?(.+?) in ([0-9.eE+-]+) sec")
_STARTED = re.compile(r"Compiling (.+?) with global shapes and types (.+)")
_LOGGERS = ("jax._src.dispatch", "jax._src.interpreters.pxla")
_ROOT = Path(__file__).resolve().parents[1]


def _lowering_signature() -> tuple[str, str]:
    """Read static lowering operands without retaining frames or numerical owners."""
    frame = inspect.currentframe()
    equations: str | None = None
    layouts: str | None = None
    try:
        while frame is not None:
            if (
                frame.f_globals.get("__name__") == "jax._src.interpreters.pxla"
                and frame.f_code.co_name == "_cached_lowering_to_hlo"
            ):
                operands = frame.f_locals
                equations = str(operands["closed_jaxpr"])
                layouts = repr(
                    (
                        operands["num_const_args"],
                        operands["out_avals"],
                        operands["out_shardings"],
                        operands["in_layouts"],
                        operands["out_layouts"],
                        operands["num_devices"],
                        operands["donated_invars"],
                        operands["inout_aliases"],
                        operands["platforms"],
                        operands["abstract_mesh"],
                        operands["all_default_mem_kind"],
                        operands["propagated_out_mem_kinds"],
                        tuple(repr(value) for value in operands["in_avals"]),
                        tuple(repr(value) for value in operands["out_avals"]),
                        repr(operands["lowering_parameters"]),
                    )
                )
            if (
                equations is not None
                and layouts is not None
                and frame.f_globals.get("__name__") == "jax._src.interpreters.pxla"
                and frame.f_code.co_name == "lower_sharding_computation"
            ):
                operands = frame.f_locals
                # Dead input leaves still affect the tracing cache, not its final ABI.
                info = operands["all_args_info"]
                names = info.debug_info.safe_arg_names(len(info.in_avals))
                frontend = tuple(
                    (name, repr(value), repr(value.sharding))
                    for name, value in zip(names, info.in_avals, strict=True)
                )
                return equations, repr((layouts, frontend, operands["kept_var_idx"]))
            frame = frame.f_back
    finally:
        del frame
    raise RuntimeError(
        "compile_counter.lowering_signature must have its JAX lowering context; got no matching frame"
    )


class CompileCounter(logging.Handler):
    """Count completed compiler events directly, regardless of stderr handlers.

    Signature and optional first-party stack evidence come from the matching
    lowering event. Completed events alone contribute to compile totals.
    """

    def __init__(
        self,
        *,
        stage: str | Callable[[], str] = "unscoped",
        source_root: Path = _ROOT,
        capture_stack: bool = False,
        capture_equations: bool = False,
    ) -> None:
        super().__init__(logging.WARNING)
        self.events: list[CompileEvent] = []
        self.stage = stage
        self.capture_stack = capture_stack
        self.capture_equations = capture_equations
        self.source_root = source_root
        self._pending: defaultdict[
            str, deque[tuple[str, tuple[str, ...], str | None, str | None]]
        ] = defaultdict(deque)
        self._context: ExitStack | None = None

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def seconds(self) -> float:
        return sum(event.compile_seconds for event in self.events)

    def __enter__(self) -> CompileCounter:
        if self._context is not None:
            raise ValueError("compile_counter.context must be inactive; got active")
        context = ExitStack()
        context.enter_context(jax.log_compiles(True))
        for name in _LOGGERS:
            logger = logging.getLogger(name)
            logger.addHandler(self)
            context.callback(logger.removeHandler, self)
        self._context = context
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback_: TracebackType | None,
    ) -> None:
        context = self._context
        if context is None:
            raise ValueError("compile_counter.context must be active; got inactive")
        self._context = None
        context.__exit__(exception_type, exception, traceback_)
        self._pending.clear()

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        started = _STARTED.fullmatch(message)
        if started is not None:
            stack: list[str] = []
            if self.capture_stack:
                for frame in traceback.extract_stack():
                    path = Path(frame.filename)
                    if path.is_relative_to(self.source_root):
                        relative = path.relative_to(self.source_root)
                        if relative.parts[0] in ("phydrax", "tools", "tests"):
                            stack.append(f"{relative}:{frame.lineno}:{frame.name}")
            equations, layouts = (
                _lowering_signature() if self.capture_equations else (None, None)
            )
            self._pending[started.group(1)].append(
                (started.group(2), tuple(stack), equations, layouts)
            )
            return
        completed = _COMPLETED.fullmatch(message)
        if completed is None:
            return
        kernel = completed.group(1)
        pending = self._pending.get(kernel)
        signature, stack, equations, layouts = (
            pending.popleft() if pending else (None, (), None, None)
        )
        stage = self.stage if isinstance(self.stage, str) else self.stage()
        self.events.append(
            CompileEvent(
                kernel,
                float(completed.group(2)),
                stage,
                signature,
                stack,
                equations,
                layouts,
            )
        )
