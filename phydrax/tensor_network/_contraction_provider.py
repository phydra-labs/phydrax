# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Explicit, immutable execution requests; importing this module is device-inert."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import final, Literal

import equinox as eqx

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._validation import finite_real_scalar, positive_finite_float
from ..typing import parse


type ContractionProvider = Literal["native", "cutensor", "cutensornet"]


@final
class ContractionExecutionPolicy(StrictModule):
    """Device-local executor and explicit external-memory/transformation bounds.

    ``maximum_cache_bytes`` bounds retained external device scratch/cache across
    the partition. Opaque vendor host descriptor sizes are unknown, not zero;
    ``maximum_cache_entries`` separately bounds their number. Numerical cache is
    disabled: descriptors may be reused, values never are. Batching uses serial
    device-local lanes. Host slice/checkpoint orchestration is not differentiable.

    The native executor keeps its existing plan resource admission and JAX
    transformations. Optional-provider transformation limits are not a claimed
    hard bound on compiler-managed native autodiff allocations.
    """

    provider: ContractionProvider = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    maximum_cache_bytes: int = eqx.field(static=True)
    maximum_cache_entries: int = eqx.field(static=True)
    maximum_batch_size: int = eqx.field(static=True)
    maximum_derivative_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        provider: ContractionProvider = "native",
        /,
        *,
        maximum_workspace_bytes: int = 2**28,
        maximum_cache_bytes: int = 2**30,
        maximum_cache_entries: int = 16,
        maximum_batch_size: int = 64,
        maximum_derivative_bytes: int = 2**31,
    ) -> None:
        provider = parse(provider, ContractionProvider, "provider")
        limits = (
            maximum_workspace_bytes,
            maximum_cache_bytes,
            maximum_cache_entries,
            maximum_batch_size,
            maximum_derivative_bytes,
        )
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) for value in limits
        ):
            raise TypeError("Contraction execution bounds must be integers.")
        if any(value < 1 for value in limits):
            raise ValueError("Contraction execution bounds must be positive.")
        if any(value >= 2**63 for value in limits) or maximum_cache_entries > 4096:
            raise ValueError(
                "Execution bounds must fit positive int64 and cache entries must not exceed 4096."
            )
        self.provider = provider
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self.maximum_cache_bytes = maximum_cache_bytes
        self.maximum_cache_entries = maximum_cache_entries
        self.maximum_batch_size = maximum_batch_size
        self.maximum_derivative_bytes = maximum_derivative_bytes
        self.policy_id = canonical_fingerprint(
            {
                "kind": "contraction-execution-policy",
                "provider": provider,
                "limits": limits,
            }
        )


@final
class ContractionPathPlanner(StrictModule):
    """Bounded, shape-only cuTensorNet FLOP-objective path search.

    The search runs in a spawned process with a whole-worker wall deadline. No
    numerical arrays or live handles cross that boundary. Slicing is not enabled.
    """

    maximum_planning_seconds: float = eqx.field(static=True)
    maximum_search_iterations: int = eqx.field(static=True)
    maximum_workspace_bytes: int = eqx.field(static=True)
    seed: int = eqx.field(static=True)
    planner_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        maximum_planning_seconds: float = 30.0,
        maximum_search_iterations: int = 64,
        maximum_workspace_bytes: int = 2**28,
        seed: int = 0,
    ) -> None:
        values = (maximum_search_iterations, maximum_workspace_bytes, seed)
        if any(
            not isinstance(value, Integral) or isinstance(value, bool) for value in values
        ):
            raise TypeError(
                "Planner iteration, workspace, and seed controls must be integers."
            )
        if (
            not 1 <= maximum_search_iterations <= 10000
            or not 1 <= maximum_workspace_bytes < 2**63
            or not 0 <= seed < 2**31
        ):
            raise ValueError(
                "Planner iterations must lie in [1, 10000], workspace must fit positive int64, and seed must fit nonnegative int32."
            )
        seconds = positive_finite_float(
            finite_real_scalar(maximum_planning_seconds, "maximum_planning_seconds"),
            "maximum_planning_seconds",
        )
        self.maximum_planning_seconds = seconds
        self.maximum_search_iterations = maximum_search_iterations
        self.maximum_workspace_bytes = maximum_workspace_bytes
        self.seed = seed
        self.planner_id = canonical_fingerprint(
            {
                "kind": "cutensornet-path-planner",
                "objective": "flops",
                "seconds": seconds,
                "controls": values,
            }
        )


@dataclass(frozen=True, slots=True)
class TensorOperation:
    """Portable FFI recipe, never a pointer or a live native handle."""

    inputs: tuple[tuple[int, ...], ...]
    output: tuple[int, ...]
    dimensions: tuple[tuple[int, int], ...]
    path: tuple[tuple[int, ...], ...]
    dtype: str
    provider: Literal["cutensor", "cutensornet"]
    workspace_limit_bytes: int
    cache_limit_bytes: int
    capacity: int
    maximum_batch_size: int
    maximum_derivative_bytes: int
    partition: int
    derivative_depth: int = 0

    @property
    def output_shape(self) -> tuple[int, ...]:
        extents = dict(self.dimensions)
        return tuple(extents[mode] for mode in self.output)

    @property
    def recipe(self) -> str:
        extents = dict(self.dimensions)
        tokens = [len(self.inputs)]
        for modes in (*self.inputs, self.output):
            tokens.extend((len(modes), *(extents[mode] for mode in modes), *modes))
        tokens.append(len(self.path))
        for step in self.path:
            tokens.extend((len(step), *step))
        return " ".join(str(value) for value in tokens)


@final
class ContractionExecutor(StrictModule):
    """Reconstructible execution identity with dynamic values kept elsewhere."""

    policy: ContractionExecutionPolicy
    operations: tuple[TensorOperation, ...] = eqx.field(static=True)
    build_identity: str = eqx.field(static=True)
    lowering_id: str = eqx.field(static=True)
    normalization_bytes: int = eqx.field(static=True)
    external_workspace_bound_bytes: int = eqx.field(static=True)
    external_cache_bound_bytes: int = eqx.field(static=True)
    vendor_host_bound_bytes: int | None = eqx.field(static=True)
    derivative_contract: str = eqx.field(static=True)


__all__ = [
    "ContractionExecutionPolicy",
    "ContractionPathPlanner",
]
