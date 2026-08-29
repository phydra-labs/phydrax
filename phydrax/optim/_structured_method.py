#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx

from .._strict import StrictModule
from ._iterative import AbstractMinimizationMethod, OptimizationTermination
from ._structured_nonlinear import (
    prepare_structured_nonlinear,
    PreparedStructuredNonlinearProgram,
    StructuredNonlinearProgram,
    StructuredNonlinearResult,
    StructuredNonlinearWarmStart,
)


class StructuredNonlinearCapabilities(StrictModule):
    """Static capabilities of one structured nonlinear method."""

    exact_sparse_jacobian: bool = eqx.field(static=True)
    exact_sparse_hessian: bool = eqx.field(static=True)
    limited_memory_hessian: bool = eqx.field(static=True)
    portable_warm_start: bool = eqx.field(static=True)
    numeric_refresh: bool = eqx.field(static=True)
    jit: bool = eqx.field(static=True)
    ordinary_batch: bool = eqx.field(static=True)
    pooled_batch: bool = eqx.field(static=True)
    implicit_differentiation: bool = eqx.field(static=True)
    device_execution: bool = eqx.field(static=True)

    def __init__(
        self,
        *,
        exact_sparse_jacobian: bool,
        exact_sparse_hessian: bool,
        limited_memory_hessian: bool,
        portable_warm_start: bool,
        numeric_refresh: bool,
        jit: bool,
        ordinary_batch: bool,
        pooled_batch: bool,
        implicit_differentiation: bool,
        device_execution: bool,
    ):
        self.exact_sparse_jacobian = bool(exact_sparse_jacobian)
        self.exact_sparse_hessian = bool(exact_sparse_hessian)
        self.limited_memory_hessian = bool(limited_memory_hessian)
        self.portable_warm_start = bool(portable_warm_start)
        self.numeric_refresh = bool(numeric_refresh)
        self.jit = bool(jit)
        self.ordinary_batch = bool(ordinary_batch)
        self.pooled_batch = bool(pooled_batch)
        self.implicit_differentiation = bool(implicit_differentiation)
        self.device_execution = bool(device_execution)


class AbstractStructuredNonlinearMethod(AbstractMinimizationMethod):
    """Method consuming one prepared fixed-topology bound-form NLP."""

    @property
    @abc.abstractmethod
    def structured_capabilities(self) -> StructuredNonlinearCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def solve_structured(
        self,
        prepared: PreparedStructuredNonlinearProgram,
        initial_coordinates: Any,
        /,
        *,
        termination: OptimizationTermination,
        warm_start: StructuredNonlinearWarmStart | None,
    ) -> StructuredNonlinearResult:
        raise NotImplementedError


def solve_structured_nonlinear(
    program: StructuredNonlinearProgram | PreparedStructuredNonlinearProgram,
    initial_coordinates: Any,
    /,
    *,
    method: AbstractStructuredNonlinearMethod,
    termination: OptimizationTermination | None = None,
    args: Any = None,
    warm_start: StructuredNonlinearWarmStart | None = None,
) -> StructuredNonlinearResult:
    """Prepare if needed, solve structurally, and validate portable identities."""
    if not isinstance(method, AbstractStructuredNonlinearMethod):
        raise TypeError("method must implement AbstractStructuredNonlinearMethod.")
    termination_ = OptimizationTermination() if termination is None else termination
    if not isinstance(termination_, OptimizationTermination):
        raise TypeError("termination must be OptimizationTermination or None.")
    if isinstance(program, StructuredNonlinearProgram):
        prepared = prepare_structured_nonlinear(program, args)
    elif isinstance(program, PreparedStructuredNonlinearProgram):
        if args is not None:
            raise ValueError(
                "args are already bound by PreparedStructuredNonlinearProgram."
            )
        prepared = program
    else:
        raise TypeError(
            "program must be StructuredNonlinearProgram or "
            "PreparedStructuredNonlinearProgram."
        )
    if warm_start is not None:
        if not method.structured_capabilities.portable_warm_start:
            raise ValueError(f"{method.method_id} does not support portable warm starts.")
        if warm_start.structure_id != prepared.structure_id:
            raise ValueError("Warm-start structure does not match the prepared program.")
    result = method.solve_structured(
        prepared,
        initial_coordinates,
        termination=termination_,
        warm_start=warm_start,
    )
    if not isinstance(result, StructuredNonlinearResult):
        raise TypeError(
            "Structured nonlinear methods must return StructuredNonlinearResult."
        )
    if result.structure_id != prepared.structure_id:
        raise ValueError(
            "Structured result does not match the prepared program structure."
        )
    if result.method_id != method.method_id:
        raise ValueError("Structured result method identity does not match the method.")
    return result


__all__ = [
    "AbstractStructuredNonlinearMethod",
    "StructuredNonlinearCapabilities",
    "solve_structured_nonlinear",
]
