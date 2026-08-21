#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any

from jaxtyping import PyTree

from ..._strict import StrictModule
from ._types import (
    IterativeStepMetrics,
    LeastSquaresResult,
    MinimizationProblem,
    MinimizationResult,
    NonlinearLeastSquaresProblem,
    OptimizationCapabilities,
    OptimizationTermination,
)


class AbstractMinimizationMethod(StrictModule):
    """Method implementing a complete scalar minimization contract."""

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> OptimizationCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        raise NotImplementedError


class AbstractScalarIterativeMethod(AbstractMinimizationMethod):
    """Scalar method that can advance one frozen objective realization."""

    @property
    @abc.abstractmethod
    def globalization_id(self) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, parameters: PyTree[Any], /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_state(
        self,
        value_function: Callable[[PyTree[Any]], Any],
        parameters: PyTree[Any],
        /,
    ) -> Any:
        """Initialize state that may depend on one objective structure."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        value_function: Callable[[PyTree[Any]], Any],
        parameters: PyTree[Any],
        state: Any,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], Any, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def step_metrics(self, state: Any, /) -> IterativeStepMetrics:
        raise NotImplementedError


class AbstractLeastSquaresMethod(StrictModule):
    """Method implementing nonlinear least squares over residual PyTrees."""

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> OptimizationCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, parameters: PyTree[Any], /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_state(
        self,
        residual_function: Callable[[PyTree[Any]], PyTree[Any]],
        parameters: PyTree[Any],
        /,
    ) -> Any:
        """Initialize state that may depend on one residual structure."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        residual_function: Callable[[PyTree[Any]], PyTree[Any]],
        parameters: PyTree[Any],
        state: Any,
        /,
        *,
        termination: OptimizationTermination | None,
    ) -> tuple[PyTree[Any], Any, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def step_metrics(self, state: Any, /) -> IterativeStepMetrics:
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        problem: NonlinearLeastSquaresProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> LeastSquaresResult:
        raise NotImplementedError


class AbstractCompositeLeastSquaresMethod(StrictModule):
    """Method for objectives combining residual squares and signed scalars."""

    @property
    @abc.abstractmethod
    def method_id(self) -> str:
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def capabilities(self) -> OptimizationCapabilities:
        raise NotImplementedError

    @abc.abstractmethod
    def init(self, parameters: PyTree[Any], /) -> Any:
        raise NotImplementedError

    @abc.abstractmethod
    def prepare_state(
        self,
        problem: Any,
        parameters: PyTree[Any],
        /,
        *,
        args: Any,
    ) -> Any:
        """Initialize state that may depend on one composite objective."""
        raise NotImplementedError

    @abc.abstractmethod
    def step(
        self,
        problem: Any,
        parameters: PyTree[Any],
        state: Any,
        /,
        *,
        termination: OptimizationTermination | None,
        args: Any,
    ) -> tuple[PyTree[Any], Any, Any]:
        raise NotImplementedError

    @abc.abstractmethod
    def step_metrics(self, state: Any, /) -> IterativeStepMetrics:
        raise NotImplementedError

    @abc.abstractmethod
    def solve(
        self,
        problem: Any,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> Any:
        raise NotImplementedError


__all__ = [
    "AbstractCompositeLeastSquaresMethod",
    "AbstractLeastSquaresMethod",
    "AbstractMinimizationMethod",
    "AbstractScalarIterativeMethod",
]
