#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import AbstractVectorSpace
from ..nonlinear import NonlinearSystemProblem
from ._core import ParameterContinuationProblem
from ._geometry import ContinuationGeometry, ContinuationRepresentationPolicy


class HomotopyEndpointStatus(IntEnum):
    """Validity of explicitly checked homotopy endpoints."""

    SUCCESS = 0
    START_NOT_CONVERGED = 1
    TARGET_NOT_CONVERGED = 2
    NONFINITE = 3


class HomotopyEndpointCertificate(StrictModule):
    """Residual evidence at the start and target of one homotopy path."""

    start_residual_norm: Array
    target_residual_norm: Array
    tolerance: Array
    finite: Array
    status: Array

    def __init__(
        self,
        *,
        start_residual_norm: Any,
        target_residual_norm: Any,
        tolerance: Any,
        finite: Any,
        status: Any,
    ):
        self.start_residual_norm = jnp.asarray(start_residual_norm)
        self.target_residual_norm = jnp.asarray(target_residual_norm)
        self.tolerance = jnp.asarray(tolerance)
        self.finite = jnp.asarray(finite, dtype=bool)
        self.status = jnp.asarray(status, dtype=jnp.int32)

    @property
    def successful(self) -> Array:
        return self.status == int(HomotopyEndpointStatus.SUCCESS)


class HomotopyProblem(StrictModule):
    """Continuation-ready path with fixed unit-interval endpoint semantics."""

    continuation_problem: ParameterContinuationProblem
    physical_parameter_function: Callable[[Array], Array] | None
    homotopy_id: str = eqx.field(static=True)
    has_physical_parameter: bool = eqx.field(static=True)

    def __init__(
        self,
        residual: Callable[[PyTree[Any], Array, Any], PyTree[Any]],
        /,
        *,
        physical_parameter: Callable[[Array], Array] | None = None,
        homotopy_id: str = "homotopy",
        state_space: AbstractVectorSpace | None = None,
        residual_space: AbstractVectorSpace | None = None,
        representation: ContinuationRepresentationPolicy | None = None,
    ):
        if not callable(residual):
            raise TypeError("residual must be callable.")
        if physical_parameter is not None and not callable(physical_parameter):
            raise TypeError("physical_parameter must be callable or None.")
        identifier = str(homotopy_id)
        if not identifier:
            raise ValueError("homotopy_id must be non-empty.")
        self.continuation_problem = ParameterContinuationProblem(
            residual,
            parameter_lower=0.0,
            parameter_upper=1.0,
            state_space=state_space,
            residual_space=residual_space,
            representation=representation,
            problem_id=identifier,
        )
        self.physical_parameter_function = physical_parameter
        self.homotopy_id = identifier
        self.has_physical_parameter = physical_parameter is not None

    def residual(
        self,
        state: PyTree[Any],
        homotopy_parameter: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        return self.continuation_problem.residual(
            state,
            homotopy_parameter,
            args,
        )

    def physical_parameter(self, homotopy_parameter: Any, /) -> Array:
        if self.physical_parameter_function is None:
            raise ValueError("This homotopy has no physical-parameter mapping.")
        value = jnp.asarray(homotopy_parameter)
        if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("homotopy_parameter must be one real scalar array.")
        mapped = jnp.asarray(self.physical_parameter_function(value))
        if mapped.shape != () or not jnp.issubdtype(mapped.dtype, jnp.floating):
            raise TypeError("The physical-parameter mapping must return a real scalar.")
        return mapped

    def parameter_derivative(
        self,
        state: PyTree[Any],
        homotopy_parameter: Any,
        args: Any = None,
        /,
    ) -> PyTree[Array]:
        parameter = jnp.asarray(homotopy_parameter)
        return jax.jvp(
            lambda value: self.residual(state, value, args),
            (parameter,),
            (jnp.ones_like(parameter),),
        )[1]

    def verify_endpoints(
        self,
        start_state: PyTree[Any],
        target_state: PyTree[Any],
        /,
        *,
        tolerance: float = 1e-8,
        args: Any = None,
        geometry: ContinuationGeometry | None = None,
    ) -> HomotopyEndpointCertificate:
        tolerance_ = float(tolerance)
        if not isfinite(tolerance_) or tolerance_ < 0.0:
            raise ValueError("tolerance must be finite and non-negative.")
        start_residual = self.residual(start_state, jnp.asarray(0.0), args)
        target_residual = self.residual(target_state, jnp.asarray(1.0), args)
        geometry_ = geometry
        if geometry_ is None:
            state_space, residual_space = self.continuation_problem.declared_spaces()
            geometry_ = ContinuationGeometry.resolve(
                start_state,
                start_residual,
                state_space=state_space,
                residual_space=residual_space,
                representation=self.continuation_problem.representation_policy(),
            )
        if not isinstance(geometry_, ContinuationGeometry):
            raise TypeError("geometry must be a ContinuationGeometry or None.")
        start_coordinates = geometry_.residual_to_execution(start_residual)
        target_coordinates = geometry_.residual_to_execution(target_residual)
        start_norm = geometry_.residual_norm(start_coordinates)
        target_norm = geometry_.residual_norm(target_coordinates)
        finite = (
            tree_allfinite(start_coordinates)
            & tree_allfinite(target_coordinates)
            & jnp.isfinite(start_norm)
            & jnp.isfinite(target_norm)
        )
        status = jnp.where(
            ~finite,
            int(HomotopyEndpointStatus.NONFINITE),
            jnp.where(
                start_norm > tolerance_,
                int(HomotopyEndpointStatus.START_NOT_CONVERGED),
                jnp.where(
                    target_norm > tolerance_,
                    int(HomotopyEndpointStatus.TARGET_NOT_CONVERGED),
                    int(HomotopyEndpointStatus.SUCCESS),
                ),
            ),
        )
        return HomotopyEndpointCertificate(
            start_residual_norm=start_norm,
            target_residual_norm=target_norm,
            tolerance=tolerance_,
            finite=finite,
            status=status,
        )


def linear_homotopy(
    start_problem: NonlinearSystemProblem,
    target_problem: NonlinearSystemProblem,
    /,
    *,
    homotopy_id: str | None = None,
) -> HomotopyProblem:
    """Build ``(1-t) G(x) + t F(x) = 0`` from two nonlinear systems."""
    if not isinstance(start_problem, NonlinearSystemProblem):
        raise TypeError("start_problem must be a NonlinearSystemProblem.")
    if not isinstance(target_problem, NonlinearSystemProblem):
        raise TypeError("target_problem must be a NonlinearSystemProblem.")
    identifier = (
        f"{start_problem.problem_id}-to-{target_problem.problem_id}"
        if homotopy_id is None
        else str(homotopy_id)
    )
    if not identifier:
        raise ValueError("homotopy_id must be non-empty.")

    def residual(state, homotopy_parameter, args):
        start = start_problem.residual(state, args)
        target = target_problem.residual(state, args)
        if jax.tree.structure(start) != jax.tree.structure(target):
            raise ValueError("Homotopy endpoint residual structures must match.")
        return jax.tree.map(
            lambda start_value, target_value: (
                (1.0 - homotopy_parameter) * start_value
                + homotopy_parameter * target_value
            ),
            start,
            target,
        )

    return HomotopyProblem(residual, homotopy_id=identifier)


def parameter_homotopy(
    problem: ParameterContinuationProblem,
    start_parameter: float,
    target_parameter: float,
    /,
    *,
    homotopy_id: str | None = None,
) -> HomotopyProblem:
    """Build an affine unit-interval path through a physical parameter."""
    if not isinstance(problem, ParameterContinuationProblem):
        raise TypeError("problem must be a ParameterContinuationProblem.")
    start = float(start_parameter)
    target = float(target_parameter)
    if not isfinite(start) or not isfinite(target):
        raise ValueError("Homotopy endpoint parameters must be finite.")
    if not (problem.coordinate_lower <= start <= problem.coordinate_upper):
        raise ValueError("start_parameter lies outside the continuation bounds.")
    if not (problem.coordinate_lower <= target <= problem.coordinate_upper):
        raise ValueError("target_parameter lies outside the continuation bounds.")
    if start == target:
        raise ValueError("Homotopy endpoint parameters must be distinct.")
    identifier = (
        f"{problem.problem_id}/parameter-homotopy"
        if homotopy_id is None
        else str(homotopy_id)
    )
    if not identifier:
        raise ValueError("homotopy_id must be non-empty.")

    def physical_parameter(homotopy_parameter):
        return (1.0 - homotopy_parameter) * start + homotopy_parameter * target

    def residual(state, homotopy_parameter, args):
        return problem.residual(
            state,
            physical_parameter(homotopy_parameter),
            args,
        )

    return HomotopyProblem(
        residual,
        physical_parameter=physical_parameter,
        state_space=problem.state_space,
        residual_space=problem.residual_space,
        representation=problem.representation,
        homotopy_id=identifier,
    )


__all__ = [
    "HomotopyEndpointCertificate",
    "HomotopyEndpointStatus",
    "HomotopyProblem",
    "linear_homotopy",
    "parameter_homotopy",
]
