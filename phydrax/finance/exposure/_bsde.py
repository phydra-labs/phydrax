# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Typed finance binding to the native BSDE evaluation substrate."""

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
from jaxtyping import Array, Key

from ..._strict import StrictModule
from ...domain import DomainFunction
from ...stochastic import (
    BSDEEvaluation,
    BSDEPathBatch,
    BSDEProblem,
    evaluate_bsde,
)
from ..core import PricingLaw


BSDEControlMode: TypeAlias = Literal["explicit", "autodiff"]
BSDEQuadrature: TypeAlias = Literal["left", "trapezoid"]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a canonical non-empty string.")
    return value


class ExposureBSDERoute(StrictModule):
    """One finance-labelled native BSDE problem and exact pricing-law binding."""

    problem: BSDEProblem
    pricing_law: PricingLaw = eqx.field(static=True)
    factor_layout_id: str = eqx.field(static=True)
    closeout_convention_id: str = eqx.field(static=True)
    control_mode: BSDEControlMode = eqx.field(static=True)
    quadrature: BSDEQuadrature = eqx.field(static=True)
    route_id: str = eqx.field(static=True)

    def __init__(
        self,
        problem: BSDEProblem,
        pricing_law: PricingLaw,
        /,
        *,
        factor_layout_id: str,
        closeout_convention_id: str,
        control_mode: BSDEControlMode,
        quadrature: BSDEQuadrature,
        route_id: str,
    ):
        if not isinstance(problem, BSDEProblem):
            raise TypeError("problem must be a BSDEProblem.")
        if not isinstance(pricing_law, PricingLaw):
            raise TypeError(
                "pricing_law must be a PricingLaw; P and stress laws are invalid."
            )
        layout = _identifier(factor_layout_id, "factor_layout_id")
        if pricing_law.factor_layout_id != layout:
            raise ValueError(
                "Pricing-law factor layout is incompatible with the BSDE route."
            )
        if control_mode not in ("explicit", "autodiff"):
            raise ValueError("control_mode must be 'explicit' or 'autodiff'.")
        if quadrature not in ("left", "trapezoid"):
            raise ValueError("quadrature must be 'left' or 'trapezoid'.")
        self.problem = problem
        self.pricing_law = pricing_law
        self.factor_layout_id = layout
        self.closeout_convention_id = _identifier(
            closeout_convention_id, "closeout_convention_id"
        )
        self.control_mode = control_mode
        self.quadrature = quadrature
        self.route_id = _identifier(route_id, "route_id")


class ExposureBSDEEvaluation(StrictModule):
    """Native BSDE residuals retained with their finance semantic identities."""

    evaluation: BSDEEvaluation
    pricing_law_id: str = eqx.field(static=True)
    closeout_convention_id: str = eqx.field(static=True)
    route_id: str = eqx.field(static=True)
    path_id: str = eqx.field(static=True)


def evaluate_exposure_bsde(
    route: ExposureBSDERoute,
    paths: BSDEPathBatch,
    value_model: DomainFunction,
    /,
    *,
    control_model: DomainFunction | None = None,
    key: Key[Array, ""],
) -> ExposureBSDEEvaluation:
    """Evaluate a typed BSDE route without introducing a second BSDE solver API."""

    if not isinstance(route, ExposureBSDERoute):
        raise TypeError("route must be an ExposureBSDERoute.")
    if not isinstance(paths, BSDEPathBatch):
        raise TypeError("paths must be a BSDEPathBatch.")
    if not isinstance(value_model, DomainFunction):
        raise TypeError("value_model must be a DomainFunction, not an opaque callable.")
    if control_model is not None and not isinstance(control_model, DomainFunction):
        raise TypeError("control_model must be a DomainFunction or None.")
    if paths.process_id != route.problem.process_id:
        raise ValueError("BSDE paths and route process identities differ.")
    if route.control_mode == "explicit" and control_model is None:
        raise ValueError("Explicit BSDE control requires a typed control_model.")
    if route.control_mode == "autodiff" and control_model is not None:
        raise ValueError("Autodiff BSDE control does not accept a control_model.")
    evaluation = evaluate_bsde(
        route.problem,
        paths,
        value_model,
        control_predictor=control_model,
        control_mode=route.control_mode,
        quadrature=route.quadrature,
        key=key,
    )
    return ExposureBSDEEvaluation(
        evaluation,
        route.pricing_law.law_id,
        route.closeout_convention_id,
        route.route_id,
        paths.path_id,
    )


__all__ = [
    "ExposureBSDEEvaluation",
    "ExposureBSDERoute",
    "evaluate_exposure_bsde",
]
