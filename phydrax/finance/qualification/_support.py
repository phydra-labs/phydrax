#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Final

from ...qualification._registry import SupportTuple


_VALUATION_ROUTES: Final = frozenset(
    {"analytic", "fourier", "pde", "simulation", "exercise", "bsde"}
)
_ADVANCED_ROUTES: Final = frozenset(
    {"martingale-transport", "rough", "deep", "operator", "tensor"}
)


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical identifier.")
    return value


def _common(
    capability: str,
    route: str,
    coordinates: dict[str, str | int | bool],
    /,
    *,
    backend: str,
    precision: str,
) -> SupportTuple:
    values: dict[str, str | int | bool] = {
        "route": _text(route, "route"),
        "backend": _text(backend, "backend"),
        "precision": _text(precision, "precision"),
    }
    values.update(coordinates)
    return SupportTuple(capability, values)


def market_resolution_support(
    route: str,
    *,
    calendar: str,
    temporal_policy: str,
    missing_data_policy: str,
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe one exact host-resolution and device-materialization route."""
    return _common(
        "finance.market-resolution",
        route,
        {
            "calendar": _text(calendar, "calendar"),
            "temporal_policy": _text(temporal_policy, "temporal_policy"),
            "missing_data_policy": _text(missing_data_policy, "missing_data_policy"),
        },
        backend=backend,
        precision=precision,
    )


def curve_support(
    route: str,
    *,
    curve_kind: str,
    interpolation: str,
    extrapolation: str,
    differentiable: bool,
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe one exact curve construction or calibration route."""
    if not isinstance(differentiable, bool):
        raise TypeError("differentiable must be boolean.")
    return _common(
        "finance.curve-calibration",
        route,
        {
            "curve_kind": _text(curve_kind, "curve_kind"),
            "interpolation": _text(interpolation, "interpolation"),
            "extrapolation": _text(extrapolation, "extrapolation"),
            "differentiable": differentiable,
        },
        backend=backend,
        precision=precision,
    )


def valuation_support(
    route: str,
    *,
    product: str,
    model: str,
    pricing_law: str,
    exercise: str = "none",
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe a supported Q-law valuation route without implying P dynamics."""
    route_ = _text(route, "route")
    if route_ not in _VALUATION_ROUTES:
        raise ValueError(
            "valuation route must be analytic, fourier, pde, simulation, "
            "exercise, or bsde."
        )
    return _common(
        "finance.valuation",
        route_,
        {
            "product": _text(product, "product"),
            "model": _text(model, "model"),
            "pricing_law": _text(pricing_law, "pricing_law"),
            "exercise": _text(exercise, "exercise"),
        },
        backend=backend,
        precision=precision,
    )


def econometrics_support(
    route: str,
    *,
    estimator: str,
    target: str,
    physical_law: str,
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe a historical/statistical P-law estimation route."""
    return _common(
        "finance.econometrics",
        route,
        {
            "estimator": _text(estimator, "estimator"),
            "target": _text(target, "target"),
            "physical_law": _text(physical_law, "physical_law"),
        },
        backend=backend,
        precision=precision,
    )


def portfolio_support(
    route: str,
    *,
    objective: str,
    constraints: str,
    scenario_law: str,
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe one portfolio optimization or allocation route."""
    return _common(
        "finance.portfolio",
        route,
        {
            "objective": _text(objective, "objective"),
            "constraints": _text(constraints, "constraints"),
            "scenario_law": _text(scenario_law, "scenario_law"),
        },
        backend=backend,
        precision=precision,
    )


def exposure_xva_support(
    route: str,
    *,
    netting: str,
    collateral: str,
    default_model: str,
    wrong_way_risk: bool,
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe one exposure/XVA route with explicit legal-set assumptions."""
    if not isinstance(wrong_way_risk, bool):
        raise TypeError("wrong_way_risk must be boolean.")
    return _common(
        "finance.exposure-xva",
        route,
        {
            "netting": _text(netting, "netting"),
            "collateral": _text(collateral, "collateral"),
            "default_model": _text(default_model, "default_model"),
            "wrong_way_risk": wrong_way_risk,
        },
        backend=backend,
        precision=precision,
    )


def execution_support(
    route: str,
    *,
    fill_model: str,
    impact_model: str,
    control: str,
    topology: str,
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe one execution/control route; it is not a live-venue claim."""
    return _common(
        "finance.execution",
        route,
        {
            "fill_model": _text(fill_model, "fill_model"),
            "impact_model": _text(impact_model, "impact_model"),
            "control": _text(control, "control"),
            "topology": _text(topology, "topology"),
        },
        backend=backend,
        precision=precision,
    )


def advanced_finance_support(
    route: str,
    *,
    representation: str,
    law: str,
    candidate_ceiling: str = "candidate",
    backend: str = "jax",
    precision: str = "float64",
) -> SupportTuple:
    """Describe an explicitly candidate-only advanced finance route."""
    route_ = _text(route, "route")
    if route_ not in _ADVANCED_ROUTES:
        raise ValueError(
            "advanced route must be martingale-transport, rough, deep, "
            "operator, or tensor."
        )
    ceiling = _text(candidate_ceiling, "candidate_ceiling")
    if ceiling != "candidate":
        raise ValueError("Advanced finance support cannot exceed the candidate ceiling.")
    return _common(
        "finance.advanced",
        route_,
        {
            "representation": _text(representation, "representation"),
            "law": _text(law, "law"),
            "maturity": ceiling,
            "live_trading": False,
        },
        backend=backend,
        precision=precision,
    )


__all__ = [
    "advanced_finance_support",
    "curve_support",
    "econometrics_support",
    "execution_support",
    "exposure_xva_support",
    "market_resolution_support",
    "portfolio_support",
    "valuation_support",
]
