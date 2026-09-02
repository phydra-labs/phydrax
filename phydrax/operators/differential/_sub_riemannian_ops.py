#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Literal

import jax.numpy as jnp

import phydrax.ein as ein
from phydrax.domain import AbstractGeometry, DomainFunction

from ..._strict import StrictModule
from ...metrix import HorizontalCometric, sub_laplacian as _sub_laplacian, VolumeDensity
from ._domain_ops import _factor_and_dim, _resolve_var, grad


def _horizontal_contract(
    function: DomainFunction,
    cometric: HorizontalCometric,
    var: str | None,
    /,
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(cometric, HorizontalCometric):
        raise TypeError("cometric must be a HorizontalCometric.")
    variable = _resolve_var(function, var)
    _, dimension = _factor_and_dim(function, variable)
    if not isinstance(function.domain.factor(variable), AbstractGeometry):
        raise ValueError("Horizontal operators require a geometry variable.")
    if dimension != cometric.chart.dimension:
        raise ValueError(
            f"Horizontal chart dimension {cometric.chart.dimension} does not match "
            f"domain variable {variable!r} dimension {dimension}."
        )
    deps = tuple(
        label
        for label in function.domain.labels
        if label == variable or label in function.deps
    )
    return variable, deps


class _HorizontalGradientCallable(StrictModule):
    differential: DomainFunction
    cometric: HorizontalCometric
    differential_positions: tuple[int, ...]
    coordinate_position: int

    def __init__(
        self,
        differential: DomainFunction,
        cometric: HorizontalCometric,
        deps: tuple[str, ...],
        variable: str,
        /,
    ):
        positions = {label: position for position, label in enumerate(deps)}
        self.differential = differential
        self.cometric = cometric
        self.differential_positions = tuple(
            positions[label] for label in differential.deps
        )
        self.coordinate_position = positions[variable]

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        differential = jnp.asarray(
            self.differential.func(
                *[args[position] for position in self.differential_positions],
                key=key,
                **kwargs,
            )
        )
        return ein.contract(
            "...ij,...j->...i",
            self.cometric(args[self.coordinate_position]),
            differential,
        )


class _SubLaplacianCallable(StrictModule):
    function: DomainFunction
    cometric: HorizontalCometric
    density: VolumeDensity | None
    argument_positions: tuple[int, ...]
    variable_argument: int
    coordinate_position: int

    def __init__(
        self,
        function: DomainFunction,
        cometric: HorizontalCometric,
        density: VolumeDensity | None,
        deps: tuple[str, ...],
        variable: str,
        /,
    ):
        positions = {label: position for position, label in enumerate(deps)}
        self.function = function
        self.cometric = cometric
        self.density = density
        self.argument_positions = tuple(positions[label] for label in function.deps)
        self.variable_argument = function.deps.index(variable)
        self.coordinate_position = positions[variable]

    def __call__(self, *args: Any, key=None, **kwargs: Any):
        def field(coordinates):
            function_args = list(args[position] for position in self.argument_positions)
            function_args[self.variable_argument] = coordinates
            return self.function.func(*function_args, key=key, **kwargs)

        return _sub_laplacian(
            field,
            self.cometric,
            args[self.coordinate_position],
            density=self.density,
        )


def horizontal_grad(
    function: DomainFunction,
    cometric: HorizontalCometric,
    /,
    *,
    var: str | None = None,
    mode: Literal["reverse", "forward"] = "reverse",
    ad_engine: Literal["auto", "reverse", "forward", "jvp"] = "auto",
) -> DomainFunction:
    """Apply a horizontal cometric to a DomainFunction differential."""
    variable, deps = _horizontal_contract(function, cometric, var)
    differential = grad(
        function,
        var=variable,
        mode=mode,
        ad_engine=ad_engine,
    )
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_HorizontalGradientCallable(
            differential,
            cometric,
            deps,
            variable,
        ),
        metadata=differential.metadata,
    )


def sub_laplacian(
    function: DomainFunction,
    cometric: HorizontalCometric,
    /,
    *,
    var: str | None = None,
    density: VolumeDensity | None = None,
) -> DomainFunction:
    """Apply the density-weighted horizontal sub-Laplacian."""
    variable, deps = _horizontal_contract(function, cometric, var)
    if variable not in function.deps:
        raise ValueError(f"Function must depend on horizontal variable {variable!r}.")
    if density is not None and not isinstance(density, VolumeDensity):
        raise TypeError("density must be a VolumeDensity or None.")
    return DomainFunction(
        domain=function.domain,
        deps=deps,
        func=_SubLaplacianCallable(
            function,
            cometric,
            density,
            deps,
            variable,
        ),
        metadata=function.metadata,
    )


__all__ = ["horizontal_grad", "sub_laplacian"]
