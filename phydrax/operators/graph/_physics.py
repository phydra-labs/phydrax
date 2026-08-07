#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from phydrax.domain import DomainFunction

from ._ops import graph_divergence, graph_incidence_laplacian


def _zero_if_none(value: Any | None, /) -> Any:
    return 0.0 if value is None else value


def graph_poisson_residual(
    u: DomainFunction,
    /,
    *,
    source: DomainFunction | Any | None = None,
    weight: DomainFunction | Any | None = None,
) -> DomainFunction:
    """Return the graph Poisson residual `div(grad(u)) - source`."""
    return graph_incidence_laplacian(u, weight=weight) - _zero_if_none(source)


def graph_diffusion_residual(
    u: DomainFunction,
    /,
    *,
    source: DomainFunction | Any | None = None,
    diffusivity: DomainFunction | Any | None = None,
) -> DomainFunction:
    """Return the conservative graph diffusion residual."""
    return graph_poisson_residual(u, source=source, weight=diffusivity)


def graph_conservation_residual(
    flux: DomainFunction,
    /,
    *,
    source: DomainFunction | Any | None = None,
) -> DomainFunction:
    """Return the graph conservation residual `div(flux) - source`."""
    return graph_divergence(flux) - _zero_if_none(source)


def graph_advection_diffusion_residual(
    u: DomainFunction,
    /,
    *,
    advective_flux: DomainFunction | None = None,
    source: DomainFunction | Any | None = None,
    diffusivity: DomainFunction | Any | None = None,
) -> DomainFunction:
    """Return `div(advective_flux) + div(diffusivity * grad(u)) - source`."""
    residual = graph_incidence_laplacian(u, weight=diffusivity)
    if advective_flux is not None:
        residual = residual + graph_divergence(advective_flux)
    return residual - _zero_if_none(source)


def graph_heat_residual(
    u_next: DomainFunction,
    u_current: DomainFunction,
    /,
    *,
    dt: float,
    source: DomainFunction | Any | None = None,
    diffusivity: DomainFunction | Any | None = None,
) -> DomainFunction:
    """Return an implicit-Euler heat residual on a graph."""
    step = (u_next - u_current) / float(dt)
    diffusion = graph_incidence_laplacian(u_next, weight=diffusivity)
    return step - diffusion - _zero_if_none(source)


def graph_euler_residual(
    u_next: DomainFunction,
    u_current: DomainFunction,
    vector_field: Callable[[DomainFunction], DomainFunction],
    /,
    *,
    dt: float,
) -> DomainFunction:
    """Return `(u_next - u_current) / dt - vector_field(u_current)`."""
    return (u_next - u_current) / float(dt) - vector_field(u_current)


__all__ = [
    "graph_advection_diffusion_residual",
    "graph_conservation_residual",
    "graph_diffusion_residual",
    "graph_euler_residual",
    "graph_heat_residual",
    "graph_poisson_residual",
]
