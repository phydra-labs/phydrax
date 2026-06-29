#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Graph operators for graph-domain `DomainFunction`s."""

from ._ops import (
    degree,
    divergence,
    gradient,
    graph_div,
    graph_divergence,
    graph_grad,
    graph_gradient,
    graph_incidence_laplacian,
    graph_laplacian,
    GraphDivergenceSign,
    GraphFlow,
    GraphReduce,
    incidence_laplacian,
    neighbor_aggregate,
)
from ._physics import (
    graph_advection_diffusion_residual,
    graph_conservation_residual,
    graph_diffusion_residual,
    graph_euler_residual,
    graph_heat_residual,
    graph_poisson_residual,
)


__all__ = [
    "GraphDivergenceSign",
    "GraphFlow",
    "GraphReduce",
    "degree",
    "divergence",
    "graph_div",
    "graph_divergence",
    "graph_grad",
    "graph_gradient",
    "graph_advection_diffusion_residual",
    "graph_conservation_residual",
    "graph_diffusion_residual",
    "graph_euler_residual",
    "graph_heat_residual",
    "graph_incidence_laplacian",
    "graph_laplacian",
    "graph_poisson_residual",
    "gradient",
    "incidence_laplacian",
    "neighbor_aggregate",
]
