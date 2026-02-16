#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""
# Operators

Operators build PDE terms such as gradients, divergences, Laplacians, and
integrals. They are designed to work with `DomainFunction` objects and preserve
structure across dense and coord-separable batches.

## Families

- **Differential**: $\\nabla u$, $\\nabla \\cdot v$, $\\Delta u$, surface operators.
- **Integral**: $\\int_\\Omega u \\, d\\Omega$ and weighted variants.
- **Functional**: $\\|u\\|_p$, inner products, and averages.
- **Linear algebra**: determinants, traces, norms.

!!! example
    ```python
    import phydrax as phx

    geom = phx.domain.Square(center=(0.0, 0.0), side=2.0)

    @geom.Function("x")
    def u(x):
        return x[0] ** 2 + x[1] ** 2

    lap_u = phx.operators.laplacian(u, var="x")
    ```
"""

from . import (
    delay,
    differential,
    functional,
    linalg,
)

# Re-export objects from submodules
from .delay import (  # noqa: F401
    delay as delay_operator,
)
from .differential import (  # noqa: F401
    bilaplacian,
    caputo_time_fractional,
    caputo_time_fractional_dw,
    cauchy_from_pk2,
    cauchy_strain,
    cauchy_stress,
    curl,
    deformation_gradient,
    deviatoric_stress,
    directional_derivative,
    div,
    div_cauchy_stress,
    div_diag_k_grad,
    div_K_grad,
    div_k_grad,
    div_tensor,
    dt,
    dt_n,
    fractional_derivative_gl_mc,
    fractional_laplacian,
    grad,
    green_lagrange_strain,
    hessian,
    hydrostatic_pressure,
    hydrostatic_stress,
    laplace_beltrami,
    laplace_beltrami_divgrad,
    laplacian,
    linear_elastic_cauchy_stress_2d,
    linear_elastic_orthotropic_stress_2d,
    material_derivative,
    maxwell_stress,
    navier_stokes_divergence,
    navier_stokes_stress,
    neo_hookean_cauchy,
    neo_hookean_pk1,
    partial_n,
    partial_t,
    partial_x,
    partial_y,
    partial_z,
    pk1_from_pk2,
    riesz_fractional_derivative_gl_mc,
    strain_rate,
    strain_rate_magnitude,
    surface_curl_scalar,
    surface_curl_vector,
    surface_div,
    surface_grad,
    svk_pk2_stress,
    tangential_component,
    viscous_stress,
    von_mises_stress,
)
from .functional import (  # noqa: F401
    spatial_inner_product,
    spatial_l2_norm,
    spatial_lp_norm,
    spatial_mean,
)
from .integral import (  # noqa: F401
    build_ball_quadrature,
    build_quadrature,
    integral,
    integrate_boundary,
    integrate_interior,
    local_integral,
    local_integral_ball,
    mean,
    nonlocal_integral,
    spatial_integral,
    time_convolution,
)
from .linalg import (  # noqa: F401
    det,
    einsum,
    norm,
    trace,
)


__all__ = [
    # subpackages
    "delay",
    "differential",
    "functional",
    "linalg",
    # delay exports
    "delay_operator",
    # differential exports
    "cauchy_from_pk2",
    "cauchy_strain",
    "cauchy_stress",
    "curl",
    "deformation_gradient",
    "deviatoric_stress",
    "directional_derivative",
    "div",
    "div_cauchy_stress",
    "div_diag_k_grad",
    "div_K_grad",
    "div_k_grad",
    "dt",
    "dt_n",
    "grad",
    "hessian",
    "green_lagrange_strain",
    "hydrostatic_pressure",
    "hydrostatic_stress",
    "bilaplacian",
    "laplacian",
    "fractional_laplacian",
    "linear_elastic_cauchy_stress_2d",
    "linear_elastic_orthotropic_stress_2d",
    "material_derivative",
    "maxwell_stress",
    "navier_stokes_stress",
    "neo_hookean_cauchy",
    "neo_hookean_pk1",
    "partial_t",
    "partial_n",
    "partial_x",
    "partial_y",
    "partial_z",
    "pk1_from_pk2",
    "svk_pk2_stress",
    "viscous_stress",
    "von_mises_stress",
    "strain_rate",
    "strain_rate_magnitude",
    "div_tensor",
    "navier_stokes_divergence",
    "laplace_beltrami",
    "laplace_beltrami_divgrad",
    "surface_curl_scalar",
    "surface_curl_vector",
    "surface_div",
    "surface_grad",
    "tangential_component",
    "fractional_derivative_gl_mc",
    "riesz_fractional_derivative_gl_mc",
    "caputo_time_fractional",
    "caputo_time_fractional_dw",
    # integral exports
    "build_ball_quadrature",
    "build_quadrature",
    "integrate_boundary",
    "integrate_interior",
    "integral",
    "local_integral",
    "local_integral_ball",
    "mean",
    "nonlocal_integral",
    "spatial_integral",
    "time_convolution",
    # functional exports
    "spatial_inner_product",
    "spatial_l2_norm",
    "spatial_lp_norm",
    "spatial_mean",
    # linalg exports
    "det",
    "einsum",
    "norm",
    "trace",
]
