"""Independent cylinder conduction and Lamé/GPS references on the actual FE forms."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax import config as pxconfig
from phydrax.applications.nuclear_fuel._forms import (
    deformation,
    mechanical_volume_residual,
    thermal_volume_residual,
)
from phydrax.applications.nuclear_fuel._fuel_constitutive import (
    multiplicative_hencky_stress,
)
from phydrax.applications.nuclear_fuel._geometry import fuel_rod_geometry
from phydrax.linalg import GMRES, LinearSolvePolicy, TolerancePolicy
from phydrax.nonlinear import NewtonKrylov, NonlinearSystemProblem, NonlinearTermination
from phydrax.operators.mechanics import finite_strain_kinematics
from tests._support.nuclear_fuel import (
    closed_end_thick_cylinder,
    solid_cylinder_temperature,
    synthetic_rod_specification,
)


_SINGLE_ZONE = (0.0, synthetic_rod_specification().rod.stack_length)


def _root(residual: Callable[[Array, None], Array], initial: Array, /) -> Array:
    result = NewtonKrylov(
        linear_policy=LinearSolvePolicy(
            GMRES(restart=64),
            tolerance=TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=128),
        )
    ).solve(
        NonlinearSystemProblem(residual),
        initial,
        termination=NonlinearTermination(
            absolute_residual=1e-10, relative_residual=1e-11, maximum_steps=20
        ),
    )
    assert bool(result.successful)
    return result.state


def test_constant_conductivity_cylinder_refines_to_independent_quadratic() -> None:
    surface_temperature, volumetric_heat, conductivity = 600.0, 1e8, 3.0
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        spec = synthetic_rod_specification()
        errors = []
        for cells in (8, 32):
            geometry = fuel_rod_geometry(
                spec,
                _SINGLE_ZONE,
                layered=True,
                fuel_cells=cells,
                cladding_cells=2,
                quadrature_order=3,
            )
            body = geometry.fuel
            source = jnp.full((body.site_count,), volumetric_heat, dtype=jnp.float64)
            tensor = jnp.broadcast_to(
                conductivity * jnp.eye(2, dtype=jnp.float64), (body.site_count, 2, 2)
            )
            zero = jnp.zeros((body.site_count,), dtype=jnp.float64)

            def residual(temperature: Array, args: None) -> Array:
                action = thermal_volume_residual(
                    body,
                    temperature,
                    zero,
                    zero,
                    tensor,
                    source,
                    jnp.asarray(1.0),
                )
                return action.at[-1].set(temperature[-1] - surface_temperature) / 1000.0

            values = _root(
                residual,
                jnp.full((body.node_count,), surface_temperature, dtype=jnp.float64),
            )
            exact = solid_cylinder_temperature(
                body.site_coordinates[:, 0],
                outer_radius=spec.pellet.radius,
                surface_temperature=surface_temperature,
                volumetric_heat=volumetric_heat,
                conductivity=conductivity,
            )
            error = jnp.sqrt(
                body.integrate((body.values(values) - exact) ** 2)
                / body.integrate(jnp.ones_like(exact))
            )
            errors.append(float(error))
        assert errors[1] < 0.4 * errors[0]
        assert errors[1] < 2.0


def test_lame_closed_end_tube_recovers_radial_displacement() -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        spec = synthetic_rod_specification()
        geometry = fuel_rod_geometry(
            spec,
            _SINGLE_ZONE,
            layered=True,
            fuel_cells=2,
            cladding_cells=32,
            quadrature_order=3,
        )
        body, ci, co = (
            geometry.cladding,
            geometry.cladding_interface,
            geometry.cladding_outer,
        )
        pi, po, young, nu = 1e4, 0.0, 1e11, 0.3
        exact, axial_strain = closed_end_thick_cylinder(
            np.asarray(body.nodes)[:, 0],
            inner_radius=spec.cladding.inner_radius,
            outer_radius=spec.cladding.outer_radius,
            inner_pressure=pi,
            outer_pressure=po,
            youngs_modulus=young,
            poisson_ratio=nu,
        )
        axial = jnp.asarray((axial_strain,), dtype=jnp.float64)

        def residual(radial: Array, args: None) -> Array:
            f = deformation(body, radial[:, None], axial)
            stress = multiplicative_hencky_stress(
                f,
                jnp.zeros((body.site_count, 3, 3), dtype=jnp.float64),
                jnp.asarray(young / (3 * (1 - 2 * nu))),
                jnp.asarray(young / (2 * (1 + nu))),
                finite_strain_kinematics(f),
            )
            action = mechanical_volume_residual(body, stress.first_piola)[:, 0]
            action -= ci.assemble(jnp.full(ci.weight.shape, pi, dtype=jnp.float64))
            action += co.assemble(jnp.full(co.weight.shape, po, dtype=jnp.float64))
            return action / young

        solved = _root(residual, jnp.zeros((body.node_count,), dtype=jnp.float64))
        np.testing.assert_allclose(solved, exact, rtol=3e-3, atol=1e-12)


@pytest.mark.parametrize(
    "zone_edges",
    [
        (0.1, 0.3),
        (0.0, 0.2, 0.1),
        (0.0, float("nan"), 0.3),
        ((0.0, 0.3), (0.0, 0.3)),
        (0.0,),
    ],
    ids=["offset", "decreasing", "nan", "rank-2", "single-edge"],
)
def test_geometry_refuses_invalid_zone_edges(
    zone_edges: tuple[float, ...] | tuple[tuple[float, float], ...],
) -> None:
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        spec = synthetic_rod_specification()
        with pytest.raises(ValueError, match="zone_edges must start at 0"):
            fuel_rod_geometry(
                spec,
                zone_edges,
                layered=True,
                fuel_cells=2,
                cladding_cells=2,
                quadrature_order=2,
            )
