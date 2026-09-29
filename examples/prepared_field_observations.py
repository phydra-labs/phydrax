#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Repeated sensor observations of a discretized field through one prepared query.

The sensor locations are located once. Every refreshed coefficient state
(here a decaying cubic temperature profile) is observed by the same prepared
route without relocating, the transpose pulls sensor residuals back to the
coefficient space for adjoint workflows, and a sensor outside the support is
refused by complete coverage and reported and excluded by masked coverage.

The same observation code then reads the temperature from substituted
methods (finite-difference nodal values with declared cubic B-spline
interpolation and a global Chebyshev spectral field): only the owner's
reconstruction changes, never the observation workflow.
"""

from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import PreparedFieldReconstruction, SimplicialLocationPolicy
from phydrax.discretization.fem import prepare_finite_element_field_reconstruction
from phydrax.discretization.finite_difference import (
    BSplineGridInterpolation,
    prepare_finite_difference_field_reconstruction,
)
from phydrax.discretization.spectral import prepare_spectral_field_reconstruction


jax.config.update("jax_enable_x64", True)


def _mesh(resolution: int) -> phx.discretization.CellMesh:
    vertices = np.asarray(
        [
            (i / resolution, j / resolution)
            for j in range(resolution + 1)
            for i in range(resolution + 1)
        ]
    )
    triangles = []
    for j in range(resolution):
        for i in range(resolution):
            corner = j * (resolution + 1) + i
            triangles.append((corner, corner + 1, corner + resolution + 2))
            triangles.append((corner, corner + resolution + 2, corner + resolution + 1))
    return phx.discretization.CellMesh(
        jnp.asarray(vertices),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", jnp.asarray(np.asarray(triangles, np.int32))
            ),
        ),
    )


def _temperature(x: np.ndarray, y: np.ndarray, time: float) -> np.ndarray:
    return np.exp(-time) * (1.0 + x**3 - 2.0 * x * y**2 + 0.5 * y)


def _temperature_dx(x: np.ndarray, y: np.ndarray, time: float) -> np.ndarray:
    return np.exp(-time) * (3.0 * x**2 - 2.0 * y**2)


type StateAt = Callable[[float], Array]


def _finite_difference() -> tuple[PreparedFieldReconstruction, StateAt]:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(17, periodic=False, endpoint=True)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray(((0.0, 0.0), (1.0, 1.0))))
    request = phx.discretization.DerivativeRequest(
        "dx", grid, "x", derivative_order=1, accuracy_order=2, boundary="one_sided"
    )
    discretization = phx.discretization.FiniteDifferencePlan(
        grid, (request,), field_name="temperature"
    ).prepare()
    x, y = (np.asarray(axis) for axis in grid.primary_entity_layout.coordinates_by_axis)
    reconstruction = prepare_finite_difference_field_reconstruction(
        discretization, interpolation=BSplineGridInterpolation(3)
    )
    return reconstruction, lambda time: jnp.asarray(
        _temperature(x[:, None], y[None, :], time)
    )


def _spectral() -> tuple[PreparedFieldReconstruction, StateAt]:
    axes = phx.discretization
    space = axes.TensorSpectralPlan(
        (axes.ChebyshevBasisPlan(6), axes.ChebyshevBasisPlan(6)),
        axis_names=("x", "y"),
        field_name="temperature",
    ).prepare((axes.AxisDomain.interval(0.0, 1.0), axes.AxisDomain.interval(0.0, 1.0)))
    x, y = (np.asarray(axis.nodes) for axis in space.axes)
    return prepare_spectral_field_reconstruction(space), lambda time: space.project(
        jnp.asarray(_temperature(x[:, None], y[None, :], time))
    )


def _substituted_errors(
    sensors: np.ndarray, observe: Callable[..., Array]
) -> dict[str, float]:
    """Observe the same sensors of the same temperature through other methods."""
    errors = {}
    for name, (reconstruction, state_at) in (
        ("finite_difference", _finite_difference()),
        ("spectral", _spectral()),
    ):
        values = reconstruction.prepare_query(sensors)
        gradients = reconstruction.prepare_query(sensors, derivative=(1, 0))
        worst = 0.0
        for time in np.linspace(0.0, 2.0, 5):
            state = state_at(time)
            exact = _temperature(sensors[:, 0], sensors[:, 1], time)
            slope = _temperature_dx(sensors[:, 0], sensors[:, 1], time)
            worst = max(
                worst,
                float(np.max(np.abs(np.asarray(observe(values, state)) - exact))),
                float(np.max(np.abs(np.asarray(observe(gradients, state)) - slope))),
            )
        errors[f"{name}_max_error"] = worst
    return errors


def run() -> dict[str, float | int | bool | str]:
    discretization = phx.discretization.FiniteElementPlan(
        _mesh(8),
        phx.discretization.FiniteElementFieldSpec(
            "temperature", phx.discretization.lagrange_element("triangle", 3)
        ),
    ).prepare()
    reconstruction = prepare_finite_element_field_reconstruction(
        discretization,
        "temperature",
        location_policy=SimplicialLocationPolicy(64, 16, 1),
    )
    sensors = np.asarray(
        ((0.15, 0.2), (0.52, 0.47), (0.83, 0.31), (0.27, 0.91), (0.66, 0.74))
    )
    values = reconstruction.prepare_query(sensors)
    gradients = reconstruction.prepare_query(sensors, derivative=(1, 0))
    observe = eqx.filter_jit(lambda query, state: query.apply(state))

    nodes = np.asarray(discretization.dof_maps[0].dof_coordinates)
    worst_value = 0.0
    worst_gradient = 0.0
    for time in np.linspace(0.0, 2.0, 5):
        state = jnp.asarray(_temperature(nodes[:, 0], nodes[:, 1], time))
        observed = np.asarray(observe(values, state))
        slope = np.asarray(observe(gradients, state))
        exact = _temperature(sensors[:, 0], sensors[:, 1], time)
        exact_slope = _temperature_dx(sensors[:, 0], sensors[:, 1], time)
        worst_value = max(worst_value, float(np.max(np.abs(observed - exact))))
        worst_gradient = max(worst_gradient, float(np.max(np.abs(slope - exact_slope))))
        print(f"t={time:.2f} sensors={np.round(observed, 6)}")

    residual = jnp.asarray((0.1, -0.2, 0.05, 0.3, -0.1))
    state = jnp.asarray(_temperature(nodes[:, 0], nodes[:, 1], 0.5))
    duality = values.duality_evidence(state, residual)
    pulled_back = values.transpose(residual)

    extended = np.concatenate((sensors, ((1.4, 0.5),)))
    masked = reconstruction.prepare_query(extended, coverage="masked")
    rejected = np.asarray(masked.evidence.status) != int(
        phx.discretization.FieldQueryStatus.VALID
    )

    summary: dict[str, float | int | bool | str] = {
        "coefficients": int(reconstruction.coefficient_shape[0]),
        "sensors": int(sensors.shape[0]),
        "max_value_error": worst_value,
        "max_gradient_error": worst_gradient,
        "duality_valid": bool(duality.valid),
        "pullback_nonzero_rows": int(np.count_nonzero(np.asarray(pulled_back))),
        "masked_admitted": int(masked.admitted_count),
        "masked_complete": bool(masked.complete),
        "rejected_sensor_status": ",".join(
            phx.discretization.FieldQueryStatus(int(code)).name
            for code in np.asarray(masked.evidence.status)[rejected]
        ),
    }
    summary.update(_substituted_errors(sensors, observe))
    return summary


if __name__ == "__main__":
    for name, value in run().items():
        print(f"{name}: {value}")
