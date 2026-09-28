#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Refinement campaigns for the surface thin-film routes.

Campaigns (closed-form references, see ``docs/guides_surface_thin_films.md``):

- planar capillary levelling rate ``sigma h^3 k^4 / (3 mu)`` under mesh
  refinement at a time step small against the decay time;
- spherical-harmonic (l = 2) decay ``sigma h^3 lambda (lambda - 2/R^2) / (3 mu)``
  on icosphere refinement, with the l = 1 neutrality defect;
- Marangoni wave quarter period for ``c_M^2 = 2 E_s / (rho h)``;
- exact discrete geometric conservation (closed-form dual-area rate) for a
  tangential slide on a fixed ruled surface, spherical scaling and arbitrary
  non-homothetic motion of a sphere and a wavy plane, against the midpoint
  rule and Gauss--Legendre integrals of the instantaneous rate, plus the
  fail-closed status of a far-translated motion;
- fixed-sphere gravity/Marangoni rest profile (Huang et al. 2020) under
  icosphere refinement after relaxation with linear air drag;
- volume/surfactant conservation residuals, statuses, preparation and warmed
  step times per resolution, and the stored bytes of each lubrication solve's
  symbolic sparse-LU plan against its factor nonzeros.

The full campaign runs planar 10/20/40 and icosphere levels 1-4 (up to 5124
lubrication unknowns). Film Newton solves reuse the redesigned pattern-only
sparse-LU plan. Gravity relaxation compiles one
bounded recurrence and reuses one numeric factor until its steady-state gate.
"""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_synchronized,
)


it = phx.interfacial_transport
_STEP = eqx.filter_jit(lambda prepared, state, step_size: prepared.step(state, step_size))
_GRAVITY_WINDOW_STEPS = 20
_GRAVITY_STEADY_WINDOWS = 2


@eqx.filter_jit
def _relax_gravity(
    prepared: Any,
    state: Any,
    step_size: jax.Array,
    minimum_steps: int,
    maximum_steps: int,
    profile_tolerance: jax.Array,
    velocity_tolerance: jax.Array,
    drainage_speed: jax.Array,
) -> tuple[Any, ...]:
    """Relax in one compiled recurrence until two macroscopic steady windows."""
    area = prepared.plan.surface.vertex_area
    status_seen = jnp.zeros(
        (max(int(status) for status in it.FilmStepStatus) + 1,), dtype=jnp.bool_
    )
    zero = jnp.zeros((), dtype=jnp.int32)
    infinity = jnp.asarray(jnp.inf, dtype=jnp.float64)
    factorization = prepared._elastic_factorization(state, step_size)
    surfactant_factorization = (
        None
        if prepared.plan.kinetics is not None
        else prepared._surfactant_factorization(state, step_size)
    )
    initial_speed = jnp.max(jnp.linalg.norm(prepared.velocity(state), axis=1))
    initial = (
        zero,
        state,
        state,
        initial_speed,
        status_seen,
        zero,
        zero,
        zero,
        infinity,
        infinity,
        infinity,
        jnp.zeros((), dtype=jnp.float64),
        jnp.zeros((), dtype=jnp.float64),
        zero,
    )

    def condition(carry: tuple[Any, ...]) -> jax.Array:
        (
            count,
            _,
            _,
            _,
            statuses,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            steady_windows,
        ) = carry
        successful = ~jnp.any(statuses[1:])
        return (
            (count < maximum_steps)
            & successful
            & ((count < minimum_steps) | (steady_windows < _GRAVITY_STEADY_WINDOWS))
        )

    def body(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        (
            count,
            current,
            reference,
            reference_speed,
            statuses,
            total_iterations,
            maximum_iterations,
            _,
            last_thickness_change,
            last_concentration_change,
            last_velocity_change,
            maximum_residual,
            _,
            steady_windows,
        ) = carry
        result = prepared._step(
            current, step_size, factorization, surfactant_factorization
        )
        next_state = result.state
        next_count = count + 1
        next_speed = jnp.max(jnp.linalg.norm(prepared.velocity(next_state), axis=1))
        thickness = next_state.liquid_volume_m3 / area
        reference_thickness = reference.liquid_volume_m3 / area
        concentration = next_state.surfactant_amount_mol / area
        reference_concentration = reference.surfactant_amount_mol / area
        thickness_change = jnp.sqrt(
            jnp.sum(area * (thickness - reference_thickness) ** 2)
            / jnp.sum(area * thickness**2)
        )
        concentration_change = jnp.sqrt(
            jnp.sum(area * (concentration - reference_concentration) ** 2)
            / jnp.sum(area * concentration**2)
        )
        velocity_change = jnp.abs(next_speed - reference_speed) / drainage_speed
        checkpoint = next_count % _GRAVITY_WINDOW_STEPS == 0
        steady = (
            (thickness_change <= profile_tolerance)
            & (concentration_change <= profile_tolerance)
            & (velocity_change <= velocity_tolerance)
            & result.accepted
        )
        next_steady_windows = jnp.where(
            checkpoint,
            jnp.where(steady, steady_windows + 1, 0),
            steady_windows,
        )
        next_reference = jax.tree.map(
            lambda old, new: jnp.where(checkpoint, new, old), reference, next_state
        )
        next_reference_speed = jnp.where(checkpoint, next_speed, reference_speed)
        next_statuses = statuses.at[result.status].set(True)
        iterations = result.evidence.nonlinear_iterations
        residual = result.evidence.nonlinear_residual_norm
        return (
            next_count,
            next_state,
            next_reference,
            next_reference_speed,
            next_statuses,
            total_iterations + iterations,
            jnp.maximum(maximum_iterations, iterations),
            iterations,
            jnp.where(checkpoint, thickness_change, last_thickness_change),
            jnp.where(checkpoint, concentration_change, last_concentration_change),
            jnp.where(checkpoint, velocity_change, last_velocity_change),
            jnp.maximum(maximum_residual, residual),
            residual,
            next_steady_windows,
        )

    return jax.lax.while_loop(condition, body, initial)


def _planar_grid(nx: int, ny: int, length_x: float, length_y: float) -> Any:
    xs = np.linspace(0.0, length_x, nx + 1)
    ys = np.linspace(0.0, length_y, ny + 1)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    index = np.arange((nx + 1) * (ny + 1)).reshape((nx + 1, ny + 1))
    lower = np.stack((index[:-1, :-1], index[1:, :-1], index[1:, 1:]), -1).reshape(
        (-1, 3)
    )
    upper = np.stack((index[:-1, :-1], index[1:, 1:], index[:-1, 1:]), -1).reshape(
        (-1, 3)
    )
    return phx.geometry.TriangleMesh(
        vertices, np.concatenate((lower, upper)).astype(np.int32)
    )


def _icosphere(level: int) -> Any:
    ratio = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, ratio, 0),
            (1, ratio, 0),
            (-1, -ratio, 0),
            (1, -ratio, 0),
            (0, -1, ratio),
            (0, 1, ratio),
            (0, -1, -ratio),
            (0, 1, -ratio),
            (ratio, 0, -1),
            (ratio, 0, 1),
            (-ratio, 0, -1),
            (-ratio, 0, 1),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        midpoints = vertices[unique[:, 0]] + vertices[unique[:, 1]]
        midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
        count = faces.shape[0]
        ab, bc, ca = (
            vertices.shape[0] + inverse[k * count : (k + 1) * count] for k in range(3)
        )
        a, b, c = faces.T
        faces = np.concatenate(
            (
                np.stack((a, ab, ca), 1),
                np.stack((b, bc, ab), 1),
                np.stack((c, ca, bc), 1),
                np.stack((ab, bc, ca), 1),
            )
        )
        vertices = np.concatenate((vertices, midpoints))
    return phx.geometry.TriangleMesh(vertices, faces.astype(np.int32))


def _amplitude(area: np.ndarray, values: np.ndarray, mode: np.ndarray) -> float:
    mean = np.sum(area * values) / np.sum(area)
    return float(np.sum(area * (values - mean) * mode) / np.sum(area * mode**2))


def _timed_prepare(plan: Any) -> tuple[Any, float]:
    started = perf_counter()
    prepared = plan.prepare()
    return prepared, 1e3 * (perf_counter() - started)


def _factor_storage(prepared: Any) -> dict[str, int | float]:
    """Return a film solver's sparse-LU plan size against its factor nonzeros."""
    plan = prepared.solver.factorization
    arrays = {id(x): x for x in jax.tree.leaves(plan) if isinstance(x, jax.Array)}
    stored = sum(leaf.size * leaf.dtype.itemsize for leaf in arrays.values())
    nonzeros = plan.factor_indices.size
    return {
        "unknowns": plan.shape[0],
        "factor_nonzeros": nonzeros,
        "factor_plan_bytes": stored,
        "factor_plan_bytes_per_nonzero": stored / nonzeros,
    }


def _timed_step(prepared: Any, state: Any, step_size: float) -> tuple[Any, float, float]:
    started = perf_counter()
    result = _STEP(prepared, state, step_size)
    jax.block_until_ready(result.state)
    first = 1e3 * (perf_counter() - started)
    started = perf_counter()
    warmed = _STEP(prepared, state, step_size)
    jax.block_until_ready(warmed.state)
    return result, first, 1e3 * (perf_counter() - started)


def planar_levelling(resolutions: tuple[int, ...]) -> list[dict[str, Any]]:
    thickness, wavenumber = 0.1, 2.0 * np.pi
    rate = thickness**3 * wavenumber**4 / 3.0
    step_size = 1e-3 / rate
    rows = []
    for count in resolutions:
        surface = it.prepare_film_surface(_planar_grid(count, count, 1.0, 1.0))
        prepared, prepare_ms = _timed_prepare(
            it.SurfaceLubricationPlan(
                surface,
                mobility_law="one-sided-substrate",
                surface_tension_n_m=1.0,
                viscosity_pa_s=1.0,
            )
        )
        x = np.asarray(surface.coordinates[:, 0])
        mode = np.cos(wavenumber * x)
        area = np.asarray(surface.vertex_area)
        state = prepared.initial_state(thickness * (1.0 + 1e-6 * mode))
        result, first_ms, warm_ms = _timed_step(prepared, state, step_size)
        ratio = _amplitude(area, np.asarray(prepared.thickness(result.state)), mode) / (
            _amplitude(area, np.asarray(prepared.thickness(state)), mode)
        )
        measured = (1.0 / ratio - 1.0) / step_size
        rows.append(
            {
                "vertices": surface.topology.num_vertices,
                "measured_rate": measured,
                "reference_rate": rate,
                "relative_error": abs(measured - rate) / rate,
                "status": int(result.status),
                "volume_residual_m3": float(result.evidence.liquid_volume_residual_m3),
                "nonlinear_iterations": int(result.evidence.nonlinear_iterations),
                "prepare_ms": prepare_ms,
                "first_step_ms": first_ms,
                "warm_step_ms": warm_ms,
                **_factor_storage(prepared),
            }
        )
    return rows


def sphere_decay(levels: tuple[int, ...]) -> list[dict[str, Any]]:
    thickness = 0.1
    reference = 8.0 * thickness**3
    step_size = 1e-3 / reference
    rows = []
    for level in levels:
        surface = it.prepare_film_surface(_icosphere(level))
        prepared, prepare_ms = _timed_prepare(
            it.SurfaceLubricationPlan(
                surface,
                mobility_law="one-sided-substrate",
                surface_tension_n_m=1.0,
                viscosity_pa_s=1.0,
            )
        )
        z = np.asarray(surface.coordinates[:, 2])
        area = np.asarray(surface.vertex_area)
        quadrupole = 3.0 * z**2 - 1.0
        state = prepared.initial_state(thickness * (1.0 + 1e-5 * quadrupole))
        result, first_ms, warm_ms = _timed_step(prepared, state, step_size)
        ratio = _amplitude(
            area, np.asarray(prepared.thickness(result.state)), quadrupole
        ) / (_amplitude(area, np.asarray(prepared.thickness(state)), quadrupole))
        dipole_state = prepared.initial_state(thickness * (1.0 + 1e-5 * z))
        dipole = _STEP(prepared, dipole_state, step_size)
        dipole_ratio = _amplitude(
            area, np.asarray(prepared.thickness(dipole.state)), z
        ) / (_amplitude(area, np.asarray(prepared.thickness(dipole_state)), z))
        measured = (1.0 / ratio - 1.0) / step_size
        rows.append(
            {
                "vertices": surface.topology.num_vertices,
                "conductance_admissible": bool(surface.evidence.conductance_admissible),
                "minimum_edge_conductance": float(
                    surface.evidence.minimum_edge_conductance
                ),
                "measured_rate": measured,
                "reference_rate": reference,
                "relative_error": abs(measured - reference) / reference,
                "dipole_rate_over_quadrupole": (1.0 / dipole_ratio - 1.0)
                / step_size
                / reference,
                "status": int(result.status),
                "volume_residual_m3": float(result.evidence.liquid_volume_residual_m3),
                "prepare_ms": prepare_ms,
                "first_step_ms": first_ms,
                "warm_step_ms": warm_ms,
                **_factor_storage(prepared),
            }
        )
    return rows


def marangoni_wave(resolutions: tuple[int, ...]) -> list[dict[str, Any]]:
    length, thickness, concentration = 1e-2, 1e-6, 2e-6
    law = it.LangmuirSurfactantLaw(0.072, 298.15, 4e-6)
    speed = np.sqrt(
        2.0 * float(law.gibbs_elasticity(concentration)) / (1000.0 * thickness)
    )
    wavenumber = 2.0 * np.pi / length
    period = 2.0 * np.pi / (speed * wavenumber)
    step_size = period / 400.0
    rows = []
    for count in resolutions:
        surface = it.prepare_film_surface(_planar_grid(count, 2, length, length / 32))
        prepared, prepare_ms = _timed_prepare(
            it.SurfacePlugFlowPlan(surface, law, density_kg_m3=1000.0)
        )
        x = np.asarray(surface.coordinates[:, 0])
        mode = np.cos(wavenumber * x)
        area = np.asarray(surface.vertex_area)
        state = prepared.initial_state(thickness, concentration * (1.0 + 1e-5 * mode))
        initial_surfactant = float(state.total_surfactant_mol())
        previous = _amplitude(area, np.asarray(state.surfactant_amount_mol) / area, mode)
        crossing = float("nan")
        statuses = set()
        started = perf_counter()
        for index in range(1, 250):
            result = _STEP(prepared, state, step_size)
            statuses.add(int(result.status))
            state = result.state
            current = _amplitude(
                area, np.asarray(state.surfactant_amount_mol) / area, mode
            )
            if current < 0.0:
                crossing = (index - 1 + previous / (previous - current)) * step_size
                break
            previous = current
        rows.append(
            {
                "vertices": surface.topology.num_vertices,
                "measured_quarter_period_s": crossing,
                "reference_quarter_period_s": 0.25 * period,
                "relative_error": abs(crossing - 0.25 * period) / (0.25 * period),
                "marangoni_courant_number": float(
                    result.plug_flow.marangoni_courant_number
                ),
                "surfactant_residual_mol": float(state.total_surfactant_mol())
                - initial_surfactant,
                "statuses": sorted(statuses),
                "prepare_ms": prepare_ms,
                "campaign_ms": 1e3 * (perf_counter() - started),
            }
        )
    return rows


def _sphere_field(points: np.ndarray) -> np.ndarray:
    """A smooth non-homothetic, non-rigid velocity field with normal parts."""
    x, y, z = points.T
    return np.stack(
        (
            0.3 * x * y + 0.05 * np.sin(3.0 * y),
            0.2 * z**2 - 0.1 * x,
            0.15 * np.sin(2.0 * x) + 0.1 * y * z,
        ),
        axis=1,
    )


def _midpoint_gcl(motion: Any) -> float:
    """Return the relative GCL residual of the midpoint-rule dual-area rate."""
    operators = motion.midpoint.operators
    divergence = jnp.sum(
        motion.mesh_velocity[operators.faces] * operators.basis_gradients, axis=(1, 2)
    )
    rate = (
        jnp.zeros_like(motion.source.vertex_area)
        .at[operators.faces.reshape((-1,))]
        .add(jnp.repeat(operators.face_area * divergence / 3.0, 3))
    )
    residual = motion.evidence.area_change_m2 - motion.step_size * rate
    return float(jnp.max(jnp.abs(residual) / motion.target.vertex_area))


def _quadrature_rate_defect(motion: Any, nodes: int) -> float:
    """Gauss--Legendre time integral of the instantaneous lumped ESFEM rate
    ``sum_f |f(t)| div_f(w)(t) / 3`` against the closed-form ``area_rate``."""
    abscissae, weights = np.polynomial.legendre.leggauss(nodes)
    step = float(motion.step_size)
    integral = jnp.zeros_like(motion.source.vertex_area)
    for abscissa, weight in zip(abscissae, weights, strict=True):
        time = 0.5 * step * (abscissa + 1.0)
        instant = it.prepare_film_surface(
            phx.geometry.TriangleMesh(
                np.asarray(motion.source.coordinates + time * motion.mesh_velocity),
                np.asarray(motion.source.operators.faces),
            )
        ).operators
        divergence = jnp.sum(
            motion.mesh_velocity[instant.faces] * instant.basis_gradients, axis=(1, 2)
        )
        integral = integral.at[instant.faces.reshape((-1,))].add(
            0.5 * step * weight * jnp.repeat(instant.face_area * divergence / 3.0, 3)
        )
    exact = motion.step_size * motion.area_rate_m2_s
    return float(jnp.max(jnp.abs(integral - exact) / motion.target.vertex_area))


def geometric_conservation() -> dict[str, Any]:
    planar = it.prepare_film_surface(_planar_grid(24, 24, 1.0, 1.0))
    points = np.asarray(planar.coordinates)
    wavy = points.copy()
    wavy[:, 2] = 0.1 * np.sin(2.0 * np.pi * points[:, 0])
    ruled = it.prepare_film_surface(
        phx.geometry.TriangleMesh(wavy, np.asarray(planar.operators.faces))
    )
    # Tangential mesh motion of a fixed surface: vertices slide along the
    # straight x-grid lines of the ruled surface z = 0.1 sin(2 pi x).
    shift = 0.02 * np.sin(np.pi * points[:, 1]) * (1.0 + 0.5 * np.cos(3.0 * points[:, 0]))
    slide = it.SurfaceMeshMotion(
        ruled, wavy + np.stack((0.0 * shift, shift, 0.0 * shift), 1), 0.1
    )
    uniform = slide.transport(
        ruled.vertex_area, slide.material_velocity(jnp.zeros_like(slide.mesh_velocity))
    )
    sphere = it.prepare_film_surface(_icosphere(3))
    sphere_points = np.asarray(sphere.coordinates)
    scaling = it.SurfaceMeshMotion(sphere, 1.05 * sphere.coordinates, 0.1)
    wobble = it.SurfaceMeshMotion(
        sphere,
        sphere.coordinates * (1.0 + 0.05 * np.asarray(sphere.coordinates[:, 2:3]) ** 2),
        0.1,
    )
    arbitrary = it.SurfaceMeshMotion(
        sphere, sphere_points + 0.2 * _sphere_field(sphere_points), 0.2
    )
    wavy_arbitrary = it.SurfaceMeshMotion(ruled, wavy + 0.2 * _sphere_field(wavy), 0.2)
    lagrangian = arbitrary.transport(sphere.vertex_area, arbitrary.mesh_velocity)
    far = it.SurfaceMeshMotion(sphere, sphere.coordinates + 1e9, 1.0)
    return {
        "ruled_slide_relative_gcl": float(slide.evidence.maximum_relative_gcl_residual),
        "ruled_slide_uniform_defect": float(
            jnp.max(jnp.abs(uniform.content / slide.target.vertex_area - 1.0))
        ),
        "ruled_slide_status": int(uniform.status),
        "sphere_scaling_relative_gcl": float(
            scaling.evidence.maximum_relative_gcl_residual
        ),
        "sphere_wobble_relative_gcl": float(
            wobble.evidence.maximum_relative_gcl_residual
        ),
        "sphere_wobble_midpoint_rule_relative_gcl": _midpoint_gcl(wobble),
        "sphere_arbitrary_relative_gcl": float(
            arbitrary.evidence.maximum_relative_gcl_residual
        ),
        "sphere_arbitrary_midpoint_rule_relative_gcl": _midpoint_gcl(arbitrary),
        "sphere_arbitrary_gauss_legendre_rate_defect": {
            str(nodes): _quadrature_rate_defect(arbitrary, nodes)
            for nodes in (1, 2, 4, 8)
        },
        "wavy_arbitrary_relative_gcl": float(
            wavy_arbitrary.evidence.maximum_relative_gcl_residual
        ),
        # Committed Lagrangian density on the target cells against the unit
        # density carried by the exactly integrated dual-area rate.
        "sphere_arbitrary_lagrangian_density_defect": float(
            jnp.max(
                jnp.abs(
                    lagrangian.content
                    / arbitrary.target.vertex_area
                    * (
                        sphere.vertex_area
                        + arbitrary.step_size * arbitrary.area_rate_m2_s
                    )
                    / sphere.vertex_area
                    - 1.0
                )
            )
        ),
        "relative_gcl_tolerance": float(arbitrary.evidence.relative_gcl_tolerance),
        "far_translation_status": int(far.evidence.status),
        "far_translation_relative_gcl": float(far.evidence.maximum_relative_gcl_residual),
    }


_GAS_CONSTANT_TEMPERATURE = 8.314462618 * 298.15


def _equilibrium_thickness(
    height: np.ndarray,
    area: np.ndarray,
    volume: float,
    ratio: float,
    capacity: float,
    density: float,
    gravity: float,
) -> np.ndarray:
    """Model-exact rest profile of an insoluble, non-diffusive Langmuir film.

    ``Gamma = ratio * h`` (materially conserved from a uniform start) and
    ``2 grad sigma(Gamma) + rho h g_t = 0`` give
    ``logit(Gamma / Gamma_inf) = C - rho g z / (2 R T ratio)``; ``C`` is fixed by
    the discrete liquid volume.
    """
    slope = density * gravity / (2.0 * _GAS_CONSTANT_TEMPERATURE * ratio)

    def thickness(offset: float) -> np.ndarray:
        return capacity / ratio / (1.0 + np.exp(slope * height - offset))

    low, high = -80.0, 20.0
    for _ in range(200):
        middle = 0.5 * (low + high)
        if np.sum(area * thickness(middle)) > volume:
            high = middle
        else:
            low = middle
    return thickness(0.5 * (low + high))


def sphere_gravity_equilibrium(
    levels: tuple[int, ...], *, step_size: float = 5e-3, duration: float = 1.0
) -> list[dict[str, Any]]:
    """Relax a uniform soap bubble under gravity to its Marangoni rest profile.

    Huang et al. (ACM TOG 39(4), 2020, eqs. 33-36) balance ``M/eta dGamma/dtheta
    = g sin theta`` with ``Gamma/eta`` materially constant and a linear tension
    law, giving ``h ~ exp(-a cos theta)`` with ``a = rho g R h_0 / (2 R T
    Gamma_0)`` in dimensional form. The plug-flow model shares the rest
    balance, the symmetric factor two, no surface diffusion and no exchange,
    but uses the Langmuir tension, whose exact rest profile is the logistic
    ``_equilibrium_thickness``; it reduces to Huang's exponential for
    ``Gamma << Gamma_inf``. The constant is fixed by the liquid volume.
    """
    radius, thickness, density, gravity, capacity = 0.02, 1e-6, 1000.0, 9.81, 4e-6
    concentration = (
        density * gravity * radius * thickness / (2.0 * _GAS_CONSTANT_TEMPERATURE)
    )
    drag = 0.05
    law = it.LangmuirSurfactantLaw(0.072, 298.15, capacity)
    maximum_steps = round(duration / step_size)
    minimum_steps = min(maximum_steps, round(0.4 / step_size))
    profile_tolerance = 1e-3
    velocity_tolerance = 1e-3
    drainage_speed = density * thickness * gravity / drag
    compiled_relaxation: Any = _relax_gravity
    rows = []
    for level in levels:
        mesh = _icosphere(level)
        surface = it.prepare_film_surface(
            phx.geometry.TriangleMesh(
                radius * np.asarray(mesh.vertices), np.asarray(mesh.topology.faces)
            )
        )
        prepared, prepare_ms = _timed_prepare(
            it.SurfacePlugFlowPlan(
                surface,
                law,
                density_kg_m3=density,
                air_drag_coefficient_kg_m2_s=drag,
                gravity_m_s2=(0.0, 0.0, -gravity),
            )
        )
        state = prepared.initial_state(thickness, concentration)
        area = np.asarray(surface.vertex_area)
        height = np.asarray(surface.coordinates[:, 2])
        volume = float(np.sum(state.liquid_volume_m3))
        surfactant = float(state.total_surfactant_mol())
        reference = _equilibrium_thickness(
            height, area, volume, concentration / thickness, capacity, density, gravity
        )
        exponent = (
            density
            * gravity
            * radius
            * thickness
            / (2.0 * _GAS_CONSTANT_TEMPERATURE * concentration)
        )
        dilute = np.exp(-exponent * height / radius)
        dilute *= volume / np.sum(area * dilute)
        arguments = (
            prepared,
            state,
            jnp.asarray(step_size, dtype=jnp.float64),
            minimum_steps,
            maximum_steps,
            jnp.asarray(profile_tolerance, dtype=jnp.float64),
            jnp.asarray(velocity_tolerance, dtype=jnp.float64),
            jnp.asarray(drainage_speed, dtype=jnp.float64),
        )
        compiled, compilation = measure_lower_and_compile(
            lambda: compiled_relaxation.lower(*arguments),
            lambda lowered: lowered.compile(),
        )
        relaxation, warm_seconds = measure_synchronized(lambda: compiled(*arguments))
        (
            step_count,
            state,
            _,
            _,
            status_seen,
            total_iterations,
            maximum_iterations,
            last_iterations,
            thickness_change,
            concentration_change,
            velocity_change,
            maximum_residual,
            last_residual,
            steady_windows,
        ) = relaxation
        executable = compiled.compiled
        cost = executable.cost_analysis()
        memory = executable.memory_analysis()
        compiler = compiler_evidence(
            cost,
            memory,
            source="jax-compiled-executable",
            unavailable_reason=(
                "Backend did not expose compiler cost or memory analysis."
                if not cost and memory is None
                else None
            ),
        )
        film = np.asarray(state.liquid_volume_m3) / area
        top, bottom = int(np.argmax(height)), int(np.argmin(height))
        final_speed = float(jnp.max(jnp.linalg.norm(prepared.velocity(state), axis=1)))
        statuses = np.flatnonzero(np.asarray(status_seen)).tolist()
        rows.append(
            {
                "vertices": surface.topology.num_vertices,
                "relative_error": float(
                    np.sqrt(np.sum(area * (film - reference) ** 2))
                    / np.sqrt(np.sum(area * reference**2))
                ),
                "top_bottom_ratio": float(film[top] / film[bottom]),
                "reference_top_bottom_ratio": float(reference[top] / reference[bottom]),
                "dilute_huang_profile_relative_difference": float(
                    np.sqrt(np.sum(area * (dilute - reference) ** 2))
                    / np.sqrt(np.sum(area * reference**2))
                ),
                "residual_speed_over_drainage_speed": final_speed / drainage_speed,
                "residual_speed_change": float(velocity_change)
                / (final_speed / drainage_speed),
                "surfactant_to_volume_ratio_drift": float(
                    np.max(
                        np.abs(
                            np.asarray(state.surfactant_amount_mol)
                            / np.asarray(state.liquid_volume_m3)
                            * thickness
                            / concentration
                            - 1.0
                        )
                    )
                ),
                "volume_residual_relative": abs(
                    float(np.sum(state.liquid_volume_m3)) - volume
                )
                / volume,
                "surfactant_residual_relative": abs(
                    float(state.total_surfactant_mol()) - surfactant
                )
                / surfactant,
                "statuses": statuses,
                "steady_state_reached": int(steady_windows) >= _GRAVITY_STEADY_WINDOWS,
                "steady_window_steps": _GRAVITY_WINDOW_STEPS,
                "steady_windows_required": _GRAVITY_STEADY_WINDOWS,
                "steady_profile_tolerance": profile_tolerance,
                "steady_velocity_tolerance": velocity_tolerance,
                "last_window_thickness_change": float(thickness_change),
                "last_window_concentration_change": float(concentration_change),
                "last_window_velocity_change_over_drainage_speed": float(velocity_change),
                "steps": int(step_count),
                "simulated_time_s": int(step_count) * step_size,
                "nonlinear_iterations_total": int(total_iterations),
                "nonlinear_iterations_maximum": int(maximum_iterations),
                "nonlinear_iterations_last": int(last_iterations),
                "nonlinear_residual_maximum": float(maximum_residual),
                "nonlinear_residual_last": float(last_residual),
                "symbolic_preparations": 1,
                "numeric_factorizations": 1,
                "prepare_ms": prepare_ms,
                "lowering_ms": 1e3 * compilation.lowering_seconds,
                "compilation_ms": 1e3 * compilation.compilation_seconds,
                "warm_recurrence_ms": 1e3 * warm_seconds,
                "logical_prepared_bytes": logical_array_bytes(prepared),
                "compiler": asdict(compiler),
                **_factor_storage(prepared),
            }
        )
    return rows


def _observed_orders(rows: list[dict[str, Any]], size_key: str) -> list[float]:
    orders = []
    for coarse, fine in zip(rows[:-1], rows[1:], strict=True):
        ratio = np.sqrt(fine[size_key] / coarse[size_key])
        orders.append(
            float(
                np.log(coarse["relative_error"] / fine["relative_error"]) / np.log(ratio)
            )
        )
    return orders


def run(*, smoke: bool = False) -> dict[str, Any]:
    planar = planar_levelling((8, 16) if smoke else (10, 20, 40))
    sphere = sphere_decay((1, 2) if smoke else (1, 2, 3, 4))
    wave = marangoni_wave((16, 32) if smoke else (32, 64, 128))
    gcl = geometric_conservation()
    gravity = sphere_gravity_equilibrium((1, 2) if smoke else (1, 2, 3))
    tolerance = gcl["relative_gcl_tolerance"]
    successful = bool(
        all(row["status"] == 0 for row in planar + sphere)
        and all(row["statuses"] == [0] for row in wave + gravity)
        and all(row["steady_state_reached"] for row in gravity)
        and planar[-1]["relative_error"] < (0.05 if smoke else 0.01)
        and sphere[-1]["relative_error"] < (0.1 if smoke else 0.02)
        and wave[-1]["relative_error"] < (0.05 if smoke else 0.01)
        and max(
            gcl["ruled_slide_relative_gcl"],
            gcl["sphere_scaling_relative_gcl"],
            gcl["sphere_wobble_relative_gcl"],
            gcl["sphere_arbitrary_relative_gcl"],
            gcl["wavy_arbitrary_relative_gcl"],
        )
        <= tolerance
        and gcl["ruled_slide_status"] == 0
        and gcl["ruled_slide_uniform_defect"] < 1e-12
        and gcl["sphere_arbitrary_lagrangian_density_defect"] < 1e-12
        and gcl["far_translation_status"] == int(it.SurfaceMotionStatus.GCL_VIOLATED)
        and all(
            fine["relative_error"] < coarse["relative_error"]
            for coarse, fine in zip(gravity[:-1], gravity[1:], strict=True)
        )
        and gravity[-1]["relative_error"] < (0.03 if smoke else 0.01)
    )
    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
            "precision": "float64",
        },
        "references": {
            "planar_levelling": "sigma h^3 k^4 / (3 mu), closed form",
            "sphere_decay": "sigma h^3 lambda (lambda - 2/R^2) / (3 mu), closed form",
            "marangoni_wave": "c_M^2 = 2 E_s / (rho h), Chomaz 2001",
            "sphere_gravity_equilibrium": (
                "Langmuir rest profile logit(Gamma/Gamma_inf) = C - rho g z / "
                "(2 R T Gamma_0/h_0); dilute limit h ~ exp(-a cos theta), "
                "Huang et al. ACM TOG 39(4) 2020 eq. 36 (volume-normalized)"
            ),
            "geometric_conservation": (
                "closed-form integral of the lumped ESFEM rate sum_f |f| div_f w / 3 "
                "along straight vertex paths"
            ),
        },
        "planar_levelling": planar,
        "planar_observed_orders": _observed_orders(planar, "vertices"),
        "sphere_decay": sphere,
        "sphere_observed_orders": _observed_orders(sphere, "vertices"),
        "marangoni_wave": wave,
        "marangoni_observed_orders": _observed_orders(wave, "vertices"),
        "geometric_conservation": gcl,
        "sphere_gravity_equilibrium": gravity,
        "sphere_gravity_observed_orders": _observed_orders(gravity, "vertices"),
        "successful": successful,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(smoke=args.smoke)
    payload = json.dumps(report, indent=2, allow_nan=False)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report["successful"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
