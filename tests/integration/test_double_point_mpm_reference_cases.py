#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded references for the exact alpha=1, S=0 constituent limit.

These are quasistatic reductions of the native mixed blocks, not claims that a
finite-inertia explicit MPM column is Terzaghi's small-strain continuum. With
constrained modulus H, epsilon_c=(load-p)/H and q=n(v_f-v_s), mixture continuity
and phase drag give q=-n**2/beta * grad(p), p_t=H*n**2/beta * p_xx. The native
Df/Df.T and inverse-weight operations supply that diffusion/flux operator.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array
from numpy.typing import NDArray

import phydrax as phx
from phydrax.applications.solid_mechanics._mpm_double_point import (
    MixedMPMPressureOperator,
    MPMToMACFaceTransferPlan,
)
from phydrax.solver._mpm_double_point import (
    DoublePointMPMPlan,
    project_double_point_velocity,
)


@dataclass(frozen=True)
class _Column:
    plan: DoublePointMPMPlan
    mixed: MixedMPMPressureOperator
    beta: float
    n: float
    h: float


def _column(
    cells: int, *, both_ends_drained: bool, pressure_drop: float = 0.0
) -> _Column:
    h = 1.0 / cells
    bounds = jnp.asarray([[0.0], [1.0]], dtype=jnp.float64)
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells),), axis_names=("x",)
    ).prepare(bounds)
    mac = phx.discretization.MACOperatorPlan(
        phx.discretization.FiniteVolumePlan(grid).prepare()
    ).prepare()
    # One nodal halo on each side admits the ordinary linear MPM basis.
    nodal_grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformAxisSpec(cells + 3, endpoint=True),), axis_names=("x",)
    ).prepare(jnp.asarray([[-h], [1.0 + h]], dtype=jnp.float64))
    n = 0.4
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(cells, dtype=jnp.int32),
        jnp.full((cells,), (1.0 - n) * h, dtype=jnp.float64),
        ambient_dimension=1,
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        nodal_grid, assignment=phx.discretization.TensorBSplineSplatAssignment(1)
    ).prepare(particles)
    solid = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "mixed-quasistatic-column",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(1),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            bounds, periodic=(False,), support_margin=h
        ),
    ).dynamics
    transfer = MPMToMACFaceTransferPlan(
        solid, mac, maximum_routes=2 * (cells + 1)
    ).prepare()
    fluid_transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(
        particles
    )
    law = phx.equations.InterphaseDragLawPlan("stokes")
    evaluation = law.evaluate(
        jnp.asarray(n, dtype=jnp.float64),
        jnp.asarray(0.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
        jnp.asarray(0.01, dtype=jnp.float64),
        jnp.asarray(997.0, dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    assert evaluation.successful
    beta = float(evaluation.beta)
    pressure_mask = np.zeros((cells,), dtype=np.bool_)
    pressure_mask[-1] = True
    pressure_mask[0] = both_ends_drained
    pressure_values = np.zeros((cells,), dtype=np.float64)
    pressure_values[0] = pressure_drop
    solid_free = jnp.zeros(solid.nodal_shape + (1,), dtype=jnp.bool_)
    fluid_free = (
        jnp.ones((cells + 1,), dtype=jnp.bool_).at[0].set(False).at[-1].set(False),
    )
    plan = DoublePointMPMPlan(
        solid,
        fluid_transfer,
        transfer,
        law,
        solid_density=1.0,
        fluid_density=997.0,
        fluid_viscosity=0.01,
        grain_diameter=1.0,
        pressure_mask=pressure_mask,
        pressure_values=pressure_values,
        solid_free=solid_free,
        fluid_free=fluid_free,
        tolerance=1.0e-11,
        maximum_iterations=4 * cells,
    )
    fractions_f = (jnp.full((cells + 1,), n, dtype=jnp.float64),)
    fractions_s = (jnp.full((cells + 1,), 1.0 - n, dtype=jnp.float64),)
    # In the exact inertia-free reduction M_f is the drag resistance beta*V,
    # NOT rho_f*n*V. Its inverse is the quasistatic mobility. No phase-density
    # rescaling is concealed in the pressure or flux oracle below.
    resistance = tuple(beta * volume for volume in mac.face_dual_measures)
    mixed = MixedMPMPressureOperator(
        transfer,
        fractions_s,
        fractions_f,
        jnp.full(solid.nodal_shape, (1.0 - n) * h, dtype=jnp.float64),
        resistance,
        solid_active=solid_free,
        fluid_active=fluid_free,
    )
    assert mixed.evidence.successful
    return _Column(plan, mixed, beta, n, h)


def _terzaghi_pressure(
    x: NDArray[np.float64], time: float, length: float, cv: float, load: float
) -> NDArray[np.float64]:
    # Uniform initial excess pressure; one impermeable and one drained end.
    # At tested T_v >= .05, 80 terms make series truncation negligible.
    modes = np.arange(80, dtype=np.float64)
    wavenumber = (modes + 0.5) * np.pi / length
    coefficients = 2.0 * load * (-1.0) ** modes / ((modes + 0.5) * np.pi)
    return np.sum(
        coefficients[:, None]
        * np.cos(wavenumber[:, None] * x[None, :])
        * np.exp(-cv * wavenumber[:, None] ** 2 * time),
        axis=0,
    )


@pytest.mark.parametrize("cells", [16, 32], ids=["coarse", "fine"])
def test_terzaghi_pressure_diffusion_and_settlement_from_native_mixed_reduction(
    cells: int,
) -> None:
    column = _column(cells, both_ends_drained=False)
    mixed = column.mixed
    mac = mixed.transfer.operators
    load = 2.0
    constrained_modulus = 50.0
    cv = constrained_modulus * column.n**2 / column.beta
    # Drainage is imposed at the last cell center; use that physical distance
    # in both the analytic solution and the settlement, not nominal grid size.
    length = 1.0 - 0.5 * column.h
    x = (np.arange(cells, dtype=np.float64) + 0.5) * column.h
    pressure = jnp.full((cells,), load, dtype=jnp.float64).at[-1].set(0.0)
    volume = mac.discretization.cell_volumes
    elapsed = 0.0
    for time_factor in (0.05, 0.2):
        target = time_factor * length**2 / cv
        # Explicit exact-limit reduction, CFL=.2 (<1/2); first-order temporal
        # and second-order spatial errors are both O(h^2) at fixed T_v.
        steps = int(np.ceil((target - elapsed) / (0.2 * column.h**2 / cv)))
        dt = (target - elapsed) / steps

        def diffuse(_: int, value: Array) -> Array:
            rate = -constrained_modulus * mixed.pressure_action(volume * value)
            return (value + dt * rate).at[-1].set(0.0)

        pressure = jax.lax.fori_loop(0, steps, diffuse, pressure)
        expected = _terzaghi_pressure(x, target, length, cv, load)
        # Reservoir-at-center treatment and O(h^2) time integration justify a
        # 3% load-scaled coarse tolerance, shrinking by four on refinement.
        relative_tolerance = 0.03 * (16.0 / cells) ** 2
        np.testing.assert_allclose(
            pressure, expected, rtol=0.0, atol=relative_tolerance * load
        )
        weights = np.full((cells,), column.h, dtype=np.float64)
        weights[-1] = 0.5 * column.h
        settlement = float(
            jnp.sum(jnp.asarray(weights) * (load - pressure)) / constrained_modulus
        )
        modes = np.arange(80, dtype=np.float64)
        average_ratio = np.sum(
            2.0
            / ((modes + 0.5) * np.pi) ** 2
            * np.exp(-(((modes + 0.5) * np.pi) ** 2) * time_factor)
        )
        exact_settlement = load * length / constrained_modulus * (1.0 - average_ratio)
        np.testing.assert_allclose(
            settlement, exact_settlement, rtol=relative_tolerance, atol=0.0
        )
        assert settlement > 0.0
        assert jnp.all(jnp.isfinite(pressure))
        elapsed = target


@pytest.mark.parametrize("cells", [8, 16])
def test_darcy_head_discharge_from_native_mixed_schur_projection(cells: int) -> None:
    density = 997.0
    gravity = 9.81
    head_drop = 0.002
    area = 0.3
    pressure_drop = density * gravity * head_drop
    column = _column(cells, both_ends_drained=True, pressure_drop=pressure_drop)
    mixed = column.mixed
    # This is the exact steady drag-mobility Schur solve, not a finite-inertia
    # transient. Solid DOFs are constrained; pressure is a physical Pa field.
    result = project_double_point_velocity(
        column.plan,
        mixed,
        jnp.zeros(mixed.transfer.solid.nodal_shape + (1,), dtype=jnp.float64),
        tuple(jnp.zeros_like(value) for value in mixed.fluid_mass),
        jnp.zeros((cells,), dtype=jnp.float64),
        jnp.asarray(1.0, dtype=jnp.float64),
    )
    assert result.successful
    assert result.iterations > 0
    length = (cells - 1) * column.h
    expected_pressure = np.linspace(pressure_drop, 0.0, cells, dtype=np.float64)
    np.testing.assert_allclose(
        result.pressure, expected_pressure, rtol=2.0e-10, atol=2.0e-10 * pressure_drop
    )
    discharge = area * column.n * result.fluid_velocity[0][1:-1]
    expected_discharge = (
        column.n**2 * density * gravity / column.beta * area * head_drop / length
    )
    # Linear pressure has zero truncation error on this uniform grid; tolerance
    # only covers the native Krylov residual and floating-point arithmetic.
    np.testing.assert_allclose(discharge, expected_discharge, rtol=5.0e-10, atol=0.0)
    np.testing.assert_array_equal(result.solid_velocity, 0.0)
    np.testing.assert_allclose(
        mixed.divergence(result.solid_velocity, result.fluid_velocity)[1:-1],
        0.0,
        atol=2.0e-9 * expected_discharge / area / column.h,
    )
