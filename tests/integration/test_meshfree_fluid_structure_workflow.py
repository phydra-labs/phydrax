#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Monolithic meshfree fluid--structure step: interface force, work, and total balance.

The example's step couples a compressible Stokes fluid and an elastic solid
across ``x = 1`` through a ``VectorTransmissionLaw``. The certificate's gated
defects prove weak velocity continuity, traction balance of both sides' conormal
rows, and zero interface power loss as dual pairings of the traction covector.
A smooth manufactured step then proves the total balance: the solid's elastic
power, assembled from its own derivative rows independently of the coupling
law, equals the received interface work plus the body-force power to the
scheme's order.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import examples.meshfree_fluid_structure as fsi
import phydrax as phx
from examples.meshfree_fluid_structure import prepare_step, resultants
from phydrax.discretization.meshfree import isotropic_elasticity_coefficients
from phydrax.solver.coupling import (
    CoupledSolution,
    PreparedCoupledProblem,
    solve_coupled_problem,
    VectorInterfaceResultants,
)


type Field = Callable[[Array], Array]


POLICY = phx.linalg.LinearSolvePolicy(
    phx.linalg.DenseLU(),
    materialization=phx.linalg.MaterializationPolicy(
        max_entries=64_000_000, max_bytes=1024 * 1024 * 1024
    ),
)


def _step(
    fluid: int, solid: int, /
) -> tuple[PreparedCoupledProblem, CoupledSolution, VectorInterfaceResultants]:
    prepared = prepare_step(fluid, solid)
    solution = solve_coupled_problem(prepared, policy=POLICY)
    return prepared, solution, resultants(prepared, solution)


@pytest.fixture(scope="module")
def coarse() -> tuple[PreparedCoupledProblem, CoupledSolution, VectorInterfaceResultants]:
    return _step(8, 6)


def test_interface_force_and_work_balance_through_the_certificate(
    coarse: tuple[PreparedCoupledProblem, CoupledSolution, VectorInterfaceResultants],
) -> None:
    _, solution, result = coarse
    assert bool(solution.native_successful) and bool(solution.accepted)
    for certificate in solution.components:
        assert bool(certificate.accepted)
        assert float(certificate.residual_norm) <= solution.tolerance * float(
            certificate.scale
        )
    report = solution.interface("wetted-interface")
    assert report.names == ("weak-continuity", "traction-balance", "interface-work")
    assert report.gated == (True, True, True)
    values, scales = np.asarray(report.values), np.asarray(report.scales)
    assert np.all(scales > 0.0)
    np.testing.assert_array_less(values, 1.0e-12 * scales)
    force = np.asarray(result.force)
    # The inflow drives the fluid into the solid, which pushes back along -x; the
    # symmetric channel carries no net shear resultant.
    assert force[0] < 0.0
    assert abs(force[1]) < 1.0e-2 * abs(force[0])
    # The fluid delivers exactly the power the solid receives.
    received = float(result.plus_power)
    assert received > 0.0
    assert float(result.minus_power) == pytest.approx(-received, rel=1.0e-12)


# --- Smooth manufactured step: the energy balance at the scheme order ---------------
#
# The physical channel has mixed Dirichlet/traction corners at (1, 0) and (1, 1),
# whose r^(1/2) singular strain limits the stored-versus-received power balance.
# The manufactured step has a smooth exact solution with the same geometry and
# laws: the fluid velocity is U, the solid rate w = U + (x - 1)(2 - x) Z(y) with Z
# chosen so that the tractions balance at x = 1, and both vanish (with every
# derivative that enters a traction) on the corner walls y = 0, 1.


def _stress(lam: float, mu: float, field: Field, /) -> Field:
    def sigma(point: Array) -> Array:
        gradient = jax.jacfwd(field)(point)
        strain = 0.5 * (gradient + gradient.T)
        return 2.0 * mu * strain + lam * jnp.trace(strain) * jnp.eye(2)

    return sigma


def _body_force(lam: float, mu: float, field: Field, /) -> Field:
    sigma = _stress(lam, mu, field)

    def force(point: Array) -> Array:
        # -div(sigma): d sigma_ij / d x_j.
        return -jnp.trace(jax.jacfwd(sigma)(point), axis1=1, axis2=2)

    return force


def _fluid_velocity(point: Array) -> Array:
    x, y = point[0], point[1]
    return (
        jnp.sin(jnp.pi * y) ** 2 * (2.0 - x) * jnp.stack((1.0 + 0.5 * x, 0.5 - 0.25 * x))
    )


FLUID = (fsi.BULK_VISCOSITY, fsi.VISCOSITY)
SOLID = (fsi.STEP * fsi.LAME_LAMBDA, fsi.STEP * fsi.SHEAR_MODULUS)


def _solid_rate(point: Array) -> Array:
    on = jnp.stack((1.0, point[1]))
    normal = jnp.asarray([1.0, 0.0])
    mismatch = (
        _stress(*FLUID, _fluid_velocity)(on) @ normal
        - _stress(*SOLID, _fluid_velocity)(on) @ normal
    )
    lam, mu = SOLID
    correction = jnp.stack((mismatch[0] / (lam + 2.0 * mu), mismatch[1] / mu))
    return _fluid_velocity(point) + (point[0] - 1.0) * (2.0 - point[0]) * correction


def _values(field: Field, points: Array, /) -> np.ndarray:
    return np.asarray(jax.vmap(field)(jnp.asarray(points)))


def _manufactured(resolution: int, /) -> tuple[float, float, float]:
    """Field error, energy-balance gap, and elastic power of one level."""
    regions = (
        ("fluid", 0.0, "left", FLUID, _fluid_velocity),
        ("solid", 1.0, "right", SOLID, _solid_rate),
    )
    components = []
    for name, x0, side, (lam, mu), field in regions:
        points = fsi.region(x0, resolution, side)[0].points
        components.append(
            fsi.block_component(
                name,
                x0,
                resolution,
                side,
                isotropic_elasticity_coefficients(lam, mu, 2),
                _values(field, points),
                source=_values(_body_force(lam, mu, field), points),
            )
        )
    prepared = fsi.coupled_step(components[0], components[1])
    solution = solve_coupled_problem(prepared, policy=POLICY)
    assert bool(solution.native_successful) and bool(solution.accepted)
    for certificate in solution.components:
        assert bool(certificate.accepted)
        assert float(certificate.residual_norm) <= solution.tolerance * float(
            certificate.scale
        )
    solid = components[1].owner
    rate = jnp.stack([solution.field("solid", field) for field in fsi.FIELDS], axis=1)
    error = float(jnp.max(jnp.abs(rate - jax.vmap(_solid_rate)(solid.points))))
    supplied = float(
        jnp.sum(
            solid.quadrature_weights
            * jnp.sum(
                jax.vmap(_body_force(*SOLID, _solid_rate))(solid.points) * rate, axis=1
            )
        )
    )
    stored = fsi.elastic_power(prepared, solution)
    received = float(resultants(prepared, solution).plus_power)
    # int sigma : eps(w) = int f . w + int_Gamma (sigma n_s) . w on a solid whose
    # other walls do not move.
    return error, abs(stored - received - supplied) / stored, stored


def test_smooth_step_closes_the_energy_balance_at_the_scheme_order() -> None:
    # The h=1/8 signed work defect cancels before the asymptotic regime;
    # independent refinement at 1/24 and 1/32 resolves the second-order tail.
    coarse_error, coarse_gap, coarse_stored = _manufactured(16)
    fine_error, fine_gap, fine_stored = _manufactured(32)
    assert coarse_stored > 0.0 and fine_stored > 0.0
    error_order = np.log2(coarse_error / fine_error)
    gap_order = np.log2(coarse_gap / fine_gap)
    # Cubic-augmented collocation converges at second order or better; the
    # quadrature energy identity closes at least as fast as the fields converge.
    assert error_order > 1.5
    assert gap_order > 1.5
