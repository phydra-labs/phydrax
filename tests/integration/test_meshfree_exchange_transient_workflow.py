#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Monolithic nonlinear bulk--surface transient with Langmuir exchange and its adjoint.

A P1 finite-element bulk concentration ``c`` on ``[0, 1]^2`` with insulated outer
walls and a uniform volumetric source ``q`` exchanges amount with a meshfree
intrinsic surface concentration ``Gamma`` on the circle ``r = 0.3`` about
``(0.5, 0.5)`` through Langmuir kinetics ``j = k_a c (Gamma_max - Gamma) - k_d
Gamma``:

``c_t - Laplace(c) = q - j delta_Gamma`` and ``Gamma_t - Laplace_Gamma(Gamma) = j``,
the surface operator in its conservative measure-paired form.

``solve_coupled_transient`` integrates the nonlinear monolithic DAE natively
(BDF stages solved by the native Newton through one coupled linearization). The
exchange routes are exact transposes, so with insulated walls the total amount
``int c + int Gamma`` grows exactly by ``q t |Omega|`` up to stage tolerance. The
reverse-mode derivative of the terminal adsorbed amount with respect to ``q``
is the native discrete implicit adjoint; it is checked against central
differences of the same discrete trajectory.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.discretization.meshfree import (
    ImplicitSurfaceGeometry,
    LocalStencilPolicy,
    SurfacePointCloudPlan,
    SurfaceQuadraturePolicy,
)
from phydrax.geometry import Rectangle
from phydrax.interfacial_transport import AdsorptionKinetics
from phydrax.linalg import DiagonalLinearOperator
from phydrax.metrix import RegularLevelSetManifold
from phydrax.solver import coupling as cpl
from tests.unit.solver.coupling._cases import triangle_mesh


CENTER = np.asarray([0.5, 0.5])
RADIUS = 0.3
SURFACE_POINTS = 24
CELLS = 6
KINETICS = AdsorptionKinetics(2.0, 0.5, 1.0)
SOURCE = 0.4
FINAL_TIME = 0.2
STEPS = 8


def _source(points: Array, args: object) -> Array:
    if not isinstance(args, phx.equations.FiniteElementExecutionContext):
        raise TypeError("The bulk source reads its amplitude from the FE context.")
    return jnp.full(points.shape[:-1], 1.0) * jnp.asarray(args.user_args)


def _bulk() -> cpl.VariationalComponent:
    space = phx.discretization.FiniteElementPlan(
        triangle_mesh(0.0, 1.0, CELLS, CELLS),
        phx.discretization.FiniteElementFieldSpec(
            "c", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    form = phx.equations.FiniteElementForm(
        "bulk-diffusion",
        "c",
        (
            phx.equations.DiffusionAction("c"),
            phx.equations.SourceAction(
                "c", phx.equations.coefficient(_source, coefficient_id="bulk-source")
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(form, space)
    return cpl.VariationalComponent("bulk", problem, field="c")


def _surface() -> cpl.MeshfreeComponent:
    angle = 2 * math.pi * (np.arange(SURFACE_POINTS) + 0.25) / SURFACE_POINTS
    points = CENTER + RADIUS * np.column_stack((np.cos(angle), np.sin(angle)))

    def constraint(point: Array) -> Array:
        offset = point - jnp.asarray(CENTER)
        return jnp.asarray([jnp.dot(offset, offset) - RADIUS**2])

    source = RegularLevelSetManifold(
        constraint, ambient_dimension=2, codimension=1, manifold_id="adsorbing-circle"
    )
    surface = SurfacePointCloudPlan(
        jnp.asarray(points),
        ImplicitSurfaceGeometry(
            source, certified_tube_radius=0.1, geometry_id="adsorbing-circle"
        ),
        8,
        quadrature=SurfaceQuadraturePolicy(
            "normalized-density",
            density=jnp.ones(SURFACE_POINTS, dtype=jnp.float64),
            total_area=2 * math.pi * RADIUS,
        ),
        stencil_policy=LocalStencilPolicy(polynomial_degree=2),
        require_tube=True,
    ).prepare()
    # The measure-paired divergence of the surface gradient is the quadrature
    # adjoint (dissipative) Laplace-Beltrami: sum_i m_i (div grad Gamma)_i = 0 on
    # the closed curve, so surface diffusion moves no amount.
    diffusion = surface.surface_divergence @ surface.surface_gradient
    native = DiagonalLinearOperator(-surface.measures, space=diffusion.target) @ diffusion
    return cpl.MeshfreeComponent(
        surface,
        native,
        surface.prepare_field_reconstruction(
            support_geometry=Rectangle((0.5, 0.5), (1.0, 1.0)).compile()
        ),
        surface.measures,
        name="surface",
        field="gamma",
        owner_id=surface.prepared_id,
    )


def _transient() -> tuple[cpl.PreparedCoupledTransient, cpl.SurfaceExchangeLaw]:
    bulk, surface = _bulk(), _surface()
    query = bulk.prepare_field_reconstruction("c").prepare_query(surface.owner.points)
    normals = np.asarray(surface.owner.points) - CENTER
    law = cpl.SurfaceExchangeLaw(
        cpl.ContributionEndpoint("bulk", "c"),
        cpl.ContributionEndpoint("surface", "gamma"),
        query,
        surface,
        normals / np.linalg.norm(normals, axis=1, keepdims=True),
        cpl.LangmuirAdsorptionFlux(KINETICS),
    )
    plan = cpl.CoupledProblemPlan(
        "langmuir-bulk-surface", components=(bulk, surface), bindings=(), laws=(law,)
    )
    prepared = cpl.prepare_coupled_problem(plan)
    runtime = bulk.problem.discretization.default_runtime

    def arguments(time: Array, parameter: object) -> dict[str, object]:
        return {
            "bulk": phx.equations.FiniteElementExecutionContext(
                runtime, time=time, user_args=parameter
            )
        }

    transient = cpl.prepare_coupled_transient(
        prepared,
        fields=(cpl.TransientField("bulk", "c"), cpl.TransientField("surface", "gamma")),
        arguments=arguments,
        arguments_id="uniform-bulk-source",
        parameters=jnp.asarray(SOURCE),
    )
    return transient, law


def _policy() -> phx.solver.DAESolvePolicy:
    # The stage residual alone certifies a BDF stage: its M / h scaling puts the
    # converged Newton correction far below the default state-step floor.
    termination = phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-12,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=12,
    )
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_termination=termination,
        initialization_termination=termination,
    )


@pytest.fixture(scope="module")
def scenario() -> tuple[cpl.PreparedCoupledTransient, cpl.SurfaceExchangeLaw]:
    return _transient()


def _initial(transient: cpl.PreparedCoupledTransient, /) -> tuple[tuple[Array, ...], ...]:
    names = transient.state_space.names
    blocks = []
    for name, member in zip(names, transient.state_space.spaces, strict=True):
        if not isinstance(member, phx.linalg.BlockSpace):
            raise TypeError("Every coupled owner publishes a block of named states.")
        fill = 1.0 if name == "bulk" else 0.1
        states = []
        for space in member.spaces:
            if not isinstance(space, phx.linalg.ArraySpace):
                raise TypeError("Every named state must be an array-valued space.")
            states.append(jnp.full(space.shape, fill, dtype=space.dtype))
        blocks.append(tuple(states))
    return tuple(blocks)


def _grid() -> phx.dynamics.TimeGrid:
    return phx.dynamics.TimeGrid(
        jnp.linspace(0.0, FINAL_TIME, STEPS + 1), time_id="langmuir-adsorption"
    )


def _solve(
    transient: cpl.PreparedCoupledTransient, parameter: Array, /
) -> cpl.CoupledTransientSolution:
    return cpl.solve_coupled_transient(
        transient,
        _initial(transient),
        _grid(),
        parameters=parameter,
        policy=_policy(),
    )


def _amounts(
    transient: cpl.PreparedCoupledTransient,
    law: cpl.SurfaceExchangeLaw,
    native: Array,
    parameter: Array,
    time: Array,
    /,
) -> tuple[Array, Array]:
    state = transient.state_view(native)
    bulk = transient.field("bulk", "c", time, state, parameter)
    gamma = transient.field("surface", "gamma", time, state, parameter)
    bulk_owner = transient.prepared.chart.component("bulk")
    if not isinstance(bulk_owner, cpl.VariationalComponent):
        raise TypeError("The bulk owner is a variational component.")
    context = phx.equations.FiniteElementExecutionContext(
        bulk_owner.problem.discretization.default_runtime, time=time
    )
    mass = bulk_owner.prepare_capacity("c").operator(context)
    return jnp.sum(mass.mv(bulk)), jnp.dot(law.measures, gamma)


def test_langmuir_transient_conserves_amount_and_adsorbs_monotonically(
    scenario: tuple[cpl.PreparedCoupledTransient, cpl.SurfaceExchangeLaw],
) -> None:
    transient, law = scenario
    solution = _solve(transient, jnp.asarray(SOURCE))
    assert bool(solution.native_successful) and bool(solution.accepted)
    assert bool(solution.derivative_valid)
    assert transient.roles == ("differential", "differential")
    parameter = jnp.asarray(SOURCE)
    totals, adsorbed = [], []
    for time, native in zip(solution.times, solution.dae.states, strict=True):
        bulk_amount, surface_amount = _amounts(transient, law, native, parameter, time)
        totals.append(float(bulk_amount + surface_amount))
        adsorbed.append(float(surface_amount))
    # Insulated walls: the only exterior supply is the uniform source on |Omega| = 1.
    expected = totals[0] + SOURCE * np.asarray(solution.times)
    np.testing.assert_allclose(totals, expected, rtol=0.0, atol=1.0e-10)
    # c = 1 adsorbs onto Gamma = 0.1: k_a c (1 - Gamma) - k_d Gamma > 0.
    assert np.all(np.diff(adsorbed) > 0.0)


def test_langmuir_transient_adjoint_matches_central_differences(
    scenario: tuple[cpl.PreparedCoupledTransient, cpl.SurfaceExchangeLaw],
) -> None:
    transient, law = scenario

    def adsorbed(parameter: Array) -> Array:
        solution = _solve(transient, parameter)
        return _amounts(
            transient,
            law,
            solution.dae.states[-1],
            parameter,
            solution.times[-1],
        )[1]

    parameter = jnp.asarray(SOURCE)
    gradient = float(jax.grad(adsorbed)(parameter))
    step = 1.0e-4
    central = float(adsorbed(parameter + step) - adsorbed(parameter - step)) / (
        2.0 * step
    )
    # More source raises the bulk concentration and therefore the adsorption.
    assert gradient > 0.0
    assert gradient == pytest.approx(central, rel=1.0e-6)
