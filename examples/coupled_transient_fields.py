#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transient conjugate heat conduction across a finite-element and a virtual-element region.

The plate ``[0, 2] x [0, 1]`` is split at ``x = 1``: P1 triangles on the left
(volumetric heat capacity ``rho c = 1``) and conforming H1 virtual elements on
quadrilaterals on the right (``rho c = 2``), with matching interface vertices.
Both regions conduct heat with unit conductivity,

``rho c u_t - Laplace(u) = q cos(t)``,

and the exterior wall temperature oscillates, ``u = sin(omega t) G(x)``: the
Dirichlet lift moves in time. One ``ScalarTransmissionLaw`` with an explicitly
selected ``MatchingElimination`` states temperature continuity and heat-flux
balance; the eliminated virtual-element interface temperatures follow the
finite-element trace, so the coupled semi-discrete system is an ODE.

``prepare_coupled_transient`` adds each owner's own capacity operator, the exact
lift-rate terms, and structural index-one admission, and lowers the rows onto the
native BDF runtime. The example prints

1. fixed-step BDF2 errors against an independent reference, the matrix
   exponential of the semi-discrete system assembled from the materialized
   steady coupled operator and the owners' capacity matrices (with the explicit
   lift-rate load), and the observed temporal order;
2. adaptive BDF over two windows joined by the accepted history continuation,
   compared with one adaptive segment;
3. the refusal, with its structural evidence, of the same transient declared with
   a mortar multiplier (an unreduced index-two system).

It raises if any certified solve is not accepted.
"""

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from phydrax.solver import coupling as cpl


jax.config.update("jax_enable_x64", True)

INTERFACE_X = 1.0
CELLS = 6
OMEGA = 2.0
SOURCE = 0.7
CAPACITY = {"conductor": 1.0, "insulator": 2.0}
FINAL_TIME = 0.4


def wall_shape(points: np.ndarray, /) -> np.ndarray:
    """Spatial shape ``G`` of the oscillating wall temperature."""
    return np.sin(np.pi * points[:, 1]) + 0.5 * points[:, 0]


def heat_source(points: Array, args: object) -> Array:
    """Uniform volumetric source whose amplitude is the context's user argument."""
    user = getattr(args, "user_args", None)
    amplitude = 0.0 if user is None else user
    return jnp.full(points.shape[:-1], 1.0) * amplitude


def heat_actions() -> tuple[phx.equations.DiffusionAction, phx.equations.SourceAction]:
    return (
        phx.equations.DiffusionAction("u"),
        phx.equations.SourceAction(
            "u", phx.equations.coefficient(heat_source, coefficient_id="heat-source")
        ),
    )


def grid_points(x0: float, x1: float, cells: int, /) -> np.ndarray:
    xs = np.linspace(x0, x1, cells + 1)
    ys = np.linspace(0.0, 1.0, cells + 1)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)


def quads(cells: int, /) -> list[tuple[int, int, int, int]]:
    result = []
    for j in range(cells):
        for i in range(cells):
            a = j * (cells + 1) + i
            result.append((a, a + 1, a + cells + 2, a + cells + 1))
    return result


def exterior_mask(points: np.ndarray, boundary: np.ndarray, /) -> np.ndarray:
    """Boundary rows except the open interface segment (those stay coupled)."""
    interface = (
        np.isclose(points[:, 0], INTERFACE_X)
        & (points[:, 1] > 1.0e-12)
        & (points[:, 1] < 1.0 - 1.0e-12)
    )
    return boundary & ~interface


def finite_element_region() -> tuple[
    cpl.VariationalComponent, IntegrationDomain, np.ndarray, np.ndarray
]:
    triangles = [t for a, b, c, d in quads(CELLS) for t in ((a, b, c), (a, c, d))]
    mesh = phx.discretization.CellMesh(
        grid_points(0.0, INTERFACE_X, CELLS),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    on = np.all(np.isclose(np.asarray(probe.sites)[..., 0], INTERFACE_X), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[on]] = True
    interface = space.integration_domain("exterior_facet", EntitySelection(edges, mask))
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    fixed = exterior_mask(points, np.asarray(space.dof_maps[0].boundary_dof_mask))
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("heat", "u", heat_actions()),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=fixed
        ),
        dirichlet_values=lambda values: jnp.zeros(values.shape[:-1]),
    )
    component = cpl.VariationalComponent("conductor", problem, field="u")
    return component, interface, points, np.where(fixed, wall_shape(points), 0.0)


def virtual_element_region() -> tuple[
    cpl.VariationalComponent, IntegrationDomain, np.ndarray, np.ndarray
]:
    mesh = phx.discretization.CellMesh.from_polygons(
        jnp.asarray(grid_points(INTERFACE_X, 2.0, CELLS)),
        tuple(np.asarray(cell, dtype=np.int32) for cell in quads(CELLS)),
    )
    connectivity = mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError("The virtual-element region is a polygon mesh.")
    space = phx.discretization.VirtualElementPlan(
        mesh,
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(1)
        ),
    ).prepare()
    exterior = space.exterior_facet_domain
    facets = np.asarray(exterior.entity_indices)
    ends = np.asarray(mesh.coordinates)[np.asarray(connectivity.edges)[facets]]
    on = np.all(np.isclose(ends[..., 0], INTERFACE_X), axis=1)
    edges = mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[facets[on]] = True
    interface = space.integration_domain("exterior_facet", EntitySelection(edges, mask))
    points = np.asarray(space.dof_map.default_dof_points)
    boundary = np.asarray(connectivity.boundary_vertices, dtype=np.bool_)
    fixed = exterior_mask(points, boundary)
    problem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm("heat", "u", heat_actions()),
        space,
        constraint=phx.discretization.virtual_element_dirichlet_constraint(
            space, "u", boundary_mask=fixed
        ),
        dirichlet_values=lambda values: jnp.zeros(values.shape[:-1]),
    )
    component = cpl.VariationalComponent("insulator", problem, field="u")
    return component, interface, points, np.where(fixed, wall_shape(points), 0.0)


def coupled_problem(
    imposition: cpl.TransmissionImposition, /
) -> tuple[cpl.PreparedCoupledProblem, dict[str, np.ndarray], dict[str, np.ndarray]]:
    left, left_interface, left_points, left_lift = finite_element_region()
    right, right_interface, right_points, right_lift = virtual_element_region()
    domain = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(domain, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    samples = pairing.component.sample(phx.domain.PointSampling(8), key=jax.random.key(1))
    binding = cpl.InterfaceBinding(
        "cut",
        cpl.InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        (
            cpl.InterfaceEndpoint(
                "left",
                cpl.PairedSupportAttachment(
                    cover, pairing.pairing_id, pairing.left_patch_id, samples
                ),
                fields={"value": left.field_space_id("u")},
            ),
            cpl.InterfaceEndpoint(
                "right",
                cpl.PairedSupportAttachment(
                    cover, pairing.pairing_id, pairing.right_patch_id, samples
                ),
                fields={"value": right.field_space_id("u")},
            ),
        ),
    )
    law = cpl.ScalarTransmissionLaw(
        "cut-heat",
        binding,
        (
            cpl.TransmissionSide("left", "conductor", "u", left_interface),
            cpl.TransmissionSide("right", "insulator", "u", right_interface),
        ),
        imposition,
    )
    plan = cpl.CoupledProblemPlan(
        "conjugate-heat", components=(left, right), bindings=(binding,), laws=(law,)
    )
    prepared = cpl.prepare_coupled_problem(plan, interface_owners=(cover,))
    return (
        prepared,
        {"conductor": left_lift, "insulator": right_lift},
        {"conductor": left_points, "insulator": right_points},
    )


def context(
    problem: phx.equations.CompiledFiniteElementProblem
    | phx.equations.CompiledVirtualElementProblem,
    lift: Array,
    amplitude: Array,
    /,
) -> object:
    if isinstance(problem, phx.equations.CompiledFiniteElementProblem):
        return phx.equations.FiniteElementExecutionContext(
            problem.discretization.default_runtime, lift=lift, user_args=amplitude
        )
    return phx.equations.VirtualElementExecutionContext(
        problem.discretization.default_runtime, lift=lift, user_args=amplitude
    )


def transient_arguments(
    prepared: cpl.PreparedCoupledProblem, lifts: Mapping[str, np.ndarray], /
) -> cpl.TransientArguments:
    """Owner contexts at time ``t``: moving wall lift and source amplitude ``q cos t``."""
    problems = {
        component.name: component.problem
        for component in prepared.components
        if isinstance(component, cpl.VariationalComponent)
    }

    def arguments(time: Array, parameter: object) -> Mapping[str, object]:
        amplitude = jnp.asarray(parameter) * jnp.cos(time)
        return {
            name: context(
                problem, jnp.sin(OMEGA * time) * jnp.asarray(lifts[name]), amplitude
            )
            for name, problem in problems.items()
        }

    return arguments


def prepare_transient(
    prepared: cpl.PreparedCoupledProblem, lifts: Mapping[str, np.ndarray], /
) -> cpl.PreparedCoupledTransient:
    return cpl.prepare_coupled_transient(
        prepared,
        fields=tuple(
            cpl.TransientField(name, "u", capacity=value)
            for name, value in CAPACITY.items()
        ),
        arguments=transient_arguments(prepared, lifts),
        arguments_id="oscillating-wall-heat",
        parameters=jnp.asarray(SOURCE),
    )


def reference_state(
    transient: cpl.PreparedCoupledTransient,
    arguments: cpl.TransientArguments,
    time: float,
    /,
) -> np.ndarray:
    """Exact semi-discrete solve state at ``time`` (native coordinates).

    The steady coupled rows of the spatial problem are affine,
    ``R(z, t) = K z + sin(w t) a + q cos(t) b``: ``K``, ``a``, and ``b`` are read
    off ``PreparedCoupledProblem.residual`` (the steady assembler, independent of
    the transient lowering and of the DAE runtime). The capacity rows
    ``C z' + w cos(w t) m`` are the transient rows minus the steady rows. The
    forcing is generated by ``(sin w t, cos w t, sin t, cos t)`` and the augmented
    linear system is evolved by one matrix exponential. (The unit tests compare
    against a fully independent host P1 assembly instead.)
    """
    prepared = transient.prepared
    size = transient.state_space.size
    zero = jnp.zeros(size)

    def native(rows: tuple[tuple[Array, ...], ...]) -> np.ndarray:
        return np.asarray(transient.native_state(rows))

    def steady(state: Array, time_: float, parameter: float) -> np.ndarray:
        view = transient.state_view(state)
        return native(prepared.residual(view, arguments(jnp.asarray(time_), parameter)))

    def capacity_rows(rate: Array) -> np.ndarray:
        view = transient.state_view(zero)
        total = transient.residual(0.0, view, transient.state_view(rate), 0.0)
        return native(total) - steady(zero, 0.0, 0.0)

    basis = np.eye(size)
    offset = steady(zero, 0.0, 0.0)
    stiffness = np.stack([steady(jnp.asarray(e), 0.0, 0.0) - offset for e in basis], 1)
    rate_load = capacity_rows(zero) / OMEGA  # at t = 0 the lift rate is w G
    capacity = np.stack([capacity_rows(jnp.asarray(e)) for e in basis], 1) - (
        OMEGA * rate_load[:, None]
    )
    lift_load = steady(zero, np.pi / (2.0 * OMEGA), 0.0) - offset
    source = steady(zero, 0.0, 1.0) - offset
    generator = np.zeros((size + 4, size + 4))
    generator[:size, :size] = -np.linalg.solve(capacity, stiffness)
    generator[:size, size] = -np.linalg.solve(capacity, lift_load)
    generator[:size, size + 1] = -OMEGA * np.linalg.solve(capacity, rate_load)
    generator[:size, size + 3] = -SOURCE * np.linalg.solve(capacity, source)
    generator[size, size + 1], generator[size + 1, size] = OMEGA, -OMEGA
    generator[size + 2, size + 3], generator[size + 3, size + 2] = 1.0, -1.0
    initial = np.zeros(size + 4)
    initial[size + 1], initial[size + 3] = 1.0, 1.0
    return (scipy.linalg.expm(time * generator) @ initial)[:size]


def stage_termination() -> phx.nonlinear.NonlinearTermination:
    """Residual-certified BDF stages.

    The stage residual is dominated by ``C / h``: at small adaptive steps the
    Newton correction that removes a ``1e-10`` residual lies below any fixed
    state step floor, so only the residual threshold decides convergence.
    """
    return phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=12,
    )


def fixed_policy() -> phx.solver.DAESolvePolicy:
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_termination=stage_termination(),
        initialization_termination=stage_termination(),
    )


def adaptive_policy() -> phx.solver.DAESolvePolicy:
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_termination=stage_termination(),
        initialization_termination=stage_termination(),
        adaptive=phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            maximum_accepted_steps=4096,
            maximum_attempts=8192,
        ),
    )


def require(solution: cpl.CoupledTransientSolution, label: str, /) -> None:
    if not bool(solution.accepted):
        raise RuntimeError(f"{label}: the coupled transient was not accepted.")


def main() -> None:
    prepared, lifts, _ = coupled_problem(cpl.MatchingElimination(eliminated="right"))
    transient = prepare_transient(prepared, lifts)
    print("solve blocks:", ", ".join("/".join(path) for path in transient.paths))
    print(
        "roles:",
        transient.roles,
        "| structural index:",
        transient.analysis.structural_index,
    )
    zero = transient.state_space.zeros()
    parameter = jnp.asarray(SOURCE)
    reference = reference_state(
        transient, transient_arguments(prepared, lifts), FINAL_TIME
    )
    scale = np.max(np.abs(reference))

    errors = []
    for steps in (10, 20, 40):
        grid = phx.dynamics.TimeGrid(
            jnp.linspace(0.0, FINAL_TIME, steps + 1), time_id=f"fixed-{steps}"
        )
        solution = cpl.solve_coupled_transient(
            transient, zero, grid, parameters=parameter, policy=fixed_policy()
        )
        require(solution, f"BDF2 with {steps} steps")
        error = float(
            np.max(np.abs(np.asarray(solution.dae.states[-1]) - reference)) / scale
        )
        errors.append(error)
        worst = float(
            jnp.max(solution.certificate.residual_norms / solution.certificate.scales)
        )
        print(
            f"BDF2 dt={FINAL_TIME / steps:.4f}: relative error {error:.3e}, "
            f"max certified row defect {worst:.2e}"
        )
    rates = [np.log2(errors[i] / errors[i + 1]) for i in range(len(errors) - 1)]
    print("observed temporal orders:", ", ".join(f"{rate:.2f}" for rate in rates))

    times = {
        "full": (0.0, 0.2, FINAL_TIME),
        "first": (0.0, 0.2),
        "second": (0.2, FINAL_TIME),
    }
    grids = {
        name: phx.dynamics.TimeGrid(jnp.asarray(value), time_id=f"adaptive-{name}")
        for name, value in times.items()
    }
    full = cpl.solve_coupled_transient(
        transient, zero, grids["full"], parameters=parameter, policy=adaptive_policy()
    )
    first = cpl.solve_coupled_transient(
        transient, zero, grids["first"], parameters=parameter, policy=adaptive_policy()
    )
    second = cpl.solve_coupled_transient(
        transient,
        zero,
        grids["second"],
        parameters=parameter,
        policy=adaptive_policy(),
        continuation=first.continuation,
    )
    for solution, label in (
        (full, "adaptive"),
        (first, "window 1"),
        (second, "window 2"),
    ):
        require(solution, label)
    steps = int(first.dae.step_history.count) + int(second.dae.step_history.count)
    print(
        f"adaptive BDF: {int(full.dae.step_history.count)} accepted steps in one segment, "
        f"{steps} across two windows joined by the accepted history; "
        f"max |window - segment| {float(jnp.max(jnp.abs(second.dae.states[-1] - full.dae.states[-1]))):.2e}; "
        f"relative error {float(np.max(np.abs(np.asarray(full.dae.states[-1]) - reference)) / scale):.2e}"
    )
    fields = {
        name: transient.field(
            name, "u", FINAL_TIME, transient.state_view(full.dae.states[-1]), parameter
        )
        for name in CAPACITY
    }
    print(
        "final temperatures: "
        + ", ".join(
            f"{name} in [{float(jnp.min(value)):.4f}, {float(jnp.max(value)):.4f}]"
            for name, value in fields.items()
        )
    )

    mortar, mortar_lifts, _ = coupled_problem(
        cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="right"))
    )
    try:
        prepare_transient(mortar, mortar_lifts)
    except ValueError as error:
        print("mortar transient refused:", str(error).split(";")[0])
    else:
        raise RuntimeError("An unreduced multiplier transient must be refused.")


if __name__ == "__main__":
    main()
