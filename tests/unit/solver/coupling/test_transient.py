"""Transient coupled heat conduction lowered onto the native index-one DAE runtime.

Two P1 finite-element regions ``[0, 1] x [0, 1]`` and ``[1, 2] x [0, 1]`` share the
cut ``x = 1`` through a matching elimination. They solve

``c(x) u_t - Laplace(u) = p cos(t)`` with ``u = sin(omega t) G(x)`` on the exterior
boundary, where the capacity ``c`` is 1 on the left and 2 on the right (or 0 for a
declared quasistatic right region). The independent reference assembles the same
P1 system on the union mesh on the host, forms the exact semi-discrete dynamics,
including the lift-rate load ``-M_fd g'_d(t)``, as one linear time-invariant system
augmented by the harmonic forcing states, and evaluates it with a matrix
exponential (SciPy) or a host array DAE.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.linalg
from jax import Array

import phydrax as phx
from phydrax.solver import coupling as cpl

from . import _cases as cases


OMEGA = 2.0
PARAMETER = 0.7
CELLS = 3


def _boundary_data(points: np.ndarray, /) -> np.ndarray:
    return np.sin(np.pi * points[:, 1]) + points[:, 0]


def _heat_source(points: Array, args: object) -> Array:
    context = args
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("The heat source reads its amplitude from the FE context.")
    amplitude = jnp.asarray(0.0 if context.user_args is None else context.user_args)
    return jnp.full(points.shape[:-1], 1.0) * amplitude


@dataclass(frozen=True, slots=True)
class HeatRegion:
    region: cases.Region
    lift_shape: np.ndarray


def _heat_region(spec: cases.RegionSpec, /) -> HeatRegion:
    space = phx.discretization.FiniteElementPlan(
        cases.triangle_mesh(spec.x0, spec.x1, spec.cells, spec.cells),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace(
        "u", exterior, rule=phx.discretization.FacetTraceRule(points=2)
    )
    on = np.all(np.isclose(np.asarray(probe.sites)[..., 0], cases.INTERFACE_X), axis=1)
    entities = space.mesh.topology.entity_sets[1]
    mask = np.zeros((entities.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[on]] = True
    interface = space.integration_domain(
        "exterior_facet", phx.discretization.EntitySelection(entities, mask)
    )
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    boundary = np.asarray(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    fixed = cases._dirichlet_mask(boundary, points, spec)
    form = phx.equations.FiniteElementForm(
        "heat",
        "u",
        (
            phx.equations.DiffusionAction("u"),
            phx.equations.SourceAction(
                "u", phx.equations.coefficient(_heat_source, coefficient_id="heat-p")
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form,
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=fixed
        ),
        dirichlet_values=lambda values: jnp.zeros(values.shape[:-1]),
    )
    region = cases.Region(
        spec,
        cpl.VariationalComponent(spec.name, problem, field="u"),
        problem,
        interface,
        points,
        np.arange(points.shape[0]),
        np.flatnonzero(~fixed),
    )
    return HeatRegion(region, np.where(fixed, _boundary_data(points), 0.0))


def _arguments(left: HeatRegion, right: HeatRegion) -> cpl.TransientArguments:
    def arguments(time: Array, parameter: object) -> Mapping[str, object]:
        amplitude = jnp.asarray(parameter) * jnp.cos(time)
        values = {}
        for name, item in (("left", left), ("right", right)):
            problem = item.region.problem
            values[name] = phx.equations.FiniteElementExecutionContext(
                problem.discretization.default_runtime,
                lift=jnp.sin(OMEGA * time) * jnp.asarray(item.lift_shape),
                user_args=amplitude,
            )
        return values

    return arguments


@dataclass(frozen=True, slots=True)
class Scenario:
    left: HeatRegion
    right: HeatRegion
    coupled: cases.Coupled


def _scenario(kind: cases.ImpositionKind, /, *, side: str = "right") -> Scenario:
    left = _heat_region(cases.RegionSpec("left", "fe", 0.0, 1.0, CELLS, 1))
    right = _heat_region(cases.RegionSpec("right", "fe", 1.0, 2.0, CELLS, 1))
    return Scenario(left, right, cases.couple(left.region, right.region, kind, side=side))


def _transient(
    scenario: Scenario,
    /,
    *,
    right_capacity: float | None = 2.0,
    scales: tuple[cpl.TransientScale, ...] = (),
) -> cpl.PreparedCoupledTransient:
    fields = (cpl.TransientField("left", "u"),)
    quasistatic: tuple[str, ...] = ("right",)
    if right_capacity is not None:
        fields = fields + (cpl.TransientField("right", "u", capacity=right_capacity),)
        quasistatic = ()
    return cpl.prepare_coupled_transient(
        scenario.coupled.prepared,
        fields=fields,
        quasistatic=quasistatic,
        arguments=_arguments(scenario.left, scenario.right),
        arguments_id="harmonic-boundary-heat",
        scales=scales,
        parameters=jnp.asarray(PARAMETER),
    )


# --- Independent host reference -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HostHeat:
    points: np.ndarray
    free: np.ndarray
    fixed: np.ndarray
    mass: np.ndarray
    stiffness: np.ndarray
    load: np.ndarray
    lift: np.ndarray


def _host_heat(right_capacity: float, /) -> HostHeat:
    """P1 mass, stiffness, and unit load on the union mesh ``[0, 2] x [0, 1]``."""
    nx, ny = 2 * CELLS, CELLS
    xs, ys = np.linspace(0.0, 2.0, nx + 1), np.linspace(0.0, 1.0, ny + 1)
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    triangles = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            triangles += [(a, a + 1, a + nx + 2), (a, a + nx + 2, a + nx + 1)]
    size = points.shape[0]
    mass, stiffness = np.zeros((size, size)), np.zeros((size, size))
    load = np.zeros(size)
    local_mass = (np.ones((3, 3)) + np.eye(3)) / 12.0
    for triangle in triangles:
        vertices = points[list(triangle)]
        jacobian = np.stack((vertices[1] - vertices[0], vertices[2] - vertices[0]), 1)
        area = 0.5 * abs(np.linalg.det(jacobian))
        gradients = np.linalg.solve(
            jacobian.T, np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
        )
        capacity = 1.0 if vertices[:, 0].mean() < 1.0 else right_capacity
        index = np.ix_(triangle, triangle)
        mass[index] += capacity * area * local_mass
        stiffness[index] += area * gradients.T @ gradients
        load[list(triangle)] += area / 3.0
    on_boundary = (
        np.isclose(points[:, 0], 0.0)
        | np.isclose(points[:, 0], 2.0)
        | np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 1], 1.0)
    )
    return HostHeat(
        points,
        np.flatnonzero(~on_boundary),
        np.flatnonzero(on_boundary),
        mass,
        stiffness,
        load,
        _boundary_data(points[on_boundary]),
    )


def _reference(host: HostHeat, time: float, /) -> np.ndarray:
    """Exact semi-discrete nodal field at ``time`` from ``u(0) = 0``.

    ``M_ff z' + K_ff z = p cos(t) b_f - sin(w t) K_fd G - w cos(w t) M_fd G`` with
    the forcing generated by ``(sin w t, cos w t, sin t, cos t)``.
    """
    free, fixed = host.free, host.fixed
    mass = host.mass[np.ix_(free, free)]
    stiffness = host.stiffness[np.ix_(free, free)]
    coupling_k = host.stiffness[np.ix_(free, fixed)] @ host.lift
    coupling_m = host.mass[np.ix_(free, fixed)] @ host.lift
    n = free.size
    generator = np.zeros((n + 4, n + 4))
    generator[:n, :n] = -np.linalg.solve(mass, stiffness)
    generator[:n, n] = -np.linalg.solve(mass, coupling_k)
    generator[:n, n + 1] = -OMEGA * np.linalg.solve(mass, coupling_m)
    generator[:n, n + 3] = PARAMETER * np.linalg.solve(mass, host.load[free])
    generator[n, n + 1], generator[n + 1, n] = OMEGA, -OMEGA
    generator[n + 2, n + 3], generator[n + 3, n + 2] = 1.0, -1.0
    initial = np.zeros(n + 4)
    initial[n + 1], initial[n + 3] = 1.0, 1.0
    evolved = scipy.linalg.expm(time * generator) @ initial
    field = np.zeros(host.points.shape[0])
    field[free] = evolved[:n]
    field[fixed] = np.sin(OMEGA * time) * host.lift
    return field


def _union_values(
    host: HostHeat, region: cases.Region, values: np.ndarray, /
) -> np.ndarray:
    """Host union-mesh values at the region's degree-of-freedom points."""
    distances = np.linalg.norm(
        region.dof_points[:, None, :] - host.points[None, :, :], axis=-1
    )
    return values[np.argmin(distances, axis=1)]


def _field_error(
    transient: cpl.PreparedCoupledTransient,
    scenario: Scenario,
    host: HostHeat,
    time: float,
    native: Array,
    /,
) -> float:
    state = transient.state_view(native)
    reference = _reference(host, time)
    errors = []
    for name, item in (("left", scenario.left), ("right", scenario.right)):
        field = np.asarray(
            transient.field(name, "u", time, state, jnp.asarray(PARAMETER))
        )
        errors.append(np.max(np.abs(field - _union_values(host, item.region, reference))))
    return float(max(errors) / np.max(np.abs(reference)))


def _stage_termination() -> phx.nonlinear.NonlinearTermination:
    # The BDF stage residual is dominated by M / h: at small adaptive steps the
    # Newton correction that removes a 1e-10 residual is far below the default
    # 1e-12 state step floor, which would report a converging stage as
    # stagnated. The residual threshold alone certifies the stage.
    return phx.nonlinear.NonlinearTermination(
        absolute_residual=1e-10,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=12,
    )


def _fixed_policy() -> phx.solver.DAESolvePolicy:
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_termination=_stage_termination(),
        initialization_termination=_stage_termination(),
    )


def _adaptive_policy() -> phx.solver.DAESolvePolicy:
    return phx.solver.DAESolvePolicy(
        method=phx.solver.BDFMethod(2),
        nonlinear_termination=_stage_termination(),
        initialization_termination=_stage_termination(),
        adaptive=phx.solver.DAEAdaptivePolicy(
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            maximum_accepted_steps=2048,
            maximum_attempts=4096,
        ),
    )


@pytest.fixture(scope="module")
def matching() -> tuple[Scenario, cpl.PreparedCoupledTransient]:
    scenario = _scenario("matching")
    return scenario, _transient(scenario)


def test_bdf_transient_heat_with_moving_lift_converges_to_semidiscrete_reference(
    matching: tuple[Scenario, cpl.PreparedCoupledTransient],
) -> None:
    scenario, transient = matching
    host = _host_heat(2.0)
    errors = []
    for steps in (16, 32):
        grid = phx.dynamics.TimeGrid(
            jnp.linspace(0.0, 0.4, steps + 1), time_id=f"heat-{steps}"
        )
        solution = cpl.solve_coupled_transient(
            transient,
            transient.state_space.zeros(),
            grid,
            parameters=jnp.asarray(PARAMETER),
            policy=_fixed_policy(),
        )
        assert bool(solution.accepted)
        assert bool(jnp.all(solution.certificate.accepted))
        errors.append(
            _field_error(transient, scenario, host, 0.4, solution.dae.states[-1])
        )

    assert transient.roles == ("differential", "differential")
    assert transient.analysis.status == "success"
    assert errors[1] < 5e-3
    # Second-order BDF: halving the step reduces the error by about four.
    assert errors[0] / errors[1] > 3.0


def test_accepted_history_continuation_reproduces_one_adaptive_segment(
    matching: tuple[Scenario, cpl.PreparedCoupledTransient],
) -> None:
    scenario, transient = matching
    host = _host_heat(2.0)
    parameters = jnp.asarray(PARAMETER)
    zero = transient.state_space.zeros()

    def grid(times: tuple[float, ...], name: str) -> phx.dynamics.TimeGrid:
        return phx.dynamics.TimeGrid(jnp.asarray(times), time_id=name)

    full = cpl.solve_coupled_transient(
        transient,
        zero,
        grid((0.0, 0.2, 0.4), "full"),
        parameters=parameters,
        policy=_adaptive_policy(),
    )
    first = cpl.solve_coupled_transient(
        transient,
        zero,
        grid((0.0, 0.2), "first"),
        parameters=parameters,
        policy=_adaptive_policy(),
    )
    second = cpl.solve_coupled_transient(
        transient,
        zero,
        grid((0.2, 0.4), "second"),
        parameters=parameters,
        policy=_adaptive_policy(),
        continuation=first.continuation,
    )

    assert bool(full.accepted & first.accepted & second.accepted)
    assert second.dae.initialization.nonlinear_result is None
    counts = (int(first.dae.step_history.count), int(second.dae.step_history.count))
    segmented = np.concatenate(
        (
            np.asarray(first.dae.step_history.accepted_times[: counts[0]]),
            np.asarray(second.dae.step_history.accepted_times[: counts[1]]),
        )
    )
    total = int(full.dae.step_history.count)
    np.testing.assert_allclose(
        segmented,
        np.asarray(full.dae.step_history.accepted_times[:total]),
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(second.dae.states[-1]),
        np.asarray(full.dae.states[-1]),
        rtol=0,
        atol=1e-12,
    )
    assert _field_error(transient, scenario, host, 0.4, second.dae.states[-1]) < 1e-5


def _host_array_dae(host: HostHeat, /) -> phx.dynamics.DifferentialAlgebraicSystem:
    """Native array DAE of the union mesh with a quasistatic right region."""
    free, fixed = host.free, host.fixed
    mass = jnp.asarray(host.mass[np.ix_(free, free)])
    stiffness = jnp.asarray(host.stiffness[np.ix_(free, free)])
    coupling_k = jnp.asarray(host.stiffness[np.ix_(free, fixed)] @ host.lift)
    coupling_m = jnp.asarray(host.mass[np.ix_(free, fixed)] @ host.lift)
    load = jnp.asarray(host.load[free])
    differential = np.any(host.mass[np.ix_(free, free)] != 0.0, axis=1)

    def residual(time: Array, state: Array, rate: Array, parameter: Array) -> Array:
        return (
            mass @ rate
            + stiffness @ state
            + jnp.sin(OMEGA * time) * coupling_k
            + OMEGA * jnp.cos(OMEGA * time) * coupling_m
            - parameter * jnp.cos(time) * load
        )

    roles = tuple("differential" if value else "algebraic" for value in differential)
    return phx.dynamics.DifferentialAlgebraicSystem(
        residual,
        state_shape=(free.size,),
        structure=phx.dynamics.DAEStructure(roles, component_axis=-1),
        system_id="host-union-heat-quasistatic-right",
    )


def test_named_block_transient_matches_native_array_dae_with_scales_and_roles() -> None:
    scenario = _scenario("matching")
    scales = (
        cpl.TransientScale("left", "u", state=2.0, rate=5.0, residual=0.25),
        cpl.TransientScale("right", "u", state=0.5, rate=3.0, residual=4.0),
    )
    transient = _transient(scenario, right_capacity=None, scales=scales)
    host = _host_heat(0.0)
    grid = phx.dynamics.TimeGrid(jnp.linspace(0.0, 0.3, 13), time_id="quasistatic")
    named = cpl.solve_coupled_transient(
        transient,
        transient.state_space.zeros(),
        grid,
        parameters=jnp.asarray(PARAMETER),
        policy=_fixed_policy(),
    )
    array_system = _host_array_dae(host)
    native = phx.solver.solve_dae(
        phx.solver.DifferentialAlgebraicProblem(
            array_system, jnp.zeros(host.free.size), args=jnp.asarray(PARAMETER)
        ),
        grid,
        policy=_fixed_policy(),
    )

    assert transient.roles == ("differential", "algebraic")
    roles = {value.path: value.role for value in transient.adapter.variables}
    assert roles == {
        ("two-regions.left.u",): "differential",
        ("two-regions.right.u",): "algebraic",
    }
    scale = transient.adapter.scale_view("state")
    assert {float(jnp.max(block)) for block in scale} == {2.0, 0.5}
    assert bool(named.accepted) and bool(native.successful)
    for index in range(1, 13):
        time = float(grid.times[index])
        state = transient.state_view(named.dae.states[index])
        union = np.zeros(host.points.shape[0])
        union[host.free] = np.asarray(native.states[index])
        union[host.fixed] = np.sin(OMEGA * time) * host.lift
        for name, item in (("left", scenario.left), ("right", scenario.right)):
            field = transient.field(name, "u", time, state, jnp.asarray(PARAMETER))
            np.testing.assert_allclose(
                np.asarray(field), _union_values(host, item.region, union), atol=1e-9
            )


def test_mortar_multiplier_transient_is_refused_with_structural_evidence() -> None:
    scenario = _scenario("mortar-side-trace")

    with pytest.raises(ValueError, match="differentiation-capacity-exceeded") as error:
        _transient(scenario)

    assert "required differentiations" in str(error.value)


def test_transient_declarations_are_refused_without_explicit_roles() -> None:
    scenario = _scenario("matching", side="right")
    prepared = scenario.coupled.prepared
    arguments = _arguments(scenario.left, scenario.right)

    with pytest.raises(ValueError, match="neither transient nor declared quasistatic"):
        cpl.prepare_coupled_transient(
            prepared,
            fields=(cpl.TransientField("left", "u"),),
            arguments=arguments,
            arguments_id="undeclared",
        )
    with pytest.raises(ValueError, match="is not differential"):
        cpl.prepare_coupled_transient(
            prepared,
            fields=(cpl.TransientField("right", "u"),),
            quasistatic=("left",),
            arguments=arguments,
            arguments_id="eliminated-transient-side",
            parameters=jnp.asarray(PARAMETER),
        )
