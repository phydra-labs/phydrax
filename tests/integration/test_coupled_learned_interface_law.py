#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A learned, monotone, dissipative contact law between two coupled regions.

The finite-element / virtual-element plate of ``tests._support.coupled_plate``
keeps its components, bindings, parameters, and interface sides; only the mortar
transmission is replaced by a ``ConservativeFluxLaw`` whose flux is
``MonotoneInterfaceConductance``: with the temperature jump
``d = u_triangles - u_polygons`` the heat leaving the triangles through ``x = 1``
is ``q(d) = h d + phi'(d) - phi'(0)``, where ``phi`` is an
``InputConvexNetwork`` bound with ``MODEL`` authority through a
``ParameterBinding``.

The temperature depends on ``x`` only, so the right region's heat balance fixes
the interface heat ``Q = -(s + g)`` independently of the contact law. The host
reference solves ``q(d) = Q`` for the jump by bisection with the network's own
derivative and shifts the analytic right-region temperature by it. Interface
integrals are recomputed on the host from the piecewise-linear traces of both
sides with composite Gauss quadrature on their common breakpoints.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from jax import Array

import phydrax as phx
from phydrax.solver import coupling as cpl
from tests._support import coupled_plate as cp


# Solve-coordinate state of the prepared plate: one block per component.
type SolveState = tuple[tuple[Array, ...], ...]

LEVEL = 0
KAPPA_LEFT = 1.0
KAPPA_RIGHT = 2.0
WALL_FLUX = 0.5
BASELINE = 0.5
# Heat leaving the triangles through x = 1 per unit depth: the right region's
# source and wall inflow cross the interface toward the Dirichlet wall.
INTERFACE_HEAT = -(cp.SOURCE + WALL_FLUX)

RESPONSE = cpl.RuntimeInput("polygons", "contact-potential")
CONTACT_PORT = phx.ValuePort(
    "contact-potential",
    event_shape=(),
    component_ids=("phi",),
    representation="scalar",
    dimensions=(cp.HEAT_FLUX_UNIT.dimension,),
)
JUMP_PORT = phx.ValuePort(
    "contact-jump",
    event_shape=(),
    component_ids=("d",),
    representation="scalar",
    dimensions=(phx.units.KELVIN.dimension,),
)


def _contact_binding() -> cpl.ParameterBinding:
    return cpl.ParameterBinding(
        "contact-potential",
        CONTACT_PORT,
        targets=(RESPONSE,),
        role="coefficient",
        derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
    )


def _network(key: int) -> phx.nn.models.InputConvexNetwork:
    return phx.nn.models.InputConvexNetwork(
        in_size="scalar", width_size=8, depth=2, key=jr.key(key)
    )


def _bound(
    model: phx.AbstractArrayModel,
    authority: phx.ComponentAuthority = phx.ComponentAuthority.MODEL,
) -> phx.ComponentBinding:
    return phx.bind_component(
        model,
        authority,
        owner_ports=phx.ModelPorts(inputs=(JUMP_PORT,), outputs=(CONTACT_PORT,)),
    )


def _parameters(potential: phx.ComponentBinding) -> dict[str, object]:
    return {
        "conductivity-left": jnp.asarray(KAPPA_LEFT, jnp.float64),
        "conductivity-right": jnp.asarray(KAPPA_RIGHT, jnp.float64),
        "heat-flux": jnp.asarray(WALL_FLUX, jnp.float64),
        "contact-potential": potential,
    }


def _plan(
    plate: cp.CoupledPlate, flux: cpl.AbstractInterfaceFlux
) -> cpl.CoupledProblemPlan:
    law = cpl.ConservativeFluxLaw("contact", plate.binding, plate.law.sides, flux)
    return cpl.CoupledProblemPlan(
        "contact-plate",
        components=(plate.triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(law,),
        parameters=(*cp.parameter_bindings(), _contact_binding()),
    )


@dataclass(frozen=True)
class _Contact:
    plate: cp.CoupledPlate
    flux: cpl.MonotoneInterfaceConductance
    prepared: cpl.PreparedCoupledProblem


@pytest.fixture(scope="module")
def contact() -> _Contact:
    plate = cp.build_plate(LEVEL)
    flux = cpl.MonotoneInterfaceConductance(RESPONSE, baseline=BASELINE)
    prepared = cpl.prepare_coupled_problem(
        _plan(plate, flux),
        interface_owners=(plate.cover,),
        parameters=_parameters(_bound(_network(0))),
    )
    return _Contact(plate, flux, prepared)


def _solve(contact: _Contact, potential: phx.ComponentBinding) -> cpl.CoupledSolution:
    return cpl.solve_coupled_problem(
        contact.prepared, parameters=_parameters(potential), policy=cp.dense_policy()
    )


# --- Host references --------------------------------------------------------------------


def _host_heat_flow(model: phx.AbstractArrayModel) -> Callable[[np.ndarray], np.ndarray]:
    """``q(d) = h d + phi'(d) - phi'(0)`` from the network's own input derivative."""
    slope = jax.jit(jax.vmap(jax.grad(lambda value: model(value))))
    offset = float(slope(jnp.zeros((1,), jnp.float64))[0])

    def heat(jumps: np.ndarray) -> np.ndarray:
        values = np.asarray(slope(jnp.asarray(np.ravel(jumps), jnp.float64)))
        return BASELINE * jumps + (values.reshape(np.shape(jumps)) - offset)

    return heat


def _contact_jump(heat: Callable[[np.ndarray], np.ndarray]) -> float:
    """The jump carrying ``INTERFACE_HEAT``: bisection on the nondecreasing ``q``."""
    lower, upper = (
        -4.0 * abs(INTERFACE_HEAT) / BASELINE,
        4.0 * abs(INTERFACE_HEAT) / BASELINE,
    )
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if heat(np.asarray([middle]))[0] < INTERFACE_HEAT:
            lower = middle
        else:
            upper = middle
    return 0.5 * (lower + upper)


def _trace(nodes: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vertex values of a piecewise-linear trace on ``x = 1``, sorted along ``y``."""
    on_interface = np.isclose(nodes[:, 0], cp.INTERFACE_X)
    order = np.argsort(nodes[on_interface, 1])
    return nodes[on_interface, 1][order], values[on_interface][order]


def _interface_integral(
    plate: cp.CoupledPlate,
    minus: np.ndarray,
    plus: np.ndarray,
    density: Callable[[np.ndarray], np.ndarray],
) -> float:
    """``int_0^1 density(d(y)) dy`` by 8-point Gauss on the common trace breakpoints."""
    minus_y, minus_values = _trace(plate.triangle_nodes, minus)
    plus_y, plus_values = _trace(plate.polygon_nodes, plus)
    breaks = np.unique(np.concatenate([minus_y, plus_y]))
    nodes, weights = np.polynomial.legendre.leggauss(8)
    lower, upper = breaks[:-1, None], breaks[1:, None]
    y = 0.5 * (lower + upper) + 0.5 * (upper - lower) * nodes
    jump = np.interp(y, minus_y, minus_values) - np.interp(y, plus_y, plus_values)
    return float(np.sum(0.5 * (upper - lower) * weights * density(jump)))


# --- Accepted nonlinear solution --------------------------------------------------------


def test_learned_contact_reproduces_the_analytic_nonlinear_plate(
    contact: _Contact,
) -> None:
    network = _network(0)
    solution = _solve(contact, _bound(network))
    assert contact.prepared.execution == "nonlinear"
    assert solution.nonlinear is not None and solution.linear is None
    assert bool(solution.native_successful) and bool(solution.accepted)

    jump = _contact_jump(_host_heat_flow(network))
    # The learned part of the law carries the jump away from the linear
    # conductance's Q / h, far beyond the discretization tolerance below.
    assert abs(jump - INTERFACE_HEAT / BASELINE) > 0.5
    plate = contact.plate
    left = cp.exact_temperature(plate.triangle_nodes, KAPPA_LEFT, KAPPA_RIGHT, WALL_FLUX)
    # The analytic right branch at x = 1 equals the left one, so the polygon
    # nodes on the interface receive the right branch shifted by the jump.
    right = (
        cp.exact_temperature(plate.polygon_nodes, KAPPA_LEFT, KAPPA_RIGHT, WALL_FLUX)
        - jump
    )
    # P1 and degree-1 virtual elements on the level-0 meshes (quadratic exact
    # temperature): the same nodal tolerance as the mortar plate.
    np.testing.assert_allclose(
        solution.field("triangles", "u"), left, rtol=0.0, atol=2.0e-2
    )
    np.testing.assert_allclose(
        solution.field("polygons", "u"), right, rtol=0.0, atol=2.0e-2
    )


def test_learned_contact_conserves_heat_and_dissipates(contact: _Contact) -> None:
    network = _network(0)
    potential = _bound(network)
    solution = _solve(contact, potential)
    assert bool(solution.accepted)
    report = solution.interface("contact")
    for name in ("law-residual", "flux-conservation"):
        scale = float(report.scales[report.names.index(name)])
        assert float(report.value(name)) <= 1.0e-10 * scale

    minus = np.asarray(solution.field("triangles", "u"))
    plus = np.asarray(solution.field("polygons", "u"))
    bound = contact.prepared.bind_arguments(parameters=_parameters(potential))
    (contribution,) = contact.prepared.chart.laws[0].contributions
    assert isinstance(contribution, cpl.ResidualContribution)
    minus_rows, plus_rows = (
        np.asarray(rows)
        for rows in contribution.residual.evaluate(
            (jnp.asarray(minus), jnp.asarray(plus)), bound.arguments
        )
    )
    heat = _host_heat_flow(network)
    # Constants lie in both trace spaces: the row sums are the heat each side
    # loses and gains, equal and opposite, and the host quadrature of q(d(y))
    # recovers the right region's balance Q = -(s + g).
    host_heat = _interface_integral(contact.plate, minus, plus, heat)
    assert host_heat == pytest.approx(INTERFACE_HEAT, rel=1.0e-8)
    assert float(np.sum(minus_rows)) == pytest.approx(host_heat, rel=1.0e-8)
    assert float(np.sum(plus_rows)) == pytest.approx(-host_heat, rel=1.0e-8)
    # Pairing the rows with the solved traces is the dissipated interface power
    # int d q(d) dy, which a monotone law makes nonnegative.
    dissipation = float(minus @ minus_rows + plus @ plus_rows)
    host_dissipation = _interface_integral(
        contact.plate, minus, plus, lambda jump: jump * heat(jump)
    )
    assert host_dissipation > 0.0
    assert dissipation == pytest.approx(host_dissipation, rel=1.0e-8)


@pytest.mark.parametrize("key", [0, 1, 2], ids=("key-0", "key-1", "key-2"))
def test_every_certified_potential_gives_a_monotone_dissipative_heat_flow(
    contact: _Contact, key: int
) -> None:
    bound = contact.prepared.bind_arguments(parameters=_parameters(_bound(_network(key))))
    # An even count keeps the symmetric grid off d = 0, where the roundoff of
    # phi'(d) - phi'(0) would decide the sign of d q(d).
    jumps = np.linspace(-6.0, 6.0, 240)
    heat = np.asarray(contact.flux.heat_flow(jnp.asarray(jumps), bound.arguments))
    assert (
        float(contact.flux.heat_flow(jnp.zeros((1,), jnp.float64), bound.arguments)[0])
        == 0.0
    )
    # Heat flows from hot to cold and grows with the jump at least at rate h:
    # d q(d) >= h d^2 and q(b) - q(a) >= h (b - a), up to float64 roundoff of
    # the O(10) heat values.
    roundoff = 1.0e-12
    assert np.all(jumps * heat >= BASELINE * jumps**2 - roundoff)
    assert np.all(np.diff(heat) >= BASELINE * np.diff(jumps) - roundoff)


def test_an_affine_potential_recovers_the_linear_conductance(contact: _Contact) -> None:
    # Zero input weights of the hidden layers leave only the network's direct
    # affine term: phi'(d) - phi'(0) = 0 and q(d) = h d, the certified linear law.
    network = _network(3)
    depth = len(network.state_layers)
    affine = eqx.tree_at(
        lambda model: tuple(layer.weight for layer in model.input_layers[:depth]),
        network,
        tuple(jnp.zeros_like(layer.weight) for layer in network.input_layers[:depth]),
    )
    learned = _solve(contact, _bound(affine))
    plate = contact.plate
    linear = cpl.prepare_coupled_problem(
        _plan(plate, cpl.InterfaceConductance(BASELINE)),
        interface_owners=(plate.cover,),
        parameters=_parameters(_bound(affine)),
    )
    assert linear.execution == "linear"
    reference = cpl.solve_coupled_problem(
        linear, parameters=_parameters(_bound(affine)), policy=cp.dense_policy()
    )
    assert bool(learned.accepted) and bool(reference.accepted)
    for component in ("triangles", "polygons"):
        np.testing.assert_allclose(
            learned.field(component, "u"),
            reference.field(component, "u"),
            rtol=0.0,
            atol=1.0e-10,
        )


# --- Refusals ---------------------------------------------------------------------------


def test_a_potential_without_a_convexity_certificate_is_refused(
    contact: _Contact,
) -> None:
    mlp = phx.nn.models.MLP(
        in_size="scalar", out_size="scalar", width_size=8, depth=2, key=jr.key(1)
    )
    with pytest.raises(ValueError, match="requires a monotone response"):
        _solve(contact, _bound(mlp))


@pytest.mark.parametrize(
    "authority",
    [phx.ComponentAuthority.SURROGATE, phx.ComponentAuthority.ACCELERATOR],
    ids=("surrogate", "accelerator"),
)
def test_a_potential_that_is_not_a_model_cannot_change_the_contact_law(
    contact: _Contact, authority: phx.ComponentAuthority
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"component with {authority.value} authority cannot supply parameter "
        r"'contact-potential'",
    ):
        contact.prepared.bind_arguments(
            parameters=_parameters(_bound(_network(0), authority))
        )


def test_a_learned_response_outside_a_parameter_binding_is_refused() -> None:
    """A raw model at the law's input would bypass the authority and port admission."""
    plate = cp.build_plate(LEVEL)
    flux = cpl.MonotoneInterfaceConductance(RESPONSE, baseline=BASELINE)
    law = cpl.ConservativeFluxLaw("contact", plate.binding, plate.law.sides, flux)
    plan = cpl.CoupledProblemPlan(
        "contact-plate",
        components=(plate.triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(law,),
        parameters=cp.parameter_bindings(),
    )
    values = _parameters(_bound(_network(0)))
    del values["contact-potential"]

    with pytest.raises(ValueError, match="which no refresh ParameterBinding targets"):
        cpl.prepare_coupled_problem(
            plan,
            interface_owners=(plate.cover,),
            arguments={"polygons": {"contact-potential": _network(0)}},
            parameters=values,
        )


def test_a_negative_baseline_conductance_is_refused() -> None:
    with pytest.raises(ValueError, match="baseline must be a nonnegative conductance"):
        cpl.MonotoneInterfaceConductance(RESPONSE, baseline=-1.0)


# --- Fixed-structure derivative route ---------------------------------------------------


def test_nonlinear_contact_refuses_implicit_parameter_derivatives(
    contact: _Contact,
) -> None:
    capability = contact.prepared.derivative_capability(cp.dense_policy())
    refused = dict(capability.refused)
    # Newton publishes no qualified implicit solution map, for the learned
    # potential and for every other bound parameter alike.
    assert capability.derivative_contract.route is phx.DerivativeRoute.STOPPED
    for binding_id in ("contact-potential", "conductivity-left", "heat-flux"):
        assert not capability.admits(binding_id)
        assert "nonlinear coupled solve" in refused[binding_id]


def test_state_design_response_is_the_derivative_of_accepted_nonlinear_solves(
    contact: _Contact,
) -> None:
    prepared = contact.prepared
    potential = _bound(_network(0))
    admission = phx.optim.StateDesignComponentAdmission(
        potential, kind=phx.ObjectiveKind.DATA_FIT
    )
    design, model_state, fixed = phx.partition_parameters(potential.model)
    bound = prepared.bind_arguments(parameters=_parameters(potential))

    def with_design(design: Any, arguments: dict[str, object]) -> dict[str, object]:
        owner = arguments["polygons"]
        assert isinstance(owner, Mapping)
        model = phx.combine_parameters(design, model_state, fixed)
        return {**arguments, "polygons": {**owner, "contact-potential": model}}

    def residual(
        state: SolveState, design: Any, arguments: dict[str, object]
    ) -> SolveState:
        return prepared.residual(state, with_design(design, arguments))

    def mean_temperature(
        state: SolveState, design: Any, arguments: dict[str, object]
    ) -> Array:
        field = prepared.field("polygons", "u", state, with_design(design, arguments))
        return jnp.mean(field)

    # The state owner re-solves from the accepted Newton state and certifies
    # the residual itself (absolute threshold 1e-10). Its least-squares stop is
    # set at the same scale: the default 1e-14 gradient-norm stop lies below the
    # roundoff of a converged state, where the method reports stagnation.
    problem = phx.optim.StateDesignProblem(
        residual,
        mean_temperature,
        state_solver=phx.optim.LeastSquaresStateSolver(
            termination=phx.optim.OptimizationTermination(
                absolute_optimality=1.0e-10, relative_optimality=0.0, maximum_steps=20
            )
        ),
        problem_id="learned-contact",
    )
    initial = _solve(contact, potential)
    assert bool(initial.accepted)
    point = phx.optim.prepare_state_design_linearization(
        problem,
        design,
        initial.state,
        args=bound.arguments,
        linear_policy=cp.dense_policy(),
        component=admission,
    )
    state_result = point.state_result
    assert state_result is not None
    assert int(state_result.status) == phx.optim.OptimizationStatus.SUCCESS
    assert bool(point.accepted)
    response = phx.optim.state_design_response_vjp(point)
    assert response.adjoint_acceptance is not None
    assert bool(response.adjoint_acceptance.accepted) and bool(response.accepted)

    leaves, structure = jax.tree.flatten(design)
    keys = jr.split(jr.key(7), len(leaves))
    raw = [jr.normal(key, leaf.shape, leaf.dtype) for key, leaf in zip(keys, leaves)]
    norm = jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in raw))
    direction = jax.tree.unflatten(structure, [leaf / norm for leaf in raw])
    adjoint = sum(
        float(jnp.vdot(left, right))
        for left, right in zip(
            jax.tree.leaves(response.design_cotangent), jax.tree.leaves(direction)
        )
    )

    def solved_mean(step: float) -> float:
        shifted = jax.tree.map(lambda left, right: left + step * right, design, direction)
        model = phx.combine_parameters(shifted, model_state, fixed)
        solution = _solve(contact, eqx.tree_at(lambda item: item.model, potential, model))
        assert bool(solution.accepted)
        return float(jnp.mean(solution.field("polygons", "u")))

    step = 1.0e-4
    central = (solved_mean(step) - solved_mean(-step)) / (2.0 * step)
    assert float(response.values) == pytest.approx(solved_mean(0.0), rel=1.0e-12)
    # Newton converges quadratically far below the 1e-8 solve tolerance; the
    # central difference carries O(step^2) truncation and eps |u| / step roundoff.
    assert adjoint == pytest.approx(central, rel=1.0e-6)
