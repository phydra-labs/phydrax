#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A physics-informed network next to a classical finite-element region.

Physics. Steady heat conduction ``-div(kappa grad u) = s`` (``s = 2``) on the
plate ``[0, 2] x [0, 1]`` with constant conductivities ``kappa_left`` and
``kappa_right`` on either side of the interface ``x = 1``, ``u = 0`` on
``x = 0``, the prescribed boundary heat flux ``g = kappa du/dn`` on ``x = 2``,
and insulated top and bottom sides. The exact solution depends on ``x`` only:

    u(x) = (-s x^2 / 2 + (2 s + g) x) / kappa_left                 for x <= 1,
    u(x) = (-s (x - 1)^2 / 2 + (s + g) (x - 1)) / kappa_right
           + (3 s / 2 + g) / kappa_left                              for x >= 1.

The left half is a classical P1 finite-element owner on a structured triangle
mesh. The right half is a physics-informed network ``u_theta(x, y)`` that keeps
its ``SURROGATE`` authority throughout; it is bound once with
``phx.ComponentBinding`` and never re-labeled as a physical model. The network
declares a typed ``temperature`` output port, the same port the FE field view
declares, so the two fields compose only with matching units and identity.

1. Direct pretraining. ``FunctionalSolver`` trains ``u_theta`` on the physical
   residuals of the right region (interior ``kappa_right lap u + s``, heat flux
   ``kappa_right u_x - g`` at ``x = 2``, insulated ``u_y`` at ``y = 0, 1``) with a
   provisional interface Dirichlet value ``u = 0``. The interface heat flux
   ``kappa_right u_x(1) = s + g`` of the right region does not depend on that
   Dirichlet value, so pretraining already fixes the flux the classical owner
   needs.
2. Accepted classical response. The finite-element owner reads the network's
   interface heat flux ``kappa_right du_theta/dx`` as a ``BoundaryLoadAction`` on
   its ``x = 1`` facets. ``StateDesignProblem`` takes the FE solve coordinates as
   state and the network's PARAMETER lane as design, admitted by
   ``StateDesignComponentAdmission(binding, kind=PHYSICAL_RESIDUAL, surfaces=
   (INPUT, MODEL_PARAMETER))``; its objective is the coupled interface-continuity
   residual ``J = 1/2 sum_w (u_theta - u_FE)^2`` at the FE interface trace
   sites. The port-declaring wrapper keeps the conservative (undeclared)
   regularity contract, so the admission uses
   ``RegularityPolicy(allow_undeclared=True)`` and records that condition.
   ``prepare_state_design_linearization`` and
   ``state_design_response_vjp`` return ``J``, its design cotangent, and
   separate accepted primal (Gauss-Newton with a dense QR step) and adjoint
   (dense LU transpose solve) evidence.
3. Dirichlet-Neumann coupling. The accepted FE solution becomes a fixed
   ``DiscreteFieldFunctionView``; ``FunctionalSolver`` continues training the
   network with the FE interface trace as its interface Dirichlet target. A new
   accepted classical response reports the coupled interface residual.

No new optimizer and no widened authority: a ``SOLUTION_MAP`` admission of the
surrogate, and an implicit ``SolverObjective`` training it, are refused.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import final, NoReturn

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from phydrax.operators.differential import laplacian, partial_n


jax.config.update("jax_enable_x64", True)

INTERFACE_X = 1.0
SOURCE = 2.0
KAPPA_LEFT = 1.3
KAPPA_RIGHT = 0.7
HEAT_FLUX = 0.5
RESOLUTION = 8  # Mesh intervals per side of the FE square (two triangles per cell).
TRACE_POINTS = 3  # Gauss points per interface facet.
PRETRAINING_STEPS = 200
COUPLING_STEPS = 200
INTERFACE_FLUX = "interface-flux"
TEMPERATURE = phx.ValuePort(
    "temperature",
    event_shape=(),
    component_ids=("T",),
    representation="scalar-field",
    dimensions=(phx.units.DimensionSignature({"temperature": 1}),),
)


def exact_temperature(points: ArrayLike) -> np.ndarray:
    """Host reference ``u(x)`` of the piecewise-constant-conductivity plate."""
    x = np.asarray(points, dtype=np.float64)[..., 0]
    left = (-0.5 * SOURCE * x**2 + (2.0 * SOURCE + HEAT_FLUX) * x) / KAPPA_LEFT
    offset = (1.5 * SOURCE + HEAT_FLUX) / KAPPA_LEFT
    right = (
        -0.5 * SOURCE * (x - INTERFACE_X) ** 2 + (SOURCE + HEAT_FLUX) * (x - INTERFACE_X)
    ) / KAPPA_RIGHT + offset
    return np.where(x <= INTERFACE_X, left, right)


RIGHT_REGION = phx.domain.HyperRectangle(
    np.asarray([INTERFACE_X, 0.0]), np.asarray([2.0, 1.0]), label="x"
)
POSITION = RIGHT_REGION.value_port("x")
POSITION_MAPPING = phx.PortMapping(inputs=[(POSITION.port_id, POSITION.port_id)])


@final
class TemperatureNetwork(phx.AbstractArrayModel):
    """A network ``u_theta(x, y)`` that declares its temperature output port.

    The FE solution enters the network's interface condition as a typed
    ``DiscreteFieldFunctionView``; field algebra between the two requires both
    to declare the same ``TEMPERATURE`` port (units, event shape, variance).
    """

    network: phx.nn.models.MLP
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, network: phx.nn.models.MLP, /) -> None:
        self.network = network
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, x: Array, /, *, key: phx.typing.PRNGKey | None = None) -> Array:
        del key
        return self.network(x)

    def model_ports(self) -> phx.ModelPorts:
        return phx.ModelPorts(inputs=(POSITION,), outputs=(TEMPERATURE,))


# --- Classical finite-element owner (left) -------------------------------------------


def _triangle_mesh(resolution: int) -> phx.discretization.CellMesh:
    axis = np.linspace(0.0, 1.0, resolution + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    triangles: list[tuple[int, int, int]] = []
    for row in range(resolution):
        for column in range(resolution):
            corner = row * (resolution + 1) + column
            upper = corner + resolution + 1
            triangles += [(corner, corner + 1, upper + 1), (corner, upper + 1, upper)]
    return phx.discretization.CellMesh(
        jnp.asarray(points),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", jnp.asarray(np.asarray(triangles, np.int32))
            ),
        ),
    )


def _interface_facets(
    space: phx.discretization.FiniteElementDiscretization,
) -> IntegrationDomain:
    """The owner's exterior facets on the interface line ``x = 1``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    selected = np.all(np.isclose(np.asarray(probe.sites)[..., 0], INTERFACE_X), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[selected]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


def _left_conductivity(points: Array, context: object) -> Array:
    del context
    return KAPPA_LEFT * jnp.ones(points.shape[:-1])


def _source(points: Array, context: object) -> Array:
    del context
    return SOURCE * jnp.ones(points.shape[:-1])


def _network(arguments: object) -> TemperatureNetwork:
    if not isinstance(arguments, Mapping):
        raise TypeError("The FE owner's user arguments must be a mapping.")
    network = arguments[INTERFACE_FLUX]
    if not isinstance(network, TemperatureNetwork):
        raise TypeError("The interface heat flux is read from the bound network.")
    return network


def interface_heat_flux(points: Array, context: object) -> Array:
    """``kappa_right du_theta/dx`` of the network at the FE interface facet points.

    Heat leaving the right region through ``x = 1`` enters the left region, whose
    outward normal there is ``+x``: the network's conormal flux is the FE
    owner's prescribed boundary heat flux.
    """
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("FE coefficients receive the execution context.")
    network = _network(context.user_args)
    flat = points.reshape(-1, points.shape[-1])
    gradient = jax.vmap(jax.grad(network))(flat)
    return KAPPA_RIGHT * gradient[:, 0].reshape(points.shape[:-1])


@dataclass(frozen=True)
class ClassicalRegion:
    """Prepared-once finite-element owner of the left half and its interface trace."""

    space: phx.discretization.FiniteElementDiscretization
    problem: phx.equations.CompiledFiniteElementProblem
    trace: phx.discretization.PreparedTraceAction
    sites: Array
    weights: Array
    nodes: np.ndarray


def classical_region(resolution: int = RESOLUTION) -> ClassicalRegion:
    space = phx.discretization.FiniteElementPlan(
        _triangle_mesh(resolution),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    nodes = np.asarray(space.dof_maps[0].dof_coordinates)
    interface = _interface_facets(space)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "left-conduction",
            "u",
            (
                phx.equations.DiffusionAction(
                    "u",
                    phx.equations.coefficient(
                        _left_conductivity, coefficient_id="conductivity-left"
                    ),
                ),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(_source, coefficient_id="source")
                ),
                phx.equations.BoundaryLoadAction(
                    "u",
                    phx.equations.coefficient(
                        interface_heat_flux, coefficient_id=INTERFACE_FLUX
                    ),
                    action_id="pinn-interface-heat-flux",
                    domain=interface,
                ),
            ),
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=np.isclose(nodes[:, 0], 0.0)
        ),
        dirichlet_values=0.0,
    )
    trace = space.prepare_side_trace(
        "u", interface, rule=FacetTraceRule(points=TRACE_POINTS)
    )
    return ClassicalRegion(
        space, problem, trace, jnp.asarray(trace.sites), jnp.asarray(trace.weights), nodes
    )


# --- Physics-informed network (right) --------------------------------------------------


def surrogate_network(key: phx.typing.PRNGKey) -> TemperatureNetwork:
    """A ``tanh`` MLP declaring the temperature port of the right region."""
    return TemperatureNetwork(
        phx.nn.models.MLP(
            in_size=2,
            out_size="scalar",
            width_size=16,
            depth=2,
            activation=jnp.tanh,
            key=key,
        )
    )


def surrogate_binding(network: TemperatureNetwork) -> phx.ComponentBinding:
    """The network's one binding: a ``SURROGATE`` temperature field of the right region."""
    ports = network.model_ports()
    return phx.ComponentBinding(
        network,
        authority=phx.ComponentAuthority.SURROGATE,
        owner_ports=ports,
        port_mapping=phx.PortMapping(
            inputs=[(POSITION.port_id, POSITION.port_id)],
            outputs=[(TEMPERATURE.port_id, TEMPERATURE.port_id)],
        ),
    )


@dataclass(frozen=True)
class Collocation:
    """Fixed collocation points of the right region's physical residuals."""

    interior: np.ndarray
    wall: np.ndarray
    insulated: np.ndarray
    interface: np.ndarray


def collocation(region: ClassicalRegion, count: int = 12) -> Collocation:
    """Tensor midpoints inside, uniform edge points, and the FE interface trace sites."""
    midpoints = (np.arange(count, dtype=np.float64) + 0.5) / count
    grid_x, grid_y = np.meshgrid(INTERFACE_X + midpoints, midpoints, indexing="ij")
    edge = np.linspace(0.0, 1.0, count + 1)
    return Collocation(
        interior=np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1),
        wall=np.stack((np.full_like(edge, 2.0), edge), axis=-1),
        insulated=np.concatenate(
            (
                np.stack((INTERFACE_X + edge, np.zeros_like(edge)), axis=-1),
                np.stack((INTERFACE_X + edge, np.ones_like(edge)), axis=-1),
            )
        ),
        interface=np.asarray(region.sites).reshape(-1, 2),
    )


def _penalty(
    condition: phx.conditions.Residual | phx.conditions.Observation, points: np.ndarray
) -> phx.terms.ResidualPenalty:
    component = condition.on
    if not isinstance(component, phx.domain.DomainComponent):
        raise TypeError("Each residual lives on one domain component.")
    batch = component.points(jnp.asarray(points))
    source = phx.integration.fixed(
        phx.integration.from_samples(phx.integration.mean_over(component), batch)
    )
    return phx.terms.ResidualPenalty(condition, source)


def _conduction(u: phx.domain.DomainFunction) -> phx.domain.DomainFunction:
    return KAPPA_RIGHT * laplacian(u, var="x") + SOURCE


def _wall_heat_flux(u: phx.domain.DomainFunction) -> phx.domain.DomainFunction:
    return KAPPA_RIGHT * partial_n(u, var="x", axis=0, order=1) - HEAT_FLUX


def _insulated(u: phx.domain.DomainFunction) -> phx.domain.DomainFunction:
    return partial_n(u, var="x", axis=1, order=1)


def pinn_solver(
    network: TemperatureNetwork,
    points: Collocation,
    interface_temperature: phx.domain.DomainFunction,
) -> phx.solver.FunctionalSolver:
    """Direct physical-residual training of the right region's network.

    ``interface_temperature`` is the interface Dirichlet value, a field of the
    right region: a constant before coupling, the fixed FE trace afterwards.
    """
    interior = RIGHT_REGION.component()
    boundary = RIGHT_REGION.component({"x": phx.domain.Boundary()})
    terms = (
        _penalty(phx.conditions.Residual("u", interior, _conduction), points.interior),
        _penalty(phx.conditions.Residual("u", boundary, _wall_heat_flux), points.wall),
        _penalty(phx.conditions.Residual("u", boundary, _insulated), points.insulated),
        _penalty(
            phx.conditions.Observation("u", boundary, interface_temperature),
            points.interface,
        ),
    )
    return phx.solver.FunctionalSolver(
        functions={"u": RIGHT_REGION.Model("x", port_mapping=POSITION_MAPPING)(network)},
        terms=terms,
    )


def trained_network(solver: phx.solver.FunctionalSolver) -> TemperatureNetwork:
    evaluator = solver.functions["u"].func
    if not isinstance(evaluator, phx.domain.ConcatenatedModelEvaluator):
        raise TypeError("The trained field is a bound domain model.")
    network = evaluator.raw_model
    if not isinstance(network, TemperatureNetwork):
        raise TypeError("The trained field wraps the surrogate network.")
    return network


def train(
    solver: phx.solver.FunctionalSolver, steps: int
) -> tuple[TemperatureNetwork, float, float]:
    """Levenberg-Marquardt on the fixed residual realization: network, loss before/after."""
    before = float(solver.loss())
    trained = solver.solve(
        num_iter=steps,
        optim=phx.optim.LevenbergMarquardt(),
        seed=0,
        jit=True,
        keep_best=False,
        log_every=0,
    )
    return trained_network(trained), before, float(trained.loss())


def constant_interface_temperature(value: float) -> phx.domain.DomainFunction:
    return phx.domain.DomainFunction(
        domain=RIGHT_REGION, deps=(), func=jnp.asarray(value, dtype=jnp.float64)
    )


# --- Accepted classical response ------------------------------------------------------

type HeldLanes = tuple[TemperatureNetwork, TemperatureNetwork]


def _bound_network(design: TemperatureNetwork, held: HeldLanes) -> TemperatureNetwork:
    network = phx.combine_parameters(design, *held)
    if not isinstance(network, TemperatureNetwork):
        raise TypeError("The design recombines into the surrogate network.")
    return network


class InterfaceFluxResidual(phx.StrictModule):
    """FE residual with the network's interface heat flux as boundary load."""

    problem: phx.equations.CompiledFiniteElementProblem

    def __call__(
        self, state: Array, design: TemperatureNetwork, held: HeldLanes
    ) -> Array:
        network = _bound_network(design, held)
        return jnp.asarray(self.problem.residual(state, {INTERFACE_FLUX: network}))


class InterfaceContinuity(phx.StrictModule):
    """``J = 1/2 sum_w (u_theta - u_FE)^2`` over the FE interface trace sites."""

    problem: phx.equations.CompiledFiniteElementProblem
    trace: phx.discretization.PreparedTraceAction
    sites: Array
    weights: Array

    def __call__(
        self, state: Array, design: TemperatureNetwork, held: HeldLanes
    ) -> Array:
        network = _bound_network(design, held)
        classical = self.trace.apply(self.problem.expand(state))
        surrogate = jax.vmap(network)(self.sites.reshape(-1, 2))
        mismatch = surrogate.reshape(self.weights.shape) - classical
        return 0.5 * jnp.sum(self.weights * mismatch**2)


def dense_policy() -> phx.linalg.LinearSolvePolicy:
    """Dense LU of the transposed state Jacobian for the explicit adjoint solve."""
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=1_000_000, max_bytes=16 * 1024 * 1024
        ),
    )


def state_solver() -> phx.optim.LeastSquaresStateSolver:
    """Gauss-Newton with a dense QR step: the affine FE state converges in one step.

    The optimality tolerance is far below the state acceptance threshold
    (``1e-10 + 1e-8 ||R(z0)||``), which certifies the realized residual on its own.
    """
    return phx.optim.LeastSquaresStateSolver(
        method=phx.optim.GaussNewton(
            linear_policy=phx.linalg.LinearSolvePolicy(
                phx.linalg.DenseQR(),
                materialization=phx.linalg.MaterializationPolicy(
                    max_entries=1_000_000, max_bytes=16 * 1024 * 1024
                ),
            )
        ),
        termination=phx.optim.OptimizationTermination(
            absolute_optimality=1e-10, relative_optimality=0.0, maximum_steps=8
        ),
    )


def classical_problem(region: ClassicalRegion) -> phx.optim.StateDesignProblem:
    return phx.optim.StateDesignProblem(
        InterfaceFluxResidual(region.problem),
        InterfaceContinuity(region.problem, region.trace, region.sites, region.weights),
        state_solver=state_solver(),
        problem_id="hybrid-interface-continuity",
    )


# The port-declaring wrapper keeps the conservative default contract of
# `AbstractArrayModel` (regularity undeclared) instead of re-declaring the MLP's
# smooth regularity, so its INPUT and MODEL_PARAMETER derivatives are admitted
# with the recorded "regularity-undeclared" condition. This is the same
# exploratory policy `FunctionalSolver` trains the surrogate under.
UNDECLARED_REGULARITY = phx.RegularityPolicy(allow_undeclared=True)


def admit(binding: phx.ComponentBinding) -> phx.optim.StateDesignComponentAdmission:
    """Admit the network's direct input and parameter derivatives as the design."""
    return phx.optim.StateDesignComponentAdmission(
        binding,
        kind=phx.ObjectiveKind.PHYSICAL_RESIDUAL,
        surfaces=(phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER),
        policy=UNDECLARED_REGULARITY,
    )


def held_lanes(binding: phx.ComponentBinding) -> HeldLanes:
    _, model_state, fixed = phx.partition_parameters(binding.model)
    return model_state, fixed


def classical_response(
    region: ClassicalRegion,
    problem: phx.optim.StateDesignProblem,
    binding: phx.ComponentBinding,
) -> tuple[phx.optim.StateDesignLinearization, phx.optim.StateDesignResponseVJP]:
    """Accepted FE solve with the network's flux and the pullback of ``J``."""
    admission = admit(binding)
    linearization = phx.optim.prepare_state_design_linearization(
        problem,
        admission.design(binding),
        region.problem.state_space.zeros(),
        args=held_lanes(binding),
        linear_policy=dense_policy(),
        component=admission,
    )
    return linearization, phx.optim.state_design_response_vjp(linearization)


# --- Dirichlet-Neumann coupling ---------------------------------------------------------


def classical_field(
    region: ClassicalRegion, state: Array
) -> phx.discretization.DiscreteFieldFunctionView:
    """The accepted FE solution as a fixed, typed numerical field of the left region."""
    reconstruction = phx.discretization.fem.prepare_finite_element_field_reconstruction(
        region.space, "u", value_port=TEMPERATURE
    )
    domain = phx.domain.GeometryDomain(reconstruction.support_geometry, label="x")
    return phx.discretization.DiscreteFieldFunctionView(
        reconstruction,
        region.problem.expand(state),
        domain,
        variable="x",
        field_name="u",
    )


def classical_interface_temperature(
    view: phx.discretization.DiscreteFieldFunctionView, sites: np.ndarray
) -> phx.domain.DomainFunction:
    """The FE interface trace as the network's interface Dirichlet value.

    The interface belongs to both closed regions. The view's reconstruction is
    read only at the interface sites, where its query evidence must be valid; it
    enters the right region's interface condition as fixed observation data.
    """
    if not bool(jnp.all(view.query(jnp.asarray(sites)).valid)):
        raise ValueError("The FE field is not defined at every interface site.")
    return phx.domain.DomainFunction(
        domain=RIGHT_REGION, deps=("x",), func=view.as_domain_function().func
    )


def surrogate_error(network: TemperatureNetwork, count: int = 21) -> float:
    axis = np.linspace(0.0, 1.0, count)
    grid_x, grid_y = np.meshgrid(INTERFACE_X + axis, axis, indexing="ij")
    points = np.stack((grid_x.ravel(), grid_y.ravel()), axis=-1)
    values = np.asarray(jax.vmap(network)(jnp.asarray(points)))
    return float(np.max(np.abs(values - exact_temperature(points))))


def classical_error(region: ClassicalRegion, state: Array) -> float:
    nodal = np.asarray(region.problem.expand(state))
    return float(np.max(np.abs(nodal - exact_temperature(region.nodes))))


def _evidence(response: phx.optim.StateDesignResponseVJP) -> str:
    primal = response.state_acceptance
    adjoint = response.adjoint_acceptance
    if adjoint is None:
        raise ValueError("A state-dependent response reports adjoint evidence.")
    return (
        f"primal accepted={bool(primal.accepted)} "
        f"(residual {float(primal.residual_norm):.2e} <= {float(primal.threshold):.2e}), "
        f"adjoint accepted={bool(adjoint.accepted)} "
        f"(transpose defect {float(adjoint.transpose_defect_norm):.2e} "
        f"<= {float(adjoint.threshold):.2e})"
    )


def _never_called(*_: object) -> NoReturn:
    raise AssertionError("a refused objective never binds or measures")


def refusals(binding: phx.ComponentBinding) -> None:
    try:
        phx.optim.StateDesignComponentAdmission(
            binding, kind=phx.ObjectiveKind.SOLUTION_MAP, policy=UNDECLARED_REGULARITY
        )
    except ValueError as error:
        print(f"   solution-map admission refused: {error}")
    else:
        raise AssertionError("a surrogate never supplies a solution-map design")
    objective = phx.solver.SolverObjective(
        None, _never_called, _never_called, objective_id="hybrid-solution-map"
    )
    try:
        objective.evaluate(binding)
    except ValueError as error:
        print(f"   implicit SolverObjective refused: {error}")
    else:
        raise AssertionError("a surrogate has no implicit solution-map signal")


def main() -> None:
    region = classical_region()
    problem = classical_problem(region)
    points = collocation(region)
    print(
        f"FE owner: {region.problem.state_space.size} unknowns; "
        f"{points.interface.shape[0]} interface trace sites"
    )

    print("1. Direct PINN pretraining (interface value u = 0):")
    network, before, after = train(
        pinn_solver(
            surrogate_network(jr.key(0)), points, constant_interface_temperature(0.0)
        ),
        PRETRAINING_STEPS,
    )
    print(f"   residual loss {before:.3e} -> {after:.3e}")

    print("2. Accepted classical response with the PINN interface heat flux:")
    binding = surrogate_binding(network)
    linearization, response = classical_response(region, problem, binding)
    if not bool(response.accepted):
        raise RuntimeError("The classical response was not accepted.")
    print(f"   {_evidence(response)}")
    cotangent = jnp.sqrt(
        sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(response.design_cotangent))
    )
    print(
        f"   J = {float(response.values):.3e}, |dJ/dtheta| = {float(cotangent):.3e}; "
        f"FE max nodal error {classical_error(region, linearization.state):.2e}"
    )

    print("3. Dirichlet-Neumann coupling through the fixed FE interface trace:")
    view = classical_field(region, linearization.state)
    coupled, before, after = train(
        pinn_solver(
            network, points, classical_interface_temperature(view, points.interface)
        ),
        COUPLING_STEPS,
    )
    print(f"   residual loss {before:.3e} -> {after:.3e}")
    coupled_binding = surrogate_binding(coupled)
    coupled_linearization, coupled_response = classical_response(
        region, problem, coupled_binding
    )
    if not bool(coupled_response.accepted):
        raise RuntimeError("The coupled classical response was not accepted.")
    print(f"   {_evidence(coupled_response)}")
    print(
        f"   J {float(response.values):.3e} -> {float(coupled_response.values):.3e}; "
        f"max error vs exact: PINN {surrogate_error(coupled):.2e}, "
        f"FE {classical_error(region, coupled_linearization.state):.2e}"
    )

    print("4. Refusals:")
    refusals(coupled_binding)


if __name__ == "__main__":
    main()
