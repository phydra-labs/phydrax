#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Learned physical and accelerator components of a coupled FE-VEM plate.

Physics. Steady heat conduction ``-div(kappa grad u) = s`` (``s = 2``) on the
plate ``[0, 2] x [0, 1]``: P1 triangles on the left half, degree-1 virtual
elements on distorted running-bond bricks on the right half, coupled across the
nonmatching interface ``x = 1`` by a mortar ``ScalarTransmissionLaw``. The left
wall is held at ``u = 0``, the right wall receives the heat flux
``g = kappa du/dn``, and the top and bottom are insulated.

Two learned roles, two permitted objectives:

1. A learned conductivity field ``kappa(x, y) = exp(c0 + c1 (x - 1/2) +
   c2 (y - 1/2))`` of the finite-element region next to the interface changes
   the accepted equations. It is a ``MODEL``-authority ``ComponentBinding``
   bound as the value of the ``conductivity-left`` ``ParameterBinding`` and
   trains through the accepted coupled solve with a ``SolverObjective``
   (implicit solution-map derivatives reusing the dense LU factors) and L-BFGS
   in ``train_components``. The data are point temperatures of an independent
   host analytic plate whose left conductivity is ``1.3 exp(0.6 (x - 1/2))``
   (for an x-only conductivity the heat flux ``kappa u' = 2 s + g - s x`` is
   fixed, so ``u`` integrates in closed form).
2. A learned right preconditioner ``(I + W0 + W1 / kappa_left + W2 /
   kappa_right)`` of the same linear solve is an accelerator
   (``AbstractPreconditioner`` slot, ``ACCELERATOR`` authority). It trains only
   through a fixed-work FGMRES ``AlgorithmicWorkObjective`` over sampled
   conductivities and heat fluxes, never through the solution map. At held-out
   parameters the production FGMRES needs far fewer iterations and still
   returns the dense LU accepted state; a singular learned action cannot make
   the solve report success.

An untrusted initial-guess provider (``LearnedInitialGuess``) is reported
beside the corrected state: the coupled solve uses its proposal only when the
original residual strictly improves, and certifies the solved state alone.
Every authority that would change the accepted equation, or train through an
objective it does not admit, is refused.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain


jax.config.update("jax_enable_x64", True)

cpl = phx.solver.coupling
M = phx.measurement

INTERFACE_X = 1.0
SOURCE = 2.0
KAPPA_RIGHT = 0.7
TRUE_FIELD = np.asarray([np.log(1.3), 0.6, 0.0])
INITIAL_FIELD = np.asarray([np.log(0.8), 0.0, 0.0])
FLUXES = np.asarray([0.0, 0.5, 1.0])
SENSORS = np.asarray(
    [
        [0.1, 0.3],
        [0.25, 0.8],
        [0.4, 0.45],
        [0.55, 0.15],
        [0.7, 0.65],
        [0.85, 0.35],
        [0.95, 0.9],
        [1.5, 0.5],
    ]
)
TEMPERATURE_SCALE = 0.01  # Explicit least-squares reference weighting (K).
WORK = 8
ACCELERATOR_STEPS = 400
PRODUCTION_RESTART = 30

WATT = phx.units.derived_unit("W", ((phx.units.JOULE, 1), (phx.units.SECOND, -1)))
CONDUCTIVITY_UNIT = phx.units.derived_unit(
    "W/(m K)", ((WATT, 1), (phx.units.METER, -1), (phx.units.KELVIN, -1))
)
HEAT_FLUX_UNIT = phx.units.derived_unit("W/m^2", ((WATT, 1), (phx.units.METER, -2)))
CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = M.QuantitySpec(
    "example", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
POSITION = phx.ValuePort(
    "plate-position",
    event_shape=(2,),
    component_ids=("x", "y"),
    representation="cartesian",
    dimensions=(phx.units.METER.dimension,) * 2,
)


def scalar_port(port_id: str, name: str, unit: phx.units.UnitDefinition) -> phx.ValuePort:
    return phx.ValuePort(
        port_id,
        event_shape=(),
        component_ids=(name,),
        representation="scalar",
        dimensions=(unit.dimension,),
    )


LEFT_CONDUCTIVITY = scalar_port("thermal-conductivity-left", "kappa", CONDUCTIVITY_UNIT)
RIGHT_CONDUCTIVITY = scalar_port("thermal-conductivity-right", "kappa", CONDUCTIVITY_UNIT)
HEAT_FLUX = scalar_port("right-wall-heat-flux", "g", HEAT_FLUX_UNIT)


# --- Host analytic plate ----------------------------------------------------------------


def graded_temperature(
    points: np.ndarray, coefficients: np.ndarray, flux: float
) -> np.ndarray:
    """``u(x)`` for ``kappa = exp(a + b x)`` left of the interface, ``KAPPA_RIGHT`` right."""
    x = np.asarray(points, dtype=np.float64)[..., 0]
    slope = float(coefficients[1])
    offset = float(coefficients[0]) - 0.5 * slope
    total = 2.0 * SOURCE + flux

    def left(t: np.ndarray) -> np.ndarray:
        decay = np.exp(-slope * t)
        first = (1.0 - decay) / slope
        second = (1.0 - decay * (1.0 + slope * t)) / slope**2
        return np.exp(-offset) * (total * first - SOURCE * second)

    right = (
        left(np.asarray(1.0))
        + (-0.5 * SOURCE * (x - 1.0) ** 2 + (SOURCE + flux) * (x - 1.0)) / KAPPA_RIGHT
    )
    return np.where(x <= INTERFACE_X, left(np.minimum(x, INTERFACE_X)), right)


def field_error(coefficients: np.ndarray) -> float:
    """Relative RMS error of a log-linear field against the analytic conductivity."""
    axis = np.linspace(0.0, 1.0, 21)
    grid = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)

    def values(c: np.ndarray) -> np.ndarray:
        return np.exp(c[0] + c[1] * (grid[:, 0] - 0.5) + c[2] * (grid[:, 1] - 0.5))

    truth = values(TRUE_FIELD)
    return float(
        np.sqrt(np.mean((values(coefficients) - truth) ** 2) / np.mean(truth**2))
    )


# --- Coupled plate ------------------------------------------------------------------


def _facets(
    space: phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization,
    x: float,
) -> IntegrationDomain:
    """The owner's exterior facets lying on the vertical line ``x``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    selected = np.all(np.isclose(np.asarray(probe.sites)[..., 0], x), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[selected]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


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


def _brick_polygons(bricks: int) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Distorted running-bond hexagons and quadrilaterals on ``[1, 2] x [0, 1]``."""
    columns = 2 * bricks + 1
    grid_x, grid_y = np.meshgrid(
        np.linspace(0.0, 1.0, columns), np.linspace(0.0, 1.0, bricks + 1), indexing="xy"
    )
    line = np.arange(bricks + 1)[:, None]
    midpoint = (np.arange(columns) % 2 == 1)[None, :]
    interior = (line > 0) & (line < bricks)
    grid_y = grid_y + np.where(midpoint & interior, 0.15 * (-1.0) ** line / bricks, 0.0)
    mapped_y = grid_y + 0.06 * np.sin(2.0 * np.pi * grid_y) * (1.0 - 0.5 * grid_x)
    mapped_x = grid_x + 0.05 * np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
    points = np.stack((INTERFACE_X + mapped_x, mapped_y), axis=-1).reshape(-1, 2)
    cells: list[np.ndarray] = []
    for row in range(bricks):
        bounds = (
            [0, *range(1, columns - 1, 2), columns - 1]
            if row % 2
            else list(range(0, columns, 2))
        )
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            lower = [row * columns + column for column in range(start, stop + 1)]
            upper = [
                (row + 1) * columns + column for column in range(stop, start - 1, -1)
            ]
            cells.append(np.asarray(lower + upper, dtype=np.int32))
    return points, tuple(cells)


def _runtime(arguments: object, name: str) -> Any:
    if not isinstance(arguments, Mapping):
        raise TypeError("Owner user arguments must be a mapping.")
    return arguments[name]


def uniform_left_conductivity(points: Array, context: object) -> Array:
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("FE coefficients receive the execution context.")
    value = jnp.asarray(_runtime(context.user_args, "conductivity"))
    return value * jnp.ones(points.shape[:-1])


def learned_left_conductivity(points: Array, context: object) -> Array:
    """The bound learned model evaluated at the FE quadrature points."""
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("FE coefficients receive the execution context.")
    model = _runtime(context.user_args, "conductivity")
    if not isinstance(model, phx.AbstractArrayModel):
        raise TypeError("The conductivity parameter is bound to a learned model.")
    return model(points)


def _right_conductivity(points: Array, args: object) -> Array:
    return jnp.asarray(_runtime(args, "conductivity")) * jnp.ones(points.shape[:-1])


def _boundary_flux(points: Array, args: object) -> Array:
    return jnp.asarray(_runtime(args, "heat-flux")) * jnp.ones(points.shape[:-1])


def _source(points: Array, args: object) -> Array:
    del args
    return SOURCE * jnp.ones(points.shape[:-1])


@dataclass(frozen=True)
class Plate:
    triangles: cpl.VariationalComponent
    polygons: cpl.VariationalComponent
    law: cpl.ScalarTransmissionLaw
    binding: cpl.InterfaceBinding
    cover: phx.domain.SubdomainCover


def _coefficient(function: Callable[[Array, Any], Array], name: str) -> Any:
    return phx.equations.coefficient(function, coefficient_id=name)


def _interface(
    triangles: cpl.VariationalComponent, polygons: cpl.VariationalComponent
) -> tuple[cpl.InterfaceBinding, phx.domain.SubdomainCover]:
    rectangle = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(rectangle, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(1))
    binding = cpl.InterfaceBinding(
        "cut",
        cpl.InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        tuple(
            cpl.InterfaceEndpoint(
                component.name,
                cpl.PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
                fields={"value": component.field_space_id("u")},
            )
            for component, patch in (
                (triangles, pairing.left_patch_id),
                (polygons, pairing.right_patch_id),
            )
        ),
    )
    return binding, cover


def build_plate(left_conductivity: Callable[[Array, Any], Array]) -> Plate:
    fe_space = phx.discretization.FiniteElementPlan(
        _triangle_mesh(4),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    nodes = np.asarray(fe_space.dof_maps[0].dof_coordinates)
    fe = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "left-conduction",
            "u",
            (
                phx.equations.DiffusionAction(
                    "u", _coefficient(left_conductivity, "conductivity-left")
                ),
                phx.equations.SourceAction("u", _coefficient(_source, "source")),
            ),
        ),
        fe_space,
        constraint=phx.discretization.dirichlet_constraint(
            fe_space, "u", boundary_mask=np.isclose(nodes[:, 0], 0.0)
        ),
        dirichlet_values=0.0,
    )
    points, cells = _brick_polygons(3)
    vem_space = phx.discretization.VirtualElementPlan(
        phx.discretization.CellMesh.from_polygons(jnp.asarray(points), cells),
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(1)
        ),
    ).prepare()
    vem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm(
            "right-conduction",
            "u",
            (
                phx.equations.DiffusionAction(
                    "u", _coefficient(_right_conductivity, "conductivity-right")
                ),
                phx.equations.SourceAction("u", _coefficient(_source, "source")),
                phx.equations.BoundaryLoadAction(
                    "u",
                    _coefficient(_boundary_flux, "heat-flux"),
                    action_id="right-wall-heat-flux",
                    domain=_facets(vem_space, 2.0),
                ),
            ),
        ),
        vem_space,
    )
    triangles = cpl.VariationalComponent("triangles", fe, field="u")
    polygons = cpl.VariationalComponent("polygons", vem, field="u")
    binding, cover = _interface(triangles, polygons)
    law = cpl.ScalarTransmissionLaw(
        "transmission",
        binding,
        (
            cpl.TransmissionSide("triangles", "triangles", "u", _facets(fe_space, 1.0)),
            cpl.TransmissionSide("polygons", "polygons", "u", _facets(vem_space, 1.0)),
        ),
        cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="polygons")),
    )
    return Plate(triangles, polygons, law, binding, cover)


def parameter_bindings() -> tuple[cpl.ParameterBinding, ...]:
    """Conductivities are physical parameters; the wall heat flux a solver argument."""
    return (
        cpl.ParameterBinding(
            "conductivity-left",
            LEFT_CONDUCTIVITY,
            targets=(cpl.RuntimeInput("triangles", "conductivity"),),
            role="coefficient",
            derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
        ),
        cpl.ParameterBinding(
            "conductivity-right",
            RIGHT_CONDUCTIVITY,
            targets=(cpl.RuntimeInput("polygons", "conductivity"),),
            role="coefficient",
            derivative=phx.DerivativeSurface.PHYSICAL_PARAMETER,
        ),
        cpl.ParameterBinding(
            "heat-flux",
            HEAT_FLUX,
            targets=(cpl.RuntimeInput("polygons", "heat-flux"),),
            role="boundary",
            derivative=phx.DerivativeSurface.SOLVER_ARGUMENT,
        ),
    )


def dense_policy() -> phx.linalg.LinearSolvePolicy:
    """Dense LU of the coupled saddle system; derivative solves reuse its factors."""
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(route="primal-factors"),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=16_000_000, max_bytes=256 * 1024 * 1024
        ),
    )


def prepare(
    plate: Plate,
    left: object,
    observations: tuple[cpl.FieldPointObservation, ...] = (),
) -> cpl.PreparedCoupledProblem:
    plan = cpl.CoupledProblemPlan(
        "learned-plate",
        components=(plate.triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(plate.law,),
        parameters=parameter_bindings(),
        observations=observations,
    )
    return cpl.prepare_coupled_problem(
        plan,
        interface_owners=(plate.cover,),
        parameters={
            "conductivity-left": left,
            "conductivity-right": jnp.asarray(KAPPA_RIGHT),
            "heat-flux": jnp.asarray(0.5),
        },
    )


def smooth_contract() -> phx.ModelExecutionContract:
    return phx.ModelExecutionContract(
        derivative=phx.DerivativeContract.smooth(
            (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
        ),
        execution=phx.ExecutionCapabilities("native-jax"),
        precision=phx.ComponentPrecisionContract.native("float64"),
        randomness=phx.RandomnessContract("deterministic"),
    )


# --- 1. Learned conductivity (MODEL authority, solution-map objective) -------------------


class ConductivityField(phx.AbstractArrayModel):
    """Positive log-linear conductivity field of the plate position."""

    coefficients: Array
    in_size: int = eqx.field(static=True)
    out_size: Literal["scalar"] = eqx.field(static=True)

    def __init__(self, coefficients: Any) -> None:
        self.coefficients = jnp.asarray(coefficients, jnp.float64)
        self.in_size = 2
        self.out_size = "scalar"

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        c = self.coefficients
        return jnp.exp(c[0] + c[1] * (x[..., 0] - 0.5) + c[2] * (x[..., 1] - 0.5))

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return smooth_contract()


def conductivity(
    coefficients: Any, authority: phx.ComponentAuthority = phx.ComponentAuthority.MODEL
) -> phx.ComponentBinding:
    return phx.bind_component(
        ConductivityField(coefficients),
        authority,
        owner_ports=phx.ModelPorts(inputs=(POSITION,), outputs=(LEFT_CONDUCTIVITY,)),
    )


def sensor_observations() -> tuple[cpl.FieldPointObservation, ...]:
    points = np.concatenate([SENSORS, np.zeros((len(SENSORS), 1))], axis=1)
    point = M.SamplingSemantics(M.SpatialSamplingKind.POINT)
    return (
        cpl.FieldPointObservation(
            "left",
            "triangles",
            "u",
            quantity=TEMPERATURE,
            support=M.PointSampleSupport(points[:-1], tuple("abcdefg"), CONTRACT),
            sampling=point,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldPointObservation(
            "right",
            "polygons",
            "u",
            quantity=TEMPERATURE,
            support=M.PointSampleSupport(points[-1:], ("h",), CONTRACT),
            sampling=point,
            field_unit=phx.units.KELVIN,
        ),
    )


class FieldSolve(phx.StrictModule):
    prepared: cpl.PreparedCoupledProblem
    kappa_right: Array


class BoundField(phx.StrictModule):
    solve: FieldSolve
    conductivity: phx.ComponentBinding


def field_measure(
    owner: BoundField, case: tuple[Array, Array]
) -> phx.solver.SolverCaseResult:
    flux, observed = case
    solution = cpl.solve_coupled_problem(
        owner.solve.prepared,
        parameters={
            "conductivity-left": owner.conductivity,
            "conductivity-right": owner.solve.kappa_right,
            "heat-flux": flux,
        },
        policy=dense_policy(),
    )
    predicted = jnp.concatenate(
        [solution.observation(name).values for name in ("left", "right")]
    )
    return phx.solver.SolverCaseResult(
        residual=(predicted - observed) / TEMPERATURE_SCALE, accepted=solution.accepted
    )


def learned_conductivity() -> tuple[
    cpl.PreparedCoupledProblem, phx.solver.SolverObjective
]:
    prepared = prepare(
        build_plate(learned_left_conductivity),
        conductivity(INITIAL_FIELD),
        sensor_observations(),
    )
    capability = prepared.derivative_capability(dense_policy())
    print(f"  admitted derivatives: {[name for name, _ in capability.admitted]}")
    observed = np.stack([graded_temperature(SENSORS, TRUE_FIELD, g) for g in FLUXES])
    objective = phx.solver.SolverObjective(
        FieldSolve(prepared, jnp.asarray(KAPPA_RIGHT, jnp.float64)),
        BoundField,
        field_measure,
        objective_id="learned-conductivity",
        cases=(jnp.asarray(FLUXES), jnp.asarray(observed)),
    )
    result = phx.solver.train_components(
        conductivity(INITIAL_FIELD),
        (objective,),
        optimizer=optax.lbfgs(),
        steps=15,
        key=jr.key(0),
    )
    trained = np.asarray(result.tree.model.coefficients)
    evaluation = objective.evaluate(result.tree)
    at_truth = objective.evaluate(conductivity(TRUE_FIELD))
    print(f"  trained groups: {result.authorities}")
    print(
        f"  L-BFGS: {result.accepted_updates} of {result.attempts} attempts accepted; "
        f"objective {float(result.values[0]):.4e} -> {float(result.values[-1]):.4e} "
        f"(true field on this mesh: {float(at_truth.value):.4e})"
    )
    print(f"  coefficients {np.round(trained, 4)} (analytic {np.round(TRUE_FIELD, 4)})")
    print(
        f"  conductivity RMS error vs analytic: {field_error(INITIAL_FIELD):.3f} -> "
        f"{field_error(trained):.3f}; every case accepted: "
        f"{bool(jnp.all(evaluation.accepted))}"
    )
    if not (
        bool(jnp.all(evaluation.accepted))
        and field_error(trained) < 0.1 * field_error(INITIAL_FIELD)
        and float(evaluation.value) < float(at_truth.value)
    ):
        raise RuntimeError(
            "The learned conductivity did not approach the analytic field."
        )
    return prepared, objective


# --- 2. Learned preconditioner (ACCELERATOR authority, fixed-work objective) -------------


class InverseAction(phx.AbstractArrayModel):
    """Learned approximate-inverse action ``(I + sum_j phi_j W_j) r``."""

    weights: Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, size: int) -> None:
        self.weights = jnp.zeros((3, size, size), jnp.float64)
        self.in_size = 3 + size
        self.out_size = size

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del key
        features, residual = x[:3], x[3:]
        correction = jnp.sum(features[:, None, None] * self.weights, axis=0)
        return residual + correction @ residual

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return smooth_contract()


class LearnedInverse(phx.linalg.AbstractPreconditioner):
    """Right preconditioner: the bound learned action at ``(1, 1/kappa_l, 1/kappa_r)``."""

    action: phx.ComponentBinding
    features: Array = phx.fixed_field()

    def __init__(
        self, action: phx.ComponentBinding, theta: Array, space: phx.linalg.BlockSpace
    ) -> None:
        self.action = action
        self.features = jnp.stack(
            (jnp.ones_like(theta[0]), 1.0 / theta[0], 1.0 / theta[1])
        )
        self.space = space
        self.properties = phx.linalg.PreconditionerProperties(
            linear=True,
            stationary=True,
            evidence={"linear": "construction", "stationary": "construction"},
        )
        self.preconditioner_id = "learned-approximate-inverse"

    def apply(self, residual: Any, /, *, iteration: Any = None) -> Any:
        del iteration
        flat = self.space.flatten(residual)
        return self.space.unflatten(
            self.action.model(jnp.concatenate((self.features, flat)))
        )


def parameters(theta: Array) -> dict[str, Array]:
    return {
        "conductivity-left": theta[0],
        "conductivity-right": theta[1],
        "heat-flux": theta[2],
    }


def krylov_policy(
    preconditioner: phx.linalg.AbstractPreconditioner | None,
    *,
    restart: int,
    relative: float,
    max_steps: int,
    mode: phx.linalg.DifferentiationMode,
) -> phx.linalg.LinearSolvePolicy:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=restart),
        preconditioning=None
        if preconditioner is None
        else phx.linalg.PreconditioningPolicy(preconditioner, side="right"),
        tolerance=phx.linalg.TolerancePolicy(
            relative=relative, absolute=0.0, max_steps=max_steps
        ),
        differentiation=phx.linalg.DifferentiationPolicy(mode),
        failure=phx.linalg.FailurePolicy("status"),
    )


class KrylovSolve(phx.StrictModule):
    prepared: cpl.PreparedCoupledProblem


class BoundKrylov(phx.StrictModule):
    solve: KrylovSolve
    action: phx.ComponentBinding


def fixed_work(owner: BoundKrylov, theta: Array) -> phx.solver.AlgorithmicWorkResult:
    """Exactly ``WORK`` preconditioned FGMRES steps from the native zero state."""
    prepared = owner.solve.prepared
    solution = cpl.solve_coupled_problem(
        prepared,
        parameters=parameters(theta),
        policy=krylov_policy(
            LearnedInverse(owner.action, theta, prepared.state_space),
            restart=WORK,
            relative=0.0,
            max_steps=WORK,
            mode="algorithmic",
        ),
    )
    linear = solution.linear
    if linear is None:
        raise RuntimeError("An affine coupled solve reports its linear result.")
    arguments = prepared.bind_arguments(parameters=parameters(theta)).arguments
    return phx.solver.AlgorithmicWorkResult(
        initial_residual=prepared.residual(prepared.state_space.zeros(), arguments),
        final_residual=prepared.residual(solution.state, arguments),
        iterations=linear.diagnostics.iterations,
        accepted=jnp.all(jnp.isfinite(prepared.state_space.flatten(solution.state))),
    )


def never_measured(owner: Any, case: Any) -> phx.solver.SolverCaseResult:
    raise RuntimeError("A refused objective never binds or measures.")


def iterations(solution: cpl.CoupledSolution) -> int:
    linear = solution.linear
    if linear is None:
        raise RuntimeError("An affine coupled solve reports its linear result.")
    return int(linear.diagnostics.iterations)


def flat(
    prepared: cpl.PreparedCoupledProblem, solution: cpl.CoupledSolution
) -> np.ndarray:
    return np.asarray(prepared.state_space.flatten(solution.state))


def learned_preconditioner(prepared: cpl.PreparedCoupledProblem) -> None:
    rng = np.random.default_rng(0)
    cases = np.stack(
        (rng.uniform(0.5, 2.0, 8), rng.uniform(0.5, 2.0, 8), rng.uniform(0.0, 1.0, 8)),
        axis=1,
    )
    objective = phx.solver.AlgorithmicWorkObjective(
        KrylovSolve(prepared),
        BoundKrylov,
        fixed_work,
        work=WORK,
        objective_id="learned-inverse-fixed-work",
        cases=jnp.asarray(cases),
    )
    initial = phx.bind_component(InverseAction(prepared.state_space.size), LearnedInverse)
    try:
        phx.solver.SolverObjective(
            KrylovSolve(prepared),
            BoundKrylov,
            never_measured,
            objective_id="solution-map",
        ).evaluate(initial)
    except ValueError as error:
        print(f"  refused: {error}")
    else:
        raise RuntimeError("An accelerator has no solution-map training signal.")
    result = phx.solver.train_components(
        initial,
        (objective,),
        optimizer=optax.adam(5e-3),
        steps=ACCELERATOR_STEPS,
        key=jr.key(0),
    )
    before = float(objective.evaluate(initial).value)
    after = float(objective.evaluate(result.tree).value)
    print(f"  trained groups: {result.authorities}")
    print(
        f"  fixed work ({WORK} FGMRES steps): mean log(|r_k| / |r_0|) "
        f"{before:.3f} -> {after:.3f} over {ACCELERATOR_STEPS} Adam steps"
    )
    for theta in ((1.3, 0.7, 0.5), (0.6, 1.8, 0.9)):
        values = jnp.asarray(theta)
        reference = flat(
            prepared,
            cpl.solve_coupled_problem(
                prepared, parameters=parameters(values), policy=dense_policy()
            ),
        )
        solves = {
            name: cpl.solve_coupled_problem(
                prepared,
                parameters=parameters(values),
                policy=krylov_policy(
                    preconditioner,
                    restart=PRODUCTION_RESTART,
                    relative=1e-12,
                    max_steps=4000,
                    mode="none",
                ),
            )
            for name, preconditioner in (
                ("plain", None),
                ("learned", LearnedInverse(result.tree, values, prepared.state_space)),
            )
        }
        differences = {
            name: float(np.max(np.abs(flat(prepared, solution) - reference)))
            for name, solution in solves.items()
        }
        print(
            f"  held-out theta {theta}: FGMRES({PRODUCTION_RESTART}) iterations "
            f"{iterations(solves['plain'])} plain, {iterations(solves['learned'])} "
            f"learned; accepted {[bool(item.accepted) for item in solves.values()]}; "
            f"max |state - dense LU| {max(differences.values()):.1e}"
        )
        if not (
            all(bool(item.accepted) for item in solves.values())
            and max(differences.values()) < 1e-9 * np.abs(reference).max()
            and 3 * iterations(solves["learned"]) < iterations(solves["plain"])
        ):
            raise RuntimeError("The learned preconditioner changed or failed the solve.")
    size = prepared.state_space.size
    singular = eqx.tree_at(
        lambda binding: binding.model.weights,
        initial,
        jnp.zeros((3, size, size)).at[0].set(-jnp.eye(size)),
    )
    values = jnp.asarray((1.3, 0.7, 0.5))
    broken = cpl.solve_coupled_problem(
        prepared,
        parameters=parameters(values),
        policy=krylov_policy(
            LearnedInverse(singular, values, prepared.state_space),
            restart=PRODUCTION_RESTART,
            relative=1e-12,
            max_steps=200,
            mode="none",
        ),
    )
    print(
        f"  singular learned action: native success {bool(broken.native_successful)}, "
        f"accepted {bool(broken.accepted)}"
    )
    if bool(broken.accepted):
        raise RuntimeError("A singular preconditioner produced an accepted solution.")


# --- 3. Untrusted initial-guess proposals -----------------------------------------------


class StoredState(phx.AbstractArrayModel):
    """An accelerator's stored proposal: one flat solve-coordinate state."""

    state: Array
    in_size: Literal["scalar"] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, state: Any) -> None:
        self.state = jnp.asarray(state, jnp.float64)
        self.in_size = "scalar"
        self.out_size = self.state.shape[0]

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        return self.state

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return smooth_contract()


class StoredProposal(phx.StrictModule):
    guess: phx.ComponentBinding
    space: phx.linalg.BlockSpace

    def __call__(self, data: Any, baseline: Any) -> Any:
        del data, baseline
        return self.space.unflatten(self.guess.model(jnp.zeros((), jnp.float64)))


def proposals(prepared: cpl.PreparedCoupledProblem) -> None:
    space = prepared.state_space
    values = parameters(jnp.asarray((1.3, 0.7, 0.5)))
    reference = flat(
        prepared,
        cpl.solve_coupled_problem(prepared, parameters=values, policy=dense_policy()),
    )
    neighbor = flat(
        prepared,
        cpl.solve_coupled_problem(
            prepared,
            parameters=parameters(jnp.asarray((1.25, 0.75, 0.5))),
            policy=dense_policy(),
        ),
    )
    policy = krylov_policy(
        None, restart=PRODUCTION_RESTART, relative=1e-12, max_steps=4000, mode="none"
    )
    native = cpl.solve_coupled_problem(prepared, parameters=values, policy=policy)
    for label, stored in (
        ("neighbor accepted state", neighbor),
        ("constant 1e3 state", np.full((space.size,), 1.0e3)),
    ):
        guess = phx.bind_component(StoredState(stored), phx.linalg.LearnedInitialGuess)
        provider = phx.linalg.LearnedInitialGuess(
            StoredProposal(guess, space), provider_id="stored-state"
        )
        guided = cpl.solve_coupled_problem(
            prepared, parameters=values, policy=policy, initial_state=provider
        )
        proposal = cpl.certify_coupled_state(
            prepared,
            space.unflatten(jnp.asarray(stored)),
            prepared.bind_arguments(parameters=values),
            policy=policy,
            linear=None,
            nonlinear=None,
            native=jnp.asarray(True),
            derivative=jnp.asarray(False),
            tolerance=guided.tolerance,
        )
        evidence = guided.initial_guess
        if evidence is None:
            raise RuntimeError("A provider solve reports its selection evidence.")
        worst = max(
            float(item.residual_norm / item.scale) for item in proposal.components
        )
        difference = float(np.max(np.abs(flat(prepared, guided) - reference)))
        print(
            f"  {label}: proposal used {bool(evidence.accepted)} (residual "
            f"{float(evidence.proposal_residual_norm):.2e} vs zero state "
            f"{float(evidence.baseline_residual_norm):.2e}); proposal accepted as a "
            f"solution {bool(proposal.accepted)} (worst relative residual {worst:.1e}); "
            f"corrected state accepted {bool(guided.accepted)}, "
            f"{iterations(guided)} vs {iterations(native)} iterations, "
            f"max |state - dense LU| {difference:.1e}"
        )
        if bool(proposal.accepted) or not bool(guided.accepted):
            raise RuntimeError("A proposal replaced the certified solution.")


# --- 4. Refusals --------------------------------------------------------------------------


def refusals(
    prepared: cpl.PreparedCoupledProblem, objective: phx.solver.SolverObjective
) -> None:
    for authority in (
        phx.ComponentAuthority.ACCELERATOR,
        phx.ComponentAuthority.SURROGATE,
    ):
        try:
            prepared.bind_arguments(
                parameters={
                    "conductivity-left": conductivity(TRUE_FIELD, authority),
                    "conductivity-right": jnp.asarray(KAPPA_RIGHT),
                    "heat-flux": jnp.asarray(0.5),
                }
            )
        except ValueError as error:
            print(f"  refused: {error}")
        else:
            raise RuntimeError(f"A {authority.value} input changed the equation.")
    try:
        objective.evaluate(conductivity(INITIAL_FIELD, phx.ComponentAuthority.SURROGATE))
    except ValueError as error:
        print(f"  refused: {error}")
    else:
        raise RuntimeError("A surrogate trained through the implicit solution map.")


def main() -> None:
    print("1. Learned conductivity through the accepted coupled solve:")
    learned, objective = learned_conductivity()
    prepared = prepare(build_plate(uniform_left_conductivity), jnp.asarray(1.0))
    print(f"2. Learned preconditioner ({prepared.state_space.size} unknowns):")
    learned_preconditioner(prepared)
    print("3. Untrusted initial-guess proposals:")
    proposals(prepared)
    print("4. Refusals:")
    refusals(learned, objective)


if __name__ == "__main__":
    main()
