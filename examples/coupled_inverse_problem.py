#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Infer a conductivity and a boundary heat flux of a coupled FE-VEM plate.

Physics. Steady heat conduction ``-div(kappa grad u) = s`` (``s = 2``) on the
plate ``[0, 2] x [0, 1]``. The left half is meshed with P1 triangles, the right
half with degree-1 virtual elements on distorted running-bond bricks whose
interface vertices do not match the triangles'; a mortar
``ScalarTransmissionLaw`` couples them at ``x = 1``. The left wall is held at
``u = 0``, the right wall receives the heat flux ``g = kappa du/dn``, and the top
and bottom are insulated. The conductivity is ``kappa_left`` on the left and
``kappa_right`` on the right.

Independent data. For constant conductivities the exact temperature depends
on ``x`` only and is evaluated on the host (``exact_temperature``); the
conormal flux content ``int kappa du/dn ds`` through the left wall (outward
normal ``-e_x``), i.e. the heat entering there, is ``-(2 s + g)`` per unit
depth: all generated and supplied heat leaves through that wall. Measurements at
the true values ``kappa_right = 0.7`` and ``g = 0.5`` (``kappa_left = 1.3``
known) are drawn from this analytic plate plus seeded Gaussian noise whose
standard deviation is declared through ``IndependentStandardUncertainty``:
four temperature sensors (two per region; the VEM ones observe the labeled H1
projection), the right-wall mean temperature (``SURFACE_AVERAGE``), and the
left-wall heat inflow (``PATH_INTEGRAL`` of the residual reaction, the
conormal flux content that ``FieldFluxObservation`` reports).

Inversion. The unknowns ``(log kappa_right, g)`` live in a MODEL-authority
``ComponentBinding``; a ``SolverObjective`` binds them into the prepared
problem, solves it (dense LU, implicit derivatives reusing the factors), and
returns the whitened ``MeasurementComparisonPlan`` residuals; L-BFGS through
``train_components`` fits them. The fixed-noise posterior of the estimate is
formed with ``posterior_problem_from_solver_objective``.
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

LEVEL = 0
INTERFACE_X = 1.0
SOURCE = 2.0
KAPPA_LEFT = 1.3
TRUTH = (0.7, 0.5)  # (kappa_right, g)
INITIAL = (1.0, 1.0)
SENSORS = np.asarray([[0.3, 0.4, 0.0], [0.7, 0.6, 0.0], [1.4, 0.3, 0.0], [1.8, 0.7, 0.0]])
TEMPERATURE_STD = 0.01
HEAT_STD = 0.02
TRAINING_STEPS = 12

WATT = phx.units.derived_unit("W", ((phx.units.JOULE, 1), (phx.units.SECOND, -1)))
CONDUCTIVITY_UNIT = phx.units.derived_unit(
    "W/(m K)", ((WATT, 1), (phx.units.METER, -1), (phx.units.KELVIN, -1))
)
HEAT_FLUX_UNIT = phx.units.derived_unit("W/m^2", ((WATT, 1), (phx.units.METER, -2)))
LINE_HEAT_UNIT = phx.units.derived_unit("W/m", ((WATT, 1), (phx.units.METER, -1)))


# --- Host analytic plate ----------------------------------------------------------------


def exact_temperature(
    points: np.ndarray, kappa_left: float, kappa_right: float, flux: float
) -> np.ndarray:
    x = np.asarray(points, dtype=np.float64)[..., 0]
    left = (-0.5 * SOURCE * x**2 + (2.0 * SOURCE + flux) * x) / kappa_left
    offset = (1.5 * SOURCE + flux) / kappa_left
    right = (
        -0.5 * SOURCE * (x - 1.0) ** 2 + (SOURCE + flux) * (x - 1.0)
    ) / kappa_right + offset
    return np.where(x <= INTERFACE_X, left, right)


def exact_measurements(kappa_right: float, flux: float) -> tuple[np.ndarray, ...]:
    """Sensors (left, right), right-wall mean ``u(2)``, and left-wall heat inflow."""
    temperature = exact_temperature(SENSORS, KAPPA_LEFT, kappa_right, flux)
    wall = exact_temperature(np.asarray([[2.0, 0.5]]), KAPPA_LEFT, kappa_right, flux)
    return temperature[:2], temperature[2:], wall, np.asarray([-(2.0 * SOURCE + flux)])


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


def _runtime(arguments: object, name: str) -> Array:
    if not isinstance(arguments, Mapping):
        raise TypeError("Owner user arguments must be a mapping.")
    return jnp.asarray(arguments[name])


def _left_conductivity(points: Array, context: object) -> Array:
    if not isinstance(context, phx.equations.FiniteElementExecutionContext):
        raise TypeError("FE coefficients receive the execution context.")
    return _runtime(context.user_args, "conductivity") * jnp.ones(points.shape[:-1])


def _right_conductivity(points: Array, args: object) -> Array:
    return _runtime(args, "conductivity") * jnp.ones(points.shape[:-1])


def _boundary_flux(points: Array, args: object) -> Array:
    return _runtime(args, "heat-flux") * jnp.ones(points.shape[:-1])


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
    left_wall: IntegrationDomain
    right_wall: IntegrationDomain


def _coefficient(function: Callable[[Array, Any], Array], name: str) -> Any:
    return phx.equations.coefficient(function, coefficient_id=name)


def build_plate(level: int) -> Plate:
    fe_space = phx.discretization.FiniteElementPlan(
        _triangle_mesh(4 * 2**level),
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
                    "u", _coefficient(_left_conductivity, "conductivity-left")
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
    points, cells = _brick_polygons(3 * 2**level)
    vem_space = phx.discretization.VirtualElementPlan(
        phx.discretization.CellMesh.from_polygons(jnp.asarray(points), cells),
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(1)
        ),
    ).prepare()
    right_wall = _facets(vem_space, 2.0)
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
                    domain=right_wall,
                ),
            ),
        ),
        vem_space,
    )
    triangles = cpl.VariationalComponent("triangles", fe, field="u")
    polygons = cpl.VariationalComponent("polygons", vem, field="u")
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
    law = cpl.ScalarTransmissionLaw(
        "transmission",
        binding,
        (
            cpl.TransmissionSide("triangles", "triangles", "u", _facets(fe_space, 1.0)),
            cpl.TransmissionSide("polygons", "polygons", "u", _facets(vem_space, 1.0)),
        ),
        cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="polygons")),
    )
    return Plate(
        triangles, polygons, law, binding, cover, _facets(fe_space, 0.0), right_wall
    )


def _scalar_port(
    port_id: str, name: str, unit: phx.units.UnitDefinition
) -> phx.ValuePort:
    return phx.ValuePort(
        port_id,
        event_shape=(),
        component_ids=(name,),
        representation="scalar",
        dimensions=(unit.dimension,),
    )


def parameter_bindings() -> tuple[cpl.ParameterBinding, ...]:
    """Conductivities are physical parameters; the wall heat flux a solver argument."""
    return tuple(
        cpl.ParameterBinding(
            binding_id,
            _scalar_port(f"thermal-{binding_id}", symbol, unit),
            targets=(cpl.RuntimeInput(component, name),),
            role=role,
            derivative=surface,
        )
        for binding_id, symbol, unit, component, name, role, surface in (
            (
                "conductivity-left",
                "kappa",
                CONDUCTIVITY_UNIT,
                "triangles",
                "conductivity",
                "coefficient",
                phx.DerivativeSurface.PHYSICAL_PARAMETER,
            ),
            (
                "conductivity-right",
                "kappa",
                CONDUCTIVITY_UNIT,
                "polygons",
                "conductivity",
                "coefficient",
                phx.DerivativeSurface.PHYSICAL_PARAMETER,
            ),
            (
                "heat-flux",
                "g",
                HEAT_FLUX_UNIT,
                "polygons",
                "heat-flux",
                "boundary",
                phx.DerivativeSurface.SOLVER_ARGUMENT,
            ),
        )
    )


def dense_policy(
    mode: phx.linalg.DifferentiationMode = "mathematical",
) -> phx.linalg.LinearSolvePolicy:
    """Dense LU of the coupled saddle system; derivative solves reuse its factors."""
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy(mode),
        derivative_solve=phx.linalg.LinearDerivativeSolvePolicy(route="primal-factors"),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=16_000_000, max_bytes=256 * 1024 * 1024
        ),
    )


# --- Measurements -----------------------------------------------------------------------


CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = M.QuantitySpec(
    "example", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
WALL_HEAT = M.QuantitySpec(
    "example",
    "wall-heat-inflow",
    "heat-rate-per-depth",
    LINE_HEAT_UNIT,
    "wall-heat-inflow",
)
WALL = M.IndexSampleSupport((1,), ("wall",))
RULE = FacetTraceRule(points=2)


@dataclass(frozen=True)
class Record:
    binding_id: str
    quantity: M.QuantitySpec
    support: M.SampleSupport
    sampling: M.SamplingSemantics
    std: float


LEFT_SENSORS = M.PointSampleSupport(SENSORS[:2], ("a", "b"), CONTRACT)
RIGHT_SENSORS = M.PointSampleSupport(SENSORS[2:], ("c", "d"), CONTRACT)
RECORDS = (
    Record(
        "left-sensors",
        TEMPERATURE,
        LEFT_SENSORS,
        M.SamplingSemantics(M.SpatialSamplingKind.POINT),
        TEMPERATURE_STD,
    ),
    Record(
        "right-sensors",
        TEMPERATURE,
        RIGHT_SENSORS,
        M.SamplingSemantics(M.SpatialSamplingKind.POINT),
        TEMPERATURE_STD,
    ),
    Record(
        "right-wall-mean",
        TEMPERATURE,
        WALL,
        M.SamplingSemantics(M.SpatialSamplingKind.SURFACE_AVERAGE),
        TEMPERATURE_STD,
    ),
    Record(
        "left-wall-heat",
        WALL_HEAT,
        WALL,
        M.SamplingSemantics(M.SpatialSamplingKind.PATH_INTEGRAL),
        HEAT_STD,
    ),
)


def observations(plate: Plate) -> tuple[cpl.AbstractObservationBinding, ...]:
    left, right, wall, heat = RECORDS
    return (
        cpl.FieldPointObservation(
            left.binding_id,
            "triangles",
            "u",
            quantity=left.quantity,
            support=LEFT_SENSORS,
            sampling=left.sampling,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldPointObservation(
            right.binding_id,
            "polygons",
            "u",
            quantity=right.quantity,
            support=RIGHT_SENSORS,
            sampling=right.sampling,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldBoundaryObservation(
            wall.binding_id,
            "polygons",
            "u",
            plate.right_wall,
            statistic="average",
            rule=RULE,
            quantity=wall.quantity,
            support=wall.support,
            sampling=wall.sampling,
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldFluxObservation(
            heat.binding_id,
            "triangles",
            "u",
            plate.left_wall,
            rule=RULE,
            quantity=heat.quantity,
            support=heat.support,
            sampling=heat.sampling,
            reaction_unit=LINE_HEAT_UNIT,
        ),
    )


def noisy_data(seed: int) -> tuple[M.PreparedQuantityField, ...]:
    """Analytic measurements at the truth plus declared Gaussian noise."""
    rng = np.random.default_rng(seed)
    return tuple(
        M.QuantityField(
            f"{record.binding_id}-data",
            record.quantity,
            M.ValueLayout.scalar(),
            record.support,
            record.sampling,
            exact + record.std * rng.standard_normal(exact.shape),
            uncertainty=M.IndependentStandardUncertainty(
                np.full(exact.shape, record.std), record.quantity.unit
            ),
        ).prepare()
        for record, exact in zip(RECORDS, exact_measurements(*TRUTH), strict=True)
    )


# --- Learned-component parameterization and objective -----------------------------------


class PlateParameters(phx.AbstractArrayModel):
    """Constant map to ``(kappa_right, g)``; ``log kappa_right`` keeps it positive."""

    log_conductivity: Array
    heat_flux: Array
    in_size: Literal["scalar"] = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, conductivity: float, heat_flux: float) -> None:
        self.log_conductivity = jnp.log(jnp.asarray(conductivity, jnp.float64))
        self.heat_flux = jnp.asarray(heat_flux, jnp.float64)
        self.in_size = "scalar"
        self.out_size = 2

    def __call__(self, x: Any, /, *, key: Any = None) -> Array:
        del x, key
        return jnp.stack((jnp.exp(self.log_conductivity), self.heat_flux))

    def model_execution_contract(self) -> phx.ModelExecutionContract:
        return phx.ModelExecutionContract(
            derivative=phx.DerivativeContract.smooth(
                (phx.DerivativeSurface.INPUT, phx.DerivativeSurface.MODEL_PARAMETER)
            ),
            execution=phx.ExecutionCapabilities("native-jax"),
            precision=phx.ComponentPrecisionContract.native("float64"),
            randomness=phx.RandomnessContract("deterministic"),
        )


class PlateSolve(phx.StrictModule):
    """Fixed prepared solve: the problem, the comparisons, and the known input.

    ``observation_ids`` names the prediction each comparison scores; objective
    callables must read every array through the solve, never through globals.
    """

    prepared: cpl.PreparedCoupledProblem
    plans: tuple[phx.observation.MeasurementComparisonPlan, ...]
    kappa_left: Array
    policy: phx.linalg.LinearSolvePolicy
    observation_ids: tuple[str, ...] = eqx.field(static=True)


class BoundPlate(phx.StrictModule):
    solve: PlateSolve
    theta: Array


def declared(term: Array | None) -> Array:
    """A comparison term the declared independent uncertainty always provides."""
    if term is None:
        raise ValueError("The comparison declares no noise model.")
    return term


def whitened(
    solve: PlateSolve, theta: Array
) -> tuple[Array, Array, tuple[phx.observation.MeasurementComparisonResult, ...]]:
    """Whitened residuals, acceptance, and comparisons at ``(kappa_l, kappa_r, g)``."""
    solution = cpl.solve_coupled_problem(
        solve.prepared,
        parameters={
            "conductivity-left": theta[0],
            "conductivity-right": theta[1],
            "heat-flux": theta[2],
        },
        policy=solve.policy,
    )
    comparisons = tuple(
        plan.evaluate(solution.observation(binding_id))
        for plan, binding_id in zip(solve.plans, solve.observation_ids, strict=True)
    )
    residual = jnp.concatenate(
        [jnp.ravel(declared(item.whitened_residual)) for item in comparisons]
    )
    accepted = solution.accepted & jnp.all(
        jnp.stack([item.successful for item in comparisons])
    )
    return residual, accepted, comparisons


def bind(solve: PlateSolve, component: phx.ComponentBinding) -> BoundPlate:
    unknowns = component.model(jnp.zeros((), jnp.float64))
    return BoundPlate(solve, jnp.concatenate((solve.kappa_left[None], unknowns)))


def measure(owner: BoundPlate, case: None) -> phx.solver.SolverCaseResult:
    del case
    residual, accepted, _ = whitened(owner.solve, owner.theta)
    return phx.solver.SolverCaseResult(residual=residual, accepted=accepted)


def main() -> None:
    plate = build_plate(LEVEL)
    plan = cpl.CoupledProblemPlan(
        "plate",
        components=(plate.triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(plate.law,),
        parameters=parameter_bindings(),
        observations=observations(plate),
    )
    prepared = cpl.prepare_coupled_problem(
        plan,
        interface_owners=(plate.cover,),
        parameters={
            "conductivity-left": jnp.asarray(KAPPA_LEFT),
            "conductivity-right": jnp.asarray(TRUTH[0]),
            "heat-flux": jnp.asarray(TRUTH[1]),
        },
    )
    solve = PlateSolve(
        prepared,
        tuple(phx.observation.MeasurementComparisonPlan(data) for data in noisy_data(7)),
        jnp.asarray(KAPPA_LEFT, jnp.float64),
        dense_policy(),
        tuple(record.binding_id for record in RECORDS),
    )

    print("Forward check at the truth (prediction vs host analytic):")
    truth_theta = jnp.asarray((KAPPA_LEFT, *TRUTH))
    solution = jax.jit(
        lambda theta: cpl.solve_coupled_problem(
            prepared,
            parameters={
                "conductivity-left": theta[0],
                "conductivity-right": theta[1],
                "heat-flux": theta[2],
            },
            policy=solve.policy,
        )
    )(truth_theta)
    if not bool(solution.accepted):
        raise RuntimeError("The coupled solve at the truth was not accepted.")
    for record, exact in zip(RECORDS, exact_measurements(*TRUTH), strict=True):
        predicted = np.asarray(solution.observation(record.binding_id).values)
        label = prepared.observation(record.binding_id).approximation
        print(
            f"  {record.binding_id:16s} [{label}] predicted {np.round(predicted, 6)} "
            f"analytic {np.round(exact, 6)} max|error| {np.max(np.abs(predicted - exact)):.2e}"
        )

    print("Derivative check of 0.5 ||r||^2 in (kappa_l, kappa_r, g) at (1.25, 0.8, 0.6):")

    def misfit(theta: Array) -> Array:
        return 0.5 * jnp.sum(whitened(solve, theta)[0] ** 2)

    point = np.asarray([1.25, 0.8, 0.6])
    gradient = np.asarray(jax.jit(jax.grad(misfit))(jnp.asarray(point)))
    value = jax.jit(misfit)
    step = 1e-6
    central = np.asarray(
        [
            (
                float(value(jnp.asarray(point + step * axis)))
                - float(value(jnp.asarray(point - step * axis)))
            )
            / (2.0 * step)
            for axis in np.eye(3)
        ]
    )
    error = np.max(np.abs(gradient - central)) / np.max(np.abs(central))
    print(
        f"  implicit adjoint {gradient}\n  central diff.    {central}\n  rel. error {error:.1e}"
    )
    if error > 1e-6:
        raise RuntimeError("The implicit gradient disagrees with central differences.")

    for mode in ("mathematical", "rhs-only"):
        capability = prepared.derivative_capability(dense_policy(mode))
        print(f"Derivatives under {mode!r}:")
        print(f"  admitted {[name for name, _ in capability.admitted]}")
        for name, reason in capability.refused:
            print(f"  refused  {name}: {reason}")

    objective = phx.solver.SolverObjective(
        solve, bind, measure, objective_id="plate-inverse"
    )
    initial = phx.bind_component(PlateParameters(*INITIAL), phx.ComponentAuthority.MODEL)
    result = phx.solver.train_components(
        initial,
        (objective,),
        optimizer=optax.lbfgs(),
        steps=TRAINING_STEPS,
        key=jr.key(0),
    )
    print(
        f"L-BFGS from (kappa_r, g) = {INITIAL}: {result.accepted_updates} accepted "
        f"of {result.attempts} attempts"
    )
    for attempt, objective_value in enumerate(np.asarray(result.values)):
        print(f"  attempt {attempt:2d}  objective {objective_value:.6e}")
    model = result.tree.model
    estimate = (float(jnp.exp(model.log_conductivity)), float(model.heat_flux))
    _, accepted, comparisons = jax.jit(lambda theta: whitened(solve, theta))(
        jnp.asarray((KAPPA_LEFT, *estimate))
    )
    quadratic = sum(float(declared(item.quadratic)) for item in comparisons)
    log_likelihood = sum(float(declared(item.log_likelihood)) for item in comparisons)
    position = jnp.stack((model.log_conductivity, model.heat_flux))
    posterior = phx.uq.posterior_problem_from_solver_objective(
        objective,
        result.tree,
        phx.uq.ParameterSpace(position, priors=phx.uq.Normal(0.0, 10.0)),
    )
    # Linearized estimator: the noise spreads the estimate with the Gauss-Newton
    # covariance C = (J^T J)^-1 of the whitened residual; the O(h^2) bias b of the
    # predictions at the truth (against the analytic plate) shifts it by C J^T b.
    jacobian = np.asarray(jax.jit(jax.jacfwd(posterior.gauss_newton_residual))(position))
    physical = jacobian * np.asarray([1.0 / estimate[0], 1.0])  # d/d(kappa_r, g)
    covariance = np.linalg.solve(physical.T @ physical, np.eye(2))
    deviation = np.sqrt(np.diag(covariance))
    bias = np.concatenate(
        [
            (np.asarray(solution.observation(record.binding_id).values) - exact)
            / record.std
            for record, exact in zip(RECORDS, exact_measurements(*TRUTH), strict=True)
        ]
    )
    shift = np.abs(covariance @ physical.T @ bias)
    print(
        f"Estimate kappa_right = {estimate[0]:.5f} +- {deviation[0]:.5f} "
        f"(truth {TRUTH[0]}, discretization shift {shift[0]:.5f})"
    )
    print(
        f"Estimate g           = {estimate[1]:.5f} +- {deviation[1]:.5f} "
        f"(truth {TRUTH[1]}, discretization shift {shift[1]:.5f})"
    )
    print(
        f"Comparison quadratic {quadratic:.4f} over {len(SENSORS) + 2} values, "
        f"normalized log-likelihood {log_likelihood:.4f}, prewhitened posterior "
        f"log-likelihood {float(posterior.log_likelihood(position)):.4f}"
    )
    if not bool(accepted) or result.accepted_updates == 0:
        raise RuntimeError("The inversion was not accepted.")
    error = np.abs(np.asarray(estimate) - np.asarray(TRUTH))
    if np.any(error > 4.0 * deviation + shift):
        raise RuntimeError(
            f"The inversion missed the truth by {error}, beyond four standard "
            f"deviations {deviation} plus the discretization shift {shift}."
        )


if __name__ == "__main__":
    main()
