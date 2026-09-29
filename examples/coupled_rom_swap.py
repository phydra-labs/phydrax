#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Swap the finite-element region of a coupled FE-VEM plate for a Galerkin ROM.

Physics. Steady heat conduction ``-div(kappa grad u) = s`` (``s = 2``) on the
plate ``[0, 2] x [0, 1]``. The left half is meshed with P1 triangles, the right
half with degree-1 virtual elements on distorted running-bond bricks whose
interface vertices do not match the triangles'; a mortar
``ScalarTransmissionLaw`` couples them at ``x = 1``. The left wall is held at
``u = 0``, the right wall receives the heat flux ``g = kappa du/dn``, and the top
and bottom are insulated. The conductivity is ``kappa_left`` on the left and
``kappa_right`` on the right; ``(kappa_left, kappa_right, g)`` are parameter
bindings of the plan.

Offline. Full-order coupled solves at seeded training parameters give
snapshots of the triangles' solve coordinates; an uncentered physical POD on
the state space of a ``ComponentResidualProvider`` of the triangles gives the
basis. For each POD rank ``r`` the triangles are replaced by
``ReducedComponent("triangles", FullResidualGalerkin(...))``; every binding,
law, parameter, and observation of the plan is unchanged. The POD's method of
snapshots resolves singular values only to about ``sqrt(eps)`` of the largest,
and the last direction of the region's solution manifold lies below that
floor, so the spanning ROM uses the leading left singular vectors of a host
SVD of the same snapshots (accurate to ``eps`` relative), whose numerical rank
is the manifold dimension.

Online. At held-out parameters the ROM-coupled solution is compared with the
full-order coupled solution (the ROM truth) and with the host analytic
temperature (the discretization reference). The mortar's weak continuity holds
exactly at every rank; its flux balance compares the full owner's reaction of
the reconstructed field with the multiplier, which is the interface residual of
the ROM: it decays with the rank, and the solution is accepted once the basis
spans the parametric solution manifold of the region.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain


jax.config.update("jax_enable_x64", True)

cpl = phx.solver.coupling
M = phx.measurement

LEVEL = 0
INTERFACE_X = 1.0
SOURCE = 2.0
REFERENCE = (1.3, 0.7, 0.5)  # (kappa_left, kappa_right, g) of the preparation
LOWER = np.asarray([0.8, 0.5, -0.5])
UPPER = np.asarray([1.6, 1.2, 1.0])
TRAINING_COUNT = 12
TRAINING_SEED = 7
HELD_OUT = np.asarray([[1.1, 0.9, 0.2], [1.45, 0.6, 0.8]])
SOURCES = ("plate-training-snapshots",)
SENSORS = np.asarray([[0.3, 0.4, 0.0], [0.7, 0.6, 0.0], [0.5, 0.9, 0.0]])

WATT = phx.units.derived_unit("W", ((phx.units.JOULE, 1), (phx.units.SECOND, -1)))
CONDUCTIVITY_UNIT = phx.units.derived_unit(
    "W/(m K)", ((WATT, 1), (phx.units.METER, -1), (phx.units.KELVIN, -1))
)
HEAT_FLUX_UNIT = phx.units.derived_unit("W/m^2", ((WATT, 1), (phx.units.METER, -2)))


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
    interface: IntegrationDomain
    triangle_nodes: np.ndarray


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
    interface = _facets(fe_space, 1.0)
    law = cpl.ScalarTransmissionLaw(
        "transmission",
        binding,
        (
            cpl.TransmissionSide("triangles", "triangles", "u", interface),
            cpl.TransmissionSide("polygons", "polygons", "u", _facets(vem_space, 1.0)),
        ),
        cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="polygons")),
    )
    return Plate(triangles, polygons, law, binding, cover, interface, nodes)


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


def dense_policy() -> phx.linalg.LinearSolvePolicy:
    return phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        differentiation=phx.linalg.DifferentiationPolicy("mathematical"),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=16_000_000, max_bytes=256 * 1024 * 1024
        ),
    )


# --- Observations of the reduced region -------------------------------------------------


CONTRACT = phx.SpatialCoordinateContract(phx.units.METER)
TEMPERATURE = M.QuantitySpec(
    "example", "temperature", "temperature", phx.units.KELVIN, "temperature"
)
OBSERVATIONS = ("left-sensors", "interface-mean")


def observations(plate: Plate) -> tuple[cpl.AbstractObservationBinding, ...]:
    """Temperatures at three points and the mean on the triangles' interface side."""
    return (
        cpl.FieldPointObservation(
            "left-sensors",
            "triangles",
            "u",
            quantity=TEMPERATURE,
            support=M.PointSampleSupport(SENSORS, ("a", "b", "c"), CONTRACT),
            sampling=M.SamplingSemantics(M.SpatialSamplingKind.POINT),
            field_unit=phx.units.KELVIN,
        ),
        cpl.FieldBoundaryObservation(
            "interface-mean",
            "triangles",
            "u",
            plate.interface,
            statistic="average",
            rule=FacetTraceRule(points=2),
            quantity=TEMPERATURE,
            support=M.IndexSampleSupport((1,), ("interface",)),
            sampling=M.SamplingSemantics(M.SpatialSamplingKind.SURFACE_AVERAGE),
            field_unit=phx.units.KELVIN,
        ),
    )


def prepare(
    plate: Plate, triangles: cpl.AbstractTraceComponent
) -> cpl.PreparedCoupledProblem:
    """The same plan with ``triangles`` (full or reduced) as the left region."""
    plan = cpl.CoupledProblemPlan(
        "plate",
        components=(triangles, plate.polygons),
        bindings=(plate.binding,),
        laws=(plate.law,),
        parameters=parameter_bindings(),
        observations=observations(plate),
    )
    names = ("conductivity-left", "conductivity-right", "heat-flux")
    return cpl.prepare_coupled_problem(
        plan,
        interface_owners=(plate.cover,),
        parameters={
            name: jnp.asarray(value) for name, value in zip(names, REFERENCE, strict=True)
        },
    )


def solver(
    prepared: cpl.PreparedCoupledProblem,
) -> Callable[[Array], cpl.CoupledSolution]:
    def solve(theta: Array) -> cpl.CoupledSolution:
        return cpl.solve_coupled_problem(
            prepared,
            parameters={
                "conductivity-left": theta[0],
                "conductivity-right": theta[1],
                "heat-flux": theta[2],
            },
            policy=dense_policy(),
        )

    return jax.jit(solve)


# --- Reduced-order model ----------------------------------------------------------------


def reduced_triangles(
    provider: cpl.ComponentResidualProvider, modes: Array
) -> cpl.ReducedComponent:
    """The Galerkin ROM of the triangles on the orthonormal ``modes``."""
    basis = phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(provider.state_space, modes, orthonormal=True),
        role="state",
        state_contract_id=provider.residual_id,
        support_id=provider.support_id,
        measure_id="euclidean-coordinates",
        geometry_id=provider.geometry_id,
        source_artifact_ids=SOURCES,
    )
    galerkin = phx.rom.FullResidualGalerkin(
        phx.rom.trial_test_reduction_from_bases(basis), provider
    )
    return cpl.ReducedComponent("triangles", galerkin)


def relative(value: Array, reference: Array) -> float:
    difference = np.linalg.norm(np.asarray(value) - np.asarray(reference))
    return float(difference / np.linalg.norm(np.asarray(reference)))


def defect(report: cpl.InterfaceDefectReport, name: str) -> float:
    """One interface defect relative to its reference magnitude."""
    index = report.names.index(name)
    return float(report.values[index] / report.scales[index])


def main() -> None:
    plate = build_plate(LEVEL)
    full = prepare(plate, plate.triangles)
    full_solve = solver(full)

    rng = np.random.default_rng(TRAINING_SEED)
    training = LOWER + (UPPER - LOWER) * rng.uniform(size=(TRAINING_COUNT, 3))
    index = [component.name for component in full.components].index("triangles")
    snapshots = []
    for theta in training:
        solution = full_solve(jnp.asarray(theta))
        if not bool(solution.accepted):
            raise RuntimeError(f"The full-order training solve at {theta} failed.")
        snapshots.append(solution.state[index][0])
    provider = cpl.ComponentResidualProvider(plate.triangles, field="u")
    pod = phx.ml.decomposition.PhysicalPODPlan(TRAINING_COUNT, centered=False).fit(
        provider.state_space, jnp.stack(snapshots), source_artifact_ids=SOURCES
    )
    singular = np.asarray(pod.singular_values)
    # Uncentered snapshot matrix (columns are snapshots) scaled like the POD's
    # uniform sample weights, so its singular values are comparable.
    columns = np.stack([np.asarray(item) for item in snapshots], axis=1)
    left, host, _ = np.linalg.svd(columns / np.sqrt(TRAINING_COUNT), full_matrices=False)
    dimension = int(
        np.sum(host > max(columns.shape) * np.finfo(host.dtype).eps * host[0])
    )
    truths = [full_solve(jnp.asarray(theta)) for theta in HELD_OUT]
    # The triangles' block solves (1 / kappa_left) K0 u = f + B lambda, so its
    # solutions span at most 1 + (multiplier size) directions; POD retains the
    # modes above the roundoff floor of the snapshot Gram matrix, the host SVD
    # resolves the full manifold.
    print(
        f"Offline: {TRAINING_COUNT} full-order solves; {provider.state_space.size} "
        f"triangle solve coordinates, {truths[0].law_state('transmission')[0].size} "
        f"multiplier coordinates; POD retains rank {pod.achieved_rank}, the host SVD "
        f"has numerical rank {dimension}"
    )
    print(f"  POD singular values      {np.array2string(singular, precision=2)}")
    print(f"  host SVD singular values {np.array2string(host, precision=2)}")
    sweep = [
        (rank, pod.subspace.basis[:, :rank], singular[rank - 1])
        for rank in range(1, min(pod.achieved_rank, dimension - 1) + 1)
    ]
    sweep.append((dimension, jnp.asarray(left[:, :dimension]), host[dimension - 1]))

    print(
        f"Held-out {HELD_OUT.tolist()} (kappa_left, kappa_right, g): worst relative "
        "error against the full-order coupled solution"
    )
    print(
        "  rank  sigma_r    triangles  polygons   multiplier observed   "
        "weak-cont. flux-bal.  accepted"
    )
    final = None
    for rank, modes, sigma in sweep:
        reduced = prepare(plate, reduced_triangles(provider, modes))
        solve = solver(reduced)
        solutions = [solve(jnp.asarray(theta)) for theta in HELD_OUT]
        errors = np.max(
            [
                [
                    relative(ours.field("triangles", "u"), truth.field("triangles", "u")),
                    relative(ours.field("polygons", "u"), truth.field("polygons", "u")),
                    relative(
                        ours.law_state("transmission")[0],
                        truth.law_state("transmission")[0],
                    ),
                    max(
                        relative(
                            ours.observation(name).values, truth.observation(name).values
                        )
                        for name in OBSERVATIONS
                    ),
                ]
                for ours, truth in zip(solutions, truths, strict=True)
            ],
            axis=0,
        )
        reports = [ours.interface("transmission") for ours in solutions]
        weak = max(defect(report, "weak-continuity") for report in reports)
        balance = max(defect(report, "flux-balance") for report in reports)
        accepted = all(bool(ours.accepted) for ours in solutions)
        print(
            f"  {rank:4d}  {sigma:.2e}   "
            + "  ".join(f"{value:.2e}" for value in errors)
            + f"   {weak:.2e}   {balance:.2e}   {accepted}"
        )
        final = (rank, reduced, solutions, errors, accepted)

    if final is None:
        raise RuntimeError("No rank was swept.")
    rank, reduced, solutions, errors, accepted = final
    print(
        f"Rank {rank} owner {reduced.components[index].owner_id[:12]}..., problem "
        f"{reduced.problem_id[:12]}... (full-order problem {full.problem_id[:12]}...)"
    )
    print("Host analytic temperature at the triangle nodes (max |error|):")
    for theta, ours, truth in zip(HELD_OUT, solutions, truths, strict=True):
        exact = exact_temperature(plate.triangle_nodes, *theta)
        reduced_error = np.max(np.abs(np.asarray(ours.field("triangles", "u")) - exact))
        full_error = np.max(np.abs(np.asarray(truth.field("triangles", "u")) - exact))
        print(f"  {theta.tolist()}: ROM {reduced_error:.4e}  full order {full_error:.4e}")
    if not accepted or np.max(errors) > 1e-8:
        raise RuntimeError(
            f"The rank-{rank} ROM does not reproduce the full-order plate: {errors}."
        )


if __name__ == "__main__":
    main()
