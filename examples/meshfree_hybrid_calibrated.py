"""Hybrid point-cloud / finite-volume overlap solve with calibrated predictive bands.

Problem: ``-div(K grad u) = f`` on the unit square with
``K = 1 + sin(pi x) sin(pi y) / 4`` and the nonpolynomial manufactured family

    u(x, y; delta, omega, phase) = exp((x - 1) / delta) (1 + sin(omega y + phase) / 2)
                                   + sin(pi x + 0.3) cos(pi y) / 2,

whose boundary layer at ``x = 1`` lies in the point-cloud region. The outer
boundary carries the exact Dirichlet data and ``f`` is the exact automatic
derivative of the closed form, independent of both discretizations.

Decomposition: a point cloud (PHS RBF-FD, degree 3) on the frame
``Omega minus (b, 1 - b)^2`` and a cell-centered finite-volume grid on
``[a, 1 - a]^2`` with ``a < b``. ``OverlapDirichletLaw`` closes the cloud's
artificial nodes with value stencils from the cell centers and the grid's
artificial faces with value stencils from the cloud, so the coupled solution
is the composite discretization of the original problem.

Solve: native GMRES preconditioned by native subspace correction with one exact
sparse local factorization per subdomain, additive (parallel Schwarz) and
multiplicative (alternating Schwarz); unpreconditioned GMRES is reported with
its native status.

Verification: analytic errors of both owners under refinement with observed
rates, the owners' overlap mismatch, law/component certificates, and a
monolithic point-cloud reference on the whole square.

Calibration: complete cases are independent manufactured problems drawn
i.i.d. from the declared parameter family (the exchangeability assumption).
The predictor is the coarse hybrid solve and the target the analytic solution
on a fixed probe set; disjoint train (score scale), calibration (finite-sample
radius), and test (coverage) cases. Separately labeled shifted sets (an
out-of-family boundary-layer regime and a narrower-overlap geometry) report
their measured coverage without any distribution-free claim.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import Literal, TypedDict

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax.linalg as la
from phydrax import uq
from phydrax.discretization import (
    ConservativeDiffusionPlan,
    FiniteVolumeDiscretization,
    FiniteVolumePlan,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    prepare_point_cloud_field_reconstruction,
    PreparedConservativeDiffusion,
    PreparedFieldReconstruction,
    PreparedPointCloudDiscretization,
    PreparedPointCloudPoisson,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    MeshfreeFunctional,
    MeshfreeNeighborhoodPlan,
    MeshfreeOperator,
    prepare_local_stencils,
)
from phydrax.geometry import Rectangle
from phydrax.solver.coupling import (
    certify_coupled_state,
    ContributionEndpoint,
    CoupledProblemPlan,
    CoupledSolution,
    FiniteVolumeComponent,
    MeshfreeComponent,
    OverlapDirichletEvidence,
    OverlapDirichletLaw,
    OverlapTransferPolicy,
    prepare_coupled_problem,
    prepare_overlap_schwarz,
    PreparedCoupledProblem,
    PreparedOverlapTransfers,
    solve_coupled_problem,
)
from phydrax.sparse import SparseCoordinateOperator


type Parameters = tuple[float, float, float]


class SolveReport(TypedDict):
    iterations: int
    status: int
    native_successful: bool
    accepted: bool


class SchwarzSubdomainReport(TypedDict):
    component: str
    size: int
    colors: int
    block_relative_error: float
    factor_entries: int
    minimum_pivot: float


class TransferReport(TypedDict):
    route: str
    targets: int
    refused_rows: int
    maximum_condition: float
    constant_defect: float
    linear_defect: float


class RefinementReport(TypedDict):
    cells: int
    spacing: float
    cloud_points: int
    grid_cells: int
    cloud_max_error: float
    grid_max_error: float
    monolithic_meshfree_max_error: float
    interface_defects: dict[str, float]
    interface_gated: dict[str, bool]
    component_residuals: dict[str, float]
    solves: dict[str, SolveReport]
    schwarz_subdomains: list[SchwarzSubdomainReport]
    transfers: list[TransferReport]
    overlap_gap: float
    node_identity_defect: float
    node_load_defect: float


class ShiftReport(TypedDict):
    label: str
    shift_kind: str
    cases: int
    all_cases_accepted: bool
    measured_simultaneous_coverage: float
    mean_width: float
    distribution_free_guarantee: bool


class CalibrationSplitReport(TypedDict):
    train: int
    calibration: int
    test: int
    disjoint: bool


class InDistributionReport(TypedDict):
    label: str
    measured_simultaneous_coverage: float
    standard_error: float
    coverage_confidence_interval: list[float]
    finite_sample_expected_coverage: float
    finite_sample_coverage_deviation: float
    mean_width: float


class CalibrationReport(TypedDict):
    exchangeability: str
    cells: int
    probes: int
    score: str
    split: CalibrationSplitReport
    all_cases_accepted: bool
    multi_rhs_iterations: list[int]
    nominal_coverage: float
    radius: float
    in_distribution: InDistributionReport
    shifted: dict[str, ShiftReport]


class DecompositionReport(TypedDict):
    cloud: str
    grid: str


class RefinementStudyReport(TypedDict):
    levels: list[RefinementReport]
    cloud_rates: list[float]
    grid_rates: list[float]
    monolithic_rates: list[float]
    overlap_mismatch_max: list[float]
    overlap_mismatch_l2: list[float]


class HybridWorkflowReport(TypedDict):
    problem: str
    decomposition: DecompositionReport
    refinement: RefinementStudyReport
    calibration: CalibrationReport


REFERENCE_PARAMETERS: Parameters = (0.2, 2.0, 0.5)
IN_FAMILY_DELTA = (0.12, 0.25)
SHIFTED_DELTA = (0.04, 0.07)
OMEGA = (1.0, 3.0)
PROBE_AXIS = (0.06, 0.35, 0.65, 0.94)
CLOUD = "cloud"
GRID = "grid"
FIELD = "u"
LAW = "overlap"
STENCIL = LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3)
CELL_STENCIL = LocalStencilPolicy(polynomial_degree=2)
NEIGHBORS = 24
CELL_NEIGHBORS = 12
JITTER = 0.15


@dataclass(frozen=True)
class OverlapGeometry:
    """Grid margin ``a`` and cloud margin ``b`` (``a < b``) of the unit square."""

    grid_margin: float
    cloud_margin: float
    label: str


BASE_GEOMETRY = OverlapGeometry(0.125, 0.25, "overlap-width-1/8")
SHIFTED_GEOMETRY = OverlapGeometry(0.1875, 0.25, "overlap-width-1/16")


# --- Manufactured family -------------------------------------------------------------


def diffusivity(points: Array) -> Array:
    return 1.0 + 0.25 * jnp.sin(jnp.pi * points[..., 0]) * jnp.sin(
        jnp.pi * points[..., 1]
    )


def exact(points: Array, parameters: Array) -> Array:
    delta, omega, phase = parameters[0], parameters[1], parameters[2]
    x, y = points[..., 0], points[..., 1]
    layer = jnp.exp((x - 1.0) / delta) * (1.0 + 0.5 * jnp.sin(omega * y + phase))
    return layer + 0.5 * jnp.sin(jnp.pi * x + 0.3) * jnp.cos(jnp.pi * y)


@jax.jit
def source(points: Array, parameters: Array) -> Array:
    """``-div(K grad u)`` by exact automatic derivatives of the closed form."""

    def flux(point: Array) -> Array:
        return diffusivity(point) * jax.grad(lambda value: exact(value, parameters))(
            point
        )

    def divergence(point: Array) -> Array:
        return jnp.trace(jax.jacfwd(flux)(point))

    flat = points.reshape((-1, points.shape[-1]))
    return -jax.vmap(divergence)(flat).reshape(points.shape[:-1])


def draw_parameters(
    rng: np.random.Generator, count: int, delta: tuple[float, float], /
) -> np.ndarray:
    """``count`` i.i.d. parameter triples ``(delta, omega, phase)`` of one family."""
    return np.stack(
        (
            rng.uniform(*delta, count),
            rng.uniform(*OMEGA, count),
            rng.uniform(0.0, 2.0 * math.pi, count),
        ),
        axis=1,
    )


# --- Owners -----------------------------------------------------------------------------


def _lattice(cells: int, /) -> np.ndarray:
    axis = np.linspace(0.0, 1.0, cells + 1)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.stack((x.ravel(), y.ravel()), axis=1)


def _jitter(points: np.ndarray, free: np.ndarray, spacing: float, seed: int, /) -> None:
    rng = np.random.default_rng(seed)
    points[free] += rng.uniform(-JITTER, JITTER, (int(np.sum(free)), 2)) * spacing


def _cloud(
    points: np.ndarray, boundary: np.ndarray, normals: np.ndarray, spacing: float, /
) -> PreparedPointCloudDiscretization:
    return PointCloudPlan(
        jnp.asarray(points),
        jnp.full((points.shape[0],), spacing * spacing),
        boundary_mask=jnp.asarray(boundary),
        boundary_normals=jnp.asarray(normals),
        stencil=STENCIL,
        neighbors=NEIGHBORS,
    ).prepare()


@dataclass(frozen=True)
class HybridSetup:
    """Prepared owners, transfers, and case-independent pieces of one level."""

    cells: int
    geometry: OverlapGeometry
    cloud: PreparedPointCloudDiscretization
    poisson: PreparedPointCloudPoisson
    native: SparseCoordinateOperator
    reconstruction: PreparedFieldReconstruction
    outer_rows: np.ndarray
    finite_volume: FiniteVolumeDiscretization
    diffusion: PreparedConservativeDiffusion
    cell_centers: np.ndarray
    law: OverlapDirichletLaw

    @property
    def spacing(self) -> float:
        return 1.0 / self.cells


def prepare_setup(
    cells: int, geometry: OverlapGeometry, /, *, seed: int = 0
) -> HybridSetup:
    """Frame cloud, interior grid, and the overlap transfers at spacing ``1/cells``."""
    spacing = 1.0 / cells
    margin, core = geometry.grid_margin, geometry.cloud_margin
    grid_cells = round((1.0 - 2.0 * margin) * cells)
    if not (
        math.isclose(core * cells, round(core * cells))
        and math.isclose(grid_cells, (1.0 - 2.0 * margin) * cells)
    ):
        raise ValueError("Both margins must be multiples of the spacing.")
    points = _lattice(cells)
    tolerance = 1.0e-12
    hole = np.all((points > core + tolerance) & (points < 1.0 - core - tolerance), axis=1)
    points = points[~hole]
    outer = np.any((points < tolerance) | (points > 1.0 - tolerance), axis=1)
    inner = ~outer & np.all(
        (points >= core - tolerance) & (points <= 1.0 - core + tolerance), axis=1
    )
    _jitter(points, ~(outer | inner), spacing, seed)
    normals = np.zeros_like(points)
    normals[outer] = np.where(
        points[outer] < tolerance, -1.0, np.where(points[outer] > 1 - tolerance, 1.0, 0.0)
    )
    offset = points[inner] - 0.5
    normals[inner] = -np.where(
        np.abs(np.abs(offset) - (0.5 - core)) < tolerance, np.sign(offset), 0.0
    )
    normals[outer | inner] /= np.linalg.norm(
        normals[outer | inner], axis=1, keepdims=True
    )
    cloud = _cloud(points, outer | inner, normals, spacing)
    outer_rows = np.flatnonzero(outer)
    inner_rows = np.flatnonzero(inner)
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition("dirichlet", outer_rows, 0.0, label="outer"),
            PointBoundaryCondition("dirichlet", inner_rows, 0.0, label="artificial"),
        ),
        row_count=points.shape[0],
    )
    # Only the physical sparse rows are consumed; the owner's own solve is unused.
    poisson = PointCloudPoissonPlan(
        cloud, boundary, linear_policy=la.LinearSolvePolicy(la.GMRES())
    ).prepare(diffusivity(cloud.points))
    full = cloud.field_spaces[0].vector_space
    physical = poisson.physical_assembly.operator
    if not isinstance(physical, SparseCoordinateOperator):
        raise TypeError("The point-cloud Poisson rows must be native sparse rows.")
    native = SparseCoordinateOperator(
        physical.relation, physical.coefficients, source=full, target=full
    )
    support = (
        Rectangle((0.5, 0.5), (1.0, 1.0))
        - Rectangle((0.5, 0.5), (1.0 - 2.0 * core, 1.0 - 2.0 * core))
    ).compile()
    reconstruction = prepare_point_cloud_field_reconstruction(
        cloud, support_geometry=support, radius=3.0 * spacing, capacity=40
    )
    grid = TensorGridPlan(
        (UniformCellAxisSpec(grid_cells), UniformCellAxisSpec(grid_cells)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[margin, margin], [1.0 - margin, 1.0 - margin]]))
    finite_volume = FiniteVolumePlan(grid, field_name=FIELD).prepare()
    cell_centers = np.asarray(finite_volume.cell_centers)
    diffusion = ConservativeDiffusionPlan(
        grid,
        boundaries={"x": ("dirichlet", "dirichlet"), "y": ("dirichlet", "dirichlet")},
    ).prepare(diffusivity(jnp.asarray(cell_centers)))
    flat = cell_centers.reshape((-1, 2))
    overlap = np.flatnonzero(~np.all((flat > core) & (flat < 1.0 - core), axis=1))
    transfers = PreparedOverlapTransfers(
        cloud,
        inner_rows,
        finite_volume,
        (("x", "lower"), ("x", "upper"), ("y", "lower"), ("y", "upper")),
        overlap,
        policy=OverlapTransferPolicy(
            point_stencil=STENCIL,
            cell_stencil=CELL_STENCIL,
            point_neighbors=NEIGHBORS,
            cell_neighbors=CELL_NEIGHBORS,
        ),
    )
    law = OverlapDirichletLaw(
        LAW,
        ContributionEndpoint(CLOUD, FIELD),
        ContributionEndpoint(GRID, FIELD),
        transfers,
    )
    return HybridSetup(
        cells=cells,
        geometry=geometry,
        cloud=cloud,
        poisson=poisson,
        native=native,
        reconstruction=reconstruction,
        outer_rows=outer_rows,
        finite_volume=finite_volume,
        diffusion=diffusion,
        cell_centers=cell_centers,
        law=law,
    )


def hybrid_problem(
    setup: HybridSetup, parameters: Parameters, /
) -> PreparedCoupledProblem:
    """Coupled problem of one manufactured case on a prepared level."""
    theta = jnp.asarray(parameters, dtype=jnp.float64)
    cloud = setup.cloud
    points = cloud.points
    load = setup.poisson.physical_rhs(
        source(points, theta),
        boundary_values={"outer": exact(points[setup.outer_rows], theta)},
    )
    meshfree = MeshfreeComponent(
        cloud,
        setup.native,
        setup.reconstruction,
        cloud.quadrature_weights,
        name=CLOUD,
        owner_id=cloud.prepared_id,
        field=FIELD,
        load=load,
    )
    finite_volume = FiniteVolumeComponent(
        GRID,
        setup.finite_volume,
        setup.diffusion,
        source=np.asarray(source(jnp.asarray(setup.cell_centers), theta)),
    )
    plan = CoupledProblemPlan(
        f"hybrid-overlap-{setup.cells}-{setup.geometry.label}",
        components=(meshfree, finite_volume),
        bindings=(),
        laws=(setup.law,),
    )
    return prepare_coupled_problem(plan)


def gmres_policy(
    builder: la.AdditiveSubspaceCorrectionBuilder
    | la.MultiplicativeSubspaceCorrectionBuilder
    | None,
    /,
    *,
    max_steps: int,
) -> la.LinearSolvePolicy:
    return la.LinearSolvePolicy(
        la.GMRES(restart=100),
        tolerance=la.TolerancePolicy(
            relative=1.0e-11, absolute=1.0e-13, max_steps=max_steps
        ),
        preconditioning=None if builder is None else la.PreconditioningPolicy(builder),
        failure=la.FailurePolicy("status"),
    )


# --- Refinement study -------------------------------------------------------------------


def _owner_errors(
    setup: HybridSetup, solution: CoupledSolution, /
) -> tuple[float, float]:
    theta = jnp.asarray(REFERENCE_PARAMETERS)
    cloud = float(
        jnp.max(jnp.abs(solution.field(CLOUD, FIELD) - exact(setup.cloud.points, theta)))
    )
    grid = float(
        jnp.max(
            jnp.abs(
                solution.field(GRID, FIELD)
                - exact(jnp.asarray(setup.cell_centers), theta)
            )
        )
    )
    return cloud, grid


def _solve_record(solution: CoupledSolution, /) -> SolveReport:
    linear = solution.linear
    if linear is None:
        raise ValueError(
            "The hybrid problem is affine; a linear solve record is expected."
        )
    return {
        "iterations": int(linear.diagnostics.iterations),
        "status": int(linear.status),
        "native_successful": bool(solution.native_successful),
        "accepted": bool(solution.accepted),
    }


def meshfree_reference(cells: int, /, *, seed: int = 0) -> float:
    """Monolithic point-cloud solve on the whole square: maximum analytic error."""
    spacing = 1.0 / cells
    points = _lattice(cells)
    tolerance = 1.0e-12
    outer = np.any((points < tolerance) | (points > 1.0 - tolerance), axis=1)
    _jitter(points, ~outer, spacing, seed)
    normals = np.where(
        points < tolerance, -1.0, np.where(points > 1 - tolerance, 1.0, 0.0)
    )
    normals[outer] /= np.linalg.norm(normals[outer], axis=1, keepdims=True)
    cloud = _cloud(points, outer, normals, spacing)
    theta = jnp.asarray(REFERENCE_PARAMETERS)
    rows = np.flatnonzero(outer)
    boundary = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet",
                rows,
                np.asarray(exact(cloud.points[rows], theta)),
                label="outer",
            ),
        ),
        row_count=points.shape[0],
    )
    result = (
        PointCloudPoissonPlan(cloud, boundary)
        .prepare(diffusivity(cloud.points))
        .solve(source(cloud.points, theta))
    )
    if not bool(result.successful):
        raise RuntimeError(
            f"Monolithic reference solve failed: status {int(result.status)}."
        )
    return float(jnp.max(jnp.abs(result.values - exact(cloud.points, theta))))


def refinement_level(
    cells: int, /, *, seed: int = 0, baseline_steps: int = 300
) -> RefinementReport:
    setup = prepare_setup(cells, BASE_GEOMETRY, seed=seed)
    prepared = hybrid_problem(setup, REFERENCE_PARAMETERS)
    schwarz = prepare_overlap_schwarz(prepared, LAW)
    solves = {
        "unpreconditioned": solve_coupled_problem(
            prepared, policy=gmres_policy(None, max_steps=baseline_steps)
        ),
        "additive-schwarz": solve_coupled_problem(
            prepared,
            policy=gmres_policy(
                la.AdditiveSubspaceCorrectionBuilder(schwarz.terms), max_steps=400
            ),
        ),
        "multiplicative-schwarz": solve_coupled_problem(
            prepared,
            policy=gmres_policy(
                la.MultiplicativeSubspaceCorrectionBuilder(schwarz.terms), max_steps=400
            ),
        ),
    }
    accepted = solves["multiplicative-schwarz"]
    cloud_error, grid_error = _owner_errors(setup, accepted)
    report = accepted.interface(LAW)
    evidence = prepared.laws[0].evidence
    if not isinstance(evidence, OverlapDirichletEvidence):
        raise TypeError("The overlap law publishes OverlapDirichletEvidence.")
    transfers = setup.law.transfers
    return {
        "cells": cells,
        "spacing": setup.spacing,
        "cloud_points": int(setup.cloud.points.shape[0]),
        "grid_cells": int(setup.cell_centers.shape[0] * setup.cell_centers.shape[1]),
        "cloud_max_error": cloud_error,
        "grid_max_error": grid_error,
        "monolithic_meshfree_max_error": meshfree_reference(cells, seed=seed),
        "interface_defects": {
            name: float(value)
            for name, value in zip(report.names, report.values, strict=True)
        },
        "interface_gated": dict(zip(report.names, report.gated, strict=True)),
        "component_residuals": {
            item.component: float(item.residual_norm / item.scale)
            for item in accepted.components
        },
        "solves": {name: _solve_record(solution) for name, solution in solves.items()},
        "schwarz_subdomains": [
            {
                "component": item.component,
                "size": item.size,
                "colors": item.colors,
                "block_relative_error": item.block_relative_error,
                "factor_entries": item.factor_entries,
                "minimum_pivot": item.minimum_pivot,
            }
            for item in schwarz.evidence
        ],
        "transfers": [
            {
                "route": item.route,
                "targets": item.targets,
                "refused_rows": item.refused_rows,
                "maximum_condition": item.maximum_condition,
                "constant_defect": item.constant_defect,
                "linear_defect": item.linear_defect,
            }
            for item in transfers.evidence
        ],
        "overlap_gap": transfers.overlap_gap,
        "node_identity_defect": evidence.node_identity_defect,
        "node_load_defect": evidence.node_load_defect,
    }


def observed_rates(errors: Sequence[float], cells: Sequence[int], /) -> list[float]:
    return [
        math.log(errors[index] / errors[index + 1])
        / math.log(cells[index + 1] / cells[index])
        for index in range(len(errors) - 1)
    ]


# --- Calibration ------------------------------------------------------------------------


def probe_points() -> np.ndarray:
    """Fixed physical probes: boundary-layer, frame, and interior-core sites."""
    x, y = np.meshgrid(np.asarray(PROBE_AXIS), np.asarray(PROBE_AXIS), indexing="ij")
    return np.stack((x.ravel(), y.ravel()), axis=1)


def _value_operator(
    sources: np.ndarray,
    targets: np.ndarray,
    neighbors: int,
    stencil: LocalStencilPolicy,
    /,
) -> MeshfreeOperator:
    """Native value stencils ``sources -> targets`` (refusing unsupported rows)."""
    neighborhood = MeshfreeNeighborhoodPlan(sources, neighbors, targets=targets).prepare()
    stencils = prepare_local_stencils(
        neighborhood,
        sources,
        targets,
        (MeshfreeFunctional(((0, 0),), (1.0,), name="value"),),
        stencil,
    )
    return MeshfreeOperator(stencils, 0)


@dataclass(frozen=True)
class CompositeProbe:
    """Composite hybrid value at the probes: grid values in the cloud's hole."""

    cloud: MeshfreeOperator
    grid: MeshfreeOperator
    cloud_probes: np.ndarray
    grid_probes: np.ndarray

    def evaluate(self, solution: CoupledSolution, /) -> Array:
        count = self.cloud_probes.size + self.grid_probes.size
        values = jnp.zeros((count,), dtype=jnp.float64)
        values = values.at[self.cloud_probes].set(
            self.cloud.apply(solution.field(CLOUD, FIELD))
        )
        return values.at[self.grid_probes].set(
            self.grid.apply(solution.field(GRID, FIELD).reshape((-1,)))
        )


def composite_probe(setup: HybridSetup, probes: np.ndarray, /) -> CompositeProbe:
    core = setup.geometry.cloud_margin
    in_core = np.all((probes > core) & (probes < 1.0 - core), axis=1)
    cloud_probes = np.flatnonzero(~in_core)
    grid_probes = np.flatnonzero(in_core)
    return CompositeProbe(
        _value_operator(
            np.asarray(setup.cloud.points), probes[cloud_probes], NEIGHBORS, STENCIL
        ),
        _value_operator(
            setup.cell_centers.reshape((-1, 2)),
            probes[grid_probes],
            CELL_NEIGHBORS,
            CELL_STENCIL,
        ),
        cloud_probes,
        grid_probes,
    )


@dataclass(frozen=True)
class CaseBatch:
    """Coarse hybrid predictions, analytic targets, and per-case solve evidence."""

    predictions: np.ndarray
    targets: np.ndarray
    accepted: np.ndarray
    iterations: np.ndarray


def solve_cases(setup: HybridSetup, parameters: np.ndarray, /) -> CaseBatch:
    """Every case through one prepared operator and preconditioner, multi-RHS.

    The cases share the operator and differ only in data, so one native
    multi-right-hand-side GMRES solve with the prepared alternating Schwarz
    preconditioner serves them all; each state is then certified against its
    own case's original coupled equations.
    """
    problems = [hybrid_problem(setup, (row[0], row[1], row[2])) for row in parameters]
    schwarz = prepare_overlap_schwarz(problems[0], LAW)
    policy = gmres_policy(
        la.MultiplicativeSubspaceCorrectionBuilder(schwarz.terms), max_steps=400
    )
    system, _ = problems[0].linear_system()
    right_hand_sides = [problem.linear_system()[1] for problem in problems]
    stacked = jax.tree.map(lambda *values: jnp.stack(values, axis=-1), *right_hand_sides)
    count = len(problems)
    result = la.solve(
        system,
        stacked,
        rhs_layout=la.RHSLayout((count,), names=("case",)),
        policy=policy,
    )
    status = np.asarray(result.status)
    probes = probe_points()
    probe = composite_probe(setup, probes)
    predictions = np.zeros((count, probes.shape[0]))
    targets = np.zeros((count, probes.shape[0]))
    accepted = np.zeros((count,), dtype=np.bool_)
    for index, problem in enumerate(problems):
        state = jax.tree.map(itemgetter((..., index)), result.value)
        solution = certify_coupled_state(
            problem,
            state,
            problem.bind_arguments(),
            policy=policy,
            linear=None,
            nonlinear=None,
            native=jnp.asarray(status[index] == 0),
            derivative=jnp.asarray(False),
            tolerance=1.0e-8,
        )
        accepted[index] = bool(solution.accepted)
        predictions[index] = np.asarray(probe.evaluate(solution))
        targets[index] = np.asarray(
            exact(jnp.asarray(probes), jnp.asarray(parameters[index]))
        )
    return CaseBatch(
        predictions,
        targets,
        accepted,
        np.asarray(result.diagnostics.iterations),
    )


def finite_sample_band(
    calibration: int, test: int, alpha: float, /
) -> tuple[float, float]:
    """Mean and standard deviation of measured test coverage under exchangeability.

    Split-conformal coverage conditional on the calibration cases is
    ``Beta(k, n + 1 - k)`` with ``k = ceil((n + 1)(1 - alpha))``; ``test``
    exchangeable cases add binomial sampling variance.
    """
    rank = math.ceil((calibration + 1) * (1.0 - alpha))
    mean = rank / (calibration + 1)
    variance = (
        rank * (calibration + 1 - rank) / ((calibration + 1) ** 2 * (calibration + 2))
    )
    sampling = (mean * (1.0 - mean) - variance) / test
    return mean, math.sqrt(variance + sampling)


def _shift_record(
    calibrator: uq.ProcessConformalCalibrator,
    batch: CaseBatch,
    scale: np.ndarray,
    label: str,
    kind: str,
    /,
) -> ShiftReport:
    center = jnp.asarray(batch.predictions)
    interval = calibrator.interval(center, jnp.broadcast_to(scale, center.shape))
    lower, upper = interval.lower.data, interval.upper.data
    pointwise = uq.interval_coverage(
        lower, upper, jnp.asarray(batch.targets), reduction="none"
    )
    simultaneous = jnp.all(pointwise > 0.5, axis=1).astype(jnp.float64)
    return {
        "label": label,
        "shift_kind": kind,
        "cases": int(center.shape[0]),
        "all_cases_accepted": bool(np.all(batch.accepted)),
        "measured_simultaneous_coverage": float(jnp.mean(simultaneous)),
        "mean_width": float(uq.interval_width(lower, upper)),
        "distribution_free_guarantee": False,
    }


def calibration_study(
    *,
    cells: int,
    train_cases: int,
    calibration_cases: int,
    test_cases: int,
    shift_cases: int,
    alpha: float,
    seed: int,
) -> CalibrationReport:
    rng = np.random.default_rng(seed + 1)
    in_family = draw_parameters(
        rng, train_cases + calibration_cases + test_cases, IN_FAMILY_DELTA
    )
    regime = draw_parameters(rng, shift_cases, SHIFTED_DELTA)
    geometry_cases = draw_parameters(rng, shift_cases, IN_FAMILY_DELTA)
    base = prepare_setup(cells, BASE_GEOMETRY, seed=seed)
    family_batch = solve_cases(base, np.concatenate((in_family, regime), axis=0))
    shifted = prepare_setup(cells, SHIFTED_GEOMETRY, seed=seed)
    geometry_batch = solve_cases(shifted, geometry_cases)
    ids = [f"case-{index:03d}" for index in range(in_family.shape[0])]
    train = slice(0, train_cases)
    calibration = slice(train_cases, train_cases + calibration_cases)
    test = slice(train_cases + calibration_cases, in_family.shape[0])
    split = uq.ProcessValidationSplit(ids[train], ids[calibration], ids[test])
    predictions = family_batch.predictions
    targets = family_batch.targets
    # Train cases fit only the per-probe score scale; they never enter the radius.
    scale = np.sqrt(np.mean((predictions[train] - targets[train]) ** 2, axis=0))
    calibrator = uq.ProcessConformalCalibrator.calibrate_observable(
        jnp.asarray(predictions[calibration]),
        jnp.asarray(targets[calibration]),
        split,
        observable_name="probe-values",
        alpha=alpha,
        scale=jnp.broadcast_to(jnp.asarray(scale), (calibration_cases, scale.size)),
    )
    test_center = jnp.asarray(predictions[test])
    test_scale = jnp.broadcast_to(jnp.asarray(scale), test_center.shape)
    diagnostics = uq.process_conformal_diagnostics(
        calibrator, test_center, jnp.asarray(targets[test]), scale=test_scale
    )
    interval = calibrator.interval(test_center, test_scale)
    mean, deviation = finite_sample_band(calibration_cases, test_cases, alpha)
    regime_batch = CaseBatch(
        predictions[in_family.shape[0] :],
        targets[in_family.shape[0] :],
        family_batch.accepted[in_family.shape[0] :],
        family_batch.iterations[in_family.shape[0] :],
    )
    return {
        "exchangeability": (
            "complete cases are independent manufactured problems drawn i.i.d. from "
            f"delta~U{IN_FAMILY_DELTA}, omega~U{OMEGA}, phase~U(0, 2pi) on one fixed "
            "coarse hybrid discretization; coverage is guaranteed only for cases "
            "exchangeable with the calibration cases"
        ),
        "cells": cells,
        "probes": int(probe_points().shape[0]),
        "score": "max over probes of |prediction - target| / train-fitted probe RMS error",
        "split": {
            "train": len(split.train_case_ids),
            "calibration": len(split.calibration_case_ids),
            "test": len(split.test_case_ids),
            "disjoint": not (
                set(split.train_case_ids) & set(split.calibration_case_ids)
                or set(split.train_case_ids) & set(split.test_case_ids)
                or set(split.calibration_case_ids) & set(split.test_case_ids)
            ),
        },
        "all_cases_accepted": bool(np.all(family_batch.accepted[: in_family.shape[0]])),
        "multi_rhs_iterations": [int(value) for value in family_batch.iterations],
        "nominal_coverage": float(diagnostics.nominal_coverage),
        "radius": float(calibrator.calibrator.radius),
        "in_distribution": {
            "label": "in-distribution test cases",
            "measured_simultaneous_coverage": float(diagnostics.empirical_coverage),
            "standard_error": float(diagnostics.standard_error),
            "coverage_confidence_interval": [
                float(diagnostics.lower),
                float(diagnostics.upper),
            ],
            "finite_sample_expected_coverage": mean,
            "finite_sample_coverage_deviation": deviation,
            "mean_width": float(
                uq.interval_width(interval.lower.data, interval.upper.data)
            ),
        },
        "shifted": {
            "regime": _shift_record(
                calibrator,
                regime_batch,
                scale,
                f"out-of-family boundary layer delta~U{SHIFTED_DELTA}",
                "parameter_regime",
            ),
            "geometry": _shift_record(
                calibrator,
                geometry_batch,
                scale,
                f"narrower overlap {SHIFTED_GEOMETRY.label}",
                "geometry",
            ),
        },
    }


def run_workflow(
    *,
    levels: Sequence[int] = (16, 24, 32),
    calibration_cells: int = 16,
    train_cases: int = 6,
    calibration_cases: int = 19,
    test_cases: int = 20,
    shift_cases: int = 10,
    alpha: float = 0.1,
    seed: int = 0,
) -> HybridWorkflowReport:
    """Refinement study, Schwarz-preconditioned solves, and calibrated bands."""
    if len(levels) < 3 or list(levels) != sorted(set(levels)):
        raise ValueError("The refinement study needs at least three increasing levels.")
    records = [refinement_level(cells, seed=seed) for cells in levels]
    cells_ = [record["cells"] for record in records]

    def series(
        name: Literal[
            "cloud_max_error", "grid_max_error", "monolithic_meshfree_max_error"
        ],
    ) -> list[float]:
        return [record[name] for record in records]

    def mismatch(name: str) -> list[float]:
        return [record["interface_defects"][name] for record in records]

    return {
        "problem": (
            "-div(K grad u) = f on [0,1]^2, K = 1 + sin(pi x) sin(pi y)/4, "
            "u = exp((x-1)/delta)(1 + sin(omega y + phase)/2) + sin(pi x + 0.3) cos(pi y)/2"
        ),
        "decomposition": {
            "cloud": f"[0,1]^2 minus ({BASE_GEOMETRY.cloud_margin}, {1 - BASE_GEOMETRY.cloud_margin})^2",
            "grid": f"[{BASE_GEOMETRY.grid_margin}, {1 - BASE_GEOMETRY.grid_margin}]^2",
        },
        "refinement": {
            "levels": records,
            "cloud_rates": observed_rates(series("cloud_max_error"), cells_),
            "grid_rates": observed_rates(series("grid_max_error"), cells_),
            "monolithic_rates": observed_rates(
                series("monolithic_meshfree_max_error"), cells_
            ),
            "overlap_mismatch_max": mismatch("overlap-mismatch-max"),
            "overlap_mismatch_l2": mismatch("overlap-mismatch-l2"),
        },
        "calibration": calibration_study(
            cells=calibration_cells,
            train_cases=train_cases,
            calibration_cases=calibration_cases,
            test_cases=test_cases,
            shift_cases=shift_cases,
            alpha=alpha,
            seed=seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(run_workflow(levels=tuple(args.levels), seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
