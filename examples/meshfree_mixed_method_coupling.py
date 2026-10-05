#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A meshfree point cloud coupled to finite elements across a material interface.

The plate ``[0, 2] x [0, 1]`` is split at ``x = 1``: P1 triangles discretize
``-div(k_minus grad u) = f`` on the left with ``k_minus = 1`` and a collocated
point cloud discretizes the same law on the right with ``k_plus = 3``. The
manufactured field is continuous with a continuous normal flux, so its normal
derivative jumps by the conductivity ratio across the interface.

The cloud publishes its interface only through geometry authority: the plate
half is a polygon whose edges on ``x = 1`` run between cloud points, sampled
with a Gauss--Lobatto rule into ``PointBoundaryCharts``. The cloud's interface
rows are homogeneous Neumann conormal rows with the charts' lumped measure,
and ``MeshfreeTraceComponent`` publishes their measure-scaled residual reaction
as the canonical conormal flux. The same physical ``ScalarTransmissionLaw``
then lowers through a mortar (either orientation) or a one-sided Nitsche
imposition whose penalty is certified by the finite-element side alone.
"""

import time
from collections.abc import Callable
from typing import assert_never, Literal, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import (
    EntitySelection,
    FacetTraceRule,
    IntegrationDomain,
    PointBoundaryCondition,
    PointBoundaryPlan,
    PointCloudPlan,
    PointCloudPoissonPlan,
    prepare_point_cloud_field_reconstruction,
)
from phydrax.discretization.meshfree import (
    LocalStencilPolicy,
    PointBoundaryCharts,
    PointGhostLayerPlan,
)
from phydrax.solver.coupling import (
    CoupledProblemPlan,
    CoupledSolution,
    InterfaceBinding,
    InterfaceEndpoint,
    InterfaceSource,
    MeshfreeBoundaryTrace,
    MeshfreeTraceComponent,
    MortarImposition,
    MortarMultiplier,
    NitscheImposition,
    PairedSupportAttachment,
    prepare_coupled_problem,
    PreparedCoupledProblem,
    ScalarTransmissionLaw,
    solve_coupled_problem,
    TransmissionImposition,
    TransmissionSide,
    VariationalComponent,
)


jax.config.update("jax_enable_x64", True)

INTERFACE_X = 1.0
K_MINUS = 1.0
K_PLUS = 3.0
LEVELS = (6, 12, 24)

Side: TypeAlias = Literal["left", "right"]
type PointField = Callable[[Array], Array]


# --- Manufactured material-interface field --------------------------------------------


def exact(points: ArrayLike) -> Array:
    """``u = e^y (1 + (x-1) + (x-1)^2)`` left, ``e^y (1 + (x-1)/3 - (x-1)^2)`` right."""
    p = jnp.asarray(points, dtype=jnp.float64)
    s = p[..., 0] - INTERFACE_X
    left = 1.0 + s + s**2
    right = 1.0 + (K_MINUS / K_PLUS) * s - s**2
    return jnp.exp(p[..., 1]) * jnp.where(s <= 0.0, left, right)


def side_source(side: Side, /) -> PointField:
    """``f = -k Laplace(u)`` of one material, including on the interface itself.

    Points on ``x = 1`` belong to both halves; a cloud collocates its own
    material's PDE there, so the source is selected by side, not by position.
    """

    def evaluate(points: ArrayLike) -> Array:
        p = jnp.asarray(points, dtype=jnp.float64)
        s = p[..., 0] - INTERFACE_X
        match side:
            case "left":
                return -K_MINUS * jnp.exp(p[..., 1]) * (1.0 + s + s**2 + 2.0)
            case "right":
                return (
                    -K_PLUS
                    * jnp.exp(p[..., 1])
                    * (1.0 + (K_MINUS / K_PLUS) * s - s**2 - 2.0)
                )
            case _:
                assert_never(side)

    return evaluate


def source(points: ArrayLike) -> Array:
    """``f = -k Laplace(u)`` of the material at each point off the interface."""
    p = jnp.asarray(points, dtype=jnp.float64)
    left, right = side_source("left")(p), side_source("right")(p)
    return jnp.where(p[..., 0] <= INTERFACE_X, left, right)


# --- Finite-element region ---------------------------------------------------------------


def conductivity(side: Side, /) -> float:
    """The material of the left (minus) or right half of the plate."""
    match side:
        case "left":
            return K_MINUS
        case "right":
            return K_PLUS
        case _:
            assert_never(side)


def origin(side: Side, /) -> float:
    return 0.0 if side == "left" else INTERFACE_X


def triangle_mesh(resolution: int, side: Side, /) -> phx.discretization.CellMesh:
    axis = np.linspace(0.0, 1.0, resolution + 1)
    points = np.stack(np.meshgrid(axis, axis, indexing="xy"), axis=-1).reshape(-1, 2)
    points[:, 0] += origin(side)
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


def on_interface(points: np.ndarray, /) -> np.ndarray:
    return (
        np.isclose(points[:, 0], INTERFACE_X)
        & (points[:, 1] > 1.0e-12)
        & (points[:, 1] < 1.0 - 1.0e-12)
    )


def interface_facets(
    space: phx.discretization.FiniteElementDiscretization,
) -> IntegrationDomain:
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    facets = np.all(np.isclose(np.asarray(probe.sites)[..., 0], INTERFACE_X), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[facets]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


def finite_element_region(
    resolution: int, /, *, side: Side = "left"
) -> tuple[VariationalComponent, IntegrationDomain, np.ndarray]:
    space = phx.discretization.FiniteElementPlan(
        triangle_mesh(resolution, side),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    dirichlet = np.asarray(space.dof_maps[0].boundary_dof_mask) & ~on_interface(dofs)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "poisson",
            "u",
            (
                phx.equations.DiffusionAction("u", conductivity(side)),
                phx.equations.SourceAction(
                    "u",
                    phx.equations.coefficient(
                        lambda points, args: source(points), coefficient_id="f"
                    ),
                ),
            ),
        ),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=dirichlet
        ),
        dirichlet_values=lambda points: np.asarray(exact(points)),
    )
    return (
        VariationalComponent("triangles", problem, field="u"),
        interface_facets(space),
        dofs,
    )


# --- Chart-authorized point cloud ------------------------------------------------------


def point_region(
    resolution: int,
    /,
    *,
    side: Side = "right",
    value: PointField = exact,
    load: PointField | None = None,
    kappa: float | None = None,
) -> tuple[MeshfreeTraceComponent, np.ndarray]:
    """Collocated cloud on one half of the plate with a chart-authorized interface.

    The cloud solves ``-div(kappa grad u) = load`` with Dirichlet data ``value``
    off the interface; ``kappa`` and ``load`` default to the half's material.
    Interface rows use boundary ghosts (``PointGhostLayerPlan``): each interface
    point collocates the PDE and its conormal condition, which keeps square
    collocation spectrally stable.
    """
    h = 1.0 / resolution
    axis = np.linspace(0.0, 1.0, resolution + 1)
    x0 = origin(side)
    x, y = np.meshgrid(x0 + axis, axis, indexing="ij")
    points = np.stack((x.ravel(), y.ravel()), axis=-1)
    count = points.shape[0]
    interior_weight = np.where((axis > 0) & (axis < 1), h, 0.5 * h)
    volumes = np.outer(interior_weight, interior_weight).ravel()
    low, high = np.isclose(points[:, 0], x0), np.isclose(points[:, 0], x0 + 1.0)
    at_interface = np.isclose(points[:, 0], INTERFACE_X)
    bottom, top = np.isclose(points[:, 1], 0.0), np.isclose(points[:, 1], 1.0)
    boundary = low | high | bottom | top
    normals = np.zeros_like(points)
    normals[low], normals[high] = (-1.0, 0.0), (1.0, 0.0)
    normals[bottom], normals[top] = (0.0, -1.0), (0.0, 1.0)
    # The half-plate polygon (counter-clockwise): its edges on x = 1 run between
    # consecutive cloud points, upward on the left half and downward on the right.
    match side:
        case "left":
            rising = np.stack((np.full(resolution, INTERFACE_X), axis[1:]), axis=1)
            vertices = np.concatenate(
                ([[0.0, 0.0], [INTERFACE_X, 0.0]], rising, [[0.0, 1.0]])
            )
            interface_charts = tuple(range(1, 1 + resolution))
        case "right":
            falling = np.stack(
                (np.full(resolution, INTERFACE_X), axis[::-1][:-1]), axis=1
            )
            vertices = np.concatenate(
                ([[INTERFACE_X, 0.0], [2.0, 0.0], [2.0, 1.0]], falling)
            )
            interface_charts = tuple(range(3, 3 + resolution))
        case _:
            assert_never(side)
    atlas = phx.geometry.Polygon([tuple(v) for v in vertices]).compile().boundary_atlas
    cloud = PointCloudPlan(
        points,
        volumes,
        boundary_mask=boundary,
        boundary_normals=normals,
        stencil=LocalStencilPolicy(approximation="phs-rbf-fd", polynomial_degree=3),
        neighbors=30,
    ).prepare()
    charts = PointBoundaryCharts(
        cloud,
        atlas,
        FacetTraceRule("gauss-lobatto-legendre", points=2),
        charts=interface_charts,
    )
    measure = np.asarray(charts.measure)
    flux_rows = np.flatnonzero(at_interface & ~bottom & ~top)
    dirichlet_rows = np.flatnonzero(boundary & ~np.isin(np.arange(count), flux_rows))
    callback_points = jnp.asarray(points, dtype=jnp.float64)
    values = np.asarray(value(callback_points))
    plan = PointBoundaryPlan(
        (
            PointBoundaryCondition(
                "dirichlet", dirichlet_rows, values[dirichlet_rows], label="exterior"
            ),
            PointBoundaryCondition(
                "neumann",
                flux_rows,
                0.0,
                label="interface",
                normals=normals[flux_rows],
                measure=measure[flux_rows],
            ),
        ),
        row_count=count,
    )
    ghosts = PointGhostLayerPlan(plan).prepare(cloud)
    poisson = PointCloudPoissonPlan(cloud, plan, ghosts=ghosts).prepare(
        conductivity(side) if kappa is None else kappa
    )
    reconstruction = prepare_point_cloud_field_reconstruction(
        cloud,
        support_geometry=phx.geometry.Rectangle((x0 + 0.5, 0.5), (1.0, 1.0)).compile(),
        radius=3.5 * h,
        capacity=36,
    )
    component = MeshfreeTraceComponent(
        poisson,
        reconstruction,
        (MeshfreeBoundaryTrace("u", "interface", charts),),
        name="cloud",
        source=np.asarray((side_source(side) if load is None else load)(callback_points)),
    )
    return component, points


# --- Interface binding and laws --------------------------------------------------------


def plate_interface(
    minus: tuple[str, str], plus: tuple[str, str], /
) -> tuple[InterfaceBinding, phx.domain.SubdomainCover]:
    """Two-sided binding of the cut ``x = 1``; ``minus`` owns the left half."""
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(1))
    endpoints = tuple(
        InterfaceEndpoint(
            component,
            PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
            fields={"value": field},
        )
        for (component, field), patch in (
            (minus, pairing.left_patch_id),
            (plus, pairing.right_patch_id),
        )
    )
    binding = InterfaceBinding(
        "cut",
        InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        endpoints,
    )
    return binding, cover


def prepare_level(
    resolution: int, imposition: TransmissionImposition, /, *, cloud: Side = "right"
) -> tuple[PreparedCoupledProblem, np.ndarray, np.ndarray]:
    """The cloud on half ``cloud`` and triangles on the other; minus is the left half."""
    fe_half: Side = "left" if cloud == "right" else "right"
    triangles, fe_interface, dofs = finite_element_region(resolution, side=fe_half)
    points_component, points = point_region(resolution, side=cloud)
    sides = (
        TransmissionSide("triangles", "triangles", "u", fe_interface),
        TransmissionSide(
            "cloud", "cloud", "u", points_component.boundary_domain("u", "interface")
        ),
    )
    fields = (
        ("triangles", triangles.field_space_id("u")),
        ("cloud", points_component.field_space_id("u")),
    )
    order = (0, 1) if cloud == "right" else (1, 0)
    binding, cover = plate_interface(fields[order[0]], fields[order[1]])
    law = ScalarTransmissionLaw(
        "transmission", binding, (sides[order[0]], sides[order[1]]), imposition
    )
    plan = CoupledProblemPlan(
        "fe-meshfree-plate",
        components=(triangles, points_component),
        bindings=(binding,),
        laws=(law,),
    )
    return prepare_coupled_problem(plan, interface_owners=(cover,)), dofs, points


def errors(
    solution: CoupledSolution, dofs: np.ndarray, points: np.ndarray, /
) -> tuple[float, float]:
    left = np.asarray(solution.field("triangles", "u")) - np.asarray(exact(dofs))
    right = np.asarray(solution.field("cloud", "u")) - np.asarray(exact(points))
    return float(np.max(np.abs(left))), float(np.max(np.abs(right)))


def rates(values: list[float], /) -> list[str]:
    return ["-"] + [
        f"{np.log2(coarse / fine):.2f}" for coarse, fine in zip(values[:-1], values[1:])
    ]


def run_transmission(
    name: str,
    imposition: TransmissionImposition,
    levels: tuple[int, ...] = LEVELS,
    /,
    *,
    cloud: Side = "right",
) -> dict[str, list[float]]:
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=64_000_000, max_bytes=1024 * 1024 * 1024
        ),
    )
    history: dict[str, list[float]] = {"fe": [], "cloud": [], "jump": []}
    for resolution in levels:
        start = time.perf_counter()
        prepared, dofs, points = prepare_level(resolution, imposition, cloud=cloud)
        solution = solve_coupled_problem(prepared, policy=policy)
        elapsed = time.perf_counter() - start
        interface = solution.interface("transmission")
        if not bool(solution.accepted):
            raise RuntimeError(
                f"{name} level {resolution} refused: native "
                f"{bool(solution.native_successful)}, defects "
                f"{dict(zip(interface.names, np.asarray(interface.values)))}."
            )
        fe_error, cloud_error = errors(solution, dofs, points)
        lookup = dict(zip(interface.names, np.asarray(interface.values), strict=True))
        jump = float(lookup.get("trace-mismatch-l2", lookup.get("trace-jump-l2", 0.0)))
        history["fe"].append(fe_error)
        history["cloud"].append(cloud_error)
        history["jump"].append(jump)
        print(
            f"{name} n={resolution:3d}: fe max {fe_error:.3e}, cloud max "
            f"{cloud_error:.3e}, interface L2 jump {jump:.3e} ({elapsed:.1f} s)"
        )
    for key, values in history.items():
        print(f"  {key:<6} rates {rates(values)}")
    return history


def main() -> None:
    run_transmission(
        "mortar, cloud on the plus side",
        MortarImposition(MortarMultiplier("side-trace", side="triangles")),
    )
    run_transmission(
        "mortar, cloud on the minus side",
        MortarImposition(MortarMultiplier("side-trace", side="triangles")),
        cloud="left",
    )
    run_transmission(
        "one-sided Nitsche (FE flux and penalty)",
        NitscheImposition(penalty_factor=4.0, weights=(1.0, 0.0)),
    )


if __name__ == "__main__":
    main()
