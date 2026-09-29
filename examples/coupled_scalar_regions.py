#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One Poisson field solved across a finite-element and a virtual-element region.

The plate ``[0, 2] x [0, 1]`` is split at ``x = 1`` into two independently
discretized regions: P1 triangles on ``[0, 1] x [0, 1]`` and conforming H1
virtual elements (degree 1) on a perturbed brick mesh of hexagons and
quadrilaterals on ``[1, 2] x [0, 1]``. The interface vertices of the two
regions do not coincide. One ``ScalarTransmissionLaw`` states continuity of the
potential and balance of its conormal flux across the paired-support interface
and is imposed by a mortar whose multiplier is the virtual-element side trace.

Both regions solve ``-Delta u = f`` with the manufactured field
``u = sin(pi x / 2) exp(y)`` and Dirichlet data on the outer boundary. The
example prints, per refinement level, nodal errors of both regions against the
host-evaluated exact field and their observed rates, the law's interface
defects, the mortar's rank/inf-sup/coverage evidence, the component
certificates, and the native and certified status. It raises when a level is
not accepted.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.typing import ArrayLike

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from phydrax.solver.coupling import (
    CoupledProblemPlan,
    CoupledSolution,
    InterfaceBinding,
    InterfaceEndpoint,
    InterfaceSource,
    MortarEvidence,
    MortarImposition,
    MortarMultiplier,
    PairedSupportAttachment,
    prepare_coupled_problem,
    PreparedCoupledProblem,
    ScalarTransmissionLaw,
    solve_coupled_problem,
    TransmissionSide,
    VariationalComponent,
)


jax.config.update("jax_enable_x64", True)

INTERFACE_X = 1.0
LEVELS = (1, 2, 3)


def exact(points: ArrayLike) -> np.ndarray:
    """Host reference ``u = sin(pi x / 2) exp(y)``."""
    values = np.asarray(points, dtype=np.float64)
    return np.sin(0.5 * np.pi * values[..., 0]) * np.exp(values[..., 1])


def source(points: Array, args: object) -> Array:
    """``f = -Delta u = (pi^2 / 4 - 1) u``."""
    del args
    return (
        (0.25 * np.pi**2 - 1.0)
        * jnp.sin(0.5 * np.pi * points[..., 0])
        * jnp.exp(points[..., 1])
    )


def poisson_actions() -> tuple[phx.equations.DiffusionAction, phx.equations.SourceAction]:
    return (
        phx.equations.DiffusionAction("u"),
        phx.equations.SourceAction(
            "u", phx.equations.coefficient(source, coefficient_id="f")
        ),
    )


def on_interface(points: np.ndarray, /) -> np.ndarray:
    """Points strictly inside the interface segment (its end points stay Dirichlet)."""
    return (
        np.isclose(points[:, 0], INTERFACE_X)
        & (points[:, 1] > 1.0e-12)
        & (points[:, 1] < 1.0 - 1.0e-12)
    )


def interface_facets(
    space: phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization,
    /,
) -> IntegrationDomain:
    """The owner's exterior-facet domain restricted to its facets on ``x = 1``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    facets = np.all(np.isclose(np.asarray(probe.sites)[..., 0], INTERFACE_X), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[facets]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


# --- Finite-element region -------------------------------------------------------------


def triangle_mesh(resolution: int, /) -> phx.discretization.CellMesh:
    """Right-diagonal triangles on ``[0, 1] x [0, 1]``."""
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


def finite_element_region(
    resolution: int, /
) -> tuple[VariationalComponent, IntegrationDomain, np.ndarray]:
    space = phx.discretization.FiniteElementPlan(
        triangle_mesh(resolution),
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", 1)
        ),
    ).prepare()
    interface = interface_facets(space)
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    dirichlet = np.asarray(space.dof_maps[0].boundary_dof_mask) & ~on_interface(dofs)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("poisson", "u", poisson_actions()),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=dirichlet
        ),
        dirichlet_values=exact,
    )
    return VariationalComponent("triangles", problem, field="u"), interface, dofs


# --- Virtual-element region ------------------------------------------------------------


def brick_polygons(bricks: int, /) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """Distorted running-bond mesh of hexagons and quadrilaterals on ``[1, 2] x [0, 1]``.

    Every horizontal line carries ``2 * bricks + 1`` vertices; even rows hold
    ``bricks`` hexagons and odd rows are offset by half a brick (two boundary
    quadrilaterals and ``bricks - 1`` hexagons). Interior brick midpoints move
    vertically by 15% of a row, alternating by line, so no hexagon is a
    rectangle; a smooth map that keeps the four sides then slides the
    interface vertices along ``x = 1`` away from the triangles' vertices.
    """
    columns = 2 * bricks + 1
    x = np.linspace(0.0, 1.0, columns)
    y = np.linspace(0.0, 1.0, bricks + 1)
    grid_x, grid_y = np.meshgrid(x, y, indexing="xy")
    line = np.arange(bricks + 1)[:, None]
    midpoint = (np.arange(columns) % 2 == 1)[None, :]
    interior = (line > 0) & (line < bricks)
    grid_y = grid_y + np.where(midpoint & interior, 0.15 * (-1.0) ** line / bricks, 0.0)
    mapped_y = grid_y + 0.06 * np.sin(2.0 * np.pi * grid_y) * (1.0 - 0.5 * grid_x)
    mapped_x = grid_x + 0.05 * np.sin(np.pi * grid_x) * np.sin(np.pi * grid_y)
    points = np.stack((INTERFACE_X + mapped_x, mapped_y), axis=-1).reshape(-1, 2)

    def vertex(row: int, column: int) -> int:
        return row * columns + column

    cells: list[np.ndarray] = []
    for row in range(bricks):
        bounds = (
            [0, *range(1, columns - 1, 2), columns - 1]
            if row % 2
            else list(range(0, columns, 2))
        )
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True):
            lower = [vertex(row, column) for column in range(start, stop + 1)]
            upper = [vertex(row + 1, column) for column in range(stop, start - 1, -1)]
            cells.append(np.asarray(lower + upper, dtype=np.int32))
    return points, tuple(cells)


def virtual_element_region(
    bricks: int, /
) -> tuple[VariationalComponent, IntegrationDomain, np.ndarray, int]:
    points, cells = brick_polygons(bricks)
    mesh = phx.discretization.CellMesh.from_polygons(jnp.asarray(points), cells)
    space = phx.discretization.VirtualElementPlan(
        mesh,
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(1)
        ),
    ).prepare()
    dofs = np.asarray(space.dof_map.default_dof_points)
    boundary = np.asarray(mesh.connectivity.boundary_vertices)
    problem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm("poisson", "u", poisson_actions()),
        space,
        constraint=phx.discretization.virtual_element_dirichlet_constraint(
            space, "u", boundary_mask=boundary & ~on_interface(dofs)
        ),
        dirichlet_values=exact,
    )
    component = VariationalComponent("polygons", problem, field="u")
    return component, interface_facets(space), dofs, max(len(cell) for cell in cells)


# --- Interface binding and coupled problem ---------------------------------------------


def plate_interface(
    left_field: str, right_field: str, /
) -> tuple[InterfaceBinding, phx.domain.SubdomainCover]:
    """Two-sided binding of the analytic cut ``x = 1`` (minus: triangles)."""
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(1))
    binding = InterfaceBinding(
        "cut",
        InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        (
            InterfaceEndpoint(
                "triangles",
                PairedSupportAttachment(
                    cover, pairing.pairing_id, pairing.left_patch_id, witness
                ),
                fields={"value": left_field},
            ),
            InterfaceEndpoint(
                "polygons",
                PairedSupportAttachment(
                    cover, pairing.pairing_id, pairing.right_patch_id, witness
                ),
                fields={"value": right_field},
            ),
        ),
    )
    return binding, cover


def prepare_level(
    level: int, /
) -> tuple[PreparedCoupledProblem, np.ndarray, np.ndarray, int]:
    triangles, left_interface, left_dofs = finite_element_region(4 * 2**level)
    polygons, right_interface, right_dofs, arity = virtual_element_region(3 * 2**level)
    binding, cover = plate_interface(
        triangles.field_space_id("u"), polygons.field_space_id("u")
    )
    law = ScalarTransmissionLaw(
        "transmission",
        binding,
        (
            TransmissionSide("triangles", "triangles", "u", left_interface),
            TransmissionSide("polygons", "polygons", "u", right_interface),
        ),
        MortarImposition(MortarMultiplier("side-trace", side="polygons")),
    )
    plan = CoupledProblemPlan(
        "fe-vem-plate",
        components=(triangles, polygons),
        bindings=(binding,),
        laws=(law,),
    )
    prepared = prepare_coupled_problem(plan, interface_owners=(cover,))
    return prepared, left_dofs, right_dofs, arity


def mortar_evidence(prepared: PreparedCoupledProblem, /) -> MortarEvidence:
    evidence = prepared.laws[0].evidence
    if not isinstance(evidence, MortarEvidence):
        raise TypeError("The transmission law was not lowered through a mortar.")
    return evidence


def nodal_errors(
    solution: CoupledSolution, component: str, dofs: np.ndarray, /
) -> tuple[float, float]:
    error = np.asarray(solution.field(component, "u")) - exact(dofs)
    return float(np.max(np.abs(error))), float(np.sqrt(np.mean(error**2)))


def rates(errors: list[float], /) -> list[str]:
    return ["-"] + [
        f"{np.log2(coarse / fine):.2f}"
        for coarse, fine in zip(errors[:-1], errors[1:], strict=True)
    ]


def report_level(
    level: int,
    prepared: PreparedCoupledProblem,
    solution: CoupledSolution,
    timings: tuple[float, float],
    arity: int,
    /,
) -> None:
    evidence = mortar_evidence(prepared)
    coverage = evidence.coverage
    interface = solution.interface("transmission")
    linear = solution.linear
    print(
        f"level {level}: prepare {timings[0]:.2f} s, solve {timings[1]:.2f} s, "
        f"unknowns {prepared.state_space.size}, largest polygon {arity} vertices"
    )
    for name, gated, value, scale in zip(
        interface.names,
        interface.gated,
        np.asarray(interface.values),
        np.asarray(interface.scales),
        strict=True,
    ):
        role = "gated" if gated else "evidence"
        print(f"  interface {name:<18} {value:.3e} (scale {scale:.3e}, {role})")
    print(
        f"  mortar: multiplier {evidence.multiplier_family} on "
        f"{evidence.multiplier_side!r}, dimension {evidence.multiplier_dimension}, "
        f"numerical rank {evidence.numerical_rank}, sigma_min/sigma_max "
        f"{evidence.singular_value_min / evidence.singular_value_max:.3e}, "
        f"inf-sup {evidence.inf_sup:.3f}, quadrature degree "
        f"{evidence.quadrature_exact_degree}, trace degrees {evidence.trace_degrees}"
    )
    print(
        f"  coverage: measures {coverage.first_measure:.12f} / "
        f"{coverage.second_measure:.12f} / common {coverage.common_measure:.12f}, "
        f"max gap {coverage.maximum_gap:.1e}, max normal defect "
        f"{coverage.maximum_normal_defect:.1e}, segments {coverage.segment_count}"
    )
    for certificate in solution.components:
        print(
            f"  component {certificate.component:<9} residual "
            f"{float(certificate.residual_norm):.3e} (scale "
            f"{float(certificate.scale):.3e}) accepted {bool(certificate.accepted)}"
        )
    relative = (
        "n/a" if linear is None else f"{float(linear.diagnostics.relative_residual):.1e}"
    )
    print(
        f"  native successful {bool(solution.native_successful)} (relative residual "
        f"{relative}), accepted {bool(solution.accepted)}"
    )


def run(levels: tuple[int, ...] = LEVELS, /) -> dict[str, list[float]]:
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.DenseLU(),
        materialization=phx.linalg.MaterializationPolicy(
            max_entries=16_000_000, max_bytes=256 * 1024 * 1024
        ),
    )
    errors: dict[str, list[float]] = {
        "triangles max": [],
        "triangles rms": [],
        "polygons max": [],
        "polygons rms": [],
    }
    for level in levels:
        start = time.perf_counter()
        prepared, left_dofs, right_dofs, arity = prepare_level(level)
        prepared_at = time.perf_counter()
        solution = solve_coupled_problem(prepared, policy=policy)
        solved_at = time.perf_counter()
        report_level(
            level,
            prepared,
            solution,
            (prepared_at - start, solved_at - prepared_at),
            arity,
        )
        if not bool(solution.accepted):
            raise RuntimeError(
                f"Level {level} was not accepted: native "
                f"{bool(solution.native_successful)}, interface "
                f"{np.asarray(solution.interfaces[0].values)}."
            )
        for component, dofs in (("triangles", left_dofs), ("polygons", right_dofs)):
            maximum, rms = nodal_errors(solution, component, dofs)
            errors[f"{component} max"].append(maximum)
            errors[f"{component} rms"].append(rms)
    return errors


def print_convergence(errors: dict[str, list[float]], /) -> None:
    print("nodal errors against the exact field (h halves per level):")
    for name, values in errors.items():
        formatted = ", ".join(f"{value:.3e}" for value in values)
        print(f"  {name:<14} [{formatted}]  rates {rates(values)}")


if __name__ == "__main__":
    print_convergence(run())
