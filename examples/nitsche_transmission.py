#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nitsche transmission with a certified penalty across a nonmatching interface.

The plate ``[0, 2] x [0, 1]`` is split at ``x = 1`` into two independently
meshed regions with diffusivities ``kappa = 1`` (left) and ``kappa = 4``
(right). Both solve ``-div(kappa grad u) = f`` for the manufactured field
``u = (1 + a (x - 1)) exp(y)`` with ``a = 2`` on the left and ``a = 1/2`` on
the right, which is continuous across the cut and balances the conormal flux
``kappa a exp(y)``. The same ``ScalarTransmissionLaw`` is imposed by a
``NitscheImposition``: each finite-element owner publishes its exact pointwise
flux and certifies its trace-inverse constants, and the penalty is the declared
factor times those constants.

The example prints, for P1 and P2 on 2:3 nonmatching refinements, nodal errors
and observed rates, the certified constants, penalty range, and coercivity
constant, and the interface certificate; compares the symmetric and
nonsymmetric variants; and couples a virtual-element right region one-sidedly
(flux weight on the finite-element side only). A symmetric factor at or below
one is refused at preparation (see the tests). It raises when a solve is not
accepted.
"""

import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain
from phydrax.solver.coupling import (
    CoupledProblemPlan,
    CoupledSolution,
    InterfaceBinding,
    InterfaceEndpoint,
    InterfaceSource,
    NitscheEvidence,
    NitscheImposition,
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
# (x0, x1, kappa, a) of the left and right regions.
REGIONS = {"left": (0.0, 1.0, 1.0, 2.0), "right": (1.0, 2.0, 4.0, 0.5)}


def exact(points: np.ndarray, region: str, /) -> np.ndarray:
    """Host reference ``(1 + a (x - 1)) exp(y)`` of one region."""
    slope = REGIONS[region][3]
    return (1.0 + slope * (points[..., 0] - 1.0)) * np.exp(points[..., 1])


def region_actions(
    region: str, /
) -> tuple[phx.equations.DiffusionAction, phx.equations.SourceAction]:
    """``-div(kappa grad u) = f`` with ``f = -kappa u`` (since ``Delta u = u``)."""
    kappa, slope = REGIONS[region][2], REGIONS[region][3]

    def source(points: Array, args: object) -> Array:
        del args
        return -kappa * (1.0 + slope * (points[..., 0] - 1.0)) * jnp.exp(points[..., 1])

    return (
        phx.equations.DiffusionAction("u", kappa),
        phx.equations.SourceAction(
            "u", phx.equations.coefficient(source, coefficient_id=f"f-{region}")
        ),
    )


def grid(region: str, cells: int, /) -> np.ndarray:
    x0, x1 = REGIONS[region][:2]
    xs, ys = np.linspace(x0, x1, cells + 1), np.linspace(0.0, 1.0, cells + 1)
    return np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)


def quads(cells: int, /) -> list[tuple[int, int, int, int]]:
    return [
        (
            j * (cells + 1) + i,
            j * (cells + 1) + i + 1,
            (j + 1) * (cells + 1) + i + 1,
            (j + 1) * (cells + 1) + i,
        )
        for j in range(cells)
        for i in range(cells)
    ]


def interface_facets(
    space: phx.discretization.FiniteElementDiscretization
    | phx.discretization.VirtualElementDiscretization,
    /,
) -> IntegrationDomain:
    """The owner's exterior facets on ``x = 1``."""
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    on = np.all(np.isclose(np.asarray(probe.sites)[..., 0], INTERFACE_X), axis=1)
    edges = space.mesh.topology.entity_sets[1]
    mask = np.zeros((edges.count,), dtype=np.bool_)
    mask[np.asarray(exterior.entity_indices)[on]] = True
    return space.integration_domain("exterior_facet", EntitySelection(edges, mask))


def dirichlet_mask(boundary: np.ndarray, points: np.ndarray, /) -> np.ndarray:
    """Outer boundary rows; the open interface segment stays free."""
    inner = (
        np.isclose(points[:, 0], INTERFACE_X)
        & (points[:, 1] > 1.0e-12)
        & (points[:, 1] < 1.0 - 1.0e-12)
    )
    return boundary & ~inner


def finite_element_region(
    region: str, cells: int, degree: int, /
) -> tuple[VariationalComponent, IntegrationDomain, np.ndarray]:
    triangles = [t for a, b, c, d in quads(cells) for t in ((a, b, c), (a, c, d))]
    mesh = phx.discretization.CellMesh(
        grid(region, cells),
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )
    space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec(
            "u", phx.discretization.lagrange_element("triangle", degree)
        ),
    ).prepare()
    dofs = np.asarray(space.dof_maps[0].dof_coordinates)
    fixed = dirichlet_mask(np.asarray(space.dof_maps[0].boundary_dof_mask), dofs)
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm("poisson", "u", region_actions(region)),
        space,
        constraint=phx.discretization.dirichlet_constraint(
            space, "u", boundary_mask=fixed
        ),
        dirichlet_values=lambda points: exact(np.asarray(points), region),
    )
    return VariationalComponent(region, problem, field="u"), interface_facets(space), dofs


def virtual_element_region(
    region: str, cells: int, /
) -> tuple[VariationalComponent, IntegrationDomain, np.ndarray]:
    mesh = phx.discretization.CellMesh.from_polygons(
        jnp.asarray(grid(region, cells)),
        tuple(np.asarray(cell, dtype=np.int32) for cell in quads(cells)),
    )
    space = phx.discretization.VirtualElementPlan(
        mesh,
        phx.discretization.VirtualElementFieldSpec(
            "u", phx.discretization.conforming_h1_virtual_element(1)
        ),
    ).prepare()
    dofs = np.asarray(space.dof_map.default_dof_points)
    fixed = dirichlet_mask(np.asarray(mesh.connectivity.boundary_vertices), dofs)
    problem = phx.equations.compile_virtual_element_problem(
        phx.equations.VirtualElementForm("poisson", "u", region_actions(region)),
        space,
        constraint=phx.discretization.virtual_element_dirichlet_constraint(
            space, "u", boundary_mask=fixed
        ),
        dirichlet_values=lambda points: exact(np.asarray(points), region),
    )
    return VariationalComponent(region, problem, field="u"), interface_facets(space), dofs


def couple(
    left: tuple[VariationalComponent, IntegrationDomain, np.ndarray],
    right: tuple[VariationalComponent, IntegrationDomain, np.ndarray],
    imposition: NitscheImposition,
    /,
) -> PreparedCoupledProblem:
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([2.0, 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(plate, "x", (2, 1), cover_id="plate")
    pairing = cover.pairings[0]
    witness = pairing.component.sample(phx.domain.PointSampling(8), key=jr.key(1))
    endpoints = tuple(
        InterfaceEndpoint(
            name,
            PairedSupportAttachment(cover, pairing.pairing_id, patch, witness),
            fields={"value": component.field_space_id("u")},
        )
        for name, patch, component in (
            ("left", pairing.left_patch_id, left[0]),
            ("right", pairing.right_patch_id, right[0]),
        )
    )
    binding = InterfaceBinding(
        "cut",
        InterfaceSource.paired_support(cover, pairing.pairing_id),
        "two-sided",
        endpoints,
    )
    law = ScalarTransmissionLaw(
        "transmission",
        binding,
        (
            TransmissionSide("left", "left", "u", left[1]),
            TransmissionSide("right", "right", "u", right[1]),
        ),
        imposition,
    )
    plan = CoupledProblemPlan(
        "nitsche-plate", components=(left[0], right[0]), bindings=(binding,), laws=(law,)
    )
    return prepare_coupled_problem(plan, interface_owners=(cover,))


def solve(prepared: PreparedCoupledProblem, /) -> CoupledSolution:
    solution = solve_coupled_problem(
        prepared, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    if not bool(solution.accepted):
        raise RuntimeError("The coupled Nitsche solve was not accepted.")
    return solution


def nodal_error(
    solution: CoupledSolution, region: str, dofs: np.ndarray, rows: int, /
) -> float:
    values = np.asarray(solution.field(region, "u"))[:rows]
    return float(np.max(np.abs(values - exact(dofs[:rows], region))))


def evidence_of(prepared: PreparedCoupledProblem, /) -> NitscheEvidence:
    evidence = prepared.laws[0].evidence
    if not isinstance(evidence, NitscheEvidence):
        raise TypeError("The transmission law was not lowered through Nitsche.")
    return evidence


def report(prepared: PreparedCoupledProblem, solution: CoupledSolution, /) -> None:
    evidence = evidence_of(prepared)
    for role, stability in zip(("left", "right"), evidence.stability, strict=True):
        if stability is None:
            print(f"    {role}: zero flux weight, no stability evidence needed")
            continue
        constants = np.asarray(stability.constants)
        print(
            f"    {role}: certified C_F in [{constants.min():.4g}, {constants.max():.4g}] "
            f"on {constants.size} facets (max eigen-residual "
            f"{float(np.max(np.asarray(stability.relative_residuals))):.1e})"
        )
    print(
        f"    penalty in [{evidence.penalty_range[0]:.4g}, {evidence.penalty_range[1]:.4g}], "
        f"coercivity constant {evidence.coercivity_constant:.3f}, flux degrees "
        f"{evidence.flux_degrees}, quadrature degree {evidence.quadrature_exact_degree}"
    )
    interface = solution.interface("transmission")
    for name, gated, value, scale in zip(
        interface.names, interface.gated, interface.values, interface.scales, strict=True
    ):
        role = "gated" if gated else "evidence"
        print(f"    {name:<18} {float(value):.3e} (scale {float(scale):.3e}, {role})")


def convergence(degree: int, ladder: tuple[int, ...], /) -> None:
    print(f"P{degree} finite elements, symmetric Nitsche, penalty factor 2:")
    sizes, errors = [], []
    for cells in ladder:
        start = time.perf_counter()
        left = finite_element_region("left", cells, degree)
        right = finite_element_region("right", 3 * cells // 2, degree)
        prepared = couple(left, right, NitscheImposition(penalty_factor=2.0))
        solution = solve(prepared)
        error = max(
            nodal_error(solution, "left", left[2], left[2].shape[0]),
            nodal_error(solution, "right", right[2], right[2].shape[0]),
        )
        sizes.append(1.0 / cells)
        errors.append(error)
        print(
            f"  cells {cells}:{3 * cells // 2}  nodal error {error:.3e}  "
            f"({time.perf_counter() - start:.1f} s)"
        )
        report(prepared, solution)
    rates = np.diff(np.log(errors)) / np.diff(np.log(sizes))
    print(f"  observed rates {np.round(rates, 2).tolist()}")


def variants() -> None:
    print("P2 symmetric versus nonsymmetric on 3:4 cells:")
    left = finite_element_region("left", 3, 2)
    right = finite_element_region("right", 4, 2)
    for variant, factor in (("symmetric", 2.0), ("nonsymmetric", 0.5)):
        solution = solve(
            couple(left, right, NitscheImposition(variant, penalty_factor=factor))
        )
        error = max(
            nodal_error(solution, "left", left[2], left[2].shape[0]),
            nodal_error(solution, "right", right[2], right[2].shape[0]),
        )
        print(f"  {variant:<12} factor {factor}: nodal error {error:.3e}, accepted True")


def one_sided_virtual_elements() -> None:
    print("P1 finite elements (weight 1) next to VEM k = 1 (weight 0), 8:12 cells:")
    left = finite_element_region("left", 8, 1)
    right = virtual_element_region("right", 12)
    prepared = couple(
        left, right, NitscheImposition(penalty_factor=2.0, weights=(1.0, 0.0))
    )
    solution = solve(prepared)
    print(
        f"  nodal errors: triangles {nodal_error(solution, 'left', left[2], left[2].shape[0]):.3e}, "
        f"polygons {nodal_error(solution, 'right', right[2], right[2].shape[0]):.3e}"
    )
    report(prepared, solution)


if __name__ == "__main__":
    convergence(1, (4, 8, 16))
    convergence(2, (2, 4, 8))
    variants()
    one_sided_virtual_elements()
