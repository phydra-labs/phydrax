#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""A chain of identical Laplace strips solved with bounded lane worksets.

The band ``[0, N] x [0, 1]`` is split into ``N`` unit strips, each an
independently compiled P1 owner of ``-Delta u = 0`` with the harmonic
Dirichlet data ``u = sin(x / 2) exp(y / 2)`` on the outer boundary. Side-trace
mortars impose continuity and flux balance on the cuts ``x = 1, ..., N - 1``.

The same plan is prepared twice: per owner, and with a
``CoupledExecutionPolicy`` that groups the strips by executable signature into
bounded lanes. The example prints the groups and their resource estimates,
the agreement of residuals and certified solutions of both executions, the
difference to the single conforming owner on the union band (the same
discrete problem on matching meshes), and the nodal error against the exact
field. It raises when a solve is not accepted or the executions disagree.
"""

import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.discretization import EntitySelection, FacetTraceRule, IntegrationDomain


jax.config.update("jax_enable_x64", True)
cpl = phx.solver.coupling

STRIPS = 12
CELLS = 4
LANE_CAPACITY = 4


def exact(points: Array) -> Array:
    return jnp.sin(0.5 * points[..., 0]) * jnp.exp(0.5 * points[..., 1])


def no_source(points: Array, args: object) -> Array:
    del args
    return jnp.zeros(points.shape[:-1], dtype=points.dtype)


def band_mesh(x0: float, width: int, /) -> phx.discretization.CellMesh:
    columns = width * CELLS
    xs = np.linspace(x0, x0 + width, columns + 1)
    ys = np.linspace(0.0, 1.0, CELLS + 1)
    points = np.stack(np.meshgrid(xs, ys, indexing="xy"), -1).reshape(-1, 2)
    triangles = []
    for j in range(CELLS):
        for i in range(columns):
            a = j * (columns + 1) + i
            triangles += [
                (a, a + 1, a + columns + 2),
                (a, a + columns + 2, a + columns + 1),
            ]
    return phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "cells", "triangle", np.asarray(triangles, dtype=np.int32)
            ),
        ),
    )


type Owner = tuple[
    phx.equations.CompiledFiniteElementProblem, dict[float, IntegrationDomain], np.ndarray
]


def laplace_owner(x0: float, width: int, cuts: tuple[float, ...], /) -> Owner:
    """P1 owner on a band; the interface rows of ``cuts`` stay free."""
    d = phx.discretization
    space = d.FiniteElementPlan(
        band_mesh(x0, width),
        d.FiniteElementFieldSpec("u", d.lagrange_element("triangle", 1)),
    ).prepare()
    points = np.asarray(space.dof_maps[0].dof_coordinates)
    dirichlet = np.array(space.dof_maps[0].boundary_dof_mask, dtype=np.bool_)
    exterior = space.exterior_facet_domain
    probe = space.prepare_side_trace("u", exterior, rule=FacetTraceRule(points=2))
    sites = np.asarray(probe.sites)[..., 0]
    edges = space.mesh.topology.entity_sets[1]
    domains: dict[float, IntegrationDomain] = {}
    for x in cuts:
        dirichlet &= ~(
            np.isclose(points[:, 0], x)
            & (points[:, 1] > 1e-12)
            & (points[:, 1] < 1 - 1e-12)
        )
        mask = np.zeros((edges.count,), dtype=np.bool_)
        mask[
            np.asarray(exterior.entity_indices)[np.all(np.isclose(sites, x), axis=1)]
        ] = True
        domains[x] = space.integration_domain(
            "exterior_facet", EntitySelection(edges, mask)
        )
    problem = phx.equations.compile_finite_element_problem(
        phx.equations.FiniteElementForm(
            "laplace",
            "u",
            (
                phx.equations.DiffusionAction("u"),
                phx.equations.SourceAction(
                    "u", phx.equations.coefficient(no_source, coefficient_id="f")
                ),
            ),
        ),
        space,
        constraint=d.dirichlet_constraint(space, "u", boundary_mask=dirichlet),
        dirichlet_values=exact,
    )
    return problem, domains, points


def strip_plan(
    execution: cpl.CoupledExecutionPolicy | None, /
) -> tuple[cpl.CoupledProblemPlan, phx.domain.SubdomainCover, dict[str, np.ndarray]]:
    plate = phx.domain.HyperRectangle(np.zeros(2), np.asarray([float(STRIPS), 1.0]))
    cover = phx.domain.cartesian_subdomain_cover(
        plate, "x", (STRIPS, 1), cover_id="strips"
    )
    strips = []
    coordinates: dict[str, np.ndarray] = {}
    for k in range(STRIPS):
        cuts = tuple(x for x in (float(k), float(k + 1)) if 0.0 < x < STRIPS)
        problem, domains, points = laplace_owner(float(k), 1, cuts)
        component = cpl.VariationalComponent(f"strip{k:02d}", problem, field="u")
        strips.append((component, domains))
        coordinates[component.name] = points
    bindings, laws = [], []
    for k, pairing in enumerate(cover.pairings):
        witness = pairing.component.sample(
            phx.domain.PointSampling(8), key=jax.random.key(1)
        )
        (minus, minus_cuts), (plus, plus_cuts) = strips[k], strips[k + 1]
        binding = cpl.InterfaceBinding(
            f"cut{k:02d}",
            cpl.InterfaceSource.paired_support(cover, pairing.pairing_id),
            "two-sided",
            tuple(
                cpl.InterfaceEndpoint(
                    role,
                    cpl.PairedSupportAttachment(
                        cover, pairing.pairing_id, patch, witness
                    ),
                    fields={"value": owner.field_space_id("u")},
                )
                for role, patch, owner in (
                    ("minus", pairing.left_patch_id, minus),
                    ("plus", pairing.right_patch_id, plus),
                )
            ),
        )
        bindings.append(binding)
        laws.append(
            cpl.ScalarTransmissionLaw(
                f"law{k:02d}",
                binding,
                (
                    cpl.TransmissionSide("minus", minus.name, "u", minus_cuts[k + 1.0]),
                    cpl.TransmissionSide("plus", plus.name, "u", plus_cuts[k + 1.0]),
                ),
                cpl.MortarImposition(cpl.MortarMultiplier("side-trace", side="plus")),
            )
        )
    plan = cpl.CoupledProblemPlan(
        "strip-chain",
        components=tuple(component for component, _ in strips),
        bindings=tuple(bindings),
        laws=tuple(laws),
        resources=cpl.CoupledResourcePolicy(execution=execution),
    )
    return plan, cover, coordinates


def conforming_values() -> dict[tuple[float, float], float]:
    """Native solve of the single conforming owner on the union band."""
    problem, _, points = laplace_owner(0.0, STRIPS, ())
    system, rhs = problem.linear_system()
    result = phx.linalg.solve(
        system, rhs, policy=phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    )
    if not bool(result.successful):
        raise RuntimeError("The conforming owner's native solve failed.")
    full = np.asarray(problem.expand(result.value, None))
    return {
        (round(float(x), 10), round(float(y), 10)): float(value)
        for (x, y), value in zip(points, full, strict=True)
    }


def main() -> None:
    policy = phx.linalg.LinearSolvePolicy(phx.linalg.DenseLU())
    prepared = {}
    coordinates: dict[str, np.ndarray] = {}
    for label, execution in (
        ("per-owner", None),
        ("lanes", cpl.CoupledExecutionPolicy(lane_capacity=LANE_CAPACITY)),
    ):
        plan, cover, coordinates = strip_plan(execution)
        started = time.perf_counter()
        prepared[label] = cpl.prepare_coupled_problem(plan, interface_owners=(cover,))
        print(f"{label}: prepared in {time.perf_counter() - started:.2f} s")
    reference, lanes = prepared["per-owner"], prepared["lanes"]
    worksets = lanes.worksets
    if worksets is None:
        raise RuntimeError("Lane execution prepares worksets.")
    for group in (*worksets.components, *worksets.interfaces):
        estimate = group.estimate
        print(
            f"{group.kind} workset: {len(group.members)} members, "
            f"{estimate.bucket_count} bucket(s) x {estimate.lane_capacity} lanes "
            f"({estimate.padded_lanes} padded), working set "
            f"{estimate.working_set_bytes / 2**20:.2f} MiB"
        )
    print(
        f"ungrouped components: {STRIPS - len(worksets.grouped_components)}, "
        f"lane data {worksets.lane_data_bytes / 2**20:.2f} MiB"
    )

    state = reference.state_space.unflatten(
        jnp.asarray(np.random.default_rng(0).normal(size=(reference.state_space.size,)))
    )
    residual_gap = float(
        jnp.max(
            jnp.abs(
                reference.row_space.flatten(reference.residual(state))
                - lanes.row_space.flatten(lanes.residual(state))
            )
        )
    )
    solutions = {
        label: cpl.solve_coupled_problem(problem, policy=policy)
        for label, problem in prepared.items()
    }
    if not all(bool(solution.accepted) for solution in solutions.values()):
        raise RuntimeError("A certified solve was not accepted.")
    solution_gap = float(
        jnp.max(
            jnp.abs(
                reference.state_space.flatten(solutions["per-owner"].state)
                - lanes.state_space.flatten(solutions["lanes"].state)
            )
        )
    )
    conforming = conforming_values()
    conforming_gap = 0.0
    nodal_error = 0.0
    for name, points in coordinates.items():
        field = np.asarray(solutions["lanes"].field(name, "u"))
        native = np.asarray(
            [conforming[(round(float(x), 10), round(float(y), 10))] for x, y in points]
        )
        conforming_gap = max(conforming_gap, float(np.max(np.abs(field - native))))
        nodal_error = max(
            nodal_error,
            float(np.max(np.abs(field - np.asarray(exact(jnp.asarray(points)))))),
        )
    print(f"residual difference lanes vs per-owner: {residual_gap:.2e}")
    print(f"solution difference lanes vs per-owner: {solution_gap:.2e}")
    print(f"difference to the conforming owner on the union band: {conforming_gap:.2e}")
    print(f"max nodal error against the exact field: {nodal_error:.2e}")
    if max(residual_gap, solution_gap, conforming_gap) > 1e-9:
        raise RuntimeError("Lane execution disagrees with its references.")


if __name__ == "__main__":
    main()
