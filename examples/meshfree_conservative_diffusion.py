# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Constrained graph diffusion, fixed-topology derivatives and upwind ledgers."""

from __future__ import annotations

import argparse
from typing import TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from jax import Array

from benchmarks._runtime import logical_array_bytes
from phydrax.discretization.meshfree import (
    edge_upwind_content,
    MeshfreeExteriorCalculusPlan,
    MeshfreeMetricPolicy,
    PreparedMeshfreeMetric,
)
from phydrax.sparse import EdgeRelation, linear_apply


WorkflowValue: TypeAlias = float | int | bool | str


def exterior_plan(
    *,
    size: int = 25,
    dimension: int = 2,
    seed: int = 0,
    maximum_pairs: int | None = None,
    maximum_symbolic_entries: int = 2_000_000,
) -> MeshfreeExteriorCalculusPlan:
    """All requested points are active in a center-first Cartesian sample.

    Nodes missing an axial neighbor are prescribed Dirichlet nodes. The domain
    is the union of their axis-aligned tensor control volumes, not a fictitious
    complete box. Seed changes translation, not polynomial accuracy or rank.
    """
    if dimension not in (1, 2, 3) or size < 3**dimension:
        raise ValueError(
            "The workflow needs dimension 1,2,3 and at least 3**dimension capacity."
        )
    width = max(3, int(np.ceil(size ** (1 / dimension))))
    if width % 2 == 0:
        width += 1
    while width**dimension < size:
        width += 2
    integers = np.stack(
        np.meshgrid(*(np.arange(width) for _ in range(dimension)), indexing="ij"), axis=-1
    ).reshape((-1, dimension))
    centered = integers - width // 2
    order = np.argsort(np.sum(centered**2, axis=1), kind="stable")
    integers = integers[order[:size]]
    lattice = integers / (width - 1)
    shift = np.random.default_rng(seed).normal(0.0, 0.001, dimension)
    points = lattice + shift
    spacing = 1 / (width - 1)
    boundary_axes = (integers == 0) | (integers == width - 1)
    volume = spacing**dimension * np.prod(np.where(boundary_axes, 0.5, 1.0), axis=1)
    nodes = {
        tuple(int(integers[index, axis]) for axis in range(dimension))
        for index in range(size)
    }
    boundary = np.zeros(size, dtype=np.bool_)
    for index in range(size):
        node = tuple(int(integers[index, axis]) for axis in range(dimension))
        for axis in range(dimension):
            for direction in (-1, 1):
                neighbor = list(node)
                neighbor[axis] += direction
                if tuple(neighbor) not in nodes:
                    boundary[index] = True
    return MeshfreeExteriorCalculusPlan(
        points,
        1.1 * spacing,
        2 * dimension * size if maximum_pairs is None else maximum_pairs,
        node_volumes=volume,
        dirichlet=boundary,
        metric_policy=MeshfreeMetricPolicy(
            maximum_symbolic_entries=maximum_symbolic_entries
        ),
    )


def _host_dilation_weights(plan: PreparedMeshfreeMetric, scale: float) -> np.ndarray:
    """Independent SciPy sparse minimum-norm oracle, retaining redundant rows."""
    relation = plan.constraint.relation
    if not isinstance(relation, EdgeRelation):
        raise TypeError(
            "Independent moment oracle requires native edge-list constraints."
        )
    rows = np.asarray(relation.target_indices)
    columns = np.asarray(relation.source_indices)
    degree = np.asarray(plan.row_degrees)[rows]
    phi = np.asarray(plan.prior)
    coefficients = (
        np.asarray(plan.constraint.coefficients)
        * scale**degree
        / np.asarray(plan.row_scaling)[rows]
    )
    design = sp.coo_matrix(
        (coefficients * np.sqrt(phi[columns]), (rows, columns)),
        shape=(plan.constraint.target.size, plan.constraint.source.size),
    ).tocsr()
    rhs = np.asarray(plan.rhs / plan.row_scaling)
    solution = spla.lsmr(
        design, rhs, atol=1e-13, btol=1e-13, maxiter=10 * sum(design.shape)
    )[0]
    return np.sqrt(phi) * solution


def run_workflow(
    *, size: int = 25, dimension: int = 2, seed: int = 0
) -> dict[str, WorkflowValue]:
    prepared = exterior_plan(size=size, dimension=dimension, seed=seed).prepare()
    metric = prepared.metric_result
    diffusion = prepared.diffusion()
    quadratic = jnp.sum(prepared.points**2, axis=1)
    laplacian = diffusion.mv(quadratic)
    quadratic_error = jnp.max(
        jnp.abs(laplacian[prepared.equation_indices] - 2 * dimension)
    )
    temperature = 1 + 0.1 * jnp.sin(jnp.sum(prepared.points, axis=1))
    relation = prepared.incidence.relation
    if not isinstance(relation, EdgeRelation):
        raise TypeError("Workflow geometry requires canonical native edge incidence.")
    offset = linear_apply(relation, prepared.incidence.coefficients, prepared.points)
    volume_flux = metric.weights * (offset @ jnp.ones((dimension,)) / dimension) * 0.1
    # This axial Cartesian fixture has degree <=2d and speed components 0.1/d.
    # Bound outgoing flow without duplicating the native CFL reduction; the
    # actual outgoing CFL and ledger are measured by edge_upwind_content.
    flow_bound = 0.2 * jnp.max(jnp.abs(metric.weights)) * jnp.max(prepared.lengths)
    dt = 0.5 * jnp.min(prepared.node_volumes) / flow_bound
    advection = edge_upwind_content(
        temperature,
        prepared.node_volumes,
        prepared.incidence,
        volume_flux,
        dt,
        metric_nonnegative=metric.nonnegative,
        metric_accepted=metric.accepted,
    )

    def dilated_weights(scale: Array) -> Array:
        return prepared.refresh(points=scale * prepared.points).metric_result.weights

    _, derivative = jax.jvp(dilated_weights, (jnp.asarray(1.0),), (jnp.asarray(1.0),))
    epsilon = 1e-5
    finite_difference = (
        _host_dilation_weights(prepared.metric_system, 1 + epsilon)
        - _host_dilation_weights(prepared.metric_system, 1 - epsilon)
    ) / (2 * epsilon)
    derivative_error = jnp.max(jnp.abs(derivative - finite_difference))
    # A one-sided node with offsets 1 and 2 has C=[[1,2],[1,4]],
    # b=[0,2]. y=[2,-1] gives C.T y=[1,0]>=0, b.T y=-2<0:
    # exact nonnegative feasibility is impossible, not merely a bad iteration.
    refused = (
        MeshfreeExteriorCalculusPlan(
            np.asarray([[0.0], [1.0], [2.0]]),
            2.1,
            3,
            node_volumes=np.ones(3),
            dirichlet=np.asarray([False, True, True]),
            metric_policy=MeshfreeMetricPolicy("nonnegative", maximum_steps=2048),
        )
        .prepare()
        .metric_result
    )
    return {
        "size": size,
        "active_size": prepared.points.shape[0],
        "dimension": dimension,
        "seed": seed,
        "domain": f"translated Cartesian control-volume union in R^{dimension}; positive tensor midpoint/trapezoid measures",
        "oracle_provenance": "analytic Delta(sum(x_i**2))=2d; independent SciPy sparse LSMR dilation derivative; exact Farkas witness",
        "retained_bytes": logical_array_bytes(prepared),
        "metric_status": int(np.asarray(metric.status)),
        "metric_accepted": bool(np.asarray(metric.accepted)),
        "moment_residual": float(np.asarray(jnp.max(jnp.abs(metric.moment_residual)))),
        "quadratic_laplacian_error": float(np.asarray(quadratic_error)),
        "conservation_residual": float(
            np.asarray(jnp.abs(diffusion.conservation_residual(temperature)))
        ),
        "advection_conservation_residual": float(
            np.asarray(jnp.abs(advection.conservation_residual))
        ),
        "cfl": float(np.asarray(advection.cfl)),
        "positivity_admitted": bool(np.asarray(advection.positivity_admitted)),
        "negative_weights": int(np.asarray(metric.negative_count)),
        "zero_weights": int(np.asarray(metric.zero_count)),
        "rank": int(np.asarray(metric.rank)),
        "rank_route": metric.rank_certificate.route,
        "redundant_constraints": int(np.asarray(metric.redundant_constraints)),
        "reduced_spd": bool(np.asarray(diffusion.evidence.spd)),
        "derivative_error": float(np.asarray(derivative_error)),
        "derivative_available": bool(np.asarray(metric.derivative_available)),
        "nonnegative_refused": not bool(np.asarray(refused.accepted)),
        "nonnegative_provider_status": int(np.asarray(refused.provider_status)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=25)
    parser.add_argument("--dimension", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print(run_workflow(size=args.size, dimension=args.dimension, seed=args.seed))


if __name__ == "__main__":
    main()
