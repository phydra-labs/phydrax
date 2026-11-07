# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Glue two transfinite blocks, certify the source, and solve a manufactured FE problem.

Set PHYDRAX_MESHCORE_LIBRARY to the intended native library and JAX_ENABLE_X64=true.
"""

import json
import os
from typing import cast

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from phydrax.geometry import ParametricCurveBoundarySource, PiecewiseLinearDomain
from phydrax.geometry.brep import LineCurve
from phydrax.meshing import (
    BlockInterfaceControl,
    CellFamilyPolicy,
    CellMeshingTarget,
    MeshingScope,
    SizeControlStrength,
    SurfaceMeshingSpec,
    TransfiniteBlock,
    TransfiniteCurveControl,
    UniformSizeControl,
)
from phydrax.meshing.providers import (
    NativeMeshingOptions,
    NativeMeshingProvider,
    NativeStructuredSource,
)


def rectangle(name: str, left: float, right: float, resolution: int) -> TransfiniteBlock:
    edges = (
        ("left", (left, 0.0), (0.0, 1.0)),
        ("right", (right, 0.0), (0.0, 1.0)),
        ("bottom", (left, 0.0), (right - left, 0.0)),
        ("top", (left, 1.0), (right - left, 0.0)),
    )
    return TransfiniteBlock(
        name,
        tuple(
            TransfiniteCurveControl(
                LineCurve(origin, direction), f"{name}:{edge}", (0.0, 1.0), resolution
            )
            for edge, origin, direction in edges
        ),
    )


def poisson_error(resolution: int) -> tuple[float, dict[str, object]]:
    # The PWL boundary and the transfinite charts describe the same segmented
    # chain, including the block junctions and every boundary mesh vertex.
    boundary_points = np.asarray(
        [(i / (2 * resolution), 0.0) for i in range(2 * resolution)]
        + [(1.0, i / resolution) for i in range(resolution)]
        + [(1.0 - i / (2 * resolution), 1.0) for i in range(2 * resolution)]
        + [(0.0, 1.0 - i / resolution) for i in range(resolution)],
        dtype=np.float64,
    )
    count = len(boundary_points)
    boundary_edges = np.column_stack(
        (np.arange(count, dtype=np.int64), np.roll(np.arange(count, dtype=np.int64), -1))
    )
    domain = PiecewiseLinearDomain(
        boundary_points,
        boundary_edges,
        np.tile(np.asarray((0, -1), dtype=np.int64), (count, 1)),
        ("square",),
        source_id="unit-square",
    )
    blocks = (
        rectangle("left", 0.0, 0.5, resolution),
        rectangle("right", 0.5, 1.0, resolution),
    )
    interfaces = (BlockInterfaceControl("left", 1, "right", 0, (0,), (False,)),)
    exterior_controls = tuple(
        control
        for block in blocks
        for face, control in enumerate(block.boundaries)
        if (block.name, face) not in {("left", 1), ("right", 0)}
    )
    if not all(
        isinstance(control, TransfiniteCurveControl) for control in exterior_controls
    ):
        raise TypeError("Planar block boundaries must be curve controls.")
    boundary_controls = cast(tuple[TransfiniteCurveControl, ...], exterior_controls)
    boundary = ParametricCurveBoundarySource(
        tuple(c.curve for c in boundary_controls),
        tuple(c.parameter_range for c in boundary_controls),
        source_id=domain.source_id,
        source_revision=domain.source_revision,
        curve_entity_ids=tuple(c.source_id for c in boundary_controls),
        occurrence_paths=((),) * len(boundary_controls),
        chart_ids=tuple(c.source_id for c in boundary_controls),
    )
    source = NativeStructuredSource(
        blocks,
        interfaces,
        domain.source_id,
        domain.source_revision,
        fidelity_source=boundary,
        maximum_deviation=0.08,
        domain=domain,
    )
    scope = MeshingScope.whole(source, 2)
    spec = SurfaceMeshingSpec(
        CellMeshingTarget(2, 2, CellFamilyPolicy(required=("quadrilateral",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
        ),
        size_controls=(
            UniformSizeControl(
                scope, 1.0 / resolution, strength=SizeControlStrength.SOFT
            ),
        ),
    )
    plan = NativeMeshingProvider(NativeMeshingOptions("structured_transfinite")).plan(
        source, spec, coordinate_contract=phx.SpatialCoordinateContract.si()
    )
    if plan.route != "structured_transfinite":
        raise RuntimeError("Unexpected native route")
    result = plan.execute()
    kinds = {b.cell_kind for b in result.mesh.blocks if b.cell_count}
    measures = np.asarray(result.quality.evaluation.measures, dtype=np.float64)
    if (
        kinds != {"quadrilateral"}
        or not np.all(measures > 0)
        or abs(float(measures.sum()) - 1.0) > 1e-10
    ):
        raise RuntimeError("Multiblock family or area evidence failed")
    if (
        not result.audit.passed
        or result.certification is None
        or not result.certification.passed
        or result.certification.inputs.domain is not domain
    ):
        raise RuntimeError("Missing original source certification")
    interface = next(
        (
            label
            for label in result.labels
            if label.name == f"structured-interface:{interfaces[0].control_id}"
        ),
        None,
    )
    if interface is None or len(interface.scope.entity_ids) != resolution:
        raise RuntimeError("Native output omitted the declared shared interface")
    connectivity = result.mesh.connectivity
    if not isinstance(connectivity, phx.discretization.PolygonalConnectivity):
        raise TypeError("Multiblock output requires polygonal edge connectivity.")
    edge_ids = np.asarray(result.mesh.entity_set(1).entity_ids, dtype=np.int64)
    interface_rows = np.flatnonzero(np.isin(edge_ids, interface.scope.entity_ids))
    vertices = np.unique(np.asarray(connectivity.edges, dtype=np.int64)[interface_rows])
    if (
        len(interface_rows) != resolution
        or len(vertices) != resolution + 1
        or np.any(np.asarray(connectivity.edge_cell_counts)[interface_rows] != 2)
    ):
        raise RuntimeError("The blocks do not share one conforming interface.")
    field = phx.discretization.FiniteElementFieldSpec(
        "u", phx.discretization.lagrange_element("quadrilateral", 1)
    )
    space = phx.discretization.FiniteElementPlan(
        result.mesh, field, coordinate_spec=result.geometry
    ).prepare()
    constraint = phx.discretization.dirichlet_constraint(space, "u")

    def source_term(points: Array, _args: object) -> Array:
        x, y = points[..., 0], points[..., 1]
        return 2.0 * (x * (1.0 - x) + y * (1.0 - y))

    form = phx.equations.FiniteElementForm(
        "multiblock-poisson",
        "u",
        (
            phx.equations.DiffusionAction("u", 1.0),
            phx.equations.SourceAction(
                "u",
                phx.equations.coefficient(
                    source_term, coefficient_id="manufactured-source"
                ),
            ),
        ),
    )
    problem = phx.equations.compile_finite_element_problem(
        form, space, constraint=constraint, dirichlet_values=0.0
    )
    operator, rhs = problem.linear_system()
    solved = phx.linalg.solve(operator, rhs)
    values = np.asarray(problem.expand(solved.value), dtype=np.float64)
    points = np.asarray(space.dof_maps[0].dof_coordinates, dtype=np.float64)
    x, y = points[:, 0], points[:, 1]
    error = float(np.max(np.abs(values - x * (1.0 - x) * y * (1.0 - y))))
    if not bool(jnp.all(solved.successful)):
        raise RuntimeError("Manufactured Poisson solve failed")
    return error, {
        "route": plan.route,
        "source_id": source.source_id,
        "source_revision": source.source_revision,
        "interface_id": interfaces[0].control_id,
        "interface_edges": len(interface.scope.entity_ids),
        "family": sorted(kinds),
        "cells": int(measures.size),
        "area": float(measures.sum()),
    }


def main() -> None:
    if not os.environ.get("PHYDRAX_MESHCORE_LIBRARY"):
        raise RuntimeError(
            "Set PHYDRAX_MESHCORE_LIBRARY to the native library before running."
        )
    coarse, evidence = poisson_error(4)
    fine, _ = poisson_error(8)
    if not 0.0 < fine < coarse / 3.0:
        raise RuntimeError(f"Manufactured Poisson convergence failed: {coarse}, {fine}")
    print(
        json.dumps(
            {
                **evidence,
                "coarse_error": coarse,
                "fine_error": fine,
                "refinement_ratio": coarse / fine,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
