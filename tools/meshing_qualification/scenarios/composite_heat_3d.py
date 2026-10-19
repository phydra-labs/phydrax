# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import qualification_failure
from tools.meshing_qualification.scenarios.gmsh import (
    _patch_controls,
    _region_controls,
    _three_dimensional_partition,
)


def qualify_composite_heat_3d() -> dict[str, object]:

    with TemporaryDirectory(prefix="phydrax-composite-heat-3d-") as temporary:
        partition = _three_dimensional_partition(Path(temporary))
        provider = phx.meshing.GmshProvider()
        boundary_scope = provider.whole_scope(partition.model, 3)
        specification = phx.meshing.VolumeMeshingSpec(
            phx.meshing.CellMeshingTarget(
                3, 3, phx.meshing.CellFamilyPolicy(required=("tetrahedron",))
            ),
            boundary_scope,
            phx.meshing.VolumeFillStrategy.SIMPLEX,
            size_controls=(phx.meshing.UniformSizeControl(boundary_scope, 0.2),),
            region_controls=_region_controls(provider, partition),
            patch_controls=_patch_controls(provider, partition),
        )
        result = provider.plan(partition.model, specification).execute()
    if not result.audit.passed:
        raise qualification_failure(
            "qualify_composite_heat_3d",
            ("Composite three-dimensional heat mesh failed audit."),
            {"result": result},
        )
    zones = {
        zone.name: zone
        for zone in result.zones
        if zone.role is phx.meshing.MeshZoneRole.REGION
    }
    field = phx.discretization.FiniteElementFieldSpec(
        "temperature",
        phx.discretization.lagrange_element("tetrahedron", 1),
    )
    discretization = phx.discretization.FiniteElementPlan(
        result.mesh,
        field,
    ).prepare()
    domains = {
        name: discretization.integration_domain(
            "cell",
            phx.meshing.resolve_mesh_scope(result.mesh, zone.scope),
        )
        for name, zone in zones.items()
    }
    exterior_patches = tuple(
        patch for patch in result.patches if len(patch.adjacent_zone_ids) == 1
    )
    exterior_scope = exterior_patches[0].scope
    for patch in exterior_patches[1:]:
        exterior_scope = exterior_scope | patch.scope
    constraint = phx.discretization.dirichlet_constraint(
        discretization,
        "temperature",
        boundary_selection=phx.meshing.resolve_mesh_scope(
            result.mesh,
            exterior_scope,
        ),
    )
    form = phx.equations.FiniteElementForm(
        "two-region-volume-diffusion",
        "temperature",
        (
            phx.equations.DiffusionAction(
                "temperature",
                1.0,
                action_id="left-volume-conductivity",
                domain=domains["left"],
            ),
            phx.equations.DiffusionAction(
                "temperature",
                2.0,
                action_id="right-volume-conductivity",
                domain=domains["right"],
            ),
        ),
    )
    compiled = phx.equations.compile_finite_element_problem(
        form,
        discretization,
        constraint=constraint,
        dirichlet_values=lambda points: jnp.where(
            points[..., 0] <= 1.0,
            points[..., 0],
            1.0 + 0.5 * (points[..., 0] - 1.0),
        ),
    )
    system, right_hand_side = compiled.linear_system()
    solved = phx.linalg.solve(system, right_hand_side)
    solution = np.asarray(compiled.expand(solved.value))
    coordinates = np.asarray(result.mesh.coordinates)
    expected = np.where(
        coordinates[:, 0] <= 1.0,
        coordinates[:, 0],
        1.0 + 0.5 * (coordinates[:, 0] - 1.0),
    )
    error = float(np.max(np.abs(solution - expected)))
    if not bool(np.all(np.asarray(solved.successful))) or error > 1.0e-6:
        raise qualification_failure(
            "qualify_composite_heat_3d",
            ("Composite volume diffusion failed its interface solution."),
            {"error": error, "solved": solved},
        )
    return {
        "mesh_id": result.mesh.mesh_id,
        "cells": result.mesh.entity_set(3).count,
        "minimum_quality": float(result.quality.minimum_mean_ratio),
        "revision": partition.revision.revision_id,
        "association_graph": partition.association_graph.graph_id,
        "region_control_count": len(specification.region_controls),
        "patch_control_count": len(specification.patch_controls),
        "dofs": discretization.dof_maps[0].global_dof_count,
        "maximum_solution_error": error,
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_composite_heat_3d()
