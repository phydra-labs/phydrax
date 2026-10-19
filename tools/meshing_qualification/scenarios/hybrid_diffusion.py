# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import phydrax as phx
from tools.meshing_qualification._controls import QualificationControls
from tools.meshing_qualification._runtime import qualification_failure
from tools.meshing_qualification.scenarios.layout import _hybrid_gmsh_result
from tools.meshing_qualification.scenarios.providers import _mesh_scope


def qualify_hybrid_diffusion() -> dict[str, object]:

    with TemporaryDirectory(prefix="phydrax-hybrid-diffusion-") as temporary:
        generated, _ = _hybrid_gmsh_result(Path(temporary) / "hybrid-slab.brep")
    mesh = generated.mesh
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        {
            block.name: phx.discretization.lagrange_element(
                block.cell_kind,
                1,
            )
            for block in mesh.blocks
        },
    )
    discretization = phx.discretization.FiniteElementPlan(mesh, field).prepare()
    cells = mesh.entity_set(3)
    scope = _mesh_scope(mesh, 3, np.asarray(cells.entity_ids))
    domain = discretization.integration_domain(
        "cell", phx.meshing.resolve_mesh_scope(mesh, scope)
    )
    form = phx.equations.FiniteElementForm(
        "hybrid-diffusion",
        "u",
        (phx.equations.DiffusionAction("u", 1.0, domain=domain),),
    )
    compiled = phx.equations.compile_finite_element_problem(form, discretization)
    system, rhs = compiled.linear_system()
    dof_count = discretization.dof_maps[0].global_dof_count
    constant = np.ones(dof_count)
    null_residual = float(np.max(np.abs(np.asarray(system.operator.mv(constant)))))
    coordinates = np.asarray(discretization.vertices)
    values = coordinates[:, 0]
    energy = float(np.vdot(values, np.asarray(system.operator.mv(values))))
    rhs_norm = float(np.max(np.abs(np.asarray(rhs))))
    if not (
        np.isclose(null_residual, 0.0, atol=1.0e-12)
        and np.isclose(rhs_norm, 0.0, atol=1.0e-12)
        and np.isclose(energy, 1.0, atol=1.0e-10)
    ):
        raise qualification_failure(
            "qualify_hybrid_diffusion",
            ("Hybrid diffusion lost its P1 null mode or volume energy."),
            {"energy": energy, "null_residual": null_residual, "rhs_norm": rhs_norm},
        )
    return {
        "mesh_id": mesh.mesh_id,
        "scope": scope.scope_id,
        "dofs": dof_count,
        "null_residual": null_residual,
        "rhs_norm": rhs_norm,
        "x_gradient_energy": energy,
        "generated_result_id": generated.result_id,
    }


def run(args: QualificationControls) -> dict[str, object]:
    """Execute this scenario with host controls."""
    return qualify_hybrid_diffusion()
