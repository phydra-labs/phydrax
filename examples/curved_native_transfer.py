# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Refine a quadratic shear map and transfer a quadratic finite-element field.

Native bisection preserves the P2 coordinate map. A nested field transfer
carries u = y**2 + 0.1*x to the refined mesh. Run from the repository root::

    JAX_ENABLE_X64=1 python -m examples.curved_native_transfer

For learned marking, atomic rollback, PDE reanalysis, derivative checks, and
JSON records, run ``python -m tools.meshing_qualification --scenario curved-native-transfer``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx


def shear(points: Array) -> Array:
    return points.at[..., 0].add(0.1 * points[..., 1] ** 2)


def main() -> None:
    """Print refinement counts, mapped measure, and the transferred-field defect."""
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    mesh = phx.meshing.canonicalize_cell_mesh(
        phx.discretization.CellMesh.from_triangles(
            np.asarray(
                ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)), dtype=np.float64
            ),
            np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
        )
    )
    element = phx.discretization.coordinate_lagrange_element("triangle", 2)
    coordinate_space = phx.discretization.FiniteElementPlan(
        mesh,
        phx.discretization.FiniteElementFieldSpec("coordinate", element),
    ).prepare()
    coordinate_map = coordinate_space.dof_maps[0]
    geometry = phx.discretization.CellGeometrySpec(
        {mesh.blocks[0].name: element},
        {mesh.blocks[0].name: coordinate_map.cell_dofs[0]},
        shear(coordinate_map.dof_coordinates),
    )
    source = phx.meshing.certify_cell_mesh(
        mesh.with_coordinates(shear(mesh.coordinates), numeric_version="quadratic-shear"),
        phx.SpatialCoordinateContract.si(),
        geometry=geometry,
    )
    adaptation = phx.meshing.execute_mesh_adaptation(
        phx.meshing.prepare_mesh_adaptation(
            source,
            phx.meshing.MarkedMeshAdaptation(
                np.asarray(source.mesh.blocks[0].global_ids)[:1]
            ),
            policy=phx.meshing.MeshAdaptationPolicy(
                phx.meshing.MeshAdaptationRoute.NATIVE_BISECTION
            ),
        )
    )
    transition = adaptation.geometry_transition
    parents = phx.solver.refinement_parent_cells(adaptation)
    if transition is None or parents is None or not adaptation.status.converged:
        raise RuntimeError(
            "adaptation.geometry must retain curved nested maps; got incomplete refinement"
        )
    field = phx.discretization.FiniteElementFieldSpec(
        "u",
        phx.discretization.lagrange_element("triangle", 2),
    )
    before = phx.discretization.FiniteElementPlan(
        source.mesh,
        field,
        coordinate_spec=source.geometry,
    ).prepare()
    after = phx.discretization.FiniteElementPlan(
        adaptation.target.mesh,
        field,
        coordinate_spec=adaptation.target.geometry,
    ).prepare()
    transfer = phx.discretization.prepare_nested_field_transfer(
        before,
        after,
        parents,
        field_name="u",
        parent_reference_vertices=adaptation.parent_reference_vertices,
        source_geometry=source.geometry,
        target_geometry=adaptation.target.geometry,
        geometry_transition=transition,
    )
    points = before.dof_maps[0].dof_coordinates
    transferred = transfer.transfer.apply(points[:, 1] ** 2 + 0.1 * points[:, 0])
    target_points = after.dof_maps[0].dof_coordinates
    expected = target_points[:, 1] ** 2 + 0.1 * target_points[:, 0]
    error = float(jnp.max(jnp.abs(transferred - expected)))
    print(
        f"cells: {source.mesh.blocks[0].cell_count} -> {adaptation.target.mesh.blocks[0].cell_count}"
    )
    print(f"mapped measure: {transition.evidence.target_measure:.6f}")
    print(f"transferred quadratic field maximum defect: {error:.3e}")


if __name__ == "__main__":
    main()
