# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Donor, fringe, and image profiles for sparse owner traffic scenarios."""

import jax
import jax.numpy as jnp
import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax._strict import StrictModule
from phydrax.discretization import CellBlock, CellMesh, PreparedFieldReconstruction
from phydrax.discretization.fem import (
    FiniteElementDiscretization,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)
from phydrax.discretization.finite_volume import (
    PiecewiseConstantReconstruction,
    prepare_finite_volume_field_reconstruction,
    UnstructuredFiniteVolumePlan,
)
from phydrax.meshing import certify_cell_mesh
from phydrax.meshing._assembly import MeshAssembly
from phydrax.meshing._mesh_part import MeshPart
from phydrax.meshing._overset import (
    OversetConnectivity,
    OversetPartSpec,
    OversetPolicy,
    prepare_overset_connectivity,
)


class OversetTrafficCase(StrictModule):
    target_mesh: CellMesh
    connectivity: OversetConnectivity
    image_connectivities: dict[str, OversetConnectivity]
    rotation: np.ndarray
    affine_matrix: np.ndarray
    fe: FiniteElementDiscretization
    fe_vector: FiniteElementDiscretization
    fv_owner: PreparedFieldReconstruction
    fe_values: jax.Array | None
    fe_vector_values: jax.Array | None
    fv_values: jax.Array | None


def build_traffic_case(rank: int) -> OversetTrafficCase:
    source_mesh = CellMesh(
        np.asarray(((0.0, 0.0), (4.0, 0.0), (0.0, 4.0), (4.0, 4.0)), dtype=np.float64),
        (
            CellBlock(
                "cells", "triangle", np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.int32)
            ),
        ),
    )
    target_mesh = CellMesh(
        np.asarray(((0.3, 0.3), (1.1, 0.3), (0.3, 1.1)), dtype=np.float64),
        (CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)),),
    )
    source_result = certify_cell_mesh(source_mesh, SpatialCoordinateContract.si())
    source = MeshPart("donor", source_result)
    target = MeshPart(
        "fringe", certify_cell_mesh(target_mesh, SpatialCoordinateContract.si())
    )
    boundary = target.scope(1, np.asarray(target_mesh.entity_set(1).entity_ids))
    connectivity = prepare_overset_connectivity(
        MeshAssembly((source, target)),
        (
            OversetPartSpec("donor", owner_rank=0),
            OversetPartSpec("fringe", boundary=boundary, owner_rank=1),
        ),
        policy=OversetPolicy(fringe_layers=1),
    )
    rotation = np.asarray(((0.0, -1.0), (1.0, 0.0)), dtype=np.float64)
    affine_matrix = np.asarray(((0.6, -0.8), (0.8, 0.6)), dtype=np.float64)
    translation = np.asarray((6.0, -2.0), dtype=np.float64)
    image_connectivities: dict[str, OversetConnectivity] = {}
    for image_name, matrix in (("rigid", rotation), ("affine", affine_matrix)):
        image_target_mesh = CellMesh(
            np.asarray(target_mesh.coordinates) @ matrix.T + translation,
            (CellBlock("cells", "triangle", np.asarray(((0, 1, 2),), dtype=np.int32)),),
        )
        image_target = MeshPart(
            "fringe", certify_cell_mesh(image_target_mesh, SpatialCoordinateContract.si())
        )
        image_connectivities[image_name] = prepare_overset_connectivity(
            MeshAssembly((source, image_target)),
            (
                OversetPartSpec(
                    "donor",
                    owner_rank=0,
                    image_rotation=matrix,
                    image_translation=translation,
                ),
                OversetPartSpec(
                    "fringe",
                    owner_rank=1,
                    boundary=image_target.scope(
                        1, np.asarray(image_target_mesh.entity_set(1).entity_ids)
                    ),
                ),
            ),
            policy=OversetPolicy(fringe_layers=1),
        )
    fe = FiniteElementPlan(
        source_mesh,
        FiniteElementFieldSpec("u", lagrange_element("triangle", 2)),
        coordinate_spec=source_result.geometry,
    ).prepare()
    fe_vector = FiniteElementPlan(
        source_mesh,
        FiniteElementFieldSpec(
            "u", lagrange_element("triangle", 2), component_shape=(2,)
        ),
        coordinate_spec=source_result.geometry,
    ).prepare()
    fv = UnstructuredFiniteVolumePlan.from_cell_mesh(source_mesh).prepare()
    fv_owner = prepare_finite_volume_field_reconstruction(
        fv, PiecewiseConstantReconstruction()
    )
    nodes = np.asarray(fe.dof_maps[0].dof_coordinates)
    fe_values = (
        jnp.asarray(3 + nodes[:, 0] ** 2 + nodes[:, 0] * nodes[:, 1] - 2 * nodes[:, 1])
        if rank == 0
        else None
    )
    fe_vector_values = (
        jnp.stack((fe_values, 1 + nodes[:, 0] - nodes[:, 1] ** 2), axis=1)
        if fe_values is not None
        else None
    )
    fv_values = (
        jnp.asarray((4.2, -9.1), dtype=jnp.float64).reshape(fv_owner.coefficient_shape)
        if rank == 0
        else None
    )
    return OversetTrafficCase(
        target_mesh,
        connectivity,
        image_connectivities,
        rotation,
        affine_matrix,
        fe,
        fe_vector,
        fv_owner,
        fe_values,
        fe_vector_values,
        fv_values,
    )
