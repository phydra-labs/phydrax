#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._meshcore import meshcore_available
from phydrax.discretization import (
    HybridDiffusionBoundary,
    UnstructuredFiniteVolumeDiscretization,
    UnstructuredFiniteVolumePlan,
)
from tests._support.cell_meshes import unit_square


pytestmark = [
    pytest.mark.strict_jax,
    pytest.mark.skipif(not meshcore_available(), reason="phydrax-meshcore unavailable"),
]


@pytest.fixture(scope="module")
def geometry() -> UnstructuredFiniteVolumeDiscretization:
    return UnstructuredFiniteVolumePlan.from_cell_mesh(unit_square()).prepare()


def test_area_integrated_boundary_laws_preserve_face_order_and_signs(
    geometry: UnstructuredFiniteVolumeDiscretization,
) -> None:
    exterior = np.flatnonzero(np.asarray(geometry.neighbor_cells) < 0)
    dirichlet, neumann, robin = (int(face) for face in exterior[:3])
    boundary = HybridDiffusionBoundary(
        geometry,
        dirichlet={dirichlet: 2.0},
        neumann={neumann: 5.0},
        robin={robin: (4.0, 2.0)},
    )
    count = boundary.kind.size
    face_values = jnp.full((count,), 3.0, dtype=jnp.float64)
    outward_sum = jnp.full((count,), 2.0, dtype=jnp.float64)
    expected = np.full(geometry.neighbor_cells.size, -2.0, dtype=np.float64)
    expected[dirichlet] = 1.0
    expected[neumann] = 3.0
    expected[robin] = 2.0
    active = np.asarray(geometry.face_block.active_mask)
    np.testing.assert_array_equal(
        boundary.face_residual(face_values, outward_sum), expected[active]
    )
    prescribed = np.full(geometry.neighbor_cells.size, 3.0, dtype=np.float64)
    prescribed[dirichlet] = 2.0
    np.testing.assert_array_equal(
        boundary.impose_dirichlet(face_values), prescribed[active]
    )
    assert boundary.geometry_id == geometry.geometry_id


def test_boundary_laws_refuse_interior_faces(
    geometry: UnstructuredFiniteVolumeDiscretization,
) -> None:
    interior = int(np.flatnonzero(np.asarray(geometry.neighbor_cells) >= 0)[0])
    with pytest.raises(ValueError, match=r"diffusion_boundary\.face"):
        HybridDiffusionBoundary(geometry, dirichlet={interior: 2.0})


def test_boundary_laws_refuse_duplicate_face_ownership(
    geometry: UnstructuredFiniteVolumeDiscretization,
) -> None:
    exterior = int(np.flatnonzero(np.asarray(geometry.neighbor_cells) < 0)[0])
    with pytest.raises(ValueError, match=r"diffusion_boundary\.face"):
        HybridDiffusionBoundary(
            geometry, dirichlet={exterior: 2.0}, neumann={exterior: 3.0}
        )


def test_boundary_laws_refuse_boolean_indices(
    geometry: UnstructuredFiniteVolumeDiscretization,
) -> None:
    with pytest.raises(TypeError, match=r"diffusion_boundary\.face"):
        HybridDiffusionBoundary(geometry, dirichlet={True: 2.0})


@pytest.mark.parametrize("target", [np.nan, np.inf], ids=["nan", "infinity"])
def test_boundary_laws_refuse_nonfinite_targets(
    geometry: UnstructuredFiniteVolumeDiscretization,
    target: float,
) -> None:
    exterior = int(np.flatnonzero(np.asarray(geometry.neighbor_cells) < 0)[0])
    with pytest.raises(ValueError, match=r"diffusion_boundary\.target"):
        HybridDiffusionBoundary(geometry, neumann={exterior: target})


@pytest.mark.parametrize(
    "conductance", [0.0, -1.0, np.inf], ids=["zero", "negative", "infinity"]
)
def test_robin_laws_require_positive_finite_conductance(
    geometry: UnstructuredFiniteVolumeDiscretization,
    conductance: float,
) -> None:
    exterior = int(np.flatnonzero(np.asarray(geometry.neighbor_cells) < 0)[0])
    with pytest.raises(ValueError, match=r"diffusion_boundary\.robin\.conductance"):
        HybridDiffusionBoundary(geometry, robin={exterior: (conductance, 2.0)})
