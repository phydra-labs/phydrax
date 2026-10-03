# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Overlap Dirichlet law: owner preconditions, site admission, and exact transposes."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from examples.meshfree_hybrid_calibrated import (
    BASE_GEOMETRY,
    CELL_NEIGHBORS,
    CELL_STENCIL,
    exact,
    hybrid_problem,
    HybridSetup,
    NEIGHBORS,
    prepare_setup,
    source,
    STENCIL,
)
from phydrax.discretization import ConservativeDiffusionPlan
from phydrax.discretization.finite_volume import ConservativeBoundaryKind
from phydrax.solver.coupling import (
    FiniteVolumeComponent,
    LinearContribution,
    MeshfreeComponent,
    OverlapTransferPolicy,
    PreparedOverlapTransfers,
)


PARAMETERS = (0.2, 2.0, 0.5)


@pytest.fixture(scope="module")
def setup() -> HybridSetup:
    return prepare_setup(8, BASE_GEOMETRY)


def _cloud_component(setup: HybridSetup, artificial: float, /) -> MeshfreeComponent:
    theta = jnp.asarray(PARAMETERS)
    points = setup.cloud.points
    rows = np.asarray(setup.law.transfers.rows)
    load = setup.poisson.physical_rhs(
        source(points, theta),
        boundary_values={
            "outer": exact(points[setup.outer_rows], theta),
            "artificial": jnp.full((rows.size,), artificial),
        },
    )
    return MeshfreeComponent(
        setup.cloud,
        setup.native,
        setup.reconstruction,
        setup.cloud.quadrature_weights,
        name="cloud",
        owner_id=setup.cloud.prepared_id,
        field="u",
        load=load,
    )


def _grid_component(
    setup: HybridSetup, lower_x: ConservativeBoundaryKind, /
) -> FiniteVolumeComponent:
    grid = setup.finite_volume.grid
    diffusion = ConservativeDiffusionPlan(
        grid, boundaries={"x": (lower_x, "dirichlet"), "y": ("dirichlet", "dirichlet")}
    ).prepare(1.0)
    return FiniteVolumeComponent("grid", setup.finite_volume, diffusion)


def test_law_contributions_publish_exact_transposes(setup: HybridSetup) -> None:
    law = hybrid_problem(setup, PARAMETERS).laws[0]
    contributions = [
        item for item in law.contributions if isinstance(item, LinearContribution)
    ]
    assert len(contributions) == 2
    for contribution in contributions:
        operator = contribution.operator
        source_shape = operator.source.structure().shape
        target_shape = operator.target.structure().shape
        left = jnp.cos(jnp.arange(np.prod(source_shape), dtype=jnp.float64)).reshape(
            source_shape
        )
        right = jnp.sin(
            1.0 + jnp.arange(np.prod(target_shape), dtype=jnp.float64)
        ).reshape(target_shape)
        forward = jnp.vdot(operator.mv(left), right)
        backward = jnp.vdot(left, operator.transpose_mv(right))
        assert float(jnp.abs(forward)) > 0.0
        np.testing.assert_allclose(forward, backward, rtol=1e-12)


def test_artificial_cloud_rows_must_carry_no_owner_data(setup: HybridSetup) -> None:
    components = {
        "cloud": _cloud_component(setup, 1.0),
        "grid": _grid_component(setup, "dirichlet"),
    }
    with pytest.raises(ValueError, match="zero-data identity rows"):
        setup.law.prepare(components, ())


def test_artificial_faces_must_be_native_dirichlet_faces(setup: HybridSetup) -> None:
    components = {
        "cloud": _cloud_component(setup, 0.0),
        "grid": _grid_component(setup, "neumann"),
    }
    with pytest.raises(ValueError, match="native neumann condition"):
        setup.law.prepare(components, ())


def test_artificial_nodes_outside_the_cell_hull_are_refused(setup: HybridSetup) -> None:
    policy = OverlapTransferPolicy(
        point_stencil=STENCIL,
        cell_stencil=CELL_STENCIL,
        point_neighbors=NEIGHBORS,
        cell_neighbors=CELL_NEIGHBORS,
    )
    with pytest.raises(ValueError, match="strictly inside the hull"):
        PreparedOverlapTransfers(
            setup.cloud,
            setup.outer_rows,
            setup.finite_volume,
            (("x", "lower"),),
            np.asarray(setup.law.transfers.overlap_cells),
            policy=policy,
        )
