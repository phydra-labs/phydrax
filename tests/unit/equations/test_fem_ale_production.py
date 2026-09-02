#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.discretization._cell_mesh import CellBlock, CellMesh
from phydrax.discretization.fem._generic import FiniteElementFieldSpec, FiniteElementPlan
from phydrax.discretization.fem._reference import discontinuous_element
from phydrax.equations._hyperbolic_systems import EulerSystem
from phydrax.equations.fem._moving_conservation import (
    ale_physical_normal_flux,
    ConservativeRemapPlan,
    finite_element_ale_metric_evidence,
    FiniteElementGeometrySnapshot,
    recover_geometry_snapshot,
)


def _discretization():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    mesh = CellMesh(
        points,
        (CellBlock("cells", "quadrilateral", np.asarray(((0, 1, 2, 3),))),),
    )
    return FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "state", discontinuous_element("quadrilateral", 2), component_shape=(4,)
        ),
    ).prepare()


def test_uniform_translation_satisfies_temporal_gcl_and_ale_flux():
    discretization = _discretization()
    coordinates = discretization.default_runtime.coordinates
    velocity = jnp.broadcast_to(jnp.asarray((0.2, -0.1)), coordinates.shape)
    current = FiniteElementGeometrySnapshot(
        coordinates,
        velocity,
        0.0,
        topology_id=discretization.mesh.topology_id,
        geometry_layout_id="quadratic-geometry",
    )
    next_snapshot = current.advance(0.05)
    evidence = finite_element_ale_metric_evidence(
        discretization, current, next_snapshot, tolerance=2.0e-9
    )
    assert evidence.passed
    assert evidence.maximum_gcl_defect <= 2.0e-9

    system = EulerSystem(2)
    state = system.primitive_to_conserved(jnp.asarray((1.0, 0.3, 0.0, 1.0)))
    normal = jnp.asarray((1.0, 0.0))
    expected = system.physical_normal_flux(state, normal) - 0.2 * state
    np.testing.assert_allclose(
        ale_physical_normal_flux(system, state, normal, jnp.asarray((0.2, -0.1))),
        expected,
        atol=2.0e-12,
    )


def test_conservative_remap_and_quality_recovery_preserve_valid_state():
    mass = jnp.asarray(((2.0, 0.5), (0.5, 1.5)))
    remap = ConservativeRemapPlan(mass, mass, mass)
    state = jnp.asarray(((1.0, 2.0), (3.0, 4.0)))
    np.testing.assert_allclose(remap.apply(state), state, atol=3.0e-12)
    assert remap.constant_defect == 0.0

    discretization = _discretization()
    coordinates = discretization.default_runtime.coordinates
    accepted = FiniteElementGeometrySnapshot(
        coordinates,
        jnp.zeros_like(coordinates),
        0.0,
        topology_id=discretization.mesh.topology_id,
        geometry_layout_id="quadratic-geometry",
    )
    invalid_coordinates = coordinates.at[2].set(coordinates[0])
    candidate = FiniteElementGeometrySnapshot(
        invalid_coordinates,
        jnp.zeros_like(coordinates),
        0.1,
        topology_id=accepted.topology_id,
        geometry_layout_id=accepted.geometry_layout_id,
    )
    recovered = recover_geometry_snapshot(
        discretization, accepted, candidate, iterations=12
    )
    assert recovered.accepted
    assert 0.0 <= recovered.accepted_fraction < 1.0
