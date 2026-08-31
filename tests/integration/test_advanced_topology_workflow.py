#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_phase_field_functorial_extended_and_uq_workflow():
    topology = phx.geometry.simplicial.TriangleTopology(
        jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
        num_vertices=3,
    ).cell_complex_topology()
    complex = phx.topology.CellSubcomplex.full(topology)
    support = phx.topology.cell_vertex_support(
        topology,
        (
            np.asarray([[0], [1], [2]], dtype=np.int32),
            np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        ),
    )
    plan = phx.applications.phase_field.phase_field_topology_plan(
        complex,
        support,
        jnp.asarray([0.25, 0.75]),
        phase="void",
    )
    first = plan.snapshot(jnp.asarray([0.0, 0.5, 1.0]), field_id="phase-0")
    second = plan.snapshot(jnp.asarray([0.1, 0.6, 1.1]), field_id="phase-1")
    extended = phx.topology.compute_extended_persistence(
        first.filtration,
        coefficients=phx.topology.PrimeField(2),
    )
    identity = phx.topology.CellularChainMap.identity(complex)
    exact = phx.topology.compute_homology(
        complex,
        coefficients=phx.topology.PrimeField(2),
        representatives="both",
    )
    induced = phx.topology.compute_induced_topology_map(identity, exact, exact)
    summary = phx.uq.TopologyEnsembleSummary((first, second))

    assert extended.extended_positive.interval_count >= 1
    np.testing.assert_array_equal(induced.homology_maps[0].matrix, [[1]])
    assert summary.sample_count == 2
