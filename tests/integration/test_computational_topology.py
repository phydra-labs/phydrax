#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_mesh_filtration_persistence_hodge_and_event_workflow():
    vertices = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    faces = np.asarray([[0, 1, 2]], dtype=np.int32)
    complex_ir = phx.graph.triangle_mesh_to_cochain_complex(vertices, faces)
    topology = complex_ir.discretization.topology
    complex = phx.topology.CellSubcomplex.full(topology)
    homology = phx.topology.compute_homology(
        complex,
        coefficients=phx.topology.PrimeField(2),
    )
    filtration = phx.topology.CellFiltration(
        complex,
        (
            jnp.asarray([0.0, 0.5, 1.0]),
            jnp.asarray([0.5, 1.0, 1.0]),
            jnp.asarray([2.0]),
        ),
        source_id="integration-triangle",
    )
    persistence = phx.topology.compute_persistence(
        filtration,
        coefficients=phx.topology.PrimeField(2),
    )
    diagram = persistence.diagram()
    packed = persistence.pack(4)
    _, report = phx.graph.validate_hodge_homology(complex_ir, 0)

    times = jnp.asarray([0.0, 0.25, 0.75, 1.5])
    zero_degree = diagram.degrees == 0
    counts = jax_vmap_component_count(
        times,
        diagram.birth_values,
        diagram.death_values,
        diagram.has_finite_death,
        zero_degree,
    )
    benchmark = phx.applications.free_boundary.topology_event_benchmark(
        counts,
        counts,
        times,
    )

    assert homology.dimensions == (1, 0, 0)
    assert int(packed.interval_count) == 2
    assert bool(report.complete)
    assert float(benchmark.component_count_correct) == 1.0
    assert bool(benchmark.event_detected)


def jax_vmap_component_count(times, births, deaths, finite, selected):
    def at_time(time):
        alive = (births <= time) & (~finite | (deaths > time))
        return jnp.sum((selected & alive).astype(jnp.int32))

    return jnp.stack(tuple(at_time(time) for time in times))
