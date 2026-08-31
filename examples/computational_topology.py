#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


mesh_topology = phx.geometry.simplicial.TriangleTopology(
    jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    num_vertices=3,
).cell_complex_topology()
complex = phx.topology.CellSubcomplex.full(mesh_topology)
homology = phx.topology.compute_homology(
    complex,
    coefficients=phx.topology.PrimeField(2),
    representatives="both",
)
rational = phx.topology.compute_betti_dimensions(
    complex,
    coefficients=phx.topology.RationalField(),
)
filtration = phx.topology.CellFiltration(
    complex,
    (
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([0.5, 1.0, 1.0]),
        jnp.asarray([2.0]),
    ),
    source_id="filled-triangle",
)
persistence = phx.topology.compute_persistence(
    filtration,
    coefficients=phx.topology.PrimeField(2),
    representatives="cycles",
)
diagram = persistence.diagram()
packed = persistence.pack(4)
frozen = phx.topology.freeze_persistence_pairing(persistence, filtration)
evaluation = frozen.evaluate(filtration.values)
metric_complex = phx.graph.triangle_mesh_to_cochain_complex(
    jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
)
harmonics, hodge_report = phx.graph.validate_hodge_homology(metric_complex, 0)
_, kernel_certificate, _ = phx.graph.cochain_harmonic_kernel_certificate(
    metric_complex,
    0,
    harmonic_subspace=harmonics,
)

print("homology", homology.dimensions)
print("rational betti", rational.dimensions)
print("persistence degrees", diagram.degrees)
print("births", diagram.birth_values)
print("deaths", diagram.death_values)
print("finite", diagram.has_finite_death)
print("packed active", packed.active_mask)
print("frozen order valid", evaluation.ordering_valid)
print("hodge rank agreement", hodge_report.ranks_match)
print("kernel certificate", kernel_certificate.valid)
