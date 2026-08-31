#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


topology = phx.geometry.simplicial.TriangleTopology(
    jnp.asarray([[0, 1, 2]], dtype=jnp.int32),
    num_vertices=3,
).cell_complex_topology()
complex = phx.topology.CellSubcomplex.full(topology)
identity = phx.topology.CellularChainMap.identity(complex)
homology = phx.topology.compute_homology(
    complex,
    coefficients=phx.topology.PrimeField(2),
    representatives="both",
)
induced = phx.topology.compute_induced_topology_map(identity, homology, homology)
cone = phx.topology.compute_mapping_cone_homology(
    identity,
    coefficients=phx.topology.PrimeField(2),
)
filtration = phx.topology.CellFiltration(
    complex,
    (
        jnp.asarray([0.0, 0.5, 1.0]),
        jnp.asarray([0.5, 1.0, 1.0]),
        jnp.asarray([2.0]),
    ),
    source_id="advanced-triangle",
)
persistence = phx.topology.compute_persistence(
    filtration,
    coefficients=phx.topology.PrimeField(2),
)
extended = phx.topology.compute_extended_persistence(
    filtration,
    coefficients=phx.topology.PrimeField(2),
)
diagram = persistence.diagram()
distance = phx.topology.diagram_bottleneck_distance(diagram, diagram)
rational = phx.topology.compute_rational_homology_basis(complex)
integral = phx.topology.compute_integral_homology(complex)
vineyard = phx.topology.compute_vineyard(
    (filtration, filtration),
    jnp.asarray([0.0, 1.0]),
    coefficients=phx.topology.PrimeField(2),
)
local = phx.topology.compute_cell_local_homology(
    complex,
    0,
    0,
    coefficients=phx.topology.PrimeField(2),
)

print("identity induced", np.asarray(induced.homology_maps[0].matrix))
print("identity cone acyclic", cone.acyclic)
print(
    "extended components",
    extended.ordinary.interval_count,
    extended.relative.interval_count,
    extended.extended_positive.interval_count,
    extended.extended_negative.interval_count,
)
print("self bottleneck", float(distance.distance))
print("rational dimensions", tuple(value.generator_count for value in rational.bases))
print(
    "integral",
    tuple((value.free_rank, value.torsion_invariants) for value in integral.degrees),
)
print("vineyard snapshots", len(vineyard.snapshots))
print("local homology", local.homology.dimensions)
