#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tests.unit.topology._fixtures import annulus_complex, filled_triangle_topology


def test_harmonic_class_frame_controls_exact_periods():
    complex_ir = annulus_complex()
    complex = phx.topology.CellSubcomplex.full(complex_ir.discretization.topology)
    rational = phx.topology.compute_rational_homology_basis(complex)
    frame = phx.graph.prepare_harmonic_class_frame(complex_ir, rational.degree(1))
    constraint = phx.solver.HarmonicConstraint(frame, jnp.asarray([2.0]))
    field = constraint.apply(jnp.zeros((complex_ir.cell_counts[1],)))

    np.testing.assert_allclose(frame.periods(field), [2.0], atol=1e-7)
    assert float(constraint.residual(field)) < 1e-7


def test_hodge_subspace_tracking_is_zero_for_identical_frames():
    complex_ir = annulus_complex()
    harmonic = phx.graph.compute_harmonic_subspace(complex_ir, max_modes=3)
    basis = harmonic.bases[1][:, : harmonic.ranks[1]]
    metric = jnp.diag(complex_ir.hodge_stars[1])
    tracking = phx.graph.HodgeSubspaceTracking(
        basis,
        basis,
        metric,
        source_id="source",
        target_id="target",
    )

    np.testing.assert_allclose(tracking.principal_angles, 0.0, atol=1e-7)
    assert float(tracking.projector_residual) < 1e-7


def test_conley_homology_requires_isolating_pair():
    topology = filled_triangle_topology()
    neighborhood = phx.topology.CellSubcomplex.full(topology)
    exit_set = phx.topology.CellSubcomplex(
        topology,
        tuple(np.zeros_like(np.asarray(mask), dtype=bool) for mask in neighborhood.masks),
    )
    relation = phx.sparse.EdgeRelation(
        np.asarray([0]),
        np.asarray([0]),
        source_size=1,
        target_size=1,
    )
    enclosure = phx.dynamics.CellMapEnclosure(
        neighborhood,
        exit_set,
        relation,
        degree=2,
    )
    index = phx.dynamics.compute_conley_homology_index(
        enclosure,
        coefficients=phx.topology.PrimeField(2),
    )
    pair = phx.topology.CellComplexPair(neighborhood, exit_set)
    pair_map = phx.topology.CellularPairMap(
        pair,
        pair,
        phx.topology.CellularChainMap.identity(neighborhood),
    )
    full_index = phx.dynamics.compute_conley_index(
        enclosure,
        pair_map,
        coefficients=phx.topology.PrimeField(2),
    )

    assert enclosure.isolating
    assert index.homology.dimensions == (1, 0, 0)
    np.testing.assert_array_equal(full_index.index_maps[0].matrix, [[1]])
