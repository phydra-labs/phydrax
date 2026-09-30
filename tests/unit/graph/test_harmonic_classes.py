#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from tests.unit.topology._fixtures import annulus_complex, filled_triangle_topology


def test_harmonic_classes_scenario_1() -> None:
    realization = annulus_complex()
    complex = phx.topology.CellSubcomplex.full(realization.topology)
    rational = phx.topology.compute_rational_homology_basis(complex)
    frame = phx.exterior.prepare_harmonic_class_frame(realization, rational.degree(1))
    constraint = phx.solver.HarmonicConstraint(frame, target_periods=jnp.asarray([2.0]))
    field = constraint.apply(jnp.zeros((realization.cell_counts[1],)))

    np.testing.assert_allclose(frame.periods(field), [2.0], atol=1e-7)
    exact_cycles = np.asarray(
        rational.degree(1).dense(realization.cell_counts[1]), dtype=np.float64
    )
    np.testing.assert_allclose(exact_cycles.T @ np.asarray(field), [2.0], atol=1e-7)
    assert float(frame.period_defect) < 1e-9
    assert bool(frame.kernel_certificate.valid)
    assert frame.solve_evidence is not None
    assert bool(
        jnp.all(frame.solve_evidence.status == int(phx.linalg.LinearSolveStatus.SUCCESS))
    )
    assert float(constraint.residual(field)) < 1e-7
    for policy in ("free", "deflated"):
        nonprescribed = phx.solver.HarmonicConstraint(frame, policy=policy)
        expected = [2.0] if policy == "free" else [0.0]
        np.testing.assert_allclose(
            frame.periods(nonprescribed.apply(field)), expected, atol=1e-7
        )
        with pytest.raises(ValueError):
            phx.solver.HarmonicConstraint(
                frame, policy=policy, target_periods=jnp.asarray([0.0])
            )
    with pytest.raises(ValueError):
        phx.solver.HarmonicConstraint(frame)
    harmonic, _ = phx.exterior.validate_harmonic_cohomology(realization, 1)
    basis = harmonic.basis
    metric = jnp.diag(realization.hodge_diagonal(1))
    tracking = phx.exterior.HodgeSubspaceTracking(
        basis,
        3.0 * basis,
        metric,
        source_id="source",
        target_id="target",
    )

    np.testing.assert_allclose(tracking.principal_angles, 0.0, atol=1e-7)
    assert float(tracking.projector_residual) < 1e-7
    assert tracking.svd_evidence is not None
    topology = filled_triangle_topology()
    neighborhood = phx.topology.CellSubcomplex.full(topology)
    exit_set = phx.topology.CellSubcomplex(
        topology,
        tuple(
            np.zeros_like(np.asarray(mask), dtype="bool") for mask in neighborhood.masks
        ),
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


def test_relative_periods_against_exact_quotient_generators() -> None:
    realization = annulus_complex()
    ambient = phx.topology.CellSubcomplex.full(realization.topology)
    boundary = phx.topology.CellSubcomplex(
        realization.topology, realization.boundary_masks
    )
    pair = phx.topology.CellComplexPair(ambient, boundary)
    rational = phx.topology.compute_rational_homology_basis(pair)
    frame = phx.exterior.prepare_harmonic_class_frame(
        realization, rational.degree(2), boundary="relative"
    )
    values = frame.with_periods(
        jnp.zeros((realization.cell_counts[2],)), jnp.asarray([1.75])
    )
    np.testing.assert_allclose(frame.periods(values), [1.75], atol=1e-8)
    exact_cycles = np.asarray(
        rational.degree(2).dense(realization.cell_counts[2]), dtype=np.float64
    )
    np.testing.assert_allclose(exact_cycles.T @ np.asarray(values), [1.75], atol=1e-8)
    assert bool(frame.kernel_certificate.valid)
