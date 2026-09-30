#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import cubical_cell_complex, simplicial_cell_complex
from phydrax.discretization._topology import (
    CellComplexTopology,
    EntitySet,
    OrientedIncidence,
)
from phydrax.exterior._coefficients import (
    bloch_coefficient_system,
    CoefficientSystem,
    curvature_evidence,
    orientation_coefficient_system,
    sheaf_laplacian,
    twisted_differential,
)
from phydrax.sparse import EdgeRelation


def test_bloch_circle_has_shifted_spectrum_and_wrap_holonomy() -> None:
    count = 7
    wave = 0.23
    system = bloch_coefficient_system(
        cubical_cell_complex((count,), periodic=True), jnp.asarray([wave])
    )
    derivative = twisted_differential(system, 0).as_dense()
    expected = -np.eye(count, dtype=np.complex128)
    expected[np.arange(count - 1), np.arange(1, count)] = 1.0
    expected[-1, 0] = np.exp(1j * wave * count)
    np.testing.assert_allclose(derivative, expected, atol=1e-14)
    frequencies = (2 * np.pi * np.arange(count) + wave * count) / count
    laplacian = sheaf_laplacian(system, 0)
    eigenvalues = np.linalg.eigvalsh(
        np.asarray(laplacian.mv_block(jnp.eye(count, dtype=jnp.complex128)))
    )
    np.testing.assert_allclose(
        eigenvalues, np.sort(4 * np.sin(frequencies / 2) ** 2), atol=1e-12
    )
    assert eigenvalues[0] > 0.01


def test_bloch_torus_is_flat_with_nontrivial_global_holonomy() -> None:
    cubical = cubical_cell_complex((3, 4, 3), periodic=True)
    system = bloch_coefficient_system(cubical, jnp.asarray([0.13, -0.29, 0.41]))
    evidence = eqx.filter_jit(curvature_evidence)(system)
    np.testing.assert_allclose(evidence.residual_norms, 0.0, atol=2e-14)
    assert bool(evidence.flat)
    values = jnp.arange(system.spaces[0].size, dtype=jnp.float64).astype(jnp.complex128)
    np.testing.assert_allclose(
        eqx.filter_jit(lambda s, v: s.chain_residual(0, v))(system, values),
        0.0,
        atol=1e-12,
    )


def _triangle() -> CellComplexTopology:
    return simplicial_cell_complex(
        (
            np.asarray([[0], [1], [2]], dtype=np.int32),
            np.asarray([[0, 1], [0, 2], [1, 2]], dtype=np.int32),
            np.asarray([[0, 1, 2]], dtype=np.int32),
        )
    )


def test_real_transport_preserves_complex_fibers_in_mixed_connection() -> None:
    topology = _triangle()
    system = CoefficientSystem(
        topology,
        (
            jnp.ones(topology.incidences[0].relation.route_shape, dtype=jnp.float64),
            jnp.full(
                topology.incidences[1].relation.route_shape, 1j, dtype=jnp.complex128
            ),
        ),
    )
    values = jnp.asarray([1j, 2j, 4j], dtype=jnp.complex128)
    np.testing.assert_array_equal(system.exterior_derivative(0, values), [1j, 3j, 2j])


def test_matrix_connection_curvature_is_holonomy_minus_identity() -> None:
    topology = _triangle()
    identity = np.eye(2, dtype=np.complex128)
    first = np.asarray([[0, 1], [-1, 0]], dtype=np.complex128)
    second = np.diag(np.exp(1j * np.asarray([0.4, -0.4])))
    edge_links = (first, identity, second)
    incidence = topology.incidences[0]
    routes = np.broadcast_to(identity, incidence.relation.route_shape + (2, 2)).copy()
    for index, (edge, sign) in enumerate(
        zip(
            np.asarray(incidence.relation.target_indices),
            np.asarray(incidence.signs),
            strict=True,
        )
    ):
        if sign > 0:
            routes[index] = edge_links[int(edge)]
    face_incidence = topology.incidences[1]
    face_routes = np.broadcast_to(
        identity, face_incidence.relation.route_shape + (2, 2)
    ).copy()
    face_routes[np.asarray(face_incidence.relation.source_indices) == 2] = first
    system = CoefficientSystem(topology, (routes, face_routes))
    evidence = curvature_evidence(system)
    expected = np.zeros((2, 6), dtype=np.complex128)
    expected[:, 4:] = first @ second - identity
    np.testing.assert_allclose(evidence.operators[0].as_dense(), expected, atol=1e-14)
    np.testing.assert_allclose(
        evidence.residual_norms, [np.linalg.norm(first @ second - identity)], atol=1e-14
    )
    assert not bool(evidence.flat)


def test_flat_matrix_gauge_change_preserves_chain_and_laplacian_pairing() -> None:
    topology = _triangle()
    angles = (
        np.asarray([0.1, -0.4, 0.8]),
        np.asarray([0.2, 0.7, -0.3]),
        np.asarray([0.6]),
    )
    frames = tuple(
        np.stack([np.diag(np.exp(1j * np.asarray([angle, -angle]))) for angle in degree])
        for degree in angles
    )
    transports = tuple(
        frames[degree + 1][np.asarray(incidence.relation.target_indices)]
        @ np.swapaxes(
            np.conj(frames[degree][np.asarray(incidence.relation.source_indices)]), -1, -2
        )
        for degree, incidence in enumerate(topology.incidences)
    )
    system = CoefficientSystem(topology, transports)
    assert bool(curvature_evidence(system).flat)
    value = jnp.asarray([1 + 2j, 3 - 1j, -2 + 1j, 0.4j, -1j, 2.0], dtype=jnp.complex128)
    laplacian = sheaf_laplacian(system, 1)
    quadratic = jnp.vdot(value, laplacian.mv(value))
    lower = twisted_differential(system, 0).adjoint_mv(value)
    upper = twisted_differential(system, 1).mv(value)
    np.testing.assert_allclose(
        quadratic, jnp.vdot(lower, lower) + jnp.vdot(upper, upper), atol=1e-12
    )


def test_prepared_coefficient_refresh_differentiates_without_rebinding() -> None:
    cubical = cubical_cell_complex((5,), periodic=True)
    base = bloch_coefficient_system(cubical, jnp.asarray([0.2]), key="circle-binding")
    incidence = cubical.topology.incidences[0]
    wraps = (np.asarray(incidence.relation.target_indices) == 4) & (
        np.asarray(incidence.signs) > 0
    )
    vector = jnp.asarray([1, 2, 3, 4, 5], dtype=jnp.complex128)

    def energy(wave: Array) -> Array:
        transports = jnp.exp(1j * wave * 5 * jnp.asarray(wraps))
        refreshed = base.refresh((transports,))
        result = refreshed.exterior_derivative(0, vector)
        return jnp.real(jnp.vdot(result, result))

    derivative = jax.jit(jax.grad(energy))(jnp.asarray(0.2))
    np.testing.assert_allclose(derivative, 50 * np.sin(1.0), atol=1e-12)
    refreshed = base.refresh((jnp.exp(1j * 0.3 * 5 * jnp.asarray(wraps)),))
    assert refreshed.system_id == base.system_id


def _klein_bottle() -> CellComplexTopology:
    nx, ny = 3, 4

    def vertex(x: int, y: int) -> int:
        return x * ny + y % ny

    def horizontal(x: int, y: int) -> int:
        return x * ny + y % ny

    def vertical(x: int, y: int) -> int:
        return nx * ny + x * ny + y % ny

    entities = tuple(
        EntitySet(f"klein_{degree}", degree, np.arange(count, dtype=np.int64))
        for degree, count in enumerate((nx * ny, 2 * nx * ny, nx * ny))
    )
    sources: list[int] = []
    targets: list[int] = []
    signs: list[float] = []
    for x in range(nx):
        for y in range(ny):
            head = vertex(x + 1, y) if x < nx - 1 else vertex(0, -y)
            for edge, tail, tip in (
                (horizontal(x, y), vertex(x, y), head),
                (vertical(x, y), vertex(x, y), vertex(x, y + 1)),
            ):
                sources.extend((tail, tip))
                targets.extend((edge, edge))
                signs.extend((-1.0, 1.0))
    edge_relation = EdgeRelation(
        np.asarray(sources, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        source_size=entities[0].count,
        target_size=entities[1].count,
    )
    edge_incidence = OrientedIncidence(
        1, entities[0], entities[1], edge_relation, np.asarray(signs, dtype=np.float64)
    )
    sources = []
    targets = []
    signs = []
    for x in range(nx):
        for y in range(ny):
            right = vertical(x + 1, y) if x < nx - 1 else vertical(0, -y - 1)
            sources.extend(
                (horizontal(x, y), right, horizontal(x, y + 1), vertical(x, y))
            )
            targets.extend((vertex(x, y),) * 4)
            signs.extend((1.0, 1.0 if x < nx - 1 else -1.0, -1.0, -1.0))
    face_relation = EdgeRelation(
        np.asarray(sources, dtype=np.int32),
        np.asarray(targets, dtype=np.int32),
        source_size=entities[1].count,
        target_size=entities[2].count,
    )
    face_incidence = OrientedIncidence(
        2, entities[1], entities[2], face_relation, np.asarray(signs, dtype=np.float64)
    )
    return CellComplexTopology(entities, (edge_incidence, face_incidence))


def test_klein_bottle_orientation_coefficients_restore_top_class() -> None:
    topology = _klein_bottle()
    ordinary = np.asarray(topology.incidences[1].exterior_derivative().as_dense())
    twisted = orientation_coefficient_system(topology)
    assert topology.entity_sets[2].count - np.linalg.matrix_rank(ordinary) == 0
    assert (
        topology.entity_sets[2].count
        - np.linalg.matrix_rank(np.asarray(twisted_differential(twisted, 1).as_dense()))
        == 1
    )
    assert bool(curvature_evidence(twisted).flat)
    assert (
        np.linalg.matrix_rank(np.asarray(twisted_differential(twisted, 0).as_dense()))
        == topology.entity_sets[0].count
    )


def test_klein_polygon_occurrences_encode_orientation_character() -> None:
    entities = tuple(
        EntitySet(f"klein_polygon_{degree}", degree, np.arange(count, dtype=np.int64))
        for degree, count in enumerate((1, 2, 1))
    )
    edges = OrientedIncidence(
        1,
        entities[0],
        entities[1],
        EdgeRelation(
            np.asarray([0, 0, 0, 0], dtype=np.int32),
            np.asarray([0, 0, 1, 1], dtype=np.int32),
            source_size=1,
            target_size=2,
        ),
        np.asarray([-1, 1, -1, 1], dtype=np.float64),
    )
    faces = OrientedIncidence(
        2,
        entities[1],
        entities[2],
        EdgeRelation(
            np.asarray([0, 1, 0, 1], dtype=np.int32),
            np.asarray([0, 0, 0, 0], dtype=np.int32),
            source_size=2,
            target_size=1,
        ),
        np.asarray([1, 1, -1, 1], dtype=np.float64),
    )
    topology = CellComplexTopology(entities, (edges, faces))
    system = orientation_coefficient_system(topology)
    np.testing.assert_array_equal(twisted_differential(system, 0).as_dense(), [[0], [-2]])
    np.testing.assert_array_equal(twisted_differential(system, 1).as_dense(), [[0, 0]])
    assert bool(curvature_evidence(system).flat)
