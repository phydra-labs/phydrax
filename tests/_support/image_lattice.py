#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent bounded host lattice enumeration for image-route tests."""

from collections import Counter
from itertools import product

import jax.numpy as jnp
import numpy as np

from phydrax.discretization.particle import ParticleImageRelation


Route = tuple[int, int, int, tuple[int, ...]]


def lattice_routes(
    positions: np.ndarray,
    vectors: np.ndarray,
    periodic: tuple[bool, ...],
    radius: float,
    ids: np.ndarray,
    *,
    case: int = 0,
    active: np.ndarray | None = None,
    receivers: set[int] | None = None,
    extent: int = 8,
) -> dict[Route, np.ndarray]:
    """Brute-force ``(case, source id, receiver id, n) -> x_t - x_s + n H``.

    Enumerates every integer translation in ``[-extent, extent]`` on periodic
    axes; only the zero translation of a particle to itself is excluded.
    """
    points = np.asarray(positions, dtype=np.float64)
    lattice = np.asarray(vectors, dtype=np.float64)
    mask = (
        np.ones(points.shape[0], dtype=np.bool_)
        if active is None
        else np.asarray(active, dtype=np.bool_)
    )
    choices = [range(-extent, extent + 1) if axis else (0,) for axis in periodic]
    routes: dict[Route, np.ndarray] = {}
    for shift in product(*choices):
        translation = np.asarray(shift, dtype=np.float64) @ lattice
        for source in np.flatnonzero(mask):
            for receiver in np.flatnonzero(mask):
                if source == receiver and not any(shift):
                    continue
                if receivers is not None and int(ids[receiver]) not in receivers:
                    continue
                displacement = points[receiver] - points[source] + translation
                if np.linalg.norm(displacement) < radius:
                    key = (case, int(ids[source]), int(ids[receiver]), tuple(shift))
                    routes[key] = displacement
    return routes


def relation_routes(
    relation: ParticleImageRelation, positions: np.ndarray, vectors: np.ndarray
) -> tuple[Counter[Route], dict[Route, np.ndarray]]:
    """Return the stored route multiset and displacement of each valid route."""
    valid = np.asarray(relation.valid)
    displacement = np.asarray(
        relation.displacement(jnp.asarray(positions), jnp.asarray(vectors))
    )
    keys = [
        (int(case), int(source), int(receiver), tuple(int(value) for value in shift))
        for case, source, receiver, shift in zip(
            np.asarray(relation.route_cases)[valid],
            np.asarray(relation.source_particle_ids)[valid],
            np.asarray(relation.receiver_particle_ids)[valid],
            np.asarray(relation.image_shifts)[valid],
            strict=True,
        )
    ]
    return Counter(keys), dict(zip(keys, displacement[valid], strict=True))


def assert_routes_match(
    relation: ParticleImageRelation,
    positions: np.ndarray,
    vectors: np.ndarray,
    expected: dict[Route, np.ndarray],
    *,
    atol: float = 1e-12,
) -> None:
    """Exact multiset equality (each route once) plus matching displacements."""
    multiset, displacement = relation_routes(relation, positions, vectors)
    assert all(count == 1 for count in multiset.values()), "duplicated route"
    assert set(multiset) == set(expected)
    for key, row in expected.items():
        np.testing.assert_allclose(displacement[key], row, atol=atol)


def assert_resolved_cutoff(
    routes: dict[Route, np.ndarray],
    positions: np.ndarray,
    vectors: np.ndarray,
    radius: float,
    particle_ids: np.ndarray,
    dtype: type[np.floating],
) -> None:
    """Require that input/translation/norm roundoff cannot flip the cutoff."""
    slots = {int(identifier): slot for slot, identifier in enumerate(particle_ids)}
    epsilon = np.finfo(dtype).eps
    row_norms = np.linalg.norm(vectors, axis=1)
    for (_, source, target, shift), displacement in routes.items():
        scale = (
            np.linalg.norm(positions[slots[source]])
            + np.linalg.norm(positions[slots[target]])
            + np.abs(np.asarray(shift)) @ row_norms
            + radius
        )
        roundoff = 32.0 * epsilon * scale / (1.0 - 32.0 * epsilon)
        assert abs(np.linalg.norm(displacement) - radius) > roundoff, (
            "fixture classification is unresolved at the requested precision",
            dtype,
            source,
            target,
            shift,
        )
