import gc
import weakref
from collections.abc import Callable, Mapping
from dataclasses import replace
from itertools import combinations, permutations
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from phydrax._host_storage import immutable_host_array, retained_host_storage_profile
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureResource,
    CoordinateEnclosureResourceError,
)
from phydrax.geometry import _affine_chart_coverage as owner
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._planar_chain_embedding import (
    _PlanarOperation,
    _PlanarTopology,
    _PreparedPlanarOperands,
    _prove_planar_embedding,
    _transport_planar_embedding,
    PlanarEmbedding,
    prepare_planar_topology,
    prove_planar_embedding,
)
from phydrax.geometry._planar_primitives import intersection
from phydrax.geometry._source_boundary_records import (
    AffineChainResourceEvidence,
    SourceBoundaryChartCover,
)
from tests._support.coordinate_references import coordinate_reference_budget


def _grid(width: int, height: int, *, hole: bool = False) -> tuple[np.ndarray, ...]:
    rows = []
    for x in range(width):
        for y in range(height):
            if hole and 0 < x < width - 1 and 0 < y < height - 1:
                continue
            square = np.asarray(((x, y, 0), (x + 1, y, 0), (x + 1, y + 1, 0), (x, y + 1, 0)), dtype=float)
            rows.extend((square[[0, 1, 2]], square[[0, 2, 3]]))
    return tuple(rows)


def _exercise(
    rows: tuple[np.ndarray, ...],
    *,
    pairwise: bool = False,
    prepared_path: bool = False,
    limits: MeshCertificateLimits | None = None,
    capacity: int = 1_000_000,
    mutate: Callable | None = None,
    topology: _PlanarTopology | None = None,
) -> tuple[PlanarEmbedding, int, int]:
    with (
        coordinate_reference_budget(limits or MeshCertificateLimits()) as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        sources = tuple(row[[2, 1, 0]] for row in reversed(rows)) if prepared_path else ()
        prepared = owner._prepare_chain_records(rows, sources, preparation)
        if mutate is not None:
            mutate(rows, prepared, preparation)
        if prepared_path:
            _, result, _ = owner._chain_embedding_proofs(
                rows, sources, prepared, preparation, capacity + len(rows),
                topology=topology,
            )
            assert preparation.identity_mapping is None
        else:
            result = prove_planar_embedding(
                rows, prepared.target, prepared.records, preparation,
                capacity=capacity, point_bits=prepared.point_bits,
            )
        if pairwise:
            overlaps = []
            for (first, plane), (second, other_plane) in combinations(prepared.target, 2):
                assert plane is not None and other_plane is not None
                if plane[0] == other_plane[0]:
                    # Independent clipping oracle, not theorem or support caches.
                    overlaps.append(intersection(
                        tuple(tuple(p[axis] for axis in plane[1]) for p in first),
                        tuple(tuple(p[axis] for axis in plane[1]) for p in second),
                        budget=budget,
                    ) != 0)
            assert not any(overlaps)
            assert result.complete
        return result, budget.required_work_units, budget.peak_bytes_upper


@pytest.mark.parametrize("kind", ("grid", "permuted", "disconnected", "hole", "island-in-hole", "planes"))
def test_planar_embedding_agrees_with_independent_pairwise(kind: str) -> None:
    rows = _grid(3, 3, hole=kind in ("hole", "island-in-hole"))
    if kind == "permuted":
        rows = tuple(row[[2, 1, 0]] if index % 2 else row[[1, 2, 0]] for index, row in enumerate(reversed(rows)))
    elif kind == "disconnected":
        rows += tuple(row + np.asarray((5, 0, 0)) for row in _grid(1, 1))
    elif kind == "island-in-hole":
        rows += tuple(row / 4 + np.asarray((1.25, 1.25, 0)) for row in _grid(1, 1))
    elif kind == "planes":
        rows += tuple(row + np.asarray((0, 0, 1)) for row in _grid(1, 1))
    result, _, _ = _exercise(rows, pairwise=True)
    assert result.components == (2 if kind in ("disconnected", "island-in-hole", "planes") else 1)
    assert result.holes == (1 if kind in ("hole", "island-in-hole") else 0)
    assert _exercise(rows)[0] == result
    prepared = _exercise(rows, prepared_path=True)[0]
    assert replace(prepared, obligations=result.obligations) == result


@pytest.mark.parametrize("kind", ("duplicate", "reversed-duplicate", "same-side", "fold", "t-junction", "nonmanifold", "pinch", "nested", "crossing", "degenerate"))
def test_planar_embedding_refuses_ambiguous_complexes(kind: str) -> None:
    rows = _grid(1, 1)
    if kind == "duplicate":
        rows += (rows[0].copy(),)
    elif kind == "reversed-duplicate":
        rows += (rows[0][::-1].copy(),)
    elif kind == "same-side":
        rows = (rows[0], np.asarray(((0, 0, 0), (1, 1, 0), (0.75, 0.25, 0))))
    elif kind == "fold":
        rows = (rows[0], np.asarray(((0, 0, 0), (1, 1, 0), (2, 0, 0))))
    elif kind == "t-junction":
        rows += (np.asarray(((1, 0.5, 0), (2, 0, 0), (2, 1, 0))),)
    elif kind == "nonmanifold":
        rows += (np.asarray(((0, 0, 0), (1, 1, 0), (-1, 1, 0))),)
    elif kind == "pinch":
        rows += tuple(row + np.asarray((1, 1, 0)) for row in _grid(1, 1))
    elif kind == "nested":
        rows += tuple(row / 4 + np.asarray((0.25, 0.25, 0)) for row in _grid(1, 1))
    elif kind == "crossing":
        rows += tuple(row + np.asarray((0.5, 0.5, 0)) for row in _grid(1, 1))
    else:
        rows += (np.zeros((3, 3)),)
    # Keep both routes above the production theorem-admission threshold,
    # without changing the faulty component's geometry.
    rows += tuple(row + np.asarray((10, 10, 0)) for row in _grid(1, 1))
    result, _, _ = _exercise(rows)
    assert not result.complete
    prepared = _exercise(rows, prepared_path=True)[0]
    assert replace(prepared, obligations=result.obligations) == result


@pytest.mark.parametrize("fault", ("row", "record", "fraction", "plane", "profile"))
@pytest.mark.parametrize("prepared_path", (False, True))
def test_planar_embedding_authenticates_exact_owners(fault: str, prepared_path: bool) -> None:
    def mutate(
        rows: tuple[np.ndarray, ...],
        prepared: owner._PreparedChains,
        preparation: owner._Preparation,
    ) -> None:
        if fault == "row":
            rows[0][0, 0] = 0.25
        elif fault == "record":
            prepared.records[id(rows[0])] = prepared.target[1]
        elif fault == "fraction":
            object.__setattr__(prepared.target[0][0][0][0], "_numerator", 7)
        elif fault == "plane":
            plane = prepared.target[0][1]
            assert plane is not None
            object.__setattr__(plane[0][0], "_numerator", 7)
        else:
            points = prepared.target[0][0]
            preparation.point_profiles[id(points)] = replace(preparation.point_profiles[id(points)], owner=object())

    with pytest.raises(ValueError, match="(owners|owner|mutated coefficients)"):
        _exercise(_grid(2, 2), mutate=mutate, prepared_path=prepared_path)


@pytest.mark.parametrize("resource", ("work", "memory", "candidate"))
@pytest.mark.parametrize("prepared_path", (False, True))
def test_planar_embedding_exact_resource_boundary(resource: str, prepared_path: bool) -> None:
    rows = _grid(3, 3, hole=True)
    result, work, memory = _exercise(rows, prepared_path=prepared_path)
    assert _exercise(
        rows, prepared_path=prepared_path,
        limits=MeshCertificateLimits(maximum_work_units=work, maximum_scratch_bytes=memory),
        capacity=result.obligations if resource == "candidate" else 1_000_000,
    )[0] == result
    if resource == "candidate":
        assert not _exercise(rows, capacity=result.obligations - 1, prepared_path=prepared_path)[0].complete
    else:
        with pytest.raises(CoordinateEnclosureResourceError):
            _exercise(rows, prepared_path=prepared_path, limits=MeshCertificateLimits(
                maximum_work_units=work - (resource == "work"),
                maximum_scratch_bytes=memory - (resource == "memory"),
            ))


@pytest.mark.parametrize("kind", ("gap", "island", "overlap", "t-junction"))
def test_planar_embedding_preserves_tiling_and_pairwise_fallback(kind: str) -> None:
    source = _grid(2, 2)
    target = source
    expected = None
    if kind == "gap":
        target = target[:-1]
    elif kind == "island":
        target += tuple(row + np.asarray((5, 0, 0)) for row in _grid(1, 1))
    elif kind == "overlap":
        target += (target[0],)
    else:
        # Two nonconforming but nonoverlapping halves; ordinary pairwise proof
        # remains admissible, while the manifold theorem must not claim them.
        source = (np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0))),
                  np.asarray(((1, 0, 0), (1, 1, 0), (0, 1, 0))))
        target = (source[0], np.asarray(((1, 0, 0), (1, 1, 0), (0.5, 0.5, 0))),
                  np.asarray(((1, 1, 0), (0, 1, 0), (0.5, 0.5, 0))))
        expected = 0.0
    cover = SourceBoundaryChartCover(np.asarray(source), np.zeros(len(source)), "certified", True, "complex", "r1")
    bound, proved, _ = owner.prove_affine_chart_chain(
        [(row, 0.0, 0.0) for row in target], cover, MeshCertificateLimits(), 0.0,
        source_identity=("complex", "r1"),
    )
    assert bound == expected
    if kind == "t-junction":
        counts = {
            row.metric: row.used for row in proved.affine_chain_resources
            if row.role == "target" and row.owner_id == cover.cover_id
        }
        assert counts["embedding_target"] == 0
        assert counts["embedding_source"] == 0


def test_planar_embedding_capped_size_saves_remaining_pair_obligations(record_property: Callable[[str, object], None]) -> None:
    # 778 triangles, independently generated planar disk: static scaling only,
    # not a capped-cylinder profile or scientific acceptance claim.
    rows = _grid(19, 20) + tuple(row + np.asarray((0, 20, 0)) for row in _grid(9, 1))
    assert len(rows) == 778
    result, work, memory = _exercise(rows)
    assert result.complete
    assert result.obligations < 4990 - 387
    record_property("theorem_obligations", result.obligations)
    record_property("theorem_with_preparation_work", work)
    record_property("theorem_with_preparation_bytes", memory)


def test_planar_embedding_keeps_ids_and_accounts_candidate_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _grid(3, 3, hole=True)
    cover = SourceBoundaryChartCover(
        np.asarray(rows), np.zeros(len(rows)), "certified", True, "complex", "r1",
        restriction_ids=("outer", "hole"),
    )

    def forbidden(
        first: tuple[owner._Triangle, owner._PlaneKey | None],
        second: tuple[owner._Triangle, owner._PlaneKey | None],
        preparation: owner._Preparation,
    ) -> bool | None:
        raise AssertionError("An admitted embedding must not enumerate triangle pairs")

    monkeypatch.setattr(owner, "_exact_overlap", forbidden)
    proxies = [(row, 0.0, 0.0) for row in rows]
    first, proved, _ = owner.prove_affine_chart_chain(
        proxies, cover, MeshCertificateLimits(), 0.0, source_identity=("complex", "r1"),
    )
    assert first == 0.0
    counts = {
        row.metric: row.used for row in proved.affine_chain_resources
        if row.role == "target" and row.owner_id == cover.cover_id
    }
    assert counts["embedding_target"] == counts["embedding_source"] == 1
    required = counts["embedding_obligations"] + len(rows)
    second, repeated, _ = owner.prove_affine_chart_chain(
        proxies, cover, MeshCertificateLimits(maximum_candidate_pairs=required), 0.0,
        source_identity=("complex", "r1"),
    )
    assert second == first
    assert repeated.restriction_ids == proved.restriction_ids == cover.restriction_ids
    repeated_counts = {
        row.metric: row.used for row in repeated.affine_chain_resources
        if row.role == "target" and row.owner_id == cover.cover_id
    }
    assert repeated_counts == counts
    refused, _, _ = owner.prove_affine_chart_chain(
        proxies, cover, MeshCertificateLimits(maximum_candidate_pairs=required - 1), 0.0,
        source_identity=("complex", "r1"),
    )
    assert refused is None


@pytest.mark.parametrize("fault", ("operation", "preparation", "rows", "order", "owners", "groups", "replay", "rearm"))
def test_prepared_planar_operands_are_scoped_and_one_shot(fault: str) -> None:
    rows = _grid(2, 2)
    with (
        coordinate_reference_budget(MeshCertificateLimits()) as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(rows, (), preparation)
        operation = _PlanarOperation()
        token = _PreparedPlanarOperands(
            operation, preparation, rows, prepared.target, prepared.records,
            (), groups={},
        )
        operation.pending = token
        records, owners = prepared.target, prepared.records
        current = preparation
        if fault == "operation":
            operation.close()
        elif fault == "preparation":
            current = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        elif fault == "rows":
            rows = tuple(row.copy() for row in rows)
        elif fault == "order":
            records = tuple(reversed(records))
        elif fault == "owners":
            owners = dict(owners)
        elif fault == "groups":
            token = replace(token, groups={})
        else:
            assert token.consume(rows, records, owners, current) is token.triangles
            if fault == "rearm":
                operation.pending = token
        with pytest.raises(ValueError, match="current operation and ordered owners"):
            token.consume(rows, records, owners, current)
        assert operation.pending is None
        with pytest.raises(ValueError, match="current operation and ordered owners"):
            token.consume(token.rows, token.records, token.owners, token.preparation)


@pytest.mark.parametrize("kind", ("grid", "planes"))
def test_prepared_planar_778_row_authentication_saving(
    kind: str,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    """Static two-role deltas; never run a capped-cylinder profile."""
    rows = _grid(19, 20) + tuple(row + np.asarray((0, 20, 0)) for row in _grid(9, 1))
    if kind == "planes":
        triangle = _grid(1, 1)[0]
        rows = tuple(triangle + np.asarray((0, 0, height)) for height in range(778))
    sources = tuple(row.copy() for row in rows)
    assert len(rows) == 778
    theorem_work: list[int] = []
    transport_work: list[int] = []

    def measured(
        current_rows: tuple[np.ndarray, ...],
        records: tuple[tuple[owner._Triangle, owner._PlaneKey | None], ...],
        owners: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *,
        capacity: int,
        point_bits: int,
        _prepared: _PreparedPlanarOperands | None = None,
    ) -> PlanarEmbedding:
        before = preparation.budget.required_work_units
        result = _prove_planar_embedding(
            current_rows, records, owners, preparation, capacity=capacity,
            point_bits=point_bits, _prepared=_prepared,
        )
        theorem_work.append(preparation.budget.required_work_units - before)
        return result

    def measured_transport(
        current_rows: tuple[np.ndarray, ...],
        records: tuple[tuple[owner._Triangle, owner._PlaneKey | None], ...],
        owners: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *,
        capacity: int,
        source: _PreparedPlanarOperands,
        target: _PreparedPlanarOperands,
        mapping: tuple[int, ...],
        evidence: PlanarEmbedding,
    ) -> PlanarEmbedding:
        before = preparation.budget.required_work_units
        result = _transport_planar_embedding(
            current_rows, records, owners, preparation, capacity=capacity,
            source=source, target=target, mapping=mapping, evidence=evidence,
        )
        transport_work.append(preparation.budget.required_work_units - before)
        return result

    monkeypatch.setattr(owner, "_prove_planar_embedding", measured)
    monkeypatch.setattr(owner, "_transport_planar_embedding", measured_transport)

    def exercise(reuse: bool) -> tuple[tuple[PlanarEmbedding, PlanarEmbedding], int, int, int]:
        with (
            coordinate_reference_budget(MeshCertificateLimits()) as budget,
            budget.live_storage() as cache,
            budget.live_storage() as sides,
        ):
            preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
            prepared = owner._prepare_chain_records(rows, sources, preparation)
            if reuse:
                identity, target, source = owner._chain_embedding_proofs(
                    rows, sources, prepared, preparation, 1_000_000,
                )
                assert identity
                # The raw run snapshots this same authenticated predecessor;
                # the difference includes numeric batching, not just the theorem.
                after_identity = 0
            else:
                assert owner._chain_identity_tiling(rows, sources, prepared, preparation)
                after_identity = budget.required_work_units
                target = measured(
                    rows, prepared.target, prepared.records, preparation,
                    capacity=1_000_000 - len(rows), point_bits=prepared.point_bits,
                )
                source = measured(
                    sources, prepared.source, prepared.records, preparation,
                    capacity=1_000_000 - len(rows) - target.obligations,
                    point_bits=prepared.point_bits,
                )
            return (target, source), budget.required_work_units, budget.peak_bytes_upper, after_identity

    raw, raw_work, _, after_identity = exercise(False)
    prepared, prepared_work, prepared_bytes, _ = exercise(True)
    assert prepared[0] == raw[0]
    assert replace(prepared[1], obligations=raw[1].obligations) == raw[1]
    assert prepared[0] is not prepared[1]
    assert all(result.complete for result in prepared)
    saving = raw_work - prepared_work
    if kind == "grid":
        assert saving > 24_407
        assert theorem_work[1] - transport_work[0] > 10_000
        # Sources here are exact ordered copies: a second prepared theorem
        # would perform the same topology work as the measured target.
        assert theorem_work[2] - transport_work[0] > 10_000
    assert theorem_work[0] - theorem_work[2] > 24_407
    assert len(theorem_work) == 3  # No second topology/Jordan theorem.
    assert after_identity > 0
    record_property("prepared_planar_target_theorem_work", theorem_work[2])
    record_property("prepared_planar_target_theorem_work_saved", theorem_work[0] - theorem_work[2])
    record_property("transported_source_work", transport_work[0])
    record_property("transported_source_work_saved", theorem_work[1] - transport_work[0])
    record_property("prepared_repeat_topology_work_saved", theorem_work[2] - transport_work[0])
    record_property("prepared_planar_two_role_work_saved", saving)
    record_property("prepared_planar_two_role_total_work", prepared_work)
    record_property("prepared_planar_two_role_peak_bytes", prepared_bytes)


@pytest.mark.parametrize("order", tuple(permutations(range(3))))
def test_transport_matches_fresh_source_for_all_vertex_permutations(order: tuple[int, ...]) -> None:
    rows = _grid(3, 3, hole=True) + tuple(row + (0, 0, 2) for row in _grid(2, 2))
    sources = tuple(rows[index][list(order)] for index in np.random.default_rng(42).permutation(len(rows)))
    with (
        coordinate_reference_budget(MeshCertificateLimits()) as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(rows, sources, preparation)
        identity, target, source = owner._chain_embedding_proofs(
            rows, sources, prepared, preparation, 1_000_000,
        )
        fresh = prove_planar_embedding(
            sources, prepared.source, prepared.records, preparation,
            capacity=1_000_000, point_bits=prepared.point_bits,
        )
        assert identity and target.complete and source.complete
        assert target is not source and source is not fresh
        assert replace(source, obligations=fresh.obligations) == fresh


@pytest.mark.parametrize("fault", (
    "row", "record", "fraction", "plane", "incidence", "group", "mapping",
    "role", "evidence", "cross-request", "replay",
))
def test_transport_reauthenticates_source_and_fresh_role(
    fault: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def intercepted(
        rows: tuple[np.ndarray, ...],
        records: tuple[tuple[owner._Triangle, owner._PlaneKey | None], ...],
        owners: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *,
        capacity: int,
        source: _PreparedPlanarOperands,
        target: _PreparedPlanarOperands,
        mapping: tuple[int, ...],
        evidence: PlanarEmbedding,
    ) -> PlanarEmbedding:
        if fault == "row":
            rows[0][0, 0] += 0.25
        elif fault == "record":
            owners[id(rows[0])] = records[1]
        elif fault == "fraction":
            object.__setattr__(records[0][0][0][0], "_numerator", 7)
        elif fault in ("plane", "incidence", "group"):
            first = source.triangles[0]
            if fault == "plane":
                changed = replace(first, plane=None)
            elif fault == "incidence":
                changed = replace(first, projected=(first.projected[0],) * 3)
            else:
                changed = replace(first, group=(first.group[0] + 1,) + first.group[1:])
            object.__setattr__(source, "triangles", (changed,) + source.triangles[1:])
        elif fault == "mapping":
            mapping = (0,) * len(rows)
        elif fault == "role":
            source = target
        elif fault == "evidence":
            evidence = replace(evidence)
        elif fault == "cross-request":
            preparation = owner._Preparation(
                budget=preparation.budget, cache_storage=preparation.cache_storage,
                side_storage=preparation.cache_storage,
            )
        else:
            result = _transport_planar_embedding(
                rows, records, owners, preparation, capacity=capacity,
                source=source, target=target, mapping=mapping, evidence=evidence,
            )
            assert result.complete
        return _transport_planar_embedding(
            rows, records, owners, preparation, capacity=capacity,
            source=source, target=target, mapping=mapping, evidence=evidence,
        )

    monkeypatch.setattr(owner, "_transport_planar_embedding", intercepted)
    with pytest.raises(ValueError, match="(owners|owner|incidence|permutation|role|theorem|coefficients)"):
        _exercise(_grid(3, 3, hole=True), prepared_path=True)


@pytest.mark.parametrize("fault", ("duplicate", "missing", "mutated"))
def test_ineligible_source_never_transports(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _grid(2, 2)
    sources = tuple(row.copy() for row in rows)
    if fault == "duplicate":
        sources = (sources[1],) + sources[1:]
    elif fault == "missing":
        sources = sources[:-1]
    else:
        sources[0][0, 0] += 0.25

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("An ineligible source must prove independently")

    monkeypatch.setattr(owner, "_transport_planar_embedding", forbidden)
    with (
        coordinate_reference_budget(MeshCertificateLimits()) as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(rows, sources, preparation)
        identity, target, source = owner._chain_embedding_proofs(
            rows, sources, prepared, preparation, 1_000_000,
        )
        assert identity is None and target.complete
        fresh = prove_planar_embedding(
            sources, prepared.source, prepared.records, preparation,
            capacity=1_000_000, point_bits=prepared.point_bits,
        )
        assert source == fresh


def test_prepared_single_plane_simplexes_avoid_graph_work(
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None],
) -> None:
    from phydrax.geometry._planar_chain_embedding import _Proof

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("A singleton plane must use the canonical simplex theorem")

    regroup = False
    theorem_work: list[int] = []

    def measured(
        current_rows: tuple[np.ndarray, ...],
        records: tuple[tuple[owner._Triangle, owner._PlaneKey | None], ...],
        owners: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *,
        capacity: int,
        point_bits: int,
        _prepared: _PreparedPlanarOperands | None = None,
    ) -> PlanarEmbedding:
        assert _prepared is not None and _prepared.groups is not None
        assert len(_prepared.groups) == 480
        if regroup:
            # Differential baseline: same authenticated operands and canonical
            # theorem, but independently rebuild their plane membership.
            object.__setattr__(_prepared, "groups", None)
        before = preparation.budget.required_work_units
        result = _prove_planar_embedding(
            current_rows, records, owners, preparation, capacity=capacity,
            point_bits=point_bits, _prepared=_prepared,
        )
        theorem_work.append(preparation.budget.required_work_units - before)
        return result

    triangle = _grid(1, 1)[0]
    rows = tuple(triangle + (0, 0, height) for height in range(480))
    monkeypatch.setattr(_Proof, "plane", forbidden)
    monkeypatch.setattr(owner, "_prove_planar_embedding", measured)
    result, reused_work, _ = _exercise(rows, prepared_path=True)
    assert result.complete and result.components == 480 and result.boundary_edges == 1440
    regroup = True
    repeated, regrouped_work, _ = _exercise(rows, prepared_path=True)
    assert repeated == result
    assert theorem_work[1] - theorem_work[0] == 480
    assert regrouped_work - reused_work == 480
    record_property("singleton_prepared_membership_work_saved", 480)
    record_property("singleton_target_theorem_work", theorem_work[0])


def _compile_topology(rows: tuple[np.ndarray, ...]) -> _PlanarTopology:
    with (
        coordinate_reference_budget(MeshCertificateLimits()) as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        sources = tuple(row[[2, 1, 0]] for row in reversed(rows))
        prepared = owner._prepare_chain_records((), sources, preparation)
        topology = prepare_planar_topology(prepared.source, preparation)
        assert topology is not None
        return topology


@pytest.mark.parametrize("kind", ("grid", "permuted", "hole", "planes", "fold", "pinch", "crossing", "t-junction"))
def test_source_numeric_schedule_runs_fresh_topology(kind: str) -> None:
    rows = _grid(3, 3, hole=kind == "hole")
    if kind == "planes":
        rows += tuple(row + (0, 0, 2) for row in _grid(2, 2))
    elif kind == "permuted":
        orders = tuple(permutations(range(3)))
        rows = tuple(
            rows[index][list(orders[offset % len(orders)])]
            for offset, index in enumerate(np.random.default_rng(17).permutation(len(rows)))
        )
    elif kind == "fold":
        rows += (np.asarray(((0, 0, 0), (1, 1, 0), (2, 0, 0))),)
    elif kind == "pinch":
        rows += tuple(row + (3, 3, 0) for row in _grid(1, 1))
    elif kind == "crossing":
        rows += tuple(row + (0.25, 0.25, 0) for row in _grid(1, 1))
    elif kind == "t-junction":
        rows += (np.asarray(((3, 0.5, 0), (4, 0, 0), (4, 1, 0))),)
    expected = _exercise(rows, prepared_path=True)[0]
    schedule = _compile_topology(rows)
    actual = _exercise(rows, prepared_path=True, topology=schedule)[0]
    assert replace(actual, obligations=expected.obligations) == expected
    assert actual.complete == (kind in ("grid", "permuted", "hole", "planes"))
    if actual.complete:
        assert actual.obligations == expected.obligations


@pytest.mark.parametrize("fault", (
    "row-count", "planes", "plane", "edges", "links", "components", "boundary", "segments",
    "coefficient", "projected-coefficient", "storage", "mapping", "permutation", "replay",
))
def test_source_numeric_schedule_rejects_changed_owners_and_mapping(
    fault: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _grid(3, 3, hole=True)
    schedule = _compile_topology(rows)

    def intercepted(
        current_rows: tuple[np.ndarray, ...],
        records: tuple[tuple[owner._Triangle, owner._PlaneKey | None], ...],
        owners: dict[int, tuple[owner._Triangle, owner._PlaneKey | None]],
        preparation: owner._Preparation,
        *,
        capacity: int,
        point_bits: int,
        _prepared: _PreparedPlanarOperands | None = None,
    ) -> PlanarEmbedding:
        assert _prepared is not None and _prepared.topology is schedule
        if fault == "row-count":
            object.__setattr__(schedule, "row_count", schedule.row_count - 1)
        elif fault == "planes":
            object.__setattr__(schedule, "planes", ())
        elif fault in ("plane", "edges", "links", "components", "boundary", "segments"):
            object.__setattr__(schedule.planes[0], fault, ())
        elif fault == "coefficient":
            scalar = schedule.planes[0].coefficients[0][0]
            object.__setattr__(scalar, "_numerator", scalar.numerator + 1)
        elif fault == "projected-coefficient":
            scalar = schedule.planes[0].points[0][0]
            object.__setattr__(scalar, "_numerator", scalar.numerator + 1)
        elif fault == "storage":
            object.__setattr__(schedule, "storage_bytes", 0)
        elif fault in ("mapping", "permutation"):
            mapping = _prepared.source_mapping
            assert mapping is not None
            changed = (0,) * len(mapping) if fault == "mapping" else mapping[1:] + mapping[:1]
            object.__setattr__(_prepared, "source_mapping", changed)
        else:
            first = _prove_planar_embedding(
                current_rows, records, owners, preparation, capacity=capacity,
                point_bits=point_bits, _prepared=_prepared,
            )
            assert first.complete
        return _prove_planar_embedding(
            current_rows, records, owners, preparation, capacity=capacity,
            point_bits=point_bits, _prepared=_prepared,
        )

    monkeypatch.setattr(owner, "_prove_planar_embedding", intercepted)
    with pytest.raises(ValueError, match="(owners|permutation|incidence)"):
        _exercise(rows, prepared_path=True, topology=schedule)


@pytest.mark.parametrize("resource", ("work", "memory", "candidate"))
def test_source_numeric_schedule_resource_boundary(resource: str) -> None:
    rows = _grid(3, 3, hole=True)
    schedule = _compile_topology(rows)
    result, work, memory = _exercise(rows, prepared_path=True, topology=schedule)
    assert _exercise(
        rows, prepared_path=True, topology=schedule,
        limits=MeshCertificateLimits(maximum_work_units=work, maximum_scratch_bytes=memory),
        capacity=result.obligations if resource == "candidate" else 1_000_000,
    )[0] == result
    if resource == "candidate":
        assert not _exercise(
            rows, prepared_path=True, topology=schedule, capacity=result.obligations - 1,
        )[0].complete
    else:
        with pytest.raises(CoordinateEnclosureResourceError):
            _exercise(rows, prepared_path=True, topology=schedule, limits=MeshCertificateLimits(
                maximum_work_units=work - (resource == "work"),
                maximum_scratch_bytes=memory - (resource == "memory"),
            ))


def test_source_numeric_schedule_149_face_work_saving(
    record_property: Callable[[str, object], None],
) -> None:
    strip = _grid(75, 1)
    rows = strip[:-2] + strip[-1:]
    assert len(rows) == 149
    baseline, raw_work, _ = _exercise(rows, prepared_path=True)
    schedule = _compile_topology(rows)
    result, work, memory = _exercise(rows, prepared_path=True, topology=schedule)
    assert result == baseline and result.complete
    assert work < raw_work
    record_property("prepared_149_face_topology_work_saved", raw_work - work)
    record_property("prepared_149_face_topology_peak_bytes", memory)


@pytest.mark.parametrize("refuse", (False, True))
def test_source_schedule_releases_preparation_owners(
    refuse: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparations: list[weakref.ReferenceType[owner._Preparation]] = []
    chains: list[weakref.ReferenceType[owner._PreparedChains]] = []
    views: list[weakref.ReferenceType[np.ndarray]] = []
    prepare = owner._prepare_chain_records
    compile_schedule = owner.prepare_planar_topology

    def capture(target: tuple[np.ndarray, ...], source: tuple[np.ndarray, ...],
                preparation: owner._Preparation) -> owner._PreparedChains:
        preparations.append(weakref.ref(preparation))
        views.extend(weakref.ref(row) for row in source)
        result = prepare(target, source, preparation)
        chains.append(weakref.ref(result))
        return result

    def compile_or_refuse(
        records: tuple[tuple[owner._Triangle, owner._PlaneKey | None], ...],
        preparation: owner._Preparation,
    ) -> _PlanarTopology | None:
        result = compile_schedule(records, preparation)
        if refuse:
            raise ValueError("source schedule refused")
        return result

    monkeypatch.setattr(owner, "_prepare_chain_records", capture)
    monkeypatch.setattr(owner, "prepare_planar_topology", compile_or_refuse)
    with coordinate_reference_budget() as budget:
        original_basis = dict(budget.retained_objects)
        if refuse:
            with pytest.raises(ValueError, match="source schedule refused"):
                owner._prepare_source_planar_topology(np.asarray(_grid(3, 3)), budget)
            assert not budget.retained_owner_objects
        else:
            schedule = owner._prepare_source_planar_topology(np.asarray(_grid(3, 3)), budget)
            assert schedule is not None
            assert not hasattr(schedule, "records")
            profiled, storage, _, _ = retained_host_storage_profile(schedule, {}, lambda _: False)
            assert storage <= schedule.storage_bytes
            assert all(
                not isinstance(value, (owner._Preparation, owner._PreparedChains, owner._PointProfile))
                for value in profiled.values()
            )
        assert budget.retained_objects == original_basis
        # Explicit close breaks owner cycles without depending on cyclic GC.
        assert preparations and all(reference() is None for reference in preparations)
        assert chains and all(reference() is None for reference in chains)
        assert views and all(reference() is None for reference in views)
        gc.collect()
        assert all(reference() is None for reference in preparations)


def test_source_schedule_778_row_live_storage_saving(
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    rows = _grid(19, 20) + tuple(row + np.asarray((0, 20, 0)) for row in _grid(9, 1))
    sources = tuple(row[[2, 1, 0]] for row in reversed(rows))
    assert len(rows) == 778
    initialize = owner._Preparation.__init__

    def retained_initialize(self: owner._Preparation, **kwargs: Any) -> None:
        # Same arithmetic/exporter and compact schedule, with the former
        # request-long exact-basis lifetime as the independent storage control.
        kwargs.pop("basis_storage", None)
        initialize(self, **kwargs)

    def exercise(
        retained: bool, memory: int | None = None,
    ) -> tuple[tuple[bool | None, PlanarEmbedding, PlanarEmbedding], int, int]:
        limits = MeshCertificateLimits() if memory is None else MeshCertificateLimits(
            maximum_scratch_bytes=memory,
        )
        with (
            monkeypatch.context() as patch,
            coordinate_reference_budget(limits) as budget,
        ):
            if retained:
                patch.setattr(owner._Preparation, "__init__", retained_initialize)
            schedule = owner._prepare_source_planar_topology(np.asarray(sources), budget)
            assert schedule is not None
            live = budget.retained_basis_bytes + budget.temporary_bytes_upper
            with budget.live_storage() as cache, budget.live_storage() as sides:
                preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
                prepared = owner._prepare_chain_records(rows, sources, preparation)
                result = owner._chain_embedding_proofs(
                    rows, sources, prepared, preparation, 1_000_000, topology=schedule,
                )
                preparation.close()
            return result, budget.peak_bytes_upper, live

    control, before, live_before = exercise(True)
    result, after, live_after = exercise(False)
    assert result == control
    assert result[0] and result[1].complete and result[2].complete
    assert before - after > 4329
    assert live_before - live_after > 4329
    assert exercise(False, after)[0] == result
    with pytest.raises(CoordinateEnclosureResourceError):
        exercise(False, after - 1)
    record_property("source_schedule_778_peak_bytes_before", before)
    record_property("source_schedule_778_peak_bytes_after", after)
    record_property("source_schedule_778_peak_bytes_saved", before - after)
    record_property("source_schedule_778_live_bytes_saved", live_before - live_after)


def _source_key_schedule(simplices: np.ndarray) -> tuple[_PlanarTopology, int]:
    with coordinate_reference_budget() as budget:
        schedule = owner._prepare_source_planar_topology(simplices, budget)
        assert schedule is not None
        return schedule, budget.required_work_units


def _source_key_mapping(
    rows: tuple[np.ndarray, ...], simplices: np.ndarray,
    schedule: _PlanarTopology | None,
    *,
    limits: MeshCertificateLimits | None = None,
    mutate: Callable[..., Any] | None = None,
) -> tuple[tuple[int, ...] | None, int, int]:
    with (
        coordinate_reference_budget(limits or MeshCertificateLimits()) as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        sources = tuple(simplices)
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(rows, sources, preparation)
        token = owner._source_identity_operands(schedule, simplices, sources, prepared, preparation)
        if mutate is not None:
            rows, sources, prepared, preparation, token = mutate(
                rows, sources, prepared, preparation, token,
            )
        try:
            mapping = owner._chain_identity_mapping(
                rows, sources, prepared, preparation, source_keys=token,
            )
            if token is not None:
                assert token.operation.pending is None
            return mapping, budget.required_work_units, budget.peak_bytes_upper
        finally:
            preparation.close()


@pytest.mark.parametrize("kind", ("grid", "permuted", "planes", "duplicate", "disjoint"))
def test_source_identity_keys_match_fresh_mapping(kind: str) -> None:
    rows = _grid(3, 3)
    if kind == "planes":
        rows += tuple(row + (0, 0, 2) for row in _grid(2, 2))
    if kind == "duplicate":
        rows = rows[:-1] + rows[:1]
    sources = immutable_host_array(np.asarray(tuple(row[::-1] for row in reversed(rows))))
    if kind == "permuted":
        orders = tuple(permutations(range(3)))
        rows = tuple(
            rows[index][list(orders[offset % len(orders)])]
            for offset, index in enumerate(np.random.default_rng(73).permutation(len(rows)))
        )
    elif kind == "disjoint":
        rows = tuple(row + (0.25, 0, 0) for row in rows)
    schedule, _ = _source_key_schedule(sources)
    fresh = _source_key_mapping(rows, sources, None)[0]
    actual = _source_key_mapping(rows, sources, schedule)[0]
    assert actual == fresh
    assert (actual is not None) == (kind not in ("duplicate", "disjoint"))
    assert all(
        type(value) is int
        for key, indices in schedule.identity_keys.items()
        for value in (*key, *indices)
    )


@pytest.mark.parametrize("fault", (
    "source-order", "source-clone", "source-layout", "source-record", "source-fraction",
    "source-plane", "keys", "target-row", "target-record", "replay", "copied-token",
    "preparation",
))
def test_source_identity_keys_refuse_changed_roles_and_replay(fault: str) -> None:
    from copy import copy

    rows = _grid(3, 3)
    sources = immutable_host_array(np.asarray(tuple(row[::-1] for row in reversed(rows))))
    schedule, _ = _source_key_schedule(sources)

    def change(
        targets: tuple[np.ndarray, ...], current_sources: tuple[np.ndarray, ...],
        prepared: owner._PreparedChains, preparation: owner._Preparation,
        token: owner._SourceIdentityOperands | None,
    ) -> tuple[Any, ...]:
        assert token is not None
        if fault == "source-order":
            current_sources = tuple(reversed(current_sources))
        elif fault == "source-clone":
            current_sources = tuple(row.copy() for row in current_sources)
        elif fault == "source-layout":
            current_sources = (current_sources[0].reshape(9), *current_sources[1:])
            object.__setattr__(token, "rows", current_sources)
        elif fault == "source-record":
            prepared.records[id(current_sources[0])] = prepared.source[1]
        elif fault in ("source-fraction", "source-plane"):
            plane = prepared.source[0][1]
            assert plane is not None
            scalar = prepared.source[0][0][0][0] if fault == "source-fraction" else plane[0][0]
            object.__setattr__(scalar, "_numerator", scalar.numerator + 1)
        elif fault == "keys":
            object.__setattr__(
                schedule, "identity_keys",
                MappingProxyType(dict(reversed(tuple(schedule.identity_keys.items())))),
            )
        elif fault == "target-row":
            targets[0][0, 0] += 0.125
        elif fault == "target-record":
            prepared.records[id(targets[0])] = prepared.target[1]
        elif fault == "replay":
            assert owner._chain_identity_mapping(
                targets, current_sources, prepared, preparation, source_keys=token,
            ) is not None
        elif fault == "copied-token":
            token = copy(token)
        elif fault == "preparation":
            preparation = owner._Preparation(
                budget=preparation.budget, cache_storage=preparation.cache_storage,
                side_storage=preparation.cache_storage,
            )
        return targets, current_sources, prepared, preparation, token

    with pytest.raises(ValueError, match="(owners|order|once|coefficients)"):
        _source_key_mapping(rows, sources, schedule, mutate=change)


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_source_identity_keys_exact_resource_boundary(resource: str) -> None:
    rows = _grid(3, 3)
    sources = immutable_host_array(np.asarray(tuple(row[::-1] for row in reversed(rows))))
    schedule, _ = _source_key_schedule(sources)
    mapping, work, memory = _source_key_mapping(rows, sources, schedule)
    assert _source_key_mapping(
        rows, sources, schedule,
        limits=MeshCertificateLimits(maximum_work_units=work, maximum_scratch_bytes=memory),
    )[0] == mapping
    with pytest.raises(CoordinateEnclosureResourceError):
        _source_key_mapping(
            rows, sources, schedule,
            limits=MeshCertificateLimits(
                maximum_work_units=work - (resource == "work"),
                maximum_scratch_bytes=memory - (resource == "memory"),
            ),
        )


def test_source_identity_keys_778_row_work_saving(
    monkeypatch: pytest.MonkeyPatch, record_property: Callable[[str, object], None],
) -> None:
    from phydrax.geometry import _planar_chain_embedding as planar

    rows = _grid(19, 20) + tuple(row + np.asarray((0, 20, 0)) for row in _grid(9, 1))
    sources = immutable_host_array(np.asarray(tuple(row[::-1] for row in reversed(rows))))
    schedule, export_work = _source_key_schedule(sources)
    fresh, fresh_work, fresh_bytes = _source_key_mapping(rows, sources, None)
    actual, work, memory = _source_key_mapping(rows, sources, schedule)
    assert actual == fresh and actual is not None
    assert work < fresh_work

    def no_keys(*args: Any) -> Mapping[object, object]:
        return MappingProxyType({})

    with monkeypatch.context() as patch:
        patch.setattr(planar, "_compile_source_identity_keys", no_keys)
        _, prior_export_work = _source_key_schedule(sources)
    assert fresh_work + prior_export_work > work + export_work
    record_property("source_identity_778_mapping_work_saved", fresh_work - work)
    record_property("source_identity_778_export_work_added", export_work - prior_export_work)
    record_property(
        "source_identity_778_export_inclusive_work_saved",
        fresh_work + prior_export_work - work - export_work,
    )
    record_property("source_identity_778_fresh_peak_bytes", fresh_bytes)
    record_property("source_identity_778_prepared_peak_bytes", memory)
    record_property("source_identity_778_schedule_bytes", schedule.storage_bytes)


@pytest.mark.parametrize("resource", ("work", "memory"))
def test_source_identity_key_export_exact_resource_boundary(resource: str) -> None:
    simplices = immutable_host_array(np.asarray(_grid(3, 3)))

    def export(limits: MeshCertificateLimits) -> tuple[int, int]:
        with coordinate_reference_budget(limits) as budget:
            assert owner._prepare_source_planar_topology(simplices, budget) is not None
            return budget.required_work_units, budget.peak_bytes_upper

    work, memory = export(MeshCertificateLimits())
    assert export(MeshCertificateLimits(
        maximum_work_units=work, maximum_scratch_bytes=memory,
    )) == (work, memory)
    with pytest.raises(CoordinateEnclosureResourceError):
        export(MeshCertificateLimits(
            maximum_work_units=work - (resource == "work"),
            maximum_scratch_bytes=memory - (resource == "memory"),
        ))


def test_source_identity_key_refusal_consumes_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = _grid(3, 3)
    simplices = immutable_host_array(np.asarray(rows))
    schedule, _ = _source_key_schedule(simplices)
    with (
        coordinate_reference_budget() as budget,
        budget.live_storage() as cache,
        budget.live_storage() as sides,
    ):
        sources = tuple(simplices)
        preparation = owner._Preparation(budget=budget, cache_storage=cache, side_storage=sides)
        prepared = owner._prepare_chain_records(rows, sources, preparation)
        token = owner._source_identity_operands(schedule, simplices, sources, prepared, preparation)
        assert token is not None

        def refuse(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("source key admission refused")

        with monkeypatch.context() as patch:
            patch.setattr(budget, "reserve", refuse)
            with pytest.raises(RuntimeError, match="source key admission refused"):
                owner._chain_identity_mapping(
                    rows, sources, prepared, preparation, source_keys=token,
                )
        assert token.operation.pending is None
        with pytest.raises(ValueError, match="once"):
            token.consume(sources, prepared, preparation)
        preparation.close()


def test_affine_resource_owners_preserve_repeated_proofs_and_archive() -> None:
    from phydrax._model._structure import (
        model_from_array_recipe,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    rows = _grid(1, 1)
    original = SourceBoundaryChartCover(
        np.stack(rows), np.zeros(len(rows)), "certified", True, "owned", "r1",
        resource_counts=(("source_chart_work", 7, 100),),
    )
    proxies = [(row, 0.0, 0.0) for row in rows]
    limits = MeshCertificateLimits()
    first_bound, first, _ = owner.prove_affine_chart_chain(
        proxies, original, limits, 0.0, source_identity=("owned", "r1"),
    )
    proxies = [(row / 2, 0.0, 0.0) for row in _grid(2, 2)]
    independent_bound, independent, _ = owner.prove_affine_chart_chain(
        proxies, original, limits, 0.0, source_identity=("owned", "r1"),
    )
    second_bound, second, _ = owner.prove_affine_chart_chain(
        proxies, first, limits, 0.0, source_identity=("owned", "r1"),
    )
    assert first_bound == independent_bound == second_bound == 0.0
    first_counts = {
        row.metric: (row.used, row.maximum)
        for row in first.affine_chain_resources
        if row.role == "target" and row.owner_id == original.cover_id
    }
    source_counts = {
        row.metric: (row.used, row.maximum)
        for row in second.affine_chain_resources
        if row.role == "source" and row.owner_id == original.cover_id
    }
    target_counts = {
        row.metric: (row.used, row.maximum)
        for row in second.affine_chain_resources
        if row.role == "target" and row.owner_id == first.cover_id
    }
    assert source_counts == first_counts
    assert target_counts == {
        row.metric: (row.used, row.maximum)
        for row in independent.affine_chain_resources
        if row.role == "target" and row.owner_id == original.cover_id
    }
    assert target_counts["total_work"] != source_counts["total_work"]
    assert second.resource_counts == original.resource_counts
    keys = tuple(
        (row.role, row.owner_id, row.metric) for row in second.affine_chain_resources
    )
    assert keys == tuple(sorted(set(keys)))
    reordered = SourceBoundaryChartCover(
        second.simplices, second.deviation_bounds, second.semantics, second.complete,
        second.source_id, second.source_revision,
        resource_counts=second.resource_counts,
        affine_chain_resources=tuple(reversed(second.affine_chain_resources)),
    )
    assert reordered.cover_id == second.cover_id != first.cover_id
    _, repeated, _ = owner.prove_affine_chart_chain(
        proxies, first, limits, 0.0, source_identity=("owned", "r1"),
    )
    assert repeated.cover_id == second.cover_id
    _, third, _ = owner.prove_affine_chart_chain(
        proxies, second, limits, 0.0, source_identity=("owned", "r1"),
    )
    assert len(third.affine_chain_resources) == 3 * len(first_counts)
    assert {row.owner_id for row in third.affine_chain_resources if row.role == "source"} == {
        original.cover_id, first.cover_id,
    }
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(third)
    restored = model_from_array_recipe(
        recipe, model_recipe_array_values(third, recipe, prefix="chain"),
        prefix="chain",
    )
    assert isinstance(restored, SourceBoundaryChartCover)
    assert restored.cover_id == third.cover_id
    assert tuple(
        (row.role, row.owner_id, row.metric, row.used, row.maximum)
        for row in restored.affine_chain_resources
    ) == tuple(
        (row.role, row.owner_id, row.metric, row.used, row.maximum)
        for row in third.affine_chain_resources
    )
    required = target_counts["required_work"][0]
    bound, exact, _ = owner.prove_affine_chart_chain(
        proxies, first, MeshCertificateLimits(maximum_work_units=required), 0.0,
        source_identity=("owned", "r1"),
    )
    assert bound == 0.0
    assert {
        row.metric: (row.used, row.maximum)
        for row in exact.affine_chain_resources
        if row.role == "source" and row.owner_id == original.cover_id
    } == first_counts
    with pytest.raises(CoordinateEnclosureResourceError):
        owner.prove_affine_chart_chain(
            proxies, first, MeshCertificateLimits(maximum_work_units=required - 1),
            0.0, source_identity=("owned", "r1"),
        )


@pytest.mark.parametrize("fault", ("duplicate", "split-role", "missing", "two-targets"))
def test_affine_resource_cover_rejects_ambiguous_evidence(fault: str) -> None:
    rows = _grid(1, 1)
    original = SourceBoundaryChartCover(
        np.stack(rows), np.zeros(len(rows)), "certified", True, "owned", "r1",
    )
    _, proved, _ = owner.prove_affine_chart_chain(
        [(row, 0.0, 0.0) for row in rows], original, MeshCertificateLimits(), 0.0,
        source_identity=("owned", "r1"),
    )
    evidence = proved.affine_chain_resources
    first = evidence[0]
    if fault == "duplicate":
        evidence = (*evidence, first)
    elif fault == "split-role":
        evidence = (
            AffineChainResourceEvidence(
                first.metric, "source", first.owner_id, first.used, first.maximum,
            ), *evidence[1:],
        )
    elif fault == "missing":
        evidence = evidence[1:]
    else:
        evidence = (*evidence, *(
            AffineChainResourceEvidence(
                row.metric, row.role, "other-owner", row.used, row.maximum,
            ) for row in evidence
        ))
    with pytest.raises(ValueError):
        SourceBoundaryChartCover(
            original.simplices, original.deviation_bounds, "certified", True,
            "owned", "r1", affine_chain_resources=evidence,
        )


@pytest.mark.parametrize("field", ("role", "metric"))
def test_affine_resource_record_rejects_open_vocabulary(field: str) -> None:
    values: dict[str, Any] = {
        "metric": "total_work", "role": "target", "owner_id": "owner",
        "used": 1, "maximum": 2,
    }
    values[field] = "unowned"
    with pytest.raises((TypeError, ValueError)):
        AffineChainResourceEvidence(**values)


def test_affine_resource_cover_rejects_legacy_unowned_names() -> None:
    with pytest.raises(ValueError, match="structured"):
        SourceBoundaryChartCover(
            np.stack(_grid(1, 1)), np.zeros(2), "certified", True, "owned", "r1",
            resource_counts=(("affine_chain_total_work", 1, 2),),
        )


@pytest.mark.parametrize(
    ("resource", "metric"),
    (
        ("coefficient_work", "refused_coefficient_work"),
        ("polynomial_storage", "refused_polynomial_storage"),
        ("retained_basis", "refused_retained_basis"),
    ),
)
def test_affine_resource_refusal_preserves_owned_phase_counts(
    resource: CoordinateEnclosureResource, metric: str,
) -> None:
    from phydrax.geometry._source_fidelity import _chart_refusal_cover

    original = SourceBoundaryChartCover(
        np.stack(_grid(1, 1)), np.zeros(2), "certified", True, "owned", "r1",
        restriction_ids=("restriction",),
    )
    first = _chart_refusal_cover(
        original, CoordinateEnclosureResourceError(resource, 10, 12, 7),
        MeshCertificateLimits(),
    )
    second = _chart_refusal_cover(
        first, CoordinateEnclosureResourceError(resource, 20, 23, 17),
        MeshCertificateLimits(),
    )
    assert second.restriction_ids == original.restriction_ids
    assert tuple(
        (row.role, row.owner_id, row.metric, row.used, row.maximum)
        for row in second.affine_chain_resources
    ) == (
        ("source", original.cover_id, metric, 7, 10),
        ("target", first.cover_id, metric, 17, 20),
    )
