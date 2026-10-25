#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import numpy as np
import numpy.typing as npt
import pytest

from phydrax._data_plane import ExecutionGroupRank
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._jax_config import x64_enabled
from phydrax.atomistic import atomistic_training_loader
from phydrax.atomistic._structure_store import (
    AbstractAtomisticStructureSource,
    AtomisticStoreField,
    AtomisticStoreIndex,
    AtomisticStoreIndexBasis,
    AtomisticStoreLevel,
    AtomisticStructureBatch,
    AtomisticStructureRecord,
    AtomisticStructureSource,
    AtomisticStructureStoreSchema,
)
from phydrax.atomistic._training_data import (
    AtomisticFieldBinding,
    AtomisticLevelCapacity,
    AtomisticLevelLayout,
    AtomisticLoaderCursor,
    AtomisticLoaderError,
    AtomisticLoaderItem,
    AtomisticLoaderPayload,
    AtomisticLoaderPlacement,
    AtomisticLoaderStaging,
    AtomisticMixtureComponent,
    AtomisticSourceMixture,
    AtomisticSplitEvidence,
    AtomisticStructureLoader,
    AtomisticWorkloadBucket,
    AtomisticWorkloadBuckets,
)
from phydrax.axes import Axis, AxisKey
from phydrax.units import (
    ANGSTROM,
    BOHR,
    conversion_factor,
    derived_unit,
    ELECTRONVOLT,
    ELEMENTARY_CHARGE,
    KELVIN,
)


CARTESIAN = Axis(AxisKey("atomistic", "cartesian"), 3, ("x", "y", "z"))
ENDPOINT = Axis(AxisKey("atomistic", "edge-endpoint"), 2, ("sender", "receiver"))
FORCE = derived_unit("eV/angstrom", ((ELECTRONVOLT, 1), (ANGSTROM, -1)))
CURVATURE = derived_unit("eV/angstrom^2", ((ELECTRONVOLT, 1), (ANGSTROM, -2)))
DERIVATION = "minimum-image-radius-3-angstrom"
FULL_LEVELS = ("atom", "molecule", "residue", "edge", "hessian_block")
ALPHA = ((2, (2,)), (3, (1, 2)), (1, ()), (2, (1,)))
BETA = ((3, (1,)), (1, ()), (2, (2,)))


def _schema() -> AtomisticStructureStoreSchema:
    levels = (
        AtomisticStoreLevel("structure", "structure"),
        AtomisticStoreLevel("atom", "atom", parent="structure"),
        AtomisticStoreLevel("molecule", "group", parent="structure"),
        AtomisticStoreLevel("residue", "segment", parent="molecule"),
        AtomisticStoreLevel("edge", "edge", parent="structure", derivation_id=DERIVATION),
        AtomisticStoreLevel(
            "hessian_block",
            "product",
            factors=("atom", "atom"),
            maximum_rows_per_structure=9,
        ),
    )
    fields = (
        AtomisticStoreField(
            "energy",
            "structure",
            "float64",
            kind="continuous",
            role="label",
            convention="total-energy",
            unit=ELECTRONVOLT,
            extensivity="extensive",
            availability="optional",
        ),
        AtomisticStoreField(
            "positions",
            "atom",
            "float64",
            kind="continuous",
            role="geometry",
            convention="cartesian",
            axes=(CARTESIAN,),
            unit=ANGSTROM,
            extensivity="intensive",
        ),
        AtomisticStoreField(
            "species",
            "atom",
            "int16",
            kind="categorical",
            role="identity",
            convention="atomic-number",
        ),
        AtomisticStoreField(
            "forces",
            "atom",
            "float64",
            kind="continuous",
            role="label",
            convention="cartesian",
            axes=(CARTESIAN,),
            unit=FORCE,
            extensivity="intensive",
            availability="optional",
        ),
        AtomisticStoreField(
            "molecule",
            "atom",
            "int64",
            kind="index",
            role="identity",
            convention="molecule-membership",
            index=AtomisticStoreIndex("molecule", sentinel=-1),
        ),
        AtomisticStoreField(
            "residue_code",
            "residue",
            "int32",
            kind="categorical",
            role="identity",
            convention="test-residue-code",
        ),
        AtomisticStoreField(
            "endpoints",
            "edge",
            "int32",
            kind="index",
            role="derived",
            convention="sender-receiver",
            axes=(ENDPOINT,),
            index=AtomisticStoreIndex("atom"),
        ),
        AtomisticStoreField(
            "hessian",
            "hessian_block",
            "float64",
            kind="continuous",
            role="label",
            convention="cartesian-atom-pair-block",
            axes=(CARTESIAN, CARTESIAN),
            unit=CURVATURE,
            extensivity="intensive",
            availability="optional",
        ),
    )
    return AtomisticStructureStoreSchema(levels, fields)


def _record(
    schema: AtomisticStructureStoreSchema,
    prefix: str,
    offset: int,
    index: int,
    atoms: int,
    residues: Sequence[int],
    *,
    energy: bool = True,
    nan_force: bool = False,
    lineage: str | None = None,
) -> AtomisticStructureRecord:
    molecules = len(residues)
    forces_available = np.arange(atoms) % 2 == 0
    forces = np.arange(atoms * 3, dtype=np.float64).reshape(atoms, 3) * 0.25 + offset
    forces[~forces_available] = 0.0
    if nan_force:
        forces[0, 0] = np.nan
    values: dict[str, Any] = {
        "positions": offset
        + 10.0 * index
        + 0.5 * np.arange(atoms * 3, dtype=np.float64).reshape(atoms, 3),
        "species": (np.arange(atoms, dtype=np.int16) % 5) + 1,
        "forces": forces,
        "molecule": (
            np.full(atoms, -1, dtype=np.int64)
            if molecules == 0
            else np.arange(atoms, dtype=np.int64) % molecules
        ),
        "residue_code": np.arange(sum(residues), dtype=np.int32) + 100 + offset,
        "endpoints": np.stack(
            (np.arange(atoms), (np.arange(atoms) + 1) % atoms), axis=-1
        ).astype(np.int32),
        "hessian": np.full((atoms * atoms, 3, 3), offset + index, dtype=np.float64),
    }
    if energy:
        values["energy"] = np.asarray([-1.5 * (index + 1) - offset], dtype=np.float64)
    return AtomisticStructureRecord(
        schema,
        record_id=f"{prefix}-{index}",
        lineage_id=f"{prefix}-lineage-{index}" if lineage is None else lineage,
        counts={
            "atom": [atoms],
            "edge": [atoms],
            "molecule": [molecules],
            "residue": np.asarray(residues, dtype=np.int64),
        },
        stable_ids=np.arange(atoms, dtype=np.int64) * 7 + 1000 * index + 100000 * offset,
        values=values,
        available={"forces": forces_available},
    )


def _source(
    prefix: str,
    offset: int,
    shapes: Sequence[tuple[int, Sequence[int]]],
    overrides: dict[int, dict[str, Any]] | None = None,
) -> AtomisticStructureSource:
    schema = _schema()
    return AtomisticStructureSource(
        schema,
        tuple(
            _record(
                schema,
                prefix,
                offset,
                index,
                atoms,
                residues,
                **(overrides or {}).get(index, {}),
            )
            for index, (atoms, residues) in enumerate(shapes)
        ),
    )


def _alpha(
    overrides: dict[int, dict[str, Any]] | None = None,
) -> AtomisticStructureSource:
    return _source("alpha", 0, ALPHA, overrides)


def _beta() -> AtomisticStructureSource:
    return _source("beta", 100, BETA)


class _RecordingSource(AbstractAtomisticStructureSource):
    """Delegating source recording projections and injecting one read failure."""

    schema: AtomisticStructureStoreSchema
    source_id: str = eqx.field(static=True)
    record_ids: tuple[str, ...] = eqx.field(static=True)
    lineage_ids: tuple[str, ...] = eqx.field(static=True)
    record_digests: tuple[str, ...] = eqx.field(static=True)
    inner: AtomisticStructureSource
    reads: list[tuple[str, ...] | None]
    fail_after: int | None = eqx.field(static=True)
    safe: bool = eqx.field(static=True)

    def __init__(
        self,
        inner: AtomisticStructureSource,
        /,
        *,
        fail_after: int | None = None,
        safe: bool = True,
    ) -> None:
        self.schema = inner.schema
        self.source_id = inner.source_id
        self.record_ids = inner.record_ids
        self.lineage_ids = inner.lineage_ids
        self.record_digests = inner.record_digests
        self.inner = inner
        self.reads = []
        self.fail_after = fail_after
        self.safe = safe

    @property
    def background_read_safe(self) -> bool:
        return self.safe

    def level_counts(self, level: str, /) -> npt.NDArray[np.int64]:
        return self.inner.level_counts(level)

    def read(
        self,
        ordinals: Any,
        /,
        *,
        fields: Sequence[str] | None = None,
        index_basis: AtomisticStoreIndexBasis = "structure-local",
    ) -> AtomisticStructureBatch:
        self.reads.append(None if fields is None else tuple(fields))
        if self.fail_after is not None and len(self.reads) > self.fail_after:
            raise RuntimeError("injected read failure")
        return self.inner.read(ordinals, fields=fields, index_basis=index_basis)


def _bindings(
    *, positions_unit: Any = ANGSTROM, **positions: Any
) -> tuple[AtomisticFieldBinding, ...]:
    return (
        AtomisticFieldBinding(
            "energy",
            "energy",
            level_kind="structure",
            value_kind="continuous",
            role="label",
            convention="total-energy",
            unit=ELECTRONVOLT,
        ),
        AtomisticFieldBinding(
            "positions",
            "positions",
            level_kind="atom",
            value_kind="continuous",
            role="geometry",
            convention="cartesian",
            axes=(CARTESIAN,),
            unit=positions_unit,
            **positions,
        ),
        AtomisticFieldBinding(
            "species",
            "species",
            level_kind="atom",
            value_kind="categorical",
            role="identity",
            convention="atomic-number",
        ),
        AtomisticFieldBinding(
            "forces",
            "forces",
            level_kind="atom",
            value_kind="continuous",
            role="label",
            convention="cartesian",
            axes=(CARTESIAN,),
            unit=FORCE,
        ),
        AtomisticFieldBinding(
            "molecule",
            "molecule",
            level_kind="atom",
            value_kind="index",
            role="identity",
            convention="molecule-membership",
        ),
        AtomisticFieldBinding(
            "residue_code",
            "residue_code",
            level_kind="segment",
            value_kind="categorical",
            role="identity",
            convention="test-residue-code",
        ),
        _endpoints(),
        AtomisticFieldBinding(
            "hessian",
            "hessian",
            level_kind="product",
            value_kind="continuous",
            role="label",
            convention="cartesian-atom-pair-block",
            axes=(CARTESIAN, CARTESIAN),
            unit=CURVATURE,
        ),
    )


def _endpoints(derivation: str | None = DERIVATION) -> AtomisticFieldBinding:
    return AtomisticFieldBinding(
        "endpoints",
        "endpoints",
        level_kind="edge",
        value_kind="index",
        role="derived",
        convention="sender-receiver",
        axes=(ENDPOINT,),
        derivation_id=derivation,
    )


def _bucket(
    name: str = "main",
    *,
    cases: int = 2,
    atoms: int = 3,
    molecules: int = 2,
    residues: int = 3,
    edges: int = 3,
    edge_layout: AtomisticLevelLayout = "per-case",
    levels: Sequence[str] = FULL_LEVELS,
    max_bytes: int = 1 << 24,
) -> AtomisticWorkloadBucket:
    rows: dict[str, tuple[AtomisticLevelLayout, int]] = {
        "atom": ("per-case", atoms),
        "molecule": ("per-case", molecules),
        "residue": ("per-case", residues),
        "edge": (edge_layout, edges),
        "hessian_block": ("per-case", atoms * atoms),
    }
    return AtomisticWorkloadBucket(
        name,
        cases=cases,
        levels=tuple(
            AtomisticLevelCapacity(level, layout=rows[level][0], rows=rows[level][1])
            for level in levels
        ),
        max_bytes=max_bytes,
    )


def _mixture(
    *sources: AbstractAtomisticStructureSource,
    weights: Sequence[int | float] | None = None,
    draw: Any = "epoch",
    objective: Any = "sampled",
    seed: int = 7,
    **options: Any,
) -> AtomisticSourceMixture:
    parts = sources or (_alpha(), _beta())
    return AtomisticSourceMixture(
        tuple(
            AtomisticMixtureComponent(source, weight=1 if weights is None else weights[i])
            for i, source in enumerate(parts)
        ),
        draw=draw,
        objective=objective,
        seed=seed,
        **options,
    )


def _loader(
    mixture: AtomisticSourceMixture | None = None,
    *,
    buckets: Sequence[AtomisticWorkloadBucket] | None = None,
    bindings: Sequence[AtomisticFieldBinding] | None = None,
    batch_size: int = 3,
    **options: Any,
) -> AtomisticStructureLoader:
    return AtomisticStructureLoader(
        _mixture() if mixture is None else mixture,
        _bindings() if bindings is None else bindings,
        AtomisticWorkloadBuckets((_bucket(),) if buckets is None else buckets),
        batch_size=batch_size,
        stream=options.pop("stream", "train"),
        **options,
    )


def _items(
    loader: AtomisticStructureLoader,
    cursor: AtomisticLoaderCursor | None = None,
    **options: Any,
) -> list[AtomisticLoaderItem]:
    with loader.epoch(loader.start() if cursor is None else cursor, **options) as epoch:
        return list(epoch)


def _valid_draws(items: Sequence[AtomisticLoaderItem]) -> list[tuple[str, str, int]]:
    return [
        (source, record, draw)
        for item in items
        for source, record, draw, valid in zip(
            item.provenance.source_ids,
            item.provenance.record_ids,
            item.provenance.draw_indices,
            item.provenance.valid,
        )
        if valid
    ]


def _leaves(payload: AtomisticLoaderPayload) -> list[np.ndarray]:
    return [np.asarray(leaf) for leaf in jax.tree.leaves(payload)]


def _assert_items_equal(
    left: Sequence[AtomisticLoaderItem], right: Sequence[AtomisticLoaderItem]
) -> None:
    assert len(left) == len(right)
    for first, second in zip(left, right):
        assert first.provenance == second.provenance
        assert first.cursor.cursor_id == second.cursor.cursor_id
        assert first.payload.layout.layout_id == second.payload.layout.layout_id
        for a, b in zip(_leaves(first.payload), _leaves(second.payload), strict=True):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert a.tobytes() == b.tobytes()


def _assert_payload_safe(payload: AtomisticLoaderPayload) -> None:
    layout = payload.layout
    cases = np.asarray(payload.case_valid).size
    valid = np.asarray(payload.case_valid)
    assert np.all(np.asarray(payload.case_weight)[~valid] == 0.0)
    rows = dict(zip(layout.levels, layout.level_rows))
    for level in payload.levels:
        owners = np.asarray(level.row_case)
        assert np.all((owners >= 0) & (owners < cases))
        if level.row_parent is not None:
            parents = np.asarray(level.row_parent)
            parent = layout.level_parents[layout.levels.index(level.level)]
            assert parent is not None
            assert np.all((parents >= 0) & (parents < rows[parent]))
    atoms = payload.level("atom")
    occupied = np.bincount(
        np.asarray(atoms.row_case)[np.asarray(atoms.row_mask)], minlength=cases
    )
    assert np.all(occupied >= 1), "every case, padding included, owns real atoms"
    assert np.all(np.isfinite(np.asarray(payload.value("positions"))))
    for slot, target, sentinel in (
        ("molecule", "molecule", -1),
        ("endpoints", "atom", None),
    ):
        if slot not in layout.slots:
            continue
        values = np.asarray(payload.value(slot))
        live = values != sentinel if sentinel is not None else np.ones(values.shape, bool)
        assert np.all((values[live] >= 0) & (values[live] < rows[target]))


# --------------------------------------------------------------- epochs/mixtures


def test_complete_epoch_visits_every_record_once_in_plan_order() -> None:
    loader = _loader()
    items = _items(loader)
    draws = _valid_draws(items)
    expected = {
        (source.source_id, record)
        for source in (_alpha(), _beta())
        for record in source.record_ids
    }
    assert len(draws) == 7
    assert {(source, record) for source, record, _ in draws} == expected
    order = [position for batch in loader.epoch_plan(0) for position in batch]
    assert [draw for _, _, draw in draws] == order
    assert sorted(order) == list(range(7))
    _assert_items_equal(items, _items(_loader()))
    epoch_one = _items(loader, loader.start(1))
    assert sorted(draw for _, _, draw in _valid_draws(epoch_one)) == list(range(7))
    assert {item.provenance.epoch for item in epoch_one} == {1}
    for item in items:
        _assert_payload_safe(item.payload)
    assert [item.closes_batch for item in items].count(True) == loader.batch_count == 3
    assert items[-1].cursor.cursor_id == loader.start(1).cursor_id


def test_repeated_draws_keep_record_identity_and_importance_weights() -> None:
    mixture = _mixture(weights=(2, 1), objective="importance-corrected")
    assert mixture.draws_per_epoch == 2 * 4 + 3
    assert mixture.record_probabilities == (2 / 11, 1 / 11)
    assert mixture.case_weights == pytest.approx(((1 / 7) / (2 / 11), (1 / 7) / (1 / 11)))
    items = _items(_loader(mixture))
    draws = _valid_draws(items)
    assert len({draw for _, _, draw in draws}) == len(draws) == 11
    alpha = _alpha()
    for record in alpha.record_ids:
        assert len([r for s, r, _ in draws if r == record and s == alpha.source_id]) == 2
    for item in items:
        payload = item.payload
        valid = np.asarray(payload.case_valid)
        component = np.asarray(payload.case_component)
        np.testing.assert_allclose(
            np.asarray(payload.case_weight)[valid],
            np.asarray(mixture.case_weights)[component[valid]],
        )
        np.testing.assert_allclose(
            np.asarray(payload.draw_probability),
            np.asarray(mixture.record_probabilities)[component],
        )


def test_mixture_refuses_uncovered_targets_and_duplicate_membership() -> None:
    with pytest.raises(AtomisticLoaderError) as uncovered:
        _mixture(weights=(1, 0), objective="importance-corrected")
    assert uncovered.value.failure == "mixture"
    sampled = _mixture(weights=(1, 0))
    assert sampled.draws_per_epoch == 4
    with pytest.raises(AtomisticLoaderError) as repeated:
        _mixture(_alpha(), _alpha())
    assert repeated.value.failure == "mixture"
    with pytest.raises(AtomisticLoaderError):
        _mixture(weights=(1.5, 1))
    with pytest.raises(ValueError):
        _mixture(draw="replacement", draws_per_epoch=4, shuffle=True)


def test_replacement_draws_are_addressed_by_epoch_and_draw() -> None:
    mixture = _mixture(weights=(3.0, 1.0), draw="replacement", draws_per_epoch=400)
    assert mixture.record_probabilities == pytest.approx((0.75 / 4, 0.25 / 3))
    assert mixture.prng_implementation is not None
    draws = np.arange(400)
    components, ordinals = mixture.resolve(0, draws)
    again = mixture.resolve(0, draws)
    np.testing.assert_array_equal(components, again[0])
    np.testing.assert_array_equal(ordinals, again[1])
    single = mixture.resolve(0, np.asarray([17]))
    assert (single[0][0], single[1][0]) == (components[17], ordinals[17])
    assert 0.65 < float(np.mean(components == 0)) < 0.85
    counts = np.asarray(mixture.record_counts)
    assert np.all((ordinals >= 0) & (ordinals < counts[components]))
    later = mixture.resolve(1, draws)
    assert not np.array_equal(np.stack(later), np.stack((components, ordinals)))
    items = _items(_loader(mixture, batch_size=50))
    observed = _valid_draws(items)
    assert sorted(draw for _, _, draw in observed) == list(range(400))
    records = [record for _, record, _ in observed]
    assert len(set(records)) < len(records)


# ------------------------------------------------------------------- padding


def test_short_tail_is_mask_safe_and_drop_remainder_refuses_partial_batch() -> None:
    mixture = _mixture(_alpha(), weights=(1,))
    loader = _loader(mixture, batch_size=3, buckets=(_bucket(cases=3),))
    items = _items(loader)
    assert loader.batch_count == 2
    tail = items[-1]
    assert tail.provenance.global_valid_draws == 1
    valid = np.asarray(tail.payload.case_valid)
    assert valid.tolist() == [True, False, False]
    assert np.asarray(tail.payload.case_weight).tolist() == [1.0, 0.0, 0.0]
    record = tail.provenance.record_ids[0]
    assert tail.provenance.record_ids == (record, record, record)
    atoms = tail.payload.level("atom")
    positions = np.asarray(tail.payload.value("positions")).reshape(3, 3, 3)
    mask = np.asarray(atoms.row_mask).reshape(3, 3)
    np.testing.assert_array_equal(mask[1], mask[0])
    np.testing.assert_array_equal(positions[1][mask[1]], positions[0][mask[0]])
    support = np.asarray(tail.payload.support("forces")).reshape(3, 3)
    assert not support[1:].any()
    _assert_payload_safe(tail.payload)
    dropped = _loader(
        mixture, batch_size=3, remainder="drop", buckets=(_bucket(cases=3),)
    )
    assert dropped.batch_count == 1
    assert len(_valid_draws(_items(dropped))) == 3
    with pytest.raises(AtomisticLoaderError):
        _loader(mixture, batch_size=5, remainder="drop")


def test_labels_keep_availability_support_and_nonfinite_values() -> None:
    source = _alpha({1: {"nan_force": True}, 2: {"energy": False}})
    loader = _loader(_mixture(source), batch_size=4, buckets=(_bucket(cases=4),))
    (item,) = _items(loader)
    payload = item.payload
    by_record = {record: case for case, record in enumerate(item.provenance.record_ids)}
    energy_available = np.asarray(payload.availability("energy"))
    assert not energy_available[by_record["alpha-2"]]
    assert energy_available[by_record["alpha-0"]]
    forces = np.asarray(payload.value("forces")).reshape(4, 3, 3)
    available = np.asarray(payload.availability("forces")).reshape(4, 3)
    nan_case = by_record["alpha-1"]
    assert np.isnan(forces[nan_case, 0, 0]) and available[nan_case, 0]
    assert not available[nan_case, 1] and available[nan_case, 2]
    mask = np.asarray(payload.level("atom").row_mask).reshape(4, 3)
    assert not np.any(available & ~mask)
    molecule = np.asarray(payload.value("molecule")).reshape(4, 3)
    empty_case = by_record["alpha-2"]
    assert molecule[empty_case, 0] == -1


# ------------------------------------------------------------------- bindings


def test_integer_physical_conditions_keep_their_unit_and_exact_values() -> None:
    schema = AtomisticStructureStoreSchema(
        (
            AtomisticStoreLevel("structure", "structure"),
            AtomisticStoreLevel("atom", "atom", parent="structure"),
        ),
        (
            AtomisticStoreField(
                "charge",
                "structure",
                "int64",
                kind="integer",
                role="condition",
                convention="total-charge",
                unit=ELEMENTARY_CHARGE,
                extensivity="extensive",
            ),
        ),
    )
    source = AtomisticStructureSource(
        schema,
        tuple(
            AtomisticStructureRecord(
                schema,
                record_id=f"charge-{index}",
                lineage_id=f"charge-{index}",
                counts={"atom": [1]},
                stable_ids=np.asarray([100 + index], dtype=np.int64),
                values={"charge": np.asarray([charge], dtype=np.int64)},
            )
            for index, charge in enumerate((1, -1))
        ),
    )
    (loaded,) = _items(atomistic_training_loader(source, shuffle=False))
    charges = np.asarray(loaded.payload.value("charge"))
    assert charges.dtype == np.dtype(np.int64)
    np.testing.assert_array_equal(charges, np.asarray([1, -1], dtype=np.int64))
    np.testing.assert_array_equal(loaded.payload.support("charge"), True)


def test_integer_physical_binding_refuses_unit_conversion() -> None:
    with pytest.raises(ValueError):
        AtomisticFieldBinding(
            "charge",
            "charge",
            level_kind="structure",
            value_kind="integer",
            role="condition",
            convention="total-charge",
            unit=ELEMENTARY_CHARGE,
            convert_from=ELEMENTARY_CHARGE,
        )


def test_unit_and_semantic_mismatches_are_refused() -> None:
    with pytest.raises(AtomisticLoaderError) as unit:
        _loader(bindings=_bindings(positions_unit=BOHR))
    assert unit.value.failure == "binding"
    converted = _loader(
        bindings=_bindings(positions_unit=BOHR, convert_from=ANGSTROM),
        batch_size=7,
        buckets=(_bucket(cases=7),),
    )
    reference = _loader(batch_size=7, buckets=(_bucket(cases=7),))
    (scaled,) = _items(converted)
    (plain,) = _items(reference)
    np.testing.assert_allclose(
        np.asarray(scaled.payload.value("positions")),
        np.asarray(plain.payload.value("positions"))
        * float(conversion_factor(ANGSTROM, BOHR)),
    )
    with pytest.raises(AtomisticLoaderError):
        _loader(bindings=_bindings(positions_unit=KELVIN, convert_from=ANGSTROM))
    bindings = list(_bindings())
    mismatched = [
        AtomisticFieldBinding(
            "energy",
            "energy",
            level_kind="structure",
            value_kind="continuous",
            role="label",
            convention="formation-energy",
            unit=ELECTRONVOLT,
        ),
        AtomisticFieldBinding(
            "energy",
            "energy",
            level_kind="atom",
            value_kind="continuous",
            role="label",
            convention="total-energy",
            unit=ELECTRONVOLT,
        ),
        AtomisticFieldBinding(
            "energy",
            "energy",
            level_kind="structure",
            value_kind="continuous",
            role="condition",
            convention="total-energy",
            unit=ELECTRONVOLT,
        ),
    ]
    for binding in mismatched:
        with pytest.raises(AtomisticLoaderError) as error:
            _loader(bindings=(binding, *bindings[1:]))
        assert error.value.failure == "binding"
    with pytest.raises(ValueError):
        AtomisticFieldBinding(
            "species",
            "species",
            level_kind="atom",
            value_kind="categorical",
            role="identity",
            convention="atomic-number",
            dtype="float32",
        )
    narrowed = _loader(bindings=_bindings(dtype="float32"))
    assert np.asarray(_items(narrowed)[0].payload.value("positions")).dtype == np.float32


def test_projected_reads_and_derived_edge_identity() -> None:
    recording = _RecordingSource(_alpha())
    projected = (
        _bindings()[1],
        _endpoints(),
    )
    loader = _loader(
        _mixture(recording),
        bindings=projected,
        buckets=(_bucket(levels=("atom", "edge")),),
        batch_size=4,
    )
    assert loader.levels == ("atom", "edge")
    items = _items(loader)
    assert set(recording.reads) == {("positions", "endpoints")}
    source = _alpha()
    for item in items:
        payload = item.payload
        edges = payload.level("edge")
        atoms = payload.level("atom")
        endpoints = np.asarray(payload.value("endpoints"))
        owners = np.asarray(edges.row_case)
        live = np.asarray(edges.row_mask)
        atom_case = np.asarray(atoms.row_case)
        assert np.all(atom_case[endpoints[live]] == owners[live][:, None])
        assert np.all(np.asarray(atoms.row_mask)[endpoints[live]])
        for case, (record, valid) in enumerate(
            zip(item.provenance.record_ids, item.provenance.valid)
        ):
            if not valid:
                continue
            batch = source.read([source.record_ids.index(record)])
            count = int(batch.structure_counts["atom"][0])
            np.testing.assert_array_equal(
                np.asarray(payload.value("positions"))[case * 3 : case * 3 + count],
                batch.values["positions"],
            )
        _assert_payload_safe(payload)
    for derivation in ("minimum-image-radius-4-angstrom", None):
        with pytest.raises(AtomisticLoaderError) as error:
            _loader(
                bindings=(_bindings()[1], _endpoints(derivation)),
                buckets=(_bucket(levels=("atom", "edge")),),
            )
        assert error.value.failure == "binding"
    with pytest.raises(AtomisticLoaderError):
        _loader(bindings=projected)


def test_segment_rows_bind_parent_groups() -> None:
    loader = _loader(batch_size=7, buckets=(_bucket(cases=7),))
    (item,) = _items(loader)
    payload = item.payload
    residues = payload.level("residue")
    molecules = payload.level("molecule")
    live = np.asarray(residues.row_mask)
    parents = np.asarray(residues.row_parent)
    assert np.all(np.asarray(molecules.row_mask)[parents[live]])
    np.testing.assert_array_equal(
        np.asarray(molecules.row_case)[parents[live]], np.asarray(residues.row_case)[live]
    )
    hessian = payload.level("hessian_block")
    np.testing.assert_array_equal(
        np.asarray(hessian.case_count), np.asarray(payload.level("atom").case_count) ** 2
    )


# -------------------------------------------------------------------- buckets


def test_oversize_records_route_to_explicit_larger_bucket_or_refuse() -> None:
    small = _bucket("small", atoms=2, molecules=1, residues=2, edges=2)
    large = _bucket("large")
    items = _items(_loader(buckets=(small, large), batch_size=7))
    seen = {
        record: item.provenance.bucket
        for item in items
        for record, valid in zip(item.provenance.record_ids, item.provenance.valid)
        if valid
    }
    assert seen == {
        "alpha-0": "small",
        "alpha-1": "large",
        "alpha-2": "small",
        "alpha-3": "small",
        "beta-0": "large",
        "beta-1": "small",
        "beta-2": "small",
    }
    assert {item.payload.layout.bucket for item in items} == {"small", "large"}
    with pytest.raises(AtomisticLoaderError) as refused:
        _loader(buckets=(small,))
    assert refused.value.failure == "capacity"
    assert "alpha-1" in str(refused.value)
    with pytest.raises(AtomisticLoaderError) as bytes_error:
        _loader(buckets=(_bucket(max_bytes=64),))
    assert bytes_error.value.failure == "capacity"
    with pytest.raises(AtomisticLoaderError) as levels_error:
        _loader(buckets=(_bucket(levels=("atom", "edge")),))
    assert levels_error.value.failure == "binding"


def test_atom_only_edge_only_and_combined_budgets() -> None:
    atom_only = _loader(buckets=(_bucket(cases=4),), batch_size=4)
    edge_only = _loader(
        buckets=(_bucket(cases=4, edges=3, edge_layout="packed"),), batch_size=4
    )
    combined = _loader(
        buckets=(_bucket(cases=2, edges=4, edge_layout="packed"),), batch_size=4
    )
    assert atom_only.payload_count(0, 0) == 1
    for loader in (edge_only, combined):
        assert loader.payload_count(0, 0) >= atom_only.payload_count(0, 0)
        items = _items(loader)
        assert sorted(draw for _, _, draw in _valid_draws(items)) == list(range(7))
        capacity = 3 if loader is edge_only else 4
        for item in items:
            edges = item.payload.level("edge")
            assert edges.layout == "packed"
            assert int(np.sum(np.asarray(edges.row_mask))) <= capacity
            assert np.asarray(edges.row_mask).size == capacity
            _assert_payload_safe(item.payload)
    assert edge_only.payload_count(0, 0) > 1


# -------------------------------------------------------------------- staging


def test_prefetch_matches_synchronous_order_and_propagates_failure() -> None:
    loader = _loader(buckets=(_bucket(cases=2, edges=3, edge_layout="packed"),))
    synchronous = _items(loader)
    staged = loader.epoch(loader.start(), staging=AtomisticLoaderStaging(host_prefetch=2))
    assert staged.host_prefetch == 2
    with staged:
        _assert_items_equal(list(staged), synchronous)
    unsafe = _loader(_mixture(_RecordingSource(_alpha(), safe=False), _beta()))
    epoch = unsafe.epoch(unsafe.start(), staging=AtomisticLoaderStaging(host_prefetch=2))
    assert epoch.host_prefetch == 0
    epoch.close()
    failing = _loader(_mixture(_RecordingSource(_alpha(), fail_after=1)), batch_size=1)
    epoch = failing.epoch(
        failing.start(), staging=AtomisticLoaderStaging(host_prefetch=2)
    )
    with pytest.raises(RuntimeError, match="injected read failure"):
        list(epoch)
    assert epoch.closed
    with pytest.raises(StopIteration):
        next(epoch)


def test_close_stops_iteration_and_device_staging_is_bounded() -> None:
    loader = _loader()
    epoch = loader.epoch(loader.start(), staging=AtomisticLoaderStaging(host_prefetch=1))
    next(epoch)
    epoch.close()
    assert epoch.closed
    with pytest.raises(StopIteration):
        next(epoch)
    layout_bytes = max(layout.nbytes for layout in loader.layouts)
    with pytest.raises(AtomisticLoaderError) as bounded:
        loader.epoch(
            loader.start(),
            staging=AtomisticLoaderStaging(
                target="device", device_ahead=1, max_device_bytes=layout_bytes
            ),
        )
    assert bounded.value.failure == "capacity"
    staging = AtomisticLoaderStaging(
        target="device", device_ahead=1, max_device_bytes=2 * layout_bytes
    )
    if not x64_enabled():
        with pytest.raises(AtomisticLoaderError):
            _items(loader, staging=staging)
        return
    device_items = _items(loader, staging=staging)
    host_items = _items(loader)
    assert all(
        isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(device_items[0].payload)
    )
    _assert_items_equal(device_items, host_items)


# --------------------------------------------------------------------- cursor


def test_interrupted_resume_equals_uninterrupted_epoch() -> None:
    def build() -> AtomisticStructureLoader:
        return _loader(
            buckets=(_bucket(cases=2, edges=3, edge_layout="packed"),), batch_size=4
        )

    loader = build()
    full = _items(loader)
    mid = next(index + 1 for index, item in enumerate(full[:-1]) if not item.closes_batch)
    assert full[mid - 1].cursor.payload_index > 0
    for split in (1, mid, len(full) - 1):
        payload = full[split - 1].cursor.to_payload()
        cursor = AtomisticLoaderCursor.from_payload(payload)
        assert cursor == full[split - 1].cursor
        resumed = _items(build(), cursor, staging=AtomisticLoaderStaging(host_prefetch=2))
        _assert_items_equal(resumed, full[split:])
    entry = full[0].cursor.composition_entry(entry_id="data/cursor", owner_id="training")
    assert entry.role == "history"
    assert entry.revision_id == full[0].cursor.cursor_id
    assert entry.structure_id == loader.structure_id
    assert entry.semantics_id == loader.semantics_id


def test_cursor_refuses_source_drift_and_policy_or_layout_change() -> None:
    loader = _loader()
    cursor = _items(loader)[0].cursor
    drifted = _loader(_mixture(_alpha({3: {"energy": False}}), _beta()))
    with pytest.raises(AtomisticLoaderError) as drift:
        drifted.require_cursor(cursor)
    assert drift.value.failure == "drift"
    assert "[0]" in str(drift.value)
    for changed in (
        _loader(batch_size=2),
        _loader(stream="validation"),
        _loader(buckets=(_bucket(cases=3),)),
        _loader(_mixture(seed=8)),
        _loader(
            placement=AtomisticLoaderPlacement(
                ExecutionGroupRank(
                    ExecutionGroupSpec("loader-pair", (0,), ((0, 0), (0, 1))), 0
                )
            )
        ),
    ):
        with pytest.raises(AtomisticLoaderError) as error:
            changed.require_cursor(cursor)
        assert error.value.failure == "cursor"
    tampered = {**cursor.to_payload(), "epoch": 3}
    with pytest.raises(AtomisticLoaderError) as forged:
        AtomisticLoaderCursor.from_payload(tampered)
    assert forged.value.failure == "cursor"


# ------------------------------------------------------------------ placement


def test_single_device_and_explicit_subgroup_placements_share_draws() -> None:
    spec = ExecutionGroupSpec("loader-subgroup", (3, 5), ((3, 0), (3, 1), (5, 2), (5, 3)))
    placements = tuple(
        AtomisticLoaderPlacement(ExecutionGroupRank(spec, process)) for process in (3, 5)
    )
    for expected, placement in enumerate(placements):
        assert placement.rank is not None
        assert placement.rank.rank == expected
    assert [p.local_shards for p in placements] == [(0, 1), (2, 3)]
    assert placements[0].placement_id == placements[1].placement_id
    assert placements[0].shard_count == 4
    assert placements[0].group_id == "loader-subgroup"
    with pytest.raises(ValueError):
        ExecutionGroupRank(spec, 4)
    single = _loader(batch_size=5)
    grouped = tuple(
        _loader(batch_size=5, placement=placement) for placement in placements
    )
    assert grouped[0].loader_id == grouped[1].loader_id != single.loader_id
    single_items = _items(single)
    process_items = tuple(_items(loader) for loader in grouped)
    assert [i.provenance.payload_count for i in process_items[0]] == [
        i.provenance.payload_count for i in process_items[1]
    ]
    for batch in range(single.batch_count):
        expected = sorted(
            draw
            for item in single_items
            if item.provenance.batch_index == batch
            for draw, valid in zip(item.provenance.draw_indices, item.provenance.valid)
            if valid
        )
        observed = sorted(
            draw
            for items in process_items
            for item in items
            if item.provenance.batch_index == batch
            for draw, valid in zip(item.provenance.draw_indices, item.provenance.valid)
            if valid
        )
        assert observed == expected
    root = jax.random.key(0)
    single_keys = {
        draw: key
        for item in single_items
        for draw, key, valid in zip(
            item.provenance.draw_indices,
            jax.random.key_data(item.payload.draw_keys(root)),
            item.provenance.valid,
        )
        if valid
    }
    cases = grouped[1].layouts[0].case_capacity
    for process, items in enumerate(process_items):
        for item in items:
            payload = item.payload
            assert np.asarray(payload.case_valid).shape == (2 * cases,)
            owners = np.asarray(payload.level("atom").row_case)
            assert np.all(owners // (2 * cases) == process)
            atom_rows = payload.layout.level_rows[payload.layout.levels.index("atom")]
            endpoints = np.asarray(payload.value("endpoints"))
            assert np.all(endpoints // (2 * atom_rows) == process)
            keys = jax.random.key_data(payload.draw_keys(root))
            for draw, key, valid in zip(
                item.provenance.draw_indices, keys, item.provenance.valid
            ):
                if valid:
                    np.testing.assert_array_equal(key, single_keys[draw])


def test_split_evidence_refuses_shared_lineage() -> None:
    evidence = AtomisticSplitEvidence(
        {"train": _mixture(_alpha()), "validation": _mixture(_beta())}
    )
    assert evidence.split_names == ("train", "validation")
    schema = _schema()
    leaking = AtomisticStructureSource(
        schema, (_record(schema, "gamma", 200, 0, 2, (1,), lineage="alpha-lineage-2"),)
    )
    with pytest.raises(AtomisticLoaderError) as leak:
        AtomisticSplitEvidence({"train": _mixture(_alpha()), "test": _mixture(leaking)})
    assert leak.value.failure == "leakage"
