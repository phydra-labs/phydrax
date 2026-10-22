#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stored certificate reconstruction preserves proof identity and refuses corruption."""

from dataclasses import fields
from typing import Any

import pytest

from phydrax._model import model_from_structure_recipe, model_structure_recipe
from phydrax.geometry import GlobalEmbeddingCertificate
from phydrax.lifecycle import register_meshing_source_artifacts
from tests._support.mesh_certificate_fixtures import (
    embedding,
    TETRAHEDRON,
    two_tetrahedra,
)


def _stored_embedding(
    certificate: GlobalEmbeddingCertificate,
    field_name: str | None = None,
    value: Any = None,
) -> GlobalEmbeddingCertificate:
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(certificate)
    if field_name is not None:
        field_names = tuple(field.name for field in fields(GlobalEmbeddingCertificate))
        recipe["items"][field_names.index(field_name)] = {
            "kind": "literal",
            "value": value,
        }
    restored = model_from_structure_recipe(recipe)
    assert isinstance(restored, GlobalEmbeddingCertificate)
    return restored


def test_global_embedding_reconstructs_stored_proof_and_expression_work() -> None:
    _, _, certificate = embedding(two_tetrahedra(TETRAHEDRON + 5.0))
    stored = _stored_embedding(certificate)
    rebuilt = GlobalEmbeddingCertificate.from_stored_fields(stored)

    assert rebuilt.certificate_id == certificate.certificate_id
    assert rebuilt.binding.binding_id == certificate.binding.binding_id
    assert rebuilt.status == certificate.status == "certified"
    assert rebuilt.evaluated_checks == certificate.evaluated_checks
    assert rebuilt.cell_count == certificate.cell_count == 2
    assert rebuilt.candidate_pair_count == certificate.candidate_pair_count
    assert rebuilt.ray_test_count == certificate.ray_test_count
    assert (
        rebuilt.source_expression_work_units == certificate.source_expression_work_units
    )
    assert (
        rebuilt.source_expression_peak_bytes == certificate.source_expression_peak_bytes
    )


@pytest.mark.parametrize(
    ("field_name", "value", "diagnostic"),
    (
        pytest.param(
            "cell_count", -1, "EmbeddingCounts.cell_count", id="negative-entity-count"
        ),
        pytest.param(
            "cell_count",
            3,
            "GlobalEmbeddingCertificate.stored_fields",
            id="changed-entity-count",
        ),
        pytest.param(
            "status",
            "unresolved",
            "GlobalEmbeddingCertificate.stored_fields",
            id="changed-verdict",
        ),
        pytest.param(
            "certificate_id",
            "foreign-proof",
            "GlobalEmbeddingCertificate.stored_fields",
            id="changed-proof-id",
        ),
        pytest.param(
            "source_expression_work_units",
            -1,
            "EmbeddingExpressionWork.work_units",
            id="negative-phase-work",
        ),
    ),
)
def test_global_embedding_refuses_corrupted_stored_fields(
    field_name: str,
    value: int | str,
    diagnostic: str,
) -> None:
    _, _, certificate = embedding(two_tetrahedra(TETRAHEDRON + 5.0))
    with pytest.raises(ValueError, match=diagnostic):
        stored = _stored_embedding(certificate, field_name, value)
        GlobalEmbeddingCertificate.from_stored_fields(stored)


def test_expression_work_is_reconstructed_without_changing_scientific_proof_identity() -> (
    None
):
    _, _, certificate = embedding(two_tetrahedra(TETRAHEDRON + 5.0))
    work = certificate.source_expression_work_units + 17
    stored = _stored_embedding(certificate, "source_expression_work_units", work)
    rebuilt = GlobalEmbeddingCertificate.from_stored_fields(stored)

    assert rebuilt.certificate_id == certificate.certificate_id
    assert rebuilt.status == certificate.status
    assert rebuilt.source_expression_work_units == work


def test_stored_record_memo_authenticates_actual_immutable_owners_and_current_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    import jax.numpy as jnp

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import (
        CoordinateEnclosureBudget,
        CoordinateEnclosureResourceError,
    )
    from phydrax.lifecycle import _meshing_source_records as owner

    @dataclass
    class Record:
        values: Any
        label: str

    value = Record(jnp.arange(1024, dtype=jnp.float64), "original")
    expected = owner.stored_record_content_id(value, excluded="")
    original = owner.prepare_model_recipe
    scans: list[object] = []

    def prepare(content: Any, **kwargs: Any) -> Any:
        scans.append(content)
        return original(content, **kwargs)

    def budget(work: int) -> CoordinateEnclosureBudget:
        return CoordinateEnclosureBudget(
            HostBatchTicket(work, bytes_upper=10_000_000),
            maximum_work_units=work,
            maximum_memory_bytes=10_000_000,
        )

    monkeypatch.setattr(owner, "prepare_model_recipe", prepare)
    ledger = budget(100_000)
    with ledger.ticket, ledger.activate():
        assert owner.stored_record_content_id(value, excluded="", budget=ledger) == expected
        first_work = ledger.work_units
        assert owner.stored_record_content_id(value, excluded="", budget=ledger) == expected
        hit_work = ledger.work_units - first_work
        assert 0 < hit_work < first_work
        assert len(scans) == 1
        assert ledger.retained_basis_bytes >= value.values.nbytes
        assert ledger.temporary_bytes_upper == 0
        value.label = "changed"
        changed = owner.stored_record_content_id(value, excluded="", budget=ledger)
        assert changed != expected
        assert len(scans) == 2
        value.values = value.values.at[0].set(-1.0)
        assert owner.stored_record_content_id(value, excluded="", budget=ledger) != changed
        assert len(scans) == 3
    # Another request must pay its own full current-content authentication.
    current = owner.stored_record_content_id(value, excluded="")
    count = len(scans)
    fresh = budget(100_000)
    with fresh.ticket, fresh.activate():
        assert owner.stored_record_content_id(value, excluded="", budget=fresh) == current
    assert len(scans) == count + 1
    required = fresh.work_units + hit_work
    exact = budget(required)
    with exact.ticket, exact.activate():
        assert owner.stored_record_content_id(value, excluded="", budget=exact) == current
        assert owner.stored_record_content_id(value, excluded="", budget=exact) == current
    assert exact.work_units == required
    short = budget(required - 1)
    with short.ticket, short.activate(), pytest.raises(CoordinateEnclosureResourceError):
        owner.stored_record_content_id(value, excluded="", budget=short)
        owner.stored_record_content_id(value, excluded="", budget=short)
    assert short.temporary_bytes_upper == 0


@pytest.mark.parametrize("readonly", (False, True))
def test_stored_record_memo_never_trusts_numpy_write_flags(readonly: bool) -> None:
    from dataclasses import dataclass

    import numpy as np

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle._meshing_source_records import stored_record_content_id

    @dataclass
    class Record:
        values: Any

    bank = np.arange(32, dtype=np.float64)
    bank.flags.writeable = not readonly
    value = Record(bank)
    with HostBatchTicket(100_000, bytes_upper=1_000_000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=1_000_000
        )
        with ledger.activate():
            expected = stored_record_content_id(value, excluded="", budget=ledger)
            work = ledger.work_units
            assert stored_record_content_id(value, excluded="", budget=ledger) == expected
            assert ledger.work_units - work >= 1 + bank.size
            cached = ledger.stored_record_content_cache[(id(value), "")]
            assert cached.mutable_banks[0] is bank
            bank.flags.writeable = True
            bank[0] = -1.0
            assert stored_record_content_id(value, excluded="", budget=ledger) != expected


def test_stored_record_memo_rescans_actual_mutable_banks_without_rescanning_jax(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    import jax.numpy as jnp
    import numpy as np

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle import _meshing_source_records as owner

    @dataclass
    class Record:
        immutable: Any
        mutable: Any

    value = Record(jnp.arange(4096, dtype=jnp.float64), np.arange(12, dtype=np.int64))
    expected = owner.stored_record_content_id(value, excluded="")
    from phydrax import _fingerprint as fingerprint_owner

    original_shards = fingerprint_owner._logical_shard_pieces
    original_chunks = fingerprint_owner._host_array_chunks
    original_bytes = fingerprint_owner.iter_array_bytes
    scanned: list[tuple[object, ...]] = []

    def shards(value: Any, *args: Any, **kwargs: Any) -> Any:
        scanned.append((value,))
        return original_shards(value, *args, **kwargs)

    def chunks(value: Any, *args: Any, **kwargs: Any) -> Any:
        scanned.append((value,))
        yield from original_chunks(value, *args, **kwargs)

    def raw_bytes(value: Any) -> Any:
        scanned.append((value,))
        yield from original_bytes(value)

    monkeypatch.setattr(fingerprint_owner, "_logical_shard_pieces", shards)
    monkeypatch.setattr(fingerprint_owner, "_host_array_chunks", chunks)
    monkeypatch.setattr(fingerprint_owner, "iter_array_bytes", raw_bytes)
    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )
        with ledger.activate():
            assert owner.stored_record_content_id(value, excluded="", budget=ledger) == expected
            assert len(scanned) == 2
            assert any(bank is value.immutable for bank in scanned[0])
            assert scanned[1][0] is value.mutable
            assert sum(bank is value.mutable for group in scanned for bank in group) == 1
            first_work = ledger.work_units
            assert owner.stored_record_content_id(value, excluded="", budget=ledger) == expected
            assert len(scanned) == 3
            assert scanned[-1][0] is value.mutable
            assert len(scanned[-1]) == 1
            assert value.mutable.size <= ledger.work_units - first_work < value.immutable.size
            value.mutable[0] = -1
            changed = owner.stored_record_content_id(value, excluded="", budget=ledger)
            assert changed != expected
            assert any(bank is value.immutable for bank in scanned[-2])
            assert scanned[-1][0] is value.mutable
            assert changed == owner.stored_record_content_id(value, excluded="")
            value.immutable.delete()
            with pytest.raises(ValueError, match="live actual JAX banks"):
                owner.stored_record_content_id(value, excluded="", budget=ledger)
            assert not ledger.stored_record_content_cache


@pytest.mark.parametrize("change", ("list-order", "mapping-value", "new-array-owner"))
def test_stored_record_memo_reauthenticates_mutable_container_structure(
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    from dataclasses import dataclass

    import jax.numpy as jnp

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle import _meshing_source_records as owner

    @dataclass
    class Record:
        parts: Any

    value = Record([{"bank": jnp.arange(256)}, {"label": "original"}])
    original = owner.prepare_model_recipe
    scans: list[object] = []

    def prepare(*args: Any, **kwargs: Any) -> Any:
        scans.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "prepare_model_recipe", prepare)
    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )
        with ledger.activate():
            expected = owner.stored_record_content_id(value, excluded="", budget=ledger)
            assert owner.stored_record_content_id(value, excluded="", budget=ledger) == expected
            assert len(scans) == 1
            if change == "list-order":
                value.parts.reverse()
            elif change == "mapping-value":
                value.parts[1]["label"] = "changed"
            else:
                value.parts[0]["bank"] = jnp.arange(256)
            current = owner.stored_record_content_id(value, excluded="", budget=ledger)
            assert len(scans) == 2
            assert current == owner.stored_record_content_id(value, excluded="")
            assert (current == expected) is (change == "new-array-owner")


def test_stored_record_memo_allows_unset_excluded_derived_identity() -> None:
    from dataclasses import dataclass, field

    import jax.numpy as jnp

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle._meshing_source_records import stored_record_content_id

    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )

        @dataclass
        class Binding:
            values: Any
            binding_id: str = field(init=False)

            def __post_init__(self) -> None:
                self.binding_id = stored_record_content_id(
                    self, excluded="binding_id", budget=ledger
                )

        with ledger.activate():
            value = Binding(jnp.arange(32, dtype=jnp.float64))
            expected = value.binding_id
            assert expected == stored_record_content_id(value, excluded="binding_id")
            # The constructor must not retain an owner whose derived field did
            # not yet exist. A later complete record may earn ordinary reuse.
            assert not ledger.stored_record_content_cache
            assert stored_record_content_id(
                value, excluded="binding_id", budget=ledger
            ) == expected
            assert (id(value), "binding_id") in ledger.stored_record_content_cache
            value.binding_id = "excluded-field-is-not-hashed"
            assert stored_record_content_id(
                value, excluded="binding_id", budget=ledger
            ) == expected
            value.values = value.values.at[0].set(-1.0)
            assert stored_record_content_id(
                value, excluded="binding_id", budget=ledger
            ) != expected


@pytest.mark.parametrize("layout", ("contiguous", "strided", "empty", "big-endian"))
def test_initial_mutable_seals_share_exact_canonical_logical_bytes(layout: str) -> None:
    from dataclasses import dataclass

    import numpy as np

    from phydrax._fingerprint import array_bytes_digest
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle._meshing_source_records import stored_record_content_id

    @dataclass
    class Record:
        values: Any

    source = np.arange(24, dtype=np.float64).reshape(4, 6)
    bank = {
        "contiguous": source, "strided": source[:, ::2],
        "empty": source[:0], "big-endian": source.astype(">f8"),
    }[layout]
    value = Record(bank)
    expected = stored_record_content_id(value, excluded="")
    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )
        with budget.activate():
            assert stored_record_content_id(value, excluded="", budget=budget) == expected
            (seal,) = budget.stored_record_content_cache[(id(value), "")].mutable_seals
            assert seal.shape == bank.shape and seal.dtype == bank.dtype.str
            assert seal.payload_digest == array_bytes_digest(bank)
            assert stored_record_content_id(value, excluded="", budget=budget) == expected


def test_mutable_seals_authenticate_current_shape_and_dtype() -> None:
    from dataclasses import dataclass

    import numpy as np

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle._meshing_source_records import stored_record_content_id

    @dataclass
    class Record:
        values: Any

    value = Record(np.arange(8, dtype=np.uint64))
    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )
        with budget.activate():
            original = stored_record_content_id(value, excluded="", budget=budget)
            value.values.shape = (2, 4)
            reshaped = stored_record_content_id(value, excluded="", budget=budget)
            assert reshaped != original
            value.values.dtype = np.dtype(">u8")
            changed = stored_record_content_id(value, excluded="", budget=budget)
            assert changed != reshaped
            assert changed == stored_record_content_id(value, excluded="")


def test_initial_mutable_seal_binds_the_same_snapshot_as_canonical_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    import numpy as np

    from phydrax import _fingerprint as fingerprint_owner
    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle._meshing_source_records import stored_record_content_id

    @dataclass
    class Record:
        values: Any

    value = Record(np.arange(32, dtype=np.float64))
    expected = stored_record_content_id(value, excluded="")
    original = fingerprint_owner._host_array_chunks
    changed = False

    def mutate_after_snapshot(bank: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        yield from original(bank, *args, **kwargs)
        if not changed:
            bank[0] = -1.0
            changed = True

    monkeypatch.setattr(fingerprint_owner, "_host_array_chunks", mutate_after_snapshot)
    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )
        with budget.activate():
            assert stored_record_content_id(value, excluded="", budget=budget) == expected
            current = stored_record_content_id(value, excluded="", budget=budget)
            assert current != expected
            assert current == stored_record_content_id(value, excluded="")


@pytest.mark.parametrize("readonly", (False, True))
def test_one_shot_record_hash_scans_without_retaining_owners(
    readonly: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import gc
    import weakref
    from dataclasses import dataclass

    import numpy as np

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle import _meshing_source_records as owner

    @dataclass
    class Record:
        values: Any

    bank = np.arange(32, dtype=np.float64)
    bank.flags.writeable = not readonly
    value = Record(bank)
    expected = owner.stored_record_content_id(value, excluded="")
    original = owner.logical_array_value_collection_digest
    scans = 0

    def digest(*args: Any, **kwargs: Any) -> str:
        nonlocal scans
        scans += 1
        assert kwargs.get("_numpy_seals") is None
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, "logical_array_value_collection_digest", digest)
    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )
        with budget.activate():
            assert owner.stored_record_content_id(
                value, excluded="", budget=budget, memoize=False
            ) == expected
            first_work = budget.work_units
            assert first_work >= bank.size
            assert owner.stored_record_content_id(
                value, excluded="", budget=budget, memoize=False
            ) == expected
            assert budget.work_units == 2 * first_work
            assert scans == 2
            bank.flags.writeable = True
            bank[0] = -1
            bank.flags.writeable = not readonly
            changed = owner.stored_record_content_id(
                value, excluded="", budget=budget, memoize=False
            )
            assert changed != expected
            assert changed == owner.stored_record_content_id(value, excluded="")
            assert not budget.stored_record_content_cache
            value_ref, bank_ref = weakref.ref(value), weakref.ref(bank)
            del value, bank
            gc.collect()
            assert value_ref() is None
            assert bank_ref() is None


@pytest.mark.parametrize("readonly", (False, True))
def test_one_shot_record_hash_bypasses_and_preserves_existing_memos(
    readonly: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from dataclasses import dataclass

    import numpy as np

    from phydrax._resource_ticket import HostBatchTicket
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle import _meshing_source_records as owner

    @dataclass
    class Record:
        values: Any

    value = Record(np.arange(32, dtype=np.float64))
    other = Record(np.arange(4, dtype=np.float64))
    value.values.flags.writeable = not readonly
    with HostBatchTicket(100_000, bytes_upper=10_000_000) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100_000, maximum_memory_bytes=10_000_000
        )
        with budget.activate():
            expected = owner.stored_record_content_id(value, excluded="", budget=budget)
            owner.stored_record_content_id(other, excluded="", budget=budget)
            entries = dict(budget.stored_record_content_cache)

            def reject_memo_authentication(*args: Any, **kwargs: Any) -> Any:
                raise AssertionError("one-shot hashing must bypass memo authentication")

            monkeypatch.setattr(
                owner, "require_coordinate_input_owners", reject_memo_authentication
            )
            assert owner.stored_record_content_id(
                value, excluded="", budget=budget, memoize=False
            ) == expected
            value.values.flags.writeable = True
            value.values[0] = -1
            value.values.flags.writeable = not readonly
            changed = owner.stored_record_content_id(
                value, excluded="", budget=budget, memoize=False
            )
            assert changed != expected
            assert changed == owner.stored_record_content_id(value, excluded="")
            assert budget.stored_record_content_cache.keys() == entries.keys()
            assert all(
                budget.stored_record_content_cache[key] is entry
                for key, entry in entries.items()
            )
