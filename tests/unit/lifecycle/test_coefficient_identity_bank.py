#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coefficient identity banks: bounded coordinate arrays and lazily derived IDs."""

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
from phydrax._model._structure import (
    model_from_array_recipe,
    model_recipe_array_inventory,
    model_recipe_array_values,
    model_recipe_from_wire_json,
    model_recipe_wire_json,
    model_structure_recipe,
)
from phydrax.discretization._spaces import (
    DiscreteFieldSpace,
    GlobalCoefficientId,
    TensorDofLayout,
)
from phydrax.lifecycle import (
    _meshing_field_records as records,
    read_meshing_source_closure,
    write_meshing_source_closure,
)
from phydrax.lifecycle._meshing_field_records import MeshingCoefficientIdentityBank
from phydrax.lifecycle._meshing_sources import (
    register_meshing_source_artifacts,
    validate_meshing_source_closure,
)
from phydrax.linalg import ArraySpace


def _space(name: str, count: int, components: tuple[int, ...] = ()) -> DiscreteFieldSpace:
    return DiscreteFieldSpace(
        name,
        "coefficient-bank-test-support",
        TensorDofLayout(("row",), (count,), component_shape=components),
        ArraySpace((count, *components)),
        representation="point_value",
    )


def _readonly(values: Any) -> np.ndarray:
    array = np.array(values, dtype=np.int64)
    array.setflags(write=False)
    return array


@pytest.fixture
def derived_count(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Count derived-ID fingerprints computed by the bank module."""
    calls = [0]
    original = records.GlobalCoefficientId

    def counted(*args: Any, **kwargs: Any) -> GlobalCoefficientId:
        if kwargs.get("coefficient_id") is None:
            calls[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(records, "GlobalCoefficientId", counted)
    return calls


def _mixed_bank() -> MeshingCoefficientIdentityBank:
    foreign = GlobalCoefficientId("another-field", 0, ()).coefficient_id
    return MeshingCoefficientIdentityBank.from_rows(
        "mixed-field",
        ((0, ()), (0, (0,)), (0, (0, 2)), (0, (1,)), (2, (0,)), (7, ())),
        ((1, "authored-coefficient"), (4, foreign)),
    )


def test_field_space_bank_is_one_array_without_deriving_ids(
    derived_count: list[int],
) -> None:
    space = _space("vector-field", 4096, (3,))
    bank = MeshingCoefficientIdentityBank.from_field_space(space)
    bank.require_field_space(space)
    assert derived_count[0] == 0
    coordinates = bank.coordinates
    assert (coordinates.shape, coordinates.dtype) == ((12288, 2), np.int64)
    assert coordinates.flags.c_contiguous and not coordinates.flags.writeable
    assert bank.explicit_ids == ()
    for row in (0, 1, 2, 3, 12287):
        assert bank.coefficient(row) == GlobalCoefficientId(
            space.field_space_id, row // 3, (row % 3,)
        )
    assert derived_count[0] == 5
    scalar = MeshingCoefficientIdentityBank.from_field_space(_space("scalar", 3))
    assert scalar.coordinates.tolist() == [[0], [1], [2]]
    empty = MeshingCoefficientIdentityBank.from_rows("empty", ())
    assert empty.coordinates.shape == (0, 1)


def test_explicit_overrides_gaps_and_mixed_components_survive_the_archive(
    tmp_path: Path,
) -> None:
    bank = _mixed_bank()
    assert bank.coordinates.tolist() == [
        [0, -1, -1],
        [0, 0, -1],
        [0, 0, 2],
        [0, 1, -1],
        [2, 0, -1],
        [7, -1, -1],
    ]
    receipt = write_meshing_source_closure(tmp_path / "coefficient-bank", bank)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert type(restored) is MeshingCoefficientIdentityBank
    assert restored == bank
    assert not restored.coordinates.flags.writeable
    expected = (
        (0, (), None),
        (0, (0,), "authored-coefficient"),
        (0, (0, 2), None),
        (0, (1,), None),
        (2, (0,), GlobalCoefficientId("another-field", 0, ()).coefficient_id),
        (7, (), None),
    )
    for row, (ordinal, component, identity) in enumerate(expected):
        assert restored.coefficient(row) == GlobalCoefficientId(
            "mixed-field", ordinal, component, coefficient_id=identity
        )
    with pytest.raises(IndexError, match=r"MeshingCoefficientIdentityBank\.row"):
        restored.coefficient(6)


def test_lf17_scale_banks_fit_manifest_with_separately_accounted_arrays(
    tmp_path: Path,
) -> None:
    register_meshing_source_artifacts()
    counts = (125, 729, 729, 512, 512, 1944, 1944, 1728, 1728) * 2
    spaces = tuple(
        _space(f"field-{index}", count) for index, count in enumerate(counts)
    ) + (_space("euler-0", 64, (5,)), _space("euler-1", 64, (5,)))
    banks = tuple(MeshingCoefficientIdentityBank.from_field_space(s) for s in spaces)
    rows = sum(bank.row_count for bank in banks)
    assert rows == 2 * 10271
    recipe = model_structure_recipe(banks)
    wire = model_recipe_wire_json(recipe)
    assert len(wire.encode("utf-8")) < DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes
    inventory = model_recipe_array_inventory(recipe, prefix="coefficient-identities")
    assert [entry.shape for entry in inventory] == [
        bank.coordinates.shape for bank in banks
    ]
    assert sum(np.prod(entry.shape) for entry in inventory) == rows - 640 + 2 * 640
    receipt = write_meshing_source_closure(tmp_path / "lf17-scale-banks", banks)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert restored == banks
    for bank, space in zip(restored, spaces, strict=True):
        bank.require_field_space(space)


def test_derived_form_override_preserves_large_banked_identity_inventory(
    tmp_path: Path,
) -> None:
    coordinates = _readonly(np.arange(8192, dtype=np.int64)[:, None])
    foreign = GlobalCoefficientId("another-field", 0, ()).coefficient_id
    bank = MeshingCoefficientIdentityBank("large-field", coordinates, ((3, foreign),))
    receipt = write_meshing_source_closure(tmp_path / "large-bank", bank)
    restored = read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert restored == bank
    assert restored.coefficient(3).coefficient_id == foreign
    assert restored.coefficient(8191).coefficient_id == (
        GlobalCoefficientId("large-field", 8191, ()).coefficient_id
    )
    wire = model_recipe_wire_json(model_structure_recipe(restored))
    assert len(wire.encode("utf-8")) < DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes


def test_wire_recipe_refuses_claimed_oversized_coordinates_before_allocation() -> None:
    register_meshing_source_artifacts()
    bank = MeshingCoefficientIdentityBank.from_field_space(_space("vector", 2, (2,)))
    wire = model_recipe_wire_json(model_structure_recipe(bank))
    claimed = DEFAULT_ARRAY_ARCHIVE_LIMITS.max_array_elements // 2 + 1
    forged = wire.replace('"shape":[4,2]', f'"shape":[{claimed},2]')
    assert forged != wire and len(forged) < len(wire) + 16
    with pytest.raises(ValueError, match="element limit"):
        model_recipe_from_wire_json(forged)


def _corrupt(bank: MeshingCoefficientIdentityBank, case: str) -> Any:
    derived = GlobalCoefficientId("mixed-field", 0, (0, 2)).coefficient_id
    match case:
        case "writable":
            return "coordinates", np.array(bank.coordinates)
        case "dtype":
            return "coordinates", _readonly(bank.coordinates).astype(np.int32)
        case "rank":
            return "coordinates", _readonly(bank.coordinates[:, 0])
        case "duplicate":
            return "coordinates", _readonly(bank.coordinates[[0, 0, 1, 2, 3, 4]])
        case "unordered":
            return "coordinates", _readonly(bank.coordinates[[1, 0, 2, 3, 4, 5]])
        case "negative":
            return "coordinates", _readonly(
                [[-2, -1, -1], *bank.coordinates[1:].tolist()]
            )
        case "interior-padding":
            return "coordinates", _readonly(
                [
                    *bank.coordinates[:2].tolist(),
                    [0, -1, 2],
                    *bank.coordinates[3:].tolist(),
                ]
            )
        case "noncanonical-width":
            return "coordinates", _readonly(
                np.pad(bank.coordinates, ((0, 0), (0, 1)), constant_values=-1)
            )
        case "empty-width":
            return "coordinates", _readonly(np.empty((0, 2)))
        case "override-range":
            return "explicit_ids", ((6, "authored-coefficient"),)
        case "override-order":
            return "explicit_ids", ((4, "b"), (1, "a"))
        case "override-repeat":
            return "explicit_ids", ((1, "a"), (4, "a"))
        case "override-untrimmed":
            return "explicit_ids", ((1, " a"),)
        case "override-type":
            return "explicit_ids", [(1, "a")]
        case "override-own-derived":
            return "explicit_ids", ((2, derived),)
        case "override-collides":
            return "explicit_ids", ((1, derived),)
        case _:
            raise ValueError(
                f"coefficient_bank.case must name a corruption; got {case!r}"
            )


@pytest.mark.parametrize(
    "case",
    (
        "writable",
        "dtype",
        "rank",
        "duplicate",
        "unordered",
        "negative",
        "interior-padding",
        "noncanonical-width",
        "empty-width",
        "override-range",
        "override-order",
        "override-repeat",
        "override-untrimmed",
        "override-type",
        "override-own-derived",
        "override-collides",
    ),
)
def test_constructor_bypass_refuses_noncanonical_bank(case: str) -> None:
    bank = copy(_mixed_bank())
    name, value = _corrupt(bank, case)
    object.__setattr__(bank, name, value)
    with pytest.raises((TypeError, ValueError), match="MeshingCoefficientIdentityBank"):
        MeshingCoefficientIdentityBank.from_stored_fields(bank)


@pytest.mark.parametrize("tamper", ("unordered", "negative", "interior-padding"))
def test_restored_bank_authenticates_coordinate_payload(tamper: str) -> None:
    register_meshing_source_artifacts()
    bank = _mixed_bank()
    recipe = model_structure_recipe(bank)
    arrays = model_recipe_array_values(bank, recipe, prefix="bank-tamper")
    ((name, values),) = arrays.items()
    rows = values.tolist()
    match tamper:
        case "unordered":
            rows[0], rows[1] = rows[1], rows[0]
        case "negative":
            rows[5] = [7, -2, -1]
        case _:
            rows[2] = [0, -1, 2]
    arrays[name] = _readonly(rows)
    restored = model_from_array_recipe(recipe, arrays, prefix="bank-tamper")
    with pytest.raises(ValueError, match=r"MeshingCoefficientIdentityBank\.coordinates"):
        validate_meshing_source_closure(restored)
