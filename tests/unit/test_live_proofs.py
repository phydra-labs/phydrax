import gc
import hashlib
from fractions import Fraction

import equinox as eqx
import jax
import numpy as np
import pytest

from phydrax._fingerprint import iter_array_bytes
from phydrax._live_proofs import (
    _ImmutableFingerprintLeaf,
    live_content_fingerprint,
    LiveProofLedger,
)
from phydrax.discretization._exact_coordinate_bank import ExactCoordinateBank


class _Operand(eqx.Module):
    values: np.ndarray
    label: str = eqx.field(static=True)


class _MixedOperand(eqx.Module):
    values: list[object]


class _RationalSubclass(Fraction):
    pass


class _ArraySubclass(np.ndarray):
    pass


class _OpaqueLeaf(_ImmutableFingerprintLeaf):
    def __init__(self, content: str) -> None:
        self.content = content

    def __repr__(self) -> str:
        return "opaque"

    def _live_leaf_fingerprint(self) -> str:
        return self.content


def _original_fingerprint(value: object) -> tuple[object, str]:
    """Reference stream produced by the original ledger guard."""
    leaves, treedef = jax.tree_util.tree_flatten(value)
    digest = hashlib.sha256()
    scalars: list[str] = []

    def flush() -> None:
        if scalars:
            text = "".join(scalars).encode("utf-8", "surrogatepass")
            digest.update(b"s" + len(text).to_bytes(8, "big"))
            digest.update(text)
            scalars.clear()

    for leaf in leaves:
        if not isinstance(leaf, (np.ndarray, np.generic, jax.Array)):
            text = f"{type(leaf).__qualname__}:{leaf!r}"
            scalars.append(f"{len(text)}:{text}")
            if isinstance(leaf, _ImmutableFingerprintLeaf):
                flush()
                content = leaf._live_leaf_fingerprint().encode("ascii")
                digest.update(b"i" + len(content).to_bytes(8, "big") + content)
            continue
        flush()
        array = np.asarray(leaf)
        header = f"{array.dtype.str}{array.shape}".encode("ascii")
        digest.update(b"a" + len(header).to_bytes(8, "big") + header)
        if array.dtype.hasobject:
            text = repr(array.tolist()).encode("utf-8", "surrogatepass")
            digest.update(len(text).to_bytes(8, "big"))
            digest.update(text)
            continue
        digest.update(array.nbytes.to_bytes(8, "big"))
        for chunk in iter_array_bytes(array):
            digest.update(chunk)
    flush()
    return treedef, digest.hexdigest()


def _record(ledger: LiveProofLedger[str], operands: tuple[object, ...]) -> None:
    ledger.record(operands, live_content_fingerprint(operands), "proved")


def test_ledger_preserves_original_guard_for_all_leaf_kinds() -> None:
    opaque = _OpaqueLeaf("first")
    values: list[object] = [
        _RationalSubclass(1, 3),
        Fraction(2, 5),
        True,
        1,
        1.25,
        "label",
        b"bytes",
        ((1, 3), (2, 5)),
        np.float64(2.5),
        np.asarray([1, 2], dtype=np.int64),
        jax.numpy.asarray([3.0, 4.0], dtype=np.float64),
        opaque,
        None,
        np.asarray([7, 8], dtype=np.int64).view(_ArraySubclass),
    ]
    operand = _MixedOperand(values)
    bank = ExactCoordinateBank.from_rows(((Fraction(1, 3), Fraction(2)),))
    operands = (operand, bank)
    ledger: LiveProofLedger[str] = LiveProofLedger()
    original = _original_fingerprint(operands)
    assert live_content_fingerprint(operands) == original
    ledger.record(operands, original, "proved")
    assert ledger.find(operands) == "proved"

    replacements: tuple[tuple[int, object], ...] = (
        (0, _RationalSubclass(2, 3)),
        (1, Fraction(3, 5)),
        (2, 1),
        (3, True),
        (4, 1.5),
        (5, "changed"),
        (6, b"changed"),
        (7, ((1, 3), (2, 7))),
        (8, np.float64(2.75)),
        (9, np.asarray([1, 3], dtype=np.int64)),
        (10, jax.numpy.asarray([3.0, 5.0], dtype=np.float64)),
        (12, 0),
        (13, np.asarray([7, 9], dtype=np.int64).view(_ArraySubclass)),
    )
    for index, replacement in replacements:
        previous = values[index]
        values[index] = replacement
        assert live_content_fingerprint(operands) == _original_fingerprint(operands)
        assert ledger.find(operands) is None
        values[index] = previous
        assert ledger.find(operands) == "proved"

    opaque.content = "changed"
    assert live_content_fingerprint(operands) == _original_fingerprint(operands)
    assert ledger.find(operands) is None
    opaque.content = "first"
    assert ledger.find(operands) == "proved"


def test_ledger_reuses_only_the_same_unchanged_live_operands() -> None:
    ledger: LiveProofLedger[str] = LiveProofLedger()
    first = _Operand(np.arange(4.0), "a")
    second = _Operand(np.ones(3), "b")
    operands = (first, second, None)
    assert ledger.find(operands) is None
    _record(ledger, operands)
    assert ledger.find(operands) == "proved"
    # A content-equal rebuild or copy is a different live object: replay.
    rebuilt = jax.tree_util.tree_map(np.copy, first)
    assert ledger.find((rebuilt, second, None)) is None
    # An absent operand binds as absent.
    assert ledger.find((first, second, second)) is None
    # An in-place mutation changes the content fingerprint: replay.
    first.values[0] = 7.0
    assert ledger.find(operands) is None
    first.values[0] = 0.0
    assert ledger.find(operands) == "proved"


def test_ledger_entry_dies_with_any_operand() -> None:
    ledger: LiveProofLedger[str] = LiveProofLedger()
    kept = _Operand(np.arange(2.0), "kept")
    dropped = _Operand(np.arange(3.0), "dropped")
    _record(ledger, (kept, dropped))
    assert len(ledger._entries) == 1
    del dropped
    gc.collect()
    assert not ledger._entries


def test_live_guard_binds_exact_coordinate_bank_content() -> None:
    first = ExactCoordinateBank.from_rows(((Fraction(1),),))
    second = ExactCoordinateBank.from_rows(((Fraction(3),),))
    assert live_content_fingerprint((first,)) != live_content_fingerprint((second,))
    ledger: LiveProofLedger[str] = LiveProofLedger()
    _record(ledger, (first,))
    assert ledger.find((first,)) == "proved"
    assert ledger.find((second,)) is None

    replacement = np.frombuffer(
        np.asarray([[3]], dtype=np.int64).tobytes(), dtype=np.int64
    ).reshape((1, 1))
    object.__setattr__(first, "_significands", replacement)
    with pytest.raises(ValueError, match="derived hash"):
        ledger.find((first,))


def test_live_guard_rejects_replaced_sparse_row_offsets() -> None:
    bank = ExactCoordinateBank.from_rows(((Fraction(1, 3),), (Fraction(1, 5),)))
    ledger: LiveProofLedger[str] = LiveProofLedger()
    _record(ledger, (bank,))
    assert ledger.find((bank,)) == "proved"
    assert tuple(bank) == ((Fraction(1, 3),), (Fraction(1, 5),))
    invalid_offsets = np.frombuffer(
        np.asarray([0, 0, 2], dtype=np.int64).tobytes(), dtype=np.int64
    )
    object.__setattr__(bank, "_sparse_row_starts", invalid_offsets)
    assert tuple(bank) == ((Fraction(1, 3),), (Fraction(1, 3),))
    with pytest.raises(ValueError, match="row offsets"):
        ledger.find((bank,))
