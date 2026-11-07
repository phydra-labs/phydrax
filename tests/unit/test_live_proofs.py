import gc

import equinox as eqx
import jax
import numpy as np

from phydrax._live_proofs import live_content_fingerprint, LiveProofLedger


class _Operand(eqx.Module):
    values: np.ndarray
    label: str = eqx.field(static=True)


def _record(ledger: LiveProofLedger[str], operands: tuple[object, ...]) -> None:
    ledger.record(operands, live_content_fingerprint(operands), "proved")


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
