import hashlib

import numpy as np
import pytest

from phydrax.tensor_network._platform_interchange import _payload_digest


@pytest.mark.parametrize(
    "value",
    [
        np.arange(48, dtype=np.float64).reshape(6, 8)[::-2, ::3],
        np.empty((0, 3), dtype=np.float64),
        np.asarray([1 + 2j, -3j], dtype=np.complex128),
        np.asarray([True, False, True], dtype=np.bool_),
        np.arange(9, dtype=">i4"),
    ],
)
def test_payload_digest_matches_independent_byte_reference(value: np.ndarray) -> None:
    assert _payload_digest(value) == hashlib.sha256(value.tobytes(order="C")).hexdigest()


def test_payload_digest_detects_changed_bytes() -> None:
    value = np.arange(12, dtype=np.float64)
    damaged = value.copy()
    damaged.view(np.uint8)[-1] ^= np.uint8(1)
    assert _payload_digest(value) != _payload_digest(damaged)
