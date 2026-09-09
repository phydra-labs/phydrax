#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import struct

import numpy as np
import pytest

from phydrax.interchange import (
    decode_segy_bytes,
    read_segy,
    ResourceLimits,
    SEGYDecodeError,
    SEGYRev1IEEEProfile,
)
from phydrax.interchange._report import AdapterStatus
from phydrax.units import PASCAL


def _limits(*, max_bytes=100_000, max_nodes=1000):
    return ResourceLimits(max_bytes, 4, max_nodes, 1000, 20)


def _segy(
    *,
    sequences=(1, 3),
    identifications=(1, 2),
    samples=((1.0, -2.0, 3.0), (4.0, 5.0, 6.0)),
):
    text = bytearray(b" " * 3200)
    label = b"C01 PHYDRAX EXPLICIT REV1 IEEE PRESSURE PROFILE"
    text[: len(label)] = label
    binary = bytearray(400)
    struct.pack_into(">H", binary, 16, 2000)
    struct.pack_into(">H", binary, 20, 3)
    struct.pack_into(">H", binary, 24, 5)
    struct.pack_into(">H", binary, 52, 1)
    struct.pack_into(">H", binary, 54, 1)
    struct.pack_into(">H", binary, 56, 2)
    struct.pack_into(">H", binary, 300, 0x0100)
    struct.pack_into(">H", binary, 302, 1)
    struct.pack_into(">h", binary, 304, 0)
    records = []
    for row, (sequence, identification, values) in enumerate(
        zip(sequences, identifications, samples, strict=True)
    ):
        header = bytearray(240)
        struct.pack_into(">i", header, 4, sequence)
        struct.pack_into(">h", header, 28, identification)
        struct.pack_into(">i", header, 40, 100 + row)
        struct.pack_into(">i", header, 44, 120 + row)
        struct.pack_into(">i", header, 48, 20)
        struct.pack_into(">h", header, 68, 1)
        struct.pack_into(">h", header, 70, 1)
        struct.pack_into(">i", header, 72, 1000 + 10 * row)
        struct.pack_into(">i", header, 76, 2000 + 10 * row)
        struct.pack_into(">i", header, 80, 1100 + 10 * row)
        struct.pack_into(">i", header, 84, 2100 + 10 * row)
        struct.pack_into(">h", header, 88, 1)
        struct.pack_into(">h", header, 108, 4 + row)
        struct.pack_into(">HH", header, 114, 3, 2000)
        struct.pack_into(">h", header, 202, 1)
        struct.pack_into(">h", header, 214, 1)
        records.append(bytes(header) + struct.pack(">3f", *values))
    return bytes(text + binary) + b"".join(records)


def test_explicit_rev1_profile_preserves_trace_gaps_clock_geometry_and_dead_validity():
    data = _segy()
    result = decode_segy_bytes(
        data,
        profile=SEGYRev1IEEEProfile(
            byte_order="big", text_encoding="ascii", pressure_polarity="positive"
        ),
        limits=_limits(),
    )
    assert result.manifest.size_bytes == len(data)
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    assert result.amplitude_unit == PASCAL
    np.testing.assert_array_equal(result.trace_sequence, [1, 3])
    np.testing.assert_array_equal(result.trace_identification, [1, 2])
    np.testing.assert_allclose(
        result.source_positions, [[1000, 2000, 100], [1010, 2010, 101]]
    )
    np.testing.assert_allclose(
        result.receiver_positions, [[1100, 2100, 100], [1110, 2110, 101]]
    )
    np.testing.assert_allclose(
        result.series.support.broadcast_coordinates(),
        [[0.004, 0.006, 0.008], [0.005, 0.007, 0.009]],
    )
    np.testing.assert_allclose(result.series.values, [[1, -2, 3], [4, 5, 6]])
    np.testing.assert_array_equal(result.series.sample_valid, [[True] * 3, [False] * 3])


def test_profile_rejects_byte_order_calibration_and_clock_ambiguity():
    data = _segy(sequences=(1,), identifications=(1,), samples=((1.0, 2.0, 3.0),))
    with pytest.raises(SEGYDecodeError, match="revision"):
        decode_segy_bytes(
            data,
            profile=SEGYRev1IEEEProfile(
                byte_order="little", text_encoding="ascii", pressure_polarity="positive"
            ),
            limits=_limits(),
        )
    with pytest.raises(SEGYDecodeError, match="conflicts"):
        decode_segy_bytes(
            data,
            profile=SEGYRev1IEEEProfile(
                byte_order="big",
                text_encoding="ascii",
                pressure_calibration=2.0,
                pressure_polarity="positive",
            ),
            limits=_limits(),
        )
    malformed = bytearray(data)
    struct.pack_into(">H", malformed, 3600 + 116, 4000)
    with pytest.raises(SEGYDecodeError, match="clocks disagree"):
        decode_segy_bytes(
            bytes(malformed),
            profile=SEGYRev1IEEEProfile(
                byte_order="big", text_encoding="ascii", pressure_polarity="positive"
            ),
            limits=_limits(),
        )


def test_decoder_accounts_before_allocation_and_file_reader_stays_under_root(tmp_path):
    data = _segy(sequences=(1,), identifications=(1,), samples=((1.0, 2.0, 3.0),))
    profile = SEGYRev1IEEEProfile(
        byte_order="big", text_encoding="ascii", pressure_polarity="positive"
    )
    with pytest.raises(ValueError, match="node"):
        decode_segy_bytes(data, profile=profile, limits=_limits(max_nodes=2))
    path = tmp_path / "line.sgy"
    path.write_bytes(data)
    loaded = read_segy(
        path.name, trusted_root=tmp_path, profile=profile, limits=_limits()
    )
    assert loaded.manifest.source_kind == "file"
    with pytest.raises(ValueError):
        read_segy("../line.sgy", trusted_root=tmp_path, profile=profile, limits=_limits())
