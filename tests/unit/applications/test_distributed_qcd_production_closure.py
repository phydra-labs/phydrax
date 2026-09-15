# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import zipfile
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


jax.config.update("jax_enable_x64", True)

from phydrax._array_archive import ArrayArchiveCorruptionError  # noqa: E402
from phydrax.applications.lattice_field._distributed_qcd import (  # noqa: E402
    DistributedGaugeTheoryPlan,
)
from phydrax.applications.lattice_field._qcd_io import (  # noqa: E402
    gauge_field_from_owned_shards,
    GaugeFieldRecord,
    read_gauge_interchange,
    read_native_gauge_archive,
    write_gauge_interchange,
    write_native_gauge_archive,
)
from phydrax.backends.lattice import (  # noqa: E402
    LatticeKernelCapabilities,
    LatticeKernelCapabilityError,
    LatticeProviderStatus,
    NativeJaxLatticeProvider,
)
from phydrax.discretization._lattice_distribution import (  # noqa: E402
    LatticeDecompositionPlan,
)


def _owned_shards(plan: LatticeDecompositionPlan, values: np.ndarray) -> np.ndarray:
    ids = np.asarray(plan.owned_global_ids)
    valid = np.asarray(plan.owned_valid)
    shards = values[ids].copy()
    return np.where(
        valid.reshape(valid.shape + (1,) * (values.ndim - 1)),
        shards,
        0,
    )


def _su2_links(shape: tuple[int, ...]) -> np.ndarray:
    sites = int(np.prod(shape))
    dimension = len(shape)
    links = np.zeros((sites, dimension, 2, 2), dtype=np.complex128)
    coordinates = np.asarray(tuple(np.ndindex(shape)))
    for site in range(sites):
        for axis in range(dimension):
            angle = 0.17 * (axis + 1) * (1 + coordinates[site, (axis + 1) % dimension])
            links[site, axis] = np.diag((np.exp(1j * angle), np.exp(-1j * angle)))
    return links


def test_global_ownership_is_exact_and_plaquettes_are_never_double_counted():
    plan = LatticeDecompositionPlan((4, 6), (2, 3), periodic=(False, False))
    owners = np.asarray(plan.ownership.site_owner)
    assert np.array_equal(np.bincount(owners), np.full((6,), 4))

    site_counts = np.stack(
        [np.asarray(plan.ownership.site_mask(part)) for part in range(6)]
    ).sum(axis=0)
    link_counts = np.stack(
        [np.asarray(plan.ownership.link_mask(part)) for part in range(6)]
    ).sum(axis=0)
    face_counts = np.stack(
        [np.asarray(plan.ownership.face_mask(part)) for part in range(6)]
    ).sum(axis=0)
    assert np.all(site_counts == 1)
    assert int(np.asarray(plan.ownership.face_valid).sum()) == 3 * 5
    assert np.all(link_counts == 1)
    assert np.array_equal(face_counts, np.asarray(plan.ownership.face_valid))


def test_parity_halo_exchange_reconstructs_reference_without_touching_other_parity():
    plan = LatticeDecompositionPlan((6, 4), (3, 2))
    global_values = jnp.arange(plan.site_count, dtype=jnp.float64) + 1.0
    owned = plan.pack_owned_sites(global_values)

    packed = plan.halo.pack(owned)
    complete = plan.halo.complete(plan.halo.start_reference(packed), owned)
    np.testing.assert_array_equal(complete, plan.pack_reference_sites(global_values))

    even_pack = plan.halo.pack(owned, parity=0)
    even_complete = plan.halo.complete(
        plan.halo.start_reference(even_pack),
        owned,
    )
    halo = np.asarray(plan.halo.distributed.local_valid) & ~np.asarray(
        plan.halo.distributed.local_owned
    )
    even = np.asarray(plan.halo.local_site_parity) == 0
    reference = np.asarray(plan.pack_reference_sites(global_values))
    assert np.array_equal(np.asarray(even_complete)[halo & even], reference[halo & even])
    assert np.all(np.asarray(even_complete)[halo & ~even] == 0.0)


def test_projected_spinor_halo_reconstructs_directional_block_rhs():
    plan = LatticeDecompositionPlan((6, 4), (3, 2))
    spinor = (
        np.arange(plan.site_count * 2 * 1 * 3, dtype=np.float64)
        .reshape((plan.site_count, 2, 1, 3))
        .astype(np.complex128)
    )
    local = plan.pack_owned_sites(spinor)
    projectors = np.zeros((plan.dimension, 2, 1, 2), dtype=np.complex128)
    reconstructors = np.zeros((plan.dimension, 2, 2, 1), dtype=np.complex128)
    for direction in range(plan.dimension):
        projectors[direction, 0, 0, 0] = 1.0
        projectors[direction, 1, 0, 1] = 1.0
        reconstructors[direction, 0, 0, 0] = 1.0
        reconstructors[direction, 1, 1, 0] = 1.0

    packed = plan.halo.pack_projected_spinor(local, projectors)
    reconstructed = plan.halo.reconstruct_projected_spinor_reference(
        packed,
        reconstructors,
    )
    values = np.asarray(reconstructed.values)
    valid = np.asarray(reconstructed.valid)
    local_ids = np.asarray(plan.halo.distributed.local_global_ids)
    neighbors = np.asarray(plan.neighbor_ids)
    for part, local_index, direction, orientation in np.argwhere(valid):
        target = int(local_ids[part, local_index])
        source = int(neighbors[target, direction, orientation])
        projected = projectors[direction, orientation] @ spinor[source].reshape((2, -1))
        expected = (reconstructors[direction, orientation] @ projected).reshape((2, 1, 3))
        np.testing.assert_array_equal(
            values[part, local_index, direction, orientation],
            expected,
        )


class _DiagonalDslash:
    def __init__(self, diagonal: np.ndarray):
        self.diagonal = jnp.asarray(diagonal)

    def mv(self, value):
        return self.diagonal * value

    def adjoint_mv(self, value):
        return jnp.conj(self.diagonal) * value


def test_native_block_dslash_preserves_the_true_rhs_axis():
    provider = NativeJaxLatticeProvider()
    diagonal = np.asarray([1.0 + 2.0j, 3.0 - 0.5j])[:, None, None]
    operator = _DiagonalDslash(diagonal)
    right_hand_sides = (
        np.arange(2 * 2 * 1 * 4).reshape((2, 2, 1, 4)).astype(np.complex128)
    )
    observed = provider.block_dslash(operator, right_hand_sides)
    expected = diagonal[..., None] * right_hand_sides
    np.testing.assert_allclose(observed, expected)
    np.testing.assert_allclose(
        provider.block_dslash(operator, right_hand_sides, adjoint=True),
        np.conj(diagonal)[..., None] * right_hand_sides,
    )


def test_native_archive_is_rank_independent_and_checksum_bound(tmp_path: Path):
    links = _su2_links((4, 4))
    one = LatticeDecompositionPlan((4, 4), (1, 1))
    four = LatticeDecompositionPlan((4, 4), (2, 2))
    first = gauge_field_from_owned_shards(
        one, _owned_shards(one, links), "fundamental-su2"
    )
    second = gauge_field_from_owned_shards(
        four,
        _owned_shards(four, links),
        "fundamental-su2",
    )
    assert first.field_id == second.field_id

    first_path = tmp_path / "one-rank.pxg"
    second_path = tmp_path / "four-rank.pxg"
    write_native_gauge_archive(first_path, first)
    write_native_gauge_archive(second_path, second)
    assert first_path.read_bytes() == second_path.read_bytes()
    restored = read_native_gauge_archive(second_path)
    assert restored.field.field_id == first.field_id
    np.testing.assert_array_equal(restored.field.links, links)

    corrupt_native = tmp_path / "corrupt-native.pxg"
    with zipfile.ZipFile(second_path, "r") as source:
        members = tuple(
            (item.filename, source.read(item.filename)) for item in source.infolist()
        )
    with zipfile.ZipFile(corrupt_native, "w", compression=zipfile.ZIP_STORED) as target:
        for name, data in members:
            if name.startswith("arrays/"):
                changed = bytearray(data)
                changed[-1] ^= 1
                data = bytes(changed)
            target.writestr(name, data)
    with pytest.raises(ArrayArchiveCorruptionError, match="checksum"):
        read_native_gauge_archive(corrupt_native)

    interchange = tmp_path / "field.ildg-like"
    write_gauge_interchange(
        interchange,
        first,
        "ildg",
        byte_order="big",
        precision="float64",
    )
    payload = bytearray(interchange.read_bytes())
    payload[-1] ^= 1
    interchange.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        read_gauge_interchange(interchange)


@pytest.mark.parametrize("kind", ("nersc", "milc", "ildg"))
@pytest.mark.parametrize(
    ("byte_order", "precision"),
    (("big", "float32"), ("little", "float64")),
)
def test_interchange_validates_endian_precision_and_round_trips(
    tmp_path: Path,
    kind: str,
    byte_order: str,
    precision: str,
):
    links = _su2_links((2, 2))
    field = GaugeFieldRecord(links, (2, 2), "fundamental-su2")
    path = tmp_path / f"{kind}-{byte_order}-{precision}.gauge"
    written = write_gauge_interchange(
        path,
        field,
        kind,
        byte_order=byte_order,
        precision=precision,
    )
    restored = read_gauge_interchange(
        path,
        expected_kind=kind,
        expected_byte_order=byte_order,
        expected_precision=precision,
    )
    assert restored.field.field_id == written.field.field_id
    tolerance = 2.0e-7 if precision == "float32" else 0.0
    np.testing.assert_allclose(restored.field.links, links, rtol=0.0, atol=tolerance)
    wrong = "little" if byte_order == "big" else "big"
    with pytest.raises(ValueError, match="byte order"):
        read_gauge_interchange(path, expected_byte_order=wrong)


def test_provider_capability_refusal_is_operation_specific():
    capabilities = LatticeKernelCapabilities(
        "gauge-action-only",
        ("gauge.wilson_action",),
        ("complex128",),
        ("single_device",),
        ("jit",),
        native_complex=True,
    )
    status = LatticeProviderStatus(
        capabilities,
        available=True,
        reason="focused test provider",
    )
    status.require("gauge.wilson_action", np.complex128)
    with pytest.raises(LatticeKernelCapabilityError) as refusal:
        status.require("gauge.wilson_force", np.complex128)
    assert refusal.value.capability == "gauge.wilson_force"


def test_serial_and_distributed_gauge_action_and_force_are_equal():
    links = _su2_links((4, 4))
    serial = DistributedGaugeTheoryPlan(
        LatticeDecompositionPlan((4, 4), (1, 1)),
        5.7,
    ).prepare()
    distributed = DistributedGaugeTheoryPlan(
        LatticeDecompositionPlan((4, 4), (2, 2)),
        5.7,
    ).prepare()

    serial_action = serial.gauge_action(links)
    distributed_action = distributed.gauge_action(links)
    assert bool(serial_action.successful)
    assert bool(distributed_action.successful)
    assert float(serial_action.value) > 0.0
    np.testing.assert_allclose(distributed_action.value, serial_action.value, rtol=1e-13)

    staples = np.roll(links, 1, axis=0) + np.roll(links, -1, axis=0)
    serial_force = serial.gauge_force(links, staples)
    distributed_force = distributed.gauge_force(links, staples)
    assert bool(serial_force.successful)
    assert bool(distributed_force.successful)
    assert float(jnp.linalg.norm(serial_force.value)) > 0.0
    np.testing.assert_allclose(distributed_force.value, serial_force.value, rtol=1e-13)
