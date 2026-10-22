"""Only actual canonical kernel issuance can earn request-local owner reuse."""

from typing import Any

import numpy as np
import pytest

from phydrax.lifecycle import _meshing_field_records as fields
from phydrax.meshing.qualification import _polyhedral as kernel
from phydrax.meshing.qualification._native_lifecycle import execute_native_consumer
from tests.unit.lifecycle.test_meshing_polyhedral_fields import actual_epoch


def _fail(stage: str, message: str, data: dict[str, object]) -> Exception:
    return ValueError(f"{stage}: {message}; {data}")


@pytest.fixture
def kernel_calls(monkeypatch):
    original = kernel.prepare_polyhedral_archive_owners
    calls = []

    def prepare(carrier, *, budget=None):
        result = original(carrier, budget=budget)
        calls.append(result)
        return result

    monkeypatch.setattr(kernel, "prepare_polyhedral_archive_owners", prepare)
    return calls


def test_kernel_issued_owner_is_reused_by_first_canonical_epoch_admission(
    actual_epoch, kernel_calls,
) -> None:
    epoch, records = actual_epoch

    def admit(budget):
        binding, prepared = fields.prepare_meshing_polyhedral_owner_binding(
            epoch.carrier, budget=budget
        )
        assert len(kernel_calls) == 1
        declaration = epoch.declarations[0]
        admitted = fields.MeshingAcceptedEpoch(
            epoch.carrier, (declaration,), epoch.fields
        )
        current, _ = fields.prepare_meshing_field_owner(
            declaration, epoch.carrier, budget=budget
        )
        assert current is prepared
        assert len(kernel_calls) == 1
        assert binding.binding_id == declaration.owner_binding.binding_id
        assert admitted.epoch_id == epoch.epoch_id
        return admitted

    execute_native_consumer(
        epoch.carrier, records["generation_specification"].limits, admit,
        failure_factory=_fail,
    )


def test_supplied_owners_do_not_issue_canonical_preparation_authority(
    actual_epoch, kernel_calls,
) -> None:
    epoch, records = actual_epoch

    def admit(budget):
        supplied = kernel.prepare_polyhedral_archive_owners(epoch.carrier, budget=budget)
        fields.prepare_meshing_polyhedral_owner_binding(
            epoch.carrier, owners=supplied, budget=budget
        )
        _, canonical = fields.prepare_meshing_polyhedral_owner_binding(
            epoch.carrier, budget=budget
        )
        assert canonical is not supplied
        assert len(kernel_calls) == 2

    execute_native_consumer(
        epoch.carrier, records["generation_specification"].limits, admit,
        failure_factory=_fail,
    )


def test_kernel_owner_reuse_expires_with_the_actual_request(actual_epoch, kernel_calls) -> None:
    epoch, records = actual_epoch
    previous = []

    def admit(budget):
        _, prepared = fields.prepare_meshing_polyhedral_owner_binding(epoch.carrier, budget=budget)
        if previous:
            assert prepared is not previous[0]
        previous.append(prepared)

    for _ in range(2):
        execute_native_consumer(
            epoch.carrier, records["generation_specification"].limits, admit,
            failure_factory=_fail,
        )
    assert len(kernel_calls) == 2


def test_kernel_owner_reuse_reauthenticates_current_mutable_output_bytes(
    actual_epoch, kernel_calls,
) -> None:
    epoch, records = actual_epoch

    def admit(budget):
        binding, _ = fields.prepare_meshing_polyhedral_owner_binding(epoch.carrier, budget=budget)
        bank = binding.coefficient_banks["vem"].coordinates
        assert isinstance(bank, np.ndarray) and bank.size
        original, writeable = bank.flat[0], bank.flags.writeable
        bank.setflags(write=True)
        bank.flat[0] = original + 1
        bank.setflags(write=writeable)
        try:
            with pytest.raises(ValueError, match="changed owner payload"):
                fields.prepare_meshing_polyhedral_owner_binding(epoch.carrier, budget=budget)
        finally:
            bank.setflags(write=True)
            bank.flat[0] = original
            bank.setflags(write=writeable)
        assert len(kernel_calls) == 1

    execute_native_consumer(
        epoch.carrier, records["generation_specification"].limits, admit,
        failure_factory=_fail,
    )


def test_kernel_owner_reuse_tracks_current_constructor_identity(
    actual_epoch, kernel_calls, monkeypatch,
) -> None:
    epoch, records = actual_epoch
    original = kernel.PolyhedralSourceStarFEMPlan.__init__

    def admit(budget):
        _, first = fields.prepare_meshing_polyhedral_owner_binding(epoch.carrier, budget=budget)

        def changed(self, *args: Any, **kwargs: Any) -> None:
            original(self, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(kernel.PolyhedralSourceStarFEMPlan, "__init__", changed)
            _, second = fields.prepare_meshing_polyhedral_owner_binding(epoch.carrier, budget=budget)
        assert second is not first
        assert len(kernel_calls) == 2

    execute_native_consumer(
        epoch.carrier, records["generation_specification"].limits, admit,
        failure_factory=_fail,
    )
