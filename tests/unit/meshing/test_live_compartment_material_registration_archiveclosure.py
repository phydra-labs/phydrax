"""Authenticate real live image issuance without portable cold authority."""

from copy import copy
import gc
import weakref

import numpy as np
import pytest

import phydrax as phx
from phydrax.geometry._compartments import CompartmentMeshingSource
from phydrax.lifecycle._meshing_registration import (
    _material_root,
    _remember_issued_compartment_material_registration,
    has_issued_compartment_material_registration,
)
from tests._support.coordinate_references import (
    coordinate_reference_budget,
    native_meshing_request,
)
from tests._support.image_sources import _execute, _source, image_specification


@pytest.fixture(scope="module")
def cold_material_publication():
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    specification = image_specification(source)
    options = phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
    # This is a real complete publication, not a constructed passed-status
    # object. Its original paid native request has ended before any lookup.
    result = _execute(source)
    assert result.certification is not None
    assert result.region_evidence is not None
    return source, specification, options, result


def test_cold_material_certificate_has_no_unbudgeted_authority(
    cold_material_publication,
) -> None:
    assert not has_issued_compartment_material_registration(
        *cold_material_publication, budget=None
    )


def test_host_ledger_cannot_authorize_material_registration(
    cold_material_publication,
) -> None:
    _, specification, _, _ = cold_material_publication
    with coordinate_reference_budget(specification.limits) as budget:
        assert _material_root(budget) is None
        assert not has_issued_compartment_material_registration(
            *cold_material_publication, budget=budget
        )
        with pytest.raises(ValueError, match="actual live native root and ledger"):
            _remember_issued_compartment_material_registration(
                *cold_material_publication, budget=budget
            )


def test_new_paid_root_cannot_promote_cold_material_publication(
    cold_material_publication,
) -> None:
    _, specification, _, _ = cold_material_publication
    with native_meshing_request(specification.limits) as (scope, budget, _):
        with budget.bind_ticket(scope.ticket_view()):
            assert _material_root(budget) is scope
            assert not has_issued_compartment_material_registration(
                *cold_material_publication, budget=budget
            )
            # A real certificate and material evidence are still not the
            # actual producer-issued authority for this root and ledger.
            with pytest.raises(ValueError):
                _remember_issued_compartment_material_registration(
                    *cold_material_publication, budget=budget
                )
            assert not has_issued_compartment_material_registration(
                *cold_material_publication, budget=budget
            )
        assert _material_root(budget) is None
        assert not has_issued_compartment_material_registration(
            *cold_material_publication, budget=budget
        )
    assert not has_issued_compartment_material_registration(
        *cold_material_publication, budget=budget
    )


def test_same_root_distinct_ledgers_still_cannot_adopt_cold_authority(
    cold_material_publication,
) -> None:
    _, specification, _, _ = cold_material_publication
    with native_meshing_request(specification.limits) as (root, first, _):
        with first.bind_ticket(root.ticket_view()):
            assert _material_root(first) is root
        with native_meshing_request(specification.limits) as (child, second, _):
            with second.bind_ticket(child.ticket_view()):
                assert first is not second
                assert child is not root
                assert _material_root(second) is root
                assert not has_issued_compartment_material_registration(
                    *cold_material_publication, budget=second
                )
                with pytest.raises(ValueError):
                    _remember_issued_compartment_material_registration(
                        *cold_material_publication, budget=second
                    )


def test_genuine_live_material_authority_reauthenticates_payload_and_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing.providers import _native_compartment as producer

    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    specification = image_specification(source)
    options = phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
    with native_meshing_request(specification.limits) as (root, budget, _):
        plan = phx.meshing.NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=source.coordinate_contract
        )
        result = plan.execute()
        # Use the exact actual producer owners, not equal reconstructed inputs.
        source, specification, options = plan.source, plan.specification, plan.options
        assert isinstance(source, CompartmentMeshingSource)
        assert result.certification is not None
        assert result.region_evidence is not None
        with budget.bind_ticket(root.ticket_view()):
            assert not budget.stored_record_content_cache
            before = budget.work_units
            assert has_issued_compartment_material_registration(
                source, specification, options, result, budget=budget
            )
            assert budget.work_units > before
            assert not budget.stored_record_content_cache
            assert not has_issued_compartment_material_registration(
                copy(source), specification, options, result, budget=budget
            )
            assert not has_issued_compartment_material_registration(
                source, copy(specification), options, result, budget=budget
            )
            assert not has_issued_compartment_material_registration(
                source, specification,
                phx.meshing.NativeMeshingOptions("image_material_tetrahedral"),
                result, budget=budget,
            )
            # Re-sealing can only authenticate the already genuinely issued
            # producer owner. It cannot mint source/material proof itself.
            _remember_issued_compartment_material_registration(
                source, specification, options, result, budget=budget
            )
            values = source.labels.asset.values
            original, writeable = values.flat[0], values.flags.writeable
            values.setflags(write=True)
            try:
                values.flat[0] = original + 1
                # The existing canonical NumPy recipe includes writeability
                # metadata. Restore it before lookup so this refusal proves
                # changed bytes, not merely different recipe metadata.
                values.setflags(write=writeable)
                assert not has_issued_compartment_material_registration(
                    source, specification, options, result, budget=budget
                )
            finally:
                values.setflags(write=True)
                values.flat[0] = original
                values.setflags(write=writeable)
            # Read-only flags confer no proof: the changed bytes above were
            # refused under these same flags, and only restoring bytes permits reuse.
            assert has_issued_compartment_material_registration(
                source, specification, options, result, budget=budget
            )
            with monkeypatch.context() as patch:
                patch.setattr(
                    producer, "_issue_compartment_material_authority",
                    producer._require_issued_compartment_material_authority,
                )
                assert not has_issued_compartment_material_registration(
                    source, specification, options, result, budget=budget
                )
            assert has_issued_compartment_material_registration(
                source, specification, options, result, budget=budget
            )
        assert not has_issued_compartment_material_registration(
            source, specification, options, result, budget=budget
        )
        # The producer's original ledger and these callers are different
        # Python owners, but share the same actual still-paid native root.
        with native_meshing_request(specification.limits) as (child, second, _):
            with second.bind_ticket(child.ticket_view()):
                assert second is not budget
                assert _material_root(second) is root
                assert has_issued_compartment_material_registration(
                    source, specification, options, result, budget=second
                )
                assert not second.stored_record_content_cache
    assert not has_issued_compartment_material_registration(
        source, specification, options, result, budget=budget
    )
    with native_meshing_request(specification.limits) as (fresh, fresh_budget, _):
        with fresh_budget.bind_ticket(fresh.ticket_view()):
            assert not has_issued_compartment_material_registration(
                source, specification, options, result, budget=fresh_budget
            )
            with pytest.raises(ValueError):
                _remember_issued_compartment_material_registration(
                    source, specification, options, result, budget=fresh_budget
                )


def _genuine_registration_weak_owners():
    source = _source(np.ones((1, 1, 1), dtype=np.int16))
    specification = image_specification(source)
    options = phx.meshing.NativeMeshingOptions("image_material_tetrahedral")
    with native_meshing_request(specification.limits) as (root, budget, _):
        plan = phx.meshing.NativeMeshingProvider(options).plan(
            source, specification, coordinate_contract=source.coordinate_contract
        )
        result = plan.execute()
        source, specification, options = plan.source, plan.specification, plan.options
        assert isinstance(source, CompartmentMeshingSource)
        report, evidence = result.certification, result.region_evidence
        assert report is not None and evidence is not None
        with budget.bind_ticket(root.ticket_view()):
            assert has_issued_compartment_material_registration(
                source, specification, options, result, budget=budget
            )
            assert not budget.stored_record_content_cache
        return (
            tuple(weakref.ref(owner) for owner in (
                source, specification, options, result, report, evidence, root, budget,
            )),
            (id(source), id(report), id(evidence), id(root)),
        )


def test_genuine_material_registration_retains_no_global_owner() -> None:
    from phydrax.lifecycle import _meshing_registration as registration
    from phydrax.meshing.providers import _native_compartment as producer

    references, key = _genuine_registration_weak_owners()
    gc.collect()
    assert all(reference() is None for reference in references)
    assert key not in registration._issued_material
    assert key not in producer._issued_compartment_material
