#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import pytest

from phydrax.qualification import CampaignRole, ScientificCampaign, ScientificCase


def _case(
    case_id: str,
    independent_unit_id: str,
    preparation_id: str,
    *,
    construct_id: str = "construct-shared",
    parent_case_ids: tuple[str, ...] = (),
    batch_id: str | None = None,
) -> ScientificCase:
    return ScientificCase(
        case_id,
        independent_unit_id,
        construct_id,
        "condition-298K",
        preparation_id,
        f"batch-{case_id}" if batch_id is None else batch_id,
        (f"manifest-{case_id}",),
        parent_case_ids,
    )


def _campaign(
    cases: tuple[ScientificCase, ...],
    calibration: tuple[str, ...],
    locked: tuple[str, ...],
    *,
    preprocessing_source_ids: tuple[str, ...] = (),
) -> ScientificCampaign:
    return ScientificCampaign(
        cases,
        (
            CampaignRole("locked_evaluation", locked),
            CampaignRole("model_selection", ()),
            CampaignRole("calibration", calibration),
        ),
        preprocessing_source_ids=preprocessing_source_ids,
        criteria_ids=("family-macro-mae",),
    )


def test_related_wild_type_and_mutation_cannot_cross_roles():
    wild_type = _case("wild-type", "family-background", "preparation-wt")
    mutation = _case(
        "mutation-a7v",
        "family-background",
        "preparation-mutant",
        parent_case_ids=(wild_type.case_id,),
    )

    with pytest.raises(ValueError, match="independent_unit_id cannot cross"):
        _campaign((wild_type, mutation), (wild_type.case_id,), (mutation.case_id,))


def test_technical_replicates_from_one_preparation_cannot_cross_roles():
    first = _case("replicate-1", "unit-1", "shared-preparation")
    second = _case("replicate-2", "unit-2", "shared-preparation")

    with pytest.raises(ValueError, match="preparation_id cannot cross"):
        _campaign((first, second), (first.case_id,), (second.case_id,))


def test_acquisitions_from_one_batch_cannot_cross_roles():
    first = _case(
        "acquisition-1",
        "unit-1",
        "preparation-1",
        batch_id="shared-batch",
    )
    second = _case(
        "acquisition-2",
        "unit-2",
        "preparation-2",
        batch_id="shared-batch",
    )

    with pytest.raises(ValueError, match="batch_id cannot cross"):
        _campaign((first, second), (first.case_id,), (second.case_id,))


def test_indirect_source_ancestry_is_closed_within_one_role():
    source = _case("source", "unit-source", "preparation-source")
    repaired = _case(
        "repaired", "unit-repaired", "preparation-repaired", parent_case_ids=("source",)
    )
    derived = _case(
        "derived", "unit-derived", "preparation-derived", parent_case_ids=("repaired",)
    )

    with pytest.raises(ValueError, match="transitive closure"):
        _campaign(
            (source, repaired, derived),
            (source.case_id,),
            (repaired.case_id, derived.case_id),
        )


def test_locked_observation_cannot_supply_preprocessing():
    calibration = _case("calibration", "unit-calibration", "preparation-calibration")
    locked = _case("locked", "unit-locked", "preparation-locked")

    with pytest.raises(ValueError, match="calibration or model_selection"):
        _campaign(
            (calibration, locked),
            (calibration.case_id,),
            (locked.case_id,),
            preprocessing_source_ids=(locked.case_id,),
        )


def test_campaign_identity_is_order_independent_and_content_verified():
    calibration = _case("calibration", "unit-calibration", "preparation-calibration")
    locked = _case("locked", "unit-locked", "preparation-locked")
    first = _campaign(
        (locked, calibration),
        (calibration.case_id,),
        (locked.case_id,),
        preprocessing_source_ids=(calibration.case_id,),
    )
    ordered = _campaign(
        (calibration, locked),
        (calibration.case_id,),
        (locked.case_id,),
        preprocessing_source_ids=(calibration.case_id,),
    )
    regrouped_locked = _case(
        "locked", "different-independent-unit", "different-preparation"
    )
    regrouped = _campaign(
        (calibration, regrouped_locked),
        (calibration.case_id,),
        (regrouped_locked.case_id,),
        preprocessing_source_ids=(calibration.case_id,),
    )
    reconstructed = ScientificCampaign.from_record(first.to_record())

    assert first.campaign_id == ordered.campaign_id == reconstructed.campaign_id
    assert regrouped.campaign_id != first.campaign_id
    assert reconstructed.to_record() == first.to_record()
    assert first.case_ids == ("calibration", "locked")

    corrupted = first.to_record()
    corrupted["campaign_id"] = "not-the-content-address"
    with pytest.raises(ValueError, match="invalid content address"):
        ScientificCampaign.from_record(corrupted)


def test_distinct_preparations_may_share_a_construct_by_campaign_design():
    calibration = _case("calibration", "unit-calibration", "preparation-calibration")
    locked = _case("locked", "unit-locked", "preparation-locked")

    campaign = _campaign(
        (calibration, locked),
        (calibration.case_id,),
        (locked.case_id,),
    )

    assert campaign.independent_unit_ids == ("unit-calibration", "unit-locked")
    assert {case.construct_id for case in campaign.cases} == {"construct-shared"}


def test_duplicate_role_declarations_are_rejected():
    calibration = _case("calibration", "unit-calibration", "preparation-calibration")
    locked = _case("locked", "unit-locked", "preparation-locked")

    with pytest.raises(ValueError, match="role names must be unique"):
        ScientificCampaign(
            (calibration, locked),
            (
                CampaignRole("calibration", (calibration.case_id,)),
                CampaignRole("calibration", ()),
                CampaignRole("locked_evaluation", (locked.case_id,)),
            ),
        )
