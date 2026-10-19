#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from dataclasses import replace

import pytest

from phydrax._model._structure import model_from_structure_recipe, model_structure_recipe
from phydrax.geometry.design import DerivativeTier, DesignQualificationEvidence
from phydrax.lifecycle import register_meshing_source_artifacts
from phydrax.meshing import FixedEpochDerivativeEvidence, MeasuredAdaptationCost
from tests._support.recipes import recipe_field_position


@pytest.fixture(scope="module", autouse=True)
def registered_artifacts() -> None:
    register_meshing_source_artifacts()


@pytest.fixture
def cost() -> MeasuredAdaptationCost:
    return MeasuredAdaptationCost(
        "sample",
        "candidate",
        preparation_seconds=1.0,
        compilation_seconds=2.0,
        solve_seconds=3.0,
        transfer_seconds=4.0,
        reanalysis_seconds=5.0,
        decision_seconds=6.0,
        peak_memory_bytes=1024,
    )


def test_measured_cost_reconstruction_preserves_phase_roles_and_identity(
    cost: MeasuredAdaptationCost,
) -> None:
    restored = MeasuredAdaptationCost.from_stored_fields(cost)
    assert restored is not cost
    assert restored.phase_seconds == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert restored.total_seconds == 21.0
    assert restored.peak_memory_bytes == cost.peak_memory_bytes
    assert restored.cost_id == cost.cost_id


@pytest.fixture
def evidence() -> FixedEpochDerivativeEvidence:
    geometry = DesignQualificationEvidence(
        DerivativeTier.Q1,
        "geometry-analysis",
        "epoch",
        "geometry-plan",
        event_ids=("geometry-margin",),
        event_margins=(1.0,),
        valid=True,
    )
    pde = DesignQualificationEvidence(
        DerivativeTier.Q2,
        "pde-analysis",
        "epoch",
        "pde-plan",
        event_ids=("pde-margin",),
        event_margins=(2.0,),
        valid=True,
    )
    transfer = DesignQualificationEvidence(
        DerivativeTier.Q3,
        "transfer-analysis",
        "epoch",
        "transfer-plan",
        event_ids=("transfer-margin",),
        event_margins=(3.0,),
        valid=True,
    )
    return FixedEpochDerivativeEvidence(
        "epoch",
        "parameters",
        geometry=geometry,
        pde=pde,
        transfer=transfer,
        routes=(
            ("geometry", "geometry-plan"),
            ("pde", "pde-plan"),
            ("transfer", "transfer-plan"),
        ),
    )


def test_fixed_epoch_reconstruction_preserves_qualification_roles_and_identity(
    evidence: FixedEpochDerivativeEvidence,
) -> None:
    geometry, pde, transfer = evidence.qualifications
    restored = FixedEpochDerivativeEvidence.from_stored_fields(evidence)
    assert restored is not evidence
    assert restored.qualifications == (geometry, pde, transfer)
    assert restored.qualifications[0] is geometry
    assert restored.qualifications[1] is pde
    assert restored.qualifications[2] is transfer
    assert restored.routes == evidence.routes
    assert restored.evidence_id == evidence.evidence_id
    restored.require_valid(
        "epoch",
        "parameters",
        routes=evidence.routes,
        event_margins=(
            ("geometry-margin", 1.0),
            ("pde-margin", 2.0),
            ("transfer-margin", 3.0),
        ),
    )


@pytest.mark.parametrize(
    ("phases", "message"),
    (
        ((1.0, 2.0), "phase table"),
        ((1.0, 2.0, -3.0, 4.0, 5.0, 6.0), "solve_seconds"),
    ),
    ids=("missing-phases", "negative-solve-cost"),
)
def test_measured_cost_reconstruction_refuses_corrupt_phases(
    cost: MeasuredAdaptationCost,
    phases: tuple[float, ...],
    message: str,
) -> None:
    recipe = model_structure_recipe(cost)
    recipe["items"][recipe_field_position(recipe, "phase_seconds")] = (
        model_structure_recipe(phases)
    )
    corrupted = model_from_structure_recipe(recipe)
    with pytest.raises(ValueError, match=message):
        MeasuredAdaptationCost.from_stored_fields(corrupted)


def test_fixed_epoch_reconstruction_refuses_missing_qualification(
    evidence: FixedEpochDerivativeEvidence,
) -> None:
    recipe = model_structure_recipe(evidence)
    recipe["items"][recipe_field_position(recipe, "qualifications")] = (
        model_structure_recipe(evidence.qualifications[:2])
    )
    corrupted = model_from_structure_recipe(recipe)
    with pytest.raises(ValueError, match="three"):
        FixedEpochDerivativeEvidence.from_stored_fields(corrupted)


def test_fixed_epoch_reconstruction_refuses_foreign_qualification_epoch(
    evidence: FixedEpochDerivativeEvidence,
) -> None:
    geometry, pde, transfer = evidence.qualifications
    foreign = replace(pde, numeric_revision_id="foreign")
    recipe = model_structure_recipe(evidence)
    recipe["items"][recipe_field_position(recipe, "qualifications")] = (
        model_structure_recipe((geometry, foreign, transfer))
    )
    corrupted = model_from_structure_recipe(recipe)
    with pytest.raises(ValueError, match="numeric epoch"):
        FixedEpochDerivativeEvidence.from_stored_fields(corrupted)
