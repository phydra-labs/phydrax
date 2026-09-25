#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax.applications.relativistic_scattering._matrix_element_revision import (
    MatrixElementRevision,
)
from phydrax.artifacts import DerivativeEstimatorKind, DerivativeEvidence
from phydrax.particle_physics._capabilities import HEPProviderBinding
from phydrax.particle_physics._host_events import HostEventRecord, HostEventWeight
from phydrax.particle_physics._identity import ReproducibilityGrade
from phydrax.particle_physics._matching_runtime import (
    assign_exclusive_matching_bins,
    ProviderExecutionRecord,
    ProviderExecutionStage,
    ProviderExecutionStatus,
    record_provider_execution,
)
from phydrax.particle_physics._operations import ProcessNormalization
from phydrax.particle_physics._weights import WeightVariationKind
from phydrax.qualification import CapabilityProfile, SupportTuple


def _provider_contracts():
    profile = CapabilityProfile(
        "external-shower",
        "provider-x",
        "1.2.3",
        (SupportTuple("hep.shower", {"event_record": "host", "signed_weights": True}),),
    )
    differentiation = DerivativeEvidence(
        phx.DerivativeContract(route=phx.DerivativeRoute.DIRECT),
        estimator=DerivativeEstimatorKind.UNSUPPORTED,
        discrete_parameters=("tune",),
        stopped_events=("accept-reject",),
        support_id="external-shower-discrete",
    )
    binding = HEPProviderBinding(
        profile,
        differentiation,
        configuration_checksum="configuration-sha256",
        input_profile_ids=("hard-events",),
        output_profile_ids=("showered-events",),
        unit_ids=("GeV",),
        frame_ids=("collision-cm",),
        devices=("cpu",),
        dtypes=("float64",),
        side_effects=("subprocess",),
        license_ids=("provider-license",),
        reproducibility=ReproducibilityGrade.SEED_REPLAYABLE,
    )
    normalization = ProcessNormalization(
        process_id="dark-pair",
        attempted_count=1,
        generated_count=1,
        accepted_count=1,
        positive_count=0,
        negative_count=1,
        sum_weights=-2.0,
        sum_absolute_weights=2.0,
        sum_squared_weights=4.0,
        cross_section=1.0,
        cross_section_uncertainty=0.1,
        cross_section_unit_id="pb",
        provider_id="provider-x",
    )
    revision = MatrixElementRevision(
        "dark-pair",
        "dark-model-r1",
        "provider-x",
        "provider-rights",
        normalization.normalization_id,
        "hard-process-support",
        "proposal-r1",
        "no-adaptation",
        "training-snapshot",
        "optimizer-none",
        "error-model-r1",
        "a" * 64,
        differentiation_id=differentiation.evidence_id,
    )
    return binding, normalization, revision


def _event(source, nominal, shape):
    return HostEventRecord(
        7,
        0,
        (),
        (),
        (
            HostEventWeight("nominal", nominal, WeightVariationKind.NOMINAL, "nominal"),
            HostEventWeight("scale-up", shape, WeightVariationKind.SHAPE, "scale"),
        ),
        "provider-x-status",
        source,
    )


def test_exclusive_matching_bins_are_disjoint_with_inclusive_highest_multiplicity():
    assignment = assign_exclusive_matching_bins(
        jnp.asarray(
            ((30.0, 0.0, 0.0), (30.0, 20.0, 0.0), (30.0, 20.0, 15.0), (5.0, 0.0, 0.0))
        ),
        jnp.asarray(
            (
                (True, False, False),
                (True, True, False),
                (True, True, True),
                (True, False, False),
            )
        ),
        multiplicities=(0, 1, 2),
        merging_scale=10.0,
    )
    np.testing.assert_array_equal(assignment.multiplicity, jnp.asarray((1, 2, 2, 0)))
    assert jnp.all(assignment.accepted)
    np.testing.assert_array_equal(assignment.resolved_count, jnp.asarray((1, 2, 3, 0)))


def test_provider_record_preserves_signed_named_weights_normalization_and_revision_provenance():
    binding, normalization, revision = _provider_contracts()
    input_event = _event("hard-source", -2.0, -1.8)
    output_event = _event("shower-source", -2.0, -1.9)
    record = record_provider_execution(
        binding,
        input_event,
        output_event,
        normalization,
        revision,
        stage=ProviderExecutionStage.SHOWER,
        required_capability="hep.shower",
        input_profile_id="hard-events",
        output_profile_id="showered-events",
        unit_contract_id="GeV",
        frame_id="collision-cm",
        frame_realization_id="f" * 64,
        process_id="dark-pair",
    )
    assert isinstance(record, ProviderExecutionRecord)
    assert record.successful
    assert record.unit_contract_id == "GeV"
    assert record.frame_id == "collision-cm"
    assert record.frame_realization_id == "f" * 64
    assert record.matrix_element_revision.revision_id == revision.revision_id
    assert record.binding.binding_id == binding.binding_id
    assert record.input_weights.names == ("nominal", "scale-up")
    np.testing.assert_allclose(record.input_weights.values, ((-2.0, -1.8),))
    np.testing.assert_allclose(record.output_weights.values, ((-2.0, -1.9),))
    assert record.normalization.sum_weights == -2.0


def test_provider_capability_mismatch_is_explicit_not_silently_accepted():
    binding, normalization, revision = _provider_contracts()
    record = record_provider_execution(
        binding,
        _event("hard-source", -2.0, -1.8),
        _event("shower-source", -2.0, -1.9),
        normalization,
        revision,
        stage="hadronization",
        required_capability="hep.hadronization",
        input_profile_id="hard-events",
        output_profile_id="showered-events",
        unit_contract_id="GeV",
        frame_id="collision-cm",
        frame_realization_id="f" * 64,
        process_id="dark-pair",
    )
    assert record.status is ProviderExecutionStatus.CAPABILITY_MISMATCH
    assert not record.successful
