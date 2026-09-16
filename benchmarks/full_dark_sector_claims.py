#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax.numpy as jnp

from benchmarks._runtime import capture_environment
from phydrax._physical import DimensionalScaleContract, RelativityScaleContract
from phydrax.applications.cosmology import _full_dark_sector_claims as claims
from phydrax.applications.relativistic_scattering._unit_contract import (
    LocalRelativisticFramePlan,
    RelativisticUnitContract,
)
from phydrax.metrix import (
    ADMGridGeometry,
    CoordinateChart,
    minkowski_metric,
    orthonormal_tetrad,
    RelativityConvention,
)
from phydrax.qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)


_FACTORIES = (
    ("relativistic-stress-energy-pm", claims.relativistic_stress_energy_pm_claim_profile),
    ("einstein-vlasov-z4c", claims.einstein_vlasov_z4c_claim_profile),
    ("dynamic-epoch-runtime", claims.dynamic_epoch_runtime_claim_profile),
    ("fixed-multiplicity", claims.fixed_multiplicity_claim_profile),
    ("parton-shower", claims.parton_shower_claim_profile),
    ("hadronization", claims.hadronization_claim_profile),
    ("quantum-uu", claims.quantum_uu_claim_profile),
    ("thermal-qft", claims.thermal_qft_claim_profile),
    ("coherent-qke", claims.coherent_qke_claim_profile),
    ("off-shell-kb", claims.off_shell_kb_claim_profile),
    ("radiation-packet", claims.radiation_packet_claim_profile),
    ("radiation-m1", claims.radiation_m1_claim_profile),
    ("radiation-vet", claims.radiation_vet_claim_profile),
    ("radiation-hierarchy", claims.radiation_hierarchy_claim_profile),
    ("fully-coupled-closure", claims.fully_coupled_closure_claim_profile),
)


def _frame():
    units = RelativisticUnitContract(
        RelativityScaleContract(DimensionalScaleContract.si(), 1, 3, 2, 1),
        RelativityConvention(metric_signature="mostly_minus"),
    )
    geometry = ADMGridGeometry(
        jnp.asarray(1.0),
        jnp.zeros((3,)),
        jnp.eye(3),
        jnp.eye(3),
        jnp.asarray(1.0),
        jnp.zeros((3, 3)),
        jnp.asarray(True),
        jnp.asarray(True),
        snapshot_token=jnp.asarray(0),
        chart_id="benchmark-minkowski",
        convention_id=units.convention.convention_id,
        scale_id=units.scale.scale_id,
        topology_id="benchmark-single-cell",
        geometry_lineage_id="benchmark-flat-adm",
    )
    chart = CoordinateChart("benchmark-minkowski", ("t", "x", "y", "z"))
    tetrad = orthonormal_tetrad(
        minkowski_metric(chart, convention="mostly_minus"),
        jnp.eye(4),
        jnp.zeros((4,)),
        convention=units.convention,
        source_id="benchmark-observer",
    )
    return units, LocalRelativisticFramePlan(
        geometry,
        tetrad,
        units,
        jnp.zeros((4,)),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
        observer_id="benchmark-observer",
        orientation_id="future-right-handed",
    )


def _campaign(criteria):
    calibration = ScientificCase(
        "benchmark-calibration",
        "benchmark-calibration-unit",
        "dark-sector",
        "calibration",
        "prepared",
        "calibration-batch",
        ("benchmark-source",),
    )
    locked = ScientificCase(
        "benchmark-locked",
        "benchmark-locked-unit",
        "dark-sector",
        "locked",
        "prepared-locked",
        "locked-batch",
        ("benchmark-source",),
    )
    return ScientificCampaign(
        (calibration, locked),
        (
            CampaignRole("calibration", (calibration.case_id,)),
            CampaignRole("locked_evaluation", (locked.case_id,)),
        ),
        criteria_ids=tuple(value.criterion_id for value in criteria),
    )


def _reference():
    return ReferenceArtifactManifest(
        "benchmark-full-dark-sector-reference",
        checksum_algorithm="sha256",
        checksum="0" * 64,
        size_bytes=1,
        license_id="benchmark-internal",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="benchmark-local",
        nondimensionalization={"energy": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("benchmark-reference",),
    )


def _inputs():
    units, frame = _frame()
    reference = _reference()
    requested_use = claims.FullDarkSectorReferenceUse()
    values = []
    for name, factory in _FACTORIES:
        metric_ids = claims.full_dark_sector_claim_metric_ids(name)
        criteria = claims.full_dark_sector_claim_criteria(
            name,
            {metric_id: 1.0 for metric_id in metric_ids},
        )
        values.append((name, factory, criteria, _campaign(criteria)))
    return units, frame, reference, requested_use, tuple(values)


def _construct_all(units, frame, reference, requested_use, values):
    return tuple(
        factory(
            campaign,
            criteria,
            (f"benchmark-{name}",),
            {
                "backend": "cpu",
                "precision": "float64",
                "maximum-resident-values": 4096,
                "provider-revision": "benchmark-pinned",
            },
            units,
            frame,
            reference_artifacts=(reference,),
            requested_use=requested_use,
        )
        for name, factory, criteria, campaign in values
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.repeats < 1:
        raise ValueError("warmup must be non-negative and repeats must be positive")

    units, frame, reference, requested_use, values = _inputs()
    expected = _construct_all(units, frame, reference, requested_use, values)
    for _ in range(arguments.warmup):
        _construct_all(units, frame, reference, requested_use, values)

    durations = []
    latest = expected
    for _ in range(arguments.repeats):
        started = time.perf_counter()
        latest = _construct_all(units, frame, reference, requested_use, values)
        durations.append(time.perf_counter() - started)

    identities_match = tuple(value.claim_id for value in latest) == tuple(
        value.claim_id for value in expected
    )
    unique_claims = len({value.claim_id for value in expected}) == len(expected)
    channels = tuple(
        claims.full_dark_sector_promotion_channel(value) for value in expected
    )
    unique_channels = len(set(channels)) == len(channels)
    successful = identities_match and unique_claims and unique_channels
    total_seconds = sum(durations)
    payload = {
        "environment": capture_environment().to_dict(),
        "support": {
            "profile_count": len(expected),
            "unit_contract_id": units.contract_id,
            "frame_id": frame.frame_id,
            "frame_realization_id": frame.realization_id(),
            "frame_snapshot_token": int(frame.frame_token),
            "reference_manifest_id": reference.manifest_id,
            "fixed_capacity": True,
            "semantic_unboundedness": "durable-finite-epoch-chain",
        },
        "execution": {
            "repeats": arguments.repeats,
            "seconds_total": total_seconds,
            "seconds_mean": total_seconds / arguments.repeats,
            "claims_per_second": (
                arguments.repeats * len(expected) / total_seconds
                if total_seconds > 0.0
                else None
            ),
        },
        "profiles": {
            name: {
                "claim_id": claim.claim_id,
                "support_tuple_id": claim.support.support_tuple_id,
                "promotion_channel": channel,
                "criterion_count": len(claim.criteria),
                "differentiation_contract_id": claims.full_dark_sector_differentiation_contract(
                    name
                ).contract_id,
                "checkpoint_product": dict(claim.support.attributes)[
                    "checkpoint_product"
                ],
                "analysis_output_products": dict(claim.support.attributes)[
                    "analysis_output_products"
                ],
            }
            for (name, _), claim, channel in zip(
                _FACTORIES, expected, channels, strict=True
            )
        },
        "verification": {
            "deterministic_identity": identities_match,
            "independent_claim_ids": unique_claims,
            "independent_promotion_channels": unique_channels,
        },
        "successful": successful,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(text)
    else:
        arguments.output.write_text(text + "\n", encoding="utf-8")
    if not successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
