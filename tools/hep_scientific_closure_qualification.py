#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import phydrax as phx


def qualification_record():
    criterion = phx.qualification.ScientificMetricCriterion(
        "maximum-relative-residual",
        "at_most",
        None,
        1.0e-8,
        "1",
        "worst_stratum",
    )
    bundle = phx.particle_physics.build_hep_qualification_bundle(
        capability_name="hep.scientific-closure",
        support_attributes={"profile": "reference", "device": "cpu", "dtype": "float64"},
        profile_name="hep.scientific-closure.reference",
        provider="phydrax",
        provider_version="current",
        release_evidence=(),
        released=False,
        campaign_id="hep-scientific-closure-campaign",
        observable_ids=("energy", "rate", "likelihood"),
        condition_domain_ids=("reference-domain",),
        required_stage_ids=(
            "source-admission",
            "numerical-validity",
            "external-transfer",
        ),
        criteria=(criterion,),
        frozen_criteria_ids=(criterion.criterion_id,),
        abstention_policy_id="fail-closed",
        invalidation_triggers=("provider-change", "conditions-change", "support-change"),
    )
    return {
        "bundle_id": bundle.bundle_id,
        "capability_profile_id": bundle.capability.profile_id,
        "scientific_claim_id": bundle.scientific_claim.claim_id,
        "released": bundle.capability.released,
        "status": "inconclusive",
        "missing_release_gates": list(phx.particle_physics.HEP_RELEASE_GATES),
        "external_authority_required": True,
        "nonclaims": [
            "no live accelerator or trigger control",
            "no official experiment calibration or data-quality certification",
            "no replacement for provider-owned generators, transport, reconstruction, or statistics",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = qualification_record()
    payload = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0 if record["released"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
