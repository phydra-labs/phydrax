#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Explicit candidate selection or externally authenticated battery admission."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from phydrax.applications import battery
from phydrax.qualification import CapabilityProfile, RuntimeDistributionAttestation


def add_admission_arguments(parser: argparse.ArgumentParser, /) -> None:
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--development",
        action="store_true",
        help="Run the unreleased equation candidate.",
    )
    mode.add_argument(
        "--deployment",
        type=Path,
        help="Signed release index and retained typed proofs JSON.",
    )
    parser.add_argument(
        "--trust-roots",
        type=Path,
        help="Separately provisioned public keys and role policy JSON.",
    )
    parser.add_argument("--profile-id", help="Exact released profile content ID.")
    parser.add_argument(
        "--runtime-attestation",
        type=Path,
        help="Executor-signed installed-runtime byte attestation JSON.",
    )
    parser.add_argument(
        "--distribution-id", help="Verified installed distribution content ID."
    )
    parser.add_argument("--admission-lifetime-s", type=int, default=300)
    parser.add_argument("--maximum-index-age-s", type=int, default=86400)


def _unique_object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"Duplicate admission record field: {key}")
        record[key] = value
    return record


def _nonfinite(token):
    raise ValueError(f"Nonfinite admission record value: {token}")


def _record(path: Path, /) -> dict:
    record = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_nonfinite,
    )
    if not isinstance(record, dict):
        raise ValueError("Admission artifact must contain one JSON object.")
    return record


def resolve_example_admission(
    parser: argparse.ArgumentParser,
    arguments: argparse.Namespace,
    candidate: CapabilityProfile,
    /,
) -> tuple[CapabilityProfile, battery.BatteryExecutionAdmission | None, str | None]:
    if arguments.development:
        if any(
            value is not None
            for value in (
                arguments.trust_roots,
                arguments.profile_id,
                arguments.distribution_id,
                arguments.runtime_attestation,
            )
        ):
            parser.error(
                "Development execution cannot consume release credentials or distribution claims."
            )
        print("admission_scope=unreleased-development")
        return candidate, None, None
    if any(
        value is None
        for value in (
            arguments.trust_roots,
            arguments.profile_id,
            arguments.distribution_id,
            arguments.runtime_attestation,
        )
    ):
        parser.error(
            "Production execution requires --trust-roots, --profile-id, --distribution-id, and --runtime-attestation."
        )
    if (
        not 0 < arguments.admission_lifetime_s <= 86400
        or not 0 < arguments.maximum_index_age_s <= 86400
    ):
        parser.error(
            "Admission lifetime and maximum index age must be within 1..86400 seconds."
        )
    now = time.time_ns()
    admission = battery.load_battery_execution_admission(
        _record(arguments.deployment),
        _record(arguments.trust_roots),
        profile_id=arguments.profile_id,
        support_tuple_id=candidate.support_tuples[0].support_tuple_id,
        distribution_id=arguments.distribution_id,
        at_time=now,
        expires_at=now + arguments.admission_lifetime_s * 1_000_000_000,
        max_index_age=arguments.maximum_index_age_s * 1_000_000_000,
        runtime_attestation=RuntimeDistributionAttestation.from_record(
            _record(arguments.runtime_attestation)
        ),
    )
    print("admission_scope=numerical-equation-support-only")
    return admission.profile, admission, admission.distribution_id
