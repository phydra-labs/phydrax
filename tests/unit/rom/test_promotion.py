#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import phydrax as phx


def test_rom_promotion_requires_accuracy_speed_memory_refusal_and_resume() -> None:
    passed = phx.rom.ROMPromotionEvidence(
        "rom.affine-steady",
        "support:affine-steady",
        relative_error=1.0e-3,
        rollout_error=2.0e-3,
        speedup=5.0,
        memory_ratio=0.2,
        support_refusal_passed=True,
        exact_resume_passed=True,
        evidence_ids=("accuracy", "performance", "resume"),
    )
    failed = phx.rom.ROMPromotionEvidence(
        "rom.affine-steady",
        "support:affine-steady",
        relative_error=1.0e-3,
        rollout_error=2.0e-3,
        speedup=1.1,
        memory_ratio=0.9,
        support_refusal_passed=True,
        exact_resume_passed=True,
        evidence_ids=("accuracy", "performance", "resume"),
    )

    assert passed.passed
    assert passed.evidence_id
    assert not failed.passed
