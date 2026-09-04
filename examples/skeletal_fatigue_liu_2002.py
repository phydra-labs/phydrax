#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Advance Liu--Brown--Yue (2002) fatigue and recovery compartments."""

from __future__ import annotations

from phydrax.applications.skeletal_muscle.fatigue import (
    commit_liu_brown_yue_2002,
    LiuBrownYue2002Parameters,
    LiuBrownYue2002Plan,
)


def main() -> None:
    prepared = LiuBrownYue2002Plan(
        LiuBrownYue2002Parameters(
            fatigue_rate_per_s=0.0206,
            recovery_rate_per_s=0.0084,
        ),
        muscle_id="example-handgrip",
        protocol_id="piecewise-constant-brain-effort",
    ).prepare()
    state = prepared.initialize()
    for _ in range(120):
        candidate = prepared.evaluate(state, 1.0, 1.0)
        if not bool(candidate.evidence.successful):
            raise RuntimeError(f"fatigue step status={int(candidate.evidence.status)}")
        state = commit_liu_brown_yue_2002(candidate, state)
    capacity = prepared.capacity(state)
    print(
        f"sustained_time_s={float(capacity.time_s):.0f} "
        f"active_relative_force={float(capacity.active_relative_force):.6f} "
        f"fatigued_fraction={float(capacity.fatigued_fraction):.6f}"
    )

    # A separate high-fatigue control demonstrates the source R flow. The
    # 2002 model does not define a later intermittent-task rest multiplier.
    recovery_state = prepared.initialize(
        uncommitted_fraction=0.0,
        active_fraction=0.05,
        fatigued_fraction=0.95,
    )
    for _ in range(60):
        candidate = prepared.evaluate(recovery_state, 0.0, 1.0)
        if not bool(candidate.evidence.successful):
            raise RuntimeError(f"recovery step status={int(candidate.evidence.status)}")
        recovery_state = commit_liu_brown_yue_2002(candidate, recovery_state)
    recovered = prepared.capacity(recovery_state)
    print(
        f"recovery_control_time_s={float(recovered.time_s):.0f} "
        f"active_relative_force={float(recovered.active_relative_force):.6f} "
        f"fatigued_fraction={float(recovered.fatigued_fraction):.6f}"
    )
    print("route=standalone LiuBrownYue2002; no D1 composition")


if __name__ == "__main__":
    main()
