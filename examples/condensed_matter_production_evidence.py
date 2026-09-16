#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Public API smoke for periodic execution, lifecycle, and closure evidence."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import jax.numpy as jnp

from phydrax.applications import (
    condensed_matter_candidate_campaigns,
    condensed_matter_candidate_closure,
    condensed_matter_candidate_profiles,
    condensed_matter_frontier_candidate_campaigns,
    condensed_matter_frontier_candidate_profiles,
)
from phydrax.chemistry.periodic import (
    periodic_candidate_profiles,
    read_periodic_artifact_archive,
    write_periodic_artifact_archive,
)
from phydrax.lifecycle import ArrayArtifactProvenance
from phydrax.operators.periodic import (
    PeriodicTranslationFamilyPlan,
    PeriodicTranslationFamilyState,
    prepare_periodic_translation_family,
)
from phydrax.sparse import EdgeRelation


def main() -> None:
    relation = EdgeRelation(
        [0, 0, 0],
        [0, 0, 0],
        source_size=1,
        target_size=1,
    )
    plan = PeriodicTranslationFamilyPlan(
        relation,
        [[-1], [0], [1]],
        [2, 1, 0],
    )
    state = PeriodicTranslationFamilyState(plan, [[[-1.0]], [[0.25]], [[-1.0]]])
    prepared = prepare_periodic_translation_family(plan, state)
    points = jnp.asarray([[0.0], [0.25], [0.5]])
    before = prepared.evaluate(points)
    profile = next(
        candidate
        for candidate in periodic_candidate_profiles()
        if candidate.capability == "chemistry.periodic.pencil.orthonormal"
    )

    provenance = ArrayArtifactProvenance(
        "phydrax-example",
        ("example:analytic-one-orbital-chain",),
        (profile.profile_id,),
        ("unit:declared-model-energy",),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "periodic-state.pxa"
        written = write_periodic_artifact_archive(path, state, provenance)
        restored, reopened = read_periodic_artifact_archive(path, state, provenance)
        after = prepare_periodic_translation_family(plan, restored).evaluate(points)

    if not bool(jnp.array_equal(before, after)):
        raise RuntimeError("Periodic lifecycle replay changed the evaluated family.")
    if written.artifact_id != reopened.artifact_id:
        raise RuntimeError("Periodic lifecycle replay changed artifact identity.")

    closure = condensed_matter_candidate_closure()
    campaigns = condensed_matter_candidate_campaigns()
    profiles = condensed_matter_candidate_profiles()
    frontier_profiles = condensed_matter_frontier_candidate_profiles()
    frontier_campaigns = condensed_matter_frontier_candidate_campaigns()
    print(
        json.dumps(
            {
                "periodic_values": before.real[:, 0, 0].tolist(),
                "artifact_id": written.artifact_id,
                "candidate_dependency_count": len(closure.dependencies),
                "candidate_ledger_id": closure.ledger_id,
                "owner_campaign_count": len(campaigns.campaigns),
                "candidate_profile_count": len(profiles),
                "campaign_owners": sorted(
                    {reference.owner_id for reference in campaigns.campaigns}
                ),
                "frontier_profile_count": len(frontier_profiles),
                "frontier_campaign_owners": sorted(
                    {reference.owner_id for reference in frontier_campaigns.campaigns}
                ),
                "release_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
