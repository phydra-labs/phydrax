"""Real multi-process distributed domain execution beyond MACE.

Every case launches two coupled local JAX processes with two forced host CPU
devices each (four owners, gloo collectives) through
`tools.atomistic_distributed_smoke.run_scenario`, which dispatches the family
scenarios of `tools._atomistic_smoke_families`. Each rank ingresses only its
own atom rows and compares the distributed result with an independent
single-device native program of the same model and terms (parity) or with the
single-device P5 MTK NPT trajectory of the same program (dynamics). Routes the
native runtime does not admit are attempted and their typed refusal recorded.

These are functional oracles of the collective program on ONE host with
random-initialized or random source-layout models: no accelerator,
multi-host, pretrained-weight or upstream-parity claim.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from tools._atomistic_smoke_families import (
    declared_routes,
    DYNAMICS,
    Family,
    FamilyScenario,
    SCENARIO_FAMILIES,
    UMA_MAX_NEIGHBORS,
)
from tools.atomistic_distributed_smoke import run_scenario, Strategy


pytestmark = [
    pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups"),
    pytest.mark.skipif(
        not sys.platform.startswith("linux"),
        reason="requires Linux boot identities for observed host mapping",
    ),
]

# Whole-job deadline of one coupled launch (JAX bootstrap plus CPU compilation
# of every route's domain and single-device reference); a hang cannot exceed it.
DEADLINE_SECONDS = 900.0
PROCESSES = 2
DEVICES_PER_PROCESS = 2
STRATEGIES: tuple[Strategy, ...] = ("halo", "graph-partition")


def _launch(
    scenario: FamilyScenario, family: Family, strategy: Strategy, directory: Path
) -> dict[str, Any]:
    record = run_scenario(
        scenario,
        directory,
        processes=PROCESSES,
        devices_per_process=DEVICES_PER_PROCESS,
        strategy=strategy,
        family=family,
        timeout_seconds=DEADLINE_SECONDS,
    )
    assert record["passed"], record["errors"]
    ranks = record["ranks"]
    assert len(ranks) == PROCESSES
    for index, rank in enumerate(ranks):
        assert rank["family"] == family and rank["strategy"] == strategy
        assert rank["runtime"]["process_index"] == index
        assert rank["runtime"]["process_count"] == PROCESSES
        assert rank["runtime"]["observed_host_count"] == 1
        _assert_declared(rank, scenario, family)
    return record


def _assert_declared(
    rank: dict[str, Any], scenario: FamilyScenario, family: Family
) -> None:
    """Every declared route is qualified, or refused with its declared text."""
    declarations = declared_routes(scenario, family)
    routes = rank["routes"]
    assert sorted(routes) == sorted(declaration.name for declaration in declarations)
    for declaration in declarations:
        outcome = routes[declaration.name]["replicated"]
        if declaration.refusal is None:
            assert outcome["status"] == "qualified", outcome
            assert outcome["refusal"] is None
        else:
            assert outcome["status"] == "refused", outcome
            assert declaration.refusal in outcome["refusal"]["message"]


PARITY_CASES = [
    (family, strategy)
    for family in SCENARIO_FAMILIES["family-parity"]
    for strategy in STRATEGIES
]


@pytest.mark.parametrize(("family", "strategy"), PARITY_CASES)
def test_family_routes_match_the_single_device_program(
    tmp_path: Path, family: Family, strategy: Strategy
) -> None:
    record = _launch("family-parity", family, strategy, tmp_path)
    owners = PROCESSES * DEVICES_PER_PROCESS
    for rank in record["ranks"]:
        for declaration in declared_routes("family-parity", family):
            if declaration.refusal is not None:
                continue
            outcome = rank["routes"][declaration.name]["replicated"]
            assert outcome["successful"]
            assert outcome["strategy"] == strategy
            assert outcome["owner_count"] == owners
            assert sum(outcome["owned_rows_by_process"]) == rank["atom_count"]
            if family in ("aimnet2", "uma"):
                assert outcome["system_context_leaves"] > 0
            if family == "uma":
                # The declared max_neighbors selection binds on the receivers.
                assert outcome["capped_receivers"] > 0
                assert outcome["maximum_selected_degree"] >= UMA_MAX_NEIGHBORS
            if family == "mace-les":
                assert outcome["reciprocal_kind"] == "latent-ewald"
                assert outcome["global_energy"] != 0.0
            if declaration.name == "predicted-charge-pme" or family in (
                "lj-pme",
                "fourier-d3",
            ):
                assert outcome["term_energies"]
                assert all(outcome["term_capacity_successful"])
            if declaration.name == "lj-pme-dispersion":
                assert outcome["term_reciprocal_kinds"] == ["lj-pme", "pme"]
            if family == "fourier-d3":
                assert outcome["term_reciprocal_kinds"] == ["fourier-d3"]
                assert outcome["term_reciprocal_schedule"] == (
                    "pencil" if declaration.name == "fourier-d3-pencil" else "slab"
                )
    # Replicated energies are identical on every process.
    for declaration in declared_routes("family-parity", family):
        assert (
            len(
                {
                    str(rank["routes"][declaration.name]["replicated"].get("energy"))
                    for rank in record["ranks"]
                }
            )
            == 1
        )


DYNAMICS_CASES = [
    (family, strategy)
    for family in SCENARIO_FAMILIES["family-dynamics"]
    for strategy in STRATEGIES
]


@pytest.mark.parametrize(("family", "strategy"), DYNAMICS_CASES)
def test_learned_dynamic_cell_trajectory_matches_single_device_p5(
    tmp_path: Path, family: Family, strategy: Strategy
) -> None:
    record = _launch("family-dynamics", family, strategy, tmp_path)
    route, static = declared_routes("family-dynamics", family)
    declaration = DYNAMICS[family]
    for rank in record["ranks"]:
        outcome = rank["routes"][route.name]["replicated"]
        assert outcome["integrator"] == "mtk-npt"
        assert outcome["dynamic_cell"] and outcome["strain_response"]
        assert outcome["step_index"] == declaration.steps
        assert outcome["cell_epoch"] == declaration.steps
        if declaration.migrates:
            assert outcome["migrated_atoms"] >= 1
        assert all(step["successful"] for step in outcome["steps"])
        assert sum(outcome["owned_rows_by_process"]) == rank["atom_count"]
        refusal = rank["routes"][static.name]["replicated"]["refusal"]
        assert refusal["stage"] == "dynamics-plan"
