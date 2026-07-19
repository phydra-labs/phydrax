#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import json
import zipfile

import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.uq._checkpoint import (
    checkpoint_compatibility,
    pack_array_tree,
    read_checkpoint_archive,
    unpack_array_tree,
    write_checkpoint_archive,
)


def _problem(observation=0.5):
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            {"offset": jnp.zeros(2), "scale": jnp.asarray(0.0)},
            priors={
                "offset": phx.uq.Normal(0.0, 1.0),
                "scale": phx.uq.LogNormal(0.0, 0.5),
            },
            bijectors={
                "offset": phx.uq.IdentityBijector(),
                "scale": phx.uq.ExpBijector(),
            },
        ),
        lambda value: -0.5 * jnp.sum((value["offset"] - observation) ** 2),
    )


def test_checkpoint_archive_is_atomic_versioned_and_pickle_free(tmp_path):
    problem = _problem()
    compatibility = checkpoint_compatibility(
        problem,
        checkpoint_id="run-a",
        settings={"algorithm": "nuts", "num_chains": 2},
    )
    arrays = {}
    tree_specification = pack_array_tree(
        "position",
        problem.initial_position,
        arrays,
    )
    destination = write_checkpoint_archive(
        tmp_path / "nested" / "state.phxckpt",
        kind="mcmc",
        compatibility=compatibility,
        state={"completed": 4, "position_tree": tree_specification},
        arrays=arrays,
    )

    with zipfile.ZipFile(destination) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["format"] == "phydrax-uq-checkpoint"
        assert manifest["schema_version"] == 1
        assert all(name.endswith(".npy") for name in archive.namelist()[1:])

    state, loaded_arrays = read_checkpoint_archive(
        destination,
        kind="mcmc",
        compatibility=compatibility,
    )
    restored = unpack_array_tree(
        state["position_tree"],
        loaded_arrays,
        problem.initial_position,
    )
    assert state["completed"] == 4
    assert jnp.array_equal(restored["offset"], problem.initial_position["offset"])
    assert jnp.array_equal(restored["scale"], problem.initial_position["scale"])
    assert not tuple(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_checkpoint_rejects_incompatible_problem_settings_and_corruption(tmp_path):
    problem = _problem()
    compatibility = checkpoint_compatibility(
        problem,
        checkpoint_id="run-a",
        settings={"algorithm": "hmc", "num_chains": 2},
    )
    destination = write_checkpoint_archive(
        tmp_path / "state.phxckpt",
        kind="mcmc",
        compatibility=compatibility,
        state={"completed": 0},
        arrays={"position": jnp.zeros(2)},
    )

    mismatched = checkpoint_compatibility(
        problem,
        checkpoint_id="run-b",
        settings={"algorithm": "hmc", "num_chains": 2},
    )
    with pytest.raises(phx.uq.CheckpointCompatibilityError, match="checkpoint id"):
        read_checkpoint_archive(
            destination,
            kind="mcmc",
            compatibility=mismatched,
        )

    destination.write_bytes(destination.read_bytes()[:40])
    with pytest.raises(phx.uq.CheckpointCorruptionError, match="Cannot read"):
        read_checkpoint_archive(
            destination,
            kind="mcmc",
            compatibility=compatibility,
        )
