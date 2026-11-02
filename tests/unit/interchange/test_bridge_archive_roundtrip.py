from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path
from collections.abc import Callable

import numpy as np

from phydrax.meshing._model_codec import read_model_archive, write_model_archive

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = Path(__file__).with_name("_bridge_archive_worker.py")


def _run(
    mode: str, directory: Path, expected: object | None = None
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(_WORKER), mode, str(directory)]
    if expected is not None:
        command.append(json.dumps(expected))
    return subprocess.run(command, cwd=_ROOT, check=False, capture_output=True, text=True)


def _find(node: object, predicate: Callable[[dict], bool]) -> dict | None:
    if isinstance(node, dict):
        if predicate(node):
            return node
        children = node.values()
    elif isinstance(node, list):
        children = node
    else:
        return None
    for child in children:
        found = _find(child, predicate)
        if found is not None:
            return found
    return None


def test_bridge_patch_cold_archive_replays_exact_seams_and_embedding(
    tmp_path: Path,
) -> None:
    written = _run("write", tmp_path)
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    assert len(expected["patches"]) == 3
    assert expected["patches"][2]["degrees"] == [3, 5]
    restored = _run("read", tmp_path, expected)
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "bridge-cold-restored"

    manifest, arrays = read_model_archive(tmp_path / "bridge.phx", limits=None)
    changed = copy.deepcopy(manifest)
    patch = _find(
        changed["model_recipe"],
        lambda n: str(n.get("type", "")).rsplit(":", 1)[-1] == "BSplineSurfacePatch",
    )
    assert patch is not None
    scalar = _find(
        patch["fields"]["exact_control_point_bank"], lambda n: n.get("kind") == "fraction"
    )
    assert scalar is not None
    scalar["numerator"] = int(scalar["numerator"]) + 1
    tampered = tmp_path / "tampered"
    tampered.mkdir()
    write_model_archive(
        tampered / "bridge.phx",
        manifest=changed,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    for name in ("left", "right", "left-model", "right-model", "published-model"):
        shutil.copyfile(tmp_path / f"{name}.phx", tampered / f"{name}.phx")
    refused = _run("read", tampered, expected)
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr

    manifest, arrays = read_model_archive(tmp_path / "published-model.phx", limits=None)
    changed = copy.deepcopy(manifest)
    patch = _find(
        changed["model_recipe"],
        lambda n: str(n.get("type", "")).rsplit(":", 1)[-1] == "BSplineSurfacePatch",
    )
    assert patch is not None
    scalar = _find(
        patch["fields"]["exact_control_point_bank"], lambda n: n.get("kind") == "fraction"
    )
    assert scalar is not None
    scalar["numerator"] = int(scalar["numerator"]) + 1
    model_tampered = tmp_path / "model-tampered"
    model_tampered.mkdir()
    write_model_archive(
        model_tampered / "published-model.phx",
        manifest=changed,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    for name in ("left", "right", "bridge", "left-model", "right-model"):
        shutil.copyfile(tmp_path / f"{name}.phx", model_tampered / f"{name}.phx")
    refused = _run("read", model_tampered, expected)
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr


def test_rooted_bridge_cold_archive_replays_lifts_and_refuses_tamper(
    tmp_path: Path,
) -> None:
    written = _run("write-rooted", tmp_path)
    assert written.returncode == 0, written.stderr
    expected = json.loads(written.stdout)
    assert expected["root"]["source_lift_count"] >= 4
    assert expected["models"][0]["vertex_roots"][3] is not None
    assert any(any(pair) for pair in expected["models"][0]["edge_endpoints"])
    assert any(any(pair) for pair in expected["models"][0]["coedge_endpoints"])
    assert len(expected["root"]["coedge_root_ids"]) == 2
    restored = _run("read-rooted", tmp_path, expected)
    assert restored.returncode == 0, restored.stderr
    assert restored.stdout.strip() == "bridge-cold-restored"

    manifest, arrays = read_model_archive(tmp_path / "published-model.phx", limits=None)
    changed = copy.deepcopy(manifest)
    endpoint = _find(
        changed["model_recipe"],
        lambda n: str(n.get("type", "")).rsplit(":", 1)[-1] == "ConicParameterEndpoint",
    )
    assert endpoint is not None
    scalar = _find(
        endpoint["fields"]["point_exact"], lambda n: n.get("kind") == "fraction"
    )
    assert scalar is not None
    scalar["numerator"] = int(scalar["numerator"]) + 1
    tampered = tmp_path / "root-tampered"
    tampered.mkdir()
    write_model_archive(
        tampered / "published-model.phx",
        manifest=changed,
        arrays={name: np.array(array, copy=True) for name, array in arrays.items()},
        limits=None,
    )
    for name in ("left", "right", "bridge", "left-model", "right-model"):
        shutil.copyfile(tmp_path / f"{name}.phx", tampered / f"{name}.phx")
    refused = _run("read-rooted", tampered, expected)
    assert refused.returncode != 0
    assert "incompatible with its registered recipe" in refused.stderr
