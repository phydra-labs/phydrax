#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Build, clean-install, and identify an unpromoted battery distribution.

This records artifact and installed-package identities, not scientific qualification,
license clearance, independent replay, or release authorization. Output must be outside
of the source tree. The immutable manifest is written only after all checks succeed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from zipfile import ZipFile

from phydrax._fingerprint import canonical_fingerprint
from phydrax.qualification._runtime_distribution import parse_distribution_manifest
from tools.battery_qualification import execution_source_build_id


_INVENTORY_SCRIPT = """
import hashlib
import importlib.metadata
import json
import pathlib
import sys
import phydrax

packages = []
for distribution in importlib.metadata.distributions():
    metadata = distribution.metadata
    files = []
    for file in sorted(distribution.files or (), key=str):
        path = pathlib.Path(distribution.locate_file(file))
        if path.is_file() and path.suffix != '.pyc':
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            files.append({'path': str(file), 'size_bytes': path.stat().st_size,
                          'sha256': digest})
    packages.append({'name': metadata['Name'], 'version': distribution.version,
                     'requires_dist': sorted(distribution.requires or ()),
                     'license_expression': metadata.get('License-Expression'),
                     'license': metadata.get('License'), 'files': files})
print(json.dumps({'python': sys.version, 'prefix': sys.prefix,
                  'phydrax_file': str(pathlib.Path(phydrax.__file__).resolve()),
                  'packages': sorted(packages, key=lambda item: item['name'].lower())},
                 sort_keys=True, allow_nan=False))
"""


def _artifact(path: Path, /) -> dict[str, object]:
    before = path.stat()
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise RuntimeError(f"Artifact changed while hashing: {path}")
    return {"filename": path.name, "size_bytes": after.st_size, "sha256": digest}


def _atomic_create(path: Path, record: dict[str, object], /) -> None:
    """Publish complete immutable bytes without overwriting a concurrent writer."""
    payload = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def _freeze_inputs(root: Path, /) -> dict[str, object]:
    # Docs/examples/harnesses are finalized before qualification, even though runtime
    # execution identity deliberately has its own narrower, stable source contract.
    paths = set()
    for name in ("phydrax", "tools", "benchmarks", "docs", "examples"):
        directory = root / name
        paths.update(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and not {"__pycache__", ".cache", ".ipynb_checkpoints"}.intersection(
                path.parts
            )
            and path.name != ".DS_Store"
            and (
                name == "docs"
                or path.suffix in (".py", ".md", ".yml", ".yaml", ".toml", ".npz")
            )
        )
    paths.update(
        root / name
        for name in ("pyproject.toml", "uv.lock", "mkdocs.yml", "LICENSE", "NOTICE")
    )
    paths.update(path for path in (root / "LICENSES").rglob("*") if path.is_file())
    records = []
    for path in sorted(paths):
        record = _artifact(path)
        record["path"] = path.relative_to(root).as_posix()
        del record["filename"]
        records.append(record)
    content = {"kind": "battery-distribution-freeze-inputs", "files": records}
    return {**content, "freeze_id": canonical_fingerprint(content)}


def _run(command: list[str], cwd: Path, /) -> str:
    result = subprocess.run(
        command, cwd=cwd, check=True, text=True, stdout=subprocess.PIPE
    )
    return result.stdout


def _unique_json_object(pairs):
    record = {}
    for key, value in pairs:
        if key in record:
            raise ValueError(f"Duplicate distribution record key: {key}")
        record[key] = value
    return record


def verify_installed_distribution(
    manifest_path: Path, source_root: Path, /
) -> dict[str, object]:
    """Verify executed package bytes against retained artifacts, without granting release."""
    import phydrax

    manifest_path = manifest_path.resolve(strict=True)
    source_root = source_root.resolve(strict=True)
    manifest = parse_distribution_manifest(manifest_path.read_text(encoding="utf-8"))
    artifacts = {}
    for name in ("wheel", "sdist", "sbom"):
        record = manifest[name]
        filename = record["filename"]
        path = manifest_path.parent / "artifacts" / filename
        if _artifact(path) != record:
            raise ValueError(f"Distribution artifact bytes differ: {name}")
        artifacts[name] = path
    sbom = json.loads(
        artifacts["sbom"].read_text(encoding="utf-8"),
        object_pairs_hook=_unique_json_object,
    )
    if not isinstance(sbom, dict) or sbom.get("sbom_id") != manifest["sbom_id"]:
        raise ValueError("Distribution SBOM identity is missing or mismatched.")
    if (
        canonical_fingerprint(
            {key: value for key, value in sbom.items() if key != "sbom_id"}
        )
        != manifest["sbom_id"]
    ):
        raise ValueError("Distribution SBOM content identity is invalid.")
    if execution_source_build_id(source_root) != manifest["source_build_id"]:
        raise ValueError("Retained source closure differs from distribution.")
    if _freeze_inputs(source_root)["freeze_id"] != manifest["freeze_id"]:
        raise ValueError("Retained source/docs/harness freeze differs from distribution.")
    package = Path(phydrax.__file__).resolve().parent
    if package == source_root / "phydrax" or source_root in package.parents:
        raise ValueError("Clean distribution execution imported the source checkout.")
    members = []
    with ZipFile(artifacts["wheel"]) as archive:
        for member in sorted(archive.namelist()):
            if not member.startswith("phydrax/") or member.endswith("/"):
                continue
            relative = Path(member).relative_to("phydrax")
            if ".." in relative.parts:
                raise ValueError("Unsafe package path in wheel.")
            installed = package / relative
            identity = _artifact(installed)
            with archive.open(member) as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if identity["sha256"] != digest:
                raise ValueError(f"Executed package differs from wheel: {relative}")
            members.append(
                {
                    "path": relative.as_posix(),
                    "sha256": digest,
                    "size_bytes": identity["size_bytes"],
                }
            )
    actual = {
        path.relative_to(package).as_posix()
        for path in package.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if not members or actual != {record["path"] for record in members}:
        raise ValueError("Executed package contains missing or unrecorded files.")
    return {
        "distribution_id": manifest["distribution_id"],
        "source_build_id": manifest["source_build_id"],
        "wheel_sha256": manifest["wheel"]["sha256"],
        "installed_package_path": str(package),
        "installed_package_digest": canonical_fingerprint({"files": members}),
    }


def build_distribution(root: Path, output: Path, /, *, uv: str = "uv") -> Path:
    """Build one candidate distribution with a lock-constrained clean installation.

    The installation belongs to this execution and is retained for reproducibility.
    A second actor must build an independent replay execution; this receipt is not it.
    """
    root = root.resolve(strict=True)
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("Distribution output must be outside the source tree.")
    if output.exists():
        raise FileExistsError("Use a fresh output directory for each build execution.")
    before = _freeze_inputs(root)
    source_id = execution_source_build_id(root)
    output.mkdir(parents=True)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    _run([uv, "build", "--no-sources", "--out-dir", str(artifacts)], root)
    wheels = tuple(artifacts.glob("*.whl"))
    sdists = tuple(artifacts.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise RuntimeError("Build must produce exactly one wheel and one source archive.")
    wheel, sdist = wheels[0], sdists[0]
    wheel_identity, sdist_identity = _artifact(wheel), _artifact(sdist)
    with ZipFile(wheel) as archive:
        if not any(name == "phydrax/__init__.py" for name in archive.namelist()):
            raise RuntimeError("Built wheel does not contain the phydrax package.")
    requirements = _run(
        [
            uv,
            "export",
            "--frozen",
            "--no-dev",
            "--no-editable",
            "--no-emit-project",
            "--extra",
            "commercial",
        ],
        root,
    )
    requirements_path = output / "locked-requirements.txt"
    requirements_path.write_text(requirements, encoding="utf-8")
    environment = output / "environment"
    _run([uv, "venv", "--python", sys.executable, str(environment)], output)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            "--require-hashes",
            "-r",
            str(requirements_path),
        ],
        output,
    )
    _run(
        [uv, "pip", "install", "--python", str(python), "--no-deps", str(wheel)],
        output,
    )
    _run([uv, "pip", "check", "--python", str(python)], output)
    inventory = json.loads(_run([str(python), "-I", "-c", _INVENTORY_SCRIPT], output))
    installed_package = Path(inventory["phydrax_file"])
    if environment.resolve() not in installed_package.parents:
        raise RuntimeError("Clean installation imported phydrax outside its environment.")
    with ZipFile(wheel) as archive:
        for member in archive.namelist():
            if member.startswith("phydrax/") and not member.endswith("/"):
                installed_file = installed_package.parent.parent / member
                if not installed_file.is_file():
                    raise RuntimeError(
                        f"Wheel member missing from installation: {member}"
                    )
                if (
                    hashlib.sha256(archive.read(member)).hexdigest()
                    != _artifact(installed_file)["sha256"]
                ):
                    raise RuntimeError(f"Installed package differs from wheel: {member}")
    snapshot, harness = output / "frozen-source", output / "harness"
    for record in before["files"]:
        relative = Path(record["path"])
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / relative, destination)
        if (
            relative.parts[0] in ("tools", "benchmarks", "examples")
            and relative.suffix == ".py"
        ):
            runnable = harness / relative
            runnable.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(destination, runnable)
    if (
        _freeze_inputs(snapshot) != before
        or execution_source_build_id(snapshot) != source_id
    ):
        raise RuntimeError(
            "Retained source snapshot differs from the frozen build inputs."
        )
    after = _freeze_inputs(root)
    if before != after or source_id != execution_source_build_id(root):
        raise RuntimeError(
            "Source/facade/docs/harness changed during distribution build."
        )
    if wheel_identity != _artifact(wheel) or sdist_identity != _artifact(sdist):
        raise RuntimeError("Built distribution changed during clean installation.")
    sbom_content = {
        "kind": "battery-installed-software-bill-of-materials",
        "wheel": wheel_identity,
        "lockfile": _artifact(root / "uv.lock"),
        "locked_requirements": _artifact(requirements_path),
        "python": inventory["python"],
        "packages": inventory["packages"],
        "rights_status": "inventory-only-not-rights-clearance",
    }
    sbom = {**sbom_content, "sbom_id": canonical_fingerprint(sbom_content)}
    sbom_path = artifacts / "sbom.json"
    _atomic_create(sbom_path, sbom)
    content = {
        "kind": "battery-unpromoted-distribution",
        "source_build_id": source_id,
        "freeze_id": before["freeze_id"],
        "wheel": wheel_identity,
        "sdist": sdist_identity,
        "sbom": _artifact(sbom_path),
        "sbom_id": sbom["sbom_id"],
    }
    manifest = {**content, "distribution_id": canonical_fingerprint(content)}
    _atomic_create(output / "freeze-inputs.json", before)
    _atomic_create(
        output / "clean-install-receipt.json",
        {
            "kind": "battery-clean-install-receipt",
            "distribution_id": manifest["distribution_id"],
            "python_executable": str(python),
            "environment": str(environment),
            "phydrax_file": str(installed_package),
            "frozen_source_root": str(snapshot),
            "harness_root": str(harness),
            "campaign_command": [
                str(python),
                "-I",
                "-c",
                "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                "runpy.run_module('tools.battery_qualification',run_name='__main__')",
                str(harness),
                "--source-root",
                str(snapshot),
                "--distribution-manifest",
                str(output / "distribution.json"),
            ],
            "example_command_prefix": [
                str(python),
                "-I",
                "-c",
                "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); "
                "runpy.run_module(sys.argv.pop(1),run_name='__main__')",
                str(harness),
            ],
            "independent_replay": False,
            "released": False,
        },
    )
    manifest_path = output / "distribution.json"
    _atomic_create(manifest_path, manifest)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--uv", default="uv")
    arguments = parser.parse_args()
    print(build_distribution(arguments.source_root, arguments.output, uv=arguments.uv))


if __name__ == "__main__":
    main()
