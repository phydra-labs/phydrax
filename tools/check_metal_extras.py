#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Check built Metal-extra wheel metadata and supported/unsupported resolutions.

The check loads no GPU. It inspects the built Phydrax and ``phydrax-metal``
wheels, then resolves them with ``uv pip compile`` for the supported
Darwin-arm64 host, unsupported hosts, a too-old macOS deployment target, the
base CPU installation, an absent companion wheel and a companion whose version
does not match the core release::

    uv run python -m tools.check_metal_extras \\
        --phydrax-wheel dist/phydrax-0.2.2-py3-none-any.whl \\
        --companion-wheel wheelhouse/phydrax_metal-0.2.2-py3-none-macosx_14_0_arm64.whl \\
        --work-dir /tmp/phydrax-metal-extras --output /tmp/phydrax-metal-extras.json

It exits nonzero unless every metadata obligation and resolution case holds.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, UTC
from email.message import Message
from pathlib import Path
from typing import Literal, TypedDict

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name, InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

from tools.check_cuda_extras import resolved_distributions, wheel_metadata


COMPANION = "phydrax-metal"
COMPANION_PACKAGE = "phydrax_metal"
RUNTIME_PLATFORM = "mps"
PLUGIN_ENTRY_POINT = ("mps", "phydrax_metal.plugin")
SUPPORTED_PLATFORM = "aarch64-apple-darwin"
# The Linux glibc tag is substituted from --linux-tag; x86_64 macOS has no JAX 0.12 jaxlib.
UNSUPPORTED_PLATFORMS = ("x86_64-{linux}", "aarch64-{linux}", "x86_64-pc-windows-msvc")
FOREIGN_PLUGINS = frozenset({"jax-mps", "jax-metal", "jax-metallib", "metaljax"})
BUILD_FIELDS = frozenset(
    {
        "version",
        "platform",
        "jax_version",
        "jaxlib_version",
        "pjrt_c_api",
        "stablehlo_version",
        "minimum_macos",
        "library",
        "assets",
        "sources",
        "toolchain",
        "source_id",
        "build_id",
    }
)
SOURCE_FIELDS = frozenset({"lock_sha256", "entries", "patches"})
EVIDENCE_TEXT_LIMIT = 20_000
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MACOS_TAG = re.compile(r"macosx_(\d+)_(\d+)_arm64")


class CommandEvidence(TypedDict):
    command: list[str]
    environment: dict[str, str]
    returncode: int
    stdout: str
    stderr: str


class CompanionFacts(TypedDict):
    version: str
    jax_version: str
    jaxlib_version: str
    minimum_macos: str
    build_id: str
    source_id: str
    tags: list[str]


class CaseEvidence(TypedDict):
    name: str
    platform: str
    requirement: str
    deployment_target: str | None
    expected: Literal["resolved", "refused"]
    successful: bool
    errors: list[str]
    resolved: dict[str, str]
    command: CommandEvidence


class Evidence(TypedDict):
    timestamp: str
    completed_at: str
    python: str
    uv: str
    wheels: dict[str, str]
    core_version: str | None
    companion: CompanionFacts | None
    metadata_errors: list[str]
    cases: list[CaseEvidence]
    errors: list[str]
    successful: bool


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _bounded(text: str) -> str:
    return text if len(text) <= EVIDENCE_TEXT_LIMIT else text[-EVIDENCE_TEXT_LIMIT:]


def _requirements(metadata: Message, errors: list[str]) -> list[Requirement]:
    parsed: list[Requirement] = []
    for text in metadata.get_all("Requires-Dist", []):
        try:
            parsed.append(Requirement(str(text)))
        except InvalidRequirement as error:
            errors.append(f"Invalid Requires-Dist {text!r}: {error}")
    return parsed


def _version_tuple(value: object) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d+(\.\d+)*", value):
        return None
    return tuple(int(part) for part in value.split("."))


def core_metadata_errors(metadata: Message) -> list[str]:
    """Check the core wheel's lockstep, unmarked Metal extra."""
    errors: list[str] = []
    version = str(metadata["Version"])
    if "metal" not in set(metadata.get_all("Provides-Extra", [])):
        errors.append("Missing metal extra")
    requirements = _requirements(metadata, errors)
    companions = [req for req in requirements if canonicalize_name(req.name) == COMPANION]
    expected = f'{COMPANION}=={version}; extra == "metal"'
    if [str(req) for req in companions] != [expected]:
        errors.append(
            f"Expected one unmarked lockstep requirement {expected!r}: "
            f"{[str(req) for req in companions]}"
        )
    foreign = sorted(
        str(req) for req in requirements if canonicalize_name(req.name) in FOREIGN_PLUGINS
    )
    if foreign:
        errors.append(f"Upstream Apple plugin requirements are published: {foreign}")
    if not [req for req in requirements if req.name == "jax" and req.marker is None]:
        errors.append("Missing unconditional base jax requirement")
    return errors


def build_record_errors(record: object, *, wheel_version: str) -> list[str]:
    """Check one companion build manifest and its content-derived identity."""
    if not isinstance(record, dict):
        return ["Metal build manifest must be a JSON object"]
    errors: list[str] = []
    if set(record) != BUILD_FIELDS:
        errors.append(
            f"Metal build manifest fields differ: missing {sorted(BUILD_FIELDS - set(record))}, "
            f"unexpected {sorted(set(record) - BUILD_FIELDS)}"
        )
        return errors
    for field in (
        "version",
        "platform",
        "jax_version",
        "jaxlib_version",
        "stablehlo_version",
        "minimum_macos",
        "library",
        "source_id",
        "build_id",
    ):
        if not isinstance(record[field], str) or not record[field]:
            errors.append(f"Metal build manifest {field} must be non-empty text")
    if errors:
        return errors
    if record["version"] != wheel_version:
        errors.append(
            f"Manifest version {record['version']} != wheel version {wheel_version}"
        )
    if record["platform"] != RUNTIME_PLATFORM:
        errors.append(
            f"Manifest platform {record['platform']!r} is not {RUNTIME_PLATFORM!r}"
        )
    if record["jax_version"] != record["jaxlib_version"]:
        errors.append("Manifest JAX and JAXlib versions are not one matched pair")
    if _version_tuple(record["minimum_macos"]) is None:
        errors.append(f"Invalid manifest minimum_macos {record['minimum_macos']!r}")
    api = record["pjrt_c_api"]
    if (
        not isinstance(api, dict)
        or set(api) != {"major", "minor"}
        or any(type(api[key]) is not int or api[key] < 0 for key in ("major", "minor"))
    ):
        errors.append(f"Manifest pjrt_c_api must hold integer major/minor: {api!r}")
    assets = record["assets"]
    if not isinstance(assets, dict) or not assets:
        errors.append("Manifest assets must be a non-empty mapping")
    else:
        for relative, digest in assets.items():
            path = Path(str(relative))
            if not isinstance(relative, str) or path.is_absolute() or ".." in path.parts:
                errors.append(
                    f"Manifest asset path is not package-relative: {relative!r}"
                )
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                errors.append(f"Manifest asset {relative!r} lacks a sha256 digest")
        if record["library"] not in assets:
            errors.append("Manifest library is not one of its hashed assets")
    sources = record["sources"]
    if (
        not isinstance(sources, dict)
        or set(sources) != SOURCE_FIELDS
        or not isinstance(sources["lock_sha256"], str)
        or not _SHA256.fullmatch(sources["lock_sha256"])
        or not isinstance(sources["entries"], list)
        or not sources["entries"]
        or not isinstance(sources["patches"], list)
    ):
        errors.append("Manifest sources must hold lock_sha256, entries and patches")
    if not isinstance(record["toolchain"], dict) or not record["toolchain"]:
        errors.append("Manifest toolchain must be a non-empty mapping")
    content = {key: value for key, value in record.items() if key != "build_id"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != record["build_id"]:
        errors.append("Manifest build_id is not the SHA-256 of its canonical content")
    return errors


def record_digest(payload: bytes) -> str:
    """Return the wheel RECORD form of a payload's SHA-256 digest."""
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    return f"sha256={encoded.decode('ascii')}"


def _record_errors(archive: zipfile.ZipFile, record_name: str) -> list[str]:
    errors: list[str] = []
    listed: set[str] = set()
    rows = csv.reader(io.StringIO(archive.read(record_name).decode("utf-8")))
    for row in rows:
        if not row:
            continue
        if len(row) != 3:
            errors.append(f"Malformed RECORD row {row!r}")
            continue
        name, digest, size = row
        listed.add(name)
        if name == record_name:
            continue
        try:
            payload = archive.read(name)
        except KeyError:
            errors.append(f"RECORD lists absent member {name}")
            continue
        if digest != record_digest(payload) or size != str(len(payload)):
            errors.append(f"RECORD hash or size mismatch for {name}")
    unlisted = sorted(
        name
        for name in archive.namelist()
        if not name.endswith("/") and name not in listed
    )
    if unlisted:
        errors.append(f"Wheel members missing from RECORD: {unlisted}")
    return errors


def _entry_points(text: str) -> dict[str, list[tuple[str, str]]]:
    groups: dict[str, list[tuple[str, str]]] = {}
    group: str | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(("#", ";")):
            continue
        if line.startswith("[") and line.endswith("]"):
            group = line[1:-1].strip()
            groups.setdefault(group, [])
        elif group is not None and "=" in line:
            name, value = (part.strip() for part in line.split("=", 1))
            groups[group].append((name, value))
    return groups


def companion_errors(
    wheel: Path, *, core_version: str, core_metadata: Message
) -> tuple[list[str], CompanionFacts | None]:
    """Check the companion wheel's identity, pins, tags, entry point and manifest."""
    errors: list[str] = []
    try:
        name, version, _, tags = parse_wheel_filename(wheel.name)
    except (InvalidWheelFilename, InvalidVersion) as error:
        return [f"Invalid companion wheel filename {wheel.name}: {error}"], None
    if name != COMPANION:
        errors.append(f"Companion wheel distribution is {name}, not {COMPANION}")
    platforms = sorted({tag.platform for tag in tags})
    macos = [_MACOS_TAG.fullmatch(platform) for platform in platforms]
    if len(platforms) != 1 or macos[0] is None:
        errors.append(f"Companion must have one macOS arm64 platform tag: {platforms}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        dist_info = f"{COMPANION_PACKAGE}-{version}.dist-info"
        metadata = wheel_metadata(wheel)
        if isinstance(metadata, str):
            return [*errors, metadata], None
        if (
            str(metadata["Version"]) != str(version)
            or str(metadata["Version"]) != core_version
        ):
            errors.append(
                f"Companion version {metadata['Version']} does not match core {core_version}"
            )
        if canonicalize_name(str(metadata["Name"])) != COMPANION:
            errors.append(f"Companion METADATA names {metadata['Name']}")
        wheel_file = f"{dist_info}/WHEEL"
        if wheel_file not in names:
            errors.append(f"Missing {wheel_file}")
        else:
            info = archive.read(wheel_file).decode("utf-8")
            if not re.search(r"^Root-Is-Purelib:\s*false\s*$", info, flags=re.MULTILINE):
                errors.append("Companion wheel must declare Root-Is-Purelib: false")
        record_name = f"{dist_info}/RECORD"
        if record_name not in names:
            errors.append(f"Missing {record_name}")
        else:
            errors.extend(_record_errors(archive, record_name))
        entry_name = f"{dist_info}/entry_points.txt"
        groups = (
            _entry_points(archive.read(entry_name).decode("utf-8"))
            if entry_name in names
            else {}
        )
        if groups.get("jax_plugins") != [PLUGIN_ENTRY_POINT]:
            errors.append(
                f"Expected exactly one jax_plugins entry {PLUGIN_ENTRY_POINT}: "
                f"{groups.get('jax_plugins')}"
            )
        manifest = f"{COMPANION_PACKAGE}/build.json"
        if manifest not in names:
            return [*errors, f"Missing {manifest}"], None
        try:
            record = json.loads(archive.read(manifest))
        except json.JSONDecodeError as error:
            return [*errors, f"Invalid {manifest}: {error}"], None
        manifest_errors = build_record_errors(record, wheel_version=str(version))
        errors.extend(manifest_errors)
        if manifest_errors:
            return errors, None
        for relative, digest in record["assets"].items():
            member = f"{COMPANION_PACKAGE}/{relative}"
            if member not in names:
                errors.append(f"Manifest asset absent from wheel: {member}")
            elif hashlib.sha256(archive.read(member)).hexdigest() != digest:
                errors.append(f"Manifest asset hash differs from wheel bytes: {member}")
    jax_version = record["jax_version"]
    requirements = _requirements(metadata, errors)
    pins = sorted(str(req) for req in requirements)
    expected_pins = [f"jax=={jax_version}", f"jaxlib=={jax_version}"]
    if pins != expected_pins:
        errors.append(f"Companion must pin exactly {expected_pins}: {pins}")
    requires_python = metadata["Requires-Python"]
    if requires_python is None or SpecifierSet(str(requires_python)) != SpecifierSet(
        ">=3.13,<3.14"
    ):
        errors.append(
            f"Companion Requires-Python must be >=3.13,<3.14, found {requires_python!r}"
        )
    for req in _requirements(core_metadata, []):
        if req.name in ("jax", "jaxlib") and req.marker is None:
            if not req.specifier.contains(jax_version, prereleases=True):
                errors.append(
                    f"Companion {req.name}=={jax_version} is outside core requirement {req}"
                )
    if macos and macos[0] is not None and len(platforms) == 1:
        tagged = (int(macos[0][1]), int(macos[0][2]))
        minimum = _version_tuple(record["minimum_macos"]) or ()
        if (minimum + (0, 0))[:2] != tagged:
            errors.append(
                f"Wheel macOS tag {platforms[0]} disagrees with manifest minimum "
                f"{record['minimum_macos']}"
            )
    return errors, {
        "version": str(version),
        "jax_version": jax_version,
        "jaxlib_version": record["jaxlib_version"],
        "minimum_macos": record["minimum_macos"],
        "build_id": record["build_id"],
        "source_id": record["source_id"],
        "tags": sorted(str(tag) for tag in tags),
    }


def variant_companion(wheel: Path, destination: Path, *, version: str) -> Path:
    """Write a consistent copy of the companion wheel declaring another version."""
    _, original, build, tags = parse_wheel_filename(wheel.name)
    if build or len(tags) != 1:
        raise ValueError(
            "The builder produces one single-tag companion wheel without a build tag."
        )
    old_info = f"{COMPANION_PACKAGE}-{original}.dist-info/"
    new_info = f"{COMPANION_PACKAGE}-{version}.dist-info/"
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{COMPANION_PACKAGE}-{version}-{next(iter(tags))}.whl"
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(wheel) as archive:
        for info in archive.infolist():
            if info.is_dir() or info.filename == f"{old_info}RECORD":
                continue
            payload = archive.read(info)
            member = info.filename
            if member.startswith(old_info):
                member = new_info + member.removeprefix(old_info)
            if member == f"{new_info}METADATA":
                text = payload.decode("utf-8")
                text, count = re.subn(
                    r"^Version: .*$",
                    f"Version: {version}",
                    text,
                    count=1,
                    flags=re.MULTILINE,
                )
                if count != 1:
                    raise ValueError("Companion METADATA has no Version field.")
                payload = text.encode("utf-8")
            members[member] = payload
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    for member, payload in members.items():
        writer.writerow((member, record_digest(payload), len(payload)))
    writer.writerow((f"{new_info}RECORD", "", ""))
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for member, payload in members.items():
            archive.writestr(member, payload)
        archive.writestr(f"{new_info}RECORD", rows.getvalue())
    return target


def resolution_errors(
    case: str, resolved: dict[str, str], *, version: str, jax_version: str
) -> list[str]:
    """Check one successful resolution's selected distributions."""
    errors: list[str] = []
    foreign = sorted(
        name
        for name in resolved
        if name in FOREIGN_PLUGINS or re.fullmatch(r"jax-cuda\d+-(plugin|pjrt)", name)
    )
    if foreign:
        errors.append(f"{case} resolved foreign PJRT plugins: {foreign}")
    if case == "metal-supported":
        expected = {COMPANION: version, "jax": jax_version, "jaxlib": jax_version}
        observed = {name: resolved.get(name) for name in expected}
        if observed != expected:
            errors.append(f"{case} expected {expected}, resolved {observed}")
    elif COMPANION in resolved:
        errors.append(
            f"{case} resolved {COMPANION}=={resolved[COMPANION]} without the extra"
        )
    return errors


def compile_case(
    name: str,
    *,
    requirement: str,
    platform: str,
    find_links: Path,
    deployment_target: str | None,
    expected: Literal["resolved", "refused"],
    version: str,
    jax_version: str,
    work_dir: Path,
) -> CaseEvidence:
    command = [
        "uv",
        "pip",
        "compile",
        "--no-config",
        "--python-platform",
        platform,
        "--python-version",
        "3.13",
        "--find-links",
        str(find_links),
        "--no-header",
        "--no-annotate",
        "-",
    ]
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("MACOSX_DEPLOYMENT_TARGET", "PYTHONPATH", "UV_FIND_LINKS")
    }
    recorded: dict[str, str] = {}
    if deployment_target is not None:
        environment["MACOSX_DEPLOYMENT_TARGET"] = deployment_target
        recorded["MACOSX_DEPLOYMENT_TARGET"] = deployment_target
    result = subprocess.run(
        command,
        cwd=work_dir,
        env=environment,
        input=requirement + "\n",
        capture_output=True,
        text=True,
        check=False,
    )
    resolved = resolved_distributions(result.stdout)
    errors: list[str] = []
    if expected == "resolved":
        if result.returncode:
            errors.append(f"{name} resolution failed")
        else:
            errors.extend(
                resolution_errors(
                    name, resolved, version=version, jax_version=jax_version
                )
            )
    elif result.returncode == 0:
        errors.append(f"{name} resolved although it must be refused: {resolved}")
    elif COMPANION not in result.stderr.lower():
        errors.append(f"{name} was refused for a reason not naming {COMPANION}")
    return {
        "name": name,
        "platform": platform,
        "requirement": requirement,
        "deployment_target": deployment_target,
        "expected": expected,
        "successful": not errors,
        "errors": errors,
        "resolved": resolved,
        "command": {
            "command": command,
            "environment": recorded,
            "returncode": result.returncode,
            "stdout": _bounded(result.stdout),
            "stderr": _bounded(result.stderr),
        },
    }


def resolution_matrix(
    *,
    core_wheel: Path,
    companion: CompanionFacts,
    links: dict[str, Path],
    linux_tag: str,
) -> list[tuple[str, str, str, Path, str | None, Literal["resolved", "refused"]]]:
    """Name every (case, requirement, platform, links, macOS target, expectation)."""
    minimum = _version_tuple(companion["minimum_macos"]) or (0,)
    target = ".".join(str(part) for part in (minimum + (0,))[:2])
    older = f"{minimum[0] - 1}.0"
    metal = f"phydrax[metal] @ {core_wheel.as_uri()}"
    base = f"phydrax @ {core_wheel.as_uri()}"
    unsupported = tuple(
        platform.format(linux=linux_tag) for platform in UNSUPPORTED_PLATFORMS
    )
    return [
        (
            "metal-supported",
            metal,
            SUPPORTED_PLATFORM,
            links["companion"],
            target,
            "resolved",
        ),
        *(
            (f"base-{platform}", base, platform, links["companion"], None, "resolved")
            for platform in unsupported
        ),
        (
            f"base-{SUPPORTED_PLATFORM}",
            base,
            SUPPORTED_PLATFORM,
            links["companion"],
            target,
            "resolved",
        ),
        *(
            (
                f"metal-unsupported-{platform}",
                metal,
                platform,
                links["companion"],
                None,
                "refused",
            )
            for platform in unsupported
        ),
        (
            "metal-below-minimum-macos",
            metal,
            SUPPORTED_PLATFORM,
            links["companion"],
            older,
            "refused",
        ),
        (
            "metal-absent-companion",
            metal,
            SUPPORTED_PLATFORM,
            links["absent"],
            target,
            "refused",
        ),
        (
            "metal-companion-version-mismatch",
            metal,
            SUPPORTED_PLATFORM,
            links["mismatch"],
            target,
            "refused",
        ),
    ]


def _write(path: Path, evidence: Evidence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phydrax-wheel", type=Path, required=True)
    parser.add_argument("--companion-wheel", type=Path, required=True)
    parser.add_argument(
        "--work-dir", type=Path, required=True, help="Fresh scratch directory"
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON evidence output")
    parser.add_argument(
        "--linux-tag",
        default="manylinux_2_28",
        help="Linux platform tag of the unsupported-host resolutions",
    )
    args = parser.parse_args()
    core_wheel: Path = args.phydrax_wheel.resolve()
    companion_wheel: Path = args.companion_wheel.resolve()
    work_dir: Path = args.work_dir.resolve()
    output: Path = args.output.resolve()
    uv = subprocess.run(["uv", "--version"], capture_output=True, text=True, check=False)
    evidence: Evidence = {
        "timestamp": datetime.now(UTC).isoformat(),
        "completed_at": "",
        "python": sys.version,
        "uv": uv.stdout.strip(),
        "wheels": {},
        "core_version": None,
        "companion": None,
        "metadata_errors": [],
        "cases": [],
        "errors": [],
        "successful": False,
    }
    for wheel in (core_wheel, companion_wheel):
        if not wheel.is_file():
            evidence["errors"].append(f"Wheel not found: {wheel}")
    if work_dir.exists() and any(work_dir.iterdir()):
        evidence["errors"].append(f"Work directory is not empty: {work_dir}")
    core_metadata = None
    if not evidence["errors"]:
        evidence["wheels"] = {
            path.name: sha256(path) for path in (core_wheel, companion_wheel)
        }
        loaded = wheel_metadata(core_wheel)
        if isinstance(loaded, str):
            evidence["errors"].append(loaded)
        else:
            core_metadata = loaded
    if core_metadata is not None:
        version = str(core_metadata["Version"])
        evidence["core_version"] = version
        evidence["metadata_errors"].extend(core_metadata_errors(core_metadata))
        found, companion = companion_errors(
            companion_wheel, core_version=version, core_metadata=core_metadata
        )
        evidence["metadata_errors"].extend(found)
        evidence["companion"] = companion
        if companion is not None:
            links = {key: work_dir / key for key in ("companion", "absent", "mismatch")}
            for directory in links.values():
                directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(companion_wheel, links["companion"] / companion_wheel.name)
            release = Version(version).release
            mismatch = ".".join(str(part) for part in (*release[:-1], release[-1] + 1))
            variant_companion(companion_wheel, links["mismatch"], version=mismatch)
            for (
                name,
                requirement,
                platform,
                find_links,
                target,
                expected,
            ) in resolution_matrix(
                core_wheel=core_wheel,
                companion=companion,
                links=links,
                linux_tag=args.linux_tag,
            ):
                case = compile_case(
                    name,
                    requirement=requirement,
                    platform=platform,
                    find_links=find_links,
                    deployment_target=target,
                    expected=expected,
                    version=version,
                    jax_version=companion["jax_version"],
                    work_dir=work_dir,
                )
                evidence["cases"].append(case)
                print(f"{name}: {'PASS' if case['successful'] else 'FAIL'}", flush=True)
    for path in (core_wheel, companion_wheel):
        if (
            path.name in evidence["wheels"]
            and sha256(path) != evidence["wheels"][path.name]
        ):
            evidence["errors"].append(f"Wheel changed during the check: {path}")
    evidence["successful"] = bool(evidence["cases"]) and not (
        evidence["errors"]
        or evidence["metadata_errors"]
        or any(not case["successful"] for case in evidence["cases"])
    )
    evidence["completed_at"] = datetime.now(UTC).isoformat()
    _write(output, evidence)
    for case in evidence["cases"]:
        for error in case["errors"]:
            print(error, file=sys.stderr)
    for error in [*evidence["errors"], *evidence["metadata_errors"]]:
        print(error, file=sys.stderr)
    print(f"Evidence: {output}")
    return 0 if evidence["successful"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
