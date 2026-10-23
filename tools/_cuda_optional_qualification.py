# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Installed-wheel mathematical and physical coverage for optional CUDA providers."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import UUID
from xml.etree import ElementTree

from tools._accelerator_process import CommandEvidence, run


@dataclass(frozen=True)
class LibraryProfile:
    files: tuple[str, ...]
    required_families: tuple[str, ...]
    precisions: tuple[str, ...] = ()
    extra_requirements: tuple[str, ...] = ()


_TENSOR_FAMILIES = (
    "test_cuda_tensor_numpy_primal_jvp_and_bilinear_vjp",
    "test_cuda_tensor_current_data_transforms_share_one_compiled_executable",
    "test_cuda_tensor_repeated_operand_chain_rule_and_second_order",
    "test_cuda_tensor_refresh_concurrency_and_policy_cache_isolation",
    "test_cuda_tensor_registry_release_rebuild_and_resource_admission",
    "test_cuda_tensor_unary_and_imported_nary_path_contract",
)
_TENSOR_PRECISIONS = ("float32", "float64", "complex64", "complex128")
LIBRARY_PROFILES = {
    "cutensor": LibraryProfile(
        ("tests/integration/test_cuda_tensor_contraction.py",),
        _TENSOR_FAMILIES,
        _TENSOR_PRECISIONS,
    ),
    "cutensornet": LibraryProfile(
        ("tests/integration/test_cuda_tensor_contraction.py",),
        _TENSOR_FAMILIES,
        _TENSOR_PRECISIONS,
    ),
    "cuest": LibraryProfile(
        (
            "tests/integration/chemistry/test_cuest_integrals.py",
            "tests/integration/chemistry/test_cuest_mean_field.py",
            "tests/integration/chemistry/test_cuest_lifecycle.py",
        ),
        (
            "test_s_t_v_match_independent_spherical_ao_reference",
            "test_jk_uses_actual_changed_damped_rank_deficient_and_zero_density",
            "test_native_scf_matches_independent_df_energy_density_and_stationarity",
            "test_analytic_forces_match_same_definition_and_external_df_energy_differences",
            "test_resource_shortage_is_native_failure_and_does_not_poison_normal_plan",
            "test_typed_force_result_archive_pes_and_born_oppenheimer_step",
        ),
        extra_requirements=("pyscf>=2.14.0,<3",),
    ),
}
_SLICE_PROFILE = LibraryProfile(
    ("tests/integration/test_cuda_tensor_slices.py",),
    (
        "test_cuda_slice_residual_identity_pure_local_jit_jvp_vjp_and_mixed_second",
        "test_cuda_slice_host_checkpoint_order_and_numeric_version",
        "test_cuda_slice_host_checkpoint_and_distributed_orchestration_refuse_transformations",
        "test_cuda_slice_placement_device_subsets_uneven_counts_and_inactive_padding",
        "test_cuda_slice_nonfinite_failure_returns_no_partial_aggregate",
    ),
    _TENSOR_PRECISIONS,
)


def unavailable_libraries(libraries: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(
        library
        for library in libraries
        if sys.platform == "win32" and library in ("cutensornet", "cuest")
    )


def hostile_environment(
    environment: dict[str, str], library_path: str | None
) -> dict[str, str]:
    result = dict(environment)
    result.pop("LD_LIBRARY_PATH", None)
    result.pop("PHYDRAX_CUDA_PROFILE_HOSTILE_PATH", None)
    if library_path:
        result["PHYDRAX_CUDA_PROFILE_HOSTILE_PATH"] = library_path
        if sys.platform == "win32":
            result["PATH"] = library_path + os.pathsep + result.get("PATH", "")
        else:
            result["LD_LIBRARY_PATH"] = library_path
    return result


HOSTILE_BOOTSTRAP = """
import os, sys
_hostile_handles = []
if sys.platform == "win32":
    for directory in os.environ.get("PHYDRAX_CUDA_PROFILE_HOSTILE_PATH", "").split(os.pathsep):
        if directory:
            _hostile_handles.append(os.add_dll_directory(directory))
"""


WINDOWS_LOADER_PROVENANCE = """
def windows_loader_provenance():
    import hashlib, importlib.metadata, json, re
    from pathlib import Path
    distribution = importlib.metadata.distribution("phydrax-cuda13")
    manifests = [
        Path(distribution.locate_file(member)).resolve()
        for member in distribution.files or ()
        if Path(str(member)).name == "phx_cuda_build.json"
    ]
    if len(manifests) != 1:
        raise RuntimeError("Expected one owned CUDA companion build manifest")
    path = manifests[0]
    payload = path.read_bytes()
    manifest = json.loads(payload)
    linkage = {"kind": "driver-hybrid", "target_dll": "nvcudart_hybrid64.dll", "owner": "display-driver"}
    if manifest.get("runtime_linkage") != linkage:
        raise RuntimeError("Windows CUDA requires canonical driver-hybrid runtime linkage")
    toolchain = manifest.get("toolchain", {})
    libraries = {}
    required = {
        "cuda_link_library": ("driver-loader", "nvcuda.dll", {"cudaDriverLoader_initializeLoader", "cuGetProcAddress_v2"}),
        "cudart_link_library": ("driver-hybrid", "nvcudart_hybrid64.dll", {"initializeHybridRuntimeLoaderOnce", "cuGetProcAddress_v2", "cuGetExportTable"}),
    }
    for name, (kind, target, symbols) in required.items():
        record = toolchain.get(name, {})
        members = record.get("archive_members")
        dispatch = record.get("dispatch_symbols")
        if (
            record.get("kind") != kind or record.get("target_dll") != target
            or not isinstance(record.get("path"), str) or not record["path"]
            or re.fullmatch(r"[0-9a-f]{64}", str(record.get("sha256"))) is None
            or not isinstance(members, list) or not members
            or not all(isinstance(member, str) and member for member in members)
            or not isinstance(dispatch, list) or not all(isinstance(symbol, str) for symbol in dispatch)
            or not symbols.issubset(dispatch)
        ):
            raise RuntimeError(f"Missing canonical vendor loader archive provenance: {name}")
        libraries[name] = record
    if not manifest.get("toolkit_wheels"):
        raise RuntimeError("Missing CUDA toolkit wheel provenance")
    link_maps = manifest.get("link_maps")
    if not isinstance(link_maps, dict) or not link_maps or not all(
        isinstance(name, str) and isinstance(relative, str) and relative
        for name, relative in link_maps.items()
    ):
        raise RuntimeError("Missing installed per-component linker maps")
    return {
        "manifest_path": str(path), "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "runtime_linkage": linkage, "loader_archives": libraries,
        "toolkit_wheels": manifest["toolkit_wheels"],
        "link_maps": link_maps,
    }

def windows_component_dispatch(loader_provenance, components=None):
    import base64, importlib.metadata, os
    from pathlib import Path
    from phydrax._pe_dependencies import inspect_cuda_runtime_linkage
    if components is None:
        from phydrax._cuda_native import cuda_native_resources
        components = cuda_native_resources().components
    if not components:
        raise RuntimeError("No loaded native components for binary dispatch inspection")
    distribution = importlib.metadata.distribution("phydrax-cuda13")
    root = Path(distribution.locate_file("")).resolve()
    records = {}
    for member in distribution.files or ():
        path = Path(distribution.locate_file(member)).resolve()
        key = os.path.normcase(str(path))
        if key in records:
            raise RuntimeError(f"Duplicate companion RECORD path: {path}")
        records[key] = member
    def require_record(path, digest):
        member = records.get(os.path.normcase(str(path)))
        if (
            not path.is_relative_to(root) or member is None or member.hash is None
            or member.hash.mode != "sha256" or member.size != path.stat().st_size
            or member.hash.value != base64.urlsafe_b64encode(bytes.fromhex(digest)).rstrip(b"=").decode("ascii")
        ):
            raise RuntimeError(f"Binary dispatch evidence lacks installed RECORD ownership: {path}")
    manifest_path = Path(loader_provenance["manifest_path"]).resolve()
    require_record(manifest_path, loader_provenance["manifest_sha256"])
    result = {}
    for component in components:
        if component.name in result:
            raise RuntimeError(f"Duplicate loaded component identity: {component.name}")
        relative = loader_provenance["link_maps"].get(component.name)
        if not isinstance(relative, str):
            raise RuntimeError(f"Missing installed linker map for {component.name}")
        map_path = (manifest_path.parent / relative).resolve()
        if not map_path.is_relative_to(manifest_path.parent):
            raise RuntimeError(f"Linker map escapes companion package: {relative}")
        path = Path(component.path).resolve()
        if map_path != path.with_suffix(".map"):
            raise RuntimeError(f"Linker map does not belong to loaded component: {component.name}")
        proof = inspect_cuda_runtime_linkage(path, map_path, loader_provenance["runtime_linkage"])
        require_record(path, proof["sha256"])
        require_record(map_path, proof["link_map_sha256"])
        result[component.name] = proof | {"build_hash": component.build_hash}
    return result
"""

_BOOTSTRAP = (
    HOSTILE_BOOTSTRAP
    + """
import json, sys
from pathlib import Path
import phydrax
if not Path(phydrax.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve()):
    raise RuntimeError('Qualification imported Phydrax outside the installed environment')
sys.path.insert(0, sys.argv[1])
from tests._support.cuda_execution_worker import cuda_runtime, ownership
observed = ownership(cuda_runtime())
print('CUDA_PROFILE_DEVICES ' + json.dumps(observed, sort_keys=True), flush=True)
"""
)
_PROFILE_RUN = (
    _BOOTSTRAP
    + WINDOWS_LOADER_PROVENANCE
    + """
expected = json.loads(sys.argv[3])
if [row['gpu_uuid'] for row in observed] != expected:
    raise RuntimeError('Optional provider execution has incorrect physical GPU ownership')
import os, pytest
status = pytest.main(sys.argv[4:])
from phydrax._cuda_environment import competing_cuda_libraries
from phydrax.backends import cuda_environment
report = cuda_environment()
owned = {os.path.normcase(str(Path(path).resolve())) for path, major in report.facts.wheel_library_majors if major == 13}
foreign = [item.path for item in report.facts.mapped_libraries if item.major != 13 or os.path.normcase(str(Path(item.path).resolve())) not in owned]
runtime_identity = dict(report.facts.runtime_identity) if sys.platform == "win32" else None
loader_provenance = windows_loader_provenance() if sys.platform == "win32" else None
component_dispatch = windows_component_dispatch(loader_provenance) if sys.platform == "win32" else None
runtime_observed = sys.platform != "win32" or bool(runtime_identity and component_dispatch)
hostile = os.environ.get('PHYDRAX_CUDA_PROFILE_HOSTILE_PATH')
challenge = competing_cuda_libraries(hostile or '', report.facts.mapped_libraries)
passed = status == 0 and report.successful and not foreign and runtime_observed and (hostile is None or bool(challenge))
print('CUDA_PROFILE_LIBRARIES ' + json.dumps({
    'passed': passed, 'foreign': foreign, 'hostile_candidates': challenge,
    'mapped': [(item.path, item.major) for item in report.facts.mapped_libraries],
    'runtime_identity': runtime_identity, 'loader_provenance': loader_provenance,
    'component_dispatch': component_dispatch,
    'compiled_cuda_major': report.facts.compiled_cuda_major,
    'driver_compatibility_version': report.facts.driver_compatibility_version,
}, sort_keys=True), flush=True)
raise SystemExit(0 if passed else 1)
"""
)


@dataclass(frozen=True)
class ProfileCoverage:
    cases: tuple[str, ...]
    missing: tuple[tuple[str, str | None], ...]
    unsuccessful: tuple[str, ...]
    duplicates: tuple[str, ...]
    report_error: str | None = None

    @property
    def passed(self) -> bool:
        return bool(self.cases) and not (
            self.missing or self.unsuccessful or self.duplicates or self.report_error
        )

    def errors(self) -> list[str]:
        errors = [self.report_error] if self.report_error else []
        if not self.cases:
            errors.append("Optional provider report contains no numerical cases")
        errors.extend(
            f"Required scenario did not pass: {case}" for case in self.unsuccessful
        )
        errors.extend(f"Duplicate scenario evidence: {case}" for case in self.duplicates)
        errors.extend(
            f"Missing optional-provider scenario: {family}"
            + (f" at {precision}" if precision is not None else "")
            for family, precision in self.missing
        )
        return errors


def collect_profile_coverage(path: Path, profile: LibraryProfile) -> ProfileCoverage:
    """Require actual completed scenarios, not just a successful pytest exit."""
    try:
        report = ElementTree.parse(path)
    except (OSError, ElementTree.ParseError) as error:
        return ProfileCoverage((), (), (), (), str(error))
    identities: set[str] = set()
    names: list[str] = []
    unsuccessful: list[str] = []
    duplicates: list[str] = []
    for case in report.getroot().iter("testcase"):
        name = case.attrib.get("name", "")
        identity = case.attrib.get("classname", "") + "::" + name
        if identity in identities:
            duplicates.append(identity)
        identities.add(identity)
        names.append(name)
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped")):
            unsuccessful.append(identity)
    missing: list[tuple[str, str | None]] = []
    for family in profile.required_families:
        matching = [name for name in names if name.split("[", 1)[0] == family]
        if not matching:
            missing.append((family, None))
        for precision in profile.precisions:
            if not any(
                precision in name.partition("[")[2].removesuffix("]").split("-")
                for name in matching
            ):
                missing.append((family, precision))
    return ProfileCoverage(
        tuple(sorted(identities)),
        tuple(missing),
        tuple(sorted(unsuccessful)),
        tuple(sorted(duplicates)),
    )


def _observed_uuids(command: CommandEvidence) -> tuple[str, ...]:
    lines = [
        line.removeprefix("CUDA_PROFILE_DEVICES ")
        for line in command["stdout"].splitlines()
        if line.startswith("CUDA_PROFILE_DEVICES ")
    ]
    if len(lines) != 1:
        raise ValueError("Expected one physical-device ownership record")
    records: object = json.loads(lines[0])
    if not isinstance(records, list) or not records:
        raise ValueError("Physical-device ownership record is empty or invalid")
    uuids = []
    for record in records:
        if not isinstance(record, dict) or record.get("platform") != "gpu":
            raise ValueError("Optional provider qualification requires actual GPUs")
        uuid = record.get("gpu_uuid")
        if not isinstance(uuid, str) or not uuid.startswith("GPU-"):
            raise ValueError("Physical-device ownership record has an invalid UUID")
        if str(UUID(uuid.removeprefix("GPU-"))) != uuid.removeprefix("GPU-"):
            raise ValueError("Physical-device UUID is not canonical")
        uuids.append(uuid)
    if len(set(uuids)) != len(uuids):
        raise ValueError("Physical-device ownership record has duplicate UUIDs")
    return tuple(uuids)


def validate_windows_runtime_evidence(report: dict[str, object]) -> None:
    """Require observed driver identity, not metadata-only or static-runtime claims."""
    identity = report.get("runtime_identity")
    if not isinstance(identity, dict) or (
        identity.get("linkage") != "driver-hybrid"
        or identity.get("owner") != "display-driver"
    ):
        raise ValueError("Missing canonical Windows driver-hybrid runtime identity")
    for name in ("runtime", "driver", "driver_dispatch"):
        if (
            not isinstance(identity.get(f"{name}_path"), str)
            or not identity[f"{name}_path"]
            or re.fullmatch(r"[0-9a-f]{64}", str(identity.get(f"{name}_sha256"))) is None
        ):
            raise ValueError(f"Incomplete Windows {name} identity")
    compatibility = report.get("driver_compatibility_version")
    if (
        report.get("compiled_cuda_major") != 13
        or not isinstance(compatibility, (tuple, list))
        or len(compatibility) != 2
        or any(type(part) is not int for part in compatibility)
        or tuple(compatibility) < (13, 0)
    ):
        raise ValueError("Missing compiled CUDA 13 / compatible driver evidence")
    provenance = report.get("loader_provenance")
    if not isinstance(provenance, dict) or (
        provenance.get("runtime_linkage")
        != {
            "kind": "driver-hybrid",
            "target_dll": "nvcudart_hybrid64.dll",
            "owner": "display-driver",
        }
        or not provenance.get("loader_archives")
        or not provenance.get("toolkit_wheels")
        or re.fullmatch(r"[0-9a-f]{64}", str(provenance.get("manifest_sha256"))) is None
    ):
        raise ValueError("Missing canonical Windows loader archive provenance")
    dispatch = report.get("component_dispatch")
    if not isinstance(dispatch, dict) or not dispatch:
        raise ValueError("Missing loaded-component binary dispatch evidence")
    for component, proof in dispatch.items():
        if not isinstance(component, str) or not isinstance(proof, dict):
            raise ValueError("Malformed loaded-component binary dispatch evidence")
        if (
            proof.get("machine") != 0x8664
            or proof.get("runtime_linkage") != provenance["runtime_linkage"]
            or proof.get("static_runtime_excluded") is not True
            or proof.get("dispatch_markers")
            != ["nvcudart_hybrid64.dll", "cuGetProcAddress_v2", "cuGetExportTable"]
        ):
            raise ValueError(f"Unproven canonical hybrid dispatch for {component}")
        for field in ("path", "link_map_path", "build_hash"):
            if not isinstance(proof.get(field), str) or not proof[field]:
                raise ValueError(f"Missing {component} binary dispatch {field}")
        for field in ("sha256", "link_map_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(proof.get(field))) is None:
                raise ValueError(f"Missing {component} binary dispatch {field}")
        for field in ("imports", "delay_imports", "closure_imports"):
            names = proof.get(field)
            if not isinstance(names, list) or any(
                not isinstance(name, str) or name.casefold().startswith("cudart")
                for name in names
            ):
                raise ValueError(f"Invalid {component} binary dispatch imports")
    builds = report.get("build_hashes")
    if isinstance(builds, dict) and any(
        name not in dispatch or dispatch[name].get("build_hash") != build
        for name, build in builds.items()
    ):
        raise ValueError(
            "Mandatory loaded components lack matching binary dispatch evidence"
        )


def _run_profile(
    python: Path,
    root: Path,
    environment: dict[str, str],
    timeout: float,
    library: str,
    profile: LibraryProfile,
    devices: tuple[str, ...],
    label: str,
    family_selector: str | None = None,
) -> tuple[dict[str, object], list[str]]:
    report = root / f"{library}-{label}.xml"
    selected_environment = environment | {"CUDA_VISIBLE_DEVICES": ",".join(devices)}
    selector = ""
    if library == "cutensor":
        selector = "cutensor and not cutensornet"
    elif library == "cutensornet":
        selector = "cutensornet"
    if profile is _SLICE_PROFILE and len(devices) == 1:
        # -k also matches function names, including the word "uneven".
        selector += (
            " and (single or not "
            "test_cuda_slice_placement_device_subsets_uneven_counts_and_inactive_padding)"
        )
    if family_selector is not None:
        selector = f"({selector}) and {family_selector}" if selector else family_selector
    command = run(
        [
            str(python),
            "-I",
            "-c",
            _PROFILE_RUN,
            str(root),
            str(python.parent.parent),
            json.dumps(devices),
            "-q",
            "-x",
            "-n",
            "0",
            "--junitxml",
            str(report),
            *profile.files,
            *(["-k", selector] if selector else []),
        ],
        cwd=root,
        environment=selected_environment,
        timeout=timeout,
    )
    coverage = collect_profile_coverage(report, profile)
    errors = coverage.errors()
    if command["returncode"] != 0 or command["timed_out"]:
        errors.append("Optional provider command failed or timed out")
    try:
        observed = _observed_uuids(command)
    except ValueError as error:
        errors.append(str(error))
    else:
        if observed != devices:
            errors.append("Optional provider physical device selection changed")
    libraries: dict[str, object] | None = None
    if sys.platform == "win32":
        try:
            lines = [
                line.removeprefix("CUDA_PROFILE_LIBRARIES ")
                for line in command["stdout"].splitlines()
                if line.startswith("CUDA_PROFILE_LIBRARIES ")
            ]
            parsed = json.loads(lines[0]) if len(lines) == 1 else None
            if not isinstance(parsed, dict) or parsed.get("passed") is not True:
                raise ValueError("Missing successful Windows mapped-library evidence")
            validate_windows_runtime_evidence(parsed)
            if library not in parsed["component_dispatch"]:
                raise ValueError(f"Missing executed {library} binary dispatch evidence")
            libraries = parsed
        except ValueError as error:
            errors.append(str(error))
    evidence: dict[str, object] = {
        "devices": list(devices),
        "command": command,
        "report": str(report),
        "coverage": asdict(coverage),
        "errors": errors,
        "passed": not errors,
    }
    if sys.platform == "win32":
        evidence["libraries"] = libraries
    # XML proves assertions, but not the process's subsequent clean exit.
    report.with_suffix(".json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    return evidence, errors


def qualify_libraries(
    python: Path,
    root: Path,
    libraries: tuple[str, ...],
    required_gpus: int,
    environment: dict[str, str],
    timeout: float,
    hostile_library_path: str,
) -> tuple[dict[str, object], list[str]]:
    """Run every selected local profile on every actual GPU, then host slicing."""
    if required_gpus < 1:
        raise ValueError("Hardware qualification requires a positive GPU count")
    unavailable = unavailable_libraries(libraries)
    if unavailable:
        return {
            "passed": False,
            "hardware_qualified": False,
            "refused": list(unavailable),
        }, [
            f"{library} has no supported Windows vendor artifacts; request refused, not numerical qualification"
            for library in unavailable
        ]
    discovery = run(
        [str(python), "-I", "-c", _BOOTSTRAP, str(root), str(python.parent.parent)],
        cwd=root,
        environment=environment,
        timeout=timeout,
    )
    evidence: dict[str, object] = {"discovery": discovery, "profiles": {}}
    if discovery["returncode"] != 0 or discovery["timed_out"]:
        return evidence, ["Optional provider device discovery failed"]
    try:
        devices = _observed_uuids(discovery)
    except ValueError as error:
        return evidence, [str(error)]
    if len(devices) < required_gpus:
        return evidence, ["Insufficient physical GPUs for selected optional profiles"]
    devices = devices[:required_gpus]
    evidence["devices"] = list(devices)
    profiles: dict[str, object] = {}
    errors = []
    for library in libraries:
        profile = LIBRARY_PROFILES[library]
        results = []
        provider_errors = []
        hostile_results = []
        for index, device in enumerate(devices):
            result, failed = _run_profile(
                python,
                root,
                environment,
                timeout,
                library,
                profile,
                (device,),
                f"device-{index}",
            )
            results.append(result)
            provider_errors.extend(f"{library}/{device}: {message}" for message in failed)
            if not failed:
                family = (
                    "test_resource_shortage_is_native_failure_and_does_not_poison_normal_plan"
                    if library == "cuest"
                    else "test_cuda_tensor_registry_release_rebuild_and_resource_admission"
                )
                hostile_profile = LibraryProfile(
                    profile.files,
                    (family,),
                    () if library == "cuest" else ("float64",),
                )
                hostile, hostile_errors = _run_profile(
                    python,
                    root,
                    hostile_environment(environment, hostile_library_path),
                    timeout,
                    library,
                    hostile_profile,
                    (device,),
                    f"device-{index}-hostile",
                    family_selector=family
                    + ("" if library == "cuest" else " and float64"),
                )
                hostile_results.append(hostile)
                if sys.platform == "win32" and not hostile_errors:
                    clean_libraries = result.get("libraries")
                    hostile_libraries = hostile.get("libraries")
                    if (
                        not isinstance(clean_libraries, dict)
                        or not isinstance(hostile_libraries, dict)
                        or any(
                            clean_libraries.get(key) != hostile_libraries.get(key)
                            for key in (
                                "runtime_identity",
                                "loader_provenance",
                                "component_dispatch",
                            )
                        )
                    ):
                        hostile_errors.append(
                            "Windows hostile audit changed driver cohort, loader provenance or component dispatch"
                        )
                        hostile["passed"] = False
                provider_errors.extend(
                    f"{library}/{device}/hostile: {message}" for message in hostile_errors
                )
        entry: dict[str, object] = {"local": results, "hostile": hostile_results}
        if library in ("cutensor", "cutensornet") and not provider_errors:
            group, failed = _run_profile(
                python,
                root,
                environment,
                timeout,
                library,
                _SLICE_PROFILE,
                devices,
                "native-host-slices",
            )
            entry["native_host_slices"] = group
            provider_errors.extend(
                f"{library}/native-host-slices: {message}" for message in failed
            )
        profiles[library] = entry
        errors.extend(provider_errors)
    evidence["profiles"] = profiles
    evidence["passed"] = not errors
    return evidence, errors
