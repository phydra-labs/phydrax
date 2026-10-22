# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Installed-wheel mathematical and physical coverage for optional CUDA providers."""

from __future__ import annotations

import json
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

_BOOTSTRAP = """
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
_PROFILE_RUN = (
    _BOOTSTRAP
    + """
expected = json.loads(sys.argv[3])
if [row['gpu_uuid'] for row in observed] != expected:
    raise RuntimeError('Optional provider execution has incorrect physical GPU ownership')
import os, pytest
status = pytest.main(sys.argv[4:])
from phydrax._cuda_environment import competing_cuda_libraries
from phydrax.backends import cuda_environment
report = cuda_environment()
owned = {path for path, major in report.facts.wheel_library_majors if major == 13}
foreign = [item.path for item in report.facts.mapped_libraries if item.major != 13 or item.path not in owned]
hostile = os.environ.get('PHYDRAX_CUDA_PROFILE_HOSTILE_PATH')
challenge = competing_cuda_libraries(hostile or '', report.facts.mapped_libraries)
passed = status == 0 and report.successful and not foreign and (hostile is None or bool(challenge))
print('CUDA_PROFILE_LIBRARIES ' + json.dumps({
    'passed': passed, 'foreign': foreign, 'hostile_candidates': challenge,
    'mapped': [(item.path, item.major) for item in report.facts.mapped_libraries],
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
    evidence: dict[str, object] = {
        "devices": list(devices),
        "command": command,
        "report": str(report),
        "coverage": asdict(coverage),
        "errors": errors,
        "passed": not errors,
    }
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
    discovery = run(
        [str(python), "-c", _BOOTSTRAP, str(root), str(python.parent.parent)],
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
                    environment
                    | {
                        "LD_LIBRARY_PATH": hostile_library_path,
                        "PHYDRAX_CUDA_PROFILE_HOSTILE_PATH": hostile_library_path,
                    },
                    timeout,
                    library,
                    hostile_profile,
                    (device,),
                    f"device-{index}-hostile",
                    family_selector=family
                    + ("" if library == "cuest" else " and float64"),
                )
                hostile_results.append(hostile)
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
