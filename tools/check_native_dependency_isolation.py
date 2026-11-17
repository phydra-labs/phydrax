from __future__ import annotations

import argparse
import email.parser
import faulthandler
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_ENGINES = ("OCP", "gmsh", "manifold3d", "open3d", "pyvista", "pymeshlab", "vtk")
# Read the lightweight authoritative registry without importing its package,
# whose CLI initialization loads Phydrax from the checkout.
NATIVE_SCENARIOS: tuple[str, ...] = runpy.run_path(
    str(ROOT / "tools/meshing_qualification/_registry.py")
)["NATIVE_SCENARIOS"]
if not NATIVE_SCENARIOS:
    raise RuntimeError("The canonical native qualification registry is empty.")


def _write_json(path: Path, record: dict[str, object]) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _command(
    arguments: list[str],
    /,
    *,
    evidence: Path,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> str:
    number = len(tuple(evidence.glob("command-*.json"))) + 1
    stem = f"command-{number:03d}"
    stdout_path = evidence / f"{stem}.stdout"
    stderr_path = evidence / f"{stem}.stderr"
    metadata_path = evidence / f"{stem}.json"
    environment = dict(os.environ if env is None else env)
    environment["PYTHONFAULTHANDLER"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    recorded_environment = {
        name: environment[name]
        for name in (
            "PATH",
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONNOUSERSITE",
            "PYTHONFAULTHANDLER",
            "PYTHONUNBUFFERED",
            "VIRTUAL_ENV",
            "TMPDIR",
            "JAX_PLATFORM_NAME",
            "JAX_PLATFORMS",
            "JAX_ENABLE_X64",
            "JAX_ENABLE_COMPILATION_CACHE",
            "PHX_MESHCORE_THREADS",
            "JAX_NUM_CPU_DEVICES",
            "XLA_FLAGS",
            "OMP_NUM_THREADS",
            "OMP_THREAD_LIMIT",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "TF_NUM_INTRAOP_THREADS",
            "TF_NUM_INTEROP_THREADS",
        )
        if name in environment
    }
    metadata: dict[str, object] = {
        "arguments": arguments,
        "cwd": str(cwd or Path.cwd()),
        "environment": recorded_environment,
        "stdout": stdout_path.name,
        "stderr": stderr_path.name,
        "returncode": None,
    }
    _write_json(metadata_path, metadata)
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            result = subprocess.run(
                arguments,
                cwd=cwd,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        except OSError as error:
            metadata["error"] = str(error)
            _write_json(metadata_path, metadata)
            raise
        stdout.flush()
        stderr.flush()
        os.fsync(stdout.fileno())
        os.fsync(stderr.fileno())
    metadata["returncode"] = result.returncode
    _write_json(metadata_path, metadata)
    output = stdout_path.read_text(encoding="utf-8")
    errors = stderr_path.read_text(encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {arguments!r}\n" + output + errors
        )
    return output.strip()


def _resolve_python(selector: str, /, *, evidence: Path) -> Path:
    candidate = Path(selector).expanduser()
    if not candidate.is_file():
        executable = shutil.which(f"python{selector}")
        if executable is None and selector == "3.13":
            executable = sys.executable
        if executable is None:
            raise RuntimeError(f"Cannot locate requested CPython {selector!r}.")
        candidate = Path(executable)
    candidate = candidate.resolve()
    version = json.loads(
        _command(
            [
                str(candidate),
                "-c",
                "import json,sys; print(json.dumps(list(sys.version_info[:3])))",
            ],
            evidence=evidence,
        )
    )
    if version[:2] != [3, 13]:
        raise RuntimeError(
            f"The isolation gate requires CPython 3.13, got {version!r} from {candidate}."
        )
    return candidate


def _wheel_version(path: Path, expected_distribution: str, /) -> str:
    with zipfile.ZipFile(path) as archive:
        metadata_names = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_names) != 1:
            raise RuntimeError(f"Wheel must contain one METADATA file: {path}")
        metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]))
    distribution = metadata.get("Name", "").replace("_", "-").lower()
    if distribution != expected_distribution.replace("_", "-").lower():
        raise RuntimeError(
            f"Expected {expected_distribution} wheel, got {distribution!r} at {path}."
        )
    version = metadata.get("Version")
    if not version:
        raise RuntimeError(f"Wheel has no version metadata: {path}")
    return version


def _built_wheel(directory: Path, prefix: str, /) -> Path:
    matches = tuple(directory.glob(f"{prefix}*.whl"))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {prefix} wheel in {directory}, found {len(matches)}."
        )
    return matches[0]


def _build_wheels(
    python: Path, directory: Path, /, *, evidence: Path
) -> tuple[Path, Path]:
    directory.mkdir(parents=True)
    phydrax_output = directory / "phydrax"
    meshcore_output = directory / "meshcore"
    phydrax_output.mkdir()
    meshcore_output.mkdir()
    _command(
        [
            "uv",
            "build",
            "--wheel",
            "--python",
            str(python),
            "--out-dir",
            str(phydrax_output),
            str(ROOT),
        ],
        evidence=evidence,
    )
    _command(
        [
            "uv",
            "build",
            "--wheel",
            "--python",
            str(python),
            "--out-dir",
            str(meshcore_output),
            str(ROOT / "native/meshcore"),
        ],
        evidence=evidence,
    )
    return _built_wheel(phydrax_output, "phydrax-"), _built_wheel(
        meshcore_output, "phydrax_meshcore-"
    )


def _venv_python(environment: Path, /) -> Path:
    scripts = environment / ("Scripts" if os.name == "nt" else "bin")
    return scripts / ("python.exe" if os.name == "nt" else "python")


def _clean_environment(runner: Path | None = None, /) -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("PHYDRAX_") or name in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
        ):
            environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    if runner is not None:
        environment["PYTHONPATH"] = str(runner)
    return environment


def _json_command(
    python: Path, code: str, /, *, cwd: Path, environment: dict[str, str], evidence: Path
) -> dict[str, object]:
    wrapped = _guarded_code("    exec(" + repr(code) + ")")
    output = _command(
        [str(python), "-c", wrapped], cwd=cwd, env=environment, evidence=evidence
    )
    try:
        record = json.loads(output.splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Installed qualification child returned invalid JSON: {output!r}"
        ) from error
    if not isinstance(record, dict):
        raise RuntimeError("Installed qualification child must return one JSON object.")
    return record


def _base_refusal(
    python: Path, /, *, cwd: Path, environment: dict[str, str], evidence: Path
) -> dict[str, object]:
    code = r"""
import json, numpy as np, jax.numpy as jnp
import phydrax as phx
from phydrax._meshcore import MeshcoreUnavailableError, meshcore_available
from phydrax._physical import SpatialCoordinateContract
from phydrax.geometry.simplicial import SegmentMesh
from phydrax.meshing._contracts import CellFamilyPolicy, CellMeshingTarget, CurveMeshingSpec, MeshingFailure, MeshingFailureCategory
from phydrax.meshing._scope import MeshingScope
from phydrax.meshing._sizing import UniformSizeControl
from phydrax.meshing.providers import NativeMeshingProvider
from phydrax.meshing.providers._native_sources import NativeCurveSource
if meshcore_available(): raise RuntimeError("Base refusal phase unexpectedly has meshcore installed.")
segments = SegmentMesh(jnp.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=jnp.float64), jnp.asarray(((0, 1),), dtype=jnp.int32), source_id="installed-missing-meshcore")
source = NativeCurveSource(segments, "revision-1", curve_entity_ids=np.asarray((1,), dtype=np.int64), vertex_entity_ids=np.asarray((1, 2), dtype=np.int64))
scope = MeshingScope.whole(source, 1)
spec = CurveMeshingSpec(CellMeshingTarget(1, 2, CellFamilyPolicy(required=("interval",))), scope, size_controls=(UniformSizeControl(scope, 0.5),), maximum_deviation=0.01)
try:
    NativeMeshingProvider().plan(source, spec, coordinate_contract=SpatialCoordinateContract.si()).execute()
except MeshingFailure as error:
    if error.category is not MeshingFailureCategory.PROVIDER_UNAVAILABLE:
        raise RuntimeError("Missing meshcore was not a provider-unavailable refusal.") from error
    print(json.dumps({"status": "passed", "meshcore_available": False, "refusal_category": error.category.value, "detail": str(error)}))
except MeshcoreUnavailableError as error:
    print(json.dumps({"status": "passed", "meshcore_available": False, "refusal_category": type(error).__name__, "detail": str(error)}))
else:
    raise RuntimeError("Native meshing unexpectedly ran without the meshcore wheel.")
"""
    return _json_command(
        python, code, cwd=cwd, environment=environment, evidence=evidence
    )


def _native_probe(
    python: Path, /, *, cwd: Path, environment: dict[str, str], evidence: Path
) -> dict[str, object]:
    code = r"""
import importlib.metadata, json, pathlib, sys
import phydrax as phx
from phydrax._meshcore import load_meshcore, meshcore_available, meshcore_identity
phydrax_path = pathlib.Path(phx.__file__).resolve()
if not phydrax_path.is_relative_to(pathlib.Path(sys.prefix).resolve()):
    raise RuntimeError(f"phydrax imported outside the isolated environment: {phydrax_path}")
if not meshcore_available():
    raise RuntimeError("Installed native phase lacks a usable ABI-matched meshcore library.")
phydrax_version = importlib.metadata.version("phydrax")
meshcore_version = importlib.metadata.version("phydrax-meshcore")
if phydrax_version != meshcore_version:
    raise RuntimeError(f"Mismatched installed wheels: phydrax {phydrax_version}, meshcore {meshcore_version}.")
print(json.dumps({"status": "passed", "python": sys.version, "prefix": sys.prefix, "phydrax_path": str(phydrax_path), "phydrax_version": phydrax_version, "meshcore_version": meshcore_version, "meshcore_identity": meshcore_identity(), "meshcore_library_path": str(load_meshcore().path.resolve())}))
"""
    return _json_command(
        python, code, cwd=cwd, environment=environment, evidence=evidence
    )


def _guarded_code(body: str, /, *, expected_runner: Path | None = None) -> str:
    roots = json.dumps(BLOCKED_ENGINES)
    qualification_root = (
        "None"
        if expected_runner is None
        else "pathlib.Path("
        + json.dumps(str((expected_runner / "tools/meshing_qualification").resolve()))
        + ")"
    )
    return f"""
import importlib.abc, pathlib, runpy, sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType
blocked = {roots}
attempted = []
class _RejectOptionalEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None) -> ModuleSpec | None:
        if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
            attempted.append(fullname)
            raise ModuleNotFoundError("optional geometry engine import rejected: " + fullname)
        return None
sys.meta_path.insert(0, _RejectOptionalEngine())
exit_code = 0
try:
{body}
except SystemExit as error:
    exit_code = int(error.code or 0)
expected_qualification_root = {qualification_root}
if expected_qualification_root is not None:
    qualification_modules = sorted(
        name for name in sys.modules
        if name == "tools.meshing_qualification"
        or name.startswith("tools.meshing_qualification.")
    )
    if not qualification_modules:
        raise RuntimeError("Installed workflow did not load the isolated qualification package.")
    for module_name in qualification_modules:
        module_path = getattr(sys.modules[module_name], "__file__", None)
        if module_path is None or not pathlib.Path(module_path).resolve().is_relative_to(expected_qualification_root):
            raise RuntimeError("Qualification callable loaded outside the isolated runner: " + module_name)
import phydrax as _phx
if not pathlib.Path(_phx.__file__).resolve().is_relative_to(pathlib.Path(sys.prefix).resolve()):
    raise RuntimeError("Installed workflow imported Phydrax from outside its isolated environment.")
from phydrax._meshcore import load_meshcore as _load_meshcore, meshcore_available as _meshcore_available
if _meshcore_available():
    import phydrax_meshcore as _meshcore_package
    _prefix = pathlib.Path(sys.prefix).resolve()
    _package_path = pathlib.Path(_meshcore_package.__file__).resolve()
    _library_path = _load_meshcore().path.resolve()
    _wheel_library_path = pathlib.Path(_meshcore_package.library_path()).resolve()
    if not _package_path.is_relative_to(_prefix) or not _library_path.is_relative_to(_prefix):
        raise RuntimeError("Native workflow loaded meshcore outside its isolated environment.")
    if _library_path != _wheel_library_path:
        raise RuntimeError("Native workflow did not load the installed meshcore wheel's library.")
loaded = sorted(name for name in sys.modules if any(name == root or name.startswith(root + ".") for root in blocked))
if attempted or loaded:
    raise RuntimeError("Engine-free workflow attempted/loaded optional engines: " + repr({{"attempted": attempted, "loaded": loaded}}))
if exit_code:
    raise SystemExit(exit_code)
"""


def _run_guarded_cli(
    python: Path, /, *, cwd: Path, environment: dict[str, str], evidence: Path
) -> dict[str, object]:
    argv = [
        "tools.meshing_qualification",
        "--scenario",
        "native-corpus",
        "--events",
        str(evidence / "native-events.jsonl"),
    ]
    body = f"    sys.argv = {argv!r}\n    runpy.run_module('tools.meshing_qualification', run_name='__main__')"
    runner = Path(environment["PYTHONPATH"]).expanduser().resolve()
    output = _command(
        [str(python), "-c", _guarded_code(body, expected_runner=runner)],
        cwd=cwd,
        env=environment,
        evidence=evidence,
    )
    try:
        record = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Native qualification CLI returned invalid JSON: {output!r}"
        ) from error
    if not isinstance(record, dict):
        raise RuntimeError("Native qualification CLI must return a JSON object.")
    return record


def _check_backend(record: dict[str, object], /, scenario: str) -> None:
    candidate: object = record.get("accepted") or record.get("native_result")
    consumer = record.get("consumer")
    if isinstance(consumer, dict):
        candidate = consumer
    if isinstance(candidate, dict) and (
        "provider" in candidate or "backend" in candidate
    ):
        provider = candidate.get("provider", candidate.get("backend"))
        if provider != "phydrax-native":
            raise RuntimeError(f"{scenario} used a non-native backend: {provider!r}")
        if not str(candidate.get("actual_version", "")).startswith("phydrax-meshcore "):
            raise RuntimeError(
                f"{scenario} did not report the actual meshcore backend identity."
            )
        return
    identity = record.get("meshcore_identity")
    if not isinstance(identity, str):
        runtime = record.get("runtime")
        if isinstance(runtime, dict):
            identity = runtime.get("meshcore")
    if not isinstance(identity, str) or not identity.startswith("phydrax-meshcore "):
        raise RuntimeError(
            f"{scenario} omitted a valid native backend identity: {identity!r}"
        )


def _validate_native_results(
    native: dict[str, object], /
) -> dict[str, dict[str, object]]:
    records = native.get("scenarios")
    if not isinstance(records, dict):
        raise RuntimeError(
            "Installed all-native qualification omitted its scenario record mapping."
        )

    expected = set(NATIVE_SCENARIOS)
    actual = set(records)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    failures: dict[str, object] = {}
    validated: dict[str, dict[str, object]] = {}
    for name in sorted(expected & actual):
        scenario_record = records[name]
        if not isinstance(scenario_record, dict):
            failures[name] = {"status": "invalid-result-record"}
            continue
        validated[name] = scenario_record
        if scenario_record.get("status") != "passed":
            failure: dict[str, object] = {"status": scenario_record.get("status")}
            for field in ("error_type", "detail", "dependency"):
                if field in scenario_record:
                    failure[field] = scenario_record[field]
            failures[name] = failure
        elif "selected_profiles" in scenario_record:
            # A profile selection is one part of a scenario, never its result.
            failures[name] = {
                "status": "partial-profile-selection",
                "selected_profiles": scenario_record["selected_profiles"],
            }
        elif scenario_record.get("scenario") != name:
            failures[name] = {
                "status": "scenario-identity-mismatch",
                "reported_scenario": scenario_record.get("scenario"),
            }
        elif not (set(scenario_record) - {"scenario", "status", "runtime"}):
            failures[name] = {"status": "missing-scenario-specific-evidence"}

    status = native.get("status")
    if missing or extra or failures or status != "passed":
        details = {
            "status": status,
            "missing": missing,
            "extra": extra,
            "failed": failures,
        }
        raise RuntimeError(
            "Installed all-native qualification did not pass the canonical native inventory: "
            + json.dumps(details, sort_keys=True)
        )
    return validated


def qualify(arguments: argparse.Namespace, *, evidence: Path) -> dict[str, object]:
    python = _resolve_python(arguments.python, evidence=evidence)
    wheel_arguments = (arguments.wheel, arguments.meshcore_wheel)
    if (wheel_arguments[0] is None) != (wheel_arguments[1] is None):
        raise RuntimeError(
            "Pass both --wheel and --meshcore-wheel, or omit both to build matching wheels."
        )
    with tempfile.TemporaryDirectory(
        prefix="phydrax-native-dependency-isolation-"
    ) as directory_name:
        directory = Path(directory_name)
        if wheel_arguments[0] is None:
            phydrax_wheel, meshcore_wheel = _build_wheels(
                python, directory / "wheels", evidence=evidence
            )
        else:
            phydrax_wheel = Path(arguments.wheel).expanduser().resolve()
            meshcore_wheel = Path(arguments.meshcore_wheel).expanduser().resolve()
        phydrax_version = _wheel_version(phydrax_wheel, "phydrax")
        meshcore_version = _wheel_version(meshcore_wheel, "phydrax-meshcore")
        if phydrax_version != meshcore_version:
            raise RuntimeError(
                f"Wheel versions do not match: phydrax {phydrax_version}, meshcore {meshcore_version}."
            )
        environment_path = directory / "venv"
        _command(
            ["uv", "venv", "--python", str(python), str(environment_path)],
            evidence=evidence,
        )
        installed_python = _venv_python(environment_path)
        isolated_env = _clean_environment()
        _command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(installed_python),
                str(phydrax_wheel),
            ],
            cwd=directory,
            env=isolated_env,
            evidence=evidence,
        )
        missing_meshcore = _base_refusal(
            installed_python, cwd=directory, environment=isolated_env, evidence=evidence
        )
        if (
            missing_meshcore.get("status") != "passed"
            or missing_meshcore.get("meshcore_available") is not False
        ):
            raise RuntimeError(
                "Base install did not prove typed refusal without meshcore."
            )
        if not arguments.all_native:
            return {
                "status": "passed",
                "base_missing_meshcore_refusal": missing_meshcore,
                "native": "not-requested",
            }
        _command(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(installed_python),
                "--no-deps",
                str(meshcore_wheel),
            ],
            cwd=directory,
            env=isolated_env,
            evidence=evidence,
        )
        native_identity = _native_probe(
            installed_python, cwd=directory, environment=isolated_env, evidence=evidence
        )
        runner = directory / "runner"
        tools_root = runner / "tools"
        tools_root.mkdir(parents=True)
        (tools_root / "__init__.py").write_text("", encoding="utf-8")
        shutil.copytree(
            ROOT / "tools/meshing_qualification",
            tools_root / "meshing_qualification",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        engine_free_env = _clean_environment(runner)
        native = _run_guarded_cli(
            installed_python,
            cwd=directory,
            environment=engine_free_env,
            evidence=evidence,
        )
        records = _validate_native_results(native)
        for name in NATIVE_SCENARIOS:
            _check_backend(records[name], name)
        return {
            "status": "passed",
            "python": str(python),
            "phydrax_wheel": str(phydrax_wheel),
            "meshcore_wheel": str(meshcore_wheel),
            "versions": {"phydrax": phydrax_version, "meshcore": meshcore_version},
            "base_missing_meshcore_refusal": missing_meshcore,
            "installed_runtime": native_identity,
            "native_scenarios": sorted(records),
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build/install matched CPython 3.13 wheels and reject optional geometry engines in installed native workflows."
    )
    parser.add_argument("--python", default="3.13")
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--meshcore-wheel", type=Path)
    parser.add_argument("--all-native", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh or empty directory for persistent evidence.",
    )
    arguments = parser.parse_args(argv)
    evidence: Path | None = None
    record: dict[str, object]
    try:
        if arguments.output_dir is None:
            evidence = Path(
                tempfile.mkdtemp(prefix="phydrax-native-isolation-evidence-")
            ).resolve()
        else:
            requested = arguments.output_dir.expanduser().resolve()
            requested.mkdir(parents=True, exist_ok=True)
            if any(requested.iterdir()):
                raise RuntimeError(f"Evidence directory must be empty: {requested}")
            evidence = requested
        print(f"Isolation evidence: {evidence}", file=sys.stderr, flush=True)
        record = qualify(arguments, evidence=evidence)
    except Exception as error:
        record = {
            "status": "failed",
            "error_type": type(error).__name__,
            "detail": str(error),
        }
    if evidence is not None:
        record["evidence_dir"] = str(evidence)
        try:
            _write_json(evidence / "result.json", record)
        except Exception as error:
            record = {
                "status": "failed",
                "error_type": type(error).__name__,
                "detail": f"Could not persist final result: {error}; original result: {record!r}",
                "evidence_dir": str(evidence),
            }
    print(json.dumps(record, indent=2, sort_keys=True, allow_nan=False))
    return 0 if record.get("status") == "passed" else 1


if __name__ == "__main__":
    faulthandler.enable()
    raise SystemExit(main())
