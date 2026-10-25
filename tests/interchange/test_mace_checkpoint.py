"""Trusted mace-torch source conversion against the independent provider oracle."""

import base64
import dataclasses
import hashlib
import json
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any, Literal, NoReturn

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
from examples import mace_checkpoint_conversion as conversion_example
from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax.atomistic._model_artifact import (
    ATOMISTIC_MODEL_ARTIFACT_LIMITS,
    AtomisticModelArtifactError,
    read_atomistic_model_artifact,
    write_atomistic_model_artifact,
)
from phydrax.atomistic.interchange import (
    _mace_checkpoint,
    _mace_worker,
    convert_mace_checkpoint,
    create_mace_provider_fixture,
    evaluate_mace_source,
    mace_source_gradients,
    MACECheckpointConversion,
    MACEConversionLimits,
    MACEProviderRuntime,
    MACESource,
    MACESourceKind,
    MACESourceProvenance,
    MACESourceRefusedError,
    TrustedTorchPickleSource,
)
from phydrax.interchange import pin_executable
from tests._support.mace_source import (
    admitted_source,
    assert_matches_oracle,
    CHARGE_SPIN_SPECS,
    committed_maceles_potential,
    committed_manifest,
    committed_source_potential,
    configurations,
    DECLARATIONS,
    execution,
    FAMILY_DECLARATIONS,
    FLOAT32_GATES,
    FLOAT64_GATES,
    for_release,
    latent_resources,
    LES_ARGUMENTS,
    les_provider,
    native_case,
    ONE_LAYER_INVARIANT_READOUT,
    provider_runtime,
    reduced_cg_provider,
    RELEASE_FIXTURES,
    structure,
    TWO_LAYER_DENSITY_MULTIHEAD,
    TWO_LAYER_REDUCED_CG,
    TWO_LAYER_RESIDUAL_AGNESI_ZBL,
    within,
)
from tools.mace_checkpoint_campaign import CAMPAIGN_ROWS


REPOSITORY = Path(__file__).resolve().parents[2]
SEED = 20261003
DIRECTION_GATE = (1.0e-10, 1.0e-7)
FLOAT32_PRECISION = phx.atomistic.AtomisticPrecisionPolicy(
    coordinate_dtype="float32",
    compute_dtype="float32",
    reduction_dtype="float32",
    output_dtype="float32",
)


@pytest.mark.parametrize("fault", ("none", "digest", "size", "algorithm"))
def test_provider_record_files_stream_and_validate_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    payload = bytes(range(256)) * 8193
    (tmp_path / "provider.bin").write_bytes(payload)
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    size = len(payload)
    algorithm = "sha256"
    if fault == "digest":
        digest = b"0" * len(digest)
    elif fault == "size":
        size += 1
    elif fault == "algorithm":
        algorithm = "sha512"
    info = tmp_path / "provider.dist-info"
    info.mkdir()
    (info / "RECORD").write_text(
        f"provider.bin,{algorithm}={digest.decode('ascii')},{size}\n",
        encoding="utf-8",
    )
    distribution = metadata.Distribution.at(info)

    def refuse_full_read(self: Path) -> NoReturn:
        pytest.fail("Provider RECORD verification must stream file bytes.")

    monkeypatch.setattr(Path, "read_bytes", refuse_full_read)
    if fault == "none":
        assert _mace_worker._verify_record_files(distribution) == 1
    else:
        with pytest.raises(_mace_worker.ProviderRefusal, match="RECORD"):
            _mace_worker._verify_record_files(distribution)


@pytest.mark.parametrize("fault", ("overwrite", "replace", "touch", "nonregular"))
def test_provider_record_files_refuse_stat_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    payload = b"provider file"
    location = tmp_path / "provider.bin"
    location.write_bytes(payload)
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
    info = tmp_path / "provider.dist-info"
    info.mkdir()
    (info / "RECORD").write_text(
        f"provider.bin,sha256={digest.decode('ascii')},{len(payload)}\n",
        encoding="utf-8",
    )
    distribution = metadata.Distribution.at(info)
    original_digest = hashlib.file_digest

    def mutate_after_digest(*args: Any, **kwargs: Any) -> Any:
        result = original_digest(*args, **kwargs)
        if fault == "overwrite":
            location.write_bytes(b"x" * len(payload))
        elif fault == "replace":
            replacement = tmp_path / "replacement.bin"
            replacement.write_bytes(payload)
            os.replace(replacement, location)
        elif fault == "touch":
            before = location.stat()
            os.utime(
                location,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
            )
        return result

    if fault == "nonregular":
        location.unlink()
        location.symlink_to(os.devnull)
        refusal = "regular file"
    else:
        monkeypatch.setattr(hashlib, "file_digest", mutate_after_digest)
        refusal = "changed while being hashed"
    with pytest.raises(_mace_worker.ProviderRefusal, match=refusal):
        _mace_worker._verify_record_files(distribution)


@pytest.mark.parametrize("fault", ("none", "digest", "size", "source-limit"))
def test_provider_fixture_publication_streams_and_checks_file_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    payload = bytes(range(256)) * 8193
    digest = hashlib.sha256(payload).hexdigest()
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    (root / "fixture.pt").write_bytes(payload)
    record = {
        "seed": SEED,
        "source_kind": "torch-state-dict",
        "source_dtype": "float64",
        "fixture_byte_size": len(payload) + (fault == "size"),
        "fixture_sha256": "0" * 64 if fault == "digest" else digest,
        "declaration": ONE_LAYER_INVARIANT_READOUT,
    }

    def provider_result(
        provider: MACEProviderRuntime,
        request: Mapping[str, Any],
        source_payload: bytes | None,
        limits: MACEConversionLimits,
        /,
        *,
        extra_outputs: Sequence[tuple[str, int]] = (),
    ) -> tuple[
        dict[str, Any], dict[str, np.ndarray], Path, tempfile.TemporaryDirectory[str]
    ]:
        assert extra_outputs == (("fixture.pt", limits.max_source_bytes),)
        return {"fixture": record, "provider": {}}, {}, root / "result.zip", directory

    def refuse_full_read(self: Path) -> NoReturn:
        pytest.fail("Provider fixture publication must stream file bytes.")

    monkeypatch.setattr(_mace_checkpoint, "_run_provider", provider_result)
    limits = MACEConversionLimits(
        max_source_bytes=len(payload) - 1
        if fault == "source-limit"
        else MACEConversionLimits().max_source_bytes
    )
    with monkeypatch.context() as streaming:
        streaming.setattr(Path, "read_bytes", refuse_full_read)
        # The test owns the provider-result boundary; no provider is executed.
        runtime: Any = None
        if fault == "none":
            fixture = create_mace_provider_fixture(
                ONE_LAYER_INVARIANT_READOUT,
                tmp_path,
                provider=runtime,
                seed=SEED,
                limits=limits,
            )
        else:
            with pytest.raises(ValueError, match="fixture"):
                create_mace_provider_fixture(
                    ONE_LAYER_INVARIANT_READOUT,
                    tmp_path,
                    provider=runtime,
                    seed=SEED,
                    limits=limits,
                )
    assert not root.exists()
    if fault == "none":
        assert fixture.path.read_bytes() == payload
        assert fixture.sha256 == digest
        assert fixture.byte_size == len(payload)
    else:
        assert not tuple(tmp_path.iterdir())


_STREAMED_PROVIDER_FIXTURE = """
import dataclasses
import hashlib
import importlib.abc
import io
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

class RefusePhydrax(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "phydrax":
            raise ImportError("worker must remain standalone")
        return None

sys.meta_path.insert(0, RefusePhydrax())
sys.path.insert(0, sys.argv[1])
import torch
worker = runpy.run_path(sys.argv[2], run_name="_standalone_mace_worker")
request = json.loads(sys.argv[3])
limits = worker["_Limits"](**request.pop("limits"))
torch.manual_seed(request["seed"])
dtype = {"float32": torch.float32, "float64": torch.float64}[request["source_dtype"]]
model = worker["_build"](request["architecture"], dtype, limits)
buffer = io.BytesIO()
payload = model.state_dict() if request["source_kind"] == "torch-state-dict" else model
torch.save(payload, buffer)
expected = buffer.getvalue()

def refuse_buffer(*args, **kwargs):
    raise AssertionError("fixture serialization must not buffer the file")

def refuse_read(self):
    raise AssertionError("fixture hashing must stream")

run_fixture = worker["_run_fixture"]
run_fixture.__globals__["io"] = SimpleNamespace(BytesIO=refuse_buffer)
Path.read_bytes = refuse_read
record = run_fixture(request, Path(sys.argv[4]) / "result.zip", limits)
with (Path(sys.argv[4]) / "fixture.pt").open("rb") as stream:
    actual = stream.read()
assert actual == expected
assert record["fixture_sha256"] == hashlib.sha256(expected).hexdigest()
assert record["fixture_byte_size"] == len(expected)
try:
    run_fixture(
        request,
        Path(sys.argv[4]) / "result.zip",
        dataclasses.replace(limits, max_source_bytes=len(expected) - 1),
    )
except worker["ProviderRefusal"] as error:
    assert "source byte limit" in str(error)
else:
    raise AssertionError("oversized serialized fixture must be refused")
assert not any(name.split(".")[0] == "phydrax" for name in sys.modules)
"""


@pytest.mark.parametrize("kind", ("torch-state-dict", "torch-full-model"))
@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_streamed_provider_fixture_preserves_buffer_serialization(
    provider: MACEProviderRuntime, tmp_path: Path, kind: str, dtype: str
) -> None:
    request = {
        "architecture": ONE_LAYER_INVARIANT_READOUT,
        "seed": SEED,
        "source_kind": kind,
        "source_dtype": dtype,
        "limits": _mace_checkpoint._worker_limits(MACEConversionLimits()),
    }
    completed = subprocess.run(
        [
            provider.interpreter.path,
            "-I",
            "-c",
            textwrap.dedent(_STREAMED_PROVIDER_FIXTURE),
            provider.site_packages,
            str(REPOSITORY / "phydrax/atomistic/interchange/_mace_worker.py"),
            json.dumps(request),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@pytest.fixture(scope="module")
def provider() -> MACEProviderRuntime:
    return provider_runtime()


@pytest.fixture(scope="module")
def fixtures(
    provider: MACEProviderRuntime, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("mace-sources")
    built = {
        name: create_mace_provider_fixture(
            declaration, root, provider=provider, seed=SEED
        )
        for name, declaration in DECLARATIONS.items()
    }
    built["full-object"] = create_mace_provider_fixture(
        ONE_LAYER_INVARIANT_READOUT,
        root,
        provider=provider,
        seed=SEED,
        kind="torch-full-model",
    )
    built["float32"] = create_mace_provider_fixture(
        TWO_LAYER_DENSITY_MULTIHEAD, root, provider=provider, seed=SEED, dtype="float32"
    )
    return built


def _safe(fixture: Any) -> MACESource:
    return admitted_source(
        fixture.path, "torch-state-dict", architecture=fixture.declaration
    )


# Converted sources keyed by (declaration name, active head).
Conversions = dict[tuple[str, str], MACECheckpointConversion]


@pytest.fixture(scope="module")
def conversions(provider: MACEProviderRuntime, fixtures: dict[str, Any]) -> Conversions:
    converted: Conversions = {}
    for name, declaration in DECLARATIONS.items():
        for head in declaration["heads"]:
            converted[name, head] = convert_mace_checkpoint(
                _safe(fixtures[name]), provider=provider, head=head
            )
    return converted


@pytest.mark.parametrize(
    ("name", "head"),
    [(name, head) for name, item in DECLARATIONS.items() for head in item["heads"]],
)
def test_admitted_layout_matches_independent_provider_oracle(
    provider: MACEProviderRuntime,
    fixtures: dict[str, Any],
    conversions: Conversions,
    name: str,
    head: str,
) -> None:
    declaration = DECLARATIONS[name]
    conversion = conversions[name, head]
    provenance = conversion.provenance
    assert provenance.head == head
    assert provenance.source_kind == "torch-state-dict"
    assert not provenance.trusted_full_object
    assert provenance.source_dtype == "float64"
    assert provenance.source_sha256 == fixtures[name].sha256
    assert dict(provenance.declaration) == dict(fixtures[name].declaration)
    assert provenance.basis_residual <= 1.0e-12
    assert not provenance.layout_normalization["legacy_head"]
    assert all(
        not provenance.layout_normalization[item]
        for item in (
            "derived_zero_flags",
            "dropped_inert_state",
            "integral_buffers",
            "precision_buffers",
        )
    )
    architecture = conversion.potential.configuration
    assert tuple(architecture.species) == tuple(declaration["atomic_numbers"])
    assert architecture.head == head
    cases = configurations(tuple(declaration["atomic_numbers"]))
    oracle = evaluate_mace_source(
        _safe(fixtures[name]), cases, provider=provider, head=head
    )
    assert oracle.provider["distributions"]["mace-torch"]["version"] == "0.3.16"
    assert_matches_oracle(conversion.potential, cases, oracle, FLOAT64_GATES)


def test_perturbed_source_parameter_fails_the_frozen_parity_gate(
    provider: MACEProviderRuntime, fixtures: dict[str, Any], conversions: Conversions
) -> None:
    conversion = conversions["one-layer-invariant-readout", "Default"]
    name = next(key for key in conversion.source_tensors if key.endswith("weights_max"))
    perturbed = conversion.reconstruct(
        {name: conversion.source_tensors[name] * (1.0 + 1.0e-4)}
    )
    finite, _ = configurations((1, 8))
    oracle = evaluate_mace_source(
        _safe(fixtures["one-layer-invariant-readout"]), [finite], provider=provider
    ).cases[0]
    assert within(
        native_case(conversion.potential, finite)["energy"],
        oracle.energy,
        FLOAT64_GATES["energy"],
    )
    assert not within(
        native_case(perturbed, finite)["forces"], oracle.forces, FLOAT64_GATES["forces"]
    )


def _native_leaves(potential: Any) -> list[np.ndarray]:
    parameters = phx.partition_parameters(potential)[0]
    return [np.asarray(leaf) for leaf in jax.tree.leaves(parameters)]


def _contraction_weights(conversion: Any) -> list[str]:
    names = [
        name
        for name in conversion.source_tensors
        if ".symmetric_contractions." in name and ".weights" in name
    ]
    assert names
    return names


def _assert_original_w(conversion: Any) -> None:
    """Every source symmetric-contraction W is itself a native parameter leaf."""

    leaves = _native_leaves(conversion.potential)
    for name in _contraction_weights(conversion):
        source = conversion.source_tensors[name]
        assert any(
            leaf.shape == source.shape and np.array_equal(leaf, source) for leaf in leaves
        ), name


def _assert_source_directions(
    conversion: Any, source: MACESource, provider: MACEProviderRuntime, head: str
) -> None:
    """Native JVPs along source-space directions equal provider gradients.

    Covers the energy and the mixed force contraction ``d(sum F * W)/d theta``.
    """

    _, periodic = configurations(tuple(conversion.potential.configuration.species))
    generator = np.random.default_rng(17)
    weights = generator.standard_normal((len(periodic.numbers), 3))
    gradients = mace_source_gradients(
        source, [periodic], [weights], provider=provider, head=head
    )
    assert set(_contraction_weights(conversion)) <= set(gradients.parameters)
    parameters, state, fixed = phx.partition_parameters(conversion.potential)
    batch = phx.atomistic.AtomisticBatch.from_structure(
        structure(conversion.potential, periodic)
    )
    graph = execution(periodic)
    payload = phx.atomistic.AtomisticGraphLayout.for_batch(
        batch, graph, cutoff=conversion.potential.configuration.cutoff
    ).prepare(batch)
    contraction = jnp.asarray(weights)

    def scalars(values: Any) -> tuple[jax.Array, jax.Array]:
        potential = phx.combine_parameters(values, state, fixed)
        result = phx.atomistic.atomistic_energy_derivatives(
            potential, payload, batch.positions
        )
        forces = result.forces
        assert forces is not None
        return result.energy[0], jnp.sum(forces[0, :3] * contraction)

    step = 1.0e-3
    for _ in range(2):
        direction = {
            name: generator.standard_normal(gradients.energy_gradients[0][name].shape)
            for name in gradients.parameters
        }
        plus = conversion.reconstruct(
            {
                name: conversion.source_tensors[name] + step * value
                for name, value in direction.items()
            }
        )
        minus = conversion.reconstruct(
            {
                name: conversion.source_tensors[name] - step * value
                for name, value in direction.items()
            }
        )
        # The source-to-native map is linear, so the central difference is
        # the exact native image of the source direction.
        native_direction = jax.tree.map(
            lambda high, low: (high - low) / (2.0 * step),
            phx.partition_parameters(plus)[0],
            phx.partition_parameters(minus)[0],
        )
        _, (energy, force) = jax.jvp(scalars, (parameters,), (native_direction,))
        provider_energy = sum(
            float(np.sum(gradients.energy_gradients[0][name] * value))
            for name, value in direction.items()
        )
        provider_force = sum(
            float(np.sum(gradients.force_gradients[0][name] * value))
            for name, value in direction.items()
        )
        assert within(float(energy), provider_energy, DIRECTION_GATE)
        assert within(float(force), provider_force, DIRECTION_GATE)


def test_original_w_is_a_native_parameter_and_source_directions_match(
    provider: MACEProviderRuntime, fixtures: dict[str, Any], conversions: Conversions
) -> None:
    conversion = conversions["two-layer-density-multihead", "r2scan"]
    _assert_original_w(conversion)
    _assert_source_directions(
        conversion, _safe(fixtures["two-layer-density-multihead"]), provider, "r2scan"
    )


def test_reduced_cg_source_keeps_its_reduced_parameterization(
    provider: MACEProviderRuntime, conversions: Conversions, tmp_path: Path
) -> None:
    reduced_provider = reduced_cg_provider(provider)
    fixture = create_mace_provider_fixture(
        TWO_LAYER_REDUCED_CG, tmp_path, provider=reduced_provider, seed=SEED
    )
    assert fixture.declaration["use_reduced_cg"] is True
    source = _safe(fixture)
    conversion = convert_mace_checkpoint(source, provider=reduced_provider)
    assert conversion.provenance.declaration["use_reduced_cg"] is True
    assert not conversion.provenance.layout_normalization["relabelled_reduced_cg"]
    original = conversions["two-layer-residual-agnesi-zbl", "Default"]
    # The reduced basis has fewer product parameters than the original one.
    assert any(
        conversion.source_tensors[name].shape[1] < original.source_tensors[name].shape[1]
        for name in _contraction_weights(conversion)
    )
    _assert_original_w(conversion)
    cases = configurations((1, 8))
    oracle = evaluate_mace_source(source, cases, provider=reduced_provider)
    assert_matches_oracle(conversion.potential, cases, oracle, FLOAT64_GATES)
    _assert_source_directions(conversion, source, reduced_provider, "Default")
    undeclared = dataclasses.replace(reduced_provider, cuequivariance_version=None)
    with pytest.raises(MACESourceRefusedError, match="declare its release"):
        convert_mace_checkpoint(source, provider=undeclared)


def test_trusted_full_object_converts_to_the_same_source_parameters(
    provider: MACEProviderRuntime, fixtures: dict[str, Any], conversions: Conversions
) -> None:
    full = fixtures["full-object"]
    with pytest.raises(PermissionError, match="TrustedTorchPickleSource"):
        admitted_source(full.path, "torch-full-model")
    with pytest.raises(PermissionError, match="different digest"):
        MACESource(
            *_admission(full.path),
            "torch-full-model",
            trust=TrustedTorchPickleSource(fixtures["float32"].sha256, "other bytes"),
        )
    trusted = admitted_source(
        full.path, "torch-full-model", trust="provider-built fixture of this test"
    )
    conversion = convert_mace_checkpoint(trusted, provider=provider)
    safe = conversions["one-layer-invariant-readout", "Default"]
    assert conversion.provenance.trusted_full_object
    assert (
        conversion.provenance.source_sha256
        == full.sha256
        != safe.provenance.source_sha256
    )
    assert set(conversion.source_tensors) == set(safe.source_tensors)
    for name, value in safe.source_tensors.items():
        np.testing.assert_array_equal(conversion.source_tensors[name], value)
    finite, periodic = configurations((1, 8))
    for case in (finite, periodic):
        assert (
            native_case(conversion.potential, case)["energy"]
            == (native_case(safe.potential, case)["energy"])
        )


def _admission(path: Path) -> tuple[Any, Any, Any]:
    source = admitted_source(
        path, "torch-state-dict", architecture=ONE_LAYER_INVARIANT_READOUT
    )
    return source.artifact, source.manifest, source.policy


def test_safe_loader_refuses_a_pickle_without_unsafe_fallback(
    provider: MACEProviderRuntime, fixtures: dict[str, Any]
) -> None:
    full = fixtures["full-object"]
    disguised = admitted_source(
        full.path, "torch-state-dict", architecture=ONE_LAYER_INVARIANT_READOUT
    )
    # The pickle's module global is refused from its metadata; it is never loaded.
    with pytest.raises(MACESourceRefusedError, match="ScaleShiftMACE, outside the safe"):
        convert_mace_checkpoint(disguised, provider=provider)


def test_declared_architecture_must_describe_the_state_exactly(
    provider: MACEProviderRuntime, fixtures: dict[str, Any]
) -> None:
    other = {**ONE_LAYER_INVARIANT_READOUT, "hidden_irreps": "6x0e"}
    mismatched = admitted_source(
        fixtures["one-layer-invariant-readout"].path,
        "torch-state-dict",
        architecture=other,
    )
    with pytest.raises(MACESourceRefusedError):
        convert_mace_checkpoint(mismatched, provider=provider)


def test_unadmitted_readout_gate_refuses_before_any_native_model(
    provider: MACEProviderRuntime, tmp_path: Path
) -> None:
    declaration = {**TWO_LAYER_RESIDUAL_AGNESI_ZBL, "gate": "tanh"}
    fixture = create_mace_provider_fixture(
        declaration, tmp_path, provider=provider, seed=3
    )
    with pytest.raises(MACESourceRefusedError, match="normalized SiLU activation"):
        convert_mace_checkpoint(_safe(fixture), provider=provider)


def test_rights_digest_size_and_release_pins_refuse(
    provider: MACEProviderRuntime, fixtures: dict[str, Any], tmp_path: Path
) -> None:
    fixture = fixtures["one-layer-invariant-readout"]
    with pytest.raises(PermissionError, match="license is not admitted"):
        admitted_source(
            fixture.path,
            "torch-state-dict",
            license_id="ASL",
            architecture=fixture.declaration,
        )
    copy = tmp_path / fixture.path.name
    copy.write_bytes(fixture.path.read_bytes())
    source = admitted_source(copy, "torch-state-dict", architecture=fixture.declaration)
    payload = bytearray(copy.read_bytes())
    payload[-1] ^= 0xFF
    copy.write_bytes(bytes(payload))
    with pytest.raises(ValueError, match="SHA-256 checksum mismatch"):
        convert_mace_checkpoint(source, provider=provider)
    with pytest.raises(ValueError, match="source byte limit"):
        convert_mace_checkpoint(
            _safe(fixture),
            provider=provider,
            limits=MACEConversionLimits(max_source_bytes=fixture.byte_size - 1),
        )
    other_torch = dataclasses.replace(provider, torch_version="0.0.0")
    with pytest.raises(MACESourceRefusedError, match="differs from the declared"):
        convert_mace_checkpoint(_safe(fixture), provider=other_torch)


def _records(path: Path) -> dict[str, int]:
    with zipfile.ZipFile(path) as archive:
        return {
            info.filename.split("/", 1)[1]: info.file_size for info in archive.infolist()
        }


def _rewritten(
    source: Path, target: Path, records: Mapping[str, tuple[bytes, int]]
) -> Path:
    """Copy a torch.save container, replacing or adding (payload, ZIP method) records."""

    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w") as copy:
        prefix = original.namelist()[0].split("/", 1)[0]
        for info in original.infolist():
            if info.filename.removeprefix(prefix + "/") not in records:
                copy.writestr(info.filename, original.read(info), zipfile.ZIP_STORED)
        for name, (payload, method) in records.items():
            copy.writestr(f"{prefix}/{name}", payload, method)
    return target


def _pickled_int(value: int) -> bytes:
    if 0 <= value < 2**31:
        return pickle.BININT + struct.pack("<i", value)
    encoded = pickle.encode_long(value)
    return pickle.LONG1 + bytes([len(encoded)]) + encoded


def _pickled_text(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return pickle.BINUNICODE + struct.pack("<I", len(encoded)) + encoded


_PICKLED_ORDERED_DICT = (
    pickle.GLOBAL + b"collections\nOrderedDict\n" + pickle.EMPTY_TUPLE + pickle.REDUCE
)


def _view_state_dict(path: Path, shapes: Sequence[tuple[int, ...]]) -> Path:
    """A torch.save state dict of zero-stride views over one stored float64.

    Written without torch, opcode by opcode as torch.save emits a state dict;
    the weights-only loader realizes each view without allocating it.
    """

    pickled = pickle.PROTO + b"\x02" + _PICKLED_ORDERED_DICT + pickle.MARK
    for index, shape in enumerate(shapes):
        storage = (
            pickle.MARK
            + _pickled_text("storage")
            + pickle.GLOBAL
            + b"torch\nDoubleStorage\n"
            + _pickled_text("0")
            + _pickled_text("cpu")
            + _pickled_int(1)
            + pickle.TUPLE
            + pickle.BINPERSID
        )
        pickled += (
            _pickled_text(f"view{index}")
            + pickle.GLOBAL
            + b"torch._utils\n_rebuild_tensor_v2\n"
            + pickle.MARK
            + storage
            + _pickled_int(0)
            + pickle.MARK
            + b"".join(_pickled_int(extent) for extent in shape)
            + pickle.TUPLE
            + pickle.MARK
            + b"".join(_pickled_int(0) for _ in shape)
            + pickle.TUPLE
            + pickle.NEWFALSE
            + _PICKLED_ORDERED_DICT
            + pickle.TUPLE
            + pickle.REDUCE
        )
    pickled += pickle.SETITEMS + pickle.STOP
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("archive/data.pkl", pickled)
        archive.writestr("archive/byteorder", "little")
        archive.writestr("archive/data/0", struct.pack("<d", 1.0))
        archive.writestr("archive/version", "3\n")
    return path


def _compressed_storage(fixture: Any, target: Path) -> tuple[Path, MACEConversionLimits]:
    # 64 MiB of zeros deflates to ~64 KiB: a bomb under the default limits.
    bomb = (bytes(64 << 20), zipfile.ZIP_DEFLATED)
    return _rewritten(fixture.path, target, {"data/0": bomb}), MACEConversionLimits()


def _oversized_storage(fixture: Any, target: Path) -> tuple[Path, MACEConversionLimits]:
    largest = max(
        size for name, size in _records(fixture.path).items() if name.startswith("data/")
    )
    return fixture.path, MACEConversionLimits(max_tensor_bytes=largest - 1)


def _many_members(fixture: Any, target: Path) -> tuple[Path, MACEConversionLimits]:
    count = len(_records(fixture.path))
    extra = {f"data/extra{index}": (b"\0", zipfile.ZIP_STORED) for index in range(64)}
    path = _rewritten(fixture.path, target, extra)
    return path, MACEConversionLimits(max_members=count + 8)


def _logical_view(fixture: Any, target: Path) -> tuple[Path, MACEConversionLimits]:
    path = _view_state_dict(target, [(1 << 12, 1 << 12)])
    return path, MACEConversionLimits(max_tensor_elements=1 << 20)


def _aggregate_views(fixture: Any, target: Path) -> tuple[Path, MACEConversionLimits]:
    path = _view_state_dict(target, [(1 << 9, 1 << 9)] * 4)
    return path, MACEConversionLimits(max_source_bytes=4 << 20)


@pytest.mark.parametrize(
    ("craft", "refusal"),
    [
        pytest.param(_compressed_storage, "stored uncompressed", id="compressed"),
        pytest.param(_oversized_storage, r"'data/\d+' has \d+ bytes", id="oversized"),
        pytest.param(_many_members, "records, above the member limit", id="members"),
        pytest.param(_logical_view, "rank or tensor element limit", id="logical-view"),
        pytest.param(_aggregate_views, "logical bytes than the source", id="aggregate"),
    ],
)
def test_safe_source_resources_refuse_before_deserialization(
    provider: MACEProviderRuntime,
    fixtures: dict[str, Any],
    tmp_path: Path,
    craft: Any,
    refusal: str,
) -> None:
    # These refusals come only from the provider's container and pickle-metadata
    # preflight, which precedes torch.load and the declared model's construction.
    fixture = fixtures["one-layer-invariant-readout"]
    path, limits = craft(fixture, tmp_path / "crafted.pt")
    source = admitted_source(path, "torch-state-dict", architecture=fixture.declaration)
    with pytest.raises(MACESourceRefusedError, match=refusal):
        convert_mace_checkpoint(source, provider=provider, limits=limits)


@pytest.mark.parametrize(
    "inflated",
    [
        pytest.param({"hidden_irreps": "65536x0e"}, id="hidden"),
        pytest.param({"max_ell": 48}, id="max-ell"),
        pytest.param({"correlation": [12]}, id="correlation"),
        pytest.param({"radial_MLP": [65536, 65536]}, id="radial"),
        pytest.param({"num_bessel": 1 << 20}, id="bessel"),
    ],
)
def test_inflated_declaration_refuses_before_provider_construction(
    provider: MACEProviderRuntime, fixtures: dict[str, Any], inflated: dict[str, Any]
) -> None:
    # Each inflated model or its coupling basis is far beyond these limits; the
    # declaration is refused against the admitted tensors before it is built.
    fixture = fixtures["one-layer-invariant-readout"]
    source = admitted_source(
        fixture.path, "torch-state-dict", architecture={**fixture.declaration, **inflated}
    )
    limits = MACEConversionLimits(max_tensor_elements=1 << 16, max_tensor_bytes=1 << 20)
    # Refused by the admission against the state dict, which precedes _build.
    with pytest.raises(MACESourceRefusedError, match="admitted state dict"):
        convert_mace_checkpoint(source, provider=provider, limits=limits)


_MANY_HEADS = [f"head{index}" for index in range(4096)]


@pytest.mark.parametrize(
    ("declaration", "limits", "refusal"),
    [
        # (2**14 + 1)**2 harmonic components, refused before e3nn expands them.
        pytest.param(
            {**ONE_LAYER_INVARIANT_READOUT, "max_ell": 1 << 14},
            MACEConversionLimits(),
            "spherical-harmonic components, above the tensor limits",
            id="harmonic-degree",
        ),
        # (13**2)**6 generalized-CG coefficients at correlation 3.
        pytest.param(
            {**ONE_LAYER_INVARIANT_READOUT, "max_ell": 12},
            MACEConversionLimits(),
            "above the tensor limits",
            id="coupling-basis",
        ),
        # A 65536 x 65536 radial layer.
        pytest.param(
            {**ONE_LAYER_INVARIANT_READOUT, "radial_MLP": [65536, 65536]},
            MACEConversionLimits(),
            "above the tensor limits",
            id="radial",
        ),
        # A 65536 x 65536 channel mixing.
        pytest.param(
            {**ONE_LAYER_INVARIANT_READOUT, "hidden_irreps": "65536x0e"},
            MACEConversionLimits(),
            "above the tensor limits",
            id="channels",
        ),
        # (4096 heads x 4096 MLP) x 4096 head readout weights.
        pytest.param(
            {
                **TWO_LAYER_RESIDUAL_AGNESI_ZBL,
                "heads": _MANY_HEADS,
                "MLP_irreps": "4096x0e",
                "atomic_energies": [[-13.6, -2041.8]] * len(_MANY_HEADS),
            },
            MACEConversionLimits(),
            "above the tensor limits",
            id="heads",
        ),
        # Every tensor fits, the whole model does not.
        pytest.param(
            ONE_LAYER_INVARIANT_READOUT,
            MACEConversionLimits(max_source_bytes=4096),
            "above the source byte limit",
            id="model-bytes",
        ),
    ],
)
def test_fixture_construction_is_planned_before_allocation(
    provider: MACEProviderRuntime,
    tmp_path: Path,
    declaration: dict[str, Any],
    limits: MACEConversionLimits,
    refusal: str,
) -> None:
    with pytest.raises(MACESourceRefusedError, match=refusal):
        create_mace_provider_fixture(
            declaration, tmp_path, provider=provider, seed=SEED, limits=limits
        )


def test_extraction_result_refuses_beyond_its_aggregate_limit(
    provider: MACEProviderRuntime, fixtures: dict[str, Any]
) -> None:
    # The result holds every source tensor plus basis evidence and a manifest.
    fixture = fixtures["two-layer-residual-agnesi-zbl"]
    stored = sum(
        size for name, size in _records(fixture.path).items() if name.startswith("data/")
    )
    with pytest.raises(MACESourceRefusedError, match="above the result byte limit"):
        convert_mace_checkpoint(
            _safe(fixture),
            provider=provider,
            limits=MACEConversionLimits(max_result_bytes=stored),
        )


# Scalar one-layer source whose every tensor fits a 10000-element limit.
_TOPOLOGY_DECLARATION = {
    **ONE_LAYER_INVARIANT_READOUT,
    "max_ell": 1,
    "correlation": [2],
}
_TOPOLOGY_LIMITS = MACEConversionLimits(max_tensor_elements=10_000)


def test_neighbor_topology_is_admitted_before_provider_discovery(
    provider: MACEProviderRuntime, tmp_path: Path
) -> None:
    fixture = create_mace_provider_fixture(
        _TOPOLOGY_DECLARATION,
        tmp_path,
        provider=provider,
        seed=SEED,
        limits=_TOPOLOGY_LIMITS,
    )
    source = _safe(fixture)
    # Lawful finite and periodic cases stay admitted under the same limits.
    lawful = evaluate_mace_source(
        source, list(configurations((1, 8))), provider=provider, limits=_TOPOLOGY_LIMITS
    )
    assert [case.edge_index.shape for case in lawful.cases] == [(2, 12), (2, 16)]
    # 100 atoms inside one 3.2 A cutoff: 9900 edges, 29700 shift elements.
    positions = np.random.default_rng(SEED).uniform(0.0, 1.5, (100, 3))
    dense = phx.atomistic.interchange.MACEProviderConfiguration(
        (1, 8) * 50, positions, np.zeros((3, 3)), False
    )
    refusal = r"configuration 0 has at least \d+ neighbor-list edges"
    with pytest.raises(MACESourceRefusedError, match=refusal):
        evaluate_mace_source(source, [dense], provider=provider, limits=_TOPOLOGY_LIMITS)
    with pytest.raises(MACESourceRefusedError, match=refusal):
        mace_source_gradients(
            source,
            [dense],
            [np.ones((100, 3))],
            provider=provider,
            head="Default",
            limits=_TOPOLOGY_LIMITS,
        )
    # A 0.4 A cell needs 21**3 image shifts around a 3.2 A cutoff.
    small_cell = phx.atomistic.interchange.MACEProviderConfiguration(
        (1, 8), np.array([[0.0, 0.0, 0.0], [0.2, 0.2, 0.2]]), np.eye(3) * 0.4, True
    )
    with pytest.raises(MACESourceRefusedError, match="9261 periodic image shifts"):
        evaluate_mace_source(
            source, [small_cell], provider=provider, limits=_TOPOLOGY_LIMITS
        )


_ASL_RELEASE = next(row.sha256 for row in CAMPAIGN_ROWS if row.weight_license == "ASL")


def _example_arguments(checkpoint: Path, *rights: str) -> list[str]:
    return [
        "--provider-venv",
        str(checkpoint.parent / "absent-venv"),
        "--provider-python-version",
        "3.12.8",
        "--torch-version",
        "2.14.1",
        "--mace-torch-version",
        "0.3.16",
        "--checkpoint",
        str(checkpoint),
        "--sha256",
        _ASL_RELEASE,
        "--trust-statement",
        "pinned release admitted by digest",
        *rights,
    ]


class _ProviderReached(Exception):
    pass


def _unreachable_provider(*_: Any) -> None:
    raise _ProviderReached


_MIT_LABEL = ("--license-id", "MIT", "--held-license", "MIT")


@pytest.mark.parametrize(
    ("rights", "refusal"),
    [
        pytest.param((), "explicit --license-id", id="no-license"),
        pytest.param(_MIT_LABEL, "whose weights are ASL", id="mislabeled"),
        pytest.param(("--license-id", "ASL"), "--held-license ASL", id="not-held"),
        pytest.param(
            ("--license-id", "ASL", "--held-license", "MIT"),
            "--held-license ASL",
            id="other-held",
        ),
    ],
)
def test_example_checkpoint_rights_refuse_before_the_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    rights: tuple[str, ...],
    refusal: str,
) -> None:
    # Bytes are never read: the pinned ASL release digest alone decides.
    monkeypatch.setattr(conversion_example, "_provider", _unreachable_provider)
    with pytest.raises(SystemExit, match=refusal):
        conversion_example.main(_example_arguments(tmp_path / "release.model", *rights))


def test_example_admits_explicit_held_rights_and_fixture_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(conversion_example, "_provider", _unreachable_provider)
    explicit = _example_arguments(
        tmp_path / "release.model", "--license-id", "ASL", "--held-license", "ASL"
    )
    assert conversion_example._source_rights(conversion_example._arguments(explicit)) == (
        "ASL",
        ("ASL",),
    )
    with pytest.raises(_ProviderReached):
        conversion_example.main(explicit)
    fixture = explicit[:8]
    assert conversion_example._source_rights(conversion_example._arguments(fixture)) == (
        "MIT",
        ("MIT",),
    )
    with pytest.raises(SystemExit, match="apply to --checkpoint"):
        conversion_example.main([*fixture, "--license-id", "ASL"])


def test_float32_source_buffers_are_realized_at_source_precision(
    provider: MACEProviderRuntime, fixtures: dict[str, Any]
) -> None:
    fixture = fixtures["float32"]
    assert fixture.dtype == "float32"
    source = _safe(fixture)
    conversion = convert_mace_checkpoint(source, provider=provider, head="pbe")
    assert conversion.provenance.source_dtype == "float32"
    tensors = [
        value for value in conversion.source_tensors.values() if value.dtype.kind == "f"
    ]
    assert tensors and all(value.dtype == np.float32 for value in tensors)
    cases = configurations((1, 6, 8))
    # The provider widens the float32 source once; native float64 executes the
    # identical widened parameters and stored coupling buffers.
    widened = evaluate_mace_source(source, cases, provider=provider, head="pbe")
    assert_matches_oracle(conversion.potential, cases, widened, FLOAT64_GATES)
    native32 = dataclasses.replace(conversion, precision=FLOAT32_PRECISION).reconstruct(
        {}
    )
    single = evaluate_mace_source(
        source, cases, provider=provider, head="pbe", evaluation_dtype="float32"
    )
    assert_matches_oracle(native32, cases, single, FLOAT32_GATES)


def test_provenance_identity_is_recomputed_not_trusted(conversions: Conversions) -> None:
    record = conversions["two-layer-density-multihead", "pbe"].provenance.to_record()
    assert (
        MACESourceProvenance.from_record(record).conversion_id == record["conversion_id"]
    )
    for field, value in (
        ("head", "r2scan"),
        ("source_sha256", "0" * 64),
        ("trusted_full_object", True),
    ):
        with pytest.raises(ValueError, match="identity"):
            MACESourceProvenance.from_record({**record, field: value})
    layout = {**record["layout_normalization"], "legacy_head": True}
    with pytest.raises(ValueError, match="identity"):
        MACESourceProvenance.from_record({**record, "layout_normalization": layout})


_NATIVE_WITHOUT_PROVIDER = """
import importlib.abc
import json
import sys


class RefuseProvider(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {"torch", "mace", "e3nn"}:
            raise ImportError(f"provider package {name} imported")
        return None


sys.meta_path.insert(0, RefuseProvider())
from phydrax import config as pxconfig
pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
import numpy as np
from phydrax.atomistic._model_artifact import read_atomistic_model_artifact
from tests._support.mace_source import configurations, native_case

path, revision = sys.argv[1], sys.argv[2]
artifact = read_atomistic_model_artifact(path, numeric_revision_id=revision)
results = [native_case(artifact.model, case) for case in configurations((1, 8))]
print(json.dumps({
    "conversion_id": artifact.manifest.source.conversion_id,
    "licenses": list(artifact.manifest.licenses),
    "energies": [float(item["energy"]) for item in results],
    "stress": np.asarray(results[1]["stress"]).tolist(),
    "provider_modules": sorted(
        name for name in sys.modules if name.split(".")[0] in {"torch", "mace", "e3nn"}
    ),
}))
"""


def test_native_artifact_restores_in_a_process_without_the_provider(
    conversions: Conversions, tmp_path: Path
) -> None:
    conversion = conversions["two-layer-residual-agnesi-zbl", "Default"]
    path = tmp_path / "imported.phydrax"
    manifest = write_atomistic_model_artifact(
        path, conversion.potential, source=conversion.provenance, licenses=["MIT"]
    )
    assert manifest.source == conversion.provenance
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_NATIVE_WITHOUT_PROVIDER),
            str(path),
            manifest.numeric_revision.revision_id,
        ],
        cwd=REPOSITORY,
        env={**os.environ, "PYTHONPATH": str(REPOSITORY)},
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    fresh = json.loads(completed.stdout.strip().splitlines()[-1])
    assert fresh["provider_modules"] == []
    assert fresh["conversion_id"] == conversion.provenance.conversion_id
    assert fresh["licenses"] == ["MIT"]
    local = [native_case(conversion.potential, case) for case in configurations((1, 8))]
    assert fresh["energies"] == [float(item["energy"]) for item in local]
    np.testing.assert_array_equal(fresh["stress"], local[1]["stress"])

    # A recorded source binding that no longer describes the model refuses.
    tampered = tmp_path / "tampered.phydrax"
    recorded, arrays = read_array_archive(path, limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS)
    recorded = json.loads(json.dumps(recorded))
    recorded.pop("arrays")
    recorded["source"]["declaration"]["atomic_numbers"] = [1, 6]
    write_array_archive(
        tampered,
        manifest=recorded,
        limits=ATOMISTIC_MODEL_ARTIFACT_LIMITS,
        arrays={name: np.array(value) for name, value in arrays.items()},
    )
    with pytest.raises(AtomisticModelArtifactError):
        read_atomistic_model_artifact(tampered)


def _foundation_source() -> tuple[Path, str]:
    directory = os.environ.get("PHYDRAX_MACE_FOUNDATION_DIRECTORY")
    if directory is None:
        pytest.skip(
            "declare PHYDRAX_MACE_FOUNDATION_DIRECTORY holding the MIT release "
            "mace_agnesi_small.model; external weights are never bundled"
        )
    return (
        Path(directory) / "mace_agnesi_small.model",
        "7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0",
    )


def test_earlier_release_layout_is_normalized_recorded_and_faithful(
    provider: MACEProviderRuntime,
) -> None:
    path, sha256 = _foundation_source()
    source = admitted_source(
        path,
        "torch-full-model",
        sha256=sha256,
        trust="pinned MIT mace-mp-0b small release admitted by digest",
    )
    conversion = convert_mace_checkpoint(source, provider=provider)
    layout = conversion.provenance.layout_normalization
    assert layout["legacy_head"] and conversion.provenance.head == "Default"
    assert layout["dropped_inert_state"] == [
        "pair_repulsion_fn.cutoff.p",
        "pair_repulsion_fn.cutoff.r_max",
        "pair_repulsion_fn.r_max",
    ]
    assert layout["integral_buffers"] == [
        "pair_repulsion_fn.p",
        "radial_embedding.cutoff_fn.p",
    ]
    water = phx.atomistic.interchange.MACEProviderConfiguration(
        (8, 1, 1),
        np.array([[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.24, 0.927, 0.0]]),
        np.zeros((3, 3)),
        False,
    )
    oracle = evaluate_mace_source(source, [water], provider=provider, head="Default")
    assert_matches_oracle(conversion.potential, [water], oracle, FLOAT64_GATES)


# --- MACE family expansion: release-specific declarations, refusals, families ---


def _declaration_refusal(declaration: Mapping[str, Any], match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _mace_checkpoint._validated_declaration(declaration)


@pytest.mark.parametrize("name", sorted(FAMILY_DECLARATIONS))
@pytest.mark.parametrize("release", ("0.3.15", "0.3.16"))
def test_family_declarations_have_release_specific_exact_fields(
    name: str, release: str
) -> None:
    declaration = FAMILY_DECLARATIONS[name]
    if release == "0.3.15" and declaration["keep_last_layer_irreps"]:
        with pytest.raises(ValueError, match="0.3.15 cannot keep"):
            for_release(declaration, release)
        return
    converted = for_release(declaration, release)
    assert _mace_checkpoint._validated_declaration(converted) == converted
    assert ("keep_last_layer_irreps" in converted) == (release == "0.3.16")
    if release == "0.3.16":
        _declaration_refusal(
            {
                key: value
                for key, value in converted.items()
                if key != "keep_last_layer_irreps"
            },
            "fields are not exact",
        )
    else:
        _declaration_refusal(
            {**converted, "keep_last_layer_irreps": False}, "fields are not exact"
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        pytest.param({"mace_torch_release": "0.3.14"}, "source release", id="release"),
        pytest.param({"model_class": "AtomicDipolesMACE"}, "model class", id="dipoles"),
        pytest.param({"model_class": "PolarMACE"}, "model class", id="polar"),
        pytest.param({"radial_type": "spline"}, "radial_type", id="radial"),
        pytest.param({"distance_transform": "Cosine"}, "transform", id="transform"),
        pytest.param({"use_so3": 1}, "boolean", id="so3-flag"),
        pytest.param({"readout_class": None}, "readout_class", id="readout-missing"),
        pytest.param(
            {"readout_class": "GeneralNonLinearBiasReadoutBlock"},
            "readout_class",
            id="readout-general",
        ),
        pytest.param(
            {"interaction_classes": ["RealAgnosticInteractionBlock", "Other"]},
            "unadmitted interaction",
            id="interaction",
        ),
        pytest.param({"edge_irreps": "3x1x"}, "invalid irreps", id="edge-irreps"),
        pytest.param(
            {"use_embedding_readout": True}, "needs declared embedding", id="readout"
        ),
        pytest.param(
            {
                "embedding_specs": {
                    "temperature": {
                        "type": "categorical",
                        "per": "graph",
                        "num_classes": 2,
                        "emb_dim": 2,
                        "offset": 0,
                    }
                }
            },
            "not admitted",
            id="embedding-name",
        ),
        pytest.param(
            {
                "embedding_specs": {
                    "total_charge": {
                        "type": "continuous",
                        "per": "graph",
                        "num_classes": 2,
                        "emb_dim": 2,
                        "offset": 0,
                    }
                }
            },
            "not canonical",
            id="embedding-continuous",
        ),
    ],
)
def test_declaration_validation_refuses_unadmitted_families(
    changes: dict[str, Any], match: str
) -> None:
    _declaration_refusal({**FAMILY_DECLARATIONS["attention"], **changes}, match)


def test_maceles_declaration_records_every_les_argument() -> None:
    declaration = FAMILY_DECLARATIONS["maceles"]
    assert _mace_checkpoint._validated_declaration(declaration) == declaration
    partial = dict(LES_ARGUMENTS)
    partial.pop("N_max")
    _declaration_refusal({**declaration, "les_arguments": partial}, "every LES argument")
    _declaration_refusal(
        {**declaration, "les_arguments": {**LES_ARGUMENTS, "sigma": float("nan")}},
        "plain finite",
    )
    missing = {key: value for key, value in declaration.items() if key != "les_arguments"}
    _declaration_refusal(missing, "fields are not exact")
    _declaration_refusal(
        {**FAMILY_DECLARATIONS["attention"], "les_arguments": LES_ARGUMENTS},
        "fields are not exact",
    )


def test_runtime_pins_admitted_source_releases(tmp_path: Path) -> None:
    interpreter = pin_executable(sys.executable, version="3", license_id="PSF-2.0")
    runtime = MACEProviderRuntime(
        interpreter,
        str(tmp_path),
        "2.0.0",
        les_version="0.2.0",
        openequivariance_version="0.4.1",
        mace_torch_version="0.3.15",
    )
    assert runtime.distributions == {
        "e3nn": "0.4.4",
        "mace-torch": "0.3.15",
        "torch": "2.0.0",
        "openequivariance": "0.4.1",
        "les": "0.2.0",
    }
    assert _mace_checkpoint.MACE_PROVIDER_RELEASES["mace-torch"] == ("0.3.15", "0.3.16")
    with pytest.raises(ValueError, match="mace_torch_version"):
        MACEProviderRuntime(
            interpreter, str(tmp_path), "2.0.0", mace_torch_version="0.3.14"
        )
    with pytest.raises(ValueError, match="les_version"):
        MACEProviderRuntime(
            interpreter,
            str(tmp_path),
            "2.0.0",
            les_version="0.1.0",
            mace_torch_version="0.3.16",
        )


def _layout(**changes: Any) -> dict[str, Any]:
    return {
        "legacy_head": False,
        "relabelled_reduced_cg": False,
        "derived_zero_flags": [],
        "dropped_inert_state": [],
        "integral_buffers": [],
        "precision_buffers": [],
        "reparameterized_u_buffers": [],
        "optimized_layout": None,
        **changes,
    }


def _provider_identity(release: str, *, les: bool = False) -> dict[str, Any]:
    def record(version: str) -> dict[str, Any]:
        return {"version": version, "record_sha256": "a" * 64, "verified_files": 3}

    distributions = {
        "e3nn": record("0.4.4"),
        "mace-torch": record(release),
        "torch": record("2.0.0"),
    }
    if les:
        distributions["les"] = record("0.2.0")
    return {
        "distributions": distributions,
        "python": "3.12.8",
        "site_packages": "/provider/site-packages",
    }


def _provenance(
    declaration: Mapping[str, Any],
    layout: Mapping[str, Any],
    provider: Mapping[str, Any],
) -> MACESourceProvenance:
    unsigned = MACESourceProvenance(
        source_sha256="b" * 64,
        source_byte_size=4096,
        source_license_id="MIT",
        source_manifest_id="manifest",
        source_admission_id="admission",
        source_kind="torch-full-model",
        trusted_full_object=True,
        source_dtype="float64",
        provider=provider,
        interpreter_sha256="c" * 64,
        interpreter_version="3.12.8",
        declaration=declaration,
        head="Default",
        basis_residual=0.0,
        layout_normalization=layout,
        conversion_id="pending",
    )
    return dataclasses.replace(
        unsigned, conversion_id=_mace_checkpoint._conversion_id(unsigned)
    )


def test_layout_normalization_records_optimized_source_conversions() -> None:
    optimized = {
        "kind": "cuequivariance",
        "converter": "mace.cli.convert_cueq_e3nn",
        "reproduction_deviation": 3.0e-15,
    }
    record = _mace_checkpoint._layout_record(_layout(optimized_layout=optimized))
    assert _mace_checkpoint._normalized_layout(record)
    assert not _mace_checkpoint._normalized_layout(_layout())
    for invalid in (
        {**optimized, "converter": "mace.cli.convert_oeq_e3nn"},
        {**optimized, "kind": "torchscript"},
        {**optimized, "reproduction_deviation": -1.0},
        {key: value for key, value in optimized.items() if key != "converter"},
    ):
        with pytest.raises(ValueError, match="optimized-layout"):
            _mace_checkpoint._layout_record(_layout(optimized_layout=invalid))
    legacy = {key: value for key, value in _layout().items() if key != "optimized_layout"}
    with pytest.raises(ValueError, match="not canonical"):
        _mace_checkpoint._layout_record(legacy)


def test_conversion_identity_changes_with_declaration_and_layout() -> None:
    provider = _provider_identity("0.3.16")
    base = _provenance(FAMILY_DECLARATIONS["attention"], _layout(), provider)
    record = base.to_record()
    assert MACESourceProvenance.from_record(record).conversion_id == base.conversion_id
    identities = {base.conversion_id}
    for declaration in (
        {**FAMILY_DECLARATIONS["attention"], "use_so3": True},
        FAMILY_DECLARATIONS["nonlinear-bias-readout"],
        FAMILY_DECLARATIONS["keep-last-layer-irreps"],
    ):
        identities.add(_provenance(declaration, _layout(), provider).conversion_id)
    optimized = _layout(
        optimized_layout={
            "kind": "openequivariance",
            "converter": "mace.cli.convert_oeq_e3nn",
            "reproduction_deviation": 0.0,
        }
    )
    identities.add(
        _provenance(FAMILY_DECLARATIONS["attention"], optimized, provider).conversion_id
    )
    assert len(identities) == 5
    # A stale binding: the recorded declaration names another provider release.
    stale = _provenance(
        for_release(FAMILY_DECLARATIONS["attention"], "0.3.15"), _layout(), provider
    )
    with pytest.raises(ValueError, match="different mace-torch release"):
        MACESourceProvenance.from_record(stale.to_record())
    maceles = _provenance(FAMILY_DECLARATIONS["maceles"], _layout(), provider)
    with pytest.raises(ValueError, match="verified LES release"):
        MACESourceProvenance.from_record(maceles.to_record())
    with_les = _provenance(
        FAMILY_DECLARATIONS["maceles"], _layout(), _provider_identity("0.3.16", les=True)
    )
    assert MACESourceProvenance.from_record(with_les.to_record()) == with_les
    other_les = _provider_identity("0.3.16", les=True)
    other_les["distributions"]["les"]["version"] = "0.1.0"
    with pytest.raises(ValueError, match="les is not a verified admitted release"):
        _mace_checkpoint._provider_record(other_les)


@pytest.mark.parametrize(
    ("text", "parity", "expected"),
    [
        ("4x0e+4x1o+4x2e", "o3", (4, 2)),
        ("4x0e+4x1e", "so3", (4, 1)),
        ("3x0e", "o3", (3, 0)),
        ("4x0e+4x1e", "o3", None),
        ("4x0e+4x1o", "so3", None),
        ("4x0e+3x1o", "o3", None),
        ("4x0e+4x0o+4x1o", "o3", None),
    ],
)
def test_hidden_and_edge_irreps_must_be_uniform_under_the_parity(
    text: str, parity: Literal["o3", "so3"], expected: tuple[int, int] | None
) -> None:
    if expected is None:
        with pytest.raises(MACESourceRefusedError, match="uniform"):
            _mace_checkpoint._parity_layout(text, "edge_irreps", parity)
    else:
        assert _mace_checkpoint._parity_layout(text, "edge_irreps", parity) == expected


def test_provider_constants_are_verified_against_the_admitted_values() -> None:
    from phydrax.nn.atomistic._mace_source import MACE_SOURCE_LAYER_NORM_EPSILON

    network = {"hs": [16, 8, 9], "layer_norm_epsilons": [MACE_SOURCE_LAYER_NORM_EPSILON]}
    _mace_checkpoint._layer_norm_network(network, [16, 8, 9], "edge network")
    with pytest.raises(MACESourceRefusedError, match="LayerNorm epsilon"):
        _mace_checkpoint._layer_norm_network(
            {**network, "layer_norm_epsilons": [1.0e-6]}, [16, 8, 9], "edge network"
        )
    with pytest.raises(MACESourceRefusedError, match="widths"):
        _mace_checkpoint._layer_norm_network(network, [16, 8, 8], "edge network")
    biased = {
        "irreps_in": "8x0e",
        "irreps_out": "8x0e",
        "instructions": [],
        "biases": [{"i_out": 0, "path_weight": 1.0}],
        "bias_numel": 8,
    }
    _mace_checkpoint._bias_normalization(biased, "linear_mid")
    for invalid in (
        {**biased, "biases": [{"i_out": 0, "path_weight": 0.5}]},
        {**biased, "biases": []},
        {**biased, "bias_numel": 7},
    ):
        with pytest.raises(MACESourceRefusedError, match="bias"):
            _mace_checkpoint._bias_normalization(invalid, "linear_mid")
    declaration = {**FAMILY_DECLARATIONS["gaussian-soft"], "r_max": 3.2, "num_bessel": 6}
    coefficient = -0.5 / (3.2 / 5) ** 2
    gaussian = {"type": "gaussian", "out_dim": 6, "coefficient": coefficient}
    _mace_checkpoint._radial_basis(gaussian, declaration, np.dtype(np.float64))
    with pytest.raises(MACESourceRefusedError, match="Gaussian basis coefficient"):
        _mace_checkpoint._radial_basis(
            {**gaussian, "coefficient": 1.01 * coefficient},
            declaration,
            np.dtype(np.float64),
        )
    with pytest.raises(MACESourceRefusedError, match="contradicts"):
        _mace_checkpoint._radial_basis(
            {"type": "bessel", "out_dim": 6}, declaration, np.dtype(np.float64)
        )


def test_source_inventory_consumes_every_tensor_exactly_once() -> None:
    from phydrax.nn.atomistic._mace_source import _SourceInventory

    inventory = _SourceInventory(
        {"flat": np.arange(6.0), "matrix": np.ones((2, 3)), "extra": np.zeros(2)}
    )
    assert inventory.array("flat", (2, 3)).shape == (2, 3)
    with pytest.raises(RuntimeError, match="consumed twice"):
        inventory.array("flat")
    with pytest.raises(ValueError, match="expected"):
        inventory.exact("matrix", (3, 2))
    with pytest.raises(ValueError, match="missing"):
        inventory.scalar("absent")
    with pytest.raises(ValueError, match=r"Unadmitted source tensors: \['extra'\]"):
        inventory.finish()
    with pytest.raises(ValueError, match="finite"):
        _SourceInventory({"bad": np.array([np.inf])}).array("bad")


def test_source_linear_split_matches_the_e3nn_instruction_order() -> None:
    """Input-major ``[mul_in, mul_out]`` paths; unmatched outputs are absent."""

    from phydrax.nn.atomistic._mace_interaction import (
        mace_irrep_layout,
        MACEPlainLinear,
    )
    from phydrax.nn.atomistic._mace_source import (
        _linear_count,
        _linear_weights,
        _matched_outputs,
    )

    inputs = mace_irrep_layout(2, 0)
    outputs = mace_irrep_layout(3, 2)
    matched = _matched_outputs(inputs, outputs)
    assert [block.degree for block in matched.blocks] == [0]
    count = _linear_count(inputs, outputs)
    assert count == 6
    flat = np.arange(1.0, count + 1.0)
    (block,) = _linear_weights(flat, inputs, matched)
    np.testing.assert_array_equal(block, flat.reshape(2, 3).T)
    plain = MACEPlainLinear(inputs, outputs, weights=(block,), dtype=jnp.float64)
    values = np.array([[0.3, -1.2]])
    result = np.asarray(plain(jnp.asarray(values)))
    # e3nn: out[w] = sum_u x[u] W[u, w] / sqrt(fan_in); higher outputs are zero.
    expected_scalars = values @ flat.reshape(2, 3) / np.sqrt(2.0)
    split = outputs.split(jnp.asarray(result))
    np.testing.assert_allclose(np.asarray(split[0])[..., 0], expected_scalars)
    assert all(not np.any(np.asarray(part)) for part in split[1:])
    # Mixed degrees: instructions run over input blocks first.
    inputs = mace_irrep_layout(2, 1)
    outputs = mace_irrep_layout(3, 1)
    flat = np.arange(1.0, _linear_count(inputs, outputs) + 1.0)
    scalars, vectors = _linear_weights(flat, inputs, outputs)
    np.testing.assert_array_equal(scalars, flat[:6].reshape(2, 3).T)
    np.testing.assert_array_equal(vectors, flat[6:].reshape(2, 3).T)
    with pytest.raises(ValueError, match="do not match"):
        _linear_weights(flat[:-1], inputs, outputs)


_PROVIDER_REFUSALS = """
import json
import runpy
import sys

sys.path.insert(0, sys.argv[1])
import torch
from mace.modules import extensions, models

worker = runpy.run_path(sys.argv[2], run_name="_standalone_mace_worker")
refusals = {}
candidates = {
    "AtomicDipolesMACE": models.AtomicDipolesMACE.__new__(models.AtomicDipolesMACE),
    "AtomicDielectricMACE": models.AtomicDielectricMACE.__new__(
        models.AtomicDielectricMACE
    ),
    "EnergyDipolesMACE": models.EnergyDipolesMACE.__new__(models.EnergyDipolesMACE),
    "PolarMACE": extensions.PolarMACE.__new__(extensions.PolarMACE),
    "TorchScript": torch.jit.script(torch.nn.Linear(2, 2)),
    "Unknown": torch.nn.Linear(2, 2),
}
for name, model in candidates.items():
    try:
        worker["_admitted_root"](model)
    except worker["ProviderRefusal"] as refusal:
        refusals[name] = str(refusal)
print(json.dumps(refusals))
"""


def test_property_models_and_compiled_executables_refuse_in_the_provider(
    provider: MACEProviderRuntime,
) -> None:
    completed = subprocess.run(
        [
            provider.interpreter.path,
            "-I",
            "-c",
            textwrap.dedent(_PROVIDER_REFUSALS),
            provider.site_packages,
            str(REPOSITORY / "phydrax/atomistic/interchange/_mace_worker.py"),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    refusals = json.loads(completed.stdout.strip().splitlines()[-1])
    for name in ("AtomicDipolesMACE", "AtomicDielectricMACE", "EnergyDipolesMACE"):
        assert "not a MACE energy-wrapper route" in refusals[name]
    assert "not a MACE energy-wrapper route" in refusals["PolarMACE"]
    assert "TorchScript" in refusals["TorchScript"]
    assert "not an admitted MACE energy model" in refusals["Unknown"]


def _family_conditions(declaration: Mapping[str, Any]) -> dict[str, int]:
    return (
        {"total_charge": 1, "total_spin": 2}
        if declaration["embedding_specs"] is not None
        else {}
    )


def _assert_family_conversion(
    provider: MACEProviderRuntime,
    declaration: Mapping[str, Any],
    tmp_path: Path,
    kind: MACESourceKind = "torch-state-dict",
) -> MACECheckpointConversion:
    """Convert one provider-built family fixture and compare with the provider."""

    from phydrax.atomistic._latent_electrostatics import LatentEwaldDefinition

    tmp_path.mkdir(parents=True, exist_ok=True)
    fixture = create_mace_provider_fixture(
        declaration, tmp_path, provider=provider, seed=SEED, kind=kind
    )
    assert dict(fixture.declaration) == dict(declaration)
    source = (
        _safe(fixture)
        if kind == "torch-state-dict"
        else admitted_source(
            fixture.path, kind, trust="provider-built fixture of this test"
        )
    )
    cases = configurations(
        tuple(declaration["atomic_numbers"]), **_family_conditions(declaration)
    )
    oracle = evaluate_mace_source(source, cases, provider=provider, head="Default")
    latent = declaration["model_class"] == "MACELES"
    definition = (
        LatentEwaldDefinition.from_source_arguments(declaration["les_arguments"])
        if latent
        else None
    )
    conversion = convert_mace_checkpoint(
        source,
        provider=provider,
        latent_resources=None
        if definition is None
        else latent_resources(definition, cases[0].cell),
        atom_capacity=len(cases[0].numbers) if latent else None,
    )
    assert dict(conversion.provenance.declaration) == dict(declaration)
    assert (
        conversion.provenance.provider["distributions"]["mace-torch"]["version"]
        == declaration["mace_torch_release"]
    )
    for case, reference in zip(cases, oracle.cases, strict=True):
        potential = conversion.potential
        if latent:
            potential = dataclasses.replace(
                conversion,
                atom_capacity=len(case.numbers),
                latent_resources=latent_resources(definition, case.cell),
            ).reconstruct({})
        native = native_case(potential, case)
        assert within(native["energy"], reference.energy, FLOAT64_GATES["energy"]), (
            native["energy"],
            reference.energy,
        )
        assert within(native["forces"], reference.forces, FLOAT64_GATES["forces"])
        if case.periodic:
            assert within(native["stress"], reference.stress, FLOAT64_GATES["stress"])
    return conversion


@pytest.mark.parametrize("name", sorted(FAMILY_DECLARATIONS))
def test_family_source_matches_independent_provider_oracle(
    provider: MACEProviderRuntime, tmp_path: Path, name: str
) -> None:
    declaration = FAMILY_DECLARATIONS[name]
    if declaration["model_class"] == "MACELES":
        les_provider(provider)
    conversion = _assert_family_conversion(provider, declaration, tmp_path)
    architecture = conversion.architecture
    match name:
        case "attention":
            assert set(architecture.interactions) == {"real-agnostic-attention-residual"}
        case "nonlinear-bias-readout":
            assert architecture.readout_variant == "nonlinear-bias"
            assert architecture.gate_scales is not None
        case "gaussian-soft":
            assert (architecture.radial_basis, architecture.distance_transform) == (
                "gaussian",
                "soft",
            )
        case "chebyshev-agnesi":
            assert architecture.radial_basis == "chebyshev"
        case "so3":
            assert architecture.parity == "so3"
        case "edge-irreps-first":
            assert architecture.edge_irreps_first
            assert (
                architecture.edge_channel_count,
                architecture.edge_feature_degree,
            ) == (
                3,
                1,
            )
        case "keep-last-layer-irreps":
            assert architecture.keep_last_layer_irreps
        case "joint-embedding-readout":
            assert architecture.embedding_readout
            assert [item.name for item in architecture.embedding_features] == [
                "total_charge",
                "total_spin",
            ]
        case "maceles":
            assert conversion.atom_capacity == 4
    # Every source tensor is a native leaf or fixed table: rebuilding from the
    # unchanged inventory reproduces the conversion exactly.
    rebuilt = conversion.reconstruct({})
    assert rebuilt.architecture_id == conversion.potential.architecture_id


def test_trusted_full_object_family_converts_like_its_state_dict(
    provider: MACEProviderRuntime, tmp_path: Path
) -> None:
    declaration = FAMILY_DECLARATIONS["nonlinear-bias-readout"]
    safe = _assert_family_conversion(provider, declaration, tmp_path / "safe")
    full = _assert_family_conversion(
        provider, declaration, tmp_path / "full", kind="torch-full-model"
    )
    assert set(full.source_tensors) == set(safe.source_tensors)
    for name, value in safe.source_tensors.items():
        np.testing.assert_array_equal(full.source_tensors[name], value)


@pytest.mark.parametrize(
    "name",
    ["attention", "nonlinear-bias-readout", "gaussian-soft", "so3", "maceles"],
)
def test_release_0_3_15_family_matches_its_provider(tmp_path: Path, name: str) -> None:
    provider = provider_runtime("0.3.15")
    declaration = for_release(FAMILY_DECLARATIONS[name], "0.3.15")
    if declaration["model_class"] == "MACELES":
        les_provider(provider)
    conversion = _assert_family_conversion(provider, declaration, tmp_path)
    assert "keep_last_layer_irreps" not in conversion.provenance.declaration


def test_release_declarations_refuse_on_another_provider_release(
    provider: MACEProviderRuntime, tmp_path: Path
) -> None:
    declaration = for_release(FAMILY_DECLARATIONS["attention"], "0.3.15")
    with pytest.raises(ValueError, match="extracted by mace-torch 0.3.15"):
        create_mace_provider_fixture(declaration, tmp_path, provider=provider, seed=SEED)


@pytest.mark.parametrize(
    ("changes", "refusal"),
    [
        pytest.param(
            {"edge_irreps": "3x0e+2x1o"}, "edge_irreps must be uniform", id="edge"
        ),
        pytest.param(
            {
                "model_class": "MACE",
                "heads": ["pbe", "r2scan"],
                "atomic_energies": [[-13.6, -2041.8], [-13.5, -2042.6]],
                "embedding_specs": CHARGE_SPIN_SPECS,
                "use_embedding_readout": True,
            },
            "broadcasts its embedding readout",
            id="multihead-embedding-readout",
        ),
    ],
)
def test_family_sources_outside_the_native_families_refuse(
    provider: MACEProviderRuntime,
    tmp_path: Path,
    changes: dict[str, Any],
    refusal: str,
) -> None:
    declaration = {**FAMILY_DECLARATIONS["attention"], **changes}
    if declaration["model_class"] == "MACE":
        del declaration["atomic_inter_scale"], declaration["atomic_inter_shift"]
    with pytest.raises(MACESourceRefusedError, match=refusal):
        fixture = create_mace_provider_fixture(
            declaration, tmp_path, provider=provider, seed=SEED
        )
        convert_mace_checkpoint(_safe(fixture), provider=provider)


def test_maceles_conversion_requires_explicit_latent_resources(
    provider: MACEProviderRuntime, tmp_path: Path
) -> None:
    les_provider(provider)
    fixture = create_mace_provider_fixture(
        FAMILY_DECLARATIONS["maceles"], tmp_path, provider=provider, seed=SEED
    )
    with pytest.raises(ValueError, match="required exactly for MACELES"):
        convert_mace_checkpoint(_safe(fixture), provider=provider)


_OPTIMIZED_SOURCE = """
import sys

sys.path.insert(0, sys.argv[1])
import torch
from mace.cli import convert_e3nn_cueq

torch.set_default_dtype(torch.float64)
model = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
converted = convert_e3nn_cueq.run(model, return_model=True)
torch.save(converted, sys.argv[3])
"""


def test_cuequivariance_source_is_expressed_in_e3nn_layout_and_recorded(
    provider: MACEProviderRuntime, tmp_path: Path
) -> None:
    reduced_provider = reduced_cg_provider(provider)
    fixture = create_mace_provider_fixture(
        TWO_LAYER_RESIDUAL_AGNESI_ZBL,
        tmp_path,
        provider=reduced_provider,
        seed=SEED,
        kind="torch-full-model",
    )
    optimized_path = tmp_path / "cueq-source.pt"
    completed = subprocess.run(
        [
            reduced_provider.interpreter.path,
            "-I",
            "-c",
            textwrap.dedent(_OPTIMIZED_SOURCE),
            reduced_provider.site_packages,
            str(fixture.path),
            str(optimized_path),
        ],
        capture_output=True,
        text=True,
        timeout=1800,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    trust = "provider-built fixture converted by the pinned provider in this test"
    plain = convert_mace_checkpoint(
        admitted_source(fixture.path, "torch-full-model", trust=trust),
        provider=reduced_provider,
    )
    optimized = convert_mace_checkpoint(
        admitted_source(optimized_path, "torch-full-model", trust=trust),
        provider=reduced_provider,
    )
    record = optimized.provenance.layout_normalization["optimized_layout"]
    assert record["kind"] == "cuequivariance"
    assert record["converter"] == "mace.cli.convert_cueq_e3nn"
    assert 0.0 <= record["reproduction_deviation"]
    assert plain.provenance.layout_normalization["optimized_layout"] is None
    # A different source layout is a different conversion of the same model.
    assert optimized.provenance.conversion_id != plain.provenance.conversion_id
    cases = configurations((1, 8))
    oracle = evaluate_mace_source(
        admitted_source(fixture.path, "torch-full-model", trust=trust),
        cases,
        provider=reduced_provider,
        head="Default",
    )
    assert_matches_oracle(optimized.potential, cases, oracle, FLOAT64_GATES)


# Committed provider-built family fixtures (generated by
# tests/fixtures/mace_torch_<release>/generate.py with the pinned provider).
_COMMITTED_FAMILY_CASES = {
    "0.3.16": (
        "two_attention",
        "two_nonlinear_bias",
        "two_gaussian_soft",
        "two_chebyshev_agnesi",
        "two_so3",
        "two_edge_irreps_first",
        "two_keep_last_layer",
        "two_joint_embedding",
        "two_maceles",
    ),
    "0.3.15": (
        "two_attention",
        "two_nonlinear_bias",
        "two_gaussian_soft",
        "two_so3",
        "two_joint_embedding",
        "two_maceles",
    ),
}


def _committed_structure(manifest: Mapping[str, Any], name: str) -> Any:
    record = manifest["structures"][name]
    numbers = tuple({"O": 8, "H": 1}[symbol] for symbol in record["symbols"])
    cell = None if record["cell"] is None else np.asarray(record["cell"])
    return phx.atomistic.interchange.MACEProviderConfiguration(
        numbers,
        np.asarray(record["positions"]),
        np.zeros((3, 3)) if cell is None else cell,
        cell is not None,
    )


@pytest.mark.parametrize(
    ("release", "case"),
    [
        (release, case)
        for release, cases in _COMMITTED_FAMILY_CASES.items()
        for case in cases
    ],
)
def test_committed_family_fixture_matches_its_provider_evaluation(
    release: str, case: str
) -> None:
    from phydrax.atomistic._latent_electrostatics import LatentEwaldDefinition

    manifest = committed_manifest(release)
    assert case in manifest["cases"], (
        f"Regenerate tests/fixtures/mace_torch_{release.replace('.', '_')} with "
        "its generate.py and the pinned provider."
    )
    declaration = manifest["cases"][case]
    fixture = np.load(RELEASE_FIXTURES[release] / f"{case}.npz")
    conditions = declaration.get("conditions", {})
    for structure_name in ("molecule", "crystal"):
        configuration = dataclasses.replace(
            _committed_structure(manifest, structure_name),
            total_charge=conditions.get("total_charge"),
            total_spin=conditions.get("total_spin"),
        )
        if declaration["class"] == "MACELES":
            definition = LatentEwaldDefinition.from_source_arguments(
                declaration["les_arguments"]
            )
            potential = committed_maceles_potential(
                case,
                len(configuration.numbers),
                latent_resources(
                    definition, configuration.cell if configuration.periodic else None
                ),
                release,
            )
        else:
            potential = committed_source_potential(case, release)
        native = native_case(potential, configuration)
        prefix = f"{structure_name}/Default/"
        assert within(
            native["energy"], fixture[prefix + "energy"][0], FLOAT64_GATES["energy"]
        )
        assert within(
            native["forces"], fixture[prefix + "forces"], FLOAT64_GATES["forces"]
        )
        if configuration.periodic:
            volume = abs(np.linalg.det(configuration.cell))
            assert within(
                native["stress"] * volume,
                -fixture[prefix + "virials"][0],
                FLOAT64_GATES["stress"],
            )
