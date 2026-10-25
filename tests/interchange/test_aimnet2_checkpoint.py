import collections
import hashlib
import io
import json
import os
import pickle
import struct
import sys
import types
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import phydrax as phx
from phydrax._external_resource import ResourceLimits
from phydrax._fingerprint import canonical_json
from phydrax.artifacts import (
    admit_external_artifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
)
from phydrax.atomistic import AtomicStructure, AtomisticBatch, AtomisticGraphExecutionPlan
from phydrax.atomistic._conditioning import AtomisticConditioning
from phydrax.atomistic._graph import AtomisticGraphLayout
from phydrax.atomistic._model_inputs import AtomisticGraphInputs
from phydrax.atomistic.interchange._aimnet2_checkpoint import (
    AIMNet2ExtractionLimits,
    AIMNet2OracleConfiguration,
    AIMNet2ProviderRuntime,
    AIMNet2Source,
    AIMNet2SourceProvenance,
    convert_aimnet2_checkpoint,
    evaluate_aimnet2_source,
    extract_aimnet2_source,
    read_safetensors_tensors,
    read_torch_v2_source,
)
from phydrax.atomistic.interchange._source_admission import TrustedTorchPickleSource
from phydrax.interchange import pin_executable
from phydrax.nn.atomistic._aimnet2_source import AIMNet2SourceRefusedError
from tests._support.aimnet2_source import reference_forward, source_inventory


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
NUMBERS = np.asarray([8, 1, 1, 6])
POSITIONS = np.asarray(
    [[0.0, 0.0, 0.0], [0.95, 0.1, 0.0], [-0.3, 0.9, 0.1], [0.4, -0.7, 1.2]]
)
METADATA = {
    "format_version": 2,
    "cutoff": 3.2,
    "needs_coulomb": True,
    "needs_dispersion": True,
    "coulomb_mode": "sr_embedded",
    "coulomb_sr_rc": 2.4,
    "coulomb_sr_envelope": "exp",
    "d3_params": {"s8": 0.3908, "a1": 0.566, "a2": 3.128, "s6": 1.0},
    "has_embedded_lr": True,
    "implemented_species": [1, 6, 7, 8],
}


def _safetensors(tensors: dict[str, np.ndarray], *, dtype: str = "F64") -> bytes:
    header: dict[str, Any] = {}
    chunks = []
    offset = 0
    for name, value in tensors.items():
        data = np.ascontiguousarray(value, dtype="<f8").tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(np.shape(value)),
            "data_offsets": [offset, offset + len(data)],
        }
        chunks.append(data)
        offset += len(data)
    encoded = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded + b"".join(chunks)


_SUFFIXES = (".safetensors", ".json", ".pt", ".jpt", ".yaml")


def _policy(root: Path) -> ExternalArtifactPolicy:
    return ExternalArtifactPolicy(
        root,
        maximum_bytes=1 << 30,
        allowed_license_ids=("test-fixture",),
        allowed_suffixes=_SUFFIXES,
    )


def _admit(path: Path, data: bytes) -> tuple[Any, Any, Any]:
    """Admit one file under the shared policy rooted at its directory."""
    path.write_bytes(data)
    policy = _policy(path.parent)
    manifest = ArtifactManifest(
        artifact_id=path.stem,
        producer="aimnetcentral",
        version="0.2.0",
        sha256=hashlib.sha256(data).hexdigest(),
        byte_size=len(data),
        source_uri=path.resolve().as_uri(),
        license_id="test-fixture",
        model="aimnet2",
        coverage="random test fixture; no pretrained weights",
    )
    return admit_external_artifact(path.name, manifest, policy=policy), manifest, policy


def _hf_source(tmp_path: Path, config: dict, tensors: dict) -> AIMNet2Source:
    artifact, manifest, policy = _admit(
        tmp_path / "ensemble_0.safetensors", _safetensors(tensors)
    )
    config_artifact, config_manifest, _ = _admit(
        tmp_path / "config.json", json.dumps(config).encode("utf-8")
    )
    return AIMNet2Source(
        artifact,
        manifest,
        policy,
        "hf-safetensors",
        declaration=config_artifact,
        declaration_manifest=config_manifest,
    )


def _energy(model: Any, charge: float) -> float:
    structure = AtomicStructure(NUMBERS, POSITIONS, np.ones(4), model.scale)
    batch = AtomisticBatch.from_structure(structure)
    execution = AtomisticGraphExecutionPlan(8, maximum_dense_atoms=8)
    cutoff = model.configuration.cutoff
    graph = (
        AtomisticGraphLayout.for_batch(batch, execution, cutoff=cutoff)
        .prepare(batch)
        .graph(cutoff=cutoff)
    )
    conditioning = AtomisticConditioning(
        model.conditioning_spec,
        {"charge": np.asarray([charge])},
        case_count=1,
        atom_capacity=4,
    )
    inputs = AtomisticGraphInputs(
        graph,
        model.scale,
        species_kind=model.capabilities.species_kind,
        conditioning=conditioning,
    )
    return float(model.evaluate_graph(inputs).energy[0])


def test_hf_safetensors_conversion_is_exact_and_records_composition(
    tmp_path: Path,
) -> None:
    declaration, tensors = source_inventory(21)
    config = {**METADATA, "model_yaml": yaml.safe_dump(declaration, sort_keys=False)}
    conversion = convert_aimnet2_checkpoint(_hf_source(tmp_path, config, tensors))
    reference = reference_forward(declaration, tensors, NUMBERS, POSITIONS, 0.0)
    np.testing.assert_allclose(
        _energy(conversion.potential, 0.0), reference["energy"], rtol=1e-11
    )
    composition = conversion.composition
    assert composition.external_coulomb and not composition.subtract_short_range
    assert composition.external_dispersion
    record = conversion.provenance.to_record()
    assert record["source_format"] == "hf-safetensors"
    assert not record["trusted_executable"]
    assert record["source_pin"]["commit"] == "b1b0fd8c8b8d8acb8811eb646f9ba7b15d72da55"
    assert record["source_license_id"] == "test-fixture"
    assert "outputs.dipole.mass" in record["reconstruction"]["inert_tensors"]
    conversion.potential.validate()


def test_provenance_record_round_trip_is_exact(tmp_path: Path) -> None:
    declaration, tensors = source_inventory(27)
    config = {**METADATA, "model_yaml": yaml.safe_dump(declaration, sort_keys=False)}
    provenance = convert_aimnet2_checkpoint(
        _hf_source(tmp_path, config, tensors)
    ).provenance
    record = json.loads(json.dumps(provenance.to_record()))
    restored = AIMNet2SourceProvenance.from_record(record)
    assert restored.to_record() == provenance.to_record()
    assert restored.conversion_id == provenance.conversion_id
    refusals = {
        "extra field": {**record, "schema": 1},
        "missing field": {
            key: value for key, value in record.items() if key != "metadata"
        },
        "tampered metadata": {
            **record,
            "metadata": {**record["metadata"], "cutoff": 6.0},
        },
        "trust flag": {**record, "trusted_executable": True},
        "source pin": {
            **record,
            "source_pin": {**record["source_pin"], "release": "0.1.0"},
        },
        "digest": {**record, "source_sha256": "A" * 64},
        "provider for safe format": {**record, "provider": {"torch": {}}},
    }
    for reason, value in refusals.items():
        with pytest.raises(ValueError):
            AIMNet2SourceProvenance.from_record(value)
        del reason


def test_config_without_model_yaml_never_falls_back(tmp_path: Path) -> None:
    _, tensors = source_inventory(22)
    with pytest.raises(AIMNet2SourceRefusedError, match="fallback"):
        extract_aimnet2_source(_hf_source(tmp_path, dict(METADATA), tensors))


def test_safetensors_admission_refuses_malformed_buffers() -> None:
    limits = AIMNet2ExtractionLimits()
    good = _safetensors({"a": np.ones(3), "b": np.zeros((2, 2))})
    tensors, metadata = read_safetensors_tensors(good, limits)
    assert tensors["b"].shape == (2, 2) and metadata == {}
    with pytest.raises(AIMNet2SourceRefusedError, match="dtype"):
        read_safetensors_tensors(_safetensors({"a": np.ones(3)}, dtype="F16"), limits)
    with pytest.raises(AIMNet2SourceRefusedError, match="trailing"):
        read_safetensors_tensors(good + b"\0" * 8, limits)
    header = json.dumps(
        {
            "a": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]},
            "b": {"dtype": "F64", "shape": [1], "data_offsets": [0, 8]},
        }
    ).encode()
    overlapping = struct.pack("<Q", len(header)) + header + b"\0" * 8
    with pytest.raises(AIMNet2SourceRefusedError, match="overlap"):
        read_safetensors_tensors(overlapping, limits)
    with pytest.raises(AIMNet2SourceRefusedError, match="tensor limits"):
        read_safetensors_tensors(good, AIMNet2ExtractionLimits(max_tensor_elements=2))


def _fake_torch(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    torch_module = types.ModuleType("torch")
    utils_module = types.ModuleType("torch._utils")
    storage = type("DoubleStorage", (), {"__module__": "torch"})

    def _rebuild_tensor_v2(*arguments: Any) -> None:
        raise AssertionError("never called")

    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    monkeypatch.setattr(torch_module, "DoubleStorage", storage, raising=False)
    monkeypatch.setattr(torch_module, "_utils", utils_module, raising=False)
    monkeypatch.setattr(
        utils_module, "_rebuild_tensor_v2", _rebuild_tensor_v2, raising=False
    )
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch._utils", utils_module)
    return storage, _rebuild_tensor_v2


def _torch_save(
    payload: dict[str, Any],
    tensors: dict[str, np.ndarray],
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra: Any = None,
    views: Mapping[str, tuple[tuple[int, ...], tuple[int, ...]]] | None = None,
) -> bytes:
    """A ``torch.save`` protocol-2 archive written without torch.

    ``views`` adds named ``(shape, stride)`` views of one shared one-element
    float64 storage (zero strides allowed), as torch would record them.
    """
    storage_class, rebuild = _fake_torch(monkeypatch)

    class Storage:
        def __init__(self, key: str, numel: int) -> None:
            self.key, self.numel = key, numel

    class Tensor:
        def __init__(self, key: str, value: np.ndarray) -> None:
            self.storage = Storage(key, value.size)
            self.shape = value.shape
            self.stride = tuple(
                int(np.prod(value.shape[index + 1 :])) for index in range(value.ndim)
            )

        def __reduce__(self) -> Any:
            return (
                rebuild,
                (
                    self.storage,
                    0,
                    self.shape,
                    self.stride,
                    False,
                    collections.OrderedDict(),
                ),
            )

    class Pickler(pickle.Pickler):
        def persistent_id(self, obj: Any) -> Any:
            if isinstance(obj, Storage):
                return ("storage", storage_class, obj.key, "cpu", obj.numel)
            return None

    class View:
        def __init__(self, storage: Any, shape: tuple, stride: tuple) -> None:
            self.storage, self.shape, self.stride = storage, shape, stride

        def __reduce__(self) -> Any:
            return (
                rebuild,
                (
                    self.storage,
                    0,
                    self.shape,
                    self.stride,
                    False,
                    collections.OrderedDict(),
                ),
            )

    state = collections.OrderedDict()
    records = {}
    for index, (name, value) in enumerate(tensors.items()):
        array = np.ascontiguousarray(value, dtype="<f8")
        state[name] = Tensor(str(index), array)
        records[f"archive/data/{index}"] = array.tobytes()
    if views:
        shared = Storage("shared", 1)
        records["archive/data/shared"] = np.zeros(1, dtype="<f8").tobytes()
        for name, (shape, stride) in views.items():
            state[name] = View(shared, shape, stride)
    stream = io.BytesIO()
    document = {**payload, "state_dict": state}
    if extra is not None:
        document["extra"] = extra
    Pickler(stream, protocol=2).dump(document)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", stream.getvalue())
        archive.writestr("archive/byteorder", b"little")
        for name, data in records.items():
            archive.writestr(name, data)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


def test_safetensors_overlap_refuses_before_any_copy() -> None:
    payload = 4096
    header = {
        f"t{index}": {
            "dtype": "F64",
            "shape": [payload // 8],
            "data_offsets": [0, payload],
        }
        for index in range(64)
    }
    encoded = json.dumps(header).encode("utf-8")
    data = struct.pack("<Q", len(encoded)) + encoded + bytes(payload)
    with pytest.raises(AIMNet2SourceRefusedError, match="overlap"):
        read_safetensors_tensors(data, AIMNet2ExtractionLimits())


def test_torch_views_are_admitted_in_aggregate_before_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = AIMNet2ExtractionLimits(max_source_bytes=64 * 1024)
    views = {f"t{index}": ((1024,), (0,)) for index in range(16)}
    data = _torch_save({"cutoff": 5.0}, {}, monkeypatch, views=views)
    # 16 zero-stride 8 KiB views of an 8-byte storage: 128 KiB logical.
    with pytest.raises(AIMNet2SourceRefusedError, match="logical bytes"):
        read_torch_v2_source(data, limits)
    small = {f"t{index}": ((4, 1), (0, 7)) for index in range(2)}
    _, tensors = read_torch_v2_source(
        _torch_save({"cutoff": 5.0}, {}, monkeypatch, views=small), limits
    )
    assert tensors["t0"].shape == (4, 1) and np.all(tensors["t1"] == 0.0)


def test_pickled_metadata_dags_and_cycles_refuse_under_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = AIMNet2ExtractionLimits(
        metadata=ResourceLimits(
            max_bytes=1 << 16,
            max_depth=32,
            max_nodes=4096,
            max_attributes=4096,
            max_losses=0,
        )
    )
    node: Any = [1]
    for _ in range(14):
        node = [node, node]
    with pytest.raises(AIMNet2SourceRefusedError, match="node budget"):
        read_torch_v2_source(
            _torch_save({"cutoff": 5.0}, {}, monkeypatch, extra=node), limits
        )
    cycle: list[Any] = []
    cycle.append(cycle)
    with pytest.raises(AIMNet2SourceRefusedError, match="cycle"):
        read_torch_v2_source(
            _torch_save({"cutoff": 5.0}, {}, monkeypatch, extra=cycle), limits
        )
    deep: Any = 0
    for _ in range(40):
        deep = [deep]
    with pytest.raises(AIMNet2SourceRefusedError, match="depth"):
        read_torch_v2_source(
            _torch_save({"cutoff": 5.0}, {}, monkeypatch, extra=deep), limits
        )
    shared = [list(range(8))] * 4
    fields, _ = read_torch_v2_source(
        _torch_save({"cutoff": 5.0}, {}, monkeypatch, extra=shared), limits
    )
    assert fields["extra"] == [list(range(8))] * 4


def _metadata_limits(max_bytes: int) -> AIMNet2ExtractionLimits:
    return AIMNet2ExtractionLimits(
        metadata=ResourceLimits(
            max_bytes=max_bytes,
            max_depth=32,
            max_nodes=4096,
            max_attributes=4096,
            max_losses=0,
        )
    )


def test_escape_heavy_text_is_charged_at_its_encoded_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = "\x01" * 2000
    data = _torch_save({"cutoff": 5.0}, {}, monkeypatch, extra=text)
    encoded = len(canonical_json({"cutoff": 5.0, "extra": text}))
    cap = 4096
    # The raw UTF-8 text fits the cap; its six-byte escapes do not.
    assert len(text.encode("utf-8")) + 64 < cap < encoded
    with pytest.raises(AIMNet2SourceRefusedError, match="byte budget"):
        read_torch_v2_source(data, _metadata_limits(cap))


def test_metadata_byte_admission_is_exact_canonical_json_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Quotes, backslash, short escapes, C0 control, DEL, Latin-1, CJK,
    # supplementary plane and a lone surrogate; plus integer keys and floats.
    text = 'a"\\\n\t\x01\x7f\u00e9\u6f22\U0001f600\ud800' * 20
    extra = {"text": text, "nested": [text, {"k": -1.5e-7}], "n": 12345678901234567890}
    integer_keyed = {1: 6.5, 6: 46.6}
    data = _torch_save(
        {"cutoff": 5.0, "keys": integer_keyed}, {}, monkeypatch, extra=extra
    )
    expected = {"cutoff": 5.0, "keys": integer_keyed, "extra": extra}
    exact = len(canonical_json(expected))
    fields, _ = read_torch_v2_source(data, _metadata_limits(exact))
    assert fields == expected
    with pytest.raises(AIMNet2SourceRefusedError, match="byte budget"):
        read_torch_v2_source(data, _metadata_limits(exact - 1))


def test_model_yaml_aliases_refuse(tmp_path: Path) -> None:
    declaration, tensors = source_inventory(26)
    text = "base: &w [1, 2]\n" + yaml.safe_dump(declaration, sort_keys=False)
    config = {
        **METADATA,
        "model_yaml": text.replace("hidden:", "hidden_unused: *w\nhidden:", 1),
    }
    with pytest.raises(AIMNet2SourceRefusedError, match="aliases"):
        extract_aimnet2_source(_hf_source(tmp_path, config, tensors))


def test_torch_v2_state_is_read_symbolically_and_converted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, tensors = source_inventory(23, moments=True)
    payload = {**METADATA, "model_yaml": yaml.safe_dump(declaration, sort_keys=False)}
    data = _torch_save(payload, tensors, monkeypatch)
    monkeypatch.delitem(sys.modules, "torch")
    monkeypatch.delitem(sys.modules, "torch._utils")
    artifact, manifest, policy = _admit(tmp_path / "model.pt", data)
    conversion = convert_aimnet2_checkpoint(
        AIMNet2Source(artifact, manifest, policy, "torch-v2-state")
    )
    reference = reference_forward(declaration, tensors, NUMBERS, POSITIONS, -1.0)
    np.testing.assert_allclose(
        _energy(conversion.potential, -1.0), reference["energy"], rtol=1e-11
    )


def test_torch_v2_refuses_executable_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    declaration, tensors = source_inventory(24)
    payload = {**METADATA, "model_yaml": yaml.safe_dump(declaration)}
    data = _torch_save(payload, tensors, monkeypatch, extra=collections.Counter)
    artifact, manifest, policy = _admit(tmp_path / "hostile.pt", data)
    with pytest.raises(AIMNet2SourceRefusedError, match="refused|outside|names"):
        extract_aimnet2_source(
            AIMNet2Source(artifact, manifest, policy, "torch-v2-state")
        )


def test_legacy_torchscript_requires_explicit_digest_trust(tmp_path: Path) -> None:
    artifact, manifest, policy = _admit(tmp_path / "model.jpt", b"PK\x05\x06")
    yaml_artifact, yaml_manifest, _ = _admit(tmp_path / "model.yaml", b"class: x\n")
    with pytest.raises(PermissionError, match="executable"):
        AIMNet2Source(
            artifact,
            manifest,
            policy,
            "torch-legacy-jit",
            declaration=yaml_artifact,
            declaration_manifest=yaml_manifest,
        )
    with pytest.raises(PermissionError, match="different digest"):
        AIMNet2Source(
            artifact,
            manifest,
            policy,
            "torch-legacy-jit",
            declaration=yaml_artifact,
            declaration_manifest=yaml_manifest,
            trust=TrustedTorchPickleSource("0" * 64, "test"),
        )
    with pytest.raises(ValueError, match="never execution-trusted"):
        AIMNet2Source(
            artifact,
            manifest,
            policy,
            "torch-v2-state",
            trust=TrustedTorchPickleSource(artifact.sha256, "test"),
        )


_PROVIDER = ("PHYDRAX_AIMNET2_PROVIDER_PYTHON", "PHYDRAX_AIMNET2_PROVIDER_SITE_PACKAGES")


@pytest.mark.skipif(
    any(name not in os.environ for name in _PROVIDER),
    reason="isolated aimnet==0.2.0 provider not configured",
)
def test_native_matches_isolated_source_oracle(tmp_path: Path) -> None:
    declaration, tensors = source_inventory(25, moments=True)
    config = {**METADATA, "model_yaml": yaml.safe_dump(declaration, sort_keys=False)}
    source = _hf_source(tmp_path, config, tensors)
    provider = AIMNet2ProviderRuntime(
        pin_executable(
            os.environ[_PROVIDER[0]], version="provider", license_id="provider"
        ),
        os.environ[_PROVIDER[1]],
        os.environ.get("PHYDRAX_AIMNET2_PROVIDER_TORCH", ""),
    )
    record, _ = extract_aimnet2_source(source)
    oracle = evaluate_aimnet2_source(
        record,
        (
            AIMNet2OracleConfiguration(
                tuple(int(number) for number in NUMBERS), POSITIONS, 0.0
            ),
        ),
        provider=provider,
    )
    conversion = convert_aimnet2_checkpoint(source)
    np.testing.assert_allclose(
        _energy(conversion.potential, 0.0), oracle.energies[0], rtol=1e-10
    )
