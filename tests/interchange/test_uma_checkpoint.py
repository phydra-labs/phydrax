"""UMA source admission: isolated safe extraction and exact reconstruction."""

import copy
import hashlib
import io
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._array_archive import read_array_archive
from phydrax.artifacts import (
    admit_external_artifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
)
from phydrax.atomistic import AtomisticScaleContract
from phydrax.atomistic.interchange import _uma_worker as worker
from phydrax.atomistic.interchange._uma_checkpoint import (
    _admitted_extraction,
    convert_uma_checkpoint,
    UMAConversionLimits,
    UMAExtractionRuntime,
    UMASource,
    UMASourceRefusedError,
)
from phydrax.interchange import pin_executable
from phydrax.nn.atomistic._uma_source import (
    parse_uma_source_config,
    uma_initial_source_tensors,
    uma_model_from_source,
    UMASourceConfigurationError,
)
from phydrax.units import ANGSTROM, ELECTRONVOLT
from tests._support.uma_source import torch_save_bytes, uma_config


phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
SCALE = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
LIMITS = UMAConversionLimits()


def _declaration(**config: Any) -> Any:
    model_config, tasks = uma_config(**config)
    return parse_uma_source_config(model_config, tasks)


def _tensors(declaration: Any) -> dict[str, np.ndarray]:
    return {
        name: value.astype(np.float32)
        for name, value in uma_initial_source_tensors(declaration, key=jr.key(1)).items()
    }


def _run_worker(
    tmp_path: Path, payload: bytes, limits: dict[str, int] | None = None
) -> tuple[dict, dict]:
    (tmp_path / "source.bin").write_bytes(payload)
    request = {
        "operation": "extract",
        "source": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "limits": LIMITS.worker_limits() if limits is None else limits,
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    worker._execute(tmp_path / "request.json", tmp_path / "result.zip")
    return read_array_archive(tmp_path / "result.zip", limits=LIMITS.archive_limits())


def _checkpoint(
    declaration_config: tuple[dict, list], tensors: dict, **options: Any
) -> bytes:
    model_config, tasks = declaration_config
    return torch_save_bytes(
        model_config=model_config,
        tasks_config=tasks,
        ema={f"module.{name}": value for name, value in tensors.items()},
        model=tensors,
        **options,
    )


# --- configuration admission ----------------------------------------------


def test_configuration_admission_records_exclusions_and_named_policies() -> None:
    declaration = _declaration()
    architecture = declaration.architecture
    assert architecture.tasks == ("omat", "omol")
    assert architecture.composition_policy == "mean"
    assert architecture.isolated_atoms == "external-reference-lookup"
    assert "backbone.otf_graph" in declaration.excluded
    assert "backbone.moe_dropout" in declaration.excluded
    assert "tasks_config.omat_energy.loss_fn" in declaration.excluded
    seeded = _declaration(model_id="UMA-S-1.2")
    assert seeded.architecture.composition_policy == "sum-over-count-plus-one"
    assert seeded.architecture.architecture_id != architecture.architecture_id


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda c, t: c["backbone"].update(unknown_switch=True), "unadmitted"),
        (lambda c, t: c["backbone"].update(moe_type="all"), "edge-count"),
        (lambda c, t: c.pop("model_id"), "model_id"),
        (lambda c, t: c["backbone"].update(radius_pbc_version=1), "radius_pbc_version"),
        (lambda c, t: t[1]["normalizer"].update(rmsd=2.0), "denormalized differently"),
        (
            lambda c, t: c["heads"].update(
                spare={"module": "fairchem.core.models.uma.escn_md.Linear_Energy_Head"}
            ),
            "serves no admitted task",
        ),
    ],
)
def test_configuration_outside_the_source_contract_refuses(
    mutation: Any, message: str
) -> None:
    model_config, tasks = uma_config()
    model_config, tasks = copy.deepcopy(model_config), copy.deepcopy(tasks)
    mutation(model_config, tasks)
    with pytest.raises(UMASourceConfigurationError, match=message):
        parse_uma_source_config(model_config, tasks)


# --- exact parameter consumption --------------------------------------------


def test_every_source_tensor_is_consumed_exactly_once() -> None:
    declaration = _declaration()
    tensors = _tensors(declaration)
    potential, residual = uma_model_from_source(SCALE, declaration, tensors)
    assert residual < 1e-6
    potential.validate()
    with pytest.raises(ValueError, match="Unadmitted UMA source tensors"):
        uma_model_from_source(
            SCALE,
            declaration,
            {**tensors, "backbone.extra.weight": np.zeros((2,), np.float32)},
        )
    missing = dict(tensors)
    missing.pop("backbone.routing_mlp.2.bias")
    with pytest.raises(ValueError, match="Missing UMA source tensor"):
        uma_model_from_source(SCALE, declaration, missing)
    reshaped = dict(tensors)
    reshaped["backbone.mix_csd.weight"] = reshaped["backbone.mix_csd.weight"].T.copy()
    with pytest.raises(ValueError, match="shape"):
        uma_model_from_source(SCALE, declaration, reshaped)
    mixture = dict(tensors)
    name = "backbone.blocks.0.edge_wise.so2_conv_1.fc_m0.weights"
    mixture[name.replace(".weights", ".weight")] = mixture.pop(name)[0]
    with pytest.raises(ValueError, match="Missing UMA source tensor"):
        uma_model_from_source(SCALE, declaration, mixture)


def test_stored_basis_evidence_must_match_the_native_relabelling() -> None:
    declaration = _declaration()
    tensors = _tensors(declaration)
    tensors["backbone.Jd_2"] = tensors["backbone.Jd_2"][:, ::-1].copy()
    with pytest.raises(ValueError, match="relabelled native basis"):
        uma_model_from_source(SCALE, declaration, tensors)


def test_source_parameter_directions_map_to_native_parameters() -> None:
    declaration = _declaration()
    tensors = _tensors(declaration)
    first, _ = uma_model_from_source(SCALE, declaration, tensors)
    changed = dict(tensors)
    changed["backbone.sphere_embedding.weight"] = (
        changed["backbone.sphere_embedding.weight"] + 1.0
    )
    second, _ = uma_model_from_source(SCALE, declaration, changed)
    np.testing.assert_allclose(
        second.network.sphere_embedding - first.network.sphere_embedding, 1.0, atol=1e-6
    )


# --- isolated safe extraction ------------------------------------------------


def test_worker_extracts_ema_parameters_configuration_and_inventory(
    tmp_path: Path,
) -> None:
    config = uma_config()
    tensors = {"backbone.a.weight": np.arange(6, dtype=np.float32).reshape(2, 3)}
    manifest, arrays = _run_worker(tmp_path, _checkpoint(config, tensors))
    checkpoint = manifest["checkpoint"]
    assert checkpoint["ema_n_averaged"] == 7
    assert checkpoint["model_state_inventory"] == [["backbone.a.weight", "<f4", [2, 3]]]
    assert checkpoint["model_config"]["backbone"]["sphere_channels"] == 8
    np.testing.assert_array_equal(
        arrays["ema/module.backbone.a.weight"], tensors["backbone.a.weight"]
    )


@pytest.mark.parametrize(
    "options, message",
    [
        ({"record_global": ("os", "system")}, "outside the admitted format"),
        ({"stride_override": {"module.backbone.a.weight": (1, 2)}}, "non-contiguous"),
        ({"compress": True}, "stored"),
    ],
)
def test_worker_refuses_unsafe_or_unbounded_sources(
    tmp_path: Path, options: dict, message: str
) -> None:
    tensors = {"backbone.a.weight": np.zeros((2, 3), dtype=np.float32)}
    with pytest.raises(worker.ProviderRefusal, match=message):
        _run_worker(tmp_path, _checkpoint(uma_config(), tensors, **options))


def test_worker_refuses_a_source_that_differs_from_its_admitted_digest(
    tmp_path: Path,
) -> None:
    payload = _checkpoint(uma_config(), {"backbone.a.weight": np.zeros((1,), np.float32)})
    (tmp_path / "source.bin").write_bytes(payload + b"x")
    request = {
        "operation": "extract",
        "source": {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "byte_size": len(payload),
        },
        "limits": LIMITS.worker_limits(),
    }
    (tmp_path / "request.json").write_text(json.dumps(request))
    with pytest.raises(worker.ProviderRefusal, match="admitted digest"):
        worker._execute(tmp_path / "request.json", tmp_path / "result.zip")


# --- bounded metadata and directory admission ---------------------------------

_RECORD_GLOBAL = (
    b"cfairchem.core.units.mlip_unit.api.inference\nMLIPInferenceCheckpoint\n"
)
_ORDERED_EMPTY = b"ccollections\nOrderedDict\n)R"


def _text(value: str) -> bytes:
    data = value.encode()
    return b"X" + struct.pack("<I", len(data)) + data


def _raw_record(model_config: bytes) -> bytes:
    """A checkpoint pickle with a raw ``model_config`` opcode stream and no tensors."""
    return b"".join(
        (
            b"\x80\x02" + _RECORD_GLOBAL + b")\x81}(",
            _text("model_config") + model_config,
            _text("model_state_dict") + _ORDERED_EMPTY,
            _text("ema_state_dict") + _ORDERED_EMPTY,
            _text("tasks_config") + b"]",
            b"ub.",
        )
    )


def _memo_dag(levels: int) -> bytes:
    """Level k = [level k-1, memo(level k-1)], nested as a pickler emits it.

    Each level is built inside its parent's MARK frame, so only the outermost
    level remains on the stack for ``SETITEM``.
    """
    parts = [b"}" + _text("backbone") + b"](" * levels + b"]" + _text("leaf") + b"aq\x00"]
    for level in range(1, levels + 1):
        parts.append(b"h" + bytes([level - 1]) + b"eq" + bytes([level]))
    return b"".join(parts) + b"s"


def _stored_zip(payload: bytes, members: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", payload)
        archive.writestr("archive/byteorder", b"little")
        for name, data in (members or {}).items():
            archive.writestr(name, data)
        archive.writestr("archive/version", b"3\n")
    return buffer.getvalue()


def _as_zip64(data: bytes, size_delta: int = 0) -> bytes:
    """Rewrite the end of a ZIP as torch's writer does: ZIP64 record + locator."""
    end = len(data) - 22
    _, _, _, _, entries, size, offset, _ = struct.unpack_from("<4s4H2LH", data, end)
    record = struct.pack(
        "<4sQ2H2L4Q",
        b"PK\x06\x06",
        44,
        45,
        45,
        0,
        0,
        entries,
        entries,
        size + size_delta,
        offset,
    )
    locator = struct.pack("<4sLQL", b"PK\x06\x07", 0, end, 1)
    classic = struct.pack(
        "<4s4H2LH", b"PK\x05\x06", 0, 0, 0xFFFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0
    )
    return data[:end] + record + locator + classic


def test_worker_charges_every_expanded_memo_reference(tmp_path: Path) -> None:
    # 2**40 leaves when expanded; the node budget refuses after a bounded prefix.
    payload = _stored_zip(_raw_record(_memo_dag(40)))
    with pytest.raises(worker.ProviderRefusal, match="max_config_nodes"):
        _run_worker(tmp_path, payload)
    small_bytes = {**LIMITS.worker_limits(), "max_manifest_bytes": 4096}
    with pytest.raises(worker.ProviderRefusal, match="max_manifest_bytes"):
        _run_worker(tmp_path, payload, small_bytes)


def test_worker_refuses_cyclic_and_overly_nested_metadata(tmp_path: Path) -> None:
    cycle = b"}" + _text("backbone") + b"]q\x00h\x00as"
    with pytest.raises(worker.ProviderRefusal, match="cyclic metadata"):
        _run_worker(tmp_path, _stored_zip(_raw_record(cycle)))
    nested = b"]"
    for _ in range(LIMITS.max_config_depth + 1):
        nested = b"](" + nested + b"e"
    deep = b"}" + _text("backbone") + nested + b"s"
    with pytest.raises(worker.ProviderRefusal, match="max_config_depth"):
        _run_worker(tmp_path, _stored_zip(_raw_record(deep)))
    with pytest.raises(ValueError, match="max_config_depth"):
        UMAConversionLimits(max_config_depth=513)


@pytest.mark.parametrize(
    "members, override",
    [
        ({f"archive/x{index}": b"" for index in range(40)}, {"max_members": 16}),
        (
            {f"archive/{'n' * 300}{index}": b"" for index in range(8)},
            {"max_manifest_bytes": 1024},
        ),
    ],
)
def test_zip_directory_is_bounded_before_zipfile_parses_it(
    monkeypatch: pytest.MonkeyPatch, members: dict[str, bytes], override: dict[str, int]
) -> None:
    data = _stored_zip(_raw_record(b"}"), members)
    limits = worker._limits({"limits": {**LIMITS.worker_limits(), **override}})

    def parse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("zipfile parsed the directory before its preflight.")

    monkeypatch.setattr(worker.zipfile, "ZipFile", parse)
    with pytest.raises(worker.ProviderRefusal, match="member or byte limits"):
        worker._container(data, limits)


def test_zip64_end_records_are_admitted_only_when_consistent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tensors = {"backbone.a.weight": np.arange(6, dtype=np.float32).reshape(2, 3)}
    payload = _as_zip64(_checkpoint(uma_config(), tensors))
    manifest, arrays = _run_worker(tmp_path, payload)
    assert manifest["checkpoint"]["ema_n_averaged"] == 7
    np.testing.assert_array_equal(
        arrays["ema/module.backbone.a.weight"], tensors["backbone.a.weight"]
    )
    limits = worker._limits({"limits": LIMITS.worker_limits()})
    misplaced = _as_zip64(_stored_zip(_raw_record(b"}")), size_delta=4096)
    truncated = bytearray(_stored_zip(_raw_record(b"}")))
    end = len(truncated) - 22
    struct.pack_into("<H", truncated, end + 10, 2)
    struct.pack_into("<H", truncated, end + 8, 2)

    def parse(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("zipfile parsed the directory before its preflight.")

    # Malformed end records refuse in preflight; zipfile never sees them.
    monkeypatch.setattr(worker.zipfile, "ZipFile", parse)
    for malformed in (misplaced, bytes(truncated)):
        with pytest.raises(worker.ProviderRefusal):
            worker._container(malformed, limits)


def _admitted(tmp_path: Path, payload: bytes) -> UMASource:
    path = tmp_path / "uma.pt"
    path.write_bytes(payload)
    policy = ExternalArtifactPolicy(
        tmp_path,
        maximum_bytes=1 << 30,
        allowed_license_ids=("test-fixture",),
        allowed_suffixes=(".pt",),
    )
    manifest = ArtifactManifest(
        artifact_id="uma",
        producer="fairchem",
        version="bb0a062",
        sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        source_uri=path.resolve().as_uri(),
        license_id="test-fixture",
        model="uma",
        coverage="random test fixture; no pretrained weights",
    )
    return UMASource(
        admit_external_artifact(path.name, manifest, policy=policy), manifest, policy
    )


def test_pinned_isolated_conversion_round_trips_without_source_packages(
    tmp_path: Path,
) -> None:
    config = uma_config()
    declaration = parse_uma_source_config(*config)
    tensors = _tensors(declaration)
    source = _admitted(tmp_path, _checkpoint(config, tensors))
    runtime = UMAExtractionRuntime(
        pin_executable(sys.executable, version="host-python", license_id="PSF-2.0")
    )
    conversion = convert_uma_checkpoint(source, runtime=runtime)
    expected, _ = uma_model_from_source(SCALE, declaration, tensors)
    np.testing.assert_array_equal(
        conversion.model.network.sphere_embedding, expected.network.sphere_embedding
    )
    assert conversion.model.architecture_id == expected.architecture_id
    record = conversion.provenance.to_record()
    assert type(conversion.provenance).from_record(record) == conversion.provenance
    assert "model_state_dict" in conversion.provenance.excluded
    replaced = conversion.reconstruct(
        {
            "backbone.sphere_embedding.weight": np.zeros_like(
                tensors["backbone.sphere_embedding.weight"]
            )
        }
    )
    assert float(jnp.max(jnp.abs(replaced.network.sphere_embedding))) == 0.0
    refused_root = tmp_path / "refused"
    refused_root.mkdir()
    refused = _admitted(
        refused_root, _checkpoint(config, tensors, record_global=("builtins", "eval"))
    )
    with pytest.raises(UMASourceRefusedError, match="outside the admitted format"):
        convert_uma_checkpoint(refused, runtime=runtime)


def test_host_refuses_ema_and_training_inventories_that_disagree(tmp_path: Path) -> None:
    config = uma_config()
    payload = _checkpoint(config, {"backbone.a.weight": np.zeros((2,), np.float32)})
    manifest, arrays = _run_worker(tmp_path, payload)
    manifest["checkpoint"]["model_state_inventory"] = []
    source = _admitted(tmp_path, payload)
    with pytest.raises(ValueError, match="inventory differs"):
        _admitted_extraction(manifest, arrays, source)


def test_moe_single_consumes_every_alias_of_a_shared_layer_exactly() -> None:
    declaration = _declaration(moe_single=True)
    assert declaration.architecture.mixture_sharing
    tensors = _tensors(declaration)
    first = "backbone.blocks.0.edge_wise.so2_conv_1.so2_m_conv.0.fc.weights"
    alias = "backbone.blocks.1.edge_wise.so2_conv_1.so2_m_conv.0.fc.weights"
    # The source state dict names one shared MOLE module under every path.
    np.testing.assert_array_equal(tensors[first], tensors[alias])
    potential, _ = uma_model_from_source(SCALE, declaration, tensors)
    assert len(potential.network.shared_mixtures) == 2
    differing = dict(tensors)
    differing[alias] = differing[alias] + 1.0
    with pytest.raises(ValueError, match="name one shared source layer"):
        uma_model_from_source(SCALE, declaration, differing)
