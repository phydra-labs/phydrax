"""Trusted mace-torch source conversion against the independent provider oracle."""

import dataclasses
import json
import os
import pickle
import struct
import subprocess
import sys
import textwrap
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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
    convert_mace_checkpoint,
    create_mace_provider_fixture,
    evaluate_mace_source,
    mace_source_gradients,
    MACECheckpointConversion,
    MACEConversionLimits,
    MACEProviderRuntime,
    MACESource,
    MACESourceProvenance,
    MACESourceRefusedError,
    TrustedTorchPickleSource,
)
from tests._support.mace_source import (
    admitted_source,
    assert_matches_oracle,
    configurations,
    DECLARATIONS,
    execution,
    FLOAT32_GATES,
    FLOAT64_GATES,
    native_case,
    ONE_LAYER_INVARIANT_READOUT,
    provider_runtime,
    reduced_cg_provider,
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
    topology = phx.atomistic.prepare_atomistic_graph_topology(
        batch, graph, cutoff=conversion.potential.configuration.cutoff
    )
    contraction = jnp.asarray(weights)

    def scalars(values: Any) -> tuple[jax.Array, jax.Array]:
        potential = phx.combine_parameters(values, state, fixed)
        result = phx.atomistic.atomistic_energy_derivatives(
            potential, batch, graph, batch.positions, topology=topology
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
    fixture = explicit[:6]
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
import jax
jax.config.update("jax_enable_x64", True)
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
