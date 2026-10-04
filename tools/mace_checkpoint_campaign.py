#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Frozen MACE source-fidelity campaign: provider oracle versus native execution.

Rows are the pinned external releases of the streamed-atomistic-execution plan
(release identifiers and SHA-256 digests are external evidence, never native
schema generations) plus one deterministic provider-built one-interaction
fixture. For every admitted row the driver converts the trusted source,
evaluates the source with its own provider and neighbor list and the native
model with Phydrax, and compares total energy per atom, forces and stress under
gates frozen below before any result is collected. It also compares source
parameter-direction derivatives of the energy and of a force contraction,
mapping source-space directions through the native factory itself.

Nothing is downloaded. Source files are supplied explicitly in ``--sources``
under their release file names. Rows whose weights are not under a license the
caller holds are reported as rights blockers and never executed; full-object
sources run only with ``--trust-pinned-releases`` (executable deserialization
of exactly the pinned digest). Failed rows remain failed with diagnostics.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from phydrax import combine_parameters, partition_parameters
from phydrax.artifacts import (
    admit_external_artifact,
    ArtifactManifest,
    ExternalArtifactPolicy,
)
from phydrax.atomistic import (
    AtomicStructure,
    atomistic_energy_derivatives,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticPrecisionPolicy,
    prepare_atomistic_graph_topology,
)
from phydrax.atomistic.interchange import (
    convert_mace_checkpoint,
    create_mace_provider_fixture,
    evaluate_mace_source,
    mace_source_gradients,
    MACECheckpointConversion,
    MACEProviderConfiguration,
    MACEProviderRuntime,
    MACESource,
    TrustedTorchPickleSource,
)
from phydrax.discretization.particle import ParticleImageCapacity
from phydrax.interchange import ExternalRuntimeError, pin_executable
from phydrax.sparse import StreamedPayloadSpec, StreamedRelationPlan


# Bytes of provider stderr/stdout retained per failed row (the tail, where the
# traceback or refusal is).
_DIAGNOSTIC_BYTES = 16_384


def failure_diagnostic(error: BaseException, /) -> dict[str, Any]:
    """The failure and, for provider runs, its bounded process evidence."""

    record: dict[str, Any] = {"error": f"{type(error).__name__}: {error}"}
    cause = error if isinstance(error, ExternalRuntimeError) else error.__cause__
    if isinstance(cause, ExternalRuntimeError):
        result = cause.result
        record["provider_process"] = {
            "evidence": {key: str(value) for key, value in cause.evidence.items()},
            **(
                {}
                if result is None
                else {
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "elapsed_seconds": result.elapsed_seconds,
                    "stderr_tail": result.stderr[-_DIAGNOSTIC_BYTES:].decode(
                        "utf-8", "replace"
                    ),
                    "stdout_tail": result.stdout[-_DIAGNOSTIC_BYTES:].decode(
                        "utf-8", "replace"
                    ),
                }
            ),
        }
    return record


@dataclass(frozen=True, slots=True)
class CampaignRow:
    """One pinned external release.

    ``weight_license`` is the license of the published weights as determined
    from the release's own terms (the MACE software license does not cover
    every weight file); rows under licenses the operator does not hold are
    rights blockers.
    """

    release: str
    filename: str
    sha256: str
    source_url: str
    head: str | None
    weight_license: str
    structures: tuple[str, ...]


_MP = "https://github.com/ACEsuit/mace-mp/releases/download"
_FOUNDATIONS = "https://github.com/ACEsuit/mace-foundations/releases/download"
_OFF = "https://github.com/ACEsuit/mace-off/raw/main/mace_off23"
_CRYSTALS = ("C64", "STO40")
_MOLECULAR = ("C64", "H2O")

CAMPAIGN_ROWS: tuple[CampaignRow, ...] = (
    CampaignRow(
        "mace-omat-0-small",
        "mace-omat-0-small.model",
        "0abfde07862cf1e93b8b4d03cb702f29ce9c344ff2fc4de2ec0d7166d6c113a5",
        f"{_FOUNDATIONS}/mace_omat_0/mace-omat-0-small.model",
        None,
        "ASL",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-omat-0-medium",
        "mace-omat-0-medium.model",
        "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a",
        f"{_FOUNDATIONS}/mace_omat_0/mace-omat-0-medium.model",
        None,
        "ASL",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-mpa-0-medium",
        "mace-mpa-0-medium.model",
        "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638",
        f"{_MP}/mace_mpa_0/mace-mpa-0-medium.model",
        None,
        "MIT",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-mp-0b-small",
        "mace_agnesi_small.model",
        "7e3a0abcaf41e03a80e69f778e1b11b29de1cca704783dc25917a736392f8cf0",
        f"{_MP}/mace_mp_0b/mace_agnesi_small.model",
        None,
        "MIT",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-mp-0b-medium",
        "mace_agnesi_medium.model",
        "ab8baff639a8f295f3eccad3d3ccf574efb6fb63220bd52cd88664211569e521",
        f"{_MP}/mace_mp_0b/mace_agnesi_medium.model",
        None,
        "MIT",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-mp-0b2-small",
        "mace-small-density-agnesi-stress.model",
        "d5773bf9440e96d6eb8c598f84bd0e6369fcfa432f626a87f890e07da3c651c9",
        f"{_MP}/mace_mp_0b2/mace-small-density-agnesi-stress.model",
        None,
        "MIT",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-mp-0b2-medium",
        "mace-medium-density-agnesi-stress.model",
        "a90be07c8aa6623c390fcc4653d3e319c4a356f8b4a53d323f96b2012d375caf",
        f"{_MP}/mace_mp_0b2/mace-medium-density-agnesi-stress.model",
        None,
        "MIT",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-mp-0b2-large",
        "mace-large-density-agnesi-stress.model",
        "348390e758e1c90011c7675e864850f8e9c5b3e7c217f79a2b4bad1baaa657ad",
        f"{_MP}/mace_mp_0b2/mace-large-density-agnesi-stress.model",
        None,
        "MIT",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-mp-0b3-medium",
        "mace-mp-0b3-medium.model",
        "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012",
        f"{_MP}/mace_mp_0b3/mace-mp-0b3-medium.model",
        None,
        "MIT",
        _CRYSTALS,
    ),
    CampaignRow(
        "mace-off23-small",
        "MACE-OFF23_small.model",
        "165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f",
        f"{_OFF}/MACE-OFF23_small.model",
        None,
        "ASL",
        _MOLECULAR,
    ),
    CampaignRow(
        "mace-off23-medium",
        "MACE-OFF23_medium.model",
        "4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7",
        f"{_OFF}/MACE-OFF23_medium.model",
        None,
        "ASL",
        _MOLECULAR,
    ),
    CampaignRow(
        "mace-off23-large",
        "MACE-OFF23_large.model",
        "a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4",
        f"{_OFF}/MACE-OFF23_large.model",
        None,
        "ASL",
        _MOLECULAR,
    ),
    CampaignRow(
        "mace-mh-0",
        "mace-mh-0.model",
        "d62ff8f293664e6556cfa49364b28ee72a70cf1e6150f3c111a578397fed609d",
        f"{_FOUNDATIONS}/mace_mh_1/mace-mh-0.model",
        "omat_pbe",
        "ASL",
        _CRYSTALS,
    ),
)

# Gates frozen before any campaign result: mixed absolute-relative bounds per
# evaluation dtype and property (eV/atom, eV/A, eV/A^3; relative parameter
# directional derivatives). Never loosened to pass a row.
GATES: Mapping[str, Mapping[str, tuple[float, float]]] = {
    "float64": {
        "energy_per_atom": (1.0e-9, 1.0e-12),
        "forces": (1.0e-8, 1.0e-10),
        "stress": (1.0e-9, 1.0e-10),
        "parameter_direction": (1.0e-10, 1.0e-7),
    },
    "float32": {
        "energy_per_atom": (1.0e-5, 1.0e-6),
        "forces": (1.0e-4, 1.0e-5),
        "stress": (1.0e-5, 1.0e-5),
    },
}

# One-interaction ScaleShiftMACE with residual interaction, Agnesi transform and
# ZBL: the deterministic lawful provider fixture of the installed provider.
FIXTURE_DECLARATION: Mapping[str, Any] = {
    "model_class": "ScaleShiftMACE",
    "r_max": 4.5,
    "num_bessel": 6,
    "num_polynomial_cutoff": 5,
    "max_ell": 2,
    "interaction_classes": ["RealAgnosticResidualInteractionBlock"],
    "num_interactions": 1,
    "atomic_numbers": [1, 8],
    "hidden_irreps": "8x0e",
    "MLP_irreps": None,
    "avg_num_neighbors": 2.5,
    "correlation": [3],
    "gate": None,
    "pair_repulsion": True,
    "distance_transform": "Agnesi",
    "radial_MLP": [16, 16],
    "radial_type": "bessel",
    "heads": ["Default"],
    "apply_cutoff": True,
    "use_reduced_cg": False,
    "use_agnostic_product": False,
    "use_last_readout_only": False,
    "atomic_energies": [-13.6, -2041.8],
    "atomic_inter_scale": 1.3,
    "atomic_inter_shift": 0.05,
}
FIXTURE_SEED = 20261003
_STRUCTURE_SEED = 20260903
_DISPLACEMENT_SIGMA = 0.03
_DIRECTION_SEED = 17


def _shaken(positions: np.ndarray, /) -> np.ndarray:
    generator = np.random.default_rng(_STRUCTURE_SEED)
    return positions + _DISPLACEMENT_SIGMA * generator.standard_normal(positions.shape)


def _supercell(
    basis: np.ndarray, numbers: Sequence[int], lattice: float, /
) -> tuple[list[int], np.ndarray, np.ndarray]:
    shifts = np.array(
        [[i, j, k] for i in range(2) for j in range(2) for k in range(2)],
        dtype=np.float64,
    )
    positions = ((basis[None, :, :] + shifts[:, None, :]) * lattice).reshape(-1, 3)
    return list(numbers) * len(shifts), positions, 2.0 * lattice * np.eye(3)


def structure(name: str, /) -> MACEProviderConfiguration:
    """Independently generated campaign structures from their declared recipes."""

    match name:
        case "C64":
            diamond = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.25, 0.25, 0.25],
                    [0.25, 0.75, 0.75],
                    [0.75, 0.25, 0.75],
                    [0.75, 0.75, 0.25],
                ]
            )
            numbers, positions, cell = _supercell(diamond, [6] * 8, 3.567)
            return MACEProviderConfiguration(
                tuple(numbers), _shaken(positions), cell, True
            )
        case "STO40":
            perovskite = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5],
                ]
            )
            numbers, positions, cell = _supercell(perovskite, [38, 22, 8, 8, 8], 3.905)
            return MACEProviderConfiguration(
                tuple(numbers), _shaken(positions), cell, True
            )
        case "H2O":
            water = np.array(
                [[0.0, 0.0, 0.0], [0.9572, 0.0, 0.0], [-0.239987, 0.927297, 0.0]]
            )
            return MACEProviderConfiguration(
                (8, 1, 1), _shaken(water), np.zeros((3, 3)), False
            )
        case "fixture-OH":
            positions = np.array(
                [[0.0, 0.0, 0.0], [0.97, 0.1, -0.05], [-0.3, 0.92, 0.2], [2.1, 0.4, 0.3]]
            )
            return MACEProviderConfiguration(
                (8, 1, 1, 8), positions, np.zeros((3, 3)), False
            )
        case "fixture-OH-periodic":
            cell = np.array([[3.4, 0.0, 0.0], [0.5, 3.1, 0.0], [0.3, 0.4, 3.6]])
            positions = np.array([[0.1, 0.2, 0.1], [1.0, 0.4, 0.2], [0.3, 1.2, 0.5]])
            return MACEProviderConfiguration((8, 1, 1), positions, cell, True)
        case other:
            raise ValueError(f"Unknown campaign structure {other!r}.")


def _execution(
    configuration: MACEProviderConfiguration, /
) -> AtomisticGraphExecutionPlan:
    atoms = len(configuration.numbers)
    if configuration.periodic:
        return AtomisticGraphExecutionPlan(
            512,
            backend="particle",
            image_capacity=ParticleImageCapacity(
                maximum_particles_per_cell=64,
                maximum_edges=512 * atoms,
                maximum_degree=512,
                maximum_images=343,
            ),
        )
    return AtomisticGraphExecutionPlan(atoms, maximum_dense_atoms=atoms)


def _bound(
    potential: Any, configuration: MACEProviderConfiguration, /
) -> tuple[Any, ...]:
    masses = np.ones(len(configuration.numbers))
    numbers = np.asarray(configuration.numbers, dtype=np.int64)
    if configuration.periodic:
        atomic = AtomicStructure(
            numbers,
            configuration.positions,
            masses,
            potential.scale,
            cell=configuration.cell,
            periodic_axes=np.ones(3, dtype=np.bool_),
            coordinate_dtype=potential.precision.coordinate_dtype,
        )
    else:
        atomic = AtomicStructure(
            numbers,
            configuration.positions,
            masses,
            potential.scale,
            coordinate_dtype=potential.precision.coordinate_dtype,
        )
    batch = AtomisticBatch.from_structure(atomic)
    execution = _execution(configuration)
    topology = prepare_atomistic_graph_topology(
        batch, execution, cutoff=potential.configuration.cutoff
    )
    return batch, execution, topology


def native_case(
    potential: Any, configuration: MACEProviderConfiguration, /
) -> dict[str, np.ndarray]:
    batch, execution, topology = _bound(potential, configuration)
    result = atomistic_energy_derivatives(
        potential,
        batch,
        execution,
        batch.positions,
        topology=topology,
        compute_stress=configuration.periodic,
    )
    atoms = len(configuration.numbers)
    if not bool(np.asarray(result.successful)[0]) or result.forces is None:
        raise RuntimeError("Native evaluation failed or overflowed its capacity.")
    record = {
        "energy": np.asarray(result.energy)[0],
        "forces": np.asarray(result.forces)[0, :atoms],
    }
    if configuration.periodic:
        if result.stress is None:
            raise RuntimeError("Native stress was requested but not returned.")
        record["stress"] = np.asarray(result.stress)[0]
    return record


def _gate(observed: float, reference: float, gate: tuple[float, float], /) -> bool:
    absolute, relative = gate
    return abs(observed - reference) <= absolute + relative * abs(reference)


def _array_gate(
    observed: np.ndarray, reference: np.ndarray, gate: tuple[float, float], /
) -> tuple[float, bool]:
    absolute, relative = gate
    difference = np.abs(observed - reference)
    passed = bool(np.all(difference <= absolute + relative * np.abs(reference)))
    return float(np.max(difference, initial=0.0)), passed


def compare(
    conversion: MACECheckpointConversion,
    source: MACESource,
    provider: MACEProviderRuntime,
    configurations: Mapping[str, MACEProviderConfiguration],
    head: str,
    dtype: str,
    /,
) -> dict[str, Any]:
    """Compare provider and native E/F/S for every configuration in one dtype."""

    match dtype:
        case "float64":
            potential = conversion.potential
        case "float32":
            potential = _with_precision(
                conversion,
                AtomisticPrecisionPolicy(
                    coordinate_dtype="float32",
                    compute_dtype="float32",
                    reduction_dtype="float32",
                    output_dtype="float32",
                ),
            )
        case other:
            raise ValueError(f"Unsupported campaign dtype {other!r}.")
    names = tuple(configurations)
    reference = evaluate_mace_source(
        source,
        [configurations[name] for name in names],
        provider=provider,
        head=head,
        evaluation_dtype=dtype,
    )
    gates = GATES[dtype]
    records: dict[str, Any] = {}
    for name, case in zip(names, reference.cases, strict=True):
        configuration = configurations[name]
        native = native_case(potential, configuration)
        atoms = len(configuration.numbers)
        energy_difference = abs(float(native["energy"]) - case.energy) / atoms
        force_difference, force_pass = _array_gate(
            native["forces"], case.forces, gates["forces"]
        )
        record: dict[str, Any] = {
            "atoms": atoms,
            "provider_edges": int(case.edge_index.shape[1]),
            "energy_per_atom_abs_difference": energy_difference,
            "energy_pass": _gate(
                float(native["energy"]) / atoms,
                case.energy / atoms,
                gates["energy_per_atom"],
            ),
            "forces_max_abs_difference": force_difference,
            "forces_pass": force_pass,
        }
        if configuration.periodic and case.stress is not None:
            stress_difference, stress_pass = _array_gate(
                native["stress"], case.stress, gates["stress"]
            )
            record["stress_max_abs_difference"] = stress_difference
            record["stress_pass"] = stress_pass
        records[name] = record
    return records


def _with_precision(
    conversion: MACECheckpointConversion, precision: AtomisticPrecisionPolicy, /
) -> Any:
    return dataclasses.replace(conversion, precision=precision).reconstruct({})


# Streamed tiles hold at most this many payload elements per edge tile.
_TILE_ELEMENTS = 1 << 23


def streaming_plan(potential: Any, /) -> StreamedRelationPlan:
    """Smallest power-of-two channel capacity admitting every layer payload."""

    from phydrax.nn.atomistic._mace_interaction import MACEStreamedLayer

    width = 0
    for layer in potential.layers:
        message, output = MACEStreamedLayer(
            layer, potential.geometry, potential.energy_reference.head
        ).payload_shapes(potential.embedding.dtype)
        payload = StreamedPayloadSpec(message, output)
        width = max(width, payload.event_elements, payload.output_elements)
    capacity = max(4096, 1 << (width - 1).bit_length())
    return StreamedRelationPlan(
        edge_tile=min(8192, max(256, _TILE_ELEMENTS // capacity)),
        channel_capacity=capacity,
    )


def with_streaming(conversion: MACECheckpointConversion, /) -> MACECheckpointConversion:
    """The conversion rebuilt under its explicit campaign streaming plan."""

    plan = streaming_plan(conversion.potential)
    planned = dataclasses.replace(conversion, streaming=plan)
    return dataclasses.replace(planned, potential=planned.reconstruct({}))


def configuration_record(conversion: MACECheckpointConversion, /) -> dict[str, Any]:
    """Explicit native configuration and the original source parameter map."""

    plan = conversion.streaming
    return {
        "streaming": None
        if plan is None
        else {
            "receiver_tile": plan.receiver_tile,
            "edge_tile": plan.edge_tile,
            "channel_capacity": plan.channel_capacity,
            "accumulation": plan.accumulation,
            "plan_id": plan.plan_id,
        },
        "architecture_id": conversion.potential.architecture_id,
        "source_parameters": {
            name: {"shape": list(value.shape), "dtype": value.dtype.name}
            for name, value in sorted(conversion.source_tensors.items())
        },
    }


def parameter_directions(
    conversion: MACECheckpointConversion,
    source: MACESource,
    provider: MACEProviderRuntime,
    configuration: MACEProviderConfiguration,
    head: str,
    /,
    *,
    directions: int = 3,
    step: float = 1.0e-3,
) -> dict[str, Any]:
    """Compare source-space directional derivatives of E and of sum(F * W).

    Directions are random in the provider's trainable parameter space (original
    W included). Their native images are central differences of the native
    factory output, exact for the factory's linear source-to-native map.
    """

    generator = np.random.default_rng(_DIRECTION_SEED)
    weights = generator.standard_normal((len(configuration.numbers), 3))
    gradients = mace_source_gradients(
        source, [configuration], [weights], provider=provider, head=head
    )
    parameters, model_state, fixed = partition_parameters(conversion.potential)
    batch, execution, topology = _bound(conversion.potential, configuration)
    atoms = len(configuration.numbers)
    weight_array = jnp.asarray(weights)

    def scalars(values: Any, /) -> tuple[jax.Array, jax.Array]:
        potential = combine_parameters(values, model_state, fixed)
        result = atomistic_energy_derivatives(
            potential, batch, execution, batch.positions, topology=topology
        )
        if result.forces is None:
            raise RuntimeError("Native forces were not returned.")
        return result.energy[0], jnp.sum(result.forces[0, :atoms] * weight_array)

    gate = GATES["float64"]["parameter_direction"]
    rows = []
    for _ in range(directions):
        direction = {
            name: generator.standard_normal(gradients.energy_gradients[0][name].shape)
            for name in gradients.parameters
        }
        source_tensors = conversion.source_tensors
        plus = conversion.reconstruct(
            {
                name: source_tensors[name] + step * value
                for name, value in direction.items()
            }
        )
        minus = conversion.reconstruct(
            {
                name: source_tensors[name] - step * value
                for name, value in direction.items()
            }
        )
        native_direction = jax.tree.map(
            lambda high, low: (high - low) / (2.0 * step),
            partition_parameters(plus)[0],
            partition_parameters(minus)[0],
        )
        _, (energy_tangent, force_tangent) = jax.jvp(
            scalars, (parameters,), (native_direction,)
        )
        provider_energy = sum(
            float(np.sum(gradients.energy_gradients[0][name] * value))
            for name, value in direction.items()
        )
        provider_force = sum(
            float(np.sum(gradients.force_gradients[0][name] * value))
            for name, value in direction.items()
        )
        rows.append(
            {
                "energy_native": float(energy_tangent),
                "energy_provider": provider_energy,
                "energy_pass": _gate(float(energy_tangent), provider_energy, gate),
                "force_native": float(force_tangent),
                "force_provider": provider_force,
                "force_pass": _gate(float(force_tangent), provider_force, gate),
            }
        )
    return {"parameters": list(gradients.parameters), "directions": rows}


def _admitted(
    path: Path, sha256: str, license_id: str, release: str, source_url: str, /
) -> tuple[Any, ArtifactManifest, ExternalArtifactPolicy]:
    policy = ExternalArtifactPolicy(
        path.parent,
        maximum_bytes=2_147_483_648,
        allowed_license_ids=[license_id],
        allowed_suffixes=[path.suffix],
    )
    manifest = ArtifactManifest(
        artifact_id=release,
        producer="ACEsuit",
        version=release,
        sha256=sha256,
        byte_size=path.stat().st_size,
        source_uri=source_url,
        license_id=license_id,
        model="mace",
        coverage="mace-fidelity-campaign",
    )
    return admit_external_artifact(path.name, manifest, policy=policy), manifest, policy


def _passed(value: Any, /) -> bool:
    if isinstance(value, Mapping):
        return all(
            _passed(item)
            for key, item in value.items()
            if key.endswith("pass") or isinstance(item, (Mapping, list))
        )
    if isinstance(value, list):
        return all(_passed(item) for item in value)
    return bool(value)


def run_row(
    row: CampaignRow,
    sources: Path,
    provider: MACEProviderRuntime,
    /,
    *,
    held_licenses: frozenset[str],
    trust_pinned: bool,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "release": row.release,
        "sha256": row.sha256,
        "source_url": row.source_url,
        "weight_license": row.weight_license,
        "head": row.head,
    }
    if row.weight_license not in held_licenses:
        return {**record, "status": "blocked-rights"}
    path = sources / row.filename
    if not path.is_file():
        return {**record, "status": "blocked-source-unavailable"}
    if not trust_pinned:
        return {**record, "status": "blocked-untrusted-pickle"}
    artifact, manifest, policy = _admitted(
        path, row.sha256, row.weight_license, row.release, row.source_url
    )
    source = MACESource(
        artifact,
        manifest,
        policy,
        "torch-full-model",
        trust=TrustedTorchPickleSource(
            row.sha256, f"Pinned campaign release {row.release} admitted by digest."
        ),
    )
    try:
        conversion = with_streaming(
            convert_mace_checkpoint(source, provider=provider, head=row.head)
        )
        head = conversion.provenance.head
        configurations = {name: structure(name) for name in row.structures}
        record["provenance"] = conversion.provenance.to_record()
        record["configuration"] = configuration_record(conversion)
        record["float64"] = compare(
            conversion, source, provider, configurations, head, "float64"
        )
        record["float32"] = compare(
            conversion, source, provider, configurations, head, "float32"
        )
        record["parameter_directions"] = parameter_directions(
            conversion,
            source,
            provider,
            configurations["H2O" if "H2O" in row.structures else row.structures[0]],
            head,
        )
    except Exception as error:  # A failed row stays failed with its diagnostic.
        return {**record, "status": "failed", **failure_diagnostic(error)}
    passed = all(
        _passed(record[name]) for name in ("float64", "float32", "parameter_directions")
    )
    record["status"] = "passed" if passed else "failed"
    return record


def run_fixture(provider: MACEProviderRuntime, /, *, reduced_cg: bool) -> dict[str, Any]:
    release = "provider-fixture-one-interaction" + ("-reduced-cg" if reduced_cg else "")
    if reduced_cg and provider.cuequivariance_version is None:
        return {"release": release, "status": "blocked-provider-prerequisite"}
    declaration = {**FIXTURE_DECLARATION, "use_reduced_cg": reduced_cg}
    with tempfile.TemporaryDirectory(prefix="mace-campaign-fixture-") as directory:
        fixture = create_mace_provider_fixture(
            declaration, directory, provider=provider, seed=FIXTURE_SEED
        )
        artifact, manifest, policy = _admitted(
            fixture.path, fixture.sha256, "MIT", "provider-fixture", "provider-generated"
        )
        source = MACESource(
            artifact,
            manifest,
            policy,
            "torch-state-dict",
            architecture=fixture.declaration,
        )
        conversion = with_streaming(convert_mace_checkpoint(source, provider=provider))
        configurations = {
            name: structure(name) for name in ("fixture-OH", "fixture-OH-periodic")
        }
        record = {
            "release": release,
            "fixture_sha256": fixture.sha256,
            "seed": fixture.seed,
            "provenance": conversion.provenance.to_record(),
            "configuration": configuration_record(conversion),
            "float64": compare(
                conversion, source, provider, configurations, "Default", "float64"
            ),
            "parameter_directions": parameter_directions(
                conversion, source, provider, configurations["fixture-OH"], "Default"
            ),
        }
    passed = _passed(record["float64"]) and _passed(record["parameter_directions"])
    record["status"] = "passed" if passed else "failed"
    return record


def main(arguments: Sequence[str] | None = None, /) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--provider-python", type=Path, required=True)
    parser.add_argument("--provider-python-version", required=True)
    parser.add_argument("--provider-site-packages", type=Path, required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--cuequivariance-version")
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--held-license", action="append", default=[])
    parser.add_argument("--trust-pinned-releases", action="store_true")
    parser.add_argument("--rows", nargs="*", default=None)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args(arguments)
    provider = MACEProviderRuntime(
        interpreter=pin_executable(
            options.provider_python,
            version=options.provider_python_version,
            license_id="PSF-2.0",
        ),
        site_packages=str(options.provider_site_packages),
        torch_version=options.torch_version,
        cuequivariance_version=options.cuequivariance_version,
    )
    selected = [
        row
        for row in CAMPAIGN_ROWS
        if options.rows is None or row.release in options.rows
    ]
    report = {
        "gates": GATES,
        "provider": {
            "interpreter": str(provider.interpreter.path),
            "interpreter_sha256": provider.interpreter.sha256,
            "interpreter_version": provider.interpreter.version,
            "site_packages": provider.site_packages,
            "distributions": provider.distributions,
        },
        "fixtures": [
            run_fixture(provider, reduced_cg=False),
            run_fixture(provider, reduced_cg=True),
        ],
        "rows": [
            run_row(
                row,
                options.sources,
                provider,
                held_licenses=frozenset(options.held_license),
                trust_pinned=options.trust_pinned_releases,
            )
            for row in selected
        ],
    }
    options.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False, default=_json)
    )
    return (
        0
        if all(
            row["status"] in ("passed",) or row["status"].startswith("blocked")
            for row in [*report["fixtures"], *report["rows"]]
        )
        else 1
    )


def _json(value: Any, /) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Campaign reports contain only finite values.")
    raise TypeError(f"Unserializable campaign value {type(value).__name__}.")


if __name__ == "__main__":
    raise SystemExit(main())
