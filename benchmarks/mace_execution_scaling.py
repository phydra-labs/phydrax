# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Phase-separated scaling of MACE's accelerated edge coupling against native JAX.

Operator tier: one radial-weighted ``uvu`` coupling with the native MACE path
enumeration, varied over node count, average degree (edge count), degree skew
into one hub receiver, channels, hidden/edge degree, fragment and program
tiles and precision. Routes share model structure, precision, topology and
requested properties: ``accelerated`` (the prepared streamed relation's
fragments aggregated by the Pallas Mosaic GPU kernels on ``--target``),
``streamed`` (the same prepared relation evaluating the native tensor product
per event) and ``dense`` (the named reference that materializes every per-edge
radial weight, harmonic and message). The bounded routes generate radial
weights and harmonics from lane displacements inside each fragment; the dense
route over the whole graph. Transforms: forward aggregate, energy+forces,
cluster strain derivative, force/strain-loss parameter gradient and
coordinate HVP.

Model tier (``--model-atoms``): a native ``MACEPotential`` over a finite cluster
through the exact model, the trainable model under the accelerated execution
policy (``with_acceleration``), and ``prepare_mace_potential`` with the
streamed or accelerated coupling, varied over correlation order and species
count, for energy+forces, strain derivative and coordinate HVP. Force-loss
parameter gradients run on the exact and trainable accelerated models;
prepared models are inference-only.

Kernel-structure lowering, coefficient-table binding, fragment schedule
preparation and routing checks are timed separately from lowering,
compilation, first and warmed execution. Each executable records XLA
argument/output/temporary/code bytes and the accelerated route's compiler
resource evidence; preparation records the declared per-fragment kernel bytes
(independent of node and edge counts), per-program workspace and logical
retained bytes, plus a sampled resident peak (an observation, never a bound).
Refused or failed configurations are retained. CPU interpreter timings verify
kernels; they are not GPU performance evidence.

``--target`` is required exactly when an accelerated route (``accelerated``,
or the model tier's ``trainable_accelerated``/``accelerated``) is selected.
Native-only runs (``--routes streamed dense``, ``--model-routes exact
streamed``) resolve no accelerator target and execute ordinary JAX on the
selected default backend; their record has ``"target": null``.

``--host-initialization`` constructs model parameters and inputs on CPU once,
then explicitly transfers their dynamic array payloads to the selected execution
device. This is input preparation, not a numerical fallback: all scientific
transforms execute on the selected device. Content fingerprints hash actual
array bytes, dtypes and shapes, not random seeds. Initialization, topology
preparation, route preparation and synchronized transfer are separate from
lowering, compilation, first execution and warmed execution.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._jax_config import default_device, default_device_selection
from phydrax._trainable import combine_parameters, partition_parameters
from phydrax.atomistic import (
    AtomicStructure,
    AtomisticBatch,
    AtomisticGraphExecutionPlan,
    AtomisticGraphInputs,
    AtomisticGraphLayout,
    AtomisticPrecisionPolicy,
    AtomisticScaleContract,
    bind_atomistic_graph,
)
from phydrax.backends._types import BackendUnavailableError
from phydrax.backends.atomistic import (
    AtomisticAccelerationTarget,
    AtomisticKernelPrecision,
)
from phydrax.execution import PhaseMemorySampler
from phydrax.nn.atomistic._mace import MACEArchitecture, MACEPotential
from phydrax.nn.atomistic._mace_interaction import (
    convolution_paths,
    harmonic_layout,
    mace_irrep_layout,
)
from phydrax.nn.atomistic._mace_kernels import (
    fragment_extent,
    MACEAcceleratedCoupling,
    MACEEdgeCouplingSpec,
    MACEKernelPlan,
    require_fragment_routing,
)
from phydrax.nn.atomistic._mace_prepare import (
    prepare_mace_potential,
    PreparedMACEPotential,
)
from phydrax.nn.operator.layers import (
    O3TensorProduct,
    O3TensorProductPath,
    O3TensorProductPlan,
)
from phydrax.nn.operator.representations import O3IrrepLayout
from phydrax.sparse import (
    EdgeRelation,
    KeyGroupAccumulation,
    PreparedStreamedRelation,
    RelationAccumulation,
    StreamedFragment,
    StreamedPayloadSpec,
    StreamedRelationPlan,
)
from phydrax.special import RealCartesianHarmonics
from phydrax.typing import parse
from phydrax.units import ANGSTROM, ELECTRONVOLT


Route: TypeAlias = Literal["accelerated", "streamed", "dense"]
Transform: TypeAlias = Literal[
    "forward", "energy_forces", "strain_derivative", "force_loss_gradient", "hvp"
]
ModelRoute: TypeAlias = Literal[
    "exact", "trainable_accelerated", "streamed", "accelerated"
]
ModelTransform: TypeAlias = Literal[
    "energy_forces", "strain_derivative", "hvp", "force_loss_gradient"
]
_RADIAL_BASIS = 8
_SPECIES = (1, 6, 7, 8, 9, 14, 16, 17)
_SAMPLE_INTERVAL = 0.02
_CUTOFF = 4.0


@dataclass(frozen=True, slots=True)
class KernelBudget:
    """Kernel program tiling and explicit fragment admission shared by all cases."""

    program_tile: int
    reduction_programs: int
    fragment_budget_bytes: int


@dataclass(frozen=True, slots=True)
class CouplingCase:
    """One operator-tier configuration; ``degree`` is the mean routes per node.

    ``receiver_tile`` and ``edge_tile`` are the streamed fragment extents.
    """

    nodes: int
    degree: int
    skew: float
    channels: int
    hidden_degree: int
    edge_degree: int
    precision: AtomisticKernelPrecision
    receiver_tile: int
    edge_tile: int
    channel_tile: int
    seed: int


@dataclass(frozen=True, slots=True)
class ModelCase:
    """One model-tier configuration; tiles are the streamed fragment extents."""

    atoms: int
    channels: int
    correlation: int
    species: int
    interactions: int
    precision: AtomisticKernelPrecision
    receiver_tile: int
    edge_tile: int
    channel_tile: int
    seed: int


def _execution_device() -> jax.Device:
    """Resolve the configured execution placement before entering a CPU scope."""
    selected = default_device_selection()
    if isinstance(selected, jax.Device):
        return selected
    return jax.devices(selected)[0] if isinstance(selected, str) else jax.devices()[0]


def _array_evidence(value: Any, /) -> dict[str, Any]:
    """Hash dynamic array content outside timed phases; report actual storage."""
    digest = hashlib.sha256()
    arrays = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(value)[0]:
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            continue
        host = np.asarray(jax.device_get(leaf))
        name = jax.tree_util.keystr(path)
        descriptor = f"{name}:{host.dtype.str}:{host.shape}:".encode()
        content = host.tobytes(order="C")
        digest.update(descriptor)
        digest.update(content)
        arrays.append(
            {
                "path": name,
                "dtype": str(host.dtype),
                "shape": list(host.shape),
                "bytes": int(host.nbytes),
                "sha256": hashlib.sha256(content).hexdigest(),
                "devices": sorted(str(device) for device in leaf.devices())
                if isinstance(leaf, jax.Array)
                else ["host"],
            }
        )
    return {"sha256": digest.hexdigest(), "arrays": arrays}


def _transfer(value: Any, device: jax.Device, /) -> Any:
    """Transfer dynamic arrays only; retain every dtype and static tree field."""
    transferred: dict[int, Array] = {}

    def transfer_leaf(leaf: Any) -> Any:
        if not isinstance(leaf, (jax.Array, np.ndarray)):
            return leaf
        identifier = id(leaf)
        if identifier not in transferred:
            transferred[identifier] = jax.device_put(leaf, device)
        return transferred[identifier]

    return jax.tree.map(transfer_leaf, value)


def failure_record(error: BaseException, /) -> dict[str, Any]:
    return {
        "status": "refused"
        if isinstance(error, (ValueError, TypeError, BackendUnavailableError))
        else "failed",
        "error_type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exception_only(error),
    }


def _graph(case: CouplingCase, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    edges = case.nodes * case.degree
    hub = round(case.skew * edges)
    receivers = np.concatenate(
        [
            np.zeros(hub, dtype=np.int32),
            rng.integers(0, case.nodes, edges - hub, dtype=np.int32),
        ]
    )
    offsets = rng.integers(1, case.nodes, edges, dtype=np.int32)
    senders = ((receivers + offsets) % case.nodes).astype(np.int32)
    # Box side keeps the mean pair distance comparable across node counts.
    positions = rng.uniform(0.0, 1.6 * case.nodes ** (1.0 / 3.0), size=(case.nodes, 3))
    return senders, receivers, positions


def _tensor_product(case: CouplingCase, dtype: np.dtype, /) -> O3TensorProduct:
    source = mace_irrep_layout(case.channels, case.hidden_degree)
    harmonics = harmonic_layout(case.edge_degree)
    message, paths = convolution_paths(source, harmonics, source)
    plan = O3TensorProductPlan(
        source,
        harmonics,
        message,
        paths=tuple(
            O3TensorProductPath(left, right, output, connection_mode="uvu")
            for left, right, output in paths
        ),
    )
    return O3TensorProduct(plan, internal_weights=False, dtype=dtype)


type _Aggregate = Callable[[O3TensorProduct, Array, Array, Array, Array], Array]
"""``(product, projection, embedding, displacements, active) -> [nodes, width]``."""


def _lane_geometry(
    product: O3TensorProduct, cutoff: float, projection: Array, vector: Array
) -> tuple[Array, Array]:
    """Harmonics and radial path weights of displacement rows ``[lanes, 3]``."""
    harmonics_layout = product.plan.right_representation
    if not isinstance(harmonics_layout, O3IrrepLayout):
        raise TypeError(
            "MACE benchmark harmonics require the declared real-irrep layout."
        )
    distance = jnp.sqrt(jnp.sum(vector * vector, axis=-1))
    inside = distance < cutoff
    safe = jnp.where(inside, distance, cutoff)
    envelope = jnp.where(inside, (1.0 - safe / cutoff) ** 3, 0.0)
    frequency = jnp.arange(1, _RADIAL_BASIS + 1, dtype=vector.dtype)[None, :] * (
        jnp.pi / cutoff
    )
    basis = jnp.sin(frequency * safe[:, None]) / safe[:, None]
    radial = (basis * envelope[:, None]) @ projection
    direction = jnp.where(inside[:, None], vector, jnp.ones_like(vector))
    harmonics = RealCartesianHarmonics(
        harmonics_layout.blocks[-1].degree,
        normalization="fully_normalized",
    )(direction)
    return harmonics, radial


def _identity_epilogue(theta: Any, receiver: Any, aggregate: Array) -> Array:
    del theta, receiver
    return aggregate


def _accelerated_aggregate(
    coupling: MACEAcceleratedCoupling,
    prepared: PreparedStreamedRelation,
    cutoff: float,
    width: int,
    dtype: np.dtype,
) -> _Aggregate:
    payload = StreamedPayloadSpec(
        jax.ShapeDtypeStruct((width,), dtype), jax.ShapeDtypeStruct((width,), dtype)
    )

    def fragment(
        theta: tuple[O3TensorProduct, Array],
        sources: Array,
        receivers: tuple[()],
        rows: Array,
        routing: StreamedFragment,
        seed: KeyGroupAccumulation,
    ) -> KeyGroupAccumulation:
        del receivers
        product, projection = theta
        harmonics, radial = _lane_geometry(product, cutoff, projection, rows)
        return coupling.fragment(
            product, routing, sources[routing.source_ids], harmonics, radial, seed
        )

    def aggregate(
        product: O3TensorProduct,
        projection: Array,
        embedding: Array,
        vectors: Array,
        active: Array,
    ) -> Array:
        result = prepared.evaluate_fragments(
            payload,
            fragment,
            _identity_epilogue,
            (product, projection),
            embedding,
            (),
            vectors,
            edge_active=active,
        )
        return result.receiver_outputs

    return aggregate


def _streamed_aggregate(
    prepared: PreparedStreamedRelation, cutoff: float, width: int, dtype: np.dtype
) -> _Aggregate:
    payload = StreamedPayloadSpec(
        jax.ShapeDtypeStruct((width,), dtype), jax.ShapeDtypeStruct((width,), dtype)
    )

    def edge(
        theta: tuple[O3TensorProduct, Array],
        source: Array,
        receiver: tuple[()],
        row: Array,
    ) -> Array:
        del receiver
        product, projection = theta
        harmonics, radial = _lane_geometry(product, cutoff, projection, row[None])
        return product(source, harmonics[0], radial[0])

    def aggregate(
        product: O3TensorProduct,
        projection: Array,
        embedding: Array,
        vectors: Array,
        active: Array,
    ) -> Array:
        result = prepared.evaluate(
            payload,
            edge,
            _identity_epilogue,
            (product, projection),
            embedding,
            (),
            vectors,
            edge_active=active,
        )
        return result.receiver_outputs

    return aggregate


def _dense_aggregate(
    senders: Array, receivers: Array, nodes: int, cutoff: float
) -> _Aggregate:
    def aggregate(
        product: O3TensorProduct,
        projection: Array,
        embedding: Array,
        vectors: Array,
        active: Array,
    ) -> Array:
        harmonics, radial = _lane_geometry(product, cutoff, projection, vectors)
        message = product(embedding[senders], harmonics, radial)
        message = jnp.where(active[:, None], message, jnp.zeros_like(message))
        return jax.ops.segment_sum(message, receivers, num_segments=nodes)

    return aggregate


def _energy(
    aggregate: _Aggregate,
    product: O3TensorProduct,
    senders: Array,
    receivers: Array,
    cutoff: float,
    coordinates: Array,
    strain: Array,
    parameters: tuple[Array, Array, Array],
) -> Array:
    projection, readout, embedding = parameters
    deformed = coordinates @ (jnp.eye(3, dtype=coordinates.dtype) + strain).T
    vector = deformed[receivers] - deformed[senders]
    active = jnp.sum(vector * vector, axis=-1) < cutoff * cutoff
    values = aggregate(product, projection, embedding, vector, active)
    return jnp.sum(jnp.tanh(values) @ readout)


def _transforms(energy: Callable[..., Array]) -> dict[Transform, Callable[..., Any]]:
    def forward(
        coordinates: Array, strain: Array, parameters: tuple[Array, Array, Array]
    ) -> Array:
        return energy(coordinates, strain, parameters)

    def energy_forces(
        coordinates: Array, strain: Array, parameters: tuple[Array, Array, Array]
    ) -> tuple[Array, Array]:
        value, gradient = jax.value_and_grad(energy)(coordinates, strain, parameters)
        return value, -gradient

    def strain_derivative(
        coordinates: Array, strain: Array, parameters: tuple[Array, Array, Array]
    ) -> Array:
        return jax.grad(energy, argnums=1)(coordinates, strain, parameters)

    def force_loss_gradient(
        coordinates: Array, strain: Array, parameters: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        def loss(values: tuple[Array, Array, Array]) -> Array:
            forces = -jax.grad(energy)(coordinates, strain, values)
            virial = jax.grad(energy, argnums=1)(coordinates, strain, values)
            return jnp.sum(forces * forces) + jnp.sum(virial * virial)

        return jax.grad(loss)(parameters)

    def hvp(
        coordinates: Array, strain: Array, parameters: tuple[Array, Array, Array]
    ) -> Array:
        def gradient(values: Array) -> Array:
            return jax.grad(energy)(values, strain, parameters)

        return jax.jvp(gradient, (coordinates,), (jnp.ones_like(coordinates),))[1]

    return {
        "forward": forward,
        "energy_forces": energy_forces,
        "strain_derivative": strain_derivative,
        "force_loss_gradient": force_loss_gradient,
        "hvp": hvp,
    }


def _measure(
    name: str,
    function: Callable[..., Any],
    arguments: tuple[Any, ...],
    repeats: int,
    coupling: MACEAcceleratedCoupling | None,
) -> dict[str, Any]:
    compiled, timing = measure_lower_and_compile(
        partial(jax.jit(function).lower, *arguments), lambda lowered: lowered.compile()
    )
    output, first = measure_synchronized(partial(compiled, *arguments))
    execution_device = _execution_device()
    output_arrays = [
        leaf for leaf in jax.tree.leaves(output) if isinstance(leaf, jax.Array)
    ]
    if any(leaf.devices() != {execution_device} for leaf in output_arrays):
        raise RuntimeError("Scientific output did not remain on the execution device.")
    devices = tuple(device for device in jax.local_devices() if device.platform != "cpu")
    with PhaseMemorySampler(
        name, interval_seconds=_SAMPLE_INTERVAL, devices=devices
    ) as sampler:
        _, warm = measure_repeated(
            partial(compiled, *arguments), warmup=1, repeats=repeats
        )
    host_outputs = tuple(np.asarray(leaf) for leaf in output_arrays)
    if any(not np.all(np.isfinite(leaf)) for leaf in host_outputs):
        raise ValueError("Measured MACE outputs must be finite.")
    record: dict[str, Any] = {
        "status": "measured",
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "first_seconds": first,
        "warm": warm.to_dict(unit="seconds"),
        "output_devices": sorted(
            {str(device) for leaf in output_arrays for device in leaf.devices()}
        ),
        "output_dtypes": sorted({str(leaf.dtype) for leaf in output_arrays}),
        "output_values": [
            {"dtype": str(leaf.dtype), "shape": list(leaf.shape), "values": leaf.tolist()}
            for leaf in host_outputs
        ],
        "compiler": asdict(
            compiler_evidence(
                compiled.cost_analysis(),
                compiled.memory_analysis(),
                source="xla",
                unavailable_reason="The selected backend did not expose compiler estimates.",
            )
        ),
        "sampled_peak_increase_bytes": sampler.evidence.sampled_peak_increase_bytes,
    }
    if coupling is not None:
        record["resource_evidence"] = coupling.resource_evidence(compiled).to_payload()
    return record


def _coupling(
    target: AtomisticAccelerationTarget | None,
    precision: AtomisticKernelPrecision,
    channel_tile: int,
    budget: KernelBudget,
    accumulation: RelationAccumulation = "deterministic",
) -> MACEAcceleratedCoupling:
    if target is None:
        raise ValueError("Accelerated MACE routes require an explicit --target.")
    return MACEAcceleratedCoupling(
        MACEKernelPlan(
            target=target,
            precision=precision,
            receiver_tile=budget.program_tile,
            edge_tile=budget.program_tile,
            channel_tile=channel_tile,
            reduction_programs=budget.reduction_programs,
            fragment_budget_bytes=budget.fragment_budget_bytes,
            accumulation=accumulation,
        )
    )


def _streamed_relation(
    case: CouplingCase,
    graph: tuple[Array, Array, Array],
    width: int,
    record: dict[str, Any],
) -> PreparedStreamedRelation:
    senders, receivers, valid = graph
    plan = StreamedRelationPlan(
        receiver_tile=case.receiver_tile,
        edge_tile=case.edge_tile,
        channel_capacity=width,
        accumulation="deterministic",
    )
    relation = EdgeRelation(
        senders, receivers, source_size=case.nodes, target_size=case.nodes, valid=valid
    )
    prepared, schedule_seconds = measure_synchronized(
        lambda: plan.prepare(relation, owner_id="mace-benchmark-graph")
    )
    record["preparation"] = {
        "schedule_seconds": schedule_seconds,
        "schedule_bytes": logical_array_bytes(prepared),
        "tile_count": prepared.schedule.tile_count,
    }
    return prepared


def _route_aggregate(
    route: Route,
    case: CouplingCase,
    target: AtomisticAccelerationTarget | None,
    budget: KernelBudget,
    graph: tuple[Array, Array, Array],
    product: O3TensorProduct,
    record: dict[str, Any],
) -> tuple[_Aggregate, MACEAcceleratedCoupling | None]:
    senders, receivers, _ = graph
    dtype = np.dtype(case.precision)
    width = product.plan.output_representation.packed_size
    match route:
        case "accelerated":
            coupling = _coupling(target, case.precision, case.channel_tile, budget)
            prepared = _streamed_relation(case, graph, width, record)
            spec, structure_seconds = measure_host(lambda: MACEEdgeCouplingSpec(product))
            extent = fragment_extent(prepared.fragments())
            admission, admission_seconds = measure_host(
                lambda: coupling.admit(spec, extent)
            )
            _, table_seconds = measure_synchronized(lambda: spec.coefficients(product))
            _, routing_seconds = measure_synchronized(
                lambda: require_fragment_routing(prepared.fragments()).lane_slots
            )
            evidence = coupling.evidence(spec, extent, prepared.binding.binding_id)
            resources = evidence.resources
            record["preparation"].update(
                {
                    "structure_seconds": structure_seconds,
                    "admission_seconds": admission_seconds,
                    "coefficient_table_seconds": table_seconds,
                    "routing_check_seconds": routing_seconds,
                    "fragment_extent": {
                        "receivers": extent.receivers,
                        "sources": extent.sources,
                        "edges": extent.edges,
                    },
                    "declared_fragment_bind_bytes": resources.peak_bind_bytes,
                    "declared_role_bind_bytes": dict(resources.role_bind_bytes),
                    "declared_layout_bytes": resources.layout_bytes,
                    "declared_coefficient_partial_bytes": resources.coefficient_partial_bytes,
                    "declared_workspace_bytes": evidence.workspace_bytes,
                    "declared_workspace_elements": dict(evidence.workspace_elements),
                    "padded_channels": evidence.padded_channels,
                    "admission_id": admission.admission_id,
                    "qualification_scope": admission.qualification_scope,
                    "executable_signature_id": evidence.signature.signature_id,
                    "lowering": coupling.target.lowering,
                    "target_id": coupling.target.target_id,
                }
            )
            return _accelerated_aggregate(
                coupling, prepared, _CUTOFF, width, dtype
            ), coupling
        case "streamed":
            prepared = _streamed_relation(case, graph, width, record)
            return _streamed_aggregate(prepared, _CUTOFF, width, dtype), None
        case "dense":
            record["preparation"] = {
                "scope": "named dense reference materializing per-edge radial, harmonic "
                "and message values"
            }
            return _dense_aggregate(senders, receivers, case.nodes, _CUTOFF), None
        case _:
            raise ValueError(f"Unsupported route {route!r}.")


def _coupling_inputs(
    case: CouplingCase, /
) -> tuple[tuple[Array, Array, Array], O3TensorProduct, tuple[Any, ...]]:
    dtype = np.dtype(case.precision)
    senders_, receivers_, positions_ = _graph(case)
    senders, receivers = jnp.asarray(senders_), jnp.asarray(receivers_)
    graph = (senders, receivers, jnp.ones(senders.shape, dtype=jnp.bool_))
    product = _tensor_product(case, dtype)
    rng = np.random.default_rng(case.seed + 1)
    parameters = (
        jnp.asarray(
            rng.normal(size=(_RADIAL_BASIS, product.plan.parameter_count)) * 0.3,
            dtype=dtype,
        ),
        jnp.asarray(
            rng.normal(size=(product.plan.output_representation.packed_size,)),
            dtype=dtype,
        ),
        jnp.asarray(
            rng.normal(size=(case.nodes, product.plan.left_representation.packed_size)),
            dtype=dtype,
        ),
    )
    arguments = (
        jnp.asarray(positions_, dtype=dtype),
        jnp.zeros((3, 3), dtype=dtype),
        parameters,
    )
    return graph, product, arguments


def coupling_rows(
    case: CouplingCase,
    routes: tuple[Route, ...],
    transforms: tuple[Transform, ...],
    target: AtomisticAccelerationTarget | None,
    budget: KernelBudget,
    repeats: int,
    host_initialization: bool = False,
) -> list[dict[str, Any]]:
    device = _execution_device()
    initialization_device = jax.devices("cpu")[0] if host_initialization else device
    with default_device(initialization_device):
        payload, initialization_seconds = measure_synchronized(
            lambda: _coupling_inputs(case)
        )
    initialized_content = _array_evidence(payload)
    if host_initialization:
        payload, transfer_seconds = measure_synchronized(
            lambda: _transfer(payload, device)
        )
    else:
        transfer_seconds = None
    execution_content = _array_evidence(payload)
    if initialized_content["sha256"] != execution_content["sha256"]:
        raise RuntimeError("Input transfer changed dynamic array content.")
    graph, product, arguments = payload
    senders, receivers, _ = graph
    rows = []
    for route in routes:
        record: dict[str, Any] = {
            "case": asdict(case),
            "route": route,
            "edges": int(senders.shape[0]),
            "initialization": {
                "host_initialization": host_initialization,
                "device": str(initialization_device),
                "execution_device": str(device),
                "initialization_seconds": initialization_seconds,
                "shared_across_routes": True,
                "content": initialized_content,
            },
            "transfer_seconds": transfer_seconds,
            "transfer_shared_across_routes": True,
            "execution_content": execution_content,
        }
        try:
            aggregate, coupling = _route_aggregate(
                route, case, target, budget, graph, product, record
            )
        except (ValueError, TypeError, RuntimeError) as error:
            rows.append({**record, **failure_record(error)})
            continue
        energy = partial(_energy, aggregate, product, senders, receivers, _CUTOFF)
        record["logical_retained_bytes"] = logical_array_bytes(
            (product, arguments, graph)
        )
        record["transforms"] = {}
        for name, function in _transforms(energy).items():
            if name not in transforms:
                continue
            try:
                record["transforms"][name] = _measure(
                    f"{route}-{name}", function, arguments, repeats, coupling
                )
            except (ValueError, TypeError, RuntimeError) as error:
                record["transforms"][name] = failure_record(error)
        rows.append(record)
    return rows


def _model(
    case: ModelCase, /
) -> tuple[MACEPotential, AtomisticBatch, Any, float, tuple[Array, Array]]:
    dtype = case.precision
    architecture = MACEArchitecture(
        species=_SPECIES[: case.species],
        cutoff=3.0,
        radial_basis_count=_RADIAL_BASIS,
        cutoff_power=5,
        channel_count=case.channels,
        hidden_degree=1,
        edge_degree=3,
        interactions=("real-agnostic",)
        + ("real-agnostic-residual",) * (case.interactions - 1),
        correlations=(case.correlation,) * case.interactions,
        radial_widths=(16,),
        readout_width=8 if case.interactions > 1 else None,
        average_neighbor_count=10.0,
    )
    scale = AtomisticScaleContract(ANGSTROM, ELECTRONVOLT)
    model = MACEPotential(
        scale,
        architecture,
        atomic_energies=np.linspace(-1.0, -2.0, case.species),
        precision=AtomisticPrecisionPolicy(
            coordinate_dtype=dtype,
            compute_dtype=dtype,
            reduction_dtype=dtype,
            output_dtype=dtype,
        ),
        key=jax.random.key(case.seed),
    )
    rng = np.random.default_rng(case.seed)
    positions = rng.uniform(0.0, 1.5 * case.atoms ** (1.0 / 3.0), size=(case.atoms, 3))
    numbers = np.asarray(_SPECIES[: case.species], dtype=np.int32)[
        rng.integers(0, case.species, case.atoms)
    ]
    structure = AtomicStructure(
        numbers,
        positions,
        np.ones(case.atoms, dtype=dtype),
        scale,
        coordinate_dtype=dtype,
        mass_accumulation_dtype=dtype,
    )
    batch = AtomisticBatch.from_structure(structure)
    execution = AtomisticGraphExecutionPlan(
        case.atoms - 1,
        maximum_dense_atoms=case.atoms,
        streamed=StreamedRelationPlan(
            receiver_tile=case.receiver_tile,
            edge_tile=case.edge_tile,
            accumulation="deterministic",
        ),
    )
    arguments = (batch.positions, jnp.zeros((3, 3), dtype=dtype))
    return model, batch, execution, 3.0, arguments


def model_rows(
    case: ModelCase,
    routes: tuple[ModelRoute, ...],
    transforms: tuple[ModelTransform, ...],
    target: AtomisticAccelerationTarget | None,
    budget: KernelBudget,
    repeats: int,
    host_initialization: bool = False,
) -> list[dict[str, Any]]:
    device = _execution_device()
    initialization_device = jax.devices("cpu")[0] if host_initialization else device
    with default_device(initialization_device):
        initialized, initialization_seconds = measure_synchronized(lambda: _model(case))
        (
            model,
            source_batch,
            execution,
            source_topology_cutoff,
            source_arguments,
        ) = initialized
        source_topology, topology_seconds = measure_synchronized(
            lambda: (
                AtomisticGraphLayout.for_batch(
                    source_batch, execution, cutoff=source_topology_cutoff
                )
                .prepare(source_batch)
                .topology
            )
        )
    source_evidence = {
        "model": _array_evidence(model),
        "inputs": _array_evidence((source_batch, source_arguments)),
        "topology": _array_evidence(source_topology),
    }
    rows = []
    for route in routes:
        record: dict[str, Any] = {
            "case": asdict(case),
            "route": route,
            "initialization": {
                "host_initialization": host_initialization,
                "device": str(initialization_device),
                "execution_device": str(device),
                "initialization_seconds": initialization_seconds,
                "topology_seconds": topology_seconds,
                "shared_across_routes": True,
                "content": source_evidence,
            },
        }
        coupling = None
        try:

            def prepare_route() -> MACEPotential | PreparedMACEPotential:
                nonlocal coupling
                match route:
                    case "exact":
                        return model
                    case "trainable_accelerated":
                        coupling = _coupling(
                            target, case.precision, case.channel_tile, budget
                        )
                        return model.with_acceleration(coupling)
                    case "streamed":
                        return prepare_mace_potential(model)
                    case "accelerated":
                        coupling = _coupling(
                            target, case.precision, case.channel_tile, budget
                        )
                        return prepare_mace_potential(model, edge_coupling=coupling)
                    case _:
                        raise ValueError(f"Unsupported model route {route!r}.")

            with default_device(initialization_device):
                potential, seconds = measure_synchronized(prepare_route)
            record["preparation_seconds"] = seconds
            if coupling is not None:
                record["lowering"] = coupling.target.lowering
            payload = (potential, source_batch, source_topology, source_arguments)
            record["prepared_content"] = _array_evidence(payload)
            if host_initialization:
                payload, transfer_seconds = measure_synchronized(
                    lambda: _transfer(payload, device)
                )
            else:
                transfer_seconds = None
            potential, batch, topology, arguments = payload
            cutoff = source_topology_cutoff
            record["transfer_seconds"] = transfer_seconds
            record["execution_content"] = _array_evidence(payload)
            if (
                record["prepared_content"]["sha256"]
                != record["execution_content"]["sha256"]
            ):
                raise RuntimeError("Input transfer changed dynamic array content.")
        except (ValueError, TypeError, RuntimeError) as error:
            rows.append({**record, **failure_record(error)})
            continue

        def energy(
            values: Array,
            strain: Array,
            current: MACEPotential | PreparedMACEPotential = potential,
        ) -> Array:
            deformed = values @ (jnp.eye(3, dtype=values.dtype) + strain).T
            graph = bind_atomistic_graph(topology, execution, deformed, cutoff=cutoff)
            result = current.evaluate_graph(
                AtomisticGraphInputs(
                    graph, batch.scale, species_kind=current.capabilities.species_kind
                )
            )
            return jnp.sum(result.energy)

        functions: dict[ModelTransform, Callable[..., Any]] = {
            "energy_forces": lambda values, strain: jax.value_and_grad(energy)(
                values, strain
            ),
            "strain_derivative": lambda values, strain: jax.grad(energy, argnums=1)(
                values, strain
            ),
            "hvp": lambda values, strain: jax.jvp(
                lambda inner: jax.grad(energy)(inner, strain),
                (values,),
                (jnp.ones_like(values),),
            )[1],
        }
        # Arguments were constructed and synchronized in the initialization phase.
        record["logical_retained_bytes"] = logical_array_bytes(
            (potential, batch, topology)
        )
        record["transforms"] = {}
        for name in transforms:
            if name == "force_loss_gradient":
                continue
            try:
                record["transforms"][name] = _measure(
                    f"model-{route}-{name}", functions[name], arguments, repeats, coupling
                )
            except (ValueError, TypeError, RuntimeError) as error:
                record["transforms"][name] = failure_record(error)
        if (
            route in ("exact", "trainable_accelerated")
            and "force_loss_gradient" in transforms
        ):
            if not isinstance(potential, MACEPotential):
                raise RuntimeError("A trainable benchmark route produced a frozen model.")
            record["transforms"]["force_loss_gradient"] = _model_force_loss(
                potential,
                route,
                batch,
                execution,
                topology,
                cutoff,
                arguments,
                repeats,
                coupling,
            )
        rows.append(record)
    return rows


def _model_force_loss(
    model: MACEPotential,
    route: ModelRoute,
    batch: AtomisticBatch,
    execution: Any,
    topology: Any,
    cutoff: float,
    arguments: tuple[Array, Array],
    repeats: int,
    coupling: MACEAcceleratedCoupling | None,
) -> dict[str, Any]:
    """Force-loss gradient in the trainable model's original parameters."""
    parameters, model_state, fixed = partition_parameters(model)

    def loss(values: Any, positions: Array, strain: Array) -> Array:
        current = combine_parameters(values, model_state, fixed)

        def energy(coordinates: Array) -> Array:
            deformed = coordinates @ (jnp.eye(3, dtype=coordinates.dtype) + strain).T
            graph = bind_atomistic_graph(topology, execution, deformed, cutoff=cutoff)
            result = current.evaluate_graph(
                AtomisticGraphInputs(
                    graph, batch.scale, species_kind=current.capabilities.species_kind
                )
            )
            return jnp.sum(result.energy)

        forces = -jax.grad(energy)(positions)
        return jnp.sum(forces * forces)

    try:
        return _measure(
            f"model-{route}-force-loss",
            jax.grad(loss),
            (parameters, *arguments),
            repeats,
            coupling,
        )
    except (ValueError, TypeError, RuntimeError) as error:
        return failure_record(error)


def _parse_all(values: list[str], form: Any, /) -> tuple[Any, ...]:
    return tuple(parse(value, form, "selector") for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=get_args(AtomisticAccelerationTarget),
        default=None,
        help="accelerator target; required only for accelerated routes",
    )
    parser.add_argument("--nodes", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--degrees", type=int, nargs="+", default=[8])
    parser.add_argument("--skews", type=float, nargs="+", default=[0.0, 0.25])
    parser.add_argument("--channels", type=int, nargs="+", default=[8])
    parser.add_argument("--hidden-degrees", type=int, nargs="+", default=[1])
    parser.add_argument("--edge-degrees", type=int, nargs="+", default=[2, 3])
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=get_args(AtomisticKernelPrecision),
        default=["float64"],
    )
    parser.add_argument(
        "--receiver-tiles",
        type=int,
        nargs="+",
        default=[4],
        help="fragment receiver slots",
    )
    parser.add_argument(
        "--edge-tiles", type=int, nargs="+", default=[32], help="fragment lanes"
    )
    parser.add_argument("--channel-tiles", type=int, nargs="+", default=[128])
    parser.add_argument(
        "--program-tile", type=int, default=4, help="kernel owner/lane rows per program"
    )
    parser.add_argument("--reduction-programs", type=int, default=32)
    parser.add_argument("--fragment-budget-bytes", type=int, default=1 << 30)
    parser.add_argument(
        "--routes", nargs="+", choices=get_args(Route), default=list(get_args(Route))
    )
    parser.add_argument(
        "--transforms",
        nargs="+",
        choices=get_args(Transform),
        default=list(get_args(Transform)),
    )
    parser.add_argument("--model-atoms", type=int, nargs="*", default=[])
    parser.add_argument("--model-correlations", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--model-species", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--model-channels", type=int, nargs="+", default=[8])
    parser.add_argument("--model-interactions", type=int, default=2)
    parser.add_argument(
        "--model-routes",
        nargs="+",
        choices=get_args(ModelRoute),
        default=list(get_args(ModelRoute)),
    )
    parser.add_argument(
        "--model-transforms",
        nargs="+",
        choices=get_args(ModelTransform),
        default=list(get_args(ModelTransform)),
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--host-initialization",
        action="store_true",
        help="initialize on CPU, then explicitly transfer inputs to the execution device",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 2 for value in args.nodes):
        raise ValueError("repeats must be positive and node counts at least two.")
    routes = _parse_all(args.routes, Route)
    transforms = _parse_all(args.transforms, Transform)
    model_routes = _parse_all(args.model_routes, ModelRoute)
    model_transforms = _parse_all(args.model_transforms, ModelTransform)
    accelerated_model_routes = {"trainable_accelerated", "accelerated"}
    accelerated = "accelerated" in routes or (
        bool(args.model_atoms) and not accelerated_model_routes.isdisjoint(model_routes)
    )
    if accelerated and args.target is None:
        parser.error("--target is required when an accelerated route is selected.")
    target = (
        None
        if args.target is None
        else parse(args.target, AtomisticAccelerationTarget, "target")
    )
    budget = KernelBudget(
        args.program_tile, args.reduction_programs, args.fragment_budget_bytes
    )
    coupling_cases = [
        CouplingCase(n, d, s, c, h, e, p, r, t, k, args.seed)
        for n, d, s, c, h, e, p, r, t, k in itertools.product(
            args.nodes,
            args.degrees,
            args.skews,
            args.channels,
            args.hidden_degrees,
            args.edge_degrees,
            _parse_all(args.precisions, AtomisticKernelPrecision),
            args.receiver_tiles,
            args.edge_tiles,
            args.channel_tiles,
        )
    ]
    model_cases = [
        ModelCase(a, c, r, s, args.model_interactions, p, rt, et, k, args.seed)
        for a, c, r, s, p, rt, et, k in itertools.product(
            args.model_atoms,
            args.model_channels,
            args.model_correlations,
            args.model_species,
            _parse_all(args.precisions, AtomisticKernelPrecision),
            args.receiver_tiles,
            args.edge_tiles,
            args.channel_tiles,
        )
    ]
    root = Path(__file__).resolve().parents[1]
    owners = (
        "phydrax/backends/atomistic.py",
        "phydrax/nn/atomistic/_mace_kernels.py",
        "phydrax/nn/atomistic/_mace_kernel_derivatives.py",
        "phydrax/nn/atomistic/_mace.py",
        "phydrax/nn/atomistic/_mace_prepare.py",
        "phydrax/sparse/_streamed.py",
    )
    record = {
        "benchmark": "mace-execution-scaling",
        "label": args.label,
        "target": target,
        "host_initialization": args.host_initialization,
        "scope": {
            "timing": "synchronized host wall time per phase",
            "compiler": "XLA executable estimates; kernel registers and shared memory excluded",
            "sampled": "PhaseMemorySampler sampled resident peak above phase baseline; never a bound",
            "workspace": "declared per-program planning bound of the accelerated kernels",
            "fragment": "declared per-fragment kernel operand/result bytes; no node/edge scaling",
            "interpret": "cpu_interpret verifies kernel semantics; not GPU performance evidence",
            "initialization": "shared constructor/input and topology phases are not per-route costs",
            "transfer": "explicit dynamic-array device_put plus synchronization; null when not requested",
            "content": "SHA-256 of actual dynamic-array paths, dtype, shape and C-order bytes; device-independent",
        },
        "environment": capture_environment().to_dict(),
        "identity": capture_benchmark_identity(
            root,
            Path(__file__),
            ("coupling", "model", "host_initialization", "environment", "source_sha256"),
        ).to_dict(),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in owners
        },
        "coupling": [
            row
            for case in coupling_cases
            for row in coupling_rows(
                case,
                routes,
                transforms,
                target,
                budget,
                args.repeats,
                host_initialization=args.host_initialization,
            )
        ],
        "model": [
            row
            for case in model_cases
            for row in model_rows(
                case,
                model_routes,
                model_transforms,
                target,
                budget,
                args.repeats,
                host_initialization=args.host_initialization,
            )
        ],
    }
    write_json_atomic(args.output, record)
    print(args.output)


if __name__ == "__main__":
    main()
