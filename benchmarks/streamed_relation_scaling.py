#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated scaling of prepared bounded streamed relations.

Every row varies one controlling capacity of an irregular nonlinear relation:
edge count (through receiver count and degree), degree skew (one high-degree
hub), message channel width, or receiver/edge tile capacity. The streamed route
is compared with the materialized reference that evaluates every event at
once and scatters complete messages, under four separately compiled
transforms: forward energy, reverse (coordinate and parameter gradient),
force-loss parameter gradient (reverse over reverse) and coordinate HVP
(forward over reverse).

Phases are measured separately and synchronized only at their boundaries:
schedule preparation, lowering, compilation, first execution and warm
repeats. Compiler argument/output/alias/temporary/code bytes are official XLA
estimates of each transformed executable; sampled peaks are
``PhaseMemorySampler`` observations, never bounds; declared bytes are the
substrate's ``StreamedRelationResources`` logical bounds, which exclude callback
internals and compiler temporaries. Capacity refusals are retained rows.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, get_args, Literal, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    benchmark_driver_fingerprint,
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax._fingerprint import canonical_fingerprint
from phydrax.execution import compiled_memory_estimate, PhaseMemorySampler
from phydrax.typing import parse


PROJECT_ROOT = Path(__file__).resolve().parents[1]

StreamedRoute: TypeAlias = Literal["streamed", "materialized"]
StreamedTransform: TypeAlias = Literal["forward", "reverse", "force-loss", "hvp"]
ROUTES: tuple[StreamedRoute, ...] = get_args(StreamedRoute)
TRANSFORMS: tuple[StreamedTransform, ...] = get_args(StreamedTransform)

type Parameters = dict[str, Array]
type Energy = Callable[[Parameters, Array], Array]


@dataclass(frozen=True, slots=True)
class StreamedCase:
    """One controlled point of the scaling campaign."""

    name: str
    receivers: int
    degree: int
    hub_degree: int
    channels: int
    hidden: int
    receiver_tile: int
    edge_tile: int
    accumulation: phx.sparse.RelationAccumulation
    seed: int

    @property
    def routes(self) -> int:
        return self.receivers * self.degree + self.hub_degree


@dataclass(frozen=True, slots=True)
class CampaignConfig:
    receivers: tuple[int, ...]
    degrees: tuple[int, ...]
    hub_degrees: tuple[int, ...]
    channels: tuple[int, ...]
    tiles: tuple[tuple[int, int], ...]
    hidden: int
    accumulation: phx.sparse.RelationAccumulation
    warmup: int
    repeats: int
    memory_interval_seconds: float
    seed: int


def _cases(config: CampaignConfig, /) -> list[StreamedCase]:
    """Vary one controlling capacity at a time around the first (base) values."""
    base_receivers, base_degree = config.receivers[0], config.degrees[0]
    base_hub, base_channels = config.hub_degrees[0], config.channels[0]
    base_tiles = config.tiles[0]

    def case(
        name: str,
        *,
        receivers: int = base_receivers,
        degree: int = base_degree,
        hub_degree: int = base_hub,
        channels: int = base_channels,
        tiles: tuple[int, int] = base_tiles,
    ) -> StreamedCase:
        return StreamedCase(
            name=name,
            receivers=receivers,
            degree=degree,
            hub_degree=hub_degree,
            channels=channels,
            hidden=config.hidden,
            receiver_tile=tiles[0],
            edge_tile=tiles[1],
            accumulation=config.accumulation,
            seed=config.seed,
        )

    cases = [case(f"receivers-{count}", receivers=count) for count in config.receivers]
    cases += [case(f"degree-{degree}", degree=degree) for degree in config.degrees[1:]]
    cases += [case(f"hub-{hub}", hub_degree=hub) for hub in config.hub_degrees[1:]]
    cases += [case(f"channels-{width}", channels=width) for width in config.channels[1:]]
    cases += [
        case(f"tiles-{receiver_tile}x{edge_tile}", tiles=(receiver_tile, edge_tile))
        for receiver_tile, edge_tile in config.tiles[1:]
    ]
    return cases


def _topology(case: StreamedCase, /) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-degree receivers plus one hub receiver of ``hub_degree`` routes."""
    rng = np.random.default_rng(case.seed)
    receivers = np.repeat(np.arange(case.receivers, dtype=np.int32), case.degree)
    receivers = np.concatenate((receivers, np.zeros(case.hub_degree, dtype=np.int32)))
    sources = rng.integers(0, case.receivers, size=receivers.size, dtype=np.int32)
    order = rng.permutation(receivers.size)
    return sources[order], receivers[order]


def _parameters(case: StreamedCase, /) -> Parameters:
    rng = np.random.default_rng(case.seed + 1)
    return {
        "w1": jnp.asarray(rng.normal(size=(4, case.hidden)) / 2.0),
        "w2": jnp.asarray(
            rng.normal(size=(case.hidden, case.channels)) / np.sqrt(case.hidden)
        ),
        "w3": jnp.asarray(
            rng.normal(size=(case.channels, case.hidden)) / np.sqrt(case.channels)
        ),
    }


def _message(theta: Parameters, source: Array, receiver: Array, scale: Array) -> Array:
    displacement = receiver - source
    distance = jnp.sqrt(jnp.sum(displacement * displacement) + 1.0)
    features = jnp.concatenate((displacement, distance[None]))
    return jnp.tanh(jnp.tanh(features @ theta["w1"]) @ theta["w2"]) * scale


def _readout(theta: Parameters, receiver: Array, aggregate: Array) -> Array:
    return jnp.sum(jnp.tanh(aggregate @ theta["w3"])) + 0.01 * jnp.sum(receiver**2)


def _streamed_plan(case: StreamedCase, /) -> phx.sparse.StreamedRelationPlan:
    return phx.sparse.StreamedRelationPlan(
        receiver_tile=case.receiver_tile,
        edge_tile=case.edge_tile,
        channel_capacity=case.channels,
        accumulation=case.accumulation,
    )


def _streamed_energy(
    prepared: phx.sparse.PreparedStreamedRelation,
    payload: phx.sparse.StreamedPayloadSpec,
    edge_scale: Array,
    /,
) -> Energy:
    def energy(theta: Parameters, positions: Array) -> Array:
        result = prepared.evaluate(
            payload, _message, _readout, theta, positions, positions, edge_scale
        )
        return jnp.sum(result.receiver_outputs)

    return energy


def _materialized_energy(
    case: StreamedCase, sources: np.ndarray, receivers: np.ndarray, edge_scale: Array, /
) -> Energy:
    """Reference route: every event's message exists at once, then one scatter."""

    def energy(theta: Parameters, positions: Array) -> Array:
        messages = jax.vmap(lambda s, r, e: _message(theta, s, r, e))(
            positions[sources], positions[receivers], edge_scale
        )
        aggregate = (
            jnp.zeros((case.receivers, case.channels), dtype=messages.dtype)
            .at[receivers]
            .add(messages)
        )
        return jnp.sum(jax.vmap(lambda r, a: _readout(theta, r, a))(positions, aggregate))

    return energy


def _transform(
    energy: Energy, transform: StreamedTransform, labels: Array, direction: Array, /
) -> Callable[[Parameters, Array], Any]:
    match transform:
        case "forward":
            return energy
        case "reverse":
            return jax.grad(energy, argnums=(0, 1))
        case "force-loss":

            def force_loss(theta: Parameters, positions: Array) -> Any:
                def loss(parameters: Parameters) -> Array:
                    forces = -jax.grad(energy, argnums=1)(parameters, positions)
                    return jnp.sum((forces - labels) ** 2)

                return jax.grad(loss)(theta)

            return force_loss
        case "hvp":

            def hvp(theta: Parameters, positions: Array) -> Array:
                return jax.jvp(
                    lambda x: jax.grad(energy, argnums=1)(theta, x),
                    (positions,),
                    (direction,),
                )[1]

            return hvp
        case _:
            raise ValueError(f"Unknown transform {transform!r}.")


def _measure_transform(
    function: Callable[[Parameters, Array], Any],
    theta: Parameters,
    positions: Array,
    *,
    label: str,
    config: CampaignConfig,
) -> tuple[Any, dict[str, Any]]:
    jitted = jax.jit(function)
    compiled, timing = measure_lower_and_compile(
        lambda: jitted.lower(theta, positions), lambda lowered: lowered.compile()
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source=f"jax-compiled-streamed-relation-{label}",
        unavailable_reason="Compiler did not expose every estimate",
    )
    estimate = compiled_memory_estimate(compiled)
    value, first_seconds = measure_synchronized(lambda: compiled(theta, positions))
    devices = tuple(device for device in jax.local_devices() if device.platform != "cpu")
    with PhaseMemorySampler(
        f"warm-{label}", interval_seconds=config.memory_interval_seconds, devices=devices
    ) as sampler:
        value, warm = measure_repeated(
            lambda: compiled(theta, positions),
            warmup=config.warmup,
            repeats=config.repeats,
        )
    return value, {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "first_seconds": first_seconds,
        "warm_samples_seconds": list(warm.samples_seconds),
        "warm_median_seconds": warm.median_seconds,
        "compiler": asdict(compiler),
        "compiler_estimate": None if estimate is None else asdict(estimate),
        "sampled_memory": sampler.evidence.to_payload(),
    }


def _defect(left: Any, right: Any, /) -> float:
    """Largest streamed/materialized difference relative to the reference scale."""
    return max(
        float(jnp.max(jnp.abs(a - b)) / jnp.maximum(1.0, jnp.max(jnp.abs(b))))
        for a, b in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True)
    )


def _case_rows(case: StreamedCase, config: CampaignConfig, /) -> list[dict[str, Any]]:
    sources, receivers = _topology(case)
    rng = np.random.default_rng(case.seed + 2)
    positions = jnp.asarray(
        rng.normal(size=(case.receivers, 3)) * case.receivers ** (1 / 3)
    )
    edge_scale = jnp.asarray(rng.uniform(0.5, 1.5, size=sources.size))
    labels = jnp.asarray(rng.normal(size=positions.shape))
    direction = jnp.asarray(rng.normal(size=positions.shape))
    theta = _parameters(case)
    relation = phx.sparse.EdgeRelation(
        sources, receivers, source_size=case.receivers, target_size=case.receivers
    )
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((case.channels,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    plan = _streamed_plan(case)
    # Host preparation is synchronized: the grouping work runs eagerly on device.
    prepared, preparation_seconds = measure_synchronized(
        lambda: plan.prepare(relation, owner_id="benchmark-streamed-relation")
    )
    resources = prepared.resources(payload, theta, positions, positions, edge_scale)
    energies: dict[StreamedRoute, Energy] = {
        "streamed": _streamed_energy(prepared, payload, edge_scale),
        "materialized": _materialized_energy(case, sources, receivers, edge_scale),
    }
    common = {
        "case": case.name,
        **asdict(case),
        "routes": case.routes,
        "preparation_seconds": preparation_seconds,
        "schedule": {
            "tile_count": prepared.schedule.tile_count,
            "logical_bytes": logical_array_bytes(prepared),
            "fragmented_receivers": int(prepared.schedule.fragmented_receivers),
            "maximum_receiver_degree": int(prepared.schedule.maximum_receiver_degree),
            "successful": bool(prepared.schedule.successful),
            "execution_id": prepared.execution_id,
        },
        "declared": _resources_record(resources),
    }
    rows: list[dict[str, Any]] = []
    for transform in TRANSFORMS:
        values: dict[StreamedRoute, Any] = {}
        for route in ROUTES:
            label = f"{case.name}-{route}-{transform}"
            value, phases = _measure_transform(
                _transform(energies[route], transform, labels, direction),
                theta,
                positions,
                label=label,
                config=config,
            )
            values[route] = value
            rows.append({**common, "route": route, "transform": transform, **phases})
        rows[-1]["streamed_materialized_defect"] = _defect(
            values["streamed"], values["materialized"]
        )
    return rows


def _resources_record(
    resources: phx.sparse.StreamedRelationResources, /
) -> dict[str, Any]:
    return {
        "tile_count": resources.tile_count,
        "receiver_tile": resources.receiver_tile,
        "edge_tile": resources.edge_tile,
        "persistent_schedule_bytes": resources.persistent_schedule_bytes,
        "edge_workspace_bytes": resources.edge_workspace_bytes,
        "receiver_workspace_bytes": resources.receiver_workspace_bytes,
        "spill_carry_bytes": resources.spill_carry_bytes,
        "replay_boundary_bytes": resources.replay_boundary_bytes,
        "replay_tiles_in_flight": resources.replay_tiles_in_flight,
        "output_bytes": resources.output_bytes,
        "output_staging_bytes": resources.output_staging_bytes,
        "cotangent_accumulator_bytes": resources.cotangent_accumulator_bytes,
        "replay": resources.replay,
        "resources_id": resources.resources_id,
    }


def _capacity_refusal(config: CampaignConfig, /) -> dict[str, Any]:
    """The plan's channel cap refuses a wider payload before any execution."""
    case = _cases(config)[0]
    sources, receivers = _topology(case)
    relation = phx.sparse.EdgeRelation(
        sources, receivers, source_size=case.receivers, target_size=case.receivers
    )
    prepared = phx.sparse.StreamedRelationPlan(
        receiver_tile=case.receiver_tile,
        edge_tile=case.edge_tile,
        channel_capacity=case.channels,
    ).prepare(relation, owner_id="benchmark-streamed-relation")
    payload = phx.sparse.StreamedPayloadSpec(
        jax.ShapeDtypeStruct((case.channels + 1,), jnp.float64),
        jax.ShapeDtypeStruct((), jnp.float64),
    )
    positions = jnp.zeros((case.receivers, 3))
    try:
        prepared.evaluate(
            payload,
            _message,
            _readout,
            _parameters(case),
            positions,
            positions,
            jnp.ones((sources.size,)),
        )
    except ValueError as error:
        return {"case": case.name, "status": "refused", "reason": str(error)}
    return {"case": case.name, "status": "admitted-unexpectedly"}


def run(config: CampaignConfig, /) -> dict[str, Any]:
    rows = [row for case in _cases(config) for row in _case_rows(case, config)]
    refusal = _capacity_refusal(config)
    passed = refusal["status"] == "refused" and all(
        row["schedule"]["successful"]
        and row.get("streamed_materialized_defect", 0.0) < 1.0e-9
        for row in rows
    )
    configuration = {
        **asdict(config),
        "tiles": [list(tiles) for tiles in config.tiles],
    }
    record: dict[str, Any] = {
        "kind": "streamed-relation-scaling",
        "config": configuration,
        "rows": rows,
        "capacity_refusal": refusal,
        "passed": bool(passed),
        "environment": capture_environment().to_dict(),
        "host_load": {
            "load_average_1_5_15": list(os.getloadavg()),
            "logical_cpus": os.cpu_count(),
        },
        "resource_measurement": {
            "declared_scope": "StreamedRelationResources logical substrate bytes; excludes callback internals and compiler temporaries",
            "retained_scope": "unique logical payload of the prepared relation visible to the runtime object walker",
            "compiler_scope": "official executable estimates of each transformed route, not process resident memory",
            "sampled_scope": "phydrax.execution.PhaseMemorySampler peak during warm repeats; sampled, never an upper bound",
        },
        "driver_dependencies": {
            Path(__file__)
            .relative_to(PROJECT_ROOT)
            .as_posix(): benchmark_driver_fingerprint(PROJECT_ROOT, Path(__file__))
        },
    }
    identity = capture_benchmark_identity(PROJECT_ROOT, Path(__file__), tuple(record))
    record["identity"] = identity.to_dict()
    record["workload_id"] = canonical_fingerprint(
        {"config": configuration, "driver": Path(__file__).name}
    )
    return record


def _integers(text: str, /) -> tuple[int, ...]:
    return tuple(int(value) for value in text.split(","))


def _tiles(text: str, /) -> tuple[tuple[int, int], ...]:
    pairs = []
    for item in text.split(","):
        receiver_tile, edge_tile = item.split("x")
        pairs.append((int(receiver_tile), int(edge_tile)))
    return tuple(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receivers", type=_integers, default=(256, 1024, 4096))
    parser.add_argument("--degrees", type=_integers, default=(16, 48))
    parser.add_argument("--hub-degrees", type=_integers, default=(0, 2048))
    parser.add_argument("--channels", type=_integers, default=(16, 64))
    parser.add_argument(
        "--tiles", type=_tiles, default=((32, 512), (8, 128), (128, 2048))
    )
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument(
        "--accumulation",
        choices=get_args(phx.sparse.RelationAccumulation),
        default="fast",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--memory-interval-seconds", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/streamed_relation_scaling.json")
    )
    arguments = parser.parse_args()
    config = CampaignConfig(
        receivers=arguments.receivers,
        degrees=arguments.degrees,
        hub_degrees=arguments.hub_degrees,
        channels=arguments.channels,
        tiles=arguments.tiles,
        hidden=arguments.hidden,
        accumulation=parse(
            arguments.accumulation, phx.sparse.RelationAccumulation, "accumulation"
        ),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        memory_interval_seconds=arguments.memory_interval_seconds,
        seed=arguments.seed,
    )
    record = run(config)
    write_json_atomic(arguments.output, record)
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
