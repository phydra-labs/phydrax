# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Learn certified sphere area from B-Rep graphs and take one radius-gradient step.

Five spheres seed incidence-graph messages with sampled face positions. An
Equinox readout learns certified area; one radius step is independently rebuilt
and certified. Run from the repository root::

    JAX_ENABLE_X64=1 python -m examples.brep_graph_surrogate
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

import phydrax as phx
from phydrax.nn.layers import Linear


@runtime_checkable
class BRepRealizer(Protocol):
    def realize(
        self, state: phx.geometry.DesignState, /
    ) -> phx.geometry.FixedTopologyBRepRealization: ...


def prepare(
    radius: float,
) -> tuple[
    phx.geometry.BRepModel,
    phx.graph.BRepIncidenceGraph,
    phx.graph.BRepGraphSamplePlan,
    float,
]:
    """Prepare one family member, its sample plan, and certified area midpoint."""
    model = phx.geometry.brep_sphere(
        radius,
        coordinate_contract=phx.SpatialCoordinateContract.si(),
        tessellation=phx.geometry.BRepTessellationPolicy(realize=False),
    )
    with phx.HostBatchTicket(1_000_000, bytes_upper=64 << 20) as ticket:
        budget = phx.discretization.CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=1_000_000,
            maximum_memory_bytes=64 << 20,
        )
        with budget.activate():
            mass = phx.geometry.brep_mass_properties(
                model,
                model.require_geometry(),
                subdivisions=1,
                coordinate_budget=budget,
            )
            plan = phx.graph.prepare_brep_graph_samples(
                model,
                grid_size=3,
                coedge_samples=3,
                policy=phx.geometry.brep.BRepQueryPolicy(
                    maximum_operations=100_000,
                    maximum_points=10_000,
                    maximum_scratch_bytes=1 << 20,
                ),
                budget=budget,
            )
    incidence = phx.graph.brep_incidence_graph(model, mass_properties=mass)
    area = float(jnp.sum(jnp.mean(mass.face_area_enclosures, axis=1)))
    return model, incidence, plan, area


def graph_feature(
    incidence: phx.graph.BRepIncidenceGraph, samples: phx.graph.BRepGraphSamples
) -> Array:
    """Pool one incidence-message pass from sampled face positions."""
    face_values = jnp.stack(
        [jnp.mean(jnp.sum(grid[:3] ** 2, axis=0)) for grid in samples.face_grids]
    )
    graph = incidence.graph
    if graph.senders is None or graph.receivers is None:
        raise ValueError(
            "incidence.graph must have explicit edge endpoints; got implicit graph"
        )
    values = jnp.zeros((len(incidence.entity_ids),), dtype=jnp.float64)
    values = values.at[: face_values.size].set(face_values)
    messages = values.at[graph.receivers].add(values[graph.senders])
    count = face_values.size + jnp.sum(graph.senders < face_values.size)
    return (jnp.sum(messages) / count)[None]


def run() -> dict[str, float]:
    """Train a graph readout and minimize a target-area objective once."""
    phx.config.configure(phx.config.RuntimeConfig(enable_x64=True))
    features, targets = [], []
    for radius in (0.75, 1.0, 1.25, 1.5, 1.75):
        model, incidence, plan, area = prepare(radius)
        source = phx.geometry.FixedTopologyBRepSource(model).compile(
            coordinate_unit=phx.units.METER,
        )
        kernel = source.kernel
        if not isinstance(kernel, BRepRealizer):
            raise TypeError(
                "source.kernel must realize fixed-topology B-Rep geometry; got unsupported kernel"
            )
        features.append(
            graph_feature(incidence, plan.evaluate(kernel.realize(source.state)))
        )
        targets.append(area)
    x = jnp.stack(features)
    y = jnp.asarray(targets, dtype=jnp.float64)
    readout = Linear(
        in_size=1, out_size="scalar", rwf=False, use_bias=False, key=jax.random.key(7)
    )

    @eqx.filter_jit
    def train_step(network: Linear) -> tuple[Linear, Array]:
        def loss(candidate: Linear) -> Array:
            return jnp.mean((jax.vmap(candidate)(x) - y) ** 2)

        value, gradient = eqx.filter_value_and_grad(loss)(network)
        updates = jax.tree.map(lambda leaf: -0.03 * leaf, gradient)
        return eqx.apply_updates(network, updates), value

    for _ in range(160):
        readout, loss = train_step(readout)
    radius_index = next(
        index
        for index, spec in enumerate(source.schema.specs)
        if spec.parameter_id.name == "radius"
    )

    def predict(radius: Array) -> Array:
        state = source.state.replace_at(radius_index, radius)
        samples = plan.evaluate(kernel.realize(state))
        return readout(graph_feature(incidence, samples))

    desired_area = jnp.asarray(4.0 * jnp.pi, dtype=jnp.float64)

    def objective(radius: Array) -> Array:
        return ((predict(radius) - desired_area) / desired_area) ** 2

    before = jnp.asarray(1.25, dtype=jnp.float64)
    after = before - 0.05 * jax.grad(objective)(before)
    _, _, _, certified_before = prepare(float(before))
    _, _, _, certified_after = prepare(float(after))
    return {
        "training_mse": float(loss),
        "radius_before": float(before),
        "radius_after": float(after),
        "prediction_before": float(predict(before)),
        "prediction_after": float(predict(after)),
        "certified_before": certified_before,
        "certified_after": certified_after,
    }


if __name__ == "__main__":
    for name, value in run().items():
        print(f"{name}: {value:.6g}")
