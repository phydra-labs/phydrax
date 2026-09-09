"""Train native SFNO spherical diffusion and deploy a native column closure.

Run from the repository root: python examples/geophysical_operator_forecast.py
This is a small scientific qualification, not a pretrained weather skill claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax.applications.geophysics._learning import (
    column_closure_dataset,
    ColumnClosureBinding,
    deploy_column_closure,
    GeophysicalForecastRequest,
    GeophysicalLearningExperiment,
    NativeGeophysicalForecast,
)
from phydrax.applications.geophysics._metrics import geophysical_forecast_metrics
from phydrax.applications.geophysics._quantities import GeophysicalQuantity
from phydrax.applications.geophysics._time import GeophysicalTimeSpec


def _provenance(size):
    return tuple(
        phx.nn.operator.OperatorCaseProvenance(
            f"case:{i}",
            identities={
                "scenario": "autonomous",
                "model": "analytic-teacher",
                "member": f"member:{i // 2}",
            },
            order={"time": float(i % 2)},
        )
        for i in range(size)
    )


def _dimension_basis(quantity):
    return tuple(name for name, _, _ in quantity.unit.dimension.terms)


def run_example(*, steps=40, artifact_directory=None):
    started = perf_counter()
    op, training = phx.nn.operator, phx.nn.operator.training
    quantity = GeophysicalQuantity(
        "temperature_anomaly",
        "temperature_anomaly",
        phx.units.KELVIN,
        reference_configuration="fixed_reference_temperature",
    )
    space = phx.discretization.SphericalSpectralPlan(3).prepare()
    transform = space.transform
    axes = (
        op.OperatorAxis(
            "theta",
            transform.theta,
            basis="sphere",
            quadrature_weights=transform.theta_quadrature_weights,
        ),
        op.OperatorAxis(
            "phi",
            transform.phi,
            basis="fourier",
            periodic=True,
            quadrature_weights=transform.phi_quadrature_weights,
        ),
    )
    # Both real l=1 harmonics decay by exp(-2*kappa*dt/R²). Choose that factor
    # as 0.92. These are exact heat-equation interval labels on the sphere.
    pattern = jnp.cos(transform.theta)[:, None] * jnp.ones((1, transform.phi.size))
    second = jnp.sin(transform.theta)[:, None] * jnp.cos(transform.phi)[None, :]
    amplitude = jnp.linspace(-2.0, 2.0, 12)
    initial = (
        amplitude[:, None, None] * pattern + (amplitude**2 - 1)[:, None, None] * second
    )
    batch = op.OperatorBatch(
        inputs={"state": op.FunctionSamples(values=initial, axes=axes)},
        queries={"query": op.FunctionSamples(values=None, axes=axes)},
        case_axes=("case",),
        case_shape=(12,),
    )
    task = op.OperatorTask(
        "geophysical-spherical-diffusion",
        dimension_basis=_dimension_basis(quantity),
        fields=(
            op.OperatorFieldSpec(
                "state",
                role="both",
                query_name="query",
                dimension=quantity.unit.dimension,
            ),
        ),
        queries=(
            op.OperatorQuerySpec(
                "query",
                geometry_kind="sphere",
                coordinate_components=("theta", "phi"),
                quadrature="physical_required",
                fixed_geometry=True,
            ),
        ),
        problem=op.OperatorProblemSpec(
            source_query_relation="coincident", query_is_fixed=True, rollout_steps=4
        ),
        metadata={
            "geophysical_target_kind": "next_state",
            "geophysical_step_seconds": 3600.0,
        },
    )
    dataset = training.OperatorDataset(
        batch,
        op.OperatorTargetBatch.from_arrays({"state": 0.92 * initial}, batch),
        provenance=_provenance(12),
    )
    experiment = GeophysicalLearningExperiment.prepare(task, dataset)
    fit = experiment.fit(
        op.architectures.SFNO(space, width=4, depth=1, source_key="state", key=jr.key(1)),
        steps=steps,
        learning_rate=0.01,
        output_field_map={"output": "state"},
        artifact_id="spherical-diffusion-qualification",
    )
    trained = fit.trained_operator
    assert trained is not None

    def artifact_roundtrip(directory):
        training.save_operator_artifact(directory, trained)
        return training.load_trained_operator(directory)

    if artifact_directory is None:
        with TemporaryDirectory(prefix="geophysical-operator-") as directory:
            restored = artifact_roundtrip(directory)
    else:
        restored = artifact_roundtrip(Path(artifact_directory))
    route = training.OperatorRolloutRoute("state", "output", "state")
    adapter = NativeGeophysicalForecast(
        restored, step_seconds=3600.0, quantities={"state": quantity}, routes=(route,)
    )
    member_initial = initial[:2]
    member_batch = op.OperatorBatch(
        inputs={"state": op.FunctionSamples(values=member_initial, axes=axes)},
        queries=batch.queries,
        case_axes=("member",),
        case_shape=(2,),
    )
    request = GeophysicalForecastRequest(
        GeophysicalTimeSpec(),
        "2000-01-01T00:00:00",
        4,
        "diffusion",
        "trained-native-SFNO",
        ("cold", "warm"),
        "diffusion-qualification",
    )
    full = adapter.start(request, member_batch, key=jr.key(9))
    first = adapter.start(request, member_batch, steps=2, key=jr.key(9))
    resumed = adapter.resume(first.continuation, steps=2)
    restart_error = float(
        jnp.max(jnp.abs(full.field("state")[2:] - resumed.field("state")))
    )
    # Verify the ensemble against its analytic ensemble-mean heat solution.
    forecast = full.field("state").reshape(4, 2, 1, -1, 1, 1)
    truth = jnp.stack(
        [0.92**lead * jnp.mean(member_initial, axis=0) for lead in range(1, 5)]
    )
    truth = truth.reshape(4, 1, -1, 1, 1)
    area = (
        transform.theta_quadrature_weights[:, None]
        * transform.phi_quadrature_weights[None, :]
    ).reshape(-1)
    metrics = geophysical_forecast_metrics(
        truth,
        forecast,
        quantities=(quantity,),
        lead_seconds=full.lead_seconds,
        member_ids=request.member_ids,
        area_weights=area,
        layer_weights=jnp.ones(1),
        time_weights=jnp.ones(1),
        mask=jnp.ones_like(truth, dtype=bool),
    )

    temperature = GeophysicalQuantity("temperature", "temperature", phx.units.KELVIN)
    column_task = op.OperatorTask(
        "geophysical-column-mixing",
        dimension_basis=_dimension_basis(temperature),
        fields=(
            op.OperatorFieldSpec(
                "state", role="source", dimension=temperature.unit.dimension
            ),
            op.OperatorFieldSpec(
                "increment",
                role="target",
                query_name="column",
                dimension=temperature.unit.dimension,
            ),
        ),
        queries=(
            op.OperatorQuerySpec(
                "column",
                geometry_kind="point_cloud",
                coordinate_components=("level",),
                quadrature="physical_required",
            ),
        ),
        problem=op.OperatorProblemSpec(
            source_query_relation="coincident", query_is_fixed=False
        ),
        metadata={
            "geophysical_target_kind": "interval_increment",
            "geophysical_step_seconds": 600.0,
        },
    )
    binding = ColumnClosureBinding(
        column_task,
        (temperature,),
        ("state",),
        ("increment",),
        ("top", "middle", "bottom"),
        "three-layer-pressure-column",
        600.0,
    )
    masses = jnp.broadcast_to(jnp.asarray([100.0, 200.0, 300.0]), (12, 3))
    before = (
        280.0 + amplitude[:, None, None] * jnp.asarray([2.0, -1.0, 0.0])[None, :, None]
    )
    mean = jnp.sum(before[..., 0] * masses, axis=-1) / jnp.sum(masses, axis=-1)
    closure = 0.1 * (mean[:, None, None] - before)
    bounds = np.broadcast_to([0.0, 600.0], (12, 2))
    columns = column_closure_dataset(
        binding,
        before,
        before + closure,
        jnp.zeros_like(before),
        layer_mass=masses,
        interval_bounds=bounds,
        forcing={},
        forcing_bounds={},
        provenance=_provenance(12),
    )
    column_experiment = GeophysicalLearningExperiment.prepare(column_task, columns)
    latent = 8
    column_model = op.architectures.DeepONet(
        branch={
            "state": phx.nn.models.MLP(
                in_size=3, out_size=latent, width_size=12, depth=1, key=jr.key(11)
            )
        },
        trunk=phx.nn.models.MLP(
            in_size=1, out_size=latent, width_size=12, depth=1, key=jr.key(12)
        ),
        coord_dim=1,
        latent_size=latent,
    )
    column_fit = column_experiment.fit(
        column_model,
        steps=steps,
        learning_rate=0.005,
        output_field_map={"output": "increment"},
    )
    assert column_fit.trained_operator is not None
    admission = deploy_column_closure(
        column_fit.trained_operator,
        binding,
        columns.batch,
        layer_mass=masses,
        target_budget=jnp.zeros((12, 1)),
        resolved_increment=jnp.zeros_like(before),
        domain_admission=lambda state: (state > 150.0) & (state < 350.0),
    )
    deployed = admission.require_state()
    budget_error = float(
        jnp.max(jnp.abs(jnp.sum((deployed - before) * masses[..., None], axis=-2)))
    )
    return {
        "sfno_initial_loss": fit.initial_loss,
        "sfno_final_loss": fit.final_loss,
        "sfno_completed_steps": fit.completed_steps,
        "column_initial_loss": column_fit.initial_loss,
        "column_final_loss": column_fit.final_loss,
        "column_completed_steps": column_fit.completed_steps,
        "restart_max_error": restart_error,
        "column_budget_max_error": budget_error,
        "column_reported_correction": np.asarray(admission.budget_correction).tolist(),
        "column_admitted": admission.admitted,
        "lead_seconds": full.lead_seconds,
        "valid_times": full.valid_times,
        "rmse_kelvin": np.asarray(metrics.fields[0].rmse.value).tolist(),
        "crps_kelvin": np.asarray(metrics.fields[0].crps.value).tolist(),
        "native_artifact_roundtrip": True,
        "seconds": perf_counter() - started,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--artifact-directory", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_example(steps=args.steps, artifact_directory=args.artifact_directory),
            indent=2,
        )
    )
