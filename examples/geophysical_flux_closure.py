"""Train and reload native conservative vapor/total-energy interface operators.

The truth is a closed, finite-volume vertical diffusion experiment, not weather
observations. Held-out grids, closure intervals and diffusivity forcing are
actually evolved independently. Reported transfer skill is limited to this law,
initial-condition family and the explicitly enumerated withheld supports.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.applications.atmosphere._column import conservative_vertical_mixing
from phydrax.applications.atmosphere._interactive_column import InteractiveMoistColumnPlan
from phydrax.applications.atmosphere._moist import MoistThermodynamicPlan
from phydrax.applications.geophysics._flux_closure import (
    column_flux_datasets,
    column_flux_space,
    column_flux_tasks,
    ColumnFluxBinding,
    conditional_column_flux_target,
    ConservativeColumnTransfer,
    deploy_column_flux,
)
from phydrax.applications.geophysics._learning import GeophysicalLearningExperiment


def _transfer(layers, refinement, thermo):
    # Unit horizontal area and 1 kg/m3 dry density: these measures are kg/m2.
    fine_mass = np.full(layers * refinement, 600.0 / (layers * refinement))
    coarse_mass = np.full(layers, 600.0 / layers)
    fine, fm = column_flux_space(fine_mass, f"transport/fine-{layers * refinement}")
    coarse, cm = column_flux_space(coarse_mass, f"transport/coarse-{layers}")
    restriction = np.zeros((layers, layers * refinement))
    for cell in range(layers):
        selection = slice(cell * refinement, (cell + 1) * refinement)
        restriction[cell, selection] = fine_mass[selection] / coarse_mass[cell]
    transfer = phx.discretization.FieldTransfer(
        fine,
        coarse,
        phx.linalg.DenseLinearOperator(
            jnp.asarray(restriction),
            source=fine.vector_space,
            target=coarse.vector_space,
        ),
        properties=phx.discretization.TransferProperties(
            conservative=True,
            constant_preserving=True,
            positivity_preserving=True,
            nested=True,
        ),
    )
    return ConservativeColumnTransfer(transfer, fm, cm, thermo.plan_id)


def _evolve(initial, mass, interval, diffusivity, substeps):
    """Execute the existing native closed vertical transport law, no fitted labels."""
    rate = diffusivity / (600.0 / mass.size) ** 2
    h = interval / substeps
    if 2 * h * rate >= 1:
        raise ValueError("Explicit native transport timestep violates positivity CFL.")

    def step(_, values):
        derivative = jax.vmap(
            lambda value: conservative_vertical_mixing(value / mass, mass, rate),
            in_axes=-1,
            out_axes=-1,
        )(values)
        return values + h * derivative

    return jax.lax.fori_loop(0, substeps, step, initial)


def _cases(tasks, thermo, *, layers, interval, forcing, count, seed, partition):
    transfer = _transfer(layers, 4, thermo)
    fm, cm = transfer.source_measure.weights, transfer.target_measure.weights
    rng = np.random.default_rng(seed)
    x = (jnp.arange(fm.size) + 0.5) / fm.size
    amplitude = jnp.asarray(rng.uniform(-1.0, 1.0, (count, 3)))
    vapor = fm * (
        0.004
        + 0.001 * amplitude[:, :1] * jnp.cos(jnp.pi * x)
        + 0.0004 * amplitude[:, 1:2] * jnp.cos(2 * jnp.pi * x)
    )
    temperature = 286 + 5 * amplitude[:, :1] * jnp.cos(jnp.pi * x) + 2 * amplitude[:, 2:3]
    dry_energy, vapor_energy, _, _ = thermo.phase_energies(temperature)
    energy = fm * dry_energy + vapor * vapor_energy
    fine_before = jnp.stack((vapor, energy), axis=-1)
    coarse_before = transfer.inventories(fine_before)
    # The coarse resolved model omits 75% of the diffusion. The conditional
    # target must learn that unresolved transfer plus spatial-discretization effects.
    diffusivity = 30.0 * forcing
    fine_after = _evolve(fine_before, fm, interval, diffusivity, 64)
    coarse_after = _evolve(coarse_before, cm, interval, 0.25 * diffusivity, 8)
    # Separate temporal numerical discrepancy using independently refined time
    # integration at both spatial resolutions. Spatial discrepancy remains in
    # the unresolved target; no analytic continuum-reference claim is made.
    fine_reference = _evolve(fine_before, fm, interval, diffusivity, 256)
    coarse_reference = _evolve(coarse_before, cm, interval, 0.25 * diffusivity, 64)
    binding = ColumnFluxBinding(
        tasks,
        transfer.transfer.target,
        transfer.target_measure,
        interval,
        600.0 / layers,
        "closed-vapor-diffusion",
        f"diffusivity-multiplier-{forcing:g}",
    )
    bounds = np.broadcast_to([0.0, interval], (count, 2))
    target = conditional_column_flux_target(
        binding,
        transfer,
        fine_before,
        fine_after,
        coarse_before,
        coarse_after,
        interval_bounds=bounds,
        fine_reference_after=fine_reference,
        coarse_reference_after=coarse_reference,
        reference_id="same-grid-temporally-refined-256-fine-64-coarse",
    )
    batch = binding.batch(
        coarse_before[..., 0], coarse_before[..., 1], forcing=np.full(count, forcing)
    )
    records = tuple(
        phx.nn.operator.OperatorCaseProvenance(
            f"{partition}/case-{index}",
            identities={
                "scenario": "closed-diffusion",
                "model": "native-vertical-mixing",
                "member": f"{partition}/seed-{seed}/member-{index}",
            },
        )
        for index in range(count)
    )
    datasets = column_flux_datasets(
        binding, batch, target, provenance=records, interval_bounds=bounds
    )
    return {
        "binding": binding,
        "datasets": datasets,
        "batch": batch,
        "target": target,
        "before": coarse_before,
        "resolved": coarse_reference,
        "truth": transfer.inventories(fine_reference),
        "mass": cm,
    }


def _model(key):
    op = phx.nn.operator
    keys = jr.split(key, 3)
    latent = 16
    # Quadrature-aware variable sensors, not a fixed-size flattened branch.
    # dt/dx are recorded in the task/batch, not used as proof of generalization.
    branches = {
        name: op.architectures.IntegralBranchEncoder(
            feature_model=phx.nn.models.MLP(
                in_size=2,
                out_size=latent,
                width_size=32,
                depth=2,
                key=keys[index],
            ),
            latent_size=latent,
            coord_dim=1,
            normalize=True,
        )
        for index, name in enumerate(("vapor_per_dry_mass", "energy_per_dry_mass"))
    }
    return op.architectures.DeepONet(
        branch=branches,
        trunk=phx.nn.models.MLP(
            in_size=1,
            out_size=latent,
            width_size=32,
            depth=2,
            key=keys[2],
        ),
        coord_dim=1,
        latent_size=latent,
    )


def _metrics(operators, data, thermo):
    binding = data["binding"]
    resolved, truth, mass = data["resolved"], data["truth"], data["mass"]
    admission = deploy_column_flux(
        operators,
        binding,
        data["batch"],
        vapor_mass=resolved[..., 0],
        total_energy=resolved[..., 1],
        thermodynamics=thermo,
        dry_mass=mass,
        artifact_ids=tuple(model.artifact_id for model in operators),
    )
    candidate = jnp.stack((admission.vapor_mass, admission.total_energy), axis=-1)
    # Rejection is explicitly counted and the consumer retains the no-closure
    # endpoint; a rejected prediction is never counted as a successful correction.
    accepted = np.asarray(admission.admitted)
    committed = jnp.where(admission.admitted[:, None, None], candidate, resolved)

    def rms(value):
        return np.asarray(
            jnp.sqrt(
                jnp.sum(value**2 * mass[None, :, None], axis=(0, 1))
                / (mass.sum() * value.shape[0])
            )
        )

    error = rms((committed - truth) / mass[None, :, None])
    baseline = rms((resolved - truth) / mass[None, :, None])
    normalized = error / baseline
    numerical = rms(data["target"].numerical_increment / mass[None, :, None])
    return {
        "vapor_per_dry_mass_rmse": float(error[0]),
        "energy_per_dry_mass_rmse_J_kg": float(error[1]),
        "no_closure_vapor_rmse": float(baseline[0]),
        "no_closure_energy_rmse_J_kg": float(baseline[1]),
        "relative_rmse_vs_no_closure": normalized.tolist(),
        "skill_vs_no_closure": (1.0 - normalized).tolist(),
        "numerical_discrepancy_rms": numerical.tolist(),
        "rejection_rate": float(1.0 - accepted.mean()),
        "correction_rate": 0.0,
        "inventory_residual_max": np.max(
            np.abs(np.asarray(admission.inventory_residual)), axis=0
        ).tolist(),
        "cases": int(accepted.size),
        "layers": int(mass.size),
        "resolution_m": binding.resolution_m,
        "closure_interval_s": binding.interval_seconds,
        "forcing_id": binding.forcing_id,
    }


def _column_restart(operators, data, thermo):
    """Exercise physical native continuation with stateless restored flux artifacts."""
    binding = data["binding"]
    mass = data["mass"]
    water, energy = data["before"][0, :, 0], data["before"][0, :, 1]
    reference = thermo.phase_energies(thermo.reference_temperature)[1]
    temperature = thermo.reference_temperature + (energy - water * reference) / (
        mass * thermo.dry_cv + water * thermo.vapor_cv
    )
    # Isolate learned vapor transport: no falling hydrometeors and no separate
    # convective-mixing closure (length=50 otherwise activates at negative N²).
    # Neither learned interface rates nor their physical interval are rescaled.
    plan = InteractiveMoistColumnPlan(
        thermodynamics=thermo,
        rain_fall_speed=0.0,
        snow_fall_speed=0.0,
        mixing_length=0.0,
    )
    initial = plan.initialize(mass, water, temperature, mass, surface_temperature=286.0)
    identities = tuple(model.artifact_id for model in operators)

    def advance(state):
        batch = binding.batch(
            state.vapor_mass[None, :], state.internal_energy[None, :], forcing=np.ones(1)
        )
        flux = deploy_column_flux(
            operators,
            binding,
            batch,
            vapor_mass=state.vapor_mass[None, :],
            total_energy=state.internal_energy[None, :],
            thermodynamics=thermo,
            dry_mass=mass,
            liquid_mass=(state.cloud_liquid_mass + state.rain_mass)[None, :],
            ice_mass=(state.cloud_ice_mass + state.snow_mass)[None, :],
            artifact_ids=identities,
        )
        flux.require_inventories()
        result = plan.step(
            state,
            binding.interval_seconds,
            wind_speed=0.0,
            water_flux=flux.water_flux[0],
            energy_flux=flux.energy_flux[0],
        )
        if not bool(result.successful):
            raise RuntimeError(
                "Physical column rejected the trained-flux continuation: "
                f"dt={binding.interval_seconds}, stable_step={float(result.stable_step)}, "
                f"water_residual={float(result.water_residual)}, "
                f"energy_residual={float(result.energy_residual)}."
            )
        return result.state

    first = advance(initial)
    uninterrupted = advance(first)
    with TemporaryDirectory(prefix="native-flux-column-restart-") as directory:
        path = Path(directory) / "column.npz"
        plan.save_checkpoint(path, first)
        resumed = advance(plan.load_checkpoint(path))
    return max(
        float(jnp.max(jnp.abs(left - right)))
        for left, right in zip(
            jax.tree.leaves(uninterrupted), jax.tree.leaves(resumed), strict=True
        )
    )


def run_example(*, steps=200, artifact_directory=None):
    jax.config.update("jax_enable_x64", True)
    thermo = MoistThermodynamicPlan()
    tasks = column_flux_tasks(
        "native-closed-vapor-diffusion",
        thermo,
        training_resolutions_m=(100.0,),
        training_intervals_s=(60.0,),
        training_regimes=("closed-vapor-diffusion",),
        training_forcing_ids=("diffusivity-multiplier-1",),
    )
    training_data = _cases(
        tasks,
        thermo,
        layers=6,
        interval=60.0,
        forcing=1.0,
        count=36,
        seed=12,
        partition="training",
    )
    models = (_model(jr.key(3)), _model(jr.key(4)))
    fits, untrained = [], []
    native = phx.nn.operator.training
    for index, (task, dataset, model) in enumerate(
        zip(tasks, training_data["datasets"], models, strict=True)
    ):
        experiment = GeophysicalLearningExperiment.prepare(task, dataset)
        fit = experiment.fit(
            model,
            steps=steps,
            learning_rate=0.003,
            output_field_map={"output": task.fields[-1].name},
            artifact_id=f"closed-flux-{index}-trained",
        )
        if fit.trained_operator is None:
            raise RuntimeError(
                "Native fit did not produce a task-bound operator artifact."
            )
        fits.append(fit)
        untrained.append(
            native.TrainedOperator(
                model,
                task,
                training_evidence=phx.nn.operator.OperatorTrainingEvidence(
                    "task_specific"
                ),
                output_field_map={"output": task.fields[-1].name},
                normalization=fit.normalization,
                artifact_id=f"closed-flux-{index}-untrained",
                provenance={"untrained_baseline": True},
            )
        )

    def reload(directory):
        restored = []
        for index, fit in enumerate(fits):
            path = Path(directory) / f"flux-{index}"
            native.save_operator_artifact(path, fit.trained_operator)
            restored.append(native.load_trained_operator(path))
        return tuple(restored)

    if artifact_directory is None:
        with TemporaryDirectory(prefix="native-column-flux-") as directory:
            trained = reload(directory)
    else:
        trained = reload(artifact_directory)
    eager_sources = []
    artifact_reports = {}
    for fit, restored, task, unit in zip(
        fits, trained, tasks, ("kg m^-2 s^-1", "W m^-2"), strict=True
    ):
        source = fit.trained_operator
        identities_match = (
            source.artifact_id == restored.artifact_id
            and source.task_fingerprint == restored.task_fingerprint
            and source.contract_fingerprint == restored.contract_fingerprint
            and source.normalization_fingerprint == restored.normalization_fingerprint
            and array_tree_fingerprint(source.execution_model)
            == array_tree_fingerprint(restored.execution_model)
        )
        if not identities_match:
            raise RuntimeError(
                f"Native artifact changed task/weights/normalization identity for {task.fields[-1].name}."
            )
        # Native artifacts restore eager execution. Compare against the original
        # weights in that SAME mode to separate serialization from JIT arithmetic.
        eager_sources.append(
            native.TrainedOperator(
                source.execution_model,
                source.task,
                training_evidence=source.training_evidence,
                output_field_map=source.output_field_map,
                fixed_query_fingerprints=source.fixed_query_fingerprints,
                output_pipeline=source.output_pipeline,
                normalization=source.normalization,
                dtype_policy=source.dtype_policy,
                sharding_policy=source.sharding_policy,
                compilation_strategy="eager",
                padding_policy=source.padding_policy,
                artifact_id=source.artifact_id,
                provenance=dict(source.provenance),
                calibration=dict(source.calibration),
            )
        )
        artifact_reports[task.fields[-1].name] = {
            "unit": unit,
            "artifact_id": source.artifact_id,
            "task_fingerprint": source.task_fingerprint,
            "normalization_fingerprint": source.normalization_fingerprint,
            "exact_task_weights_normalization_identity": True,
            "original_execution_strategy": source.compilation_strategy,
            "restored_execution_strategy": restored.compilation_strategy,
            "same_eager_max_error": 0.0,
            "max_absolute_error": 0.0,
            "max_relative_linf_error": 0.0,
            "max_eps_scaled_linf_error": 0.0,
            # jnp.allclose defaults used by native artifact roundtrip regressions.
            "rtol": 1e-5,
            "atol_SI": 1e-8,
            "max_tolerance_ratio": 0.0,
        }
    held_out = {
        "independent_members": (6, 60.0, 1.0, 30),
        "withheld_resolution": (8, 60.0, 1.0, 31),
        "withheld_interval": (6, 75.0, 1.0, 32),
        "withheld_forcing": (6, 60.0, 1.15, 33),
        "joint_withheld": (8, 75.0, 1.15, 34),
    }
    reports = {}
    for name, (layers, interval, forcing, seed) in held_out.items():
        data = _cases(
            tasks,
            thermo,
            layers=layers,
            interval=interval,
            forcing=forcing,
            count=12,
            seed=seed,
            partition=name,
        )
        reports[name] = {
            "trained": _metrics(trained, data, thermo),
            "untrained": _metrics(tuple(untrained), data, thermo),
        }
        reports[name]["case_ids"] = [
            record.case_id for record in data["datasets"][0].provenance
        ]
        for fit, restored, eager, task in zip(
            fits, trained, eager_sources, tasks, strict=True
        ):
            field = task.fields[-1].name
            original = np.asarray(
                fit.trained_operator.predict(data["batch"]).field(field).values
            )
            reloaded = np.asarray(restored.predict(data["batch"]).field(field).values)
            same_mode = np.asarray(eager.predict(data["batch"]).field(field).values)
            if not all(
                np.all(np.isfinite(values)) for values in (original, reloaded, same_mode)
            ):
                raise RuntimeError(
                    f"Native artifact produced nonfinite {field} predictions on {name}."
                )
            absolute = float(np.max(np.abs(original - reloaded)))
            scale = max(float(np.max(np.abs(original))), np.finfo(original.dtype).tiny)
            record = artifact_reports[field]
            record["same_eager_max_error"] = max(
                record["same_eager_max_error"],
                float(np.max(np.abs(same_mode - reloaded))),
            )
            record["max_absolute_error"] = max(record["max_absolute_error"], absolute)
            record["max_relative_linf_error"] = max(
                record["max_relative_linf_error"], absolute / scale
            )
            record["max_eps_scaled_linf_error"] = max(
                record["max_eps_scaled_linf_error"],
                absolute / (np.finfo(original.dtype).eps * scale),
            )
            record["max_tolerance_ratio"] = max(
                record["max_tolerance_ratio"],
                float(
                    np.max(
                        np.abs(original - reloaded)
                        / (record["atol_SI"] + record["rtol"] * np.abs(original))
                    )
                ),
            )
    return {
        "evidence_scope": "native closed vapor/total-energy diffusion only; not real-Earth skill",
        "initial_loss": [fit.initial_loss for fit in fits],
        "final_loss": [fit.final_loss for fit in fits],
        "completed_steps": [fit.completed_steps for fit in fits],
        "native_artifact_roundtrip": artifact_reports,
        "native_column_restart_max_error": _column_restart(
            trained, training_data, thermo
        ),
        "training_case_ids": [
            record.case_id for record in training_data["datasets"][0].provenance
        ],
        "training_resolution_m": [100.0],
        "training_closure_interval_s": [60.0],
        "training_forcing_ids": ["diffusivity-multiplier-1"],
        "evaluation": reports,
        "numerical_reference": "same spatial operators with refined temporal substeps; not continuum truth",
        "correction_policy": "none; physically inadmissible proposed fluxes are rejected",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--artifact-directory", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            run_example(steps=args.steps, artifact_directory=args.artifact_directory),
            indent=2,
        )
    )
