#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from tools.operator_benchmarks.scenarios import periodic_burgers_scenario


class _SelectedQuadratic(phx.rom.AbstractSampledNonlinearProvider):
    provider_id: str = eqx.field(static=True)
    support_id: str = eqx.field(static=True)
    geometry_id: str = eqx.field(static=True)

    def __init__(self):
        self.provider_id = "qualification-selected-quadratic"
        self.support_id = "qualification-support"
        self.geometry_id = "qualification-geometry"

    def evaluate_selected(self, state, node_indices, inputs, /):
        del inputs
        values = jnp.asarray([state[0] ** 2, state[1] ** 2, 0.0])
        return values[node_indices]


def _basis(space, matrix, *, role, source):
    return phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            space,
            matrix,
            orthonormal=True,
            subspace_id=f"qualification-{role}-subspace",
        ),
        role=role,
        state_contract_id="qualification-state",
        support_id="qualification-support",
        measure_id="qualification-measure",
        geometry_id="qualification-geometry",
        source_artifact_ids=(source,),
    )


def affine_thermal_block_qualification() -> dict[str, object]:
    dimension = 8
    base = 0.1 * np.eye(dimension)
    left = np.zeros((dimension, dimension))
    right = np.zeros((dimension, dimension))
    for edge in range(dimension + 1):
        incidence = np.zeros((dimension,))
        if edge > 0:
            incidence[edge - 1] -= 1.0
        if edge < dimension:
            incidence[edge] += 1.0
        target = left if (edge + 0.5) / (dimension + 1) < 0.5 else right
        target += np.outer(incidence, incidence)
    matrices = tuple(
        jnp.asarray(value, dtype=jnp.float64) for value in (base, left, right)
    )
    forcing = jnp.ones((dimension,), dtype=jnp.float64)
    training = []
    for left_value in (0.5, 1.0, 2.0):
        for right_value in (0.5, 1.0, 2.0):
            assembled = base + left_value * left + right_value * right
            training.append(np.linalg.solve(assembled, np.asarray(forcing)))
    full = phx.linalg.ArraySpace(
        (dimension,), dtype=jnp.float64, space_id="qualification-full"
    )
    pod = phx.ml.decomposition.PhysicalPODPlan(
        4, retained_energy=0.999999, centered=False
    ).fit(
        full,
        jnp.asarray(np.stack(training)),
        source_artifact_ids=("thermal-block-training",),
    )
    basis = phx.rom.ReducedBasisArtifact(
        pod.subspace,
        role="state",
        state_contract_id="thermal-temperature",
        support_id="qualification-support",
        measure_id="qualification-measure",
        geometry_id="qualification-geometry",
        source_artifact_ids=("thermal-block-training",),
        evidence_ids=(pod.result_id,),
    )
    reduction = phx.rom.trial_test_reduction_from_bases(basis)
    dual = phx.linalg.DualSpace(full)
    operators = tuple(
        phx.linalg.DenseLinearOperator(
            matrix,
            source=full,
            target=dual,
            operator_id=f"thermal-block-A{index}",
        )
        for index, matrix in enumerate(matrices)
    )
    problem = phx.rom.AffineLinearROMProblem(
        reduction,
        operators,
        (forcing,),
        operator_term_ids=("mass", "left-conductivity", "right-conductivity"),
        right_hand_side_term_ids=("forcing",),
        source_artifact_ids=("qualification-thermal-block-family",),
    )
    coefficients = phx.rom.ArrayAffineCoefficientMap(
        jnp.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=jnp.float64),
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.zeros((1, 2), dtype=jnp.float64),
        jnp.ones((1,), dtype=jnp.float64),
        operator_term_ids=("mass", "left-conductivity", "right-conductivity"),
        right_hand_side_term_ids=("forcing",),
        lower=jnp.asarray([0.5, 0.5], dtype=jnp.float64),
        upper=jnp.asarray([2.0, 2.0], dtype=jnp.float64),
        input_contract_id="thermal-conductivities",
        unit_contract_id="thermal-conductivity",
        support_id="qualification-support",
    )
    model = phx.rom.prepare_affine_linear_rom(problem, coefficients)
    parameters = jnp.asarray([0.8, 1.4], dtype=jnp.float64)
    result = model.evaluate(parameters, reconstruct=True)
    assembled = (
        np.asarray(matrices[0])
        + float(parameters[0]) * np.asarray(matrices[1])
        + float(parameters[1]) * np.asarray(matrices[2])
    )
    truth = np.linalg.solve(assembled, np.asarray(forcing))
    relative_error = float(
        np.linalg.norm(np.asarray(result.reconstructed_state) - truth)
        / np.linalg.norm(truth)
    )
    refused = model.evaluate(jnp.asarray([0.1, 1.4], dtype=jnp.float64))
    return {
        "passed": (
            bool(result.valid)
            and not bool(refused.valid)
            and model.reduction.rank < dimension
            and relative_error <= 5.0e-3
        ),
        "relative_state_error": relative_error,
        "full_dimension": dimension,
        "reduced_rank": model.reduction.rank,
        "reduced_residual": float(jnp.linalg.norm(result.reduced_residual)),
        "model_id": model.model_id,
    }


def operator_inference_qualification() -> dict[str, object]:
    time = jnp.linspace(0.0, 1.0, 32, dtype=jnp.float64)
    states = jnp.stack((jnp.sin(5.0 * time), jnp.cos(3.0 * time)), axis=-1)[None, ...]
    inputs = jnp.sin(2.0 * time)[None, :, None]
    z0 = states[..., 0]
    z1 = states[..., 1]
    control = inputs[..., 0]
    derivatives = jnp.stack(
        (
            0.5 + 1.2 * z0 - 0.3 * z1 + 0.7 * control + 0.2 * z0 * z1,
            -0.1 + 0.4 * z0 - 0.8 * z1 - 0.2 * control + 0.3 * z1**2,
        ),
        axis=-1,
    )
    state_layout = phx.dynamics.StateLayout(
        (2,),
        component_names=("z0", "z1"),
        layout_id="qualification-opinf-state",
    )
    input_layout = phx.dynamics.InputLayout(
        (1,),
        component_names=("u",),
        layout_id="qualification-opinf-input",
    )
    data = phx.dynamics.TrajectoryData(
        time,
        states,
        state_layout=state_layout,
        inputs=inputs,
        input_layout=input_layout,
        input_alignment="samples",
        derivatives=derivatives,
        case_axes=("case",),
        source_id="qualification-opinf-data",
    )
    library = phx.dynamics.identification.OperatorInferenceFeatureLibrary(
        state_layout,
        input_layout=input_layout,
    )
    result = phx.dynamics.identification.fit_operator_inference(
        phx.dynamics.identification.SINDyProblem(
            data=data,
            library=library,
            formulation=phx.dynamics.identification.StrongSINDyFormulation(),
        ),
        (1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12),
    )
    prediction = result.evaluate(states, inputs)
    relative = float(
        jnp.linalg.norm(prediction - derivatives)
        / jnp.maximum(jnp.linalg.norm(derivatives), 1.0e-12)
    )
    return {
        "passed": bool(result.valid) and relative <= 1.0e-5,
        "relative_derivative_error": relative,
        "feature_count": library.num_features,
        "method_id": result.method_id,
    }


def hyperreduction_qualification() -> dict[str, object]:
    full = phx.linalg.ArraySpace(
        (3,), dtype=jnp.float64, space_id="qualification-hyper-full"
    )
    state_basis = _basis(
        full,
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64),
        role="state",
        source="hyper-state-data",
    )
    reduction = phx.rom.trial_test_reduction_from_bases(state_basis)
    nonlinear_basis = _basis(
        phx.linalg.DualSpace(full),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=jnp.float64),
        role="nonlinear-term",
        source="hyper-nonlinear-data",
    )
    provider = _SelectedQuadratic()
    artifact = phx.rom.prepare_deim(reduction, nonlinear_basis, provider)
    state = jnp.asarray([2.0, 3.0, 0.0], dtype=jnp.float64)
    reduced = artifact.evaluate(provider, state)
    error = float(jnp.linalg.norm(reduced - jnp.asarray([4.0, 9.0])))
    return {
        "passed": error <= 1.0e-12,
        "reduced_error": error,
        "sample_count": int(artifact.node_indices.size),
        "artifact_id": artifact.artifact_id,
    }


def burgers_projection_qualification() -> dict[str, object]:
    scenario = periodic_burgers_scenario(
        train_resolution=32,
        test_resolution=48,
        num_cases=8,
        initial_condition="shock",
        rollout_steps=2,
        target_steps=2,
        seed=31,
    )
    source = jnp.asarray(scenario.train_batch.input("state").values)
    target = jnp.asarray(scenario.train_target)
    training = jnp.concatenate((source[:6], target[:6]), axis=0)
    fit = phx.ml.decomposition.POD(
        10,
        centered=False,
        query_layout_provenance=("periodic-unit-interval", "burgers-state"),
    ).fit_batch(phx.ml.MLBatch(training))
    model = fit.as_trainable()
    projected = model.project(target[6:])
    relative = float(
        jnp.linalg.norm(projected - target[6:])
        / jnp.maximum(jnp.linalg.norm(target[6:]), 1.0e-12)
    )
    metadata = dict(scenario.metadata)
    mass_drift = float(metadata["maximum_absolute_mass_drift"])
    return {
        "passed": relative <= 0.75 and mass_drift <= 1.0e-10,
        "held_out_projection_error": relative,
        "projection_error_gate": 0.75,
        "maximum_absolute_mass_drift": mass_drift,
        "scenario": scenario.name,
    }


def run() -> dict[str, object]:
    scenarios = {
        "affine_thermal_block": affine_thermal_block_qualification(),
        "operator_inference": operator_inference_qualification(),
        "hyperreduction": hyperreduction_qualification(),
        "burgers_projection": burgers_projection_qualification(),
    }
    capabilities = phx.rom.rom_capability_catalog()
    return {
        "kind": "phydrax-rom-qualification",
        "passed": all(bool(value["passed"]) for value in scenarios.values()),
        "capabilities": [
            {
                "capability": value.capability,
                "maturity": value.maturity.name.lower(),
                "declaration_id": value.declaration_id,
                "required_gates": list(value.required_gates),
            }
            for value in capabilities
        ],
        "scenarios": scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is None:
        print(payload)
    else:
        arguments.output.write_text(payload + "\n", encoding="utf-8")
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
