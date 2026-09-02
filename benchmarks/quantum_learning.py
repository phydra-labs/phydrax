#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from _quantum_learning_data import QuantumLearningDataset, two_curves
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

import phydrax as phx
from phydrax._model import AbstractArrayModel


def _stratified_holdout(
    dataset: QuantumLearningDataset,
    /,
    *,
    holdout_per_class: int,
) -> tuple[phx.ml.MLBatch, jax.Array, jax.Array]:
    negative = jnp.nonzero(dataset.targets == 0.0)[0]
    positive = jnp.nonzero(dataset.targets == 1.0)[0]
    if negative.size <= holdout_per_class or positive.size <= holdout_per_class:
        raise ValueError("Dataset is too small for the requested stratified holdout.")
    validation_indices = jnp.concatenate(
        (negative[:holdout_per_class], positive[:holdout_per_class])
    )
    train_indices = jnp.concatenate(
        (negative[holdout_per_class:], positive[holdout_per_class:])
    )
    feature_schema = phx.ml.FeatureSchema(
        tuple(f"x{index}" for index in range(dataset.features.shape[-1]))
    )
    target_schema = phx.ml.TargetSchema(
        "binary",
        class_labels=(0.0, 1.0),
    )
    training = phx.ml.MLBatch(
        dataset.features[train_indices],
        dataset.targets[train_indices],
        feature_schema=feature_schema,
        target_schema=target_schema,
    )
    return (
        training,
        dataset.features[validation_indices],
        dataset.targets[validation_indices],
    )


def _compiled_case(
    name: str,
    function,
    arguments: tuple[Any, ...],
    /,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(*arguments),
        warmup=warmup,
        repeats=repeats,
    )
    evidence = compiler_evidence(
        compiled.compiled.cost_analysis(),
        compiled.compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "case_id": name,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "compiler": asdict(evidence),
        "logical_input_bytes": logical_array_bytes(arguments),
        "logical_output_bytes": logical_array_bytes(result),
    }


def _predict(model: AbstractArrayModel, features: jax.Array, /) -> jax.Array:
    if model.input_binding().batch_mode == "pointwise":
        return jax.vmap(model)(features)
    return model(features)


def _fit_case(
    name: str,
    recipe: phx.ml.AbstractRecipe,
    training: phx.ml.MLBatch,
    validation_features: jax.Array,
    validation_targets: jax.Array,
    /,
    *,
    key: jax.Array | None,
    score_threshold: float,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    result, execution = measure_repeated(
        lambda: recipe.fit_batch(training, key=key),
        warmup=warmup,
        repeats=repeats,
    )
    model = result.as_trainable()
    scores = _predict(model, validation_features)
    predictions = (scores >= score_threshold).astype(validation_targets.dtype)
    accuracy = jnp.mean(predictions == validation_targets)
    return {
        "case_id": name,
        "fit": execution.to_milliseconds_dict(),
        "valid": bool(result.valid),
        "status": int(result.status),
        "holdout_accuracy": float(accuracy),
        "logical_model_bytes": logical_array_bytes(model),
    }


def _binary_accuracy_scorer(
    predictions: jax.Array,
    targets: jax.Array,
    /,
    *,
    sample_weight: jax.Array,
    mask: jax.Array,
) -> dict[str, jax.Array]:
    weights = jnp.where(mask, sample_weight, 0.0)
    mass = jnp.maximum(jnp.sum(weights), jnp.finfo(weights.dtype).tiny)
    labels = (predictions >= 0.0).astype(targets.dtype)
    return {"accuracy": jnp.sum(weights * (labels == targets)) / mass}


def _run(profile: str, /) -> dict[str, Any]:
    if profile == "smoke":
        sample_count, holdout, warmup, repeats, fit_iterations = 12, 2, 0, 1, 3
    else:
        sample_count, holdout, warmup, repeats, fit_iterations = 32, 4, 1, 3, 12
    dataset = two_curves(jr.key(10), sample_count, noise=0.03)
    training, validation_features, validation_targets = _stratified_holdout(
        dataset,
        holdout_per_class=holdout,
    )
    layout = phx.operators.quantum.HilbertRegisterLayout(("q0", "q1"), (2, 2))
    edges = (("q0", "q1"),)

    state_model = phx.ml.quantum.iqp_state_feature_map(
        layout,
        entanglement_edges=edges,
    )
    fidelity = phx.kernels.ExactQuantumStateFidelityKernel(
        state_model,
        state_model.model_id,
    )
    gram_function = eqx.filter_jit(fidelity.matrix)
    cases = [
        _compiled_case(
            "entangling-iqp-fidelity-gram",
            gram_function,
            (training.features, training.features),
            warmup=warmup,
            repeats=repeats,
        )
    ]

    autodiff_model = phx.ml.quantum.data_reuploading_feature_map(
        2,
        layout,
        1,
        jr.key(21),
        entanglement_edges=edges,
        gradient_method="autodiff",
    )
    shift_model = phx.ml.quantum.data_reuploading_feature_map(
        2,
        layout,
        1,
        jr.key(21),
        entanglement_edges=edges,
        gradient_method="parameter-shift",
    )
    gradient_point = training.dense_features()[0]
    cases.extend(
        (
            _compiled_case(
                "data-reuploading-autodiff-jacobian",
                eqx.filter_jit(jax.jacrev(autodiff_model)),
                (gradient_point,),
                warmup=warmup,
                repeats=repeats,
            ),
            _compiled_case(
                "data-reuploading-parameter-shift-jacobian",
                eqx.filter_jit(jax.jacrev(shift_model)),
                (gradient_point,),
                warmup=warmup,
                repeats=repeats,
            ),
        )
    )

    projected = phx.ml.quantum.projected_iqp_feature_map(
        layout,
        entanglement_edges=edges,
        axes=("Z",),
    )
    projected_pipeline = phx.ml.compose.Pipeline(
        (
            ("scale", phx.ml.preprocessing.MinMaxScaler((-jnp.pi, jnp.pi))),
            (
                "quantum",
                phx.ml.quantum.CircuitFeatureTransformRecipe(
                    projected,
                    output_names=("z_q0", "z_q1"),
                ),
            ),
            (
                "classifier",
                phx.ml.kernel_methods.SupportVectorClassifierRecipe(
                    phx.kernels.SquaredExponentialKernel(length_scale=1.0),
                    iterations=30,
                ),
            ),
        )
    )
    fidelity_pipeline = phx.ml.compose.Pipeline(
        (
            ("scale", phx.ml.preprocessing.MinMaxScaler((-jnp.pi, jnp.pi))),
            (
                "classifier",
                phx.ml.kernel_methods.SupportVectorClassifierRecipe(
                    fidelity,
                    iterations=30,
                ),
            ),
        )
    )
    variational_recipe = phx.ml.quantum.VariationalCircuitClassifierRecipe(
        shift_model,
        max_iterations=fit_iterations,
        learning_rate=0.05,
        tolerance=0.0,
    )
    fits = [
        _fit_case(
            "linear-svc-baseline",
            phx.ml.kernel_methods.SupportVectorClassifierRecipe(
                phx.kernels.LinearKernel(),
                iterations=30,
            ),
            training,
            validation_features,
            validation_targets,
            key=None,
            score_threshold=0.0,
            warmup=warmup,
            repeats=repeats,
        ),
        _fit_case(
            "projected-iqp-rbf-svc",
            projected_pipeline,
            training,
            validation_features,
            validation_targets,
            key=None,
            score_threshold=0.0,
            warmup=warmup,
            repeats=repeats,
        ),
        _fit_case(
            "iqp-fidelity-svc",
            fidelity_pipeline,
            training,
            validation_features,
            validation_targets,
            key=None,
            score_threshold=0.0,
            warmup=warmup,
            repeats=repeats,
        ),
        _fit_case(
            "parameter-shift-data-reuploading-classifier",
            variational_recipe,
            training,
            validation_features,
            validation_targets,
            key=jr.key(31),
            score_threshold=0.5,
            warmup=warmup,
            repeats=repeats,
        ),
    ]
    nested_selection = None
    if profile == "standard":

        def projected_recipe(*, c=1.0):
            return phx.ml.compose.Pipeline(
                (
                    ("scale", phx.ml.preprocessing.MinMaxScaler((-jnp.pi, jnp.pi))),
                    (
                        "quantum",
                        phx.ml.quantum.CircuitFeatureTransformRecipe(
                            projected,
                            output_names=("z_q0", "z_q1"),
                        ),
                    ),
                    (
                        "classifier",
                        phx.ml.kernel_methods.SupportVectorClassifierRecipe(
                            phx.kernels.SquaredExponentialKernel(length_scale=1.0),
                            c=c,
                            iterations=30,
                        ),
                    ),
                )
            )

        nested_result, nested_timing = measure_repeated(
            lambda: phx.ml.model_selection.nested_cross_validate(
                phx.ml.model_selection.GridSearch(
                    {"c": (0.5, 1.0, 2.0)},
                    primary_metric="accuracy",
                ),
                projected_recipe,
                training,
                phx.ml.model_selection.NestedSplitPlan(
                    phx.ml.model_selection.StratifiedKFoldPlan(2),
                    phx.ml.model_selection.StratifiedKFoldPlan(2),
                ),
                _binary_accuracy_scorer,
                key=jr.key(41),
            ),
            warmup=0,
            repeats=1,
        )
        nested_selection = {
            "valid": bool(nested_result.valid),
            "status": int(nested_result.status),
            "outer_fold_count": len(nested_result.folds),
            "inner_candidate_count": 3,
            "outer_accuracy": float(
                nested_result.outer_cross_validation.aggregate_score.value["accuracy"]
            ),
            "execution": nested_timing.to_milliseconds_dict(),
        }
    return {
        "profile": profile,
        "semantic_mode": "exact",
        "representation": "dense",
        "environment": capture_environment().to_dict(),
        "dataset": {
            "dataset_id": dataset.dataset_id,
            "parameters": dataset.parameters,
            "training_samples": training.sample_count,
            "holdout_samples": int(validation_targets.shape[0]),
        },
        "circuit_cases": cases,
        "fit_cases": fits,
        "nested_selection": nested_selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    payload = _run(arguments.profile)
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
