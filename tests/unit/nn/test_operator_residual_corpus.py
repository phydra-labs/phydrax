#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _case(index, *, identities=None, values=None, mask=None):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, 3),
        quadrature_weights=jnp.full((3,), 1.0 / 3.0),
    )
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "residual": phx.nn.operator.FunctionSamples(
                values=(
                    jnp.asarray([index + 1.0, index + 2.0, index + 3.0])
                    if values is None
                    else jnp.asarray(values)
                ),
                axes=(axis,),
                mask=mask,
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(axis,),
                mask=mask,
            )
        },
    )
    return phx.nn.operator.OperatorCase(
        batch,
        phx.nn.operator.OperatorTargetBatch({}),
        provenance=phx.nn.operator.OperatorCaseProvenance(
            f"case-{index}",
            identities=(
                {
                    "solver_execution_id": f"solve-{index}",
                    "operator_id": "operator-a",
                }
                if identities is None
                else identities
            ),
            order={"iteration": float(index), "residual_norm": float(index + 1)},
        ),
    )


def _corpus(cases, *, binding_id="binding-a", loss_id="loss-a"):
    return phx.nn.operator.training.prepare_operator_residual_corpus(
        cases,
        binding_id=binding_id,
        task_fingerprint="task-a",
        residual_loss_fingerprint=loss_id,
        source_artifact_ids=("solver-artifact-a",),
        required_identity_keys=("solver_execution_id", "operator_id"),
    )


def test_residual_corpus_is_targetless_provenance_complete_and_content_addressed():
    cases = (_case(0), _case(1))

    first = _corpus(cases)
    second = _corpus(cases)

    assert first.dataset.size == 2
    assert not first.dataset.targets.fields
    assert first.corpus_id == second.corpus_id
    assert first.training_provenance["operator_residual_corpus_id"] == first.corpus_id
    assert first.training_provenance["operator_correction_binding_id"] == "binding-a"
    assert _corpus(cases, binding_id="binding-b").corpus_id != first.corpus_id
    assert _corpus(cases, loss_id="loss-b").corpus_id != first.corpus_id


def test_residual_corpus_rejects_missing_identity_and_nonfinite_active_values():
    missing = _case(0, identities={"solver_execution_id": "solve-0"})
    with pytest.raises(ValueError, match="missing.*operator_id"):
        _corpus((missing,))

    invalid = _case(0, values=[1.0, jnp.nan, 3.0])
    with pytest.raises(ValueError, match="non-finite active"):
        _corpus((invalid,))

    masked = _case(
        0,
        values=[1.0, jnp.nan, 3.0],
        mask=jnp.asarray([True, False, True]),
    )
    corpus = _corpus((masked,))
    assert corpus.dataset.size == 1
