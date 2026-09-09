#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


class _DiagonalInverseOperator(phx.nn.operator.AbstractOperatorModel):
    inverse_diagonal: jax.Array
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, inverse_diagonal):
        self.inverse_diagonal = jnp.asarray(inverse_diagonal, dtype=jnp.float32)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("FNO")

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        residual = batch.input("residual").values
        assert residual is not None
        return self.inverse_diagonal * residual

    def __call__(self, batch, /, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _field_space(name, support_id, vector_space):
    return phx.discretization.DiscreteFieldSpace(
        name,
        support_id,
        phx.discretization.TensorDofLayout(("x",), (4,)),
        vector_space,
        representation="point_value",
        field_space_id=f"{name}-field-space",
    )


def test_transferred_operator_correction_solves_and_certifies_original_system():
    task = phx.nn.operator.OperatorTask(
        "transferred-linear-correction",
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "residual", role="source", source_name="residual"
            ),
            phx.nn.operator.OperatorFieldSpec(
                "correction", role="target", query_name="query"
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                fixed_geometry=True,
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=True,
        ),
    )
    axis = phx.nn.operator.OperatorAxis("x", jnp.arange(4.0))
    template = phx.nn.operator.OperatorBatch(
        inputs={
            "residual": phx.nn.operator.FunctionSamples(
                values=jnp.zeros((4,), dtype=jnp.float32),
                axes=(axis,),
            )
        },
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
    )
    model_support = template.input("residual").support_id
    solver_vector = phx.linalg.ArraySpace(
        (4,), dtype=jnp.float32, space_id="fine-solver-vectors"
    )
    model_vector = phx.linalg.ArraySpace(
        (4,), dtype=jnp.float32, space_id="model-correction-vectors"
    )
    solver_space = _field_space("solver", "fine-support", solver_vector)
    residual_space = _field_space("model-residual", model_support, model_vector)
    correction_space = _field_space("model-correction", model_support, model_vector)
    permutation = jnp.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    residual_transfer = phx.discretization.FieldTransfer(
        solver_space,
        residual_space,
        phx.linalg.DenseLinearOperator(
            permutation, source=solver_vector, target=model_vector
        ),
        transfer_id="fine-to-model-residual",
    )
    correction_transfer = phx.discretization.FieldTransfer(
        correction_space,
        solver_space,
        phx.linalg.DenseLinearOperator(
            permutation.T, source=model_vector, target=solver_vector
        ),
        transfer_id="model-to-fine-correction",
    )
    inverse_in_model_order = jnp.asarray([1.0 / 3.0, 1.0, 0.25, 0.5])
    trained = phx.nn.operator.training.TrainedOperator(
        _DiagonalInverseOperator(inverse_in_model_order),
        task,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "correction"},
        fixed_query_fingerprints={
            "query": template.query("query").geometry_fingerprint()
        },
        artifact_id="transferred-inverse-artifact",
    )
    binding = phx.nn.operator.OperatorCorrectionBinding(
        trained,
        template,
        solver_space,
        residual_space,
        correction_space,
        residual_transfer,
        correction_transfer,
        residual_source_name="residual",
        correction_field_name="correction",
        condition_ids=("problem:diagonal", "transfer:permutation"),
    )
    builder = phx.nn.operator.TrainedOperatorPreconditionerBuilder(
        binding,
        phx.nn.operator.OperatorCorrectionCost(
            preparation_workspace_bytes=0,
            inference_workspace_bytes_per_rhs=512,
        ),
    )
    diagonal = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float32)
    operator = phx.linalg.DiagonalLinearOperator(diagonal, space=solver_vector)
    right_hand_side = jnp.asarray([2.0, -4.0, 3.0, 8.0], dtype=jnp.float32)
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=4),
        preconditioning=phx.linalg.PreconditioningPolicy(builder, side="right"),
        differentiation=phx.linalg.DifferentiationPolicy("none"),
    )

    result = phx.linalg.solve(
        phx.linalg.LinearSystem(operator),
        right_hand_side,
        policy=policy,
    )
    original_residual = right_hand_side - operator.mv(result.value)

    assert bool(result.successful)
    assert jnp.linalg.norm(original_residual) < 2e-5
    assert jnp.allclose(result.value, right_hand_side / diagonal, atol=2e-5)
