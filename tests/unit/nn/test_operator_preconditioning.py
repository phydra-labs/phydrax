#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


class _ResidualCorrectionOperator(phx.nn.operator.AbstractOperatorModel):
    gain: jax.Array
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self, gain=0.5):
        self.gain = jnp.asarray(gain, dtype=jnp.float32)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("FNO")

    def __call_operator_batch__(self, batch, /, *, key=None):
        del key
        values = batch.input("residual").values
        assert values is not None
        return self.gain * values

    def __call__(self, batch, /, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _operator_task():
    return phx.nn.operator.OperatorTask(
        "linear-residual-correction",
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "residual",
                role="source",
                source_name="residual",
            ),
            phx.nn.operator.OperatorFieldSpec(
                "correction",
                role="target",
                query_name="query",
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


def _template(count=3):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, count),
        quadrature_weights=jnp.full((count,), 1.0 / count),
    )
    samples = phx.nn.operator.FunctionSamples(
        values=jnp.zeros((count,), dtype=jnp.float32),
        axes=(axis,),
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"residual": samples},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=(axis,))},
    )


def _field_space(name, support_id, vector_space):
    count = vector_space.size
    return phx.discretization.DiscreteFieldSpace(
        name,
        support_id,
        phx.discretization.TensorDofLayout(("x",), (count,)),
        vector_space,
        representation="point_value",
        field_space_id=f"{name}-field-space",
    )


def _transfer(source, target, *, transfer_id):
    matrix = jnp.eye(
        source.vector_space.size, dtype=source.vector_space.structure().dtype
    )
    return phx.discretization.FieldTransfer(
        source,
        target,
        phx.linalg.DenseLinearOperator(
            matrix,
            source=source.vector_space,
            target=target.vector_space,
        ),
        transfer_id=transfer_id,
    )


def _binding(*, gain=0.5, artifact_id="correction-artifact", normalization=None):
    template = _template()
    support = template.input("residual").support_id
    vector = phx.linalg.ArraySpace((3,), dtype=jnp.float32)
    solver_space = _field_space("solver", support, vector)
    residual_space = _field_space("model-residual", support, vector)
    correction_space = _field_space("model-correction", support, vector)
    trained = phx.nn.operator.training.TrainedOperator(
        _ResidualCorrectionOperator(gain),
        _operator_task(),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "correction"},
        fixed_query_fingerprints={
            "query": template.query("query").geometry_fingerprint()
        },
        normalization=normalization,
        artifact_id=artifact_id,
    )
    binding = phx.nn.operator.OperatorCorrectionBinding(
        trained,
        template,
        solver_space,
        residual_space,
        correction_space,
        _transfer(solver_space, residual_space, transfer_id="residual-transfer"),
        _transfer(correction_space, solver_space, transfer_id="correction-transfer"),
        residual_source_name="residual",
        correction_field_name="correction",
        condition_ids=("problem:diagonal", "boundary:homogeneous"),
    )
    return binding


def test_prepared_correction_preserves_physical_normalization_and_is_jittable():
    normalizer = phx.nn.operator.training.AffineNormalizer(
        mean=jnp.asarray(0.0),
        scale=jnp.asarray(2.0),
        channel_axis=None,
        epsilon=1e-6,
    )
    normalization = phx.nn.operator.training.OperatorNormalizationPolicy(
        input_values={"residual": normalizer},
        targets={"correction": normalizer},
        input_coordinates={},
        query_coordinates={},
    )
    prepared = _binding(normalization=normalization).prepare()
    residual = jnp.asarray([2.0, -4.0, 1.0], dtype=jnp.float32)

    eager = prepared.apply(residual)
    compiled = jax.jit(lambda value: prepared.apply(value))(residual)

    assert jnp.allclose(eager, 0.5 * residual)
    assert jnp.allclose(compiled, eager)
    assert jnp.array_equal(
        prepared.prepared_input.physical_batch.input("residual").values,
        jnp.zeros_like(residual),
    )


def test_direct_correction_is_fgmres_only_and_original_residual_is_authoritative():
    binding = _binding(gain=0.4)
    builder = phx.nn.operator.TrainedOperatorPreconditionerBuilder(
        binding,
        phx.nn.operator.OperatorCorrectionCost(
            preparation_workspace_bytes=64,
            inference_workspace_bytes_per_rhs=128,
        ),
    )
    matrix = jnp.diag(jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float32))
    operator = phx.linalg.DenseLinearOperator(
        matrix,
        source=binding.solver_space.vector_space,
        target=binding.solver_space.vector_space,
    )
    problem = phx.linalg.LinearSystem(operator)
    policy = phx.linalg.LinearSolvePolicy(
        phx.linalg.FGMRES(restart=3),
        preconditioning=phx.linalg.PreconditioningPolicy(builder, side="right"),
        differentiation=phx.linalg.DifferentiationPolicy("none"),
    )
    right_hand_side = jnp.asarray([1.0, -2.0, 3.0], dtype=jnp.float32)

    result = phx.linalg.solve(problem, right_hand_side, policy=policy)

    assert bool(result.successful)
    assert jnp.linalg.norm(operator.mv(result.value) - right_hand_side) < 2e-5
    properties = builder.properties_for(operator)
    assert properties.certifies("stationary")
    assert not properties.linear
    assert builder.cost_for(operator).apply_workspace_bytes_per_rhs == 128
    with pytest.raises(ValueError, match="linear|stationary"):
        phx.linalg.plan(
            problem,
            phx.linalg.LinearSolvePolicy(
                phx.linalg.GMRES(restart=3),
                preconditioning=phx.linalg.PreconditioningPolicy(builder),
                differentiation=phx.linalg.DifferentiationPolicy("none"),
            ),
        )


def test_binding_identity_tracks_artifact_and_rejects_wrong_transfer_direction():
    first = _binding(artifact_id="artifact-a")
    second = _binding(artifact_id="artifact-b")
    assert first.binding_id != second.binding_id

    with pytest.raises(ValueError, match="Residual transfer"):
        phx.nn.operator.OperatorCorrectionBinding(
            first.trained_operator,
            first.template,
            first.solver_space,
            first.model_residual_space,
            first.model_correction_space,
            first.correction_transfer,
            first.correction_transfer,
            residual_source_name="residual",
            correction_field_name="correction",
            condition_ids=("problem:diagonal",),
        )


def test_operator_basis_lowers_with_hilbert_adjoint_and_rejects_dependent_modes():
    sample = phx.nn.operator.FunctionSamples(
        values=jnp.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            dtype=jnp.float32,
        ),
        coordinates=jnp.arange(3.0)[:, None],
    )
    model_vector = phx.linalg.ArraySpace((3,), dtype=jnp.float32)
    solver_vector = phx.linalg.ArraySpace(
        (3,),
        dtype=jnp.float32,
        pairing=phx.linalg.DiagonalPairing(
            jnp.asarray([1.0, 2.0, 4.0], dtype=jnp.float32)
        ),
    )
    model_space = _field_space("basis-model", sample.support_id, model_vector)
    solver_space = _field_space("basis-solver", sample.support_id, solver_vector)
    transfer = _transfer(model_space, solver_space, transfer_id="basis-transfer")

    prepared = phx.nn.operator.prepare_operator_subspace_correction(
        sample,
        model_space,
        solver_space,
        transfer,
        phx.linalg.DenseInversePreconditionerBuilder(),
    )
    residual = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)

    assert jnp.allclose(
        prepared.term.restriction.mv(residual),
        jnp.asarray([1.0, 2.0], dtype=jnp.float32),
    )
    assert prepared.transferred_subspace.space.compatible(solver_vector)
    dependent = phx.nn.operator.function_samples_with_values(
        sample,
        jnp.asarray(
            [[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
            dtype=jnp.float32,
        ),
    )
    with pytest.raises(Exception, match="linearly independent"):
        phx.nn.operator.prepare_operator_subspace_correction(
            dependent,
            model_space,
            solver_space,
            transfer,
            phx.linalg.DenseInversePreconditionerBuilder(),
        )
