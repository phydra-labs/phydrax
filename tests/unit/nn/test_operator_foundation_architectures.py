#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import opt_einsum as oe
import pytest
import trimesh

import phydrax as phx
import phydrax._spectral as multiresolution
import phydrax.discretization as spectral


def _assert_finite_model_gradient(model, loss):
    value, gradient = eqx.filter_value_and_grad(loss)(model)
    leaves = [
        leaf for leaf in jax.tree_util.tree_leaves(gradient) if eqx.is_inexact_array(leaf)
    ]
    assert jnp.isfinite(value)
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def _point_batch(
    values,
    *,
    source_mask=None,
    query_mask=None,
    source_name="u",
    query_coordinates=None,
):
    source_coordinates = jnp.array([[0.0], [0.3], [0.7], [1.0]])
    if query_coordinates is None:
        query_coordinates = jnp.array([[0.1], [0.5], [0.9]])
    if source_mask is None:
        source_mask = jnp.ones((4,), dtype=bool)
    if query_mask is None:
        query_mask = jnp.ones((query_coordinates.shape[0],), dtype=bool)
    return phx.nn.operator.OperatorBatch(
        inputs={
            source_name: phx.nn.operator.FunctionSamples(
                values=jnp.asarray(values),
                coordinates=source_coordinates,
                quadrature_weights=jnp.full((4,), 0.25),
                mask=source_mask,
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=query_coordinates,
                mask=query_mask,
            )
        },
    )


def _case_point_batch(values):
    source_mask = jnp.array([[True, True, True, False], [True, True, False, False]])
    query_mask = jnp.array([[True, True, False], [True, False, False]])
    return phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.asarray(values),
                coordinates=jnp.array([[0.0], [0.3], [0.7], [1.0]]),
                quadrature_weights=jnp.full((4,), 0.25),
                mask=source_mask,
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.array([[0.1], [0.5], [0.9]]),
                mask=query_mask,
            )
        },
        case_axes=("case",),
    )


def _grid_batch(values, *, query_mask=None, source_name="state"):
    values = jnp.asarray(values)
    size = int(values.shape[0])
    points = jnp.linspace(0.0, 1.0, size, endpoint=False)
    axis = phx.nn.operator.OperatorAxis(
        "x",
        points,
        quadrature_weights=jnp.full((size,), 1.0 / size),
        periodic=True,
    )
    if query_mask is None:
        query_mask = jnp.ones((size,), dtype=bool)
    return phx.nn.operator.OperatorBatch(
        inputs={
            source_name: phx.nn.operator.FunctionSamples(
                values=values,
                axes=(axis,),
                mask=jnp.ones((size,), dtype=bool),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                axes=(axis,),
                mask=query_mask,
            )
        },
    )


@pytest.fixture
def masked_point_batch():
    return _point_batch(
        jnp.sin(jnp.pi * jnp.array([0.0, 0.3, 0.7, 1.0])),
        source_mask=jnp.array([True, True, True, False]),
        query_mask=jnp.array([True, True, False]),
    )


def test_coordinate_conditioned_operator_film_decode_is_masked_jittable_and_finite(
    masked_point_batch,
):
    branch = phx.nn.operator.architectures.IntegralBranchEncoder(
        feature_model=phx.nn.models.MLP(
            in_size=2,
            out_size=4,
            width_size=8,
            depth=1,
            key=jr.key(0),
        ),
        latent_size=4,
        coord_dim=1,
    )
    decoder = phx.nn.operator.architectures.FiLMCoordinateDecoder(
        latent_size=4,
        coord_dim=1,
        out_size="scalar",
        width=8,
        depth=2,
        key=jr.key(1),
    )
    model = phx.nn.operator.architectures.CoordinateConditionedOperator(
        branch={"u": branch},
        decoder=decoder,
        coord_dim=1,
        latent_size=4,
        source_key="u",
    )
    assert model.operator_contract.capabilities.encode_once_decode_many

    eager = model(masked_point_batch)
    compiled = eqx.filter_jit(lambda item, batch: item(batch))(model, masked_point_batch)
    prediction = model.predict(masked_point_batch)

    assert eager.shape == (3,)
    assert jnp.all(jnp.isfinite(eager))
    assert jnp.allclose(compiled, eager)
    assert eager[-1] == 0.0
    output = prediction.field("output")
    assert prediction.query_geometry(output.query_name) is masked_point_batch.query(
        "query"
    )
    assert output.spec.channels == "scalar"
    _assert_finite_model_gradient(
        model, lambda item: jnp.sum(item(masked_point_batch) ** 2)
    )


def test_wavelet_operators_reconstruct_and_execute_scalar_and_channel_fields():
    scalar_values = jnp.sin(2.0 * jnp.pi * jnp.arange(8) / 8.0)
    channel_values = jnp.stack((scalar_values, jnp.cos(scalar_values)), axis=-1)
    query_mask = jnp.array([True, True, True, True, True, True, True, False])

    wavelet = multiresolution.DiscreteWaveletTransform(
        (-2,), levels=2, wavelet="db2", boundary="periodization"
    )
    multiwavelet = multiresolution.AlpertMultiwaveletTransform(
        order=2, levels=2, boundary="periodization"
    )
    assert jnp.allclose(
        wavelet.synthesis(wavelet.analysis(channel_values)),
        channel_values,
        rtol=1e-5,
        atol=1e-5,
    )
    assert jnp.allclose(
        multiwavelet.synthesis(multiwavelet.analysis(channel_values)),
        channel_values,
        rtol=1e-5,
        atol=1e-5,
    )

    wno = phx.nn.operator.architectures.WaveletNeuralOperator(
        1,
        in_channels=2,
        out_channels=2,
        levels=2,
        wavelet="haar",
        width=4,
        depth=1,
        source_key="state",
        key=jr.key(2),
    )
    mwt = phx.nn.operator.architectures.MultiwaveletOperator(
        in_channels="scalar",
        out_channels="scalar",
        order=2,
        levels=2,
        width=4,
        depth=1,
        source_key="state",
        key=jr.key(3),
    )
    channel_batch = _grid_batch(channel_values, query_mask=query_mask)
    scalar_batch = _grid_batch(scalar_values, query_mask=query_mask)

    wno_eager = wno(channel_batch)
    mwt_eager = mwt(scalar_batch)
    wno_jit = eqx.filter_jit(lambda item, batch: item(batch))(wno, channel_batch)
    mwt_jit = eqx.filter_jit(lambda item, batch: item(batch))(mwt, scalar_batch)

    assert wno_eager.shape == (8, 2)
    assert mwt_eager.shape == (8,)
    assert jnp.all(jnp.isfinite(wno_eager))
    assert jnp.all(jnp.isfinite(mwt_eager))
    assert jnp.allclose(wno_jit, wno_eager)
    assert jnp.allclose(mwt_jit, mwt_eager)
    assert jnp.array_equal(wno_eager[-1], jnp.zeros((2,)))
    assert mwt_eager[-1] == 0.0
    _assert_finite_model_gradient(wno, lambda item: jnp.sum(item(channel_batch) ** 2))
    _assert_finite_model_gradient(mwt, lambda item: jnp.sum(item(scalar_batch) ** 2))


def test_wavelet_operators_reuse_one_model_across_resolutions():
    sizes = (17, 29)
    batches = tuple(
        _grid_batch(jnp.sin(2.0 * jnp.pi * jnp.arange(size, dtype=float) / size))
        for size in sizes
    )
    wno = phx.nn.operator.architectures.WaveletNeuralOperator(
        1,
        in_channels="scalar",
        out_channels="scalar",
        levels=2,
        wavelet="db2",
        width=4,
        depth=1,
        source_key="state",
        key=jr.key(51),
    )
    mwt = phx.nn.operator.architectures.MultiwaveletOperator(
        in_channels="scalar",
        out_channels="scalar",
        order=3,
        levels=2,
        width=4,
        depth=1,
        source_key="state",
        key=jr.key(52),
    )
    compiled = eqx.filter_jit(lambda model, data: model(data))

    wno_outputs = tuple(compiled(wno, batch) for batch in batches)
    mwt_outputs = tuple(compiled(mwt, batch) for batch in batches)

    assert tuple(output.shape for output in wno_outputs) == ((17,), (29,))
    assert tuple(output.shape for output in mwt_outputs) == ((17,), (29,))
    assert all(jnp.all(jnp.isfinite(output)) for output in wno_outputs)
    assert all(jnp.all(jnp.isfinite(output)) for output in mwt_outputs)


def test_manifold_spectral_operator_runs_valid_small_laplacian_plan():
    laplacian = np.array(
        [
            [2.0, -1.0, 0.0, -1.0],
            [-1.0, 2.0, -1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0],
            [-1.0, 0.0, -1.0, 2.0],
        ]
    )
    plan = spectral.SpectralDecomposition.from_stiffness(
        laplacian, np.ones((4,)), n_modes=4, decomposition_id="cycle-4"
    )
    coordinates = jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.array([1.0, 0.0, -1.0, 0.0]),
                coordinates=coordinates,
                mask=jnp.array([True, True, True, False]),
            )
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=coordinates,
                mask=jnp.array([True, True, False, False]),
            )
        },
    )
    model = phx.nn.operator.architectures.ManifoldSpectralOperator(
        plan,
        width=4,
        depth=1,
        source_key="u",
        key=jr.key(4),
    )

    eager = model(batch)
    compiled = eqx.filter_jit(lambda item, data: item(data))(model, batch)

    assert plan.analysis.shape == (4, 4)
    assert jnp.allclose(plan.analysis @ plan.synthesis, jnp.eye(4), atol=1e-5)
    assert eager.shape == (4,)
    assert jnp.all(jnp.isfinite(eager))
    assert jnp.allclose(compiled, eager)
    assert jnp.array_equal(eager[2:], jnp.zeros((2,)))
    _assert_finite_model_gradient(model, lambda item: jnp.sum(item(batch) ** 2))


def test_stiffness_plan_rejects_negative_semidefinite_operator():
    differential_laplacian = np.array(
        [
            [-2.0, 1.0, 0.0, 1.0],
            [1.0, -2.0, 1.0, 0.0],
            [0.0, 1.0, -2.0, 1.0],
            [1.0, 0.0, 1.0, -2.0],
        ]
    )

    with pytest.raises(ValueError, match="positive semidefinite"):
        spectral.SpectralDecomposition.from_stiffness(
            differential_laplacian,
            np.ones((4,)),
            n_modes=4,
        )


def test_triangle_mesh_plan_preserves_sparse_sphere_eigenspace_multiplicities():
    mesh = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
    triangle_mesh = phx.geometry.simplicial.TriangleMesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
    )
    plan = phx.graph.spectral_discretization_from_triangle_mesh(
        triangle_mesh,
        n_modes=9,
    )
    eigenvalues = np.asarray(plan.eigenvalues)

    assert plan.analysis.shape == (9, mesh.vertices.shape[0])
    assert plan.synthesis.shape == (mesh.vertices.shape[0], 9)
    assert np.allclose(plan.analysis @ plan.synthesis, np.eye(9), atol=1e-6)
    assert np.isclose(eigenvalues[0], 0.0, atol=1e-8)
    assert np.allclose(eigenvalues[1:4], 2.0, rtol=0.02)
    assert np.allclose(eigenvalues[4:9], 6.0, rtol=0.02)
    assert np.array_equal(
        np.bincount(np.asarray(plan.group_ids)),
        np.array([1, 3, 5]),
    )
    assert np.ptp(np.asarray(plan.synthesis)[:, 0]) < 1e-8


def test_upt_and_abupt_preserve_case_and_source_query_masks():
    values = jnp.array([[0.0, 0.5, 1.0, 1000.0], [1.0, 0.5, -1000.0, 2000.0]])
    changed_padding = values.at[0, 3].set(-9000.0).at[1, 2:].set(7000.0)
    batch = _case_point_batch(values)
    changed_batch = _case_point_batch(changed_padding)
    expected_query_mask = batch.query("query").mask_array(case_shape=(2,))

    upt = phx.nn.operator.architectures.UPT(
        in_channels="scalar",
        out_channels="scalar",
        coord_dim=1,
        width=4,
        num_tokens=2,
        depth=1,
        num_heads=1,
        source_key="u",
        key=jr.key(5),
    )
    upt_state = upt.encode_inputs(batch)
    upt_output = upt(batch)
    upt_compiled = eqx.filter_jit(lambda item, data: item(data))(upt, batch)

    assert upt_state.case_shape == (2,)
    assert upt_state.values.shape == (2, 2, 4)
    assert jnp.array_equal(upt_state.mask, jnp.ones((2, 2), dtype=bool))
    assert upt_output.shape == (2, 3)
    assert jnp.array_equal(upt_output[~expected_query_mask], jnp.zeros((3,)))
    assert jnp.allclose(upt_output, upt(changed_batch))
    assert jnp.allclose(upt_compiled, upt_output)

    graph = phx.nn.operator.OperatorBranchGraph(
        (
            phx.nn.operator.OperatorBranchSpec(
                "field",
                role="both",
                geometry_kind="point_cloud",
                source_name="u",
                output_spec=phx.nn.operator.OperatorOutputSpec("scalar"),
                query_name="query",
            ),
        )
    )
    abupt = phx.nn.operator.architectures.ABUPT(
        graph,
        input_channels={"field": "scalar"},
        coord_dims={"field": 1},
        anchor_counts={"field": 3},
        width=4,
        depth=1,
        num_heads=1,
        key=jr.key(6),
    )
    abupt_state = abupt.encode_inputs(batch)
    branch_state = abupt_state.branch("field")
    abupt_output = abupt(batch)
    abupt_compiled = eqx.filter_jit(lambda item, data: item(data))(abupt, batch)
    abupt_prediction = abupt.predict(batch)
    selected = jnp.rint(jnp.linspace(0, 3, 3)).astype(jnp.int32)
    expected_anchor_mask = jnp.take(
        batch.input("u").mask_array(case_shape=(2,)), selected, axis=-1
    )

    assert abupt_state.case_shape == (2,)
    assert branch_state.values.shape == (2, 3, 4)
    assert jnp.array_equal(branch_state.mask, expected_anchor_mask)
    assert abupt_output.shape == (2, 3)
    assert tuple(abupt_prediction.fields) == ("field",)
    assert abupt_prediction.field("field").query_name == "query"
    assert jnp.array_equal(abupt_output[~expected_query_mask], jnp.zeros((3,)))
    assert jnp.allclose(abupt_output, abupt(changed_batch))
    assert jnp.allclose(abupt_compiled, abupt_output)
    _assert_finite_model_gradient(upt, lambda item: jnp.sum(item(batch) ** 2))
    _assert_finite_model_gradient(abupt, lambda item: jnp.sum(item(batch) ** 2))


def test_abupt_predicts_named_fields_on_distinct_queries():
    source_coordinates = jnp.linspace(0.0, 1.0, 4)[:, None]
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "u": phx.nn.operator.FunctionSamples(
                values=jnp.linspace(-1.0, 1.0, 4),
                coordinates=source_coordinates,
            )
        },
        queries={
            "spatial": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.linspace(0.0, 1.0, 3)[:, None],
            ),
            "sensors": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=jnp.asarray([[0.25], [0.75]]),
            ),
        },
    )
    graph = phx.nn.operator.OperatorBranchGraph(
        (
            phx.nn.operator.OperatorBranchSpec(
                "state",
                role="both",
                geometry_kind="point_cloud",
                source_name="u",
                query_name="spatial",
                output_spec=phx.nn.operator.OperatorOutputSpec("scalar"),
            ),
            phx.nn.operator.OperatorBranchSpec(
                "flux",
                role="both",
                geometry_kind="point_cloud",
                source_name="u",
                query_name="sensors",
                output_spec=phx.nn.operator.OperatorOutputSpec(
                    2,
                    component_names=("x", "y"),
                ),
            ),
        )
    )
    model = phx.nn.operator.architectures.ABUPT(
        graph,
        input_channels={"state": "scalar", "flux": "scalar"},
        coord_dims={"state": 1, "flux": 1},
        anchor_counts={"state": 2, "flux": 2},
        width=4,
        depth=1,
        num_heads=1,
        key=jr.key(61),
    )

    prediction = model.predict(batch)

    assert tuple(prediction.fields) == ("state", "flux")
    assert tuple(prediction.queries) == ("spatial", "sensors")
    assert prediction.field("state").values.shape == (3,)
    assert prediction.field("flux").values.shape == (2, 2)
    assert prediction.field("flux").spec.component_names == ("x", "y")


def test_codano_executes_heterogeneous_typed_fields_and_exact_query_mask():
    coordinates = jnp.array([[0.0], [0.3], [0.7], [1.0]])
    query_mask = jnp.array([True, True, True, False])
    batch = phx.nn.operator.OperatorBatch(
        inputs={
            "pressure": phx.nn.operator.FunctionSamples(
                values=jnp.array([1.0, 0.5, -0.5, -1.0]),
                coordinates=coordinates,
                quadrature_weights=jnp.full((4,), 0.25),
                mask=jnp.array([True, True, True, False]),
            ),
            "velocity": phx.nn.operator.FunctionSamples(
                values=jnp.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]),
                coordinates=coordinates,
                quadrature_weights=jnp.full((4,), 0.25),
                mask=jnp.array([True, True, False, False]),
            ),
        },
        queries={
            "query": phx.nn.operator.FunctionSamples(
                values=None,
                coordinates=coordinates,
                mask=query_mask,
            )
        },
    )
    fields = (
        phx.nn.operator.OperatorFieldSpec(
            "pressure", channels="scalar", role="source", scale=2.0
        ),
        phx.nn.operator.OperatorFieldSpec(
            "velocity",
            channels=2,
            role="both",
            component_names=("vx", "vy"),
            query_name="query",
            scale=(2.0, 3.0),
        ),
    )
    model = phx.nn.operator.architectures.CoDANO(
        fields,
        (4,),
        n_modes=2,
        width=4,
        depth=1,
        num_heads=1,
        head_dim=4,
        key=jr.key(7),
    )

    state = model.encode_inputs(batch)
    eager = model(batch)
    compiled = eqx.filter_jit(lambda item, data: item(data))(model, batch)
    prediction = model.predict(batch)

    assert model.in_size == (1, 2)
    assert model.out_size == 2
    assert state.values.shape == (4, 2, 4)
    assert jnp.array_equal(state.field_mask, jnp.array([True, True]))
    assert eager.shape == (4, 2)
    assert jnp.all(jnp.isfinite(eager))
    assert tuple(prediction.fields) == ("velocity",)
    assert prediction.field("velocity").spec.component_names == ("vx", "vy")
    assert jnp.allclose(compiled, eager)
    assert jnp.array_equal(eager[-1], jnp.zeros((2,)))
    _assert_finite_model_gradient(model, lambda item: jnp.sum(item(batch) ** 2))


def test_eqgino_is_rotation_and_reflection_equivariant():
    representation = phx.nn.operator.representations.O3Representation(vectors=1)
    model = phx.nn.operator.architectures.EqGINO(
        representation,
        representation,
        radius=3.0,
        radial_basis_size=3,
        depth=1,
        source_key="field",
        key=jr.key(8),
    )
    source_coordinates = jnp.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    query_coordinates = jnp.array([[0.2, 0.1, 0.0], [0.0, 0.4, 0.3], [0.3, 0.2, 0.1]])
    source_values = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [4.0, 4.0, 4.0]]
    )
    source_mask = jnp.array([True, True, True, False])
    query_mask = jnp.array([True, True, False])

    def batch_for(coordinates, values, queries):
        return phx.nn.operator.OperatorBatch(
            inputs={
                "field": phx.nn.operator.FunctionSamples(
                    values=values,
                    coordinates=coordinates,
                    quadrature_weights=jnp.full((4,), 0.25),
                    mask=source_mask,
                )
            },
            queries={
                "query": phx.nn.operator.FunctionSamples(
                    values=None,
                    coordinates=queries,
                    mask=query_mask,
                )
            },
        )

    batch = batch_for(source_coordinates, source_values, query_coordinates)
    reference = model(batch)
    compiled = eqx.filter_jit(lambda item, data: item(data))(model, batch)
    transforms = (
        jnp.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        jnp.diag(jnp.array([-1.0, 1.0, 1.0])),
    )

    for orthogonal in transforms:
        transformed_batch = batch_for(
            oe.contract("ij,pj->pi", orthogonal, source_coordinates),
            representation.transform(source_values, orthogonal),
            oe.contract("ij,pj->pi", orthogonal, query_coordinates),
        )
        transformed = model(transformed_batch)
        expected = representation.transform(reference, orthogonal)
        assert jnp.allclose(transformed, expected, rtol=2e-4, atol=2e-5)

    assert reference.shape == (3, 3)
    assert jnp.all(jnp.isfinite(reference))
    assert jnp.allclose(compiled, reference)
    assert jnp.array_equal(reference[-1], jnp.zeros((3,)))
    _assert_finite_model_gradient(model, lambda item: jnp.sum(item(batch) ** 2))


def test_in_context_operator_prompt_mask_and_permutation_are_semantic():
    query_batch = _point_batch(
        jnp.array([0.0, 0.2, 0.6, 1000.0]),
        source_mask=jnp.array([True, True, True, False]),
        query_mask=jnp.array([True, True, False]),
    )
    first_batch = _point_batch(jnp.array([0.0, 0.3, 0.7, 1.0]))
    second_batch = _point_batch(jnp.array([1.0, 0.7, 0.3, 0.0]))
    masked_batch = _point_batch(jnp.full((4,), 2.0))
    changed_masked_batch = _point_batch(jnp.full((4,), 9000.0))
    first = phx.nn.operator.OperatorSupervisedExample(
        first_batch, jnp.array([0.1, 0.5, 0.9])
    )
    second = phx.nn.operator.OperatorSupervisedExample(
        second_batch, jnp.array([-0.1, -0.5, -0.9])
    )
    masked = phx.nn.operator.OperatorSupervisedExample(masked_batch, jnp.full((3,), 3.0))
    changed_masked = phx.nn.operator.OperatorSupervisedExample(
        changed_masked_batch, jnp.full((3,), -8000.0)
    )
    prompt = phx.nn.operator.OperatorPrompt(
        (first, second, masked), mask=jnp.array([True, True, False])
    )
    changed_prompt = phx.nn.operator.OperatorPrompt(
        (first, second, changed_masked), mask=jnp.array([True, True, False])
    )
    model = phx.nn.operator.architectures.InContextOperator(
        in_channels="scalar",
        out_channels="scalar",
        coord_dim=1,
        width=4,
        num_tokens=2,
        prompt_depth=1,
        processor_depth=1,
        num_heads=1,
        source_key="u",
        key=jr.key(9),
    )

    prompted = phx.nn.operator.PromptedOperatorBatch(query_batch, prompt)
    reference = model(prompted)
    permuted = model(
        phx.nn.operator.PromptedOperatorBatch(query_batch, prompt.permute((2, 0, 1)))
    )
    changed_masked_output = model(
        phx.nn.operator.PromptedOperatorBatch(query_batch, changed_prompt)
    )
    compiled = eqx.filter_jit(lambda item, data: item(data))(model, prompted)
    prompt_state = model.encode_prompt(prompt)

    assert reference.shape == (3,)
    assert jnp.all(jnp.isfinite(reference))
    assert reference[-1] == 0.0
    assert jnp.allclose(permuted, reference, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(changed_masked_output, reference, rtol=1e-5, atol=1e-6)
    assert jnp.allclose(compiled, reference)
    assert jnp.array_equal(
        prompt_state.mask,
        jnp.array(
            [True, True, True, True, True, True, True, True, False, False, False, False]
        ),
    )
    _assert_finite_model_gradient(model, lambda item: jnp.sum(item(prompted) ** 2))


def test_gaussian_function_operator_has_coherent_shape_sampling_and_masked_nll():
    values = jnp.sin(2.0 * jnp.pi * jnp.arange(8) / 8.0)
    query_mask = jnp.array([True, True, True, True, True, True, True, False])
    batch = _grid_batch(values, query_mask=query_mask)
    base = phx.nn.operator.architectures.FNO(
        n_modes=(3,),
        in_channels="scalar",
        out_channels=3,
        width=4,
        depth=1,
        coordinate_embedding=False,
        source_key="state",
        key=jr.key(10),
    )
    model = phx.nn.operator.architectures.GaussianFunctionOperator(
        base, out_channels="scalar", factor_rank=1, min_scale=1e-3
    )

    distribution = model.distribution(batch)
    samples = model.sample(batch, num_samples=3, key=jr.key(11))
    nll = phx.nn.operator.architectures.gaussian_operator_nll(
        model, batch, distribution.mean, reduction="none"
    )
    changed_target = distribution.mean.at[-1].set(1e9)
    changed_nll = phx.nn.operator.architectures.gaussian_operator_nll(
        model, batch, changed_target, reduction="none"
    )
    covariance = distribution.dense_covariance()
    factor = distribution.factors[:, 0]
    expected_covariance = jnp.diag(
        jnp.where(query_mask, distribution.scale**2, 0.0)
    ) + jnp.outer(factor * query_mask, factor * query_mask)
    compiled_mean = eqx.filter_jit(lambda item, data: item(data))(model, batch)

    assert distribution.mean.shape == (8,)
    assert distribution.scale.shape == (8,)
    assert distribution.factors.shape == (8, 1)
    assert distribution.event_shape == (8,)
    assert samples.shape == (3, 8)
    assert jnp.array_equal(samples[:, -1], jnp.zeros((3,)))
    assert covariance.shape == (8, 8)
    assert jnp.allclose(covariance, expected_covariance)
    assert jnp.max(jnp.abs(covariance - jnp.diag(jnp.diag(covariance)))) > 1e-8
    assert jnp.all(jnp.isfinite(samples))
    assert jnp.all(jnp.isfinite(nll))
    assert jnp.allclose(nll, changed_nll)
    assert jnp.allclose(compiled_mean, distribution.mean)
    _assert_finite_model_gradient(
        model,
        lambda item: phx.nn.operator.architectures.gaussian_operator_nll(
            item, batch, distribution.mean, reduction="mean"
        ),
    )


def _pde_problem(lhs):
    return phx.equations.PDEProblemIR(
        coordinates=(phx.equations.PDECoordinate("x", "space"),),
        fields=(phx.equations.PDEField("u", coordinates=("x",)),),
        equations=(phx.equations.PDEEquation("governing", lhs),),
    )


def test_pde_condition_encoder_respects_semantic_hash_and_attaches_case_condition():
    field = phx.equations.PDEExpression.field("u")
    equivalent_a = _pde_problem(field + 1.0)
    equivalent_b = _pde_problem(1.0 + field)
    changed = _pde_problem(field + 2.0)
    tokens_a = phx.equations.tokenize_pde_ir(equivalent_a)
    tokens_b = phx.equations.tokenize_pde_ir(equivalent_b)
    tokens_changed = phx.equations.tokenize_pde_ir(changed)
    encoder = phx.nn.operator.architectures.PDEConditionEncoder(
        width=4,
        depth=1,
        dimension_rank=0,
        key=jr.key(12),
    )

    encoded_a = encoder(tokens_a)
    encoded_b = encoder(tokens_b)
    encoded_changed = encoder(tokens_changed)
    compiled = eqx.filter_jit(lambda item, tokens: item(tokens))(encoder, tokens_a)
    batch = _case_point_batch(jnp.array([[0.0, 0.5, 1.0, 2.0], [1.0, 0.5, 0.0, -1.0]]))
    conditioned = phx.nn.operator.architectures.attach_pde_condition(
        batch, tokens_a, encoder
    )
    condition = conditioned.input("equation")

    assert equivalent_a.canonical_hash == equivalent_b.canonical_hash
    assert equivalent_a.canonical_hash != changed.canonical_hash
    assert tokens_a.canonical_hashes == tokens_b.canonical_hashes
    assert tokens_a.canonical_hashes != tokens_changed.canonical_hashes
    assert encoded_a.shape == (4,)
    assert jnp.allclose(encoded_a, encoded_b, rtol=1e-5, atol=1e-6)
    assert not jnp.allclose(encoded_a, encoded_changed)
    assert jnp.allclose(compiled, encoded_a)
    assert conditioned.case_axes == batch.case_axes
    assert conditioned.case_shape == batch.case_shape
    assert condition.values is not None
    assert conditioned.query("query") is batch.query("query")
    assert condition.values.shape == (2, 1, 4)
    assert jnp.allclose(condition.values[:, 0], jnp.broadcast_to(encoded_a, (2, 4)))
    assert jnp.array_equal(condition.mask_array(case_shape=(2,)), jnp.ones((2, 1)))
    _assert_finite_model_gradient(encoder, lambda item: jnp.sum(item(tokens_a) ** 2))


def _semantic_token_arrays(tokens):
    return tuple(
        getattr(tokens, name)
        for name in (
            "kind",
            "operator",
            "attribute",
            "symbol",
            "scalar",
            "physical_dimension",
            "slot",
            "parent",
            "depth",
            "mask",
        )
    )


def _semantic_problem(
    *,
    coordinates=None,
    fields=None,
    parameters=(),
    expression=None,
    regions=(),
    conditions=(),
    nondimensionalization=(),
    metadata=(),
):
    coordinates = (
        (phx.equations.PDECoordinate("x", "space"),)
        if coordinates is None
        else coordinates
    )
    fields = (
        (phx.equations.PDEField("u", coordinates=("x",)),) if fields is None else fields
    )
    equations = (
        ()
        if expression is None
        else (phx.equations.PDEEquation("governing", expression),)
    )
    return phx.equations.PDEProblemIR(
        coordinates=coordinates,
        fields=fields,
        parameters=parameters,
        equations=equations,
        regions=regions,
        conditions=conditions,
        nondimensionalization=nondimensionalization,
        metadata=metadata,
    )


def test_pde_condition_encoder_distinguishes_execution_semantics():
    expression = phx.equations.PDEExpression
    u = expression.field("u")
    vector_fields = (
        phx.equations.PDEField(
            "u",
            representation="vector",
            components=2,
            coordinates=("x",),
        ),
    )
    two_fields = (
        phx.equations.PDEField("u", coordinates=("x",)),
        phx.equations.PDEField("v", coordinates=("x",)),
    )
    xy_coordinates = (
        phx.equations.PDECoordinate("x", "space"),
        phx.equations.PDECoordinate("y", "space"),
    )
    xy_fields = (phx.equations.PDEField("u", coordinates=("x", "y")),)
    boundary_region = phx.equations.PDERegion(
        "restricted",
        "boundary",
        ("x",),
    )
    initial_region = phx.equations.PDERegion(
        "restricted",
        "initial",
        ("x",),
    )
    pairs = (
        (
            _semantic_problem(expression=u.derivative("x", order=1)),
            _semantic_problem(expression=u.derivative("x", order=2)),
        ),
        (
            _semantic_problem(
                coordinates=xy_coordinates,
                fields=xy_fields,
                expression=u.derivative("x"),
            ),
            _semantic_problem(
                coordinates=xy_coordinates,
                fields=xy_fields,
                expression=u.derivative("y"),
            ),
        ),
        (
            _semantic_problem(
                fields=vector_fields,
                expression=u.component(0),
            ),
            _semantic_problem(
                fields=vector_fields,
                expression=u.component(1),
            ),
        ),
        (
            _semantic_problem(fields=two_fields, expression=u + u),
            _semantic_problem(
                fields=two_fields,
                expression=u + expression.field("v"),
            ),
        ),
        (
            _semantic_problem(
                fields=(phx.equations.PDEField("u", representation="scalar"),),
            ),
            _semantic_problem(fields=vector_fields),
        ),
        (
            _semantic_problem(
                parameters=(
                    phx.equations.PDEParameter(
                        "a",
                        value=(1.0, 2.0),
                        components=2,
                        scale=(1.0, 1.0),
                    ),
                ),
            ),
            _semantic_problem(
                parameters=(
                    phx.equations.PDEParameter(
                        "a",
                        value=(1.0, 3.0),
                        components=2,
                        scale=(1.0, 2.0),
                    ),
                ),
            ),
        ),
        (
            _semantic_problem(
                coordinates=(
                    phx.equations.PDECoordinate(
                        "x",
                        "space",
                        bounds=(0.0, 1.0),
                        periodic=False,
                    ),
                ),
            ),
            _semantic_problem(
                coordinates=(
                    phx.equations.PDECoordinate(
                        "x",
                        "space",
                        bounds=(-1.0, 1.0),
                        periodic=True,
                    ),
                ),
            ),
        ),
        (
            _semantic_problem(
                regions=(boundary_region,),
                conditions=(
                    phx.equations.PDECondition(
                        "restriction",
                        "boundary",
                        u,
                        region="restricted",
                    ),
                ),
            ),
            _semantic_problem(
                regions=(initial_region,),
                conditions=(
                    phx.equations.PDECondition(
                        "restriction",
                        "initial",
                        u,
                        region="restricted",
                    ),
                ),
            ),
        ),
        (
            _semantic_problem(
                coordinates=xy_coordinates,
                fields=xy_fields,
                regions=(
                    phx.equations.PDERegion(
                        "x_boundary",
                        "boundary",
                        ("x",),
                    ),
                    phx.equations.PDERegion(
                        "y_boundary",
                        "boundary",
                        ("y",),
                    ),
                ),
                conditions=(
                    phx.equations.PDECondition(
                        "restriction",
                        "boundary",
                        u,
                        region="x_boundary",
                    ),
                ),
            ),
            _semantic_problem(
                coordinates=xy_coordinates,
                fields=xy_fields,
                regions=(
                    phx.equations.PDERegion(
                        "x_boundary",
                        "boundary",
                        ("x",),
                    ),
                    phx.equations.PDERegion(
                        "y_boundary",
                        "boundary",
                        ("y",),
                    ),
                ),
                conditions=(
                    phx.equations.PDECondition(
                        "restriction",
                        "boundary",
                        u,
                        region="y_boundary",
                    ),
                ),
            ),
        ),
        (
            _semantic_problem(nondimensionalization=(("x", 1.0),)),
            _semantic_problem(nondimensionalization=(("x", 2.0),)),
        ),
    )
    encoder = phx.nn.operator.architectures.PDEConditionEncoder(
        width=16,
        depth=2,
        dimension_rank=0,
        key=jr.key(120),
    )

    for left, right in pairs:
        left_tokens = phx.equations.tokenize_pde_ir(left)
        right_tokens = phx.equations.tokenize_pde_ir(right)
        assert any(
            left_array.shape != right_array.shape
            or not jnp.array_equal(left_array, right_array)
            for left_array, right_array in zip(
                _semantic_token_arrays(left_tokens),
                _semantic_token_arrays(right_tokens),
                strict=True,
            )
        )
        assert not jnp.allclose(
            encoder(left_tokens),
            encoder(right_tokens),
            rtol=1e-8,
            atol=1e-9,
        )


def test_pde_condition_encoder_is_alpha_renaming_invariant():
    expression = phx.equations.PDEExpression
    original = _semantic_problem(
        coordinates=(phx.equations.PDECoordinate("x", "space"),),
        fields=(
            phx.equations.PDEField("u", coordinates=("x",)),
            phx.equations.PDEField("v", coordinates=("x",)),
        ),
        expression=expression.field("u") + expression.field("v"),
        nondimensionalization=(("x", 2.0),),
    )
    renamed = _semantic_problem(
        coordinates=(phx.equations.PDECoordinate("position", "space"),),
        fields=(
            phx.equations.PDEField("temperature", coordinates=("position",)),
            phx.equations.PDEField("pressure", coordinates=("position",)),
        ),
        expression=(expression.field("temperature") + expression.field("pressure")),
        nondimensionalization=(("position", 2.0),),
    )
    original_tokens = phx.equations.tokenize_pde_ir(original)
    renamed_tokens = phx.equations.tokenize_pde_ir(renamed)
    encoder = phx.nn.operator.architectures.PDEConditionEncoder(
        width=16,
        depth=2,
        dimension_rank=0,
        key=jr.key(121),
    )

    assert original.canonical_hash != renamed.canonical_hash
    assert jnp.allclose(
        encoder(original_tokens),
        encoder(renamed_tokens),
        rtol=1e-8,
        atol=1e-8,
    )


def test_pde_token_padding_and_stacking_preserve_semantic_channels():
    first = phx.equations.tokenize_pde_ir(
        _semantic_problem(
            expression=phx.equations.PDEExpression.field("u").derivative(
                "x",
                order=2,
            ),
        )
    )
    second = phx.equations.tokenize_pde_ir(
        _semantic_problem(nondimensionalization=(("x", 3.0),))
    )
    padded = phx.equations.pad_pde_tokens(first, first.max_tokens + 3)
    stacked = phx.equations.stack_pde_tokens((first, second))

    assert jnp.array_equal(padded.attribute[: first.max_tokens], first.attribute)
    assert jnp.array_equal(padded.slot[: first.max_tokens], first.slot)
    assert jnp.array_equal(padded.slot[-3:], -jnp.ones((3,), dtype=jnp.int32))
    assert stacked.batch_shape == (2,)
    assert stacked.attribute.shape == stacked.mask.shape
    assert stacked.slot.shape == stacked.mask.shape


def test_arbitrary_pde_metadata_stays_outside_neural_semantics():
    first = _semantic_problem(metadata=(("provenance", "experiment-a"),))
    second = _semantic_problem(metadata=(("provenance", "experiment-b"),))
    first_tokens = phx.equations.tokenize_pde_ir(first)
    second_tokens = phx.equations.tokenize_pde_ir(second)
    encoder = phx.nn.operator.architectures.PDEConditionEncoder(
        width=8,
        depth=1,
        dimension_rank=0,
        key=jr.key(122),
    )

    assert first.canonical_hash != second.canonical_hash
    assert all(
        jnp.array_equal(left, right)
        for left, right in zip(
            _semantic_token_arrays(first_tokens),
            _semantic_token_arrays(second_tokens),
            strict=True,
        )
    )
    assert jnp.allclose(encoder(first_tokens), encoder(second_tokens))
