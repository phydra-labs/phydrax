import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _operator_inference_data():
    time = jnp.linspace(0.0, 1.0, 24, dtype=jnp.float64)
    states = jnp.stack((jnp.sin(5.0 * time), jnp.cos(3.0 * time)), axis=-1)[None, ...]
    inputs = jnp.sin(2.0 * time)[None, :, None]
    z0 = states[..., 0]
    z1 = states[..., 1]
    u = inputs[..., 0]
    derivatives = jnp.stack(
        (
            0.5 + 1.2 * z0 - 0.3 * z1 + 0.7 * u + 0.2 * z0 * z1,
            -0.1 + 0.4 * z0 - 0.8 * z1 - 0.2 * u + 0.3 * z1**2,
        ),
        axis=-1,
    )
    state_layout = phx.dynamics.StateLayout(
        (2,), component_names=("z0", "z1"), layout_id="opinf-state"
    )
    input_layout = phx.dynamics.InputLayout(
        (1,), component_names=("u",), layout_id="opinf-input"
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
        source_id="manufactured-opinf",
    )
    return data


def test_operator_inference_uses_only_c_a_b_and_symmetric_h_blocks():
    data = _operator_inference_data()
    library = phx.dynamics.identification.OperatorInferenceFeatureLibrary(
        data.state_layout,
        input_layout=data.input_layout,
    )
    result = phx.dynamics.identification.fit_operator_inference(
        phx.dynamics.identification.SINDyProblem(
            data=data,
            library=library,
            formulation=phx.dynamics.identification.StrongSINDyFormulation(),
        ),
        (1.0e-12, 1.0e-12, 1.0e-12, 1.0e-12),
    )

    assert bool(result.valid)
    assert library.feature_names == (
        "1",
        "state:z0",
        "state:z1",
        "input:u",
        "state:z0 * state:z0",
        "state:z0 * state:z1",
        "state:z1 * state:z1",
    )
    prediction = result.evaluate(data.states, data.inputs)
    np.testing.assert_allclose(prediction, data.derivatives, atol=2e-5)


def test_reduced_trajectory_projection_preserves_offset_and_derivatives():
    data = _operator_inference_data()
    full = phx.linalg.ArraySpace((2,), dtype=jnp.float64, space_id="opinf-full")
    basis = phx.rom.ReducedBasisArtifact(
        phx.linalg.LinearSubspace(
            full,
            jnp.eye(2, dtype=jnp.float64),
            orthonormal=True,
            subspace_id="opinf-identity-subspace",
        ),
        role="state",
        state_contract_id="opinf-state",
        support_id="opinf-support",
        measure_id="opinf-measure",
        geometry_id="opinf-geometry",
        source_artifact_ids=("opinf-snapshots",),
    )
    offset = jnp.asarray([0.25, -0.5], dtype=jnp.float64)
    projected = phx.rom.project_trajectory_data(
        data,
        basis,
        offset,
        source_id="projected-opinf",
    )

    np.testing.assert_allclose(projected.states, data.states - offset)
    np.testing.assert_allclose(projected.derivatives, data.derivatives)

    library = phx.dynamics.identification.OperatorInferenceFeatureLibrary(
        projected.state_layout,
        input_layout=projected.input_layout,
    )
    result = phx.dynamics.identification.fit_operator_inference(
        phx.dynamics.identification.SINDyProblem(
            data=projected,
            library=library,
            formulation=phx.dynamics.identification.StrongSINDyFormulation(),
        ),
        (1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10),
    )
    composition = phx.rom.IdentifiedReducedDynamics(
        basis,
        offset,
        result.to_system(),
        identification_id=result.method_id,
        partition_id="train-validation-test-partition",
    )
    state = jnp.asarray([0.7, -0.2], dtype=jnp.float64)
    np.testing.assert_allclose(composition.decode(composition.encode(state)), state)
