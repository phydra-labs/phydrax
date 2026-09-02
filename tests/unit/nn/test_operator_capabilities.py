#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _grid_batch(
    *,
    size=5,
    periodic=True,
    mask=None,
    physical_quadrature=True,
    basis="uniform",
    query_shift=0.0,
    nodes=None,
):
    nodes = (
        jnp.arange(size, dtype=float) / max(size, 1)
        if nodes is None
        else jnp.asarray(nodes, dtype=float)
    )
    size = int(nodes.size)
    weights = jnp.ones(size) if physical_quadrature else None
    axes = (
        phx.nn.operator.OperatorAxis(
            "x",
            nodes,
            quadrature_weights=weights,
            basis=basis,
            periodic=periodic,
        ),
        phx.nn.operator.OperatorAxis(
            "y",
            nodes,
            quadrature_weights=weights,
            basis=basis,
            periodic=periodic,
        ),
    )
    query_axes = tuple(
        phx.nn.operator.OperatorAxis(
            axis.name,
            axis.nodes + query_shift,
            quadrature_weights=axis.quadrature_weights,
            basis=axis.basis,
            periodic=axis.periodic,
        )
        for axis in axes
    )
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((2, size, size)),
        axes=axes,
        mask=mask,
    )
    return phx.nn.operator.OperatorBatch(
        inputs={"u": source},
        queries={"query": phx.nn.operator.FunctionSamples(values=None, axes=query_axes)},
        case_axes=("case",),
    )


def _graph_batch(*, quadrature=True):
    graph = phx.graph.GraphIR(
        nodes=jnp.zeros((3, 1)),
        senders=jnp.asarray([0, 1, 2]),
        receivers=jnp.asarray([1, 2, 0]),
        n_node=jnp.asarray([3]),
        n_edge=jnp.asarray([3]),
    )
    topology = phx.nn.operator.OperatorTopology.from_graph(graph)
    coordinates = jnp.arange(3.0)[:, None]
    weights = jnp.ones((3,)) if quadrature else None
    source = phx.nn.operator.FunctionSamples(
        values=jnp.ones((3,)),
        coordinates=coordinates,
        quadrature_weights=weights,
        topology=topology,
    )
    query = phx.nn.operator.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=weights,
        topology=topology,
    )
    return phx.nn.operator.OperatorBatch(inputs={"u": source}, queries={"query": query})


def test_every_registered_architecture_has_one_runtime_and_training_contract():
    for name, status in phx.nn.operator.OPERATOR_ARCHITECTURE_STATUSES.items():
        assert status.name == name
        assert isinstance(status.capabilities, phx.nn.operator.OperatorCapabilitySpec)
        assert isinstance(status.training, phx.nn.operator.OperatorTrainingRequirement)
        assert status.recommendation_eligible is (status.tier == "stable")


def test_configured_contract_preserves_registered_configuration_and_rejects_conflicts():
    tfno = phx.nn.operator.operator_architecture_contract(
        "TFNO", configuration={"rank": 4}
    )
    assert tfno.architecture == "FNO"
    assert tfno.configuration == (("factorization", "tucker"), ("rank", 4))

    with pytest.raises(ValueError, match="conflicts"):
        phx.nn.operator.operator_architecture_contract(
            "TFNO", configuration={"factorization": "dense"}
        )


def test_model_instance_contract_tracks_capability_affecting_constructor_state():
    fno = phx.nn.operator.architectures.FNO(
        n_modes=(3,),
        width=4,
        depth=1,
        factorization="tucker",
        rank=2,
        source_key="u",
        key=jr.key(0),
    )
    fno_configuration = dict(fno.operator_contract.configuration)
    assert fno.operator_contract.architecture == "FNO"
    assert fno_configuration["n_modes"] == (3,)
    assert fno_configuration["factorization"] == "tucker"
    assert fno_configuration["rank"] == 2
    assert fno_configuration["source_key"] == "u"


def test_capability_report_accepts_supported_fno_inputs():
    masked = jnp.ones((2, 5, 5), dtype=bool).at[0, 0, 0].set(False)
    batch = _grid_batch(periodic=False, mask=masked)
    report = phx.nn.operator.validate_operator_architecture("FNO", batch)

    assert report.accepted


@pytest.mark.parametrize(
    "architecture",
    ("FNO", "CNO", "UNO", "Flower", "IFNO", "UPT"),
)
def test_grid_operator_contracts_accept_resolution_transfer_and_rollout(architecture):
    report = phx.nn.operator.validate_operator_architecture(
        architecture,
        _grid_batch(),
        problem=phx.nn.operator.OperatorProblemSpec(
            requires_resolution_transfer=True,
            rollout_steps=4,
        ),
    )

    assert report.accepted


def test_cno_and_uno_declare_their_actual_measure_and_mask_support():
    masked = jnp.ones((2, 5, 5), dtype=bool).at[0, 2, 3].set(False)
    cno_masked = phx.nn.operator.validate_operator_architecture(
        "CNO", _grid_batch(mask=masked)
    )
    cno_missing_measure = phx.nn.operator.validate_operator_architecture(
        "CNO", _grid_batch(physical_quadrature=False)
    )
    uno_masked = phx.nn.operator.validate_operator_architecture(
        "UNO", _grid_batch(mask=masked)
    )

    assert cno_masked.accepted
    assert "MISSING_PHYSICAL_QUADRATURE" in cno_missing_measure.codes
    assert uno_masked.accepted


def test_cno_family_catalog_requires_exact_periodic_uniform_fourier_axes():
    for architecture in ("CNO", "UNO"):
        capabilities = phx.nn.operator.operator_architecture_contract(
            architecture
        ).capabilities
        assert capabilities.axis_requirement == "periodic_fourier_uniform"
        assert capabilities.minimum_axis_size == 2

        assert (
            "NONPERIODIC_AXIS"
            in phx.nn.operator.validate_operator_architecture(
                architecture,
                _grid_batch(periodic=False),
            ).codes
        )
        assert "UNSUPPORTED_AXIS_BASIS" in (
            phx.nn.operator.validate_operator_architecture(
                architecture,
                _grid_batch(basis="legendre"),
            ).codes
        )
        assert (
            "AXIS_TOO_SMALL"
            in phx.nn.operator.validate_operator_architecture(
                architecture,
                _grid_batch(size=1),
            ).codes
        )
        assert (
            "NONUNIFORM_AXIS"
            in phx.nn.operator.validate_operator_architecture(
                architecture,
                _grid_batch(nodes=jnp.asarray([0.0, 0.2, 0.7, 1.0])),
            ).codes
        )
        assert "SOURCE_QUERY_RELATION" in (
            phx.nn.operator.validate_operator_architecture(
                architecture,
                _grid_batch(query_shift=0.125),
                problem=phx.nn.operator.OperatorProblemSpec(
                    source_query_relation="coincident"
                ),
            ).codes
        )


def test_spectral_operator_contracts_match_runtime_invariants():
    wavelet = phx.nn.operator.operator_architecture_contract(
        "WaveletNeuralOperator"
    ).capabilities
    multiwavelet = phx.nn.operator.operator_architecture_contract(
        "MultiwaveletOperator"
    ).capabilities
    sfno = phx.nn.operator.operator_architecture_contract("SFNO").capabilities

    assert wavelet.spatial_dimensions == (1, 2, 3)
    assert wavelet.resolution_transfer
    assert wavelet.masks == "supported"
    assert multiwavelet.spatial_dimensions == (1,)
    assert multiwavelet.resolution_transfer
    assert multiwavelet.masks == "supported"
    assert sfno.source_geometries == ("sphere",)
    assert sfno.query_geometries == ("sphere",)
    assert sfno.quadrature == "physical_required"
    assert sfno.masks == "all_valid_only"
    assert not sfno.resolution_transfer


def test_fixed_query_is_separate_from_source_query_structure():
    variable_query_problem = phx.nn.operator.OperatorProblemSpec(
        source_query_relation="coincident",
        query_is_fixed=False,
    )
    fno = phx.nn.operator.validate_operator_architecture(
        "FNO",
        _grid_batch(),
        problem=variable_query_problem,
    )
    pod = phx.nn.operator.validate_operator_architecture(
        "PODDeepONet",
        _grid_batch(),
        problem=variable_query_problem,
    )

    assert fno.accepted
    assert "FIXED_QUERY_REQUIRED" not in fno.codes
    assert "FIXED_QUERY_REQUIRED" in pod.codes


def test_function_frame_contract_is_research_scoped_and_query_independent():
    status = phx.nn.operator.operator_architecture_status("function_encoder")
    capabilities = status.capabilities
    report = phx.nn.operator.validate_operator_architecture(
        "FunctionFrameReconstructor",
        _grid_batch(),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="independent",
            query_is_fixed=False,
            requires_resolution_transfer=True,
        ),
    )
    unsupported_graph = phx.nn.operator.validate_operator_architecture(
        "FunctionFrameReconstructor",
        _graph_batch(),
    )

    assert status.tier == "research"
    assert not status.recommendation_eligible
    assert capabilities.source_geometries == ("tensor_grid", "point_cloud")
    assert capabilities.query_geometries == ("tensor_grid", "point_cloud")
    assert capabilities.encode_once_decode_many
    assert capabilities.resolution_transfer
    assert report.accepted
    assert "UNSUPPORTED_GEOMETRY" in unsupported_graph.codes


def test_graph_contract_requires_native_topology_and_physical_measure():
    valid = phx.nn.operator.validate_operator_architecture(
        "GraphNeuralOperator", _graph_batch()
    )
    missing_measure = phx.nn.operator.validate_operator_architecture(
        "GraphNeuralOperator", _graph_batch(quadrature=False)
    )
    coordinate_only = phx.nn.operator.validate_operator_architecture(
        "GraphNeuralOperator", _grid_batch()
    )

    assert valid.accepted
    assert "MISSING_PHYSICAL_QUADRATURE" in missing_measure.codes
    assert "TOPOLOGY_REQUIRED" in coordinate_only.codes
    assert "UNSUPPORTED_GEOMETRY" in coordinate_only.codes


def test_training_requirements_separate_task_specific_from_foundation_claims():
    batch = _grid_batch()
    poseidon = phx.nn.operator.validate_operator_architecture("Poseidon", batch)
    assert "MISSING_PRETRAINED_WEIGHTS" in poseidon.codes

    pretrained = phx.nn.operator.validate_operator_architecture(
        "Poseidon",
        batch,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="pretrained_system",
            checkpoint_id="immutable-checkpoint",
            corpus_id="multi-pde-corpus",
        ),
    )
    assert pretrained.accepted

    in_context = phx.nn.operator.validate_operator_architecture(
        "InContextOperator", batch
    )
    assert "MISSING_TASK_DISTRIBUTION_TRAINING" in in_context.codes
    trained = phx.nn.operator.validate_operator_architecture(
        "InContextOperator",
        batch,
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_distribution",
            corpus_id="prompted-operator-distribution",
        ),
    )
    assert trained.accepted
