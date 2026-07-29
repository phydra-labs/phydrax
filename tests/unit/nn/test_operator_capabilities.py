#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _grid_batch(*, size=5, periodic=True, mask=None):
    nodes = jnp.arange(size, dtype=float) / size
    axes = (
        phx.nn.OperatorAxis("x", nodes, periodic=periodic),
        phx.nn.OperatorAxis("y", nodes, periodic=periodic),
    )
    source = phx.nn.FunctionSamples(
        values=jnp.ones((2, size, size)),
        axes=axes,
        mask=mask,
    )
    return phx.nn.OperatorBatch(
        inputs={"u": source},
        queries={"query": phx.nn.FunctionSamples(values=None, axes=axes)},
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
    topology = phx.nn.OperatorTopology.from_graph(graph)
    coordinates = jnp.arange(3.0)[:, None]
    weights = jnp.ones((3,)) if quadrature else None
    source = phx.nn.FunctionSamples(
        values=jnp.ones((3,)),
        coordinates=coordinates,
        quadrature_weights=weights,
        topology=topology,
    )
    query = phx.nn.FunctionSamples(
        values=None,
        coordinates=coordinates,
        quadrature_weights=weights,
        topology=topology,
    )
    return phx.nn.OperatorBatch(inputs={"u": source}, queries={"query": query})


def test_every_registered_architecture_has_one_runtime_and_training_contract():
    for name, status in phx.nn.OPERATOR_ARCHITECTURE_STATUSES.items():
        assert status.name == name
        assert isinstance(status.capabilities, phx.nn.OperatorCapabilitySpec)
        assert isinstance(status.training, phx.nn.OperatorTrainingRequirement)
        assert status.recommendation_eligible is (status.tier == "stable")


def test_configured_contract_preserves_registered_configuration_and_rejects_conflicts():
    tfno = phx.nn.operator_architecture_contract("TFNO", configuration={"rank": 4})
    assert tfno.architecture == "FNO"
    assert tfno.configuration == (("factorization", "tucker"), ("rank", 4))

    with pytest.raises(ValueError, match="conflicts"):
        phx.nn.operator_architecture_contract(
            "TFNO", configuration={"factorization": "dense"}
        )


def test_model_instance_contract_tracks_capability_affecting_constructor_state():
    fno = phx.nn.FNO(
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
    report = phx.nn.validate_operator_architecture("FNO", batch)

    assert report.accepted


@pytest.mark.parametrize(
    "architecture",
    ("FNO", "CNO", "UNO", "Flower", "IFNO", "UPT"),
)
def test_grid_operator_contracts_accept_resolution_transfer_and_rollout(architecture):
    report = phx.nn.validate_operator_architecture(
        architecture,
        _grid_batch(),
        problem=phx.nn.OperatorProblemSpec(
            requires_resolution_transfer=True,
            rollout_steps=4,
        ),
    )

    assert report.accepted


def test_fixed_query_is_separate_from_source_query_structure():
    variable_query_problem = phx.nn.OperatorProblemSpec(
        source_query_relation="coincident",
        query_is_fixed=False,
    )
    fno = phx.nn.validate_operator_architecture(
        "FNO",
        _grid_batch(),
        problem=variable_query_problem,
    )
    pod = phx.nn.validate_operator_architecture(
        "PODDeepONet",
        _grid_batch(),
        problem=variable_query_problem,
    )

    assert fno.accepted
    assert "FIXED_QUERY_REQUIRED" not in fno.codes
    assert "FIXED_QUERY_REQUIRED" in pod.codes


def test_graph_contract_requires_native_topology_and_physical_measure():
    valid = phx.nn.validate_operator_architecture("GraphNeuralOperator", _graph_batch())
    missing_measure = phx.nn.validate_operator_architecture(
        "GraphNeuralOperator", _graph_batch(quadrature=False)
    )
    coordinate_only = phx.nn.validate_operator_architecture(
        "GraphNeuralOperator", _grid_batch()
    )

    assert valid.accepted
    assert "MISSING_PHYSICAL_QUADRATURE" in missing_measure.codes
    assert "TOPOLOGY_REQUIRED" in coordinate_only.codes
    assert "UNSUPPORTED_GEOMETRY" in coordinate_only.codes


def test_training_requirements_separate_task_specific_from_foundation_claims():
    batch = _grid_batch()
    poseidon = phx.nn.validate_operator_architecture("Poseidon", batch)
    assert "MISSING_PRETRAINED_WEIGHTS" in poseidon.codes

    pretrained = phx.nn.validate_operator_architecture(
        "Poseidon",
        batch,
        training_evidence=phx.nn.OperatorTrainingEvidence(
            regime="pretrained_system",
            checkpoint_id="immutable-checkpoint",
            corpus_id="multi-pde-corpus",
        ),
    )
    assert pretrained.accepted

    in_context = phx.nn.validate_operator_architecture("InContextOperator", batch)
    assert "MISSING_TASK_DISTRIBUTION_TRAINING" in in_context.codes
    trained = phx.nn.validate_operator_architecture(
        "InContextOperator",
        batch,
        training_evidence=phx.nn.OperatorTrainingEvidence(
            regime="task_distribution",
            corpus_id="prompted-operator-distribution",
        ),
    )
    assert trained.accepted
