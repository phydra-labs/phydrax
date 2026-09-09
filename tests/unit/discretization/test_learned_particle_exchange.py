from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx


class _ScalarPairOperator(phx.nn.operator.AbstractOperatorModel):
    in_size: int = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.in_size = 2
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("pair_features").values
        assert values is not None
        return values[..., 0]

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


class _VectorPairOperator(phx.nn.operator.AbstractOperatorModel):
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self):
        self.in_size = 2
        self.out_size = 2

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("pair_features").values
        assert values is not None
        return values

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _pairs_and_geometry():
    relation = phx.sparse.EdgeRelation(
        jnp.asarray((0, 0, 1), dtype=jnp.int32),
        jnp.asarray((1, 2, 2), dtype=jnp.int32),
        source_size=3,
        target_size=3,
        valid=jnp.asarray((True, True, False)),
    )
    pairs = phx.discretization.particle.ParticlePairRelation(
        relation,
        jnp.asarray((10, 10, 20), dtype=jnp.int64),
        jnp.asarray((20, 30, 20), dtype=jnp.int64),
        source_support_id="particles",
        target_support_id="particles",
        same_set=True,
        unordered=True,
        relation_schema_id="fixed-pairs",
    )
    positions = jnp.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    return pairs, phx.discretization.particle.particle_pair_geometry(positions, pairs)


def _task(channels):
    return phx.nn.operator.OperatorTask(
        "pair-exchange",
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "pair_features",
                channels=2,
                role="source",
                component_names=("a", "b"),
            ),
            phx.nn.operator.OperatorFieldSpec(
                "pair_exchange",
                channels=channels,
                role="target",
                query_name="pairs",
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "pairs",
                geometry_kind="point_cloud",
                coordinate_components=("dx", "dy"),
                fixed_geometry=False,
            ),
        ),
        problem=phx.nn.operator.OperatorProblemSpec(
            source_query_relation="coincident",
            query_is_fixed=False,
        ),
    )


def _trained(model, channels, artifact):
    return phx.nn.operator.training.TrainedOperator(
        model,
        _task(channels),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence("task_specific"),
        output_field_map={"output": "pair_exchange"},
        artifact_id=artifact,
    )


def _schema():
    return phx.nn.operator.adapters.PairwiseExchangeFeatureSchema(
        ("a", "b"),
        ("1", "1"),
        dtype=jnp.float32,
        relation_schema_id="fixed-pairs",
    )


def test_central_learned_exchange_conserves_linear_and_angular_momentum():
    pairs, geometry = _pairs_and_geometry()
    plan = phx.nn.operator.adapters.PairwiseExchangeBindingPlan(
        _schema(),
        exchange_kind="central_force",
        model_artifact_id="central",
        accumulation="compensated",
        conservation_tolerance=1e-6,
    )
    prepared = plan.prepare(
        _trained(_ScalarPairOperator(), "scalar", "central"), pairs, geometry
    )
    features = jnp.asarray(
        ((2.0, 0.0), (3.0, 0.0), (jnp.nan, jnp.nan)), dtype=jnp.float32
    )
    result = prepared(features, velocities=jnp.zeros((3, 2), dtype=jnp.float32))

    assert bool(result.successful)
    np.testing.assert_allclose(result.ledger.total_exchange, 0.0, atol=1e-6)
    np.testing.assert_allclose(result.ledger.torque, 0.0, atol=1e-6)
    assert result.ledger.pair_count == 2
    np.testing.assert_allclose(result.pair_values[2], 0.0)


def test_noncentral_vector_exchange_only_claims_linear_conservation():
    pairs, geometry = _pairs_and_geometry()
    plan = phx.nn.operator.adapters.PairwiseExchangeBindingPlan(
        _schema(),
        exchange_kind="vector",
        model_artifact_id="vector",
        accumulation="deterministic",
        conservation_tolerance=1e-6,
    )
    prepared = plan.prepare(_trained(_VectorPairOperator(), 2, "vector"), pairs, geometry)
    features = jnp.asarray(((0.0, 1.0), (0.0, 0.0), (0.0, 0.0)), dtype=jnp.float32)
    result = prepared(features)

    assert bool(result.successful)
    np.testing.assert_allclose(result.ledger.total_exchange, 0.0, atol=1e-6)
    assert float(jnp.abs(result.ledger.torque)) > 0.0


def test_relation_and_artifact_mismatches_are_rejected():
    pairs, geometry = _pairs_and_geometry()
    plan = phx.nn.operator.adapters.PairwiseExchangeBindingPlan(
        _schema(),
        exchange_kind="central_force",
        model_artifact_id="expected",
    )
    with np.testing.assert_raises(ValueError):
        plan.prepare(_trained(_ScalarPairOperator(), "scalar", "other"), pairs, geometry)
