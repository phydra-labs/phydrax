#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


class _LinearOperator(phx.nn.operator.AbstractOperatorModel):
    weight: jax.Array
    in_size: str = eqx.field(static=True)
    out_size: str = eqx.field(static=True)

    def __init__(self):
        self.weight = jnp.asarray([[1.0]], dtype=jnp.float32)
        self.in_size = "scalar"
        self.out_size = "scalar"

    @property
    def operator_contract(self):
        return phx.nn.operator.operator_architecture_contract("DeepONet")

    def __call_operator_batch__(self, batch, *, key=None):
        del key
        values = batch.input("state").values
        assert values is not None
        return (values[..., None] @ self.weight)[..., 0]

    def __call__(self, batch, *, key=None):
        return self.__call_operator_batch__(batch, key=key)


def _dataset(cases: int = 8, resolution: int = 16):
    axis = phx.nn.operator.OperatorAxis(
        "x",
        jnp.linspace(0.0, 1.0, resolution),
        quadrature_weights=jnp.full((resolution,), 1.0 / resolution),
    )
    values = jnp.arange(cases, dtype=float)[:, None] + axis.nodes[None, :]
    return phx.nn.operator.training.operator_dataset_from_arrays(
        {"state": values},
        {"output": 2.0 * values},
        source_axes={"state": (axis,)},
        query_axes=(axis,),
    )


def _privacy(iterations: int):
    definition = phx.privacy.PrivacyDefinition(
        phx.privacy.PrivacyUnit("operator-case"),
        phx.privacy.NeighboringRelation.ADD_OR_REMOVE_ONE,
    )
    scope = phx.privacy.PrivateDataScope("operator-benchmark", definition)
    return phx.privacy.PrivateTrainingPlan(
        phx.privacy.DPSGDPlan(
            scope,
            phx.privacy.PrivacyBudget(5.0, 1e-5),
            sampling_probability=0.25,
            iterations=iterations,
            clipping_norm=1.0,
            normalize_by=2.0,
            dtype="float32",
        )
    )


def _fit(dataset, privacy):
    result = phx.nn.operator.training.fit_operator(
        _LinearOperator(),
        dataset,
        steps=privacy.mechanism.iterations,
        privacy=privacy,
        include_model_losses=False,
        loss_terms=(phx.nn.operator.training.SupervisedOperatorLoss(),),
        key=jr.key(7),
        jit=True,
    )
    jax.block_until_ready(result.execution_model.weight)
    return result


def main() -> None:
    dataset = _dataset()
    privacy = _privacy(iterations=4)

    started = time.perf_counter()
    first = _fit(dataset, privacy)
    cold_seconds = time.perf_counter() - started

    started = time.perf_counter()
    second = _fit(dataset, privacy)
    warm_seconds = time.perf_counter() - started

    if first.privacy_certificate is None or second.privacy_certificate is None:
        raise RuntimeError("Private benchmark did not produce privacy certificates.")
    print(
        json.dumps(
            {
                "cases": dataset.size,
                "iterations": privacy.mechanism.iterations,
                "cold_seconds": cold_seconds,
                "warm_seconds": warm_seconds,
                "epsilon": second.privacy_certificate.guarantee.epsilon,
                "delta": second.privacy_certificate.guarantee.delta,
                "public_release_allowed": second.privacy_certificate.public_release_allowed,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
