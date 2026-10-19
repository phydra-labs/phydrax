# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Periodic task and trained-operator fixtures with explicit solution bindings."""

from typing import Any

import jax.random as jr

import phydrax as phx


def periodic_task(*, revision: Any = "1") -> Any:
    return phx.nn.operator.OperatorTask(
        "periodic-map",
        revision=revision,
        dimension_basis=("length",),
        fields=(
            phx.nn.operator.OperatorFieldSpec(
                "input",
                role="source",
                source_name="u",
                dimension=phx.units.DIMENSIONLESS,
                scale=2.0,
                offset=1.0,
            ),
            phx.nn.operator.OperatorFieldSpec(
                "solution",
                role="target",
                query_name="solution-query",
                dimension=phx.units.DIMENSIONLESS,
                scale=3.0,
                offset=4.0,
            ),
        ),
        queries=(
            phx.nn.operator.OperatorQuerySpec(
                "solution-query",
                geometry_kind="tensor_grid",
                coordinate_components=("x",),
                coordinate_dimensions=(phx.units.LENGTH,),
            ),
        ),
        metadata={"source": {"family": "synthetic"}, "tags": ["periodic"]},
    )


def solution_binding(task: Any) -> Any:
    port = task.field_by_name["solution"].value_port()
    return {
        "output_ports": {"output": port},
        "port_mapping": phx.PortMapping(outputs=((port.port_id, port.port_id),)),
    }


def trained_operator(
    *,
    model: Any = None,
    revision: Any = "1",
    output_pipeline: Any = None,
    compilation_strategy: Any = "eager",
    dtype_policy: Any = None,
) -> Any:
    if model is None:
        model = phx.nn.operator.architectures.FNO(
            n_modes=(3,),
            width=4,
            depth=1,
            coordinate_embedding=False,
            source_key="u",
            key=jr.key(12),
        )
    return phx.nn.operator.training.TrainedOperator(
        model,
        periodic_task(revision=revision),
        training_evidence=phx.nn.operator.OperatorTrainingEvidence(
            regime="task_specific"
        ),
        **solution_binding(periodic_task(revision=revision)),
        output_pipeline=output_pipeline,
        dtype_policy=dtype_policy,
        compilation_strategy=compilation_strategy,
        provenance={"dataset": "immutable-dataset"},
    )
