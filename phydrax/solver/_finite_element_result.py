#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..equations import FiniteElementExecutionPolicy


class FiniteElementRunConfiguration(StrictModule, NonTrainableState):
    representation: str = eqx.field(static=True)
    accumulation: str = eqx.field(static=True)
    nonlinear_method: str = eqx.field(static=True)
    linear_method: str = eqx.field(static=True)
    schema_version: int = eqx.field(static=True)
    configuration_id: str = eqx.field(static=True)

    def __init__(
        self,
        /,
        *,
        representation: str = "matrix-free",
        accumulation: str = "fast",
        nonlinear_method: str = "newton-krylov",
        linear_method: str = "auto",
        schema_version: int = 1,
    ):
        representation_ = str(representation)
        accumulation_ = str(accumulation)
        nonlinear = str(nonlinear_method)
        linear = str(linear_method)
        version = int(schema_version)
        if representation_ not in (
            "matrix-free",
            "element-tensor",
            "partial",
            "sparse",
        ):
            raise ValueError("Unknown finite-element representation.")
        if accumulation_ not in ("fast", "deterministic", "compensated"):
            raise ValueError("Unknown finite-element accumulation mode.")
        if not nonlinear or not linear or version != 1:
            raise ValueError("Finite-element run configuration is invalid.")
        self.representation = representation_
        self.accumulation = accumulation_
        self.nonlinear_method = nonlinear
        self.linear_method = linear
        self.schema_version = version
        self.configuration_id = canonical_fingerprint(
            {
                "kind": "finite-element-run-configuration",
                "representation": representation_,
                "accumulation": accumulation_,
                "nonlinear_method": nonlinear,
                "linear_method": linear,
                "schema_version": version,
            }
        )

    def execution_policy(self, /) -> FiniteElementExecutionPolicy:
        realization = {
            "matrix-free": "matrix-free",
            "element-tensor": "element-tensor",
            "partial": "partial",
            "sparse": "sparse",
        }[self.representation]
        return FiniteElementExecutionPolicy(
            realization=realization,
            accumulation=self.accumulation,
        )


class FiniteElementSolveDiagnostics(StrictModule):
    successful: Array
    residual_norm: Array
    nonlinear_iterations: Array
    linear_iterations: Array
    constraint_defect: Array
    conservation_defect: Array
    energy_defect: Array
    status: str = eqx.field(static=True)

    def __init__(
        self,
        successful: ArrayLike,
        residual_norm: ArrayLike,
        /,
        *,
        nonlinear_iterations: ArrayLike = 0,
        linear_iterations: ArrayLike = 0,
        constraint_defect: ArrayLike = 0.0,
        conservation_defect: ArrayLike = 0.0,
        energy_defect: ArrayLike = 0.0,
        status: str = "completed",
    ):
        successful_ = jnp.asarray(successful, dtype=bool)
        residual = jnp.asarray(residual_norm)
        nonlinear = jnp.asarray(nonlinear_iterations, dtype=jnp.int32)
        linear = jnp.asarray(linear_iterations, dtype=jnp.int32)
        constraint = jnp.asarray(constraint_defect)
        conservation = jnp.asarray(conservation_defect)
        energy = jnp.asarray(energy_defect)
        if any(
            value.shape != ()
            for value in (
                successful_,
                residual,
                nonlinear,
                linear,
                constraint,
                conservation,
                energy,
            )
        ):
            raise ValueError("Finite-element diagnostics must contain scalar summaries.")
        identifier = str(status)
        if not identifier:
            raise ValueError("Diagnostic status must be non-empty.")
        self.successful = successful_
        self.residual_norm = residual
        self.nonlinear_iterations = nonlinear
        self.linear_iterations = linear
        self.constraint_defect = constraint
        self.conservation_defect = conservation
        self.energy_defect = energy
        self.status = identifier


class FiniteElementResult(StrictModule, NonTrainableState):
    field_names: tuple[str, ...] = eqx.field(static=True)
    fields: tuple[Array, ...]
    time: Array
    prepared_id: str = eqx.field(static=True)
    compilation_id: str = eqx.field(static=True)
    diagnostics: FiniteElementSolveDiagnostics
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        field_names: Sequence[str],
        fields: Sequence[ArrayLike],
        time: ArrayLike,
        prepared_id: str,
        compilation_id: str,
        diagnostics: FiniteElementSolveDiagnostics,
        /,
    ):
        names = tuple(str(value) for value in field_names)
        values = tuple(jnp.asarray(value) for value in fields)
        time_ = jnp.asarray(time)
        prepared = str(prepared_id)
        compilation = str(compilation_id)
        if (
            not names
            or len(names) != len(values)
            or len(set(names)) != len(names)
            or any(not name for name in names)
        ):
            raise ValueError("Finite-element result fields are invalid.")
        if time_.shape != () or not prepared or not compilation:
            raise ValueError("Finite-element result metadata are invalid.")
        if not isinstance(diagnostics, FiniteElementSolveDiagnostics):
            raise TypeError("diagnostics must be FiniteElementSolveDiagnostics.")
        self.field_names = names
        self.fields = values
        self.time = time_
        self.prepared_id = prepared
        self.compilation_id = compilation
        self.diagnostics = diagnostics
        self.result_id = canonical_fingerprint(
            {
                "kind": "finite-element-result",
                "field_names": list(names),
                "field_shapes": [list(value.shape) for value in values],
                "prepared": prepared,
                "compilation": compilation,
                "status": diagnostics.status,
            }
        )


def write_finite_element_result(path: str | Path, result: FiniteElementResult, /) -> None:
    if not isinstance(result, FiniteElementResult):
        raise TypeError("result must be FiniteElementResult.")
    metadata = {
        "schema_version": 1,
        "field_names": list(result.field_names),
        "prepared_id": result.prepared_id,
        "compilation_id": result.compilation_id,
        "result_id": result.result_id,
        "status": result.diagnostics.status,
    }
    arrays = {
        "time": np.asarray(result.time),
        **{
            f"field_{index}": np.asarray(value)
            for index, value in enumerate(result.fields)
        },
        "successful": np.asarray(result.diagnostics.successful),
        "residual_norm": np.asarray(result.diagnostics.residual_norm),
        "nonlinear_iterations": np.asarray(result.diagnostics.nonlinear_iterations),
        "linear_iterations": np.asarray(result.diagnostics.linear_iterations),
        "constraint_defect": np.asarray(result.diagnostics.constraint_defect),
        "conservation_defect": np.asarray(result.diagnostics.conservation_defect),
        "energy_defect": np.asarray(result.diagnostics.energy_defect),
    }
    np.savez(Path(path), metadata=np.asarray(json.dumps(metadata)), **arrays)


def read_finite_element_result(path: str | Path, /) -> FiniteElementResult:
    archive = np.load(Path(path), allow_pickle=False)
    metadata = json.loads(str(archive["metadata"]))
    if int(metadata["schema_version"]) != 1:
        raise ValueError("Unsupported finite-element result schema version.")
    diagnostics = FiniteElementSolveDiagnostics(
        archive["successful"],
        archive["residual_norm"],
        nonlinear_iterations=archive["nonlinear_iterations"],
        linear_iterations=archive["linear_iterations"],
        constraint_defect=archive["constraint_defect"],
        conservation_defect=archive["conservation_defect"],
        energy_defect=archive["energy_defect"],
        status=metadata["status"],
    )
    fields = tuple(
        archive[f"field_{index}"] for index in range(len(metadata["field_names"]))
    )
    result = FiniteElementResult(
        metadata["field_names"],
        fields,
        archive["time"],
        metadata["prepared_id"],
        metadata["compilation_id"],
        diagnostics,
    )
    if result.result_id != metadata["result_id"]:
        raise ValueError("Finite-element result identity mismatch.")
    return result


__all__ = [
    "FiniteElementResult",
    "FiniteElementRunConfiguration",
    "FiniteElementSolveDiagnostics",
    "read_finite_element_result",
    "write_finite_element_result",
]
