#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.conditions import (
    ArrayCodomain,
    bind_condition,
    Condition,
    Equality,
    FieldSpec,
    MatrixLinearFunctional,
    ProductFieldSpec,
)
from phydrax.enforcement import (
    ConstraintLinearCorrectionProvider,
    prepare_affine_projector,
)


def _ready(value):
    return jax.block_until_ready(value)


def _case(size: int, iterations: int) -> dict[str, object]:
    codomain = ArrayCodomain.from_shape((size,), dtype=float)
    fields = ProductFieldSpec((FieldSpec("u", codomain),))
    matrix = jnp.eye(size, dtype=float)
    target = jnp.linspace(-1.0, 1.0, size)
    condition = Condition(
        f"constraint-benchmark-{size}",
        fields,
        MatrixLinearFunctional(("u",), ((size,),), (matrix,)),
        codomain,
        Equality(target),
    )
    initial = {"u": jnp.linspace(3.0, 5.0, size)}
    bound = bind_condition(condition, initial)
    start = time.perf_counter()
    prepared = prepare_affine_projector(
        (bound,),
        ConstraintLinearCorrectionProvider(),
        correction_fields=("u",),
    )
    preparation_seconds = time.perf_counter() - start

    apply = eqx.filter_jit(prepared.apply)
    start = time.perf_counter()
    first = _ready(apply(initial))
    first_seconds = time.perf_counter() - start
    start = time.perf_counter()
    value = first
    for _ in range(iterations):
        value = apply(value)
    value = _ready(value)
    steady_seconds = (time.perf_counter() - start) / iterations
    defect = jnp.max(jnp.abs(value["u"] - target))
    return {
        "constraint_rows": size,
        "source_size": size,
        "iterations": iterations,
        "preparation_seconds": preparation_seconds,
        "first_execution_seconds": first_seconds,
        "steady_execution_seconds": steady_seconds,
        "maximum_constraint_defect": float(defect),
        "rank": prepared.correction.evidence.rank,
        "nullity": prepared.correction.evidence.nullity,
        "right_inverse_defect": float(prepared.correction.evidence.identity_defect),
        "projector_id": prepared.prepared_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    sizes = (1, 4) if args.smoke else (1, 4, 16, 32)
    iterations = 2 if args.smoke else 20
    result = {
        "backend": jax.default_backend(),
        "devices": tuple(str(device) for device in jax.devices()),
        "cases": tuple(_case(size, iterations) for size in sizes),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
