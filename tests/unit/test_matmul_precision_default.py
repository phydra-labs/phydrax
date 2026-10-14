#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax  # noqa: F401  (importing Phydrax establishes the precision default)


@pytest.mark.parametrize(
    ("environment", "prelude", "expected"),
    [
        ({}, "", "highest"),
        ({"JAX_DEFAULT_MATMUL_PRECISION": "tensorfloat32"}, "", "tensorfloat32"),
        (
            {},
            "jax.config.update('jax_default_matmul_precision', 'bfloat16'); ",
            "bfloat16",
        ),
    ],
    ids=("unset", "environment-choice", "prior-config-choice"),
)
def test_import_defaults_float32_contractions_without_overriding_user_choice(
    environment: dict[str, str], prelude: str, expected: str
) -> None:
    child_environment = {
        name: value
        for name, value in os.environ.items()
        if name != "JAX_DEFAULT_MATMUL_PRECISION"
    }
    child_environment.update(environment, JAX_PLATFORMS="cpu")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                f"import jax; {prelude}import phydrax; "
                "print(jax.config.jax_default_matmul_precision)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=child_environment,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == expected


@pytest.mark.skipif(
    jax.default_backend() != "gpu", reason="reduced-precision tensor cores need a GPU"
)
def test_default_float32_matmul_on_gpu_has_float32_accuracy() -> None:
    # TF32 keeps a 10-bit mantissa (relative error near 1e-3 for this product);
    # IEEE float32 accumulation stays near 1e-6.
    generator = np.random.default_rng(7)
    left = generator.standard_normal((256, 256)).astype(np.float32)
    right = generator.standard_normal((256, 256)).astype(np.float32)
    reference = left.astype(np.float64) @ right.astype(np.float64)
    product = np.asarray(jnp.asarray(left) @ jnp.asarray(right), dtype=np.float64)
    relative = np.linalg.norm(product - reference) / np.linalg.norm(reference)
    assert relative < 1e-5
