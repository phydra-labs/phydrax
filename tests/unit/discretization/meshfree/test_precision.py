from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.meshfree import MeshfreePrecisionPolicy


pytestmark = pytest.mark.strict_jax


def test_float32_profile_keeps_wide_certification_and_reports_native_roles() -> None:
    policy = MeshfreePrecisionPolicy(
        geometry_dtype="float32", certification_dtype="float64"
    )
    assert policy.compute_dtype == "float32"
    assert policy.fit_dtype == "float32"
    assert policy.certification_dtype == "float64"
    assert policy.checkpoint_dtype == "float32"
    evidence = dict(policy.evidence({"compute": jnp.float32}).observed)
    assert evidence["storage"] == "float32"
    assert evidence["factorization"] == "float32"
    assert evidence["certification"] == "float64"
    assert policy.resource_assumptions.itemsize("certification") == 8
    with pytest.raises(ValueError, match="differs from resolved"):
        policy.evidence({"compute": jnp.float64})


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        pytest.param(
            {"geometry_dtype": "float64", "certification_dtype": "float32"},
            "Certification precision cannot be narrower than geometry",
            id="certification-below-geometry",
        ),
        pytest.param(
            {
                "geometry_dtype": "float32",
                "fit_dtype": "float64",
                "certification_dtype": "float32",
            },
            "Certification precision cannot be narrower than fit",
            id="certification-below-fit",
        ),
        pytest.param(
            {"compute_dtype": "float64", "residual_dtype": "float32"},
            "Residual precision cannot be narrower than compute",
            id="residual-below-compute",
        ),
        pytest.param(
            {"compute_dtype": "float64", "communication_dtype": "float32"},
            "Communication precision cannot narrow",
            id="communication-below-compute",
        ),
        pytest.param(
            {"geometry_dtype": "float64", "checkpoint_dtype": "float32"},
            "Checkpoint precision cannot narrow",
            id="checkpoint-below-state",
        ),
        pytest.param(
            {"geometry_dtype": "float16"},
            "geometry_dtype",
            id="unsupported-half-precision",
        ),
    ),
)
def test_unsupported_narrowing_is_refused(
    overrides: dict[str, str], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MeshfreePrecisionPolicy(**overrides)


def test_certification_cast_refuses_silent_downcast_of_near_ties() -> None:
    # Two sources whose float64 distances to the target differ by less than the
    # float32 spacing: float64 orders them, float32 rounding makes them equal.
    target = np.asarray([[0.1, 0.2]])
    sources = np.asarray([[0.1 + 3.0e-9, 0.2], [0.1, 0.2 + 2.0e-9]])
    exact = np.sum((sources - target) ** 2, axis=1)
    assert exact[1] < exact[0]

    wide = MeshfreePrecisionPolicy(geometry_dtype="float64")
    certified = wide.cast("certification", sources) - wide.cast("certification", target)
    distances = jnp.sum(certified * certified, axis=1)
    assert distances.dtype == jnp.float64
    assert bool(distances[1] < distances[0])

    narrow = MeshfreePrecisionPolicy(geometry_dtype="float32")
    assert narrow.certification_dtype == "float32"
    with pytest.raises(ValueError, match="Refusing to downcast float64 certification"):
        narrow.cast("certification", sources)
    assert narrow.cast("geometry", sources).dtype == jnp.float32
    with pytest.raises(TypeError, match="real floating point"):
        wide.cast("compute", jnp.arange(3))
