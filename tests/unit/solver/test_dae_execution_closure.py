import jax
import jax.numpy as jnp

from phydrax.metrix import EuclideanStateGeometry
from phydrax.solver._dae_events import (
    certify_dae_regularity,
    DAERegularityCertificatePlan,
    DAERegularityDomain,
    manifold_bdf_stage,
    ManifoldBDFMethod,
)


def test_bounded_dae_regularity_marks_singularity_crossing_uncovered():
    domain = DAERegularityDomain(
        jnp.asarray([[1.0], [-0.5]]),
        jnp.asarray([[2.0], [0.5]]),
        domain_id="two-cells",
    )
    plan = DAERegularityCertificatePlan(
        domain,
        lambda center, args: (
            jnp.asarray([[center[0]]]),
            jnp.asarray(0.5),
        ),
        operator_id="affine-stage",
    )
    certificate = certify_dae_regularity(plan)
    assert certificate.covered[0]
    assert not certificate.covered[1]
    assert not certificate.certified


def test_manifold_bdf_stage_uses_fixed_local_chart_and_is_jittable():
    geometry = EuclideanStateGeometry()
    method = ManifoldBDFMethod(2)
    stage = jax.jit(
        lambda local: manifold_bdf_stage(
            method,
            geometry,
            jnp.asarray([1.0, 2.0]),
            (jnp.asarray([1.0, 2.0]), jnp.asarray([0.5, 1.5])),
            local,
            0.25,
        )
    )(jnp.asarray([0.25, 0.5]))
    assert stage.chart_valid
    assert jnp.allclose(stage.state, jnp.asarray([1.25, 2.5]))
    assert jnp.all(jnp.isfinite(stage.state_rate))
