# Embedded geometry

`EmbeddedChart` represents one local parameterization `X(q)` into Euclidean
ambient space. It derives the embedding Jacobian, induced metric, intrinsic
volume density, tangent/normal projectors, second fundamental form, mean
curvature vector, and shape operator.

It is intentionally a local-chart abstraction. It does not claim global
coverage, choose atlas transitions, infer mesh charts, or invent a retraction for
an arbitrary immersion.

```python
import jax.numpy as jnp
import phydrax as phx

chart = phx.metrix.CoordinateChart("sphere", ("theta", "phi"))


def embedding(q):
    theta, phi = q
    return jnp.array(
        [
            jnp.sin(theta) * jnp.cos(phi),
            jnp.sin(theta) * jnp.sin(phi),
            jnp.cos(theta),
        ]
    )


sphere = phx.metrix.EmbeddedChart(
    chart,
    embedding,
    3,
    retraction=lambda x: x / jnp.linalg.norm(x),
)

q = jnp.array([1.1, 0.4])
g = sphere.induced_metric()(q)
projector = sphere.tangent_projector(q)
mean_curvature = sphere.mean_curvature_vector(q)
```

::: phydrax.metrix.EmbeddedChart
    options:
        members:
            - __init__
            - __call__
            - tangent_basis
            - embedding_hessian
            - induced_metric
            - volume_density
            - tangent_projector
            - normal_projector
            - project_tangent
            - project_normal
            - retract
            - second_fundamental_form
            - mean_curvature_vector
            - shape_operator

## Projector from an explicit normal

For an implicit hypersurface or an existing normal-based geometry, use the
shared Euclidean helper. It normalizes each nonzero normal and returns
`I - n nᵀ`; zero normals fail explicitly.

::: phydrax.metrix.tangent_projector_from_normal

Phydrax's sampled surface operators reuse this projector kernel. Their sampling,
normal construction, masking, and component semantics remain in the domain and
operator layers rather than moving into Metrix.
