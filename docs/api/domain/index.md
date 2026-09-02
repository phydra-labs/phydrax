# Domains

A PhydraX `Domain` is an ordered product of atomic `JointFactor` objects. A factor
owns one or more public labels, its support, reference-coordinate transport, and
base measure. This distinction preserves intrinsically coupled coordinates—such
as trajectory row and time—without treating every label as independent.

```python
import phydrax as phx

space = phx.domain.GeometryDomain(
    phx.geometry.Square(center=(0.0, 0.0), side=2.0).compile(),
    label="x",
)
time = phx.domain.TimeInterval(0.0, 1.0)
problem_domain = space @ time

interior = problem_domain.component()
spatial_boundary = problem_domain.component(
    {"x": phx.domain.Boundary(), "t": phx.domain.Interior()}
)
```

Construction, sampling, integration, and field execution are separate layers:

1. `phydrax.geometry` sources compile representation-specific geometry to one
   JAX-safe kernel.
2. `GeometryDomain` labels that kernel and joins the domain algebra.
3. `Domain.component(...)` binds selections, restrictions, and densities to a
   typed base measure.
4. `PointSampling` or `GridSampling` materializes explicit batch schemas.
5. Integration lowers component and probability targets to weighted
   realizations.

See [Domains and sampling](../../guides_domain.md) for the full conceptual model.

## Base contracts

::: phydrax.domain.Domain
    options:
        members:
            - labels
            - joint_factors
            - factor
            - same_support
            - schema_compatible
            - component
            - restrict
            - drop
            - relabel
            - Function
            - Model
            - Parameter
            - __matmul__

---

::: phydrax.domain.JointFactor

## Product and coordinate domains

::: phydrax.domain.ProductDomain

---

::: phydrax.domain.ScalarInterval

---

::: phydrax.domain.TimeInterval

---

::: phydrax.domain.HyperRectangle

---

::: phydrax.domain.GeometryDomain

## Empirical and probability domains

::: phydrax.domain.DatasetDomain

---

::: phydrax.domain.ProbabilityDomain

---

::: phydrax.domain.ReferenceTransport

---

::: phydrax.domain.ReferenceTransportEvidence

---

::: phydrax.domain.construct_reference_transport

## Related APIs

- [Components and selections](components.md)
- [Sampling plans and batches](sampling.md)
- [Domain functions](functions.md)
- [Geometry substrate](../geometry.md)
- [Integration](../integration.md)
