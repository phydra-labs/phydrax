# Exterior calculus

`phydrax.exterior` owns scientific form metadata, exterior algebra and bridges
between smooth forms and discrete realizations. It does not replace chart fields,
`DomainFunction`, cell topology or the native linear algebra substrate.

## Form semantics

```python
from phydrax.exterior import FormType, FormValueSpec

circulation = FormType(3, 1, twist="untwisted")
flux = FormType(3, 2, twist="twisted")
flux_values = FormValueSpec(flux, proxy="flux")
```

`FormType(dimension, degree, /, *, twist="untwisted", fiber_shape=(),
ambient_dimension=None)` is immutable scientific metadata. Dimension, degree,
twist, fiber and ambient dimension participate in `form_type_id`.
`FormValueSpec(form_type, /, *, proxy)` declares the physical value representation
and exposes `value_spec_id`. Equal shapes do not establish scientific identity.

| Proxy | Degree | Physical value |
|---|---|---|
| `scalar` | 0 | Scalar |
| `circulation` | 1 | Covariant vector components |
| `flux` | n−1 | Vector v with ι_v vol = ω |
| `density` | n | Density |
| `components` | Any | Lexicographic form coefficients |

Ambiguous proxies require an explicit choice, including circulation versus flux
in two dimensions. The 2-D flux packs as `(−v_y, v_x)`; the 3-D components
`(xy, xz, yz)` are `(v_z, −v_y, v_x)`. Piola maps use signed determinants for
untwisted forms and absolute determinants for twisted forms. Embedded maps require
the declared coorientation where twist demands it.

Smooth coefficient shape is `(*batch, choose(ambient_dimension, degree), *fiber)`.
The component axis is present even for degree zero and top degree. Basis indices
are increasing and lexicographically ordered. Wedge XORs twist; d, contraction and
Lie derivative preserve it. Star flips twist; an orientation is supplied only to
explicit `to_untwisted`/`to_twisted` conversions. Embedded star requires tracing to
an intrinsic form first.

## Owners and entry points

- [Complexes](complexes.md): `AbstractDeRhamComplex`, `DiscreteForm`, metric-only
  Hodges and the `phydrax.linalg.HilbertComplex` operator substrate.
- [Bridges](bridges.md): de Rham integration, chain kernels, boundary traces,
  products and coefficient systems.
- [Executable guide](../../guides_exterior_calculus.md): smooth calculus,
  realization workflows, failure and evidence contracts.
- [Chart forms](../metrix/forms.md): `DifferentialForm`.
- [Domain differential operators](../operators/differential.md):
  `DomainDifferentialForm` over labeled programs.

The exterior facade is lazy. Core metadata and realization protocols do not
import discretization; cell-specific protocol extensions belong to
`phydrax.discretization.AbstractCellDeRhamComplex`.

::: phydrax.exterior.FormType

::: phydrax.exterior.FormValueSpec

::: phydrax.exterior.FormTwist

::: phydrax.exterior.FormProxy

::: phydrax.exterior.vector_to_form

::: phydrax.exterior.form_to_vector

::: phydrax.exterior.map_reference_values

## Algebra kernel

These functions act on explicitly typed lexicographic coefficients. They do not
infer batch, blade or fiber identity from coincident shapes. Matrix wedge uses
`product="matrix"` and preserves multiplication order. For smooth value carriers,
use the corresponding metrix or domain operations.

::: phydrax.exterior.wedge

::: phydrax.exterior.interior

::: phydrax.exterior.exterior_derivative_from_jacobian

::: phydrax.exterior.hodge_star

::: phydrax.exterior.inner

::: phydrax.exterior.pullback

::: phydrax.exterior.to_twisted

::: phydrax.exterior.to_untwisted

::: phydrax.exterior.hodge_square_sign

::: phydrax.exterior.codifferential_sign

::: phydrax.exterior.exterior_indices

::: phydrax.exterior.wedge_sign
