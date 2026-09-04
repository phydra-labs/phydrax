# Physical dimensions and units

PhydraX keeps physical metadata static and numerical kernels array-native. Units
are resolved at construction, import, or export boundaries; prepared solvers and
models operate on homogeneous raw JAX arrays in the units declared by their
owning contract.

This separation is deliberate:

- a **dimension** states algebraic powers such as length/time;
- a **unit** selects a numerical coordinate for one dimension;
- a **domain contract** adds meaning such as frame, epoch, coordinate kind,
  sign, support, species, gauge, or reference configuration;
- a **normalization** supplies numerical scale and offset for conditioning.

Equal dimensions do not make domain values interchangeable. A comoving length
is not converted into a physical length without cosmological context, and a UTC
epoch is not a duration in seconds.

## Exact dimensions

`DimensionSignature` stores a canonical sparse map of named axes to exact
rational exponents.

```python
from fractions import Fraction

import phydrax as phx

velocity = phx.units.DimensionSignature({"length": 1, "time": -1})
square_root_length = phx.units.LENGTH ** Fraction(1, 2)

assert velocity == phx.units.VELOCITY
assert (velocity / velocity).is_dimensionless
```

Exponents must be integers or `Fraction` values. Floating-point exponents are
rejected because they cannot define stable semantic identity. Numerical PDE
execution may still evaluate arbitrary floating powers when the base is
physically dimensionless.

PhydraX treats electric charge as the independent electrical axis and angle as
a distinct semantic dimension. Current is charge/time; radians are therefore
not silently interchangeable with an arbitrary dimensionless ratio.

## Unit definitions

`UnitDefinition` is immutable multiplicative metadata. A unit declares:

- a canonical display symbol;
- an exact dimension signature;
- an explicit reference-system identity;
- an exact positive rational scale to that reference;
- a content-addressed unit ID.

```python
import phydrax as phx

values_m = phx.units.convert_value(
    [1.0, 2.0],
    source=phx.units.KILOMETER,
    target=phx.units.METER,
)
```

Conversion is allowed only when dimensions and reference-system IDs match
exactly. Unit labels do not trigger parsing or inference. Domain adapters may
map a closed set of accepted source tokens to canonical unit definitions.

Cardiovascular and skeletal-muscle quantity specifications store the same
`UnitDefinition` objects while retaining their domain-owned quantity kind,
axes, sign, support, and reference configuration. Their displayed kernel/SI
units and exact conversion factors are derived from that one definition rather
than maintained in a parallel string conversion table. Electrophysiology uses
the same algebra behind its closed boundary-token map and fixed prepared kernel
units.

Derived units are composed statically:

```python
speed_unit = phx.units.derived_unit(
    "km/s",
    ((phx.units.KILOMETER, 1), (phx.units.SECOND, -1)),
)
```

## Prepared execution

Convert external values before constructing or entering a prepared numerical
state. Bind the resulting unit contract to the plan, then carry only raw arrays
through JIT, VMAP, scans, implicit solves, and differentiation.

A fixed multiplicative conversion remains differentiable: its JVP and VJP use
the same static factor. Unit metadata does not attempt to infer the dimensions
of optimizer cotangents or third-party internal work buffers.

Exact contract IDs remain the default compatibility rule. Convertibility is a
separate boundary relation and never causes automatic conversion when plans,
states, learned models, or persisted artifacts are composed.

## Reduced and code units

A reduced or code system has its own explicit reference-system identity. It is
not SI-convertible unless a separate calibrated physical contract supplies that
relationship. Merely naming a value `code_length` or `reduced_energy` does not
establish a physical scale.

## Persistence

Unit-bearing artifacts store the canonical descriptor and its content ID.
Readers reconstruct the descriptor and verify that its content hashes to the
claimed ID. IDs are evidence, not lookups into a process-local registry.

## Deliberate boundaries

The native unit layer does not provide:

- array-wrapper quantities;
- implicit conversion or unit stripping;
- arbitrary string-expression parsing;
- affine temperature or epoch conversions;
- logarithmic units;
- global equivalencies such as mass-energy or spectral conversion;
- mixed-unit packed arrays;
- primitive-level guesses for Fourier, quadrature, or probability-density
  measures.

Those operations require explicit domain transformations with enough context to
state their physical meaning.
