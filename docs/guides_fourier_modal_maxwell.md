# Fourier-modal Maxwell

`phydrax.solver.maxwell.fourier_modal` solves frequency-domain Maxwell problems that
are periodic in one or two transverse directions and piecewise invariant along the
stack direction. It complements, rather than replaces, compatible cochain Maxwell.

Use this substrate for gratings, metasurfaces, photonic-crystal slabs, layered
emitters, diffraction orders, and many-source studies. Use compatible Maxwell for
general three-dimensional topology, time dynamics, nonlinear constitutive state, or
boundaries that are not naturally represented by homogeneous ports.

## Conventions

- Fields have time dependence `exp(−iωt)`.
- The stack runs from the superstrate to the substrate along +z.
- Primitive and reciprocal vectors satisfy `aᵢ · bⱼ = 2π δᵢⱼ`.
- Material arrays are sampled at unit-cell pixel centers.
- Angular frequency is canonical. Any wavelength conversion must name its wave speed.
- Port scattering maps left-forward and right-backward incident amplitudes to
  right-forward and left-backward outgoing amplitudes.
- Time-average flux is `0.5 Re(E × H*)`.

## Harmonic lattice

A `LatticeHarmonicPlan` owns discrete integer harmonic coefficients and the physical
sample shape. Preparing it with numerical primitive vectors produces reciprocal
vectors, pixel centers, transforms, and pairwise-difference convolution indexing.

```python
import jax.numpy as jnp

from phydrax.discretization.spectral import LatticeHarmonicPlan

period = 1.0
harmonic_plan = LatticeHarmonicPlan.parallelogramic((1,), (3,))
harmonics = harmonic_plan.prepare(((period, 0.0),))
epsilon_grid = jnp.full(harmonics.sample_shape, 2.25)
thickness = 0.2
angular_frequency = 2.0 * jnp.pi
bloch_wavevector = jnp.asarray((0.0, 0.0))
```

One-dimensional plans are genuinely one-dimensional. A two-dimensional circular
plan selects a conjugate-closed set against a reference lattice. The integer layout
is static; changing it requires replanning. Numerical primitive vectors remain
differentiable while the layout is fixed.

The physical grid must resolve every pairwise harmonic difference. Planning rejects
an undersampled grid before constructing a convolution matrix.

## Problem lifecycle

A problem has two homogeneous semi-infinite ports and an ordered tuple of finite
layers and optional named source planes.

```python
from phydrax.solver.maxwell import fourier_modal as fm

vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
layer_material = fm.FrequencyMaxwellMaterial(epsilon_grid, material_id="pattern")
layer = fm.FourierModalLayer(
    layer_material,
    thickness,
    fm.DirectFourierFactorizationPlan(),
    layer_id="patterned-layer",
)
problem = fm.FourierModalMaxwellProblem(
    harmonics,
    angular_frequency,
    bloch_wavevector,
    fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
    (layer,),
    fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
)
plan = fm.plan_fourier_modal_maxwell(problem)
prepared = fm.prepare_fourier_modal_maxwell(problem, plan)
```

Planning owns resource limits and static propagation choices. Preparation performs
material transforms, full-tensor layer assembly, boundary propagation, port
normalization, and stack composition. Sources are applied only after preparation.

## Materials

`FrequencyMaxwellMaterial` accepts:

- one scalar;
- one homogeneous 3 × 3 tensor;
- a scalar array with the lattice sample shape;
- a tensor array with shape `sample_shape + (3, 3)`.

Finite layers support full ε and μ tensors. Exterior ports are currently homogeneous
and isotropic. Magnetoelectric coupling and patterned ports are rejected.

`passive` and `reciprocal` are scientific claims recorded with the material; they do
not silently alter the supplied tensors.

## Fourier factorization

- `DirectFourierFactorizationPlan`: direct Laurent multiplication.
- `InverseFourierFactorizationPlan`: scalar isotropic inverse-rule transverse blocks.
- `VectorFourierFactorizationPlan`: scalar local-frame factorization.

Vector factorization accepts an analytic tangent field or a Jones-direct Fourier
least-squares frame. Jones frames expose their derivative contract:

- `mathematical`: differentiate the smooth target and least-squares solution;
- `frozen`: recompute the frame but hold it fixed during differentiation;
- `none`: fixed external frame.

A frozen derivative is a partial derivative and diagnostics set
`frame_gradient_omitted=True`. Vector factorization of arbitrary full tensors is not
implemented and fails closed.

## Boundary propagation

The default backend forms a first-order tangential Maxwell operator for
`[Eₓ, Eᵧ, Hₓ, Hᵧ]`. It initializes a short-interval transfer polynomial, converts it
to a mixed boundary relation, and repeatedly doubles that relation. Growing transfer
states are never propagated across the full layer.

`BoundaryCascadePolicy` fixes:

- the number of doublings;
- Taylor initializer order;
- paired N versus N+1 error evidence;
- absolute and relative tolerances.

A result reports initializer, solve, and paired propagation residuals. Increase the
cascade order when `PROPAGATION_TOLERANCE_NOT_MET` is returned.

The optional modal backend uses Phydrax's dense general eigensolver. It exists for
modal observables and independent small-problem comparisons. Modal eigenvectors are
not differentiable through this API.

## Excitations and source planes

`plane_wave_excitation` addresses port modes by stable harmonic ID and TE/TM label.
Propagating modes are normalized to unit absolute longitudinal power.

Internal currents live at a named `FourierModalSourcePlane`. A source plane must
split one host material into adjacent finite layers. This makes the source z location
part of static problem topology while source amplitudes remain trailing RHS columns.
Electric and magnetic current sheets may contain tangential and normal harmonic
components.

`point_source_coefficients` and `gaussian_source_coefficients` construct transverse
Fourier profiles. Each RHS column is coherent. `channel_weights` combine columns
incoherently for reported aggregate power.

## Brillouin-zone integration

`BrillouinZonePlan` constructs a Γ-containing periodic trapezoid rule in the
reciprocal unit cell. Use:

- `integrate_brillouin_fields` for coherent field reconstruction;
- `integrate_brillouin_power` for quadrature-weighted k-resolved power.

`prepare_brillouin_zone_maxwell` prepares one independently identified case at every
rule wavevector. `solve_fourier_modal_case_batch` restores the declared Brillouin
grid before the field or power reduction, while each case retains its complete
diagnostics and provenance.

The two reductions are intentionally distinct. No batch axis is guessed to be
coherent or incoherent from shape alone.

## Fields and far fields

`fields_in_layer` evaluates a partial boundary relation at a requested layer offset,
recovers E_z and H_z from the constitutive elimination maps, and reconstructs either
the physical pixel grid or caller-supplied xy coordinates.

`diffraction_order_far_field` returns exact discrete order wavevectors, directions,
angles, propagating masks, and powers. It does not interpolate a continuous angular
surface. Evanescent orders carry zero reported radiated power.

## Refresh

`refresh_fourier_modal_maxwell` accepts an explicit `FourierModalRefreshSpec`.
Per-layer changes are classified as `unchanged`, `thickness`, `translation`, or
`material`. Frequency and Bloch changes invalidate all dependent layer operators.
Source amplitudes never require refresh.

No array-equality heuristic infers sharing. Repeated layers share scientific identity
through explicit material IDs.

## Differentiation

The boundary backend differentiates the mathematical matrix products and linear
solves. Supported smooth parameters include material tensors, thickness, angular
frequency, Bloch vector, fixed-layout primitive vectors, translations, and source
amplitudes.

Discrete harmonic layouts, layer topology, source-plane placement, backend choice,
and cascade order are static. Their changes require replanning.

Near a material discontinuity, changing a raster topology is not a differentiable
operation. Optimize a smooth material parameterization and separately qualify the
final discretized structure.

## Precision and convergence

Complex128 is the correctness default. Complex64 is opt-in and must be qualified with
propagation residuals, power balance, and a nested harmonic study.

A successful numerical status does not establish transverse Fourier convergence.
Run the same observable on nested harmonic layouts and use
`fourier_modal_convergence_report`. Also vary Jones regularization and boundary
cascade order for difficult metallic or cornered structures.

## Limitations

- Dense harmonic work scales cubically with harmonic count per changed layer.
- Sharp corners and metals can converge nonmonotonically.
- Semi-infinite ports must be homogeneous and isotropic.
- Full-tensor layers currently use direct or inverse/Li factorization, not Jones
  factorization.
- Continuous-z media require explicit slicing.
- Bianisotropy, lateral PML, and continuous angular interpolation are not included.
