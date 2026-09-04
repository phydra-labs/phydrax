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

`material_id` is a required logical slot ID, not a digest of traced values. Two
occurrences of one slot must have exactly equal canonical ε, μ, ξ, and ζ samples.
Concrete equal occurrences may share prepared convolutions; unequal occurrences fail.
Under tracing, equality is asserted on device and every occurrence is recomputed, so
independent equal-valued leaves retain independent JVP/VJP paths. Distinct slot IDs
never authorize sharing.

`material_role` is either `physical` or `artificial_pml`.
`origin_evidence_id` identifies the stable physical response law/dataset,
rasterization origin, or transformation-optics PML evidence across frequency.
Dispersive samples therefore keep one origin ID while their full numeric revisions
bind each frequency's actual tensors. Translation and refresh retain both fields. A
user or rasterized material is physical; `transform_fourier_modal_material` always
produces `artificial_pml`. `passive` and `reciprocal` are recorded scientific claims
and never alter supplied tensors.

Finite layers support full ε and μ tensors. Periodic or anisotropic exteriors require
`PeriodicMaxwellPort`; the homogeneous port path is scalar isotropic.

## Geometry rasterization

`FourierModalRasterizationPlan` maps one compiled two-dimensional or sliced
three-dimensional geometry onto the prepared lattice pixel centers. The plan owns
fixed subpixel coordinates; `rasterize_fourier_modal_material` evaluates the current
geometry state and returns a `FrequencyMaxwellMaterial`, fill fractions, and explicit
field-certificate evidence.

With `smoothing_width=None`, rasterization uses the exact region predicate and is a
discrete derivative boundary. A positive smoothing width applies Phydrax's compact
regularized Heaviside to the geometry boundary field. Geometry parameters are then
differentiable only when the geometry's `FieldCertificate` says they are.

```text
raster_plan = fm.FourierModalRasterizationPlan(
    harmonics,
    fm.FourierModalRasterizationPolicy(
        samples_per_axis=3,
        smoothing_width=0.02,
    ),
)
material = fm.rasterize_fourier_modal_material(
    raster_plan,
    geometry,
    inside_permittivity=12.0,
    outside_permittivity=1.0,
    material_id="inclusion",
).material
```

The initial contract supports scalar inside/outside permittivity and permeability on
a two-dimensional lattice. It performs arithmetic fill averaging, not tensor
interface factorization. The resulting material may still use direct, inverse, or
vector Fourier factorization during layer preparation.

## Fourier factorization

- `DirectFourierFactorizationPlan`: direct Laurent multiplication.
- `InverseFourierFactorizationPlan`: scalar isotropic inverse-rule transverse blocks.
- `VectorFourierFactorizationPlan`: scalar local-frame factorization.

Vector factorization accepts an analytic tangent field or a Jones-direct Fourier
least-squares frame. A dynamic `AnalyticInterfaceFramePlan` requires an explicit
`frame_id`. Equal IDs with unequal tangent values fail; a shape match is never a
reuse certificate. Jones frames expose their derivative contract:

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

Both exterior port types accept a nonnegative `reference_distance` measured outward
from the adjacent stack interface. Incident amplitudes are propagated from that
reference to the interface before scattering; outgoing amplitudes are propagated
back afterward. The left circuit coordinate is `−reference_distance`; the right
coordinate is `total_thickness + reference_distance`. Periodic ports retain distinct
incoming/outgoing invariant bases and propagation exponents throughout this path.

Internal currents live at a named `FourierModalSourcePlane`. A source plane must
split one host slot into adjacent finite layers. Both the slot ID and exact canonical
host samples must match; an equal shape or a repeated ID with unequal values fails.
This makes the source z location part of static problem topology while source
amplitudes remain trailing RHS columns. Electric and magnetic current sheets may
contain tangential and normal harmonic components.

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

## Directional port power and physical loss

Every solve reports `left_incoming_power`, `right_incoming_power`,
`left_outgoing_power`, and `right_outgoing_power` per coherent RHS. The signed net
power entering the stack is
`left_incoming + right_incoming − left_outgoing − right_outgoing`.
Weighted forms apply the declared incoherent `channel_weights`. These quantities are
directional port fluxes: the ordinary solve does not label them reflection,
transmission, absorption, or a power-balance certificate. Right-only and coherent
two-sided incidence therefore retain the same unambiguous contract.

`power_audit_residual` is an independent terminal check: it compares directional
modal power with cell-integrated Poynting flux reconstructed from the terminal E/H
fields. `POWER_AUDIT_TOLERANCE_NOT_MET` reports a basis, normalization, or terminal
field inconsistency. It does not rename the signed port deficit as absorption.

`evaluate_fourier_modal_loss` is an opt-in, independent reconstruction. Its first
eligible envelope is real positive frequency, port drive, retained boundary fields,
one or more finite piecewise-constant physical layers, no source planes, no
artificial PML, and ξ = ζ = 0. For the `exp(−iωt)` convention it integrates
`(ω/2) [E† Im_H(ε) E + H† Im_H(μ) H]` over cell and z, where
`Im_H(A) = (A − A†)/(2i)`. Elementwise imaginary parts are not substituted for the
Hermitian loss operator.

`FourierModalLossEvidence` independently retains both port fluxes, each layer's two
face fluxes and Poynting drop, each physical volume loss, net port power, unresolved
closure, embedded z-quadrature defect, passive-claim PSD violations, eligibility,
status, and exactly one `harmonic_discretization_id`. Internal-source work,
artificial-PML work, continuous-z profiles, surface loss, and general 6 × 6
magnetoelectric work are typed ineligible. Active negative loss is permitted, but it
invalidates a `passive=True` claim.

Call `fourier_modal_numeric_revision(prepared)` only at a host-materialized accepted
point and pass it to loss evaluation. Tracers are rejected rather than omitted from
the digest. Revision metadata separates a `physical_state_digest` over primitive
vectors, frequency/Bloch values, ports/reference distances, raw constitutive values,
per-layer thickness/translation, frame data, and source parents from a
`physical_stack_digest` that removes frequency, angle, overall thickness, and sampled
constitutive values while retaining geometry, primitive vectors, normalized layer
distribution, stable material-response origins/claims, and factorization/frame
provenance. The full revision digest additionally binds the harmonic layout and
canonical discretized values. Without a revision the
differentiable material-loss observable remains available, but evidence has
`NUMERIC_REVISION_REQUIRED` and cannot be accepted.

A single solve never claims transverse convergence.
`assess_fourier_modal_loss_convergence` requires at least two accepted loss results
with distinct full numeric revisions, identical physical-state digests, and distinct,
nested harmonic discretizations. It reports port, material-loss, and closure
convergence separately.

## Refresh

`refresh_fourier_modal_maxwell` accepts an explicit `FourierModalRefreshSpec`, but
the entries are hints rather than permission to reuse stale arrays. Reuse occurs only
when exact concrete primitive-vector values, harmonic layout, canonical material
samples, factorization/frame leaves, translation, thickness, frequency, and Bloch
values prove it safe. A changed or traced dependency promotes the affected route to
recomputation even when a hint says `unchanged`.

Stack IDs, material slots/origins, factorization plans, and harmonic layouts are
semantic structure and cannot change in refresh. Continuous-z profiles always
reprepare and every z evaluation is uncached; no logical profile ID authorizes reuse
between coordinates. Source amplitudes never require refresh.

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
propagation residuals, directional port-power stability, independent physical-loss
closure where eligible, and a nested harmonic study.

A successful numerical status does not establish transverse Fourier convergence.
Run the same observable on nested harmonic layouts and use
`fourier_modal_convergence_report`. Physical loss uses the stricter, separate
`assess_fourier_modal_loss_convergence` evidence. Also vary Jones regularization and
boundary cascade order for difficult metallic or cornered structures.

## Advanced media, ports, and bounded adaptation

`FrequencyMaxwellMaterial` carries all four constitutive blocks: permittivity,
permeability, magnetoelectric ξ, and magnetoelectric ζ. Periodic or anisotropic
semi-infinite exteriors use `PeriodicMaxwellPort` and a flux/decay-separated
Schur-QZ invariant basis; `port_mode_excitation` addresses its stable mode IDs.
Unresolved dimensions or a missing spectral-gap certificate fail closed.

`ContinuousFourierModalLayer` prepares a fixed segment epoch with an embedded
second/fourth-order commutator-free Magnus estimate. It never bisects inside a
trace: exhausted segment capacity reports refinement required and the accepted
epoch remains unchanged. `LateralTransformationOpticsPMLPlan` transforms every
constitutive block and rejects singular, active, orientation-reversing, or
nonperiodic-seam transforms.

Preparation retains fixed-capacity prefix boundary relations for every accepted
continuous segment. `fields_in_layer` selects the containing prefix, performs one
partial fourth-order step, and evaluates the local constitutive operator at the
requested coordinate. Its result reports the selected segment, embedded dense-output
defect, and continuous integration status; it never substitutes one representative
midpoint operator for a varying profile.

`diffraction_order_far_field` remains the exact discrete radiation API for an
infinite periodic stack. Continuous directions are available only through
`FiniteApertureFarFieldPlan`, with an explicit rectangular or sampled finite
aperture and an aperture-power defect. `FourierModalHarmonicAdaptationPolicy`
selects from a finite nested, conjugate-closed candidate set between epochs and
uses the canonical spectral modal transfer; harmonic count never changes in a
compiled solve.

## Equivalent-slab retrieval and local-isotropic qualification

`prepare_maxwell_modal_sweep` creates a host-revision-bound Maxwell record. At every
frequency it retains the original unit-flux `ModalWaveReference`, selected harmonic
and polarization IDs, longitudinal wave number, modal admittance, reference plane,
preparation ID, numeric revision, and both referenced and directly de-embedded
Maxwell scattering entries. It never converts modal coordinates to electrical
voltage/current or S/Y/Z/ABCD networks.

`retrieve_equivalent_slab` accepts a normal-incidence, symmetric homogeneous-port
sweep and an `EquivalentSlabRetrievalPlan`. Exactly one decoupled TE or TM channel
per side is selected. Cross-polarization conversion, grazing, extra propagating
diffraction orders, asymmetric terminations, complex/nonpositive frequency,
transmission zeros, singular formulas, or a mismatched numeric revision fail closed.
The plan requires a bounded integer branch window and an explicit low-frequency,
known-index, or cross-thickness anchor.

`EquivalentSlabRetrieval` returns every propagation-root/impedance-sign candidate,
branch numbers, reconstructed-S residuals, admissibility, branch margins, and the
selected per-frequency modal chart. It retains both impedance relative to the
symmetric exterior (used to reconstruct scattering) and absolute effective
impedance `Z_eff = z_relative / Y_exterior` (used for
`ε_eff = n_eff / Z_eff` and `μ_eff = n_eff Z_eff`). Tied or insufficiently separated
branches are `AMBIGUOUS`, not silently unwrapped. Finite-band Kramers–Kronig and
stable-fit residuals are diagnostics only, never causality certificates. The result
describes one equivalent slab at one thickness/polarization; it does not claim a
local constitutive material.

`qualify_local_isotropic_medium` requires valid de-embedded retrievals for both
polarizations at two or more commensurate thicknesses, host-revision-bound nonzero
angle sweeps for both polarizations, nonzero branch margins, reconstructed-S
agreement, passive decay/sign consistency when claimed, accepted independent loss
evidence, and accepted multi-discretization loss convergence. Every retrieval,
angle sweep, and loss item must share one semantic stack ID and one content-bound
physical-stack digest before numerical gates are considered. It reports typed
`QUALIFIED`, `INELIGIBLE`, or `AMBIGUOUS` evidence and explicit rejection reasons for
unrelated stacks, termination, diffraction, singular branches,
thickness/polarization/angle disagreement, spatial dispersion, and uncertified loss.
It never creates a `FrequencyMaxwellMaterial`; promotion remains an explicit caller
decision.

## Limitations

- Dense harmonic work scales cubically with harmonic count per changed layer.
- Sharp corners and metals can converge nonmonotonically.
- Adaptive harmonic selection and mode-cluster transitions are stopped events;
  only a fixed separated subspace/layout has a derivative.
- Continuous angular intensity is not claimed for an infinite periodic structure.
