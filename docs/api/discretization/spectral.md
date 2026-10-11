# Global spectral methods

## Exterior realizations

`FourierDeRhamComplex(space, /, *, nyquist_policy="zero-self-conjugate")`
requires an explicit Nyquist policy and composes the existing spectral owner.
Its compact vectors exclude padded/forbidden modes. `from_modal` / `to_modal`
use an explicit last component axis and orthonormal Fourier Parseval pairing.
`hilbert_complex(boundary="absolute")` supplies degree spaces; relative boundary
is refused on this periodic realization.

`SphericalDeRhamComplex(space, /)` uses independent real harmonic coordinates
and normalized poloidal/toroidal modes, with radius² pairing and Betti (1, 0, 1).
Degree0/2 physical values are scalar and degree1 values use east/north components.
Both expose metric Riesz `hodge_star` / `inverse_hodge_star`; `metric_star` is the
separate smooth complement rotation.
Spectral `hodge_decomposition` retains its optimized analytic route and accepts
optional `harmonic` / `lower_harmonic` artifacts only after complex/degree/kernel
rank, metric-orthonormality and current-evidence checks. Degree0 refuses a lower
artifact. Non-None iterative `policy` is explicitly refused rather than silently
ignored by a closed-form solve.


::: phydrax.discretization.FourierDeRhamComplex

::: phydrax.discretization.SphericalDeRhamComplex


## Basis and tensor spaces

::: phydrax.discretization.AxisDomain

---

::: phydrax.discretization.AbstractSpectralBasisPlan

---

::: phydrax.discretization.FourierBasisPlan

---

::: phydrax.discretization.SineBasisPlan

---

::: phydrax.discretization.CosineBasisPlan

---

::: phydrax.discretization.ChebyshevBasisPlan

---

::: phydrax.discretization.LegendreBasisPlan

---
::: phydrax.discretization.RationalChebyshevLineBasisPlan

---

::: phydrax.discretization.RationalChebyshevHalfLineBasisPlan

---

::: phydrax.discretization.ConstrainedBasisPlan

---

::: phydrax.discretization.TensorSpectralPlan

---

::: phydrax.discretization.TensorSpectralDiscretization

### Field views

`TensorSpectralDiscretization.evaluate` and `derivative_at` synthesize modal
coefficients at arbitrary points; `prepare_spectral_field_reconstruction`
wraps that synthesis as a smooth `PreparedFieldReconstruction` for
`DiscreteFieldFunctionView` and prepared queries.
`TensorSpectralDiscretization.prepare_side_trace` publishes the exact trace on
bounded faces through a sum-factorized `SpectralFaceRoute`.

::: phydrax.discretization.prepare_spectral_field_reconstruction

---

::: phydrax.discretization.SpectralFieldReconstructionKernel

---

::: phydrax.discretization.SpectralFaceRoute

## Transfer and diagnostics

::: phydrax.discretization.SpectralModalTransferPlan

---

::: phydrax.discretization.PreparedSpectralModalTransfer

---

::: phydrax.discretization.SpectralModalTransferReport

---

::: phydrax.discretization.SpectralModalDiagnosticsPlan

---

::: phydrax.discretization.PreparedSpectralModalDiagnostics

---

::: phydrax.discretization.ModalDecayReport

---

::: phydrax.discretization.SpectralEigenResolutionPolicy

---

::: phydrax.discretization.SpectralEigenResolutionReport

---

::: phydrax.discretization.compare_spectral_eigen_resolutions

---

## Reciprocal-lattice harmonics

::: phydrax.discretization.LatticeHarmonicLayout

---

::: phydrax.discretization.LatticeHarmonicPlan

---

::: phydrax.discretization.LatticeHarmonicDiscretization

---

::: phydrax.discretization.BrillouinZonePlan

---

::: phydrax.discretization.PreparedBrillouinZone

## Exact-sampling spherical spaces

Analysis and synthesis support forward- and reverse-mode differentiation with
respect to sampled fields and harmonic coefficients, including spin fields,
real-linear conjugacy, batches, and channel-last arrays. A fixed transform's
JVP is the same linear transform applied to the input tangent; its VJP is the
actual sampling/normalization adjoint, not an assumed unweighted inverse.

Recursive recurrence tables are fixed numeric preparation state, not
differentiable model parameters. Attempting to differentiate those tables
raises explicitly in either AD direction. The prepared plan is
`NonTrainableState`, so its arrays are FIXED in every training tree. The
precomputed execution retains its native array differentiation semantics,
including kernel derivatives when explicitly requested outside solver parameter
partitioning.

`SphericalSpectralDiscretization.evaluate_angles(
coefficients, theta, phi, /, *, frame_angle=0)` broadcasts the three real
numerical angle arguments and evaluates in the longitude-labeled
`(east=e_phi, north=-e_theta)` frame. The frame is oriented by
`east × north = radial`. A frame rotation by `chi` transforms spin-$s$ values
by `exp(-1j*s*chi)`. At the poles, `phi` continues to label the limiting frame.

`SphericalSpectralDiscretization.evaluate(
coefficients, directions, /, *, tangent_frame=None)` accepts real Cartesian
directions ending in length three. Spin zero is gauge independent. For nonzero
spin, finite nonpolar directions use
`east=normalize(z_axis × radial)`, `north=radial × east`; an exact pole
requires an explicit broadcastable orthonormal `(east, north)` frame with the
same orientation. Unframed nonzero-spin poles and invalid frames are rejected.
Zero or nonfinite direction lanes and nonfinite angle lanes return lane-local
complex `NaN`.

Both methods require coefficient-leading `(L, 2*L-1)` axes followed by
arbitrary payload axes. If the broadcast evaluation prefix is `D` and the
payload is `P`, the result shape is `D + P`. Invalid padded `|m| > ell`
capacity is inert. Nonzero-spin layouts are complex and retain both order
signs. Real spin-zero layouts canonicalize
`c[ell, -m] = (-1)**m * conjugate(c[ell, m])` and return real values.

Coefficients, angles and frame angles, finite nonpolar directions, and supplied
frame coordinates carry JAX derivatives of the same evaluation function. The
fixed-spin Price--McEwen and Risbo recurrence regimes do not define different
derivatives. Evaluation is fused and adds neither an all-mode table nor a
public pairwise spin-harmonic API.

::: phydrax.discretization.SphericalModeLayout

---

::: phydrax.discretization.SphericalHarmonicPlan

---

::: phydrax.discretization.SphericalSpectralPlan

---

::: phydrax.discretization.SphericalSpectralDiscretization

---

::: phydrax.discretization.SolidHarmonicPlan

---

::: phydrax.discretization.PreparedSolidHarmonicSynthesis

---

::: phydrax.discretization.spherical_laplacian_operator

---

`SphericalSamplePlan` instead prepares a fixed-capacity sample geometry with
masks, weights, and bounded dense evaluate/fit operators. It remains the route
for repeated fitting or evaluation at the same admitted sample rows; it is not
the dynamic-direction overload above.

::: phydrax.discretization.SphericalSamplePlan

---

::: phydrax.discretization.PreparedSphericalSampleOperator

---

::: phydrax.discretization.SphericalSpinOperatorPlan

---

::: phydrax.discretization.SphericalCoordinateDerivativeResult

---

::: phydrax.discretization.SphericalRotationPlan

---

::: phydrax.discretization.SphericalClebschGordanPlan

### Intrinsic tangent vector calculus

`PreparedSphericalVectorOperators(space, mean_policy="reject")` prepares real
tangent operators on an existing real, spin-zero `SphericalSpectralDiscretization`
with bandlimit at least two. The radius is inherited from the space. The
implementation uses the existing spin-one transform and diagonal spin ladders,
not dense derivative matrices or coordinate derivatives divided by a polar sine.

The physical frame is east = `e_phi`, north = `-e_theta` (theta is colatitude).
Positive curl is radially outward. A tangent vector is encoded as the spin-one
field `north - 1j * east`. The returned components at sampled poles are the
longitude-labeled limiting tangent frame: they may depend on longitude even
when the corresponding Cartesian vector is single valued.

| Method | Input | Output |
| --- | --- | --- |
| `gradient(coefficients)` | Real scalar harmonic coefficients | `(east, north)` physical arrays |
| `divergence(east, north)` | Real physical tangent components | Scalar harmonic coefficients |
| `curl(east, north)` | Real physical tangent components | Outward relative-vorticity coefficients |
| `wind(vorticity, divergence)` | Scalar harmonic coefficients | `(east, north)` physical arrays |
| `null_mode_defect(coefficients)` | Scalar harmonic coefficients | Maximum absolute constant coefficient |

The Helmholtz convention is
`wind = grad(inverse_laplacian(divergence)) + k × grad(inverse_laplacian(vorticity))`.
Thus `k × (east, north) = (-north, east)`,
`div(grad(f)) = laplacian(f)`, `curl(grad(f)) = 0`, and
`div(k × grad(f)) = 0`. One spatial derivative carries one inverse-radius factor.
The scalar Laplacian eigenvalue is `-ell * (ell + 1) / radius**2`.

Coefficients have shape `(..., L, 2*L-1)` or
`(..., L, 2*L-1, channels)`; sampled components have the corresponding
`(..., n_theta, n_phi)` or `(..., n_theta, n_phi, channels)` shape.
The underlying transform's scalar-before-channel-last axis precedence applies.
Scalar coefficients must satisfy full real-field conjugacy, including real
zonal (`m=0`) modes. Missing conjugate orders are rejected rather than silently
filled. Invalid padded modes remain inert. Nonfinite active values and complex
physical tangent components are rejected.

Scalar constants have zero gradient. In contrast, a nonzero spherical mean
vorticity or divergence has no tangent wind solution. By default, `wind` rejects
incompatible source constants using a runtime guard that remains active under
JIT. Its roundoff tolerance is 256 machine eps times the larger of one and the
largest coefficient magnitude. `mean_policy="project"` explicitly removes
those constants; both policies choose zero-mean potentials. The JIT-safe
`null_mode_defect` reports the unprojected source evidence. The policy and
prepared geometry participate in `operator_id`.

```python
import jax.numpy as jnp

from phydrax.discretization import (
    PreparedSphericalVectorOperators,
    SphericalSpectralPlan,
)

space = SphericalSpectralPlan(16, sampling="mwss").prepare(radius=6_371_000.0)
operators = PreparedSphericalVectorOperators(space)
temperature = jnp.broadcast_to(
    280.0 + 10.0 * jnp.cos(space.transform.theta)[:, None], space.sample_shape
)
east, north = operators.gradient(space.project(temperature))
laplacian_coefficients = operators.divergence(east, north)
```

Run `python -m tools.spherical_vector_qualification --sampling mwss` from the
repository root to measure analytic tangent gradients, Hodge identities,
Helmholtz inversion, oblique solid rotation, and constant modes. The report
separates scalar/vector preparation, JIT tracing, compilation, first execution,
and repeated synchronized execution. `mwss` includes both poles; `gl` exercises
a grid without sampled poles. Results are generated at runtime, not release
qualification claims. Nonlinear products and dealiasing are outside this
linear operator's contract.

::: phydrax.discretization.PreparedSphericalVectorOperators

## Finite-radius cylindrical transforms

`CylindricalHankelPlan` prepares a fixed-order Bessel-zero quadrature on one
finite disk. The result retains the physical radial and transverse-wavenumber
measures together with root, inverse, orthogonality, Parseval, and resource
evidence. Nonlinear optical propagation initially admits only azimuthal order
zero; the transform itself does not imply modal closure.

::: phydrax.discretization.CylindricalHankelPlan

---

::: phydrax.discretization.PreparedCylindricalHankel

---

::: phydrax.discretization.CylindricalHankelEvidence

### Shared-grid quasi-cylindrical transforms

`SharedGridHankelPlan(radius, radial_count, mode_count)` prepares, for every
azimuthal mode `m`, the synthesis matrices of Bessel orders `m − 1, m, m + 1` on
the cell-centered radii and one k-grid from the zeros of `J_m`, together with
their Moore–Penrose pseudoinverses and rank, conditioning, and Penrose-residual
evidence. The quasi-cylindrical PSATD solver of
`phydrax.solver.maxwell.spectral` uses it for its circular field components.

::: phydrax.discretization.SharedGridHankelPlan

---

::: phydrax.discretization.PreparedSharedGridHankel

---

::: phydrax.discretization.SharedGridHankelEvidence

---

::: phydrax.discretization.SharedGridHankelOffset


## Radial-spherical and rotational transforms

::: phydrax.discretization.RadialLaguerrePlan

---

::: phydrax.discretization.FourierLaguerrePlan

---

::: phydrax.discretization.WignerTransformPlan

---

::: phydrax.discretization.WignerLaguerrePlan

---

::: phydrax.discretization.DirectionalBallWaveletPlan

---

::: phydrax.discretization.BallWaveletCoefficients


## Pseudospectral realization

::: phydrax.discretization.PseudospectralMethodPlan


---

::: phydrax.discretization.PreparedPseudospectralMethod

---

::: phydrax.discretization.PaddingDealiasingPlan

---
::: phydrax.discretization.PolynomialClosureDealiasingPlan

---


::: phydrax.discretization.ModalFilterPlan

---

::: phydrax.discretization.NoDealiasingPlan

---

::: phydrax.discretization.PreparedSpectralOperator

---

::: phydrax.discretization.spectral_hilbert_operator

---

::: phydrax.equations.CompiledSpectralDynamics

---

::: phydrax.equations.compile_spectral_residual

---

::: phydrax.equations.CompiledSpectralResidual

---

::: phydrax.equations.SpectralResidualCompilationReport

## Conservation and entropy

::: phydrax.discretization.SpectralConservationMethodPlan

---

::: phydrax.discretization.SpectralSplitFormPlan

---

::: phydrax.discretization.SpectralSplitFormReport

---


---

::: phydrax.discretization.PreparedSpectralConservationDynamics

---

::: phydrax.discretization.SpectralConservationDiagnostics

---

::: phydrax.discretization.SpectralEntropyDiagnostics

## Distributed full-complex execution

`DistributedSpectralExecutionPlan` binds an explicit scientific `owner_id`, one
`SpectralPrecisionPolicy`, canonical exact `admitted_payload_shapes`, and a slab,
pencil, or channel layout to one real `SpectralMeshTopology`. The admitted shapes
argument is required, canonicalized to a sorted unique tuple, and automatically
includes `state_shape`. Admission occurs before placement; an equal extent or
undeclared payload is not accepted by coincidence.

`from_discretization` derives `owner_id` and `precision` from the prepared
discretization. The direct constructor requires `owner_id`; its optional `precision`
defaults to float32/complex64. Raw coefficient/accumulation dtypes and caller-supplied
`stage_count`, `checkpoint_count`, and `closure_workspace_bytes` are not part of the
API.

The precision policy distinguishes physical, coefficient-storage, transform,
nonlinear, reduction, certification, output, and checkpoint roles. FFT arithmetic
uses transform dtype and returns coefficient-storage dtype. Reductions cast before
magnitude-square and summation. Preparation refuses a requested role that the active
JAX dtype policy cannot honor; it does not silently narrow it.

The identity boundary is explicit. `numerical_id` covers precision and
transform/normalization/storage/reduction semantics while excluding topology,
layouts, resources, and schedule. `execution_id` covers topology, layouts, ordered
stages, admitted payloads, and FFT resources. Owner-bound `plan_id` composes
`owner_id`, `numerical_id`, and `execution_id`.

Preparation reports `forward_sequence_id`, `inverse_sequence_id`,
`padded_forward_sequence_id`, and `padded_inverse_sequence_id`. These identify the
same private immutable local-transform and public `SpectralTranspose` operations that
execute. No public stage abstraction is introduced, and `SpectralTranspose` remains
the public atomic redistribution contract.

`SpectralResourceReport` owns `canonical_storage_bytes`,
`padded_storage_bytes`, `transform_workspace_bytes`, and `peak_live_bytes`.
`collective_payload_bytes` is algorithmic communication traffic and is excluded from
the live-memory ceiling. Workflow stages, closure work, and checkpoint storage belong
to LES, PSATD/PIC, mixed cosmology, or another consumer and are not FFT resources.

Execution keeps JAX `NamedSharding`, performs no host gather, and refuses unavailable
process-qualified device keys or incompatible global arrays. Slab requires a
one-dimensional mesh and rank at least two; pencil requires a two-dimensional mesh,
rank at least three, and divisible canonical/padded dimensions. Channel accepts
exactly Fourier--Chebyshev--Fourier, fingerprints its ordered horizontal axes,
replicates Chebyshev axis 1, and only invokes a supplied modal action; it is not a
distributed `ChannelStokesPlan`.

The unreleased qualification scope is JAX global arrays, full-complex C2C, regular
divisible slab/pencil shards, and the current horizontal channel action. R2C/C2R,
rank-local or vendor providers, uneven shards, mixed transforms, dynamic scheduling,
and implicit gather are not claimed. Forced-device, one-device, and same-host runs
cannot establish physical multi-device or multi-host qualification.

Consumer checkpoint boundaries are not interchangeable. LES and PSATD/PIC retain
exact `execution_id` through restart. Mixed cosmology admits scalar payload only and
uses a topology-neutral checkpoint schema bound to scientific owner plus
`numerical_id`; topology/layout/stage/resource identity is deliberately excluded from
compatibility while each concrete shard artifact still receives its exact execution
identity.

::: phydrax.discretization.SpectralMeshTopology

---

::: phydrax.discretization.SpectralLayout

---

::: phydrax.discretization.SpectralTranspose

---

::: phydrax.discretization.SpectralResourceReport

---

::: phydrax.discretization.DistributedSpectralExecutionPlan

---

::: phydrax.discretization.DistributedSpectralPreparationReport

---

::: phydrax.discretization.SpectralExecutionResult

---

::: phydrax.discretization.SpectralGlobalDiagnostics

### Distributed periodic LES

`DistributedPeriodicLESPlan` places one prepared scientific action on a real slab
or pencil JAX mesh. `compile_distributed_periodic_les` adds complete rotational
flow, `DistributedPeriodicLESMethodPlan` adds ETDRK/SSPRK admission, and
`DistributedPeriodicLESProductionPlan` keeps runtime segments, statistics,
checkpoints, and returned states device-resident. No host gather occurs in the
numerical path. Backend qualification remains exact and is never inherited. See
the [LES guide](../../guides_large_eddy_simulation.md#distributed-periodic-fourier-action).

::: phydrax.discretization.DistributedPeriodicLESPlan

---

::: phydrax.discretization.PreparedDistributedPeriodicLES

---

::: phydrax.discretization.DistributedPeriodicLESPreparationEvidence

---

::: phydrax.discretization.DistributedPeriodicLESStage

---

::: phydrax.discretization.DistributedPeriodicLESStepRestriction

---

::: phydrax.discretization.DistributedPeriodicLESRestartEvidence

---

::: phydrax.discretization.DistributedPeriodicLESParityEvidence

---

::: phydrax.applications.incompressible_flow.CompiledDistributedPeriodicLESDynamics

---

::: phydrax.applications.incompressible_flow.DistributedPeriodicLESMethodPlan

---

::: phydrax.applications.incompressible_flow.DistributedPeriodicLESProductionPlan

---

## Incompressible channel solves

The default `ultraspherical_banded` channel route uses pressure-eliminated
fixed-band systems and fixed-rank tau corrections internally while retaining
primitive velocity, pressure, and affine pressure-gradient results. The zero
horizontal mode owns wall tangential data, pressure recovery, and pressure-gradient
or bulk-flux control; nonzero modes use wall-normal velocity/vorticity elimination.
`dense_reference` is an explicit oracle and does not inherit banded-route production
or qualification evidence. The preparation report gives route,
bandwidth/rank, byte counts, pivot margin, and the required unsharded wall-normal
axis. Implicit variable-coefficient Stokes and distributed line solves are excluded;
channel LES adds state-dependent SGS stress explicitly.

The live periodic-flow and ETDRK state is full complex.
`HermitianSpectralCoordinates` may encode selected checkpoint leaves into independent
real coordinates, but it does not change callback state or peak nonlinear work.


::: phydrax.discretization.ChannelStokesPlan

---

::: phydrax.discretization.ChannelStokesPreparationReport

---

::: phydrax.discretization.HermitianSpectralCoordinates

## Incompressible forcing and statistics

Constant-power forcing uses volume-mean requested power and the native full-complex
inner product; insufficient forced-shell energy returns inactive, unsuccessful zero
forcing. OU forcing uses exact stochastic transitions in independent real modal
coordinates, but fluid-stage exactness requires explicit coupling by the caller.
Periodic shell statistics use unit weight per admissible full-complex mode and report
native integrals plus per-wavenumber densities. Channel statistics retain separate
signed wall shears, friction magnitudes, and half-height wall coordinates.

::: phydrax.applications.incompressible_flow.ConstantPowerFourierForcingPlan

---

::: phydrax.applications.incompressible_flow.SolenoidalHermitianFourierBasis

---

::: phydrax.applications.incompressible_flow.SolenoidalOUForcingPlan

---

::: phydrax.applications.incompressible_flow.PeriodicModalTurbulenceStatisticsPlan

---

::: phydrax.applications.incompressible_flow.SpectralChannelStatisticsPlan

## Incompressible spectral production

`PeriodicSpectralProductionPlan(dynamics, method, statistics, case, /, *, ...)`
binds exact compiled dynamics and initial modal content. Static LES requires
`PreparedLESStabilityGuardedETDRKMethod`; dynamic LES takes matching ordinary
prepared ETDRK and installs a transactional dynamic wrapper with optional
Lagrangian continuation. OU forcing is not composable with dynamic continuation.
`DistributedPeriodicLESProductionPlan` is the device-resident slab/pencil
counterpart. `SpectralChannelProductionPlan` uses exact-step SBDF2; its optional
equilibrium-traction owner retains a separate restart state. See
[Production and restart](../../guides_large_eddy_simulation.md#production-restart-and-statistics).

::: phydrax.applications.incompressible_flow.PeriodicSpectralProductionPlan

---

::: phydrax.applications.incompressible_flow.PeriodicSpectralProductionCase

---

::: phydrax.applications.incompressible_flow.PreparedPeriodicSpectralProduction

---

::: phydrax.applications.incompressible_flow.SpectralChannelProductionPlan

---

::: phydrax.applications.incompressible_flow.PreparedSpectralChannelProduction

---

## Bounded formulations

::: phydrax.discretization.SpectralBoundaryConditionPlan

---

::: phydrax.discretization.SpectralTraceTerm

---

::: phydrax.discretization.SpectralTraceConstraint

---

::: phydrax.discretization.BoundaryLiftPlan

---

::: phydrax.discretization.SpectralGalerkinMethodPlan

---

::: phydrax.discretization.PreparedSpectralGalerkin

---

::: phydrax.discretization.GeneralizedTauPlan

---

::: phydrax.discretization.PreparedTauSystem

### Paired periodic seam rows

`periodic_trace_row(prepared, /, *, source_terms, target_terms, transport=1.0)`
returns the exact host coefficient row `r` of one seam relation on a prepared
axis: for `u = sum_n c_n phi_n` in the axis' own coefficient convention,
`r @ c = sum_k t_k d^k u(upper) - transport * sum_k s_k d^k u(lower)`, with
physical-coordinate derivatives. This is the one-axis form of the
`phydrax.conditions.Periodic` relation `T[u](upper) - Gamma S[u](lower) = g`.

- Chebyshev (`T_n`) and Legendre (orthonormal `sqrt((2n + 1) / L) P_n`) rows reuse
  the owner endpoint traces; sine and cosine rows use the orthonormal
  point-synthesis rows.
- Fourier modes are `exp(2 pi i m_n (x - lower) / L) / sqrt(L)` in FFT order
  (complex packing; the even-count Nyquist mode uses `m = -N/2`). Both seam faces
  share these values, so the row is `(T_n - Gamma S_n) / sqrt(L)` with the exact
  modal jet multipliers. Identity transport with equal actions gives an exactly
  zero row: ordinary Fourier periodicity is structural, while antiperiodic, Bloch,
  and unequal-action relations yield nonzero transported rows.
- Rows are `float64` for real terms on real-synthesis bases and `complex128` for
  Fourier axes or any complex coefficient or transport. Rational axes (no finite
  seam endpoints) and constrained axes (nullspace coordinates) are refused with
  `ValueError`.

A tensor-product coefficient representation assembles the complete seam relation
as `kron(r, I_transverse)` along the identified axis, so every transverse
coefficient participates; no transverse sampling is involved. The hard coefficient
route consumes these rows through
`phydrax.enforcement.prepare_periodic_projection(..., route="coefficient",
representation=...)`.

::: phydrax.discretization.periodic_trace_row

## Precision

::: phydrax.discretization.SpectralPrecisionPolicy
