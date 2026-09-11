# Global spectral methods

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
raises explicitly in either AD direction. This follows the prepared plan's
`NonTrainableState` solver contract. The precomputed execution retains its native
array differentiation semantics, including kernel derivatives when explicitly
requested outside solver trainable partitioning.

`SphericalSpectralDiscretization.evaluate_angles(
coefficients, theta, phi, /, *, frame_angle=0)` broadcasts the three real
numerical angle arguments and evaluates in the longitude-labelled
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
longitude-labelled limiting tangent frame: they may depend on longitude even
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

`DistributedSpectralExecutionPlan` binds slab, pencil, or channel layouts to one real
`SpectralMeshTopology`. Preparation fixes every physical/modal redistribution,
canonical/padded shape, precision, transform scale, collective count, local shape, and
byte bound. Execution keeps JAX `NamedSharding`, performs no host gather, and refuses
unavailable devices or incompatible global arrays.

Slab and pencil routes provide full-complex Fourier transforms. The channel schedule
partitions horizontal Fourier axes while replicating Chebyshev axis 1 and only invokes
a supplied modal action; it is not a distributed `ChannelStokesPlan`. One-device
topology is local. Caller meshes provide actual multi-device execution, while
multi-host launch and scaling evidence remain outside this plan.

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

## Precision

::: phydrax.discretization.SpectralPrecisionPolicy
