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

::: phydrax.discretization.SphericalModeLayout

---

::: phydrax.discretization.SphericalHarmonicPlan

---

::: phydrax.discretization.SphericalSpectralPlan

---

::: phydrax.discretization.SphericalSpectralDiscretization

---

::: phydrax.discretization.spherical_laplacian_operator

---

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

## Incompressible channel solves

The default `ultraspherical_banded` channel route uses pressure-eliminated
fixed-band systems and fixed-rank tau corrections internally while retaining
primitive velocity, pressure, and affine pressure-gradient results. The zero
horizontal mode owns wall tangential data, pressure recovery, and pressure-gradient
or bulk-flux control; nonzero modes use wall-normal velocity/vorticity elimination.
`dense_reference` is an explicit oracle and does not inherit banded-route production
or qualification evidence. The preparation report gives route,
bandwidth/rank, byte counts, pivot margin, and the required unsharded wall-normal
axis. Variable viscosity and distributed line solves are excluded.

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

`PeriodicSpectralProductionPlan` takes an already prepared Hermitian ETDRK method,
periodic statistics, a source problem identity, absolute start/end times, nominal
step, and checkpoint cadence. Optional constant-power forcing is either identity-
verified as already compiled or explicitly added by the adapter; adapter wiring
requires the supplied drift to be unforced. `SpectralChannelProductionPlan` takes
the exact-step prepared SBDF2 method plus velocity/pressure Hermitian coordinates and
derives its step from the method; end, output, and statistics-window bounds must lie
on that lattice. Both prepare a durable checkpoint root before initialization.

::: phydrax.applications.incompressible_flow.PeriodicSpectralProductionPlan

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
