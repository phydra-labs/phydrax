# Black-hole perturbations, QNMs, scattering, and Hawking flux

The perturbation surface uses explicit mode, branch, boundary, normalization, and
source identities. Complex-frequency quasinormal modes (QNMs), real-frequency
scattering, and semiclassical Hawking flux are three different scientific products.
None accepts another as an interchangeable shortcut.

## Shared separation convention

`PerturbationConvention` fixes

$$
\Psi\propto e^{-i\omega t+i m\phi},
$$

an orthonormal spin-weighted spherical-harmonic expansion, unit Euclidean norm for
the angular coefficients, a real-positive target coefficient for phase fixing, the
Kinnersley tetrad for Kerr radial coefficients, and unit maximum modulus only for a
displayed finite-domain radial profile. `SeparatedMode` records spin weight
$s\in\{-2,-1,0,1,2\}$, $(\ell,m)$, overtone, sector, family, background, and
convention. It never selects a branch by eigenvalue ordering.

`RadialBoundaryCondition` records the horizon and infinity wave senses. The QNM
choice is ingoing at the future horizon and outgoing at infinity. A real-frequency
scattering channel has incident, reflected, and transmitted amplitudes and its own
flux normalization. Display normalization is never interpreted as physical flux.

`PerturbationStatus` distinguishes invalid mode/domain, nonfinite evaluation,
angular or radial nonconvergence, residual or asymptotic failure, nonisolated modes,
and invalid derivatives.

## Angular and radial equations

`SpheroidalAngularPlan` projects the spin-weighted spheroidal equation on a fixed
spin-spherical Galerkin basis at spheroidicity $c=a\omega$. It retains the selected
branch, eigenpair residual, phase, truncation comparison, and separation constant.
The dense operator is certified self-adjoint only for real $c$.

`SchwarzschildRadialPlan` uses a two-sided Riccati/log-amplitude solve for the
Regge--Wheeler or Zerilli family. Chebyshev radial nodes are mapped to
$x=\log[(r-2M)/M]$. From a first-order horizon series and a declared fixed-order
infinity series, it evolves $(\log R,R'/R)$ inward/outward with exactly
`integration_substeps` RK4 substeps on every spectral interval. The default infinity
order is 12 and default substep count is 32; both are fixed plan identity, never
adaptive work.

The infinity recurrence is generated from the generic expansion
$V/f=z^2W(z)$ with $z=1/r$, not from a Regge--Wheeler-only hard-coded tail.
Regge--Wheeler supplies its finite $W$ coefficients. The polar Zerilli branch generates
the exact rational denominator expansion recursively, so its order-12 boundary data
use the same generic amplitude recurrence without replacing the rational potential by
an unrelated polynomial.

The two log-amplitude branches are aligned at the declared match index before exponent
reconstruction and finite-domain display normalization. The returned complex matching
residual is the normalized left/right logarithmic-derivative mismatch, not its absolute
value. Native Chebyshev derivative matrices independently evaluate the reconstructed
ODE residual on interior nodes. `RadialResidualEvidence.qualified` requires finite
profile/residuals, valid domain and asymptotics, complex matching magnitude within
`matching_tolerance`, and relative ODE residual within `residual_tolerance`; both are
acceptance gates.

The public Regge--Wheeler potential is

$$
V_{\rm RW}=f\left[\frac{\ell(\ell+1)}{r^2}+
\frac{2M(1-s^2)}{r^3}\right],\qquad f=1-\frac{2M}{r},
$$

and the gravitational even-parity Zerilli potential is also available.
`KerrTeukolskyRadialPlan` remains the separate fixed two-sided matcher for the
untransformed equation $\Delta R''+B R'+C R=0$ in Boyer--Lindquist radius; its result
has its own boundary, asymptotic, complex match and independent ODE-residual gates. No
infinite-domain convergence is inferred from either finite grid.

## Complex-frequency QNMs

`QnmSolvePlan` couples angular and radial Leaver residuals through the native
nonlinear and linear-sensitivity owners. Frequencies are returned as dimensionless
$M\omega$; the recurrence uses $2M=1$ internally while $a\omega$ remains invariant.
The plan supports $s=-2,-1,0$, requires an explicit root seed, branch ID,
subextremal spin interval, nonlinear method and termination, implicit-forward
sensitivity policy, and separate bounded continued-fraction plans. The inversion
indices select the declared angular branch and radial overtone.

`solve_qnm` reports the nonlinear result, angular/radial depth comparisons,
independent angular and radial resolution results, root singular values and
condition, branch continuation coordinate, reference differences when an explicit
`QnmReferenceMode` is supplied, and the branch-local spin derivatives. The built-in
`schwarzschild_qnm_reference` supplies only a small versioned regression set; it is
not the numerical solver or a broad catalog.

The `perturbations-scattering-hawking` qualification profile admits exactly the
Schwarzschild $M=1$, spin-$-2$, $(\ell,m,n)=(2,2,0)$ Regge--Wheeler fundamental
against `schwarzschild-leaver-Momega-reference-v1`. Its independent radial gate is
fixed at 65 nodes, $r_{\rm out}=30M$, 32 RK4 substeps per spectral interval, order-12
infinity asymptotics, `matching_tolerance=1e-7`, and relative
`residual_tolerance=1e-5`. The qualified evaluation records approximately
$1.9\times10^{-9}$ matching magnitude and $1.85\times10^{-7}$ relative ODE residual.
Those values qualify that exact axial mode/plan only, not another grid, outer radius,
mode or Kerr radial solve. The profile separately requires a strict axial/polar
isospectral regression: the Regge--Wheeler and Zerilli fundamental solves must each
pass their own radial gates before their frequencies are compared. Isospectral
agreement does not let one sector inherit the other's residual evidence.

`QnmStatus` separates nonfinite values, nonlinear nonconvergence, unresolved
continued-fraction depth, unresolved angular resolution, nondecaying roots, and
nonsimple roots. `QnmStatus.RADIAL_RESOLUTION_UNRESOLVED` is the distinct terminal
status when the independent `radial_resolution.qualified` gate fails, and that gate is
required for `converged`. `QnmDerivativeStatus` separately distinguishes primal, depth,
angular-branch, simple-root, finite, and sensitivity-solve failures;
`RADIAL_RESOLUTION_UNRESOLVED` and `RADIAL_DERIVATIVE_UNRESOLVED` distinguish its two
radial gates. The implicit derivative requires `radial_resolution.derivative_valid`.
A complex root can therefore remain unqualified or derivative-invalid.
`qnm_continuation_problem` exposes the same residual to the native parameter
continuation runtime; it does not add automatic branch switching.

## Real-frequency scattering and superradiance

`BlackHoleScatteringPlan` accepts only `family="scattering"`, spin weight zero, and
the declared massless-scalar Killing-energy normalization:

$$
F_{\rm in}=\omega |I|^2,\quad
F_{\rm ref}=\omega |R|^2,\quad
F_H=(\omega-m\Omega_H)|T|^2.
$$

`solve_black_hole_scattering` recomputes the flux residual
$F_{\rm in}-F_{\rm ref}-F_H$ from amplitudes while independently checking the radial
Wronskian residual. A channel is superradiant only when
$\omega-m\Omega_H<0$, reflected flux is amplified, and the signed greybody factor is
negative. `SuperradianceStatus` distinguishes nonsuperradiant, superradiant,
threshold, and inconsistent lanes. `BlackHoleScatteringStatus` retains finite,
frequency, incident-flux, Wronskian, conservation, and regime failures.

Qualification requires a `ScatteringQualificationEvidence` that binds the exact mode
and independently solved radial source. The derivative is not admitted in the
corotation tolerance band; the caller supplies the independently resolved slope
$d\Gamma/d\omega$ used by the paired thermal limit.

`SchwarzschildScatteringSolvePlan` is the native computed path for a scalar
Schwarzschild channel. It requires the ingoing-horizon/outgoing-infinity basis
convention, integrates the horizon solution, decomposes it against both finite-radius
infinity waves with native dense linear algebra, and normalizes incident/reflected/
horizon amplitudes by their actual basis fluxes. Coarse/refined RK4 work, asymptotic
support, decomposition conditioning, incident amplitude, canonical relative and
absolute flux/Wronskian ledgers, and the low-frequency scalar control are separate
gates. The reported `dGamma/domega` uses a centered half-step estimate; comparison
with the full-step estimate is performed on the scale-invariant derivative
`dGamma/d(M omega)`, and every neighboring solve must pass the same radial and flux
ledgers before `derivative_valid` is true. Invalid frequencies return a typed result
status rather than triggering a traced exception. The nested independent-amplitude
ledger remains unqualified; the computed wrapper owns its distinct numerical
qualification evidence.

## Semiclassical Hawking spectrum and evaporation

`QuantumFieldSpecies` declares one free species, spin-statistics assignment,
multiplicity, and Compton wavenumber. `HawkingSpectrumPlan` fixes the species,
frequency quadrature, $(\ell,m)$ slots, active mask, scale, and absolute/relative
tail tolerances. Explicit $\hbar$ and $k_B$ are required.

`HawkingScatteringData` is a neutral carrier of independently qualified signed
greybody factors and corotation slopes. It does not solve scattering. A bosonic
superradiant lane retains negative greybody factor; the slope resolves the removable
Bose limit at corotation. `HawkingTailEvidence` separately bounds omitted frequency
and angular-mode contributions to number, Killing-energy, and axial-angular-momentum
flux. `evaluate_hawking_spectrum` requires the exact `KerrEvaporationState` alongside
the scattering and horizon inputs. `HawkingSpectrumResult.source_state` retains that
mass, angular momentum, elapsed time, step index and lineage; `bound_to(state)` checks
all of them rather than accepting a matching string ID.

`KerrEvaporationPlan` performs a fixed-capacity, fail-closed coevolution of geometric
$M$ and $J$. Every proposed step requires a qualified fixed-shape spectrum bound to the
exact proposed state and passes adiabatic, fractional mass, spin-change, subextremal,
and semiclassical bounds. A stale spectrum terminates with
`STALE_SPECTRUM_BINDING`; a step that would enter the quantum-gravity regime is not
committed. The last accepted semiclassical state remains authoritative.
`HawkingEvaporationTermination` also distinguishes capacity, initial/Kerr state,
coverage, scattering, flux, adiabaticity, semiclassical, and nonfinite-flux
termination. This is a bounded semiclassical evolution, not quantum gravity or a
complete Standard Model evaporation history.

## Advanced perturbative capabilities

The same explicit-evidence pattern covers:

- charged massive-scalar quasi-bound-state shooting on Kerr--Newman;
- simple-pole excitation residues and Green-function assembly;
- second-order quadratic ringdown mode coupling;
- fixed-capacity first-order mode-sum self-force with supplied regularization
  parameters and an explicitly fitted high-$\ell$ tail.

These are separate plans with separate source and branch identities. They do not imply
second-order self-force, nonlinear spacetime evolution, generic field content, or
complete resonance catalogs.

`FixedBranchInverseAdapter` provides JVP/VJP pairing and central-difference evidence
only while the numeric branch signature remains unchanged.
`kerr_geometry_inverse_adapter`, `kerr_thermodynamics_inverse_adapter`, and
`simple_qnm_root_inverse_adapter` keep chart, ensemble, and simple-root identities
separate. Their model-evaluation helpers preserve source domain/root status; they do
not differentiate discrete branch selection or reinterpret an invalid primal result.

## Derivative boundary

Smooth angular/radial kernels remain differentiable inside one fixed mode, basis,
chart, grid, boundary family, root branch, simple-pole branch, active mask, and
qualification source. Mode selection, overtone/sector changes, root switching,
continued-fraction capacity changes, corotation classification, tail truncation,
termination, and failed acceptance are discrete boundaries. Use each result's
`derivative_valid` and more specific derivative status; never infer differentiability
from `successful` alone.