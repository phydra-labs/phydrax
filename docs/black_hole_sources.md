# Black-hole sources, provenance, rights, and qualification

This page is the source and claim ledger for the black-hole closure. It distinguishes
mathematical provenance, implementation ownership, external byte admission, numerical
qualification, scientific validation, production engineering, and deployment
permission. A citation explains a formulation; it does not license unrelated data,
software, tables, weights, or outputs.

## Authored implementation and bundled content

The Python implementation, documentation, synthetic examples, and synthetic
qualification inputs in this repository are authored Phydrax content under the
repository license. No external black-hole field archive, simulation checkpoint,
opacity/greybody table, waveform catalog, image, visibility dataset, executable,
model weight, or serialized Python object is bundled by the closure.

The small Schwarzschild QNM regression values exposed by
`schwarzschild_qnm_reference` are versioned numeric reference constants used only when
qualification is explicitly requested. They do not provide a catalog, a solver, or an
external asset runtime. Physical constants are supplied by `RelativityScaleContract`;
Hawking and entropy paths require explicit $\hbar$ and $k_B$.

## Mathematical provenance

The implementation follows standard formulations identified below. Links point to
primary or foundational literature for scientific traceability. They do not indicate
that external source code or datasets are part of the package.

| Capability | Formulation represented | Primary/foundational sources |
| --- | --- | --- |
| Schwarzschild and Kerr geometry | exact stationary metrics, horizon/ergosurface formulae, curvature invariants | [Schwarzschild (1916)](https://doi.org/10.1007/BF01595635); [Kerr (1963)](https://doi.org/10.1103/PhysRevLett.11.237) |
| Kerr--Newman equilibrium | charged rotating stationary horizon thermodynamics | [Newman et al. (1965)](https://doi.org/10.1063/1.1704351) |
| black-hole thermodynamics | area law, temperature, first law, Smarr relation, Einstein--Hilbert Wald equivalence | [Bardeen, Carter & Hawking (1973)](https://doi.org/10.1007/BF01645742); [Hawking (1975)](https://doi.org/10.1007/BF02345020); [Wald (1993)](https://doi.org/10.1103/PhysRevD.48.R3427) |
| Schwarzschild perturbations | Regge--Wheeler axial and Zerilli polar master equations | [Regge & Wheeler (1957)](https://doi.org/10.1103/PhysRev.108.1063); [Zerilli (1970)](https://doi.org/10.1103/PhysRevLett.24.737) |
| Kerr perturbations and QNMs | Kinnersley tetrad, Teukolsky separation, Leaver continued fractions | [Kinnersley (1969)](https://doi.org/10.1063/1.1664958); [Teukolsky (1973)](https://doi.org/10.1086/152444); [Leaver (1985)](https://doi.org/10.1098/rspa.1985.0119) |
| aligned BBH remnant | UIB2016v2 nonprecessing final-mass and final-spin fit inside a bounded calibration envelope | [Jiménez-Forteza et al. (2017)](https://doi.org/10.1103/PhysRevD.95.064024) |
| aligned NR mode surrogates | caller-content-bound, explicitly unauthenticated/unqualified integer-polynomial empirical-node reconstruction with nonprecessing negative-m symmetry and spin-minus-two synthesis | [Field et al. (2014)](https://doi.org/10.1103/PhysRevX.4.031006); [Varma et al. (2019)](https://doi.org/10.1103/PhysRevD.99.064045) |
| Hawking flux | Unruh-state mode occupations, signed greybody factors, bounded semiclassical balance | [Unruh (1976)](https://doi.org/10.1103/PhysRevD.14.870); [Page (1976)](https://doi.org/10.1103/PhysRevD.13.198) |
| relativistic fluids | Michel accretion and Valencia conservative hydrodynamics | [Michel (1972)](https://doi.org/10.1007/BF02710092); [Banyuls et al. (1997)](https://doi.org/10.1086/303604) |
| equilibrium torus | constant-angular-momentum Fishbone--Moncrief initial data | [Fishbone & Moncrief (1976)](https://doi.org/10.1086/154565) |
| GRMHD magnetic update | finite-volume ideal GRMHD with compatible constrained transport | [Gammie, McKinney & Tóth (2003)](https://doi.org/10.1086/374594); [Evans & Hawley (1988)](https://doi.org/10.1086/166684) |
| Z4c evolution | conformal Z4 with constraint damping and moving-puncture gauge families | [Bona et al. (2003)](https://doi.org/10.1103/PhysRevD.67.104005); [Bernuzzi & Hilditch (2010)](https://doi.org/10.1103/PhysRevD.81.084003) |
| puncture initial data | Brill--Lindquist and Bowen--York conformal data | [Brill & Lindquist (1963)](https://doi.org/10.1103/PhysRev.131.471); [Bowen & York (1980)](https://doi.org/10.1103/PhysRevD.21.2047) |
| marginal and dynamical horizons | null expansion, MOTS stability, quasilocal area/angular-momentum and flux balance | [Andersson, Mars & Simon (2005)](https://doi.org/10.1103/PhysRevLett.95.111102); [Ashtekar & Krishnan (2004)](https://doi.org/10.12942/lrr-2004-10) |
| radiation at null infinity | Weyl/$\Psi_4$ extraction, Bondi--Sachs characteristic fields and BMS charges | [Newman & Penrose (1962)](https://doi.org/10.1063/1.1724257); [Bondi et al. (1962)](https://doi.org/10.1098/rspa.1962.0161); [Sachs (1962)](https://doi.org/10.1098/rspa.1962.0206) |
| relativistic transfer | invariant $I_\nu/\nu^3$ transfer along geodesics and polarized Stokes propagation | [Lindquist (1966)](https://doi.org/10.1016/0003-4916(66)90207-7) |
| thermal synchrotron | MNY96 equation 31 Stokes-$I$ ultra-relativistic shape only; polarization and Faraday forms are independent and reference-unqualified | [Mahadevan, Narayan & Yi (1996)](https://doi.org/10.1086/177422), [astro-ph/9601073](https://arxiv.org/abs/astro-ph/9601073) |
| direct interferometry | signed Fourier visibility, closure phase and closure amplitude | [Jennison (1958)](https://doi.org/10.1017/S036839310010376X) |

Specific numerical policies—fixed capacities, stable identities, fail-closed status,
independent residuals, host/device boundaries, JAX transformations, rollback,
checkpointing, and resource accounting—are Phydrax architecture, not claims made by
those papers.

The normalized polynomial-EIM waveform runtime, external NR-surrogate model-data
rights, and PSD-weighted comparison provenance are detailed separately in the
[gravitational-wave source ledger](gravitational_wave_sources.md).

## External artifact rights

External black-hole inputs enter only through caller-directed host APIs. The shared
artifact-admission boundary requires a trusted local root, bounded bytes, allowed
suffixes and license IDs, regular files, no symbolic-link traversal, and exact SHA-256/size
matching. Pickle, dill, and joblib suffixes are refused.

`BlackHoleArtifactRights` binds artifact kind, source artifact, producer and version,
model identity, coverage, exact digest/size, license/source/attribution, and separate
permission bits for commercial use, training, redistribution, derivative use, model
execution, and export. A `BlackHoleArtifactUsePolicy` must request only allowed uses.
`BlackHoleArtifactSchema` then binds the exact chart/frame/quantity/topology/unit or
image/visibility/waveform/model semantics. Interpretation-changing losses require an
explicit valid `AdapterReport`; unwaived loss fails.

The production `NumericalRelativityArtifactBinding` retains the complete rights and
use-policy objects plus every material identity and permission; it does not reduce them
to a license string. `phydrax.service.ArtifactRights` is the separate service-delivery
record joining a scientific artifact ID to rights/use-policy IDs and permitted
principals/actions. It does not replace scientific interchange rights or grant a
permission absent there.

These records do not infer permission from publication, accessibility, a source URI, or
an upstream software license. Code rights, data rights, model-weight rights, generated-
output rights, and authorized egress are independent. `NeutralBlackHoleArtifact` is an
inert, content-bound host record; numeric-model mapping never deserializes or executes
model bytes.

## Qualification levels and authorities

| Level | Required evidence | Explicit nonclaim |
| --- | --- | --- |
| numerical kernel | finite state, declared-domain membership, residual/conservation/constraint gates, stable status, relevant derivative evidence | continuum convergence, external agreement, or physical fidelity beyond the case |
| cross-resolution/reference | independent coarse/fine or analytic/reference comparison, exact mode/chart/topology/precision/source binding | a different mode, chart, spin, EOS, resolution, or boundary |
| scientific profile | every required observable and invariant passes for one registered `SupportTuple` with source and uncertainty where applicable | observational validation, broad astrophysical prediction, or deployment |
| execution profile | setup/lowering/compile/synchronized execution, resource evidence, determinism, restart, and scaling on the exact exercised topology | scientific accuracy or unexercised hardware scaling |
| production profile | exact domain/support/run/case/precision/resource/checkpoint/output/failure bindings | release approval, license rights, or operational authorization |
| deployment decision | independent legal, security, quality, operational, intended-use, and environment review | any tuple or use beyond the explicit decision |

Product-local `qualified` values are necessary inputs to a larger dossier, not global
production labels. Synthetic profiles demonstrate implementation behavior only.

## Executable lanes

The benchmark entry points are:

| Lane | Entry point | Evidence scope |
| --- | --- | --- |
| exact geometry | `benchmarks/black_hole_geometry.py` | exterior Kerr Boyer--Lindquist metric/domain, Kretschmann/Pontryagin values, and one coordinate JVP |
| GR imaging | `benchmarks/gr_imaging.py` | public Kerr `GRRayPlan` exterior null-ray history, invariant scalar transfer, and an emission-scale derivative |
| perturbations | `benchmarks/black_hole_perturbations.py` | scalar $s=\ell=m=0$ Schwarzschild $M=1$ Riccati/log-amplitude solve at $\omega=0.4-0.05i$, default 65 nodes to $40M$, fixed 32 RK4 substeps/interval and order-12 infinity series; retains complex match, gated Chebyshev ODE residual and residual JVP; explicitly not a solved QNM |
| waveform surrogates | `benchmarks/gravitational_wave_surrogates.py` | aligned-spin polynomial-EIM mode reconstruction and JVP, canonical-grid phase/time match, and UIB2016v2 remnant evaluation |
| relativistic matter | `benchmarks/relativistic_matter.py` | SRHD recovery, flux, causal bounds, and JVP |
| numerical relativity | `benchmarks/numerical_relativity.py` | periodic linear-wave Z4c RHS, constraints, and JVP |
| production runtime | `benchmarks/black_hole_runtime.py` | atomic two-dimensional periodic all-active Valencia GRMHD SSPRK3/CT rollout and step-size JVP |

Each lane measures bounded local JAX setup, lowering, compile, synchronized execution,
logical byte estimates, relevant compiler properties, and landed-kernel status. It does
not establish convergence outside configured capacities, observational accuracy,
distributed scaling not actually exercised, production qualification, or deployment
authorization.

### Public examples

Examples exercise public APIs and print status; they are not qualification evidence.

| Example | Exact scope and nonclaim |
| --- | --- |
| `examples/kerr_horizon_thermodynamics.py` | stationary Kerr Killing horizon, SI entropy/temperature and first-law/Smarr evidence; no dynamical-horizon claim |
| `examples/kerr_shadow_rays.py` | future-ingoing Kerr bounded observer-screen ray fan with capture/escape and conservation evidence; no rendering |
| `examples/polarized_fast_light_grrt.py` | exact `GRRayResult` plus metric to `PolarizedRayPath`; snapshot chart matched to path chart; path/snapshot-bound active-segment midpoint sampling; invariant transfer; numerical path/transfer and MNY96 Stokes-$I$ support only; active polarization/Faraday and the composite prediction remain reference-unqualified |
| `examples/qnm_scattering_hawking.py` | qualified Schwarzschild QNM kept separate from native real-frequency scalar radial solves; computed qualified greybody factors and neighboring-frequency slope feed the Hawking spectrum, whose omitted-tail evidence remains explicitly unqualified |
| `examples/compact_object_accretion.py` | transonic Michel and constant-angular-momentum Fishbone--Moncrief initial data; no evolution |
| `examples/fixed_grid_z4c.py` | two bounded periodic fixed-grid SSPRK33 Z4c steps |
| `examples/binary_black_hole_extraction.py` | Brill--Lindquist candidate-surface, null/quasilocal diagnostics and finite-radius $\Psi_4$ multipoles; no MOTS certification, evolution, or asymptotic waveform |
| `examples/simulation_product_visibility.py` | physical Stokes image to neutral FITS/UVFITS payloads, direct visibilities, polarization and closures; no rendering or external files |

Run all deterministic qualification controls, or repeat `--profile` for a selected
subset. `--valid-at` is required; it binds the observation time and must be the
operator's actual nonnegative Unix timestamp:

```console
VALID_AT="$(date +%s)"
python tools/black_hole_qualification.py --valid-at "$VALID_AT" --output black-hole-qualification.json
python tools/black_hole_qualification.py --valid-at "$VALID_AT" --profile geometry --profile thermodynamics
```

The tool has eleven exact profiles. Each row below is the exercised support, not the
broader `api_domain` retained by the report.

| Profile | Exact exercised support | Important limitation |
| --- | --- | --- |
| `geometry` | Kerr $M=2$, $a=0.7$; exterior point $(0.3,7,1.1,-0.2)$; horizon polar angle $1$; Boyer--Lindquist/future-ingoing charts; mostly-plus | only that subextremal point/horizon; no dynamical or numerical metric |
| `thermodynamics` | Kerr $M=2$, $J=0.8$; tangent $(dM,dJ)=(0.3,-0.2)$; SI scale | only that smooth directional derivative; stationary Killing horizon only |
| `rays` | two analytic mostly-plus Minkowski null lanes with tangents $(1,\mp1,0,0)$; affine nodes $(0,0.125,0.25,0.375,0.5)$; event roots at $0.25$ | no curved-spacetime image convergence |
| `grrt-interferometry` | scalar slab $(L,j,\alpha,I_0)=(2,4,0.5,1)$; $3.5$ Jy point at $(0.125,-0.25)$ rad, $230$ GHz, two listed UV points; four-station closure | synthetic scalar/visibility control; no slow-light or polarized-microphysics qualification |
| `perturbations-scattering-hawking` | Schwarzschild $M=1$, spin-$-2$ $(\ell,m,n)=(2,2,0)$ Regge--Wheeler QNM against `schwarzschild-leaver-Momega-reference-v1`, 65 nodes, $r_{\rm out}=30M$, 32 fixed RK4 substeps/interval, order-12 infinity series, match tolerance $10^{-7}$ and relative ODE tolerance $10^{-5}$; achieved match $\approx1.9\times10^{-9}$ and relative ODE residual $\approx1.85\times10^{-7}$; separate scalar $(0,0)$ off-root radial control at $0.4-0.05i$ on 17 nodes to $40M$; scalar $(1,0)$ scattering at $0.2,0.4$ with $0.7/0.3$ reflected/transmitted flux; one massless-scalar Hawking mode | QNM evidence qualifies only that fundamental/plan; no external greybody reference; absent qualified tail evidence deliberately keeps Hawking unqualified |
| `grhd-grmhd` | gamma-law $5/3$ and $4/3$ EOS; Minkowski; 6-cell 1D periodic GRHD SSPRK33 at $dt=10^{-3}$; $2\times2$ periodic vector-potential CT GRMHD SSPRK33 at $dt=10^{-4}$; Michel critical-radius $M=1$ | uniform/runtime and critical-radius controls only; no shock/turbulence/production evidence |
| `z4c-initial-data-horizons-waves` | periodic $5^3$ flat vacuum Z4c SSPRK33 at $dt=0.05$; isotropic Schwarzschild $M=1$ at $r=4$; spherical isotropic radius $0.5$, bandlimit 3; vacuum $\Psi_4$ sign control, bandlimit 3 | flat and single-hole controls only; no binary evolution or waveform convergence |
| `coupling` | analytic same-stage SSPRK33 Z4c--GRHD participants; fixed two-lane topology; initial values $(0.4,1)$; $dt=0.1$; atomic commit | coordinator/ledger control; physical kernels retain separate qualification |
| `scaling-restart` | one device; Z4c global $4^3$ with $(1,1,1)$ decomposition; $2^3$ checkpoint state; exact same-topology restart | no multi-device/multi-host scaling or performance evidence |
| `production-interchange` | $5^3$ periodic flat-vacuum Z4c SSPRK33 at $dt=0.01$; float32/local JAX; one CPU and 1,000,000 host-memory bytes; 500,000-byte checkpoint/output staging bounds; one 1,024-byte input and one 1,024-byte output manifest/artifact; 64-byte cancellation detail; scientific support `qualification:fixed-grid-z4c`; locally generated CC0 checksum-pinned opaque inert bytes with `source_format="opaque-binary"`, producer `qualification:phydrax` at the installed version, model `qualification:opaque-field-byte-admission`, and coverage `checksum-rights-resource-and-production-binding-only` | no deployment support; bytes are not parsed or claimed to be openPMD/HDF5; technical binding only |
| `advanced` | exact listed KN--AdS $(M,a,Q,L)=(2,0.3,0.2,10)$; Schwarzschild massive-field $(M,\mu,\ell,n)=(2,0.1,1,0)$; self-force $\ell_{max}=14$; Kerr inverse $(M,a)=(2,0.4)$ along $(0.25,-0.1)$; two-temperature $(100,400)$ for $dt=2$; single-cell grey M1/Ohm/force-free; characteristic $(9\text{ times},3\text{ radii},\ell=2,m=2)$; octahedral BMS through $\ell=1$; complete flat five-time/two-generator offline event control | only those controls; event result remains unqualified without a qualified terminal surface; no learned/posterior/external reference evidence |

For the gravitational radial gate, the order-12 infinity series is generated from
$V/f=z^2W(z)$ coefficients. Regge--Wheeler contributes finite $W$ coefficients;
Zerilli generates the exact rational-denominator expansion recursively. The profile
also requires strict isospectral agreement between separately gated axial
Regge--Wheeler and polar Zerilli fundamental solves. Neither sector borrows the other's
matching, asymptotic or Chebyshev ODE evidence.

The runtime manifest binds the qualification runner SHA-256, installed Phydrax Python
source-tree SHA-256, exact Diffrax/Equinox/JAX/JAXlib/NumPy versions, Python
implementation/version, OS/release, machine, byte order, JAX backend/process
count/index, every device identity, and precision disposition. Scientific controls use
float64; production and native electromagnetic controls use float32. Every profile
carries the same `manifest_id`.

Validity is observation-time-only: `evaluated_at_unix_seconds` and
`expires_at_unix_seconds` both equal `--valid-at`, so the report asserts no future
validity. A missing, failed, mismatched, or reused profile/runtime manifest is not a
pass. Documentation, examples, and benchmark output are never qualification evidence.

## Supported capability domains and nonclaims

The landed code supports the exact domains documented in the focused guides:

- [exact black-hole geometry and stationary equilibrium](guides_black_hole_geometry.md);
- [separated perturbations, QNMs, real scattering, and Hawking flux](guides_black_hole_perturbations.md);
- [relativistic material, plasma, and radiation systems](guides_relativistic_matter.md);
- [general-relativistic rays, transfer, images, and interferometry](guides_black_hole_imaging.md);
- [fixed-grid/fixed-capacity numerical relativity, distinct horizon products,
  radiation extraction, block AMR, distribution, and restart](guides_numerical_relativity.md); and
- [resource, artifact, qualification, and production boundaries](guides_black_hole_execution.md).

There is no claim of unrestricted coordinates, arbitrary matter or quantum field
content, complete QNM/greybody catalogs, quantum-gravity evaporation, generic nonlinear
characteristic evolution, second-order self-force, long-duration binary-black-hole
accuracy, external-code parity, multi-host scale, real-observation validity, or a
released deployment profile unless an exact current dossier states otherwise.

## PNPL is separate from technical evidence

The repository license is PNPL. Numerical success, scientific qualification, execution
evidence, a production binding, or a technically complete dossier does not grant
commercial, service, redistribution, training, model-execution, data-egress, or
deployment rights. Those uses remain subject to PNPL, every external artifact's terms,
and any required written authorization and independent deployment decision. The
software cannot manufacture license authority, data rights, signatures, attestations,
or approval.