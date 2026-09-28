# Electromagnetic radiation sources

The native electromagnetic-radiation implementation shares no upstream runtime
or source code. External works define the physics, reference formulas, and
comparison problems; GPL-licensed programs are consulted only as literature or
pinned external oracles, never copied.

## Trajectory radiation

| Source | Role | Native boundary |
|---|---|---|
| J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999), §14.1–14.6 and problem 14.15 | Liénard–Wiechert fields, acceleration-form spectrum, Liénard power, Schott harmonics of circular motion | Formulas re-derived in SI with the `exp(−iωt)` convention; test references evaluate them with `phydrax.special.jv`. |
| L. D. Landau and E. M. Lifshitz, *The Classical Theory of Fields*, 4th ed. (1975), §73–74 | Retarded potentials, radiation of a rotating charge, harmonic decomposition | Reference only. |
| G. A. Schott, *Electromagnetic Radiation*, Cambridge University Press (1912) | Harmonic power of circular orbits | Reference only. |
| J. Schwinger, Phys. Rev. 75, 1912 (1949) | Synchrotron spectrum `F(x) = x ∫ₓ^∞ K₅/₃` and `ω_c = (3/2) γ³ ω₀` | `phydrax.special.synchrotron_f` owns `F`; used for the synchrotron comb and the Liénard-total tail. |
| A. H. Barnett, J. Magland, L. af Klinteberg, SIAM J. Sci. Comput. 41, C479 (2019) | Exponential-of-semicircle kernel and Type-3 nonuniform transforms | The node-gridded route uses the native Type-3 plan; its floor is the plan's requested tolerance times the jump mass. |
| Lorentz invariance of `(1/ω²) d²W/(dω dΩ)` (photon-number phase-space density) | Frame transformation of complete vacuum emission | `phydrax.transform_spectral_energy`; truncated windows and media are refused. |
| O. Heaviside, *Phil. Mag.* 27, 324 (1889); R. P. Feynman, *Lectures on Physics* II, §26-2 and eq. 21.1 | Field of a uniformly moving charge written in its present position; Feynman's retarded-field form | Near-zone test reference, evaluated with the cancellation-free `1/γ² + (β·R̂)²`. |
| M. Born, *Ann. Phys.* 30, 1 (1909); T. Fulton and F. Rohrlich, *Ann. Phys.* 9, 499 (1960) | Closed-form field of hyperbolic motion | Near-zone test and example reference; re-derived and checked against an independent retarded-time solve. |
| G. E. Alefeld, F. A. Potra, Y. Shi, *ACM TOMS* 21, 327 (1995) | Bracketed root enclosure (TOMS748) | `phydrax.nonlinear.scalar_root` solves the retarded condition after the index bisection; derivatives come from the implicit function theorem. |

## Kinetic dielectric and cyclotron maser growth

| Source | Role | Native boundary |
|---|---|---|
| T. H. Stix, *Waves in Plasmas*, AIP (1992), ch. 10–11 | Hot magnetized susceptibility, bi-Maxwellian harmonic sum, Bernstein waves | Tensor re-derived from the unperturbed-orbit integral for drifting bi-Maxwellians; the Bernstein test reference is an independent SciPy root of the perpendicular relation. |
| B. D. Fried and S. D. Conte, *The Plasma Dispersion Function*, Academic Press (1961); L. D. Landau, J. Phys. USSR 10, 25 (1946) | `Z(ζ)`, its continuation, Langmuir roots | `Z = i√π w` from `phydrax.special.wofz`; published roots at `kλ_D = 0.3–0.5` and a quadrature-plus-residue reference below the real axis. |
| S. P. Gary, *Theory of Space Plasma Microinstabilities*, Cambridge (1993), ch. 7 | Parallel bi-Maxwellian R/L dispersion, whistler anisotropy instability | Scalar reference relation solved independently in the test. |
| I. P. Shkarofsky, Phys. Fluids 9, 561 (1966); M. Bornatici, R. Cano, O. De Barbieri, F. Engelmann, Nucl. Fusion 23, 1153 (1983) | Weakly relativistic tensor, `F_q(z, a)`, Dnestrovskii absorption `Im F_q(z, 0)` | Closed forms in `Z`, the `q` recurrence and the momentum-space quadrature are derived natively and checked against arbitrary-precision contour integrals. |
| C. S. Wu and L. C. Lee, ApJ 230, 621 (1979); D. B. Melrose and G. A. Dulk, ApJ 259, 844 (1982) | Resonance-ellipse growth of the cyclotron maser; loss-cone drivers | Anti-Hermitian relativistic susceptibility integrated natively; the test reference is the closed-form small-FLR loss-cone integral at perpendicular propagation. |
| K. D. Dory, G. E. Guest, E. G. Harris, Phys. Rev. Lett. 14, 131 (1965) | Loss-cone distribution `u⊥^{2j} exp(−u²/u_t²)` | `LossConeDistribution`. |

## Magnetobremsstrahlung, thermal synchrotron and free–free

| Source | Role | Native boundary |
|---|---|---|
| D. B. Melrose, Ap&SS 2, 171 (1968); R. Ramaty, ApJ 158, 753 (1969); D. B. Melrose and R. C. McPhedran, *Electromagnetic Processes in Dispersive Media*, Cambridge (1991) | Mode-resolved gyromagnetic emissivity and absorption, mode-energy normalization | Re-derived in SI from the mode-energy formalism with the C1 polarization vectors; Kirchhoff checked exactly. |
| G. Bekefi, *Radiation Processes in Plasmas*, Wiley (1966), ch. 6 | Thermal cyclotron harmonics of a nonrelativistic Maxwellian | Leading-order line-integrated harmonic re-derived in the test. |
| G. D. Fleishman and A. A. Kuznetsov, ApJ 721, 1127 (2010) | Continuous-harmonic approximation of the gyrosynchrotron sum | Native exact-Bessel continuous integral; error measured against the harmonic sum. |
| V. A. Razin (1960); V. L. Ginzburg and S. I. Syrovatskii, ARA&A 3, 297 (1965) | Razin suppression `x = (ω/ω_c)(1 + γ²ω_p²/ω²)^{3/2}` | Closed-form single-particle reference evaluated with SciPy in the test. |
| M. Pandya, Z. Zhang, M. Chandra, C. F. Gammie, ApJ 822, 34 (2016) | Relativistic kappa distribution | `KappaDistribution`. |
| R. Mahadevan, R. Narayan, I. Yi, ApJ 465, 327 (1996), eq. 31 | Angle-averaged thermal synchrotron fit | `ThermalSynchrotronModel`, moved unchanged from the GR imaging owner; constants from the SI scale. |
| G. B. Rybicki and A. P. Lightman, *Radiative Processes in Astrophysics*, Wiley (1979), eq. 5.14; W. J. Karzas and R. Latter, ApJS 6, 167 (1961) | Thermal free–free emissivity; Born thermal Gaunt factor | `ThermalFreeFreeModel`; the Planck mean of the Born Gaunt factor is exactly `2√3/π`. |

UFGC (Kuznetsov & Fleishman 2021) and Symphony (Pandya et al. 2016) are GPL
programs and remain literature or pinned external oracles only; no provider
adapter ships with this capability.

## Plasma rays and mode-resolved transfer

| Source | Role | Native boundary |
|---|---|---|
| T. H. Stix, *Waves in Plasmas*, AIP (1992), ch. 1–4 | Cold-plasma biquadratic `A n⁴ − B n² + C`, cutoffs and resonances | Written in Cartesian refractive-index components by the cold-plasma owner; the ray tests use independent Appleton–Hartree formulas. |
| K. G. Budden, *The Propagation of Radio Waves*, Cambridge (1985), ch. 14; Yu. A. Kravtsov and Yu. I. Orlov, *Geometrical Optics of Inhomogeneous Media*, Springer (1990) | Hamiltonian ray equations, reflection at cutoffs | Linear-ramp turning points and parabolic paths derived in the test. |
| E. Hairer, C. Lubich, G. Wanner, *Geometric Numerical Integration*, 2nd ed., Springer (2006), §VI.3 | Symplecticity of the implicit midpoint rule | `DispersionRayPlan` solves the midpoint with `VectorLocalRootPlan`; the tangent map is the Cayley transform of `h J ∇²H`. |
| G. Bekefi, *Radiation Processes in Plasmas*, Wiley (1966), ch. 1 | Ray refractive index `n_r`, invariance of `I/n_r²` | `ColdPlasmaRayPath.ray_index_squared`; test reference by finite differences of the Appleton–Hartree branch. |
| M. H. Cohen, ApJ 131, 664 (1960); V. V. Zheleznyakov, *Radiation in Astrophysical Plasmas*, Kluwer (1996), ch. 7 | Weak and strong mode coupling, quasi-transverse regions | Coupling parameter `|dŝ/ds|/(k|Δn|)`; the twisted-field Stokes equation is solved exactly in the co-rotating frame in the test. |
| D. B. Melrose and R. C. McPhedran, *Electromagnetic Processes in Dispersive Media*, Cambridge (1991), ch. 12 | Polarized transfer with Faraday rotation and conversion in a magnetoionic medium | `PlasmaRayTransferPlan` composes `PolarizedRadiativeTransferPlan`; the rotation measure is checked against quadrature of `(ω/2c)(n_L − n_R)`. |

## One-way Maxwell antennas

| Source | Role | Native boundary |
|---|---|---|
| A. E. H. Love, Phil. Trans. R. Soc. A 197, 1 (1901); S. A. Schelkunoff, Bell Syst. Tech. J. 15, 92 (1936) | Surface equivalence: `K = n̂ × H`, `K_m = −n̂ × E` radiate the prescribed field on one side and nothing on the other | `SampledPlaneCurrentAntennaPlan` sheets; one-way ratio measured against the launched plane wave. |
| A. Taflove and S. C. Hagness, *Computational Electrodynamics*, 3rd ed., Artech House (2005), ch. 5 | Total-field/scattered-field connecting condition on the Yee lattice | Electric sheet on node planes, magnetic sheet half a cell upstream, each driven by the other sheet's incident field; re-derived on the cochain lattice. |
| A. E. Siegman, *Lasers*, University Science Books (1986), ch. 17 | Paraxial Gaussian beam, Rayleigh range, Gouy phase | `sample_focused_gaussian_pulse_envelope` and the waist/Gouy test reference (with the Yee lattice diffraction wavenumber). |
| J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999), §11.9–11.10 | Four-current and field transformation, relativistic Doppler shift | Moving sheets deposit `K'(τ)/γ` at rest-frame retarded times from `phydrax.boost_event`; Doppler reference `F_lab(ω) = F'(ω/D)`. |

## Radiation reaction

| Source | Role | Native boundary |
|---|---|---|
| L. D. Landau and E. M. Lifshitz, *The Classical Theory of Fields*, 4th ed. (1975), §76 | Landau–Lifshitz radiation-reaction force including the convective field-derivative term; planar cooling `γ(t) = coth(τω_B²t + arccoth γ₀)` | `RadiationReactionPlan` models `"landau-lifshitz"`/`"landau-lifshitz-reduced"` re-derived in SI with the scale's `ε₀`; the cooling law is the test reference. |
| M. Tamburini et al., New J. Phys. 12, 123005 (2010) | Reduced Landau–Lifshitz force without field derivatives for PIC | `"landau-lifshitz-reduced"`; first-order operator split after the push. |
| V. N. Baier and V. M. Katkov, Sov. Phys. JETP 26, 854 (1968); A. A. Sokolov and I. M. Ternov, *Radiation from Relativistic Electrons*, AIP (1986) | Quantum synchrotron photon spectrum in the locally constant field | `RadiationReactionTables` integrate it with `phydrax.special.synchrotron_f`/`synchrotron_g` by native Gauss–Legendre quadrature. |
| F. Niel, C. Riconda, F. Amiranoff, R. Duclous, M. Grech, Phys. Rev. E 97, 043209 (2018) | Quantum correction `g(χ)`, diffusion `h(χ)`, Fokker–Planck drift/diffusion and moment equations | `"quantum-corrected-landau-lifshitz"` and `"stochastic-fokker-planck"`; tests evaluate Niel's integrals independently with SciPy Bessel functions. |
| M. Sands, SLAC-121 (1970) | Classical second photon-energy moment `55/(24√3)` | Normalization of `h(0) = 1`. |

## Comparison programs

Programs that compute trajectory radiation (for example SRW, SPECTRA, or
lwrad) are external oracles only. No provider adapter ships with this
capability; an adapter belongs with the consuming milestone and runs through
the pinned external runtime.

## Data rule

Constants come from the declared `ElectromagneticScaleContract` (CODATA 2022 for
`ElectromagneticScaleContract.si()`); no module carries a private `ε₀`, `e`, or
`c`. No tabulated data enter the trajectory-radiation route.
