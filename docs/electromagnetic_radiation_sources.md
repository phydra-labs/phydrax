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
| J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999), §11.9–11.10 | Field transformation, moving-boundary jump conditions, relativistic Doppler shift | Moving sheets are smoothed moving TFSF boundaries driven by the Lorentz-transformed incident fields at rest-frame retarded times from `phydrax.boost_event`, plus the convective currents `−D ∂Θ/∂t`, `−B ∂Θ/∂t`; Doppler reference `F_lab(ω) = F'(ω/D)`. |
| I. Haber et al., Proc. Sixth Conf. Numerical Simulation of Plasmas, 46 (1973); J.-L. Vay, I. Haber, B. B. Godfrey, J. Comput. Phys. 243, 260 (2013) | Exact PSATD interval propagator with prescribed currents | Spectral sheets: the same equivalent currents (with the moving-boundary convective terms) on a band-limited normal delta, integrated exactly with an added magnetic current; re-derived for the PSATD grid, one-way and Doppler measured. |
| O. Shapoval, J.-L. Vay, H. Vincenti, Comput. Phys. Commun. 235, 102 (2019) | Two-step split-field PSATD PML | Its non-divergence-preserving damping is booked as layer-confined absorber charge (a Phydrax derivation, not from the source). |

## Frequency-domain moving charges

| Source | Role | Native boundary |
|---|---|---|
| I. E. Tamm and I. M. Frank, Dokl. Akad. Nauk SSSR 14, 107 (1937); J. D. Jackson, *Classical Electrodynamics*, 3rd ed., Wiley (1999), §13.4–13.5 | Uniform-motion field in a dispersive medium (`K₀`, `K₁` of `sρ`), Frank–Tamm spectrum | `UniformMotionFieldPlan`; the Frank–Tamm law is the independent test reference for the analytic and the cochain routes. |
| V. L. Ginzburg and I. M. Frank, J. Phys. USSR 9, 353 (1945) | Transition radiation of a charge entering a perfect conductor | Scattered-field cochain test reference, `q²β²sin²θ/(4π³ε₀c(1 − β²cos²θ)²)`. |
| S. J. Smith and E. M. Purcell, Phys. Rev. 92, 1069 (1953) | Smith–Purcell relation `λ = (d/|n|)(1/β − cosθ)` | Fourier-modal diffraction-order directions. |
| A. Szczepkowicz, L. Schächter, R. J. England, Appl. Opt. 59 (2020), arXiv:2009.03811, Table 2 | Frequency-domain SI Smith–Purcell energies for lamellar gratings (FEM, ±20%) | Fused-silica test oracle for `MovingPointChargeQuadrature`. The total is matched within 2% at converged harmonics. The published up/down split lies outside its stated ±20% of the converged Fourier-modal split. That Fourier-modal split is checked against the cochain solver for the line charge. |
| P. Lalanne and J.-P. Hugonin, J. Opt. Soc. Am. A 17, 1033 (2000), Tables 1–2 (inverse-rule RCWA column); L. Li, J. Opt. Soc. Am. A 13, 1870 (1996) | Metallic lamellar-grating efficiencies versus harmonic count; Fourier factorization rules | Test reference for `VectorFourierFactorizationPlan` on metals and for power-wave cascade stability. |
| P. M. van den Berg, J. Opt. Soc. Am. 63, 689 and 1588 (1973); J. H. Brownell, J. Walsh, G. Doucas, Phys. Rev. E 57, 1075 (1998) | Rigorous and surface-current Smith–Purcell intensities for perfectly conducting gratings | Literature only: the tables were not obtainable with their normalization. |

## Radiation reaction

| Source | Role | Native boundary |
|---|---|---|
| L. D. Landau and E. M. Lifshitz, *The Classical Theory of Fields*, 4th ed. (1975), §76 | Landau–Lifshitz radiation-reaction force including the convective field-derivative term; planar cooling `γ(t) = coth(τω_B²t + arccoth γ₀)` | `RadiationReactionPlan` models `"landau-lifshitz"`/`"landau-lifshitz-reduced"` re-derived in SI with the scale's `ε₀`; the cooling law is the test reference. |
| M. Tamburini et al., New J. Phys. 12, 123005 (2010) | Reduced Landau–Lifshitz force without field derivatives for PIC | `"landau-lifshitz-reduced"`; first-order operator split after the push. |
| V. N. Baier and V. M. Katkov, Sov. Phys. JETP 26, 854 (1968); A. A. Sokolov and I. M. Ternov, *Radiation from Relativistic Electrons*, AIP (1986) | Quantum synchrotron photon spectrum in the locally constant field | `RadiationReactionTables` integrate it with `phydrax.special.synchrotron_f`/`synchrotron_g` by native Gauss–Legendre quadrature. |
| F. Niel, C. Riconda, F. Amiranoff, R. Duclous, M. Grech, Phys. Rev. E 97, 043209 (2018) | Quantum correction `g(χ)`, diffusion `h(χ)`, Fokker–Planck drift/diffusion and moment equations | `"quantum-corrected-landau-lifshitz"` and `"stochastic-fokker-planck"`; tests evaluate Niel's integrals independently with SciPy Bessel functions. |
| M. Sands, SLAC-121 (1970) | Classical second photon-energy moment `55/(24√3)` | Normalization of `h(0) = 1`. |

## Strong-field QED

| Source | Role | Native boundary |
|---|---|---|
| V. I. Ritus, J. Sov. Laser Res. 6, 497 (1985); V. N. Baier and V. M. Katkov, Sov. Phys. JETP 26, 854 (1968) | Locally constant field nonlinear Compton and Breit–Wheeler spectra | `QEDTable` rewrites both brackets with `F`, `G` from `phydrax.special` and integrates them by native Gauss–Legendre quadrature; tests evaluate them independently with SciPy Bessel functions. |
| T. Erber, Rev. Mod. Phys. 38, 626 (1966) | Pair-creation function `T(χ)` and its asymptotes (Erber's `χ` is half the modern `χ_γ`) | Below-table continuation `T → (3/16)√(3/2) e^{−8/(3χ)}`; test references at both ends. |
| A. Di Piazza, M. Tamburini, S. Meuren, C. H. Keitel, Phys. Rev. A 99, 022125 (2019), Eqs. (10)–(12) | Improved LCFA: field-variation time along the trajectory, formation time, infrared threshold with tuning `ζ = 0.7` | `NonlinearComptonPlan(model="improved-lcfa")`; the formation-time ratio is the per-event validity evidence. The Breit–Wheeler formation time is its crossing-symmetric continuation. |
| C. P. Ridgers et al., J. Comput. Phys. 260, 273 (2014); M. Lobet et al., J. Phys. Conf. Ser. 688, 012058 (2016) | Optical-depth Monte Carlo for QED-PIC | Event sampling with identity-addressed randomness and bounded adaptive subcycling. |
| T. Grismayer et al., Phys. Plasmas 23, 056706 (2016); Phys. Rev. E 95, 023210 (2017) | Cascade growth rate in a rotating electric field, Eqs. (9)–(10) with recoil-free cycle averages | Qualification gate `rotating-field-cascade-growth-vs-grismayer` (25 % tolerance). |
| M. Vranic et al., Comput. Phys. Commun. 191, 65 (2015) | Momentum-conserving macroparticle merging for cascades | `ParticleMergePlan(minimum_occupancy=...)` consumes the cascade's occupancy trigger. |
| D. Seipt, B. King, Phys. Rev. A 102, 052805 (2020), Eqs. (38), (44), (59) | Fully spin- and photon-polarization-resolved LCFA rates of nonlinear Compton and Breit–Wheeler; small-`χ` channel asymptotics | `NonlinearComptonPlan`/`NonlinearBreitWheelerPlan` `channel_spectrum` and the `"positive"`/`"negative"` `QEDTable` components, rewritten with `F`, `G`, `H = xK_{1/3}`; tests evaluate the Airy-function form with SciPy. |
| Y.-F. Li et al., Phys. Rev. Lett. 122, 154801 (2019); Phys. Rev. Lett. 124, 014801 (2020) | Spin quantization axis `v̂ × â`, spin collapse at emission, T-BMT between events | `"spin-and-photon-polarized"` Monte Carlo, extended by the exact no-emission (mass-operator) spin evolution. |
| A. A. Sokolov, I. M. Ternov, *Radiation from Relativistic Electrons*, AIP (1986) | Radiative polarization: equilibrium `8/(5√3)`, flip rate `(5√3/8) α χ³ mc²/(ħγ)` | Qualification gate `sokolov-ternov-flip-rate-and-equilibrium`. |
| V. Bargmann, L. Michel, V. L. Telegdi, Phys. Rev. Lett. 2, 435 (1959) | Spin precession with anomalous moment | `RelativisticPushPlan.precess`, integrated with the pusher's own Cayley rotation. |
| V. N. Baier, V. M. Katkov, V. M. Strakhovenko, *Electromagnetic Processes at High Energies in Oriented Single Crystals*, World Scientific (1998) | Pair creation by polarized photons (`W_∥/W_⊥ → 1/2` as `χ_γ → 0`) | Gate `polarized-pair-rates-and-pair-spins`. |

## Coherent synchrotron radiation

| Source | Role | Native boundary |
|---|---|---|
| E. L. Saldin, E. A. Schneidmiller, M. V. Yurkov, Nucl. Instrum. Methods A 398, 373 (1997) | Steady-state 1-D CSR wake `∝ (z − z′)^{−1/3}` and entrance transients | `CSRPlan(model="1d-steady")`; the transient route is checked against the entrance transient. |
| C. Mayes and G. Hoffstaetter, Phys. Rev. ST Accel. Beams 12, 024401 (2009) | Exact 1-D retarded line-charge model over bends and drifts, bunch compression, parallel-plate images | `CSRPlan(model="1d-transient-shielded")`. |
| J. B. Murphy, S. Krinsky, R. L. Gluckstern, Part. Accel. 57, 9 (1997) | Parallel-plate shielding of the steady CSR impedance | Shielding test reference. |
| Y. Cai and Y. Ding, Phys. Rev. Accel. Beams 23, 014402 (2020) | Steady 3-D longitudinal potential with its Coulomb term | Used for the longitudinal potential only. Their published transverse potentials set `β_s ≈ β` in the magnetic term, which moves the test particle rigidly at `β(1 + x/ρ)c` and yields a residual centripetal coefficient `2 ≤ Λ ≤ 4`; Phydrax uses the exact Lorentz force instead. |
| Y. S. Derbenev and V. D. Shiltsev, SLAC-PUB-7181 (1996); G. Stupakov, Phys. Rev. Accel. Beams 25, 014401 (2022), Eq. (4) and its discussion of Ref. [22] | Residual centripetal transverse force `−2λκ` on a steady circular orbit (factor 2 for every aspect ratio) | Test oracle for `3d-steady-igf` and `3d-retarded-mesh`; both routes give `Λ ≈ 2`, cross-checked by a 40-digit Liénard–Wiechert field evaluation. |

## Comparison programs

External programs are oracles only. Each adapter lives with the route it
checks, translates a Phydrax plan into the provider's input deck (no second
physics implementation), runs a caller-pinned executable or interpreter
through `run_pinned_command` with byte-capped file artifacts, converts the
output through the plan's `ElectromagneticScaleContract`, and reports an
`AdapterReport` whose losses enumerate what the provider cannot represent.
Inputs outside the supported subset are refused before the provider runs.
GPL programs are pinned external executables only; no source is copied or
linked. Live tests skip only when the environment variables are absent. The
validated version is the release the live comparison ran against.

| Program | Route checked | Adapter | Validated version | License | Environment variables |
|---|---|---|---|---|---|
| SRW (Chubar & Elleaume, EPAC 1998) | A1 single-electron spectra on trajectories and X1a field maps | `run_srw`, `SRWProvider` (`phydrax.electromagnetics`) | srwpy 4.2.1 | EPICS | `PHYDRAX_SRW_PYTHON`, `PHYDRAX_SRW_PYTHON_VERSION` |
| UFGC (Kuznetsov & Fleishman, ApJ 922, 103, 2021) | C2 gyrosynchrotron coefficients, harmonic-sum and continuous routes | `run_ufgc`, `UFGCProvider` (`phydrax.electromagnetics`) | kuznetsov-radio/gyrosynchrotron 5e014ba | GPL-3.0-only | `PHYDRAX_UFGC_PYTHON`, `PHYDRAX_UFGC_PYTHON_VERSION`, `PHYDRAX_UFGC_LIBRARY`, `PHYDRAX_UFGC_VERSION` |
| Symphony (Pandya et al., ApJ 822, 34, 2016) | C2 vacuum synchrotron coefficients | `run_symphony`, `SymphonyProvider` (`phydrax.electromagnetics`) | AFD-Illinois/symphony a869c6b | GPL-3.0-only | `PHYDRAX_SYMPHONY_PYTHON`, `PHYDRAX_SYMPHONY_PYTHON_VERSION`, `PHYDRAX_SYMPHONY_MODULE`, `PHYDRAX_SYMPHONY_VERSION` |
| WarpX (Fedeli et al., SC22) | P periodic Yee and PSATD plasma (plasma oscillation, drifting-plasma NCI); A1 on a WarpX single-particle track; P3 laser wakefield on WarpX RZ | `run_warpx`, `run_warpx_track`, `run_warpx_wakefield`, `WarpXProvider` (`phydrax.solver`) | 26.01 | BSD-3-Clause-LBNL | `PHYDRAX_WARPX`, `PHYDRAX_WARPX_RZ`, `PHYDRAX_WARPX_VERSION` |
| Smilei (Derouillat et al., CPC 222, 351, 2018) | P periodic Yee plasma | `run_smilei`, `SmileiProvider` (`phydrax.solver`) | 5.1 | CECILL-B | `PHYDRAX_SMILEI`, `PHYDRAX_SMILEI_VERSION` |
| PIConGPU (Burau et al., IEEE TPS 38, 2831, 2010) | P periodic Yee plasma; compiled per setup by a pinned build driver | `run_picongpu`, `PIConGPUProvider` (`phydrax.solver`) | 0.8.0 | GPL-3.0-or-later | `PHYDRAX_PICONGPU`, `PHYDRAX_PICONGPU_VERSION` |
| FBPIC (Lehe et al., CPC 203, 66, 2016) | P3 quasi-cylindrical laser wakefield | `fbpic_laser_wakefield`, `FBPICProvider` (`phydrax.solver.maxwell.spectral`) | 0.27.0 | BSD-3-Clause-LBNL | `PHYDRAX_FBPIC_PYTHON`, `PHYDRAX_FBPIC_PYTHON_VERSION` |
| Genesis 1.3 version 4 (Reiche, NIM A 429, 243, 1999) | X5 averaged time-dependent FEL | `run_genesis4` (`phydrax.applications.accelerator.fel`) | 4.6.15 | GPL-3.0-only | `PHYDRAX_GENESIS4`, `PHYDRAX_GENESIS4_VERSION` |
| Puffin (Campbell & McNeil, Phys. Plasmas 19, 093119, 2012) | X5 time-dependent FEL against the unaveraged 1-D model | `run_puffin`, `PuffinProvider` (`phydrax.applications.accelerator.fel`) | 2.1.0a+157f473 | BSD-3-Clause | `PHYDRAX_PUFFIN`, `PHYDRAX_PUFFIN_VERSION` |
| elegant (Borland, APS LS-287, 2000) | X4 1-D steady and transient CSR tracking | `run_elegant_csr`, `ElegantCSRProvider` (`phydrax.applications.accelerator`) | 2026.3.0 | EPICS | `PHYDRAX_ELEGANT`, `PHYDRAX_ELEGANT_VERSION` |
| Ocelot (Agapov et al., NIM A 768, 151, 2014) | X4 1-D transient CSR chicane tracking | `ocelot_csr_tracking`, `OcelotCSRProvider` (`phydrax.applications.accelerator`) | no live run recorded | GPL-3.0 | `PHYDRAX_OCELOT_PYTHON`, `PHYDRAX_OCELOT_PYTHON_VERSION` |
| PyCSR3D (Mayes, Lou et al.) | X4 steady 3-D CSR wake | `pycsr3d_longitudinal_wake`, `PyCSR3DProvider` (`phydrax.applications.accelerator`) | no live run recorded | Apache-2.0 | `PHYDRAX_PYCSR3D_PYTHON`, `PHYDRAX_PYCSR3D_PYTHON_VERSION` |
| Geant4 through geant4_pybind (Allison et al., NIM A 835, 186, 2016) | M1 electromagnetic shower profiles; M2 Cherenkov yield, cone, and polarization; M2 Fresnel/TIR/absorption/Rayleigh transport | `run_geant4_shower`, `run_geant4_cherenkov`, `run_geant4_optical`, `Geant4Provider` (`phydrax.applications.detector`) | geant4_pybind 0.1.3 (Geant4 11.4.p01) | LicenseRef-Geant4 | `PHYDRAX_GEANT4_PYTHON`, `PHYDRAX_GEANT4_PYTHON_VERSION`, `PHYDRAX_GEANT4_DATA` |
| openPMD-api `openpmd-pipe` | I1b ADIOS2 BP4 conversion | `OpenPMDADIOS2Provider` (`phydrax.interchange`) | 0.17.1 | LGPL-3.0-or-later | `PHYDRAX_OPENPMD_PIPE`, `PHYDRAX_OPENPMD_API_VERSION` |

Trajectory-radiation programs without an adapter (SPECTRA, lwrad) remain
literature references only.

## Data rule

Constants come from the declared `ElectromagneticScaleContract` (CODATA 2022 for
`ElectromagneticScaleContract.si()`); no module carries a private `ε₀`, `e`, or
`c`. No tabulated data enter the trajectory-radiation route.
