# Crystalline elasticity, phonons, and lattice heat transport

The lattice-materials path is a composition of atomistic, periodic-operator, chemistry, and linear-algebra owners. It does not introduce a second Fourier engine, atomistic potential implementation, eigensolver, or release registry.

## Conventions and units

`SecondOrderForceConstants` stores one `phydrax.sparse.EdgeRelation`, one integer lattice translation per route, and one 3 × 3 Cartesian block per route. An edge has `target=i`, `source=j` and represents Φ(iα,jβ,R). Every valid edge has an explicit reverse edge `(j,i,−R)` whose block is the transpose-conjugate. Fractional reciprocal coordinates use the canonical periodic convention `exp(+i 2π q·R)` through `phydrax.operators.periodic`.

IFC2 values carry an energy/length² `UnitDefinition`; IFC3 values carry energy/length³. Masses use the bound `AtomisticUnitSystem`. Harmonic output is signed angular frequency in inverse-time units: a negative dynamical-matrix eigenvalue is returned as a negative square root, never hidden by an absolute value. Group velocities are length/time. Thermodynamic energies are per primitive cell. Conductivity uses the primitive-cell volume and the corresponding atomistic unit system.

`PrimitiveSupercellImageMap` is mandatory for native finite displacement. It binds stable primitive and supercell particle IDs, each supercell atom to a primitive atom and integer image, one zero-image representative for every primitive atom, primitive fractional positions, and the primitive cell. IFC routes are extracted from the supercell Hessian only through this exact map.

## Raw and corrected force constants

`FiniteDisplacementIFC2Plan` is restricted to prepared native EAM, Stillinger–Weber, and Tersoff potential programs. It evaluates central differences at h and h/2, rebuilds the prepared bounded neighborhood, retains equilibrium-force, plus/minus antisymmetry, and refinement residuals, and rejects force or graph failures. `IFCConstraintPolicy` constructs one host-side linear projection for reverse-pair symmetry, the acoustic sum rule, and Born–Huang rotational invariance. `IFCConstraintEvidence` retains residuals before and after projection, projector rank and condition, and relative correction. A large correction or failed raw force path cannot become successful merely because the projected tensor is finite.

Provider data enter through `LatticeDynamicsRequest`, `LatticeDynamicsProviderCapabilities`, and `LatticeDynamicsArtifactSet`. Requests bind the exact structure, cell, units, method, basis, pseudopotential, spin, relativity, input artifacts, and route capacities. Results retain provider/version/build, request/input hashes, rights, raw residuals, and optional first-principles provenance. Provider-generated arrays remain provider-backed after normalization.

## Harmonic phonons and LO–TO splitting

`HarmonicPhononPlan.prepare()` compiles IFC2 blocks into one `PeriodicTranslationFamilyPlan`. `PreparedHarmonicPhonons.evaluate()` mass-weights the canonical family, applies the atomistic kinetic conversion, and uses Phydrax eigensolvers. `PhononDispersionResult` retains fractional and Cartesian q points, signed angular frequencies, mass-weighted eigenvectors, dynamical matrices, imaginary/acoustic masks, eigen residuals, orthogonality, Hermiticity, and acoustic evidence.

Polar q→0 behavior is a separate `NonanalyticPhononCorrection`. It requires charge-neutral Born effective charges, a symmetric positive-definite relative dielectric tensor, the same rank-3 cell and unit identity, and one explicit Cartesian direction for each Γ occurrence. The 3D `4π kₑ/Ω` expression is refused for rank-1 or rank-2 periodic systems and is never applied at finite q. Low-dimensional polar electrostatics therefore has no implicit fallback.

## Thermodynamics and QHA

`HarmonicThermodynamicsPlan` accepts only a Γ-excluding integration mesh with positive normalized weights and strictly positive stable modes. It evaluates stable Bose functions and returns zero-point energy, free energy, internal energy, entropy, and heat capacity per primitive cell over a positive temperature grid. Dispersion-path samples, acoustic zeros, and imaginary modes are rejected.

`QuasiHarmonicPlan` binds at least five ordered volumes, fixed q/branch topology, stable modes, static energies, and one temperature grid. At every temperature the discrete minimum must be interior, the local three-point quadratic curvature must be positive, and its interpolated minimum must remain strictly inside the bracket. Endpoint clipping and equation-of-state extrapolation are not performed. The result retains the raw free-energy surface, bracket indices, curvature, interpolation residual, equilibrium volume, and volumetric expansion.

## Intrinsic three-phonon RTA

Production RTA begins with provider-normalized `ThirdOrderForceConstants`. `IFC3ModeVertexPlan` binds those IFC3 values to one complete regular q mesh, positive harmonic frequencies, mass-weighted eigenvectors, masses, units, and a phonon result ID. It derives separate decay and coalescence vertices using the explicit momentum conventions `(-q₁,q₂,q₁−q₂)` and `(q₁,q₂,−q₁−q₂)` and the `sqrt(ħ/(2 M ω))` displacement normalization. Arbitrary caller-supplied mode vertices are not accepted by `ThreePhononRTAPlan`.

The RTA evaluator includes decay and coalescence Bose factors, normalized Gaussian energy deltas, q weights, modulo-reciprocal-lattice momentum routing, and detailed-balance evidence. It reports channel-resolved rates, lifetimes, mean free paths, mode conductivity, tensor symmetry/eigenvalues, and ballistic masks. A heat-carrying mode with zero rate produces an unsuccessful unbounded/ballistic result with no rate floor and no fabricated finite conductivity.

## Resource policy and frontier boundary

IFC2, IFC3, q-point, eigensolve, channel, and byte limits are checked before large allocations. These constants are conservative code admission policy, not release envelopes. `benchmarks/cm_lattice_dynamics.py` records preparation/evaluation phases, raw synchronized samples, logical artifact bytes, analytic residuals, and unavailable measured peaks explicitly. `tools/lattice_dynamics_qualification.py` is an analytic smoke qualification, not signed release evidence.

`lattice_frontier_contracts()` describes bounded candidate-only native IFC3, iterative BTE, atomistic Green–Kubo, electron-phonon, spin-phonon, defect, and interface workflows. These contracts require method-specific conservation, provenance, convergence, and resource evidence. They implement no release promotion and make no material-phase, experimental-conductivity, hydrodynamic, sign-problem, or universal-accuracy claim.
