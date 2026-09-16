# Magnetism

This candidate ladder keeps three magnetic physical state types non-substitutable:

- `ClassicalSpinState` stores dimensionless unit vectors on the product sphere S².
- `QuantumSpinModel` stores exact finite spin-I operators through the canonical quantum-lattice compiler.
- `LinearSpinWaveResult` stores bosonic Holstein–Primakoff modes normalized in a Krein metric.

A successful smoke, test, or benchmark does not release any of these profiles.

## Classical reduced Hamiltonian

Every physical bond occurs exactly once, oriented by increasing stable particle ID. The implemented energy is

```text
E = -sum_b m_i^T (J_b I + Gamma_b) m_j
    -sum_b D_b dot (m_i cross m_j)
    -sum_i K_i (m_i dot n_i)^2
    -sum_i mu_i B_i dot m_i.
```

`J > 0` is ferromagnetic. `Gamma` must be symmetric and traceless. Reversing a bond transposes the exchange tensor and changes `D` to `-D`; callers must not supply both orientations. `K > 0` is easy-axis. The field is magnetic flux density. In the SI profile energy is joule, field tesla, moment joule/tesla, time second, positive gyromagnetic ratio rad/(second tesla), and Gilbert damping dimensionless. Reduced atomistic unit systems use the corresponding declared reduced quantities.

The effective field is exactly `H_eff,i = -(1/mu_i) dE/dm_i`. The evaluation returns the five energy channels, their reconstruction residual, magnetization, effective field, and tangent torque. Graph overflow, duplicate or missing reverse edges, ambiguous endpoint images, nonunit spins/axes, nonpositive moments, invalid Gamma, and nonfinite coefficients fail before a result is accepted.

## Geometric LLG

Deterministic dynamics uses the existing `RKMK` solver on `GeodesicManifoldStateGeometry(SphereManifold(3))`:

```text
dm/dt = -gamma/(1 + alpha^2)
        [m cross H_eff + alpha m cross (m cross H_eff)].
```

Thermal dynamics is a separate Stratonovich profile using `SRKMK` and one caller-supplied `WienerRealization`. Its field-noise amplitude satisfies

```text
sigma_i^2 = 2 alpha_i k_B T_i / (gamma_i mu_i).
```

No Euclidean renormalization, Itô reinterpretation, or per-step ad-hoc PRNG stream is used. A checkpoint binds the Hamiltonian, integrator, unit system, step, and Wiener realization identity. Pathwise differentiation is meaningful only for a fixed realization; burn-in, equilibrium, effective sample size, and ensemble uncertainty remain caller-owned statistical evidence.

## Exact quantum spins

Quantum spins use ascending doubled-projection basis labels and dimensionless operators in hbar units. `twice_spin=1` is spin one-half with `S=sigma/2`, not `sigma`. The Heisenberg sign is the same as the classical sign:

```text
H = -sum_b J_b S_i dot S_j.
```

The XXZ, transverse-field Ising, arbitrary oriented DMI-vector, and spin-one axial-anisotropy factories emit only `LocalOperatorPlan`, `QuantumLatticeTerm`, and `QuantumLatticeSpecification`. They do not introduce a second term language or backend dispatcher. Longitudinal DMI preserves fixed total Sz; transverse DMI is rejected by a fixed-Sz lowering after the compiler proves its nonzero charge changes. A fixed total-Sz sector is constructed only through `FixedSpinProjectionBasis`.

## Collinear linear spin waves

The released target is a caller-supplied torque-stationary collinear reference. `spin_exchange_tensor` combines once-oriented exchange, symmetric Gamma, and DMI as `J I + Gamma - [D]_x` in the bond Hamiltonian `-S_i^T tensor S_j`. `lower_collinear_spin_wave_bonds` performs the quadratic Holstein–Primakoff lowering in caller-supplied right-handed transverse frames and emits canonical periodic normal/pairing families plus the raw reference torque. `prepare_linear_spin_wave` evaluates exactly those families on one `ReciprocalMeshPlan`; it never performs another Fourier sum.

For `eta = diag(I,-I)`, LSWT solves the bosonic dynamical problem `eta H_B v = omega v`. Positive branches must have positive Krein norm and are normalized to `v^dagger eta v = 1`. Evidence includes reference torque, quadratic Hermiticity, positive/negative frequency pairing, full paraunitarity, energetic stability, and an exact caller-declared Goldstone count at every q point. Complex frequencies, negative energetic directions, missing positive-Krein branches, or undeclared zero modes fail closed. These bosonic modes cannot be passed to fermionic BdG APIs.

## Supplied orbital spin-orbit coupling

`SpinorBasisConvention` fixes orbital-major `(orbital up, orbital down)` order with `S=sigma/2`. For supplied dimensionless Hermitian orbital matrices, the native assembly is

```text
H_SOC = lambda (L_x tensor S_x + L_y tensor S_y + L_z tensor S_z).
```

When requested, the supplied `L` matrices must satisfy the angular-momentum commutators. The onsite matrix is lowered to one canonical `PeriodicTranslationFamily`; no private reciprocal evaluator is introduced. This is reduced-model assembly only: it does not derive lambda, relativistic pseudopotentials, four-component states, exchange, or DMI.

Spin-resolved observables accept explicit eigenvectors and, for a generalized basis, the explicit overlap metric. Isolated bands receive scalar spin expectations. Degenerate clusters retain projected spin matrices, because individual eigenvector labels inside a degenerate subspace are gauge-dependent.


## Caller-supplied magnetic symmetry

`applications.magnetism` adds physical semantics to an existing `FiniteMetricIsometryGroup`; it contains no magnetic group database or operation discovery. An operation `(R, theta)` acts on axial spin by `(-1)^theta det(R) R` and on momentum by `(-1)^theta R`. Antiunitary bits must form a Z2 homomorphism. Supplied site/route permutations and coefficient representations must compose according to the existing group table.

The compiler converts unitary and antiunitary invariance into real-linear equations, computes a rank-certified `LinearSubspace`, and refuses a rank unresolved at the cutoff. External magnetic field is axial and time-odd; DMI is axial and time-even.

## Candidate frontier contracts and nonclaims

The implemented types are usable reduced-model contracts, but remain candidate until exact SupportTuple dependencies, independent campaigns, resource envelopes, lifecycle restore, runtime attestations, documentation hashes, and signed release gates exist. Candidate and released SupportTuple contents must remain byte-identical; maturity changes only the capability profile.

Noncollinear LSWT, dipolar Ewald/demagnetization, spin-lattice dynamics, LLB, spin-transfer/spin-orbit torque, and magnon topology/thermal Hall remain frontier work. They must extend these conventions with complete physics and evidence rather than enter as placeholder methods, fallback flags, or broadened claims.
