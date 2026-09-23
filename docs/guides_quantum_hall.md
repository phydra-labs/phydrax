# Quantum Hall workflows

Phydrax separates four approximation stacks that must not be interchanged without an explicit scientific argument:

1. periodic independent-particle QH, QAH, and QSH models;
2. Landau-level-projected fractional Hall Hamiltonians;
3. unprojected monopole-sphere variational Monte Carlo;
4. finite coherent device transport.

All workflows retain model identity, boundary conditions, resource admission, convergence, and failure evidence.

## Periodic Hall models

`HaldaneModelPlan`, `KaneMeleModelPlan`, and `HofstadterModelPlan` lower to the canonical periodic orbital pencil. Chern numbers use normalized overlap links and oriented reciprocal plaquettes. Kane–Mele Z2 evaluation verifies the antiunitary time-reversal action before tracking the Wilson phases of one occupied rank-two Kramers pair over half the Brillouin zone.

A ribbon calculation is not a topology calculation. `HallRibbonPlan` reports energies and left/right edge weights, while bulk Chern or Z2 evidence remains separately identified.

## Projected Haldane sphere

`HaldaneSpherePlan` uses doubled monopole flux `2Q`, exact particle statistics, and an optional exact filling/shift relation

`2Q = N / nu - shift + flux_offset`.

The shift is supplied explicitly. It is never inferred from the denominator of the filling fraction.

Public pseudopotentials are indexed by relative angular momentum. `MonopoleLandauLevel` keeps physical monopole strength `2Q` separate from orbital spin `2l = 2Q + 2n`. `HaldanePseudopotentialPlan` converts relative channels to the doubled total-pair-spin convention used by the SU(2) substrate. `coulomb_haldane_pseudopotentials` evaluates the exact finite-sphere angular-momentum expression for any declared Landau level.

Preparation selects particle number and twice-Lz directly. The Hamiltonian is exposed as a matrix-free native linear operator. `HaldaneSphereSpectrumPlan` uses native restarted Lanczos and retains residuals and solver status.

Neutral and charge gaps are distinct result types. Finite-size extrapolation retains every raw point and model variation.

## Landau-level mixing VMC

`LandauLevelMixingVMCPlan` composes:

- a permutation-equivariant monopole-attention determinant;
- exact monopole-harmonic phase envelopes;
- a symmetric single-electron sphere proposal;
- the gauge-covariant spherical kinetic operator;
- the existing persistent-chain stochastic-reconfiguration solver.

The initial support is fully spin polarized. The local kinetic operator computes only selected Hessian diagonal actions. Particle coincidences and unresolved gauge-patch poles produce explicit local-operator status rather than clipped energies.

## Finite cylinder

`HallCylinderPlan` builds center-of-mass-conserving pseudopotential projectors on an orbital chain. Interaction-range and coefficient truncation are explicit, and the omitted coefficient norm is retained.

The canonical quantum-lattice lowering constructs an exact Abelian MPO carrying particle number and orbital momentum. `HallCylinderDMRGPlan` uses the existing Abelian residual-sweep solver and reports residuals, discarded weights, and charge drift. This is finite-cylinder DMRG; no infinite-cylinder or thermodynamic-limit claim is made.
## Infinite cylinder

`InfiniteHallCylinderPlan` binds a filling-compatible uniform Abelian MPS/MPO unit cell to native VUMPS. A filling `p/q` requires a `q`-orbital unit cell carrying `p` particles. Matrix-free transfer fixed points, injectivity, correlation length, and Galerkin residual remain visible. Infinite length does not remove finite-circumference, interaction-range, or bond-dimension error.


## Disorder and transport

`BottIndexPlan` evaluates a bounded real-space invariant for an orthonormal finite realization. It requires a resolved occupied spectral gap and reports projector, unitary, and integer-quantization residuals.

Periodic principal-layer leads are prepared by matrix decimation with native linear solves. Multi-terminal transport retains lead causality, broadening positivity, pointwise open-system status, integrated currents, and continuity residuals. Numerical broadening is never treated as physical scattering.

`HallBarPlan` maps explicitly selected contacts to Hall and longitudinal voltage differences. It does not derive resistance plateaus from clean spectral gaps.
Voltage probes enforce zero integrated probe current. Dephasing probes enforce zero current at every energy. Both remain phenomenological. `KeldyshSCBAProblem` supplies a matrix optical-phonon Fock self-energy with fixed-point, Keldysh, and collision-balance evidence. `OpenHallTransportPlan` is explicitly finite and Markovian. `LocalizedHallNetworkPlan` is a calibrated mesoscopic rate network, not an ab initio plateau predictor.

`PeriodicSheetHallPlan` reports two-dimensional sheet conductance in siemens and keeps the effective Chern weight visible. It is distinct from volumetric finite-frequency optical Kubo response.


## Finite width

`SubbandCoulombFormFactorPlan` consumes a normalized vertical probability density in SI units and evaluates the quasi-two-dimensional Coulomb form factor. The zero-momentum limit, normalization, bounds, and quadrature identity are retained. Confinement remains owned by the semiconductor application; the Hall layer only consumes its density.
