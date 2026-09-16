# Superconductivity

Phydrax keeps microscopic, mesoscopic, continuum, thin-film, and cable states non-substitutable. Every route below is an unreleased candidate until its exact support tuple has independent scientific, resource, lifecycle, runtime, rights, derivative, and documentation evidence.

## Fermionic BdG and finite-channel closure

`PreparedFermionicBdG` uses the canonical Nambu coordinate `(c_k, c†_-k)` and requires `Delta(k) = -Delta(-k)^T`. `SuperconductingMeanFieldPlan` separates fixed chemical potential from fixed filling, fixes one positive-real phase anchor, retains the Nambu one-half and double-counting terms, and rejects a nonconverged or gapless branch. `BdGChernPlan` remains limited to fully gapped two-dimensional class D with explicit basis connections and refinement evidence.

## Thin-film London/Pearl electrodynamics

`ThinFilmLondonPlan` accepts one planar `TriangleMesh`, a nonnegative Pearl length, and optional independent linear constraints for holes, fluxoids, or terminal currents. A scalar stream function generates the face sheet current. The prepared dense London KKT operator combines a DDG kinetic term and a softened nonlocal magnetostatic energy and is factorized once. Results retain stream function, face current, normal field, constraint reactions, separated magnetic/kinetic/applied energies, linear and constraint residuals, and current-divergence evidence. Curved films and arbitrary three-dimensional screening are not claimed.

## Gauge-covariant GL and TDGL

`ChargedScalarGaugePlan` owns local U(1) links. Its transform is

```text
psi_i -> exp(i q chi_i) psi_i
A_ij -> A_ij + chi_j - chi_i.
```

The covariant edge difference and plaquette curvature transform exactly. `GaugeCovariantGLPlan` combines condensation, covariant-gradient, and magnetic energies on one fixed simplicial mesh. The static solver and transactional TDGL step use real coordinates for the complex field, retain a fixed global phase anchor, and report gauge-energy and covariance residuals. The landed profile does not claim gauge-covariant AMR transfer or vortex-event differentiability.

## Equilibrium quasiclassics and retarded spectroscopy

`FermiSurfacePlan`, `MatsubaraQuadraturePlan`, and `RiccatiTrajectoryPlan` define spin-degenerate, spin-singlet equilibrium trajectories with specular caller-prepared routes. Stable Riccati amplitudes reconstruct normalized quasiclassical Green functions. `QuasiclassicalSuperconductivityPlan` solves a finite pairing-channel gap equation and returns current and free-energy evidence. `RetardedSpectroscopyPlan` is a separate real-energy profile consuming a converged equilibrium branch; it is not implicit analytic continuation of Matsubara samples. Spin-active boundaries, nonequilibrium Keldysh transport, and general impurity self-energies are not claimed.

## Engineering cable, current sharing, and quench

`SuperconductingMaterialLawPlan` imports a governed critical-current surface over temperature, field, and angle and refuses extrapolation. `SuperconductingCablePlan` evolves one series current over a fixed one-dimensional conductor, splits it between superconductor and stabilizer by equal-electric-field current sharing, routes Joule heat into solid cells, couples axial conduction and coolant heat capacity/advection, and activates a dump resistance after the declared temperature trigger. Electrical, thermal, current, and rollback evidence are returned for every step. This is not strand-resolved three-dimensional FEM.

## Cross-fidelity evidence

The bridge plans compare common observables only after the caller supplies the missing physical scaling or projection:

- `BdGQuasiclassicalBridgePlan` compares gap scales.
- `QuasiclassicalGLBridgePlan` compares the declared gap/order-parameter scale.
- `GLLondonBridgePlan` compares currents through a caller-supplied projection.
- `LondonCableBridgePlan` compares one-winding inductance.

A successful bridge is evidence over one common support; it does not make either model interchangeable or infer material parameters automatically.

## Qualification and nonclaims

`superconductivity_candidate_profiles()` and `superconductivity_candidate_campaigns()` retain separate support and locked-evaluation membership for every rung. Synthetic qualification establishes numerical validity only. A release requires governed analytic, experimental, or independent reference artifacts and signed gates. See [Superconductivity sources](superconductivity_sources.md).
