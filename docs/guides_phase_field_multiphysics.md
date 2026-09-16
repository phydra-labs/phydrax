# Coupled phase-field multiphysics

`phydrax.applications.phase_field` includes one compositional closure for
nonisothermal solidification, anti-trapping alloy transport, nucleation, mechanics,
incompressible Model-H flow, and electrostatic/electrochemical coupling.

The coupled route does not accept a candidate merely because each subsystem returned
finite values. It requires one complete energy, entropy, exchange, conservation, and
event ledger.

## Coupling graph

`PhaseFieldCouplingTerm` declares:

- input and residual-output fields;
- storage, dissipation, and external-work channels;
- internal exchange inputs and outputs;
- conservation channels.

`PhaseFieldCouplingGraph` rejects duplicate storage/work ownership, duplicate residual
owners, missing field producers, or an internal exchange without exactly one source
and one target.

## Total ledger

`CoupledPhaseFieldLedger` combines:

- thermochemical and interfacial energy;
- thermal internal energy;
- elastic and kinetic energy;
- electrostatic energy;
- nucleation interfacial energy;
- phase, thermal, viscous, mechanical, and electrical dissipation;
- heat, mechanical, flow, electric, stochastic, and reservoir work;
- component and charge inventories;
- entropy production;
- internal exchange cancellation.

An accepted step satisfies

$$
\Delta E_{\mathrm{total}} + D_{\mathrm{total}} - W_{\mathrm{external}} = R_E
$$

while every internal exchange appears once with each sign. Component, charge, and
entropy defects are gated independently.

## Power-adjoint transfers

`PowerAdjointTransferPair` binds a primal state transfer and dual force transfer. It
checks

$$
\langle T u, f\rangle_{\mathrm{target}}
 = \langle u, T^* f\rangle_{\mathrm{source}}
$$

and constant reproduction. It is the coupling boundary for FE–MAC, FE–cochain, and
nonmatching-mesh mechanics transfers.

## Nonisothermal solidification

`NonisothermalGrandPotentialPhase` provides a thermodynamically consistent
constant-heat-capacity grand potential. It returns composition, susceptibility,
entropy, internal energy, heat capacity, and stability evidence from one potential.

`NonisothermalSolidificationModel` interpolates phase thermodynamics through softmax
phase weights. `NonisothermalSolidificationPlan` performs the local enthalpy/temperature
constitutive inversion used by FE or FV thermal residuals. Temperature must remain
positive, and both energy and entropy evidence are returned.

This constitutive layer composes with the existing finite-volume
`SolidLiquidEnthalpyPlan`; it does not introduce a second latent-heat convention.

## Anti-trapping

`AntiTrappingCurrentPlan` implements a calibrated current of the form

$$
\mathbf j_{at} = -a_t W A(\phi,U,T)\,\dot\phi\,\mathbf n.
$$

The normal is evaluated only where the phase gradient exceeds the declared tolerance.
The current is exactly zero away from a moving interface. Calibration coefficient,
interface width, partition coefficient, and model identity are inseparable parts of
the plan.

Anti-trapping is a component flux, so it must enter conservation through a discrete
divergence. It is not an unconstrained source term.

## Nucleation

`ClassicalNucleationRateLaw` provides homogeneous or heterogeneous classical rates,
critical radius, and energy barrier in two or three dimensions.

`NucleationEventPlan` uses `PoissonClockRealization` thresholds keyed by stable channel
and event IDs. It stores integrated hazard, prefix-stable event counts, marks,
orientation, proposed radius, component demand, and energy demand.

`transact` sorts overlapping event proposals deterministically and admits them only
when component and thermal/reservoir inventories are sufficient. Capacity or inventory
failure leaves the accepted clock state unchanged.

## Mechanics

`LinearElasticPhaseMaterial` owns one symmetric positive-definite stiffness and
transformation strain. `PhaseMechanicalModel` interpolates stiffness and eigenstrain
by phase weight and derives elastic energy, stress, and phase driving force from one
potential.

`PhaseHyperelasticModel` composes identified native `MixedHyperelasticLaw` instances
for finite-strain work. Existing Newmark and material-transaction runtimes remain the
time-integration substrate.

## Model-H flow

`PhaseFluidMaterial` provides positive phase densities and viscosities.
`ModelHCouplingPlan` evaluates:

- mixture density and viscosity;
- kinetic energy;
- viscous dissipation;
- phase advection;
- capillary force;
- phase/flow power exchange;
- incompressibility evidence.

The canonical force route is $\mu\nabla\phi$. Its kinetic power cancels the phase
advection power pointwise. The alternative $-\phi\nabla\mu$ route includes the
corresponding pressure-shift power before acceptance.

## Electrostatics and electrochemistry

`PhasePermittivityLaw` preserves positive phase-dependent permittivity.
`PhaseElectrostaticCouplingPlan` distinguishes fixed-charge and fixed-voltage
ensembles and returns field energy, electric displacement, phase driving force,
Maxwell stress, and Gauss-law evidence.

`ElectrochemicalCouplingPlan` combines chemical and electric potentials, builds ionic
flux, tracks charge, and reports electrical dissipation and Joule heating. Joule heat
is an internal electrical-to-thermal exchange and must not be added twice.

## Coupled step and production runtime

`CoupledMultiphysicsPlan` evaluates thermal, anti-trapping, nucleation, mechanical,
flow, electrostatic, and electrochemical terms in one atomic candidate. The result is
promoted only when every sub-evaluation and the global ledger pass.

`CoupledMultiphysicsFixedStepMethod` exposes this route through the native fixed-step,
retry, segmented-production, and durable-checkpoint runtime.
`CoupledMultiphysicsEpochIdentity` binds phase, thermal, mechanical, flow,
electrostatic, partition, and event-realization identities.

## Support profiles

`coupled_phase_field_candidate_profiles()` defines exact support tuples for:

1. nonisothermal solidification;
2. quantitative anti-trapping alloy solidification;
3. nucleating solidification;
4. elastochemical phase transformation;
5. Model-H phase flow;
6. electrochemical phase transformation;
7. thermofluid solidification;
8. the electro-elasto-hydrodynamic flagship.

`coupled_phase_field_released_profiles()` binds accepted, time-bounded qualification
evidence to those tuples.

## Qualification

Run the coupled campaign with two CPU devices when accelerator multiplicity is not
available:

```console
PYTHONPATH=. JAX_ENABLE_X64=1 \
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
python tools/phase_field_multiphysics_qualification.py
```

It records PFHub-style thermal, mechanical, flow, and electrostatic observables;
anti-trapping and nucleation evidence; power-adjoint transfer; an actual two-device
collective; and one full flagship step.

Performance evidence:

```console
PYTHONPATH=. JAX_ENABLE_X64=1 \
python benchmarks/phase_field_multiphysics.py
```

## Explicit nonclaims

The coupled closure does not claim:

- compressible multiphase flow;
- full electrodynamics, magnetic induction, or magnetohydrodynamics;
- ab-initio nucleation-rate prediction;
- arbitrary anti-trapping outside its declared thin-interface calibration;
- fracture coupling;
- chemical reaction networks beyond declared phase transformation;
- bitwise equivalence across different collective layouts;
- global smooth differentiation through event, active-set, AMR, retry, or repartition
  transitions.
