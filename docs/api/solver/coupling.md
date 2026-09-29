# Partitioned coupling

The public API lives under `phydrax.solver.coupling`. See the
[partitioned coupling guide](../../guides_partitioned_coupling.md) for numerical,
transactional, transfer, measurement, temporal-conversion, waveform, and
differentiation contracts.

## Ports, measurements, participants, and exchanges

::: phydrax.solver.coupling.CouplingPort

---

::: phydrax.solver.coupling.CouplingQuantity

---

::: phydrax.solver.coupling.CouplingMeasurement

---

::: phydrax.solver.coupling.CouplingMeasurementRepresentation

---

::: phydrax.solver.coupling.CouplingTransferRequirement

---

::: phydrax.solver.coupling.CouplingExchange

---

::: phydrax.solver.coupling.CouplingSubsystemCapabilities

---

::: phydrax.solver.coupling.AbstractCouplingSubsystem

---

::: phydrax.solver.coupling.CallableCouplingSubsystem

---

::: phydrax.solver.coupling.CouplingSubsystemResult

## Graph preparation and refresh

::: phydrax.solver.coupling.CouplingGraph

---

::: phydrax.solver.coupling.CouplingStagePlan

---

::: phydrax.solver.coupling.CouplingResourcePolicy

---

::: phydrax.solver.coupling.CouplingResourceEstimate

---

::: phydrax.solver.coupling.CouplingPreparationReport

---

::: phydrax.solver.coupling.PreparedCoupling

---

::: phydrax.solver.coupling.prepare_coupling

---

::: phydrax.solver.coupling.refresh_coupling

## Numerical and differentiation policies

::: phydrax.solver.coupling.CouplingSweep

---

::: phydrax.solver.coupling.CouplingTolerance

---

::: phydrax.solver.coupling.ExplicitCouplingPolicy

---

::: phydrax.solver.coupling.ImplicitCouplingPolicy

---

::: phydrax.solver.coupling.CouplingDifferentiationPolicy

## Window execution and evidence

::: phydrax.solver.coupling.CouplingWindow

---

::: phydrax.solver.coupling.CouplingState

---

::: phydrax.solver.coupling.CouplingWindowDiagnostics

---

::: phydrax.solver.coupling.CouplingProvenance

---

::: phydrax.solver.coupling.CouplingWindowResult

---

::: phydrax.solver.coupling.CouplingStatus

---

::: phydrax.solver.coupling.coupling_status_message

---

::: phydrax.solver.coupling.advance_coupling_window

## Fixed-window rollout

::: phydrax.solver.coupling.CouplingProblem

---

::: phydrax.solver.coupling.CouplingRolloutPlan

---

::: phydrax.solver.coupling.CouplingSolution

---

::: phydrax.solver.coupling.solve_coupling

## Fixed-capacity waveforms, adaptive windows, and epochs

::: phydrax.solver.coupling.CouplingWaveformPlan

---

::: phydrax.solver.coupling.CouplingWaveformGrid

---

::: phydrax.solver.coupling.CouplingWaveform

---

::: phydrax.solver.coupling.BarycentricCouplingTemporalTransfer

---

::: phydrax.solver.coupling.CouplingTemporalConversion

---

::: phydrax.solver.coupling.CouplingTemporalConversionKind

---

::: phydrax.solver.coupling.CouplingTemporalKind

---

::: phydrax.solver.coupling.CouplingWaveformAdaptationPolicy

---

::: phydrax.solver.coupling.adapt_coupling_waveform_grid

---

::: phydrax.solver.coupling.AdaptiveCouplingWindowPolicy

---

::: phydrax.solver.coupling.AdaptiveCouplingRolloutPlan

---

::: phydrax.solver.coupling.rollout_adaptive_coupling

---

::: phydrax.solver.coupling.AdaptiveCouplingSolution

---

::: phydrax.solver.coupling.PreparedCouplingEpoch

---

::: phydrax.solver.coupling.CouplingEpochTransitionPlan

---

::: phydrax.solver.coupling.transition_coupling_epoch

---

::: phydrax.solver.coupling.FixedGridSubcyclingSubsystem

## Method participants and lowering

Native fixed-step, DAE, and steady-response owners bound as participants, and the
declaration lowered onto `CouplingProblem`. See the
[partitioned coupling guide](../../guides_partitioned_coupling.md#native-method-participants).

::: phydrax.solver.coupling.AbstractMethodCouplingParticipant

---

::: phydrax.solver.coupling.FixedStepCouplingParticipant

---

::: phydrax.solver.coupling.DAECouplingParticipant

---

::: phydrax.solver.coupling.DAEParticipantNative

---

::: phydrax.solver.coupling.SteadyResponseCouplingParticipant

---

::: phydrax.solver.coupling.SteadyResponse

---

::: phydrax.solver.coupling.MethodParticipantState

---

::: phydrax.solver.coupling.MethodParticipantRandomness

---

::: phydrax.solver.coupling.MethodWindowBinding

---

::: phydrax.solver.coupling.PartitionedCouplingDeclaration

---

::: phydrax.solver.coupling.lower_partitioned_coupling

## Host execution

Explicit host orchestration of mixed host/native declarations, such as an FMI
co-simulation slave coupled to native participants. Native preparation refuses host
participants. See the
[partitioned coupling guide](../../guides_partitioned_coupling.md#host-participants)
and [FMI coupling](../../guides_energy_interchange.md#coupling-an-fmu-to-native-participants).

::: phydrax.solver.coupling.AbstractHostCouplingParticipant

---

::: phydrax.solver.coupling.HostParticipantState

---

::: phydrax.solver.coupling.prepare_host_coupling

---

::: phydrax.solver.coupling.PreparedHostCoupling

---

::: phydrax.solver.coupling.advance_host_coupling_window

---

::: phydrax.solver.coupling.HostCouplingWindowResult

---

::: phydrax.solver.coupling.HostRollback

---

::: phydrax.solver.coupling.HostWindowCommit

---

::: phydrax.solver.coupling.solve_host_coupling

---

::: phydrax.solver.coupling.HostCouplingSolution

## Interface bindings

Revision-bound physical interfaces across mesh, analytic, and sheet owners. See
[Numerical interoperability](../../guides_numerical_interoperability.md#interface-bindings).

::: phydrax.solver.coupling.InterfaceBinding

---

::: phydrax.solver.coupling.InterfaceEndpoint

---

::: phydrax.solver.coupling.InterfaceSource

---

::: phydrax.solver.coupling.PairedSupportAttachment

---

::: phydrax.solver.coupling.SheetViewAttachment

---

::: phydrax.solver.coupling.InterfaceIncidence

---

::: phydrax.solver.coupling.EmbeddedMeaning

## Spatial coupled problems

Steady problems assembled from native owners' published components,
contributions, and laws, solved natively and certified on their original
equations and interface defects. See
[Numerical interoperability](../../guides_numerical_interoperability.md#spatial-coupled-problems).

### Components

::: phydrax.solver.coupling.AbstractSpatialComponent

---

::: phydrax.solver.coupling.AbstractTraceComponent

---

::: phydrax.solver.coupling.AbstractReconstructionComponent

---

::: phydrax.solver.coupling.AbstractCapacityComponent

---

::: phydrax.solver.coupling.AbstractPreparedCapacity

---

::: phydrax.solver.coupling.ComponentBlock

---

::: phydrax.solver.coupling.ComponentField

---

::: phydrax.solver.coupling.VariationalComponent

---

::: phydrax.solver.coupling.VariationalCapacity

---

::: phydrax.solver.coupling.GalerkinBoundaryComponent

---

::: phydrax.solver.coupling.ComponentResidualProvider

---

::: phydrax.solver.coupling.ReducedComponent

---

::: phydrax.solver.coupling.ComponentSpace

### Existing 3-D FEM–BEM products

The prepared matching scalar Johnson–Nédélec and static-elasticity Costabel
products publish their own named blocks, operator, right-hand side, and
`prepared_id` verbatim; their preparation refusals are unchanged. The
nonmatching dense and dynamic convolution-quadrature products publish no
separate named blocks and are not components.

::: phydrax.solver.coupling.ScalarLaplaceFEMBEMComponent

---

::: phydrax.solver.coupling.ScalarLaplaceFEMBEMArguments

---

::: phydrax.solver.coupling.ElasticityFEMBEMComponent

---

::: phydrax.solver.coupling.ElasticityFEMBEMArguments

### Contributions

::: phydrax.solver.coupling.ContributionEndpoint

---

::: phydrax.solver.coupling.AbstractContribution

---

::: phydrax.solver.coupling.LinearContribution

---

::: phydrax.solver.coupling.LoadContribution

---

::: phydrax.solver.coupling.AbstractContributionResidual

---

::: phydrax.solver.coupling.ResidualContribution

---

::: phydrax.solver.coupling.EliminationContribution

---

::: phydrax.solver.coupling.LawBlock

---

::: phydrax.solver.coupling.LawImposition

---

::: phydrax.solver.coupling.Contribution

---

::: phydrax.solver.coupling.ContributionSpace

### Laws, impositions, and evidence

::: phydrax.solver.coupling.AbstractCouplingLaw

---

::: phydrax.solver.coupling.PreparedLaw

---

::: phydrax.solver.coupling.AbstractLawCertificate

---

::: phydrax.solver.coupling.InterfaceDefectReport

---

::: phydrax.solver.coupling.ScalarTransmissionLaw

---

::: phydrax.solver.coupling.TransmissionSide

---

::: phydrax.solver.coupling.MatchingElimination

---

::: phydrax.solver.coupling.MortarMultiplier

---

::: phydrax.solver.coupling.MultiplierFamily

---

::: phydrax.solver.coupling.MortarImposition

---

::: phydrax.solver.coupling.TransmissionImposition

---

::: phydrax.solver.coupling.MortarEvidence

---

::: phydrax.solver.coupling.EliminationEvidence

---

::: phydrax.solver.coupling.NitscheImposition

---

::: phydrax.solver.coupling.NitscheVariant

---

::: phydrax.solver.coupling.NitscheEvidence

### Flux, port, and transfer laws

`ConservativeFluxLaw` evaluates one shared interface flux at the common
quadrature and injects it into both sides with consistent orientation;
`IntegralPortLaw` closes a field boundary port with a connector of a
`phydrax.system_modeling` acausal network; `FieldTransferLaw` couples two
co-located fields through a native `FieldTransfer`, its dual pullback, and the
target measure.

A flux whose density reads the coupled runtime arguments names them in
`AbstractInterfaceFlux.runtime_inputs`: preparation never evaluates it without
them, it cannot be certified affine, and each named input must be the target of
a refresh `ParameterBinding`. `MonotoneInterfaceConductance` is such
a flux: it reads a learned input-convex potential bound through a
`ParameterBinding` and is monotone and dissipative for every parameter value.
See
[Learned interface laws](../../guides_numerical_interoperability.md#learned-interface-laws).

::: phydrax.solver.coupling.AbstractInterfaceFlux

---

::: phydrax.solver.coupling.InterfaceConductance

---

::: phydrax.solver.coupling.GapRadiation

---

::: phydrax.solver.coupling.MonotoneInterfaceConductance

---

::: phydrax.solver.coupling.ConservativeFluxLaw

---

::: phydrax.solver.coupling.ConservativeFluxEvidence

---

::: phydrax.solver.coupling.PortSide

---

::: phydrax.solver.coupling.IntegralPortLaw

---

::: phydrax.solver.coupling.IntegralPortEvidence

---

::: phydrax.solver.coupling.FieldTransferLaw

---

::: phydrax.solver.coupling.FieldTransferEvidence

### Boundary-integral transmission

`BoundaryIntegralTransmissionLaw` couples a volume trace component to a
`GalerkinBoundaryComponent` through the bordered Johnson–Nédélec relation: the
volume rows receive the DP0 conormal load, the law owns the declared continuous
P1 projection of the volume trace, and the boundary equation receives
`(M/2 - K)` of that projection. See
[Numerical interoperability](../../guides_numerical_interoperability.md#spectral-element-virtual-element-and-boundary-element-transmission).

::: phydrax.solver.coupling.BoundaryIntegralTransmissionLaw

---

::: phydrax.solver.coupling.BoundaryIntegralSide

---

::: phydrax.solver.coupling.BoundaryIntegralEvidence

### Interface quadrature

::: phydrax.solver.coupling.InterfaceQuadraturePolicy

---

::: phydrax.solver.coupling.prepare_interface_quadrature

---

::: phydrax.solver.coupling.InterfaceQuadrature

---

::: phydrax.solver.coupling.InterfaceSideQuadrature

---

::: phydrax.solver.coupling.FacetResampling

---

::: phydrax.solver.coupling.InterfaceCoverageEvidence

### Preparation

::: phydrax.solver.coupling.CoupledProblemPlan

---

::: phydrax.solver.coupling.CoupledGauge

---

::: phydrax.solver.coupling.CoupledResourcePolicy

---

::: phydrax.solver.coupling.prepare_coupled_problem

---

::: phydrax.solver.coupling.PreparedCoupledProblem

---

::: phydrax.solver.coupling.CoupledExecution

### Bounded lane execution

Signature-grouped lane worksets of homogeneous components and contribution blocks.
See [Coupled lane worksets](../../guides_numerical_interoperability.md#coupled-lane-worksets).

::: phydrax.solver.coupling.CoupledExecutionPolicy

---

::: phydrax.solver.coupling.PreparedCoupledExecution

---

::: phydrax.solver.coupling.LaneWorkset

---

::: phydrax.solver.coupling.LaneWorksetEstimate

---

::: phydrax.solver.coupling.LaneDerivatives

---

::: phydrax.solver.coupling.InterfaceLaneSubject

---

::: phydrax.solver.coupling.LaneKind

### Parameter bindings

A `ParameterBinding` binds one scientific parameter (a `ValuePort`) to named
runtime inputs of components with a role, a derivative surface, and a
`"refresh"` or `"reprepare"` change. Refresh values are arrays of the port's
event shape or model-authority `ComponentBinding`s, supplied at every solve.
See [Parameter bindings](../../guides_numerical_interoperability.md#parameter-bindings).

::: phydrax.solver.coupling.ParameterBinding

---

::: phydrax.solver.coupling.RuntimeInput

---

::: phydrax.solver.coupling.ParameterRole

---

::: phydrax.solver.coupling.ParameterChange

---

::: phydrax.solver.coupling.CoupledArguments

---

::: phydrax.solver.coupling.PreparedParameters

### Observation bindings

Observation bindings name a component field and the measurement it predicts.
Preparation lowers them through the owner's prepared field queries, exact side
traces, and residual-reaction fluxes; every `CoupledSolution` evaluates them
into `PreparedQuantityField` values that compare with data prepared from the
same records. See
[Observation bindings](../../guides_numerical_interoperability.md#observation-bindings).

::: phydrax.solver.coupling.AbstractObservationBinding

---

::: phydrax.solver.coupling.AbstractPreparedObservation

---

::: phydrax.solver.coupling.FieldPointObservation

---

::: phydrax.solver.coupling.FieldBoundaryObservation

---

::: phydrax.solver.coupling.FieldFluxObservation

---

::: phydrax.solver.coupling.PreparedPointObservation

---

::: phydrax.solver.coupling.PreparedBoundaryObservation

---

::: phydrax.solver.coupling.PreparedFluxObservation

---

::: phydrax.solver.coupling.MeasurementIdentity

---

::: phydrax.solver.coupling.BoundaryStatistic

### Solving and acceptance

::: phydrax.solver.coupling.solve_coupled_problem

---

::: phydrax.solver.coupling.CoupledSolution

---

::: phydrax.solver.coupling.ComponentCertificate

---

::: phydrax.solver.coupling.certify_coupled_state

### Exact static condensation

Named pivot blocks eliminated through a qualified owner factorization,
reconstructed, and certified on the original coupled equations. Approximate
pivot actions stay preconditioners of the original system. See
[Numerical interoperability](../../guides_numerical_interoperability.md#block-coordinates-condensation-and-transient-systems).

::: phydrax.solver.coupling.condense_coupled_problem

---

::: phydrax.solver.coupling.CondensedCoupledProblem

---

::: phydrax.solver.coupling.solve_condensed_problem

---

::: phydrax.solver.coupling.CondensedSolution

---

::: phydrax.solver.coupling.CondensationEvidence

### Transient coupled problems

Differential fields with their owners' capacity operators, quasistatic
components, and law unknowns lowered onto one structurally admitted index-one
native DAE, integrated by the native BDF/theta runtime, and certified on the
original transient rows.

::: phydrax.solver.coupling.TransientField

---

::: phydrax.solver.coupling.TransientScale

---

::: phydrax.solver.coupling.TransientArguments

---

::: phydrax.solver.coupling.prepare_coupled_transient

---

::: phydrax.solver.coupling.PreparedCoupledTransient

---

::: phydrax.solver.coupling.solve_coupled_transient

---

::: phydrax.solver.coupling.CoupledTransientSolution

---

::: phydrax.solver.coupling.TransientCertificate

### Coupled transitions and observation ports

One accepted step of a prepared coupled transient between two physical times,
for the discrete-time control and state-space consumers: the native DAE solve is
prepared once on a template window and rebound to each interval, the original
transient rows are certified, and a failed step rolls back to its source state.
`CoupledObservationPort` evaluates the problem's observation bindings on a
transient state through the control output and state-space location ABIs.

::: phydrax.solver.coupling.prepare_coupled_transition

---

::: phydrax.solver.coupling.PreparedCoupledTransition

---

::: phydrax.solver.coupling.CoupledTransitionStep

---

::: phydrax.solver.coupling.CoupledTransitionStatus

---

::: phydrax.solver.coupling.coupled_transition_status_name

---

::: phydrax.solver.coupling.COUPLED_TRANSITION_SUCCESS

---

::: phydrax.solver.coupling.COUPLED_TRANSITION_NATIVE_FAILURE

---

::: phydrax.solver.coupling.COUPLED_TRANSITION_CERTIFICATE_FAILURE

---

::: phydrax.solver.coupling.COUPLED_TRANSITION_NONFINITE

---

::: phydrax.solver.coupling.CoupledObservationPort

## Lifecycle composition entries

An accepted `CouplingState` split into lifecycle composition entries, restored
from a published composition, and one changed exchange boundary value re-derived
through a rebound epoch's route. See
[Numerical interoperability](../../guides_numerical_interoperability.md#lifecycle-adaptation-and-junctions).

::: phydrax.solver.coupling.coupling_composition_entries

---

::: phydrax.solver.coupling.coupling_state_from_composition

---

::: phydrax.solver.coupling.coupling_exchange_transport
