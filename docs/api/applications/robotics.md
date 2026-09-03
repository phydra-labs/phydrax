# Robotics applications

## Robot adaptation, kinematics, environments, and backends

::: phydrax.applications.robotics
    options:
      members: true
      show_root_heading: true
      members_order: source

## Reduced articulation

::: phydrax.discretization.ReducedArticulationPlan

---

::: phydrax.discretization.PreparedReducedArticulation

---

::: phydrax.discretization.ReducedArticulationState

---

::: phydrax.discretization.ArticulationKinematics

---

::: phydrax.discretization.ArticulationDualityEvidence

## Reduced rigid dynamics

::: phydrax.discretization.ReducedDynamicsStatus

---

::: phydrax.discretization.ReducedEnergyResult

---

::: phydrax.discretization.ReducedMassMatrixResult

---

::: phydrax.discretization.ReducedBiasTermsResult

---

::: phydrax.discretization.ReducedInverseDynamicsResult

---

::: phydrax.discretization.ReducedForwardDynamicsResult

---

::: phydrax.discretization.ReducedSymplecticStepPolicy

---

::: phydrax.discretization.ReducedSymplecticStepDiagnostics

---

::: phydrax.discretization.ReducedSymplecticStepResult

---

::: phydrax.discretization.reduced_energy

---

::: phydrax.discretization.reduced_mass_matrix

---

::: phydrax.discretization.reduced_bias_terms

---

::: phydrax.discretization.reduced_inverse_dynamics

---

::: phydrax.discretization.reduced_forward_dynamics

---

::: phydrax.discretization.reduced_symplectic_step

## Rigid inertial realization

::: phydrax.discretization.RigidInertialCoordinates

---

::: phydrax.discretization.RigidInertialParameters

---

::: phydrax.discretization.RigidInertialEvaluation

---

::: phydrax.discretization.RigidInertialParameterization

---

::: phydrax.discretization.realize_rigid_body_plans

## Articulated contact

::: phydrax.applications.contact.make_articulated_contact_participant

---

::: phydrax.applications.contact.build_contact_velocity_operator

---

::: phydrax.applications.contact.build_delassus_operator

---

::: phydrax.applications.contact.contact_duality_evidence

---

::: phydrax.applications.contact.prepare_articulated_contact

---

::: phydrax.applications.contact.PreparedArticulatedContact

---

::: phydrax.applications.contact.solve_articulated_contact

---

::: phydrax.applications.contact.apply_articulated_contact_impulse

---

::: phydrax.applications.contact.ArticulatedContactPreparationEvidence

---

::: phydrax.applications.contact.ArticulatedContactDualityEvidence

---

::: phydrax.applications.contact.ArticulatedContactEvidence

---

::: phydrax.applications.contact.ArticulatedContactResult

## Status-aware rollout and sampling MPC

::: phydrax.control.DiscreteControlDynamics

---

::: phydrax.control.ControlTrajectory

---

::: phydrax.control.SamplingMPCStatus

---

::: phydrax.control.SamplingMPCPlan

---

::: phydrax.control.SamplingMPCState

---

::: phydrax.control.SamplingMPCEvidence

---

::: phydrax.control.SamplingMPCResult

---

::: phydrax.control.plan_sampling_mpc

---

::: phydrax.control.initialize_sampling_mpc

---

::: phydrax.control.shift_sampling_mpc_state

---

::: phydrax.control.solve_sampling_mpc

## Manifold Radau transcription

::: phydrax.control.ManifoldCollocationStages

---

::: phydrax.control.ManifoldCollocationEvidence

---

::: phydrax.control.ManifoldRadauCollocationDefects

---

::: phydrax.control.manifold_radau_stages

---

::: phydrax.control.manifold_radau_collocation_defects

## Reduced rods

::: phydrax.applications.solid_mechanics.ReducedRodPlan

---

::: phydrax.applications.solid_mechanics.PreparedReducedRod

---

::: phydrax.applications.solid_mechanics.ReducedRodState

---

::: phydrax.applications.solid_mechanics.ReducedRodLiftEvidence

---

::: phydrax.applications.solid_mechanics.ReducedRodPowerEvidence

---

::: phydrax.applications.solid_mechanics.ReducedRodStrainEvidence

---

::: phydrax.applications.solid_mechanics.ReducedRodEvaluation

---

::: phydrax.applications.solid_mechanics.prepare_reduced_rod

---

::: phydrax.applications.solid_mechanics.reduced_rod_lift_operator

---

::: phydrax.applications.solid_mechanics.lift_reduced_rod_state

---

::: phydrax.applications.solid_mechanics.lift_reduced_rod_velocity

---

::: phydrax.applications.solid_mechanics.pullback_reduced_rod_loads

---

::: phydrax.applications.solid_mechanics.reduced_rod_power_evidence

---

::: phydrax.applications.solid_mechanics.reduced_rod_potential_energy

---

::: phydrax.applications.solid_mechanics.reduced_rod_kinetic_energy

---

::: phydrax.applications.solid_mechanics.evaluate_reduced_rod

## Interchange negotiation and composition

::: phydrax.interchange.AdapterStatus

---

::: phydrax.interchange.AdapterLoss

---

::: phydrax.interchange.AdapterRequirement

---

::: phydrax.interchange.AdapterCapability

---

::: phydrax.interchange.AdapterWaiver

---

::: phydrax.interchange.AdapterNegotiationResult

---

::: phydrax.interchange.AdapterReport

---

::: phydrax.interchange.negotiate_adapter

---

::: phydrax.interchange.compose_adapter_reports

---

::: phydrax.interchange.require_lossless
