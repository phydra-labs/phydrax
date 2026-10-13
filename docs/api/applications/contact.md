# Contact applications

## Closure and materials

::: phydrax.applications.contact.ContactMaterialPairTable

---

::: phydrax.applications.contact.ContactClosurePlan

---

::: phydrax.applications.contact.ContactRouteState

Route remapping uses stable key ordering and binary-search ranges with linear workspace. The first valid old slot supplies inherited state when old keys repeat; duplicate old matches and duplicate valid new keys retain explicit refusal flags. Candidate ordering, defaults for invalid/unmatched slots, lifecycle counts, commit, and rollback are unchanged.

---

::: phydrax.applications.contact.BarrierNormalContactLaw

---

::: phydrax.applications.contact.GeometricContactNormalLaw

---

::: phydrax.applications.contact.CompliantNormalContactLaw

---

::: phydrax.applications.contact.AdhesiveBarrierNormalLaw

---

::: phydrax.applications.contact.RegularizedCoulombContactLaw

---

::: phydrax.applications.contact.AnisotropicCoulombContactLaw

---

::: phydrax.applications.contact.RateStateFrictionContactLaw

---

::: phydrax.applications.contact.FrictionWearEvolutionLaw

## Formulations

::: phydrax.applications.contact.ConvergentContactPotentialPlan

---

::: phydrax.applications.contact.ContactConeProgram

---

::: phydrax.applications.contact.ContactConeSolverPlan

---

::: phydrax.applications.contact.MortarContactPlan

---

::: phydrax.applications.contact.OneSidedNitscheContactPlan

---

::: phydrax.applications.contact.UnbiasedNitscheContactPlan

---

::: phydrax.applications.contact.MeshTiePlan

## Patch and multiphysics contact

::: phydrax.applications.contact.HydroelasticMaterialPlan

---

::: phydrax.applications.contact.HomogenizedRoughContactPlan

---

::: phydrax.applications.contact.PeriodicRoughContactPlan

---

::: phydrax.applications.contact.CoupledContactTransportLaw

---

::: phydrax.applications.contact.LubricationContactPlan

---

::: phydrax.applications.contact.ReynoldsFilmPlan

---

::: phydrax.applications.contact.ReynoldsPressureBoundaryConditions

## Solvers, coupling, and derivatives

::: phydrax.applications.contact.ContactSolvePolicy

---

::: phydrax.applications.contact.MonolithicContactPlan

---

::: phydrax.applications.contact.CrossDiscretizationContactResult

---

::: phydrax.applications.contact.ContactStateTransferPlan

---

::: phydrax.applications.contact.ContactBlockPreconditionerPlan

---

::: phydrax.applications.contact.contact_closure_gap_jvp

---

::: phydrax.applications.contact.contact_cone_solution_jvp

---

::: phydrax.applications.contact.mortar_gap_jvp

---

::: phydrax.applications.contact.SAPContactSolverPlan

---

::: phydrax.applications.contact.SemismoothContactSolverPlan

---

::: phydrax.applications.contact.PrimalDualContactSolverPlan

---

::: phydrax.applications.contact.RigidContactParticipant

---

::: phydrax.applications.contact.prepare_point_contact_participant

---

::: phydrax.applications.contact.prepare_mpm_contact_participant

## Shared residual participants and affine bodies

See [Monolithic implicit contact](../../guides_monolithic_contact.md) for
equations, the potential-consistency gate, conservative envelopes, and atomic
acceptance. Geometry participants remain distinct from mechanics participants.

::: phydrax.applications.contact.AbstractImplicitContactParticipant

::: phydrax.applications.contact.ContactParticipantEvaluation

::: phydrax.applications.contact.ContactParticipantStepLimit

::: phydrax.applications.contact.ContactKinematicLinearization

::: phydrax.applications.contact.make_implicit_contact_participant

::: phydrax.applications.contact.prepare_finite_element_contact_participant

::: phydrax.applications.contact.prepare_implicit_mpm_contact_participant

::: phydrax.applications.contact.prepare_iga_contact_participant

::: phydrax.applications.contact.prepare_implicit_reduced_rod_contact_geometry

::: phydrax.applications.contact.AffineBodyPlan

::: phydrax.applications.contact.AffineBodyHistory

::: phydrax.applications.contact.AffineBodyParticipant

::: phydrax.applications.contact.prepare_affine_body_participant

## Monolithic lifecycle and local derivatives

`MonolithicContactPlan(participants, scene, contact, search, ccd,
geometry_owner_indices=..., fixed_geometry_states=..., friction=None, policy=None)`
binds an ordered tuple of native mechanics spaces to geometry. Policy fields are
`merit`, `method`, `termination`, `maximum_lag_iterations`, and `lag_tolerance`.
Coordinates/history are mechanics-block tuples; fixed walls are explicit geometry
states. Results retain the frozen candidate epoch for local sensitivity.

::: phydrax.applications.contact.MonolithicContactState

::: phydrax.applications.contact.initialize_monolithic_contact_state

::: phydrax.applications.contact.PreparedMonolithicContactStep

::: phydrax.applications.contact.prepare_monolithic_contact_step

::: phydrax.applications.contact.solve_monolithic_contact_step

::: phydrax.applications.contact.PreparedMonolithicContactEquilibrium

::: phydrax.applications.contact.prepare_monolithic_contact_equilibrium

::: phydrax.applications.contact.solve_monolithic_contact_equilibrium

::: phydrax.applications.contact.ContactSolveDiagnostics

::: phydrax.applications.contact.MonolithicContactResult

::: phydrax.applications.contact.monolithic_contact_solution_jvp

::: phydrax.applications.contact.monolithic_contact_solution_vjp
