# Relativistic solver runtimes

Bounded conserved-to-primitive recovery; boundary-aware Valencia GRHD/GRMHD;
compatible constrained transport; gray, multigroup, neutrino, and polarized radiation;
and atomic ideal, radiation-coupled, resistive, and force-free-transition runtimes.
Equation semantics remain in `phydrax.equations`; Z4c/matter coupling remains in
`phydrax.applications.numerical_relativity`.

::: phydrax.solver
    options:
      show_root_heading: true
      members:
        - AtmosphereFloorStatus
        - AtmosphereFloorPolicy
        - AtmosphereCorrectionLedger
        - GRHDC2PStatus
        - GRHDC2PCandidateRecord
        - GRHDC2PResult
        - GRHDC2PPolicy
        - GRHDFiniteVolumeRunStatus
        - GRHDBoundaryCondition
        - GRHDBoundaryPair
        - GRHDBoundaryTrace
        - metric_aware_grhd_boundary_trace
        - GRHDFaceFluxResult
        - GRHDFaceFluxPlan
        - ValenciaFiniteVolumeStageGeometry
        - lower_valencia_stage_geometry
        - GRHDFiniteVolumeEvaluation
        - GRHDConservationLedger
        - GRHDFiniteVolumeState
        - GRHDFiniteVolumeStepResult
        - FixedGridGRHDSSPRK3Plan
        - VectorPotentialGaugeKind
        - GRMHDMagneticStateLayout
        - GRMHDVectorPotentialGauge
        - GRMHDCTState
        - GRMHDCTRate
        - GRMHDCTDefectLedger
        - GRMHDConstrainedTransportPlan
        - GRMHDRunStatus
        - GRMHDState
        - GRMHDSpatialRate
        - GRMHDStageProposal
        - GRMHDStageEvidence
        - GRMHDDefectLedger
        - GRMHDStepResult
        - GRMHDSSPRK3Plan
        - GRM1BoundaryKind
        - GRM1BoundaryCondition
        - GRM1BoundaryPair
        - GRM1ReconstructionKind
        - GRM1FiniteVolumeRunStatus
        - GRM1FiniteVolumeState
        - GRM1SpatialRate
        - GRM1ConservationLedger
        - GRM1StepResult
        - FixedGridGRM1SSPRK3Plan
        - GRRadiationExchangeLedger
        - GRRMHDSourceStatus
        - GRRMHDSourceResult
        - GRRMHDImplicitSourcePlan
        - GRRMHDRunStatus
        - GRRMHDState
        - GRRMHDStageProposal
        - GRRMHDStageEvidence
        - GRRMHDDefectLedger
        - GRRMHDStepResult
        - FixedGridGRRMHDIMEXPlan
        - GRMultigroupM1State
        - GRMultigroupM1StepResult
        - FixedGridGRMultigroupM1SSPRK3Plan
        - GRNeutrinoM1State
        - GRNeutrinoLeptonLedger
        - GRNeutrinoM1StepResult
        - FixedGridGRNeutrinoM1Plan
        - ResistiveGRRMHDRunStatus
        - ResistiveGRRMHDState
        - ResistiveGRRMHDLedger
        - ResistiveGRRMHDStepResult
        - FixedGridResistiveGRRMHDIMEXPlan
        - GRMHDForceFreeHybridState
        - GRMHDForceFreeTransitionLedger
        - GRMHDForceFreeTransitionResult
        - GRMHDForceFreeTransitionPlan
        - GRPolarizedRadiationFeedbackState
        - GRPolarizedRadiationFeedbackLedger
        - GRPolarizedRadiationFeedbackResult
        - GRPolarizedRadiationFeedbackPlan
        - polarized_propagation_matrix
