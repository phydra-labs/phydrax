# Relativistic solver runtimes

Bounded conserved-to-primitive recovery, fixed-grid Valencia GRHD, periodic
all-cells-active compatible constrained transport, and atomic GRMHD stepping.
Bounded/excision or mixed inactive grids are rejected until a boundary-aware UCT
implementation exists. Equation semantics remain in `phydrax.equations`; Z4c/matter
coupling remains in `phydrax.applications.numerical_relativity`.

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
        - GRHDStageGeometry
        - lower_grhd_stage_geometry
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
