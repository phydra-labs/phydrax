# Relativistic equations

Relativistic EOS, SRHD/Valencia GRHD, ideal GRMHD, grey M1 radiation, resistive Ohm,
and force-free physical systems. Numerical recovery and stepping remain in
`phydrax.solver`. See the [matter and radiation guide](../../guides_relativistic_matter.md).

::: phydrax.equations
    options:
      show_root_heading: true
      members:
        - AbstractRelativisticEOS
        - RelativisticEOSStatus
        - RelativisticEOSDomainEvidence
        - RelativisticEOSState
        - RelativisticEOSTableEvidence
        - GammaLawEOS
        - PiecewisePolytropicEOS
        - HybridColdThermalEOS
        - TabulatedFiniteTemperatureEOS
        - relativistic_eos_status_name
        - RelativisticHydrodynamicsLayout
        - RelativisticFluidEvaluation
        - ValenciaGeometrySource
        - SRHDSystem
        - ValenciaGRHDSystem
        - ValenciaRecoveryStatus
        - ValenciaPrimitiveRecovery
        - ValenciaHLLEBounds
        - ValenciaHLLEFlux
        - ValenciaMetricDerivatives
        - IdealValenciaGRMHDSystem
        - GRGreyM1ClosureEvaluation
        - GRRadiationMatterExchange
        - GRGreyM1RadiationSystem
        - RelativisticOhmEvaluation
        - ResistiveGRMHDOhmicClosure
        - ForceFreeConstraintEvaluation
        - ForceFreeCurrentEvaluation
        - ForceFreeProjectionResult
        - GRForceFreeSystem
