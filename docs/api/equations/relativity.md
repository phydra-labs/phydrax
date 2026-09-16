# Relativistic equations

Relativistic EOS, SRHD/Valencia GRHD, ideal GRMHD, grey/multigroup/neutrino
radiation, higher-angular closures, resistive Ohm, and force-free physical systems.
Numerical recovery and stepping remain in `phydrax.solver`. See the
[matter and radiation guide](../../guides_relativistic_matter.md).

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
        - valencia_geometric_source_from_projection
        - IdealValenciaGRMHDSystem
        - GRGreyM1ClosureEvaluation
        - GRRadiationMatterExchange
        - GRGreyM1RadiationSystem
        - AbstractGRGreyOpacityPlan
        - ConstantGRGreyOpacityPlan
        - CompositeGRGreyOpacityPlan
        - GRGreyOpacityEvaluation
        - GRGreyRadiationInteractionPlan
        - GRMultigroupM1ClosureEvaluation
        - GRMultigroupM1RadiationSystem
        - GRMultigroupRadiationMatterExchange
        - GRMultigroupRadiationInteractionPlan
        - NeutrinoSpecies
        - GRNeutrinoM1ClosureEvaluation
        - GRNeutrinoM1System
        - GRNeutrinoMatterExchange
        - GRNeutrinoInteractionPlan
        - GRRadiationAngularClosureEvaluation
        - VariableEddingtonTensorClosurePlan
        - DiscreteOrdinatesRadiationPlan
        - MonteCarloRadiationClosureEvaluation
        - MonteCarloRadiationClosurePlan
        - RelativisticOhmEvaluation
        - ResistiveGRMHDOhmicClosure
        - ForceFreeConstraintEvaluation
        - ForceFreeCurrentEvaluation
        - ForceFreeProjectionResult
        - GRForceFreeSystem
