# Bubble dynamics

## Environment and law contracts

::: phydrax.bubble_dynamics.BubbleEnvironment

---

::: phydrax.bubble_dynamics.BubbleScales

---

::: phydrax.bubble_dynamics.BubbleGasState

---

::: phydrax.bubble_dynamics.BubbleGasEvaluation

---

::: phydrax.bubble_dynamics.BubbleGasCapabilities

---

::: phydrax.bubble_dynamics.AbstractBubbleGasLaw

---

::: phydrax.bubble_dynamics.AbstractBubbleCompartmentGasLaw

---

::: phydrax.bubble_dynamics.BubbleGasMergeResult

---

::: phydrax.bubble_dynamics.BubbleGasSplitResult

---

::: phydrax.bubble_dynamics.AbstractBubbleLiquidLaw

---

::: phydrax.bubble_dynamics.BubbleLiquidEvaluation

---

::: phydrax.bubble_dynamics.AbstractBubbleInterfaceLaw

---

::: phydrax.bubble_dynamics.AbstractSmoothBubbleInterfaceLaw

---

::: phydrax.bubble_dynamics.BubbleInterfaceEvaluation

---

::: phydrax.bubble_dynamics.AbstractBubblePressureDrive

---

::: phydrax.bubble_dynamics.PressureDriveEvaluation

---

::: phydrax.bubble_dynamics.MOLAR_GAS_CONSTANT

## Gas laws

::: phydrax.bubble_dynamics.PolytropicBubbleGasLaw

---

::: phydrax.bubble_dynamics.HardCorePolytropicBubbleGasLaw

---

::: phydrax.bubble_dynamics.BoundaryLayerThermalBubbleGasLaw

---

::: phydrax.bubble_dynamics.ReducedTransferBubbleGasLaw

---

::: phydrax.bubble_dynamics.preston_transfer_coefficient

---

::: phydrax.bubble_dynamics.SpectralThermalBubbleGasLaw

---

::: phydrax.bubble_dynamics.MaterialBubbleGasLaw

---

::: phydrax.bubble_dynamics.IsothermalIdealBubbleGasLaw

---

::: phydrax.bubble_dynamics.CaloricIdealBubbleGasLaw

---

::: phydrax.bubble_dynamics.sphere_volume

---

::: phydrax.bubble_dynamics.sphere_radius

## Liquid laws

::: phydrax.bubble_dynamics.NewtonianBubbleLiquidLaw

---

::: phydrax.bubble_dynamics.PowerLawBubbleLiquidLaw

---

::: phydrax.bubble_dynamics.KelvinVoigtBubbleLiquidLaw

---

::: phydrax.bubble_dynamics.ZenerBubbleLiquidLaw

---

::: phydrax.bubble_dynamics.OldroydBBubbleLiquidLaw

## Interfaces and shells

::: phydrax.bubble_dynamics.CleanBubbleInterfaceLaw

---

::: phydrax.bubble_dynamics.TolmanCorrectionPolicy

---

::: phydrax.bubble_dynamics.MarmottantShell

---

::: phydrax.bubble_dynamics.GompertzMarmottantShell

---

::: phydrax.bubble_dynamics.HoffShell

---

::: phydrax.bubble_dynamics.ChurchShell

---

::: phydrax.bubble_dynamics.SarkarShell

---

::: phydrax.bubble_dynamics.DoinikovShearThinningShell

---

::: phydrax.bubble_dynamics.MaxwellShell

## Drives

::: phydrax.bubble_dynamics.ConstantPressureDrive

---

::: phydrax.bubble_dynamics.HarmonicPressureDrive

---

::: phydrax.bubble_dynamics.PulsedPressureDrive

---

::: phydrax.bubble_dynamics.SampledPressureDrive

---

`SampledDriveInterpolation` is the closed selector `"linear" | "cubic_hermite"`.

## Radial model

`RadialBubbleEquation` is the closed selector `"rayleigh_plesset" |
"rayleigh_plesset_radiation" | "rayleigh_plesset_gas_radiation" |
"keller_miksis" | "gilmore"`.

---

::: phydrax.bubble_dynamics.RadialBubbleModel

---

::: phydrax.bubble_dynamics.BubbleState

---

::: phydrax.bubble_dynamics.BubbleEquilibrium

---

::: phydrax.bubble_dynamics.BubbleWallPressure

---

::: phydrax.bubble_dynamics.RadialBubbleRates

## Single-bubble solves

`SingleBubbleIntegrator` is the closed selector `"auto" | "explicit" | "stiff"`.

---

`BubbleDifferentiation` is the closed selector `"reverse" | "forward"`.

---

::: phydrax.bubble_dynamics.BubbleEventPolicy

---

::: phydrax.bubble_dynamics.SingleBubblePlan

---

::: phydrax.bubble_dynamics.PreparedSingleBubble

---

::: phydrax.bubble_dynamics.solve_single_bubble

---

::: phydrax.bubble_dynamics.SingleBubbleResult

---

::: phydrax.bubble_dynamics.SingleBubbleTrajectory

---

::: phydrax.bubble_dynamics.BubbleDynamicsEvidence

---

::: phydrax.bubble_dynamics.BubbleRegimeTape

---

::: phydrax.bubble_dynamics.BubbleDynamicsStatus

---

::: phydrax.bubble_dynamics.BubbleEventKind

---

::: phydrax.bubble_dynamics.bubble_status_successful

## Validity

::: phydrax.bubble_dynamics.BubbleValidityPolicy

---

::: phydrax.bubble_dynamics.BubbleValidityEvidence

---

::: phydrax.bubble_dynamics.neglected_terms

---

::: phydrax.bubble_dynamics.BOLTZMANN_CONSTANT

## Linear response

::: phydrax.bubble_dynamics.linear_bubble_response

---

::: phydrax.bubble_dynamics.LinearBubbleResponse

---

::: phydrax.bubble_dynamics.LinearBubbleResponseEvidence

---

::: phydrax.bubble_dynamics.minnaert_angular_frequency

---

::: phydrax.bubble_dynamics.prosperetti_polytropic_index

## Dissolution and surface nanobubbles

::: phydrax.bubble_dynamics.GasSolutionProperties

---

`EpsteinPlessetRoute` is the closed selector `"quasi_static" | "full_history"`.

---

::: phydrax.bubble_dynamics.EpsteinPlessetPlan

---

::: phydrax.bubble_dynamics.PreparedEpsteinPlesset

---

::: phydrax.bubble_dynamics.solve_epstein_plesset

---

::: phydrax.bubble_dynamics.GasDissolutionState

---

::: phydrax.bubble_dynamics.GasDissolutionResult

---

::: phydrax.bubble_dynamics.DissolutionEvidence

---

::: phydrax.bubble_dynamics.quasi_static_dissolution_time

---

`SurfaceBubbleContact` is the closed selector `"pinned" | "unpinned"`.

---

::: phydrax.bubble_dynamics.PinnedSurfaceBubblePlan

---

::: phydrax.bubble_dynamics.PreparedSurfaceBubble

---

::: phydrax.bubble_dynamics.solve_surface_bubble

---

::: phydrax.bubble_dynamics.SurfaceBubbleState

---

::: phydrax.bubble_dynamics.SurfaceBubbleEquilibrium

---

::: phydrax.bubble_dynamics.SurfaceBubbleResult

---

::: phydrax.bubble_dynamics.SurfaceBubbleEvidence

---

::: phydrax.bubble_dynamics.popov_flux_factor

## Bubble clouds

::: phydrax.bubble_dynamics.BubbleSpeciesGroup

---

`BubbleCloudRoute` is the closed selector `"auto" | "dense" | "fmm"`.

---

`BubbleCloudCoupling` is the closed selector `"incompressible" | "retarded"`.

---

`BubbleCloudPressureFieldKind` is the closed selector `"uniform" | "standing_wave"`.

---

::: phydrax.bubble_dynamics.BubbleCloudPressureField

---

::: phydrax.bubble_dynamics.BubbleTranslation

---

::: phydrax.bubble_dynamics.BubbleCloudResourcePolicy

---

::: phydrax.bubble_dynamics.BubbleCloudPlan

---

::: phydrax.bubble_dynamics.PreparedBubbleCloud

---

::: phydrax.bubble_dynamics.BubbleCloudState

---

::: phydrax.bubble_dynamics.BubbleCloudRates

---

::: phydrax.bubble_dynamics.solve_bubble_cloud

---

::: phydrax.bubble_dynamics.BubbleCloudResult

---

::: phydrax.bubble_dynamics.BubbleCloudTrajectory

---

::: phydrax.bubble_dynamics.BubbleCloudEvidence

---

::: phydrax.bubble_dynamics.mean_bjerknes_forces

---

::: phydrax.bubble_dynamics.BjerknesForceResult

## Far-field emission

::: phydrax.bubble_dynamics.FarFieldEmissionPlan

---

::: phydrax.bubble_dynamics.FarFieldEmissionResult

---

::: phydrax.bubble_dynamics.FarFieldEmissionEvidence

## Reference solutions and profiles

::: phydrax.bubble_dynamics.rayleigh_collapse_time

---

::: phydrax.bubble_dynamics.quasi_static_dissolution_radius

---

::: phydrax.bubble_dynamics.bubble_dynamics_candidate_profiles

---

::: phydrax.bubble_dynamics.bubble_cloud_candidate_profiles
