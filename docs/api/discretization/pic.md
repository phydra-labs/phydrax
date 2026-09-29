# Particle-in-cell methods

## Charged particle support

::: phydrax.discretization.ChargedParticlePlan

---

::: phydrax.discretization.PreparedChargedParticles

## Spatial transfer and current

::: phydrax.discretization.pic
    options:
      show_root_heading: false
      members_order: source

## Quasi-cylindrical azimuthal transfer

`QuasiCylindricalGrid` is the cell-centered radial Hankel grid and periodic axial
grid of the quasi-cylindrical PSATD solver. `AzimuthalTransferPlan` deposits and
gathers azimuthal modes with `e^{−ipθ}` weights, mirror folding, and
near-axis-corrected volumes for shape orders 1–3.

::: phydrax.discretization.pic.QuasiCylindricalGrid

---

::: phydrax.discretization.pic.AzimuthalTransferPlan

---

::: phydrax.discretization.pic.PreparedAzimuthalTransfer

## Curved cell location

::: phydrax.discretization.SimplicialLocationPolicy

---

::: phydrax.discretization.AbstractCellLocator

---

::: phydrax.discretization.PreparedSimplicialCellLocator

---

::: phydrax.discretization.CellLocationResult


## Compatible electrostatic and PIC runtimes

::: phydrax.solver.CochainElectrostaticPlan

---

::: phydrax.solver.CochainElectrostaticResult

---

::: phydrax.solver.ElectrostaticPICPlan

---

::: phydrax.solver.ElectrostaticPICState

---

::: phydrax.solver.ElectrostaticPICStepResult

---

::: phydrax.solver.ElectromagneticPICPlan

---

::: phydrax.solver.ElectromagneticPICState

---

::: phydrax.solver.ElectromagneticPICStepResult

---

::: phydrax.solver.ElectromagneticPICDiagnostics

---

::: phydrax.solver.ElectromagneticPICFixedStepMethod

---

::: phydrax.solver.PICRestartCheckpoint

---

::: phydrax.solver.PICFieldHistory

## Distributed PIC and restart

::: phydrax.solver.distribute_pic_field_solver

---

::: phydrax.solver.AbstractDistributedPICFieldSolver

---

::: phydrax.solver.pic_distribution_support

---

::: phydrax.solver.PICDistributionSupport

---

::: phydrax.solver.DistributedElectromagneticPICPlan

---

::: phydrax.solver.DistributedPICStepResult

---

::: phydrax.solver.DistributedPICExecutor

---

::: phydrax.solver.PICDistributedEvidence

---

::: phydrax.solver.PICDistributedRoute

---

::: phydrax.discretization.pic.PICDomainDecomposition

---

::: phydrax.discretization.pic.PICMigrationPlan

---

::: phydrax.discretization.pic.PICMigrationEvidence

---

::: phydrax.discretization.pic.PICSlotGroup

---

::: phydrax.discretization.pic.PICGuardWindow

---

::: phydrax.discretization.pic.PICIdentityAllocator

---

::: phydrax.discretization.pic.PICDistributedProcess

---

::: phydrax.discretization.pic.PICProcessStatePartition

---

::: phydrax.discretization.pic.PICProcessBank

---

::: phydrax.discretization.pic.AbstractPICParticleAllocator

---

::: phydrax.discretization.pic.allocate_particles

---

::: phydrax.discretization.pic.AbstractPICParticleExecutor

---

::: phydrax.discretization.pic.PICParticleExchangeResult

---

::: phydrax.solver.PICRestartPlan

---

::: phydrax.solver.PICRestartManifest

---

::: phydrax.solver.PICRestartResult

## PIC field-solver protocol

::: phydrax.solver.AbstractPreparedPICFieldSolver

---

::: phydrax.solver.CochainMaxwellPICFieldSolver

---

::: phydrax.solver.ReducedMaxwellPICFieldSolver

---

::: phydrax.solver.UnstructuredMaxwellPICFieldSolver

---

::: phydrax.solver.PICFieldDeposit

---

::: phydrax.solver.PICFieldSample

---

::: phydrax.solver.PICFieldAdvance

---

::: phydrax.solver.PICPrecisionPolicy

---

::: phydrax.solver.AbstractPICFieldFilter

---

::: phydrax.solver.PICFilterPlan

---

::: phydrax.solver.PICFilterContinuityReport

---

::: phydrax.solver.PICTensorLayout

---

::: phydrax.solver.PICCherenkovGuard

---

::: phydrax.solver.PICSpectralSymbol

---

::: phydrax.solver.PICHuygensSampling

---

::: phydrax.solver.PICMultiDeposit

---

::: phydrax.solver.PICWindowShift

---

::: phydrax.solver.PICGaussProjection

---

::: phydrax.solver.PICGaussProjectionResult

---

::: phydrax.solver.PICRestartState

---

::: phydrax.solver.PICRestartComponent

---

::: phydrax.solver.PICEnergyAccounting

---

::: phydrax.solver.PICOpenDomain

---

::: phydrax.solver.PICFieldSolverCapability

---

::: phydrax.solver.PICCapabilityRecord

---

::: phydrax.solver.PICFieldSolverCapabilities

## PIC field-solver state handoff

::: phydrax.solver.hand_off_pic_state

---

::: phydrax.solver.PICFieldHandoffResult

---

::: phydrax.solver.PICFieldHandoffEvidence

---

::: phydrax.solver.PICFieldHandoffRoute

## Moving window

::: phydrax.solver.PICMovingWindowPlan

---

::: phydrax.solver.PICMovingWindowState

---

::: phydrax.solver.PICMovingWindowResult

---

::: phydrax.solver.PICWindowInjection

## External PIC oracles

Pinned WarpX, Smilei, and PIConGPU runs of one `ElectromagneticPICPlan`; see the
particle-in-cell guide.

::: phydrax.solver.PICOracleScenario

---

::: phydrax.solver.PICOracleCode

---

::: phydrax.solver.pic_oracle_case

---

::: phydrax.solver.PICOracleCase

---

::: phydrax.solver.PICOracleSpecies

---

::: phydrax.solver.PICOracleSpectralSolver

---

::: phydrax.solver.PICOracleFields

---

::: phydrax.solver.PICOracleResult

---

::: phydrax.solver.WarpXProvider

---

::: phydrax.solver.warpx_input

---

::: phydrax.solver.run_warpx

---

::: phydrax.solver.read_pic_oracle_openpmd

---

::: phydrax.solver.PICOracleTrack

---

::: phydrax.solver.PICOracleTrackResult

---

::: phydrax.solver.run_warpx_track

---

::: phydrax.solver.read_pic_oracle_track

---

::: phydrax.solver.PICOracleLaser

---

::: phydrax.solver.PICOracleWakefieldCase

---

::: phydrax.solver.pic_oracle_wakefield_case

---

::: phydrax.solver.PICOracleModalFields

---

::: phydrax.solver.PICOracleWakefieldResult

---

::: phydrax.solver.warpx_preroll_steps

---

::: phydrax.solver.run_warpx_wakefield

---

::: phydrax.solver.read_warpx_wakefield

---

::: phydrax.solver.SmileiProvider

---

::: phydrax.solver.smilei_input

---

::: phydrax.solver.run_smilei

---

::: phydrax.solver.read_smilei_fields

---

::: phydrax.solver.PIConGPUProvider

---

::: phydrax.solver.picongpu_input

---

::: phydrax.solver.run_picongpu
