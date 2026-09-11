# Optimize water and compute RRHO thermochemistry

This recipe requires the optional PySCF provider. It performs no molecular
lookup or download.

```text
import numpy as np
import phydrax as phx

units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
water = phx.atomistic.AtomicStructure(
    [8, 1, 1],
    [[0.0, 0.0, 0.0], [0.98, 0.02, 0.0], [-0.25, 0.95, 0.03]],
    [15.999, 1.008, 1.008],
    units.scale,
    particle_ids=[100, 101, 102],
    name="water",
)
system = phx.atomistic.AtomisticSystemPlan.from_structure(
    water,
    units,
    molecule_ids=[0, 0, 0],
)
state = phx.chemistry.MolecularElectronicStatePlan(0, 1)
method = phx.chemistry.ElectronicMethodPlan(
    phx.chemistry.ElectronicMethodFamily.KOHN_SHAM_DFT,
    "b3lyp",
    phx.chemistry.ElectronicReferenceKind.RESTRICTED,
)
model = phx.chemistry.ElectronicModelChemistryPlan(
    method,
    basis=phx.chemistry.BasisSetReference("6-31g", "pyscf-basis-library"),
)
force_calculation = phx.chemistry.ElectronicCalculationPlan(
    system,
    state,
    model,
    phx.chemistry.ElectronicPropertyRequest.energy_and_forces(),
)
provider = phx.chemistry.interchange.PySCFProvider(
    convergence_tolerance=1e-10,
    maximum_cycles=100,
)
prepared_force = force_calculation.prepare(provider)
surface = phx.chemistry.ElectronicPotentialEnergySurface(prepared_force)

optimized = phx.chemistry.MolecularGeometryOptimizationPlan(
    system,
    surface,
    convergence=phx.chemistry.MolecularGeometryConvergencePlan(
        maximum_force=1e-4,
        rms_force=5e-5,
    ),
).run(water)
if not bool(optimized.successful):
    raise RuntimeError("Water geometry did not satisfy every convergence gate")

hessian = phx.chemistry.MolecularHessianPlan(
    system,
    surface,
    displacement=1e-3,
).evaluate(optimized.final_structure)
vibration = phx.chemistry.VibrationalAnalysisPlan(system).evaluate(
    optimized.final_structure,
    hessian,
)
if vibration.stationary_point is not phx.chemistry.StationaryPointKind.MINIMUM:
    raise RuntimeError("Optimized water did not qualify as a minimum")

pressure = 101325.0 * float(
    phx.units.conversion_factor(phx.units.PASCAL, units.pressure_unit)
)
rrho = phx.chemistry.HarmonicThermochemistryPlan(
    system,
    298.15,
    pressure,
    symmetry_number=2,
    electronic_degeneracy=1,
).evaluate(
    optimized.final_structure,
    vibration,
    optimized.final_evaluation.energy,
)
molar = phx.chemistry.to_molar_thermochemistry(
    rrho,
    phx.units.KILOJOULE_PER_MOLE,
)
```

`vibration.wavenumbers` uses cm^-1. `rrho` remains per molecule; `molar` is the
explicit Avogadro conversion. Inspect all status and residual fields before
publishing either result.

To compute IR line strengths, prepare a second request containing energy,
forces, and dipole, then pass it to `IRSpectrumPlan` with the same system,
optimized structure, and qualified vibration result. Dipole finite differences
are independent electronic evaluations and use the plan's declared displacement.
