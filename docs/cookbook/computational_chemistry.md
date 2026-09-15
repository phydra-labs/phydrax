# External DFT geometry, vibrations, thermochemistry, and IR

This recipe uses an optional host provider while keeping system, method, task,
units, results, and downstream workflows native.

```text
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
sector = phx.chemistry.MolecularElectronicSectorPlan(0, 1)
functional = phx.chemistry.DensityFunctionalPlan.b3lyp()
method = phx.chemistry.KohnShamMethodPlan(
    functional,
    phx.chemistry.ElectronicReferenceKind.RESTRICTED,
)
model = phx.chemistry.ElectronicModelChemistryPlan(
    method,
    basis=phx.chemistry.BasisSetReference("6-31g", "provider-basis-library"),
)
task = phx.chemistry.GroundStateTaskPlan(
    (
        phx.chemistry.ElectronicProperty.ENERGY,
        phx.chemistry.ElectronicProperty.FORCES,
        phx.chemistry.ElectronicProperty.DIPOLE,
    )
)
calculation = phx.chemistry.ElectronicCalculationPlan(
    system,
    sector,
    model,
    task,
)
provider = phx.chemistry.interchange.PySCFProvider(
    convergence_tolerance=1e-10,
    maximum_cycles=100,
)
prepared = calculation.prepare(provider)
surface = phx.chemistry.ElectronicPotentialEnergySurface(prepared)

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

infrared = phx.chemistry.IRSpectrumPlan(
    system,
    prepared,
    line_shape=phx.chemistry.SpectralLineShape.VOIGT,
).evaluate(optimized.final_structure, vibration)
if not bool(infrared.successful):
    raise RuntimeError("IR displacement or finite-grid area evidence failed")
```

The method plan describes B3LYP; it does not imply native B3LYP execution. The
selected provider must declare the matching method family, reference, task,
geometry, and environment support. `vibration.wavenumbers` uses cm^-1. `rrho`
remains per molecule; `molar` is the explicit Avogadro conversion. IR dipole
finite differences are independent provider evaluations and retain all source
result IDs.
