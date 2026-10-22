# Molecular chemistry: typed energy, forces, PES, and native spectra

## Typed H2 energy, forces, and potential-energy surface

This complete example uses the published hydrogen STO-3G exponents and
contraction coefficients and a 0.74 Å H–H separation. It uses native exact
integrals by default; the optional DF continuation below changes the integral
definition explicitly. It reports computed quantities rather than supplying
reference energies or treating convergence as chemical-accuracy evidence.

```python
import numpy as np
from jax.typing import ArrayLike

import phydrax.atomistic as atomistic
import phydrax.chemistry as chemistry
import phydrax.operators.quantum.gaussian as gaussian

units = atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
structure = atomistic.AtomicStructure(
    np.asarray([1, 1], dtype=np.int32),
    np.asarray([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]], dtype=np.float64),
    np.asarray([1.008, 1.008], dtype=np.float64),
    units.scale,
    particle_ids=np.asarray([11, 17], dtype=np.int64),
)
system = atomistic.AtomisticSystemPlan.from_structure(
    structure, units, molecule_ids=[0, 0]
)
hydrogen_exponents = np.asarray(
    [3.42525091, 0.62391373, 0.16885540], dtype=np.float64
)
hydrogen_coefficients = np.asarray(
    [0.15432897, 0.53532814, 0.44463454], dtype=np.float64
)
basis = gaussian.GaussianBasisPlan.from_contracted_s(
    [11, 17],
    np.stack((hydrogen_exponents, hydrogen_exponents)),
    np.stack((hydrogen_coefficients, hydrogen_coefficients)),
    source_id="sto-3g-hydrogen",
).prepare(system)


def prepare_rhf(
    integrals: gaussian.MolecularIntegralPlan,
    task: chemistry.GroundStateTaskPlan,
) -> chemistry.PreparedMolecularRHFCalculation:
    plan = chemistry.MolecularHartreeFockPlan(
        system,
        basis,
        chemistry.MolecularElectronicSectorPlan(0, 1),
        chemistry.ElectronicReferenceKind.RESTRICTED,
        integrals=integrals,
        convergence=chemistry.SCFConvergencePlan(
            energy_tolerance=1.0e-10,
            density_tolerance=1.0e-8,
            commutator_tolerance=1.0e-8,
            maximum_iterations=128,
        ),
        acceleration=chemistry.SCFAccelerationPlan(
            (
                chemistry.SCFAccelerationKind.DAMPING,
                chemistry.SCFAccelerationKind.DIIS,
            ),
            damping=0.2,
        ),
    )
    model = chemistry.ElectronicModelChemistryPlan(
        chemistry.HartreeFockMethodPlan(),
        basis=chemistry.BasisSetReference(
            "STO-3G hydrogen", basis.plan.source_id, convention="cartesian"
        ),
    )
    provider = chemistry.MolecularRHFProvider(plan, model.model_chemistry_id)
    calculation = chemistry.ElectronicCalculationPlan(
        system,
        plan.sector,
        model,
        task,
        numerical=provider.numerical,
    )
    return provider.prepare(calculation)


def evaluate_forces(
    prepared: chemistry.PreparedMolecularRHFCalculation, positions: ArrayLike
) -> chemistry.ElectronicEnergyForceEvaluation:
    result = prepared.evaluate(positions)
    if not isinstance(result, chemistry.ElectronicEnergyForceEvaluation):
        raise TypeError("The prepared task must request energy and forces")
    if not bool(result.successful):
        raise RuntimeError("RHF convergence or stationary-force admission failed")
    return result


exact = gaussian.MolecularIntegralPlan()
energy_calculation = prepare_rhf(exact, chemistry.GroundStateTaskPlan.energy())
energy_result = energy_calculation.evaluate(structure.positions)
if not isinstance(energy_result, chemistry.ElectronicEnergyEvaluation):
    raise TypeError("Expected an energy-only result")
if not bool(energy_result.successful):
    raise RuntimeError("RHF energy convergence evidence failed")

force_calculation = prepare_rhf(
    exact, chemistry.GroundStateTaskPlan.energy_and_forces()
)
force_result = evaluate_forces(force_calculation, structure.positions)
print("RHF energy [eV]:", energy_result.energy)
print("RHF forces [eV/angstrom]:", force_result.forces)

surface = chemistry.ElectronicPotentialEnergySurface(force_calculation)
stretched_positions = np.asarray(structure.positions).copy()
stretched_positions[0, 2] -= 0.01
stretched_positions[1, 2] += 0.01
surface_result = surface.evaluate(stretched_positions)
if not bool(surface_result.successful):
    raise RuntimeError("The stretched-geometry PES evaluation failed")
print("Stretched H2 energy [eV]:", surface_result.energy)
print("Stretched H2 forces [eV/angstrom]:", surface_result.forces)
```

`ElectronicEnergyEvaluation`, `ElectronicEnergyForceEvaluation`, and
`PotentialEnergySurfaceEvaluation` are distinct typed results. Preserve the
electronic result header's source units, geometry, scientific artifact IDs,
numerical identity, convergence status, and work evidence. The PES keeps the
source result identity. These calls are host-orchestrated, not an AD or
JIT-SCF interface. Scalar `bool`/printing above deliberately inspect results at
the host boundary.

### Optional cuEST DF continuation

First install `pip install 'phydrax[cuda13,cuest]'` and follow
[installation](../installation.md): installing the optional CUDA/cuEST runtime
wheels is not the same as installing/building the PhydraX native extension or
qualifying a device/runtime.

This continuation is self-contained after the preceding example. Its explicit
custom auxiliary basis contains uncontracted normalized spherical s functions
with all same-center STO-3G primitive-product exponents
`alpha_i + alpha_j`. These numbers are derived from the actual orbital
exponents above, not a mislabeled published fitting basis. This auxiliary space
does **not** include all intercenter AO products and is not a production-quality
or complete auxiliary-basis claim. A chemical-accuracy study must choose and
record an appropriate independently sourced fitting basis and compare against
matched references.

```python
product_exponents = tuple(
    float(hydrogen_exponents[left] + hydrogen_exponents[right])
    for left in range(hydrogen_exponents.size)
    for right in range(left, hydrogen_exponents.size)
)
auxiliary_basis = gaussian.GaussianBasisPlan(
    tuple(
        gaussian.GaussianShellPlan(
            center,
            0,
            np.asarray([exponent], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            representation=gaussian.GaussianShellRepresentation.REAL_SPHERICAL,
            role="density-fitting",
        )
        for center in (11, 17)
        for exponent in product_exponents
    ),
    maximum_basis_functions=12,
    source_id="hydrogen-sto-3g-same-center-primitive-products",
    role="density-fitting",
).prepare(system)
cuest_source = gaussian.CuestIntegralPlan(
    auxiliary_basis,
    fitting_cutoff=1.0e-12,
    relative_conditioning=False,
    fitting_algorithm="qr",
    screening_threshold=0.0,
    variable_workspace_bytes=64 * 1024 * 1024,
    maximum_workspace_bytes=512 * 1024 * 1024,
    maximum_cached_geometries=1,
    density_factorization_tolerance=1.0e-12,
)
df_calculation = prepare_rhf(
    gaussian.MolecularIntegralPlan(cuest_source),
    chemistry.GroundStateTaskPlan.energy_and_forces(),
)
df_result = evaluate_forces(df_calculation, structure.positions)
df_surface = chemistry.ElectronicPotentialEnergySurface(df_calculation)
df_stretched = df_surface.evaluate(stretched_positions)
if not bool(df_stretched.successful):
    raise RuntimeError("DF-RHF stretched-geometry stationary forces failed")
print("DF-RHF energy [eV]:", df_result.energy)
print("DF-RHF forces [eV/angstrom]:", df_result.forces)
print("DF-RHF stretched energy [eV]:", df_stretched.energy)
```

The metric cutoff, absolute conditioning, QR algorithm, zero screening,
native-FP64 mode, density-factorization tolerance, and resource budgets are
explicit choices, not universal recommended settings. Zero screening does not
make DF exact. The metric rank is not exposed; a cutoff is not a rank or
energy-error certificate. Workspace bounds do not include all JAX,
vendor-private, driver, or JIT/process memory.

The same DF definition supplies energy, J/K, and first contracted analytic
nuclear forces. PhydraX retains SCF ownership; exchange uses the actual damped
density, and forces include Pulay/nuclear terms and stationary-density/orbital
checks. This does not authorize arbitrary AD, basis derivatives, Hessians,
CPHF/CPKS, orbital-Hessian stability, or response/correlation under substituted
exact integrals. The provider refuses unsupported tasks/contexts and numerical
definitions.

For direct integral reuse, `cuest_source.prepare(basis).bind_geometry(...)`
accepts FP64 bohr positions and nuclear charges. Reuse that bound object for
one-electron, actual-density J/K, and contracted derivatives. Use
`bound.refresh(positions=..., charges=...)` for changed data or
`bound.restore()` for same-content restored arrays/key. Changed source contents
require preparation again; raw tree replacement is not authorization.
Bindings are ephemeral and not persisted in SCF states. Runtime component
release clears native memos, and the same valid dynamic recipe rebuilds them on
next use; there is no bound-object `release()` method. See
[the lifecycle contract](../guides_computational_chemistry.md#rank-deficient-diis-and-geometry-lifecycle).
These examples make no GPU-performance or release-promotion claim.

## Native H2: RHF, MP2, TDHF, and a Voigt spectrum

This separate bounded recipe uses native exact general-Gaussian, molecular
mean-field, correlation, excited-manifold, and spectroscopy paths without an
external program call. Keep the exact definition for the response/correlation
steps; they must not be reused with a cuEST/fixed/direct state by silently
replacing its integrals.

```python
import numpy as np
import phydrax as phx

units = phx.atomistic.AtomisticUnitSystem.electronvolt_angstrom_dalton_femtosecond()
structure = phx.atomistic.AtomicStructure(
    [1, 1],
    [[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
    [1.008, 1.008],
    units.scale,
    particle_ids=[11, 17],
)
system = phx.atomistic.AtomisticSystemPlan.from_structure(
    structure,
    units,
    molecule_ids=[0, 0],
)

exponents = [3.42525091, 0.62391373, 0.16885540]
coefficients = [0.15432897, 0.53532814, 0.44463454]
basis = phx.operators.quantum.gaussian.GaussianBasisPlan.from_contracted_s(
    [11, 17],
    [exponents, exponents],
    [coefficients, coefficients],
    source_id="sto-3g-hydrogen",
).prepare(system)
positions_bohr = structure.positions * float(
    phx.units.conversion_factor(units.scale.length_unit, phx.units.BOHR)
)

rhf = phx.chemistry.MolecularHartreeFockPlan(
    system,
    basis,
    phx.chemistry.MolecularElectronicSectorPlan(0, 1),
    phx.chemistry.ElectronicReferenceKind.RESTRICTED,
    acceleration=phx.chemistry.SCFAccelerationPlan(
        (
            phx.chemistry.SCFAccelerationKind.DAMPING,
            phx.chemistry.SCFAccelerationKind.DIIS,
        ),
        level_shift=0.1,
    ),
)
state = rhf.solve_atomic_units(positions_bohr)
if not bool(state.evidence.successful):
    raise RuntimeError("RHF convergence evidence failed")

gradient = rhf.analytic_gradient_atomic_units(positions_bohr, state)
response = phx.chemistry.MeanFieldResponsePlan().electric_response(
    rhf,
    positions_bohr,
    state,
)
if not bool(gradient.successful) or not bool(response.successful):
    raise RuntimeError("RHF derivative or CPHF evidence failed")

correlation = phx.chemistry.electronic_structure.correlation
partition = correlation.CorrelatedOrbitalPartition.from_restricted_state(state)
store = correlation.MolecularIntegralTransformationPlan().transform_restricted(
    basis,
    positions_bohr,
    [1.0, 1.0],
    state,
    partition,
)
mp2 = correlation.MP2Plan().evaluate(store)
if not bool(mp2.successful):
    raise RuntimeError("MP2 denominator or finite-value evidence failed")

excited_response = phx.chemistry.HartreeFockExcitedResponsePlan(
    rhf,
    state,
    positions_bohr,
    spin_sector="singlet",
)
manifold = excited_response.tdhf(1).solve()
if not bool(manifold.successful):
    raise RuntimeError("TDHF residual or symplectic normalization failed")

line_position = phx.chemistry.transition_energy_axis(
    manifold.excitation_energies,
    manifold.energy_unit,
    phx.chemistry.SpectralAxis.ENERGY_EV,
)
spectrum = phx.chemistry.SpectralProfilePlan(
    phx.chemistry.SpectralLineShape.VOIGT,
    0.1,
    40.0,
    grid_size=8001,
    fwhm=0.2,
    lorentzian_fwhm=0.05,
).evaluate(line_position, manifold.oscillator_strengths)
if not bool(spectrum.successful):
    raise RuntimeError("Finite-grid oscillator-strength area did not close")
```

`state.total_energy`, `mp2.correlation_energy`, and the TDHF excitation energy
are in Hartree. `gradient.gradient` is the positive energy derivative;
`gradient.forces` is its negative. The CPHF result includes residuals and
condition estimates. The excited result stores an `RPAStateRepresentation`, not
a TDA-shaped amplitude alias. The Voigt result is oscillator-strength density on
an eV axis and retains its integrated-area residual.

For CCSD/CCSD(T), pass `store` to
`interchange.PySCFCoupledClusterProvider`. Preserve `result.checkpoint()` to
restart the same provider/plan/store identity. Molecular analytic
CCSD/CCSD(T) gradients use the separately configured optional molecular
provider because geometry and derivative-integral semantics are not recoverable
from an MO tensor store alone.
