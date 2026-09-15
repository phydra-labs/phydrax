# Native H2: RHF, MP2, TDHF, and a Voigt spectrum

This bounded recipe uses the native general-Gaussian, molecular mean-field,
correlation, excited-manifold, and spectroscopy paths. It performs no external
program call.

```text
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
