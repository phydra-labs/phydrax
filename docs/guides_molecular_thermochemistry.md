# Molecular vibrational analysis and thermochemistry

Molecular thermochemistry begins from one exact geometry, one electronic energy,
one Hessian, isotope-resolved masses, and one `AtomisticUnitSystem`. It is not a
fit to continuum species calorics and does not silently create NASA polynomial
coefficients.

## Stationary-point admission

Run `MolecularHessianPlan` and `VibrationalAnalysisPlan` first. Vibrational
analysis reports the raw projected curvatures and classifies the point as a
minimum, first-order saddle, higher-order saddle, or inconclusive.

- Atomic systems remove three translations.
- Linear molecules remove three translations and two rotations.
- Nonlinear molecules remove three translations and three rotations.

The inertia/SVD rank and external-mode projection residual are retained. A
numerically small negative curvature may be treated as nonsignificant only by
the explicit `imaginary_wavenumber_threshold`; its raw value remains available.

## RRHO inputs

`HarmonicThermochemistryPlan` requires:

- temperature in the system temperature unit;
- pressure in the system pressure unit;
- rotational symmetry number;
- electronic degeneracy;
- a CODATA-2018 SI-referenced atomistic unit system.

Symmetry and electronic degeneracy are never inferred from names, formulas, or
spin multiplicity. Strict RRHO requires every retained vibrational mode to have
positive frequency. No low-frequency replacement or quasi-RRHO policy is
implicitly applied.

The result stores electronic, translational, rotational, and vibrational
internal-energy and entropy components. Enthalpy includes the ideal-gas pressure
volume contribution. Gibbs energy is evaluated from the same temperature and
entropy realization.

## Per-system and molar values

`MolecularThermochemistryResult` remains in ordinary single-system energy and
energy/temperature units. Convert explicitly:

```text
molar = phx.chemistry.to_molar_thermochemistry(
    result,
    phx.units.KILOJOULE_PER_MOLE,
)
```

The conversion records both source result and target unit identities. Ordinary
unit conversion still rejects energy versus energy/amount because Avogadro
semantics are not an ordinary scale conversion.

## Binding to reaction kinetics

The current molecular result is not automatically inserted into
`ChemicalSpeciesSchema`. A future explicit species binding must verify elemental
composition, total charge, phase standard state, conformer choice, and model
chemistry before it can lower molecular partition functions into mechanism
thermodynamics.
