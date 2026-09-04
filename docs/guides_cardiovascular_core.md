# Cardiovascular core contracts

`phydrax.applications.cardiovascular` uses a fixed numeric kernel scale of millimetres,
milliseconds, milligrams, millivolts, kilopascals, and cubic millimetres. Runtime arrays
remain ordinary arrays: `CardiovascularQuantitySpec` is immutable host metadata, not a
second array, unit, solver, or archive framework.

## Quantity specifications

Every specification stores a semantic `quantity_kind` and one canonical
`UnitDefinition`, together with axes, sign convention, support association, and
reference configuration. Display symbols and the exact SI factor are computed
from that unit; they are not duplicated stored fields. Its `quantity_id` hashes
the unit ID and all remaining semantic content. The alias `spec_id` is provided
for runtime manifests that bind specification identities.

```python
from fractions import Fraction

from phydrax.applications import cardiovascular as cardio
from phydrax.units import MILLIVOLT

voltage = cardio.CardiovascularQuantitySpec(
    "paced_voltage",
    "electric_potential",
    MILLIVOLT,
    support_association="stimulus nodes",
    reference_configuration="extracellular potential",
)
assert voltage.to_si(Fraction(25)) == Fraction(1, 40)
assert voltage.from_si(voltage.to_si(Fraction(25))) == Fraction(25)

pressure = cardio.CARDIOVASCULAR_QUANTITIES["pressure"]
pressure_pa = pressure.to_si(13.3)  # 13.3 kPa -> 13300 Pa
```

The application accepts only exact multiplicative `UnitDefinition` values whose
dimension and reference system match the declared quantity kind. Textual units,
dimension mismatches, reference-system shifts, offsets, and logarithmic conversions
are rejected rather than parsed or guessed. `kernel_unit`, `si_unit`, and `si_factor`
remain computed display/conversion properties. Pass a `Fraction` when scalar
arithmetic itself must remain rational; NumPy and JAX arrays use their normal numeric
dtype.

The canonical catalog has the following exact factors, where
`SI value = kernel value * factor`:

| Quantity | Kernel unit | SI unit | Exact factor |
| --- | --- | --- | ---: |
| time | ms | s | 1/1000 |
| length | mm | m | 1/1000 |
| area | mm2 | m2 | 1/1000000 |
| volume | mm3 | m3 | 1/1000000000 |
| mass | mg | kg | 1/1000000 |
| transmembrane potential | mV | V | 1/1000 |
| electric field | mV/mm | V/m | 1 |
| membrane current | uA | A | 1/1000000 |
| membrane current density | uA/mm2 | A/m2 | 1 |
| membrane capacitance | uF | F | 1/1000000 |
| membrane capacitance density | uF/mm2 | F/m2 | 1 |
| electrical conductivity | mS/mm | S/m | 1 |
| chemical amount | mmol | mol | 1/1000 |
| species concentration | mM | mol/m3 | 1 |
| chemical diffusivity | mm2/ms | m2/s | 1/1000 |
| concentration rate | mM/ms | mol/(m3*s) | 1000 |
| molar surface flux | mmol/(mm2*ms) | mol/(m2*s) | 1000000 |
| pressure and stress | kPa | Pa | 1000 |
| velocity | mm/ms | m/s | 1 |
| acceleration | mm/ms2 | m/s2 | 1000 |
| mass density | mg/mm3 | kg/m3 | 1000 |
| force | mg*mm/ms2 | N | 1/1000 |
| strain | 1 | 1 | 1 |
| strain rate | 1/ms | 1/s | 1000 |
| energy | mg*mm2/ms2 | J | 1/1000000 |
| power | mg*mm2/ms3 | W | 1/1000 |
| dynamic viscosity | kPa*ms | Pa*s | 1 |
| volumetric flow rate | mm3/ms | m3/s | 1/1000000 |
| hydraulic resistance | kPa*ms/mm3 | Pa*s/m3 | 1000000000 |
| hydraulic inertance | kPa*ms2/mm3 | Pa*s2/m3 | 1000000 |
| hydraulic compliance | mm3/kPa | m3/Pa | 1/1000000000000 |
| hydraulic elastance | kPa/mm3 | Pa/m3 | 1000000000000 |

Do not infer a sign or frame from a unit. For example, transmembrane voltage is
intracellular minus extracellular potential, membrane current is outward-positive,
pressure is compression-positive, and circulation flow follows its terminal-port
orientation. The catalog records these conventions explicitly.

## Case manifests

`CardiovascularCaseManifest` is a host-only binding of immutable identities. It binds a
case to anatomy, model, protocol, observations, support profile, release, build, SBOM,
licenses, and data-rights records. It deliberately contains neither numerical solver
state nor a `solve` method, checkpoint behavior, archive encoding, or schema-version
field.

```python
case = cardio.CardiovascularCaseManifest(
    case_id="case-demo-001",
    anatomy_id="anatomy:sha256:001",
    model_id="model:sha256:002",
    protocol_id="protocol:sha256:003",
    support_profile_id="support:qualified-cpu",
    release_id="release:2026.09",
    build_id="build:sha256:004",
    sbom_id="sbom:sha256:005",
    observation_ids=("observation:ecg:006", "observation:pressure:007"),
    license_ids=("license:proprietary:008",),
    data_rights_ids=("rights:research:009",),
    metadata={
        "purpose": "research-use-only",
        "data_classification": "deidentified-aggregate",
    },
)
assert case.content_id == case.manifest_id
```

Identity collections are copied, checked for duplicates, and sorted before hashing.
Metadata is also copied and key-sorted. Consequently caller-owned mutable mappings do
not enter the record and semantically identical unordered inputs produce the same
`manifest_id`.

The metadata surface is intentionally narrow and fail-closed. Only the exported
`CARDIOVASCULAR_CASE_METADATA_KEYS` allowlist is accepted, and values must be bounded
canonical policy tokens. Patient, subject, participant, name, birth date, medical
record, address, email, phone, and similar person-linking fields are prohibited in both
identities and metadata. A case identity must be a non-person-derived technical ID;
pseudonymizing a clinical identifier does not make it suitable for this manifest.
Clinical linkage belongs in an access-controlled external system and must not be copied
into PhydraX manifests, logs, checkpoints, or artifacts.
