# Nuclear fuel model sources and qualification boundaries

The implementation is independent. Published equations, coefficients, physical
definitions, and evaluated data are cited in the code that implements them.
BISON, FAST, FRAPCON, MATPRO, and SCIANTIX are references and comparison
targets; their solver frameworks and implementation text are not transplanted.

## Physical laws

| Owner / quantity | Authority and equation family | Important boundary |
|---|---|---|
| UO₂ enthalpy and heat capacity | J. K. Fink, *Thermophysical properties of uranium dioxide*, J. Nucl. Mater. 279 (2000), 1–18; equations/tables in [IAEA-TECDOC-1496](https://www-pub.iaea.org/MTCD/Publications/PDF/te_1496_web.pdf), §6.1.1.1 | Consistent enthalpy/derivative pair; 298.15 K reference; solid-phase domain |
| UO₂ conductivity | Modified NFI, [PNNL-19417 Rev. 2](https://fast.labworks.org/file-download/download/public/108), Eq. 2.19; FRAPCON-4.0 Eq. 2.52 | Fabricated density and rod-average burnup limits differ from runtime densification pores and local burnup; no double-counted bubble damage |
| Fuel expansion, elasticity, emissivity | PNNL-19417 and MATPRO NUREG/CR-6150 Vol. 4, FTHEXP/FELMOD/FPOIR/FEMISS | Formula/reference conventions and fitted ranges are carried by each law |
| Fuel densification | FRAPCON FUDENS, PNNL-19417 Eqs. 2.29/2.32 | Explicit resintering data; irreversible candidate update and local-root evidence; no invented sintering-data fallback |
| Solid-fission-product swelling | FRAPCON-4.0, PNNL-19418 Vol. 1 Rev. 2 §2.4; MatLib-1.2.1 PNNL-35702 Eq. 2-17 | Source volume strain and source burnup convention are explicit |
| Fuel relocation | PNNL-19418 Vol. 1 Rev. 2 Eq. 2.141 and §2.4 | Full thermal/gas envelope versus permanent rigid fraction; exact caller history and cold activation are not specified by the accessible algebraic equation |
| Fuel creep | MATPRO-11 NUREG/CR-0497, FCREEP, Table A-6.1 | Primary stress/age history included; source-listing disagreements with prose/printed activation energies documented in the owner |
| Zircaloy thermoelasticity and heat transport | MATPRO CELMOD/CSHEAR/CTHEXP/CTHCON/CCP, PNNL-19417 | Zircaloy-4 and SRA identity explicit; texture is not inferred from an isotropic modulus |
| Cladding creep | M. Limbäck and T. Andersson, ASTM STP 1295 (1996), 448–468; FRAPCON CREPR | Significant stress changes/reversals affect primary history; measured temperature/stress/flux ranges are not replaced by a temperature-only check |
| Cladding plasticity and growth | FRAPCON CKMN and Franklin growth model, PNNL material reports | Return-map and tangent evidence; volume-conserving principal growth convention |
| Oxidation/hydrogen | FRAPCON corrosion/pickup models, PNNL-19417/19418 | Cubic/linear transition integration, oxygen/metal balance, accumulated H mass versus derived ppm; no hydride failure prediction |
| Oxide emissivity | MATPRO ZOEMIS, NUREG/CR-6150 Eqs. 5-25–5-31 | Measurement temperature is instantaneous surface temperature; historical maximum only controls high-temperature degradation. Waterside oxide is not bore emissivity |
| Intragranular gas | Booth AECL-496 (1957); Speight, Nucl. Sci. Eng. 37 (1969), 180; Turnbull et al., J. Nucl. Mater. 107 (1982), 168 | Effective trapping/re-solution reduction; conservative unresolved modal inventory; rapid-transient assumptions remain visible |
| Grain-face bubbles, venting, swelling | G. Pastore et al., Nucl. Eng. Des. 256 (2013), 75–86; INL/CON-13-28389 Eqs. 1–17/Table I | One total nonlinear microstructure problem with passive species transport; no independent duplicate Kr/Xe bubble populations |
| Grain growth/sweeping | Ainscough et al., J. Nucl. Mater. 49 (1973), 117; INL/EXT-13-29930 Rev. 3 Eqs. 8.126–8.127 | Thermal growth law plus declared irradiation drag; primary Barani thesis Eq. A.7 uses 1 + 0.002 × burnup, not a conflicting modern-document coefficient |
| Transient burst release | T. Barani et al., J. Nucl. Mater. 486 (2017), 96–110, Eqs. 3–11; [accepted manuscript](https://www.osti.gov/pages/biblio/1372265) | Heating/cooling micro-cracking with healing; transferred gas and connected cavity volume are tracked separately |
| HBS formation and pore evolution | T. Barani et al., J. Nucl. Mater. 539 (2020), 152296; Part II, J. Nucl. Mater. 563 (2022), 153627, [DOI](https://doi.org/10.1016/j.jnucmat.2022.153627), [primary manuscript](https://re.public.polimi.it/retrieve/e0c31c12-87ff-4599-e053-1705fe0aef77/Journal_of_Nuclear_Materials_563_%282022%29_153627_1-9.pdf), Eqs. 1–18/Table 1 | Positive dimer-anchored Gauss–Radau moment closure, native four-variable root and conditioning evidence; gas hard-sphere volume is distinct from vacancy volume; closed HBS gas is not free gas |
| Helium diffusion | L. Luzzi et al., Nucl. Eng. Des. 330 (2018), 265–271, [DOI](https://doi.org/10.1016/j.nucengdes.2018.01.044) | Irradiated diffusivity fit approximately 973–1800 K |
| Helium solubility/trapping | L. Cognini et al., Nuclear Engineering and Technology 53 (2021), 562–571, [DOI](https://doi.org/10.1016/j.net.2020.07.009) | Henry data approximately 1073–1773 K; finite-pressure boundary and signed absorption, not xenon transport or a perfect sink |
| Gap transfer | MATPRO GTHCON/CMHARD and FRAPCON GAPCON, PNNL-19418 Eqs. 2.16–2.19 | Source rounded contact branches are not exactly continuous; coefficients are not changed to enforce an invented continuity guarantee |
| Gas accommodation | Ullman fits and NUREG/CR-0330 Garnier/Begej gap comparisons | Direct accommodation fit 500–1000 K versus integral gap comparison 293–873 K at about 0.103 MPa; cold rod-pressure reproduction is not qualified comparison |
| Radial power/composition | K. Lassmann et al., J. Nucl. Mater. 208 (1994), 223–231; FRAPCON-4.0 TUBRNP | Spectrum-collapsed scattering and radial-depth unit are declared inputs; six-nuclide energy and reaction fractions are distinct |
| Water thermodynamics | [IAPWS R7-97(2012)](https://iapws.org/public/documents/UWTF-/IF97-Rev.pdf), region 1/4 and backward T(p,h) | Region and saturation constraints carried separately from a rectangular support box |
| Water transport | [IAPWS viscosity release](https://iapws.org/technical-guidance/release/viscosity.download), R12-08; [thermal conductivity release](https://iapws.org/technical-guidance/release/ThCond.download), R15-11 | Industrial viscosity convention and industrial conductivity critical enhancement are source-faithful |
| PWR wall transfer | Dittus–Boelter (1930); Jens–Lottes ANL-4627 (1951); FRAPCON §2.3 | Original Fahrenheit/Btu/psia units converted consistently; source comparison is not CHF/dryout support |

### Printed-source discrepancies

The HBS pore moments are derived from the primary cluster generator. A printed
first-moment destruction term has inconsistent dimensions, and the printed
variance equation does not recover the constant-rate birth/destruction control.
The implementation therefore does not encode those typographical forms or
claim to implement the printed second-order Taylor truncation. Its explicit
positive two-node Gauss–Radau reconstruction on cluster sizes at least two
reproduces pore number, first gas moment, and integrated central variance.
Source capture/destruction rates are unchanged; the representation and node
count enter the model fingerprint. Part II Eq. 17 and its accompanying
packing-fraction definition give the gas atomic volume as
π × hard-sphere diameter³ / 6, distinct from the Schottky vacancy volume.
Root convergence, conditioning availability, positivity, flux balance, and the
resulting atom ledger remain independent acceptance requirements.

For the Radau representation, let `m = A/N - 2` and `v = B/N`. When
`m > 0`, the nodes are `2` and `2 + m + v/m`, with respective weights
`v/(m*m + v)` and `m*m/(m*m + v)`. Zero variance is monodisperse;
`m = v = 0` is the dimer-only limit. These are derived moment-preserving
coordinates, not fitted pore-distribution coefficients. The source dimer-birth
extent is bounded by its available two-atom monomer budget. The separate
source monodisperse impingement stage conserves gas and vacancy volume and
holds integrated central variance fixed.

The gas hard-sphere diameter in Eq. 17 is
`4.45e-10 * (0.8542 - 0.03996 * log(T/231.2))` m. With the accompanying
packing-fraction definition, it supplies `omega(T)` in
`V = omega(T)*n + Omega*nv` (Eq. 13);
`Omega = 4.09e-29 m³` is the separate Schottky volume from Table 1.
The retained HBS inventories are pore gas and uncaptured HBS boundary gas.
Neither is communicating rod gas, and neither is a normal-operation
HBS-to-free-volume release law.

FCREEP primary time constants and reset conventions differ between some printed
text and its subcode listing. The selected convention is identified explicitly.
No disagreement is concealed by relabeling one model as another.

## Evaluated nuclear data

`tools/extract_fission_gas_yields.py` regenerates
`phydrax/nuclear/data/fission_gas_yields.json` from ENDF/B-VIII.0 evaluated files.
The reader verifies the packaged byte count/digest through native reference
artifact admission. Per-system incident energy, MT459 cumulative isotope
selection, MT454 checks, MT458 energy definition, source files/digests, and
processing choices remain in the data record.

- Gas isotopes are explicitly selected stable/long-lived Kr and Xe products.
- Xe-129 is not obtained by pretending I-129 decays on a rod-irradiation time
  scale.
- U-238 uses its tabulated fast incident-energy point; fissile thermal points
  remain declared rather than silently becoming a spectrum average.
- MT458 recoverable energy excludes neutrinos and capture gamma heating; delayed
  components and the fixed-energy approximation are stated.

`phydrax.nuclear.HeliumProductionPlan` requires explicit yields for every
member of `FISSIONING_NUCLIDES`, a declared helium `NuclideKey`,
`NuclearDataProvenance`, a `NuclideInventory` template, and decay-only native
`InventoryTransition` values that actually produce that isotope. There is no
bundled universal ternary yield or inferred actinide alpha rate.
Ternary production is fission-rate density times the declared yield mixture;
decay production is the transition rate times helium product multiplicity
times parent inventory in moles times the exact Avogadro constant, divided
by the declared reference volume. `HeliumProductionRates` reports the two
contributions separately in atoms/m³/s. Native transition conservation,
inventory nuclide order, and network identity remain authoritative. Missing
half-lives or yields are not inferred from displayed element names or
equal-length arrays.

The coupled fuel bundle consumes that native plan with an explicitly declared
fixed decay inventory and reference volume. It requires ground-state He-4 and
matching inventory identities; it does not replace isotope-resolved production
with a universal local coefficient.

`tools.nuclear_fuel_qualification.numerical_helium_production()` declares a
synthetic 1 m³ reference volume containing 1 mol of U-238, a ledger-closed
U-238 → Th-234 + He-4 transition with rate `1e10 / AVOGADRO_PER_MOL` s⁻¹,
and a ternary yield of `1e-4` atoms/fission for each named fissioning nuclide.
Its constant alpha source is therefore `1e10` atoms/m³/s. These numbers preserve
the declared numerical fixture; they are **not evaluated physical decay data**,
fabrication inventory measurements, or a universal helium source. The synthetic
parameter artifact has explicit provenance and no external-use authorization.

## Source prerequisites, not fake defaults

The following are material/scientific inputs, not implementation placeholders:

1. Spectrum-averaged transport scattering per initial heavy-metal atom,
   including oxygen, or the original TUBRNP scattering datum. Bound scattering
   is not an equivalent substitute.
2. The radial form-depth unit. The accessible FRAPCON text does not establish
   the original numeric unit strongly enough to hide it as an inferred default.
3. Low-temperature helium diffusion/solubility data if the complete cold startup
   is to be source-qualified. A supplied transport table is validated against
   its own grid and provenance, not manufactured from high-temperature fits.
4. Bore-side gray-surface data; waterside oxide models do not provide it.
5. Through-life GTRLOC history/activation semantics. PNNL-31159 §1.1 describes
   registered FAST access; public algebraic equations do not supply the caller
   initialization/retention routines. Authorized source/caller access or an
   authoritative publication is needed for an exact FAST/FRAPCON reproduction
   claim. A declared native history/geometric translation remains a distinct
   model requiring qualification.

Numerical fixtures explicitly identify synthetic input data and equation
reproduction. They exercise the real application without becoming measurements,
external code outputs, or experimental validation.

### Native coupled coolant support

`CoupledChannelCoolant` consumes
`phydrax.applications.thermofluids.LiquidChannelHeatBalance`, not an
unqualified external success flag or a fuel-private thermal-fluid solver.
The native balance is prescribed-flow, reference-mass upwind enthalpy
transport with storage; wall exchange is paired with the rod on the same
current interface area, and direct coolant deposition is separate.
Water and wall-transfer equations retain the IAPWS and FRAPCON authorities
in the table above.

Its exact `thermofluids.channel-heat` support tuple binds the history,
wall-transfer law, cell count, IF97 region-1 water, and reference-mass-upwind
method. Admission uses native `require_profile` with a `ReleaseIndex`,
`ReleaseTrustPolicy`, exact profile ID, and declared admission time.
This independently trusted participant support neither supplies IFPE
experimental data rights nor releases the fuel capability. Momentum,
bulk two-phase flow, CHF, and dryout remain nonclaims.

## Experimental data and release

The reviewed public [NEA IFPE request procedure](https://oecd-nea.org/jcms/pl_37221/international-fuel-performance-experiments-ifpe-request)
describes individual/full-collection package requests, approval notification,
and delivery. The [NEA-1664 abstract](https://oecd-nea.org/tools/abstract/detail/nea-1664),
§9, identifies NEA-1664/03 as **Masterfiled restricted**, status dated
08-APR-2024. These pages establish access metadata, not a local originator
agreement. Public reports do not imply possession or redistribution rights for
complete power/coolant histories, sensor metadata, uncertainty, or held-out rods.

Release requires separate-effects controls, multiple integral observables,
uncertainty/covariance, and locked independent evaluation grouped by physical rod,
fabrication batch, and ancestry. Temperature alone cannot distinguish compensating
conductivity, initial-gap, relocation, and diffusivity errors. FGR, pressure,
diameter, elongation, oxide/hydrogen, and sensor-equivalent temperature must be
used according to the exact claim.

Scientific support, numerical convergence, physical admissibility, data rights,
resource evidence, and regulatory acceptance are separate. Current candidate
profiles contain no fabricated experimental or release evidence.

### Public-document technical review and source pins

The native source IDs are `nea-ifpe-access-metadata` and
`iaea-fumex-public-report`. They identify only the public artifacts actually
read on 2026-09-30. The no-`www` NEA host provided the authoritative abstract
and request-policy HTML; the `www` host returned HTTP 403 during this review.
No restricted IFPE package or experiment input/history was acquired.

The reviewed [FUMEX-III report, IAEA-TECDOC-1697 (2013)](https://www-pub.iaea.org/mtcd/publications/pdf/te-1697_cd/pdf/iaea-tecdoc-1697.pdf)
provides the comparison protocol: §1.2 describes blind predictions and
simplified cases, §2.2 describes comparison requirements, and §3.1.2 describes
an idealized case and uncertainty boundaries. Numerical V1–V15 requirements
cite this protocol, **not** FUMEX as the author of their analytic solutions or
as evidence that its experimental cases were run. Each requirement retains its
specific independent reference in its rationale: analytic cylinder, Kirchhoff
transform, Lamé closed-end tube, Rivlin tube, stress-free eigenstrain, Lamé
shrink fit, Booth sphere series, communicating-volume ideal gas,
concentric-cylinder heat flow, CREPR closed form/Norton tube, manufactured
independent forcing, independent reductions, energy/gas conservation,
accepted-checkpoint replay, and Richardson refinement, respectively.
Associating a reviewed source never resolves a missing numerical campaign,
data-rights gate, or locked evaluation.

The ledger pins the following actual byte content (SHA-256):

| Reviewed artifact / rights content | Bytes | Digest |
| --- | ---: | --- |
| Original NEA-1664 public abstract HTML | 48,969 | `06dd6311afdb172400eff4b869a6f9065c2da8bfe257633eea8b9fb1ce4308bf` |
| Original IFPE request-policy HTML | 84,006 | `0f1d930b166b713c4623578a45b131412f99a1ac708693c346b389a6cd57bf75` |
| Exact UTF-8 copyright excerpt from the request HTML | 22 | `9d161f36fc51bceccb5cd5c4c1fcbfb6ef72ede32fbe0f0abe932a21e4b5f865` |
| Original IAEA-TECDOC-1697 PDF | 5,041,075 | `e6efbf9cf532713743f4a1926fabdd78c06e833a672288e30bab81b5f069cd14` |
| Extracted PDF page 5 rights text, including layout and form feed | 2,582 | `9fbb783ab8f9789608c2f00cc3af415c838363c98bd0a1bedcd1ec8a3421fb52` |

`archive_digest` identifies the original abstract HTML or original PDF, not a
normalized metadata receipt, a source-code tree, or a restricted data archive.
The request HTML is independently pinned in `relevant_documents`. The IFPE
`license_digest` identifies the exact 22 UTF-8 bytes `Copyright © 2025 OECD`
(no newline) present in that HTML; it is a copyright notice, not a grant of
permission. The FUMEX rights digest identifies the full output of
`pdftotext -f 5 -l 5 -layout iaea-tecdoc-1697.pdf rights.txt`; this is a derived
text extraction, **not** the original PDF bytes. Its page-5 locator is pinned
alongside the original full report. Neither rights digest is a fabricated
license or a digest of a descriptive status string.

The report copyright page requires permission for use of publication text,
normally subject to royalty agreements. Both native records are
`TECHNICALLY_REVIEWED`, with `copying_permitted=False`, no code/behavior
inspection claim, and `legal_reviewer="unreviewed"`. This is a public-document
technical review only, not legal approval, release authorization, or permission
to redistribute report contents, benchmark decks, or experiment histories.
The actual originator agreement, authorized machine-readable histories with
uncertainty/sensor metadata, and locked ancestry/fabrication-grouped independent
evaluation remain external scientific-release blockers.


## Rights

No BISON/FAST/FRAPCON/SCIANTIX source text is translated into package code. A
publication/reference license does not automatically cover embedded data or a
benchmark deck. Packaged evaluated data retain exact source/processing and
artifact rights; experimental histories require their own originator agreement.
Any redistributed artifact is admitted through the native source/reference
manifest and rights review.
