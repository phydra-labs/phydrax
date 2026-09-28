# Radiation-transport source boundaries

The native radiation implementation shares no upstream runtime or source code. External works define scientific processes, comparison problems, or workflow vocabulary.

| Source | Role | Native boundary |
|---|---|---|
| [LLNL RadSim](https://github.com/LLNL/RadSim) | Source → transport → detector workflow; MIT | No Java/JPype/provider orchestration. |
| [EGSnrc](https://github.com/nrc-cnrc/EGSnrc) | Photon/electron/positron process and condensed-history reference; AGPL-3.0-or-later | No source translation or linked runtime. The duplicate supplied reference is one source. |
| [MCGPU](https://github.com/DIDSR/MCGPU) | Voxel delta tracking, alias spectrum, scatter-class image, KERMA; US public domain | Independent fixed-capacity JAX implementation; no PENELOPE-derived tables. |
| [XRayMClib/OpenXRayMC](https://github.com/medicalphysics/XRayMClib) | Diagnostic X-ray sources, geometries, KERMA, detector workflow; GPL-3.0 | No C++/Qt/VTK/segmentation runtime or EPICS download. |
| [Eradiate](https://github.com/eradiate/eradiate) | Correlated-k, polarization, scene/sensor composition; LGPL-3.0 | No Mitsuba runtime, scene hierarchy, or bundled databases. |
| [SuperNu](https://github.com/lanl/SuperNu) | IMC/DDMC, census, thick-limit leakage; GPL-3.0 | No Fortran/runtime/atomic data; current native support is bounded one-dimensional packet coupling. |
| [OpenSn](https://github.com/Open-Sn/opensn) | Discrete-ordinates groupsets, sweeps, acceleration; MIT | No C++/PETSc runtime; landed support is one-dimensional slab S_n. |
| [LegPy](https://github.com/JaimeRosado/LegPy) | Educational low-energy cases; GPL-3.0 | Not a production oracle and no source/data reuse. |
| [NIST XCOM](https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html) | Photon photoelectric, coherent/incoherent, and nuclear/electron-field pair-production mass attenuation; NIST-authored United States public-domain data under 17 U.S.C. § 105 | No table is bundled or downloaded. Caller-admitted `DiagnosticPhotonCoefficientTable` artifacts retain source URL, digest, interpolation, units, and rights. |
| [NIST ESTAR](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html) | Electron stopping power and range; NIST-authored United States public-domain data under 17 U.S.C. § 105 | Provider capability only until exact caller-supplied rows and a pinned manifest are admitted; no values are reconstructed. |
| [Seltzer--Berger / NISTIR 4999](https://physics.nist.gov/PhysRefData/Star/Text/ref.html) | Differential bremsstrahlung spectra, 1 keV--10 GeV and Z=1--100; Seltzer & Berger, NIM B 12 (1985) 95--134 and ADNDT 35 (1986) 345--418; NISTIR 4999 (1993) | `tools/import_nist_seltzer_berger.py` verifies a pinned caller-supplied long-form CSV and emits a canonical artifact. No spectrum is bundled, scraped, inferred, or fabricated. NIST-authored data are United States public domain under 17 U.S.C. § 105; publication text retains its publisher rights. |
| Tsai, Rev. Mod. Phys. 46 (1974) 815 | Screened Bethe--Heitler pair/bremsstrahlung analytic reference, including erratum RMP 49 (1977) 421 | Equations independently implemented; no tables or source copied. |
| Migdal, Phys. Rev. 103 (1956) 1811; Ter-Mikaelian, JETP 25 (1953) 289 | LPM and dielectric suppression limits | Independent asymptotic factors; no source or provider data copied. |
| Garibian, JETP 6 (1958) 1079; Cherry & Müller, Phys. Rev. D 10 (1974) 3594 | Formation-zone interference and absorption in foil transition-radiation radiators | Independent coherent-interface implementation; no source copied. |
| Frank & Tamm, Dokl. Akad. Nauk SSSR 14 (1937) 107 | Cherenkov photon count and radiated energy over a declared wavelength band; dispersive spectral yield along linear-speed steps for optical-photon sources | Independent analytic integration. |
| Birks, Proc. Phys. Soc. A 64 (1951) 874 | Scintillation quenching `E / (1 + kB dE/dx)` of optical-photon sources | Independent formula; no provider data copied. |
| [Geant4](https://geant4.web.cern.ch/) | Optional shower/transition-radiation/Cherenkov comparison oracle; Geant4 Software License | Provider-only. No executable, data library, source, or generated oracle is vendored. Qualification skips only when a separately pinned executable/artifact is absent at the provider boundary. |

## Data rule

Cross sections enter through canonical `DiagnosticPhotonCoefficientTable` values with
`NuclearDataProvenance`; the prepared runtime retains all source identities and
combines mass attenuation only with caller-declared density. Stopping/scattering
powers, emission spectra, atmospheric coefficients, material maps, and external
benchmark outputs likewise require exact manifests and requested-use rights. Core
execution never downloads EPICS, NIST, PENELOPE, detector, or patient data.
Photon-local deposition is explicitly KERMA; absorbed-dose claims require a
separately qualified charged-particle profile.

`SeltzerBergerBremsstrahlungTable` refuses absent, incomplete, nonmonotone, or
rights-inadmissible data. The physical no-table route is the independently
implemented complete-screening Bethe--Heitler/Tsai spectrum, not placeholder
values. Atomic relaxation requires caller-supplied vacancy energies and yields;
no EADL/EPDL binding-energy or transition-probability data are bundled.
