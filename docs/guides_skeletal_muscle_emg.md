# Skeletal surface electromyography

Phydrax exposes distinct, source-bounded surface-EMG observation fidelities.
None accepts activation, force, calcium, or recruited-unit count as a voltage
waveform. The source-current and cylindrical families below are independent
numerical implementations, not a claim of individualized anatomy or iEMG.

## Explicit MUAP template superposition

`MotorUnitActionPotentialTemplatePlan` accepts a caller-supplied voltage template
bank with shape `(motor_unit, channel, sample)`, an explicit sample period and zero
index, unique unit/channel IDs, and a provenance ID. Runtime event times and masks
have fixed `(motor_unit, event_slot)` shape. Linear fractional-delay interpolation
and superposition are differentiable only with respect to continuous template/sample
values while event indices and masks remain fixed.

This is the exact event-to-MUAP-train boundary used by the Fuglevand–Winter–Patla
lineage. It is not labeled as a complete FWP waveform generator unless the supplied
templates themselves come from a licensed, source-pinned FWP/Fuglevand-1992 model.
Masked events contribute exactly zero; an active event whose template support never
intersects the output grid fails completeness evidence.

## Petersen–Rostalski planar conductor

`PetersenRostalski2019PlanarConductorPlan` implements the Fourier-domain transfer
of Petersen and Rostalski 2019, DOI
[`10.3389/fphys.2019.00176`](https://doi.org/10.3389/fphys.2019.00176),
for an infinite planar anisotropic muscle layer under isotropic fat and skin. Inputs
are a charge-neutral discrete source-current spectrum, conductivities in S/m, layer
thickness/depth in m, a supplied single-electrode transfer, and a charge-neutral
spatial electrode montage. Output is surface potential in V under the declared FFT
normalization.

The zero mode is removed, the source and montage must be neutral, and the inverse
FFT must be real within the declared tolerance. The transfer reproduces source-depth
attenuation. It does not model intramuscular electrodes, cylindrical limbs, arbitrary
anatomy, fatigue, time-varying geometry, or a fiber current generator. The associated
Dryad source is DOI `10.5061/dryad.326qs26`; the GPL `semgsim` implementation is an
external behavioral oracle and no code is copied.

Run:

```text
python examples/skeletal_muscle_emg.py
python tools/skeletal_muscle_emg_qualification.py
python benchmarks/skeletal_muscle_emg.py
```

## Committed fiber-current owner

`PereiraBotelho2019FiberCurrentPlan` independently implements the core-conductor
relation in Pereira Botelho, Curran and Lowery (2019),
[Eq. 3, DOI `10.1371/journal.pcbi.1007267`](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007267):
outward line current equals intracellular conductivity times fiber cross-sectional
area times the second spatial derivative of membrane voltage. This is the
core-conductor observation approximation, **not** an extraction of ionic current
or a bidomain feedback solve. Ionic/capacitive cell currents must not be added
to this source: there is exactly one physical current owner per fiber.

Preparation binds the existing fixed-geometry `PreparedSkeletalFiberBundle`,
unique fiber IDs, every world-coordinate node in m, one radius in m per fiber,
and explicit geometry provenance/reuse rights. Every edge must match the bound
uniform monodomain metric. No radius, anatomy, conductivity or voltage waveform
is inferred from muscle force. Intracellular conductivity is derived consistently
from the bound diffusivity and membrane capacitance as `2 C_m D / radius`, with
explicit mm²/ms → m²/s and microfarad/cm² → F/m² conversion. Membrane voltage
is converted from mV to V exactly once.

The numerical realization uses edge voltage differences and their conservative
divergence, zero axial flux at both fiber terminals, and half-width endpoint
control volumes. Output `transmembrane_current_A` is the outward current integrated
over a node control volume; `transmembrane_line_current_A_per_m` is that current
divided by its control length. Terminal contributions are retained; neutrality is
not manufactured by subtracting a mean. Signed per-fiber integrated current must
sum to zero within the declared absolute/relative tolerance.

The lifecycle is Plan → Prepared → `propose` → Candidate → `commit`.
The input must be an accepted fiber state (call the fiber candidate's `commit`
first), with the exact fiber-prepared and geometry identities. Commit requires
the observer snapshot and the accepted voltage/time snapshot to remain unchanged.
Nonfinite, foreign, stale, backward-time or non-neutral proposals do not advance
the observer. The current observer never writes cellular or mechanical state.
The new moving structured fiber family is deliberately not accepted by this
fixed-metric adapter; it needs its own registered current-geometry binding.

This current uses physical integrated A and A/m units. It is not implicitly
coerced into the existing planar API's supplied discrete spectrum: a future
planar projection must explicitly bind spatial-cell area and FFT normalization.

## Farina concentric cylindrical conductor

`Farina2004CylindricalConductorPlan` implements the muscle/fat/skin subset of
Farina, Mesin, Martina and Merletti (2004),
[DOI `10.1109/TBME.2003.820998`](https://doi.org/10.1109/TBME.2003.820998).
The [author manuscript](https://iris.polito.it/handle/11583/1402967)
provides Eqs. 7, 11, 13–17 and 32 used here. Its IEEE copyright permits
personal reading but not redistribution; no manuscript or third-party code is
packaged. The prepared identities pin the acquired manuscript/XML content hashes.

Inputs require three strictly increasing tissue radii, transverse/longitudinal
muscle conductivity and isotropic fat/skin conductivity in S/m, plus a named
registered Cartesian frame whose cylinder axis is +z. Fibers must be straight,
parallel to that axis and wholly inside muscle. There are no default anatomical
dimensions or universal material properties.

The radial Green function uses scaled modified Bessel functions from
`phydrax.special` and a row-equilibrated `phydrax.linalg` solve. Potential and
normal-current continuity hold at both tissue interfaces; the cylinder axis is
regular and the external skin/air boundary is insulating. Interface residual
failure aborts preparation rather than selecting another conductor. There is
no bone, sphincter, finite-end limb, heterogeneous anatomical mesh or implanted
contact in this fidelity.

The caller declares the axial Fourier period and both spectral truncations.
The period describes periodic continuation of the source, **not** an insulating
finite-limb end condition. Axial period and mode counts require independent
refinement for each workload. Angular modes are physical integer Fourier modes.
The axial zero mode vanishes separately for each neutral straight fiber, so it
is excluded without changing any source current. Contact potentials use the
zero-axial-average gauge; signed leads must have weights summing to zero.

Each contact has an explicit azimuth, axial location, positive circumferential
arc width and positive axial length. Eq. 32 gives its rectangular-aperture average.
`contact_potential_V` is passive finite-area recording under the stated gauge;
`lead_voltage_V` is the declared gauge-invariant signed montage. Neither is a
complete-electrode impedance model. Point contacts are not silently installed.
Preparation forms reciprocal source-to-contact weights once, summing angular
modes before node contractions to avoid a contact × fiber × node × angular ×
longitudinal temporary. Runtime is a fixed-shape source/lead contraction.

Only committed `FiberCurrentState` snapshots are consumed. Source/prepared and
geometry identities, source neutrality and accepted time are checked, and the
candidate requires unchanged source and observer snapshots at commit. A moved
geometry cannot reuse this lead field. EMG owns observation only, never force.

## Qualification scope and remaining data gates

The dedicated driver below compares the source with an analytic cosine second
derivative under mesh refinement, and the axisymmetric layered cylinder with
an independent radial finite-volume solve containing no Bessel functions.
Focused regressions also compare the full retained homogeneous-cylinder Green
function and finite-aperture voltage against SciPy's Bessel oracle, and defend
terminal charge balance, foreign geometry, nonfinite/stale rollback and JIT
observation. These are **implementation oracles**, not held-out physiological
validation or a replay of published Farina waveforms. The example advances and
commits an actual Shorten fiber before observing it in a labeled idealized case.
The benchmark separates preparation, compilation and accepted observation timing
and reports retained lead-field storage; its manufactured fibers are not anatomy.

```text
python tools/skeletal_muscle_emg_current_cylinder_qualification.py
python examples/skeletal_muscle_emg_current_cylinder.py
python benchmarks/skeletal_muscle_emg_current_cylinder.py --smoke
```

Static anatomical FEM and intramuscular EMG do not ship from this implementation.
The Pereira Botelho 2019 article's data statement says data are in the paper and
supporting information, but its only listed supporting asset is an experimental
rig TIF: no reusable MRI/DTI, segmentation, tissue mesh, fiber-frame field,
registered contacts or raw matching recordings were acquired. The
[Lowery–Weir–Kuiken IMES source](https://doi.org/10.1109/TBME.2006.881774)
does not supply a complete licensed, registered input bundle here.
The [IT'IS Virtual Population FAQ](https://itis.swiss/virtual-population/virtual-population/overview/faq)
requires model-specific licensing/download authorization; access to a paper or
tissue-property table does not grant deployment rights to its geometry.

L3 requires a coherent licensed same-anatomy mesh/regions/material tensors/fiber
field/contact and encapsulation bundle, an explicit gauge and contact model,
an independent forward/reciprocity oracle, and separately held-out recordings.
L4 additionally requires committed registered continuum geometry, non-inverted
mesh checks, synchronized electrode poses and a source-supported conductivity
transport law. No synthetic anatomy defaults, generic contact impedance,
deformation scaling shortcut or stale lead-field fallback closes those gates.
