# Skeletal surface electromyography

Phydrax exposes two distinct surface-EMG boundaries. Neither accepts activation,
force, calcium, or recruited-unit count as a voltage waveform.

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
