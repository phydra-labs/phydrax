# Commercial MPM transfers and schedules

`ExplicitMPMMethodPlan` owns independent velocity-transfer, position-advection,
and stress-update schedule plans.

## Velocity transfer

Available MPM-specific plans:

```text
PICTransferPlan
FLIPTransferPlan
PICFLIPTransferPlan
APICTransferPlan
```

They do not reuse electromagnetic PIC or free-surface FLIP solver state.

```text
v_PIC = sum_i N_ip v_i_new

v_FLIP = v_p_old + sum_i N_ip (v_i_new - v_i_before)

v_new = pic_fraction v_PIC + (1 - pic_fraction) v_FLIP
```

APIC retains its affine `C` state. PIC/FLIP/blended plans set `C = 0` and exclude
affine momentum from P2G. `PICFLIPTransferPlan.pic_fraction` is fingerprinted with
the displayed convention.

Position advection is independently selected:

```text
PICAdvectionPlan
TransferredVelocityAdvectionPlan
MidpointAdvectionPlan
```

This prevents FLIP momentum update from silently choosing its own advection law.

## Schedule closure

In addition to USF, USL-minus, and classical translational MUSL:

- `AffineMUSLMPMSchedule` includes updated APIC affine momentum in its second
  pre-advection P2G.
- `PostAdvectionMUSLMPMSchedule` advances particle position, updates assignment
  input, rebuilds routes, repeats P2G/constraints, and records the second topology
  digest.

Every second grid reconstruction reapplies the complete simultaneous field,
rigid, and essential constraint transaction.

## Multifield transfer

A field plan owns one `KWayMPMContactPlan`. Each field receives independent mass,
momentum, force, mass-gradient, and transfer reconstruction. USF and all MUSL
variants update every field under the same accepted schedule. Pairwise
`project_two_field_contact` is no longer a public authority.

## Evidence

Transfer/schedule evidence includes mass, momentum, affine/angular behavior,
PIC dissipation, FLIP grid-delta behavior, second-P2G balance, second route digest,
constraint work/dissipation, field leakage, and schedule identity.
