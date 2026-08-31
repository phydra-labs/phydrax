# Wet granular and rotational contact

`DEMContactModelPlan` composes four independent channels:

```text
contact = phx.discretization.DEMContactModelPlan(
    normal,
    cohesion=cohesion,
    tangential=tangential,
    rotational=rotational,
)
```

Each channel owns typed history and diagnostics. The combined plan declares the maximum finite interaction range needed by particle neighborhoods.

## Contact history

`DEMContactHistory` contains `normal`, `cohesion`, `tangential`, and `rotational` subtrees plus stable pair keys and activity. Rebuilds call one generic pair-value remapper; neighborhood slot order is never physical contact identity.

The normal subtree stores maximum, plastic, and previous overlap. Tangential history stores the objective displacement and prior normal. Rotational history stores objective rolling and torsional displacements and prior normal. Composite cohesion stores one history subtree per contribution.

## DMT cohesion

`DMTContactCohesionPlan` supplies only the attractive DMT contribution. Combine it with an explicit compressive normal law such as `HertzNormalContactPlan`. This avoids a second monolithic normal-force hierarchy and allows DMT to coexist with capillarity and lubrication.

## Capillary bridges

`LinearCapillaryBridgePlan` manages bridge birth, active volume, rupture distance, liquid source, liquid release, and residual. Birth and rupture are explicit guard events. A bridge cannot silently create or destroy liquid inventory.

The plan's finite separation range expands neighborhood reach. Capacity and skin therefore need to be sized for the largest configured cohesive range, not only particle diameter.

## Near-contact lubrication

`NearContactLubricationPlan` regularizes the normal viscous resistance with a minimum gap and truncates it at a declared finite separation. The contribution acts in both small positive gaps and overlap according to its regularized gap. It reports its own dissipated work and validity.

This is a contact-scale lubrication closure, not a complete hydrodynamic added-mass or many-body Stokes solver.

## Rolling and torsion

`ConstantRollingResistancePlan` is stateless. `ElasticRollingTorsionalResistancePlan` stores rolling and torsional elastic displacement, transports history into the current frame, applies damping, and projects trial moments to rolling and torsional friction caps. It reports:

- left and right moments;
- rolling and torsional elastic energy;
- dissipated work;
- yield booleans and cap margins;
- objective-frame transport validity.

The public channel name is `rotational`; the old `rolling=` constructor route is not retained.

## Composition

`CompositeDEMCohesionPlan` sums ordered cohesion contributions while retaining each component's history, event margins, work, and range. Force composition does not collapse diagnostics into an anonymous scalar.

```text
cohesion = phx.discretization.CompositeDEMCohesionPlan(
    (
        phx.discretization.DMTContactCohesionPlan(surface_energy, cutoff),
        phx.discretization.LinearCapillaryBridgePlan(
            surface_tension, contact_angle, bridge_volume, rupture_distance
        ),
        phx.discretization.NearContactLubricationPlan(
            viscosity, cutoff, minimum_gap
        ),
    )
)
```

Run `examples/wet_granular_bridge.py` for an executable composition and `tools/extended_dem_qualification_campaign.py` for separation sweeps.
