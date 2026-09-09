# Fusion source, neutronics, and activation

The engineering boundary is explicit:

```text
core transport
→ thermal fusion branch
→ spatial neutron source
→ transport response
→ material flux
→ activation and decay
```

## Fusion source

`ThermalFusionReactionPlan` combines two reactant densities with a tabulated Maxwellian reactivity. Identical reactants receive the one-half symmetry factor. Two-body product energies are derived from the admitted mass table, and branch-resolved particle, charge, and energy ledgers are retained.

The initial source is isotropic and line-energy. Thermal broadening, beam-target reactions, anisotropy, non-Maxwellian distributions, and reacting PIC are not represented.

## Neutron response

`NeutronResponsePlan` maps core neutron birth rates to one material region's multigroup scalar flux. The source identity and differentiation policy are part of the plan. A response imported from Monte Carlo is constant; a caller cannot differentiate through an external OpenMC run by relabeling it native.

`FusionActivationScenarioPlan` executes a staggered core-transport, fusion, response, and activation transaction. Charged-product heating is returned for the next transport exchange. The complete scenario succeeds only when every stage succeeds; activation rolls back when an upstream stage fails.

## OpenMC boundary

`phydrax.nuclear.interchange` uses the existing pinned external-energy runtime. `run_openmc` stages caller-provided inputs in a private directory, invokes an exact executable pin without a shell, and detaches bounded outputs.

`import_openmc_multigroup_flux` supports one deliberately narrow statepoint profile:

- one dedicated scalar-flux tally;
- explicitly declared result shape and axis labels;
- trailing energy-group axis;
- volume-normalized per-source values;
- explicit physical source rate;
- stored tally sum and sum of squares;
- realization count and OpenMC release identity.

It converts the tally to a governed `MeasurementAsset` with independent standard uncertainty. It does not guess filters, score ordering, material mapping, volume normalization, or energy order.

A `DagmcGeometryArtifact` keeps exact HDF5 bytes, material-region mapping, unit, converter identity, overlap-check claim, and source geometry lineage. DAGMC geometry remains opaque to JAX.

## Activation

Activation uses a fixed nuclide closure and piecewise-constant group flux. Every transition declares incident and escaped baryon/charge/lepton quantities. Matrix-exponential diagnostics, inventory minimum, activity, and decay heat are retained. Dynamic graph discovery, pathway pruning during JIT, and silent negative clipping are forbidden.

## Differentiation boundary

Native core transport, tabulated reactivity inside one support interval, native response matrices, and fixed activation networks may be differentiated. Geometry topology preparation, table support changes, external OpenMC/DAGMC execution, stochastic histories, and source-data selection are discrete or constant boundaries.
