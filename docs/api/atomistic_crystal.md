# Crystal prediction

`phydrax.atomistic.crystal` owns exact space-group settings, molecular crystal
input, asymmetric-unit packing, recipe persistence, relaxation handoff, radial
screening, certified equivalence and deduplication, and same-definition ranking.
See the [crystal prediction guide](../guides_atomistic_crystal_prediction.md) for
the contracts and their nonclaims. Literal selectors and type aliases are listed
with their owners:

- settings: `CrystalSystem`, `MonoclinicAxis`, `OrthorhombicAxes`,
  `RhombohedralAxes`, `OriginChoice`, `CellChoice`; `HallLatticeSymbol`;
- lattices: `NiggliStatus` (`"reduced"`, `"iteration-limit"`);
- site symmetry: `LatticeShift` (an integer `(n1, n2, n3)` lattice translation)
  and `StabilizerElement` (an operation index with its lattice shift);
- molecular input: `ElementQuantity` (`"mass"`, `"covalent-radius"`,
  `"double-bond-radius"`, `"triple-bond-radius"`, `"contact-radius"`,
  `"atomic-volume"`), `ValencePolicy` (`"standard"`, `"unchecked"`),
  `UndeclaredStereoPolicy` (`"refuse"`, `"unspecified"`), `ConformerSource`,
  `ContactRule` (`"radius-sum"`, `"donor-acceptor-reduced"`), `AtomTypeMode`
  (`"connectivity"`, `"chemical"`, `"stereo"`);
- correspondences: `CorrespondenceStatus` (`"complete"`, `"incompatible"`,
  `"mapping-limit"`, `"node-limit"`);
- native conformers: `LocalGeometry` (`"terminal"`, `"linear"`, `"trigonal"`,
  `"tetrahedral"`), `ConformerAttemptStatus` (`"accepted"`,
  `"embedding-rank-deficient"`, `"embedding-nonfinite"`, `"optimizer-failed"`,
  `"bounds-violated"`, `"stereo-violated"`, `"planarity-violated"`,
  `"minimum-distance"`, `"cleanup-failed"`), `ConformerGenerationStatus`
  (`"complete"`, `"attempt-budget-exhausted"`, `"bounds-inconsistent"`,
  `"empty"`);
- asymmetric units and packing: `ChiralityPolicy` (`"enantiopure"`,
  `"racemic"`); packing and storage: `CrystalRecordStage` (`"packed"`,
  `"relaxed"`);
- site tables: `SiteTableStatus` (`"complete"`, `"site-limit"`, `"system-limit"`);
- equivalence and deduplication: `SupercellPolicy` (`"identical-lattice"`,
  `"primitive-reduction"`), `EnantiomorphPolicy` (`"proper-only"`,
  `"allow-improper"`), `DedupMode` (`"greedy"`, `"components"`), `PriorityOrder`,
  `DedupRole`;
- screening and relaxation: `RadialScreenMode` (`"untyped"`, `"center-typed"`,
  `"neighbor-typed"`), `CrystalRelaxationChartKind` (`"variable-cell"`,
  `"symmetry-atomic"`, `"symmetry-rigid"`).

`SPACE_GROUP_SOURCES` and `SPACE_GROUP_DATA_ID` record the admitted provenance of
the packaged operation tables; `RANKING_NONCLAIM` is the text every ranking
carries. `AROMATIC_BOND` (`4`) is the aromatic bond-order code of
`MolecularComponent`, and `CONTACT_OWNER` (`"crystal-contact-relief"`) is the
owner label of the packers' contact-relief `IMAGE_PAIR_SUM` neighborhood
requirement.

## Space-group data, Hall symbols, and settings

::: phydrax.atomistic.crystal.SpaceGroupDataSource

::: phydrax.atomistic.crystal.HallSymbol

::: phydrax.atomistic.crystal.admitted_hall_numbers

::: phydrax.atomistic.crystal.OperationDim

::: phydrax.atomistic.crystal.SettingTransformation

::: phydrax.atomistic.crystal.SpaceGroupSetting

::: phydrax.atomistic.crystal.standard_space_group_setting

## Lattices and metric charts

::: phydrax.atomistic.crystal.enumerate_sublattice_transforms

::: phydrax.atomistic.crystal.metric_from_cell_parameters

::: phydrax.atomistic.crystal.niggli_reduce

::: phydrax.atomistic.crystal.NiggliReduction

::: phydrax.atomistic.crystal.SettingCellResult

::: phydrax.atomistic.crystal.SettingMetricChart

::: phydrax.atomistic.crystal.sublattice_count

::: phydrax.atomistic.crystal.SublatticeDim

## Site symmetry

::: phydrax.atomistic.crystal.enumerate_site_orbits

::: phydrax.atomistic.crystal.site_of_point

::: phydrax.atomistic.crystal.SiteOrbit

::: phydrax.atomistic.crystal.SitePosition

::: phydrax.atomistic.crystal.SiteSymmetryTable

## Molecular input

::: phydrax.atomistic.crystal.ConformerPool

::: phydrax.atomistic.crystal.ContactPolicy

::: phydrax.atomistic.crystal.ElementTable

::: phydrax.atomistic.crystal.FormulaAtomDim

::: phydrax.atomistic.crystal.MolecularAtomDim

::: phydrax.atomistic.crystal.MolecularComponent

::: phydrax.atomistic.crystal.MolecularConformer

::: phydrax.atomistic.crystal.MolecularCrystalInput

::: phydrax.atomistic.crystal.stereo_realization

::: phydrax.atomistic.crystal.StereoRealization

::: phydrax.atomistic.crystal.VolumePolicy

## Atom types and correspondences

::: phydrax.atomistic.crystal.atom_type_assignment

::: phydrax.atomistic.crystal.AtomTypeAssignment

::: phydrax.atomistic.crystal.component_atom_orbits

::: phydrax.atomistic.crystal.CorrespondenceResult

::: phydrax.atomistic.crystal.map_connectivity_types

::: phydrax.atomistic.crystal.molecular_correspondences

## Native conformers

::: phydrax.atomistic.crystal.ChiralCenterDim

::: phydrax.atomistic.crystal.conformer_rmsd

::: phydrax.atomistic.crystal.ConformerAttempt

::: phydrax.atomistic.crystal.ConformerConstraints

::: phydrax.atomistic.crystal.ConformerEnergyCleanup

::: phydrax.atomistic.crystal.ConformerGenerationPlan

::: phydrax.atomistic.crystal.ConformerGenerationResult

::: phydrax.atomistic.crystal.ConformerGeometryTable

::: phydrax.atomistic.crystal.ConformerQualification

::: phydrax.atomistic.crystal.ConformerRMSD

::: phydrax.atomistic.crystal.GeometryQualification

::: phydrax.atomistic.crystal.metric_embedding

::: phydrax.atomistic.crystal.MetricEmbedding

::: phydrax.atomistic.crystal.PlanarityDim

::: phydrax.atomistic.crystal.prepare_conformer_constraints

::: phydrax.atomistic.crystal.prune_conformers

::: phydrax.atomistic.crystal.qualify_geometry

## Asymmetric units

`AsymmetricUnitPlan` is the host owner with the molecular witnesses;
`AsymmetricUnitLayout(plan)` is its prepared numerical layout (JAX arrays only),
held by compiled packers and symmetry charts. `expand_asymmetric_unit` accepts
either; compiled consumers pass the layout.

::: phydrax.atomistic.crystal.AsymmetricUnitLayout

::: phydrax.atomistic.crystal.AsymmetricUnitPlan

::: phydrax.atomistic.crystal.AsymmetricUnitSite

::: phydrax.atomistic.crystal.AsymmetricUnitState

::: phydrax.atomistic.crystal.CopyDim

::: phydrax.atomistic.crystal.expand_asymmetric_unit

::: phydrax.atomistic.crystal.ExpandedAtomDim

::: phydrax.atomistic.crystal.ExpandedCrystal

## Symmetry charts

::: phydrax.atomistic.crystal.atomic_state_from_expansion

::: phydrax.atomistic.crystal.atomic_tangent

::: phydrax.atomistic.crystal.AtomicExpansion

::: phydrax.atomistic.crystal.AtomicSymmetricState

::: phydrax.atomistic.crystal.AtomicSymmetryPlan

::: phydrax.atomistic.crystal.expand_symmetric_atoms

::: phydrax.atomistic.crystal.retract_atomic

::: phydrax.atomistic.crystal.RigidSymmetryLayout

::: phydrax.atomistic.crystal.strain_generators

::: phydrax.atomistic.crystal.symmetry_residual

::: phydrax.atomistic.crystal.symmetry_residuals

::: phydrax.atomistic.crystal.SymmetryResidual

::: phydrax.atomistic.crystal.SymmetryTangentOperator

## Packing and storage

::: phydrax.atomistic.crystal.crystal_structure_record

::: phydrax.atomistic.crystal.crystal_structure_schema

::: phydrax.atomistic.crystal.CrystalCellPolicy

::: phydrax.atomistic.crystal.CrystalContactResources

::: phydrax.atomistic.crystal.CrystalPackingPlan

::: phydrax.atomistic.crystal.CrystalPackingResult

::: phydrax.atomistic.crystal.CrystalReliefMethod

::: phydrax.atomistic.crystal.CrystalReliefPolicy

::: phydrax.atomistic.crystal.CrystalSpaceGroupPolicy

::: phydrax.atomistic.crystal.native_conformer_pools

::: phydrax.atomistic.crystal.PackedCrystal

::: phydrax.atomistic.crystal.PreparedCrystalGroupPacker

::: phydrax.atomistic.crystal.PreparedCrystalPacker

## Recipe persistence

A `CrystalPackingRecipe` binds a prepared packer to its generation stream
configuration; `write_crystal_packing_recipe` stores every reconstructible
immutable input as one bounded pickle-free archive, and
`read_crystal_packing_recipe` replays every owning constructor and refuses unless
the re-encoding is byte-exact. Replay is exact in the writing numerical profile
only.

::: phydrax.atomistic.crystal.CrystalPackingRecipe

::: phydrax.atomistic.crystal.CrystalPackingRecipeReceipt

::: phydrax.atomistic.crystal.write_crystal_packing_recipe

::: phydrax.atomistic.crystal.read_crystal_packing_recipe

## Equivalence certificates

::: phydrax.atomistic.crystal.certify_equivalence

::: phydrax.atomistic.crystal.crystal_structure_from_expansion

::: phydrax.atomistic.crystal.CrystalStructure

::: phydrax.atomistic.crystal.EquivalenceCertificate

::: phydrax.atomistic.crystal.EquivalencePolicy

::: phydrax.atomistic.crystal.EquivalenceResult

::: phydrax.atomistic.crystal.EquivalenceStatus

## Persistent deduplication

::: phydrax.atomistic.crystal.create_dedup_bank

::: phydrax.atomistic.crystal.DedupBank

::: phydrax.atomistic.crystal.DedupBankCommit

::: phydrax.atomistic.crystal.DedupBankError

::: phydrax.atomistic.crystal.DedupBankLimits

::: phydrax.atomistic.crystal.DedupCandidate

::: phydrax.atomistic.crystal.DedupComparison

::: phydrax.atomistic.crystal.DedupDecision

::: phydrax.atomistic.crystal.DedupPolicy

::: phydrax.atomistic.crystal.DedupSubmission

::: phydrax.atomistic.crystal.open_dedup_bank

::: phydrax.atomistic.crystal.submit_to_dedup_bank

## Screening, relaxation, and ranking

::: phydrax.atomistic.crystal.ChemicalPotentialReference

::: phydrax.atomistic.crystal.CrystalRanking

::: phydrax.atomistic.crystal.CrystalRankingCandidate

::: phydrax.atomistic.crystal.CrystalRankingEntry

::: phydrax.atomistic.crystal.CrystalRankingExclusion

::: phydrax.atomistic.crystal.CrystalRankingPolicy

::: phydrax.atomistic.crystal.CrystalRelaxationEvidence

::: phydrax.atomistic.crystal.CrystalRelaxationInput

::: phydrax.atomistic.crystal.CrystalScreeningOutcome

::: phydrax.atomistic.crystal.CrystalScreeningRecord

::: phydrax.atomistic.crystal.radial_fingerprints

::: phydrax.atomistic.crystal.radial_scores

::: phydrax.atomistic.crystal.RadialFingerprints

::: phydrax.atomistic.crystal.RadialScreenCapacityError

::: phydrax.atomistic.crystal.RadialScreenDecision

::: phydrax.atomistic.crystal.RadialScreenPolicy

::: phydrax.atomistic.crystal.RadialScreenResult

::: phydrax.atomistic.crystal.rank_crystal_candidates

::: phydrax.atomistic.crystal.relax_crystal_candidates

::: phydrax.atomistic.crystal.screen_dedup_rank

::: phydrax.atomistic.crystal.screen_radial
