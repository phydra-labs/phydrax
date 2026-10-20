# Python 3.13: end-to-end migration plan

## 0. Status and chosen scope

Implemented on branch `python-313`; section 13 records the delivered changes, observed evidence, and deviations from this plan. Sections 0–12 are the original plan.

The plan implements a **clean replacement of Python 3.12 with ordinary, GIL-enabled CPython 3.13**:

- The main and native companion distributions and the standalone Taylor-threshold generator use `requires-python = ">=3.13,<3.14"`. The separate CUDA-major intent marker is unchanged.
- The development interpreter pin is `3.13`; qualification records the exact patch release, interpreter build, platform, and dependencies actually used.
- Python 3.12 and Python 3.14 are outside the new supported range. Retaining 3.12 would be a different, dual-interpreter support policy, not an implementation detail.
- Free-threaded `3.13t` and the experimental CPython JIT are not included. Do not mistake CUDA major 13 for a Python interpreter selection.
- Keep ty, Ruff, the existing native typing vocabulary, and `typing_extensions`. No wholesale annotation modernization, type-parameter defaults, `TypeIs`/`ReadOnly` adoption, new runtime checker, or checker replacement.
- Preserve scientific algorithms, dtype/precision policies, reduction order, RNG addressing, public object structure, static/dynamic partitions, failure evidence, and persistence formats.
- Preserve the existing callable/compiler fingerprint algorithm. Interpreter-dependent executable identities may differ; do not normalize bytecode, retain historical aliases, suppress identity refusals, or relabel old artifacts.
- Move the separate generator environment to 3.13 as well, but require its canonical numerical table to remain unchanged.
- Preserve external provider interpreter pins and historical measurements. They are not project interpreter requirements.

A successful migration does not promise transparent continuation of every 3.12 code-bound checkpoint. Existing exact identity admission remains authoritative. Portable data/model artifacts must be tested through their owning validators; changed executable bindings require re-preparation or a new run. A requirement for transparent continuation despite changed code IDs would need an explicitly approved identity-migration design outside this plan.

No package release number is selected by this plan. Keep the current lockstep versions during development. If publishing requires a new release, the main distribution, meshcore, CUDA companion, exact dependency pins, and embedded native release identities must move together through the existing release process; never overwrite already-published wheel metadata.

## 1. Current evidence and prerequisites

| Observed surface | Evidence | Consequence |
| --- | --- | --- |
| Root interpreter policy | `.python-version` is `3.12`; `pyproject.toml:35` is `>=3.12,<3.13`; `uv.lock:3` is `==3.12.*` | All three must change coherently; editing only the interpreter pin cannot produce a supported installation. |
| Existing development environment | `.venv/pyvenv.cfg` records CPython 3.12.8 | Create a fresh environment in the implementation worktree. Do not mutate or reuse the parent environment. |
| Checker target | `pyproject.toml:364` selects Python 3.12; installed-wheel typing defaults to 3.12 | Retarget both static language semantics and actual consumer installation. |
| Existing dependency mismatch | Root metadata requests JAX `>=0.12.0,<0.13`; the inspected lock and installed distribution metadata still contain JAX/JAXlib 0.11.2 | Reconcile the already-requested JAX change separately from the interpreter comparison. Do not downgrade current source or blame its pre-existing lock mismatch on Python 3.13. |
| Native companion policy | Meshcore and CUDA companion metadata exclude 3.13; both set `wheel.py-api = "py3"` | Change metadata and qualify their Python consumers. They are platform-specific C ABI libraries, not CPython-minor extension modules. |
| Interpreter-sensitive runtime ownership | `_identity._code_payload` encodes bytecode/constants; `_trainable._code_global_names` disassembles code recursively | Verify identities, hidden-state refusal, and downstream restore admission; no proactive algorithm rewrite. |
| Installed CUDA dependency test | `tests/unit/test_elf_dependencies.py:383` uses `/phydrax/.venv/lib/python3.12/site-packages` | Remove the fixed host/interpreter path so the test actually inspects the target environment. |
| Synthetic provider discovery | `tests/unit/backends/test_backend_contracts.py:178` names its synthetic extension `fakeprov.cpython-312-x86_64-linux-gnu.so` | Derive a supported extension suffix from the running interpreter. Discovery occurs before CUDA refusal, so the filename is behaviorally significant even though the ELF bytes are synthetic. |
| Standalone generator | Its own metadata and lock restrict Python to 3.12; python-flint 0.9.0 is locked with stable-ABI wheels | Migrate the generator's environment independently; prove identical generated output. |

The source already records the JAX 0.12 transition separately in `CHANGELOG.md` under Unreleased/Breaking and in `docs/installation.md`. Retain that distinction.

Upstream release-file inspection found ordinary Python 3.13 artifacts for JAXlib 0.12.0, NumPy 2.5.3, SciPy 1.18.1, h5py 3.16.0, cadquery-ocp-novtk 8.0.1.0.0, and selected optional providers. Python-flint 0.9.0 documents CPython 3.10–3.14 support. These facts are installation prerequisites, not proof that the entire resolver graph or any Phydrax runtime scenario passes.

### Worktree and evidence ownership

On a later `//code` request:

1. Create a fresh worktree from the current branch under the prescribed `../phydra-labs/.worktrees/` location, with a branch/worktree name such as `python-313`. Copy the authoritative ignored context file into it before implementation work. `//code--` is the explicit exception.
2. Establish the current branch state and changes since the latest merge into `dev` once. Record concurrent/uncommitted work; do not reset it or assume a new worktree automatically contains it. Bring required approved changes forward without overwriting another owner's work.
3. Carry this plan into the implementation worktree if it is not yet in the branch snapshot. Reconcile only inputs that have actually changed since planning.
4. Keep one task-level evidence record outside published scientific artifacts: exact source/lock state, command, interpreter/JAX/backend/provider identity, result, skips, failing nodes, and relevant duration.
5. Give root metadata/locks, qualification tools, and final integration one owner. Identity/persistence qualification and native/provider qualification can proceed independently after their shared environment prerequisites pass. Benchmarks run without competing agents, tests, or builds.

## 2. Implementation sequence

| Phase | Work and dependency | Exit condition |
| --- | --- | --- |
| A. Resolve the baseline | In the implementation worktree, reconcile the current JAX 0.12 requirement while still using a separate ordinary 3.12 reference environment. Preserve unrelated locked versions where the resolver permits. Capture only the narrow runtime, artifact, and benchmark baseline needed below. | The reference source and dependency versions are known; source/metadata/installed-environment disagreement is not hidden. |
| B. Cut over metadata and tools | Apply the exact file changes in section 3, then resolve the root and generator locks for 3.13. Create a fresh 3.13 development environment with the required QA/test dependencies. | All project-owned interpreter selectors agree and installation no longer relies on 3.12 artifacts. |
| C. Prove the narrow runtime | Apply the configured formatter to changed Python files before identity-sensitive verification. Run installed-wheel/runtime smokes, focused typing/identity checks, and the meaningful regression cases in sections 4–6. | The changed paths work, and focused failures are resolved before broadening verification. |
| D. Integrate on CPU | Run repository typing/audits and the conservative CPU test matrix, accounting for the global interpreter/configuration replacement and all other changes since the required comparison point. | One complete, recorded CPU result with no unexplained skips or repeated valid checks. |
| E. Qualify native/platform routes | Build/install matching companions; run real meshcore, CUDA, and site-specific MPI gates separately. Reuse a passed metadata gate rather than invoking it again inside the GPU qualifier. | Required deployment claims have actual matching hardware/provider evidence. |
| F. Compare and document | Run the isolated phase-separated benchmark pair, verify unchanged generated data/public exports, finish authored documentation and changelog, and record the actual outcome in this plan. | All acceptance criteria in section 11 are satisfied; otherwise completion remains blocked at the specifically named external qualification gate. |

The 3.12 reference is temporary migration evidence, not a retained supported environment, compatibility branch, or new test-suite requirement. It must use the same intended JAX/NumPy/SciPy/etc. versions as 3.13 wherever compatible. If Python markers force a dependency difference, record it and label the comparison as a combined environment change; do not manufacture a Python-only attribution.

Do not use `uv lock --upgrade` or an all-packages refresh merely to change Python. Re-resolve the necessary graph using existing compatible selections. Do not hand-edit lockfile wheel URLs, hashes, resolver markers, or the externally defined uv lock format fields.

## 3. Exact file-by-file changes

### 3.1 Interpreter, metadata, and qualification tools

| File | Required change | Invariant / verification |
| --- | --- | --- |
| `.python-version` | Change `3.12` to `3.13`. | The new worktree environment reports ordinary CPython 3.13 with the GIL enabled; record the exact patch/build rather than assuming the pin proves it. |
| `pyproject.toml` | Change the root Python requirement to `>=3.13,<3.14` and `[tool.ty.environment].python-version` to `3.13`. | Keep current dependency bounds, extras, explicit `fmi`/`privacy-jax` conflict, uv sources, Ruff rule selection, and pytest configuration unless a demonstrated compatibility defect requires a surgical change. No new optional-dependency suppressions. |
| `uv.lock` | Regenerate from the finalized metadata for 3.13, including the already-declared JAX 0.12 requirement and native metadata changes. | Root requirement becomes the resolver's canonical 3.13 range. Matching JAX/JAXlib/plugin versions, companion pins, platform markers, and supported extra combinations resolve; no stale 3.12-only dependency selection is reused as target evidence. |
| `native/meshcore/pyproject.toml` | Set `requires-python = ">=3.13,<3.14"`. | Keep `wheel.py-api = "py3"`, package layout, CMake settings, and lockstep release identity. Inspect/install the actual platform wheel. |
| `native/cuda13/pyproject.toml` | Set `requires-python = ">=3.13,<3.14"`. | Retain the wheel-only backend, CUDA 13 dependencies, current JAX/JAXlib build requirements, architecture policy, and generic Python API tag. |
| `tools/check_installed_typing.py` | Change `DEFAULT_PYTHONS` to `("3.13",)`. | Preserve explicit `--python` support, outside-checkout consumer isolation, `py.typed` validation, pinned ty, and error reporting. Run the default invocation once so actual output establishes the new default. LSP references currently show only the declaration and `main` consumer. |
| `tools/check_cuda_extras.py` | Change `compile_case`'s resolver target from 3.12 to 3.13. | Preserve all platform/marker/companion checks and failure evidence. No new selector or target-forwarding abstraction is needed. |
| `tools/qualify_cuda_wheels.py` | Change automatic fresh-environment creation from 3.12 to 3.13. | Use a fresh environment for qualification; retain required-GPU coverage, hostile-path audits, teardown checks, and all missing/skipped-evidence refusals. Do not use `--reuse-venv` to conceal the wrong interpreter. |
| `tools/generate_exponential_taylor_thresholds/pyproject.toml` | Set `requires-python = ">=3.13,<3.14"`. | Keep python-flint bounds and the independent non-package project structure. |
| `tools/generate_exponential_taylor_thresholds/uv.lock` | Regenerate the standalone lock for 3.13. | Preserve python-flint 0.9.0 if resolution permits; its existing stable-ABI wheels do not require a made-up `cp313` filename. |

Ruff currently infers its target from the discovered project `requires-python`; the existing gate does not pass an explicit Ruff configuration file that disables that inference. Do not add a redundant target setting or enable modernization rules as part of this change.

### 3.2 Required test repairs

| File and owning scenario | Required change | Consumer-visible proof |
| --- | --- | --- |
| `tests/unit/test_elf_dependencies.py::test_cuda_wheel_dependencies_resolve_inside_wheels` | Replace the absolute `/phydrax/.venv/lib/python3.12/site-packages` assumption with installed-distribution discovery. Follow the existing `importlib.metadata.distributions()` / `Distribution.files` / `locate_file` pattern in `tests/unit/test_cuda_environment.py`. Locate the active CUDA 13 plugin and NVIDIA wheel libraries without importing/loading the provider. Derive search directories and allowed wheel ownership from those records, not another literal `site-packages` path. | In a real 3.13 CUDA wheel environment, cuBLAS dependencies resolve to the installed NVIDIA wheel contents. An absent optional provider may skip at its metadata boundary; an installed provider with missing expected library files or unresolved dependencies must fail, not be relabeled absent. Preserve the dependency-closure/ownership assertions. |
| `tests/unit/backends/test_backend_contracts.py::test_cuda_provider_dependency_mismatch_refuses_before_import` | Inside the existing subprocess scenario, construct the synthetic `fakeprov` filename using `importlib.machinery.EXTENSION_SUFFIXES[0]` rather than a `cpython-312` suffix. Keep the foreign-CUDA ELF dependency and actual `probe_backend` call. | The module is discoverable under 3.13, then refused with `provider-major-mismatch` before import; `fakeprov` must remain absent from `sys.modules`. Do not weaken this to a generic unavailable result or assert the suffix text. |

The second repair is necessary despite the fixture containing inspectable rather than executable ELF bytes: `_availability.probe_backend` performs module discovery and returns `not_installed` before checking CUDA dependencies if no specification exists. A foreign-interpreter extension filename would exercise the wrong branch. No production availability change is planned.

Do not change historical CPython version values in provider provenance, external-profile recipes, or benchmark JSON. Do not add permanent tests of version literals, command forwarding, TOML source text, or incidental defaults.

### 3.3 Documentation and release notes

| File | Required content |
| --- | --- |
| `README.md` | State Python 3.13 as the supported interpreter. Keep CPU/CUDA installation semantics unchanged. |
| `docs/index.md` | Align the installation summary with the same 3.13-only policy. |
| `docs/installation.md` | Update prerequisites; explain fresh-environment installation, ordinary versus free-threaded Python, companion metadata, and the separate JAX transition. Add a concise migration subsection distinguishing portable artifacts from code-bound checkpoint refusal/re-preparation. Mark all retained 3.12/JAX 0.11.2 GPU/MPI results as historical, not new qualification. |
| `docs/guides_typing.md` | Describe the 3.13 checker and installed-wheel target. State that native runtime contracts and `typing_extensions` remain required; no automatic static shape inference or annotation modernization is part of the bump. Preserve the distinction between static-only Protocol annotations and explicit runtime provider checks. |
| `docs/cookbook/nucleic_acid_biophysics.md` | Change the current checkout prerequisite to Python 3.13; preserve fixture provenance, launch forms, numerical claims, and retained results. |
| `docs/cookbook/protein_folding.md` | Change the current checkout prerequisite to Python 3.13; preserve the separately specified OpenMM/provider contract and source provenance. |
| `docs/cookbook/radiation_biophysics.md` | Change the prerequisite to Python 3.13; do not rewrite the historical smoke observation or imply a new external radiation campaign. |
| `docs/cookbook/single_cell_transcripts.md` | Change the prerequisite to Python 3.13; preserve event-capacity refusal and statistical qualification language. |
| `CHANGELOG.md` | Add an Unreleased/Breaking entry for dropping 3.12, requiring ordinary 3.13, and possible code-derived identity changes. Keep the existing JAX entry separate. State the re-preparation/refusal boundary without claiming all checkpoints are incompatible. |
| `docs/plans/python-313.md` | After implementation, append exact implemented changes, verification/artifact locations, qualified platform/provider scope, and any explicitly unqualified external routes. Do not replace planned scenarios with fabricated pass statements. |

No navigation or public API documentation restructure is needed for a prerequisite-only migration. The four cookbook commands themselves do not change; do not launch their entire scientific campaigns solely because the prerequisite sentence changes.

### 3.4 Files intentionally unchanged unless evidence requires a fix

| Files / owner | Required disposition |
| --- | --- |
| `phydrax/typing.py`, `_typing_forms.py`, `_typing_plan.py`, `_typing_signature.py` | Preserve vocabulary identity, generic alias behavior, explicit forward-reference namespaces, fresh dimension scopes, static-only unions/protocols, cache lifetimes, argument forwarding, and conversion ownership. Fix only a reproduced 3.13 incompatibility at its existing owner. |
| `phydrax/_strict.py` | Preserve abstract/final handling, Equinox construction, field order, post-construction validation, and freeze. Do not blanket-filter new class attributes or change field discovery without a failing consumer scenario. |
| `phydrax/_config_resolution.py` | Preserve `TypeAliasType` and `get_type_hints` behavior, selector parsing, precedence, and identity. Include its existing tests because annotation evaluation is interpreter-sensitive. |
| `phydrax/_identity.py`, `_trainable.py` | Keep the fingerprint representation and imported-helper boundary. A demonstrated 3.13 opcode-discovery gap may receive a narrow fix plus regression; do not rewrite identity to preserve a historical digest. |
| `phydrax/solver/_functional_checkpoint.py`, `_functional_training.py`, `_functional_gradient.py`, `_kfac_solver.py` | Keep exact training-plan admission and existing checkpoint semantics. LSP references identify functional-gradient and KFAC restore consumers. |
| `phydrax/nn/operator/training/_checkpoint.py`, `phydrax/_training_kernel.py`, `_training_checkpoint.py` | Keep exact executable binding, parameter/RNG/optimizer payload validation, atomic publication, and corruption refusal. |
| `phydrax/_array_archive.py`, `_model/_structure.py`, `_model/_artifacts.py`, `ml/artifacts/_portable.py`, `atomistic/_model_artifact.py`, `atomistic/_checkpoint.py` | Preserve canonical data/recipe formats and registered reconstruction boundaries. No Python-version compatibility generation or unvalidated restore path. |
| `phydrax/_jax_program.py`, `_precision_rewrite.py` | Retain current JAX compiler admission and precision semantics. Investigate failures against the intended JAX baseline; do not conflate them with Python or weaken foreign/effectful operation refusal. |
| `phydrax/_meshcore.py`, `_cuda_native.py`, `_cuda_environment.py`, `backends/_availability.py` | Keep library, distribution-release, CUDA-major, provider, and pre-import failure checks. The two test repairs must not require relaxing production admission. |
| `tools/build_cuda_wheels.py`, `tools/qualify_cuda_mpi.py` | Already accept interpreter paths. Supply a provisioned 3.13 environment; do not add a duplicate Python-version CLI or hardcoded target. |
| `native/meshcore/CMakeLists.txt`, native C/C++/CUDA sources, `native/cuda13/_build_backend.py` | No kernel/ABI change is implied by Python metadata. Rebuild/package as required, but change source only for a demonstrated incompatibility. |
| `packaging/cuda-major/13/pyproject.toml` | Keep the metadata-only CUDA intent marker unchanged. Its major is CUDA 13, not Python 3.13. |
| `tools/generate_exponential_taylor_thresholds/generate.py`, `phydrax/linalg/_generated_exponential_taylor_thresholds.py` | Generator algorithm and canonical table remain unchanged. Regenerate to scratch and compare exact bytes; do not adopt numerical changes as interpreter housekeeping. |
| `tools/check_typing.py`, existing installed typing fixtures, selector/contract audit tools | Reuse the current gates. Fix truthful annotations or genuine checker defects only if new diagnostics demonstrate a problem; do not blanket-suppress or re-pin incidental implementation expectations. |
| `benchmarks/_runtime.py`, `benchmarks/configuration.py`, `benchmarks/exterior_calculus.py`, committed benchmark JSON | Reuse existing environment identities and phase-separated drivers. Write new evidence to task-owned outputs; do not overwrite historical values or forge same-environment comparisons. |
| `docs/data/public_api.json`, `docs/data/capabilities.json`, related capability manifests | No exported symbol/capability change is planned. Check existing generated consistency; regenerate only if an actual approved public-surface change requires it, after implementation stabilizes. Do not promote a scientific capability merely because its package imports on 3.13. |
| External-provider environment recipes and historical plans | Preserve provider-owned Python requirements and prior evidence. In particular, the separate MACE/QDesignOptimizer/SRW profiles are not automatically retargeted. |

## 4. Identity and artifact migration contract

### 4.1 Required cross-interpreter experiment

Before replacing the reference environment, produce small real artifacts using the existing writers under the matched 3.12 dependency baseline. Consume those exact bytes under 3.13. Preserve producer source/dependency identity and artifact digests in the task evidence; never regenerate the supposed 3.12 input with 3.13.

| Scenario | Producer/consumer ownership | Required result |
| --- | --- | --- |
| Array-only archive | `_array_archive` packing/unpacking and canonical templates | Exact leaf paths, dtype, shape, values, and checksum evidence survive compatible restoration; invalid inventory/shape/dtype/corruption remains refused. |
| Portable native ML model | `ml/artifacts/_portable.py`, registered recipes | Restore through the real public reader; preserve scientific/parameter identities and numerical predictions when the recipe and dependency semantics are unchanged. Recorded runtime versions remain provenance, not silently rewritten metadata. |
| Portable atomistic model | `atomistic/_model_artifact.py` | Restore identities and energy/force behavior against the owning independent/reference contract. A runtime checkpoint additionally requires the rebuilt prepared plan and template to match. |
| Functional-training checkpoint with a code-derived relaxation map | `ResidualRelaxationMap.map_id` → `PseudoTransientPolicy.policy_id` → `FunctionalTrainingPlan.plan_id` → persisted `plan_id` | Recompute the target plan. Equal IDs allow the normal validated resume path; differing IDs must produce the existing training-plan mismatch before continuation. No array-only fallback. |
| Operator-training checkpoint with a plain function in `static_callables` | `callable_payload` → executable signature → persisted binding | Equal binding permits the existing validated restore; a changed binding raises the existing `ValueError` identifying the executable-signature mismatch. Preserve model, optimizer, RNG, policy, and schema checks. |
| In-memory prepared/runtime state | Execution worksets, probabilistic ODE, Maxwell refresh, compiler-bound surface owners | Rebuild in the 3.13 process. Verify existing stale-signature/owner refusals; do not invent a cross-process serialization promise for in-memory objects. |

A code-derived ID changing is not automatically a product defect: accepting an incompatible old binding silently is. A portable data/model artifact failing under an otherwise matched supported contract is a real migration failure to investigate.

Cross-interpreter experiments are throwaway qualification drivers using existing APIs, not a permanent requirement to install Python 3.12 in every future test run. No new persistence schema, generation suffix, historical hash alias, or broad migration registry is introduced.

### 4.2 Focused permanent regression additions

These target uncertain, consumer-visible edges uncovered by inspection. The new regression scenarios below are proposals, not claims that the cases already exist.

| Existing test file | Proposed addition | Bug it detects |
| --- | --- | --- |
| `tests/unit/test_array_roles.py` | Add a parametrized hidden-array refusal case for a module-global array referenced only through a nested function or generator/comprehension code. Exercise `require_parameter_roles`/the existing public role-admission path, not the opcode helper directly. | A 3.13 bytecode/disassembly change silently hiding numeric callback state. Keep imported-helper recursion semantics unchanged. |
| `tests/unit/solver/test_functional_training_runtime.py` | Add a real checkpoint lifecycle using a default code-derived `ResidualRelaxationMap` in the pseudo-transient policy. Prove valid same-plan continuation, then refusal after changing the callback's scientific behavior and rebuilding the plan. Keep parameters and unrelated policy settings constant. | Training resume dropping the callback-derived plan identity while ordinary checkpoint/parameter cases still pass. |
| `tests/unit/nn/test_operator_training.py` | Add a real checkpoint lifecycle with a stateless module-level function in `static_callables`. Preserve the function's module/qualified name while changing a nested computation's literal/behavior through controlled test setup; prove the callback's behavior changed, then require executable-binding refusal. Keep dynamic model arrays unchanged. | A persisted executable binding omitting plain-function or nested-code semantics. This also covers the meaningful code-constant edge without adding a redundant low-level hash test. |

Use existing setup and writers/readers. Keep each lifecycle independently collectable. Do not hardcode hashes, rely on Python versions necessarily producing different IDs for a particular function, or make a different function name the only reason an identity changes. Restore any test mutation through pytest's scoped fixtures. New fixtures/helpers must be fully annotated and deterministic.

If existing nearby coverage added concurrently already owns one of these exact contracts, reuse it rather than adding a duplicate. If a production fix becomes necessary, retain a failing-before/passing-after regression for that defect and update the direct consumer matrix via LSP references before editing exported or cross-module behavior.

## 5. CPU, typing, and smoke verification

All commands run only after a later implementation request, from the implementation worktree unless explicitly isolated. `PY313`, `RUN`, `DIST`, `WHEEL`, and provider/build environment paths below are concrete absolute paths selected and recorded for that run; they are not new repository configuration fields. `RUN` is outside committed historical evidence.

### 5.1 Environment and dependency proof

- Create ordinary CPython 3.13 environments; inspect the actual version/build and GIL status once. Use the intended CPU backend and x64 configuration through existing configuration/environment conventions.
- Resolve the root with test/QA dependencies and each relevant supported extra combination. Do not use `--all-extras`: `fmi` and `privacy-jax` are intentionally incompatible and remain separate.
- Confirm the final locks are current and that installed JAX/JAXlib/plugin versions match the intended resolution. Do not accept an editable package being importable as proof that wheel metadata admits 3.13.
- Build a real main wheel, install it in a fresh environment outside the checkout with repository uv source overrides disabled, and confirm the import resolves to that wheel.
- Verify the same wheel is rejected under 3.12 by normal `Requires-Python` admission, without `--ignore-requires-python`; no permanent negative-test subprocess matrix is needed for this metadata contract.
- Build/install the meshcore wheel separately; do not make an optional provider mandatory for the base CPU installation.

Existing installed-wheel gate:

```sh
"$PY313" -m tools.check_installed_typing
```

This builds its own wheel and checks consumer fixtures outside the repository, including the PEP 561 marker. Capture its reported interpreter. Do not also rerun it with `--python 3.13` merely to restate the same evidence. A retained runtime-smoke environment is separate because this command removes its temporary environments on exit; avoid adding a permanent CLI abstraction solely to share that temporary directory.

### 5.2 Direct runtime smoke, not just tests

Use a throwaway, fully typed driver outside the checkout against the installed wheel:

1. Import the lazy root/configuration surface and verify the expected startup/backend behavior.
2. Exercise `geometry.radial_closest_point` using the existing `tests/integration/test_configuration_startup.py` circle case: point `[2, 0]`, center `[0, 0]`, radius `1`, expected closest point `[1, 0]` and normal coordinate `1`, with explicit float64 arrays and scientific geometry identity.
3. Construct an opted-in StrictModule using native nominal dimensions, run a checked callable eagerly and through a stable JIT entry point, and exercise a derivative. Compare numerical values to a simple independent analytic result.
4. Refuse wrong dtype/rank/shared extent and a wrong nominal argument; verify the owning error category. Conversion remains explicit and occurs once; no hidden device synchronization or returned-value validation is introduced.
5. Exercise one small real CAD/OCP operation through the existing geometry tests/provider surface, because CAD is a base compiled dependency, not an optional import that can be ignored.
6. Perform the section 4 artifact experiment with actual writer/reader calls. Record acceptance/refusal and IDs, not merely successful file creation or nonempty outputs.

Keep the smoke driver and temporary artifacts out of package code and remove scaffolding after retaining the evidence. A smoke that only imports modules, constructs an object, or asserts no exception is insufficient.

### 5.3 Focused existing test families

Run ordinary focused tests with `-n auto` in one CPU worker pool. The exact selected nodes and dependencies must be reconciled with changes since the latest merge into `dev`; do not repeat passed nodes on unchanged relevant inputs.

| Test surface | Why included |
| --- | --- |
| `tests/unit/typing/` | Forms, conversion, scope rollback, Strict construction, descriptors, forward references, tracing, custom derivatives, weak caches, scoped x64, and first-call concurrency. |
| `tests/unit/test_module_fields.py`, `tests/unit/test_contract_declarations.py` | Package-wide field/abstract declaration ownership and runtime-resolvable effective checked plans. |
| `tests/typing/test_typecheck.py`, `tests/typing/cases/checked_signatures.py` through its owning gate | Positive/negative static expectations, PEP 695 generics, descriptor signatures, and unused-suppression detection under the new target. Do not execute negative typing fixtures as runtime scripts. |
| `tests/integration/test_audit_type_alias_syntax.py` | AST-based selector/contract tooling continues to interpret the canonical syntax. |
| `tests/unit/test_configuration_resolution.py`, `test_configuration_records.py`, `test_configuration_environment.py`, `test_configuration_runtime.py`; `tests/integration/test_configuration_startup.py` | Annotation-driven configuration, canonical identities/precedence, fresh-process startup, and no unintended global numerical policy. |
| `tests/unit/test_identity.py`, `tests/unit/test_array_roles.py`, `tests/unit/test_benchmark_runtime.py` | Callable/compiler identity, hidden state, failure admission, and honest interpreter-dependent provenance. |
| `tests/unit/test_array_archive_security.py` | Existing bounded archive, inventory, integrity, and data-only decoding refusal. |
| `tests/unit/solver/test_functional_training_runtime.py` checkpoint cases and new code-bound-map case | Exact continuation, parameter/plan admission, and the callback-bound restore boundary. Include the existing KFAC checkpoint case because it consumes the same loader. |
| `tests/unit/nn/test_operator_training.py` checkpoint cases and new plain-function case | Optimizer/RNG/policy restoration and executable binding. If the checkpoint owner changes, also include its LSP-identified task-bundle and geometry-neural-operator consumers. |
| `tests/unit/ml/test_artifacts.py`, `tests/unit/atomistic/test_model_artifact.py` relevant round trips/refusals | Portable registered recipes versus code-bound identities. |
| `tests/integration/test_coupled_restart.py` portable-array/rebuilt-owner cases; `tests/integration/test_meshfree_runtime_restart.py::test_stale_identity_refuses_before_any_checkpoint_value_is_used` | Normal data restore and fail-closed stale runtime/source/program ownership. Run topology-specific cases only in their dedicated subprocesses. |
| `tests/unit/test_cuda_build_tools.py`, `test_cuda_wheel_qualification.py`, `test_cuda_mpi_qualification.py`, `test_cuda_environment.py`, `test_cuda_native_loader.py`, `test_elf_dependencies.py`; `tests/unit/geometry/test_meshcore_loader.py`; `tests/integration/test_cuda_import_contract.py` | Packaging/qualification refusal semantics, CPU-only import behavior, and active-environment discovery. Separate live CUDA/provider nodes from the CPU shard. |

Illustrative first focused invocation, after the target environment is installed:

```sh
"$PY313" -m pytest -n auto \
  tests/unit/typing \
  tests/unit/test_module_fields.py \
  tests/unit/test_contract_declarations.py \
  tests/typing/test_typecheck.py \
  tests/integration/test_audit_type_alias_syntax.py \
  tests/unit/test_identity.py \
  tests/unit/test_array_roles.py
```

The actual node list is owned by the integration record, not by a broad `-k` expression that accidentally drops a required consumer. Live CUDA cases include `test_cuda_provider_probe_order_is_independent`, `test_cuda_provider_dependency_mismatch_refuses_before_import`, and runtime-major/device checks; defer those to the matching provider environment rather than treating a CPU skip as coverage.

For this migration, replacing the interpreter and the global checker/installation target is a cross-cutting configuration/runtime change. The final CPU integration obligation is the **full default CPU-eligible suite**, not every optional real-provider/topology campaign. Run the remaining eligible nodes after subtracting still-valid focused evidence, in one `-n auto` invocation/list; if a later shared fix invalidates focused evidence, include those nodes again with that reason recorded. Do not run a broad 3.12 baseline suite. Missing optional providers may skip only at their canonical boundaries; installed-but-broken providers and missing required qualification evidence are failures.

### 5.4 Static, declaration, and generated consistency gates

```sh
"$PY313" tools/check_typing.py check
"$PY313" tools/audit_contract_candidates.py --signatures
"$PY313" tools/audit_selectors.py
"$PY313" -m tools.check_public_api_manifest
```

- Require zero ty/annotation diagnostics and no unused suppressions.
- Require zero redundant checked-boundary guards and zero selector findings. `audit_selectors.py` prints its count; the evidence reviewer must not infer zero findings from exit status alone.
- Use configured Ruff formatting/linting for changed Python files only; no repository-wide restyling or public callable-source edits as cleanup.
- Public exports/capability data should remain unchanged. If an approved compatibility fix changes them, regenerate after the last source change and run only the checks invalidated by that output plus the final affected matrix.
- A metadata/interpreter-only change does not justify another full numerical qualification campaign for unchanged algorithms. Any actual shared-runtime, numerical, persistence, or provider fix expands the matrix according to its semantic consumers.

## 6. Generator and generated artifact verification

The standalone generator already has a canonical `OUTPUT` binding and a `main()` that writes only when content differs. Use a throwaway driver in its locked 3.13 environment that imports that module, redirects `OUTPUT` to a task-owned scratch path, and invokes the real `main()`.

- Compare the complete generated bytes/digest with `phydrax/linalg/_generated_exponential_taylor_thresholds.py`.
- Preserve all 53 tolerance rows, 55 Taylor degrees, finite-series root enclosure assertions, downward float64 rounding, and the current finite-series-versus-infinite-tail distinction.
- Do not shorten the scientific generation or relax its enclosure checks to make the interpreter migration pass.
- Do not add a permanent alternate generator API, schema generation, or new benchmark for this one-off comparison.
- A numerical/table difference is a blocker to investigate against interpreter/dependency inputs, not an automatically accepted regenerated file.
- Keep the canonical generated file untouched when output matches. Keep any investigation output separately if it does not.

## 7. Native packaging and platform qualification

### 7.1 Preserve platform boundaries

| Installation / target | Required evidence |
| --- | --- |
| Base CPU wheel, macOS ARM64 and supported Linux CPU deployment | Real wheel installation and numerical/import/typing smoke. Other claimed release platforms need their own install smoke; metadata resolution does not prove execution there. |
| Optional meshcore | Matching platform wheel, actual C ABI load, numerical geometry cases, and missing/mismatched-library refusal. No GPU required. |
| `cuda13`, Linux x86-64 | CUDA plugin/PJRT, intent marker, exact companion, NVIDIA dependencies, and real CUDA runtime qualification. |
| `cuda13`, Linux aarch64 | Preserve JAX CUDA resolution without the x86-64-only Phydrax companion. Do not advertise companion support not currently shipped. |
| `cuda13`, Darwin ARM64 / Windows x86-64 | Resolver preserves the intent marker without CUDA plugin/NVIDIA/companion packages; this is not GPU support on those hosts. |
| MPI | Rebuild/provision the site-matched Python/JAX/mpi4py/mpi4jax/MPI/UCX stack and exercise the real dedicated qualifier. Ordinary package wheels do not prove CUDA-aware transport or multi-host RDMA. |

### 7.2 Companion build and metadata gate

Build meshcore using its existing scikit-build/CMake route in an isolated build directory. Install the main wheel with the matching meshcore wheel from outside the checkout and run `tests/unit/geometry/test_meshcore_loader.py` and `tests/unit/geometry/test_meshcore.py`, then a direct native geometry smoke. Verify that the provider actually loaded; a missing-provider skip is not a successful optional-companion qualification.

CUDA release command shape, with actual absolute paths selected for the target host:

```sh
"$PY313" tools/build_cuda_wheels.py \
  --python "$CUDA_BUILD_PYTHON" \
  --wheelhouse "$WHEELHOUSE" \
  --build-dir "$CUDA_BUILD_DIR" \
  --download-dir "$CUDA_DOWNLOAD_CACHE" \
  --jobs 2 \
  --container podman

"$PY313" -m tools.check_cuda_extras \
  --work-dir "$RUN/cuda-extras" \
  --evidence "$RUN/cuda-extras/evidence.json" \
  --companion-wheelhouse "$WHEELHOUSE" \
  --linux-tag manylinux_2_28
```

`--jobs 2` is a conservative example, not a fixed resource policy; allocate from the task's global CPU/memory budget. The build interpreter must exist inside the selected build/container environment with the declared build dependencies. Use the existing supported container/runtime configuration rather than assuming a host path is visible inside it.

Leave `--architectures` and `--components` unspecified for the full release defaults. Do not pass `native` or a single GPU architecture and label that wheel portable. Retain inspected architecture coverage, ELF/auditwheel results, build manifest, and wheel digest. The metadata gate builds the main/marker wheels itself and resolves its four platform cases. A supplied companion wheelhouse is required even if the command is run on a CPU host.

The main and companion Python requirements must all admit 3.13 while companion distribution/embedded release versions match the main distribution. Retain `py3` API tags for the C ABI companions and their actual platform tags; do not retag binaries by hand.

### 7.3 Real CUDA qualification

```sh
CUDA_VISIBLE_DEVICES=0,1 "$PY313" tools/qualify_cuda_wheels.py \
  --venv "$RUN/cuda13-py313" \
  --required-gpus 2 \
  --phydrax-wheel "$WHEEL" \
  --wheelhouse "$WHEELHOUSE" \
  --hostile-library-path "$HOSTILE_CUDA_LIBRARY_PATH" \
  --linux-tag manylinux_2_28 \
  --evidence "$RUN/cuda13-wheel-qualification.json"
```

Use the actual visible-device inventory and a real hostile library-path scenario. The example requests two GPUs because concurrent/device-isolation claims require real coverage; a one-GPU run must not be reported as multi-GPU qualification. The wheelhouse contains both the companion and CUDA-major marker.

Omit `--extras-gate-work-dir` when the same unchanged wheels/metadata already passed section 7.2; otherwise use it instead of a separate duplicate metadata-gate run. Omit `--reuse-venv` for the fresh install. The qualifier owns its serial `-n 0` provider suite and coverage adjudication.

Required existing families include native loader/lifecycle, cuDSS, AmgX, QUDA/lattice, and backend contracts. Keep normal/hostile teardown, per-device execution, simultaneous-use, mismatch-before-import, and unsupported-topology refusal checks. Missing, skipped, or partial required device evidence is not a pass.

Additionally run the repaired `test_cuda_wheel_dependencies_resolve_inside_wheels` in this installed CUDA environment: that node is not established by a CPU shard or by changing a site-packages literal. Run the repaired pre-import mismatch node through the qualifier's existing backend-contract suite; do not duplicate it when its evidence remains valid.

### 7.4 MPI qualification

```sh
"$PY313" tools/qualify_cuda_mpi.py \
  --python "$MPI_ENV/bin/python" \
  --mpi-prefix "$OPENMPI_PREFIX" \
  --ucx-prefix "$UCX_PREFIX" \
  --visible-devices 0,1 \
  --sizes 8 1024 65536 1048576 16777216 67108864 \
  --repeats 31 \
  --maximum-buffer-bytes 67108864 \
  --timeout 900 \
  --output "$RUN/cuda-mpi-py313.json"
```

The tool accepts an interpreter path, derives the rank count from distinct visible devices, and invokes its existing serial/distributed scenarios. Supply its real GNU `env`, MPI, UCX, and CUDA-aware bridge prerequisites. Preserve both host-staged and CUDA-direct routes, precisions, per-rank identity/evidence, communicator lifecycle, and resource bounds. This command qualifies the existing single-host scope, not multi-host networking or zero-copy/RDMA claims. Profiling, if used, remains separate from timing runs.

### 7.5 Native source changes and release scope

- Unchanged C++/CUDA kernels do not require a redundant native algorithm test campaign merely because `Requires-Python` changed. Rebuilt wheel installation and Python/native consumer qualification remain required.
- If meshcore C++ changes, enable its tests explicitly and run all six existing CTest targets; the wheel configuration itself disables them.
- If CUDA source/build semantics change, rerun the affected build/architecture audits and real-GPU qualification. Do not invent a CUDA CTest step; no such suite was found.
- Publish only after the matching required gates pass, preserving marker/companion/main release ordering. No implicit PR, merge, commit, or publication is authorized by this plan.
- If required hardware/toolchains are unavailable, finish reachable work and report the precise unqualified route and missing prerequisite. Do not substitute mocked providers, infer success from wheels, or call the complete deployment migration qualified.

## 8. Phase-separated benchmark comparison

Use two existing drivers; do not add a Python-version benchmark framework. Capture the matched 3.12 baseline before the metadata cutover, then the 3.13 candidate after correctness passes. Run sequentially on the same otherwise-idle hardware with the same shapes, dtypes, numerical policy, dependency versions where possible, and explicit cache policy.

```sh
JAX_PLATFORMS=cpu "$PY313" -m benchmarks.configuration \
  --package-root "$PY313_SITE_PACKAGES" \
  --scenario root \
  --scenario configuration \
  --repeats 20 \
  --output "$RUN/configuration-py313.json"

JAX_PLATFORMS=cpu JAX_ENABLE_X64=true \
  "$PY313" -m benchmarks.exterior_calculus \
  --sizes 8 16 \
  --repeats 20 \
  --output "$RUN/exterior-calculus-py313.json"
```

Run the analogous baseline with the reference interpreter and separate paths. `--package-root` selects the installed package for the configuration driver's fresh-process cold-import children; the parent resolution/scope measurements use the current checkout under that interpreter, so record both paths and keep their source synchronized. The exterior-calculus benchmark runs against the chosen checkout. Do not claim its command proves installed-wheel isolation; the separate wheel smoke owns that contract.

The first driver separates cold root/configuration import, configuration resolution, and scope costs. The second separates preparation/IR lowering, executable lowering, compilation, first synchronized execution, and prepared reuse with changing coefficient inputs. It also records compiler memory/cost evidence and retained logical bytes and checks an independent Fourier oracle.

Acceptance is correct behavior and honest evidence, not an invented speedup threshold. Report distributions and any material regressions; investigate unexplained changes without changing algorithms or tolerances to obtain a favorable comparison. Python's runtime version and the changed source/lock identity should differ in the records. Do not run a same-environment comparison gate by falsifying either identity, overwrite committed 3.12 measurements, or attribute JAX/dependency changes solely to Python.

## 9. Typing boundaries that must not change

- Python 3.13 adds optional type-parameter defaults, `TypeIs`, `ReadOnly`, protocol introspection, and nested `ClassVar`/`Final`; none is needed to retarget the project.
- `type` aliases and generic parameter syntax already exist in Python 3.12. Do not convert existing scientific functions/classes solely for spelling or novelty.
- Default deferred evaluation of ordinary annotations belongs to Python 3.14, not 3.13. Do not introduce `annotationlib`, remove future annotations, or rely on 3.14 semantics.
- `TypeForm`, forward-reference evaluation, and annotation helpers make `typing_extensions` an active dependency even on 3.13.
- Native forms remain statically their base JAX/NumPy/metadata types and dynamically Phydrax contracts. There is no new static proof of nominal scientific axis identity or array shape.
- Checked inputs retain original values, conversion ownership, descriptor signatures, static-only unions/protocols, trace-time execution, and custom-derivative placement. Do not add return checking, host transfers, numeric-value checks, or callback invocation.
- Strict field order, PyTree leaves, constructor-bypassing restore validation, and source/compiler identity protections remain unchanged. A checker diagnostic is not permission to reorder fields, mark array-bearing callbacks static, or erase types with `Any`/casts.
- If a pinned checker/library defect appears, establish the minimal defect and use the existing line-local suppression/upgrade policy. Do not replace ty, add a second checker, or weaken all diagnostics.

## 10. Integration discipline and evidence validity

- One integration owner maintains the final file/contract/test map. Parallelize only genuinely independent work with distinct file ownership; the three small target-literal edits do not need three agents.
- Use one CPU pytest worker pool at a time. Native builds have isolated build directories; provider/topology tests use their owning subprocesses and real resources. Benchmarks have no competing builds, tests, or agents.
- Reuse passed evidence while all relevant source, test, config, dependency, generated-input, backend, hardware, and environment inputs remain unchanged. The version migration intentionally changes some of these inputs; explain exactly which results it invalidates.
- Do not use reruns to dismiss flaky/statistical failures. Re-run only exact failed nodes after a relevant fix, then extend coverage.
- A public/cross-module compatibility fix requires LSP references before editing, direct consumer tests, and the relevant docs/generated consistency checks. The reviewed `probe_backend` references span MPI, solver providers, and robotics, so changing that owner would expand scope substantially; the planned fixture repair avoids doing so.
- A source audit showing no known removed stdlib APIs is not runtime proof. A wheel resolving is not a native/provider qualification. A skipped test is not passed coverage.
- Keep historical benchmark/provider records intact. Author new migration notes and actual evidence separately, then remove throwaway drivers after recording their exact scenarios/results.

## 11. Completion checklist

- [ ] Implementation remains on the authorized worktree; concurrent user changes are preserved.
- [ ] Root, native companion, generator, development pin, ty, and installed/qualification target policies consistently select ordinary CPython 3.13 only.
- [ ] Root/generator locks are regenerated canonically; the existing JAX requirement/lock mismatch is resolved and documented separately.
- [ ] A clean built-wheel 3.13 install and numerical smoke pass outside the checkout; normal 3.12 installation is refused.
- [ ] The two environment-dependent tests discover the actual target libraries/extensions and retain their original dependency/refusal contracts.
- [ ] Typing, declaration, selector, signature, and generated-export checks meet their existing zero-diagnostic/finding contracts.
- [ ] Focused and final CPU-eligible coverage is complete, with no repeated valid work or unexplained required skips.
- [ ] Portable 3.12-produced artifacts are actually consumed under 3.13; code-bound checkpoint outcomes follow exact recomputed identity admission, with no relabeling or fallback.
- [ ] New meaningful hidden-state/checkpoint regression cases pass, or demonstrably equivalent existing coverage is recorded.
- [ ] The standalone generator executes under 3.13 and produces the exact canonical table.
- [ ] Meshcore wheel and real consumer qualification pass; claimed CUDA/MPI routes have real matching hardware/provider results, not metadata-only evidence.
- [ ] Baseline/candidate benchmarks preserve correctness and report separate phases, capacities, memory, interpreter/dependency identity, and any attribution limits.
- [ ] README, installation/index, typing guide, four cookbook prerequisites, changelog, and this implementation record agree with the delivered scope.
- [ ] Public scientific APIs, identity algorithms, static representation, generated capability claims, persistence formats, historical evidence, and external interpreter profiles are unchanged except for separately justified and approved fixes.

## 12. Primary references

- [Python 3.13 release and porting notes](https://docs.python.org/3.13/whatsnew/3.13.html)
- [CPython bytecode portability contract](https://docs.python.org/3.13/library/dis.html)
- [Python 3.14 deferred annotation boundary](https://docs.python.org/3.14/whatsnew/3.14.html#pep-649-pep-749-deferred-evaluation-of-annotations)
- [Ruff target inference](https://docs.astral.sh/ruff/configuration/#inferring-the-python-version)
- [JAX Python support policy](https://docs.jax.dev/en/latest/deprecation.html)
- [JAXlib 0.12.0 release files](https://pypi.org/project/jaxlib/0.12.0/#files)
- [CUDA 13 JAX plugin 0.12.0 release files](https://pypi.org/project/jax-cuda13-plugin/0.12.0/#files)
- [CAD/OCP 8.0.1.0.0 release files](https://pypi.org/project/cadquery-ocp-novtk/8.0.1.0.0/#files)
- [NumPy 2.5.3 release files](https://pypi.org/project/numpy/2.5.3/#files)
- [SciPy 1.18.1 release files](https://pypi.org/project/scipy/1.18.1/#files)
- [h5py 3.16.0 release files](https://pypi.org/project/h5py/3.16.0/#files)
- [Python-flint 0.9.0 support information](https://pypi.org/project/python-flint/0.9.0/)

Local ownership/policy references: [typing guide](../guides_typing.md), [testing guide](../guides_testing.md), [existing runtime-contract identity protections](runtime-contracts.md), [CUDA installation/release contract](single-step-cuda-install.md), and [native CUDA build instructions](../../native/cuda/README.md).

## 13. Implementation record

Environment: macOS 15 arm64, uv-managed CPython 3.13.11 (GIL enabled) and 3.12.8 reference, JAX/JAXlib 0.12.0, NumPy 2.5.3, SciPy 1.18.1, CPU backend. No NVIDIA GPU, container runtime, or CUDA-aware MPI stack was available.

### Delivered changes beyond the planned metadata, tool, test, and documentation edits

| Change | Reason |
| --- | --- |
| `s2fft>=1.5.0,<2` | s2fft 1.4.0 publishes `cp312-abi3` wheels whose extension is named `_s2fft.cpython-312-*.so`, which CPython 3.13 cannot import; resolvers still select them. 1.5.0 ships `_s2fft.abi3.so`. A scan of the 3.13 environment found no other foreign-interpreter extension. |
| `modepy.**` in `tool.ty.analysis.replace-imports-with-any` | `tools/generate_cubature_data.py` imports ModePy from an explicit `--modepy-root`; the typing gate only passed when ModePy happened to be installed. |
| Structural `AxisArray.__repr__` | Functional discretization bundle IDs fingerprint term representations, which embedded `AxisArray` object addresses; functional checkpoints could not resume in any new process, independent of the interpreter. Regression: fresh-subprocess bundle-ID equality in `tests/unit/solver/test_integral_functional.py`. |
| Generator import spacing | The committed table had been formatted after generation; the generator now emits the canonical spacing, and 3.13 regeneration is byte-identical (SHA-256 `d6ab2949…818a`). |

The root `uv.lock` is git-ignored local state; it was regenerated in the worktree (JAX/JAXlib unified at 0.12.0, s2fft 1.5.0, vtk 9.3.1 → 9.7.1 because 9.3.1 has no 3.13 wheel). The standalone generator lock is tracked and was regenerated.

### Evidence

| Scenario | Result |
| --- | --- |
| Built wheel metadata | `Requires-Python: <3.14,>=3.13`; installation into a fresh 3.12 environment is refused by the resolver. |
| Fresh 3.13 wheel install outside the checkout | Lazy root import leaves backends uninitialized; checked/Strict contracts pass eager, JIT, and gradient execution with dtype/extent/rank refusals; radial closest point exact; OCP box import measure 6 with unit boundary-gradient norm; s2fft/spectral import. |
| 3.12-written artifacts consumed by 3.13 | Array archive, portable ML model, and atomistic MACE model restore with identical IDs and bitwise-equal predictions/energy. Operator checkpoint without static callables restores. A non-code-bound functional checkpoint resumes to step 2 bitwise-equal to the 3.12 uninterrupted run (after the `AxisArray` fix). Plain-function semantic IDs all differ between 3.12 and 3.13; the code-bound functional and operator checkpoints are refused with their existing identity-mismatch errors. |
| New regressions | Nested-code hidden-global refusal (2 cases), code-derived relaxation-map checkpoint binding, and nested-constant static-callable binding pass, and each fails under a deliberately injected fault. The synthetic `cpython-312` provider filename is undiscoverable on 3.13 while the active suffix is discovered. |
| Static gates | `tools/check_typing.py check`: zero diagnostics. Signature audit: no redundant checked-boundary guards. Selector audit: 0 findings. Public API manifest: unchanged. `tools.check_installed_typing` (default target 3.13): 3 fixture files pass. |
| Focused CPU tests | Typing, module-field, contract-declaration, ty fixture, alias-audit, identity, array-role, configuration-resolution, and startup families: 160 passed. Delegated determinism fix: 13 focused tests passed. |
| Meshcore | 3.13 wheel `py3-none-macosx_15_0_arm64`, `Requires-Python <3.14,>=3.13`; loader and native geometry tests: 10 passed, none skipped. |

Per the user's instruction, only the minimal affected test set above was run; the full CPU suite was not.

### Benchmarks (CPU, 20 repeats, medians; matched JAX/NumPy/SciPy)

| Phase | 3.12.8 | 3.13.11 |
| --- | --- | --- |
| Cold `import phydrax` | 0.363 s | 0.422 s |
| Cold `from phydrax import config` | 0.394 s | 0.403 s |
| Configuration resolve, 256 weights | 1.30 ms | 1.28 ms |
| Exterior calculus 8×8: realization preparation / compilation / prepared reuse | 2.16 s / 28.7 ms / 66.3 µs | 2.33 s / 34.7 ms / 70.0 µs |
| Exterior calculus 16×16: realization preparation / compilation / prepared reuse | 1.35 s / 26.7 ms / 68.8 µs | 1.41 s / 27.5 ms / 61.8 µs |

Compiler flops, bytes accessed, argument/output/temporary bytes, and retained logical bytes are identical; both runs pass the Fourier-symbol oracle. Cold phases are single samples per case except the import distributions. Dependency differences (s2fft 1.4.0 vs 1.5.0, vtk) are outside the measured paths; the root-import increase is an observation, not attributed further.

### Not qualified here

- CUDA 13 companion build, `tools/check_cuda_extras.py`, `tools/qualify_cuda_wheels.py`, and the repaired `test_cuda_wheel_dependencies_resolve_inside_wheels` / `test_cuda_provider_dependency_mismatch_refuses_before_import` nodes require Linux x86-64 with NVIDIA GPUs (both nodes skip on this host).
- `tools/qualify_cuda_mpi.py` requires a CUDA-aware MPI/UCX stack and two GPUs.
- Release publication and lockstep version advancement were not performed.
