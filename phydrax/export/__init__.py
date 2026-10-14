from importlib import import_module
from typing import Any, TYPE_CHECKING

from .._deferred_exports import deferred_exports

#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from ._array_archive import __all__ as _array_archive_all


_FACADE_EXPORT_MODULES = ("._array_archive", "._atomistic")


def _facade_getattr(name: str) -> Any:
    for module_name in reversed(_FACADE_EXPORT_MODULES):
        module = import_module(module_name, __package__)
        if name in module.__all__:
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _facade_dir() -> list[str]:
    return sorted(set(globals()) | set(__all__))


# Re-exports load on first use, so importing one owner never imports the
# whole package and its import cycles.
_DEFERRED_EXPORTS = {
    "HostInferenceAdapter": ".._external_runtime",
    "complex_coefficients_to_frame": "._complex_parameters",
    "ComplexImportPolicy": "._complex_parameters",
    "ComplexInterchangeEntry": "._complex_parameters",
    "ComplexInterchangeSemantics": "._complex_parameters",
    "ComplexInterchangeState": "._complex_parameters",
    "export_complex_parameters": "._complex_parameters",
    "frame_coefficients_to_complex": "._complex_parameters",
    "import_complex_parameters": "._complex_parameters",
    "ComplexOptimizerInterchangeState": "._complex_training",
    "ComplexOptimizerRouteKind": "._complex_training",
    "ComplexOptimizerStateGroup": "._complex_training",
    "ComplexOptimizerStateLayout": "._complex_training",
    "ComplexTrainingInterchangeState": "._complex_training",
    "export_complex_training_state": "._complex_training",
    "import_complex_training_state": "._complex_training",
    "ImportedComplexTrainingState": "._complex_training",
    "prepare_complex_optimizer_state_layout": "._complex_training",
    "prepare_complex_training_interchange": "._complex_training",
    "PreparedComplexTrainingInterchange": "._complex_training",
    "read_complex_training_checkpoint": "._complex_training",
    "RNGInterchangeState": "._complex_training",
    "write_complex_training_checkpoint": "._complex_training",
    "save_compressible_kinetic_iree": "._compressible_kinetic_iree",
    "discrete_velocity_iree_availability": "._discrete_velocity_iree",
    "DiscreteVelocityIREEContract": "._discrete_velocity_iree",
    "DiscreteVelocityIREEExportBundle": "._discrete_velocity_iree",
    "DiscreteVelocityIREEExportMode": "._discrete_velocity_iree",
    "prepare_discrete_velocity_iree_contract": "._discrete_velocity_iree",
    "save_discrete_velocity_iree": "._discrete_velocity_iree",
    "IREEArtifactManifest": "._iree",
    "IREEExecutable": "._iree",
    "IREEExportPolicy": "._iree",
    "IREEExportResult": "._iree",
    "load_iree": "._iree",
    "save_iree": "._iree",
    "lattice_boltzmann_iree_availability": "._lattice_boltzmann_iree",
    "LatticeBoltzmannIREEContract": "._lattice_boltzmann_iree",
    "LatticeBoltzmannIREEExportBundle": "._lattice_boltzmann_iree",
    "LatticeBoltzmannIREEExportMode": "._lattice_boltzmann_iree",
    "prepare_lattice_boltzmann_iree_contract": "._lattice_boltzmann_iree",
    "save_lattice_boltzmann_iree": "._lattice_boltzmann_iree",
    "NeutralAdapterBoundary": "._neutral",
    "NeutralFieldSchema": "._neutral",
    "NeutralGeometrySchema": "._neutral",
    "NeutralMaterialSchema": "._neutral",
    "NeutralPointCloudSchema": "._neutral",
    "NeutralSchemaKind": "._neutral",
    "load_onnx": "._onnx",
    "OnnxExportResult": "._onnx",
    "save_onnx": "._onnx",
}


__getattr__, __dir__ = deferred_exports(
    __name__, globals(), _DEFERRED_EXPORTS, fallback=_facade_getattr, listing=_facade_dir
)


if TYPE_CHECKING:
    from .._external_runtime import HostInferenceAdapter
    from ._complex_parameters import (
        complex_coefficients_to_frame,
        ComplexImportPolicy,
        ComplexInterchangeEntry,
        ComplexInterchangeSemantics,
        ComplexInterchangeState,
        export_complex_parameters,
        frame_coefficients_to_complex,
        import_complex_parameters,
    )
    from ._complex_training import (
        ComplexOptimizerInterchangeState,
        ComplexOptimizerRouteKind,
        ComplexOptimizerStateGroup,
        ComplexOptimizerStateLayout,
        ComplexTrainingInterchangeState,
        export_complex_training_state,
        import_complex_training_state,
        ImportedComplexTrainingState,
        prepare_complex_optimizer_state_layout,
        prepare_complex_training_interchange,
        PreparedComplexTrainingInterchange,
        read_complex_training_checkpoint,
        RNGInterchangeState,
        write_complex_training_checkpoint,
    )
    from ._compressible_kinetic_iree import save_compressible_kinetic_iree
    from ._discrete_velocity_iree import (
        discrete_velocity_iree_availability,
        DiscreteVelocityIREEContract,
        DiscreteVelocityIREEExportBundle,
        DiscreteVelocityIREEExportMode,
        prepare_discrete_velocity_iree_contract,
        save_discrete_velocity_iree,
    )
    from ._iree import (
        IREEArtifactManifest,
        IREEExecutable,
        IREEExportPolicy,
        IREEExportResult,
        load_iree,
        save_iree,
    )
    from ._lattice_boltzmann_iree import (
        lattice_boltzmann_iree_availability,
        LatticeBoltzmannIREEContract,
        LatticeBoltzmannIREEExportBundle,
        LatticeBoltzmannIREEExportMode,
        prepare_lattice_boltzmann_iree_contract,
        save_lattice_boltzmann_iree,
    )
    from ._neutral import (
        NeutralAdapterBoundary,
        NeutralFieldSchema,
        NeutralGeometrySchema,
        NeutralMaterialSchema,
        NeutralPointCloudSchema,
        NeutralSchemaKind,
    )
    from ._onnx import load_onnx, OnnxExportResult, save_onnx


__all__ = [
    "AtomisticExportRoute",
    "AtomisticExportFailurePolicy",
    "AtomisticExportRefusal",
    "AtomisticIREEContract",
    "AtomisticIREEEvaluation",
    "AtomisticIREEExportBundle",
    "LoadedAtomisticIREE",
    "load_atomistic_iree",
    "prepare_atomistic_iree_contract",
    "save_atomistic_iree",
    "ComplexImportPolicy",
    "ComplexInterchangeEntry",
    "ComplexInterchangeSemantics",
    "ComplexInterchangeState",
    "ComplexOptimizerInterchangeState",
    "ComplexOptimizerRouteKind",
    "ComplexOptimizerStateGroup",
    "ComplexOptimizerStateLayout",
    "ComplexTrainingInterchangeState",
    "DiscreteVelocityIREEContract",
    "DiscreteVelocityIREEExportBundle",
    "DiscreteVelocityIREEExportMode",
    "HostInferenceAdapter",
    "ImportedComplexTrainingState",
    "PreparedComplexTrainingInterchange",
    "RNGInterchangeState",
    "IREEArtifactManifest",
    "IREEExecutable",
    "IREEExportPolicy",
    "IREEExportResult",
    "LatticeBoltzmannIREEExportMode",
    "LatticeBoltzmannIREEContract",
    "LatticeBoltzmannIREEExportBundle",
    "NeutralAdapterBoundary",
    "NeutralFieldSchema",
    "NeutralGeometrySchema",
    "NeutralMaterialSchema",
    "NeutralPointCloudSchema",
    "NeutralSchemaKind",
    "discrete_velocity_iree_availability",
    "complex_coefficients_to_frame",
    "export_complex_parameters",
    "export_complex_training_state",
    "import_complex_training_state",
    "prepare_complex_optimizer_state_layout",
    "prepare_complex_training_interchange",
    "read_complex_training_checkpoint",
    "write_complex_training_checkpoint",
    "frame_coefficients_to_complex",
    "load_iree",
    "lattice_boltzmann_iree_availability",
    "prepare_lattice_boltzmann_iree_contract",
    "prepare_discrete_velocity_iree_contract",
    "save_discrete_velocity_iree",
    "save_compressible_kinetic_iree",
    "save_lattice_boltzmann_iree",
    "import_complex_parameters",
    "load_onnx",
    "OnnxExportResult",
    "save_onnx",
    "save_iree",
]
__all__ += [name for name in _array_archive_all if name not in __all__]
