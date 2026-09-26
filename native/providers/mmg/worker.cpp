// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Persistent Mmg library-API worker for phydrax.meshing.providers._mmg.
//
// Operations "mmg2d" (planar triangles), "mmgs" (surface triangles in 3D) and
// "mmg3d" (tetrahedra) exchange 0-based int64/float64 arrays and run one Mmg
// program: "remesh" (MMG*_mmg*lib), "levelset" (MMG*_mmg*ls) or "lagrangian"
// (MMG*_mmg*mov). Region references ride on cells, boundary references on
// edges (planar and surface meshes) or triangles (volume meshes). Declared
// vertex fields are transferred by P1 interpolation in the source (for
// Lagrangian motion: the source displaced by Mmg's own motion).
//
// With PHYDRAX_MMG_WITH_PARMMG the executable is the collective ParMmg worker:
// "mmg3d" remeshing runs through PMMG_parmmglib_centralized and every rank
// returns its partition as the exchange part "rank-<r>".

#include "locator.hpp"
#include "phydrax_worker.hpp"

#ifdef PHYDRAX_MMG_WITH_PARMMG
#include <parmmg/libparmmg.h>
#endif
#include <mmg/common/mmgversion.h>
#include <mmg/mmg3d/libmmg3d.h>
#ifndef PHYDRAX_MMG_WITH_PARMMG
#include <mmg/mmg2d/libmmg2d.h>
#include <mmg/mmgs/libmmgs.h>
#endif

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

namespace json = phydrax::json;
namespace exchange = phydrax::exchange;
using phydrax::worker::Failure;

enum class Backend { mmg2d, mmgs, mmg3d };
enum class Program { remesh, levelset, lagrangian };

Backend parse_backend(std::string const& operation) {
  if (operation == "mmg2d") return Backend::mmg2d;
  if (operation == "mmgs") return Backend::mmgs;
  if (operation == "mmg3d") return Backend::mmg3d;
  throw Failure("invalid_request", "Unknown Mmg operation '" + operation + "'");
}

Program parse_program(std::string const& name) {
  if (name == "remesh") return Program::remesh;
  if (name == "levelset") return Program::levelset;
  if (name == "lagrangian") return Program::lagrangian;
  throw Failure("invalid_request", "Unknown Mmg program '" + name + "'");
}

char const* backend_name(Backend backend) {
  switch (backend) {
    case Backend::mmg2d: return "mmg2d";
    case Backend::mmgs: return "mmgs";
    case Backend::mmg3d: return "mmg3d";
  }
  throw std::logic_error("Unknown backend");
}

// Host-side simplicial complex with 0-based indices.
struct Mesh {
  int dimension = 0;
  int arity = 0;
  std::vector<double> vertices;
  std::vector<std::uint8_t> vertex_required;
  std::vector<std::int64_t> cells, cell_references;
  std::vector<std::uint8_t> cell_required;
  // Boundary facets: edges for planar/surface meshes, triangles for volumes.
  std::vector<std::int64_t> triangles, triangle_references;
  std::vector<std::uint8_t> triangle_required;
  std::vector<std::int64_t> edges, edge_references;
  std::vector<std::uint8_t> edge_required, edge_ridges;

  std::int64_t vertex_count() const { return static_cast<std::int64_t>(vertices.size()) / dimension; }
  std::int64_t cell_count() const { return static_cast<std::int64_t>(cells.size()) / arity; }
};

struct Material {
  std::int64_t reference = 0;
  bool split = true;
  std::int64_t interior = 0, exterior = 0;
};

struct Controls {
  Program program = Program::remesh;
  double hausdorff_distance = 0.0;
  std::optional<double> minimum_size, maximum_size, gradation, angle_detection;
  bool insertion = true, swapping = true, relocation = true, surface_modification = true,
       optimize = false;
  double isovalue = 0.0;
  std::int64_t interface_reference = 0;
  std::vector<Material> materials;
  int lagrangian_mode = -1;
  std::int64_t memory_megabytes = 0;
};

struct Solutions {
  int metric_width = 0;  // 0: none, 1: scalar sizes, 3/6: symmetric tensors (Mmg order)
  std::vector<double> metric;
  std::vector<double> level_set;
  std::vector<double> displacement;
};

struct Adapted {
  Mesh mesh;
  std::vector<std::int64_t> facets, facet_references;
  int metric_width = 0;
  std::vector<double> metric;
  // A ridge metric Mmg stored without its tangent frame cannot be rebuilt.
  bool metric_unrecoverable = false;
};

MMG5_int to_mmg(std::int64_t value) {
  if (value < 0 || value > static_cast<std::int64_t>(MMG5_INTMAX))
    throw Failure("unsupported", "Mesh entity count or reference exceeds this Mmg build's integer range");
  return static_cast<MMG5_int>(value);
}

std::vector<MMG5_int> one_based(std::vector<std::int64_t> const& indices) {
  std::vector<MMG5_int> result(indices.size());
  for (std::size_t item = 0; item < indices.size(); ++item) result[item] = to_mmg(indices[item] + 1);
  return result;
}

std::vector<MMG5_int> references(std::vector<std::int64_t> const& values) {
  std::vector<MMG5_int> result(values.size());
  for (std::size_t item = 0; item < values.size(); ++item) result[item] = to_mmg(values[item]);
  return result;
}

void check(int status, char const* call) {
  if (status != 1) throw Failure("library_failure", std::string("Mmg rejected ") + call);
}

// ---------------------------------------------------------------- inputs

template <class T>
std::vector<T> take(exchange::Input const& input, std::string const& name, exchange::DType dtype,
                    std::vector<std::int64_t> const& shape) {
  return input.require(name, dtype, shape).to_vector<T>();
}

void check_indices(std::vector<std::int64_t> const& indices, std::int64_t bound, char const* name) {
  for (std::int64_t value : indices)
    if (value < 0 || value >= bound)
      throw std::invalid_argument(std::string(name) + " index an undeclared vertex");
}

void check_flags(std::vector<std::uint8_t> const& flags, char const* name) {
  for (std::uint8_t value : flags)
    if (value > 1) throw std::invalid_argument(std::string(name) + " must hold 0/1 flags");
}

Mesh read_mesh(exchange::Input const& input, Backend backend) {
  Mesh mesh;
  mesh.dimension = backend == Backend::mmg2d ? 2 : 3;
  mesh.arity = backend == Backend::mmg3d ? 4 : 3;
  using exchange::DType;
  auto const& vertices = input.require("vertices", DType::float64, {-1, mesh.dimension});
  std::int64_t const n = static_cast<std::int64_t>(vertices.shape[0]);
  mesh.vertices = vertices.to_vector<double>();
  for (double value : mesh.vertices)
    if (!std::isfinite(value)) throw std::invalid_argument("Vertices must be finite");
  mesh.vertex_required = take<std::uint8_t>(input, "vertex_required", DType::uint8, {n});
  auto const& cells = input.require("cells", DType::int64, {-1, mesh.arity});
  std::int64_t const m = static_cast<std::int64_t>(cells.shape[0]);
  if (n == 0 || m == 0) throw std::invalid_argument("Mmg requires a nonempty simplicial mesh");
  mesh.cells = cells.to_vector<std::int64_t>();
  mesh.cell_references = take<std::int64_t>(input, "cell_references", DType::int64, {m});
  mesh.cell_required = take<std::uint8_t>(input, "cell_required", DType::uint8, {m});
  auto const& edges = input.require("edges", DType::int64, {-1, 2});
  std::int64_t const e = static_cast<std::int64_t>(edges.shape[0]);
  mesh.edges = edges.to_vector<std::int64_t>();
  mesh.edge_references = take<std::int64_t>(input, "edge_references", DType::int64, {e});
  mesh.edge_required = take<std::uint8_t>(input, "edge_required", DType::uint8, {e});
  mesh.edge_ridges = take<std::uint8_t>(input, "edge_ridges", DType::uint8, {e});
  auto const& triangles = input.require("triangles", DType::int64, {-1, 3});
  std::int64_t const t = static_cast<std::int64_t>(triangles.shape[0]);
  if (backend != Backend::mmg3d && t != 0)
    throw std::invalid_argument("Boundary triangles are declared only for tetrahedral meshes");
  mesh.triangles = triangles.to_vector<std::int64_t>();
  mesh.triangle_references = take<std::int64_t>(input, "triangle_references", DType::int64, {t});
  mesh.triangle_required = take<std::uint8_t>(input, "triangle_required", DType::uint8, {t});
  check_indices(mesh.cells, n, "cells");
  check_indices(mesh.edges, n, "edges");
  check_indices(mesh.triangles, n, "triangles");
  for (auto const* flags : {&mesh.vertex_required, &mesh.cell_required, &mesh.edge_required,
                            &mesh.edge_ridges, &mesh.triangle_required})
    check_flags(*flags, "Entity flags");
  if (backend == Backend::mmg2d)
    for (std::uint8_t ridge : mesh.edge_ridges)
      if (ridge) throw Failure("unsupported", "mmg2d has no ridge edges");
  return mesh;
}

Solutions read_solutions(exchange::Input const& input, Mesh const& mesh, Program program) {
  using exchange::DType;
  Solutions solutions;
  std::int64_t const n = mesh.vertex_count();
  if (input.has("metric")) {
    auto const& metric = input.get("metric");
    int const tensor = mesh.dimension == 2 ? 3 : 6;
    if (metric.dtype != DType::float64 || metric.shape.empty() ||
        metric.shape[0] != static_cast<std::uint64_t>(n) ||
        !((metric.shape.size() == 1) ||
          (metric.shape.size() == 2 && metric.shape[1] == static_cast<std::uint64_t>(tensor))))
      throw std::invalid_argument("Metric must hold one scalar size or symmetric tensor per vertex");
    solutions.metric_width = metric.shape.size() == 1 ? 1 : tensor;
    solutions.metric = metric.to_vector<double>();
    for (double value : solutions.metric)
      if (!std::isfinite(value)) throw std::invalid_argument("Metric values must be finite");
  }
  if (program == Program::levelset)
    solutions.level_set = take<double>(input, "level_set", DType::float64, {n});
  if (program == Program::lagrangian)
    solutions.displacement = take<double>(input, "displacement", DType::float64, {n, mesh.dimension});
  return solutions;
}

std::optional<double> optional_number(json::Value const& parameters, char const* key) {
  json::Value const& value = parameters.at(key);
  if (value.is_null()) return std::nullopt;
  double const number = value.as_double();
  if (!std::isfinite(number) || number <= 0.0)
    throw std::invalid_argument(std::string(key) + " must be positive and finite");
  return number;
}

std::int64_t memory_megabytes() {
  char const* limit = std::getenv("PHYDRAX_WORKER_MEMORY_LIMIT_BYTES");
  if (limit == nullptr) return 0;
  return static_cast<std::int64_t>(std::stoull(limit) >> 20);
}

Controls read_controls(json::Value const& parameters) {
  Controls controls;
  controls.program = parse_program(parameters.at("program").as_string());
  controls.hausdorff_distance = parameters.at("hausdorff_distance").as_double();
  if (!std::isfinite(controls.hausdorff_distance) || controls.hausdorff_distance <= 0.0)
    throw std::invalid_argument("hausdorff_distance must be positive and finite");
  controls.minimum_size = optional_number(parameters, "minimum_size");
  controls.maximum_size = optional_number(parameters, "maximum_size");
  controls.gradation = optional_number(parameters, "gradation");
  controls.angle_detection = optional_number(parameters, "angle_detection");
  controls.insertion = parameters.at("insertion").as_bool();
  controls.swapping = parameters.at("swapping").as_bool();
  controls.relocation = parameters.at("relocation").as_bool();
  controls.surface_modification = parameters.at("surface_modification").as_bool();
  controls.optimize = parameters.at("optimize").as_bool();
  controls.memory_megabytes = memory_megabytes();
  switch (controls.program) {
    case Program::remesh: break;
    case Program::levelset: {
      controls.isovalue = parameters.at("isovalue").as_double();
      controls.interface_reference = parameters.at("interface_reference").as_int();
      for (auto const& row : parameters.at("materials").as_array()) {
        auto const& values = row.as_array();
        if (values.size() != 4) throw std::invalid_argument("Materials are [reference, split, interior, exterior]");
        controls.materials.push_back(
            {values[0].as_int(), values[1].as_bool(), values[2].as_int(), values[3].as_int()});
      }
      break;
    }
    case Program::lagrangian: {
      controls.lagrangian_mode = static_cast<int>(parameters.at("lagrangian_mode").as_int());
      if (controls.lagrangian_mode < 0 || controls.lagrangian_mode > 2)
        throw std::invalid_argument("lagrangian_mode must be 0, 1 or 2");
      break;
    }
  }
  return controls;
}

// A referenced facet shared by two cells needs Mmg's open-boundary mode to
// survive when both cells carry the same region reference.
bool has_interior_referenced_facets(Mesh const& mesh, Backend backend) {
  bool const volume = backend == Backend::mmg3d;
  auto const& facets = volume ? mesh.triangles : mesh.edges;
  auto const& refs = volume ? mesh.triangle_references : mesh.edge_references;
  int const width = volume ? 3 : 2;
  std::map<std::vector<std::int64_t>, int> uses;
  for (std::size_t facet = 0; facet < refs.size(); ++facet) {
    if (refs[facet] == 0) continue;
    std::vector<std::int64_t> key(facets.begin() + facet * width, facets.begin() + (facet + 1) * width);
    std::sort(key.begin(), key.end());
    uses.emplace(std::move(key), 0);
  }
  if (uses.empty()) return false;
  int const arity = mesh.arity;
  for (std::int64_t cell = 0; cell < mesh.cell_count(); ++cell)
    for (int omitted = 0; omitted < arity; ++omitted) {
      std::vector<std::int64_t> key;
      for (int corner = 0; corner < arity; ++corner)
        if (corner != omitted) key.push_back(mesh.cells[cell * arity + corner]);
      std::sort(key.begin(), key.end());
      auto found = uses.find(key);
      if (found != uses.end() && ++found->second == 2) return true;
    }
  return false;
}

// ---------------------------------------------------------------- Mmg backends

struct ParameterIds {
  int verbose, mem, angle, noinsert, noswap, nomove, nosurf, optim, opnbdy, number_of_materials, isoref,
      lag, angle_detection, hmin, hmax, hausd, hgrad, ls;
};

template <class SetInteger, class SetReal, class SetMaterial>
void apply_controls(Controls const& controls, ParameterIds const& ids, bool open_boundaries,
                    SetInteger&& integer, SetReal&& real, SetMaterial&& material) {
  check(integer(ids.verbose, -1), "verbosity");
  if (controls.memory_megabytes > 0)
    check(integer(ids.mem, to_mmg(controls.memory_megabytes)), "memory bound");
  check(real(ids.hausd, controls.hausdorff_distance), "hausd");
  if (controls.minimum_size) check(real(ids.hmin, *controls.minimum_size), "hmin");
  if (controls.maximum_size) check(real(ids.hmax, *controls.maximum_size), "hmax");
  if (controls.gradation) check(real(ids.hgrad, *controls.gradation), "hgrad");
  check(integer(ids.angle, controls.angle_detection ? 1 : 0), "angle detection");
  if (controls.angle_detection)
    check(real(ids.angle_detection, *controls.angle_detection), "angle detection threshold");
  check(integer(ids.noinsert, controls.insertion ? 0 : 1), "noinsert");
  check(integer(ids.noswap, controls.swapping ? 0 : 1), "noswap");
  check(integer(ids.nomove, controls.relocation ? 0 : 1), "nomove");
  if (ids.nosurf >= 0) {
    check(integer(ids.nosurf, controls.surface_modification ? 0 : 1), "nosurf");
  } else if (!controls.surface_modification) {
    throw Failure("unsupported", "mmgs remeshes surfaces only; surface modification cannot be disabled");
  }
  check(integer(ids.optim, controls.optimize ? 1 : 0), "optim");
  if (open_boundaries && ids.opnbdy >= 0) check(integer(ids.opnbdy, 1), "opnbdy");
  switch (controls.program) {
    case Program::remesh: break;
    case Program::levelset: {
      check(real(ids.ls, controls.isovalue), "isovalue");
      check(integer(ids.isoref, to_mmg(controls.interface_reference)), "isoref");
      check(integer(ids.number_of_materials, to_mmg(static_cast<std::int64_t>(controls.materials.size()))),
            "material count");
      for (Material const& value : controls.materials)
        check(material(to_mmg(value.reference), value.split ? MMG5_MMAT_Split : MMG5_MMAT_NoSplit,
                       to_mmg(value.interior), to_mmg(value.exterior)),
              "material");
      break;
    }
    case Program::lagrangian: {
      if (ids.lag < 0) throw Failure("unsupported", "mmgs has no Lagrangian motion");
      if (integer(ids.lag, controls.lagrangian_mode) != 1)
        throw Failure("unsupported",
                      "Lagrangian motion requires Mmg built with USE_ELAS (ISCD LinearElasticity)");
      break;
    }
  }
}

void check_program(int status) {
  switch (status) {
    case MMG5_SUCCESS: return;
    case MMG5_LOWFAILURE:
      throw Failure("library_failure",
                    "Mmg returned MMG5_LOWFAILURE: the mesh is conforming but the requested "
                    "adaptation could not be completed");
    default:
      throw Failure("library_failure", "Mmg returned MMG5_STRONGFAILURE: no usable mesh was produced");
  }
}

std::vector<std::int64_t> zero_based(std::vector<MMG5_int> const& values) {
  std::vector<std::int64_t> result(values.size());
  for (std::size_t item = 0; item < values.size(); ++item) result[item] = static_cast<std::int64_t>(values[item]) - 1;
  return result;
}

std::vector<std::int64_t> widen(std::vector<MMG5_int> const& values) {
  return std::vector<std::int64_t>(values.begin(), values.end());
}

#ifndef PHYDRAX_MMG_WITH_PARMMG

class Mmg2dSession {
 public:
  Mmg2dSession() {
    check(MMG2D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met, MMG5_ARG_ppLs,
                          &ls, MMG5_ARG_ppDisp, &disp, MMG5_ARG_end),
          "MMG2D_Init_mesh");
  }
  ~Mmg2dSession() {
    MMG2D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met, MMG5_ARG_ppLs, &ls,
                   MMG5_ARG_ppDisp, &disp, MMG5_ARG_end);
  }
  Mmg2dSession(Mmg2dSession const&) = delete;
  Mmg2dSession& operator=(Mmg2dSession const&) = delete;
  MMG5_pMesh mesh = nullptr;
  MMG5_pSol met = nullptr, ls = nullptr, disp = nullptr;
};

class MmgsSession {
 public:
  MmgsSession() {
    check(MMGS_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met, MMG5_ARG_ppLs, &ls,
                         MMG5_ARG_end),
          "MMGS_Init_mesh");
  }
  ~MmgsSession() {
    MMGS_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met, MMG5_ARG_ppLs, &ls,
                  MMG5_ARG_end);
  }
  MmgsSession(MmgsSession const&) = delete;
  MmgsSession& operator=(MmgsSession const&) = delete;
  MMG5_pMesh mesh = nullptr;
  MMG5_pSol met = nullptr, ls = nullptr;
};

#endif

class Mmg3dSession {
 public:
  Mmg3dSession() {
    check(MMG3D_Init_mesh(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met, MMG5_ARG_ppLs,
                          &ls, MMG5_ARG_ppDisp, &disp, MMG5_ARG_end),
          "MMG3D_Init_mesh");
  }
  ~Mmg3dSession() {
    MMG3D_Free_all(MMG5_ARG_start, MMG5_ARG_ppMesh, &mesh, MMG5_ARG_ppMet, &met, MMG5_ARG_ppLs, &ls,
                   MMG5_ARG_ppDisp, &disp, MMG5_ARG_end);
  }
  Mmg3dSession(Mmg3dSession const&) = delete;
  Mmg3dSession& operator=(Mmg3dSession const&) = delete;
  MMG5_pMesh mesh = nullptr;
  MMG5_pSol met = nullptr, ls = nullptr, disp = nullptr;
};

// Copies a solution array into an Mmg solution through the backend setters.
template <class SetSize, class SetValues>
void load_solution(MMG5_pMesh mesh, MMG5_pSol sol, std::vector<double> const& values, int kind,
                   std::int64_t count, SetSize&& set_size, SetValues&& set_values) {
  if (values.empty()) return;
  check(set_size(mesh, sol, MMG5_Vertex, to_mmg(count), kind), "solution size");
  std::vector<double> copy(values);
  check(set_values(sol, copy.data()), "solution values");
}

int metric_kind(int width) { return width == 1 ? MMG5_Scalar : MMG5_Tensor; }

}  // namespace

// Mmg stores anisotropic metrics at ridge points in a tangent-frame form
// (info.metRidTyp); its own .sol writer rebuilds the ambient tensor with this
// function, which libmmgs/libmmg3d export but declare only privately.
extern "C" void MMG5_build3DMetric(MMG5_pMesh mesh, MMG5_pSol sol, MMG5_int ip, double dbuf[6]);

namespace {

void ambient_tensors(MMG5_pMesh mesh, MMG5_pSol met, Adapted& adapted) {
  std::size_t const count = adapted.metric.size() / 6;
  for (std::size_t vertex = 0; vertex < count; ++vertex) {
    double* tensor = adapted.metric.data() + 6 * vertex;
    MMG5_build3DMetric(mesh, met, static_cast<MMG5_int>(vertex + 1), tensor);
    if (std::all_of(tensor, tensor + 6, [](double value) { return value == 0.0; })) {
      adapted.metric.clear();
      adapted.metric_width = 0;
      adapted.metric_unrecoverable = true;
      return;
    }
  }
}

template <class GetSize, class GetScalar, class GetTensor>
void extract_metric(MMG5_pMesh mesh, MMG5_pSol met, std::int64_t vertex_count, int tensor_width,
                    Adapted& adapted, GetSize&& get_size, GetScalar&& get_scalar, GetTensor&& get_tensor) {
  int entity = 0, kind = 0;
  MMG5_int count = 0;
  if (get_size(mesh, met, &entity, &count, &kind) != 1 || entity != MMG5_Vertex ||
      count != vertex_count || count == 0)
    return;
  if (kind == MMG5_Scalar) {
    adapted.metric.resize(static_cast<std::size_t>(count));
    check(get_scalar(met, adapted.metric.data()), "metric extraction");
    adapted.metric_width = 1;
  } else if (kind == MMG5_Tensor) {
    adapted.metric.resize(static_cast<std::size_t>(count) * tensor_width);
    check(get_tensor(met, adapted.metric.data()), "metric extraction");
    adapted.metric_width = tensor_width;
    if (tensor_width == 6) ambient_tensors(mesh, met, adapted);
  }
}

#ifndef PHYDRAX_MMG_WITH_PARMMG

Adapted run_mmg2d(Mesh const& source, Solutions const& solutions, Controls const& controls) {
  Mmg2dSession session;
  MMG5_pMesh mesh = session.mesh;
  std::int64_t const n = source.vertex_count(), m = source.cell_count();
  std::int64_t const e = static_cast<std::int64_t>(source.edge_references.size());
  check(MMG2D_Set_meshSize(mesh, to_mmg(n), to_mmg(m), 0, to_mmg(e)), "MMG2D_Set_meshSize");
  std::vector<double> vertices(source.vertices);
  check(MMG2D_Set_vertices(mesh, vertices.data(), nullptr), "MMG2D_Set_vertices");
  auto cells = one_based(source.cells);
  auto cell_refs = references(source.cell_references);
  check(MMG2D_Set_triangles(mesh, cells.data(), cell_refs.data()), "MMG2D_Set_triangles");
  if (e > 0) {
    auto edges = one_based(source.edges);
    auto edge_refs = references(source.edge_references);
    check(MMG2D_Set_edges(mesh, edges.data(), edge_refs.data()), "MMG2D_Set_edges");
  }
  for (std::int64_t k = 0; k < n; ++k)
    if (source.vertex_required[k]) check(MMG2D_Set_requiredVertex(mesh, to_mmg(k + 1)), "required vertex");
  for (std::int64_t k = 0; k < m; ++k)
    if (source.cell_required[k]) check(MMG2D_Set_requiredTriangle(mesh, to_mmg(k + 1)), "required triangle");
  for (std::int64_t k = 0; k < e; ++k)
    if (source.edge_required[k]) check(MMG2D_Set_requiredEdge(mesh, to_mmg(k + 1)), "required edge");
  load_solution(mesh, session.met, solutions.metric, metric_kind(solutions.metric_width), n,
                MMG2D_Set_solSize, solutions.metric_width == 1 ? MMG2D_Set_scalarSols : MMG2D_Set_tensorSols);
  load_solution(mesh, session.ls, solutions.level_set, MMG5_Scalar, n, MMG2D_Set_solSize,
                MMG2D_Set_scalarSols);
  if (!solutions.displacement.empty()) {
    // MMG2D_Set_vectorSols (Mmg 5.8.0) stores vertex k at m[2k-1..2k] while
    // the Lagrangian motion (mmg2d9.c, velextls_2d.c) and the .sol reader use
    // m[2k..2k+1]; write the layout the motion consumes.
    check(MMG2D_Set_solSize(mesh, session.disp, MMG5_Vertex, to_mmg(n), MMG5_Vector), "solution size");
    for (std::int64_t k = 0; k < n; ++k)
      for (int axis = 0; axis < 2; ++axis)
        session.disp->m[2 * (k + 1) + axis] = solutions.displacement[2 * k + axis];
  }
  ParameterIds const ids{MMG2D_IPARAM_verbose,  MMG2D_IPARAM_mem,        MMG2D_IPARAM_angle,
                         MMG2D_IPARAM_noinsert, MMG2D_IPARAM_noswap,     MMG2D_IPARAM_nomove,
                         MMG2D_IPARAM_nosurf,   MMG2D_IPARAM_optim,      MMG2D_IPARAM_opnbdy,
                         MMG2D_IPARAM_numberOfMat, MMG2D_IPARAM_isoref,  MMG2D_IPARAM_lag,
                         MMG2D_DPARAM_angleDetection, MMG2D_DPARAM_hmin, MMG2D_DPARAM_hmax,
                         MMG2D_DPARAM_hausd,    MMG2D_DPARAM_hgrad,      MMG2D_DPARAM_ls};
  apply_controls(
      controls, ids, has_interior_referenced_facets(source, Backend::mmg2d),
      [&](int id, MMG5_int value) { return MMG2D_Set_iparameter(mesh, session.met, id, value); },
      [&](int id, double value) { return MMG2D_Set_dparameter(mesh, session.met, id, value); },
      [&](MMG5_int ref, int split, MMG5_int in, MMG5_int ex) {
        return MMG2D_Set_multiMat(mesh, session.met, ref, split, in, ex);
      });
  switch (controls.program) {
    case Program::remesh: check_program(MMG2D_mmg2dlib(mesh, session.met)); break;
    case Program::levelset: check_program(MMG2D_mmg2dls(mesh, session.ls, session.met)); break;
    case Program::lagrangian: check_program(MMG2D_mmg2dmov(mesh, session.met, session.disp)); break;
  }
  MMG5_int np = 0, nt = 0, nquad = 0, na = 0;
  check(MMG2D_Get_meshSize(mesh, &np, &nt, &nquad, &na), "MMG2D_Get_meshSize");
  if (nquad != 0) throw Failure("library_failure", "mmg2d returned quadrilaterals");
  Adapted adapted;
  adapted.mesh.dimension = 2;
  adapted.mesh.arity = 3;
  adapted.mesh.vertices.resize(static_cast<std::size_t>(np) * 2);
  std::vector<MMG5_int> vertex_refs(np);
  std::vector<int> corners(np), required(np);
  check(MMG2D_Get_vertices(mesh, adapted.mesh.vertices.data(), vertex_refs.data(), corners.data(),
                           required.data()),
        "MMG2D_Get_vertices");
  std::vector<MMG5_int> triangles(static_cast<std::size_t>(nt) * 3), triangle_refs(nt);
  std::vector<int> triangle_required(nt);
  check(MMG2D_Get_triangles(mesh, triangles.data(), triangle_refs.data(), triangle_required.data()),
        "MMG2D_Get_triangles");
  std::vector<MMG5_int> edges(static_cast<std::size_t>(na) * 2), edge_refs(na);
  std::vector<int> ridges(na), edge_required(na);
  if (na > 0)
    check(MMG2D_Get_edges(mesh, edges.data(), edge_refs.data(), ridges.data(), edge_required.data()),
          "MMG2D_Get_edges");
  adapted.mesh.cells = zero_based(triangles);
  adapted.mesh.cell_references = widen(triangle_refs);
  adapted.facets = zero_based(edges);
  adapted.facet_references = widen(edge_refs);
  extract_metric(mesh, session.met, np, 3, adapted, MMG2D_Get_solSize, MMG2D_Get_scalarSols,
                 MMG2D_Get_tensorSols);
  return adapted;
}

Adapted run_mmgs(Mesh const& source, Solutions const& solutions, Controls const& controls) {
  if (controls.program == Program::lagrangian)
    throw Failure("unsupported", "mmgs has no Lagrangian motion");
  MmgsSession session;
  MMG5_pMesh mesh = session.mesh;
  std::int64_t const n = source.vertex_count(), m = source.cell_count();
  std::int64_t const e = static_cast<std::int64_t>(source.edge_references.size());
  check(MMGS_Set_meshSize(mesh, to_mmg(n), to_mmg(m), to_mmg(e)), "MMGS_Set_meshSize");
  std::vector<double> vertices(source.vertices);
  check(MMGS_Set_vertices(mesh, vertices.data(), nullptr), "MMGS_Set_vertices");
  auto cells = one_based(source.cells);
  auto cell_refs = references(source.cell_references);
  check(MMGS_Set_triangles(mesh, cells.data(), cell_refs.data()), "MMGS_Set_triangles");
  if (e > 0) {
    auto edges = one_based(source.edges);
    auto edge_refs = references(source.edge_references);
    check(MMGS_Set_edges(mesh, edges.data(), edge_refs.data()), "MMGS_Set_edges");
  }
  for (std::int64_t k = 0; k < n; ++k)
    if (source.vertex_required[k]) check(MMGS_Set_requiredVertex(mesh, to_mmg(k + 1)), "required vertex");
  for (std::int64_t k = 0; k < m; ++k)
    if (source.cell_required[k]) check(MMGS_Set_requiredTriangle(mesh, to_mmg(k + 1)), "required triangle");
  for (std::int64_t k = 0; k < e; ++k) {
    if (source.edge_required[k]) check(MMGS_Set_requiredEdge(mesh, to_mmg(k + 1)), "required edge");
    if (source.edge_ridges[k]) check(MMGS_Set_ridge(mesh, to_mmg(k + 1)), "ridge");
  }
  load_solution(mesh, session.met, solutions.metric, metric_kind(solutions.metric_width), n,
                MMGS_Set_solSize, solutions.metric_width == 1 ? MMGS_Set_scalarSols : MMGS_Set_tensorSols);
  load_solution(mesh, session.ls, solutions.level_set, MMG5_Scalar, n, MMGS_Set_solSize,
                MMGS_Set_scalarSols);
  ParameterIds const ids{MMGS_IPARAM_verbose,  MMGS_IPARAM_mem,       MMGS_IPARAM_angle,
                         MMGS_IPARAM_noinsert, MMGS_IPARAM_noswap,    MMGS_IPARAM_nomove,
                         -1,                   MMGS_IPARAM_optim,     -1,
                         MMGS_IPARAM_numberOfMat, MMGS_IPARAM_isoref, -1,
                         MMGS_DPARAM_angleDetection, MMGS_DPARAM_hmin, MMGS_DPARAM_hmax,
                         MMGS_DPARAM_hausd,    MMGS_DPARAM_hgrad,     MMGS_DPARAM_ls};
  apply_controls(
      controls, ids, false,
      [&](int id, MMG5_int value) { return MMGS_Set_iparameter(mesh, session.met, id, value); },
      [&](int id, double value) { return MMGS_Set_dparameter(mesh, session.met, id, value); },
      [&](MMG5_int ref, int split, MMG5_int in, MMG5_int ex) {
        return MMGS_Set_multiMat(mesh, session.met, ref, split, in, ex);
      });
  switch (controls.program) {
    case Program::remesh: check_program(MMGS_mmgslib(mesh, session.met)); break;
    case Program::levelset: check_program(MMGS_mmgsls(mesh, session.ls, session.met)); break;
    case Program::lagrangian: throw std::logic_error("unreachable");
  }
  MMG5_int np = 0, nt = 0, na = 0;
  check(MMGS_Get_meshSize(mesh, &np, &nt, &na), "MMGS_Get_meshSize");
  Adapted adapted;
  adapted.mesh.dimension = 3;
  adapted.mesh.arity = 3;
  adapted.mesh.vertices.resize(static_cast<std::size_t>(np) * 3);
  std::vector<MMG5_int> vertex_refs(np);
  std::vector<int> corners(np), required(np);
  check(MMGS_Get_vertices(mesh, adapted.mesh.vertices.data(), vertex_refs.data(), corners.data(),
                          required.data()),
        "MMGS_Get_vertices");
  std::vector<MMG5_int> triangles(static_cast<std::size_t>(nt) * 3), triangle_refs(nt);
  std::vector<int> triangle_required(nt);
  check(MMGS_Get_triangles(mesh, triangles.data(), triangle_refs.data(), triangle_required.data()),
        "MMGS_Get_triangles");
  std::vector<MMG5_int> edges(static_cast<std::size_t>(na) * 2), edge_refs(na);
  std::vector<int> ridges(na), edge_required(na);
  if (na > 0)
    check(MMGS_Get_edges(mesh, edges.data(), edge_refs.data(), ridges.data(), edge_required.data()),
          "MMGS_Get_edges");
  adapted.mesh.cells = zero_based(triangles);
  adapted.mesh.cell_references = widen(triangle_refs);
  adapted.facets = zero_based(edges);
  adapted.facet_references = widen(edge_refs);
  extract_metric(mesh, session.met, np, 6, adapted, MMGS_Get_solSize, MMGS_Get_scalarSols,
                 MMGS_Get_tensorSols);
  return adapted;
}

#endif

void load_mmg3d(Mmg3dSession& session, Mesh const& source) {
  MMG5_pMesh mesh = session.mesh;
  std::int64_t const n = source.vertex_count(), m = source.cell_count();
  std::int64_t const t = static_cast<std::int64_t>(source.triangle_references.size());
  std::int64_t const e = static_cast<std::int64_t>(source.edge_references.size());
  check(MMG3D_Set_meshSize(mesh, to_mmg(n), to_mmg(m), 0, to_mmg(t), 0, to_mmg(e)), "MMG3D_Set_meshSize");
  std::vector<double> vertices(source.vertices);
  check(MMG3D_Set_vertices(mesh, vertices.data(), nullptr), "MMG3D_Set_vertices");
  auto cells = one_based(source.cells);
  auto cell_refs = references(source.cell_references);
  check(MMG3D_Set_tetrahedra(mesh, cells.data(), cell_refs.data()), "MMG3D_Set_tetrahedra");
  if (t > 0) {
    auto triangles = one_based(source.triangles);
    auto triangle_refs = references(source.triangle_references);
    check(MMG3D_Set_triangles(mesh, triangles.data(), triangle_refs.data()), "MMG3D_Set_triangles");
  }
  if (e > 0) {
    auto edges = one_based(source.edges);
    auto edge_refs = references(source.edge_references);
    check(MMG3D_Set_edges(mesh, edges.data(), edge_refs.data()), "MMG3D_Set_edges");
  }
  for (std::int64_t k = 0; k < n; ++k)
    if (source.vertex_required[k]) check(MMG3D_Set_requiredVertex(mesh, to_mmg(k + 1)), "required vertex");
  for (std::int64_t k = 0; k < m; ++k)
    if (source.cell_required[k])
      check(MMG3D_Set_requiredTetrahedron(mesh, to_mmg(k + 1)), "required tetrahedron");
  for (std::int64_t k = 0; k < t; ++k)
    if (source.triangle_required[k])
      check(MMG3D_Set_requiredTriangle(mesh, to_mmg(k + 1)), "required triangle");
  for (std::int64_t k = 0; k < e; ++k) {
    if (source.edge_required[k]) check(MMG3D_Set_requiredEdge(mesh, to_mmg(k + 1)), "required edge");
    if (source.edge_ridges[k]) check(MMG3D_Set_ridge(mesh, to_mmg(k + 1)), "ridge");
  }
}

ParameterIds mmg3d_parameters() {
  return {MMG3D_IPARAM_verbose,  MMG3D_IPARAM_mem,        MMG3D_IPARAM_angle,
          MMG3D_IPARAM_noinsert, MMG3D_IPARAM_noswap,     MMG3D_IPARAM_nomove,
          MMG3D_IPARAM_nosurf,   MMG3D_IPARAM_optim,      MMG3D_IPARAM_opnbdy,
          MMG3D_IPARAM_numberOfMat, MMG3D_IPARAM_isoref,  MMG3D_IPARAM_lag,
          MMG3D_DPARAM_angleDetection, MMG3D_DPARAM_hmin, MMG3D_DPARAM_hmax,
          MMG3D_DPARAM_hausd,    MMG3D_DPARAM_hgrad,      MMG3D_DPARAM_ls};
}

Adapted run_mmg3d(Mesh const& source, Solutions const& solutions, Controls const& controls) {
  Mmg3dSession session;
  MMG5_pMesh mesh = session.mesh;
  std::int64_t const n = source.vertex_count();
  load_mmg3d(session, source);
  load_solution(mesh, session.met, solutions.metric, metric_kind(solutions.metric_width), n,
                MMG3D_Set_solSize, solutions.metric_width == 1 ? MMG3D_Set_scalarSols : MMG3D_Set_tensorSols);
  load_solution(mesh, session.ls, solutions.level_set, MMG5_Scalar, n, MMG3D_Set_solSize,
                MMG3D_Set_scalarSols);
  load_solution(mesh, session.disp, solutions.displacement, MMG5_Vector, n, MMG3D_Set_solSize,
                MMG3D_Set_vectorSols);
  apply_controls(
      controls, mmg3d_parameters(), has_interior_referenced_facets(source, Backend::mmg3d),
      [&](int id, MMG5_int value) { return MMG3D_Set_iparameter(mesh, session.met, id, value); },
      [&](int id, double value) { return MMG3D_Set_dparameter(mesh, session.met, id, value); },
      [&](MMG5_int ref, int split, MMG5_int in, MMG5_int ex) {
        return MMG3D_Set_multiMat(mesh, session.met, ref, split, in, ex);
      });
  switch (controls.program) {
    case Program::remesh: check_program(MMG3D_mmg3dlib(mesh, session.met)); break;
    case Program::levelset: check_program(MMG3D_mmg3dls(mesh, session.ls, session.met)); break;
    case Program::lagrangian: check_program(MMG3D_mmg3dmov(mesh, session.met, session.disp)); break;
  }
  MMG5_int np = 0, ne = 0, nprism = 0, nt = 0, nquad = 0, na = 0;
  check(MMG3D_Get_meshSize(mesh, &np, &ne, &nprism, &nt, &nquad, &na), "MMG3D_Get_meshSize");
  if (nprism != 0 || nquad != 0) throw Failure("library_failure", "mmg3d returned prisms or quadrilaterals");
  Adapted adapted;
  adapted.mesh.dimension = 3;
  adapted.mesh.arity = 4;
  adapted.mesh.vertices.resize(static_cast<std::size_t>(np) * 3);
  std::vector<MMG5_int> vertex_refs(np);
  std::vector<int> corners(np), required(np);
  check(MMG3D_Get_vertices(mesh, adapted.mesh.vertices.data(), vertex_refs.data(), corners.data(),
                           required.data()),
        "MMG3D_Get_vertices");
  std::vector<MMG5_int> tetrahedra(static_cast<std::size_t>(ne) * 4), tetrahedron_refs(ne);
  std::vector<int> tetrahedron_required(ne);
  check(MMG3D_Get_tetrahedra(mesh, tetrahedra.data(), tetrahedron_refs.data(), tetrahedron_required.data()),
        "MMG3D_Get_tetrahedra");
  std::vector<MMG5_int> triangles(static_cast<std::size_t>(nt) * 3), triangle_refs(nt);
  std::vector<int> triangle_required(nt);
  if (nt > 0)
    check(MMG3D_Get_triangles(mesh, triangles.data(), triangle_refs.data(), triangle_required.data()),
          "MMG3D_Get_triangles");
  adapted.mesh.cells = zero_based(tetrahedra);
  adapted.mesh.cell_references = widen(tetrahedron_refs);
  adapted.facets = zero_based(triangles);
  adapted.facet_references = widen(triangle_refs);
  extract_metric(mesh, session.met, np, 6, adapted, MMG3D_Get_solSize, MMG3D_Get_scalarSols,
                 MMG3D_Get_tensorSols);
  return adapted;
}

Adapted run(Backend backend, Mesh const& source, Solutions const& solutions, Controls const& controls) {
  switch (backend) {
#ifndef PHYDRAX_MMG_WITH_PARMMG
    case Backend::mmg2d: return run_mmg2d(source, solutions, controls);
    case Backend::mmgs: return run_mmgs(source, solutions, controls);
#else
    case Backend::mmg2d:
    case Backend::mmgs:
      throw Failure("unsupported", "The ParMmg worker adapts tetrahedral meshes only");
#endif
    case Backend::mmg3d: return run_mmg3d(source, solutions, controls);
  }
  throw std::logic_error("Unknown backend");
}

// ---------------------------------------------------------------- transfer

// The configuration Mmg's Lagrangian motion gives the source: the same program
// in mode 0 (pure displacement, no remeshing) keeps the source connectivity.
std::vector<double> displaced_source(Backend backend, Mesh const& source, Solutions const& solutions,
                                     Controls const& controls) {
  Controls motion = controls;
  motion.lagrangian_mode = 0;
  Adapted const moved = run(backend, source, solutions, motion);
  if (moved.mesh.vertices.size() != source.vertices.size() || moved.mesh.cells != source.cells)
    throw Failure("library_failure",
                  "Mmg's pure Lagrangian displacement renumbered the source; its displaced "
                  "configuration cannot carry the declared fields");
  return moved.mesh.vertices;
}

std::vector<std::int64_t> required_vertex_targets(Mesh const& source, Mesh const& target) {
  std::map<std::array<double, 3>, std::int64_t> positions;
  int const d = target.dimension;
  for (std::int64_t vertex = target.vertex_count() - 1; vertex >= 0; --vertex) {
    std::array<double, 3> key{0.0, 0.0, 0.0};
    for (int axis = 0; axis < d; ++axis) key[axis] = target.vertices[vertex * d + axis];
    positions[key] = vertex;
  }
  std::vector<std::int64_t> targets;
  for (std::int64_t vertex = 0; vertex < source.vertex_count(); ++vertex) {
    if (!source.vertex_required[vertex]) continue;
    std::array<double, 3> key{0.0, 0.0, 0.0};
    for (int axis = 0; axis < d; ++axis) key[axis] = source.vertices[vertex * d + axis];
    auto found = positions.find(key);
    targets.push_back(found == positions.end() ? -1 : found->second);
  }
  return targets;
}

json::Value transfer_fields(Backend backend, exchange::Input const& input, Mesh const& source,
                            Solutions const& solutions, Controls const& controls, Adapted const& adapted,
                            std::vector<std::uint8_t> const& counted, exchange::Output& output) {
  if (!input.has("fields")) return nullptr;
  auto const& fields = input.require("fields", exchange::DType::float64,
                                     {source.vertex_count(), -1});
  int const width = static_cast<int>(fields.shape[1]);
  if (width == 0) throw std::invalid_argument("Declared fields must have at least one component");
  std::vector<double> const values = fields.to_vector<double>();
  for (double value : values)
    if (!std::isfinite(value)) throw std::invalid_argument("Declared fields must be finite");
  bool const displaced = controls.program == Program::lagrangian;
  std::vector<double> const configuration =
      displaced ? displaced_source(backend, source, solutions, controls) : source.vertices;
  phydrax::mmg::TransferEvidence evidence;
  std::vector<double> const transferred =
      phydrax::mmg::transfer(configuration, source.dimension, source.cells, source.arity, values, width,
                             adapted.mesh.vertices, 1.0e-12, counted, evidence);
  output.add<double>("fields", {static_cast<std::uint64_t>(adapted.mesh.vertex_count()),
                                static_cast<std::uint64_t>(width)},
                     transferred);
  return json::Object{
      {"located", evidence.located},
      {"maximum_projection_distance", evidence.maximum_projection_distance},
      {"method", "p1-barycentric-closest-simplex"},
      {"projected", evidence.projected},
      {"source_configuration", displaced ? "lagrangian-displaced" : "source"},
      {"tolerance", evidence.tolerance},
  };
}

char const* metric_label(int width) {
  return width == 0 ? "none" : width == 1 ? "scalar" : "tensor";
}

char const* adapted_metric_label(Adapted const& adapted) {
  return adapted.metric_unrecoverable ? "unrecoverable-ridge-metric"
                                      : metric_label(adapted.metric_width);
}

void write_adapted(exchange::Output& output, Adapted const& adapted) {
  auto const n = static_cast<std::uint64_t>(adapted.mesh.vertex_count());
  auto const m = static_cast<std::uint64_t>(adapted.mesh.cell_count());
  int const facet_width = adapted.mesh.arity == 4 ? 3 : 2;
  output.add<double>("vertices", {n, static_cast<std::uint64_t>(adapted.mesh.dimension)},
                     adapted.mesh.vertices);
  output.add<std::int64_t>("cells", {m, static_cast<std::uint64_t>(adapted.mesh.arity)}, adapted.mesh.cells);
  output.add<std::int64_t>("cell_references", {m}, adapted.mesh.cell_references);
  output.add<std::int64_t>("facets",
                           {adapted.facet_references.size(), static_cast<std::uint64_t>(facet_width)},
                           adapted.facets);
  output.add<std::int64_t>("facet_references", {adapted.facet_references.size()}, adapted.facet_references);
  if (adapted.metric_width == 1)
    output.add<double>("metric", {n}, adapted.metric);
  else if (adapted.metric_width > 1)
    output.add<double>("metric", {n, static_cast<std::uint64_t>(adapted.metric_width)}, adapted.metric);
}

#ifndef PHYDRAX_MMG_WITH_PARMMG

json::Value adapt(phydrax::worker::Request const& request) {
  Backend const backend = parse_backend(request.operation);
  Controls const controls = read_controls(request.parameters);
  exchange::Input const input = request.input();
  Mesh const source = read_mesh(input, backend);
  Solutions const solutions = read_solutions(input, source, controls.program);
  Adapted const adapted = run(backend, source, solutions, controls);
  exchange::Output output = request.output();
  write_adapted(output, adapted);
  std::vector<std::int64_t> const targets = required_vertex_targets(source, adapted.mesh);
  output.add<std::int64_t>("required_vertex_targets", {targets.size()}, targets);
  json::Value interpolation = transfer_fields(backend, input, source, solutions, controls, adapted, {}, output);
  output.finish();
  char const* programs[] = {"remesh", "levelset", "lagrangian"};
  return json::Object{
      {"adapted_metric", adapted_metric_label(adapted)},
      {"backend", backend_name(backend)},
      {"input_metric", metric_label(solutions.metric_width)},
      {"interpolation", std::move(interpolation)},
      {"memory_megabytes", controls.memory_megabytes},
      {"program", programs[static_cast<int>(controls.program)]},
  };
}

// Mmg records USE_ELAS nowhere in its installed headers; the linked library
// is asked directly whether it accepts a Lagrangian mode.
template <class Probe>
bool quietly(Probe&& probe) {
  std::fflush(stderr);
  int const saved = dup(STDERR_FILENO);
  int const sink = open("/dev/null", O_WRONLY);
  if (saved >= 0 && sink >= 0) dup2(sink, STDERR_FILENO);
  bool const result = probe();
  std::fflush(stderr);
  if (saved >= 0) {
    dup2(saved, STDERR_FILENO);
    close(saved);
  }
  if (sink >= 0) close(sink);
  return result;
}

bool mmg3d_lagrangian() {
  return quietly([] {
    Mmg3dSession session;
    MMG3D_Set_iparameter(session.mesh, session.met, MMG3D_IPARAM_verbose, -1);
    return MMG3D_Set_iparameter(session.mesh, session.met, MMG3D_IPARAM_lag, 0) == 1;
  });
}

bool mmg2d_lagrangian() {
  return quietly([] {
    Mmg2dSession session;
    MMG2D_Set_iparameter(session.mesh, session.met, MMG2D_IPARAM_verbose, -1);
    return MMG2D_Set_iparameter(session.mesh, session.met, MMG2D_IPARAM_lag, 0) == 1;
  });
}
#endif

// ---------------------------------------------------------------- identity

json::Value identity() {
#ifndef PHYDRAX_MMG_WITH_PARMMG
  bool const collective = false;
  char const* worker = "phydrax-mmg-worker";
  json::Array backends = {"mmg2d", "mmg3d", "mmgs"};
  json::Array programs = {"lagrangian", "levelset", "remesh"};
  json::Value lagrangian = json::Object{{"mmg2d", mmg2d_lagrangian()}, {"mmg3d", mmg3d_lagrangian()}};
  json::Value parmmg = nullptr;
#else
  bool const collective = true;
  char const* worker = "phydrax-parmmg-worker";
  json::Array backends = {"mmg3d"};
  json::Array programs = {"remesh"};
  json::Value lagrangian = json::Object{{"mmg3d", false}};
  json::Value parmmg = json::Object{{"git_commit", PMMG_GIT_COMMIT},
                                    {"release", PMMG_VERSION_RELEASE},
                                    {"release_date", PMMG_RELEASE_DATE}};
#endif
  return json::Object{
      {"backends", std::move(backends)},
      {"collective", collective},
      {"compiler", __VERSION__},
      {"git_commit", MMG_GIT_COMMIT},
      {"integer_bytes", static_cast<std::int64_t>(sizeof(MMG5_int))},
      {"interpolation", "p1-barycentric-closest-simplex"},
      {"lagrangian", std::move(lagrangian)},
      {"library", "mmg"},
      {"parmmg", std::move(parmmg)},
      {"programs", std::move(programs)},
      {"release", MMG_VERSION_RELEASE},
      {"release_date", MMG_RELEASE_DATE},
      {"worker", worker},
  };
}

}  // namespace

#ifdef PHYDRAX_MMG_WITH_PARMMG
#include "parmmg_route.hpp"
#endif

int main(int argc, char** argv) {
#ifdef PHYDRAX_MMG_WITH_PARMMG
  MPI_Init(&argc, &argv);
  int const status = phydrax::worker::serve_collective(MPI_COMM_WORLD, identity(), collective_adapt);
  MPI_Finalize();
  return status;
#else
  (void)argc;
  (void)argv;
  return phydrax::worker::serve(identity(), adapt);
#endif
}
