// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Persistent Omega_h metric-adaptation worker (phydrax-omega-h-worker).
//
// One "adapt" request imports a simplex carrier on rank zero, classifies it
// (region class IDs on cells, patch/zone class IDs on facets, remaining model
// entities by feature angle), distributes it over the worker communicator,
// approaches and adapts to the requested metric with explicit AdaptOpts, and
// transfers declared vertex (linear) and cell (conservative) fields. Every rank
// writes its own ghosted partition; with one rank the partition is the output.
#include <Omega_h_config.h>

#if defined(OMEGA_H_USE_MPI) != defined(PHYDRAX_WORKER_WITH_MPI)
#error "PHYDRAX_WORKER_WITH_MPI must match the MPI configuration of Omega_h"
#endif

#include "phydrax_worker.hpp"

#include <Omega_h_adapt.hpp>
#include <Omega_h_array_ops.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_class.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_matrix.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_metric.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace oh = Omega_h;
using namespace phydrax;

// Tag names are private to the worker so they never collide with Omega_h's
// reserved tags (coordinates, metric, class_dim, ...).
std::string field_tag(std::size_t index) {
  return "phydrax_field_" + std::to_string(index);
}
std::string integral_tag(std::size_t index) {
  return "phydrax_integral_" + std::to_string(index);
}

enum class FieldTransfer { linear, conserve };

struct FieldRequest {
  std::string name;
  FieldTransfer transfer = FieldTransfer::linear;
  std::optional<double> diffusion_tolerance;
};

struct AdaptRequest {
  int dimension = 0;
  double feature_angle = 0.0;
  std::int64_t maximum_iterations = 0;
  std::optional<double> gradation_rate;
  double min_length_desired = 0.0;
  double max_length_desired = 0.0;
  double max_length_allowed = 0.0;
  std::optional<double> min_quality_allowed;
  std::optional<double> min_quality_desired;
  std::int64_t nsliver_layers = 0;
  bool should_refine = true;
  bool should_coarsen = true;
  bool should_swap = true;
  bool should_coarsen_slivers = true;
  bool should_prevent_coarsen_flip = false;
  std::uint64_t maximum_vertices = 0;
  std::uint64_t maximum_cells = 0;
  std::uint64_t maximum_connectivity_entries = 0;
  std::vector<FieldRequest> fields;
};

double finite(json::Value const& value, char const* name) {
  double const number = value.as_double();
  if (!std::isfinite(number))
    throw std::invalid_argument(std::string(name) + " must be finite");
  return number;
}

std::optional<double> optional_finite(json::Value const& object, char const* name) {
  if (object.at(name).is_null()) return std::nullopt;
  return finite(object.at(name), name);
}

std::uint64_t positive_count(json::Value const& object, char const* name) {
  std::int64_t const value = object.at(name).as_int();
  if (value <= 0 || value > std::numeric_limits<oh::LO>::max())
    throw std::invalid_argument(std::string(name) + " must fit Omega_h local indexing");
  return static_cast<std::uint64_t>(value);
}

AdaptRequest parse(json::Value const& parameters) {
  AdaptRequest request;
  request.dimension = static_cast<int>(parameters.at("dimension").as_int());
  if (request.dimension != 2 && request.dimension != 3)
    throw std::invalid_argument("dimension must be 2 or 3");
  request.feature_angle = finite(parameters.at("feature_angle"), "feature_angle");
  if (!(request.feature_angle > 0.0 && request.feature_angle < M_PI))
    throw std::invalid_argument("feature_angle must lie in (0, pi)");
  request.maximum_iterations = parameters.at("maximum_iterations").as_int();
  if (request.maximum_iterations < 1)
    throw std::invalid_argument("maximum_iterations must be positive");
  request.gradation_rate = optional_finite(parameters, "gradation_rate");
  if (request.gradation_rate && *request.gradation_rate <= 0.0)
    throw std::invalid_argument("gradation_rate must be positive");
  json::Value const& adapt = parameters.at("adapt");
  request.min_length_desired = finite(adapt.at("min_length_desired"), "min_length_desired");
  request.max_length_desired = finite(adapt.at("max_length_desired"), "max_length_desired");
  request.max_length_allowed = finite(adapt.at("max_length_allowed"), "max_length_allowed");
  if (!(0.0 < request.min_length_desired &&
        request.min_length_desired < request.max_length_desired &&
        request.max_length_desired <= request.max_length_allowed))
    throw std::invalid_argument(
        "Length targets require 0 < min_length_desired < max_length_desired <= "
        "max_length_allowed");
  request.min_quality_allowed = optional_finite(adapt, "min_quality_allowed");
  request.min_quality_desired = optional_finite(adapt, "min_quality_desired");
  request.nsliver_layers = adapt.at("nsliver_layers").as_int();
  if (request.nsliver_layers < 0 || request.nsliver_layers >= 100)
    throw std::invalid_argument("nsliver_layers must lie in [0, 100)");
  request.should_refine = adapt.at("should_refine").as_bool();
  request.should_coarsen = adapt.at("should_coarsen").as_bool();
  request.should_swap = adapt.at("should_swap").as_bool();
  request.should_coarsen_slivers = adapt.at("should_coarsen_slivers").as_bool();
  request.should_prevent_coarsen_flip =
      adapt.at("should_prevent_coarsen_flip").as_bool();
  request.maximum_vertices = positive_count(parameters, "maximum_vertices");
  request.maximum_cells = positive_count(parameters, "maximum_cells");
  std::int64_t const connectivity =
      parameters.at("maximum_connectivity_entries").as_int();
  if (connectivity <= 0)
    throw std::invalid_argument("maximum_connectivity_entries must be positive");
  request.maximum_connectivity_entries = static_cast<std::uint64_t>(connectivity);
  for (auto const& entry : parameters.at("fields").as_array()) {
    FieldRequest field;
    field.name = entry.at("name").as_string();
    std::string const transfer = entry.at("transfer").as_string();
    if (transfer == "linear") {
      field.transfer = FieldTransfer::linear;
    } else if (transfer == "conserve") {
      field.transfer = FieldTransfer::conserve;
    } else {
      throw std::invalid_argument("Unknown field transfer '" + transfer + "'");
    }
    field.diffusion_tolerance = optional_finite(entry, "diffusion_tolerance");
    if (field.diffusion_tolerance &&
        (field.transfer != FieldTransfer::conserve || *field.diffusion_tolerance <= 0.0))
      throw std::invalid_argument(
          "diffusion_tolerance applies to conservative fields and must be positive");
    request.fields.push_back(std::move(field));
  }
  return request;
}

oh::LO local_count(std::uint64_t value, char const* name) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<oh::LO>::max()))
    throw std::length_error(std::string(name) + " exceeds Omega_h local indexing");
  return static_cast<oh::LO>(value);
}

template <class T>
oh::Read<T> device_copy(T const* values, std::uint64_t count, char const* name) {
  oh::HostWrite<T> host(local_count(count, name));
  std::copy(values, values + count, host.data());
  return oh::Read<T>(host.write());
}

template <class T>
std::vector<T> host_copy(oh::Read<T> values) {
  oh::HostRead<T> host(values);
  return std::vector<T>(host.data(), host.data() + host.size());
}

// Rank-local stages are agreed collectively, so a failure on one rank raises
// the same Failure on every rank before the next collective Omega_h call.
template <class Stage>
void agreed(oh::CommPtr const& comm, Stage&& stage) {
#ifdef PHYDRAX_WORKER_WITH_MPI
  worker::agreed(comm->get_impl(), std::forward<Stage>(stage));
#else
  (void)comm;
  stage();
#endif
}

using FacetKey = std::array<oh::LO, 3>;

FacetKey facet_key(oh::LO const* vertices, int count) {
  FacetKey key{-1, -1, -1};
  std::copy(vertices, vertices + count, key.begin());
  std::sort(key.begin(), key.begin() + count);
  return key;
}

struct DisjointSets {
  std::vector<oh::LO> parent;
  explicit DisjointSets(oh::LO count) : parent(static_cast<std::size_t>(count)) {
    std::iota(parent.begin(), parent.end(), 0);
  }
  oh::LO find(oh::LO item) {
    while (parent[item] != item) item = parent[item] = parent[parent[item]];
    return item;
  }
  void join(oh::LO first, oh::LO second) {
    first = find(first);
    second = find(second);
    if (first != second) parent[std::max(first, second)] = std::min(first, second);
  }
};

// Classifies a serial mesh. Cells carry region class IDs, requested facets
// their patch/zone class IDs, and every other boundary or region-interface
// facet the ID of its smooth same-region-pair component, split at sharp
// hinges (classify_by_angles) and junctions. finalize_classification then
// projects consistent classification onto hinges and vertices.
struct ClassificationSummary {
  std::int64_t generated_facet_classes = 0;
};

ClassificationSummary classify(oh::Mesh& mesh, double feature_angle,
                               std::vector<oh::ClassId> const& cell_classes,
                               exchange::Array const& facet_vertices,
                               exchange::Array const& facet_classes) {
  int const dim = mesh.dim();
  oh::classify_by_angles(&mesh, feature_angle);
  oh::LO const nsides = mesh.nents(dim - 1);
  auto const side_verts = host_copy(mesh.ask_verts_of(dim - 1));
  auto const sides2elems = mesh.ask_up(dim - 1, dim);
  auto const side_elem_offsets = host_copy(sides2elems.a2ab);
  auto const side_elems = host_copy(sides2elems.ab2b);
  auto const angle_side_dims = host_copy(mesh.get_array<oh::I8>(dim - 1, "class_dim"));
  auto const hinge_dims = host_copy(mesh.get_array<oh::I8>(dim - 2, "class_dim"));
  std::vector<oh::I8> side_dims(angle_side_dims);
  std::vector<oh::ClassId> side_ids(static_cast<std::size_t>(nsides), -1);

  std::map<FacetKey, oh::LO> lookup;
  for (oh::LO side = 0; side < nsides; ++side)
    lookup.emplace(facet_key(side_verts.data() + side * dim, dim), side);
  std::int32_t const* requested = facet_vertices.data<std::int32_t>();
  std::int32_t const* requested_ids = facet_classes.data<std::int32_t>();
  oh::ClassId next_id = 0;
  for (std::uint64_t row = 0; row < facet_classes.count(); ++row) {
    auto found = lookup.find(facet_key(requested + row * dim, dim));
    if (found == lookup.end())
      throw std::invalid_argument("A classified facet is not a facet of the mesh");
    if (side_ids[found->second] != -1)
      throw std::invalid_argument("A facet is classified more than once");
    if (requested_ids[row] < 0)
      throw std::invalid_argument("Facet class IDs must be non-negative");
    side_ids[found->second] = requested_ids[row];
    side_dims[found->second] = static_cast<oh::I8>(dim - 1);
    next_id = std::max(next_id, requested_ids[row] + 1);
  }

  // Region key of each side: (smaller, larger) adjacent region, -1 if exposed.
  std::vector<std::pair<oh::ClassId, oh::ClassId>> keys(static_cast<std::size_t>(nsides));
  for (oh::LO side = 0; side < nsides; ++side) {
    oh::LO const begin = side_elem_offsets[side], end = side_elem_offsets[side + 1];
    bool const exposed = end - begin != 2;
    oh::ClassId const first = cell_classes[side_elems[begin]];
    oh::ClassId const second = exposed ? -1 : cell_classes[side_elems[begin + 1]];
    keys[side] = exposed ? std::make_pair(first, second)
                         : std::make_pair(std::min(first, second), std::max(first, second));
    if (exposed || first != second) side_dims[side] = static_cast<oh::I8>(dim - 1);
  }

  auto const hinges2sides = mesh.ask_up(dim - 2, dim - 1);
  auto const hinge_side_offsets = host_copy(hinges2sides.a2ab);
  auto const hinge_sides = host_copy(hinges2sides.ab2b);
  DisjointSets components(nsides);
  for (oh::LO hinge = 0; hinge + 1 < static_cast<oh::LO>(hinge_side_offsets.size()); ++hinge) {
    if (hinge_dims[hinge] < dim - 1) continue;  // sharp by feature angle
    std::vector<oh::LO> incident;
    for (oh::LO item = hinge_side_offsets[hinge]; item < hinge_side_offsets[hinge + 1]; ++item)
      if (side_dims[hinge_sides[item]] == dim - 1) incident.push_back(hinge_sides[item]);
    if (incident.size() == 2 && side_ids[incident[0]] == -1 &&
        side_ids[incident[1]] == -1 && keys[incident[0]] == keys[incident[1]])
      components.join(incident[0], incident[1]);
  }
  ClassificationSummary summary;
  std::map<oh::LO, oh::ClassId> generated;
  for (oh::LO side = 0; side < nsides; ++side) {
    if (side_dims[side] != dim - 1 || side_ids[side] != -1) continue;
    auto inserted = generated.emplace(components.find(side), next_id);
    if (inserted.second) {
      ++next_id;
      ++summary.generated_facet_classes;
    }
    side_ids[side] = inserted.first->second;
  }
  mesh.add_tag<oh::ClassId>(dim, "class_id", 1,
                            device_copy(cell_classes.data(), cell_classes.size(), "cells"));
  mesh.set_tag<oh::I8>(dim - 1, "class_dim",
                       device_copy(side_dims.data(), side_dims.size(), "facets"));
  mesh.add_tag<oh::ClassId>(dim - 1, "class_id", 1,
                            device_copy(side_ids.data(), side_ids.size(), "facets"));
  oh::finalize_classification(&mesh);
  return summary;
}

struct ImportSummary {
  ClassificationSummary classification;
};

// Rank zero imports the carrier on its self communicator; tags then migrate
// with the mesh when it is partitioned out over the worker communicator.
ImportSummary import_carrier(oh::Library& library, oh::Mesh& mesh,
                             worker::Request const& request, AdaptRequest const& adapt) {
  exchange::Input const input = request.input();
  int const dim = adapt.dimension;
  int const metric_width = oh::symm_ncomps(dim);
  auto const& vertex_ids = input.require("vertex_global_ids", exchange::DType::int64, {-1});
  std::uint64_t const nverts = vertex_ids.count();
  auto const& coordinates =
      input.require("coordinates", exchange::DType::float64, {static_cast<std::int64_t>(nverts), dim});
  auto const& cells = input.require("cells", exchange::DType::int32, {-1, dim + 1});
  std::uint64_t const ncells = cells.shape[0];
  auto const& cell_ids =
      input.require("cell_global_ids", exchange::DType::int64, {static_cast<std::int64_t>(ncells)});
  auto const& cell_classes =
      input.require("cell_class_ids", exchange::DType::int32, {static_cast<std::int64_t>(ncells)});
  auto const& metric = input.require("metric", exchange::DType::float64,
                                     {static_cast<std::int64_t>(nverts), metric_width});
  auto const& facet_vertices = input.require("facet_vertices", exchange::DType::int32, {-1, dim});
  auto const& facet_classes = input.require(
      "facet_class_ids", exchange::DType::int32, {static_cast<std::int64_t>(facet_vertices.shape[0])});
  if (nverts == 0 || ncells == 0) throw std::invalid_argument("The carrier is empty");
  if (nverts > adapt.maximum_vertices || ncells > adapt.maximum_cells ||
      cells.count() > adapt.maximum_connectivity_entries)
    throw std::length_error("The carrier exceeds its entity bounds");
  for (double value : coordinates.to_vector<double>())
    if (!std::isfinite(value)) throw std::invalid_argument("Coordinates must be finite");
  for (double value : metric.to_vector<double>())
    if (!std::isfinite(value)) throw std::invalid_argument("Metric values must be finite");
  for (std::int32_t vertex : cells.to_vector<std::int32_t>())
    if (vertex < 0 || static_cast<std::uint64_t>(vertex) >= nverts)
      throw std::invalid_argument("Cell connectivity references a missing vertex");
  for (std::int32_t vertex : facet_vertices.to_vector<std::int32_t>())
    if (vertex < 0 || static_cast<std::uint64_t>(vertex) >= nverts)
      throw std::invalid_argument("Facet connectivity references a missing vertex");
  std::vector<oh::ClassId> regions = cell_classes.to_vector<std::int32_t>();
  for (oh::ClassId region : regions)
    if (region < 0) throw std::invalid_argument("Cell class IDs must be non-negative");

  oh::build_from_elems2verts(
      &mesh, library.self(), OMEGA_H_SIMPLEX, dim,
      device_copy(cells.data<std::int32_t>(), cells.count(), "connectivity"),
      device_copy(vertex_ids.data<std::int64_t>(), nverts, "vertices"));
  mesh.add_coords(device_copy(coordinates.data<double>(), coordinates.count(), "coordinates"));
  mesh.set_tag(dim, "global", device_copy(cell_ids.data<std::int64_t>(), ncells, "cells"));
  if (oh::get_min(mesh.ask_sizes()) <= 0.0)
    throw std::invalid_argument("Cells must be positively oriented with positive measure");
  ImportSummary summary;
  summary.classification =
      classify(mesh, adapt.feature_angle, regions, facet_vertices, facet_classes);
  oh::add_implied_metric_tag(&mesh);
  oh::add_metric_tag(
      &mesh,
      oh::symms_inria2osh(dim, device_copy(metric.data<double>(), metric.count(), "metric")),
      "target_metric");
  for (std::size_t index = 0; index < adapt.fields.size(); ++index) {
    FieldRequest const& field = adapt.fields[index];
    bool const vertex = field.transfer == FieldTransfer::linear;
    auto const& values = input.require(
        field.name, exchange::DType::float64,
        {static_cast<std::int64_t>(vertex ? nverts : ncells), -1});
    if (values.shape[1] == 0 || values.shape[1] > 64)
      throw std::invalid_argument("Field '" + field.name + "' needs 1 to 64 components");
    for (double value : values.to_vector<double>())
      if (!std::isfinite(value))
        throw std::invalid_argument("Field '" + field.name + "' must be finite");
    mesh.add_tag<oh::Real>(vertex ? 0 : dim, field_tag(index),
                           static_cast<oh::Int>(values.shape[1]),
                           device_copy(values.data<double>(), values.count(), "field"));
  }
  return summary;
}

// Integral of a cell density over owned cells, per component, reproducibly.
std::vector<double> owned_integral(oh::Mesh& mesh, std::string const& tag) {
  int const dim = mesh.dim();
  oh::Int const width = mesh.get_tagbase(dim, tag)->ncomps();
  auto const densities = host_copy(mesh.get_array<oh::Real>(dim, tag));
  auto const sizes = host_copy(mesh.ask_sizes());
  oh::HostWrite<oh::Real> masses(static_cast<oh::LO>(densities.size()));
  for (std::size_t cell = 0; cell < sizes.size(); ++cell)
    for (oh::Int component = 0; component < width; ++component)
      masses[cell * width + component] = densities[cell * width + component] * sizes[cell];
  auto const owned = mesh.owned_array(dim, oh::Reals(masses.write()), width);
  std::vector<double> integral(static_cast<std::size_t>(width));
  oh::repro_sum(mesh.comm(), owned, width, integral.data());
  return integral;
}

json::Array json_reals(std::vector<double> const& values) {
  return json::Array(values.begin(), values.end());
}

void configure(oh::AdaptOpts& options, AdaptRequest const& adapt) {
  options.verbosity = oh::SILENT;
  options.min_length_desired = adapt.min_length_desired;
  options.max_length_desired = adapt.max_length_desired;
  options.max_length_allowed = adapt.max_length_allowed;
  if (adapt.min_quality_allowed) options.min_quality_allowed = *adapt.min_quality_allowed;
  if (adapt.min_quality_desired) options.min_quality_desired = *adapt.min_quality_desired;
  if (!(0.0 <= options.min_quality_allowed &&
        options.min_quality_allowed <= options.min_quality_desired &&
        options.min_quality_desired <= 1.0))
    throw std::invalid_argument(
        "Quality targets require 0 <= min_quality_allowed <= min_quality_desired <= 1");
  options.nsliver_layers = static_cast<oh::Int>(adapt.nsliver_layers);
  options.should_refine = adapt.should_refine;
  options.should_coarsen = adapt.should_coarsen;
  options.should_swap = adapt.should_swap;
  options.should_coarsen_slivers = adapt.should_coarsen_slivers;
  options.should_prevent_coarsen_flip = adapt.should_prevent_coarsen_flip;
  for (std::size_t index = 0; index < adapt.fields.size(); ++index) {
    FieldRequest const& field = adapt.fields[index];
    std::string const tag = field_tag(index);
    switch (field.transfer) {
      case FieldTransfer::linear:
        options.xfer_opts.type_map[tag] = OMEGA_H_LINEAR_INTERP;
        break;
      case FieldTransfer::conserve:
        options.xfer_opts.type_map[tag] = OMEGA_H_CONSERVE;
        options.xfer_opts.integral_map[tag] = integral_tag(index);
        options.xfer_opts.integral_diffuse_map[integral_tag(index)] =
            field.diffusion_tolerance
                ? oh::VarCompareOpts{oh::VarCompareOpts::RELATIVE, *field.diffusion_tolerance, 0.0}
                : oh::VarCompareOpts::none();
        break;
    }
  }
}

json::Value effective_options(oh::AdaptOpts const& options, AdaptRequest const& adapt) {
  return json::Object{
      {"feature_angle", adapt.feature_angle},
      {"gradation_rate", adapt.gradation_rate ? json::Value(*adapt.gradation_rate) : json::Value()},
      {"max_length_allowed", options.max_length_allowed},
      {"max_length_desired", options.max_length_desired},
      {"maximum_iterations", adapt.maximum_iterations},
      {"min_length_desired", options.min_length_desired},
      {"min_quality_allowed", options.min_quality_allowed},
      {"min_quality_desired", options.min_quality_desired},
      {"nsliver_layers", static_cast<std::int64_t>(options.nsliver_layers)},
      {"should_coarsen", options.should_coarsen},
      {"should_coarsen_slivers", options.should_coarsen_slivers},
      {"should_prevent_coarsen_flip", options.should_prevent_coarsen_flip},
      {"should_refine", options.should_refine},
      {"should_swap", options.should_swap},
  };
}

// Writes one rank's ghosted partition in Omega_h local order.
void write_partition(oh::Mesh& mesh, AdaptRequest const& adapt, exchange::Output& output) {
  int const dim = mesh.dim();
  std::uint64_t const nverts = static_cast<std::uint64_t>(mesh.nverts());
  std::uint64_t const ncells = static_cast<std::uint64_t>(mesh.nelems());
  auto const coordinates = host_copy(mesh.coords());
  auto const metric =
      host_copy(oh::symms_osh2inria(dim, mesh.get_array<oh::Real>(0, "metric")));
  for (double value : coordinates)
    if (!std::isfinite(value)) throw std::runtime_error("Omega_h returned nonfinite coordinates");
  for (double value : metric)
    if (!std::isfinite(value)) throw std::runtime_error("Omega_h returned a nonfinite metric");
  auto const vertex_owners = mesh.ask_owners(0);
  auto const cell_owners = mesh.ask_owners(dim);
  output.add<std::int64_t>("vertex_global_ids", {nverts}, host_copy(mesh.globals(0)));
  output.add<double>("coordinates", {nverts, static_cast<std::uint64_t>(dim)}, coordinates);
  output.add<std::int32_t>("vertex_owner_ranks", {nverts}, host_copy(vertex_owners.ranks));
  output.add<std::int32_t>("vertex_owner_indices", {nverts}, host_copy(vertex_owners.idxs));
  output.add<double>("metric", {nverts, static_cast<std::uint64_t>(oh::symm_ncomps(dim))},
                     metric);
  output.add<std::int64_t>("cell_global_ids", {ncells}, host_copy(mesh.globals(dim)));
  output.add<std::int32_t>("cells", {ncells, static_cast<std::uint64_t>(dim + 1)},
                           host_copy(mesh.ask_elem_verts()));
  output.add<std::int32_t>("cell_owner_ranks", {ncells}, host_copy(cell_owners.ranks));
  output.add<std::int32_t>("cell_owner_indices", {ncells}, host_copy(cell_owners.idxs));
  output.add<std::int32_t>("cell_class_ids", {ncells},
                           host_copy(mesh.get_array<oh::ClassId>(dim, "class_id")));

  // Facets classified on model faces (boundaries, interfaces, patches).
  auto const side_dims = host_copy(mesh.get_array<oh::I8>(dim - 1, "class_dim"));
  auto const side_ids = host_copy(mesh.get_array<oh::ClassId>(dim - 1, "class_id"));
  auto const side_verts = host_copy(mesh.ask_verts_of(dim - 1));
  auto const side_globals = host_copy(mesh.globals(dim - 1));
  auto const side_owner_ranks = host_copy(mesh.ask_owners(dim - 1).ranks);
  std::vector<std::int32_t> facet_vertices, facet_classes, facet_owner_ranks;
  std::vector<std::int64_t> facet_globals;
  for (std::size_t side = 0; side < side_dims.size(); ++side) {
    if (side_dims[side] != dim - 1) continue;
    facet_vertices.insert(facet_vertices.end(), side_verts.begin() + side * dim,
                          side_verts.begin() + (side + 1) * dim);
    facet_classes.push_back(side_ids[side]);
    facet_globals.push_back(side_globals[side]);
    facet_owner_ranks.push_back(side_owner_ranks[side]);
  }
  std::uint64_t const nfacets = facet_classes.size();
  output.add<std::int32_t>("facet_vertices", {nfacets, static_cast<std::uint64_t>(dim)},
                           facet_vertices);
  output.add<std::int32_t>("facet_class_ids", {nfacets}, facet_classes);
  output.add<std::int64_t>("facet_global_ids", {nfacets}, facet_globals);
  output.add<std::int32_t>("facet_owner_ranks", {nfacets}, facet_owner_ranks);

  for (std::size_t index = 0; index < adapt.fields.size(); ++index) {
    FieldRequest const& field = adapt.fields[index];
    int const entity = field.transfer == FieldTransfer::linear ? 0 : dim;
    std::string const tag = field_tag(index);
    auto const values = host_copy(mesh.get_array<oh::Real>(entity, tag));
    for (double value : values)
      if (!std::isfinite(value))
        throw std::runtime_error("Omega_h returned a nonfinite field '" + field.name + "'");
    output.add<double>(field.name,
                       {static_cast<std::uint64_t>(mesh.nents(entity)),
                        static_cast<std::uint64_t>(mesh.get_tagbase(entity, tag)->ncomps())},
                       values);
  }
}

json::Value adapt_operation(oh::Library& library, worker::Request const& request) {
  oh::CommPtr const world = library.world();
  int const rank = world->rank(), size = world->size();
  AdaptRequest adapt;
  agreed(world, [&] { adapt = parse(request.parameters); });

  oh::Mesh mesh(&library);
  ImportSummary summary;
  agreed(world, [&] {
    if (rank == 0) summary = import_carrier(library, mesh, request, adapt);
  });
  agreed(world, [&] { mesh.set_comm(world); });
  if (size > 1) agreed(world, [&] { mesh.balance(); });
  agreed(world, [&] { mesh.set_parting(OMEGA_H_GHOSTED); });
  if (adapt.gradation_rate) {
    agreed(world, [&] {
      auto const graded = oh::limit_metric_gradation(
          &mesh, mesh.get_array<oh::Real>(0, "target_metric"),
          *adapt.gradation_rate);
      mesh.set_tag(0, "target_metric", graded);
    });
  }

  std::unique_ptr<oh::AdaptOpts> options;
  agreed(world, [&] {
    options = std::make_unique<oh::AdaptOpts>(&mesh);
    configure(*options, adapt);
  });

  std::vector<std::vector<double>> before;
  agreed(world, [&] { before.resize(adapt.fields.size()); });
  for (std::size_t index = 0; index < adapt.fields.size(); ++index) {
    if (adapt.fields[index].transfer == FieldTransfer::conserve)
      agreed(world, [&] { before[index] = owned_integral(mesh, field_tag(index)); });
  }

  std::int64_t iterations = 0;
  while (true) {
    bool approaching = false;
    agreed(world, [&] { approaching = oh::approach_metric(&mesh, *options); });
    if (!approaching) break;
    ++iterations;
    if (iterations > adapt.maximum_iterations)
      agreed(world, [&] {
        throw worker::Failure(
            "library_failure",
            "Omega_h did not reach the metric within maximum_iterations");
      });
    agreed(world, [&] { oh::adapt(&mesh, *options); });
  }
  agreed(world, [&] { oh::adapt(&mesh, *options); });
  agreed(world, [&] { mesh.set_parting(OMEGA_H_ELEM_BASED); });
  if (size > 1) agreed(world, [&] { mesh.balance(); });
  agreed(world, [&] { mesh.set_parting(OMEGA_H_GHOSTED, 1, false); });

  double minimum_quality = 0.0, maximum_quality = 0.0;
  double minimum_length = 0.0, maximum_length = 0.0;
  std::uint64_t global_vertices = 0, global_cells = 0, global_facets = 0;
  agreed(world, [&] {
    auto const qualities = oh::get_minmax(world, mesh.ask_qualities());
    auto const lengths = oh::get_minmax(world, mesh.ask_lengths());
    minimum_quality = qualities.min;
    maximum_quality = qualities.max;
    minimum_length = lengths.min;
    maximum_length = lengths.max;
    global_vertices = static_cast<std::uint64_t>(mesh.nglobal_ents(0));
    global_cells = static_cast<std::uint64_t>(mesh.nglobal_ents(adapt.dimension));
    global_facets = static_cast<std::uint64_t>(oh::get_sum(
        world, oh::land_each(
                   mesh.owned(adapt.dimension - 1),
                   oh::each_eq_to(
                       mesh.get_array<oh::I8>(adapt.dimension - 1, "class_dim"),
                       static_cast<oh::I8>(adapt.dimension - 1)))));
  });

  json::Array fields;
  agreed(world, [&] {
    for (std::size_t index = 0; index < adapt.fields.size(); ++index) {
      FieldRequest const& field = adapt.fields[index];
      json::Object record{{"name", field.name}};
      switch (field.transfer) {
        case FieldTransfer::linear:
          record.emplace("method", "OMEGA_H_LINEAR_INTERP");
          break;
        case FieldTransfer::conserve:
          record.emplace("method", "OMEGA_H_CONSERVE");
          record.emplace("integral_before", json_reals(before[index]));
          record.emplace("integral_after",
                         json_reals(owned_integral(mesh, field_tag(index))));
          break;
      }
      fields.push_back(std::move(record));
    }
  });

  constexpr std::uint64_t root_reserve = 4096;
  agreed(world, [&] {
    if (global_vertices > adapt.maximum_vertices ||
        global_cells > adapt.maximum_cells ||
        global_cells * static_cast<std::uint64_t>(adapt.dimension + 1) >
            adapt.maximum_connectivity_entries)
      throw std::length_error("The adapted mesh exceeds its entity bounds");
    if (request.maximum_output_bytes <= root_reserve)
      throw std::length_error("The output byte bound cannot hold a partition");
  });
  agreed(world, [&] {
    if (size == 1) {
      exchange::Output output = request.output();
      write_partition(mesh, adapt, output);
      output.finish();
      return;
    }
    exchange::Output part = exchange::Output::create_part(
        request.output_directory, "rank-" + std::to_string(rank),
        (request.maximum_output_bytes - root_reserve) /
            static_cast<std::uint64_t>(size));
    write_partition(mesh, adapt, part);
    part.finish();
  });
  agreed(world, [&] {
    if (size > 1 && rank == 0) {
      exchange::Output output = request.output();
      for (int part = 0; part < size; ++part)
        output.declare_part("rank-" + std::to_string(part));
      output.finish();
    }
  });
  return json::Object{
      {"cell_count", global_cells},
      {"classification",
       json::Object{{"generated_facet_classes",
                     summary.classification.generated_facet_classes}}},
      {"facet_count", global_facets},
      {"fields", std::move(fields)},
      {"iterations", iterations},
      {"maximum_length", maximum_length},
      {"maximum_quality", maximum_quality},
      {"minimum_length", minimum_length},
      {"minimum_quality", minimum_quality},
      {"options", effective_options(*options, adapt)},
      {"ranks", size},
      {"vertex_count", global_vertices},
  };
}

json::Value identity() {
  json::Object record{
      {"commit", OMEGA_H_COMMIT},
      {"library", "Omega_h"},
      {"operations", json::Array{"adapt"}},
      {"version", OMEGA_H_SEMVER},
      {"worker", "phydrax-omega-h-worker"},
  };
#ifdef OMEGA_H_CMAKE_ARGS
  record.emplace("cmake_arguments", OMEGA_H_CMAKE_ARGS);
#else
  record.emplace("cmake_arguments", json::Value());
#endif
#ifdef OMEGA_H_USE_MPI
  char text[MPI_MAX_LIBRARY_VERSION_STRING];
  int length = 0;
  MPI_Get_library_version(text, &length);
  std::string const mpi(text, static_cast<std::size_t>(std::find(text, text + length, '\0') - text));
  record.emplace("mpi", mpi.substr(0, mpi.find_first_of("\r\n")));
#else
  record.emplace("mpi", json::Value());
#endif
#ifdef OMEGA_H_USE_KOKKOS
  record.emplace("kokkos", true);
#else
  record.emplace("kokkos", false);
#endif
#ifdef OMEGA_H_USE_OPENMP
  record.emplace("openmp", true);
#else
  record.emplace("openmp", false);
#endif
  return record;
}

}  // namespace

int main(int argc, char** argv) {
  // Library initializes MPI when Omega_h is MPI-enabled and finalizes it last.
  oh::Library library(&argc, &argv);
  auto handler = [&library](worker::Request const& request) -> json::Value {
    if (request.operation != "adapt")
      throw worker::Failure("unsupported", "Unknown operation '" + request.operation + "'");
    return adapt_operation(library, request);
  };
#ifdef PHYDRAX_WORKER_WITH_MPI
  return worker::serve_collective(library.world()->get_impl(), identity(), handler);
#else
  return worker::serve(identity(), handler);
#endif
}
