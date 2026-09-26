// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Collective ParMmg route of worker.cpp (compiled with PHYDRAX_MMG_WITH_PARMMG).
//
// Every rank reads the same centralized exchange input; rank zero hands the
// mesh to PMMG_parmmglib_centralized, ParMmg partitions and remeshes it across
// the communicator and keeps the result distributed. Each rank writes its part
// "rank-<r>" (local vertices with ParMmg's global vertex numbering and owner
// rank, cells, boundary triangles, metric, transferred fields); rank zero
// declares the parts after every rank finished its own directory.
#pragma once

namespace {

class ParMmgSession {
 public:
  explicit ParMmgSession(MPI_Comm comm) {
    if (PMMG_Init_parMesh(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_pMesh, PMMG_ARG_pMet,
                          PMMG_ARG_dim, 3, PMMG_ARG_MPIComm, comm, PMMG_ARG_end) != 1)
      throw Failure("library_failure", "ParMmg rejected PMMG_Init_parMesh");
  }
  ~ParMmgSession() { PMMG_Free_all(PMMG_ARG_start, PMMG_ARG_ppParMesh, &parmesh, PMMG_ARG_end); }
  ParMmgSession(ParMmgSession const&) = delete;
  ParMmgSession& operator=(ParMmgSession const&) = delete;
  PMMG_pParMesh parmesh = nullptr;
};

std::vector<int> as_int(std::vector<MMG5_int> const& values) { return {values.begin(), values.end()}; }

void load_parmmg(PMMG_pParMesh parmesh, Mesh const& source, Solutions const& solutions) {
  int const n = to_mmg(source.vertex_count()), m = to_mmg(source.cell_count());
  int const t = to_mmg(static_cast<std::int64_t>(source.triangle_references.size()));
  int const e = to_mmg(static_cast<std::int64_t>(source.edge_references.size()));
  check(PMMG_Set_meshSize(parmesh, n, m, 0, t, 0, e), "PMMG_Set_meshSize");
  std::vector<double> vertices(source.vertices);
  check(PMMG_Set_vertices(parmesh, vertices.data(), nullptr), "PMMG_Set_vertices");
  auto cells = as_int(one_based(source.cells));
  auto cell_refs = as_int(references(source.cell_references));
  check(PMMG_Set_tetrahedra(parmesh, cells.data(), cell_refs.data()), "PMMG_Set_tetrahedra");
  if (t > 0) {
    auto triangles = as_int(one_based(source.triangles));
    auto triangle_refs = as_int(references(source.triangle_references));
    check(PMMG_Set_triangles(parmesh, triangles.data(), triangle_refs.data()), "PMMG_Set_triangles");
  }
  if (e > 0) {
    auto edges = as_int(one_based(source.edges));
    auto edge_refs = as_int(references(source.edge_references));
    check(PMMG_Set_edges(parmesh, edges.data(), edge_refs.data()), "PMMG_Set_edges");
  }
  for (int k = 0; k < n; ++k)
    if (source.vertex_required[k]) check(PMMG_Set_requiredVertex(parmesh, k + 1), "required vertex");
  for (int k = 0; k < m; ++k)
    if (source.cell_required[k]) check(PMMG_Set_requiredTetrahedron(parmesh, k + 1), "required tetrahedron");
  for (int k = 0; k < t; ++k)
    if (source.triangle_required[k]) check(PMMG_Set_requiredTriangle(parmesh, k + 1), "required triangle");
  for (int k = 0; k < e; ++k) {
    if (source.edge_required[k]) check(PMMG_Set_requiredEdge(parmesh, k + 1), "required edge");
    if (source.edge_ridges[k]) check(PMMG_Set_ridge(parmesh, k + 1), "ridge");
  }
  if (solutions.metric_width == 0) return;
  check(PMMG_Set_metSize(parmesh, MMG5_Vertex, n, metric_kind(solutions.metric_width)), "PMMG_Set_metSize");
  std::vector<double> metric(solutions.metric);
  check(solutions.metric_width == 1 ? PMMG_Set_scalarMets(parmesh, metric.data())
                                    : PMMG_Set_tensorMets(parmesh, metric.data()),
        "PMMG metric values");
}

void apply_parmmg_controls(PMMG_pParMesh parmesh, Controls const& controls) {
  auto integer = [&](int id, int value) { return PMMG_Set_iparameter(parmesh, id, value); };
  auto real = [&](int id, double value) { return PMMG_Set_dparameter(parmesh, id, value); };
  check(integer(PMMG_IPARAM_verbose, -1), "verbosity");
  check(integer(PMMG_IPARAM_mmgVerbose, -1), "Mmg verbosity");
  if (controls.memory_megabytes > 0) check(integer(PMMG_IPARAM_mem, to_mmg(controls.memory_megabytes)), "memory bound");
  check(integer(PMMG_IPARAM_distributedOutput, 1), "distributed output");
  check(integer(PMMG_IPARAM_globalNum, 1), "global numbering");
  check(real(PMMG_DPARAM_hausd, controls.hausdorff_distance), "hausd");
  if (controls.minimum_size) check(real(PMMG_DPARAM_hmin, *controls.minimum_size), "hmin");
  if (controls.maximum_size) check(real(PMMG_DPARAM_hmax, *controls.maximum_size), "hmax");
  if (controls.gradation) check(real(PMMG_DPARAM_hgrad, *controls.gradation), "hgrad");
  check(integer(PMMG_IPARAM_angle, controls.angle_detection ? 1 : 0), "angle detection");
  if (controls.angle_detection) check(real(PMMG_DPARAM_angleDetection, *controls.angle_detection), "angle");
  check(integer(PMMG_IPARAM_noinsert, controls.insertion ? 0 : 1), "noinsert");
  check(integer(PMMG_IPARAM_noswap, controls.swapping ? 0 : 1), "noswap");
  check(integer(PMMG_IPARAM_nomove, controls.relocation ? 0 : 1), "nomove");
  check(integer(PMMG_IPARAM_nosurf, controls.surface_modification ? 0 : 1), "nosurf");
  check(integer(PMMG_IPARAM_optim, controls.optimize ? 1 : 0), "optim");
}

void check_parmmg(int status) {
  switch (status) {
    case PMMG_SUCCESS: return;
    case PMMG_LOWFAILURE:
      throw Failure("library_failure",
                    "ParMmg returned PMMG_LOWFAILURE: the distributed mesh is conforming but the "
                    "requested adaptation could not be completed");
    default:
      throw Failure("library_failure", "ParMmg returned PMMG_STRONGFAILURE: no usable mesh was produced");
  }
}

struct Part {
  Adapted adapted;
  std::vector<std::int64_t> global_vertex_ids, vertex_owners;
};

Part extract_parmmg(PMMG_pParMesh parmesh) {
  int np = 0, ne = 0, nprism = 0, nt = 0, nquad = 0, na = 0;
  check(PMMG_Get_meshSize(parmesh, &np, &ne, &nprism, &nt, &nquad, &na), "PMMG_Get_meshSize");
  if (nprism != 0 || nquad != 0) throw Failure("library_failure", "ParMmg returned prisms or quadrilaterals");
  Part part;
  Adapted& adapted = part.adapted;
  adapted.mesh.dimension = 3;
  adapted.mesh.arity = 4;
  adapted.mesh.vertices.resize(static_cast<std::size_t>(np) * 3);
  std::vector<int> vertex_refs(np), corners(np), required(np);
  check(PMMG_Get_vertices(parmesh, adapted.mesh.vertices.data(), vertex_refs.data(), corners.data(),
                          required.data()),
        "PMMG_Get_vertices");
  std::vector<int> tetrahedra(static_cast<std::size_t>(ne) * 4), tetrahedron_refs(ne), tetrahedron_required(ne);
  check(PMMG_Get_tetrahedra(parmesh, tetrahedra.data(), tetrahedron_refs.data(), tetrahedron_required.data()),
        "PMMG_Get_tetrahedra");
  std::vector<int> triangles(static_cast<std::size_t>(nt) * 3), triangle_refs(nt), triangle_required(nt);
  if (nt > 0)
    check(PMMG_Get_triangles(parmesh, triangles.data(), triangle_refs.data(), triangle_required.data()),
          "PMMG_Get_triangles");
  std::vector<int> global(np), owner(np);
  check(PMMG_Get_verticesGloNum(parmesh, global.data(), owner.data()), "PMMG_Get_verticesGloNum");
  adapted.mesh.cells = zero_based(std::vector<MMG5_int>(tetrahedra.begin(), tetrahedra.end()));
  adapted.mesh.cell_references.assign(tetrahedron_refs.begin(), tetrahedron_refs.end());
  adapted.facets = zero_based(std::vector<MMG5_int>(triangles.begin(), triangles.end()));
  adapted.facet_references.assign(triangle_refs.begin(), triangle_refs.end());
  part.global_vertex_ids.assign(global.begin(), global.end());
  part.vertex_owners.assign(owner.begin(), owner.end());
  int entity = 0, count = 0, kind = 0;
  if (PMMG_Get_metSize(parmesh, &entity, &count, &kind) == 1 && entity == MMG5_Vertex && count == np &&
      count > 0) {
    if (kind == MMG5_Scalar) {
      adapted.metric.resize(static_cast<std::size_t>(np));
      check(PMMG_Get_scalarMets(parmesh, adapted.metric.data()), "PMMG metric extraction");
      adapted.metric_width = 1;
    } else if (kind == MMG5_Tensor) {
      adapted.metric.resize(static_cast<std::size_t>(np) * 6);
      check(PMMG_Get_tensorMets(parmesh, adapted.metric.data()), "PMMG metric extraction");
      adapted.metric_width = 6;
      ambient_tensors(parmesh->listgrp[0].mesh, parmesh->listgrp[0].met, adapted);
    }
  }
  return part;
}

json::Value collective_adapt(phydrax::worker::Request const& request) {
  MPI_Comm const comm = MPI_COMM_WORLD;
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  Backend const backend = parse_backend(request.operation);
  Controls const controls = read_controls(request.parameters);
  if (backend != Backend::mmg3d || controls.program != Program::remesh)
    throw Failure("unsupported", "The ParMmg worker performs tetrahedral remeshing only");
  exchange::Input const input = request.input();
  Mesh const source = read_mesh(input, backend);
  Solutions const solutions = read_solutions(input, source, controls.program);
  ParMmgSession session(comm);
  if (rank == 0) load_parmmg(session.parmesh, source, solutions);
  apply_parmmg_controls(session.parmesh, controls);
  check_parmmg(PMMG_parmmglib_centralized(session.parmesh));
  Part const part = extract_parmmg(session.parmesh);
  std::string const name = "rank-" + std::to_string(rank);
  exchange::Output output = exchange::Output::create_part(request.output_directory, name,
                                                          request.maximum_output_bytes / size);
  write_adapted(output, part.adapted);
  auto const n = static_cast<std::uint64_t>(part.adapted.mesh.vertex_count());
  output.add<std::int64_t>("global_vertex_ids", {n}, part.global_vertex_ids);
  output.add<std::int64_t>("vertex_owners", {n}, part.vertex_owners);
  std::vector<std::int64_t> const targets = required_vertex_targets(source, part.adapted.mesh);
  output.add<std::int64_t>("required_vertex_targets", {targets.size()}, targets);
  std::vector<std::uint8_t> owned(part.vertex_owners.size());
  for (std::size_t vertex = 0; vertex < owned.size(); ++vertex)
    owned[vertex] = part.vertex_owners[vertex] == rank ? 1 : 0;
  json::Value interpolation =
      transfer_fields(backend, input, source, solutions, controls, part.adapted, owned, output);
  output.finish();
  // Collective evidence: counts sum over ranks, the projection distance is the maximum.
  std::int64_t counts[2] = {0, 0}, totals[2] = {0, 0};
  double distance = 0.0, maximum = 0.0;
  if (!interpolation.is_null()) {
    counts[0] = interpolation.at("located").as_int();
    counts[1] = interpolation.at("projected").as_int();
    distance = interpolation.at("maximum_projection_distance").as_double();
  }
  MPI_Allreduce(counts, totals, 2, MPI_INT64_T, MPI_SUM, comm);
  MPI_Allreduce(&distance, &maximum, 1, MPI_DOUBLE, MPI_MAX, comm);
  MPI_Barrier(comm);
  if (rank == 0) {
    exchange::Output root = request.output();
    for (int index = 0; index < size; ++index) root.declare_part("rank-" + std::to_string(index));
    root.finish();
  }
  if (!interpolation.is_null())
    interpolation = json::Object{
        {"located", totals[0]},
        {"maximum_projection_distance", maximum},
        {"method", interpolation.at("method")},
        {"projected", totals[1]},
        {"source_configuration", interpolation.at("source_configuration")},
        {"tolerance", interpolation.at("tolerance")},
    };
  return json::Object{
      {"adapted_metric", adapted_metric_label(part.adapted)},
      {"backend", backend_name(backend)},
      {"input_metric", metric_label(solutions.metric_width)},
      {"interpolation", std::move(interpolation)},
      {"memory_megabytes", controls.memory_megabytes},
      {"program", "remesh"},
      {"ranks", size},
  };
}

}  // namespace
