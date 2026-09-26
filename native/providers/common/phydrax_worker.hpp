// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Persistent provider-worker protocol shared with phydrax/_external_runtime.py.
//
// The worker keeps its original stdout as a private control channel and points
// descriptor 1 at stderr, so library output can never corrupt the protocol.
// Every control line is PREFIX followed by one canonical JSON object:
//   hello    {"hello":{"identity":{...},"memory_enforcement":...,"ranks":n},"ok":true}
//   request  {"input":dir,"maximum_input_bytes":n,"maximum_output_bytes":n,
//             "operation":name,"output":dir,"parameters":{...},"sequence":k}
//   response {"elapsed_seconds":s,"ok":true,"peak_rss_bytes":n,"result":{...},"sequence":k}
//            {"error":text,"kind":k,"ok":false,"peak_rss_bytes":n,"sequence":k}
// Requests arrive on stdin (rank zero only in collective mode). The "close"
// operation, or end of input, ends the session after a final response.
#pragma once

#include "phydrax_exchange.hpp"
#include "phydrax_json.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

#include <sys/resource.h>
#include <unistd.h>

#ifdef PHYDRAX_WORKER_WITH_MPI
#include <mpi.h>
#endif

namespace phydrax::worker {

constexpr char const* prefix = "@phydrax-worker ";
constexpr std::size_t maximum_request_bytes = 1 << 20;

// Kinds reported to the host: invalid_request, unsupported, resource_exhausted,
// library_failure. The host maps them onto its scientific failure categories.
class Failure : public std::runtime_error {
 public:
  Failure(std::string kind, std::string const& message)
      : std::runtime_error(message), kind(std::move(kind)) {}
  std::string kind;
};

struct Request {
  std::int64_t sequence = 0;
  std::string operation;
  json::Value parameters;
  std::string input_directory;
  std::string output_directory;
  std::uint64_t maximum_input_bytes = 0;
  std::uint64_t maximum_output_bytes = 0;

  exchange::Input input() const {
    return exchange::Input::read(input_directory, maximum_input_bytes);
  }
  exchange::Output output() const {
    return exchange::Output(output_directory, maximum_output_bytes);
  }
};

using Handler = std::function<json::Value(Request const&)>;

inline std::uint64_t peak_rss_bytes() {
  rusage usage{};
  getrusage(RUSAGE_SELF, &usage);
#ifdef __APPLE__
  return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
}

namespace detail {

struct Session {
  int protocol = -1;
  std::uint64_t memory_limit = 0;
  std::string memory_enforcement = "none";
};

// Called once on every rank before any library prints.
inline Session open_session(bool root) {
  Session session;
  std::fflush(stdout);
  if (root) {
    session.protocol = dup(STDOUT_FILENO);
    if (session.protocol < 0) throw std::runtime_error("Cannot reserve control channel");
  }
  if (dup2(STDERR_FILENO, STDOUT_FILENO) < 0)
    throw std::runtime_error("Cannot redirect library output");
  char const* limit = std::getenv("PHYDRAX_WORKER_MEMORY_LIMIT_BYTES");
  if (limit != nullptr) {
    session.memory_limit = std::stoull(limit);
    session.memory_enforcement = "peak-rss-audit";
#ifdef __linux__
    rlimit bound{static_cast<rlim_t>(session.memory_limit),
                 static_cast<rlim_t>(session.memory_limit)};
    if (setrlimit(RLIMIT_AS, &bound) == 0)
      session.memory_enforcement = "address-space-rlimit";
#endif
  }
  return session;
}

inline void send(Session const& session, json::Value const& message) {
  std::string line = prefix + json::dump(message) + "\n";
  char const* cursor = line.data();
  std::size_t remaining = line.size();
  while (remaining > 0) {
    ssize_t const written = write(session.protocol, cursor, remaining);
    if (written <= 0) throw std::runtime_error("Control channel closed");
    cursor += written;
    remaining -= static_cast<std::size_t>(written);
  }
}

// Returns false at end of input.
inline bool receive(std::string& line) {
  if (!std::getline(std::cin, line)) return false;
  if (line.size() > maximum_request_bytes)
    throw std::length_error("Control request exceeds its byte bound");
  return true;
}

inline Request decode(std::string const& line) {
  json::Value const message = json::parse(line, 32);
  Request request;
  request.sequence = message.at("sequence").as_int();
  request.operation = message.at("operation").as_string();
  if (request.operation == "close") return request;
  request.parameters = message.at("parameters");
  request.parameters.as_object();
  request.input_directory = message.at("input").as_string();
  request.output_directory = message.at("output").as_string();
  request.maximum_input_bytes =
      static_cast<std::uint64_t>(message.at("maximum_input_bytes").as_int());
  request.maximum_output_bytes =
      static_cast<std::uint64_t>(message.at("maximum_output_bytes").as_int());
  return request;
}

struct Outcome {
  bool ok = true;
  std::string kind;
  std::string error;
  json::Value result = json::Object{};
};

inline Outcome run(Handler const& handler, Request const& request) {
  Outcome outcome;
  try {
    outcome.result = handler(request);
    outcome.result.as_object();
  } catch (Failure const& failure) {
    outcome = Outcome{false, failure.kind, failure.what(), json::Object{}};
  } catch (std::bad_alloc const&) {
    outcome = Outcome{false, "resource_exhausted", "Worker allocation failed", json::Object{}};
  } catch (std::length_error const& error) {
    outcome = Outcome{false, "resource_exhausted", error.what(), json::Object{}};
  } catch (std::invalid_argument const& error) {
    outcome = Outcome{false, "invalid_request", error.what(), json::Object{}};
  } catch (std::exception const& error) {
    outcome = Outcome{false, "library_failure", error.what(), json::Object{}};
  }
  return outcome;
}

inline json::Value response(Request const& request, Outcome const& outcome,
                            std::uint64_t peak, double elapsed) {
  if (outcome.ok)
    return json::Object{{"elapsed_seconds", elapsed},
                        {"ok", true},
                        {"peak_rss_bytes", peak},
                        {"result", outcome.result},
                        {"sequence", request.sequence}};
  return json::Object{{"error", outcome.error},
                      {"kind", outcome.kind},
                      {"ok", false},
                      {"peak_rss_bytes", peak},
                      {"sequence", request.sequence}};
}

inline json::Value hello(Session const& session, json::Value const& identity, int ranks) {
  return json::Object{
      {"hello", json::Object{{"identity", identity},
                             {"memory_enforcement", session.memory_enforcement},
                             {"ranks", ranks}}},
      {"ok", true}};
}

inline Outcome enforce_memory(Session const& session, Outcome outcome,
                              std::uint64_t peak) {
  if (outcome.ok && session.memory_limit != 0 && peak > session.memory_limit)
    return Outcome{false, "resource_exhausted",
                   "Worker peak resident memory exceeds the configured limit",
                   json::Object{}};
  return outcome;
}

inline double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace detail

// Serial worker. Returns the process exit status.
inline int serve(json::Value const& identity, Handler const& handler) {
  detail::Session const session = detail::open_session(true);
  detail::send(session, detail::hello(session, identity, 1));
  std::string line;
  while (detail::receive(line)) {
    Request request;
    try {
      request = detail::decode(line);
    } catch (std::exception const& error) {
      detail::send(session, detail::response(
                                request, {false, "invalid_request", error.what(), json::Object{}},
                                peak_rss_bytes(), 0.0));
      return 1;
    }
    if (request.operation == "close") {
      detail::send(session, detail::response(request, {}, peak_rss_bytes(), 0.0));
      return 0;
    }
    auto const start = std::chrono::steady_clock::now();
    detail::Outcome outcome = detail::run(handler, request);
    std::uint64_t const peak = peak_rss_bytes();
    outcome = detail::enforce_memory(session, std::move(outcome), peak);
    detail::send(session,
                 detail::response(request, outcome, peak, detail::seconds_since(start)));
    // Exhausted memory cannot be returned to a clean state; end the session.
    if (!outcome.ok && outcome.kind == "resource_exhausted") return 1;
  }
  return 0;
}

#ifdef PHYDRAX_WORKER_WITH_MPI
namespace detail {

inline void broadcast(MPI_Comm comm, std::string& text, int root) {
  unsigned long long size = text.size();
  MPI_Bcast(&size, 1, MPI_UNSIGNED_LONG_LONG, root, comm);
  text.resize(static_cast<std::size_t>(size));
  if (size != 0) MPI_Bcast(text.data(), static_cast<int>(size), MPI_CHAR, root, comm);
}

// Every rank learns the lowest failing rank's outcome.
inline Outcome agree(MPI_Comm comm, Outcome local) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  int const candidate = local.ok ? size : rank;
  int failing = size;
  MPI_Allreduce(&candidate, &failing, 1, MPI_INT, MPI_MIN, comm);
  if (failing == size) return local;
  std::string kind = local.kind, error = local.error;
  broadcast(comm, kind, failing);
  broadcast(comm, error, failing);
  if (failing != 0) error = "rank " + std::to_string(failing) + ": " + error;
  return Outcome{false, kind, error, json::Object{}};
}

}  // namespace detail

// Collective worker: rank zero owns the control channel and broadcasts each
// request; every rank runs the handler; failures are agreed collectively and
// rank zero reports the result (handlers aggregate rank data themselves).
inline int serve_collective(MPI_Comm comm, json::Value const& identity,
                            Handler const& handler) {
  int rank = 0, size = 1;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);
  detail::Session const session = detail::open_session(rank == 0);
  if (rank == 0) detail::send(session, detail::hello(session, identity, size));
  while (true) {
    std::string line;
    if (rank == 0 && !detail::receive(line))
      line = "{\"operation\":\"close\",\"sequence\":-1}";
    detail::broadcast(comm, line, 0);
    Request request;
    try {
      request = detail::decode(line);
    } catch (std::exception const& error) {
      if (rank == 0)
        detail::send(session, detail::response(
                                  request, {false, "invalid_request", error.what(), json::Object{}},
                                  peak_rss_bytes(), 0.0));
      return 1;
    }
    if (request.operation == "close") {
      if (rank == 0 && request.sequence >= 0)
        detail::send(session, detail::response(request, {}, peak_rss_bytes(), 0.0));
      return 0;
    }
    auto const start = std::chrono::steady_clock::now();
    detail::Outcome outcome = detail::run(handler, request);
    unsigned long long local_peak = peak_rss_bytes(), peak = 0;
    MPI_Allreduce(&local_peak, &peak, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, comm);
    outcome = detail::enforce_memory(session, std::move(outcome), peak);
    outcome = detail::agree(comm, std::move(outcome));
    if (rank == 0)
      detail::send(session, detail::response(request, outcome, peak,
                                             detail::seconds_since(start)));
    if (!outcome.ok && outcome.kind == "resource_exhausted") return 1;
  }
}
#endif

}  // namespace phydrax::worker
