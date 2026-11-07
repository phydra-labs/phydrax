// Copyright © 2026 PHYDRA, Inc. All rights reserved.
#pragma once

#include <cstdint>
#include <vector>

namespace phx::mc {

// Owns validated bytecode and constant banks; no mutable views escape. Workspace
// contents are supplied anew for each execution and retain public semantics.
class PreparedIntervalProgram {
 public:
  PreparedIntervalProgram() = default;
  PreparedIntervalProgram(const PreparedIntervalProgram&) = delete;
  PreparedIntervalProgram& operator=(const PreparedIntervalProgram&) = delete;
  PreparedIntervalProgram(PreparedIntervalProgram&&) = delete;
  PreparedIntervalProgram& operator=(PreparedIntervalProgram&&) = delete;
  int32_t bind(const int64_t* instructions, int64_t instruction_count,
               const int64_t* indices, int64_t index_count,
               const double* parameters, int64_t parameter_count, int64_t workspace_size);
  int32_t execute(double* lower, double* upper, int64_t workspace_size) const;
  const std::vector<int64_t>& instructions() const noexcept { return instructions_; }
  const std::vector<int64_t>& indices() const noexcept { return indices_; }
  const std::vector<double>& parameters() const noexcept { return parameters_; }
  void reset() noexcept { workspace_size_ = -1; }

 private:
  std::vector<int64_t> instructions_, indices_;
  std::vector<double> parameters_;
  int64_t workspace_size_ = -1;
};

}  // namespace phx::mc
