#pragma once

namespace phx::cuda {
// The caller owns status diagnostics and partition poisoning. Never retire
// resources unless their owning device is established and outstanding work drains.
template <class Operations> class TensorCleanup {
  Operations& operations_;
  int previous_ = -1;
  bool restore_ = false;
  bool ready_ = false;
 public:
  TensorCleanup(Operations& operations, int device) noexcept : operations_(operations) {
    if (device < 0) { ready_ = true; return; } // Host-only network planner.
    if (!operations_.get_device(previous_)) return;
    restore_ = previous_ != device;
    if (restore_ && !operations_.set_device(device)) return;
    ready_ = operations_.drain();
  }
  TensorCleanup(const TensorCleanup&) = delete;
  TensorCleanup& operator=(const TensorCleanup&) = delete;
  ~TensorCleanup() noexcept {
    if (restore_) operations_.set_device(previous_);
  }
  bool ready() const noexcept { return ready_; }
  // A failed free leaves both ownership and its exact budget charge intact.
  // Clearing the charge after success makes repeated retirement harmless.
  template <class Bytes, class Free, class Retire>
  bool release_scratch(void*& pointer, Bytes& charge, Free&& free, Retire&& retire) noexcept {
    if (!ready_) return false;
    if (pointer && !free(pointer)) return false;
    pointer = nullptr;
    if (charge) {
      retire(charge);
      charge = 0;
    }
    return true;
  }
};
}  // namespace phx::cuda
