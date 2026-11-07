// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Append-only face history in an owned temporary file and bounded page cache.
#pragma once

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <system_error>
#include <unordered_map>

namespace phx::mc::remesh {

template <class Row, size_t kWindows>
class ScratchRows {
  static constexpr size_t kRows = 1024;
  static constexpr size_t kNone = std::numeric_limits<size_t>::max();
  struct Page {
    size_t chunk = kNone;
    std::unique_ptr<Row[]> rows;
    size_t previous = kNone, next = kNone;
    bool dirty = false;
  };
 public:
  ~ScratchRows() { if (file_) std::fclose(file_); }
  size_t size() const { return size_; }
  Row get(size_t row) const { return access(row / kRows, true).rows[row % kRows]; }
  void set(size_t row, const Row& value) {
    Page& page = access(row / kRows, true);
    page.rows[row % kRows] = value;
    page.dirty = true;
  }
  void push(const Row& row) {
    Page& page = access(size_ / kRows, size_ % kRows != 0);
    page.rows[size_ % kRows] = row;
    page.dirty = true;
    ++size_;
  }
  void pop() { --size_; }
  void clear() {
    pages_ = {};
    locations_.clear();
    if (file_) std::fclose(file_);
    file_ = nullptr;
    newest_ = oldest_ = kNone;
    used_ = 0;
    size_ = 0;
  }
 private:
  static void fail(const char* operation) {
    const int code = errno ? errno : EIO;
    if (code == ENOMEM) throw std::bad_alloc();
    throw std::system_error(code, std::generic_category(), operation);
  }
  void seek(size_t chunk) const {
    constexpr size_t bytes = kRows * sizeof(Row);
    if (chunk > static_cast<size_t>(std::numeric_limits<int64_t>::max() / bytes))
      throw std::length_error("scratch face offset overflow");
    const int64_t offset = static_cast<int64_t>(chunk * bytes);
    errno = 0;
#if defined(_WIN32)
    if (_fseeki64(file_, offset, SEEK_SET) != 0) fail("scratch seek");
#else
    if (fseeko(file_, static_cast<off_t>(offset), SEEK_SET) != 0) fail("scratch seek");
#endif
  }
  void detach(size_t slot) const {
    Page& page = pages_[slot];
    if (page.previous == kNone) newest_ = page.next;
    else pages_[page.previous].next = page.next;
    if (page.next == kNone) oldest_ = page.previous;
    else pages_[page.next].previous = page.previous;
  }
  void attach(size_t slot) const {
    Page& page = pages_[slot];
    page.previous = kNone;
    page.next = newest_;
    if (newest_ == kNone) oldest_ = slot;
    else pages_[newest_].previous = slot;
    newest_ = slot;
  }
  Page& access(size_t chunk, bool load) const {
    if (!file_) {
      errno = 0;
      file_ = std::tmpfile();
      if (!file_) fail("scratch creation");
    }
    if (newest_ != kNone && pages_[newest_].chunk == chunk) {
      return pages_[newest_];
    }
    auto found = locations_.find(chunk);
    if (found != locations_.end()) {
      Page& page = pages_[found->second];
      if (found->second != newest_) {
        detach(found->second);
        attach(found->second);
      }
      return page;
    }
    const size_t slot = used_ < kWindows ? used_++ : oldest_;
    Page& page = pages_[slot];
    if (page.chunk != kNone) {
      detach(slot);
      if (page.dirty) {
        seek(page.chunk);
        errno = 0;
        if (std::fwrite(page.rows.get(), sizeof(Row), kRows, file_) != kRows) fail("scratch write");
      }
      locations_.erase(page.chunk);
    }
    if (!page.rows) page.rows.reset(new Row[kRows]);
    if (load) {
      seek(chunk);
      errno = 0;
      if (std::fread(page.rows.get(), sizeof(Row), kRows, file_) != kRows) fail("scratch read");
    } else {
      std::fill_n(page.rows.get(), kRows, Row{});
    }
    page.chunk = chunk;
    page.dirty = false;
    locations_.emplace(chunk, slot);
    attach(slot);
    return page;
  }
  mutable std::FILE* file_ = nullptr;
  mutable std::array<Page, kWindows> pages_{};
  mutable std::unordered_map<size_t, size_t> locations_;
  mutable size_t newest_ = kNone, oldest_ = kNone, used_ = 0;
  size_t size_ = 0;
};

using ScratchFaces = ScratchRows<std::array<int32_t, 3>, 1024>;

}  // namespace phx::mc::remesh
