// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Binary array exchange directory shared with phydrax/_external_exchange.py.
//
// A directory holds one little-endian C-order NPY (format 1.0) file per array,
// named "<name>.npy", and "manifest.json": canonical JSON
//   {"arrays":[{"dtype":"<f8","name":"x","sha256":"...","shape":[n,3]},...],
//    "parts":["rank-0",...]}
// with arrays and parts sorted by name; sha256 digests the raw payload after
// the NPY header. A part is a nested exchange directory without parts (one per
// rank of a collective worker). No other files are permitted; readers verify
// every record before use.
#pragma once

#include "phydrax_json.hpp"
#include "phydrax_sha256.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <dirent.h>
#include <sys/stat.h>

namespace phydrax::exchange {

static_assert(sizeof(double) == 8, "Exchange requires IEEE binary64");

inline bool host_is_little_endian() {
  std::uint16_t const value = 1;
  unsigned char first = 0;
  std::memcpy(&first, &value, 1);
  return first == 1;
}

enum class DType { float64, int64, int32, uint64, uint8, int8 };

inline char const* descr(DType dtype) {
  switch (dtype) {
    case DType::float64: return "<f8";
    case DType::int64: return "<i8";
    case DType::int32: return "<i4";
    case DType::uint64: return "<u8";
    case DType::uint8: return "|u1";
    case DType::int8: return "|i1";
  }
  throw std::logic_error("Unknown exchange dtype");
}

inline std::size_t itemsize(DType dtype) {
  switch (dtype) {
    case DType::float64:
    case DType::int64:
    case DType::uint64: return 8;
    case DType::int32: return 4;
    case DType::uint8:
    case DType::int8: return 1;
  }
  throw std::logic_error("Unknown exchange dtype");
}

inline DType parse_descr(std::string const& text) {
  for (DType dtype : {DType::float64, DType::int64, DType::int32, DType::uint64,
                      DType::uint8, DType::int8})
    if (text == descr(dtype)) return dtype;
  throw std::invalid_argument("Unsupported exchange dtype '" + text + "'");
}

template <class T>
constexpr DType dtype_of() {
  if constexpr (std::is_same_v<T, double>) return DType::float64;
  else if constexpr (std::is_same_v<T, std::int64_t>) return DType::int64;
  else if constexpr (std::is_same_v<T, std::int32_t>) return DType::int32;
  else if constexpr (std::is_same_v<T, std::uint64_t>) return DType::uint64;
  else if constexpr (std::is_same_v<T, std::uint8_t>) return DType::uint8;
  else if constexpr (std::is_same_v<T, std::int8_t>) return DType::int8;
  else static_assert(sizeof(T) == 0, "Unsupported exchange element type");
}

inline bool valid_name(std::string const& name) {
  if (name.empty() || name.size() > 128) return false;
  for (std::size_t index = 0; index < name.size(); ++index) {
    char const character = name[index];
    bool const alphanumeric = (character >= 'a' && character <= 'z') ||
                              (character >= 'A' && character <= 'Z') ||
                              (character >= '0' && character <= '9');
    if (!alphanumeric && (index == 0 || (character != '_' && character != '-')))
      return false;
  }
  return true;
}

inline std::uint64_t element_count(std::vector<std::uint64_t> const& shape) {
  std::uint64_t count = 1;
  for (auto extent : shape) {
    if (extent != 0 && count > std::numeric_limits<std::uint64_t>::max() / extent)
      throw std::overflow_error("Exchange array extent overflows");
    count *= extent;
  }
  return count;
}

struct Array {
  std::string name;
  DType dtype = DType::float64;
  std::vector<std::uint64_t> shape;
  std::vector<unsigned char> bytes;

  std::uint64_t count() const { return element_count(shape); }

  template <class T>
  T const* data() const {
    if (dtype_of<T>() != dtype)
      throw std::invalid_argument("Exchange array '" + name + "' has dtype " +
                                  descr(dtype));
    return reinterpret_cast<T const*>(bytes.data());
  }

  template <class T>
  std::vector<T> to_vector() const {
    T const* values = data<T>();
    return std::vector<T>(values, values + count());
  }
};

namespace detail {

inline std::string npy_header(DType dtype, std::vector<std::uint64_t> const& shape) {
  std::string text = "{'descr': '";
  text += descr(dtype);
  text += "', 'fortran_order': False, 'shape': (";
  for (std::size_t index = 0; index < shape.size(); ++index) {
    text += std::to_string(shape[index]);
    if (shape.size() == 1 || index + 1 < shape.size()) text += ",";
    if (index + 1 < shape.size()) text += " ";
  }
  text += "), }";
  // NumPy aligns the payload to 64 bytes: magic(6)+version(2)+length(2)+text.
  std::size_t const unpadded = 10 + text.size() + 1;
  text.append((64 - unpadded % 64) % 64, ' ');
  text.push_back('\n');
  return text;
}

inline std::string field(std::string const& header, std::string const& key) {
  auto const position = header.find("'" + key + "':");
  if (position == std::string::npos)
    throw std::invalid_argument("NPY header lacks '" + key + "'");
  std::size_t cursor = position + key.size() + 3;
  while (cursor < header.size() && header[cursor] == ' ') ++cursor;
  return header.substr(cursor);
}

inline void parse_header(std::string const& header, DType& dtype,
                         std::vector<std::uint64_t>& shape) {
  std::string const descr_text = field(header, "descr");
  if (descr_text.empty() || descr_text[0] != '\'')
    throw std::invalid_argument("Invalid NPY descr");
  auto const close = descr_text.find('\'', 1);
  if (close == std::string::npos) throw std::invalid_argument("Invalid NPY descr");
  dtype = parse_descr(descr_text.substr(1, close - 1));
  if (field(header, "fortran_order").rfind("False", 0) != 0)
    throw std::invalid_argument("Exchange arrays must be C-ordered");
  std::string const shape_text = field(header, "shape");
  if (shape_text.empty() || shape_text[0] != '(')
    throw std::invalid_argument("Invalid NPY shape");
  auto const end = shape_text.find(')');
  if (end == std::string::npos) throw std::invalid_argument("Invalid NPY shape");
  shape.clear();
  std::string token;
  for (std::size_t index = 1; index <= end; ++index) {
    char const character = shape_text[index];
    if (character >= '0' && character <= '9') {
      token.push_back(character);
    } else if (character == ',' || character == ')') {
      if (!token.empty()) shape.push_back(std::stoull(token));
      token.clear();
    } else if (character != ' ') {
      throw std::invalid_argument("Invalid NPY shape");
    }
  }
}

inline std::vector<std::string> directory_entries(std::string const& directory) {
  DIR* handle = opendir(directory.c_str());
  if (handle == nullptr)
    throw std::invalid_argument("Cannot open exchange directory " + directory);
  std::vector<std::string> names;
  while (dirent* entry = readdir(handle)) {
    std::string const name = entry->d_name;
    if (name != "." && name != "..") names.push_back(name);
  }
  closedir(handle);
  return names;
}

}  // namespace detail

class Input {
 public:
  static Input read(std::string const& directory, std::uint64_t maximum_bytes) {
    std::uint64_t consumed = 0;
    return read_bounded(directory, maximum_bytes, consumed, true);
  }

  bool has(std::string const& name) const { return arrays_.count(name) != 0; }

  Array const& get(std::string const& name) const {
    auto found = arrays_.find(name);
    if (found == arrays_.end())
      throw std::invalid_argument("Exchange input lacks array '" + name + "'");
    return found->second;
  }

  Input const& part(std::string const& name) const {
    auto found = parts_.find(name);
    if (found == parts_.end())
      throw std::invalid_argument("Exchange input lacks part '" + name + "'");
    return found->second;
  }

  // Negative expected extents are wildcards.
  Array const& require(std::string const& name, DType dtype,
                       std::vector<std::int64_t> const& shape) const {
    Array const& array = get(name);
    bool matches = array.dtype == dtype && array.shape.size() == shape.size();
    for (std::size_t axis = 0; matches && axis < shape.size(); ++axis)
      matches = shape[axis] < 0 ||
                array.shape[axis] == static_cast<std::uint64_t>(shape[axis]);
    if (!matches)
      throw std::invalid_argument("Exchange array '" + name +
                                  "' has an unexpected dtype or shape");
    return array;
  }

  std::vector<std::string> names() const {
    std::vector<std::string> out;
    for (auto const& item : arrays_) out.push_back(item.first);
    return out;
  }

  std::string manifest_sha256;

 private:
  static Input read_bounded(std::string const& directory, std::uint64_t maximum_bytes,
                            std::uint64_t& total, bool allow_parts) {
    if (!host_is_little_endian())
      throw std::runtime_error("Exchange requires a little-endian host");
    Input input;
    std::ifstream manifest_stream(directory + "/manifest.json", std::ios::binary);
    if (!manifest_stream) throw std::invalid_argument("Exchange manifest is missing");
    std::string manifest((std::istreambuf_iterator<char>(manifest_stream)),
                         std::istreambuf_iterator<char>());
    if (manifest.size() > maximum_bytes - std::min(total, maximum_bytes))
      throw std::length_error("Exchange manifest exceeds its byte bound");
    total += manifest.size();
    input.manifest_sha256 = sha256::hexdigest(manifest.data(), manifest.size());
    json::Value const parsed = json::parse(manifest, 8);
    if (parsed.as_object().size() != 2 || !parsed.has("arrays") || !parsed.has("parts"))
      throw std::invalid_argument("Exchange manifest has unexpected fields");
    std::set<std::string> expected_files = {"manifest.json"};
    std::string previous;
    for (auto const& record : parsed.at("arrays").as_array()) {
      if (record.as_object().size() != 4)
        throw std::invalid_argument("Exchange record has unexpected fields");
      Array array;
      array.name = record.at("name").as_string();
      if (!valid_name(array.name) || (!previous.empty() && array.name <= previous))
        throw std::invalid_argument("Exchange names must be valid and sorted");
      previous = array.name;
      array.dtype = parse_descr(record.at("dtype").as_string());
      for (auto const& extent : record.at("shape").as_array()) {
        if (extent.as_int() < 0) throw std::invalid_argument("Negative exchange extent");
        array.shape.push_back(static_cast<std::uint64_t>(extent.as_int()));
      }
      std::uint64_t const count = array.count();
      if (count > maximum_bytes / itemsize(array.dtype))
        throw std::length_error("Exchange array exceeds its byte bound");
      std::uint64_t const payload = count * itemsize(array.dtype);
      if (payload > maximum_bytes - std::min(total, maximum_bytes))
        throw std::length_error("Exchange directory exceeds its byte bound");
      total += payload;
      std::string const file = array.name + ".npy";
      expected_files.insert(file);
      std::ifstream stream(directory + "/" + file, std::ios::binary | std::ios::ate);
      if (!stream) throw std::invalid_argument("Exchange array file is missing");
      auto const file_size = static_cast<std::uint64_t>(stream.tellg());
      stream.seekg(0);
      char prefix[10];
      stream.read(prefix, 10);
      if (!stream || std::memcmp(prefix, "\x93NUMPY\x01\x00", 8) != 0)
        throw std::invalid_argument("Exchange arrays must be NPY format 1.0");
      std::uint16_t header_length = static_cast<unsigned char>(prefix[8]) |
                                    (static_cast<unsigned char>(prefix[9]) << 8);
      std::string header(header_length, '\0');
      stream.read(header.data(), header_length);
      if (!stream) throw std::invalid_argument("Truncated NPY header");
      DType header_dtype;
      std::vector<std::uint64_t> header_shape;
      detail::parse_header(header, header_dtype, header_shape);
      if (header_dtype != array.dtype || header_shape != array.shape)
        throw std::invalid_argument("NPY header contradicts exchange manifest");
      if (file_size != 10 + header_length + payload)
        throw std::invalid_argument("Exchange array file size is inconsistent");
      array.bytes.resize(static_cast<std::size_t>(payload));
      if (payload != 0) {
        stream.read(reinterpret_cast<char*>(array.bytes.data()),
                    static_cast<std::streamsize>(payload));
        if (!stream) throw std::invalid_argument("Truncated exchange payload");
      }
      if (sha256::hexdigest(array.bytes.data(), array.bytes.size()) !=
          record.at("sha256").as_string())
        throw std::invalid_argument("Exchange checksum mismatch for '" + array.name + "'");
      input.arrays_.emplace(array.name, std::move(array));
    }
    previous.clear();
    for (auto const& entry : parsed.at("parts").as_array()) {
      std::string const& name = entry.as_string();
      if (!allow_parts || !valid_name(name) || (!previous.empty() && name <= previous) ||
          expected_files.count(name + ".npy") || name == "manifest")
        throw std::invalid_argument("Exchange parts must be valid, sorted, and unnested");
      previous = name;
      expected_files.insert(name);
      input.parts_.emplace(name,
                           read_bounded(directory + "/" + name, maximum_bytes, total, false));
    }
    auto const entries = detail::directory_entries(directory);
    if (std::set<std::string>(entries.begin(), entries.end()) != expected_files ||
        entries.size() != expected_files.size())
      throw std::invalid_argument("Exchange directory contains undeclared files");
    return input;
  }

  std::map<std::string, Array> arrays_;
  std::map<std::string, Input> parts_;
};

class Output {
 public:
  Output(std::string directory, std::uint64_t maximum_bytes)
      : directory_(std::move(directory)), maximum_bytes_(maximum_bytes) {}

  // Creates the nested directory of a part; its owner finishes it independently
  // and the parent declares it with declare_part before finishing itself.
  static Output create_part(std::string const& parent, std::string const& name,
                            std::uint64_t maximum_bytes) {
    if (!valid_name(name)) throw std::invalid_argument("Invalid exchange part name");
    std::string const directory = parent + "/" + name;
    if (mkdir(directory.c_str(), 0700) != 0)
      throw std::runtime_error("Cannot create exchange part " + directory);
    return Output(directory, maximum_bytes);
  }

  void declare_part(std::string const& name) {
    if (finished_) throw std::logic_error("Exchange output is already finished");
    if (!valid_name(name) || parts_.count(name))
      throw std::invalid_argument("Invalid or duplicate exchange part '" + name + "'");
    parts_.insert(name);
  }

  template <class T>
  void add(std::string const& name, std::vector<std::uint64_t> const& shape,
           T const* values) {
    if (finished_) throw std::logic_error("Exchange output is already finished");
    if (!valid_name(name) || records_.count(name))
      throw std::invalid_argument("Invalid or duplicate exchange output '" + name + "'");
    DType const dtype = dtype_of<T>();
    std::uint64_t const count = element_count(shape);
    if (count > maximum_bytes_ / sizeof(T) ||
        count * sizeof(T) > maximum_bytes_ - std::min(total_, maximum_bytes_))
      throw std::length_error("Exchange output exceeds its byte bound");
    std::uint64_t const payload = count * sizeof(T);
    total_ += payload;
    std::string const header = detail::npy_header(dtype, shape);
    std::ofstream stream(directory_ + "/" + name + ".npy", std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot create exchange output");
    stream.write("\x93NUMPY\x01\x00", 8);
    unsigned char const length[2] = {
        static_cast<unsigned char>(header.size() & 0xff),
        static_cast<unsigned char>((header.size() >> 8) & 0xff)};
    stream.write(reinterpret_cast<char const*>(length), 2);
    stream.write(header.data(), static_cast<std::streamsize>(header.size()));
    if (payload != 0)
      stream.write(reinterpret_cast<char const*>(values),
                   static_cast<std::streamsize>(payload));
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write exchange output");
    json::Array extents;
    for (auto extent : shape) extents.emplace_back(extent);
    records_.emplace(name, json::Object{
                               {"dtype", descr(dtype)},
                               {"name", name},
                               {"sha256", sha256::hexdigest(values, payload)},
                               {"shape", std::move(extents)},
                           });
  }

  template <class T>
  void add(std::string const& name, std::vector<std::uint64_t> const& shape,
           std::vector<T> const& values) {
    if (element_count(shape) != values.size())
      throw std::logic_error("Exchange output shape does not match its values");
    add(name, shape, values.data());
  }

  // Writes the manifest last so a partial directory is never well-formed.
  std::string finish() {
    if (finished_) throw std::logic_error("Exchange output is already finished");
    json::Array arrays;
    for (auto const& item : records_) arrays.push_back(item.second);
    json::Array parts;
    for (auto const& name : parts_) parts.emplace_back(name);
    std::string const manifest = json::dump(
        json::Object{{"arrays", std::move(arrays)}, {"parts", std::move(parts)}});
    if (manifest.size() > maximum_bytes_ - std::min(total_, maximum_bytes_))
      throw std::length_error("Exchange output exceeds its byte bound");
    std::ofstream stream(directory_ + "/manifest.json", std::ios::binary);
    stream.write(manifest.data(), static_cast<std::streamsize>(manifest.size()));
    stream.close();
    if (!stream) throw std::runtime_error("Cannot write exchange manifest");
    finished_ = true;
    return sha256::hexdigest(manifest.data(), manifest.size());
  }

 private:
  std::string directory_;
  std::uint64_t maximum_bytes_;
  std::uint64_t total_ = 0;
  bool finished_ = false;
  std::map<std::string, json::Value> records_;
  std::set<std::string> parts_;
};

}  // namespace phydrax::exchange
