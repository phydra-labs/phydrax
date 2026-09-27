// Copyright © 2026 PHYDRA, Inc. All rights reserved.
// Minimal canonical JSON for provider-worker control messages and exchange
// manifests. Objects keep keys sorted, so dump() is the canonical form used by
// the Python runtime (compact separators, sorted keys, ASCII-only strings).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace phydrax::json {

class Value;
using Array = std::vector<Value>;
using Object = std::map<std::string, Value>;

class Value {
 public:
  enum class Kind { null, boolean, integer, real, string, array, object };

  Value() = default;
  Value(std::nullptr_t) {}
  Value(bool value) : kind_(Kind::boolean), boolean_(value) {}
  template <class T,
            std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>, int> = 0>
  Value(T value) : kind_(Kind::integer) {
    if constexpr (std::is_unsigned_v<T>) {
      if (static_cast<std::uint64_t>(value) >
          static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
        throw std::overflow_error("JSON integer exceeds int64");
    }
    integer_ = static_cast<std::int64_t>(value);
  }
  Value(double value) : kind_(Kind::real), real_(value) {
    if (!std::isfinite(value)) throw std::domain_error("JSON numbers must be finite");
  }
  Value(char const* value) : kind_(Kind::string), string_(value) {}
  Value(std::string value) : kind_(Kind::string), string_(std::move(value)) {}
  Value(Array value)
      : kind_(Kind::array), array_(std::make_shared<Array>(std::move(value))) {}
  Value(Object value)
      : kind_(Kind::object), object_(std::make_shared<Object>(std::move(value))) {}

  Kind kind() const { return kind_; }
  bool is_null() const { return kind_ == Kind::null; }

  bool as_bool() const {
    require(Kind::boolean, "boolean");
    return boolean_;
  }
  std::int64_t as_int() const {
    require(Kind::integer, "integer");
    return integer_;
  }
  double as_double() const {
    if (kind_ == Kind::integer) return static_cast<double>(integer_);
    require(Kind::real, "number");
    return real_;
  }
  std::string const& as_string() const {
    require(Kind::string, "string");
    return string_;
  }
  Array const& as_array() const {
    require(Kind::array, "array");
    return *array_;
  }
  Object const& as_object() const {
    require(Kind::object, "object");
    return *object_;
  }
  bool has(std::string const& key) const {
    return as_object().count(key) != 0;
  }
  Value const& at(std::string const& key) const {
    auto const& object = as_object();
    auto found = object.find(key);
    if (found == object.end())
      throw std::invalid_argument("Missing JSON field '" + key + "'");
    return found->second;
  }

 private:
  void require(Kind kind, char const* name) const {
    if (kind_ != kind)
      throw std::invalid_argument(std::string("Expected JSON ") + name);
  }

  Kind kind_ = Kind::null;
  bool boolean_ = false;
  std::int64_t integer_ = 0;
  double real_ = 0.0;
  std::string string_;
  std::shared_ptr<Array> array_;
  std::shared_ptr<Object> object_;
};

namespace detail {

inline void escape_code(std::uint32_t code, std::string& out) {
  char buffer[8];
  if (code >= 0x10000) {
    code -= 0x10000;
    escape_code(0xd800 + (code >> 10), out);
    escape_code(0xdc00 + (code & 0x3ff), out);
    return;
  }
  std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(code));
  out += buffer;
}

// Decodes one UTF-8 sequence at position; malformed bytes become U+FFFD.
inline std::uint32_t next_code(std::string const& text, std::size_t& position) {
  auto const lead = static_cast<unsigned char>(text[position++]);
  int extra = lead >= 0xf0 && lead < 0xf8 ? 3 : lead >= 0xe0 ? 2 : lead >= 0xc2 ? 1 : -1;
  if (lead >= 0xf8 || extra < 0) return 0xfffd;
  std::uint32_t code = lead & (0x3f >> extra);
  for (int index = 0; index < extra; ++index) {
    if (position >= text.size() ||
        (static_cast<unsigned char>(text[position]) & 0xc0) != 0x80)
      return 0xfffd;
    code = (code << 6) | (static_cast<unsigned char>(text[position++]) & 0x3f);
  }
  if (code > 0x10ffff || (code >= 0xd800 && code < 0xe000)) return 0xfffd;
  return code;
}

inline void dump_string(std::string const& text, std::string& out) {
  out.push_back('"');
  std::size_t position = 0;
  while (position < text.size()) {
    auto const character = static_cast<unsigned char>(text[position]);
    if (character >= 0x80) {
      escape_code(next_code(text, position), out);
      continue;
    }
    ++position;
    switch (character) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      default:
        if (character < 0x20) escape_code(character, out);
        else out.push_back(static_cast<char>(character));
    }
  }
  out.push_back('"');
}

inline void dump(Value const& value, std::string& out) {
  switch (value.kind()) {
    case Value::Kind::null: out += "null"; break;
    case Value::Kind::boolean: out += value.as_bool() ? "true" : "false"; break;
    case Value::Kind::integer: out += std::to_string(value.as_int()); break;
    case Value::Kind::real: {
      // Seventeen significant digits round-trip every binary64 value.
      char buffer[32];
      std::snprintf(buffer, sizeof(buffer), "%.17g", value.as_double());
      std::string text(buffer);
      if (text.find_first_of(".eEn") == std::string::npos) text += ".0";
      out += text;
      break;
    }
    case Value::Kind::string: dump_string(value.as_string(), out); break;
    case Value::Kind::array: {
      out.push_back('[');
      bool first = true;
      for (auto const& item : value.as_array()) {
        if (!first) out.push_back(',');
        first = false;
        dump(item, out);
      }
      out.push_back(']');
      break;
    }
    case Value::Kind::object: {
      out.push_back('{');
      bool first = true;
      for (auto const& [key, item] : value.as_object()) {
        if (!first) out.push_back(',');
        first = false;
        dump_string(key, out);
        out.push_back(':');
        dump(item, out);
      }
      out.push_back('}');
      break;
    }
  }
}

class Parser {
 public:
  Parser(std::string const& text, std::size_t maximum_depth)
      : text_(text), maximum_depth_(maximum_depth) {}

  Value parse() {
    Value value = parse_value(0);
    skip_space();
    if (position_ != text_.size())
      throw std::invalid_argument("Trailing JSON content");
    return value;
  }

 private:
  void skip_space() {
    while (position_ < text_.size() &&
           (text_[position_] == ' ' || text_[position_] == '\t' ||
            text_[position_] == '\n' || text_[position_] == '\r'))
      ++position_;
  }

  char peek() {
    skip_space();
    if (position_ >= text_.size()) throw std::invalid_argument("Truncated JSON");
    return text_[position_];
  }

  void expect(char const* literal) {
    for (char const* cursor = literal; *cursor; ++cursor, ++position_)
      if (position_ >= text_.size() || text_[position_] != *cursor)
        throw std::invalid_argument("Invalid JSON literal");
  }

  Value parse_value(std::size_t depth) {
    if (depth > maximum_depth_) throw std::invalid_argument("JSON nesting too deep");
    switch (peek()) {
      case 'n': expect("null"); return Value();
      case 't': expect("true"); return Value(true);
      case 'f': expect("false"); return Value(false);
      case '"': return Value(parse_string());
      case '[': return parse_array(depth);
      case '{': return parse_object(depth);
      default: return parse_number();
    }
  }

  static void append_utf8(std::uint32_t code, std::string& out) {
    if (code < 0x80) {
      out.push_back(static_cast<char>(code));
    } else if (code < 0x800) {
      out.push_back(static_cast<char>(0xc0 | (code >> 6)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else if (code < 0x10000) {
      out.push_back(static_cast<char>(0xe0 | (code >> 12)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    } else {
      out.push_back(static_cast<char>(0xf0 | (code >> 18)));
      out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3f)));
      out.push_back(static_cast<char>(0x80 | (code & 0x3f)));
    }
  }

  std::uint32_t parse_hex4() {
    if (position_ + 4 > text_.size()) throw std::invalid_argument("Truncated JSON escape");
    std::uint32_t code = 0;
    for (int index = 0; index < 4; ++index) {
      char const digit = text_[position_++];
      code <<= 4;
      if (digit >= '0' && digit <= '9') code |= static_cast<std::uint32_t>(digit - '0');
      else if (digit >= 'a' && digit <= 'f') code |= static_cast<std::uint32_t>(digit - 'a' + 10);
      else if (digit >= 'A' && digit <= 'F') code |= static_cast<std::uint32_t>(digit - 'A' + 10);
      else throw std::invalid_argument("Invalid JSON escape");
    }
    return code;
  }

  std::string parse_string() {
    ++position_;
    std::string out;
    while (true) {
      if (position_ >= text_.size()) throw std::invalid_argument("Unterminated JSON string");
      char const character = text_[position_++];
      if (character == '"') return out;
      if (static_cast<unsigned char>(character) < 0x20)
        throw std::invalid_argument("Control character in JSON string");
      if (character != '\\') {
        out.push_back(character);
        continue;
      }
      if (position_ >= text_.size()) throw std::invalid_argument("Truncated JSON escape");
      char const escape = text_[position_++];
      switch (escape) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          std::uint32_t code = parse_hex4();
          if (code >= 0xd800 && code < 0xdc00) {
            if (position_ + 2 > text_.size() || text_[position_] != '\\' ||
                text_[position_ + 1] != 'u')
              throw std::invalid_argument("Unpaired JSON surrogate");
            position_ += 2;
            std::uint32_t const low = parse_hex4();
            if (low < 0xdc00 || low >= 0xe000)
              throw std::invalid_argument("Invalid JSON surrogate");
            code = 0x10000 + ((code - 0xd800) << 10) + (low - 0xdc00);
          }
          append_utf8(code, out);
          break;
        }
        default: throw std::invalid_argument("Invalid JSON escape");
      }
    }
  }

  Value parse_number() {
    std::size_t const start = position_;
    bool real = false;
    if (position_ < text_.size() && text_[position_] == '-') ++position_;
    while (position_ < text_.size()) {
      char const character = text_[position_];
      if (character >= '0' && character <= '9') {
        ++position_;
      } else if (character == '.' || character == 'e' || character == 'E' ||
                 character == '+' || character == '-') {
        real = true;
        ++position_;
      } else {
        break;
      }
    }
    std::string const token = text_.substr(start, position_ - start);
    if (token.empty() || token == "-") throw std::invalid_argument("Invalid JSON number");
    std::size_t parsed = 0;
    if (!real) {
      std::int64_t const value = std::stoll(token, &parsed);
      if (parsed != token.size()) throw std::invalid_argument("Invalid JSON integer");
      return Value(value);
    }
    double const value = std::stod(token, &parsed);
    if (parsed != token.size() || !std::isfinite(value))
      throw std::invalid_argument("Invalid JSON real");
    return Value(value);
  }

  Value parse_array(std::size_t depth) {
    ++position_;
    Array items;
    if (peek() == ']') {
      ++position_;
      return Value(std::move(items));
    }
    while (true) {
      items.push_back(parse_value(depth + 1));
      char const separator = peek();
      ++position_;
      if (separator == ']') return Value(std::move(items));
      if (separator != ',') throw std::invalid_argument("Expected JSON array separator");
    }
  }

  Value parse_object(std::size_t depth) {
    ++position_;
    Object items;
    if (peek() == '}') {
      ++position_;
      return Value(std::move(items));
    }
    while (true) {
      if (peek() != '"') throw std::invalid_argument("Expected JSON object key");
      std::string key = parse_string();
      if (peek() != ':') throw std::invalid_argument("Expected JSON key separator");
      ++position_;
      if (items.count(key)) throw std::invalid_argument("Duplicate JSON key '" + key + "'");
      items.emplace(std::move(key), parse_value(depth + 1));
      char const separator = peek();
      ++position_;
      if (separator == '}') return Value(std::move(items));
      if (separator != ',') throw std::invalid_argument("Expected JSON object separator");
    }
  }

  std::string const& text_;
  std::size_t position_ = 0;
  std::size_t maximum_depth_;
};

}  // namespace detail

inline std::string dump(Value const& value) {
  std::string out;
  detail::dump(value, out);
  return out;
}

inline Value parse(std::string const& text, std::size_t maximum_depth = 64) {
  return detail::Parser(text, maximum_depth).parse();
}

}  // namespace phydrax::json
