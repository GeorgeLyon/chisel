#ifndef MLIR_BINDINGS_SUPPORT_SCOPE_ID_H_
#define MLIR_BINDINGS_SUPPORT_SCOPE_ID_H_

#include <cassert>
#include <cstdint>
#include <limits>
#include <mutex>
#include <type_traits>

namespace mlir {
namespace bindings {
namespace detail {

template <typename RawValue, RawValue start, int step> class ScopeID {
  static_assert(std::is_integral_v<RawValue> && std::is_unsigned_v<RawValue>);
  RawValue rawValue;

  class Generator {
    std::mutex mutex;
    RawValue next = start;

  public:
    Generator() {}

    RawValue generate() {
      std::lock_guard<std::mutex> lock(mutex);
      auto result = next;
      next += step;
      assert(next != start && "ScopeID overflowed.");
      return result;
    }
  };

public:
  ScopeID()
      : rawValue([] {
          static Generator generator;
          return generator.generate();
        }()) {}
  bool operator==(const ScopeID &other) const {
    return rawValue == other.rawValue;
  }
};

} // namespace detail
} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_SCOPE_ID_H_