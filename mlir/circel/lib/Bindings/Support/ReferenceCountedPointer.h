#ifndef MLIR_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_INTERNAL_H_
#define MLIR_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_INTERNAL_H_

#include "mlir-bindings/Support/ReferenceCountedPointer.h"

#include <atomic>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>

namespace mlir {
namespace bindings {

/**
 A container that maintains a reference count
 */
class ReferenceCountedPointer::Implementation {
  struct ReleaseResult {
    bool containerShouldBeDestroyed;
  };
  struct ReferenceCount {
    friend class ReferenceCountedContainer;

    using RawValue = uint32_t;

    static inline constexpr RawValue min = std::numeric_limits<RawValue>::min();
    static inline constexpr RawValue initial =
        std::numeric_limits<RawValue>::min() + 1;
    static inline constexpr RawValue max = std::numeric_limits<RawValue>::max();
    std::atomic<RawValue> rawValue = ReferenceCount::initial;

    void increment() {
      auto count = ++rawValue;
      assert(count != ReferenceCount::max);
    }
    ReleaseResult decrement() {
      auto count = --rawValue;
      assert(count != ReferenceCount::min);
      return {count == ReferenceCount::initial};
    };
  };

  ReferenceCount referenceCount;

protected:
  Implementation() {}

public:
  virtual ~Implementation() = default;

  void retain() { referenceCount.increment(); }
  void release() {
    auto result = referenceCount.decrement();
    if (result.containerShouldBeDestroyed) {
      delete this;
    }
  }
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_INTERNAL_H_
