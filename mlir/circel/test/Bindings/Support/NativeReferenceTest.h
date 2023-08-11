#pragma once

#include "circel/Bindings/Support/NativeReference.h"

struct TestHelper {
  int storedInt = 0;

  TestHelper() {}
  ~TestHelper();
};

template <>
class mlir::bindings::NativeReference<TestHelper>
    : public detail::TypedNativeReference<TestHelper> {
  explicit NativeReference(
      mlir::bindings::AnyNativeReference::Storage<TestHelper> *storage)
      : detail::TypedNativeReference<TestHelper>(storage) {}

  explicit NativeReference<TestHelper>(void *opaqueReference)
      : detail::TypedNativeReference<TestHelper>(opaqueReference) {}

public:
  explicit NativeReference(TestHelper &&value)
      : detail::TypedNativeReference<TestHelper>(
            new mlir::bindings::AnyNativeReference::Storage<TestHelper>(
                std::forward<TestHelper>(value))) {}

  template <typename... Args>
  static NativeReference<TestHelper> create(Args &&...args) {
    return NativeReference<TestHelper>(
        new mlir::bindings::AnyNativeReference::Storage<TestHelper>(
            std::forward<Args>(args)...));
  }

  static NativeReference<TestHelper>
  getFromOpaqueReference(void *opaqueReference) {
    auto reference = NativeReference<TestHelper>(opaqueReference);
    reference.template assertStorageTypeMatches<TestHelper>();
    return reference;
  }
};
