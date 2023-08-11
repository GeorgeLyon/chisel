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
public:
  using detail::TypedNativeReference<TestHelper>::TypedNativeReference;
  using detail::TypedNativeReference<TestHelper>::create;
  using detail::TypedNativeReference<TestHelper>::getFromOpaqueReference;
};
