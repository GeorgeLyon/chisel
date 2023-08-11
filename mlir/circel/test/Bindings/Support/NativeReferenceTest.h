#pragma once

#include "circel/Bindings/Support/NativeReference.h"

struct TestHelper {
  int storedInt = 0;

  TestHelper() {}
  ~TestHelper();
};

template <>
class mlir::bindings::NativeReference<TestHelper>
    : public detail::NativeReferenceImpl<TestHelper> {
public:
  using detail::NativeReferenceImpl<TestHelper>::NativeReferenceImpl;
  using detail::NativeReferenceImpl<TestHelper>::create;
  using detail::NativeReferenceImpl<TestHelper>::getFromOpaqueReference;
};
