// RUN: mlir-bindings-support-test 2>&1 | FileCheck %s

#include "circel/Bindings/Support/NativeReference.h"
#include "NativeReferenceTest.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/TypeID.h>

using namespace mlir::bindings;

static int deinitializations = 0;
TestHelper::~TestHelper() {
  { deinitializations++; }
}

void testMutability() {
  // CHECK-LABEL: @testMutability
  std::cout << "@testMutability" << std::endl;
  deinitializations = 0;
  void *opaqueReference = nullptr;
  {
    auto reference = NativeReference<TestHelper>::create();
    opaqueReference = reference.getRetainedOpaqueReference();
    reference->storedInt = 42;
    // CHECK-NEXT: 42
    std::cout << reference->storedInt << std::endl;
  }
  {
    auto reference =
        NativeReference<TestHelper>::getFromOpaqueReference(opaqueReference);
    AnyNativeReference::releaseOpaqueReference(opaqueReference);
    // CHECK-NEXT: 42
    std::cout << reference->storedInt << std::endl;
  }
  assert(deinitializations == 1);
}

void testReferenceCounting() {
  // CHECK-LABEL: @TestReferenceCounting
  std::cout << "@TestReferenceCounting" << std::endl;
  deinitializations = 0;
  void *opaqueReference = nullptr;
  {
    auto reference = NativeReference<TestHelper>::create();
    opaqueReference = reference.getRetainedOpaqueReference();

    // CHECK-NEXT: 0
    std::cout << deinitializations << std::endl;
  }

  // CHECK-NEXT: 0
  std::cout << deinitializations << std::endl;

  {
    auto reference =
        NativeReference<TestHelper>::getFromOpaqueReference(opaqueReference);

    AnyNativeReference::releaseOpaqueReference(opaqueReference);

    // CHECK-NEXT: 0
    std::cout << deinitializations << std::endl;
  }

  // CHECK-NEXT: 1
  std::cout << deinitializations << std::endl;
}

int main() {
  testMutability();
  testReferenceCounting();
  return 0;
}
