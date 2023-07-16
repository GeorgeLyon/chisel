#include "circel/Bindings/Support/Support.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>

namespace mlir {
namespace bindings {

// -- Context

std::atomic<uint64_t> nextContextID{0};

Context::Context() {
#ifdef MLIR_BINDINGS_AT_MOST_ONE_CONTEXT
  assert(nextContextID++ == 0);
#else
  id = nextContextID++;
#endif

  value = std::make_shared<mlir::MLIRContext>();
}

} // namespace bindings
} // namespace mlir
