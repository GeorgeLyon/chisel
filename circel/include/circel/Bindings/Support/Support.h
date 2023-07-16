#include <cassert>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>
#include <vector>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace bindings {

// -- Context
struct Context final {
#ifdef MLIR_BINDINGS_AT_MOST_ONE_CONTEXT
#else
  uint64_t id;
#endif
  std::shared_ptr<mlir::MLIRContext> value;

  Context();
  operator mlir::MLIRContext *() { return value.get(); }
};

// -- Contextual Values

template <typename T> class Contextual {
#ifdef MLIR_BINDINGS_AT_MOST_ONE_CONTEXT
#else
  uint64_t contextID;
#endif
  T value;

  Contextual(Context context, T value) {
#ifdef MLIR_BINDINGS_AT_MOST_ONE_CONTEXT
#else
    contextID = context.id;
#endif
    this->value = value;
  }

public:
  template <typename... Args>
  static Contextual get(Context context, Args... args) {
    return Contextual(context, T::get(context.value.get(), args...));
  }

  T unwrap(Context in) const {
#ifdef MLIR_BINDINGS_AT_MOST_ONE_CONTEXT
#else
    assert(in.id == contextID);
#endif
    return value;
  }
};

// -- IR Builder

class IRBuilder {

  // -- Context

  Context context;

  // A helper function for unwrapping contextuals
  template <typename T> T unwrap(Contextual<T> contextual) {
    return contextual.unwrap(context);
  }

  // -- Locations

  std::vector<mlir::Location> locationStack;

  mlir::Location currentLocation() {
    assert(!locationStack.empty());
    return locationStack.back();
  }

public:
  IRBuilder(Context context = Context()) : context(context){};

  // -- Location Stack Management

  void pushUnknownLocation() {
    locationStack.push_back(mlir::UnknownLoc::get(context));
  }
  void pushLocation(mlir::StringRef filename, unsigned int line,
                    unsigned int column) {
    locationStack.push_back(
        mlir::FileLineColLoc::get(context, filename, line, column));
  }
  void pushLocation(Contextual<mlir::StringRef> filename, unsigned int line,
                    unsigned int column) {
    locationStack.push_back(
        mlir::FileLineColLoc::get(context, unwrap(filename), line, column));
  }
  void popLocation() { locationStack.pop_back(); }

  void test() {
    auto attribute =
        Contextual<mlir::StringAttr>::get(context, "Hello, IRBuilder!");
    attribute.unwrap(context).dump();
  }
};

} // namespace bindings
} // namespace mlir
