#ifndef MLIR_BINDINGS_SUPPORT_SCOPED_BUILDER_H_
#define MLIR_BINDINGS_SUPPORT_SCOPED_BUILDER_H_

#include "mlir-bindings/Support/OpaquePointer.h"
#include "mlir-bindings/Support/ReferenceCountedPointer.h"
#include "mlir-bindings/Support/ScopedContext.h"

#include "llvm/Support/Casting.h"
#include <mlir/IR/OpDefinition.h>
#include <type_traits>

namespace mlir {
class OpBuilder;

namespace bindings {

class ScopedBuilder : ReferenceCountedPointer {
public:
  struct Implementation;

private:
  Implementation *get() const;

  using ReferenceCountedPointer::ReferenceCountedPointer;

public:
  ScopedBuilder(ScopedContext context);
  ScopedContext getContext() const;

public: // Conversion to/from OpaquePointer
  static ScopedBuilder getFromOpaquePointer(OpaquePointer);
  OpaquePointer toRetainedOpaquePointer();

public: // Builder
  OpBuilder *operator->();

public: // Wrapping
  OpaquePointer wrap(const Type &) const;
  OpaquePointer wrap(const Attribute &) const;
  OpaquePointer wrap(const Location &) const;
  OpaquePointer wrap(const OpState &) const;
  OpaquePointer wrap(Block *) const;
  OpaquePointer wrap(Region *) const;
  OpaquePointer wrap(const Value &) const;

private: // Unwrapping logic
  template <typename T, typename = void> struct Unwrapper {
    static T unwrap(const ScopedBuilder &builder, OpaquePointer opaquePointer) {
      // Fall back on context for other types
      return builder.getContext().unwrap<T>(opaquePointer);
    };
  };
  Value unwrapValue(OpaquePointer opaquePointer) const;
  template <typename T>
  struct Unwrapper<T, std::enable_if_t<std::is_base_of_v<Value, T>>> {
    static T unwrap(const ScopedBuilder &builder, OpaquePointer opaquePointer) {
      return llvm::cast<T>(builder.unwrapValue(opaquePointer));
    }
  };
  OpState unwrapOpState(OpaquePointer opaquePointer) const;
  template <typename T>
  struct Unwrapper<T, std::enable_if_t<std::is_base_of_v<OpState, T>>> {
    static T unwrap(const ScopedBuilder &builder, OpaquePointer opaquePointer) {
      return llvm::cast<T>(builder.unwrapOpState(opaquePointer));
    }
  };
  template <> struct Unwrapper<Operation *> {
    static Operation *unwrap(const ScopedBuilder &builder,
                             OpaquePointer opaquePointer) {
      return builder.unwrapOpState(opaquePointer).getOperation();
    }
  };
  Block *unwrapBlock(OpaquePointer opaquePointer) const;
  template <> struct Unwrapper<Block *> {
    static Block *unwrap(const ScopedBuilder &builder,
                         OpaquePointer opaquePointer) {
      return builder.unwrapBlock(opaquePointer);
    }
  };
  Region *unwrapRegion(OpaquePointer opaquePointer) const;
  template <> struct Unwrapper<Region *> {
    static Region *unwrap(const ScopedBuilder &builder,
                          OpaquePointer opaquePointer) {
      return builder.unwrapRegion(opaquePointer);
    }
  };

public:
  template <typename T> T unwrap(OpaquePointer opaquePointer) const {
    return Unwrapper<T>::unwrap(*this, opaquePointer);
  }
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_SCOPED_BUILDER_H_
