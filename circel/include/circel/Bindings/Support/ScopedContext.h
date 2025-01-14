#ifndef CIRCEL_BINDINGS_SUPPORT_SCOPED_CONTEXT_H_
#define CIRCEL_BINDINGS_SUPPORT_SCOPED_CONTEXT_H_

#include "circel/Bindings/Support/OpaquePointer.h"
#include "circel/Bindings/Support/ReferenceCountedPointer.h"

#include <llvm/Support/Casting.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Location.h>
#include <type_traits>

namespace mlir {
class MLIRContext;
}

namespace circel {

class OpaquePointer;

class ScopedContext : ReferenceCountedPointer {
public:
  // -- Lifecycle
  static ScopedContext create();

  // -- Conversion to/from OpaquePointer
  static ScopedContext getFromOpaquePointer(OpaquePointer);
  OpaquePointer toRetainedOpaquePointer();

  // -- Accessors
  operator mlir::MLIRContext *() const;

  // -- User Data
  /**
   Sets user data for this context. The support infrastructure does not use this
   pointer, but it can be used to store platform-specific information relevant
   to bindings. An optional deleter function may be provided which will be
   called if the user data is replaced, or when the context is destroyed.
   */
  void setUserData(void *userData, void (*userDataDeleter)(void *) = nullptr);
  void *getUserData() const;

  // -- Wrapping
  OpaquePointer wrap(const mlir::Type &) const;
  OpaquePointer wrap(const mlir::Attribute &) const;
  OpaquePointer wrap(const mlir::Location &) const;

  // -- Unwrapping
  template <typename T> T unwrap(OpaquePointer opaquePointer) const {
    return Unwrapper<T>::unwrap(*this, opaquePointer);
  }

  // -- Reference Counted Pointer Internals
  struct Implementation;

private:
  // -- Reference Counted Pointer Internals
  Implementation *get() const;
  using ReferenceCountedPointer::ReferenceCountedPointer;

  // -- Lifecycle
  ScopedContext();

  // -- Unwrapping Logic
  friend class ScopedBuilder;
  template <typename T, typename = void> struct Unwrapper {
    static T unwrap(const ScopedContext &context,
                    OpaquePointer opaquePointer) = delete;
  };
  mlir::Attribute unwrapAttribute(OpaquePointer opaquePointer) const;
  template <typename T>
  struct Unwrapper<T, std::enable_if_t<std::is_base_of_v<mlir::Attribute, T>>> {
    static T unwrap(const ScopedContext &context, OpaquePointer opaquePointer) {
      return llvm::cast<T>(context.unwrapAttribute(opaquePointer));
    }
  };
  template <> struct Unwrapper<mlir::Location> {
    static mlir::Location unwrap(const ScopedContext &context,
                                 OpaquePointer opaquePointer) {
      return mlir::Location(llvm::cast<mlir::LocationAttr>(
          context.unwrapAttribute(opaquePointer)));
    }
  };
  mlir::Type unwrapType(OpaquePointer opaquePointer) const;
  template <typename T>
  struct Unwrapper<T, std::enable_if_t<std::is_base_of_v<mlir::Type, T>>> {
    static T unwrap(const ScopedContext &context, OpaquePointer opaquePointer) {
      return llvm::cast<T>(context.unwrapType(opaquePointer));
    }
  };
};

} // namespace circel

#endif // CIRCEL_BINDINGS_SUPPORT_SCOPED_CONTEXT_H_
