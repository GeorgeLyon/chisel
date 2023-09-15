#ifndef MLIR_BINDINGS_SUPPORT_SCOPED_CONTEXT_H_
#define MLIR_BINDINGS_SUPPORT_SCOPED_CONTEXT_H_

#include "mlir-bindings/Support/OpaquePointer.h"
#include "mlir-bindings/Support/ReferenceCountedPointer.h"

#include "llvm/Support/Casting.h"
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Location.h>
#include <type_traits>

namespace mlir {
class MLIRContext;

namespace bindings {

class OpaquePointer;

class ScopedContext : ReferenceCountedPointer {
public:
  struct Implementation;

private:
  Implementation *get() const;
  using ReferenceCountedPointer::ReferenceCountedPointer;

public: // Initialization
  static ScopedContext create();

private:
  ScopedContext();

public: // Conversion to/from OpaquePointer
  static ScopedContext getFromOpaquePointer(OpaquePointer);
  OpaquePointer toRetainedOpaquePointer();

public: // MLIR Context
  operator MLIRContext *() const;

public: // User Data
  /**
   Sets user data for this context. The support infrastructure does not use this
   pointer, but it can be used to store platform-specific information relevant
   to bindings. An optional deleter function may be provided which will be
   called if the user data is replaced, or when the context is destroyed.
   */
  void setUserData(void *userData, void (*userDataDeleter)(void *) = nullptr);
  void *getUserData() const;

public: // Wrapping
  OpaquePointer wrap(const Type &) const;
  OpaquePointer wrap(const Attribute &) const;
  OpaquePointer wrap(const Location &) const;

private: // Unwrapping Logic
  friend class ScopedBuilder;
  template <typename T, typename = void> struct Unwrapper {
    static T unwrap(const ScopedContext &context,
                    OpaquePointer opaquePointer) = delete;
  };
  Attribute unwrapAttribute(OpaquePointer opaquePointer) const;
  template <typename T>
  struct Unwrapper<T, std::enable_if_t<std::is_base_of_v<Attribute, T>>> {
    static T unwrap(const ScopedContext &context, OpaquePointer opaquePointer) {
      return llvm::cast<T>(context.unwrapAttribute(opaquePointer));
    }
  };
  template <> struct Unwrapper<Location> {
    static Location unwrap(const ScopedContext &context,
                           OpaquePointer opaquePointer) {
      return Location(
          llvm::cast<LocationAttr>(context.unwrapAttribute(opaquePointer)));
    }
  };
  Type unwrapType(OpaquePointer opaquePointer) const;
  template <typename T>
  struct Unwrapper<T, std::enable_if_t<std::is_base_of_v<Type, T>>> {
    static T unwrap(const ScopedContext &context, OpaquePointer opaquePointer) {
      return llvm::cast<T>(context.unwrapType(opaquePointer));
    }
  };

public: // Unwrapping
  template <typename T> T unwrap(OpaquePointer opaquePointer) const {
    return Unwrapper<T>::unwrap(*this, opaquePointer);
  }
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_SCOPED_CONTEXT_H_
