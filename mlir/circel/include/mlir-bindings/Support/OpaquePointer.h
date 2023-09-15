#ifndef MLIR_BINDINGS_SUPPORT_OPAQUE_POINTER_H_
#define MLIR_BINDINGS_SUPPORT_OPAQUE_POINTER_H_

namespace mlir {
namespace bindings {

class OpaquePointer {
  void *rawValue;

public:
  OpaquePointer() = delete;
  explicit OpaquePointer(void *rawValue);
  void *get() const;

  void releaseUnderlyingResource();
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_OPAQUE_POINTER
