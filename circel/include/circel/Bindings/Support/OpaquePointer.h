#ifndef CIRCEL_BINDINGS_SUPPORT_OPAQUE_POINTER_H_
#define CIRCEL_BINDINGS_SUPPORT_OPAQUE_POINTER_H_

namespace circel {

class OpaquePointer {
  void *rawValue;

public:
  OpaquePointer() = delete;
  explicit OpaquePointer(void *rawValue);
  void *get() const;

  void releaseUnderlyingResource();
};

} // namespace circel

#endif // CIRCEL_BINDINGS_SUPPORT_OPAQUE_POINTER
