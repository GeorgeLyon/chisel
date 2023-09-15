#ifndef MLIR_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_H_
#define MLIR_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_H_

namespace mlir {
namespace bindings {

class ReferenceCountedPointer {
protected:
  class Implementation;

private: // Access to the implementation must go through `get()`
  mutable Implementation *impl;

protected:
  explicit ReferenceCountedPointer(Implementation *impl);
  Implementation *get() const;

public: // Reference counting
  ~ReferenceCountedPointer();
  ReferenceCountedPointer(const ReferenceCountedPointer &other);
  ReferenceCountedPointer operator=(const ReferenceCountedPointer &ptr);
};

} // namespace bindings
} // namespace mlir

#endif
