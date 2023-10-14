#ifndef CIRCEL_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_H_
#define CIRCEL_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_H_

namespace circel {

class ReferenceCountedPointer {
public:
  // -- Lifecycle
  ~ReferenceCountedPointer();
  ReferenceCountedPointer(const ReferenceCountedPointer &other);
  ReferenceCountedPointer operator=(const ReferenceCountedPointer &other);

protected:
  class Implementation;
  explicit ReferenceCountedPointer(Implementation *impl);
  Implementation *get() const;

private:
  // Access to the implementation must go through `get()`
  mutable Implementation *impl;
};

} // namespace circel

#endif // CIRCEL_BINDINGS_SUPPORT_REFERENCE_COUNTED_POINTER_H_
