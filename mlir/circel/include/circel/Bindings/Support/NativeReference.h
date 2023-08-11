#ifndef MLIR_BINDINGS_SUPPORT_NATIVE_REFERENCE_H_
#define MLIR_BINDINGS_SUPPORT_NATIVE_REFERENCE_H_

#include <atomic>
#include <cassert>
#include <mlir/IR/Types.h>
#include <mlir/Support/TypeID.h>

namespace mlir {
namespace bindings {

class AnyNativeReference {
  class AnyStorage {
    friend class AnyNativeReference;

    std::atomic_uint64_t referenceCount = 1;
    mlir::TypeID typeID;
    using StoredDestructor = void (*)(AnyStorage *storage);
    StoredDestructor storedDestructor;

  protected:
    AnyStorage(mlir::TypeID typeID, StoredDestructor storedDestructor)
        : typeID(typeID), storedDestructor(storedDestructor) {}
  };
  AnyStorage *storage;

protected:
  template <typename T> class Storage : public AnyStorage {
    friend class AnyNativeReference;
    T value;

  public:
    template <typename... Args>
    Storage(Args &&...args)
        : AnyStorage(mlir::TypeID::get<T>(),
                     [](AnyStorage *anyStorage) {
                       auto storage = static_cast<Storage<T> *>(anyStorage);
                       delete storage;
                     }),
          value(std::forward<Args>(args)...) {}

    Storage(T value)
        : AnyStorage(mlir::TypeID::get<T>(),
                     [](AnyStorage *anyStorage) {
                       auto storage = static_cast<Storage<T> *>(anyStorage);
                       delete storage;
                     }),
          value(value) {}
  };

  explicit AnyNativeReference(AnyStorage *storage) : storage(storage) {}

  explicit AnyNativeReference(void *opaqueReference)
      : storage(static_cast<AnyStorage *>(opaqueReference)) {
    // Increment reference count to balance the decrement when this value is
    // destroyed.
    storage->referenceCount++;
  }

  template <typename T> void assertStorageTypeMatches() const {
    assert(mlir::TypeID::get<T>() == storage->typeID);
  }

  template <typename T> T *getPointerToValue() const {
    // We may eventually remove this check, as it we already check the type is
    // consistent when creating a NativeReference, but it is a helpful sanity
    // check while things are evolving rapidly.
    assertStorageTypeMatches<T>();
    return &static_cast<Storage<T> *>(storage)->value;
  }

public:
  AnyNativeReference(const AnyNativeReference &source)
      : storage(source.storage) {
    storage->referenceCount++;
  }
  ~AnyNativeReference() {
    if (--storage->referenceCount == 0)
      storage->storedDestructor(storage);
  }

  void *getRetainedOpaqueReference() {
    storage->referenceCount++;
    return reinterpret_cast<void *>(storage);
  }

  static void releaseOpaqueReference(void *opaqueReference) {
    auto reference = AnyNativeReference(opaqueReference);
    // Consume the unbalanced retain from `getRetainedOpaqueReference`, which
    // cannot be the last retain because creating `reference` incremented the
    // retain count, which will be decremented when `reference` is destroyed.
    assert(reference.storage->referenceCount-- > 0);
  }
};

namespace detail {
template <typename T> class TypedNativeReference : public AnyNativeReference {

protected:
  explicit TypedNativeReference(void *opaqueReference)
      : AnyNativeReference(opaqueReference) {
    assertStorageTypeMatches<T>();
  }

  explicit TypedNativeReference(Storage<T> *storage)
      : AnyNativeReference(storage) {}

  template <typename... Args>
  static TypedNativeReference<T> create(Args &&...args) {
    return TypedNativeReference<T>(
        new mlir::bindings::AnyNativeReference::Storage<T>(
            std::forward<Args>(args)...));
  }

  static TypedNativeReference<T> getFromOpaqueReference(void *opaqueReference) {
    auto reference = TypedNativeReference<T>(opaqueReference);
    reference.template assertStorageTypeMatches<T>();
    return reference;
  }

public:
  TypedNativeReference(const TypedNativeReference &source)
      : AnyNativeReference(source) {}

  T *operator->() const { return getPointerToValue<T>(); }
  T operator*() const { return *getPointerToValue<T>(); }
};
} // namespace detail

template <typename T>
class NativeReference : public detail::TypedNativeReference<T> {};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_NATIVE_REFERENCE_H_
