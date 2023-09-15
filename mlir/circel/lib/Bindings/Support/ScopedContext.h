#ifndef MLIR_BINDINGS_SUPPORT_SCOPED_CONTEXT_INTERNAL_H_
#define MLIR_BINDINGS_SUPPORT_SCOPED_CONTEXT_INTERNAL_H_

#include "mlir-bindings/Support/ScopedContext.h"

#include "OpaquePointer.h"
#include "ReferenceCountedPointer.h"
#include "ScopeID.h"

#include "mlir/IR/MLIRContext.h"

namespace mlir {
namespace bindings {

struct ScopedContext::Implementation
    : public ReferenceCountedPointer::Implementation,
      public OpaquePointerRepresentable<
          ScopedContext::Implementation,
          AnyOpaquePointerRepresentable::Kind::Context> {
  friend class Context;

  /**
   Used fot other types to get access to a Context's implementation
   */
  static Implementation *get(ScopedContext context) { return context.get(); }

  struct ContextID
      : detail::ScopeID<uint32_t, std::numeric_limits<uint32_t>::min(), 1> {
    using ScopeID::ScopeID;
    using ScopeID::operator==;
  };
  ContextID id;

  mutable void *userData = nullptr;
  void (*userDataDeleter)(void *) = nullptr;

  ~Implementation() override {
    if (userDataDeleter)
      userDataDeleter(userData);
  }

  mutable MLIRContext mlirContext;

  struct ScopedContainer {
#ifdef MLIR_BINDINGS_SUPPORT_SINGULAR_CONTEXT
    ContextScopedContainer() {}
    bool isValidInContext(const ScopedContext &context) const { return true; }
#else
    ContextID contextID;
    ScopedContainer(const ScopedContext &context)
        : contextID(ScopedContext::Implementation::get(context)->id) {}
    bool isValidInContext(const ScopedContext &context) const {
      return ScopedContext::Implementation::get(context)->id == contextID;
    }
#endif
  };

  struct AttributeContainer
      : OpaquePointerRepresentable<
            AttributeContainer, AnyOpaquePointerRepresentable::Kind::Attribute>,
        ScopedContainer {
    Attribute attribute;
    AttributeContainer(ScopedContext context, Attribute attribute)
        : ScopedContainer(context), attribute(attribute) {}
  };
  struct TypeContainer
      : OpaquePointerRepresentable<TypeContainer,
                                   AnyOpaquePointerRepresentable::Kind::Type>,
        ScopedContainer {
    Type type;
    TypeContainer(ScopedContext context, Type type)
        : ScopedContainer(context), type(type) {}
  };
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_SCOPED_CONTEXT_INTERNAL_H_
