#ifndef MLIR_BINDINGS_SUPPORT_SCOPED_BUILDER_INTERNAL_H_
#define MLIR_BINDINGS_SUPPORT_SCOPED_BUILDER_INTERNAL_H_

#include "mlir-bindings/Support/ScopedBuilder.h"

#include "OpaquePointer.h"
#include "ReferenceCountedPointer.h"
#include "ScopedContext.h"

#include "mlir/IR/Builders.h"

namespace mlir {
namespace bindings {

struct ScopedBuilder::Implementation
    : public ReferenceCountedPointer::Implementation,
      public OpaquePointerRepresentable<
          ScopedBuilder::Implementation,
          AnyOpaquePointerRepresentable::Kind::ScopedBuilder> {
  friend class ScopedBuilder;

  /**
   Builder IDs are allocated starting from the maximum value, decreasing by 1
   every time. This allows us to distinguish context and builder IDs visually
   when debugging, and makes it less likely for a collision to occur (though a
   collision would only matter if we accidentally reinterpreted a builder ID as
   a context ID).
   */
  struct BuilderID
      : detail::ScopeID<uint32_t, std::numeric_limits<uint32_t>::max(), -1> {
    using ScopeID::ScopeID;
    using ScopeID::operator==;
  };
  BuilderID id;

  bool isValid = true;
  ScopedContext context;
  OpBuilder builder;
  Implementation(ScopedContext context) : context(context), builder(context) {}

  struct ScopedContainer {
    BuilderID builderID;
    ScopedContainer(const ScopedBuilder &builder)
        : builderID(builder.get()->id) {}
    bool isValidInScope(const ScopedBuilder &builder) const {
      auto impl = builder.get();
      return impl->isValid && impl->id == builderID;
    }
  };

  struct OpStateContainer
      : OpaquePointerRepresentable<
            OpStateContainer, AnyOpaquePointerRepresentable::Kind::OpState>,
        ScopedContainer {
    OpState opState;
    OpStateContainer(const ScopedBuilder &builder, OpState opState)
        : ScopedContainer(builder), opState(opState) {}
  };
  struct RegionContainer
      : OpaquePointerRepresentable<RegionContainer,
                                   AnyOpaquePointerRepresentable::Kind::Region>,
        ScopedContainer {
    Region *region;
    RegionContainer(const ScopedBuilder &builder, Region *region)
        : ScopedContainer(builder), region(region) {}
  };
  struct BlockContainer
      : OpaquePointerRepresentable<BlockContainer,
                                   AnyOpaquePointerRepresentable::Kind::Block>,
        ScopedContainer {
    Block *block;
    BlockContainer(const ScopedBuilder &builder, Block *block)
        : ScopedContainer(builder), block(block) {}
  };
  struct ValueContainer
      : OpaquePointerRepresentable<ValueContainer,
                                   AnyOpaquePointerRepresentable::Kind::Value>,
        ScopedContainer {
    Value value;
    ValueContainer(const ScopedBuilder &builder, Value value)
        : ScopedContainer(builder), value(value) {}
  };
};

} // namespace bindings
} // namespace mlir

#endif // MLIR_BINDINGS_SUPPORT_SCOPED_BUILDER_INTERNAL_H_
