//===- MapMemRefStorageCLassPass.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to map numeric MemRef memory spaces to
// symbolic ones defined in the SPIR-V specification.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRVPass.h"

#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

namespace mlir {
#define GEN_PASS_DEF_MAPMEMREFSTORAGECLASS
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

#define DEBUG_TYPE "mlir-map-memref-storage-class"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Mappings
//===----------------------------------------------------------------------===//

/// Mapping between SPIR-V storage classes to memref memory spaces.
///
/// Note: memref does not have a defined semantics for each memory space; it
/// depends on the context where it is used. There are no particular reasons
/// behind the number assignments; we try to follow NVVM conventions and largely
/// give common storage classes a smaller number.
#define VULKAN_STORAGE_SPACE_MAP_LIST(MAP_FN)                                  \
  MAP_FN(spirv::StorageClass::StorageBuffer, 0)                                \
  MAP_FN(spirv::StorageClass::Generic, 1)                                      \
  MAP_FN(spirv::StorageClass::Workgroup, 3)                                    \
  MAP_FN(spirv::StorageClass::Uniform, 4)                                      \
  MAP_FN(spirv::StorageClass::Private, 5)                                      \
  MAP_FN(spirv::StorageClass::Function, 6)                                     \
  MAP_FN(spirv::StorageClass::PushConstant, 7)                                 \
  MAP_FN(spirv::StorageClass::UniformConstant, 8)                              \
  MAP_FN(spirv::StorageClass::Input, 9)                                        \
  MAP_FN(spirv::StorageClass::Output, 10)

std::optional<spirv::StorageClass>
spirv::mapMemorySpaceToVulkanStorageClass(Attribute memorySpaceAttr) {
  // Handle null memory space attribute specially.
  if (!memorySpaceAttr)
    return spirv::StorageClass::StorageBuffer;

  // Unknown dialect custom attributes are not supported by default.
  // Downstream callers should plug in more specialized ones.
  auto intAttr = dyn_cast<IntegerAttr>(memorySpaceAttr);
  if (!intAttr)
    return std::nullopt;
  unsigned memorySpace = intAttr.getInt();

#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case space:                                                                  \
    return storage;

  switch (memorySpace) {
    VULKAN_STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
  default:
    break;
  }
  return std::nullopt;

#undef STORAGE_SPACE_MAP_FN
}

std::optional<unsigned>
spirv::mapVulkanStorageClassToMemorySpace(spirv::StorageClass storageClass) {
#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case storage:                                                                \
    return space;

  switch (storageClass) {
    VULKAN_STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
  default:
    break;
  }
  return std::nullopt;

#undef STORAGE_SPACE_MAP_FN
}

#undef VULKAN_STORAGE_SPACE_MAP_LIST

#define OPENCL_STORAGE_SPACE_MAP_LIST(MAP_FN)                                  \
  MAP_FN(spirv::StorageClass::CrossWorkgroup, 0)                               \
  MAP_FN(spirv::StorageClass::Generic, 1)                                      \
  MAP_FN(spirv::StorageClass::Workgroup, 3)                                    \
  MAP_FN(spirv::StorageClass::UniformConstant, 4)                              \
  MAP_FN(spirv::StorageClass::Private, 5)                                      \
  MAP_FN(spirv::StorageClass::Function, 6)                                     \
  MAP_FN(spirv::StorageClass::Image, 7)

std::optional<spirv::StorageClass>
spirv::mapMemorySpaceToOpenCLStorageClass(Attribute memorySpaceAttr) {
  // Handle null memory space attribute specially.
  if (!memorySpaceAttr)
    return spirv::StorageClass::CrossWorkgroup;

  // Unknown dialect custom attributes are not supported by default.
  // Downstream callers should plug in more specialized ones.
  auto intAttr = dyn_cast<IntegerAttr>(memorySpaceAttr);
  if (!intAttr)
    return std::nullopt;
  unsigned memorySpace = intAttr.getInt();

#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case space:                                                                  \
    return storage;

  switch (memorySpace) {
    OPENCL_STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
  default:
    break;
  }
  return std::nullopt;

#undef STORAGE_SPACE_MAP_FN
}

std::optional<unsigned>
spirv::mapOpenCLStorageClassToMemorySpace(spirv::StorageClass storageClass) {
#define STORAGE_SPACE_MAP_FN(storage, space)                                   \
  case storage:                                                                \
    return space;

  switch (storageClass) {
    OPENCL_STORAGE_SPACE_MAP_LIST(STORAGE_SPACE_MAP_FN)
  default:
    break;
  }
  return std::nullopt;

#undef STORAGE_SPACE_MAP_FN
}

#undef OPENCL_STORAGE_SPACE_MAP_LIST

//===----------------------------------------------------------------------===//
// Type Converter
//===----------------------------------------------------------------------===//

spirv::MemorySpaceToStorageClassConverter::MemorySpaceToStorageClassConverter(
    const spirv::MemorySpaceToStorageClassMap &memorySpaceMap)
    : memorySpaceMap(memorySpaceMap) {
  // Pass through for all other types.
  addConversion([](Type type) { return type; });

  addConversion([this](BaseMemRefType memRefType) -> std::optional<Type> {
    std::optional<spirv::StorageClass> storage =
        this->memorySpaceMap(memRefType.getMemorySpace());
    if (!storage) {
      LLVM_DEBUG(llvm::dbgs()
                 << "cannot convert " << memRefType
                 << " due to being unable to find memory space in map\n");
      return std::nullopt;
    }

    auto storageAttr =
        spirv::StorageClassAttr::get(memRefType.getContext(), *storage);
    if (auto rankedType = dyn_cast<MemRefType>(memRefType)) {
      return MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                             rankedType.getLayout(), storageAttr);
    }
    return UnrankedMemRefType::get(memRefType.getElementType(), storageAttr);
  });

  addConversion([this](FunctionType type) {
    SmallVector<Type> inputs, results;
    inputs.reserve(type.getNumInputs());
    results.reserve(type.getNumResults());
    for (Type input : type.getInputs())
      inputs.push_back(convertType(input));
    for (Type result : type.getResults())
      results.push_back(convertType(result));
    return FunctionType::get(type.getContext(), inputs, results);
  });
}

//===----------------------------------------------------------------------===//
// Conversion Target
//===----------------------------------------------------------------------===//

/// Returns true if the given `type` is considered as legal for SPIR-V
/// conversion.
static bool isLegalType(Type type) {
  if (auto memRefType = dyn_cast<BaseMemRefType>(type)) {
    Attribute spaceAttr = memRefType.getMemorySpace();
    return spaceAttr && isa<spirv::StorageClassAttr>(spaceAttr);
  }
  return true;
}

/// Returns true if the given `attr` is considered as legal for SPIR-V
/// conversion.
static bool isLegalAttr(Attribute attr) {
  if (auto typeAttr = dyn_cast<TypeAttr>(attr))
    return isLegalType(typeAttr.getValue());
  return true;
}

/// Returns true if the given `op` is considered as legal for SPIR-V conversion.
static bool isLegalOp(Operation *op) {
  if (auto funcOp = dyn_cast<FunctionOpInterface>(op)) {
    return llvm::all_of(funcOp.getArgumentTypes(), isLegalType) &&
           llvm::all_of(funcOp.getResultTypes(), isLegalType) &&
           llvm::all_of(funcOp.getFunctionBody().getArgumentTypes(),
                        isLegalType);
  }

  auto attrs = llvm::map_range(op->getAttrs(), [](const NamedAttribute &attr) {
    return attr.getValue();
  });

  return llvm::all_of(op->getOperandTypes(), isLegalType) &&
         llvm::all_of(op->getResultTypes(), isLegalType) &&
         llvm::all_of(attrs, isLegalAttr);
}

std::unique_ptr<ConversionTarget>
spirv::getMemorySpaceToStorageClassTarget(MLIRContext &context) {
  auto target = std::make_unique<ConversionTarget>(context);
  target->markUnknownOpDynamicallyLegal(isLegalOp);
  return target;
}

//===----------------------------------------------------------------------===//
// Conversion Pattern
//===----------------------------------------------------------------------===//

namespace {
/// Converts any op that has operands/results/attributes with numeric MemRef
/// memory spaces.
struct MapMemRefStoragePattern final : public ConversionPattern {
  MapMemRefStoragePattern(MLIRContext *context, TypeConverter &converter)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override;
};
} // namespace

LogicalResult MapMemRefStoragePattern::matchAndRewrite(
    Operation *op, ArrayRef<Value> operands,
    ConversionPatternRewriter &rewriter) const {
  llvm::SmallVector<NamedAttribute, 4> newAttrs;
  newAttrs.reserve(op->getAttrs().size());
  for (auto attr : op->getAttrs()) {
    if (auto typeAttr = dyn_cast<TypeAttr>(attr.getValue())) {
      auto newAttr = getTypeConverter()->convertType(typeAttr.getValue());
      newAttrs.emplace_back(attr.getName(), TypeAttr::get(newAttr));
    } else {
      newAttrs.push_back(attr);
    }
  }

  llvm::SmallVector<Type, 4> newResults;
  (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

  OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                       newResults, newAttrs, op->getSuccessors());

  for (Region &region : op->getRegions()) {
    Region *newRegion = state.addRegion();
    rewriter.inlineRegionBefore(region, *newRegion, newRegion->begin());
    TypeConverter::SignatureConversion result(newRegion->getNumArguments());
    (void)getTypeConverter()->convertSignatureArgs(
        newRegion->getArgumentTypes(), result);
    rewriter.applySignatureConversion(newRegion, result);
  }

  Operation *newOp = rewriter.create(state);
  rewriter.replaceOp(op, newOp->getResults());
  return success();
}

void spirv::populateMemorySpaceToStorageClassPatterns(
    spirv::MemorySpaceToStorageClassConverter &typeConverter,
    RewritePatternSet &patterns) {
  patterns.add<MapMemRefStoragePattern>(patterns.getContext(), typeConverter);
}

//===----------------------------------------------------------------------===//
// Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
class MapMemRefStorageClassPass final
    : public impl::MapMemRefStorageClassBase<MapMemRefStorageClassPass> {
public:
  explicit MapMemRefStorageClassPass() {
    memorySpaceMap = spirv::mapMemorySpaceToVulkanStorageClass;
  }
  explicit MapMemRefStorageClassPass(
      const spirv::MemorySpaceToStorageClassMap &memorySpaceMap)
      : memorySpaceMap(memorySpaceMap) {}

  LogicalResult initializeOptions(StringRef options) override;

  void runOnOperation() override;

private:
  spirv::MemorySpaceToStorageClassMap memorySpaceMap;
};
} // namespace

LogicalResult MapMemRefStorageClassPass::initializeOptions(StringRef options) {
  if (failed(Pass::initializeOptions(options)))
    return failure();

  if (clientAPI == "opencl") {
    memorySpaceMap = spirv::mapMemorySpaceToOpenCLStorageClass;
  }

  if (clientAPI != "vulkan" && clientAPI != "opencl")
    return failure();

  return success();
}

void MapMemRefStorageClassPass::runOnOperation() {
  MLIRContext *context = &getContext();
  Operation *op = getOperation();

  if (spirv::TargetEnvAttr attr = spirv::lookupTargetEnv(op)) {
    spirv::TargetEnv targetEnv(attr);
    if (targetEnv.allows(spirv::Capability::Kernel)) {
      memorySpaceMap = spirv::mapMemorySpaceToOpenCLStorageClass;
    } else if (targetEnv.allows(spirv::Capability::Shader)) {
      memorySpaceMap = spirv::mapMemorySpaceToVulkanStorageClass;
    }
  }

  auto target = spirv::getMemorySpaceToStorageClassTarget(*context);
  spirv::MemorySpaceToStorageClassConverter converter(memorySpaceMap);

  RewritePatternSet patterns(context);
  spirv::populateMemorySpaceToStorageClassPatterns(converter, patterns);

  if (failed(applyFullConversion(op, *target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<>> mlir::createMapMemRefStorageClassPass() {
  return std::make_unique<MapMemRefStorageClassPass>();
}
