//===- CreateAsyncGroups.cpp - Create async device copies -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/NVGPU/Transforms/Transforms.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/NVGPU/Transforms/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;

/// Return "true" if the given vector transfer op is contiguous and suitable
/// for replacement with an async copy.
template <typename OpTy>
static bool isContiguousXferOp(OpTy op) {
  return op.getPermutationMap().isMinorIdentity() && op.isDimInBounds(0) &&
         op.hasBufferSemantics() &&
         isLastMemrefDimUnitStride(
             cast<MemRefType>(nvgpu::getMemrefOperand(op).getType()));
}

/// Return "true" if the given op is a contiguous and suitable
/// vector.transfer_write or vector.store op.
static bool isContiguousStore(Operation *write) {
  if (auto transferWrite = dyn_cast<vector::TransferWriteOp>(write))
    return isContiguousXferOp(transferWrite) && !transferWrite.getMask();
  // vector.store are always contiguous.
  return isa<vector::StoreOp>(write);
}

/// Return "true" if the given op is a contiguous and suitable
/// vector.transfer_read or vector.load op.
static bool isContiguousRead(Operation *read) {
  if (auto transferRead = dyn_cast<vector::TransferReadOp>(read))
    return isContiguousXferOp(transferRead);
  // vector.load are always contiguous.
  return isa<vector::LoadOp>(read);
}

/// If the given vector load op has a mask that is defined by
/// vector.create_mask, return that op.
static vector::CreateMaskOp getMaskOp(Operation *loadOp) {
  auto transferRead = dyn_cast<vector::TransferReadOp>(loadOp);
  if (!transferRead || !transferRead.getMask())
    return {};
  auto maskOp = transferRead.getMask().getDefiningOp<vector::CreateMaskOp>();
  // TODO: Support 2D masks and higher. Ops with a >1D mask are ignored at the
  // moment.
  if (maskOp.getVectorType().getRank() != 1)
    return {};
  return maskOp;
}

/// Return "true" if the conversion to async copy is supported by "async copy".
static bool resultsInSupportedAsyncCopy(MemRefType memrefType,
                                        Operation::operand_range indices,
                                        VectorType vecType) {
  assert(vecType.getRank() == 1 && "expected 1-D vector");
  constexpr int64_t kSupportedCpAsyncAlignmentsInBytes[3] = {4, 8, 16};

  // Condition 1: the copy size must be supported.
  bool supportedCopySize = false;
  int64_t numElements = vecType.getNumElements();
  Type elementType = vecType.getElementType();
  for (int64_t alignmentInBytes : kSupportedCpAsyncAlignmentsInBytes) {
    if (alignmentInBytes * 8 ==
        numElements * elementType.getIntOrFloatBitWidth()) {
      supportedCopySize = true;
      break;
    }
  }
  if (!supportedCopySize)
    return false;

  // TODO: Condition 2: the alignments must be supported. For cp.async the
  // NVIDIA doc (section 6.4.1) says: "The address must be naturally aligned to
  // a multiple of the access size. If an address is not properly aligned, the
  // resulting behavior is undefined.".
  return true;
}

void nvgpu::createAsyncGroups(RewriterBase &rewriter, Operation *op,
                              bool bypassL1) {
  llvm::SmallSetVector<Operation *, 16> copyToSharedMem;

  // Look for all the copy that can be converted to async copy ops.
  op->walk([&](Operation *writeOp) {
    // Look for contiguous 1D vector store into shared memory.
    if (!isContiguousStore(writeOp))
      return;
    Value vectorVal = nvgpu::getValueStored(writeOp);
    if (cast<VectorType>(vectorVal.getType()).getRank() != 1)
      return;
    Value storeBase = nvgpu::getMemrefOperand(writeOp);
    if (!nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(
            cast<MemRefType>(storeBase.getType())))
      return;

    // The stored vector must originate from a contiguous 1D vector load.
    Operation *readOp = vectorVal.getDefiningOp();
    if (readOp == nullptr || !isContiguousRead(readOp))
      return;
    Value loadBase = nvgpu::getMemrefOperand(readOp);
    // Should be reading from global memory (not shared memory).
    if (nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(
            cast<MemRefType>(loadBase.getType())))
      return;

    // Look for compatible mask and padding.
    if (auto transferRead = dyn_cast<vector::TransferReadOp>(readOp)) {
      if (Value mask = transferRead.getMask()) {
        if (getConstantIntValue(transferRead.getPadding()) ==
            static_cast<int64_t>(0))
          return;
        if (!getMaskOp(readOp))
          return;
      }
    }

    // Check whether both accesses are supported before we emit: this is
    // necessary to ensure the correctness of DeviceAsyncCopyOp.
    VectorType vecType = cast<VectorType>(vectorVal.getType());

    if (!resultsInSupportedAsyncCopy(cast<MemRefType>(loadBase.getType()),
                                     nvgpu::getIndices(readOp), vecType) ||
        !resultsInSupportedAsyncCopy(cast<MemRefType>(storeBase.getType()),
                                     nvgpu::getIndices(writeOp), vecType))
      return;

    copyToSharedMem.insert(writeOp);
    return;
  });

  while (!copyToSharedMem.empty()) {
    // Start a group with the first write.
    SmallVector<Operation *> group;
    Operation *writeOp = *copyToSharedMem.begin();
    copyToSharedMem.remove(writeOp);
    group.push_back(writeOp);
    Operation *nextNode = writeOp;

    // Look in the next nodes for more copies to add to the same group.
    while ((nextNode = nextNode->getNextNode())) {
      // Ignore ops without side effects.
      auto memInterface = dyn_cast<MemoryEffectOpInterface>(nextNode);
      if (memInterface && memInterface.hasNoEffect() &&
          !nextNode->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      // Ignore read from a different address space.
      if (isa<vector::TransferReadOp, vector::LoadOp>(nextNode)) {
        Operation *readOp = nextNode;
        Value memrefOperand = nvgpu::getMemrefOperand(readOp);
        if (!nvgpu::NVGPUDialect::hasSharedMemoryAddressSpace(
                cast<MemRefType>(memrefOperand.getType()))) {
          continue;
        }
      }
      if (copyToSharedMem.count(nextNode)) {
        // Found another copy, add it to the group.
        copyToSharedMem.remove(nextNode);
        group.push_back(nextNode);
        continue;
      }
      // If the op is something else stop the accumulating op in the group.
      break;
    }

    // Emit the group.
    SmallVector<Value> tokens;
    for (Operation *writeOp : group) {
      rewriter.setInsertionPoint(writeOp);
      Value vectorVal = nvgpu::getValueStored(writeOp);
      auto vectorType = cast<VectorType>(vectorVal.getType());
      int64_t numElements = vectorType.getNumElements();
      Operation *readOp = vectorVal.getDefiningOp();
      Value storeBase = nvgpu::getMemrefOperand(writeOp);
      Value loadBase = nvgpu::getMemrefOperand(readOp);
      Value numReadElements;
      if (vector::CreateMaskOp maskOp = getMaskOp(readOp)) {
        assert(maskOp.getNumOperands() == 1 && "expected single operand");
        numReadElements = maskOp.getOperand(0);
      }
      auto dstMemref = cast<MemRefType>(storeBase.getType());
      int64_t sizeInBytes =
          (dstMemref.getElementTypeBitWidth() * numElements) / 8;
      // bypass_l1 only possible with 16 byte transfer.
      Value token = rewriter.create<nvgpu::DeviceAsyncCopyOp>(
          writeOp->getLoc(), nvgpu::DeviceAsyncTokenType::get(op->getContext()),
          /*dst=*/storeBase, /*dstIndices=*/nvgpu::getIndices(writeOp),
          /*src=*/loadBase,
          /*srcIndices=*/nvgpu::getIndices(readOp),
          /*dstElements=*/rewriter.getIndexAttr(numElements),
          /*srcElements=*/numReadElements,
          /*bypassL1=*/bypassL1 && sizeInBytes == 16 ? rewriter.getUnitAttr()
                                                     : UnitAttr());
      tokens.push_back(token);
    }

    // Create the group and wait for it right after.
    Value groupToken = rewriter.create<nvgpu::DeviceAsyncCreateGroupOp>(
        op->getLoc(), nvgpu::DeviceAsyncTokenType::get(op->getContext()),
        tokens);
    rewriter.create<nvgpu::DeviceAsyncWaitOp>(op->getLoc(), groupToken,
                                              nullptr);
    // Clean up old stores.
    for (Operation *writeOp : group)
      rewriter.eraseOp(writeOp);
  }
}
