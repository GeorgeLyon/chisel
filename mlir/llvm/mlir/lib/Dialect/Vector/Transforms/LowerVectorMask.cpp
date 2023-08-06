//===- LowerVectorMask.cpp - Lower 'vector.mask' operation ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.mask' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "lower-vector-mask"

namespace mlir {
namespace vector {
#define GEN_PASS_DEF_LOWERVECTORMASKPASS
#include "mlir/Dialect/Vector/Transforms/Passes.h.inc"
} // namespace vector
} // namespace mlir

using namespace mlir;
using namespace mlir::vector;

//===----------------------------------------------------------------------===//
// populateVectorMaskOpLoweringPatterns
//===----------------------------------------------------------------------===//

namespace {
/// Progressive lowering of CreateMaskOp.
/// One:
///   %x = vector.create_mask %a, ... : vector<dx...>
/// is replaced by:
///   %l = vector.create_mask ... : vector<...>  ; one lower rank
///   %0 = arith.cmpi "slt", %ci, %a       |
///   %1 = select %0, %l, %zeroes    |
///   %r = vector.insert %1, %pr [i] | d-times
///   %x = ....
/// until a one-dimensional vector is reached.
class CreateMaskOpLowering : public OpRewritePattern<vector::CreateMaskOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::CreateMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto dstType = cast<VectorType>(op.getResult().getType());
    int64_t rank = dstType.getRank();
    if (rank <= 1)
      return rewriter.notifyMatchFailure(
          op, "0-D and 1-D vectors are handled separately");

    auto loc = op.getLoc();
    auto eltType = dstType.getElementType();
    int64_t dim = dstType.getDimSize(0);
    Value idx = op.getOperand(0);

    VectorType lowType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    Value trueVal = rewriter.create<vector::CreateMaskOp>(
        loc, lowType, op.getOperands().drop_front());
    Value falseVal = rewriter.create<arith::ConstantOp>(
        loc, lowType, rewriter.getZeroAttr(lowType));
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstType, rewriter.getZeroAttr(dstType));
    for (int64_t d = 0; d < dim; d++) {
      Value bnd =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(d));
      Value val = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 bnd, idx);
      Value sel = rewriter.create<arith::SelectOp>(loc, val, trueVal, falseVal);
      result = rewriter.create<vector::InsertOp>(loc, dstType, sel, result, d);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

/// Progressive lowering of ConstantMaskOp.
/// One:
///   %x = vector.constant_mask [a,b]
/// is replaced by:
///   %z = zero-result
///   %l = vector.constant_mask [b]
///   %4 = vector.insert %l, %z[0]
///   ..
///   %x = vector.insert %l, %..[a-1]
/// until a one-dimensional vector is reached. All these operations
/// will be folded at LLVM IR level.
class ConstantMaskOpLowering : public OpRewritePattern<vector::ConstantMaskOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ConstantMaskOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto dstType = op.getType();
    auto eltType = dstType.getElementType();
    auto dimSizes = op.getMaskDimSizes();
    int64_t rank = dstType.getRank();

    if (rank == 0) {
      assert(dimSizes.size() == 1 &&
             "Expected exactly one dim size for a 0-D vector");
      bool value = cast<IntegerAttr>(dimSizes[0]).getInt() == 1;
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, dstType,
          DenseIntElementsAttr::get(
              VectorType::get(ArrayRef<int64_t>{}, rewriter.getI1Type()),
              ArrayRef<bool>{value}));
      return success();
    }

    // Scalable constant masks can only be lowered for the "none set" case.
    if (cast<VectorType>(dstType).isScalable()) {
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, DenseElementsAttr::get(dstType, false));
      return success();
    }

    int64_t trueDim = std::min(dstType.getDimSize(0),
                               cast<IntegerAttr>(dimSizes[0]).getInt());

    if (rank == 1) {
      // Express constant 1-D case in explicit vector form:
      //   [T,..,T,F,..,F].
      SmallVector<bool> values(dstType.getDimSize(0));
      for (int64_t d = 0; d < trueDim; d++)
        values[d] = true;
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(
          op, dstType, rewriter.getBoolVectorAttr(values));
      return success();
    }

    VectorType lowType =
        VectorType::get(dstType.getShape().drop_front(), eltType);
    SmallVector<int64_t> newDimSizes;
    for (int64_t r = 1; r < rank; r++)
      newDimSizes.push_back(cast<IntegerAttr>(dimSizes[r]).getInt());
    Value trueVal = rewriter.create<vector::ConstantMaskOp>(
        loc, lowType, rewriter.getI64ArrayAttr(newDimSizes));
    Value result = rewriter.create<arith::ConstantOp>(
        loc, dstType, rewriter.getZeroAttr(dstType));
    for (int64_t d = 0; d < trueDim; d++)
      result =
          rewriter.create<vector::InsertOp>(loc, dstType, trueVal, result, d);
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // namespace

void mlir::vector::populateVectorMaskOpLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<CreateMaskOpLowering, ConstantMaskOpLowering>(
      patterns.getContext(), benefit);
}

//===----------------------------------------------------------------------===//
// populateVectorMaskLoweringPatternsForSideEffectingOps
//===----------------------------------------------------------------------===//

namespace {

/// The `MaskOpRewritePattern` implements a pattern that follows a two-fold
/// matching:
///   1. It matches a `vector.mask` operation.
///   2. It invokes `matchAndRewriteMaskableOp` on `MaskableOpInterface` nested
///      in the matched `vector.mask` operation.
///
/// It is required that the replacement op in the pattern replaces the
/// `vector.mask` operation and not the nested `MaskableOpInterface`. This
/// approach allows having patterns that "stop" at every `vector.mask` operation
/// and actually match the traits of its the nested `MaskableOpInterface`.
template <class SourceOp>
struct MaskOpRewritePattern : OpRewritePattern<MaskOp> {
  using OpRewritePattern<MaskOp>::OpRewritePattern;

private:
  LogicalResult matchAndRewrite(MaskOp maskOp,
                                PatternRewriter &rewriter) const final {
    auto maskableOp = cast<MaskableOpInterface>(maskOp.getMaskableOp());
    SourceOp sourceOp = dyn_cast<SourceOp>(maskableOp.getOperation());
    if (!sourceOp)
      return failure();

    return matchAndRewriteMaskableOp(sourceOp, maskOp, rewriter);
  }

protected:
  virtual LogicalResult
  matchAndRewriteMaskableOp(SourceOp sourceOp, MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const = 0;
};

/// Lowers a masked `vector.transfer_read` operation.
struct MaskedTransferReadOpPattern
    : public MaskOpRewritePattern<TransferReadOp> {
public:
  using MaskOpRewritePattern<TransferReadOp>::MaskOpRewritePattern;

  LogicalResult
  matchAndRewriteMaskableOp(TransferReadOp readOp, MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    // TODO: The 'vector.mask' passthru is a vector and 'vector.transfer_read'
    // expects a scalar. We could only lower one to the other for cases where
    // the passthru is a broadcast of a scalar.
    if (maskingOp.hasPassthru())
      return rewriter.notifyMatchFailure(
          maskingOp, "Can't lower passthru to vector.transfer_read");

    // Replace the `vector.mask` operation.
    rewriter.replaceOpWithNewOp<TransferReadOp>(
        maskingOp.getOperation(), readOp.getVectorType(), readOp.getSource(),
        readOp.getIndices(), readOp.getPermutationMap(), readOp.getPadding(),
        maskingOp.getMask(), readOp.getInBounds().value_or(ArrayAttr()));
    return success();
  }
};

/// Lowers a masked `vector.transfer_write` operation.
struct MaskedTransferWriteOpPattern
    : public MaskOpRewritePattern<TransferWriteOp> {
public:
  using MaskOpRewritePattern<TransferWriteOp>::MaskOpRewritePattern;

  LogicalResult
  matchAndRewriteMaskableOp(TransferWriteOp writeOp,
                            MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    Type resultType =
        writeOp.getResult() ? writeOp.getResult().getType() : Type();

    // Replace the `vector.mask` operation.
    rewriter.replaceOpWithNewOp<TransferWriteOp>(
        maskingOp.getOperation(), resultType, writeOp.getVector(),
        writeOp.getSource(), writeOp.getIndices(), writeOp.getPermutationMap(),
        maskingOp.getMask(), writeOp.getInBounds().value_or(ArrayAttr()));
    return success();
  }
};

/// Lowers a masked `vector.gather` operation.
struct MaskedGatherOpPattern : public MaskOpRewritePattern<GatherOp> {
public:
  using MaskOpRewritePattern<GatherOp>::MaskOpRewritePattern;

  LogicalResult
  matchAndRewriteMaskableOp(GatherOp gatherOp, MaskingOpInterface maskingOp,
                            PatternRewriter &rewriter) const override {
    Value passthru = maskingOp.hasPassthru()
                         ? maskingOp.getPassthru()
                         : rewriter.create<arith::ConstantOp>(
                               gatherOp.getLoc(),
                               rewriter.getZeroAttr(gatherOp.getVectorType()));

    // Replace the `vector.mask` operation.
    rewriter.replaceOpWithNewOp<GatherOp>(
        maskingOp.getOperation(), gatherOp.getVectorType(), gatherOp.getBase(),
        gatherOp.getIndices(), gatherOp.getIndexVec(), maskingOp.getMask(),
        passthru);
    return success();
  }
};

struct LowerVectorMaskPass
    : public vector::impl::LowerVectorMaskPassBase<LowerVectorMaskPass> {
  using Base::Base;

  void runOnOperation() override {
    Operation *op = getOperation();
    MLIRContext *context = op->getContext();

    RewritePatternSet loweringPatterns(context);
    populateVectorMaskLoweringPatternsForSideEffectingOps(loweringPatterns);

    if (failed(applyPatternsAndFoldGreedily(op, std::move(loweringPatterns))))
      signalPassFailure();
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
  }
};

} // namespace

/// Populates instances of `MaskOpRewritePattern` to lower masked operations
/// with `vector.mask`. Patterns should rewrite the `vector.mask` operation and
/// not its nested `MaskableOpInterface`.
void vector::populateVectorMaskLoweringPatternsForSideEffectingOps(
    RewritePatternSet &patterns) {
  patterns.add<MaskedTransferReadOpPattern, MaskedTransferWriteOpPattern,
               MaskedGatherOpPattern>(patterns.getContext());
}

std::unique_ptr<Pass> mlir::vector::createLowerVectorMaskPass() {
  return std::make_unique<LowerVectorMaskPass>();
}
