//===- VectorEmulateNarrowType.cpp - Narrow type emulation ----*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/NarrowTypeEmulationConverter.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include <cassert>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ConvertVectorLoad
//===----------------------------------------------------------------------===//

struct ConvertVectorLoad final : OpConversionPattern<vector::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto convertedType = cast<MemRefType>(adaptor.getBase().getType());
    Type oldElementType = op.getType().getElementType();
    Type newElementType = convertedType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

    // Adjust the number of elements to load when emulating narrow types,
    // and then cast back to the original type with vector.bitcast op.
    // Here only the 1-D vector load is considered, and the N-D memref types
    // should be linearized.
    // For example, to emulate i4 to i8, the following op:
    //
    // %1 = vector.load %0[%c0, %c0] : memref<3x4xi4>, vector<4xi4>
    //
    // can be replaced with
    //
    // %1 = vector.load %0[%linear_index] : memref<12xi8>, vector<2xi8>
    // %2 = vector.bitcast %1 : vector<2xi8> to vector<4xi4>
    //
    // TODO: Currently, only the even number of elements loading is supported.
    // To deal with the odd number of elements, one has to extract the
    // subvector at the proper offset after bit-casting.

    auto origElements = op.getVectorType().getNumElements();
    if (origElements % scale != 0)
      return failure();

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getBase());

    OpFoldResult linearizedIndices;
    std::tie(std::ignore, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    auto numElements = (origElements + scale - 1) / scale;
    auto newLoad = rewriter.create<vector::LoadOp>(
        loc, VectorType::get(numElements, newElementType), adaptor.getBase(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices));

    auto bitCast =
        rewriter.create<vector::BitCastOp>(loc, op.getType(), newLoad);

    rewriter.replaceOp(op, bitCast->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertVectorTransferRead
//===----------------------------------------------------------------------===//

struct ConvertVectorTransferRead final
    : OpConversionPattern<vector::TransferReadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::TransferReadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto convertedType = cast<MemRefType>(adaptor.getSource().getType());
    Type oldElementType = op.getType().getElementType();
    Type newElementType = convertedType.getElementType();
    int srcBits = oldElementType.getIntOrFloatBitWidth();
    int dstBits = newElementType.getIntOrFloatBitWidth();

    if (dstBits % srcBits != 0) {
      return rewriter.notifyMatchFailure(
          op, "only dstBits % srcBits == 0 supported");
    }
    int scale = dstBits / srcBits;

    auto origElements = op.getVectorType().getNumElements();
    if (origElements % scale != 0)
      return failure();

    auto newPadding = rewriter.create<arith::ExtUIOp>(loc, newElementType,
                                                      adaptor.getPadding());

    auto stridedMetadata =
        rewriter.create<memref::ExtractStridedMetadataOp>(loc, op.getSource());

    OpFoldResult linearizedIndices;
    std::tie(std::ignore, linearizedIndices) =
        memref::getLinearizedMemRefOffsetAndSize(
            rewriter, loc, srcBits, dstBits,
            stridedMetadata.getConstifiedMixedOffset(),
            stridedMetadata.getConstifiedMixedSizes(),
            stridedMetadata.getConstifiedMixedStrides(),
            getAsOpFoldResult(adaptor.getIndices()));

    auto numElements = (origElements + scale - 1) / scale;
    auto newReadType = VectorType::get(numElements, newElementType);

    auto newRead = rewriter.create<vector::TransferReadOp>(
        loc, newReadType, adaptor.getSource(),
        getValueOrCreateConstantIndexOp(rewriter, loc, linearizedIndices),
        newPadding);

    auto bitCast =
        rewriter.create<vector::BitCastOp>(loc, op.getType(), newRead);

    rewriter.replaceOp(op, bitCast->getResult(0));
    return success();
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// Public Interface Definition
//===----------------------------------------------------------------------===//

void vector::populateVectorNarrowTypeEmulationPatterns(
    arith::NarrowTypeEmulationConverter &typeConverter,
    RewritePatternSet &patterns) {

  // Populate `vector.*` conversion patterns.
  patterns.add<ConvertVectorLoad, ConvertVectorTransferRead>(
      typeConverter, patterns.getContext());
}
