//===- LLHDOps.h - Declare LLHD dialect operations --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the operation class for the LLHD IR.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_LLHD_IR_LLHDOPS_H
#define CIRCT_DIALECT_LLHD_IR_LLHDOPS_H

#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDEnums.h.inc"
#include "circt/Dialect/LLHD/IR/LLHDTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace circt {
namespace llhd {

unsigned getLLHDTypeWidth(Type type);
Type getLLHDElementType(Type type);

} // namespace llhd
} // namespace circt

/// Retrieve the class declarations generated by TableGen
#define GET_OP_CLASSES
#include "circt/Dialect/LLHD/IR/LLHD.h.inc"

#endif // CIRCT_DIALECT_LLHD_IR_LLHDOPS_H
