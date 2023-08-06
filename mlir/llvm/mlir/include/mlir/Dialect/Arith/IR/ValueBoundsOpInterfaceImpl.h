//===- ValueBoundsOpInterfaceImpl.h - Impl. of ValueBoundsOpInterface -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_IR_VALUEBOUNDSOPINTERFACEIMPL_H
#define MLIR_DIALECT_ARITH_IR_VALUEBOUNDSOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace arith {
void registerValueBoundsOpInterfaceExternalModels(DialectRegistry &registry);
} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_IR_VALUEBOUNDSOPINTERFACEIMPL_H
