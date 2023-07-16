// RUN: circel-swift-bindings-test 2>&1 | FileCheck %s

import CxxStdlib
import MLIR_Bindings_Support

let context = mlir.bindings.Context()
let builder = mlir.bindings.IRBuilder(context)
// CHECK: "Hello, IRBuilder!"
builder.test()
