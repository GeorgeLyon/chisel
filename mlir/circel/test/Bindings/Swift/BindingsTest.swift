// RUN: mlir-bindings-swift-test 2>&1 | FileCheck %s

import CxxStdlib
import MLIR_Bindings_Support

let context = mlir.bindings.ScopedContext.create()

// CHECK: Hello, Swift!
print("Hello, Swift!")
