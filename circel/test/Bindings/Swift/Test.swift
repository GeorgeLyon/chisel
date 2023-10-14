// RUN: circel-bindings-swift-test 2>&1 | FileCheck %s

import CxxStdlib
import CIRCEL_Bindings_Support

let context = circel.ScopedContext.create()

// CHECK: Hello, Swift!
print("Hello, Swift!")
